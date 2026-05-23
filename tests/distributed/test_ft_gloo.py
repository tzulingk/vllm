# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.distributed.elastic_ep.ft_gloo.

External dependencies (torch.distributed process groups, the gloo
backend, real network rendezvous) are mocked so the wrapper's logic
can be exercised without spawning processes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed import HashStore, ReduceOp

from vllm.distributed.elastic_ep import ft_gloo as ft_gloo_mod
from vllm.distributed.elastic_ep.ft_gloo import (
    DPFTGlooManager,
    FaultTolerantGlooGroup,
    RebuildTimeoutError,
)

# ------------------------------ Test helpers ------------------------------ #


def _make_mock_pg(name: str) -> MagicMock:
    """A MagicMock standing in for a ProcessGroup."""
    pg = MagicMock(name=name)
    return pg


def _make_mock_work(should_raise: Exception | None = None) -> MagicMock:
    """A MagicMock standing in for a c10d Work object.

    If ``should_raise`` is set, ``work.wait(...)`` raises it.
    """
    work = MagicMock(name="Work")
    if should_raise is None:
        work.wait = MagicMock(return_value=True)
    else:
        work.wait = MagicMock(side_effect=should_raise)
    return work


@pytest.fixture(autouse=True)
def _reset_singleton():
    DPFTGlooManager.reset_instance()
    yield
    DPFTGlooManager.reset_instance()


@pytest.fixture
def store():
    """A fresh in-memory HashStore for each test."""
    return HashStore()


@pytest.fixture
def patched_init():
    """Patch the stateless-init helper to return successive mock PGs.

    Yields the mock so tests can assert call_args / call_count.
    """
    with patch.object(
        ft_gloo_mod, "stateless_init_torch_distributed_process_group"
    ) as m:
        counter = {"i": 0}

        def _factory(*args, **kwargs):
            counter["i"] += 1
            return _make_mock_pg(f"pg_gen{counter['i']}")

        m.side_effect = _factory
        yield m


@pytest.fixture
def patched_destroy():
    with patch.object(
        ft_gloo_mod, "stateless_destroy_torch_distributed_process_group"
    ) as m:
        yield m


@pytest.fixture
def patched_all_reduce():
    with patch.object(ft_gloo_mod.dist, "all_reduce") as m:
        m.return_value = _make_mock_work()
        yield m


# --------------------------- FaultTolerantGlooGroup ----------------------- #


def test_init_validation_rejects_bad_rank(store):
    with pytest.raises(ValueError, match="out of range"):
        FaultTolerantGlooGroup(
            store=store, master_addr="127.0.0.1", my_global_rank=4, total_world_size=4
        )


def test_first_call_builds_group(
    store, patched_init, patched_destroy, patched_all_reduce
):
    ft = FaultTolerantGlooGroup(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    tensor = torch.tensor([1], dtype=torch.int32)
    _, valid = ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
    assert valid is True
    assert patched_init.call_count == 1
    assert ft.generation == 1
    assert ft.current_active_set == frozenset({0, 1, 2, 3})
    patched_destroy.assert_not_called()  # no old group on first build


def test_no_rebuild_on_same_mask(
    store, patched_init, patched_destroy, patched_all_reduce
):
    ft = FaultTolerantGlooGroup(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    tensor = torch.tensor([1], dtype=torch.int32)
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
    assert patched_init.call_count == 1
    assert ft.generation == 1
    assert patched_all_reduce.call_count == 3


def test_rebuild_on_mask_change(
    store, patched_init, patched_destroy, patched_all_reduce
):
    ft = FaultTolerantGlooGroup(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    tensor = torch.tensor([1], dtype=torch.int32)
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 0, 1])  # peer 2 dies
    assert patched_init.call_count == 2
    assert ft.generation == 2
    assert ft.current_active_set == frozenset({0, 1, 3})
    patched_destroy.assert_called_once()  # the gen-1 group was torn down


def test_generation_increments_monotonically(
    store, patched_init, patched_destroy, patched_all_reduce
):
    ft = FaultTolerantGlooGroup(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    tensor = torch.tensor([1], dtype=torch.int32)
    assert ft.generation == 0
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
    assert ft.generation == 1
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 0, 1])
    assert ft.generation == 2
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 0, 0, 1])
    assert ft.generation == 3


def test_all_reduce_skip_when_local_dead(
    store, patched_init, patched_destroy, patched_all_reduce
):
    ft = FaultTolerantGlooGroup(
        store=store, master_addr="127.0.0.1", my_global_rank=1, total_world_size=4
    )
    tensor = torch.tensor([5], dtype=torch.int32)
    out, valid = ft.all_reduce(
        tensor,
        ReduceOp.SUM,
        active_mask=[1, 0, 1, 1],  # rank 1 dead
    )
    assert valid is False
    assert torch.equal(out, torch.tensor([5], dtype=torch.int32))
    patched_init.assert_not_called()  # no group built when local is dead
    patched_all_reduce.assert_not_called()


def test_rebuild_failure_raises_RebuildTimeoutError(store, patched_destroy):
    with patch.object(
        ft_gloo_mod,
        "stateless_init_torch_distributed_process_group",
        side_effect=RuntimeError("rendezvous timed out"),
    ):
        ft = FaultTolerantGlooGroup(
            store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
        )
        tensor = torch.tensor([1], dtype=torch.int32)
        with pytest.raises(RebuildTimeoutError, match="rebuild of generation 1"):
            ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])


def test_online_all_reduce_failure_returns_false(store, patched_init, patched_destroy):
    """When dist.all_reduce / Work.wait raises, return (tensor, False)."""
    with patch.object(ft_gloo_mod.dist, "all_reduce") as m_ar:
        m_ar.return_value = _make_mock_work(should_raise=TimeoutError("hung peer"))
        ft = FaultTolerantGlooGroup(
            store=store,
            master_addr="127.0.0.1",
            my_global_rank=0,
            total_world_size=4,
        )
        tensor = torch.tensor([1], dtype=torch.int32)
        _, valid = ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
        assert valid is False
        # The group was still built (gen=1) so subsequent calls with the
        # same mask won't rebuild.
        assert ft.generation == 1


def test_destroy_old_before_rebuild(
    store, patched_init, patched_destroy, patched_all_reduce
):
    ft = FaultTolerantGlooGroup(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    tensor = torch.tensor([1], dtype=torch.int32)
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 0, 1])
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 0, 0, 1])
    # gen1 destroyed before gen2 build; gen2 destroyed before gen3 build.
    assert patched_destroy.call_count == 2


def test_port_published_through_store(store, patched_destroy, patched_all_reduce):
    """Master writes ft_gloo_port_<gen>; the wrapper passes it to stateless_init."""
    captured_ports: list[int] = []

    def _capture_init(*, host, port, rank, world_size, backend):
        captured_ports.append(port)
        return _make_mock_pg(f"pg_port_{port}")

    with patch.object(
        ft_gloo_mod, "stateless_init_torch_distributed_process_group"
    ) as m_init:
        m_init.side_effect = _capture_init
        ft = FaultTolerantGlooGroup(
            store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
        )
        tensor = torch.tensor([1], dtype=torch.int32)
        ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])

    assert len(captured_ports) == 1
    port = captured_ports[0]
    assert 1024 < port < 65536
    # And the master should have written that port into the store.
    raw = store.get("ft_gloo_port_1")
    assert int(raw.decode()) == port


def test_mask_length_mismatch_raises_value_error(store):
    ft = FaultTolerantGlooGroup(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    tensor = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError, match="active_mask length"):
        ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1])  # wrong size


def test_destroy_cleans_up_current_group(
    store, patched_init, patched_destroy, patched_all_reduce
):
    ft = FaultTolerantGlooGroup(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    tensor = torch.tensor([1], dtype=torch.int32)
    ft.all_reduce(tensor, ReduceOp.SUM, active_mask=[1, 1, 1, 1])
    patched_destroy.assert_not_called()
    ft.destroy()
    patched_destroy.assert_called_once()


# ------------------------------ DPFTGlooManager --------------------------- #


def test_dpft_manager_initially_uninitialized():
    assert DPFTGlooManager.instance() is None
    assert DPFTGlooManager.is_initialized() is False


def test_dpft_manager_init_returns_wrapper(store):
    ft = DPFTGlooManager.init(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    assert isinstance(ft, FaultTolerantGlooGroup)
    assert DPFTGlooManager.instance() is ft
    assert DPFTGlooManager.is_initialized() is True


def test_dpft_manager_init_idempotent(store):
    first = DPFTGlooManager.init(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    second = DPFTGlooManager.init(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    assert first is second


def test_dpft_manager_reset_instance_destroys_and_clears(store, patched_destroy):
    ft = DPFTGlooManager.init(
        store=store, master_addr="127.0.0.1", my_global_rank=0, total_world_size=4
    )
    # Force a group to exist via patched build.
    with (
        patch.object(
            ft_gloo_mod, "stateless_init_torch_distributed_process_group"
        ) as m_init,
        patch.object(ft_gloo_mod.dist, "all_reduce") as m_ar,
    ):
        m_init.return_value = _make_mock_pg("pg1")
        m_ar.return_value = _make_mock_work()
        ft.all_reduce(torch.tensor([1]), ReduceOp.SUM, [1, 1, 1, 1])
    DPFTGlooManager.reset_instance()
    assert DPFTGlooManager.instance() is None
    patched_destroy.assert_called()
