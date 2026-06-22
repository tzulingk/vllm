# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fault-tolerant DP gloo sub-group.

The multiprocess test is the important one: it spawns a real DP world,
"kills" one rank (that rank never participates), and asserts the
survivors rebuild a gloo sub-group and all-reduce correctly over just
the survivors. The demo branch only ever mocked the group, which is why
its rebuild/rendezvous bugs went unnoticed.
"""

import pytest
import torch
import torch.multiprocessing as mp
from torch.distributed import ReduceOp, TCPStore

from vllm.distributed.elastic_ep.ft_gloo import (
    FaultTolerantGlooGroup,
    get_dp_ft_gloo,
    init_dp_ft_gloo,
    reset_dp_ft_gloo,
)
from vllm.utils.network_utils import get_open_port

MASTER_ADDR = "127.0.0.1"


# ------------------------------ unit (single proc) ------------------------- #


def test_init_rejects_out_of_range_rank():
    store = TCPStore(
        MASTER_ADDR, get_open_port(), 1, is_master=True, wait_for_workers=False
    )
    for bad in (-1, 4):
        try:
            FaultTolerantGlooGroup(store, MASTER_ADDR, bad, total_world_size=4)
            raise AssertionError(f"expected ValueError for rank {bad}")
        except ValueError:
            pass


def test_all_reduce_before_rebuild_is_invalid():
    store = TCPStore(
        MASTER_ADDR, get_open_port(), 1, is_master=True, wait_for_workers=False
    )
    ft = FaultTolerantGlooGroup(store, MASTER_ADDR, 0, total_world_size=4)
    assert ft.has_group is False
    t = torch.ones(1, dtype=torch.float32)
    out, valid = ft.all_reduce(t, ReduceOp.SUM)
    assert valid is False
    # Tensor is left untouched so the caller can use its local value.
    assert out.item() == 1.0


def test_singleton_accessor_lifecycle():
    reset_dp_ft_gloo()
    assert get_dp_ft_gloo() is None
    store = TCPStore(
        MASTER_ADDR, get_open_port(), 1, is_master=True, wait_for_workers=False
    )
    ft = init_dp_ft_gloo(store, MASTER_ADDR, 0, total_world_size=4)
    assert get_dp_ft_gloo() is ft
    # idempotent
    assert init_dp_ft_gloo(store, MASTER_ADDR, 0, 4) is ft
    reset_dp_ft_gloo()
    assert get_dp_ft_gloo() is None


# ------------------------------ multiprocess ------------------------------- #


def _survivor_worker(rank, world_size, store_port, dead_rank, result_queue):
    survivors = frozenset(r for r in range(world_size) if r != dead_rank)
    if rank == dead_rank:
        # This rank is "dead": it never joins the rebuilt group. It must
        # not host the coordination store, so dead_rank is never 0.
        return
    store = TCPStore(
        MASTER_ADDR,
        store_port,
        world_size,
        is_master=(rank == 0),
        wait_for_workers=False,
    )
    ft = FaultTolerantGlooGroup(store, MASTER_ADDR, rank, total_world_size=world_size)
    try:
        ft.rebuild_for_survivors(survivors)
        assert ft.has_group is True
        assert ft.current_survivors == survivors
        assert ft.generation == 1
        # Each survivor contributes 1.0; the reduced sum must equal the
        # survivor count -- proving the group really excludes the dead rank.
        t = torch.ones(1, dtype=torch.float32)
        out, valid = ft.all_reduce(t, ReduceOp.SUM)
        result_queue.put((rank, valid, out.item()))
    finally:
        ft.destroy()


def test_survivors_rebuild_and_all_reduce():
    # The rebuild path calls stateless_init_torch_distributed_process_group,
    # which imports vllm.config (for the gloo timeout). That pulls the full
    # vLLM runtime dep tree, absent from a bare precompiled/CPU venv. Skip
    # cleanly there; runs fully in the CUDA/image test env.
    try:
        import vllm.config  # noqa: F401
    except ModuleNotFoundError as e:
        pytest.skip(f"full vLLM runtime deps not installed: {e}")

    world_size = 4
    dead_rank = 1  # not 0, so the coordination store host survives
    store_port = get_open_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_survivor_worker,
            args=(rank, world_size, store_port, dead_rank, result_queue),
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    results = {}
    while not result_queue.empty():
        rank, valid, value = result_queue.get()
        results[rank] = (valid, value)

    expected_survivors = {0, 2, 3}
    assert set(results) == expected_survivors
    for rank in expected_survivors:
        valid, value = results[rank]
        assert valid is True, f"rank {rank} all_reduce invalid"
        assert value == float(len(expected_survivors)), (
            f"rank {rank} got {value}, expected {len(expected_survivors)}"
        )
