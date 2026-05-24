# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.distributed.elastic_ep.peer_state."""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.elastic_ep.peer_state import (
    PeerActiveState,
    PeerActiveStateManager,
    apply_kernel_mask,
)


@pytest.fixture(autouse=True)
def _reset_manager_singleton():
    """Make every test start with a fresh PeerActiveStateManager."""
    PeerActiveStateManager.reset_instance()
    yield
    PeerActiveStateManager.reset_instance()


# ----------------------------- PeerActiveState ----------------------------- #


def test_init_all_alive():
    state = PeerActiveStateManager.init(ep_size=4)
    assert state.active_ranks.tolist() == [1, 1, 1, 1]
    assert state.last_active_ranks.tolist() == [1, 1, 1, 1]
    assert state.active_ranks_cpu.tolist() == [1, 1, 1, 1]


def test_is_active_equal_last_initially_true():
    state = PeerActiveStateManager.init(ep_size=4)
    assert state.is_active_equal_last() is True


def test_apply_kernel_mask_no_change_keeps_all_alive():
    state = PeerActiveStateManager.init(ep_size=4)
    apply_kernel_mask(state, torch.zeros(4, dtype=torch.int32))
    assert state.active_ranks.tolist() == [1, 1, 1, 1]
    assert state.active_ranks_cpu.tolist() == [1, 1, 1, 1]
    assert state.is_active_equal_last() is True


def test_apply_kernel_mask_inverts_convention():
    """Kernel mask convention: 1=dead. State convention: 1=alive."""
    state = PeerActiveStateManager.init(ep_size=4)
    apply_kernel_mask(state, torch.tensor([1, 0, 1, 0], dtype=torch.int32))
    # kernel: 1,0,1,0 → state: 0,1,0,1
    assert state.active_ranks.tolist() == [0, 1, 0, 1]
    assert state.active_ranks_cpu.tolist() == [0, 1, 0, 1]


def test_apply_kernel_mask_keeps_active_ranks_cpu_in_sync():
    state = PeerActiveStateManager.init(ep_size=3)
    apply_kernel_mask(state, torch.tensor([0, 1, 0], dtype=torch.int32))
    assert state.active_ranks.tolist() == state.active_ranks_cpu.tolist()


def test_apply_kernel_mask_shape_mismatch_raises():
    state = PeerActiveStateManager.init(ep_size=4)
    with pytest.raises(ValueError, match="shape"):
        apply_kernel_mask(state, torch.zeros(5, dtype=torch.int32))


def test_is_active_equal_last_false_after_change_then_true_after_snapshot():
    state = PeerActiveStateManager.init(ep_size=4)
    apply_kernel_mask(state, torch.tensor([0, 1, 0, 0], dtype=torch.int32))
    assert state.is_active_equal_last() is False
    state.snapshot_active_to_last()
    assert state.is_active_equal_last() is True


def test_newly_dead_peers_empty_when_no_change():
    state = PeerActiveStateManager.init(ep_size=4)
    assert state.newly_dead_peers() == []


def test_newly_dead_peers_returns_flipped_indices():
    state = PeerActiveStateManager.init(ep_size=5)
    # All alive initially; flip peers 1 and 3 to dead.
    apply_kernel_mask(state, torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32))
    assert state.newly_dead_peers() == [1, 3]


def test_newly_dead_peers_ignores_already_dead():
    state = PeerActiveStateManager.init(ep_size=4)
    # Peer 1 dies first.
    apply_kernel_mask(state, torch.tensor([0, 1, 0, 0], dtype=torch.int32))
    state.snapshot_active_to_last()
    # Now peer 2 dies; peer 1 is still dead but not newly dead.
    apply_kernel_mask(state, torch.tensor([0, 1, 1, 0], dtype=torch.int32))
    assert state.newly_dead_peers() == [2]


def test_alive_ranks():
    state = PeerActiveStateManager.init(ep_size=5)
    apply_kernel_mask(state, torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32))
    assert state.alive_ranks() == [0, 2, 4]


def test_reset_marks_all_alive_and_snapshots():
    state = PeerActiveStateManager.init(ep_size=4)
    apply_kernel_mask(state, torch.tensor([1, 1, 0, 0], dtype=torch.int32))
    state.snapshot_active_to_last()
    state.reset()
    assert state.active_ranks.tolist() == [1, 1, 1, 1]
    assert state.last_active_ranks.tolist() == [1, 1, 1, 1]
    assert state.active_ranks_cpu.tolist() == [1, 1, 1, 1]
    assert state.is_active_equal_last() is True


# --------------------------- PeerActiveStateManager ------------------------ #


def test_manager_initially_uninitialized():
    assert PeerActiveStateManager.instance() is None
    assert PeerActiveStateManager.is_initialized() is False


def test_manager_init_returns_state_and_flags_initialized():
    state = PeerActiveStateManager.init(ep_size=2)
    assert isinstance(state, PeerActiveState)
    assert PeerActiveStateManager.instance() is state
    assert PeerActiveStateManager.is_initialized() is True


def test_manager_init_idempotent_returns_same_instance():
    first = PeerActiveStateManager.init(ep_size=4)
    second = PeerActiveStateManager.init(ep_size=4)
    assert first is second


def test_manager_init_size_mismatch_returns_existing(caplog):
    first = PeerActiveStateManager.init(ep_size=4)
    second = PeerActiveStateManager.init(ep_size=8)
    assert first is second
    assert first.active_ranks.numel() == 4


def test_manager_reset_instance_clears_state():
    PeerActiveStateManager.init(ep_size=4)
    assert PeerActiveStateManager.is_initialized() is True
    PeerActiveStateManager.reset_instance()
    assert PeerActiveStateManager.instance() is None
    assert PeerActiveStateManager.is_initialized() is False


# ----------------------------- TP > 1 derivations ----------------------- #


def test_manager_init_stores_tp_size():
    state = PeerActiveStateManager.init(ep_size=8, tp_size=2)
    assert state.tp_size == 2
    assert state.ep_size == 8
    assert state.dp_size == 4


def test_manager_init_rejects_ep_not_multiple_of_tp():
    with pytest.raises(ValueError, match="multiple of tp_size"):
        PeerActiveStateManager.init(ep_size=5, tp_size=2)


def test_dp_active_mask_tp1_identity():
    state = PeerActiveStateManager.init(ep_size=4, tp_size=1)
    apply_kernel_mask(state, torch.tensor([0, 1, 0, 0], dtype=torch.int32))
    # TP=1 -> DP view matches the EP view bit for bit.
    assert state.dp_active_mask() == [1, 0, 1, 1]


def test_dp_active_mask_tp2_or_reduce_keeps_partial_dp_alive():
    # 4 DP ranks, TP=2 -> 8 EP slots.
    # Kill EP slot 4 (DP rank 2's first TP sibling). Slot 5 (DP rank 2's
    # other sibling) stays alive -> DP rank 2 still alive for the
    # cross-DP collective.
    state = PeerActiveStateManager.init(ep_size=8, tp_size=2)
    apply_kernel_mask(state, torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.int32))
    assert state.dp_active_mask() == [1, 1, 1, 1]


def test_dp_active_mask_tp2_drops_fully_dead_dp_rank():
    # Kill both TP siblings of DP rank 1 (EP slots 2, 3).
    state = PeerActiveStateManager.init(ep_size=8, tp_size=2)
    apply_kernel_mask(state, torch.tensor([0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.int32))
    assert state.dp_active_mask() == [1, 0, 1, 1]


def test_dp_dead_ranks_tp2_and_reduce_keeps_partial_alive():
    # One sibling dead in DP rank 2 -> DP rank 2 is NOT in the dead set
    # (the surviving sibling can still serve).
    state = PeerActiveStateManager.init(ep_size=8, tp_size=2)
    apply_kernel_mask(state, torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.int32))
    assert state.dp_dead_ranks() == set()


def test_dp_dead_ranks_tp2_full_kill_in_set():
    state = PeerActiveStateManager.init(ep_size=8, tp_size=2)
    apply_kernel_mask(state, torch.tensor([0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.int32))
    assert state.dp_dead_ranks() == {1}


def test_dp_dead_ranks_tp4_partial_alive():
    # 2 DP ranks, TP=4 -> 8 EP slots. Kill 3 of the 4 siblings of DP 0;
    # DP 0 still has 1 alive -> not dead.
    state = PeerActiveStateManager.init(ep_size=8, tp_size=4)
    apply_kernel_mask(state, torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32))
    assert state.dp_dead_ranks() == set()
    assert state.dp_active_mask() == [1, 1]


def test_newly_dead_peers_returns_ep_indexed():
    """newly_dead_peers must report EP-level indices (not DP)."""
    state = PeerActiveStateManager.init(ep_size=8, tp_size=2)
    apply_kernel_mask(state, torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.int32))
    # EP slot 4 died -- reporting MUST be EP-indexed for the EPLB
    # redistribution to work.
    assert state.newly_dead_peers() == [4]
