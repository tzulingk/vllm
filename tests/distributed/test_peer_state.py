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
