# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.distributed.elastic_ep.eplb_redistribute.

Pure algorithm tests -- no torch.distributed, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.elastic_ep.eplb_redistribute import (
    dead_dp_to_ep_ranks,
    mark_dead_columns_inplace,
    reassign_missing_experts_inplace,
    rebuild_derived_maps_inplace,
)

# ----------------------------- dead_dp_to_ep_ranks ----------------------- #


def test_dead_dp_to_ep_ranks_tp1():
    assert dead_dp_to_ep_ranks({1, 3}, tp_size=1) == {1, 3}


def test_dead_dp_to_ep_ranks_tp2():
    # DP 1 -> EP 2,3; DP 3 -> EP 6,7
    assert dead_dp_to_ep_ranks({1, 3}, tp_size=2) == {2, 3, 6, 7}


def test_dead_dp_to_ep_ranks_empty():
    assert dead_dp_to_ep_ranks(set(), tp_size=4) == set()


# --------------------------- mark_dead_columns --------------------------- #


def test_mark_dead_columns_zeroes_target_ranks():
    # 2 MoE layers, ep_world_size=4, num_local=2 -> num_physical=8
    p2l = torch.arange(16, dtype=torch.int32).view(2, 8)
    mark_dead_columns_inplace(p2l, dead_ep_ranks={2}, num_local_experts=2)
    # EP rank 2 owns columns 4..6.
    assert p2l[0, 4].item() == -1
    assert p2l[0, 5].item() == -1
    assert p2l[1, 4].item() == -1
    assert p2l[1, 5].item() == -1
    # Other columns untouched.
    assert p2l[0, 0].item() == 0
    assert p2l[0, 7].item() == 7
    assert p2l[1, 0].item() == 8


def test_mark_dead_columns_multiple_ranks():
    p2l = torch.arange(16, dtype=torch.int32).view(2, 8)
    mark_dead_columns_inplace(p2l, dead_ep_ranks={0, 3}, num_local_experts=2)
    # Ranks 0 (cols 0..1) and 3 (cols 6..7) all -1.
    for r, cols in [(0, [0, 1, 6, 7])]:
        for c in cols:
            assert p2l[r, c].item() == -1
    # Middle rank columns unchanged.
    assert p2l[0, 2].item() == 2


def test_mark_dead_columns_silently_ignores_out_of_range():
    p2l = torch.arange(8, dtype=torch.int32).view(1, 8)
    mark_dead_columns_inplace(p2l, dead_ep_ranks={99}, num_local_experts=2)
    # No change.
    assert p2l.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


# --------------------- reassign_missing_experts_inplace ------------------ #


def test_reassign_no_missing_returns_false():
    # 1 layer, 4 logical, 4 physical, no redundancy, no dead.
    p2l = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    assert reassign_missing_experts_inplace(p2l, num_logical=4) is False
    assert p2l.tolist() == [[0, 1, 2, 3]]


def test_reassign_missing_uses_redundant_slot():
    # 1 layer, 4 logical, 6 physical. Layout: 0,1,2,3,0,1 -> all logical
    # present. Now mark slot 3 dead -> 0,1,2,-1,0,1, missing {3}.
    # Should reassign onto one of the redundant SURVIVING slots (4 or 5);
    # the -1 slot is "dead" -- it belongs to a dead rank and stays -1.
    p2l = torch.tensor([[0, 1, 2, -1, 0, 1]], dtype=torch.int32)
    changed = reassign_missing_experts_inplace(p2l, num_logical=4)
    assert changed is True
    # Logical 3 now has exactly one replica somewhere alive.
    survivors = [int(x) for x in p2l[0].tolist() if x >= 0]
    assert survivors.count(3) == 1
    # Dead slot is untouched.
    assert p2l[0, 3].item() == -1


def test_reassign_missing_picks_most_redundant_first():
    # Layer with 4 logical, 6 physical: 0,0,0,1,1,2. Missing 3.
    # Most-redundant is logical 0 (count=3); a slot holding 0 should be
    # chosen first.
    p2l = torch.tensor([[0, 0, 0, 1, 1, 2]], dtype=torch.int32)
    reassign_missing_experts_inplace(p2l, num_logical=4)
    counts = {int(x): (p2l[0] == x).sum().item() for x in p2l[0].unique().tolist()}
    assert counts.get(3, 0) == 1
    assert counts.get(0, 0) == 2
    assert counts.get(1, 0) == 2
    assert counts.get(2, 0) == 1


def test_reassign_missing_raises_when_no_redundancy():
    # No redundancy: each logical has exactly 1 replica. Now slot 0 is -1.
    p2l = torch.tensor([[-1, 1, 2, 3]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="redundancy is insufficient"):
        reassign_missing_experts_inplace(p2l, num_logical=4)


def test_reassign_missing_handles_multiple_layers():
    p2l = torch.tensor(
        [
            [0, 1, 2, -1, 0, 1],  # missing 3
            [0, 1, 2, 3, -1, 0],  # nothing missing
        ],
        dtype=torch.int32,
    )
    reassign_missing_experts_inplace(p2l, num_logical=4)
    # Layer 0 should now contain 3 somewhere.
    assert 3 in p2l[0].tolist()
    # Layer 1's -1 stays (no missing experts).
    assert -1 in p2l[1].tolist()


# --------------------- rebuild_derived_maps_inplace --------------------- #


def test_rebuild_derived_maps_basic():
    p2l = torch.tensor(
        [
            [0, 1, 2, 0, -1, 1],  # logical 0 has 2 replicas; 1 has 2; 2 has 1
        ],
        dtype=torch.int32,
    )
    num_logical = 3
    max_replicas = 4
    l2p = torch.full((1, num_logical, max_replicas), -2, dtype=torch.int32)
    lrc = torch.full((1, num_logical), -1, dtype=torch.int32)

    rebuild_derived_maps_inplace(p2l, l2p, lrc)

    # Replica counts.
    assert lrc.tolist() == [[2, 2, 1]]

    # Logical 0 should have phys 0, 3.
    l2p_0 = sorted(x for x in l2p[0, 0].tolist() if x >= 0)
    assert l2p_0 == [0, 3]
    # Logical 1 should have phys 1, 5.
    l2p_1 = sorted(x for x in l2p[0, 1].tolist() if x >= 0)
    assert l2p_1 == [1, 5]
    # Logical 2 should have phys 2.
    l2p_2 = sorted(x for x in l2p[0, 2].tolist() if x >= 0)
    assert l2p_2 == [2]
    # Padding stays -1.
    assert l2p[0, 0, 2].item() == -1


def test_rebuild_derived_maps_ignores_negative_p2l():
    # All -1 row -> all counts 0, all l2p -1.
    p2l = torch.tensor([[-1, -1, -1, -1]], dtype=torch.int32)
    l2p = torch.full((1, 2, 2), -5, dtype=torch.int32)
    lrc = torch.full((1, 2), -3, dtype=torch.int32)
    rebuild_derived_maps_inplace(p2l, l2p, lrc)
    assert lrc.tolist() == [[0, 0]]
    assert (l2p == -1).all().item()


# --------------------- end-to-end: mark + reassign + rebuild ------------- #


def test_end_to_end_one_rank_dies():
    """ep_world_size=4, num_local=2, 1 layer, 8 logical experts each twice.

    DP rank 2 dies -> columns 4..5 become -1.
    Two logical experts lose their (only) replica on rank 2 -> reassign
    onto the most-redundant remaining slots. After redistribution every
    logical id must appear at least once.
    """
    # Place each of 8 logical experts in 2 slots, in a known pattern.
    p2l = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.int32,
    ).reshape(1, 16)
    mark_dead_columns_inplace(p2l, dead_ep_ranks={2}, num_local_experts=2)
    # Logical 4 still has a replica at slot 12; logical 5 still has slot 13.
    # So nothing should be missing.
    changed = reassign_missing_experts_inplace(p2l, num_logical=8)
    assert changed is False

    # Now make rank 2's columns hold *unique* logical IDs so killing them
    # creates missing experts.
    p2l2 = torch.tensor(
        [[0, 1, 2, 3, 8, 9, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.int32,
    ).reshape(1, 16)
    mark_dead_columns_inplace(p2l2, dead_ep_ranks={2}, num_local_experts=2)
    # Columns 4..5 (logical 8, 9) become -1; 8, 9 are now missing entirely.
    changed = reassign_missing_experts_inplace(p2l2, num_logical=10)
    assert changed is True
    survivors = [int(x) for x in p2l2[0].tolist() if x >= 0]
    assert 8 in survivors
    assert 9 in survivors

    # Rebuild derived maps and sanity-check.
    num_logical = 10
    l2p = torch.full((1, num_logical, 4), -2, dtype=torch.int32)
    lrc = torch.full((1, num_logical), -1, dtype=torch.int32)
    rebuild_derived_maps_inplace(p2l2, l2p, lrc)
    for lid in range(num_logical):
        assert lrc[0, lid].item() >= 1, f"logical {lid} has no replicas"
