# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB placement-table redistribution after a peer death.

Adapted from the scale-down flow in PR #38862
(``vllm/distributed/elastic_ep/elastic_execute.py`` on the
``feat/ep-fault-tolerance`` branch by Tzu-Ling), with one important
change: PR #38862 ties redistribution to elastic scale-down (the EP
topology shrinks to ``dp_size - 1``), here **the topology stays the
same**. The dead peer's physical slots remain in
``physical_to_logical_map`` -- they're marked ``-1`` (no expert) and
the missing logical experts are reassigned onto the most-redundant
surviving slots.

What this module does *not* do (yet): move expert weights. PR #38862
uses NCCL P2P + a disk-reload fallback to put the new expert into the
correct physical slot. For the demo, only the placement table is
updated; deployments running with ``replica_count >= 2`` per expert
will already have a surviving copy of every logical expert in some
other rank's slot, so the placement table update alone preserves
correctness as long as the router can find that surviving copy. When
no surviving replica exists, the expert is effectively unreachable
until weight reload is wired in.
"""

from __future__ import annotations

import torch


def dead_dp_to_ep_ranks(
    dead_dp_ranks: set[int] | list[int],
    tp_size: int,
) -> set[int]:
    """Expand dead DP ranks to the corresponding dead EP ranks.

    Matches the helper of the same name in PR #38862. For ``tp_size == 1``
    (the demo assumption) the result is just ``set(dead_dp_ranks)``.
    """
    dead_ep: set[int] = set()
    for dp_rank in dead_dp_ranks:
        for tp_offset in range(tp_size):
            dead_ep.add(dp_rank * tp_size + tp_offset)
    return dead_ep


def mark_dead_columns_inplace(
    physical_to_logical_map: torch.Tensor,
    dead_ep_ranks: set[int],
    num_local_experts: int,
) -> None:
    """Mark the dead EP ranks' columns as ``-1`` (no expert) in place.

    ``physical_to_logical_map`` is ``[num_moe_layers, num_physical]``;
    each EP rank owns the ``num_local_experts`` consecutive columns
    starting at ``rank * num_local_experts``.

    Unlike PR #38862's ``strip_dead_columns`` (which returns a smaller
    tensor with dead columns removed for the scale-down case), this
    keeps the shape constant so the NIXL EP topology and CUDA-graph
    capture are unaffected.
    """
    num_physical = physical_to_logical_map.shape[1]
    ep_world_size = num_physical // num_local_experts
    for ep_rank in dead_ep_ranks:
        if ep_rank < 0 or ep_rank >= ep_world_size:
            continue
        start = ep_rank * num_local_experts
        end = start + num_local_experts
        physical_to_logical_map[:, start:end] = -1


def reassign_missing_experts_inplace(
    physical_to_logical_map: torch.Tensor,
    num_logical: int,
) -> bool:
    """Reassign logical experts that have lost all physical replicas.

    Operates layer-by-layer on ``physical_to_logical_map`` in place.
    For each layer:

      1. Count how many physical slots host each logical expert.
      2. Any logical id ``0 <= lid < num_logical`` that's absent from the
         layer is "missing."
      3. Pick the most-redundant remaining slots (those whose donor
         logical has ``> 1`` replica in the layer) and overwrite them
         with the missing ids, one missing id per slot. A donor's
         counter is decremented as we steal from it -- we never take a
         donor below ``1`` replica.

    ``num_logical`` is taken from the caller (typically
    ``eplb_model_state.logical_replica_count.shape[1]``) rather than
    inferred from the data, because a logical id with zero surviving
    replicas would otherwise be invisible.

    Returns ``True`` if at least one slot was reassigned, ``False`` if
    every logical expert was already present in every layer (no-op).
    Raises ``RuntimeError`` when redundancy is insufficient to cover
    every missing expert -- the caller should detect this and either
    scale down or reload from disk.

    Adapted from PR #38862's ``ElasticEPScalingExecutor.reassign_missing_experts``.
    """
    if physical_to_logical_map.ndim != 2:
        raise ValueError(
            f"physical_to_logical_map must be 2D; got shape "
            f"{tuple(physical_to_logical_map.shape)}"
        )

    num_layers, num_physical = physical_to_logical_map.shape
    all_logical = set(range(num_logical))
    any_reassigned = False

    for layer_idx in range(num_layers):
        layer = physical_to_logical_map[layer_idx]

        replica_count: dict[int, int] = {}
        for phys_idx in range(num_physical):
            lid = int(layer[phys_idx].item())
            if lid >= 0:
                replica_count[lid] = replica_count.get(lid, 0) + 1

        missing = sorted(all_logical - set(replica_count.keys()))
        if not missing:
            continue
        any_reassigned = True

        # Build a list of (redundancy, phys_idx) for slots whose donor
        # still has > 1 replica. Sorted from most-redundant down.
        candidates: list[tuple[int, int]] = []
        for phys_idx in range(num_physical):
            lid = int(layer[phys_idx].item())
            if lid >= 0 and replica_count.get(lid, 0) > 1:
                candidates.append((replica_count[lid], phys_idx))
        candidates.sort(reverse=True)
        slot_iter = iter(candidates)

        for logical_id in missing:
            placed = False
            while True:
                candidate = next(slot_iter, None)
                if candidate is None:
                    raise RuntimeError(
                        f"reassign_missing_experts_inplace: layer "
                        f"{layer_idx}: no redundant slot left to host "
                        f"missing logical expert {logical_id}. EPLB "
                        f"redundancy is insufficient for the surviving "
                        f"topology."
                    )
                _, global_slot = candidate
                old_lid = int(layer[global_slot].item())
                if replica_count.get(old_lid, 0) > 1:
                    layer[global_slot] = logical_id
                    replica_count[old_lid] -= 1
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    f"reassign_missing_experts_inplace: layer "
                    f"{layer_idx}: failed to place logical expert "
                    f"{logical_id} despite candidate slots remaining."
                )

    return any_reassigned


def rebuild_derived_maps_inplace(
    physical_to_logical_map: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> None:
    """Rebuild ``logical_to_physical_map`` and ``logical_replica_count``
    from ``physical_to_logical_map``, in place.

    Call after any modification to ``physical_to_logical_map``
    (column-marking, reassignment) to keep the derived maps consistent.
    Modifies the tensors in place so existing views (held by FusedMoE
    layers) see the updates.

    Layout (matching PR #38862's ``rebuild_eplb_derived_maps``):

    * ``physical_to_logical_map``: ``[num_layers, num_physical]`` int.
    * ``logical_to_physical_map``: ``[num_layers, num_logical, max_replicas]``
      int, padded with ``-1``.
    * ``logical_replica_count``: ``[num_layers, num_logical]`` int.
    """
    num_layers, num_physical = physical_to_logical_map.shape
    logical_replica_count.zero_()
    logical_to_physical_map.fill_(-1)
    for layer_idx in range(num_layers):
        for phys_idx in range(num_physical):
            lid = int(physical_to_logical_map[layer_idx, phys_idx].item())
            if lid < 0:
                continue
            c = int(logical_replica_count[layer_idx, lid].item())
            if c < logical_to_physical_map.shape[2]:
                logical_to_physical_map[layer_idx, lid, c] = phys_idx
            logical_replica_count[layer_idx, lid] += 1
