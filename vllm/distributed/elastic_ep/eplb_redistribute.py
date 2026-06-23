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

When a logical expert had **at least one surviving replica** elsewhere,
the placement-table rewrite alone is enough -- the model's MoE router
will find that surviving slot the next time tokens are routed to that
expert. When a logical expert lost **all** of its replicas (every copy
was on the dead rank), the slot we reassigned to that expert still
holds the *donor* expert's weights in its GPU buffer; the placement
table lies. ``reassign_missing_experts_inplace`` therefore returns the
set of ``(layer_idx, new_logical_id)`` pairs it created so the caller
can reload those expert weights from the HF checkpoint via
:func:`vllm.distributed.elastic_ep.eplb_reload.reload_experts_from_disk`.

How ``eplb_state`` stays in sync across surviving ranks
-------------------------------------------------------

Every surviving worker runs the three primitives in this module
(``mark_dead_columns_inplace``, ``reassign_missing_experts_inplace``,
``rebuild_derived_maps_inplace``) on its **own** copy of the EPLB
state, with **no cross-rank communication** during the recovery itself.
The result is identical on every rank because the inputs are identical
on every rank. The invariants:

1. **Initial state is deterministic.** ``EplbState.add_model`` builds
   ``physical_to_logical_map`` from a pure function of the model config
   (``num_routed_experts``, ``num_redundant_experts``) -- see
   ``EplbState.build_initial_global_physical_to_logical_map`` in
   ``vllm/distributed/eplb/eplb_state.py``. No RNG, no broadcasts. Every
   worker constructs the same tensor at startup.

2. **Any rearrangement that happens later syncs its inputs.**
   ``EplbState.step``/``rearrange`` are the only normal-operation paths
   that mutate the placement table. Before recomputing it, they call
   ``_allreduce_list`` over the EP group to make the load tensors
   identical on every rank, then feed those to the deterministic
   ``policy.rebalance_experts`` (sort-by-load, greedy assignment, no
   RNG). Identical input + deterministic algorithm = identical output.
   (Under FT NIXL EP the load all-reduce reroutes onto the FT-gloo
   survivors after a death; the rearrangement weight shuffle itself is
   suppressed while a peer is dead -- see ``EplbState._ep_all_reduce``.)

3. **Recovery broadcasts identical args, then runs deterministic ops.**
   The engine calls ``self.collective_rpc("recover_from_dead_peers",
   args=(newly_dead,))`` once; that worker method rebuilds the FT-gloo
   survivor group and then runs ``eplb_redistribute_for_dead_peers``.
   ``collective_rpc`` delivers the same ``newly_dead`` tuple to every
   worker. Each worker then runs the three deterministic placement-table
   primitives -- ``mark_dead_columns_inplace``,
   ``reassign_missing_experts_inplace``, and
   ``rebuild_derived_maps_inplace`` -- against its (still-in-sync)
   placement table:

   * ``mark_dead_columns_inplace``: for each ``ep_rank`` in
     ``dead_ep_ranks``, write ``-1`` to the contiguous slice. No order
     dependence beyond the (stable) integer iteration over a set.
   * ``reassign_missing_experts_inplace``: iterates layer-by-layer, then
     ``phys_idx`` from 0 upward to build ``replica_count``. The donor
     candidate list is sorted by ``(redundancy, phys_idx)`` -- stable
     tie-break. The ``missing`` set is ``sorted(...)`` before
     assignment. Donor counts are decremented in lock-step. No RNG.
   * ``rebuild_derived_maps_inplace``: nested ``layer_idx`` / ``phys_idx``
     loops with the same iteration order on every rank.

   Same input + same args + deterministic algorithms = same output on
   every rank.

Together these mean ``eplb_state.physical_to_logical_map`` (and the
derived ``logical_to_physical_map`` / ``logical_replica_count``) stay
**bit-identical** across surviving ranks after recovery, with no
explicit cross-rank handshake during the recovery itself.

Note that the local ``expert_load_pass`` / ``expert_load_window``
tensors are explicitly **not** in sync across ranks during normal
operation -- each rank only counts the tokens it actually processed.
That divergence is normalized by ``_allreduce_list`` inside
``rearrange``. The recovery path does not read these tensors, so their
divergence is irrelevant here.

If you suspect sync has broken: the
``Worker.eplb_redistribute_for_dead_peers`` orchestration in
``vllm/v1/worker/gpu_worker.py`` logs a SHA-1 hash of
``physical_to_logical_map`` after every redistribute. A mismatch
across survivors is a smoking gun for an in-sync invariant violation
in one of the three places above.
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
            raise ValueError(
                f"FT EP: ep_rank={ep_rank} is out of bounds for "
                f"ep_world_size={ep_world_size} (num_physical="
                f"{num_physical}, num_local_experts={num_local_experts}). "
                "The caller passed an invalid dead-EP-rank -- this is a "
                "programming error, not a recoverable runtime state."
            )
        start = ep_rank * num_local_experts
        end = start + num_local_experts
        physical_to_logical_map[:, start:end] = -1


def reassign_missing_experts_inplace(
    physical_to_logical_map: torch.Tensor,
    num_logical: int,
) -> set[tuple[int, int]]:
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
    ``eplb_model_state.logical_replica_count.shape[1]``, which always
    carries one column per logical expert regardless of how many
    replicas survive) rather than inferred from
    ``physical_to_logical_map`` (the input tensor this function reads
    and rewrites). Inferring from ``physical_to_logical_map`` would be
    wrong here because that tensor only encodes logical ids that
    currently occupy at least one physical slot -- a logical id whose
    last physical replica was just zeroed out by
    ``mark_dead_columns_inplace`` leaves no trace in the tensor and
    would be silently treated as "doesn't exist" instead of "missing."
    That logical expert is precisely the one this function is supposed
    to find and reassign, so the cardinality has to come from outside
    the tensor.

    Returns the set of ``(layer_idx, logical_id)`` pairs that were
    reassigned. Each pair indicates a slot whose placement-table entry
    now points at ``logical_id`` but whose GPU weight buffer still
    holds the donor expert's weights -- the caller must reload those
    weights from disk (e.g. via
    :func:`vllm.distributed.elastic_ep.eplb_reload.reload_experts_from_disk`)
    before the model produces correct output for that expert. Empty set
    means no reassignment occurred (every logical expert already had at
    least one surviving replica in every layer).

    Raises ``RuntimeError`` when redundancy is insufficient to cover
    every missing expert -- the caller should detect this and either
    scale down or extend redundancy.

    Adapted from PR #38862's ``ElasticEPScalingExecutor.reassign_missing_experts``.
    """
    if physical_to_logical_map.ndim != 2:
        raise ValueError(
            f"physical_to_logical_map must be 2D; got shape "
            f"{tuple(physical_to_logical_map.shape)}"
        )

    num_layers, num_physical = physical_to_logical_map.shape
    all_logical = set(range(num_logical))
    reassignments: set[tuple[int, int]] = set()

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
                    reassignments.add((layer_idx, logical_id))
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    f"reassign_missing_experts_inplace: layer "
                    f"{layer_idx}: failed to place logical expert "
                    f"{logical_id} despite candidate slots remaining."
                )

    return reassignments


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
