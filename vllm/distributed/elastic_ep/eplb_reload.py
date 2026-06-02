# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed weight reload for experts displaced by peer death.

After
:func:`vllm.distributed.elastic_ep.eplb_redistribute.reassign_missing_experts_inplace`
rewrites the placement table to point a physical slot at a logical
expert that had no surviving replica anywhere, the *table* says the
slot now hosts that expert but the slot's GPU weight buffer still holds
the donor expert's weights -- the table lies, and the MoE router will
route tokens to the wrong weights.

This module closes that gap by reading the affected expert tensors from
the HF safetensors checkpoint on disk and writing them into the slot.
We reuse the same two pieces of vLLM's loader path used at startup:

* :class:`~vllm.model_executor.model_loader.default_loader.DefaultModelLoader`
  to enumerate every tensor in the checkpoint. We explicitly **disable**
  its iterator-level expert filter (``loader.local_expert_ids = None``)
  because that filter uses ``compute_local_expert_ids(...)`` which
  doesn't know about the placement-table mutation we just performed.
* The model's own ``load_weights(weights)`` method to route each
  ``(name, tensor)`` pair through ``FusedMoE.weight_loader``. The
  loader consults the *current* placement table via
  ``_map_global_expert_id_to_local_expert_id`` (which reads the
  just-rebuilt ``_expert_map``) to decide which (if any) local
  physical slot to write into. A Python-level filter narrows the
  iterator output to the reassigned ``(layer, logical)`` pairs so we
  don't pay disk I/O for experts that haven't moved.

Cost: one safetensors mmap pass over the checkpoint, but only the
selected expert tensors are actually copied to GPU. For DeepSeek-V2-Lite
a single expert is a few hundred KB; reloading ~half the experts across
all layers is on the order of seconds. PR #38862 measured 3.81s
end-to-end for the disk-reload case on a comparable model.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_layer_expert(name: str) -> tuple[int, int] | None:
    """Extract ``(layer_idx, logical_expert_id)`` from a tensor name.

    Returns ``None`` for tensors that aren't expert-scoped (attention
    weights, layer norms, the embedding etc.). Caller filters those
    out before yielding to ``model.load_weights``.

    The HF convention used by DeepSeek-V2 and most MoE models is
    ``model.layers.{L}.mlp.experts.{E}.{shard}.weight`` where ``L`` is
    the transformer layer index and ``E`` is the logical expert id.
    """
    parts = name.split(".")
    try:
        layer_pos = parts.index("layers")
        expert_pos = parts.index("experts")
    except ValueError:
        return None
    try:
        layer_idx = int(parts[layer_pos + 1])
        expert_id = int(parts[expert_pos + 1])
    except (IndexError, ValueError):
        return None
    return layer_idx, expert_id


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    reload_set: set[tuple[int, int]],
) -> int:
    """Reload specific ``(layer, logical_expert)`` weights from disk.

    Args:
        model: the worker's loaded model. Must expose ``load_weights``
            (every vLLM model does).
        vllm_config: the worker's :class:`VllmConfig`, used to resolve
            the HF safetensors folder.
        reload_set: ``{(layer_idx, logical_expert_id), ...}`` --
            typically the return value of
            :func:`reassign_missing_experts_inplace`. Each pair names
            a logical expert whose weights this rank should overwrite
            on disk-read.

    Returns:
        Number of parameter names actually loaded
        (``len(model.load_weights(...))``). Useful for sanity-checking
        that the filter matched at least one tensor per requested
        ``(layer, expert)`` triple.

    The reload is best-effort: if any (layer, expert) tensor is missing
    from the checkpoint we log a warning but do not raise -- the
    surrounding redistribute should still succeed for the experts we
    *did* find. The MoE layer's weight_loader is responsible for
    skipping experts that aren't locally owned (returns
    ``success=False``), so calling this on every surviving rank with
    the same ``reload_set`` is safe and idempotent.
    """
    if not reload_set:
        return 0

    # Go through the same DefaultModelLoader path used at startup, but
    # with its iterator-level expert filter disabled. The iterator-level
    # filter (should_skip_weight in ep_weight_filter.py) decides yield-vs-
    # skip based on the static compute_local_expert_ids set computed at
    # __init__. After mark_dead_columns + reassign_missing_experts, that
    # set no longer matches which experts this rank actually hosts -- the
    # placement table is the authority now, not the static computation.
    # Setting loader.local_expert_ids = None hits the early-return branch
    # in should_skip_weight (returns False, "never skip"), so every
    # expert tensor flows through.
    #
    # Per-rank routing is then handled by FusedMoE.weight_loader, which
    # consults the just-rebuilt _expert_map via
    # _map_global_expert_id_to_local_expert_id. If a tensor's logical
    # expert isn't local to this rank after reassignment, weight_loader
    # returns False and model.load_weights moves on; no harm done.
    from vllm.model_executor.model_loader.default_loader import (
        DefaultModelLoader,
    )

    loader = DefaultModelLoader(vllm_config.load_config)
    loader.local_expert_ids = None

    all_weights = loader.get_all_weights(vllm_config.model_config, model)

    def filtered_iter() -> Generator[tuple[str, torch.Tensor], None, None]:
        # Python-level filter narrows to the reassigned (layer, logical)
        # pairs we want to overwrite. The weight_loader would skip non-
        # local tensors anyway, but this avoids re-loading expert
        # weights that haven't moved (saves disk -> GPU bandwidth).
        # Non-expert tensors (layernorms, embeddings, etc.) are dropped
        # too -- they don't need reloading.
        for name, tensor in all_weights:
            parsed = _parse_layer_expert(name)
            if parsed is None:
                continue
            if parsed in reload_set:
                yield name, tensor

    logger.info(
        "FT EP: reloading %d (layer, logical-expert) pair(s) via "
        "DefaultModelLoader (iterator filter disabled; per-rank "
        "routing via FusedMoE.weight_loader + rebuilt _expert_map).",
        len(reload_set),
    )

    loaded = model.load_weights(filtered_iter())
    if not loaded:
        logger.warning(
            "FT EP: disk reload requested for %d (layer, expert) pair(s) "
            "but model.load_weights consumed 0 tensors. The HF checkpoint "
            "may not contain expert-scoped tensors, or the local rank "
            "owned none of the reassigned slots. Affected pairs: %s",
            len(reload_set),
            sorted(reload_set)[:8],
        )
    return len(loaded)


def expected_tensors_per_expert(checkpoint_index_path: str | None = None) -> int:
    """Best-effort hint: how many tensors are stored per (layer, expert).

    For DeepSeek-style MoE this is 3 (gate_proj, up_proj, down_proj),
    or sometimes 2 if gate+up are pre-fused. Used only for log lines
    and basic sanity in tests; do not rely on this for correctness.

    Returns ``3`` as a conservative default when no index is supplied.
    """
    if not checkpoint_index_path or not os.path.exists(checkpoint_index_path):
        return 3
    try:
        with open(checkpoint_index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        per_expert: dict[tuple[int, int], int] = {}
        for name in weight_map:
            parsed = _parse_layer_expert(name)
            if parsed is None:
                continue
            per_expert[parsed] = per_expert.get(parsed, 0) + 1
        if not per_expert:
            return 3
        # Mode of the counts.
        values = list(per_expert.values())
        return max(set(values), key=values.count)
    except Exception:
        return 3
