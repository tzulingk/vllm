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
We reuse two pieces of vLLM's existing loader path:

* :func:`vllm.model_executor.model_loader.weight_utils.safetensors_weights_iterator`
  to read only the experts we need (the ``local_expert_ids`` kwarg
  skips reads for experts this rank doesn't care about).
* The model's own ``load_weights(weights)`` method to route each
  ``(name, tensor)`` pair through ``FusedMoE.weight_loader``. The
  loader consults the *current* placement table via
  ``_map_global_expert_id_to_local_expert_id`` to decide which (if
  any) local physical slot to write into. Because we've already updated
  the table, this naturally routes each reassigned slot to the right
  rank.

Cost: roughly one safetensors mmap read per (layer, expert) pair, plus
a small Host→GPU copy per expert. For DeepSeek-V2-Lite, a single expert
is a few hundred KB; reloading ~half the model's experts across all
layers is on the order of seconds. PR #38862 measured 3.81s end-to-end
for the disk-reload case on a comparable model.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Generator

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    safetensors_weights_iterator,
)

logger = init_logger(__name__)


def _resolve_hf_weight_files(vllm_config: VllmConfig) -> list[str]:
    """Locate the safetensors files for the configured model.

    Returns absolute paths. Downloads from HF if the cache is empty
    (rare on the recovery path -- normally the cache is populated by
    the initial model load at engine startup).
    """
    model_config = vllm_config.model_config
    load_config = vllm_config.load_config
    model_name_or_path = model_config.model
    revision = model_config.revision

    if os.path.isdir(model_name_or_path):
        hf_folder = model_name_or_path
    else:
        # Pull from HF (no-op if cache already has it; mirrors the
        # path DefaultModelLoader uses at startup).
        hf_folder = download_weights_from_hf(
            model_name_or_path,
            cache_dir=load_config.download_dir,
            allow_patterns=["*.safetensors", "*.bin", "*.pt"],
            revision=revision,
            ignore_patterns=getattr(load_config, "ignore_patterns", None) or [],
        )

    files: list[str] = []
    for pattern in ("*.safetensors",):
        files.extend(glob.glob(os.path.join(hf_folder, pattern)))
    if not files:
        raise RuntimeError(
            "FT EP disk reload requires safetensors weights; none found in "
            f"{hf_folder}. Reload from .bin/.pt is not supported."
        )

    # Dedupe against an index file if one exists, then drop optimizer
    # state etc. Mirrors DefaultModelLoader._prepare_weights.
    index_file = "model.safetensors.index.json"
    if os.path.exists(os.path.join(hf_folder, index_file)):
        files = filter_duplicate_safetensors_files(files, hf_folder, index_file)
    files = filter_files_not_needed_for_inference(files)
    return files


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

    hf_weights_files = _resolve_hf_weight_files(vllm_config)
    wanted_logical_ids = {lid for _, lid in reload_set}

    def filtered_iter() -> Generator[tuple[str, torch.Tensor], None, None]:
        # local_expert_ids prunes IO inside the iterator -- it only
        # reads tensors whose expert id is in this set. Pass the
        # logical ids we want; layer filtering happens here in the
        # outer loop.
        for name, tensor in safetensors_weights_iterator(
            hf_weights_files,
            use_tqdm_on_load=False,
            local_expert_ids=wanted_logical_ids,
        ):
            parsed = _parse_layer_expert(name)
            if parsed is None:
                # Non-expert tensor (e.g. layernorm) -- iterator may
                # yield those when local_expert_ids is set. Skip; we
                # only want expert reloads.
                continue
            if parsed in reload_set:
                yield name, tensor

    logger.info(
        "FT EP: reloading %d (layer, logical-expert) pair(s) from %d "
        "safetensors file(s).",
        len(reload_set),
        len(hf_weights_files),
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
