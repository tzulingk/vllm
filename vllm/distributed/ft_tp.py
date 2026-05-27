# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault-tolerant TP all_reduce via FT NCCL.

Wraps ``ft_collective.FTProcessGroup`` (torch.distributed backend "ft_nccl")
to provide a TP all_reduce that survives a TP-sibling death.  The underlying
NCCL communicator is never shrunk -- world size stays N forever -- so
captured CUDA graphs and torch.compile artifacts remain valid even after a
peer dies.  FT NCCL's LSA device kernel masks dead slots in-place and signals
timeout via ``FT_TIMEOUT``; the caller polls ``get_result_mask()`` to learn
which ranks responded.

Enabled by ``VLLM_FT_TP_NCCL=1``.  Use only with an FT-capable EP backend
(currently ``--all2all-backend nixl_ep``) where TP-sibling death is an
expected scenario.

KNOWN LIMITATIONS
-----------------
1. **fp32 only at the kernel.** FT NCCL's ``FaultTolerantLsaAllReduceKernel``
   template is only instantiated for ``float`` today.  Inference activations
   in every production LLM (DeepSeek-V2/V3/V4, Llama 3, Mixtral) are bf16,
   so this wrapper does a bf16 -> fp32 -> bf16 cast pair around every FT
   collective.  That doubles the on-wire payload and adds two cast kernels
   per TP all_reduce.  Remove the cast once NCCL ships
   ``__nv_bfloat16`` / ``__half`` instantiations and the eligibility check
   in ``FTProcessGroup._is_ft_eligible`` is widened.

2. **Not torch.compile-captured.** The FT path is a Python call, not a
   registered custom op like ``vllm::all_reduce``.  When
   ``VLLM_FT_TP_NCCL=1``, every step containing a TP all_reduce falls off
   the captured graph and runs eager.  Production fix: register a
   ``vllm::ft_all_reduce`` custom op that mirrors the eager path.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from ft_collective.ft_process_group import FTProcessGroup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_FT_TP: FtTpProcessGroup | None = None


def is_ft_tp_enabled() -> bool:
    """Return True iff ``VLLM_FT_TP_NCCL=1`` is set in the environment."""
    return os.environ.get("VLLM_FT_TP_NCCL", "0") == "1"


def get_ft_tp() -> FtTpProcessGroup | None:
    """Return the per-process FT TP process group, or ``None`` if not initialized."""
    return _FT_TP


def init_ft_tp(
    world_size: int,
    tp_size: int,
    my_rank: int,
    max_numel: int = 1_048_576,
) -> FtTpProcessGroup | None:
    """Initialize the FT NCCL TP process group on this worker.

    COLLECTIVE: every rank in the default group must call this with the same
    arguments.  Internally enumerates all TP slices using vLLM's standard
    consecutive-rank layout and calls ``dist.new_group`` for each so the
    backend factory bootstraps a sub-store per TP group.

    Idempotent: subsequent calls return the existing singleton.

    Args:
        world_size: Total number of ranks in the default group.
        tp_size: TP world size; must divide ``world_size``.  When ``tp_size``
            is 1 there are no TP collectives to fault-tolerate, so this
            returns ``None`` early.
        my_rank: This process's global rank.
        max_numel: Maximum float32 element count per rank for the
            pre-allocated symmetric scratch buffer.  Larger TP all_reduces
            fall back to non-FT NCCL.  Default: 1 048 576 (4 MiB/rank).

    Returns:
        The :class:`FtTpProcessGroup` for this process, or ``None`` when the
        feature flag is off or ``tp_size == 1``.
    """
    global _FT_TP
    if _FT_TP is not None:
        return _FT_TP

    if not is_ft_tp_enabled():
        return None

    if tp_size <= 1:
        logger.info(
            "FT TP: tp_size=%d so there are no TP collectives to fault-tolerate; "
            "VLLM_FT_TP_NCCL is a no-op.",
            tp_size,
        )
        return None

    if world_size % tp_size != 0:
        raise ValueError(
            f"init_ft_tp: world_size {world_size} must be divisible by "
            f"tp_size {tp_size}"
        )

    # Import here so non-FT deployments don't pay the import cost.  Side-effect
    # of the import: registers the "ft_nccl" backend with torch.distributed.
    import ft_collective  # noqa: F401
    from ft_collective.ft_process_group import get_ft_process_group

    # Enumerate all TP slices using vLLM's consecutive-rank layout
    # (see vllm/distributed/parallel_state.py:1581-1593, where
    # all_ranks.reshape(-1, ..., tp_size).view(-1, tp_size).unbind(0)
    # produces consecutive runs).  Special layouts under enable_elastic_ep
    # need to mirror local_all_ranks instead -- not yet supported here.
    group_ranks = [list(range(i, i + tp_size)) for i in range(0, world_size, tp_size)]

    my_local_rank: int | None = None
    for tp_ranks in group_ranks:
        # dist.new_group is collective across the default group: every rank
        # participates in every iteration so the backend factory can
        # bootstrap a sub-store, but only ranks listed in ``tp_ranks`` get
        # a usable handle back.
        dist.new_group(ranks=tp_ranks, backend="ft_nccl")
        if my_rank in tp_ranks:
            my_local_rank = tp_ranks.index(my_rank)

    if my_local_rank is None:
        # Shouldn't happen: every rank in [0, world_size) belongs to exactly
        # one TP slice under the consecutive layout above.
        raise RuntimeError(
            f"init_ft_tp: my_rank {my_rank} not in any TP slice of "
            f"world_size {world_size} with tp_size {tp_size}"
        )

    # The factory registers each FTProcessGroup instance in a per-process
    # registry keyed by local rank within its subgroup.  The C-level
    # ProcessGroup wrapper from ``dist.new_group`` can't access Python-only
    # methods (``empty()``, ``get_result_mask()``), hence the lookup here.
    ft_pg = get_ft_process_group(rank=my_local_rank)
    if ft_pg is None:
        raise RuntimeError(
            "init_ft_tp: dist.new_group(backend='ft_nccl') succeeded but "
            f"get_ft_process_group(rank={my_local_rank}) returned None. "
            "The ft_collective backend factory may not have registered the "
            "FTProcessGroup instance for this rank."
        )

    _FT_TP = FtTpProcessGroup(ft_pg, max_numel=max_numel)
    logger.info(
        "FT TP: initialized FtTpProcessGroup (my_rank=%d local_rank=%d "
        "tp_size=%d max_numel=%d).",
        my_rank,
        my_local_rank,
        tp_size,
        max_numel,
    )
    return _FT_TP


class FtTpProcessGroup:
    """Vllm-side wrapper over ``ft_collective.FTProcessGroup``.

    Provides ``all_reduce`` with the bf16 <-> fp32 cast pair needed until
    FT NCCL gains bf16 kernel template instantiations.  Surfaces FT NCCL's
    per-collective result mask via ``last_active_mask`` so the death-detection
    plumbing can inspect it after each call.
    """

    def __init__(self, ft_pg: FTProcessGroup, *, max_numel: int) -> None:
        self._pg = ft_pg
        self._max_numel = max_numel

        # Symmetric scratch buffer (collective allocation -- all TP ranks must
        # have called ft_pg.empty() simultaneously when the wrapper was
        # constructed).  Used to stage bf16 -> fp32 casts.  Sized for the
        # largest expected TP all_reduce.  Allocated once at init; reused.
        self._scratch_fp32: torch.Tensor = ft_pg.empty(max_numel, dtype=torch.float32)

        # Local mirror of which TP slots are alive.  Refreshed from the FT
        # kernel's result mask after every collective.
        self._last_active_mask: list[bool] = [True] * ft_pg.size()

    @property
    def size(self) -> int:
        return self._pg.size()

    @property
    def last_active_mask(self) -> list[bool]:
        """Return the active mask from the most recent FT collective."""
        return list(self._last_active_mask)

    @torch.compiler.disable
    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """In-place fault-tolerant all_reduce; returns the same tensor.

        On a TP-sibling timeout, the FT kernel masks the dead slot, the
        surviving ranks compute the partial sum, and this function returns
        without hanging.  ``self.last_active_mask`` reflects which ranks
        responded.

        Falls back to ``dist.all_reduce`` (NOT fault-tolerant -- will hang on
        a dead peer) for tensors larger than ``max_numel``.  Log a warning
        once so misconfiguration is visible.
        """
        if t.numel() > self._max_numel:
            logger.warning(
                "FtTpProcessGroup.all_reduce: tensor numel %d exceeds "
                "max_numel %d.  Falling back to non-FT NCCL all_reduce -- "
                "this WILL hang if a TP peer is dead.  Increase "
                "max_numel in init_ft_tp() to cover the largest TP "
                "all_reduce in the model.",
                t.numel(),
                self._max_numel,
            )
            dist.all_reduce(t, group=self._pg)
            return t

        view = self._scratch_fp32[: t.numel()].view(t.shape)
        # bf16/fp16 -> fp32 cast on the activation.  .copy_() fuses the cast
        # into one kernel.  Required until FT NCCL ships bf16/fp16 kernel
        # instantiations -- see module docstring.
        view.copy_(t)
        work = self._pg.allreduce([view])
        work.wait()
        # fp32 -> original dtype store back into the caller's tensor.
        t.copy_(view)

        # Refresh local view of who responded.  get_result_mask() syncs the
        # FT stream and self-heals the input mask if a timeout occurred.
        self._last_active_mask = self._pg.get_result_mask()
        return t
