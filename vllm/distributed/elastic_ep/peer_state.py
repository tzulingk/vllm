# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-layer mask state for FT NIXL EP / elastic EP.

Modeled on SGLang's ``ElasticEPState``
(``python/sglang/srt/elastic_ep/elastic_ep.py``): the engine -- not the
kernel -- owns the set of currently-active EP peers. The kernel-side
primitive (``NixlEPAll2AllManager.query_mask()``) is read into this
state each step; ``is_active_equal_last()`` is the diff that triggers
the request-abort path; ``active_ranks_cpu`` is what an FT-gloo wrapper
consults to scope a collective to surviving ranks.

Convention: ``active_ranks[i] == 1`` means rank ``i`` is alive. The NIXL
EP buffer reports the inverse (``1`` = masked / dead); :func:`apply_kernel_mask`
inverts when ingesting.

This module deliberately stops at "what is the current alive set." It
does **not** trigger scale-down, EPLB reshuffle, or request abort -- those
are layered on top in subsequent commits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PeerActiveState:
    """Snapshot of which EP peers are currently active.

    The mask is **EP-indexed** (one bit per EP slot = one bit per GPU);
    ``active_ranks.numel() == dp_size * tp_size``. DP-level views (for
    cross-DP collectives + the AsyncLLM dispatcher) are derived from
    the EP mask via :meth:`dp_active_mask` / :meth:`dp_dead_ranks`.

    Why EP-indexed: the failure unit is a single GPU. For TP > 1, a
    DP rank may have some TP siblings dead and some alive; the
    surviving siblings must keep stepping so the cross-DP EP all-to-all
    sees the DP rank as present. Tracking aliveness per EP slot lets
    each consumer fold the right way:
      - Cross-DP collectives (wave-sync, KV-mem-sync): DP rank ``d`` is
        alive if **any** of its TP siblings is alive (OR-reduce).
      - AsyncLLM request routing: DP rank ``d`` is dead only if **all**
        of its TP siblings are dead (AND-reduce). New requests go to
        DP ranks that still have at least one live sibling.

    For TP=1 the EP index = DP index and the derivations are
    identities.

    Attributes:
        active_ranks: ``[ep_size]`` int32 tensor; ``1`` = alive, ``0`` =
            dead.
        last_active_ranks: Snapshot of ``active_ranks`` at the previous
            engine step. Compared via :meth:`is_active_equal_last` to
            detect per-step changes.
        active_ranks_cpu: CPU mirror of ``active_ranks``.
        tp_size: Number of TP workers per DP rank. ``ep_size`` /
            ``tp_size`` = DP size.
    """

    active_ranks: torch.Tensor
    last_active_ranks: torch.Tensor
    active_ranks_cpu: torch.Tensor
    tp_size: int = 1

    @property
    def ep_size(self) -> int:
        return int(self.active_ranks.numel())

    @property
    def dp_size(self) -> int:
        return self.ep_size // self.tp_size

    def is_active_equal_last(self) -> bool:
        return torch.equal(self.active_ranks, self.last_active_ranks)

    def sync_active_to_cpu(self) -> None:
        self.active_ranks_cpu = self.active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self) -> None:
        self.last_active_ranks = self.active_ranks.clone()

    def newly_dead_peers(self) -> list[int]:
        """EP ranks that flipped from alive (last step) to dead (now).

        Reads CPU mirrors only; safe to call from any thread.
        """
        was = self.last_active_ranks.detach().cpu()
        now = self.active_ranks_cpu
        diff = ((was == 1) & (now == 0)).nonzero(as_tuple=False).flatten()
        return diff.tolist()

    def alive_ranks(self) -> list[int]:
        """EP ranks currently alive (CPU read)."""
        return self.active_ranks_cpu.nonzero(as_tuple=False).flatten().tolist()

    def dp_active_mask(self) -> list[int]:
        """DP-level active mask, OR-reduced across each DP rank's TP siblings.

        Length ``dp_size``; entry ``d`` is ``1`` if any of DP rank
        ``d``'s TP siblings (EP slots ``d*tp .. (d+1)*tp - 1``) is alive.
        Used by :func:`vllm.distributed.elastic_ep.ft_gloo.ft_or_raw_all_reduce`
        so cross-DP collectives stay correct when one TP sibling in a
        DP rank dies but others survive.
        """
        cpu = self.active_ranks_cpu.tolist()
        tp = self.tp_size
        dp = self.dp_size
        return [1 if any(cpu[d * tp + t] for t in range(tp)) else 0 for d in range(dp)]

    def dp_dead_ranks(self) -> set[int]:
        """DP ranks where every TP sibling is dead (AND-reduce).

        Used by the AsyncLLM dispatcher to decide which DP ranks to
        skip for new requests. A DP rank with even one live TP sibling
        is still routable -- that sibling's worker can serve the
        request (and its EP all-to-all peers will reach it via the
        NIXL mask).
        """
        cpu = self.active_ranks_cpu.tolist()
        tp = self.tp_size
        dp = self.dp_size
        return {d for d in range(dp) if not any(cpu[d * tp + t] for t in range(tp))}

    def reset(self) -> None:
        """Mark every rank alive again. Use after a recovery / re-include."""
        self.active_ranks.fill_(1)
        self.snapshot_active_to_last()
        self.sync_active_to_cpu()


def apply_kernel_mask(state: PeerActiveState, kernel_mask: torch.Tensor) -> None:
    """Update ``state.active_ranks`` from a NIXL EP kernel mask.

    Args:
        state: the :class:`PeerActiveState` to update in place.
        kernel_mask: ``[ep_size]`` int tensor as returned by
            ``NixlEPAll2AllManager.query_mask()`` --
            ``1`` = masked (dead), ``0`` = active. Inverted here to match
            the ``1 = alive`` convention used internally.

    Caller is responsible for calling :meth:`PeerActiveState.snapshot_active_to_last`
    *after* the per-step consumers have inspected the diff.
    """
    if kernel_mask.shape != state.active_ranks.shape:
        raise ValueError(
            f"kernel_mask shape {tuple(kernel_mask.shape)} does not match "
            f"PeerActiveState.active_ranks shape "
            f"{tuple(state.active_ranks.shape)}"
        )
    new_active = (kernel_mask == 0).to(state.active_ranks.dtype)
    state.active_ranks.copy_(new_active.to(state.active_ranks.device))
    state.sync_active_to_cpu()


class PeerActiveStateManager:
    """Singleton holder for the per-process :class:`PeerActiveState`.

    Each engine-core / worker process keeps one ``PeerActiveState`` and
    refers to it through this manager. Distinct from the existing
    ``ElasticEPScalingState`` (in ``elastic_state.py``), which tracks
    explicit scale-up / scale-down progress; ``PeerActiveStateManager``
    tracks involuntary peer death without any topology change.
    """

    _instance: PeerActiveState | None = None

    @classmethod
    def instance(cls) -> PeerActiveState | None:
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._instance is not None

    @classmethod
    def init(
        cls,
        ep_size: int,
        *,
        tp_size: int = 1,
        device: torch.device | None = None,
    ) -> PeerActiveState:
        """Idempotent initializer. Subsequent calls return the existing instance.

        Args:
            ep_size: total EP world size = ``dp_size * tp_size``.
                For TP=1 callers may pass ``dp_size`` directly and the
                DP-level derivations are identities.
            tp_size: TP workers per DP rank. Defaults to ``1`` (the demo
                topology); for TP>1 ``ep_size`` must be a multiple of
                ``tp_size``.
        """
        if cls._instance is not None:
            existing_ep = cls._instance.active_ranks.numel()
            if existing_ep != ep_size or cls._instance.tp_size != tp_size:
                logger.warning(
                    "PeerActiveStateManager.init(ep_size=%d, tp_size=%d) called "
                    "but an instance with ep_size=%d, tp_size=%d already exists; "
                    "returning existing.",
                    ep_size,
                    tp_size,
                    existing_ep,
                    cls._instance.tp_size,
                )
            return cls._instance

        if ep_size % tp_size != 0:
            raise ValueError(
                f"PeerActiveStateManager.init: ep_size {ep_size} must be a "
                f"multiple of tp_size {tp_size}"
            )

        dev = device if device is not None else torch.device("cpu")
        active = torch.ones(ep_size, dtype=torch.int32, device=dev)
        cls._instance = PeerActiveState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.detach().cpu().clone(),
            tp_size=tp_size,
        )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton. Intended for test teardown."""
        cls._instance = None
