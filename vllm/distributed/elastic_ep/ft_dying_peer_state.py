# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State machine for fault-tolerant handling of a peer DP rank death.

Modeled on
:class:`vllm.distributed.elastic_ep.elastic_state.ElasticEPScalingState`:
each surviving engine drives its own copy of the state machine through
its ``run_busy_loop``, advancing at most one state per loop tick.
Cross-DP synchronization is two-layered: (1) a TCPStore counter that
each survivor increments once on arrival, used as a non-blocking
"all peers received the notify" check; (2) a survivors-only barrier
via :class:`FaultTolerantGlooGroup` all_reduce -- this rebuilds the
gloo sub-group to exclude the dead rank (which EPLB collectives in
REDISTRIBUTE need anyway) and serves as the actual cross-rank
synchronization, with a 5-second first-attempt timeout. Engines that
arrive at the barrier early fall back to a normal forward pass and
retry on the next tick, so peers slightly behind in their loops can
catch up without anyone being silent on NIXL EP.

Why FT-gloo instead of ``torch.distributed.barrier(dp_group)``: the
elastic-EP ``_staged_barrier`` uses TCPStore-poll + dp_group barrier,
where the dp_group barrier ensures the leader cleanup of arrival
keys can't race with slow joiners. We can't use dp_group directly
because it still includes the dead rank (would hang forever).
FT-gloo gives the same all-ranks-here semantics on a survivor-only
sub-group.

This shape exists because of DYN-3121: if a single fast-path notify
handler runs the slow ``eplb_redistribute_for_dead_peers`` inline,
the engine that processes the notification first starts that
multi-second disk reload while its peers are still doing forward
passes. The disk-bound engine's NIXL EP dispatch kernel never
launches, peers' dispatches time out on it, and the kernel mask
flips for an alive peer. The barrier serializes the "all engines
stop NIXL EP at the same step boundary" transition, eliminating
that cascade window.
"""

import enum
import time
import weakref
from typing import TYPE_CHECKING

import torch
from torch.distributed import ReduceOp, Store

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc

logger = init_logger(__name__)


class DyingPeerEngineState(enum.IntEnum):
    """States for the per-engine FT dying-peer state machine."""

    ENTER_BARRIER = 0
    REDISTRIBUTE = 1
    COMPLETE = 2


# Match elastic-EP's _staged_barrier semantics: first time the
# barrier is attempted, 5s wall-clock timeout (some engines may not
# have arrived yet); on subsequent attempts (after the sync_key is
# set), no timeout (wait indefinitely, because every engine will
# eventually arrive in normal operation).
_BARRIER_FIRST_ATTEMPT_TIMEOUT_S = 5.0


class FtDyingPeerState:
    """Per-engine state machine for handling one peer DP rank's death.

    Lifecycle:
      ENTER_BARRIER -> REDISTRIBUTE -> COMPLETE

    Created when ``notify_engine_death`` is processed on a surviving
    engine; the engine clears its reference when ``is_complete()``
    returns True.
    """

    def __init__(
        self,
        dead_dp_rank: int,
        engine_core: "DPEngineCoreProc",
    ):
        self.dead_dp_rank = dead_dp_rank
        self.engine_core_ref = weakref.ref(engine_core)
        self.dp_store: Store = engine_core.dp_store
        self.dp_rank: int = engine_core.dp_rank
        self.dp_world_size: int = engine_core.dp_group.size()
        # Number of survivors (excluding the dead rank).
        self.survivor_count: int = self.dp_world_size - 1
        # Lowest-numbered surviving DP rank, used for "leader" cleanup
        # operations on the TCPStore (analogous to dp_rank==0 in
        # elastic-EP but skipping the dead rank).
        self.leader_rank: int = 0 if dead_dp_rank != 0 else 1
        # Shared key suffix across surviving engines: same dead rank
        # implies same barrier. The per-rank suffix is enough -- a given
        # dead rank cannot die twice in the same process lifetime (the
        # rank is gone), so we don't need a per-event generation tag.
        # Earlier attempts to include time.monotonic_ns() here ARE WHY
        # each engine ran a barrier on its OWN key and they never met.
        self._key_suffix: str = f"dead_{dead_dp_rank}"
        self._barrier_count_key: str = f"ft_dying_peer_count_{self._key_suffix}"
        self._barrier_name: str = f"ft_dying_peer_{self._key_suffix}"
        # Whether this engine has already incremented the counter once.
        self._announced_arrival: bool = False
        # Current state of the machine.
        self.state: DyingPeerEngineState = DyingPeerEngineState.ENTER_BARRIER
        # High-precision wall-clock for cross-engine correlation. All log
        # lines below include time.time() in floating-point epoch seconds
        # so we can match "DP 0 reached barrier at t=X" with "DP 2 reached
        # barrier at t=Y" across separate actor log files.
        self._t_create: float = time.time()
        logger.warning(
            "FT EP: FtDyingPeerState created for dead DP rank %d "
            "(survivor count=%d, leader_rank=%d, key=%s, dp_rank=%d) "
            "wall_t=%.6f",
            dead_dp_rank,
            self.survivor_count,
            self.leader_rank,
            self._key_suffix,
            self.dp_rank,
            self._t_create,
        )

    @property
    def engine_core(self) -> "DPEngineCoreProc":
        ec = self.engine_core_ref()
        if ec is None:
            raise RuntimeError("Engine core has been garbage collected")
        return ec

    def is_complete(self) -> bool:
        return self.state == DyingPeerEngineState.COMPLETE

    def progress(self) -> bool:
        """Advance the state machine at most one step.

        Returns ``True`` if state advanced (the caller continues to the
        forward pass), ``False`` if blocked waiting for peers (caller
        falls back to a normal forward pass and retries on the next
        tick). The ``False`` return is what makes the barrier
        "non-blocking" -- the engine never blocks its run loop while
        waiting for peers to arrive.
        """
        if self.state == DyingPeerEngineState.ENTER_BARRIER:
            return self._progress_enter_barrier()
        if self.state == DyingPeerEngineState.REDISTRIBUTE:
            self._progress_redistribute()
            return True
        # COMPLETE: caller should detach us; be safe and report progress.
        return True

    def _progress_enter_barrier(self) -> bool:
        """ENTER_BARRIER: wait for all survivors to acknowledge the death.

        Each engine increments a TCPStore counter once on its first
        visit here, then polls until the counter reaches the survivor
        count. Once everyone has incremented (via the counter), call the
        staged barrier to synchronize at a precise wall-clock boundary,
        then advance.
        """
        if not self._announced_arrival:
            self.dp_store.add(self._barrier_count_key, 1)
            self._announced_arrival = True
            t_now = time.time()
            logger.warning(
                "FT EP: dying-peer barrier %s: dp_rank=%d announced "
                "arrival; waiting for %d survivors. wall_t=%.6f "
                "(t_since_create=%.3fs)",
                self._barrier_name,
                self.dp_rank,
                self.survivor_count,
                t_now,
                t_now - self._t_create,
            )
        try:
            arrived = int(self.dp_store.get(self._barrier_count_key))
        except (ValueError, KeyError):
            arrived = 0
        if arrived < self.survivor_count:
            # Not all peers here yet -- fall back to a forward pass.
            return False
        if not self._staged_barrier():
            # First-attempt timeout -- caller falls back; we'll retry
            # on the next tick with no-timeout barrier.
            return False
        if self.dp_rank == self.leader_rank:
            self.dp_store.delete_key(self._barrier_count_key)
        self.state = DyingPeerEngineState.REDISTRIBUTE
        t_now = time.time()
        logger.warning(
            "FT EP: dying-peer barrier %s passed; dp_rank=%d advancing to "
            "REDISTRIBUTE. wall_t=%.6f (t_since_create=%.3fs)",
            self._barrier_name,
            self.dp_rank,
            t_now,
            t_now - self._t_create,
        )
        return True

    def _staged_barrier(self) -> bool:
        """Survivors-only staged barrier via :class:`FaultTolerantGlooGroup`.

        Replaces the original "per-rank arrival key + leader cleanup"
        polling barrier, which had a race where the leader could
        ``delete_key`` arrival keys before a slow survivor observed
        them, deadlocking the slow survivor on its next poll (DYN-3121
        sub-issue, seen in the 2026-05-28 run where DP3 announced
        arrival last and then went silent forever).

        Mirrors elastic-EP's pattern of TCPStore-poll + collective
        barrier (``elastic_state.py``, line 215), but on a survivors-
        only sub-group instead of ``dp_group`` -- because ``dp_group``
        still includes the dead rank and ``torch.distributed.barrier``
        on it hangs.

        The all_reduce serves three purposes at once:
          1. Synchronization barrier (every survivor must arrive
             before any survivor can proceed).
          2. Rebuilds the FT-gloo sub-group to exclude the dead rank,
             which is the exact group that EPLB cross-DP collectives
             in REDISTRIBUTE need next.
          3. Implicitly fails if any "survivor" turns out to be also
             dead (the rebuild rendezvous times out), surfacing as
             ``valid=False`` for the caller to retry.

        Sync key pattern preserved: first attempt uses 5s timeout;
        if any rank times out the first attempt is marked sync_key
        on the TCPStore and the next attempt uses a long timeout
        (60s) to allow stragglers to arrive.
        """
        from vllm.distributed.elastic_ep.ft_gloo import DPFTGlooManager

        sync_key = f"{self._barrier_name}_sync"
        first_attempt = not self.dp_store.check([sync_key])
        timeout_ms = (
            int(_BARRIER_FIRST_ATTEMPT_TIMEOUT_S * 1000) if first_attempt else 60_000
        )

        ft = DPFTGlooManager.instance()
        if ft is None:
            # FT NIXL EP deployments always init DPFTGlooManager at engine
            # startup; if we got here without one, something is wrong with
            # the deployment. Surface it loudly rather than silently
            # falling through (which would re-introduce the cleanup race).
            raise RuntimeError(
                "FT EP: FtDyingPeerState requires DPFTGlooManager to be "
                "initialized for the survivors-only barrier."
            )

        # active_mask: 1 for surviving DP ranks, 0 for the dead rank.
        # FaultTolerantGlooGroup rebuilds the underlying gloo subgroup
        # if the active set differs from its last call -- on the first
        # FtDyingPeerState event this rebuild happens here (which is
        # exactly what we want for REDISTRIBUTE's downstream collectives).
        survivor_mask = [
            1 if r != self.dead_dp_rank else 0 for r in range(self.dp_world_size)
        ]
        # Dummy tensor: any all_reduce on the survivor sub-group serves
        # as a barrier; SUM of zeros stays zero so the result is unused.
        dummy = torch.zeros(1, dtype=torch.float32)
        _, valid = ft.all_reduce(
            dummy,
            op=ReduceOp.SUM,
            active_mask=survivor_mask,
            timeout_ms=timeout_ms,
        )
        if valid:
            # Leader clears the sync_key so the next FT event on a
            # different dead rank starts with a fresh first-attempt
            # timeout.
            if self.dp_rank == self.leader_rank and self.dp_store.check([sync_key]):
                self.dp_store.delete_key(sync_key)
            return True

        if first_attempt:
            # Mark sync_key so the next progress() tick uses the long
            # timeout; the caller (progress_enter_barrier) will fall
            # back to a normal forward pass meanwhile.
            self.dp_store.compare_set(sync_key, "", b"1")
            return False
        raise RuntimeError(
            f"FT EP: FT-gloo survivors-only barrier failed even on the "
            f"long-timeout retry for {self._barrier_name}; survivor set "
            f"{[r for r in range(self.dp_world_size) if r != self.dead_dp_rank]} "
            f"may be split."
        )

    def _progress_redistribute(self) -> None:
        """REDISTRIBUTE: update PeerActiveState and run the slow disk reload.

        Synchronous (blocks the engine's run loop for seconds of disk
        I/O). The cascade is avoided because every surviving engine
        enters this state at the same step boundary (enforced by the
        ENTER_BARRIER state's barrier), so they're all silent on NIXL
        EP for the same window. No engine is dispatching while another
        is silent.
        """
        from vllm.distributed.elastic_ep.peer_state import (
            PeerActiveStateManager,
        )
        from vllm.v1.request import RequestStatus

        ec = self.engine_core
        ec._confirmed_dead_dp_ranks.add(self.dead_dp_rank)

        # Abort any requests that are still queued -- they may have
        # been routed to us between notify_engine_death arriving and
        # the barrier passing. (The notify handler aborted the
        # already-running ones; anything that came in via zmq during
        # the barrier wait needs to be cleaned up here too.)
        running = list(getattr(ec.scheduler, "running", []))
        if running:
            ec.scheduler.finish_requests(
                [r.request_id for r in running],
                RequestStatus.FINISHED_ERROR,
            )

        state = PeerActiveStateManager.instance()
        if state is None:
            logger.warning(
                "FT EP: REDISTRIBUTE for dead DP %d skipped -- "
                "PeerActiveState not initialized on this engine.",
                self.dead_dp_rank,
            )
            self.state = DyingPeerEngineState.COMPLETE
            return

        tp_size = state.tp_size
        newly_dead_ep_slots = [self.dead_dp_rank * tp_size + t for t in range(tp_size)]
        flipped = False
        for ep_slot in newly_dead_ep_slots:
            if state.active_ranks[ep_slot].item() != 0:
                state.active_ranks[ep_slot] = 0
                flipped = True
        state.sync_active_to_cpu()
        t_redistribute_start = time.time()
        logger.warning(
            "FT EP: REDISTRIBUTE running for dead DP %d (EP slots %s, "
            "aborted %d in-flight req(s)). dp_rank=%d wall_t=%.6f "
            "(t_since_create=%.3fs)",
            self.dead_dp_rank,
            newly_dead_ep_slots,
            len(running),
            self.dp_rank,
            t_redistribute_start,
            t_redistribute_start - self._t_create,
        )

        if flipped:
            t_rpc_start = time.time()
            logger.warning(
                "FT EP: DP %d entering collective_rpc("
                "eplb_redistribute_for_dead_peers, ep_slots=%s) "
                "wall_t=%.6f",
                self.dp_rank,
                newly_dead_ep_slots,
                t_rpc_start,
            )
            try:
                ec.collective_rpc(
                    "eplb_redistribute_for_dead_peers",
                    args=(newly_dead_ep_slots,),
                )
                t_rpc_end = time.time()
                logger.warning(
                    "FT EP: DP %d collective_rpc("
                    "eplb_redistribute_for_dead_peers) RETURNED "
                    "took=%.3fs wall_t=%.6f",
                    self.dp_rank,
                    t_rpc_end - t_rpc_start,
                    t_rpc_end,
                )
            except Exception as e:
                t_rpc_end = time.time()
                logger.warning(
                    "FT EP: DP %d collective_rpc("
                    "eplb_redistribute_for_dead_peers) FAILED "
                    "after=%.3fs: %s",
                    self.dp_rank,
                    t_rpc_end - t_rpc_start,
                    e,
                )

        state.snapshot_active_to_last()
        self.state = DyingPeerEngineState.COMPLETE
        t_complete = time.time()
        logger.warning(
            "FT EP: dying-peer state machine for dead DP %d -> COMPLETE. "
            "dp_rank=%d wall_t=%.6f (t_since_create=%.3fs, "
            "redistribute_took=%.3fs)",
            self.dead_dp_rank,
            self.dp_rank,
            t_complete,
            t_complete - self._t_create,
            t_complete - t_redistribute_start,
        )
