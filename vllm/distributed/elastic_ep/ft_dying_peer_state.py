# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State machine for fault-tolerant handling of a peer DP rank death.

Modeled on
:class:`vllm.distributed.elastic_ep.elastic_state.ElasticEPScalingState`:
each surviving engine drives its own copy of the state machine through
its ``run_busy_loop``, advancing at most one state per loop tick.
Cross-DP synchronization is a TCPStore-based non-blocking barrier with
a 5-second first-attempt timeout (mirroring elastic-EP's
``_staged_barrier``): engines that arrive at the barrier early fall
back to a normal forward pass and retry on the next tick, so peers
slightly behind in their loops can catch up without anyone being
silent on NIXL EP.

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
from datetime import timedelta
from typing import TYPE_CHECKING

from torch.distributed import Store

from vllm.distributed import sched_yield
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


class _BarrierTimeoutError(RuntimeError):
    """First-attempt staged-barrier timeout signal."""


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
        logger.warning(
            "FT EP: FtDyingPeerState created for dead DP rank %d "
            "(survivor count=%d, leader_rank=%d, key=%s)",
            dead_dp_rank,
            self.survivor_count,
            self.leader_rank,
            self._key_suffix,
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
            logger.warning(
                "FT EP: dying-peer barrier %s: announced arrival; "
                "waiting for %d survivors.",
                self._barrier_name,
                self.survivor_count,
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
        logger.warning(
            "FT EP: dying-peer barrier %s passed; advancing to REDISTRIBUTE.",
            self._barrier_name,
        )
        return True

    def _staged_barrier(self) -> bool:
        """TCPStore-only staged barrier.

        Skips the ``torch.distributed.barrier(dp_group)`` that
        elastic-EP's version uses: our ``dp_group`` still includes the
        dead rank and would hang forever. The TCPStore polling barrier
        on its own gives wall-clock synchronization for surviving
        ranks.
        """
        sync_key = f"{self._barrier_name}_sync"
        timeout = (
            None
            if self.dp_store.check([sync_key])
            else timedelta(seconds=_BARRIER_FIRST_ATTEMPT_TIMEOUT_S)
        )
        try:
            self._execute_tcp_store_barrier(timeout=timeout)
            if self.dp_rank == self.leader_rank:
                for r in range(self.dp_world_size):
                    if r == self.dead_dp_rank:
                        continue
                    self.dp_store.delete_key(self._arrival_key(r))
                if self.dp_store.check([sync_key]):
                    self.dp_store.delete_key(sync_key)
            return True
        except _BarrierTimeoutError as e:
            if timeout is None:
                raise RuntimeError(
                    "FT EP: unexpected timeout on second-stage barrier "
                    f"{self._barrier_name} (should not happen with timeout=None)"
                ) from e
            # First-stage timeout: mark the sync key so the next attempt
            # uses no timeout and blocks until everyone arrives.
            self.dp_store.compare_set(sync_key, "", b"1")
            return False

    def _arrival_key(self, rank: int) -> str:
        return f"arrival_{self._barrier_name}_{rank}"

    def _execute_tcp_store_barrier(self, timeout):
        arrival_key = self._arrival_key(self.dp_rank)
        self.dp_store.set(arrival_key, b"1")

        start = time.time()
        expected = {r for r in range(self.dp_world_size) if r != self.dead_dp_rank}
        arrived: set[int] = set()
        while arrived != expected:
            if timeout is not None and time.time() - start > timeout.total_seconds():
                raise _BarrierTimeoutError(
                    f"FT EP: barrier {self._barrier_name} first-attempt "
                    f"timeout after {timeout.total_seconds()}s; "
                    f"arrived={sorted(arrived)}, expected={sorted(expected)}"
                )
            for r in expected:
                if r in arrived:
                    continue
                if self.dp_store.check([self._arrival_key(r)]):
                    arrived.add(r)
            if arrived != expected:
                sched_yield()

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
        logger.warning(
            "FT EP: REDISTRIBUTE running for dead DP %d (EP slots %s, "
            "aborted %d in-flight req(s)).",
            self.dead_dp_rank,
            newly_dead_ep_slots,
            len(running),
        )

        if flipped:
            try:
                ec.collective_rpc(
                    "eplb_redistribute_for_dead_peers",
                    args=(newly_dead_ep_slots,),
                )
            except Exception as e:
                logger.warning(
                    "FT EP: REDISTRIBUTE for dead DP %d -- "
                    "eplb_redistribute_for_dead_peers RPC failed: %s",
                    self.dead_dp_rank,
                    e,
                )

        state.snapshot_active_to_last()
        self.state = DyingPeerEngineState.COMPLETE
        logger.warning(
            "FT EP: dying-peer state machine for dead DP %d -> COMPLETE.",
            self.dead_dp_rank,
        )
