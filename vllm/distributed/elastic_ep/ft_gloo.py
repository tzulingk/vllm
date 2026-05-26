# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault-tolerant gloo process group wrapper.

When a peer dies, ``torch.distributed.new_group(ranks=[...])`` is the
wrong tool to rescope a collective onto survivors -- ``new_group`` is
itself a collective over the *parent* group, so every member (including
the dead ones) must call it, and the call hangs on the survivors.

The fix is to abandon the parent group and **re-rendezvous** with a
fresh process group built only from the surviving ranks, using
``stateless_init_torch_distributed_process_group``. Each rebuild needs
a fresh rendezvous port (a stale port can collide with a previous
incarnation's TCP listener), so the master of the new active set
allocates one and publishes it to a shared :class:`Store` under
``ft_gloo_port_<generation>``; non-master survivors read it back.

The API surface is small:

* :meth:`FaultTolerantGlooGroup.all_reduce` takes the current active
  mask + a per-call timeout, rebuilds if the mask changed, runs the
  collective, returns ``(tensor, valid)``. ``valid=False`` means the
  online collective failed (gloo timeout / runtime error).
* Rebuild itself may fail (a rank the caller thought was alive doesn't
  show up at the rendezvous). That's a distinct category of failure
  surfaced as :class:`RebuildTimeoutError` -- the caller should drop
  the unreachable rank from the mask and try again.
* :attr:`FaultTolerantGlooGroup.generation` lets the caller tag a
  result with the membership epoch it was computed under.

This wrapper does **not** consult :class:`PeerActiveStateManager` --
the caller passes ``active_mask`` in explicitly. Separation of
concerns: the wrapper knows about gloo + rendezvous, the caller knows
about peer-health state.
"""

from __future__ import annotations

import socket
import threading
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp, Store

from vllm.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class RebuildTimeoutError(RuntimeError):
    """Raised when the rendezvous for a rebuilt sub-group timed out.

    Distinct from a per-call collective failure (which returns
    ``valid=False`` from ``all_reduce``). A rebuild timeout means one
    or more ranks the caller listed in ``active_mask`` failed to reach
    the rendezvous; the caller should refine the mask and retry.
    """


def _pick_open_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class FaultTolerantGlooGroup:
    """Active-mask-aware gloo group with stateless-init-based rebuild.

    The constructor records the rendezvous coordinates but does **not**
    eagerly create a gloo group; the first :meth:`all_reduce` builds
    one for whatever active set it's given.

    Args:
        store: shared :class:`Store` used to publish per-rebuild
            rendezvous ports under ``ft_gloo_port_<generation>``. The
            store must remain reachable across rebuilds.
        master_addr: rendezvous host that every survivor can reach.
        my_global_rank: this process's global rank ID (the value used
            to index into ``active_mask``).
        total_world_size: size of the active_mask the caller will pass
            in. Used for validation.
    """

    def __init__(
        self,
        store: Store,
        master_addr: str,
        my_global_rank: int,
        total_world_size: int,
    ) -> None:
        if my_global_rank < 0 or my_global_rank >= total_world_size:
            raise ValueError(
                f"my_global_rank {my_global_rank} out of range for "
                f"total_world_size {total_world_size}"
            )
        self._store = store
        self._master_addr = master_addr
        self._my_global_rank = my_global_rank
        self._total_world_size = total_world_size
        self._current_group: ProcessGroup | None = None
        self._current_active_set: frozenset[int] | None = None
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def generation(self) -> int:
        """Monotonic counter; increments on every successful rebuild."""
        return self._generation

    @property
    def current_active_set(self) -> frozenset[int] | None:
        return self._current_active_set

    def _allocate_rebuild_port(self, generation: int, am_master: bool) -> int:
        """Rendezvous on the TCP port the next gloo subgroup will bind to.

        The new gloo subgroup needs a port the surviving ranks agree on.
        The lowest-rank survivor (the "master") picks an open port and
        publishes it to the parent TCP store; the other survivors block
        on a ``get`` of the same key until the master writes it.

        Args:
            generation: Monotonically increasing rebuild count. Used to
                construct a per-generation key ``ft_gloo_port_<gen>`` so
                a later rebuild can't race with an earlier rebuild's
                port and pick up a stale value.
            am_master: ``True`` if this rank is the rendezvous master
                (lowest-ranked survivor in the new active set).
                Exactly one rank in the rebuild has ``am_master=True``.

        Returns:
            int: TCP port on ``self._master_addr`` to bind the new gloo
            subgroup to. For the master this is the newly-picked open
            port (already published to the store); for non-masters it
            is the port read from the store, which blocks until the
            master publishes.
        """
        key = f"ft_gloo_port_{generation}"
        if am_master:
            port = _pick_open_port()
            self._store.set(key, str(port).encode())
            return port
        raw = self._store.get(key)
        return int(raw.decode() if isinstance(raw, bytes | bytearray) else raw)

    def _rebuild(
        self,
        active_set: frozenset[int],
        rebuild_timeout_ms: int,
    ) -> None:
        if self._my_global_rank not in active_set:
            raise ValueError(
                f"my_global_rank {self._my_global_rank} not in active_set "
                f"{sorted(active_set)}; cannot rebuild"
            )

        if self._current_group is not None:
            try:
                stateless_destroy_torch_distributed_process_group(self._current_group)
            except Exception as e:
                logger.warning(
                    "FaultTolerantGlooGroup: error destroying previous group "
                    "(gen %d): %s. Continuing with rebuild.",
                    self._generation,
                    e,
                )
            self._current_group = None

        sorted_active = sorted(active_set)
        my_new_rank = sorted_active.index(self._my_global_rank)
        am_master = my_new_rank == 0
        next_generation = self._generation + 1

        port = self._allocate_rebuild_port(
            generation=next_generation, am_master=am_master
        )

        logger.info(
            "FaultTolerantGlooGroup: rebuild gen=%d active=%s my_rank=%d "
            "master=%s port=%d timeout_ms=%d",
            next_generation,
            sorted_active,
            my_new_rank,
            am_master,
            port,
            rebuild_timeout_ms,
        )

        try:
            new_group = stateless_init_torch_distributed_process_group(
                host=self._master_addr,
                port=port,
                rank=my_new_rank,
                world_size=len(sorted_active),
                backend="gloo",
            )
        except Exception as e:
            # A rank we believed was alive failed to show up at the
            # rendezvous (or some other transport-level failure).
            raise RebuildTimeoutError(
                f"FaultTolerantGlooGroup: rebuild of generation "
                f"{next_generation} failed (active={sorted_active}, "
                f"port={port}): {e}"
            ) from e

        self._current_group = new_group
        self._current_active_set = active_set
        self._generation = next_generation

    def _ensure_group(
        self,
        active_mask: list[int] | tuple[int, ...],
        rebuild_timeout_ms: int,
    ) -> ProcessGroup | None:
        if len(active_mask) != self._total_world_size:
            raise ValueError(
                f"active_mask length {len(active_mask)} != total_world_size "
                f"{self._total_world_size}"
            )
        active_set = frozenset(i for i, alive in enumerate(active_mask) if alive)
        if self._my_global_rank not in active_set:
            logger.warning_once(
                "FaultTolerantGlooGroup: this rank (%d) is masked as dead; "
                "skipping collectives.",
                self._my_global_rank,
            )
            return None

        if active_set != self._current_active_set:
            self._rebuild(active_set, rebuild_timeout_ms=rebuild_timeout_ms)
        return self._current_group

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp,
        active_mask: list[int] | tuple[int, ...],
        timeout_ms: int = 5000,
        rebuild_timeout_ms: int = 10000,
    ) -> tuple[torch.Tensor, bool]:
        """All-reduce over the currently-alive sub-group.

        Args:
            tensor: tensor to reduce in place.
            op: reduction op.
            active_mask: ``[total_world_size]`` truthy values for currently-alive
                ranks. Caller fetches this from
                :class:`PeerActiveStateManager`.
            timeout_ms: per-call timeout, enforced via
                :meth:`torch.distributed.Work.wait`.
            rebuild_timeout_ms: budget for the rendezvous when a rebuild
                is triggered. Currently advisory only --
                :func:`stateless_init_torch_distributed_process_group`
                does not accept a per-call timeout arg, so the effective
                rebuild timeout is the PG-level gloo default (env var
                ``VLLM_CPU_DISTRIBUTED_TIMEOUT_SECONDS``). TODO: thread
                through if/when the helper grows a timeout parameter.

        Returns:
            ``(tensor, valid)``. ``valid=False`` means the local rank is
            dead or the online collective itself failed (timeout / runtime
            error). A rebuild failure surfaces as
            :class:`RebuildTimeoutError`.
        """
        with self._lock:
            group = self._ensure_group(active_mask, rebuild_timeout_ms)
            if group is None:
                return tensor, False
            try:
                # Per-call timeout is enforced via Work.wait(timeout=...).
                # The high-level dist.all_reduce(...) signature does NOT
                # accept a timeout kwarg; we use async_op=True and wait
                # explicitly so a hung peer surfaces as TimeoutError here.
                work = dist.all_reduce(tensor, op=op, group=group, async_op=True)
                work.wait(timeout=timedelta(milliseconds=timeout_ms))
                return tensor, True
            except (TimeoutError, RuntimeError) as e:
                logger.warning(
                    "FaultTolerantGlooGroup: all_reduce failed at gen=%d active=%s: %s",
                    self._generation,
                    sorted(self._current_active_set)
                    if self._current_active_set
                    else None,
                    e,
                )
                return tensor, False

    def destroy(self) -> None:
        if self._current_group is not None:
            try:
                stateless_destroy_torch_distributed_process_group(self._current_group)
            except Exception as e:
                logger.warning(
                    "FaultTolerantGlooGroup: error destroying group on destroy(): %s",
                    e,
                )
            self._current_group = None
            self._current_active_set = None


def ft_or_raw_all_reduce(
    tensor: torch.Tensor,
    op: ReduceOp,
    dp_group: ProcessGroup,
) -> None:
    """Route a DP collective through the FT wrapper, or fall back to raw.

    Used by the three small control-plane DP collectives in
    :mod:`vllm.config.parallel` (and any other caller that wants the
    same fault-tolerant routing) to avoid duplicating the if-FT-else-raw
    pattern at every call site.

    Behavior:
    * If both :class:`DPFTGlooManager` and
      :class:`vllm.distributed.elastic_ep.peer_state.PeerActiveStateManager`
      are initialized: fetch ``active_ranks_cpu`` from peer state, hand
      it to ``DPFTGlooManager.instance().all_reduce(...)``. On
      ``valid=False`` log a warning and leave the tensor as-is (caller's
      downstream behavior is degraded-mode -- e.g. ``has_unfinished_dp``
      returns its local value).
    * Otherwise: invoke ``torch.distributed.all_reduce(group=dp_group)``
      directly. Non-NIXL-EP deployments pay zero overhead.
    """
    # Import here to avoid an import cycle: peer_state is part of the
    # same package, but importing it at module load would create a
    # tightly-coupled circular chain through vllm.config consumers.
    from vllm.distributed.elastic_ep.peer_state import PeerActiveStateManager

    ft = DPFTGlooManager.instance()
    state = PeerActiveStateManager.instance()
    if ft is None or state is None:
        dist.all_reduce(tensor, op=op, group=dp_group)
        return

    # PeerActiveState is EP-indexed (one bit per GPU). The DP FT gloo
    # group is DP-indexed. dp_active_mask OR-reduces across each DP
    # rank's TP siblings -- a DP rank is alive for the collective if
    # any of its TP siblings can still drive it. For TP=1 this is an
    # identity.
    active_mask = state.dp_active_mask()
    _, valid = ft.all_reduce(tensor, op=op, active_mask=active_mask)
    if not valid:
        logger.warning(
            "FT NIXL EP: FT all_reduce returned valid=False at gen=%d; the "
            "local rank may be masked dead or the online collective failed. "
            "Tensor left as-is.",
            ft.generation,
        )


class DPFTGlooManager:
    """Per-process singleton holder for the DP-group FT gloo wrapper.

    Mirrors the access pattern of
    :class:`vllm.distributed.elastic_ep.peer_state.PeerActiveStateManager`.
    Initialized once at engine startup when the NIXL EP backend is
    selected; callers in ``vllm/config/parallel.py`` look up the
    singleton and fall back to raw ``torch.distributed.all_reduce`` if
    it's absent (non-FT deployments).
    """

    _instance: FaultTolerantGlooGroup | None = None

    @classmethod
    def instance(cls) -> FaultTolerantGlooGroup | None:
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._instance is not None

    @classmethod
    def init(
        cls,
        store: Store,
        master_addr: str,
        my_global_rank: int,
        total_world_size: int,
    ) -> FaultTolerantGlooGroup:
        if cls._instance is not None:
            logger.warning(
                "DPFTGlooManager.init() called but an instance already "
                "exists; returning existing."
            )
            return cls._instance
        cls._instance = FaultTolerantGlooGroup(
            store=store,
            master_addr=master_addr,
            my_global_rank=my_global_rank,
            total_world_size=total_world_size,
        )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        if cls._instance is not None:
            cls._instance.destroy()
        cls._instance = None
