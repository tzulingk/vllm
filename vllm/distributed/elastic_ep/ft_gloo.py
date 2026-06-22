# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault-tolerant gloo sub-group for per-step DP coordination.

When a DP peer dies, ``torch.distributed.new_group(ranks=[...])`` is the
wrong tool to rescope a collective onto the survivors -- ``new_group`` is
itself a collective over the *parent* group, so every member (including
the dead one) must call it, and the call hangs on the survivors.

The fix is to abandon the parent group and **re-rendezvous** a fresh
process group built only from the surviving ranks, via
``stateless_init_torch_distributed_process_group``.

Two design rules this module enforces (both decided in the FT-gloo
design discussion, see DYN-3253):

* **Rebuild is separate from collective.** :meth:`rebuild_for_survivors`
  is the *only* way membership changes. :meth:`all_reduce` runs on
  whatever group currently exists and never rebuilds. The caller of the
  per-step DP collective (``_run_ar``) therefore *cannot* react to a raw,
  unconfirmed kernel-mask blip -- it only ever consumes a group that was
  last (re)built at a consensus-confirmed beat (the
  ``recover_from_dead_peers`` RPC).

* **Content-keyed rendezvous.** The rebuild port is published under a key
  derived from the *content* of the survivor set
  (``ft_gloo_rdzv_<sorted-survivors>``), not a per-process rebuild
  counter. Every rank that computes the same survivor set computes the
  same key, so a rank that has rebuilt a different number of times still
  meets its peers. Since survivors agree on the mask by consensus, they
  agree on the key by construction.

This module holds no mask state -- the caller passes the survivor set in
explicitly. The active mask comes straight from the NIXL-EP kernel
(``query_mask``); there is intentionally no ``PeerActiveState`` here.
"""

from __future__ import annotations

import socket
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
    """Raised when the rendezvous for a rebuilt survivor group timed out.

    Distinct from a per-call collective failure (which returns
    ``valid=False`` from :meth:`FaultTolerantGlooGroup.all_reduce`). A
    rebuild timeout means a rank the caller listed as a survivor failed
    to reach the rendezvous; the caller should refine the survivor set
    and retry.
    """


def _bind_open_socket(host: str) -> tuple[socket.socket, int]:
    """Bind a listening socket on ``host`` and return ``(socket, port)``.

    The socket is returned still bound and open so it can be handed to
    ``stateless_init_torch_distributed_process_group(listen_socket=...)``,
    which avoids the TOCTOU race between picking a free port and binding
    it (the server is created directly on the pre-bound socket).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    return sock, port


class FaultTolerantGlooGroup:
    """Survivor-scoped gloo group with stateless-init-based rebuild.

    The constructor records the rendezvous coordinates but does **not**
    create a group; :meth:`rebuild_for_survivors` builds the first one.
    Until then :meth:`all_reduce` reports ``valid=False``.

    Args:
        store: shared :class:`Store` used to publish per-rebuild
            rendezvous coordinates under ``ft_gloo_rdzv_<survivors>``.
            Must stay reachable across rebuilds (hosted by a rank that
            survives -- in the single-node demo this is the DP store on
            rank 0).
        master_addr: rendezvous host the survivors bind/connect to. On a
            single node every DP rank shares the node address, so this is
            correct regardless of which survivor becomes master. See the
            multi-node TODO in :meth:`rebuild_for_survivors`.
        my_global_rank: this process's DP rank; the value used to index
            into the survivor set.
        total_world_size: original DP world size. Used for validation.
    """

    def __init__(
        self,
        store: Store,
        master_addr: str,
        my_global_rank: int,
        total_world_size: int,
    ) -> None:
        if not 0 <= my_global_rank < total_world_size:
            raise ValueError(
                f"my_global_rank {my_global_rank} out of range for "
                f"total_world_size {total_world_size}"
            )
        self._store = store
        self._master_addr = master_addr
        self._my_global_rank = my_global_rank
        self._total_world_size = total_world_size
        self._current_group: ProcessGroup | None = None
        self._current_survivors: frozenset[int] | None = None
        self._generation = 0

    @property
    def generation(self) -> int:
        """Monotonic counter; increments on each successful rebuild."""
        return self._generation

    @property
    def current_survivors(self) -> frozenset[int] | None:
        return self._current_survivors

    @property
    def has_group(self) -> bool:
        return self._current_group is not None

    @staticmethod
    def _rdzv_key(survivors: list[int]) -> str:
        return "ft_gloo_rdzv_" + "-".join(str(r) for r in survivors)

    def rebuild_for_survivors(
        self,
        survivors: set[int] | frozenset[int],
        rebuild_timeout_ms: int = 10000,
    ) -> None:
        """(Re)build the gloo group so it contains exactly ``survivors``.

        Called only at a consensus-confirmed beat (the
        ``recover_from_dead_peers`` RPC). The lowest-ranked survivor is
        the rendezvous master: it binds an open socket, publishes
        ``master_addr:port`` under the content-keyed store key, then
        creates the new group's :class:`Store` server on that socket.
        Every other survivor reads the coordinates back and connects.

        Args:
            survivors: DP ranks that should be in the rebuilt group. Must
                contain ``my_global_rank``.
            rebuild_timeout_ms: advisory rendezvous budget. The effective
                timeout is the gloo PG default
                (``VLLM_CPU_DISTRIBUTED_TIMEOUT_SECONDS``); threaded
                through if/when the stateless-init helper accepts one.

        Raises:
            ValueError: ``my_global_rank`` is not in ``survivors``.
            RebuildTimeoutError: a listed survivor failed to rendezvous.
        """
        survivor_set = frozenset(survivors)
        if self._my_global_rank not in survivor_set:
            raise ValueError(
                f"my_global_rank {self._my_global_rank} not in survivors "
                f"{sorted(survivor_set)}; cannot rebuild"
            )
        if not survivor_set - {self._my_global_rank} and len(survivor_set) == 1:
            logger.warning(
                "FT gloo: rebuilding a single-member group (survivors=%s); "
                "the all_reduce will be a local no-op.",
                sorted(survivor_set),
            )

        self._destroy_current_group()

        sorted_survivors = sorted(survivor_set)
        my_new_rank = sorted_survivors.index(self._my_global_rank)
        am_master = my_new_rank == 0
        key = self._rdzv_key(sorted_survivors)

        # TODO(multi-node / rank-0 death): master publishes self._master_addr,
        # which on a single node is the shared node address and is correct for
        # any survivor-master. Across nodes a relocated master must publish its
        # own reachable IP, and the coordination ``store`` itself must be
        # hosted by a surviving rank (today it lives on original rank 0).
        listen_socket: socket.socket | None = None
        if am_master:
            listen_socket, port = _bind_open_socket(self._master_addr)
            self._store.set(key, f"{self._master_addr}:{port}".encode())
            host = self._master_addr
        else:
            raw = self._store.get(key)
            coords = raw.decode() if isinstance(raw, bytes | bytearray) else raw
            host, port_str = coords.rsplit(":", 1)
            port = int(port_str)

        next_generation = self._generation + 1
        logger.info(
            "FT gloo: rebuild gen=%d survivors=%s my_new_rank=%d master=%s "
            "host=%s port=%d",
            next_generation,
            sorted_survivors,
            my_new_rank,
            am_master,
            host,
            port,
        )

        try:
            new_group = stateless_init_torch_distributed_process_group(
                host=host,
                port=port,
                rank=my_new_rank,
                world_size=len(sorted_survivors),
                backend="gloo",
                listen_socket=listen_socket,
            )
        except Exception as e:
            if listen_socket is not None:
                listen_socket.close()
            raise RebuildTimeoutError(
                f"FT gloo: rebuild of gen {next_generation} failed "
                f"(survivors={sorted_survivors}, host={host}, port={port}): {e}"
            ) from e

        self._current_group = new_group
        self._current_survivors = survivor_set
        self._generation = next_generation

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        timeout_ms: int = 5000,
    ) -> tuple[torch.Tensor, bool]:
        """All-reduce over the current survivor group. Never rebuilds.

        Args:
            tensor: tensor to reduce in place.
            op: reduction op.
            timeout_ms: per-call timeout, enforced via
                :meth:`torch.distributed.Work.wait`.

        Returns:
            ``(tensor, valid)``. ``valid=False`` means no group has been
            built yet, or the online collective failed (timeout / runtime
            error). The caller (``_run_ar``) degrades to its local-only
            contribution on ``valid=False``.
        """
        group = self._current_group
        if group is None:
            return tensor, False
        # dist.all_reduce(...) has no timeout kwarg; use async_op + an
        # explicit Work.wait(timeout=...) so a hung peer surfaces as a
        # bounded TimeoutError instead of blocking the worker forever.
        work = dist.all_reduce(tensor, op=op, group=group, async_op=True)
        try:
            work.wait(timeout=timedelta(milliseconds=timeout_ms))
        except (TimeoutError, RuntimeError) as e:
            logger.warning(
                "FT gloo: all_reduce failed at gen=%d survivors=%s: %s",
                self._generation,
                sorted(self._current_survivors) if self._current_survivors else None,
                e,
            )
            return tensor, False
        return tensor, True

    def _destroy_current_group(self) -> None:
        if self._current_group is None:
            return
        try:
            stateless_destroy_torch_distributed_process_group(self._current_group)
        except RuntimeError as e:
            logger.warning(
                "FT gloo: error destroying previous group (gen %d): %s. Continuing.",
                self._generation,
                e,
            )
        self._current_group = None

    def destroy(self) -> None:
        self._destroy_current_group()
        self._current_survivors = None


# --------------------------------------------------------------------------- #
# Thin process-local accessor.
#
# Build (rebuild_for_survivors, via the recover_from_dead_peers RPC) and read
# (all_reduce, via _run_ar) both happen in the *same* worker process, so a
# module-global holder has no cross-process staleness hazard (unlike the
# engine-vs-worker split that bit PeerActiveState). All logic lives on the
# class above; this is only a holder + accessor, kept testable via reset().
# --------------------------------------------------------------------------- #

_DP_FT_GLOO: FaultTolerantGlooGroup | None = None


def get_dp_ft_gloo() -> FaultTolerantGlooGroup | None:
    """Return the worker's DP FT-gloo group, or None if not initialized."""
    return _DP_FT_GLOO


def init_dp_ft_gloo(
    store: Store,
    master_addr: str,
    my_global_rank: int,
    total_world_size: int,
) -> FaultTolerantGlooGroup:
    """Initialize the process-local DP FT-gloo holder (idempotent)."""
    global _DP_FT_GLOO
    if _DP_FT_GLOO is not None:
        logger.warning(
            "init_dp_ft_gloo() called but an instance already exists; "
            "returning existing."
        )
        return _DP_FT_GLOO
    _DP_FT_GLOO = FaultTolerantGlooGroup(
        store=store,
        master_addr=master_addr,
        my_global_rank=my_global_rank,
        total_world_size=total_world_size,
    )
    return _DP_FT_GLOO


def reset_dp_ft_gloo() -> None:
    """Tear down and clear the process-local holder (used by tests)."""
    global _DP_FT_GLOO
    if _DP_FT_GLOO is not None:
        _DP_FT_GLOO.destroy()
    _DP_FT_GLOO = None
