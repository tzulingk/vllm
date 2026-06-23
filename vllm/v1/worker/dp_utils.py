# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import torch
import torch.distributed as dist

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    is_last_ubatch_empty,
)

logger = init_logger(__name__)

# FT NIXL EP: fail-fast bound for the death-step full-group DP all_reduce.
# When a peer dies/hangs, the survivors' full-group all_reduce must not stall
# this step beyond the NIXL all-to-all window (VLLM_NIXL_EP_TIMEOUT_MS, 5s) --
# a longer stall desyncs the survivors so the kernel false-flags innocent peers
# (see runbook "DP3 false-positive"). The synchronous all_reduce only honors
# the process group's timeout (gloo's 1800s default unless the group was made
# with --cpu-distributed-timeout-seconds), and a *hung* peer never closes its
# socket to fail early, so we bound the wait explicitly. Kept well under 5s.
_RUN_AR_FAILFAST_TIMEOUT_MS = 1000


def _get_device_and_group(parallel_config: ParallelConfig):
    # Use the actual device assigned to the DP group, not just the device type
    device = get_dp_group().device
    group = get_dp_group().device_group

    # Transferring this tensor from GPU to CPU will introduce a GPU sync
    # point that could adversely affect performance of vllm with asynch
    # scheduling. This environment variable exists to quickly disable
    # this optimization if we run into this case.
    if parallel_config.disable_nccl_for_dp_synchronization:
        logger.info_once(
            "Using CPU all reduce to synchronize DP padding between ranks.",
        )
        device = "cpu"
        group = get_dp_group().cpu_group
    return device, group


def _run_ar(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device, group = _get_device_and_group(parallel_config)
    # Populate this rank's contribution on CPU to reduce GPU syncs.
    tensor_cpu = torch.zeros(4, dp_size, dtype=torch.int32)
    tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0
    tensor_cpu[3][dp_rank] = cudagraph_mode
    # FT NIXL EP: after a peer dies, ``recover_from_dead_peers`` rebuilds a
    # survivors-only gloo group. When that group exists, run the per-step DP
    # coordination over the survivors only. The gloo group is CPU-only, so we
    # reduce the CPU tensor regardless of the deployment's normal NCCL/CPU
    # choice; this path engages only after a death, so steady state keeps the
    # normal full-group all_reduce below with zero overhead. ``_run_ar`` never
    # rebuilds -- it only consumes the group the recovery RPC last built at a
    # consensus-confirmed beat (see ft_gloo.py).
    from vllm.distributed.elastic_ep.ft_gloo import get_dp_ft_gloo

    ft = get_dp_ft_gloo()
    if ft is not None and ft.has_group:
        _, valid = ft.all_reduce(tensor_cpu, dist.ReduceOp.SUM)
        if valid:
            # Dead-rank columns are 0 after the survivors-only reduce.
            # Backfill them with this rank's own contribution so they don't
            # veto the ubatch (all==1), cudagraph (min), or padding (max/min)
            # consensus among survivors. This rank is itself a survivor, so
            # the survivor consensus is unchanged.
            survivors = ft.current_survivors or frozenset()
            for d in range(dp_size):
                if d not in survivors:
                    tensor_cpu[0, d] = orig_num_tokens_per_ubatch
                    tensor_cpu[1, d] = padded_num_tokens_per_ubatch
                    tensor_cpu[2, d] = 1 if should_ubatch else 0
                    tensor_cpu[3, d] = cudagraph_mode
        else:
            logger.warning_once(
                "FT NIXL EP: survivor DP all_reduce returned invalid on "
                "dp_rank=%d; proceeding with local-only contribution this "
                "step. Ubatching + CUDA-graph disabled this step.",
                dp_rank,
            )
        return tensor_cpu.to(device, non_blocking=True)

    tensor = tensor_cpu.to(device, non_blocking=True)
    # Cascade guard + fail-fast: on the death step itself -- before
    # ``recover_from_dead_peers`` has rebuilt the survivor group -- the
    # full-group all_reduce still includes the just-dead peer. We must not
    # block here: a multi-second stall desyncs the survivors past the NIXL
    # all-to-all window and the kernel false-flags innocent peers. So issue
    # the all_reduce async and bound the wait to _RUN_AR_FAILFAST_TIMEOUT_MS
    # (the synchronous form would instead wait the gloo PG timeout, and a hung
    # peer never closes its socket to fail early). On any failure/timeout we
    # proceed local-only for this one step; the next step uses the rebuilt
    # survivor group above. Without the guard the survivors' loop cascade-dies.
    try:
        work = dist.all_reduce(tensor, group=group, async_op=True)
        work.wait(timeout=timedelta(milliseconds=_RUN_AR_FAILFAST_TIMEOUT_MS))
    except (RuntimeError, ValueError, TimeoutError) as e:
        logger.warning_once(
            "FT NIXL EP cascade guard: DP all_reduce on dp_rank=%d failed or "
            "timed out after %dms (%s: %s); proceeding with local-only "
            "contribution this step. Ubatching + CUDA-graph disabled this step.",
            dp_rank,
            _RUN_AR_FAILFAST_TIMEOUT_MS,
            type(e).__name__,
            e,
        )
    return tensor


def _post_process_ubatch(tensor: torch.Tensor, num_ubatches: int) -> bool:
    orig_num_tokens_tensor = tensor[0, :]
    padded_num_tokens_tensor = tensor[1, :]

    # First determine if we are going to be ubatching.
    should_ubatch: bool = bool(torch.all(tensor[2] == 1).item())
    if not should_ubatch:
        return False
    # If the DP ranks are planning to ubatch, make sure that
    # there are no "empty" second ubatches
    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens, num_ubatches):
        logger.debug(
            "Aborting ubatching %s %s", orig_min_num_tokens, padded_max_num_tokens
        )
        should_ubatch = False
    return should_ubatch


def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    num_tokens_across_dp = tensor[1, :]
    if should_dp_pad:
        # If DP padding is enabled, ensure that each rank is processing the same number
        # of tokens
        max_num_tokens = int(num_tokens_across_dp.max().item())
        return torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
    else:
        return num_tokens_across_dp.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[3, :].min().item())


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do.

    2. Determines the total number of tokens that each rank will run.
    When running microbatched or if cudagraph is enabled (synced across ranks),
    all ranks will be padded out so that they run with the same number of tokens.

    3. Synchronizes cudagraph_mode across ranks by taking the minimum.

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including any DP padding.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded

    # Coordinate between the DP ranks via an All Reduce
    # to determine the total number of tokens that each rank
    # will run and if we are using ubatching or not.
    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )

    # Synchronize cudagraph_mode across ranks first (take min).
    # This is needed before DP padding decision since we use the synced
    # cudagraph mode to determine whether DP padding is needed.
    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)

    # Check conditions for microbatching
    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)

    # DP padding is needed when cudagraph is enabled (synced across ranks)
    # or when ubatching/DBO is active (ubatching requires uniform batch
    # sizes across DP ranks currently).
    # Use the synced runtime cudagraph mode rather than the compilation config
    # so we can avoid padding when cudagraph is not enabled for this step.
    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch

    # Pad all DP ranks up to the maximum token count across ranks if
    # should_dp_pad is True
    num_tokens_after_padding = _post_process_dp_padding(
        tensor,
        should_dp_pad,
    )

    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode


def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Args:
        num_tokens_unpadded: Number of tokens without accounting for padding
        allow_microbatching: If microbatching should be attempted
        parallel_config: The parallel config
        num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,
            TP, etc)
        uniform_decode: Only used if allow_microbatching is True. True if the batch
            only contains single token decodes
        cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL).
            DP padding is enabled when synced cudagraph mode across ranks is not NONE.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        padded up to the max value across all DP ranks when cudagraph is enabled.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]

    """
    if parallel_config.data_parallel_size == 1:
        # Early exit.
        return False, None, cudagraph_mode

    # If the caller has explicitly enabled microbatching.
    should_attempt_ubatching = False
    if allow_microbatching:
        # Check preconditions for microbatching
        assert uniform_decode is not None
        should_attempt_ubatching = check_ubatch_thresholds(
            parallel_config,
            num_tokens_unpadded,
            uniform_decode=uniform_decode,
        )

    if num_tokens_padded is None:
        num_tokens_padded = num_tokens_unpadded

    (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode) = (
        _synchronize_dp_ranks(
            num_tokens_unpadded,
            num_tokens_padded,
            should_attempt_ubatching,
            cudagraph_mode,
            parallel_config,
        )
    )

    return (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)
