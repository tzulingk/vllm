# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    Branches to the FT NCCL path when ``VLLM_FT_TP_NCCL=1`` and
    :func:`vllm.distributed.ft_tp.init_ft_tp` has run on this worker.  The FT
    path keeps the TP comm at constant size N even when a peer dies (the FT
    LSA kernel masks dead slots in place), so captured CUDA graphs and
    torch.compile artifacts stay valid across a TP-sibling failure -- but the
    FT call itself is not torch.compile-captured today (see
    ``ft_tp.py`` docstring).
    """
    from .ft_tp import get_ft_tp

    ft = get_ft_tp()
    if ft is not None:
        return ft.all_reduce(input_)
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
