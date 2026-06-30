<!-- markdownlint-disable -->
# Integrating FT-NCCL into vLLM for the tensor-parallel all-reduce

> Companion to [`fault-tolerance-overview.md`](./fault-tolerance-overview.md) (the *why*)
> and [`ft-nixl-ep-runbook.md`](./ft-nixl-ep-runbook.md) (the *how to run*). This doc is
> the concrete *how to build & wire* for making the **TP all-reduce** fault-tolerant —
> the one model-run collective that FT-gloo cannot cover, and the prerequisite the
> overview calls "FT NCCL on the TP group" / the "Single-TP-peer recovery" open question.

## Table of contents

1. [TL;DR](#tldr)
2. [Why this is needed](#why-this-is-needed)
3. [What FT-NCCL is](#what-ft-nccl-is)
4. [Where it plugs into vLLM](#where-it-plugs-into-vllm)
5. [The integration, in three layers](#the-integration-in-three-layers)
   1. [Layer 1 — the native libraries (one NCCL for the whole process)](#layer-1--the-native-libraries-one-nccl-for-the-whole-process)
   2. [Layer 2 — the vLLM code changes](#layer-2--the-vllm-code-changes)
   3. [Layer 3 — the runtime environment](#layer-3--the-runtime-environment)
6. [Build & package](#build--package)
7. [Validation](#validation)
8. [Scope & limitations](#scope--limitations)
9. [Status](#status)
10. [References](#references)

## TL;DR

vLLM's tensor-parallel all-reduce runs on **standard NCCL**, which hangs forever if a TP
peer dies mid-collective. That is the only model-run collective the FT NIXL EP work has
*not* made fault-tolerant — the cross-DP collectives are routed over fail-fast **FT-gloo**,
but the TP all-reduce is GPU-resident, per-layer, and high-bandwidth, so gloo is a
non-starter. The fix is **FT-NCCL**: a fork of NCCL whose device-side collective kernels
take a `timeout_us` budget and an active-rank mask, so a dead peer is masked instead of
deadlocking the survivors.

Integration is three layers:

1. **Native libs** — build the fork's `libnccl.so.2` + `libft_collective.so` for
   `arm64 / sm_100` (GB200) and make the fork NCCL the *single* NCCL the whole process
   loads (PyTorch included), via `LD_PRELOAD`.
2. **vLLM code** — gated by `VLLM_USE_FT_NCCL_TP=1`: import `ft_collective` (registers the
   `"ft_nccl"` torch backend), create the **TP** group with backend `"ft_nccl"`, and stage
   the TP all-reduce input into an FT symmetric tensor in `CudaCommunicator.all_reduce`.
3. **Runtime env** — disable vLLM's faster TP all-reduce paths so the call reaches the
   torch-distributed fallback (the only path `ft_nccl` participates in), and point the FT
   library at its `.so`s.

For the first cut, an FT timeout is **fail-fast**: the in-flight requests on that TP group
are errored (retryable); resharding the dead TP shard is out of scope.

## Why this is needed

A TP group is a set of GPUs that each hold a *shard of every layer's weights*; on every
transformer layer they exchange partial results through a synchronized all-reduce
(attention `o_proj`, MLP `down_proj`, …). Under standard NCCL, if one TP peer dies, the
survivors' next all-reduce never completes — NCCL has no abort/timeout on the collective
itself, so the forward pass hangs until the NCCL watchdog crashes the process.

The FT NIXL EP work already protects every *cross-DP* model-run collective by routing it
over a survivor-scoped, fail-fast **FT-gloo** group (the per-forward DP `num_tokens`
all-reduce in `dp_utils._run_ar`, the wave-sync, and the EPLB load-aggregation). FT-gloo
works there because those tensors are small, infrequent, and CPU-friendly. The TP
all-reduce is the opposite: it fires on every layer, is GPU-resident, and is
bandwidth-bound — routing it over CPU gloo would be orders of magnitude too slow.

Consequence: FT NIXL EP today is effectively **TP=1 only** (at TP=1 the TP group has one
member, so there is no TP collective to hang). To run **TP>1** with fault tolerance, the
TP all-reduce needs an abortable/bounded collective — that is FT-NCCL.

## What FT-NCCL is

FT-NCCL is a fork of NCCL — [`tstamler/nccl` @ `fault-tolerant-allgather`](https://gitlab-master.nvidia.com/tstamler/nccl/-/merge_requests/2),
NCCL **2.30-a17 + GPUNetIO 2.0** — with two additions:

- **Core patch:** a *timeout-bounded* `sync()` overload on the Device-API barrier
  (`ncclGinBarrierSession::sync` / the LSA equivalent). This is the FT primitive, and it
  exists **only in the fork's Device-API headers**. "**Stock NCCL**" — the official,
  unmodified NVIDIA NCCL (e.g. the `nvidia-nccl-cu13` wheel PyTorch bundles, or the NCCL in
  a base CUDA image) — does **not** have this overload *even at the same 2.30 version*. So
  you cannot satisfy the dependency with "any NCCL 2.30"; it must be **this fork**.
- **`contrib/fault_tolerant_collectives/`** — device-side FT AllReduce/AllGather kernels
  (`libft_collective.so`) plus a pure-Python `ft_collective` package that exposes them as a
  torch `ProcessGroup`.

The Python surface is `FTProcessGroup`, a subclass of `torch.distributed.ProcessGroupNCCL`:

- Importing `ft_collective` **registers the `"ft_nccl"` torch-distributed backend**
  (`dist.Backend.register_backend("ft_nccl", …, devices=["cuda"])`).
- It overrides only `allreduce`/`allgather`. A collective takes the **FT kernel path** iff
  the input tensor is a **symmetric tensor allocated by `pg.empty()`** (backed by
  `ncclMemAlloc`, symmetric virtual address across ranks) and within the scratch-buffer
  size. Otherwise it falls back to ordinary NCCL with a `UserWarning`.
- On timeout the kernel returns a **partial result + a `result_mask`** of which ranks
  responded — it does *not* hang or abort. (Liveness, not correctness: see the overview's
  "liveness, not correctness" caveat.)
- Timeout is `FT_TIMEOUT_US` (default 10 s); scratch cap is `FT_NCCL_MAX_COUNT`.

That `pg.empty()` requirement is the single most important integration constraint — it
dictates the staging step in Layer 2.

## Where it plugs into vLLM

The TP all-reduce enters here:

```
tensor_model_parallel_all_reduce(...)            # vllm/distributed/communication_op.py
  └─ GroupCoordinator.all_reduce(...)            # vllm/distributed/parallel_state.py
       └─ CudaCommunicator.all_reduce(input_)    # .../device_communicators/cuda_communicator.py:254
```

[`CudaCommunicator.all_reduce`](https://github.com/tzulingk/vllm/blob/97038bb6e207b85b4ce31c11e1434525a6aaecff/vllm/distributed/device_communicators/cuda_communicator.py#L254-L311)
tries up to **seven** dispatch paths in order. The only one that goes through
`torch.distributed` (and therefore can reach an `FTProcessGroup`) is the **last** one,
which this doc calls **path 7**:

| # | Path | Disable it via |
|---|------|----------------|
| 1 | NCCL symmetric-memory custom op | `VLLM_ALLREDUCE_USE_SYMM_MEM=0` |
| 2 | quick reduce (ROCm) | n/a on CUDA |
| 3 | FlashInfer all-reduce | `VLLM_ALLREDUCE_USE_FLASHINFER=0` |
| 4 | custom all-reduce | `--disable-custom-all-reduce` |
| 5 | torch symmetric-memory | `VLLM_ALLREDUCE_USE_SYMM_MEM=0` |
| 6 | PyNCCL | `VLLM_DISABLE_PYNCCL=1` |
| **7** | **`torch.distributed.all_reduce(out, group=self.device_group)`** — the only `torch.distributed` path | ← the FT-NCCL path |

"Path 7" = the final `torch.distributed.all_reduce(out, group=self.device_group)` branch
of `all_reduce` ([the `pynccl is None / disabled` and `out is None` fallbacks](https://github.com/tzulingk/vllm/blob/97038bb6e207b85b4ce31c11e1434525a6aaecff/vllm/distributed/device_communicators/cuda_communicator.py#L297-L311)).
When the TP `device_group` is an `FTProcessGroup` (backend `"ft_nccl"`), that call dispatches
into `FTProcessGroup.allreduce` automatically — **but only takes the FT kernel if `out` is
symmetric.** So the integration is: (a) make the TP group an `FTProcessGroup`, (b) ensure
the TP all-reduce reaches a `torch.distributed.all_reduce` on it (the Layer-3 env vars
disable paths 1-6 so they aren't even constructed), and (c) stage the input into an FT
symmetric tensor first.

## The integration, in three layers

### Layer 1 — the native libraries (one NCCL for the whole process)

`ft_collective` (pure-Python ctypes) loads two native libraries at runtime:

| Library | Provides | Found via |
|---|---|---|
| `libnccl.so.2` (the **fork's**) | symmetric-memory + Device-API + the FT barrier-timeout | `NCCL_HOME` / `LD_LIBRARY_PATH` |
| `libft_collective.so` | the FT device kernels (`ftHandleCreate`, `ftAllReduce`, …) | `FT_COLLECTIVE_LIB` / `LD_LIBRARY_PATH` |

**The critical model: exactly one NCCL is loaded in the process, and it is the fork's.**
PyTorch normally brings its own NCCL (the `nvidia-nccl-cu13` wheel that `torch==2.11.0`
depends on, shipped at `.../site-packages/nvidia/nccl/lib/libnccl.so.2`). If two different
`libnccl.so.2` were loaded, the FT symbols and PyTorch's collectives would not share state.
We force a single NCCL — the fork's — for the whole process:

```bash
export LD_PRELOAD=/ftnccl/nccl/lib/libnccl.so.2:$LD_PRELOAD
```

Because the fork is the same 2.30 generation PyTorch expects, this is normally
ABI-safe — but it **must be smoke-tested** (see [Validation](#validation)) before trusting
an image. (`LD_PRELOAD` also satisfies `dlopen("libnccl.so.2")` by soname, so torch's
link-time *and* lazy-loaded NCCL both resolve to the fork's.)

Building the libs is the heavy step (the foundation) — see [Build & package](#build--package).

### Layer 2 — the vLLM code changes

All four changes are **gated behind `VLLM_USE_FT_NCCL_TP`** and are no-ops when it is unset.

**(a) Add the flag** — `vllm/envs.py`:

```python
# Route the tensor-parallel all-reduce through the fault-tolerant NCCL
# ("ft_nccl") backend so a dead TP peer is masked on a timeout instead of
# hanging the forward. Requires the FT-NCCL libs in the image (see
# ft-nccl-tp-integration.md). TP-only; a timeout is fail-fast.
"VLLM_USE_FT_NCCL_TP": lambda: bool(int(os.getenv("VLLM_USE_FT_NCCL_TP", "0"))),
```

**(b) Register the backend before distributed init.** Importing `ft_collective` registers
`"ft_nccl"`; this must happen before any process group is created. Guarded import in the
worker/distributed bootstrap:

```python
if envs.VLLM_USE_FT_NCCL_TP:
    import ft_collective  # noqa: F401  registers the "ft_nccl" torch backend
```

**(c) Create the TP group with the `"ft_nccl"` backend** — `initialize_model_parallel`,
[parallel_state.py:1786-1792](https://github.com/tzulingk/vllm/blob/97038bb6e207b85b4ce31c11e1434525a6aaecff/vllm/distributed/parallel_state.py#L1786-L1792).
The TP group is the *only* group we move to `ft_nccl`; world/DP/EP/PP stay on the normal
backend:

```python
tp_backend = "ft_nccl" if envs.VLLM_USE_FT_NCCL_TP else backend
_TP = init_model_parallel_group(
    group_ranks,
    get_world_group().local_rank,
    tp_backend,                      # was: backend
    use_message_queue_broadcaster=True,
    group_name="tp",
)
```

Because TP is the only `ft_nccl` group, each worker process constructs **exactly one**
`FTProcessGroup`, so `ft_collective.get_ft_process_group()` (the rank-keyed registry's
single entry) unambiguously returns the TP group's FT process group — no per-group
plumbing needed for the TP-only milestone.

**(d) Stage the TP all-reduce into an FT symmetric tensor** — inserted at the **top** of
[`CudaCommunicator.all_reduce`](https://github.com/tzulingk/vllm/blob/97038bb6e207b85b4ce31c11e1434525a6aaecff/vllm/distributed/device_communicators/cuda_communicator.py#L254-L311),
short-circuiting the TP group *before* the seven dispatch paths and going straight to a
`torch.distributed.all_reduce` on the `ft_nccl` group (i.e. path 7). The reason staging is
required: `FTProcessGroup` only takes the FT kernel when the input is **symmetric** (from
`pg.empty()`); vLLM's ordinary fallback does `out = input_.clone()`, and a clone is *not*
symmetric, so it would silently run plain (non-FT) NCCL. Stage into a symmetric tensor
instead:

```python
# FT-NCCL TP: stage into a symmetric tensor so FTProcessGroup takes the FT kernel.
if envs.VLLM_USE_FT_NCCL_TP and self.unique_name.split(":")[0] == "tp":
    import ft_collective
    ft_pg = ft_collective.get_ft_process_group()
    out = ft_pg.empty(*input_.shape, dtype=input_.dtype).reshape_as(input_)
    out.copy_(input_)
    torch.distributed.all_reduce(out, group=self.device_group)
    return out
```

Line by line:

- `ft_pg.empty(*input_.shape, dtype=input_.dtype)` allocates a **fresh, uninitialized**
  device buffer via `ncclMemAlloc` and registers its NCCL window. "Symmetric" means the
  allocation lands at the **same virtual address on every rank** — the property the FT
  kernel relies on to read peers' buffers directly. `empty()` returns a flat **1-D** tensor
  of `prod(shape)` elements (it only uses `shape` to size the allocation), so
  `.reshape_as(input_)` restores the original N-D shape. The buffer holds **garbage** at
  this point — like `torch.empty`, it is *not* a copy of `input_`.
- `out.copy_(input_)` is therefore required: it moves the actual all-reduce input data into
  the symmetric buffer. Without it we would reduce uninitialized memory. (We can't reduce
  `input_` directly — it's an ordinary model-forward tensor, not symmetric, so it would not
  take the FT kernel path.)
- `torch.distributed.all_reduce(out, …)` reduces `out` **in place** across the TP ranks via
  the FT kernel (it's symmetric, so the FT path is taken), and `out` — now holding the
  summed result in `input_`'s shape — is returned.

> Detection of a dead peer (reading `ft_pg.get_result_mask()` and erroring the in-flight
> requests) is the fail-fast policy layered on top — see [Scope & limitations](#scope--limitations).

### Layer 3 — the runtime environment

Set on the serving pod so the TP all-reduce reaches path 7 and the FT libs are found:

```bash
# --- make the fork NCCL the single process NCCL ---
export LD_PRELOAD=/ftnccl/nccl/lib/libnccl.so.2:$LD_PRELOAD
export NCCL_HOME=/ftnccl/nccl
export LD_LIBRARY_PATH=/ftnccl/nccl/lib:$LD_LIBRARY_PATH
export FT_COLLECTIVE_LIB=/ftnccl/ft/libft_collective.so

# --- FT behaviour ---
export VLLM_USE_FT_NCCL_TP=1
export FT_TIMEOUT_US=10000000        # 10s GPU-side collective timeout (fail-fast budget)

# --- force the TP all-reduce onto the torch-distributed (ft_nccl) fallback ---
export VLLM_DISABLE_PYNCCL=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
# ...and pass --disable-custom-all-reduce to `vllm serve`
```

## Build & package

The fork NCCL must be built **from source** — stock NCCL lacks the Device-API timeout
overload. Good news for the build: the GPUNetIO/DOCA dependency is **vendored in-tree**
(`src/transport/net_ib/gdaki/doca-gpunetio/`), so the build is self-contained on a stock
`nvidia/cuda:13.0.2-devel` arm64 image — no external DOCA SDK. The TP all-reduce is
intra-node NVLink (the **LSA** transport); GIN (multi-node IB) is compiled in but unused.

<details>
<summary><b>What are LSA, GIN, and DOCA/GPUNetIO?</b> (transport primer)</summary>

The FT collective kernels can move data two ways; the FT handle auto-picks one at runtime
based on how the ranks are wired together:

- **LSA (Load/Store Access)** — *intra-node, over NVLink.* GPUs in the same NVLink domain
  can read and write each other's memory with ordinary load/store instructions, as if it
  were local memory (the access simply travels over NVLink). The FT kernel performs the
  collective by directly touching peers' buffers. Lowest latency; single-node only. **This
  is the path a TP group uses** — a TP group is always NVLink-local.
- **GIN (GPU-Initiated Networking)** — *multi-node, over InfiniBand.* When ranks live on
  different nodes (no NVLink between them), the GPU kernel issues the network sends/receives
  **itself**, without bouncing through the CPU. That GPU-driven RDMA is provided by **DOCA
  GPUNetIO**:
  - **DOCA** is NVIDIA's SDK for DPU/GPU networking (a bundle of libraries + drivers).
  - **GPUNetIO** is the DOCA library that lets CUDA *device code* post InfiniBand RDMA
    operations directly. NCCL 2.30's GIN transport is built on it.

"**Vendored in-tree**" means the fork ships a *copy* of the DOCA GPUNetIO source + headers
inside its own repo, so compiling NCCL does **not** require installing the external DOCA
SDK — the build is self-contained. For our TP all-reduce we only use LSA, so GIN/DOCA is
compiled but never exercised at runtime; it matters here only because vendoring removes
what would otherwise be a heavy build dependency.

</details>

Arch note: GB200 = B200 = **`sm_100`**. The fork's `CUDA13_GENCODE` defaults to `sm_110`,
so `NVCC_GENCODE` **must** be overridden.

Two-image pipeline (keeps the slow NCCL build separate from the fast vLLM overlay):

**Step A — FT-NCCL libs artifact image** (`build-ftnccl-libs.yaml`):

```dockerfile
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04 AS ftbuild
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ca-certificates rdma-core libibverbs-dev python3
RUN git clone --depth 80 --branch fault-tolerant-allgather \
      https://github.com/tzulingk/nccl.git /src/nccl
WORKDIR /src/nccl
RUN make -j"$(nproc)" src.build CUDA_HOME=/usr/local/cuda \
      NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"
RUN make -C contrib/fault_tolerant_collectives/ft_handle \
      NCCL_HOME=/src/nccl/build CUDA_HOME=/usr/local/cuda
# → collect build/lib/libnccl.so*, build/include, ft_handle/libft_collective.so,
#   and ft_handle/python/ into /ftnccl
FROM ubuntu:22.04 AS libs
COPY --from=ftbuild /out /ftnccl
```

(The cluster cannot resolve `gitlab-master.nvidia.com`, so the build clones the GitHub
mirror `github.com/tzulingk/nccl`, kept in sync with the fork.)

**Step B — vLLM image overlay.** Build the vLLM image from the `ft-nixl-ep-ftnccl-tp`
branch (precompiled overlay) and, in the final stage:

```dockerfile
COPY --from=<ftnccl-libs-image> /ftnccl /ftnccl
RUN pip install /ftnccl/py            # the pure-python ft_collective package
ENV NCCL_HOME=/ftnccl/nccl \
    FT_COLLECTIVE_LIB=/ftnccl/ft/libft_collective.so \
    LD_LIBRARY_PATH=/ftnccl/nccl/lib:${LD_LIBRARY_PATH}
# LD_PRELOAD is set on the serving pod (Layer 3), not baked in, so non-FT runs
# of the same image are unaffected.
```

## Validation

Validate the foundation **before** trusting the image — in this order:

1. **Build sanity (in the libs build):** `nm -D libft_collective.so` exports
   `ftHandleCreate`/`ftAllReduce`; `nm -D libnccl.so.2` exports
   `ncclCommWindowRegister`/`ncclMemAlloc`.
2. **2-GPU smoke test (the key gate):** on a GB200 node, `LD_PRELOAD` the fork NCCL, then
   `import ft_collective`, `dist.init_process_group("ft_nccl", …)`, allocate
   `t = pg.empty(N)`, run `pg.allreduce([t]).wait()`, and confirm the result is correct
   **and** that the FT path was taken (no NCCL-fallback `UserWarning`). Then kill one rank
   and confirm the survivor's call returns within `FT_TIMEOUT_US` with that rank cleared in
   `get_result_mask()` — proving the timeout-barrier is really linked.
3. **TP=2 serve + kill test:** deploy DeepSeek-V2-Lite at **TP=2** (+ DP + nixl_ep), drive
   sustained load, kill one TP peer, and confirm the survivors fail fast (in-flight
   requests errored/retryable) instead of hanging — the same kill-test methodology used for
   the DP/EP work in [`ft-nixl-ep-runbook.md`](./ft-nixl-ep-runbook.md).

## Scope & limitations

- **Fail-fast only (P0).** An FT timeout masks the dead TP peer so the survivors don't hang,
  but the masked all-reduce is missing that peer's shard — the result is *wrong*, so the
  in-flight requests on that TP group must be errored (retryable). This removes the
  indefinite-hang failure mode; it does not keep the dead TP rank serving.
- **No TP recovery.** A TP peer holds a shard of *every* layer, so surviving a TP death
  in-place would require resharding/reloading the dead shard onto a replacement and
  rebuilding the TP communicator — materially harder than the EP case and **out of scope**
  here. (Contrast: a DP rank is a replica, which is why in-place DP survival works.)
- **Intra-node / LSA.** The TP group is NVLink-local; the FT LSA kernel is the path used.
  Multi-node TP (GIN) is not exercised.
- **`ft_nccl` on TP only.** World/DP/EP/PP stay on the normal backend; cross-DP FT stays on
  FT-gloo. EP all-to-all is handled by the FT NIXL EP kernels, not this path.

## Status

- ✅ FT-NCCL API + build dependencies analyzed; GitHub mirror synced to the fork HEAD.
- 🔧 **Layer 1** — FT-NCCL libs build (`arm64/sm_100`) launched as `build-ftnccl-libs.yaml`
  → `nvcr.io/nvidian/dynamo-dev/tzulingk-ftnccl-libs:cuda13-sm100`.
- ⬜ **Layer 2** — vLLM code changes on branch `ft-nixl-ep-ftnccl-tp` (the four edits above).
- ⬜ **Step B** vLLM image overlay + **Layer 3** env wiring.
- ⬜ 2-GPU smoke test, then TP=2 serve + kill test.

Tracking issue: **DYN-3314** (FT-NCCL for the TP all-reduce).

## References

- [`fault-tolerance-overview.md`](./fault-tolerance-overview.md) — the five FT pieces and the
  three failure scenarios; "FT NCCL on the TP group" is its Path-2 prerequisite.
- [`ft-nixl-ep-runbook.md`](./ft-nixl-ep-runbook.md) — build/serve/kill-test commands.
- FT-NCCL fork: [`tstamler/nccl` MR !2](https://gitlab-master.nvidia.com/tstamler/nccl/-/merge_requests/2),
  `contrib/fault_tolerant_collectives/` (design doc `FT_TP_ONLY_FT_NCCL.md`, `CLAUDE.md`).
- vLLM TP all-reduce (permalinks @ `97038bb6e2`):
  [cuda_communicator.py:254-311](https://github.com/tzulingk/vllm/blob/97038bb6e207b85b4ce31c11e1434525a6aaecff/vllm/distributed/device_communicators/cuda_communicator.py#L254-L311),
  [parallel_state.py:1786-1792](https://github.com/tzulingk/vllm/blob/97038bb6e207b85b4ce31c11e1434525a6aaecff/vllm/distributed/parallel_state.py#L1786-L1792).
