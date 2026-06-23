# Triton-distributed XPU — MLIR Distributed-dialect lowering (Phase T3)

**Status:** design spec (not yet built — requires the Triton-fork LLVM/MLIR
build; see "Build feasibility" below). This is the op-by-op plan to add
`TritonDistributedTo{TritonGPU,LLVM}/XPU`, mirroring the NVIDIA / AMD / METAX
lowerings already in `lib/Conversion/`.

## Why this phase is gated

The Python frontend (`language/extra/xpu`, Phase T2) and the device binding
(`shmem/ishmem_bind`, Layer B) are usable on their own. But importing the
`triton_dist` **package** pulls `triton._C.libtriton.distributed` — the compiled
**Distributed MLIR dialect** — which only exists when the **Triton fork**
(`3rdparty/triton`, the ByteDance fork) is built with the distributed dialect +
a per-vendor lowering. There is no Intel lowering yet, so this phase adds it.

## The ops to lower (from `Dialect/Distributed/IR/DistributedOps.td`)

| Distributed op | Semantics | XPU lowering |
|---|---|---|
| `tt.get_rank` | current PE id | extern call → `axon_ishmem_my_pe` (bitcode) |
| `tt.get_num_ranks` | PE count | extern call → `axon_ishmem_n_pes` |
| `tt.symm_at` | symmetric addr of a buffer on peer PE | extern call → `axon_ishmem_ptr` (returns peer device ptr) |
| `tt.extern_call` | call a named device comm fn | **generic** pattern (see below) |
| `tt.notify` | update a remote signal word | extern call → `axon_ishmem_signal_op` |
| `tt.wait` | spin until signal `<cmp> val`, emit token | SPIR-V spin-loop (see WaitOp below) |
| `tt.consume_token` | token sink | erase (no-op, like NVIDIA/METAX) |

### The generic ExternCall pattern (the bulk, and the easy part)

METAX's `GenericOpToMXSHMEMDevice` (in
`TritonDistributedToLLVM/METAX/DistributedOpToLLVM.cpp`) is **backend-agnostic
plumbing**: it (1) addrspace-casts pointer operands to the generic addrspace(0)
the bitcode functions expect, (2) builds an `LLVM::LLVMFuncOp` extern decl for
the callee symbol with `libname`/`libpath` attributes, (3) emits `LLVM::call`,
(4) addrspace-casts the result back. The bitcode linker resolves the symbol
later (`extern_libs`). **This pattern ports to XPU almost verbatim** — the only
change is the address-space convention: Intel SPIR-V uses addrspace 1 (global)
/ 4 (generic); the Intel Triton backend's `TritonGPUToLLVMTypeConverter`
already handles SPIR-V address spaces, so the cast targets change from
NVVM's `addrspace(0)` to SPIR-V generic `addrspace(4)`.

The callee symbols are exactly our shim's flat names: `axon_ishmem_putmem`,
`axon_ishmem_putmem_signal`, `axon_ishmem_signal_op`, `axon_ishmem_my_pe`, … —
already built into `libishmem_device.bc`.

### WaitOp (the one genuinely vendor-specific lowering)

METAX's `WaitOpConversion` builds an `scf.while` spin-loop that polls the
barrier word with the requested memory `scope`/`semantic`, then issues a
work-group barrier via `llvm.mxc.barrier.inst` and assumes **warpSize = 64**.
The XPU version changes two things:
- **barrier intrinsic:** `llvm.mxc.barrier.inst` → the SPIR-V control-barrier
  (`__spirv_ControlBarrier`, work-group scope) — or reuse Triton-XPU's existing
  `gpu.barrier` lowering rather than emit a raw intrinsic.
- **sub-group width:** 64 → the BMG/Battlemage SIMD width (**16**; query from
  the target instead of hard-coding). The `laneId = tid % warpSize` lane-gating
  that limits the poll to the first `num_barriers` lanes must use the XPU width.

Everything else in WaitOp (the `scf.while` poll, the acquire/release ordering
map, the gep into the barrier array) is dialect-generic and ports as-is.

### Pass entry + conversion target

Mirror `ConvertMETAXDistributedToLLVM.cpp`: a `ConvertXPUDistributedToLLVM`
pass that sets up a `TritonGPUToLLVMTypeConverter` (index bitwidth 32), marks
the Distributed/Triton/TritonGPU/GPU dialects illegal and LLVM + SPIR-V legal,
then `populateDistributedOpToLLVMPatterns` + the standard arith/math/cf/view/
assert/memory/make_range populates. The METAX file pulls
`mlir::populateGpuToMACAConversionPatterns`; the XPU file pulls the Intel
backend's GPU→SPIR-V/LLVM populate from the fork's `third_party/intel`.

## Files to add (mirrors the METAX commit `6fcf524`)

```
lib/Conversion/TritonDistributedToLLVM/XPU/ConvertXPUDistributedToLLVM.cpp   (~100 LOC; from METAX)
lib/Conversion/TritonDistributedToLLVM/XPU/DistributedOpToLLVM.cpp           (~350 LOC; generic + Wait/Notify retargeted)
lib/Conversion/TritonDistributedToTritonGPU/XPU/TritonDistributedToTritonGPU.cpp  (~1.4k LOC; layout/encoding, the bulk)
include/TritonDistributed/Conversion/.../Passes.{td,h}  (+ GEN_PASS_DEF_CONVERTXPUDISTRIBUTEDTOLLVM)
python/src/passes.cc                                    (register the pass)
CMakeLists.txt                                          (TRITON_USE_XPU gate — scaffolded)
```

The `TritonDistributedToTritonGPU/XPU` file (the largest) handles encoding /
layout conversion. The Intel Triton backend already defines its TritonGPU →
SPIR-V layout machinery; this file adapts the Distributed-op-specific
encodings, largely structurally identical to the NVIDIA/METAX versions.

## Build feasibility (this environment, 2026-06)

**Not buildable in this session.** Building the lowering requires compiling the
ByteDance **Triton fork** (`3rdparty/triton`, not checked out) against LLVM/MLIR
+ the Intel backend — a multi-hour build needing a writable LLVM toolchain. The
readable oneAPI (`/data/ss/basekit_2025.2`) provides `icpx`/SYCL but the fork
build also wants a matching LLVM/MLIR dev tree and the
`intel-xpu-backend-for-triton` sources wired into `third_party/intel`. Per the
project's "don't ship unverified code" rule, the lowering is specified here
rather than committed as untested ~1.8k-LOC MLIR.

**Estimated effort once the fork build is stood up:** the generic ExternCall +
rank/symm_at/notify patterns are ~1–2 days (close to mechanical from METAX);
WaitOp retarget ~1 day; the TritonGPU layout file ~1–2 weeks (the real work);
integration/debug on BMG ~1 week. Total ≈ 3–4 weeks, matching the original
plan's T3 estimate.

## Dependency on T1b

Even with the lowering built, end-to-end in-kernel ISHMEM comm needs the
`libishmem_device.bc` to translate through Triton's `llvm-spirv` step, which
currently rejects ISHMEM's device-global SPIR-V extension
(`SPV_INTEL_global_variable_decorations`) — see `docs/xpu-build.md` (T1b). That
is an ISHMEM-rebuild / translator-allowlist item, independent of this lowering.
