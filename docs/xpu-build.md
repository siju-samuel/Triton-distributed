# Triton-distributed on Intel XPU — build, status & next steps

This documents the Intel-XPU (Battlemage / BMG, PVC, Xe2) port of
Triton-distributed: what works, what's blocked, the exact recipe, and the one
residual toolchain wall. Authored against upstream `main` @ `2b4c24b` on a
4×–8× BMG dev box (oneAPI 2025.2 basekit, torch 2.11+xpu, Triton 3.7).

Hardware-honest: every "works" claim below was run on this box; blocked items
say so plainly.

## TL;DR status

| Phase | What | Status |
|---|---|---|
| **T0** | XPU-Triton toolchain env; plain Triton-XPU kernel on BMG | ✅ **works** (`xpu_env.sh`) |
| **T1** | Triton-XPU `extern_call` of a custom SYCL device fn via SPIR-V bitcode | ✅ **PROVEN on BMG** |
| **T1b** | ISHMEM C-ABI device shim → one self-contained `libishmem_device.bc` | ✅ builds, 0 undefined symbols |
| | …its in-kernel SPIR-V translation | ⛔ **blocked**: ISHMEM device-global needs `SPV_INTEL_global_variable_decorations` in Triton's `llvm-spirv` allowlist |
| **Layer B** | `shmem/ishmem_bind/` (shim + `build.sh`) | ✅ reproducible artifact |
| **T2 / Layer C** | Python frontend: `language/extra/xpu`, `is_xpu()`, ModuleProxy | ✅ imports clean (standalone) |
| **T3** | MLIR Distributed-dialect XPU lowering (`…/XPU`) | 📋 **specced** (`xpu-mlir-lowering.md`); needs Triton-fork build |
| **T4** | Host-orchestrated AllGather on BMG (Triton compute + peer comm) | ✅ **PASSES** (world 2,4) |
| **T5** | Host-orchestrated all-to-all + EP dispatch/combine on BMG | ✅ **PASSES** (world 2,4) |
| **T6** | Cross-node (IB) | ⛔ hardware-gated (no IB NIC here → `anbmghdr`) |

**One-line:** every layer of the port is built and the compute+comm substrate
runs on BMG; the single thing between here and in-kernel fused ISHMEM comm is a
SPIR-V translator extension for ISHMEM's device globals (T1b) — an
ISHMEM-rebuild / allowlist item, not a design dead-end.

## The environment (T0)

`/opt/intel/oneapi` (→ `/home/sdp`) is permission-locked for this user; the
readable oneAPI is `/data/ss/basekit_2025.2`. torch 2.11+xpu (conda env
`my_env_py_3_12`) ships its own SYCL runtime. Sourcing `setvars.sh` shadows
torch's runtime and breaks `import torch`. So `xpu_env.sh` exposes ONLY the
compiler (`icpx` on PATH, SYCL headers on CPATH) and leaves `LD_LIBRARY_PATH`
alone:

```sh
source Triton-distributed/xpu_env.sh
python Triton-distributed/python/triton_dist/test/xpu/test_allgather_xpu_hostorch.py
```

Verified: a plain `@triton.jit` kernel JIT-compiles to SPIR-V and runs
correctly on BMG (8 devices visible).

## The device-comm linkage proof (T1) — the make-or-break result

Triton-distributed's model = link a device-bitcode library of flat C-ABI comm
functions and call them from inside the generated kernel (NVSHMEM ships
`libnvshmem_device.bc`). We proved this mechanism works on Intel XPU:

1. A custom `extern "C" SYCL_EXTERNAL int axon_shim_add_one(int)` compiled with
   `icpx -fsycl -fsycl-targets=spir64 -fsycl-device-only` → LLVM bitcode.
2. A Triton-XPU kernel `extern_elementwise`-calling it with
   `extern_libs={"shim": "shim_dev.bc"}` **compiled, linked, and ran correctly
   on BMG** (`o == x+1`). Triton's `make_llir` → `llvm.link_extern_libs` →
   SPIR-V path consumes the bitcode exactly like the CUDA NVSHMEM path.

This is the load-bearing assumption of the whole port, and it holds.

## The ISHMEM device shim (T1b / Layer B)

`shmem/ishmem_bind/`:
- `ishmemi/ishmem_device_shim.cpp` — 15 ISHMEM comm primitives + 3 thread-id
  intrinsics, each `extern "C" SYCL_EXTERNAL` with a flat `axon_*` name
  (verified against `ishmem.h`/`ishmemx.h`).
- `build.sh` — compiles the shim to `spir64` JIT bitcode, extracts ISHMEM's 30
  device images from `libishmem.a` (via `clang-offload-bundler`; they are
  themselves LLVM bitcode), and `llvm-link`s into one `libishmem_device.bc`
  with **0 undefined non-template ISHMEM symbols**.

```sh
source Triton-distributed/xpu_env.sh
ISHMEM_DIR=/path/to/ishmem/install ./Triton-distributed/shmem/ishmem_bind/build.sh
# -> shmem/ishmem_bind/libishmem_device.bc
```

### The residual wall (T1b) — precise

Linking `libishmem_device.bc` into a Triton-XPU kernel and JIT-compiling fails
at the SPIR-V translation step:

```
RequiresExtension: Feature requires the following SPIR-V extension:
 SPV_INTEL_global_variable_decorations
```

ISHMEM's device code uses decorated device **global variables** (its
symmetric-heap / PE state). ISHMEM itself AOT-compiles for `spir64_gen`
(`xe-hpc,xe2`) and is fine; but Triton's bundled `llvm-spirv` translator
(`intel.translate_to_spirv`, a compiled `_C` call) does not enable that
extension. Even `my_pe()` alone trips it (it reads a device global).

**Resolutions (either unblocks in-kernel comm):**
1. **Rebuild ISHMEM as relinkable `-fsycl-targets=spir64` JIT bitcode** with the
   device-globals lowered compatibly, OR build a thin ISHMEM device variant
   whose state is passed by pointer instead of device globals.
2. **Extend Triton-XPU's `llvm-spirv` allowlist** to include
   `SPV_INTEL_global_variable_decorations` (a translator flag / a patch to the
   intel-xpu-backend-for-triton SPIR-V step).

Both are bounded toolchain tasks. Neither changes the port's design; the shim,
build, frontend, and lowering spec are all ready to use the bitcode the moment
the translation passes.

## Python frontend (T2 / Layer C)

`python/triton_dist/language/extra/xpu/{libishmem_device,language_extra,__init__}.py`
+ `is_xpu()` in `utils.py` + ModuleProxy registration in `libshmem_device.py`.
Mirrors the MetaX (`maca`) backend exactly. The 15 device primitives + `tid`
import cleanly standalone (validated — see
`test/xpu/test_ishmem_device_smoke.py`). Note: importing through the full
`triton_dist` package needs the compiled Distributed dialect (T3 fork build);
until then the leaf modules import directly.

## Runnable multi-GPU substrate (T4/T5)

`test/xpu/test_allgather_xpu_hostorch.py` and `test_all_to_all_xpu_hostorch.py`
run the AllGather / all-to-all / EP-dispatch-combine data-movement patterns in
the **host-orchestrated** form (Triton-XPU compute kernels + torch.xpu peer
copies — the `host-proxy` access method). **All pass on 2 and 4 BMG GPUs.**
These are the correctness oracle the future in-kernel (T3+T1b) fused kernels
must match, and they prove the BMG multi-GPU substrate end to end today.

Run as scripts (pytest collection needs the fork-built `triton_dist`):
```sh
source Triton-distributed/xpu_env.sh
python Triton-distributed/python/triton_dist/test/xpu/test_allgather_xpu_hostorch.py
python Triton-distributed/python/triton_dist/test/xpu/test_all_to_all_xpu_hostorch.py
```

## MLIR lowering (T3)

Specced in `docs/xpu-mlir-lowering.md` (op-by-op NVVM/MACA→SPIR-V map, the
WaitOp sub-group-width + barrier retarget, the file list mirroring the MetaX
commit). The `TRITON_USE_XPU` CMake gate is scaffolded. The lowering itself
needs the Triton-fork LLVM/MLIR build (multi-hour, not stood up here), so it is
specified rather than committed as untested MLIR (project rule: no unverified
code).

## Recommended next steps (in order)

1. **Unblock T1b** — rebuild ISHMEM as `spir64` JIT bitcode (or patch the
   Triton-XPU SPIR-V allowlist), then flip
   `test_ishmem_device_smoke::test_triton_xpu_links_ishmem_bitcode` from xfail
   to a real 2-rank `my_pe`/`putmem`/`signal_wait_until` data test on 2× BMG.
2. **Stand up the Triton-fork build** with `third_party/intel`, implement T3 per
   `xpu-mlir-lowering.md` (generic ExternCall first → WaitOp → layout file).
3. **Port the first fused kernel** (`kernels/nvidia/allgather_gemm.py` →
   `kernels/xpu/`, oneDNN/Triton-XPU GEMM) and validate vs the T4 oracle.
4. **EP all-to-all** fused kernel (DeepEP-style) — the AXON-EP convergence
   point.
5. **Cross-node** on `anbmghdr` (IB) — no "works" claim until measured there.
