#!/bin/bash
# Copyright (c) 2026 The AXON Authors
# SPDX-License-Identifier: MIT
#
# Build the ISHMEM device-bitcode library for Triton-distributed on Intel XPU.
#
# Produces `libishmem_device.bc` — a self-contained SPIR-V LLVM bitcode module
# of flat `axon_ishmem_*` C-ABI device symbols (from ishmemi/ishmem_device_shim.cpp)
# with ISHMEM's own device implementations llvm-link'd in. Triton links this via
# `extern_libs` at JIT time so a generated XPU kernel can call ISHMEM device
# routines by name — the Intel analogue of NVSHMEM's `libnvshmem_device.bc`.
#
# Mirrors shmem/nvshmem_bind/nvshmemi/build_nvshmemi_bc.sh and
# shmem/mxshmem_bind/build.sh.
#
# Env:
#   ISHMEM_DIR  : ISHMEM install prefix (has include/ishmem.h + lib/libishmem.a)
#   ONEAPI_BIN  : dir with icpx + compiler/llvm-{link,nm,ar}, clang-offload-bundler
#                 (default: derived from `icpx` on PATH)
#
# Output: ./libishmem_device.bc  (also copied to the triton intel backend lib dir
#         if TRITON_INTEL_LIB is set).
set -euo pipefail

CUR_DIR="$(cd "$(dirname "$0")" && pwd)"
ISHMEM_DIR="${ISHMEM_DIR:-}"
if [[ -z "$ISHMEM_DIR" || ! -f "$ISHMEM_DIR/include/ishmem.h" ]]; then
  echo "ERROR: set ISHMEM_DIR to an ISHMEM install prefix (with include/ishmem.h + lib/libishmem.a)." >&2
  exit 2
fi

ICPX="$(command -v icpx || true)"
if [[ -z "$ICPX" ]]; then
  echo "ERROR: icpx not on PATH. Source an Intel oneAPI/DPC++ env first." >&2
  exit 2
fi
ONEAPI_BIN="${ONEAPI_BIN:-$(dirname "$ICPX")}"
CBIN="$ONEAPI_BIN/compiler"
LLVM_LINK="$CBIN/llvm-link"
LLVM_AR="$CBIN/llvm-ar"
LLVM_NM="$CBIN/llvm-nm"
BUNDLER="$CBIN/clang-offload-bundler"

WORK="$CUR_DIR/build"
rm -rf "$WORK" && mkdir -p "$WORK"

echo "[ishmem_bind] 1/3  compile C-ABI shim -> device bitcode (spir64 JIT)"
# NOTE: spir64 (JIT bitcode), NOT spir64_gen (AOT), so the result is relinkable
# LLVM bitcode that Triton's link_extern_libs / llvm-spirv pipeline can consume.
"$ICPX" -fsycl -fsycl-targets=spir64 -fsycl-device-only -fno-sycl-instrument-device-code \
        -I"$ISHMEM_DIR/include" \
        -c "$CUR_DIR/ishmemi/ishmem_device_shim.cpp" -o "$WORK/shim.bc"

echo "[ishmem_bind] 2/3  extract ISHMEM device images from libishmem.a"
pushd "$WORK" >/dev/null
"$LLVM_AR" x "$ISHMEM_DIR/lib/libishmem.a"
imgs=()
for o in *.o; do
  # Each ISHMEM .o is a SYCL fat object; pull the device (spir64_gen) image,
  # which is itself LLVM bitcode (verified: magic 0x4243c0de).
  if "$BUNDLER" --type=o --input="$o" \
        --targets=sycl-spir64_gen-unknown-unknown --output="$o.dev.bc" --unbundle 2>/dev/null; then
    imgs+=("$o.dev.bc")
  fi
done
echo "[ishmem_bind]      extracted ${#imgs[@]} device images"
popd >/dev/null

echo "[ishmem_bind] 3/3  llvm-link shim + ISHMEM device images -> libishmem_device.bc"
# (Triples differ: shim is spir64, images are spir64_gen — llvm-link warns but
#  links; both are SPIR LLVM bitcode.)
"$LLVM_LINK" "$WORK/shim.bc" "${imgs[@]/#/$WORK/}" -o "$CUR_DIR/libishmem_device.bc" 2>/dev/null

echo "[ishmem_bind] verify: undefined non-template ISHMEM symbols (want 0):"
UNDEF=$("$LLVM_NM" "$CUR_DIR/libishmem_device.bc" 2>/dev/null | { grep -E " U " || true; } \
          | { grep -iE "ishmem" || true; } | { grep -vE "work_group|sub_group|group" || true; } | wc -l)
echo "[ishmem_bind]      undefined = $UNDEF"
echo "[ishmem_bind] defined axon_* device symbols (comm + thread-id intrinsics):"
"$LLVM_NM" "$CUR_DIR/libishmem_device.bc" 2>/dev/null | { grep -E " T axon_" || true; } | awk '{print "  "$3}'

if [[ -n "${TRITON_INTEL_LIB:-}" ]]; then
  cp "$CUR_DIR/libishmem_device.bc" "$TRITON_INTEL_LIB/"
  echo "[ishmem_bind] copied to $TRITON_INTEL_LIB/"
fi

echo "[ishmem_bind] DONE -> $CUR_DIR/libishmem_device.bc"
echo "[ishmem_bind] NOTE: see docs/xpu-build.md — a Triton-XPU kernel can link"
echo "[ishmem_bind]       this, but the llvm-spirv translation of ISHMEM device"
echo "[ishmem_bind]       globals currently needs SPV_INTEL_global_variable_decorations"
echo "[ishmem_bind]       in the translator allowlist (residual integration item T1b)."
