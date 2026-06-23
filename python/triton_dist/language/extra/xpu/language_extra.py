################################################################################
#
# Copyright (c) 2026 The AXON Authors
# SPDX-License-Identifier: MIT
#
# Device intrinsics for Triton-distributed on Intel XPU. Mirrors
# triton_dist.language.extra.maca.language_extra and .cuda.language_extra: the
# kernel-facing names (`tid`, `__syncthreads`) are identical across backends;
# only the lowering differs.
#
# On NVIDIA, tid maps to `llvm.nvvm.read.ptx.sreg.tid.{x,y,z}`; on MACA to
# `llvm.mxc.thread.id.{x,y,z}`. On Intel/SPIR-V the local invocation id is a
# SPIR-V builtin with no stable LLVM intrinsic name usable from Triton, so we
# bind flat C-ABI wrappers (`axon_get_local_id_{x,y,z}`) from the
# libishmem_device bitcode (built by shmem/ishmem_bind) — the same extern-call
# mechanism the SHMEM primitives use.
################################################################################

import triton.language as tl
from triton.language import core
from triton_dist.language.core import extern_call

_LIB = "libishmem_device"


@core.extern
def __syncthreads(_semantic=None):
    # Work-group barrier. Triton's debug_barrier lowers to the SPIR-V
    # control-barrier on XPU.
    return tl.debug_barrier(_semantic=_semantic)


@core.extern
def __tid__(axis: core.constexpr, _semantic=None):
    sym = {"x": "axon_get_local_id_x", "y": "axon_get_local_id_y", "z": "axon_get_local_id_z"}[axis.value]
    return extern_call(_LIB, "", [], {(): (sym, core.dtype("int32"))}, is_pure=True,
                       _semantic=_semantic)


@core.extern
def tid(axis: core.constexpr, _semantic=None):
    if axis == 0:
        return __tid__(core.constexpr("x"), _semantic=_semantic)
    elif axis == 1:
        return __tid__(core.constexpr("y"), _semantic=_semantic)
    elif axis == 2:
        return __tid__(core.constexpr("z"), _semantic=_semantic)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2")


__all__ = [
    "__syncthreads",
    "tid",
]
