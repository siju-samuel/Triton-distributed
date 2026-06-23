################################################################################
#
# Copyright (c) 2026 The AXON Authors
# SPDX-License-Identifier: MIT
#
# ISHMEM device-side bindings for Triton-distributed on Intel XPU.
#
# Each function declares an `extern_call` into the `libishmem_device` bitcode
# (built by shmem/ishmem_bind/build.sh) by its flat C-ABI symbol
# (`axon_ishmem_*`). This mirrors triton_dist.language.extra.maca.
# libmxshmem_device (MetaX) and .cuda.libnvshmem_device (NVIDIA): the only
# backend-specific part is the lib key + symbol names; the kernel-facing API
# (my_pe, putmem, putmem_signal_block, signal_op, signal_wait_until, fence,
# quiet, ...) is identical across backends and dispatched through ModuleProxy.
#
# This is the minimal-viable subset (≈ the MetaX subset) — identity, put/get,
# put-with-signal, signal op + wait, ordering — enough for AllGather and EP
# all-to-all kernels. Grow as kernels require more.
#
# Signal opcodes / compare ops follow ISHMEM's headers (NOT NVSHMEM's):
#   ISHMEM_SIGNAL_SET = 0, ISHMEM_SIGNAL_ADD = 1
#   ISHMEM_CMP_EQ=1, NE=2, GT=3, GE=4, LT=5, LE=6
################################################################################
from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call
import sys

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))
void_ptr = core.pointer_type(core.void)

# --- ISHMEM compare ops (src/ishmem.h) -------------------------------------
ISHMEM_CMP_EQ = 1
ISHMEM_CMP_NE = 2
ISHMEM_CMP_GT = 3
ISHMEM_CMP_GE = 4
ISHMEM_CMP_LT = 5
ISHMEM_CMP_LE = 6
ISHMEM_CMP_SENTINEL = sys.maxsize

# --- ISHMEM signal ops (src/ishmem.h) --------------------------------------
ISHMEM_SIGNAL_SET = 0
ISHMEM_SIGNAL_ADD = 1

# Compatibility aliases so backend-agnostic kernels written against the NVSHMEM
# naming still resolve (values are ISHMEM's, not NVSHMEM's).
NVSHMEM_CMP_EQ = ISHMEM_CMP_EQ
NVSHMEM_SIGNAL_SET = ISHMEM_SIGNAL_SET
NVSHMEM_SIGNAL_ADD = ISHMEM_SIGNAL_ADD

_LIB = "libishmem_device"


@core.extern
def my_pe(_semantic=None):
    return extern_call(_LIB, "", [], {(): ("axon_ishmem_my_pe", core.dtype("int32"))},
                       is_pure=True, _semantic=_semantic)


@core.extern
def n_pes(_semantic=None):
    return extern_call(_LIB, "", [], {(): ("axon_ishmem_n_pes", core.dtype("int32"))},
                       is_pure=True, _semantic=_semantic)


@core.extern
def remote_ptr(local_ptr, pe, _semantic=None):
    return extern_call(
        _LIB, "", [local_ptr, pe], {
            (core.pointer_type(core.dtype(core_dtype)), core.dtype(pe_dtype)):
            ("axon_ishmem_ptr", core.pointer_type(core.dtype(core_dtype)))
            for core_dtype in core.dtype.SINT_TYPES + core.dtype.UINT_TYPES + core.dtype.FP_TYPES +
            core.dtype.OTHER_TYPES
            for pe_dtype in ["int32", "uint32"]
        }, is_pure=False, _semantic=_semantic)


@core.extern
def int_p(dest, value, pe, _semantic=None):
    return extern_call(
        _LIB, "", [dest, value, pe], {
            (core.pointer_type(core.dtype("int32")), core.dtype("int32"), core.dtype("int32")):
            ("axon_ishmem_int_p", ())
        }, is_pure=False, _semantic=_semantic)


def _putget(symbol):
    """Builder for the putmem/getmem family (dest, source, nbytes, pe) -> void."""

    def _fn(dest, source, nbytes, pe, _semantic=None):
        return extern_call(
            _LIB, "", [
                tl.cast(dest, tl.pointer_type(tl.int8), _builder=_semantic.builder),
                tl.cast(source, tl.pointer_type(tl.int8), _builder=_semantic.builder),
                tl.cast(nbytes, tl.uint64, _builder=_semantic.builder),
                tl.cast(pe, tl.int32, _builder=_semantic.builder),
            ], {
                (tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64, tl.int32):
                (symbol, ())
            }, is_pure=False, _semantic=_semantic)

    return _fn


putmem_block = core.extern(_putget("axon_ishmem_putmem"))
putmem_nbi_block = core.extern(_putget("axon_ishmem_putmem_nbi"))
getmem_block = core.extern(_putget("axon_ishmem_getmem"))
getmem_nbi_block = core.extern(_putget("axon_ishmem_getmem_nbi"))


def _putsignal(symbol):
    """Builder for putmem_signal{,_nbi} (dest, src, nbytes, sig_addr, signal, sig_op, pe)."""

    def _fn(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
        return extern_call(
            _LIB, "", [
                tl.cast(dest, tl.pointer_type(tl.int8), _builder=_semantic.builder),
                tl.cast(source, tl.pointer_type(tl.int8), _builder=_semantic.builder),
                tl.cast(nbytes, tl.uint64, _builder=_semantic.builder),
                sig_addr,  # uint64* — no cast (must stay aligned)
                tl.cast(signal, tl.uint64, _builder=_semantic.builder),
                tl.cast(sig_op, tl.int32, _builder=_semantic.builder),
                tl.cast(pe, tl.int32, _builder=_semantic.builder),
            ], {
                (tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64, pi_u64_t, tl.uint64,
                 tl.int32, tl.int32): (symbol, ())
            }, is_pure=False, _semantic=_semantic)

    return _fn


putmem_signal_block = core.extern(_putsignal("axon_ishmem_putmem_signal"))
putmem_signal_nbi_block = core.extern(_putsignal("axon_ishmem_putmem_signal_nbi"))


@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        _LIB, "", [
            sig_addr,
            tl.cast(signal, tl.uint64, _builder=_semantic.builder),
            tl.cast(sig_op, tl.int32, _builder=_semantic.builder),
            tl.cast(pe, tl.int32, _builder=_semantic.builder),
        ], {(pi_u64_t, tl.uint64, tl.int32, tl.int32): ("axon_ishmem_signal_op", ())},
        is_pure=False, _semantic=_semantic)


@core.extern
def signal_fetch(sig_addr, _semantic=None):
    return extern_call(_LIB, "", [sig_addr],
                       {(pi_u64_t, ): ("axon_ishmem_signal_fetch", tl.uint64)},
                       is_pure=False, _semantic=_semantic)


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
    return extern_call(
        _LIB, "", [
            sig_addr,
            tl.cast(cmp_, tl.int32, _builder=_semantic.builder),
            tl.cast(cmp_val, tl.uint64, _builder=_semantic.builder),
        ], {(pi_u64_t, tl.int32, tl.uint64): ("axon_ishmem_uint64_wait_until", ())},
        is_pure=False, _semantic=_semantic)


@core.extern
def fence(_semantic=None):
    return extern_call(_LIB, "", [], {(): ("axon_ishmem_fence", ())}, is_pure=False,
                       _semantic=_semantic)


@core.extern
def quiet(_semantic=None):
    return extern_call(_LIB, "", [], {(): ("axon_ishmem_quiet", ())}, is_pure=False,
                       _semantic=_semantic)
