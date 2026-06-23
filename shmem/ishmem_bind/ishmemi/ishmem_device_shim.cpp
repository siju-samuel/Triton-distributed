/*
 * Copyright (c) 2026 The AXON Authors
 * SPDX-License-Identifier: MIT
 *
 * ISHMEM C-ABI device shim for Triton-distributed on Intel XPU.
 *
 * Triton-distributed device kernels call SHMEM device primitives by name
 * (e.g. NVSHMEM's `nvshmemx_putmem_signal_block`) through Triton's
 * `extern_call` / `extern_elementwise`, which links a device-bitcode library
 * into the generated kernel and references the comm functions by their FLAT,
 * UNMANGLED C symbol. NVSHMEM ships exactly that (`libnvshmem_device.bc`);
 * AMD's mori/rocSHMEM and MetaX's mxshmem do the same.
 *
 * Intel SHMEM (ISHMEM)'s device API, however, is a SYCL/C++ API — templated
 * functions taking `sycl::group`, with C++-mangled names. This shim wraps the
 * subset Triton-distributed needs behind `extern "C" SYCL_EXTERNAL` symbols
 * (prefix `axon_ishmem_`) so the names are stable and unmangled, exactly like
 * the other backends' device bitcode.
 *
 * Build: compiled by `build.sh` to SPIR-V LLVM bitcode and llvm-link'd with
 * ISHMEM's own device images (extracted from libishmem.a) into one
 * self-contained `libishmem_device.bc`. See build.sh for the device-link
 * recipe and docs/xpu-build.md for the integration status.
 *
 * Subset rationale: this is the minimal-viable set (≈ what MetaX's
 * libmxshmem_device ships) for AllGather/EP kernels — identity, put/get,
 * put-with-signal, signal op + wait, and ordering. Grow as kernels need more.
 */

#include <sycl/sycl.hpp>
#include <ishmem.h>
#include <ishmemx.h>

#include <cstddef>
#include <cstdint>

extern "C" {

// ---- identity / topology --------------------------------------------------
SYCL_EXTERNAL int axon_ishmem_my_pe() { return ishmem_my_pe(); }
SYCL_EXTERNAL int axon_ishmem_n_pes() { return ishmem_n_pes(); }
SYCL_EXTERNAL void *axon_ishmem_ptr(const void *dest, int pe) {
    return ishmem_ptr(dest, pe);
}

// ---- scalar put (the NVSHMEM `int_p` analogue) ----------------------------
SYCL_EXTERNAL void axon_ishmem_int_p(int *dest, int value, int pe) {
    ishmem_int_p(dest, value, pe);
}

// ---- bulk put / get (blocking + non-blocking) -----------------------------
SYCL_EXTERNAL void axon_ishmem_putmem(void *dest, const void *source, size_t nbytes, int pe) {
    ishmem_putmem(dest, source, nbytes, pe);
}
SYCL_EXTERNAL void axon_ishmem_putmem_nbi(void *dest, const void *source, size_t nbytes, int pe) {
    ishmem_putmem_nbi(dest, source, nbytes, pe);
}
SYCL_EXTERNAL void axon_ishmem_getmem(void *dest, const void *source, size_t nbytes, int pe) {
    ishmem_getmem(dest, source, nbytes, pe);
}
SYCL_EXTERNAL void axon_ishmem_getmem_nbi(void *dest, const void *source, size_t nbytes, int pe) {
    ishmem_getmem_nbi(dest, source, nbytes, pe);
}

// ---- put-with-signal (data + remote signal in one op) ---------------------
// ISHMEM uses uint64 signal words and sig_op { ISHMEM_SIGNAL_SET=0,
// ISHMEM_SIGNAL_ADD=1 } — see libishmem_device.py for the Python-side enum.
SYCL_EXTERNAL void axon_ishmem_putmem_signal(void *dest, const void *source, size_t nbytes,
                                             uint64_t *sig_addr, uint64_t signal, int sig_op,
                                             int pe) {
    ishmem_putmem_signal(dest, source, nbytes, sig_addr, signal, sig_op, pe);
}
SYCL_EXTERNAL void axon_ishmem_putmem_signal_nbi(void *dest, const void *source, size_t nbytes,
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                 int pe) {
    ishmem_putmem_signal_nbi(dest, source, nbytes, sig_addr, signal, sig_op, pe);
}

// ---- signal op + wait -----------------------------------------------------
SYCL_EXTERNAL void axon_ishmem_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    // ISHMEM exposes signal-set/add via the typed atomic + signal API; the
    // putmem_signal path above carries the common case. A standalone signal
    // update maps to ishmemx_signal_set/add on the uint64 signal word.
    if (sig_op == ISHMEM_SIGNAL_ADD)
        ishmemx_signal_add(sig_addr, signal, pe);
    else
        ishmemx_signal_set(sig_addr, signal, pe);
}
SYCL_EXTERNAL uint64_t axon_ishmem_signal_fetch(uint64_t *sig_addr) {
    return ishmem_signal_fetch(sig_addr);
}
// Spin until *sig_addr <cmp> cmp_val. cmp uses ISHMEM_CMP_* (ishmem.h):
//   EQ=1, NE=2, GT=3, GE=4, LT=5, LE=6.
SYCL_EXTERNAL void axon_ishmem_uint64_wait_until(uint64_t *ivar, int cmp, uint64_t cmp_val) {
    ishmem_uint64_wait_until(ivar, cmp, cmp_val);
}

// ---- ordering -------------------------------------------------------------
SYCL_EXTERNAL void axon_ishmem_fence() { ishmem_fence(); }
SYCL_EXTERNAL void axon_ishmem_quiet() { ishmem_quiet(); }

// ---- device thread-id intrinsics (SPIR-V) ---------------------------------
// Triton-distributed kernels need the in-work-group thread id (NVSHMEM/MACA
// expose it as `tid(axis)`). On SPIR-V the local invocation id is a builtin;
// SYCL surfaces it via `this_work_item::get_nd_item()`. We wrap it as flat
// C-ABI so language_extra.py's `tid()` binds identically to the other backends.
SYCL_EXTERNAL int axon_get_local_id_x() {
    return static_cast<int>(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2));
}
SYCL_EXTERNAL int axon_get_local_id_y() {
    return static_cast<int>(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(1));
}
SYCL_EXTERNAL int axon_get_local_id_z() {
    return static_cast<int>(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(0));
}

}  // extern "C"
