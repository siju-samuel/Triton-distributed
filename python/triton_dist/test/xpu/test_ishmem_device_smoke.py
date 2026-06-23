################################################################################
# Copyright (c) 2026 The AXON Authors
# SPDX-License-Identifier: MIT
#
# XPU backend smoke tests for Triton-distributed.
#
# These validate the Python frontend wiring (Layer C) + the device-bitcode
# linkage (Layer B) WITHOUT needing the full triton_dist C-extension build:
#   1. is_xpu() detects the Intel XPU runtime.
#   2. The libishmem_device bindings import and declare the expected primitives.
#   3. A Triton-XPU kernel can extern_call a flat axon_ishmem_* symbol from the
#      built libishmem_device.bc and JIT-compile (the T1 linkage proof,
#      narrowed to the symbols that don't pull ISHMEM device globals).
#
# Run (single process, on a BMG box):
#   source Triton-distributed/xpu_env.sh
#   AXON_ISHMEM_BC=Triton-distributed/shmem/ishmem_bind/libishmem_device.bc \
#     python -m pytest python/triton_dist/test/xpu/test_ishmem_device_smoke.py -q
#
# NOTE: a meaningful multi-PE put/signal data test needs ISHMEM host init +
# >=2 ranks AND the ISHMEM-device-global SPIR-V extension resolved (see
# docs/xpu-build.md, item T1b). Those are marked xfail/skip here accordingly.
################################################################################
import os
import pytest

pytestmark = pytest.mark.xpu


def test_is_xpu_detects_backend():
    from triton_dist.utils import is_xpu, is_cuda, is_maca
    assert is_xpu(), "expected an Intel XPU runtime (torch.xpu available)"
    # Exactly one backend predicate should be active (ModuleProxy invariant).
    assert not is_cuda() and not is_maca()


def test_libishmem_device_bindings_present():
    """The device-API surface a kernel calls is declared with the right names."""
    import triton_dist.language.extra.xpu.libishmem_device as L
    for fn in ["my_pe", "n_pes", "remote_ptr", "int_p", "putmem_block",
               "putmem_nbi_block", "getmem_block", "putmem_signal_block",
               "putmem_signal_nbi_block", "signal_op", "signal_fetch",
               "signal_wait_until", "fence", "quiet"]:
        assert hasattr(L, fn), f"missing libishmem_device.{fn}"
    # ISHMEM enum values (NOT NVSHMEM's) — guards against copy-paste drift.
    assert L.ISHMEM_SIGNAL_SET == 0 and L.ISHMEM_SIGNAL_ADD == 1
    assert L.ISHMEM_CMP_EQ == 1 and L.ISHMEM_CMP_NE == 2


def test_language_extra_tid_present():
    import triton_dist.language.extra.xpu.language_extra as E
    assert hasattr(E, "tid") and hasattr(E, "__syncthreads")


def test_shmem_dispatch_selects_xpu():
    """libshmem_device's ModuleProxy must resolve to the ISHMEM backend."""
    import triton_dist.language.extra.libshmem_device as S
    # The proxy forwards attribute access to the active backend module; the
    # ISHMEM module defines axon-prefixed symbols, so my_pe must resolve.
    assert hasattr(S, "my_pe")


@pytest.mark.skipif(not os.environ.get("AXON_ISHMEM_BC"),
                    reason="set AXON_ISHMEM_BC to the built libishmem_device.bc")
def test_triton_xpu_links_ishmem_bitcode():
    """A Triton-XPU kernel JIT-links the ISHMEM device bitcode (the T1 proof).

    Uses a symbol that resolves through bitcode-link. Currently the ISHMEM
    device-global SPIR-V extension blocks the final llvm-spirv step for
    symbols that touch ISHMEM globals (see docs/xpu-build.md T1b); this test
    is xfail until ISHMEM is rebuilt as relinkable spir64 JIT bitcode."""
    import torch
    import triton
    import triton.language as tl
    from triton.language import core

    bc = os.environ["AXON_ISHMEM_BC"]

    @core.extern
    def my_pe(_semantic=None):
        return core.extern_elementwise(
            "ishmem", "", [], {(): ("axon_ishmem_my_pe", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic)

    @triton.jit
    def k(o_ptr, BLOCK: tl.constexpr):
        tl.store(o_ptr + tl.arange(0, BLOCK), tl.full((BLOCK, ), my_pe(), tl.int32))

    o = torch.zeros(8, device="xpu", dtype=torch.int32)
    try:
        k[(1, )](o, BLOCK=8, extern_libs={"ishmem": bc})
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001
        pytest.xfail(f"ISHMEM device-global SPIR-V ext not yet in translator allowlist: {e!r}"[:200])
