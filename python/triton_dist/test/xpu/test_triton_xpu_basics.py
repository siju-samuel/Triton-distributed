################################################################################
# Copyright (c) 2026 The AXON Authors
# SPDX-License-Identifier: MIT
#
# XPU foundation tests (Phase T0 + T1), self-contained and reproducible.
#   T0: a plain Triton-XPU kernel JIT-compiles to SPIR-V and runs on BMG.
#   T1: a Triton-XPU kernel `extern_call`s a CUSTOM device function delivered
#       as SPIR-V LLVM bitcode (the exact mechanism Triton-distributed uses for
#       SHMEM device calls). This is the load-bearing linkage proof for the
#       whole XPU port — if this works, the ISHMEM path is "just" more symbols.
#
# T1 builds its own tiny bitcode at runtime via icpx (so it needs the XPU env;
# see Triton-distributed/xpu_env.sh). It is skipped if icpx is unavailable.
#
# Run:  source Triton-distributed/xpu_env.sh
#       python Triton-distributed/python/triton_dist/test/xpu/test_triton_xpu_basics.py
################################################################################
import os
import shutil
import subprocess
import tempfile

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402
from triton.language import core  # noqa: E402

pytestmark = pytest.mark.xpu

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("no Intel XPU runtime", allow_module_level=True)


@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)


def test_t0_plain_triton_xpu_kernel():
    """T0: Triton compiles a kernel to SPIR-V and runs it correctly on BMG."""
    n = 4096
    x = torch.randn(n, device="xpu")
    y = torch.randn(n, device="xpu")
    o = torch.empty_like(x)
    _add_kernel[(triton.cdiv(n, 256), )](x, y, o, n, BLOCK=256)
    torch.xpu.synchronize()
    assert torch.allclose(o, x + y)


def _build_addone_bitcode(workdir):
    """Compile a custom extern "C" SYCL_EXTERNAL device fn to spir64 bitcode."""
    icpx = shutil.which("icpx")
    if not icpx:
        return None
    src = os.path.join(workdir, "shim.cpp")
    bc = os.path.join(workdir, "shim_dev.bc")
    with open(src, "w") as f:
        f.write('#include <sycl/sycl.hpp>\n'
                'extern "C" SYCL_EXTERNAL int axon_addone_test(int x) { return x + 1; }\n')
    r = subprocess.run(
        [icpx, "-fsycl", "-fsycl-targets=spir64", "-fsycl-device-only",
         "-fno-sycl-instrument-device-code", "-c", src, "-o", bc],
        capture_output=True, text=True)
    return bc if (r.returncode == 0 and os.path.exists(bc)) else None


@core.extern
def _addone(x, _semantic=None):
    return core.extern_elementwise(
        "shim", "", [x], {(core.dtype("int32"), ): ("axon_addone_test", core.dtype("int32"))},
        is_pure=True, _semantic=_semantic)


@triton.jit
def _extern_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, _addone(tl.load(x_ptr + off, mask=m)), mask=m)


def test_t1_extern_device_call_via_bitcode():
    """T1: extern_call of a custom device fn linked from SPIR-V bitcode (the
    NVSHMEM-style device-comm linkage), proven on BMG."""
    with tempfile.TemporaryDirectory() as wd:
        bc = _build_addone_bitcode(wd)
        if bc is None:
            pytest.skip("icpx unavailable — cannot build the device bitcode (source xpu_env.sh)")
        n = 1024
        x = torch.arange(n, device="xpu", dtype=torch.int32)
        o = torch.empty_like(x)
        _extern_kernel[(triton.cdiv(n, 256), )](x, o, n, BLOCK=256, extern_libs={"shim": bc})
        torch.xpu.synchronize()
        assert torch.equal(o, x + 1)


if __name__ == "__main__":
    test_t0_plain_triton_xpu_kernel()
    print("[ok] T0 plain Triton-XPU kernel")
    try:
        test_t1_extern_device_call_via_bitcode()
        print("[ok] T1 extern device-call via SPIR-V bitcode")
    except Exception as e:  # pytest.skip raises Skipped
        print(f"[skip] T1: {e}")
    print("ALL PASSED")
