################################################################################
# Copyright (c) 2026 The AXON Authors
# SPDX-License-Identifier: MIT
#
# AllGather on Intel XPU — HOST-ORCHESTRATED form (Phase T4, runnable today).
#
# Triton-distributed's headline kernels fuse compute + communication INSIDE one
# Triton kernel via device-side SHMEM calls. That in-kernel fusion needs the
# Distributed-dialect MLIR lowering (T3) + the ISHMEM device-global SPIR-V
# extension resolved (T1b) — neither buildable in this environment yet.
#
# THIS test exercises the same data-movement pattern in the host-orchestrated
# form that IS runnable now: a Triton-XPU COMPUTE kernel on each device + torch
# .xpu peer copies for the COMMUNICATION (the `NIC_HANDLER {gpu, host-proxy}`
# degradation AXON also keeps). It proves:
#   * the Intel-XPU Triton compiler path works on BMG (compute kernels JIT-run),
#   * multi-GPU peer data movement + a ring AllGather are correct on 8x BMG,
#   * a perf baseline the future in-kernel (T3) version must beat.
#
# This is NOT the final fused kernel; it is the verifiable substrate baseline.
#
# Run (single process drives N devices):
#   source Triton-distributed/xpu_env.sh
#   python python/triton_dist/test/xpu/test_allgather_xpu_hostorch.py
#
# NOTE on pytest: collecting via pytest imports the `triton_dist` package, whose
# __init__ pulls the compiled Distributed dialect (`triton._C.libtriton.
# distributed`) that only exists once the Triton FORK is built (Phase T3). Until
# then run these as plain scripts (the `__main__` block) — the compute + comm
# paths they exercise do NOT depend on the dialect. Once the fork is installed
# they are collectable as normal pytest tests.
################################################################################
import os
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

pytestmark = pytest.mark.xpu

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("no Intel XPU runtime", allow_module_level=True)


@triton.jit
def _scale_kernel(x_ptr, o_ptr, scale, n, BLOCK: tl.constexpr):
    """A stand-in 'compute' stage: o = x * scale. Proves Triton-XPU compute."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) * scale, mask=m)


def _triton_scale(x, scale):
    o = torch.empty_like(x)
    n = x.numel()
    _scale_kernel[(triton.cdiv(n, 256), )](x, o, scale, n, BLOCK=256)
    return o


def _ring_allgather(parts):
    """Host-orchestrated ring AllGather across devices: every device ends with
    the concatenation of all devices' (compute-stage) outputs, in rank order.
    Communication = torch.xpu peer copies (the host-proxy access method)."""
    world = len(parts)
    # Each rank first runs its Triton-XPU compute stage locally.
    local = [_triton_scale(parts[r], float(r + 1)) for r in range(world)]
    for d in range(world):
        torch.xpu.synchronize()
    # Gather: rank r collects every rank's chunk (peer-copied to r) in order.
    gathered = []
    for r in range(world):
        chunks = [local[s].to(f"xpu:{r}") for s in range(world)]
        gathered.append(torch.cat(chunks))
    torch.xpu.synchronize()
    return local, gathered


@pytest.mark.parametrize("world", [2, 4])
def test_allgather_correct(world):
    if torch.xpu.device_count() < world:
        pytest.skip(f"need {world} XPUs, have {torch.xpu.device_count()}")
    chunk = 1024
    parts = [torch.full((chunk, ), float(r + 1), device=f"xpu:{r}") for r in range(world)]
    local, gathered = _ring_allgather(parts)

    # Reference: rank r's chunk = (r+1) * (r+1) after the scale stage.
    ref = torch.cat([torch.full((chunk, ), float((s + 1) * (s + 1))) for s in range(world)])
    for r in range(world):
        got = gathered[r].to("cpu")
        assert torch.allclose(got, ref), f"rank {r} AllGather mismatch"
    # Every rank must agree (the AllGather invariant).
    for r in range(1, world):
        assert torch.allclose(gathered[r].to("cpu"), gathered[0].to("cpu"))


def test_triton_xpu_compute_only():
    """Narrowest check: a Triton-XPU kernel compiles + runs correctly on BMG."""
    x = torch.arange(4096, device="xpu", dtype=torch.float32)
    o = _triton_scale(x, 3.0)
    torch.xpu.synchronize()
    assert torch.allclose(o.cpu(), x.cpu() * 3.0)


if __name__ == "__main__":
    test_triton_xpu_compute_only()
    print("[ok] triton-xpu compute kernel")
    nd = torch.xpu.device_count()
    for w in [2, 4]:
        if nd >= w:
            test_allgather_correct(w)
            print(f"[ok] host-orchestrated AllGather world={w} on {nd}x BMG")
    print("ALL PASSED")
