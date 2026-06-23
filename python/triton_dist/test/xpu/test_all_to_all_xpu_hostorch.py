################################################################################
# Copyright (c) 2026 The AXON Authors
# SPDX-License-Identifier: MIT
#
# MoE EP all-to-all on Intel XPU — HOST-ORCHESTRATED form (Phase T5, runnable).
#
# The EP dispatch/combine pattern: each rank holds tokens destined for experts
# spread across all ranks; dispatch sends each token to the rank owning its
# expert (an all-to-all), experts run, combine sends results back. The
# in-kernel fused version (DeepEP-style, with libshmem_device putmem_signal)
# needs T3 + T1b; this host-orchestrated version proves the routing + movement
# on the BMG multi-GPU substrate today, and is the correctness oracle the fused
# kernel must match.
################################################################################
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.xpu

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("no Intel XPU runtime", allow_module_level=True)


def _all_to_all(send):
    """send[r] is a list of length `world`: send[r][d] = tensor rank r sends to
    rank d. Returns recv where recv[d][r] = what rank d got from rank r (peer
    copied onto device d). This is the EP-dispatch movement."""
    world = len(send)
    recv = [[send[r][d].to(f"xpu:{d}") for r in range(world)] for d in range(world)]
    torch.xpu.synchronize()
    return recv


@pytest.mark.parametrize("world", [2, 4])
def test_all_to_all_correct(world):
    if torch.xpu.device_count() < world:
        pytest.skip(f"need {world} XPUs, have {torch.xpu.device_count()}")
    # Each rank r sends a distinct, identifiable payload to each rank d:
    # value = r*100 + d, length 8.
    send = [[torch.full((8, ), float(r * 100 + d), device=f"xpu:{r}") for d in range(world)]
            for r in range(world)]
    recv = _all_to_all(send)
    # rank d must receive from each r exactly r*100 + d.
    for d in range(world):
        for r in range(world):
            got = recv[d][r].to("cpu")
            assert torch.allclose(got, torch.full((8, ), float(r * 100 + d))), \
                f"all_to_all: rank {d} from {r} mismatch"


def test_ep_dispatch_combine_roundtrip():
    """A token routed out and combined back must return to its origin unchanged
    (the EP dispatch->expert->combine invariant), host-orchestrated."""
    world = min(4, torch.xpu.device_count())
    if world < 2:
        pytest.skip("need >=2 XPUs")
    # rank r owns expert r. Each rank has one token for every expert.
    tokens = [[torch.full((4, ), float(r * 10 + e), device=f"xpu:{r}") for e in range(world)]
              for r in range(world)]
    # dispatch: token (r,e) goes to rank e.
    dispatched = [[tokens[r][e].to(f"xpu:{e}") for r in range(world)] for e in range(world)]
    torch.xpu.synchronize()
    # expert: identity (real expert is a GEMM; identity keeps it a movement test).
    # combine: send token (r,e) back to rank r.
    combined = [[dispatched[e][r].to(f"xpu:{r}") for e in range(world)] for r in range(world)]
    torch.xpu.synchronize()
    for r in range(world):
        for e in range(world):
            assert torch.allclose(combined[r][e].to("cpu"), torch.full((4, ), float(r * 10 + e))), \
                f"dispatch/combine roundtrip rank {r} expert {e} mismatch"


if __name__ == "__main__":
    nd = torch.xpu.device_count()
    for w in [2, 4]:
        if nd >= w:
            test_all_to_all_correct(w)
            print(f"[ok] host-orchestrated all_to_all world={w}")
    test_ep_dispatch_combine_roundtrip()
    print("[ok] EP dispatch/combine roundtrip")
    print("ALL PASSED")
