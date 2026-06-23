# Triton-distributed XPU environment (AXON port, Phase T0).
#
# Source this to get a working Intel-XPU Triton toolchain on the BMG dev box.
#
#   source Triton-distributed/xpu_env.sh
#   python3 your_triton_xpu_script.py
#
# WHY this exact shape (load-bearing):
#   * torch 2.11.0+xpu (conda env `my_env_py_3_12`) bundles its OWN SYCL /
#     Unified-Runtime libs. Sourcing oneAPI `setvars.sh` prepends a DIFFERENT
#     libur_loader / libsycl onto LD_LIBRARY_PATH, which shadows torch's and
#     breaks `import torch` with an undefined-symbol (LIBUR_LOADER version skew).
#   * Triton's XPU launcher, however, must COMPILE a host stub that
#     `#include <sycl/sycl.hpp>` and links SYCL — it needs `icpx` + the SYCL
#     headers at JIT time.
#   * Resolution: expose ONLY the compiler bin (PATH) + SYCL headers (CPATH)
#     from the readable basekit; do NOT touch LD_LIBRARY_PATH (leave torch's
#     bundled runtime in front). This is the inverse of the AXON
#     python-test-env-ld-loader note.
#
# The /opt/intel/oneapi symlink (-> /home/sdp) is permission-locked for user
# `ss`; the readable toolchain is /data/ss/basekit_2025.2.

# --- Python env (torch 2.11+xpu, triton 3.7 with the `intel` backend) --------
export AXON_TD_PY="${AXON_TD_PY:-/home/ss/miniforge3/envs/my_env_py_3_12/bin/python3}"

# --- Intel oneAPI basekit (readable; compiler + headers only) ----------------
export AXON_TD_BASEKIT="${AXON_TD_BASEKIT:-/data/ss/basekit_2025.2}"
export PATH="${AXON_TD_BASEKIT}/compiler/2025.2/bin:${PATH}"
export CPATH="${AXON_TD_BASEKIT}/2025.2/include:${AXON_TD_BASEKIT}/2025.2/include/sycl:${CPATH}"
export CC=icx
export CXX=icpx

# Sanity (printed once, harmless):
echo "[xpu_env] python=${AXON_TD_PY}"
echo "[xpu_env] icpx=$(command -v icpx 2>/dev/null || echo MISSING)"
echo "[xpu_env] NOTE: LD_LIBRARY_PATH intentionally NOT modified (torch owns its SYCL runtime)"
