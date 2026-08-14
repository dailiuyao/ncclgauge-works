#!/bin/bash
set -e

# ---- Toolchain locations (override in environment if needed) --------------
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MPI_HOME=${MPI_HOME:-/opt/amazon/openmpi}
export LIBFABRIC_HOME=${LIBFABRIC_HOME:-/opt/amazon/efa}
export LD_LIBRARY_PATH=${MPI_HOME}/lib:${LD_LIBRARY_PATH}

# ---- Source / install trees (override in environment if needed) -----------
# Default to the standard layout used in this repo (../../.. resolves to the
# 2026-summer-intern root regardless of where the script was invoked from).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_GAUGE_HOME=${NCCL_GAUGE_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}

export NCCL_SRC=${NCCL_SRC:-/home/liuyaod/software/nccl_profile}
export AWS_OFI_NCCL_SRC=${AWS_OFI_NCCL_SRC:-/home/liuyaod/software/aws-ofi-nccl_profile}
export NCCL_HOME=${NCCL_HOME:-${NCCL_SRC}/build}

# Directory holding libprofiling_arrays.so (defines the gauge↔plugin shared
# timing arrays). The plugin links against this lib and bakes its directory
# into RPATH so it resolves at runtime.
export PROFILING_LIB_DIR=${PROFILING_LIB_DIR:-${NCCL_GAUGE_HOME}/gauge}

# P6 B200 uses Blackwell architecture (compute capability 10.0)
export NVCC_GENCODE=${NVCC_GENCODE:-"-gencode=arch=compute_100,code=sm_100"}

echo "=== Build configuration ==="
echo "  CUDA_HOME          = ${CUDA_HOME}"
echo "  MPI_HOME           = ${MPI_HOME}"
echo "  LIBFABRIC_HOME     = ${LIBFABRIC_HOME}"
echo "  NCCL_SRC           = ${NCCL_SRC}"
echo "  AWS_OFI_NCCL_SRC   = ${AWS_OFI_NCCL_SRC}"
echo "  NCCL_GAUGE_HOME    = ${NCCL_GAUGE_HOME}"
echo "  PROFILING_LIB_DIR  = ${PROFILING_LIB_DIR}"
echo "  NVCC_GENCODE       = ${NVCC_GENCODE}"
echo ""

# Sanity check: libprofiling_arrays.so must exist before linking the plugin.
if [ ! -f "${PROFILING_LIB_DIR}/libprofiling_arrays.so" ]; then
    echo "ERROR: ${PROFILING_LIB_DIR}/libprofiling_arrays.so not found."
    echo "Build it first (g++ -shared -fPIC -O2 -o libprofiling_arrays.so profiling_arrays.cpp)"
    echo "or set PROFILING_LIB_DIR to the directory that contains it."
    exit 1
fi

# ---- Build NCCL (profile version) -----------------------------------------
# Using v2.27.6 approach: build library only (skip utilities like ncclparam).
# The library has undefined symbols for profiling variables (lyd_*) which
# are defined in the gauge executable, not in NCCL.
echo "=== Building NCCL (profile) for P6 B200 ==="
pushd "${NCCL_SRC}/src"
make -j$(nproc) lib NVCC_GENCODE="${NVCC_GENCODE}"
popd

# ---- Build aws-ofi-nccl (profile version) ---------------------------------
# Pass the PROFILING_LIB_DIR via LDFLAGS so the Makefile.am's plain
# `-lprofiling_arrays` resolves and gets baked into RPATH for runtime.
echo "=== Building aws-ofi-nccl (profile) ==="
pushd "${AWS_OFI_NCCL_SRC}"
./autogen.sh
LDFLAGS="-L${PROFILING_LIB_DIR} -Wl,-rpath,${PROFILING_LIB_DIR}" \
./configure --prefix=${AWS_OFI_NCCL_SRC}/install \
            --with-mpi=${MPI_HOME} \
            --with-libfabric=${LIBFABRIC_HOME} \
            --with-nccl=${NCCL_HOME} \
            --with-cuda=${CUDA_HOME}
make -j$(nproc) && make install
popd

echo "=== Profile build complete for P6 B200 ==="
