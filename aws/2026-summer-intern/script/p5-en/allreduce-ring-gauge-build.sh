#!/bin/bash
#SBATCH -N1 --exclusive
#SBATCH -p p5en-odcr-queue
#SBATCH -J allreduce-ring-gauge-build
# SLURM does not expand env vars in #SBATCH directives; output goes to the
# sbatch submit directory. cd to the desired log dir before submitting.
#SBATCH -o %x-%j.out
#SBATCH -t 0:15:00
# Build on a compute node (not the head node) so the CUDA runtime / driver ABI
# matches what's seen at run time — building on the head node produces a binary
# whose cuGetProcAddress PFN table doesn't match the compute-node driver,
# causing SEGV. Exclude the old-EFA (FABRIC_1.8-only) nodes to stay aligned
# with the plugin build / run set: 15-22, 40, 41, 42, 44, 45, 46.
#SBATCH --exclude=p5en-odcr-queue-dy-p5en48xlarge-[15-22,40-42,44-46]

set -e

# ============================================================================
# User-configurable paths — MUST be exported in the shell before `sbatch`.
#   NCCL_SRC        : path to the NCCL profile source tree
#   NCCL_GAUGE_HOME : path to this repo's root (contains gauge/, script/, out/)
# ============================================================================
export NCCL_SRC="/home/liuyaod/software/nccl_profile"
export NCCL_GAUGE_HOME="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern"

: "${NCCL_SRC:?please export NCCL_SRC to the nccl_profile source tree}"
: "${NCCL_GAUGE_HOME:?please export NCCL_GAUGE_HOME to the root of this gauge tree}"

# ---- Fixed toolchain locations (edit here if your cluster differs) --------
# Pin to CUDA 12.8: p5en nodes' /usr/local/cuda defaults to 13.1 which
# exceeds the driver's supported CUDA (13.0), triggering PFN mismatches.
export CUDA_HOME=/usr/local/cuda-12.8
export MPI_HOME=/opt/amazon/openmpi

# ---- Derived paths --------------------------------------------------------
export NCCL_HOME=$NCCL_SRC/build
export GET_CYCLES_HOME="${NCCL_GAUGE_HOME}/gauge"
export PROFILING_LIB_DIR="${NCCL_GAUGE_HOME}/gauge"
# Build output directory (kept generic — not tied to a specific instance type).
# The matching run script must reference the same path when launching the exe.
export GAUGE_OUT_DIR="${NCCL_GAUGE_HOME}/gauge/build"

export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

echo "=== Build host ==="
echo "  hostname = $(hostname)"
echo "  nvidia driver:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/    /'
echo ""

export PATH=$CUDA_HOME/bin:$MPI_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$MPI_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$MPI_HOME/lib:$LD_LIBRARY_PATH

mkdir -p $GAUGE_OUT_DIR

n_messages=(1)

for i in "${n_messages[@]}"; do
    nvcc $NVCC_GENCODE -ccbin g++ \
        -I"${NCCL_HOME}/include" -I"${MPI_HOME}/include" -I"${GET_CYCLES_HOME}" \
        -L"${NCCL_HOME}/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -L"${PROFILING_LIB_DIR}" \
        -lnccl -lcudart -lmpi -lprofiling_arrays \
        -Xlinker --export-dynamic -Xlinker --allow-shlib-undefined \
        -Xlinker -rpath="${PROFILING_LIB_DIR}" \
        -D GAUGE_N_MESSAGES=${i} \
        "${GET_CYCLES_HOME}/get_clock.cu" "${GET_CYCLES_HOME}/allreduce_ring_gauge.cu" \
        -o "${GAUGE_OUT_DIR}/allreduce_ring_gauge_n_${i}.exe"
    if [ -f "${GAUGE_OUT_DIR}/allreduce_ring_gauge_n_${i}.exe" ]; then
        echo "Compilation successful: ${GAUGE_OUT_DIR}/allreduce_ring_gauge_n_${i}.exe"
    else
        echo "Compilation failed: allreduce_ring_gauge_n_${i}.exe"
        exit 1
    fi
done

echo "=== AllReduce Ring Gauge build complete for P5 ==="
echo "Binaries location: $GAUGE_OUT_DIR"
