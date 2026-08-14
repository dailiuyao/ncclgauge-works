#!/bin/bash
#SBATCH -N1 --exclusive
#SBATCH -p p6-odcr-queue
#SBATCH -J build_nccl_p6b200
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p6-b200/rough_roofline/nccl_gdr/%x-%j.out
#SBATCH -t 0:45:00
#SBATCH --exclude=p6-odcr-queue-dy-p6b20048xlarge-[6,8]

# Build NCCL + aws-ofi-nccl plugin + nccl-tests on a P6 B200 compute node.
# Targets only sm_100 (B200 / Blackwell) so the build is fast (~5 min instead
# of ~15 min when building all archs).
#
# Output binaries land in:
#   $NCCL_SRC/build/lib/libnccl.so
#   $AWS_OFI_NCCL_SRC/install/lib/libnccl-net.so
#   $NCCL_TEST_SRC/build/all_reduce_perf, sendrecv_perf, ...
#
# These are the SAME paths the current run_nccl_*.sh / job_internode_*.sh
# scripts in this directory reference, so you can submit those right after
# this build job finishes.

set -e

echo "=========================================="
echo "Build NCCL stack on P6 B200 compute node"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Host:      $(hostname)"
echo "Started:   $(date)"
echo "=========================================="
echo ""

# ---- Environment ----------------------------------------------------------
export CUDA_HOME=/usr/local/cuda
export MPI=1
export MPI_HOME=/opt/amazon/openmpi
export LD_LIBRARY_PATH=${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}

# Source trees (these are the ones the runner scripts in this dir already use).
export NCCL_SRC=/home/liuyaod/software/nccl
export AWS_OFI_NCCL_SRC=/home/liuyaod/software/aws-ofi-nccl
export NCCL_TEST_SRC=/home/liuyaod/software/nccl-tests

# B200 only — keeps the build fast (≈5 min vs ≈15 min when emitting all archs).
export NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"

echo "CUDA_HOME=${CUDA_HOME}"
echo "MPI_HOME=${MPI_HOME}"
echo "NCCL_SRC=${NCCL_SRC}"
echo "AWS_OFI_NCCL_SRC=${AWS_OFI_NCCL_SRC}"
echo "NCCL_TEST_SRC=${NCCL_TEST_SRC}"
echo "NVCC_GENCODE=${NVCC_GENCODE}"
echo ""
nvidia-smi -L 2>&1 | head -3
echo ""

# ---- 1. Build NCCL --------------------------------------------------------
echo "=========================================="
echo "[1/3] Building NCCL ($(basename ${NCCL_SRC}))"
echo "=========================================="
pushd "${NCCL_SRC}"
make clean
rm -rf build
make -j$(nproc) src.build NVCC_GENCODE="${NVCC_GENCODE}"
popd
echo "    libnccl.so → ${NCCL_SRC}/build/lib/libnccl.so"
ls -la "${NCCL_SRC}/build/lib/libnccl.so"* 2>&1 | head -3
echo ""

export NCCL_HOME=${NCCL_SRC}/build

# ---- 2. Build aws-ofi-nccl plugin -----------------------------------------
echo "=========================================="
echo "[2/3] Building AWS OFI NCCL plugin"
echo "=========================================="
pushd "${AWS_OFI_NCCL_SRC}"
make distclean || true
./autogen.sh
./configure --prefix=${AWS_OFI_NCCL_SRC}/install \
            --with-mpi=${MPI_HOME} \
            --with-libfabric=/opt/amazon/efa \
            --with-nccl=${NCCL_HOME} \
            --with-cuda=${CUDA_HOME}
# Do NOT abort on test-binary linker errors (they fail because they reference
# gauge-side symbols at link time, but the main libnccl-net.so still builds
# successfully). Verify it landed below.
make -j$(nproc) || true
make install || true
popd
if [ ! -f "${AWS_OFI_NCCL_SRC}/install/lib/libnccl-net.so" ]; then
    echo "ERROR: libnccl-net.so was not produced."
    exit 1
fi
echo "    libnccl-net.so → ${AWS_OFI_NCCL_SRC}/install/lib/libnccl-net.so"
ls -la "${AWS_OFI_NCCL_SRC}/install/lib/libnccl-net.so"* 2>&1 | head -3
echo ""

# ---- 3. Build nccl-tests --------------------------------------------------
echo "=========================================="
echo "[3/3] Building nccl-tests"
echo "=========================================="
pushd "${NCCL_TEST_SRC}"
make clean
make -j$(nproc) MPI=1 MPI_HOME=${MPI_HOME} NCCL_HOME=${NCCL_HOME} CUDA_HOME=${CUDA_HOME} \
                NVCC_GENCODE="${NVCC_GENCODE}"
popd
echo "    nccl-tests binaries:"
ls -la "${NCCL_TEST_SRC}/build/"*_perf 2>&1 | head
echo ""

echo "=========================================="
echo "Build completed successfully on $(hostname) at $(date)"
echo "=========================================="
