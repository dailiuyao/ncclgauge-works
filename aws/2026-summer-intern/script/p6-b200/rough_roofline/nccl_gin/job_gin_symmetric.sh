#!/bin/bash
#SBATCH --job-name=gin_symmetric
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=gin_symmetric_%j.out

echo "=========================================="
echo "GIN P2P Test with Symmetric Memory"
echo "=========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo ""

cd /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p6-b200/rough_roofline/nccl_gin

# Setup
MPI_HOME="/opt/amazon/openmpi5"
NCCL_DIR="/home/liuyaod/software/nccl"
AWS_OFI_DIR="/home/liuyaod/software/aws-ofi-nccl"

export PATH="${MPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NCCL_DIR}/build/lib:${AWS_OFI_DIR}/lib:${MPI_HOME}/lib:${LD_LIBRARY_PATH}"

# GIN configuration for AWS EFA
export NCCL_GIN_TYPE=2              # PROXY mode (required for AWS EFA/OFI)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export CUDA_LAUNCH_BLOCKING=1

echo "Configuration:"
echo "  NCCL_GIN_TYPE: $NCCL_GIN_TYPE (PROXY mode for AWS EFA)"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "NCCL Version:"
strings ${NCCL_DIR}/build/lib/libnccl.so 2>/dev/null | grep "NCCL version" | head -1
echo ""

echo "Running GIN test with SYMMETRIC memory allocation..."
echo "Key: Uses ncclMemAlloc() instead of cudaMalloc()"
echo ""

mpirun -np 2 ./gin_p2p_symmetric

echo ""
echo "Test completed: $(date)"
