#!/bin/bash
#SBATCH --job-name=nccl_p2p_gin
#SBATCH --nodes=2                      # Request 2 nodes
#SBATCH --ntasks-per-node=1            # 1 task per node (for inter-node)
#SBATCH --gres=gpu:1                   # 1 GPU per node
#SBATCH --time=00:30:00                # 30 minutes
#SBATCH --output=p2p_gin_%j.out        # Output file
#SBATCH --error=p2p_gin_%j.err         # Error file

# Print job information
echo "=========================================="
echo "NCCL Inter-Node P2P Test - GIN Mode"
echo "=========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo ""

# Change to working directory (this script lives in rough_roofline/nccl_gin/)
cd /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p6-b200/rough_roofline/nccl_gin

# Setup environment
MPI_HOME="/opt/amazon/openmpi5"
NCCL_DIR="/home/liuyaod/software/nccl"
AWS_OFI_NCCL_DIR="/home/liuyaod/software/aws-ofi-nccl"
NCCL_TESTS_DIR="/home/liuyaod/software/nccl-tests"

export PATH="${MPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NCCL_DIR}/build/lib:${AWS_OFI_NCCL_DIR}/lib:${MPI_HOME}/lib:${LD_LIBRARY_PATH}"

# Optional: Enable NCCL debug output
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET,INIT

# Test parameters
NGPUS=1                          # 1 GPU per task
MIN_SIZE="8"
MAX_SIZE="1G"
STEP_FACTOR="2"
WARMUP_ITERS="5"
TEST_ITERS="20"

echo "Environment:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Verify node distribution
echo "Verifying node distribution:"
mpirun -np $SLURM_NTASKS hostname
echo ""

echo "=========================================="
echo "GIN Mode Test (ncclPutSignal)"
echo "=========================================="
echo ""

# GIN mode using custom test
if [ -f "./gin_p2p_test" ]; then
    mpirun -np $SLURM_NTASKS \
        ./gin_p2p_test

    echo ""
    echo "✓ GIN mode test completed"
else
    echo "WARNING: gin_p2p_test not found. Run 'make' to build it."
fi

echo ""
echo "=========================================="
echo "End time: $(date)"
echo ""
