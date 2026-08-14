#!/bin/bash
#SBATCH --job-name=nccl_p2p_gdr
#SBATCH --nodes=2                      # Request 2 nodes
#SBATCH --nodelist=p6-odcr-queue-dy-p6b20048xlarge-2,p6-odcr-queue-dy-p6b20048xlarge-7
#SBATCH --exclude=p6-odcr-queue-dy-p6b20048xlarge-6
#SBATCH --ntasks-per-node=1            # 1 task per node (for inter-node)
#SBATCH --gres=gpu:1                   # 1 GPU per node
#SBATCH --time=00:30:00                # 30 minutes

# Print job information
echo "=========================================="
echo "NCCL Inter-Node P2P Test - GDR Mode"
echo "=========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo ""

# Extract node indices from nodelist and create output filename
# Convert nodelist like "node-[0,1]" or "node-0,node-1" to "0_1"
NODE_IDS=$(scontrol show hostname $SLURM_JOB_NODELIST | sed 's/.*-//' | tr '\n' '_' | sed 's/_$//')
OUTPUT_FILE="p2p_gdr_${NODE_IDS}_${SLURM_JOB_ID}.out"
ERROR_FILE="p2p_gdr_${NODE_IDS}_${SLURM_JOB_ID}.err"

# Redirect stdout and stderr to the new files
exec > >(tee "$OUTPUT_FILE")
exec 2> >(tee "$ERROR_FILE" >&2)

echo "Output file: $OUTPUT_FILE"
echo "Error file: $ERROR_FILE"
echo ""

# Change to working directory (this script lives in rough_roofline/nccl_gdr/)
cd /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p6-b200/rough_roofline/nccl_gdr

# Setup environment
MPI_HOME="/opt/amazon/openmpi5"
NCCL_DIR="/home/liuyaod/software/nccl"
AWS_OFI_NCCL_DIR="/home/liuyaod/software/aws-ofi-nccl"
NCCL_TESTS_DIR="/home/liuyaod/software/nccl-tests"

export PATH="${MPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NCCL_DIR}/build/lib:${AWS_OFI_NCCL_DIR}/lib:${MPI_HOME}/lib:${LD_LIBRARY_PATH}"

export NCCL_DEBUG=INFO

# Optional: Enable NCCL debug output
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET,INIT

# Test parameters
NGPUS=1                          # 1 GPU per task
MIN_SIZE="8"
MAX_SIZE="1G"
STEP_FACTOR="2"
WARMUP_ITERS="50"
TEST_ITERS="500"

echo "Environment:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Verify node distribution
echo "Verifying node distribution:"
mpirun -np $SLURM_NTASKS hostname
echo ""

echo "=========================================="
echo "GDR Mode Test (ncclSend/ncclRecv)"
echo "=========================================="
echo ""

# Let NCCL automatically choose optimal channel settings
# GDR mode using nccl-tests
mpirun -np $SLURM_NTASKS \
    ${NCCL_TESTS_DIR}/build/sendrecv_perf \
    -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR -g $NGPUS \
    -w $WARMUP_ITERS -n $TEST_ITERS

echo ""
echo "✓ GDR mode test completed"
echo ""
echo "=========================================="
echo "End time: $(date)"
echo ""
