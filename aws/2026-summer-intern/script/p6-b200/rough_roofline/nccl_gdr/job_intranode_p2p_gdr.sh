#!/bin/bash
#SBATCH --job-name=nccl_p2p_gdr_intra
#SBATCH --nodes=1                      # Request 1 node (intra-node test)
#SBATCH --nodelist=p6-odcr-queue-dy-p6b20048xlarge-2
#SBATCH --ntasks-per-node=2            # 2 tasks (one per GPU in each pair) on the same node
#SBATCH --gres=gpu:8                   # 8 GPUs (need access to all to test all adjacent pairs)
#SBATCH --time=00:30:00                # 30 minutes

# Print job information
echo "=========================================="
echo "NCCL Intra-Node P2P Test - GDR Mode"
echo "(All adjacent GPU pairs on the same node, traffic goes over NVLink/NVSwitch,"
echo " NOT EFA — this measures intra-node GPU-to-GPU latency/bw for each pair)"
echo "=========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo ""

# Extract node index from nodelist and create output filename
NODE_ID=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1 | sed 's/.*-//')
OUTPUT_FILE="p2p_gdr_intranode_${NODE_ID}_${SLURM_JOB_ID}.out"
ERROR_FILE="p2p_gdr_intranode_${NODE_ID}_${SLURM_JOB_ID}.err"

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

# Quiet NCCL log per pair (set to INFO if you want detailed transport choice)
export NCCL_DEBUG=WARN

# Test parameters
NGPUS=1                          # 1 GPU per task (each MPI rank owns 1 GPU)
MIN_SIZE="8"
MAX_SIZE="1G"
STEP_FACTOR="2"
WARMUP_ITERS="50"
TEST_ITERS="500"

echo "Environment:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Adjacent GPU pairs to test: (0,1), (1,2), ..., (6,7), and (7,0) closing the ring
GPU_PAIRS=("0 1" "1 2" "2 3" "3 4" "4 5" "5 6" "6 7" "7 0")

for pair in "${GPU_PAIRS[@]}"; do
    A=$(echo $pair | awk '{print $1}')
    B=$(echo $pair | awk '{print $2}')

    echo ""
    echo "########################################################################"
    echo "# Pair: GPU ${A} <-> GPU ${B}"
    echo "# (CUDA_VISIBLE_DEVICES=${A},${B} → rank 0 uses GPU${A}, rank 1 uses GPU${B})"
    echo "########################################################################"

    # CUDA_VISIBLE_DEVICES=A,B remaps so:
    #   - MPI rank 0 (localRank=0) sees visible[0] = real GPU A
    #   - MPI rank 1 (localRank=1) sees visible[1] = real GPU B
    # nccl-tests assigns gpus[i] = localRank*nThreads*nGpus + i, so each rank
    # naturally picks its own visible GPU.
    mpirun -np $SLURM_NTASKS \
        -x CUDA_VISIBLE_DEVICES=${A},${B} \
        ${NCCL_TESTS_DIR}/build/sendrecv_perf \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR -g $NGPUS \
        -w $WARMUP_ITERS -n $TEST_ITERS

    echo ""
    echo "✓ Pair (${A}, ${B}) done"
done

echo ""
echo "=========================================="
echo "All adjacent GPU pairs tested."
echo "End time: $(date)"
echo "=========================================="
