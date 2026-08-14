#!/bin/bash
#SBATCH --job-name=alltoall_gin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=alltoall_gin_%j.out
#SBATCH --partition=p6-odcr-queue

echo "=========================================="
echo "NCCL AlltoAll Test with GIN"
echo "=========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo ""

# Setup paths
export NCCL_HOME=/home/liuyaod/software/nccl
export LD_LIBRARY_PATH=$NCCL_HOME/build/lib:$LD_LIBRARY_PATH

# GIN configuration for AWS EFA
export NCCL_GIN_TYPE=2              # PROXY mode
export NCCL_GIN_ENABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1

echo "Configuration:"
echo "  Nodes: 2"
echo "  GPUs per node: 8"
echo "  Total ranks: 16"
echo "  NCCL_GIN_TYPE: $NCCL_GIN_TYPE"
echo ""

echo "NCCL Version:"
strings ${NCCL_HOME}/build/lib/libnccl.so 2>/dev/null | grep "NCCL version" | head -1
echo ""

# Look for RMA debug output
echo "RMA Support Status:"
echo "Looking for 'RMA Support Debug' in output..."
echo ""

# Run AlltoAll benchmark
echo "Running AlltoAll performance test..."
echo "Message sizes: 8B to 128MB"
echo ""

srun --mpi=pmix \
  /home/liuyaod/software/nccl-tests/build/alltoall_perf \
  -b 8 -e 128M -f 2 -g 1 2>&1 | tee alltoall_output.txt

echo ""
echo "=========================================="
echo "RMA Support Debug Output:"
echo "=========================================="
grep -i "rma support debug" alltoall_output.txt || echo "No RMA debug output found"
echo ""

echo "Test completed: $(date)"
