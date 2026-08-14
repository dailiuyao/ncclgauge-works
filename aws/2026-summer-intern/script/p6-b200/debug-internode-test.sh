#!/bin/bash
#SBATCH -N2 --exclusive
#SBATCH -p p6-odcr-queue
#SBATCH -J debug-pping-internode
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p6-b200/debug-internode-%j.out
#SBATCH -t 0:10:00

set -ex

export libfabric_dir=/opt/amazon/efa
export ppn=1

export NCCL_GAUGE_HOME="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern"

nccl=/home/liuyaod/software/nccl_profile
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl_profile

export NCCL_PROTO="Simple"

export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

GAUGE_MIN_NTHREADS=512
GAUGE_MAX_NTHREADS=512

GAUGE_MIN_NCHANNELS=1
GAUGE_MAX_NCHANNELS=1

# Only test one message size for quick debug (in bytes)
MESSAGE_SIZE=1024

message_number=(1)
test_mode=("pping")

export GAUGE_MODE=${test_mode}
export NCCL_MIN_NCHANNELS=${GAUGE_MIN_NCHANNELS}
export NCCL_MAX_NCHANNELS=${GAUGE_MAX_NCHANNELS}
export NCCL_NTHREADS=${GAUGE_MIN_NTHREADS}
export NCCL_NCHANNELS_PER_NET_PEER=${GAUGE_MIN_NCHANNELS}
export GAUGE_ITERATION=1
export GAUGE_MESSAGE_SIZE=${MESSAGE_SIZE}
export GAUGE_STEP_SIZE=524288
export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/out/p6-b200"

mkdir -p $GAUGE_OUT_DIRE

echo "=============================================="
echo "DEBUG TEST: Inter-node, Message size = $MESSAGE_SIZE bytes"
echo "Looking for debug output from NCCL and Plugin"
echo "=============================================="

/opt/amazon/openmpi/bin/mpirun \
    -x LD_LIBRARY_PATH=$NCCL_GAUGE_HOME/gauge:$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:$LD_LIBRARY_PATH \
    -x LD_PRELOAD=$NCCL_GAUGE_HOME/gauge/libprofiling_arrays.so \
    -x NCCL_DEBUG=WARN \
    -x NCCL_NET="AWS Libfabric" \
    -x NCCL_GIN_ENABLE=0 \
    -x NCCL_PROTO=$NCCL_PROTO \
    -x NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS \
    -x NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS \
    -x NCCL_NTHREADS=$NCCL_NTHREADS \
    -x GAUGE_MODE=$GAUGE_MODE \
    -x GAUGE_ITERATION=$GAUGE_ITERATION \
    -x GAUGE_MESSAGE_SIZE=$GAUGE_MESSAGE_SIZE \
    -x GAUGE_STEP_SIZE=$GAUGE_STEP_SIZE \
    -x GAUGE_HEO=$GAUGE_HEO \
    -x GAUGE_OUT_DIRE=$GAUGE_OUT_DIRE \
    -N $ppn \
    --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    $NCCL_GAUGE_HOME/gauge/p6-b200/pping_gauge_n_1.exe "0" "0"

echo ""
echo "=== Debug test complete ==="
echo "Output file: Check slurm output file for debug messages"
