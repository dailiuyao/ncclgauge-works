#!/bin/bash
#SBATCH -N4 --exclusive
#SBATCH -p p5en-odcr-queue  # Specify p5en partition
#SBATCH -J nccl_test
#SBATCH -o %x-%j.out
#SBATCH -t 5:00:00

set -ex

export libfabric_dir=/opt/amazon/efa
export NCCL_ALGO=NVLSTree
export mode="allreduce"
export ppn=8 # p5.48xlarge has 8 GPU per node.
# export CUDA_VISIBLE_DEVICES=0,1

export NCCL_GAUGE_HOME="/home/liuyaod/software/gauge-test/ncclguage"

nccl=/home/liuyaod/software/nccl_profile
nccl_tests=/home/liuyaod/software/nccl-tests
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl

time_start=`date +%s`

export GAUGE_HEO="inter"
export GAUGE_MODE=$mode
export GAUGE_NCHANNELS=16
export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/aws/out"
mkdir -p "$GAUGE_OUT_DIRE"
export NCCL_PROTO=Simple

# # NCCL perf configs
# NCCL_BUFFSIZE=8388608
# NCCL_P2P_NET_CHUNKSIZE=524288

GAUGE_NITERATIONS=1

N_ITERS=10000

run_gauge_test() {
    local nchannels=$1
    local msg_sizes=("${@:2}")  # Get all arguments after the first one as array
    
    export GAUGE_NCHANNELS=$nchannels
    echo "Running tests with GAUGE_NCHANNELS=$nchannels for message sizes: ${msg_sizes[*]}"
    
    for msg in "${msg_sizes[@]}"; do
        for ((iteration=0; iteration<GAUGE_NITERATIONS; iteration++)); do
            export GAUGE_ITERATION=$iteration
            export GAUGE_MESSAGE_SIZE=$msg
            /opt/amazon/openmpi/bin/mpirun \
                -x LD_LIBRARY_PATH=$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:$LD_LIBRARY_PATH \
                -x NCCL_DEBUG=WARN \
                -x NCCL_NET="AWS Libfabric" \
                -N $ppn \
                --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
                $NCCL_GAUGE_HOME/gauge/${mode}_tree_gauge_n_${N_ITERS}.exe
        done
    done
}

if [ "$NCCL_PROTO" = "Simple" ]; then
    run_gauge_test 1 8 
    run_gauge_test 2 16
    run_gauge_test 4 32
    run_gauge_test 8 64
    run_gauge_test 16 128 256 512
    run_gauge_test 16 $(seq 1024 1024 16384)
    run_gauge_test 16 $(seq 65536 65536 1048576)
fi

time_end=`date +%s`
echo
echo Total Execution Time: `expr $time_end - $time_start` seconds.

