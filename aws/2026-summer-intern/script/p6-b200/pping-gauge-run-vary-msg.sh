#!/bin/bash
#SBATCH -N2 --exclusive
#SBATCH -p p6-odcr-queue
#SBATCH -J pping-gauge-msg-p6b200
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p6-b200/%x-%j.out
#SBATCH -t 1:00:00

set -ex

export libfabric_dir=/opt/amazon/efa
export ppn=1

export NCCL_GAUGE_HOME="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern"

nccl=/home/liuyaod/software/nccl_profile
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl_profile

export NCCL_PROTO="Simple"

cd $NCCL_GAUGE_HOME/script/p6-b200

export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/out/p6-b200"
mkdir -p $GAUGE_OUT_DIRE

export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

GAUGE_MIN_NTHREADS=512
GAUGE_MAX_NTHREADS=512

GAUGE_MIN_NCHANNELS=4
GAUGE_MAX_NCHANNELS=4

GAUGE_STEP_SIZE_LARGE=128

# Test message sizes (in BYTES). GAUGE_MESSAGE_SIZE is now interpreted as bytes by gauge.

MESSAGE_SIZES=(8 16 32 64 128 256 512 \
               1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 \
               1048576 2097152 4194304 8388608 16777216 33554432 67108864 \
               134217728 268435456 536870912 1073741824 2147483648)
MESSAGE_LABELS=("8B" "16B" "32B" "64B" "128B" "256B" "512B" \
                "1KB" "2KB" "4KB" "8KB" "16KB" "32KB" "64KB" "128KB" "256KB" "512KB" \
                "1MB" "2MB" "4MB" "8MB" "16MB" "32MB" "64MB" \
                "128MB" "256MB" "512MB" "1GB" "2GB")

message_number=(1)
test_mode=("pping")

# This script varies MESSAGE size, so output files end with "-vary-msg.out".
# The gauge picks this up from GAUGE_OUT_SUFFIX (set per-test in the mpirun call below).
export GAUGE_OUT_SUFFIX="msg"

# Clear previous output files so this run starts fresh; per-message-size results
# will be appended within this run (gauge opens the file in append mode).
rm -f "$GAUGE_OUT_DIRE"/nccl_pping_${GAUGE_HEO}_r-0-vary-${GAUGE_OUT_SUFFIX}.out \
      "$GAUGE_OUT_DIRE"/nccl_pping_${GAUGE_HEO}_r-1-vary-${GAUGE_OUT_SUFFIX}.out

for n in "${message_number[@]}"; do
    for idx in "${!MESSAGE_SIZES[@]}"; do
        msize=${MESSAGE_SIZES[$idx]}
        mlabel=${MESSAGE_LABELS[$idx]}
        for mode in "${test_mode[@]}"; do
            export GAUGE_MODE=${mode}
            export NCCL_MIN_NCHANNELS=${GAUGE_MIN_NCHANNELS}
            export NCCL_MAX_NCHANNELS=${GAUGE_MAX_NCHANNELS}
            export NCCL_NTHREADS=${GAUGE_MIN_NTHREADS}
            export NCCL_NCHANNELS_PER_NET_PEER=${GAUGE_MIN_NCHANNELS}
            export GAUGE_ITERATION=400
            export GAUGE_MESSAGE_SIZE=${msize}
            export GAUGE_STEP_SIZE=524288

            echo "=============================================="
            echo "Testing: Message size = $mlabel ($msize bytes)"
            echo "         Mode = $mode"
            echo "         Iteration = $GAUGE_ITERATION"
            echo "=============================================="

            # Header into the gauge output files (rank 0 and rank 1) before the run.
            # Gauge opens the file in append mode, so this header marks the start
            # of this message-size case in the appended output.
            for r in 0 1; do
                outfile="$GAUGE_OUT_DIRE/nccl_pping_${GAUGE_HEO}_r-${r}-vary-${GAUGE_OUT_SUFFIX}.out"
                {
                  echo ""
                  echo "########################################################################"
                  echo "# Test case: Message size = $mlabel ($msize bytes), mode=$mode, iter=$GAUGE_ITERATION"
                  echo "########################################################################"
                } >> "$outfile"
            done

            /opt/amazon/openmpi/bin/mpirun \
                -x LD_LIBRARY_PATH=/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/gauge:$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:$LD_LIBRARY_PATH \
                -x LD_PRELOAD=/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/gauge/libprofiling_arrays.so \
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
                -x GAUGE_OUT_SUFFIX=$GAUGE_OUT_SUFFIX \
                -N $ppn \
                --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
                $NCCL_GAUGE_HOME/gauge/p6-b200/${mode}_gauge_n_${n}.exe "0" "0"
        done
    done
done

echo "=== Gauge run complete for P6 B200 ==="
echo "Output directory: $GAUGE_OUT_DIRE"
