#!/bin/bash
#SBATCH -N3 --exclusive
#SBATCH -p p5en-odcr-queue  # Specify p5en partition
#SBATCH --exclude=p5en-odcr-queue-dy-p5en48xlarge-[20,39]
#SBATCH -J pping-test
#SBATCH -o %x-%j.out
#SBATCH -t 5:00:00

set -ex

export libfabric_dir=/opt/amazon/efa
export ppn=1 # p5.48xlarge has 8 GPU per node.

export NCCL_GAUGE_HOME="/home/liuyaod/software/gauge-test/ncclguage"

nccl=/home/liuyaod/software/nccl_p2p_profile
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl

export NCCL_PROTO="Simple"

cd $NCCL_GAUGE_HOME/aws

export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/aws/out"
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

GAUGE_MIN_NTHREADS=512
GAUGE_MAX_NTHREADS=512

GAUGE_MIN_NCHANNELS=1
GAUGE_MAX_NCHANNELS=4

GAUGE_STEP_SIZE_SMALL=8
GAUGE_STEP_SIZE_MEDIUM=64
GAUGE_STEP_SIZE_LARGE=128

# when test G for small message:
# ch2 MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL / 16)) 
# ch1 MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL / 16))
# else, GAUGE_STEP_SIZE_SMALL 
message_size() {
    if [ "$nch" -eq 4 ]; then

        declare -g MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL * 4))
        declare -g MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 15))
        declare -g MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL * 4))

        declare -g MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 8))
        declare -g MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 56))
        declare -g MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM * 8))

        declare -g MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 4096))
        declare -g MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 4096))
        declare -g MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 32))

    elif [ "$nch" -eq 2 ]; then

        declare -g MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL * 2))
        declare -g MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 7))
        declare -g MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL * 2))

        declare -g MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 4))
        declare -g MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 28))
        declare -g MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM * 4))

        declare -g MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 4096))
        declare -g MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 4096))
        declare -g MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 16))

    elif [ "$nch" -eq 1 ]; then

        declare -g MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL))
        declare -g MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 3))
        declare -g MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL))

        declare -g MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 4))
        declare -g MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 14))
        declare -g MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM * 2))

        declare -g MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 4096))
        declare -g MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 4096))
        declare -g MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 8))

    fi
}

MESSAGE_SIZE_EXTRA_START=65536
MESSAGE_SIZE_EXTRA_END=524288
MESSAGE_SIZE_EXTRA_STEP=65536

# Read the list of allocated nodes
message_number=(10)
test_mode=("pping_3ranks")

chunk_size_set=(131072)

run_experiment() {

    for GAUGE_STEP_SIZE in "128"; do
        export GAUGE_STEP_SIZE
        case $GAUGE_STEP_SIZE in
            "64")
                START_VAR="MESSAGE_SIZE_MEDIUM_START"
                END_VAR="MESSAGE_SIZE_MEDIUM_END"
                STEP_VAR="MESSAGE_SIZE_MEDIUM_STEP"
                ;;
            "128")
                START_VAR="MESSAGE_SIZE_LARGE_START"
                END_VAR="MESSAGE_SIZE_LARGE_END"
                STEP_VAR="MESSAGE_SIZE_LARGE_STEP"
                ;;
            "524288")
                START_VAR="MESSAGE_SIZE_EXTRA_START"
                END_VAR="MESSAGE_SIZE_EXTRA_END"
                STEP_VAR="MESSAGE_SIZE_EXTRA_STEP"
                ;;
            *)
                START_VAR="MESSAGE_SIZE_SMALL_START"
                END_VAR="MESSAGE_SIZE_SMALL_END"
                STEP_VAR="MESSAGE_SIZE_SMALL_STEP"
                ;;
        esac
        for ((nch = GAUGE_MIN_NCHANNELS; nch <= GAUGE_MAX_NCHANNELS; nch *= 2)); do
            message_size
            for n in "${message_number[@]}"; do
                for ((msize=${!START_VAR}; msize<=${!END_VAR}; msize+=${!STEP_VAR})); do
                    for mode in "${test_mode[@]}"; do
                        for ((nth = GAUGE_MIN_NTHREADS; nth <= GAUGE_MAX_NTHREADS; nth *= 2)); do
                            for combo in "0,0"; do
                                IFS=',' read -r send_d recv_d <<< "$combo"
                                for chunk_size in "${chunk_size_set[@]}"; do 
                                    for group_sz in "1" "2"; do
                                        export GAUGE_MODE=${mode}
                                        export NCCL_MIN_NCHANNELS=${nch}
                                        export NCCL_MAX_NCHANNELS=${nch}
                                        export NCCL_NTHREADS=${nth}
                                        export NCCL_NCHANNELS_PER_NET_PEER=${nch}                                   
                                        export GAUGE_ITERATION=100
                                        export GAUGE_MESSAGE_SIZE=${msize}
                                        export NCCL_P2P_NET_CHUNKSIZE=${chunk_size}

                                        export GAUGE_GROUP_SIZE=${group_sz}

                                        /opt/amazon/openmpi/bin/mpirun \
                                            -x LD_LIBRARY_PATH=$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:$LD_LIBRARY_PATH \
                                            -x NCCL_DEBUG=WARN \
                                            -x NCCL_NET="AWS Libfabric" \
                                            -N $ppn \
                                            --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
                                            $NCCL_GAUGE_HOME/gauge/${mode}_groupsz_${group_sz}_gauge_n_${n}.exe "$send_d" "$recv_d"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
}

d_number=(1000)
run_experiment