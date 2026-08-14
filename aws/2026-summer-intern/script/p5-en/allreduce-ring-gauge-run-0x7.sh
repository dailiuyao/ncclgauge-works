#!/bin/bash
#SBATCH -N4 --exclusive
#SBATCH -p p5en-odcr-queue
#SBATCH -J allreduce-ring-gauge-p5en
# SLURM does not expand env vars in #SBATCH directives; output goes to the
# sbatch submit directory. cd to the desired log dir before submitting.
#SBATCH -o %x-%j.out
#SBATCH -t 6:00:00
# Old-EFA (FABRIC_1.8-only) nodes: 15-22, 40, 41, 42, 44, 45, 46. The plugin
# is built against FABRIC_1.9, so dlopen fails on any of these — exclude them.
#SBATCH --exclude=p5en-odcr-queue-dy-p5en48xlarge-[15-22,40-42,44-46]

set -ex

# ============================================================================
# User-configurable paths — MUST be exported in the shell before `sbatch`.
#   NCCL_GAUGE_HOME : path to this repo's root (contains gauge/, script/, out/)
#   nccl            : path to the NCCL profile source tree (whose build/ was
#                     populated by the plugin build script)
#   ofi_plugin      : path to the aws-ofi-nccl profile source tree (whose
#                     install/ was populated by the plugin build script)
# ============================================================================
export NCCL_GAUGE_HOME="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern"
export nccl="/home/liuyaod/software/nccl_profile"
export ofi_plugin="/home/liuyaod/software/aws-ofi-nccl_profile"

: "${NCCL_GAUGE_HOME:?please export NCCL_GAUGE_HOME to the root of this gauge tree}"
: "${nccl:?please export nccl to the nccl_profile source tree}"
: "${ofi_plugin:?please export ofi_plugin to the aws-ofi-nccl_profile source tree}"

# ---- Fixed toolchain locations (edit here if your cluster differs) --------
export libfabric_dir=/opt/amazon/efa

# ---- Derived / run-time knobs --------------------------------------------
export ppn=1
export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/out"

# Work around OMPI-3.x PMIX dstore OUT-OF-RESOURCE errors under sbatch by
# forcing the PMIX gds framework to the hash backend (skip ds12/ds21 shm).
export PMIX_MCA_gds=hash

export NCCL_ALGO="Ring"
export NCCL_TESTS_SPLIT_MASK="0x7"

PROTOCOLS=("Simple" "LL" "LL128")

mkdir -p $GAUGE_OUT_DIRE

export GAUGE_HEO="inter"

# ---- Iterative O_net solver parameters (consumed by ComputeIterativeOnet in the gauge) ----
# 2L = round-trip latency (ms); a = 24.92us from pping -> 2L = a = 0.02492 ms.
export GAUGE_TWO_L_MS=0.02492
# NIC bandwidth pool (GB/s): dual EFA vs single EFA, and the per-GPU byte threshold
# below which nch==1 falls back to a single rail.
export GAUGE_BW_DUAL_GBS=48.75
export GAUGE_BW_SINGLE_GBS=24.375
export GAUGE_SINGLE_RAIL_THRESH_KB=128

# Set to 1 to have only rank 0 write its per-rank .out file (r-1..N-1 files
# are suppressed; those ranks keep stdout to mpirun). Unset / 0 = every rank
# writes its own file (default behavior before the flag was introduced).
export GAUGE_STDOUT_ONLY_RANK0=1

# Sweep over nchannels (NCCL_MIN_NCHANNELS == NCCL_MAX_NCHANNELS = nc).
NCHANNELS_LIST=(8 16)

# Sweep over NCCL_NTHREADS ∈ {64, 128, 256, 512, 1024}
NTHREADS_LIST=(512)

# Per-protocol buffsize sweep. NCCL uses THREE independent env vars:
#   Simple → NCCL_BUFFSIZE       (default 4 MiB)
#   LL     → NCCL_LL_BUFFSIZE    (default ~256 KB, protocol-derived)
#   LL128  → NCCL_LL128_BUFFSIZE (default ~512 KB, protocol-derived)
# NCCL_BUFFSIZE has NO effect on LL/LL128 — must set the matching env.

# BUFFSIZE_LIST_Simple=(65536 131072 262144 524288 1048576 2097152 4194304 8388608)
# BUFFSIZE_LABELS_Simple=("64KB" "128KB" "256KB" "512KB" "1MB" "2MB" "4MB" "8MB")
# BUFFSIZE_LIST_LL=( 16384 32768 65536 131072 262144 524288)
# BUFFSIZE_LABELS_LL=("16KB" "32KB" "64KB" "128KB" "256KB" "512KB")
# BUFFSIZE_LIST_LL128=(307200 614400 1228800 2457600 4915200)
# BUFFSIZE_LABELS_LL128=("300KB" "600KB" "1200KB" "2400KB" "4800KB")

BUFFSIZE_LIST_Simple=()
BUFFSIZE_LABELS_Simple=()
BUFFSIZE_LIST_LL=( 2048 4096 16384 32768 65536 131072 262144 524288)
BUFFSIZE_LABELS_LL=("2KB" "4KB" "16KB" "32KB" "64KB" "128KB" "256KB" "512KB")
BUFFSIZE_LIST_LL128=()
BUFFSIZE_LABELS_LL128=()

# Per-protocol message size: Simple=128MB, LL=32MB, LL128=128MB.
# The active MESSAGE_SIZES / MESSAGE_LABELS below get picked per-proto in the
# loop via MESSAGE_SIZES_<PROTO> / MESSAGE_LABELS_<PROTO>.
MESSAGE_SIZES_Simple=(1073741824)
MESSAGE_LABELS_Simple=("1GB")
MESSAGE_SIZES_LL=(8388608)
MESSAGE_LABELS_LL=("8MB")
MESSAGE_SIZES_LL128=(134217728)
MESSAGE_LABELS_LL128=("128MB")

export GAUGE_OUT_SUFFIX="0x7"
export GAUGE_MODE="allreduce_ring"

# Rank count (nnodes * ppn) — used to fan out per-rank output files. When
# GAUGE_STDOUT_ONLY_RANK0=1, only rank 0 writes a file, so the pre-clear /
# header loops below are collapsed to r=0.
n_ranks=$(( SLURM_JOB_NUM_NODES * ppn ))
if [ "${GAUGE_STDOUT_ONLY_RANK0:-0}" = "1" ]; then
    rank_hi=1
else
    rank_hi=$n_ranks
fi

# Clear previous output ONCE (all rank files for this suffix) — every
# (protocol, ...) combination appends to the same per-rank file.
for (( r=0; r<rank_hi; r++ )); do
    rm -f "$GAUGE_OUT_DIRE/nccl_allreduce_ring_${GAUGE_HEO}_r-${r}-${GAUGE_OUT_SUFFIX}.out"
done

for proto in "${PROTOCOLS[@]}"; do
    export NCCL_PROTO=$proto

    # Pick per-protocol MESSAGE_SIZES / MESSAGE_LABELS.
    sizes_var="MESSAGE_SIZES_${proto}[@]"
    labels_var="MESSAGE_LABELS_${proto}[@]"
    MESSAGE_SIZES=("${!sizes_var}")
    MESSAGE_LABELS=("${!labels_var}")

    # Pick per-protocol BUFFSIZE_LIST / BUFFSIZE_LABELS.
    bs_var="BUFFSIZE_LIST_${proto}[@]"
    bl_var="BUFFSIZE_LABELS_${proto}[@]"
    BUFFSIZE_LIST=("${!bs_var}")
    BUFFSIZE_LABELS=("${!bl_var}")

    # Which NCCL env var controls buffsize for this protocol.
    case "$proto" in
        Simple) buffsize_env="NCCL_BUFFSIZE"       ;;
        LL)     buffsize_env="NCCL_LL_BUFFSIZE"    ;;
        LL128)  buffsize_env="NCCL_LL128_BUFFSIZE" ;;
        *)      echo "Unknown protocol: $proto" >&2; exit 1 ;;
    esac

    echo ""
    echo "##############################################"
    echo "### Protocol = $proto  (msg sizes: ${MESSAGE_LABELS[*]}, buffsize env: $buffsize_env, buffsizes: ${BUFFSIZE_LABELS[*]})"
    echo "##############################################"

  for nc in "${NCHANNELS_LIST[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "### nchannels = $nc"
    echo "----------------------------------------------"

   for nth in "${NTHREADS_LIST[@]}"; do
    echo ""
    echo ".............................................."
    echo "### nthreads = $nth"
    echo ".............................................."

    for bidx in "${!BUFFSIZE_LIST[@]}"; do
     bsize=${BUFFSIZE_LIST[$bidx]}
     blabel=${BUFFSIZE_LABELS[$bidx]}
     echo ""
     echo "::::::::::::::::::::::::::::::::::::::::::::::"
     echo "### buffsize = $blabel ($bsize bytes)"
     echo "::::::::::::::::::::::::::::::::::::::::::::::"

    for idx in "${!MESSAGE_SIZES[@]}"; do
        msize=${MESSAGE_SIZES[$idx]}
        mlabel=${MESSAGE_LABELS[$idx]}

        export NCCL_MIN_NCHANNELS=${nc}
        export NCCL_MAX_NCHANNELS=${nc}
        export NCCL_NTHREADS=${nth}
        export NCCL_NCHANNELS_PER_NET_PEER=${nc}
        # Set only the protocol-relevant buffsize env; unset the other two
        # so a stale value from a prior iteration doesn't leak in.
        unset NCCL_BUFFSIZE NCCL_LL_BUFFSIZE NCCL_LL128_BUFFSIZE
        export ${buffsize_env}=${bsize}
        export GAUGE_ITERATION=128
        export GAUGE_MESSAGE_SIZE=${msize}

        echo "=============================================="
        echo "Testing: Protocol=$proto, nchannels=$nc, nthreads=$nth, buffsize=$blabel, Message size = $mlabel ($msize bytes)"
        echo "         Mode = allreduce_ring (SPLIT_MASK=$NCCL_TESTS_SPLIT_MASK)"
        echo "         Iteration = $GAUGE_ITERATION"
        echo "=============================================="

        # Write config marker to each rank's output file (r-limit set above).
        for (( r=0; r<rank_hi; r++ )); do
            outfile="$GAUGE_OUT_DIRE/nccl_allreduce_ring_${GAUGE_HEO}_r-${r}-${GAUGE_OUT_SUFFIX}.out"
            {
              echo ""
              echo "########################################################################"
              echo "# Test case: Protocol=$proto, nchannels=$nc, nthreads=$nth, buffsize=$blabel, Message size = $mlabel ($msize bytes), mode=allreduce_ring, iter=$GAUGE_ITERATION"
              echo "########################################################################"
            } >> "$outfile"
        done

        /opt/amazon/openmpi/bin/mpirun \
            -x LD_LIBRARY_PATH=$NCCL_GAUGE_HOME/gauge:$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:$LD_LIBRARY_PATH \
            -x LD_PRELOAD=$NCCL_GAUGE_HOME/gauge/libprofiling_arrays.so \
            -x NCCL_DEBUG=INFO \
            -x NCCL_NET="AWS Libfabric" \
            -x NCCL_GIN_ENABLE=0 \
            -x NCCL_PROTO=$NCCL_PROTO \
            -x NCCL_ALGO=$NCCL_ALGO \
            -x NCCL_TESTS_SPLIT_MASK=$NCCL_TESTS_SPLIT_MASK \
            -x NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS \
            -x NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS \
            -x NCCL_NTHREADS=$NCCL_NTHREADS \
            -x ${buffsize_env}=${bsize} \
            -x GAUGE_MODE=$GAUGE_MODE \
            -x GAUGE_ITERATION=$GAUGE_ITERATION \
            -x GAUGE_MESSAGE_SIZE=$GAUGE_MESSAGE_SIZE \
            -x GAUGE_HEO=$GAUGE_HEO \
            -x GAUGE_OUT_DIRE=$GAUGE_OUT_DIRE \
            -x GAUGE_OUT_SUFFIX=$GAUGE_OUT_SUFFIX \
            -x GAUGE_TWO_L_MS=$GAUGE_TWO_L_MS \
            -x GAUGE_BW_DUAL_GBS=$GAUGE_BW_DUAL_GBS \
            -x GAUGE_BW_SINGLE_GBS=$GAUGE_BW_SINGLE_GBS \
            -x GAUGE_SINGLE_RAIL_THRESH_KB=$GAUGE_SINGLE_RAIL_THRESH_KB \
            -x GAUGE_STDOUT_ONLY_RANK0=$GAUGE_STDOUT_ONLY_RANK0 \
            -x GAUGE_INPLACE=1 \
            -N $ppn \
            --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
            $NCCL_GAUGE_HOME/gauge/build/allreduce_ring_gauge_n_1.exe "0" "0"
    done
    done
   done
  done
done

echo "=== AllReduce Ring Gauge run complete ==="
echo "Output directory: $GAUGE_OUT_DIRE"
