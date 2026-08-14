#!/bin/bash
#SBATCH -N4 --exclusive
#SBATCH -p p5en-odcr-queue
#SBATCH -J chan-verify-p5en
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p5-en/channel_verify_AR/%x-%j.out
#SBATCH -t 2:00:00

# Channel-selection verification sweep on P5en (per channel_selection_verification.md).
# 4 nodes x 8 GPUs = 32 ranks. Runs nccl-tests all_reduce_perf across
# (mask, algo, proto) = 2 x 2 x 3 = 12 forced combos + 2 auto combos.
# For every op, rank 0 emits at NCCL_TUNING level:
#     "AllReduce: <N> Bytes -> Algo <A> proto <P> channel{Lo..Hi}={lo..hi}"
# from nccl/src/enqueue.cc:805-807 — that's the measured nChannels per op.
# The init log also prints "Channel XX/YY" lines: YY = comm->nChannels (N).

set -eu

export PMIX_MCA_gds=hash
export libfabric_dir=/opt/amazon/efa
export ppn=8

NCCL_GAUGE_HOME=/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern
# Use the plain (non-profiling) NCCL and aws-ofi-nccl builds — the *_profile
# builds have unresolved profiling shims (net_post_time, ...) that require
# LD_PRELOAD of libprofiling_arrays.so, unnecessary for reading TUNING logs.
nccl=/home/liuyaod/software/nccl
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl
nccl_tests=/home/liuyaod/software/nccl-tests

OUT_DIR="$NCCL_GAUGE_HOME/out/p5-en/channel_verify_AR"
mkdir -p "$OUT_DIR"

# One row per (mask, algo, proto). "AUTO" = don't force NCCL_ALGO/PROTO.
CONFIGS=(
  "0x0 Ring   Simple"
  "0x0 Ring   LL"
  "0x0 Ring   LL128"
  "0x0 Tree   Simple"
  "0x0 Tree   LL"
  "0x0 Tree   LL128"
  "0x7 Ring   Simple"
  "0x7 Ring   LL"
  "0x7 Ring   LL128"
  "0x7 Tree   Simple"
  "0x7 Tree   LL"
  "0x7 Tree   LL128"
  "0x0 AUTO   AUTO"
  "0x7 AUTO   AUTO"
)

run_one () {
  local mask="$1" algo="$2" proto="$3"
  local tag="mask${mask}_algo${algo}_proto${proto}"
  local log="$OUT_DIR/${tag}.log"

  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] Running $tag"
  echo "  mask=$mask algo=$algo proto=$proto"
  echo "  log=$log"
  echo "=================================================================="

  # Build the -x extra flags for algo/proto if forced.
  local algo_flag="" proto_flag=""
  if [[ "$algo" != "AUTO" ]]; then algo_flag="-x NCCL_ALGO=$algo"; fi
  if [[ "$proto" != "AUTO" ]]; then proto_flag="-x NCCL_PROTO=$proto"; fi

  # -b 1K -e 1G -f 2 sweeps 1K,2K,4K,...,1G (21 sizes).
  # -n 20 -w 5: 5 warmup + 20 iter each (rank-0 TUNING log fires once per op,
  #   so 20 identical lines per size — take the first).
  # Redirect to $log for parsing.
  /opt/amazon/openmpi/bin/mpirun \
      -x LD_LIBRARY_PATH=$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:${LD_LIBRARY_PATH:-} \
      -x NCCL_DEBUG=INFO \
      -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV,GRAPH \
      -x NCCL_NET="AWS Libfabric" \
      -x NCCL_TESTS_SPLIT_MASK="$mask" \
      $algo_flag $proto_flag \
      -N $ppn \
      --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
      "$nccl_tests/build/all_reduce_perf" -b 1K -e 1G -f 2 -n 20 -w 5 \
      > "$log" 2>&1 || {
        echo "  !!! mpirun failed for $tag (exit=$?), see $log"
      }
  echo "  done $(wc -l < "$log") log lines"
}

for row in "${CONFIGS[@]}"; do
  read -r mask algo proto <<<"$row"
  run_one "$mask" "$algo" "$proto"
done

echo "=== sweep complete ==="
ls -la "$OUT_DIR"
