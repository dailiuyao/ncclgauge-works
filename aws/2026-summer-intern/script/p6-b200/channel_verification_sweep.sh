#!/bin/bash
#SBATCH -N4 --exclusive
#SBATCH -p p6-odcr-queue
#SBATCH -J chan-verify-p6b200
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p6-b200/channel_verify/%x-%j.out
#SBATCH -t 2:00:00

# Channel-selection verification sweep on P6-B200 (per channel_selection_verification.md).
# 4 nodes x 8 GPUs = 32 ranks. Runs nccl-tests all_reduce_perf across
# (mask, algo, proto) = 2 x 2 x 3 = 12 forced combos + 2 AUTO runs.
#
# On 4-node B200 with 0x0:
#   nRanks=32, num_nodes*8 == num_ranks -> B200 plugin OVERRIDE ACTIVE for
#   AllReduce+Tree+LL128 in [4MiB, 32MiB] (regions.cpp:2114-2119). We WILL
#   see nChannels ∈ {16,24,32} in that window regardless of Layer-A.
#
# On 4-node B200 with 0x7:
#   nRanks=4, num_nodes*8 != num_ranks -> override does NOT fire. Pure
#   Layer-A + Phase-2 (same shape as P5en 0x7).
#
# Measurement (identical to the P5en sweep):
#   Per-op measured nChannels comes from rank-0's NCCL_TUNING line
#     "AllReduce: <N> Bytes -> Algo <A> proto <P> channel{Lo..Hi}={lo..hi}"
#   emitted by enqueue.cc:805-807 -> measured = hi - lo + 1.
#   N = comm->nChannels from init "Channel XX/YY" lines.
#   The B200 override ALSO prints "Setting nChannels to X at nBytes=Y" from
#   regions.cpp:2128 -- capture those too to sanity-check the override.

set -eu

export PMIX_MCA_gds=hash
export libfabric_dir=/opt/amazon/efa
export ppn=8

NCCL_GAUGE_HOME=/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern
# Plain (non-profiling) NCCL + plugin -- same choice as the P5en sweep to
# avoid unresolved profiling shims (net_post_time).
nccl=/home/liuyaod/software/nccl
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl
nccl_tests=/home/liuyaod/software/nccl-tests

OUT_DIR="$NCCL_GAUGE_HOME/out/p6-b200/channel_verify"
mkdir -p "$OUT_DIR"

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
  echo "  log=$log"
  echo "=================================================================="

  local algo_flag="" proto_flag=""
  if [[ "$algo" != "AUTO" ]]; then algo_flag="-x NCCL_ALGO=$algo"; fi
  if [[ "$proto" != "AUTO" ]]; then proto_flag="-x NCCL_PROTO=$proto"; fi

  # -b 1K -e 1G -f 2: 21 sizes.
  # -n 20 -w 5: 5 warmup + 20 iters (rank-0 TUNING logs once per op).
  #
  # NCCL_TUNER_PLUGIN=libnccl-tuner-ofi.so: force-load the aws-ofi-nccl tuner
  # so the B200 override path (regions.cpp) is active. NCCL 2.27+ usually
  # auto-discovers it if in LD_LIBRARY_PATH, but be explicit -- the entire
  # point is to trigger the B200-only override rule.
  /opt/amazon/openmpi/bin/mpirun \
      -x LD_LIBRARY_PATH=$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:${LD_LIBRARY_PATH:-} \
      -x NCCL_DEBUG=INFO \
      -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV,GRAPH \
      -x NCCL_NET="AWS Libfabric" \
      -x NCCL_TUNER_PLUGIN=libnccl-tuner-ofi.so \
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
