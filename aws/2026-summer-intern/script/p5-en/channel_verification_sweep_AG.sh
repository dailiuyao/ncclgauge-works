#!/bin/bash
#SBATCH -N16 --exclusive
#SBATCH -p p5en-odcr-queue
#SBATCH -J chan-verify-p5en-AG
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p5-en/channel_verify_AG/%x-%j.out
#SBATCH -t 2:00:00

# AllGather channel-selection verification on P5en, 16 nodes x 8 GPUs = 128 ranks.
#
# Only 0x7 mask (each sub-comm: 16 ranks, one rank/node, num_nodes==num_ranks).
# Only Ring algorithm — AG in NCCL supports {Ring, PAT, NVLS}; NCCL_ALGO=Tree
# fails with 'invalid usage'.
#
# Per-protocol size ranges (chosen to bracket the ramp-up region for each proto):
#   Simple:  1K .. 8G   (24 sizes)   Layer-A cost = 32 KiB
#   LL:      1K .. 128M (18 sizes)   Layer-A cost = 4 KiB (nRanks factor pushes ramp late)
#   LL128:   8K .. 128M (15 sizes)   Layer-A cost = 5 KiB
#
# Log parsing: rank-0 emits "AllGather: <B> Bytes -> Algo <A> proto <P> channel{Lo..Hi}={lo..hi}"
# at enqueue.cc:794-796 where <B> = task->count * eltSize = per-rank sendbytes = S/nRanks.
# The analyzer reconstructs S = B * nRanks.
#
# Memory note: recvbuf per rank = nRanks * S. For 0x7 sub-comm nRanks=16:
#   S=8G -> recvbuf=128 GiB per rank. H200 has 141 GiB, tight but fits.
# nccl-tests auto-reduces maxBytes if needed (common.cu:1306-1309).

set -eu

export PMIX_MCA_gds=hash
export libfabric_dir=/opt/amazon/efa
export ppn=8

NCCL_GAUGE_HOME=/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern
nccl=/home/liuyaod/software/nccl
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl
nccl_tests=/home/liuyaod/software/nccl-tests

OUT_DIR="$NCCL_GAUGE_HOME/out/p5-en/channel_verify_AG"
mkdir -p "$OUT_DIR"

# CONFIGS: "mask algo proto min_size max_size"
CONFIGS=(
  "0x7 Ring Simple 1K 8G"
  "0x7 Ring LL     1K 128M"
  "0x7 Ring LL128  8K 128M"
)

run_one () {
  local mask="$1" algo="$2" proto="$3" min_b="$4" max_b="$5"
  local tag="mask${mask}_algo${algo}_proto${proto}"
  local log="$OUT_DIR/${tag}.log"

  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] Running $tag  (-b $min_b -e $max_b)"
  echo "  log=$log"
  echo "=================================================================="

  /opt/amazon/openmpi/bin/mpirun \
      -x LD_LIBRARY_PATH=$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:${LD_LIBRARY_PATH:-} \
      -x NCCL_DEBUG=INFO \
      -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV,GRAPH \
      -x NCCL_NET="AWS Libfabric" \
      -x NCCL_TUNER_PLUGIN=libnccl-tuner-ofi.so \
      -x NCCL_TESTS_SPLIT_MASK="$mask" \
      -x NCCL_ALGO=$algo \
      -x NCCL_PROTO=$proto \
      -N $ppn \
      --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
      "$nccl_tests/build/all_gather_perf" -b $min_b -e $max_b -f 2 -n 20 -w 5 \
      > "$log" 2>&1 || {
        echo "  !!! mpirun failed for $tag (exit=$?), see $log"
      }
  echo "  done $(wc -l < "$log") log lines"
}

for row in "${CONFIGS[@]}"; do
  read -r mask algo proto min_b max_b <<<"$row"
  run_one "$mask" "$algo" "$proto" "$min_b" "$max_b"
done

echo "=== sweep complete ==="
ls -la "$OUT_DIR"
