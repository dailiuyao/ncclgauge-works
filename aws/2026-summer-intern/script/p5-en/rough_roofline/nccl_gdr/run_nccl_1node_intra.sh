#!/bin/bash
#SBATCH -N1 --exclusive
#SBATCH -p p5en-odcr-queue
#SBATCH -J nccl_test_1node_intra_p5en
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p5-en/rough_roofline/nccl_gdr/%x.out
#SBATCH --open-mode=append
#SBATCH -t 1:59:00

# Single-node intra-machine (NVLink only) experiment to isolate protocol overhead
# from network. Hypothesis: if LL is still much slower than Simple on large
# messages WITHOUT any network in the path, then the GPU-side receive kernel
# (per-line flag poll + reduce) is the dominant cost, not the network.

set -e

export libfabric_dir=/opt/amazon/efa
export nccl_test="all_reduce"
export ppn=8 # p5en.48xlarge has 8 GPU per node.

nccl=/home/liuyaod/software/nccl
nccl_tests=/home/liuyaod/software/nccl-tests
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl

# Display SLURM job information
echo "=========================================="
echo "NCCL Test Job Information (P5EN, single node, NVLink only)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "Expanded Nodelist:"
scontrol show hostname $SLURM_JOB_NODELIST
echo "=========================================="
echo ""

time_start=`date +%s`

# Resume support: skip configs whose "Config:" header is followed by a
# "# Avg bus bandwidth" line in the existing output file.
out_file="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p5-en/rough_roofline/nccl_gdr/${SLURM_JOB_NAME}.out"
done_configs=""
if [ -f "$out_file" ]; then
  done_configs=$(awk '
    /^Config: NCCL_ALGO=/ {
      if (cfg != "" && have_bw) print cfg
      cfg = $0; have_bw = 0; next
    }
    /^# Avg bus bandwidth/ { have_bw = 1 }
    END { if (cfg != "" && have_bw) print cfg }
  ' "$out_file")
fi

for algo in Tree Ring; do
  for proto in Simple LL LL128; do
    if [ "$proto" = "Simple" ]; then
      max_size="4G"
    else
      max_size="1G"
    fi
    for nch in 1 2 4 8 16; do
      config_line="Config: NCCL_ALGO=${algo}, NCCL_PROTO=${proto}, NCCL_MIN_NCHANNELS=${nch}, NCCL_MAX_NCHANNELS=${nch}, max_size=${max_size}"
      if echo "$done_configs" | grep -qxF "$config_line"; then
        echo ""
        echo "Skipping already-completed config: ${algo}/${proto}/nch=${nch}"
        continue
      fi

      echo ""
      echo "=========================================="
      echo "$config_line"
      echo "=========================================="

      /opt/amazon/openmpi/bin/mpirun \
      -x LD_LIBRARY_PATH=$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:$LD_LIBRARY_PATH \
      -x NCCL_PROTO=${proto} \
      -x NCCL_ALGO=${algo} \
      -x NCCL_DEBUG=INFO \
      -x NCCL_DEBUG_SUBSYS=ENV \
      -x NCCL_MIN_NCHANNELS=${nch} \
      -x NCCL_MAX_NCHANNELS=${nch} \
      -x NCCL_TESTS_SPLIT_MASK=0x0 \
      -x NCCL_NET="AWS Libfabric" \
      -N $ppn \
      --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
      bash -c '
        '"$nccl_tests/build/${nccl_test}_perf"' -b 1K -e '"${max_size}"' -f 2 -g 1 -c 1 -w 20 -n 100
      '
    done
  done
done

time_end=`date +%s`
echo
echo Total Execution Time: `expr $time_end - $time_start` seconds.
