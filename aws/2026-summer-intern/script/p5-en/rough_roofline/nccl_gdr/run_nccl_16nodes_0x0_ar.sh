#!/bin/bash
#SBATCH -N16 --exclusive
#SBATCH -p p5en-odcr-queue
#SBATCH -J nccl_test_16nodes_0x0_p5en_ringtopo
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p5-en/rough_roofline/nccl_gdr/%x.out
#SBATCH --open-mode=truncate
#SBATCH -t 0:30:00

set -e

export libfabric_dir=/opt/amazon/efa
export nccl_test="all_reduce"
export ppn=8 # p5en.48xlarge has 8 GPU per node.

nccl=/home/liuyaod/software/nccl
nccl_tests=/home/liuyaod/software/nccl-tests
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl

# Display SLURM job information
echo "=========================================="
echo "NCCL Test Job Information (P5EN)"
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

# This script writes its combined stdout+stderr to `${SLURM_JOB_NAME}.out` via
# the -o directive above. Point `out_file` at the same path so we can grep the
# NCCL init lines we just printed after each mpirun.
out_file="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p5-en/rough_roofline/nccl_gdr/${SLURM_JOB_NAME}.out"

# Ring-topology probe: for each nch in {8, 16}, launch a tiny (1 KB / 1 iter)
# all_reduce with NCCL_DEBUG=INFO (subsys INIT|GRAPH) and dump the per-channel
# "Channel XX/YY : R0 R1 ... " ring order + "R -> R'" send/recv edges, then
# grep out rank 0's send targets across the nch channels.
for nch in 8 16; do
  section="=== nch=${nch} (Ring, Simple, SPLIT_MASK=0x0, 16 nodes x ppn=${ppn} = 128 ranks) ==="
  echo ""
  echo "$section"

  /opt/amazon/openmpi/bin/mpirun \
    -x LD_LIBRARY_PATH=$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:$LD_LIBRARY_PATH \
    -x NCCL_PROTO=Simple \
    -x NCCL_ALGO=Ring \
    -x NCCL_DEBUG=INFO \
    -x NCCL_DEBUG_SUBSYS=INIT,GRAPH \
    -x NCCL_MIN_NCHANNELS=${nch} \
    -x NCCL_MAX_NCHANNELS=${nch} \
    -x NCCL_TESTS_SPLIT_MASK=0x0 \
    -x NCCL_NET="AWS Libfabric" \
    -N $ppn \
    --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    bash -c '
      '"$nccl_tests/build/${nccl_test}_perf"' -b 1K -e 1K -f 2 -g 1 -c 0 -w 0 -n 1
    '

  echo ""
  echo "--- Summary: rank 0 send targets (nch=${nch}) ---"
  echo "Per-channel ring order (lines matching 'Channel NN/${nch} : ...'):"
  grep -E "Channel [0-9]+/${nch} : " "$out_file" | sort -u | head -${nch}
  echo ""
  echo "Rank 0 [send] edges printed by NCCL (each is one send target):"
  grep -E "Channel [0-9]+/[0-9]+ : 0\[[0-9]+\] -> [0-9]+\[[0-9]+\] \[send\]" "$out_file" | sort -u
done

time_end=`date +%s`
echo
echo Total Execution Time: `expr $time_end - $time_start` seconds.
