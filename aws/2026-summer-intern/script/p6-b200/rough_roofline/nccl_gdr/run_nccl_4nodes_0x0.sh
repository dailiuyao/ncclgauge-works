#!/bin/bash
#SBATCH -N4 --exclusive
#SBATCH -p p6-odcr-queue
#SBATCH --nodelist=p6-odcr-queue-dy-p6b20048xlarge-[1,2,4,5]
#SBATCH --exclude=p6-odcr-queue-dy-p6b20048xlarge-[6,8]
#SBATCH -J nccl_test_4nodes_0x0_ring
#SBATCH -o /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p6-b200/rough_roofline/nccl_gdr/%x-%j.out
#SBATCH -t 0:29:00

set -ex

export libfabric_dir=/opt/amazon/efa
export nccl_test="all_reduce"
export ppn=8 # p6.48xlarge has 8 GPU per node.

nccl=/home/liuyaod/software/nccl
nccl_tests=/home/liuyaod/software/nccl-tests
ofi_plugin=/home/liuyaod/software/aws-ofi-nccl

# Display SLURM job information
echo "=========================================="
echo "NCCL Test Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "Expanded Nodelist:"
scontrol show hostname $SLURM_JOB_NODELIST
echo "=========================================="
echo ""

# === Pre-flight check: GPU NVL fabric must be in "ready" state on every node ===
# B200 uses NVL Fabric (NVLink 5 + NVSwitch 4) which is initialized by the
# nvidia-fabricmanager service. If any GPU on any allocated node is still in
# "System is not in ready state", cudaSetDevice on that GPU will fail with
# "CUDA-capable device(s) is/are busy or unavailable". Detect this up front
# and bail out instead of running the test for 10+ minutes only to fail.
echo "=== Pre-flight: checking GPU fabric state on each node ==="
fabric_problem=0
# Temporarily disable -e for the loop below: grep -c returns exit code 1 when
# it finds 0 matches, which would otherwise kill the script under set -e.
set +e
for node in $(scontrol show hostname $SLURM_JOB_NODELIST); do
    not_ready=$(srun --nodes=1 --nodelist=$node --ntasks=1 \
                     bash -c 'nvidia-smi -q 2>&1 | grep -c "GPU Fabric GUID.*not in ready state"' 2>/dev/null)
    not_ready=${not_ready:-0}
    if [ "$not_ready" -gt 0 ]; then
        echo "  ✗ $node: $not_ready GPU(s) have fabric NOT ready — skipping job"
        echo "    --- raw nvidia-smi fabric state on $node ---"
        srun --nodes=1 --nodelist=$node --ntasks=1 \
             bash -c 'nvidia-smi -q 2>&1 | awk "/GPU 00000000:/{gpu=\$0} /GPU Fabric GUID/{print gpu \" => \" \$0}"' 2>&1 | sed 's/^/      /'
        echo "    --- nvidia-fabricmanager status on $node ---"
        srun --nodes=1 --nodelist=$node --ntasks=1 \
             bash -c 'systemctl is-active nvidia-fabricmanager; systemctl status nvidia-fabricmanager --no-pager -n 15 2>&1' 2>&1 | sed 's/^/      /'
        echo ""
        fabric_problem=1
    else
        echo "  ✓ $node: all 8 GPUs have fabric ready"
    fi
done
set -e
if [ "$fabric_problem" -eq 1 ]; then
    echo ""
    echo "ABORTING: at least one allocated node has GPUs whose NVL fabric is not"
    echo "ready. Re-submit with --exclude or pick different --nodelist nodes."
    exit 1
fi
echo "=========================================="

time_start=`date +%s`

/opt/amazon/openmpi/bin/mpirun \
-x LD_LIBRARY_PATH=$nccl/build/lib:/usr/local/cuda/lib64:${libfabric_dir}/lib:/opt/amazon/openmpi/lib:$ofi_plugin/install/lib:$LD_LIBRARY_PATH \
-x NCCL_PROTO=Simple \
-x NCCL_ALGO=Ring \
-x NCCL_DEBUG=INFO \
-x NCCL_TESTS_SPLIT_MASK=0x0 \
-x NCCL_NET="AWS Libfabric" \
-N $ppn \
--mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
bash -c '
  '"$nccl_tests/build/${nccl_test}_perf"' -b 1K -e 1G -f 2 -g 1 -c 1 -w 20 -n 100
'

time_end=`date +%s`
echo
echo Total Execution Time: `expr $time_end - $time_start` seconds.
