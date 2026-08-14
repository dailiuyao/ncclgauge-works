#!/bin/bash
# Automatically detect optimal CPU core and EFA device pairs
# This ensures NUMA-local configuration for minimal latency

echo "=========================================="
echo "NUMA Topology Detection"
echo "=========================================="
echo ""

# Check if running on compute node
if [ ! -d "/sys/class/infiniband" ]; then
    echo "Warning: Not on a compute node with InfiniBand devices"
    echo "Run this with: srun -N 1 -w <node-name> ./detect_numa_config.sh"
    exit 1
fi

echo "Hostname: $(hostname)"
echo ""

echo "----------------------------------------"
echo "NUMA Nodes and CPU Cores"
echo "----------------------------------------"
for node in /sys/devices/system/node/node[0-9]*; do
    name=$(basename $node)
    cpus=$(cat $node/cpulist)
    # Extract first physical core (before comma)
    first_core=$(echo $cpus | cut -d',' -f1 | cut -d'-' -f1)
    echo "  $name: CPUs $cpus"
    echo "         First physical core: $first_core"
done
echo ""

echo "----------------------------------------"
echo "EFA Devices and NUMA Affinity"
echo "----------------------------------------"
for dev in /sys/class/infiniband/rdma*; do
    name=$(basename $dev)
    numa=$(cat $dev/device/numa_node 2>/dev/null || echo "unknown")
    if [ "$numa" != "unknown" ]; then
        # Get first physical core for this NUMA node
        cpus=$(cat /sys/devices/system/node/node${numa}/cpulist)
        first_core=$(echo $cpus | cut -d',' -f1 | cut -d'-' -f1)
        echo "  $name -> NUMA node $numa (recommended CPU: $first_core)"
    fi
done
echo ""

echo "----------------------------------------"
echo "Recommended Configurations"
echo "----------------------------------------"
echo ""
echo "For run_efa_latency_test.sh, use one of these combinations:"
echo ""

# Generate recommendations for each NUMA node
for node_num in 0 1; do
    if [ -d "/sys/devices/system/node/node${node_num}" ]; then
        cpus=$(cat /sys/devices/system/node/node${node_num}/cpulist)
        first_core=$(echo $cpus | cut -d',' -f1 | cut -d'-' -f1)

        # Find EFA devices on this NUMA node
        echo "NUMA Node ${node_num}:"
        for dev in /sys/class/infiniband/rdma*; do
            name=$(basename $dev)
            numa=$(cat $dev/device/numa_node 2>/dev/null)
            if [ "$numa" = "$node_num" ]; then
                echo "  EFA_DEVICE=\"$name\""
                echo "  CPU_CORE=\"$first_core\""
                echo ""
                break
            fi
        done
    fi
done
