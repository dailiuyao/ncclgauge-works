# EFA Latency Testing with Perftest

This directory contains scripts to build perftest and run EFA latency measurements between p6 instances.

## Directory Structure

```
rough_roofline/
├── build_perftest.sh           # Script to build perftest
├── run_efa_latency_test.sh     # Script to run EFA latency test
├── detect_numa_config.sh       # Auto-detect optimal CPU/EFA pairs
├── README_perftest.md          # This file (quick start guide)
├── NUMA_TOPOLOGY.md            # Deep dive: Why CPU pinning matters
├── output/                      # Test results directory
│   ├── client_*.log            # Client-side test output
│   ├── server_*.log            # Server-side test output
│   └── summary_*.txt           # Test summary (auto-generated)
└── [other job scripts...]
```

## Perftest Installation

Perftest is installed to: **`~/software/perftest/`**

### First-time Setup

1. Build perftest (only needs to be done once):
   ```bash
   ./build_perftest.sh
   ```

This will:
- Clone perftest v25.10.0-0.128 from GitHub
- Build with EFA and CUDA support
- Install to `~/software/perftest/`
- Source code will be in `~/software/perftest-src/`

## Running Latency Tests

### Quick Start

```bash
./run_efa_latency_test.sh
```

### What It Does

1. Allocates two p6 nodes (p6b20048xlarge-1 and p6b20048xlarge-2)
2. Starts perftest server on node 1
3. Runs perftest client on node 2
4. Measures RDMA write latency across message sizes (2B to 8MB)
5. Saves results to `output/` directory with timestamp

### Test Configuration

- **EFA Device**: rdmap113s0 (NUMA node 1)
- **CPU Core**: 48 (pinned for consistent performance)
- **Transport**: SRD (Scalable Reliable Datagram)
- **Iterations**: 10,000 per message size
- **Message Sizes**: 2B, 4B, 8B, 16B, 32B, 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB, 1MB, 2MB, 4MB, 8MB

### Output Files

Each test run creates three files in `output/`:
- `client_<timestamp>.log` - Full client-side output with all latency measurements
- `server_<timestamp>.log` - Server-side output
- `summary_<timestamp>.txt` - Extracted key metrics for quick review

### Customizing the Test

Edit `run_efa_latency_test.sh` to change:
- `NODE1` / `NODE2` - Target nodes
- `EFA_DEVICE` - EFA NIC to use
- `CPU_CORE` - CPU core for pinning

**Important**: `CPU_CORE` and `EFA_DEVICE` must be on the **same NUMA node** for accurate results!

Use the detection script to find optimal pairs:
```bash
srun -N 1 -w <node-name> ./detect_numa_config.sh
```

For detailed explanation of why this matters, see **[NUMA_TOPOLOGY.md](NUMA_TOPOLOGY.md)**

## Understanding the Results

### Latency Metrics

Each row in the output shows:
- `#bytes` - Message size
- `t_min` - Minimum latency (μs)
- `t_max` - Maximum latency (μs)
- `t_typical` - Typical latency (μs)
- `t_avg` - Average latency (μs)
- `t_stdev` - Standard deviation (μs)
- `99% percentile` - 99th percentile latency (μs)
- `99.9% percentile` - 99.9th percentile latency (μs)

### Example Results

From the test run on 2026-06-03:

| Message Size | Typical (μs) | Average (μs) | 99th %ile (μs) |
|-------------|--------------|--------------|----------------|
| 2 B         | 14.63        | 14.90        | 18.41          |
| 64 B        | 14.63        | 14.70        | 16.79          |
| 1 KB        | 14.86        | 14.92        | 16.74          |
| 4 KB        | 15.44        | 15.51        | 18.50          |
| 64 KB       | 21.61        | 21.70        | 24.73          |
| 1 MB        | 49.27        | 49.47        | 53.97          |
| 8 MB        | 219.54       | 219.97       | 227.26         |

## Finding EFA Devices

To find available EFA devices on your nodes:

```bash
# List all InfiniBand devices
ls /sys/class/infiniband/

# Find EFA devices specifically
fi_info -p efa -t FI_EP_RDM | grep "domain:"

# Check NUMA node for a device
cat /sys/class/infiniband/<device_name>/device/numa_node

# Find CPUs for that NUMA node
cat /sys/devices/system/node/node<X>/cpulist
```

## Other Perftest Tools

The following tools are also available in `~/software/perftest/bin/`:
- `ib_write_lat` / `ib_write_bw` - Write latency/bandwidth
- `ib_read_lat` / `ib_read_bw` - Read latency/bandwidth
- `ib_send_lat` / `ib_send_bw` - Send latency/bandwidth
- `ib_atomic_lat` / `ib_atomic_bw` - Atomic operation latency/bandwidth
- `raw_ethernet_*` - Raw Ethernet tests

## Troubleshooting

### "perftest not found" error
Run `./build_perftest.sh` first to build perftest.

### "Failed to allocate nodes" error
Check node availability with:
```bash
sinfo -N | grep p6-odcr
```

### Permission denied on nodes
Ensure your SSH key is authorized on the compute nodes (usually handled by Slurm).

## References

- [perftest GitHub](https://github.com/linux-rdma/perftest)
- [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Perftest README](https://github.com/linux-rdma/perftest/blob/master/README)
