# NCCL P2P Profiling Scripts for P6 B200

Production-ready scripts for profiling NCCL point-to-point communication on AWS P6 B200 instances with timing metrics.

## Hardware Specifications

- **Instance**: P6 (p6.48xlarge)
- **GPU**: NVIDIA B200 (Blackwell architecture)
- **Compute Capability**: 10.0 (sm_100)
- **Network**: EFA (Elastic Fabric Adapter)
- **Queue**: p6-odcr-queue

## Quick Start

```bash
cd /home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/script/p6-b200

# 1. Build NCCL library for B200
bash nccl_profile_build.sh

# 2. Build gauge executables for B200
bash pping-gauge-build.sh

# 3. Run profiling test
sbatch pping-gauge-run-vary-msg.sh

# 4. Check results
ls -lh ../../out/p6-b200/
grep "^INFO:" ../../out/p6-b200/pping-gauge-msg-p6b200-*.out
```

## Output Locations

All P6 B200 outputs are isolated in dedicated directories:

- **Binaries**: `../../gauge/p6-b200/`
- **Test Results**: `../../out/p6-b200/`
- **SLURM Logs**: `../../out/p6-b200/pping-gauge-msg-p6b200-<jobid>.out`

## Scripts

### `nccl_profile_build.sh`
Builds NCCL library and AWS OFI plugin for B200 (sm_100).
- NCCL: `/home/liuyaod/software/nccl_2_30_4_profile`
- Uses `make lib` (skips utilities)
- Profiling variables undefined in library (resolved by gauge at runtime)
- **Architecture**: Blackwell (compute_100, sm_100)

### `pping-gauge-build.sh`
Builds gauge executables for P2P profiling on B200.
- Output: `../../gauge/p6-b200/pping*_gauge_n_1.exe`
- Gauge defines profiling variables and timing arrays
- **Architecture**: sm_100 (B200 specific)

### `pping-gauge-run-vary-msg.sh`
Runs P2P profiling on 2 P6 nodes via SLURM.
- **Message sizes**: 8KB to 64KB (8KB steps)
- **Iterations**: 100 per size
- **Protocol**: Simple, 1 channel
- **Output**: `../../out/p6-b200/pping-gauge-msg-p6b200-<jobid>.out`

## Key Differences from P5-EN

| Parameter | P5-EN (H100) | P6-B200 |
|-----------|--------------|---------|
| GPU Architecture | Hopper (sm_90) | Blackwell (sm_100) |
| Compute Capability | 9.0 | 10.0 |
| SLURM Queue | p5en-odcr-queue | p6-odcr-queue |
| Binary Location | `gauge/` | `gauge/p6-b200/` |
| Output Location | `out/` | `out/p6-b200/` |

## Environment Variables

**NCCL:**
- `NCCL_PROTO=Simple` - Force Simple protocol
- `NCCL_MIN_NCHANNELS=1` / `NCCL_MAX_NCHANNELS=1` - Force 1 channel
- `NCCL_NTHREADS=512` - Threads per channel

**Gauge:**
- `GAUGE_MODE=pping` - P2P message mode
- `GAUGE_ITERATION=100` - Iterations per test
- `GAUGE_MESSAGE_SIZE=<size>` - Message size in bytes (8KB-64KB range)
- `GAUGE_HEO=inter` - Inter-node communication

## Architecture

**Symbol Resolution (v2.27.6 approach):**
- NCCL library: Undefined symbols for profiling variables/timing arrays
- Gauge executable: Defines these variables/arrays
- Dynamic linker: Resolves at runtime → both access same storage

## Expected Output Format

```
INFO: heo(inter)_mode(pping)_message size(8192)_nchannels(1)_nthreads(512)_nmessages(1)_chunksize(524288)_protocol(simple)_send-d(0)_recv-d(0)_iteration(100)
```

**Profiling Metrics:**
- `nchannels`: Number of channels used
- `protocol`: NCCL protocol (simple/LL/LL128)
- `chunksize`: Chunk size in bytes
- **T0-T7**: Per-chunk network timing
- **T2, T8**: Function-level timing
- **T9-1, T10-1**: Per-message timing

## Troubleshooting

**If compilation fails with "unsupported GPU architecture":**
- Verify CUDA version supports compute_100: `nvcc --version`
- B200 requires CUDA 12.0 or later

**If runtime fails with NCCL errors:**
- Check EFA is available: `fi_info -p efa`
- Verify NCCL plugin: `ls $ofi_plugin/install/lib/`

**If SLURM job queues indefinitely:**
- Check queue status: `squeue -p p6-odcr-queue`
- Verify node availability: `sinfo -p p6-odcr-queue`

## Migration Notes

This is adapted from P5-EN scripts. Key changes:
1. GPU architecture: sm_90 → sm_100
2. SLURM queue: p5en-odcr-queue → p6-odcr-queue
3. Isolated output directories for P6 B200
4. Message size range optimized for P6 network characteristics
