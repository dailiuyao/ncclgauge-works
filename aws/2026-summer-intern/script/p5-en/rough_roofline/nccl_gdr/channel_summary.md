# channel_verify_AR — Number of Channels Report

## 1. Test Configuration

| Field | Value |
| --- | --- |
| Collective | `all_reduce_perf -b 1K -e 1G -f 2 -n 20 -w 5` |
| Cluster | 4 × `p5en.48xlarge` (32 GPUs, NVIDIA H200) |
| Stack | NCCL 2.30.4 + aws-ofi-nccl |
| Slurm job | 46366 (6:18 elapsed) |
| Sweep dimensions | 21 message sizes (1 KiB → 1 GiB, ×2 step) × 14 configs |
| Configs | 2 masks × (3 forced protos × 2 algos + 1 AUTO/AUTO) |
| Signal source | `NCCL_TUNING` line — per-op `nc = channelHi − channelLo + 1` |
| N (upper bound) | Read from init-time `Channel XX/YY` lines |

**Masks:**

| Mask | comm nRanks | N (`comm->nChannels`) | Ring hops | Notes |
| --- | ---: | ---: | ---: | --- |
| `0x0` | 32 | 16 | 32 | Global comm — one channel per pair of nodes |
| `0x7` | 4  | 4  | 4  | ppn-split — 8 sub-comms of 4 ranks each; one rank per node |

Combined predictor `nc = min(Layer-A cap, phase-2 cap)` matches **294/294 measured points** across all 14 configs.

## 2. Number-of-channels Table (all points)

Legend: values are the measured `nc` (per-op channel count). Column groups are the 14 (mask, algo, proto) configs. `AUTO` columns show what the tuner picked at each size.

### mask 0x0 (N = 16, nRanks = 32)

| size | Ring-LL | Ring-LL128 | Ring-Simple | Tree-LL | Tree-LL128 | Tree-Simple | AUTO nc | AUTO (algo/proto) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 K   | 1  | 1  | 1  | 1  | 1  | 1  | 1  | Tree/LL      |
| 2 K   | 1  | 1  | 1  | 1  | 1  | 1  | 1  | Tree/LL      |
| 4 K   | 1  | 1  | 1  | 1  | 1  | 1  | 1  | Tree/LL      |
| 8 K   | 1  | 1  | 1  | 2  | 1  | 1  | 2  | Tree/LL      |
| 16 K  | 1  | 1  | 1  | 4  | 1  | 1  | 4  | Tree/LL      |
| 32 K  | 1  | 2  | 1  | 8  | 2  | 1  | 8  | Tree/LL      |
| 64 K  | 1  | 4  | 2  | 16 | 4  | 2  | 16 | Tree/LL      |
| 128 K | 1  | 8  | 4  | 16 | 8  | 4  | 16 | Tree/LL      |
| 256 K | 2  | 16 | 8  | 16 | 16 | 8  | 16 | Tree/LL128   |
| 512 K | 4  | 16 | 16 | 16 | 16 | 16 | 16 | Tree/LL128   |
| 1 M   | 8  | 16 | 16 | 16 | 16 | 16 | 16 | Tree/LL128   |
| 2 M   | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Tree/LL128   |
| 4 M   | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Tree/LL128   |
| 8 M   | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Tree/LL128   |
| 16 M  | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Tree/LL128   |
| 32 M  | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Ring/LL128   |
| 64 M  | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Ring/LL128   |
| 128 M | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Ring/LL128   |
| 256 M | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Ring/LL128   |
| 512 M | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Ring/Simple  |
| 1 G   | 16 | 16 | 16 | 16 | 16 | 16 | 16 | Ring/Simple  |

### mask 0x7 (N = 4, nRanks = 4)

| size | Ring-LL | Ring-LL128 | Ring-Simple | Tree-LL | Tree-LL128 | Tree-Simple | AUTO nc | AUTO (algo/proto) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 K   | 1 | 1 | 1 | 1 | 1 | 1 | 1 | Tree/LL      |
| 2 K   | 1 | 1 | 1 | 1 | 1 | 1 | 1 | Tree/LL      |
| 4 K   | 1 | 1 | 1 | 1 | 1 | 1 | 1 | Tree/LL      |
| 8 K   | 1 | 1 | 1 | 2 | 1 | 1 | 2 | Tree/LL      |
| 16 K  | 1 | 1 | 1 | 4 | 1 | 1 | 4 | Tree/LL      |
| 32 K  | 2 | 2 | 1 | 4 | 2 | 1 | 4 | Tree/LL      |
| 64 K  | 4 | 4 | 2 | 4 | 4 | 2 | 4 | Tree/LL      |
| 128 K | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Tree/LL128   |
| 256 K | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Tree/LL128   |
| 512 K | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Tree/Simple  |
| 1 M   | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/LL128   |
| 2 M   | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/LL128   |
| 4 M   | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/LL128   |
| 8 M   | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/LL128   |
| 16 M  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/Simple  |
| 32 M  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/Simple  |
| 64 M  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/Simple  |
| 128 M | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/Simple  |
| 256 M | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/Simple  |
| 512 M | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/Simple  |
| 1 G   | 4 | 4 | 4 | 4 | 4 | 4 | 4 | Ring/Simple  |

## 3. Saturation Points (nc first reaches N)

The size at which each config first attains its full N channels:

| Config | mask 0x0 (N=16) | mask 0x7 (N=4) |
| --- | ---: | ---: |
| Ring-LL     | 2 MiB  | 64 KiB  |
| Ring-LL128  | 256 KiB | 64 KiB  |
| Ring-Simple | 512 KiB | 128 KiB |
| Tree-LL     | 64 KiB  | 16 KiB  |
| Tree-LL128  | 256 KiB | 64 KiB  |
| Tree-Simple | 512 KiB | 128 KiB |

Analytical model per protocol (matches all measured points):

- **Simple**: saturates at `N × 32 KiB` (0x0: 16·32K = 512 K ✓, 0x7: 4·32K = 128 K ✓)
- **LL128**: saturates at `N × 16 KiB` (phase-2 traffic-cell cap dominates the Layer-A 5 KiB prediction)
- **Tree-LL**: saturates at `N × 4 KiB`
- **Ring-LL**: saturates at `N × 4 KiB × nRanks` (0x0: 16·4K·32 = 2 M ✓, 0x7: 4·4K·4 = 64 K ✓)

## 4. Key Observations

1. **Ring and Tree share the same N per mask** — determined by comm topology, not algorithm.
2. **`nc` is monotonically non-decreasing in size** for every one of the 14 configs.
3. **Protocol ramp-up ordering (fastest → slowest to saturate)**:
   `Tree-LL  <  {Ring-LL128, Tree-LL128}  <  {Ring-Simple, Tree-Simple}  <  Ring-LL`
   Ring-LL is slowest because its Layer-A cost scales with `nRanks`.
4. **Tuner behavior on P5en (AUTO/AUTO)**:
   - Latency → BW handoff: `Tree/LL → Tree/LL128 → Ring/LL128 → Ring/Simple`.
   - AUTO `nc` at every size equals the forced (algo, proto) run's `nc` at that size — confirming that on P5en the tuner only steers algo/proto and does **not** override the channel count.
5. **LL128 correction to the doc's §2 Layer-A formula**: predicted saturation was `N × 5 KiB`; the measured `N × 16 KiB` is set by phase-2 traffic cells (`MinTrafficPerChannel = 32 KiB`, `trafficPerByte = 2` for AllReduce → 16 KiB cell), documented in [enqueue.cc:652-668](../../../../software/nccl/src/enqueue.cc#L652-L668).
6. **B200 override for Tree+LL128 at 4–32 MiB** — not testable on P5en (analytical only).

## 5. Cross-check with Prediction Model

`nc_pred = min(N, ceil(size / cellSize), Layer-A cap)`

- All 294 measured points match `nc_pred` exactly (see [analysis.txt](analysis.txt)).
- Per-config match: 21/21 for every one of the 14 configs.

## References

- Detailed raw table: [analysis.txt](analysis.txt)
- Original results write-up: [RESULTS.md](RESULTS.md)
- Sweep script: [channel_verification_sweep.sh](../../../script/p5-en/channel_verification_sweep.sh)
- Analyzer: [analyze_channel_verification.py](../../../script/p5-en/analyze_channel_verification.py)
