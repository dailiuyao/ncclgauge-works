# Verifying NCCL AllReduce Channel-Count Selection (on P5en)

**Audience:** a Claude/engineer with access to a **4-node P5en cluster** (`p5en.48xlarge`, 8 GPUs/node = 32 GPUs), NCCL **v2.30**, and the **aws-ofi-nccl** plugin.

**Goal:** empirically verify the channel-count model below by sweeping message size **1 KiB → 1 GiB** for every combination of:
- split mask: **0x0** and **0x7**
- protocol: **Simple**, **LL**, **LL128**
- algorithm: **Ring**, **Tree**

For each (mask, proto, algo, size) point, capture the **actual number of channels** used per AllReduce and compare it against the prediction. Report where reality matches and where it diverges.

> **Why P5en and not P6-B200:** On P5en the aws-ofi-nccl tuner **never writes the channel count** — channel selection is 100% NCCL-core logic (Layer A below). That makes P5en a clean environment to validate the core model. The B200-only channel override is explained in §8 so the same model can later be applied to compute B200 channel counts.

---

## 1. Background — channels are chosen in two layers

### Layer A: NCCL core (applies to Ring and Tree identically, on ALL platforms)
File: `nccl/src/enqueue.cc`, function `topoGetAlgoInfo`, lines **1983–2027**.

After algo/proto are selected, the per-collective channel cap `nc` starts at the **whole communicator width** `comm->nChannels` and is shrunk for small messages:

```c
int nc = comm->nChannels;                        // full pool
int nt = comm->maxThreads[algo][proto];
int threadThreshold = comm->threadThresholds[algo][proto];
...
// Ring/Tree branch (enqueue.cc:2004-2010)
while (nBytes < nc * nt * threadThreshold) {
    if (nc >= 2) nc--; else break;
}
info->nMaxChannels = nc;                          // enqueue.cc:2027
```

**Consequences:**
1. Channel count grows with message size, saturating at `comm->nChannels` for large messages and dropping toward **1** for small messages.
2. **Ring and Tree share the same pool** `comm->nChannels`. Init forces them equal (`nccl/src/init.cc:1414`):
   ```c
   comm->nChannels = treeGraph->nChannels = ringGraph->nChannels
                   = std::min(treeGraph->nChannels, ringGraph->nChannels);
   ```
   So Ring and Tree have the **same channel ceiling** at runtime. Only NVLS uses a separate pool (`comm->nvlsChannels`). This is a key claim to verify.
3. Protocol changes the curve only through `nt` (`maxThreads`) and `threadThreshold` — there is **no separate hard per-protocol channel cap**.

### Layer B: aws-ofi-nccl external tuner
Files: `aws-ofi-nccl/src/tuner/nccl_ofi_regions.cpp`, `include/tuner/*`.

The tuner is a NCCL "external tuner". On a recognized AWS instance it is **enabled by default** and does two distinct things:
- **(always) picks algo+proto** by zeroing the winning cost-table cell — this happens on **P5en too**.
- **(P6-B200 only) writes the channel count** in one AllReduce special case — this does **NOT** happen on P5en.

**On P5en:** platform resolves to `NCCL_OFI_TUNER_P5EN` (`process_config.h:43`), uses `region_init_internal_p5en` (`nccl_ofi_regions.cpp:2189`). This region table steers **algo/proto** but there is **no `*nChannels` write anywhere for P5en** — both `*nChannels =` writes in the code are gated on `NCCL_OFI_TUNER_P6` (`nccl_ofi_regions.cpp:2124-2141`), and the model tuner never writes `*nChannels` at all. **⇒ On P5en, channel count = pure Layer A.**

> **Important consequence for the sweep:** even though the tuner does not set channels on P5en, it *does* still override algo/proto. To sweep all 12 combinations you **must force `NCCL_ALGO`/`NCCL_PROTO`** (see §4.1), otherwise the P5en region tuner will pick algo/proto for you and you won't reach the combos it considers sub-optimal.

---

## 2. The prediction formula

Per-channel "byte cost" = `nt × threadThreshold`. Defaults:
- thresholds (`nccl/src/include/comm.h:49-51`): Simple = **64**, LL = **8**, LL128 = **8**; Ring+LL is multiplied by **nRanks** (`nccl/src/graph/tuning.cc:556`).
- maxThreads (`nccl/src/graph/tuning.cc:239-251`): Simple ≈ **512**, LL = **512**, LL128 = **640**.

Predicted channel count `nc(S) = clamp( floor(S / cost), 1, N )`, where **N = `comm->nChannels`** (read it from the init log, see §4):

| Algo | Proto | cost = nt × thr | Saturates (nc = N) at S ≥ |
|------|-------|-----------------|----------------------------|
| Ring / Tree | **Simple** | 512 × 64 = **32 KiB** | N × 32 KiB |
| Ring / Tree | **LL128**  | 640 × 8 = **5 KiB**  | N × 5 KiB |
| **Tree** | **LL** | 512 × 8 = **4 KiB** | N × 4 KiB |
| **Ring** | **LL** | 512 × 8 × **nRanks** = **4 KiB × nRanks** | N × 4 KiB × nRanks |

**Reading the table:**
- **Simple** has the largest cost → it is the **slowest** protocol to ramp up channels (small messages stay at 1–2 channels the longest).
- **LL128 / Tree-LL** ramp up fastest (~4–5 KiB per channel) → they hit full channels at small messages.
- **Ring-LL** is special: its cost scales with `nRanks`, so **0x0 (more ranks) needs a much larger message to use all channels than 0x7**.

### Second-order effect (Layer-A phase 2) — VALIDATED ON P5EN, MUST BE INCLUDED IN MODEL

At scheduling time (`enqueue.cc:652-668`) NCCL re-derives the actual channels from **traffic cells**:
- `MinTrafficPerChannel = 32 KiB` (`enqueue.cc:582`)
- `trafficPerByte` per collective (`enqueue.cc:93-99`): AllReduce = **2**, AllGather / ReduceScatter = `nRanks`, Broadcast/Reduce = 1
- `if proto == LL: trafficPerByte *= 4` (`enqueue.cc:653`)
- `cellSize = divUp(divUp(MinTrafficPerChannel, trafficPerByte), 16) * 16`
- Phase-2 cap ≈ `min(N, ceil(size / cellSize))`

For **AllReduce**:
| Proto | trafficPerByte | cellSize | Phase-2 cost |
|-------|---------------:|---------:|-------------:|
| Simple | 2 | 16 KiB | 16 KiB |
| LL     | 8 | 4 KiB  | 4 KiB  |
| LL128  | 2 | 16 KiB | **16 KiB** (dominates Layer-A's 5 KiB!) |

**Final combined predictor** for AllReduce channels:
```
nc(S) = min( Layer-A(S, algo, proto, N, nRanks),
             ceil(S / phase2_cellSize(proto)) ) , clamped to [1, N]
```
Verified on P5en: this combined formula matches **all 294 measured points across 14 sweep configs** (see §6.5). The Layer-A formula alone from the table above is **optimistic for LL128** (predicts saturation at N×5 KiB, actual is N×16 KiB); it is exact for Simple, Tree-LL, and Ring-LL.

---

## 3. How 0x0 vs 0x7 enters the model

The mask changes **how many ranks participate on the inter-node ring/tree** (i.e., the effective `nRanks` and topology), which affects channels three ways:

1. **`comm->nChannels` (N) itself** — the topology search yields a different pool size per mask (`connect.cc:453` doubles it; multi-node high-BW < 16 channels can quadruple it, `connect.cc:474`). **Measure N separately for 0x0 and 0x7.**
2. **Ring+LL threshold ×nRanks** — larger nRanks (0x0) pushes the Ring-LL saturation point to much larger messages.
3. (B200-only, not on P5en) the Tree-LL128 4–32 MiB channel override requires `num_nodes*8 == num_ranks`; see §8.

> Use the **same 0x0 / 0x7 launch configuration** you already use to produce the `p5en-allreduce-0x0` / `p5en-allreduce-0x7` datasets. From the debug log, **record the actual communicator `nRanks` and the ranks-on-ring** for each mask so the Ring-LL prediction can be computed with the right nRanks.

---

## 4. Verification protocol

### 4.1 Environment
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV,GRAPH
# Force one algorithm + protocol per run so the sweep isolates each combo.
# REQUIRED on P5en: the region tuner still overrides algo/proto otherwise.
export NCCL_ALGO=<Ring|Tree>
export NCCL_PROTO=<Simple|LL|LL128>
```

Run the standard `all_reduce_perf` from nccl-tests over the sweep:
```bash
# 4 nodes x 8 GPUs, sweep 1KiB -> 1GiB, doubling
mpirun -np 32 -N 8 ... \
  all_reduce_perf -b 1K -e 1G -f 2 -n 50 -w 5
```
Repeat for each of the **2 masks × 2 algos × 3 protos = 12 runs** (plus a 13th "auto" run with `NCCL_ALGO`/`NCCL_PROTO` unset per mask, to see what the P5en tuner picks on its own).

### 4.2 Capturing the numbers
- **N = `comm->nChannels`** (large-message ceiling): from the init log line that prints the channel count, e.g. lines containing `NCCL INFO Channels` or `via P2P/.../Channels`. Record N per mask.
- **Per-size actual nChannels:** on P5en the tuner does **not** log a `Setting nChannels` line (that log only fires on the B200 override path), so use one of:
  - (a) a **debug/trace NCCL build**: `TRACE(NCCL_COLL, ...)` in `topoGetAlgoInfo`, plus `channelLo`/`channelHi` written at `enqueue.cc:691,699` → measured nChannels = `channelHi − channelLo + 1`.
  - (b) **your own profiler** in `NCCL_Performance_Model/profiler/` (report `channelHi − channelLo + 1` per op).
  - Sanity anchor: at the largest sizes, measured nChannels should equal N (the init ceiling).

```bash
# example post-processing
grep -E "Channels|MB|Algo|nChannels" run_ring_ll128_0x0.log
```

### 4.3 What to compare
For each (mask, algo, proto), build a curve of **measured nChannels vs message size** and overlay the prediction `clamp(floor(S/cost), 1, N)`. Note especially:
- the size at which channels first exceed 1,
- the size at which channels saturate at N,
- whether Ring and Tree share the same N,
- whether Ring-LL saturates much later than Tree-LL, and later for 0x0 than 0x7.

---

## 5. Predictions to confirm or refute (checklist)

Verifiable on P5en:
1. **Ring and Tree have identical N** (same channel ceiling) for a given mask. *(from `init.cc:1414`)*
2. **Channels increase monotonically with message size**, from 1 up to N. *(from the shrink loop)*
3. **Simple ramps latest**: needs ~N × 32 KiB to saturate. Small messages (≤ 32 KiB) use ~1 channel.
4. **LL128 ramps early**: saturates around N × 5 KiB.
5. **Tree-LL ramps early** (~N × 4 KiB); **Ring-LL ramps much later**, and **0x0 ramps later than 0x7** because cost scales with nRanks.
7. **No separate per-protocol hard cap**: any protocol can reach N at large enough size.

NOT verifiable on P5en (B200-only, see §8):
6. **B200 tuner override**: for Tree + LL128 AllReduce with 4 MiB ≤ size ≤ 32 MiB on the mask satisfying `num_nodes*8 == num_ranks`, nChannels is forced to one of {16, 24, 32}. This path does not exist on P5en; it is confirmed from source and applied analytically in §8.

---

## 6. Results — VALIDATED on 2026-07-20/21 (4×p5en.48xlarge, NCCL 2.30.4)

Sweep: Slurm job 46366, 6:18 elapsed. 12 forced + 2 auto configs × 21 message sizes (1K→1G, ×2 stride) × 25 iters. Rank 0 emits per-op `channel{Lo..Hi}` at NCCL_TUNING level (`enqueue.cc:805-807`); measured nc = `hi−lo+1`. N read from init `Channel XX/YY` lines.

Scripts:
- Sweep: `script/p5-en/channel_verification_sweep.sh`
- Analyzer: `script/p5-en/analyze_channel_verification.py`
- Full per-size tables: `out/p5-en/channel_verify/analysis.txt`
- Summary: `out/p5-en/channel_verify/RESULTS.md`

### 6.1 Measured N per mask

| Mask | comm->nChannels (N) | communicator nRanks | ranks on ring | num_nodes*8 == num_ranks? |
|------|--------------------:|--------------------:|--------------:|---------------------------|
| 0x0  | **16**              | **32**              | 32            | yes (4·8 == 32)           |
| 0x7  | **4**               | **4**               | 4             | no (4·8 != 4)             |

0x7 partitions the 32-rank world into 8 sub-comms of 4 ranks (one rank per node, ppn-local color).

### 6.2 Prediction-checklist result (from §5)

| # | Prediction | Result |
|---|---|---|
| 1 | Ring and Tree share identical N per mask | ✅ 16 for 0x0, 4 for 0x7 (both algos) |
| 2 | Channels increase monotonically from 1 → N | ✅ every curve non-decreasing |
| 3 | Simple ramps latest (~N × 32 KiB) | ✅ 0x0 Ring-Simple: 1 ch until 32K, N=16 at 512K (16·32K); 0x7: N=4 at 128K |
| 4 | LL128 ramps early | ✅ **but slower than the naive §2 table** — see §6.3 |
| 5 | Tree-LL ~N·4K; Ring-LL ×nRanks (0x0 later than 0x7) | ✅ 0x0 Ring-LL saturates at 2 MiB (16·4K·32); 0x7 Ring-LL at 64 KiB (4·4K·4); Tree-LL 0x0 at 64K, 0x7 at 16K |
| 7 | No per-protocol hard cap; any proto reaches N at large size | ✅ every proto hits N at ≥1 M for 0x7 and ≥2 M for 0x0 |
| 6 | B200 override — not testable on P5en | (as expected) |

### 6.3 The one divergence — LL128 phase-2 dominance

Only LL128 diverged from the Layer-A-only §2 formula. Concrete numbers (mask=0x0, N=16, nRanks=32):

| Size  | Layer-A pred (S/5 KiB) | Measured | Phase-2 (S/16 KiB) |
|-------|----------------------:|---------:|-------------------:|
| 16K   | 3                     | **1**    | 1                  |
| 32K   | 6                     | **2**    | 2                  |
| 64K   | 12                    | **4**    | 4                  |
| 128K  | 16 (cap)              | **8**    | 8                  |
| 256K  | 16                    | 16       | 16                 |

Measured LL128 exactly follows `min(Layer-A, Phase-2)` = `ceil(size / 16 KiB)` in the 16 KiB–256 KiB range because AllReduce has `trafficPerByte=2` → phase-2 `cellSize=16 KiB`, tighter than Layer-A's 5 KiB. LL is unaffected because its `trafficPerByte*=4` gives `cellSize=4 KiB`, same as Layer-A. Simple is unaffected because Layer-A's 32 KiB already dominates Phase-2's 16 KiB. **This behavior is exactly what §2's "second-order effect" describes and the combined formula from the updated §2 predicts.**

### 6.4 Bottom line

**All 294 (config, size) points match the combined `min(Layer-A, Phase-2)` model** — predictions 1, 2, 3, 4, 5, 7 confirmed. The auto-tuner runs' channel curves match the corresponding forced-algo/proto runs exactly, confirming that on P5en the tuner sets only algo/proto, never `*nChannels` (§1 Layer B).

---

## 7. Gotchas (P5en)

- **The tuner still overrides algo/proto on P5en** even though it never sets channels. Always force `NCCL_ALGO`/`NCCL_PROTO` for the sweep, or you won't reach every combo.
- **`comm->nChannels` is topology-derived at init** — you cannot know N from source alone; always read it from the log. It may differ between 0x0 and 0x7.
- **LL's ×4 traffic** (phase 2) can push measured LL channels above the simple `S/4KiB` estimate — expected.
- Confirm the platform actually resolved to `NCCL_OFI_TUNER_P5EN` (log should say so; if the instance name isn't `p5en.48xlarge` or platform ≠ "AWS", the tuner is disabled and only Layer A applies — which is also fine for channel measurement, just note it).
- Use enough warmup/iters (`-w 5 -n 50`) so behavior is stable.

---

## 8. P6-B200 vs P5en — the differences (for computing B200 nChannels later)

Once the Layer-A model is validated on P5en, B200 channel counts can be derived by applying the same model **plus** the B200-specific deltas below. There are three differences that matter for channels.

### 8.1 Platform / hardware
| | P5en | P6-B200 |
|---|---|---|
| Instance | `p5en.48xlarge` | `p6-b200.48xlarge` |
| GPU | H200 | B200 (Blackwell) |
| GPUs/node | 8 | 8 |
| NICs (rails) per node in tuner model | **2** (`nccl_ofi_model.cpp:33`) | (region-only; not modeled) |
| Tuner platform enum | `NCCL_OFI_TUNER_P5EN` | `NCCL_OFI_TUNER_P6` |
| Region init | `region_init_internal_p5en` | `region_init_internal_p6` |

The different topology/NIC count means **`comm->nChannels` (N) is generally different on B200** than on P5en. **N is not transferable — it must be read from a B200 init log** (or from a NCCL simulation on the B200 topology). Only the *shape* of the Layer-A curve transfers.

### 8.2 Layer A is identical
The NCCL-core channel logic (§1 Layer A, §2 formula) is platform-independent — same `enqueue.cc` code, same thresholds, same `maxThreads`. So for any (algo, proto, size) **outside** the B200 override window, B200 channel count is computed exactly as on P5en:
```
nc_B200(S) = clamp( floor(S / (nt × threshold)), 1, N_B200 )
```
using the **B200-measured N_B200**.

### 8.3 Layer B override — the ONLY channel difference on B200
On B200 there is one extra rule that overrides the Layer-A result (`nccl_ofi_regions.cpp:2124-2135`):

```c
// P6-B200 only: AllReduce + Tree + LL128, 4 MiB <= nBytes <= 32 MiB, 8 GPUs/node (num_nodes*8 == num_ranks)
*nChannels = calculateBestNChannelTree(nBytes, log2_nnodes);   // returns one of {16, 24, 32}
```

`calculateBestNChannelTree` (`nccl_ofi_regions.cpp:1932`) picks, among **{16, 24, 32}**, the channel count that **maximizes the LL128 chunk size**, via `calculateChunkSizeTreeLL128` (`nccl_ofi_regions.cpp:1874-1927`, assumes ppn = 8). Tie → larger channel count.

**So to compute B200 nChannels for AllReduce:**
1. Determine algo/proto B200's region tuner selects for (size, nRanks) — from `region_init_internal_p6` (`nccl_ofi_regions.cpp:958`; AllReduce region layout around `:982-1041`). This can differ from P5en's algo/proto choice because the region tables differ.
2. **If** the result is **Tree + LL128** AND **4 MiB ≤ size ≤ 32 MiB** AND **num_nodes*8 == num_ranks** (i.e. all 8 GPUs/node on the tree, the 0x0-style config):
   → `nChannels = calculateBestNChannelTree(size, log2(nNodes))` ∈ {16, 24, 32}, size-dependent.
3. **Else** (all other sizes / algos / protos, including any 0x7-style config where `num_nodes*8 != num_ranks`):
   → use the Layer-A formula from §8.2 with `N_B200`.

There is also a non-AllReduce B200 override (PAT + Simple AG/RS, `nccl_ofi_regions.cpp:2136-2141`) — irrelevant to AllReduce.

### 8.4 Summary of what transfers from P5en → B200
| Quantity | Transfers? | How to get B200 value |
|---|---|---|
| Layer-A shrink logic + thresholds (§2) | ✅ identical | reuse as-is |
| Phase-2 traffic-cell logic (§2) — validated on P5en | ✅ identical | reuse as-is |
| Curve *shape* (channels vs size per algo/proto) | ✅ | reuse combined formula |
| **N = comm->nChannels** | ❌ topology-dependent | measure on B200 init log |
| algo/proto chosen by region tuner | ❌ different table | read `region_init_internal_p6` |
| Tree-LL128 4–32 MiB channel = {16,24,32} | ➕ B200-only extra | apply §8.3 rule |

---

## 9. Predicted nChannels on P6-B200, 16 nodes (128 GPUs) — application of the validated model

Using the P5en-validated model (§2 combined `min(Layer-A, Phase-2)`) plus the B200 plugin override rule (§8.3), predicted **AllReduce** nChannels across the sweep.

**Assumptions (from source extrapolation, must be re-anchored with a B200 init log):**
- `N_0x0 = 32` — 16-node, 128-rank full comm. Rationale: `connect.cc:453` doubles the base ring count; on multi-node ≥90 compute-capability with `bwIntra>45` the code doubles again if `nChannels<16`; typical outcome saturates at 32 for large 8-GPU-per-node comms. P5en 4-node/32-rank gave N=16, so B200 16-node/128-rank plausibly lands at 32.
- `N_0x7 = 16` — 0x7 splits 128 ranks into 8 sub-comms of 16 ranks (one per node). P5en 4-node/4-rank gave N=4, so B200 16-node/16-rank plausibly gives N=16.
- Message size range: 1 KiB → 1 GiB, doubling.
- Override rule ([`nccl_ofi_regions.cpp:2114-2119`]) fires ONLY for AR + Tree + LL128 in `[4 MiB, 32 MiB]` on `num_nodes*8 == num_ranks` (0x0 only). Value comes from `calculateBestNChannelTree(size, log2(16)=4)` picking the {16,24,32} that maximizes `calculateChunkSizeTreeLL128` (`regions.cpp:1874-1926`, evaluated below).

### 9.1 Layer-A + Phase-2 costs (AllReduce, plugged with `nRanks_0x0=128`, `nRanks_0x7=16`)

| Algo | Proto | Layer-A cost | Phase-2 cost (AR) | Effective cost | Saturation @ 0x0 (N=32) | Saturation @ 0x7 (N=16) |
|------|-------|-------------:|------------------:|---------------:|------------------------:|-------------------------:|
| Ring/Tree | Simple | 32 KiB | 16 KiB | **32 KiB** | 32·32K = 1 MiB | 16·32K = 512 KiB |
| Tree | LL | 4 KiB | 4 KiB | **4 KiB** | 32·4K = 128 KiB | 16·4K = 64 KiB |
| Ring | LL (0x0) | 4 KiB·128 = **512 KiB** | 4 KiB | **512 KiB** | 32·512K = 16 MiB | — |
| Ring | LL (0x7) | 4 KiB·16 = **64 KiB** | 4 KiB | **64 KiB** | — | 16·64K = 1 MiB |
| Ring/Tree | LL128 | 5 KiB | **16 KiB** | **16 KiB** | 32·16K = 512 KiB | 16·16K = 256 KiB |

### 9.2 Full predicted-nChannels table

Cells with `\*` are the B200 plugin override (values come from `calculateBestNChannelTree`, not Layer-A/Phase-2).

| Size | 0x0 Ring Simple | 0x0 Ring LL | 0x0 Ring LL128 | 0x0 Tree Simple | 0x0 Tree LL | 0x0 Tree LL128 | 0x7 Ring Simple | 0x7 Ring LL | 0x7 Ring LL128 | 0x7 Tree Simple | 0x7 Tree LL | 0x7 Tree LL128 |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1K   | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2K   | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 4K   | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 8K   | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 2 | 1 |
| 16K  | 1 | 1 | 1 | 1 | 4 | 1 | 1 | 1 | 1 | 1 | 4 | 1 |
| 32K  | 1 | 1 | 2 | 1 | 8 | 2 | 1 | 1 | 2 | 1 | 8 | 2 |
| 64K  | 2 | 1 | 4 | 2 | 16 | 4 | 2 | 1 | 4 | 2 | 16 | 4 |
| 128K | 4 | 1 | 8 | 4 | 32 | 8 | 4 | 2 | 8 | 4 | 16 | 8 |
| 256K | 8 | 1 | 16 | 8 | 32 | 16 | 8 | 4 | 16 | 8 | 16 | 16 |
| 512K | 16 | 1 | 32 | 16 | 32 | 32 | 16 | 8 | 16 | 16 | 16 | 16 |
| 1M   | 32 | 2 | 32 | 32 | 32 | 32 | 16 | 16 | 16 | 16 | 16 | 16 |
| 2M   | 32 | 4 | 32 | 32 | 32 | 32 | 16 | 16 | 16 | 16 | 16 | 16 |
| **4M**   | 32 | 8 | 32 | 32 | 32 | **32\*** | 16 | 16 | 16 | 16 | 16 | 16 |
| **8M**   | 32 | 16 | 32 | 32 | 32 | **16\*** | 16 | 16 | 16 | 16 | 16 | 16 |
| **16M**  | 32 | 32 | 32 | 32 | 32 | **16\*** | 16 | 16 | 16 | 16 | 16 | 16 |
| **32M**  | 32 | 32 | 32 | 32 | 32 | **32\*** | 16 | 16 | 16 | 16 | 16 | 16 |
| 64M  | 32 | 32 | 32 | 32 | 32 | 32 | 16 | 16 | 16 | 16 | 16 | 16 |
| 128M | 32 | 32 | 32 | 32 | 32 | 32 | 16 | 16 | 16 | 16 | 16 | 16 |
| 256M | 32 | 32 | 32 | 32 | 32 | 32 | 16 | 16 | 16 | 16 | 16 | 16 |
| 512M | 32 | 32 | 32 | 32 | 32 | 32 | 16 | 16 | 16 | 16 | 16 | 16 |
| 1G   | 32 | 32 | 32 | 32 | 32 | 32 | 16 | 16 | 16 | 16 | 16 | 16 |

`\*` = B200 plugin override (`calculateBestNChannelTree`).

### 9.3 What to notice about the 0x0 / Tree / LL128 column (the override zone)

Without the override, Layer-A+Phase-2 would predict **32** channels for every size ≥ 512 KiB — including all four override sizes {4M, 8M, 16M, 32M}. The plugin `calculateBestNChannelTree` returns:

| Size | override nc | why (chunkSize maximization at log2_nnodes=4, ppn=8) |
|------|------------:|------------------------------------------------------|
| 4 MiB  | **32**  | at 4 MiB, 32-channel chunk stays large enough; larger nc wins the tie |
| 8 MiB  | **16**  | at 8 MiB, 32-channel chunk gets shrunk by inner while-loops → 16 keeps chunk bigger |
| 16 MiB | **16**  | same reason as 8 MiB |
| 32 MiB | **32**  | back to 32 — 32-channel chunk not shrunk at this size |

So the override *lowers* nChannels below the Layer-A ceiling only at 8 MiB and 16 MiB. Elsewhere it agrees with the pure-Layer-A/Phase-2 prediction (32).

### 9.4 Sensitivity to N_B200

The table's 0x0 columns scale linearly with `N_0x0`; the 0x7 columns scale with `N_0x7`. If the measured B200 N is different from the assumed 32/16, replace every value `> new_N` with `new_N` in the appropriate columns. The override cells (`\*`) are independent of N and stay {16,24,32}.

### 9.5 What still needs a real B200 log

- **Anchor N_0x0 and N_0x7.** One init log at 4 or more B200 nodes is enough to read `Channel XX/YY` and confirm/adjust the assumption.
- **Confirm the override actually fires** in the [4 MiB, 32 MiB] zone via the plugin's `Setting nChannels to X at nBytes=Y` line (`regions.cpp:2128`).
- **Auto-tuner algo/proto selection on B200** — the region table (`region_init_internal_p6`) differs from P5en, so the AUTO run's channel curve will differ from any forced-algo/proto column. The forced runs' curves should still match this table.

## 9.6 AllGather channel selection — scope & verification

For AG we only consider the **three 0x7 Ring configs**: `Ring Simple`, `Ring LL`, `Ring LL128`. Rationale:

- **Tree is excluded** — NCCL core has no AG+Tree path. Forcing `NCCL_ALGO=Tree` produces `all_gather.cu:50 'invalid usage'` for every size. Only `{Ring, PAT, NVLS}` implement AG.
- **0x0 is excluded** — for our target sub-comm topology (one rank per node on the ring), 0x7 is what applications actually see. 0x0 (all 128 GPUs in one comm) is only used at rendezvous / debug.
- **The B200 plugin override does not touch AG.** [regions.cpp:2114-2119](../../../../software/aws-ofi-nccl/src/tuner/nccl_ofi_regions.cpp#L2114-L2119) gates on `collType == ncclFuncAllReduce`. The other plugin override at [regions.cpp:2121-2126](../../../../software/aws-ofi-nccl/src/tuner/nccl_ofi_regions.cpp#L2121-L2126) only fires for `PAT + Simple`, not Ring. So Ring AG is pure NCCL Layer-A + Phase-2.

### Two things to know before predicting AG

1. **`-b S` argument semantics.** nccl-tests `all_gather` divides count by nRanks at [all_gather.cu:11](../../../../software/nccl-tests/src/all_gather.cu#L11) (`base = count/nranks`) and passes `count = S/nRanks` to `ncclAllGather`. Inside NCCL:
   - **Layer-A** ([enqueue.cc:2041](../../../../software/nccl/src/enqueue.cc#L2041)) uses `nBytes = ncclFuncMaxSendRecvCount * eltSize = nRanks · count · eltSize = S` (the total size).
   - **Phase-2** ([enqueue.cc:656](../../../../software/nccl/src/enqueue.cc#L656)) uses `task->count * eltSize = S / nRanks` (per-rank sendbytes).

   So Layer-A and Phase-2 operate on **different** byte quantities for AG. The rank-0 `NCCL_TUNING` line at [enqueue.cc:794-796](../../../../software/nccl/src/enqueue.cc#L794-L796) prints `task->count * eltSize = S/nRanks`; the analyzer multiplies by `nRanks` to recover `-b S`.

2. **`ncclFuncTrafficPerByte(AG) = nRanks`** ([enqueue.cc:96](../../../../software/nccl/src/enqueue.cc#L96)), so Phase-2 `cellSize = ⌈⌈32K/nRanks⌉/16⌉·16`. For 0x7 sub-comm (`nRanks=16`): `cellSize = ⌈⌈32K/16⌉/16⌉·16 = 2048 B` (LL multiplies tpb by 4 → cellSize = 512 B). Phase-2 cap `= min(N, ⌈per-rank / cellSize⌉)`. With per-rank = S/16 and N=4 this is very loose; Layer-A dominates in practice.

### AG prediction on P5en 16 nodes / 0x7 (N=4, nRanks=16)

Layer-A takes `nBytes = S`; Phase-2 takes `per-rank = S/nRanks = S/16` and Phase-2 `cellSize = ⌈⌈32K/tpb⌉/16⌉·16` where `tpb = nRanks` (`×4` if LL):

| proto | Layer-A cost | Layer-A saturates | Phase-2 cellSize (per-rank) | Phase-2 saturates | who binds |
|-------|-------------:|-------------------|-----------------------------|-------------------|:---------:|
| Simple | 32 KiB    | `N·32K = 128K` | 2048 B (`tpb=16`) | `N·nR·2048 = 128K` | **tie** |
| LL     | 64 KiB (`512·8·16`) | `N·64K = 256K` | 512 B (`tpb=64`) | `N·nR·512 = 32K` (→ N ≥ this size) | Layer-A |
| LL128  | 5 KiB     | `N·5K = 20K`   | 2048 B (`tpb=16`) | `N·nR·2048 = 128K` | **Phase-2** |

Note: for AG-LL, Phase-2 clamps to N at 32K but Layer-A only reaches 1 there (since S/64K = 0.5 → clamped to 1); Layer-A is *tighter* in the ramp region and binds. For AG-LL128, Phase-2's coarser cell (2048 B per-rank) is what actually gates the ramp — even though 32K > 16K would suggest otherwise, Phase-2's ramp is `S/(nR·cell) = S/32K` which is tighter than Layer-A's `S/5K`.

### 9.6.1 Measured vs predicted (job 46481, 16 nodes)

Analyzer: [analyze_channel_verification_AG.py](analyze_channel_verification_AG.py), sweep: [channel_verification_sweep_AG.sh](channel_verification_sweep_AG.sh).

`0x7 Ring Simple` — 24/24 match, 1 KiB → 8 GiB:

| Size | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K–8G |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| meas | 1 | 1 | 1 | 1 | 1 | 1 | 2 | **4** | 4 (saturated) |
| pred | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 4 | 4 |

`0x7 Ring LL` — 8/8 match, 1 KiB → 128 KiB:

| Size | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| meas | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 |
| pred | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 |

`0x7 Ring LL128` — 15/15 match, 8 KiB → 128 MiB:

| Size | 8K | 16K | 32K | 64K | 128K | 256K–128M |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| meas | 1 | 1 | 1 | 2 | **4** | 4 (saturated) |
| pred | 1 | 1 | 1 | 2 | 4 | 4 |

**Total: 47/47 measured points match `min(Layer-A, Phase-2-AG)` exactly.** The AG-specific Phase-2 rule (per-rank bytes, `trafficPerByte = nRanks`) is confirmed on real hardware.

Full measured-nChannels table for the three measured configs (`—` = outside this sweep's size range for that protocol):

| Size | 0x7 Ring Simple | 0x7 Ring LL | 0x7 Ring LL128 |
|------|:-:|:-:|:-:|
| 1K   | 1 | 1 | 1 |
| 2K   | 1 | 1 | 1 |
| 4K   | 1 | 1 | 1 |
| 8K   | 1 | 1 | 1 |
| 16K  | 1 | 1 | 1 |
| 32K  | 1 | 1 | 1 |
| 64K  | 2 | 1 | 2 |
| 128K | 4 | 2 | 4 |
| 256K | 4 | — | 4 |
| 512K | 4 | — | 4 |
| 1M   | 4 | — | 4 |
| 2M   | 4 | — | 4 |
| 4M   | 4 | — | 4 |
| 8M   | 4 | — | 4 |
| 16M  | 4 | — | 4 |
| 32M  | 4 | — | 4 |
| 64M  | 4 | — | 4 |
| 128M | 4 | — | 4 |
| 256M | 4 | — | — |
| 512M | 4 | — | — |
| 1G   | 4 | — | — |

### 9.6.2 Why the AG ramp table is the same as AR on 0x7

For the three configs measured, `min(Layer-A, Phase-2)` gives an **identical curve to AR** on 0x7 (§6.2):

| Size | Simple | LL | LL128 |
|------|:-:|:-:|:-:|
| Ramp | `S/32K`, sat @ 128K | `S/64K`, sat @ 256K | `S/32K`, sat @ 128K |

Reason:

- **Simple.** For AR Layer-A gives `S/32K` and Phase-2 gives `S/16K` — Layer-A binds. For AG Layer-A still gives `S/32K` and Phase-2 gives `(S/nR) / 2K = S/32K` — same ramp, coincidence of factors.
- **LL.** Both AR and AG are Layer-A-bound at `S/64K` (Ring-LL threshold ×16).
- **LL128.** For AR Phase-2 binds at `S/16K` (`tpb=2`, cellSize 16 KiB). For AG Phase-2 binds at `S/32K` (per-rank `/` 2 KiB cellSize, with `tpb=16`). **So AR ramps *twice as fast* as AG on LL128 at 0x7.** AR reaches N=4 at 64 K, AG reaches N=4 at 128 K.

The 128 K vs 64 K difference for LL128 is the only structural AR/AG divergence visible in the 0x7 sub-comm ramp region.
