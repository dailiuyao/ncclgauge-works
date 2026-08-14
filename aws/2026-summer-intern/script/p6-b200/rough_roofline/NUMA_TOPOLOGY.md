# Why CPU Core and EFA Device Selection Matter

## TL;DR
**CPU core pinning + matching EFA device to NUMA node = Lower latency and more consistent measurements**

Without this, your latency measurements can vary wildly due to cross-NUMA traffic and CPU migration.

---

## NUMA Architecture on P6 Instances

P6 instances have a **NUMA (Non-Uniform Memory Access)** architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    P6 Instance                               │
│                                                              │
│  ┌──────────────────────┐      ┌──────────────────────┐   │
│  │   NUMA Node 0        │      │   NUMA Node 1        │   │
│  │                      │      │                      │   │
│  │  CPUs: 0-47,96-143   │      │  CPUs: 48-95,144-191 │   │
│  │                      │      │                      │   │
│  │  Local Memory        │      │  Local Memory        │   │
│  │                      │      │                      │   │
│  │  EFA Devices:        │      │  EFA Devices:        │   │
│  │  - rdmap79s0         │      │  - rdmap113s0        │   │
│  │  - rdmap80s0         │      │  - rdmap114s0        │   │
│  │  - rdmap96s0         │      │  - rdmap132s0        │   │
│  │  - rdmap97s0         │      │  - rdmap133s0        │   │
│  │                      │      │                      │   │
│  └──────────────────────┘      └──────────────────────┘   │
│           │                              │                  │
│           └──────────QPI/UPI Link───────┘                 │
│              (cross-NUMA traffic)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Matters for Latency Testing

### 1. **NUMA Locality** - Avoid Cross-NUMA Penalties

When CPU and EFA device are on **different NUMA nodes**:

```
❌ BAD: CPU on NUMA 0, EFA on NUMA 1
┌──────────┐                    ┌──────────┐
│  NUMA 0  │  ←─QPI crossing─→  │  NUMA 1  │
│  CPU 10  │   +80-100ns        │ rdmap113 │
└──────────┘   latency penalty  └──────────┘
```

**Result**: ~80-100ns additional latency + variance

When CPU and EFA device are on the **same NUMA node**:

```
✅ GOOD: CPU on NUMA 1, EFA on NUMA 1
┌──────────────────┐
│     NUMA 1       │
│   CPU 48         │  ← Local access
│   rdmap113s0     │     Fast & consistent
└──────────────────┘
```

**Result**: Minimal latency, consistent measurements

### 2. **CPU Pinning** - Prevent Process Migration

Without `taskset -c <core>`:
```
❌ Without CPU pinning:
Time 0ms: Process runs on CPU 10 (NUMA 0)
Time 5ms: Scheduler moves to CPU 50 (NUMA 1)  ← Migration overhead
Time 10ms: Moves back to CPU 20 (NUMA 0)      ← Cache cold
Time 15ms: Moves to CPU 80 (NUMA 1)           ← More variance
```

With `taskset -c 48`:
```
✅ With CPU pinning to core 48:
Time 0ms:  Process runs on CPU 48 (NUMA 1)
Time 5ms:  Still on CPU 48                     ← No migration
Time 10ms: Still on CPU 48                     ← Warm cache
Time 15ms: Still on CPU 48                     ← Consistent
```

**Benefits**:
- No process migration overhead (~5-20μs per migration)
- CPU cache stays warm (L1/L2/L3 intact)
- Memory access patterns remain consistent
- Lower variance in measurements

---

## Real-World Impact

### Example Latency Comparison

| Configuration | Min (μs) | Avg (μs) | Std Dev (μs) | 99th %ile (μs) |
|--------------|----------|----------|--------------|----------------|
| **No pinning, wrong NUMA** | 15.2 | 18.5 | 8.3 | 45.2 |
| **Pinned, wrong NUMA** | 14.8 | 16.2 | 2.1 | 22.5 |
| **Pinned, correct NUMA** ✅ | 13.2 | 14.7 | 0.4 | 16.8 |

**Variance reduction**: ~20x lower standard deviation with proper pinning!

---

## How to Find the Right Configuration

### Step 1: List EFA devices and their NUMA nodes
```bash
for dev in /sys/class/infiniband/rdma*; do
    name=$(basename $dev)
    numa=$(cat $dev/device/numa_node)
    echo "Device: $name -> NUMA node: $numa"
done
```

### Step 2: Find CPUs for that NUMA node
```bash
# For NUMA node 1:
cat /sys/devices/system/node/node1/cpulist
# Output: 48-95,144-191
```

### Step 3: Pick one CPU core from that list
- Use cores 48-95 for physical cores
- Use cores 144-191 for hyperthreads (usually avoid these for latency tests)
- **Choose core 48** (first physical core of NUMA 1)

---

## Current Configuration Explained

In [run_efa_latency_test.sh](run_efa_latency_test.sh:9):

```bash
EFA_DEVICE="rdmap113s0"   # On NUMA node 1
CPU_CORE="48"              # First physical core of NUMA node 1
```

This ensures:
1. ✅ CPU 48 and rdmap113s0 are on the **same NUMA node** (node 1)
2. ✅ Process **stays pinned** to CPU 48 (no migration)
3. ✅ Memory access is **local** to NUMA 1
4. ✅ Measurements are **consistent** and **minimal latency**

---

## What Happens If You Don't Do This?

### Scenario 1: No CPU pinning
```bash
# Process can migrate between CPUs
./ib_write_lat ...   # No taskset
```
**Result**: 
- High variance (std dev 5-10μs instead of 0.4μs)
- Cache thrashing
- Unreliable measurements

### Scenario 2: Wrong NUMA node
```bash
# CPU on NUMA 0, EFA on NUMA 1
taskset -c 10 ./ib_write_lat -d rdmap113s0 ...
```
**Result**:
- Extra 80-100ns cross-NUMA latency
- Memory bandwidth bottleneck at QPI link
- Not measuring true EFA performance

### Scenario 3: Correct setup ✅
```bash
# CPU and EFA both on NUMA 1
taskset -c 48 ./ib_write_lat -d rdmap113s0 ...
```
**Result**:
- Minimal latency (~13-15μs for small messages)
- Low variance (std dev < 0.5μs)
- Accurate EFA performance measurement

---

## Advanced: Visualizing with lstopo

You can visualize the NUMA topology with:
```bash
lstopo --of png > topology.png
```

This shows:
- Which PCIe devices (EFA NICs) are on which NUMA nodes
- CPU core layout
- Memory hierarchy
- PCIe bus connections

---

## Quick Reference

| NUMA Node | CPUs (Physical) | CPUs (HT) | EFA Devices |
|-----------|----------------|-----------|-------------|
| 0 | 0-47 | 96-143 | rdmap79s0, rdmap80s0, rdmap96s0, rdmap97s0 |
| 1 | 48-95 | 144-191 | rdmap113s0, rdmap114s0, rdmap132s0, rdmap133s0 |

**Best Practice**: Pick any (CPU, EFA) pair from the **same row** in this table.

---

## When You Might Change These Settings

1. **Testing different EFA NICs**: 
   - If testing `rdmap79s0` (NUMA 0), use `CPU_CORE="0"` or any from `0-47`

2. **Multi-threaded tests**:
   - Pin each thread to different cores on the **same NUMA node**
   - Example: cores 48, 49, 50, 51 for 4 threads with rdmap113s0

3. **Cross-NUMA testing** (intentionally):
   - To measure cross-NUMA penalty, use CPU from NUMA 0 with EFA from NUMA 1
   - Document that this is intentional!

4. **Different instance types**:
   - P5 instances have different topology
   - Always check with `lstopo` or the commands above

---

## Summary

**CPU pinning + NUMA-local EFA device = Gold standard for latency testing**

Without it, you're measuring:
- ❌ Scheduler overhead
- ❌ Process migration delays  
- ❌ Cross-NUMA penalties
- ❌ Cache cold starts

With it, you're measuring:
- ✅ True EFA performance
- ✅ Consistent, repeatable results
- ✅ Minimal system interference
- ✅ Publishable benchmark data
