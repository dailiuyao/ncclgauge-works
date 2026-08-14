# MicroBenchmark: NCCL Profiling

This repository provides some micro-benchmarks for NCCL with some lightweight profiling code. Follow the steps below to set up, build, and run the benchmark.

---

## Cluster

P5en

---

## 1. Build NCCL with Profiling Patch

1. In a clean clone of the NCCL repo, check out the `v2.27.3-1` tag. 
   ```bash
   git clone git@github.com:NVIDIA/nccl.git
   cd nccl
   git checkout v2.27.3-1
   ``` 
2. Copy the `NCCL_LogGP_Profile.patch` file into the NCCL repository, then apply the patch:
   ```bash
   git apply NCCL_LogGP_Profile.patch
   ```
3. Build NCCL

---

## 2. Build the plugin

## 3. Build the Micro-Benchmarks

Edit `gauge-build.sh` to set `NCCL_SRC`, `NCCL_GAUGE_HOME`. Then run:
```bash
bash gauge-build.sh
```
This compiles:
- `allreduce_tree.cu`, `allreduce_ring.cu`, `allgather_ring.cu`, `reducescatter_ring.cu`

---

## 4. Run the Benchmark


Edit `allgather-gauge-ring-run.sh`, `allreduce-gauge-NVLSTree-run.sh`, `allreduce-gauge-ring-run.sh`, `allreduce-gauge-tree-run.sh`, `reducescatter-gauge-ring-run.sh` to set `NCCL_GAUGE_HOME`, `nccl`, `ofi_plugin`.

Run the benchmarks:
```bash
sbatch xxx-run.sh
```

profiling files are in your `GAUGE_OUT_DIRE`.

---

## 5. Logging & Output

- Each MPI rank writes its results to  
  `nccl_{collective}_heo-inter_r-<rank>.out`  
- Files contain timestamp and per-chunk latency entries.

---

## 6. NCCL-Tests Execution Guide

### Test Configuration
- Number of nodes: 16
- Environment variable: `NCCL_TESTS_SPLIT_MASK=0x0`

### Test Matrix
Run NCCL-Tests for the following combinations:
1. Collectives:
   - AllReduce
   - AllGather
   - ReduceScatter

2. Algorithms:
   - Ring
   - Tree
   - NVLSTree

3. Protocols:
   - LL
   - LL128
   - Simple

### Output Organization
Store test results using the following naming convention:
```
nccl_test-<collective>-<algorithm>.out
```

Example:
- `nccl_test-reducescatter-ring.out`
- `nccl_test-allgather-tree.out`
- `nccl_test-allreduce-NVLSTree.out`

### File Structure
Copy results to respective directories:
```
AllGather-Ring/AllReduce-NVLSTree/AllReduce-Ring/AllReduce-Tree/ReduceScatter-Ring
└── protocol_profiling_results/
      ├── LL/
      ├── LL128/
      └── simple/

```

## 7. Results Verification

### Prerequisites
- Completed profiling data collection
- NCCL-test results stored in appropriate directories
- Verification Jupyter notebook available

### Steps
1. Ensure all profiling and NCCL-test results are properly organized in their respective directories:
   - Profiling results
   - NCCL-test outputs

2. Launch and Execute Verification Notebook
   ```bash
   jupyter notebook verify.ipynb
   ```