# NCCL Performance Modeling and Optimization

This repository contains the codebase for Liuyao Dai's 2025 Summer internship project. The project focuses on NCCL performance profiling and modeling.

## Repository Structure

The repository is organized into four main components:

### 1. NCCL Profiling Code
**Location:** `NCCL_LogGP_Profile/`

This directory contains micro-benchmarks for NCCL collective operations with lightweight profiling instrumentation. The benchmarks support the following operations (using 0x0 mask):
- AllReduce Tree
- AllReduce NVLS Tree
- AllReduce Ring
- AllGather Ring
- ReduceScatter Ring

### 2. Profiling Experiment Results & Model Building
**Location:** `AllReduce-Tree/`, `AllReduce-NVLSTree/`, `AllReduce-Ring/`, `AllGather-Ring/`, `ReduceScatter-Ring/`

Each directory contains:
- Profiling experiment results from 4 p5en nodes
- Jupyter notebooks for:
  - Extracting timestamps from profiling data
  - Calculating model parameter values
  - Building performance models
  - Comparing model predictions with nccl-test measurements on 16 nodes
- Results for different protocols: LL, LL128, and Simple

### 3. Model-Based Tuner
**Location:** `Model-Based-Tuner/`

Contains a patch file (`model-based-tuner.patch`) that implements the model-based tuning functionality.

**Usage:**
```bash
# Apply the patch to aws-ofi-nccl before building
git apply model-based-tuner.patch
```

### 4. Nightly Data & Model Prediction Performance Comparison
**Location:** `Nightly-Data-Comparison/`

This directory includes:
- `model.py`: Builds performance models using data from the profiling experiments
- `plotter.py`: Visualization tools for performance comparison
- `standard-8.ipynb`: Jupyter notebook comparing nightly performance data with model predictions