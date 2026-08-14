#!/bin/bash

set -e

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda

# Set MPI environment
export MPI_HOME=/opt/amazon/openmpi
export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

export NCCL_SRC=/home/liuyaod/software/nccl_profile

# Set NCCL environment
export NCCL_HOME=$NCCL_SRC/build

# Set environment variables

# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export PATH=$CUDA_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include

# NCCL source location
NCCL_SRC_LOCATION=$NCCL_SRC

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

export NCCL_GAUGE_HOME="/home/liuyaod/software/gauge-test/ncclguage"

nmessages=(10000)

for i in "${nmessages[@]}"; do
    for mode in allreduce_tree allreduce_ring allgather_ring reducescatter_ring; do
        # Use proper variable expansion and quoting in the command
        nvcc "$NVCC_GENCODE" -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
            -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
            -D N_ITERS=${i} \
            "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}.exe"

        # Verification of the output
        if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}.exe" ]; then
            echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}.exe"
        else
            echo "Compilation failed."
        fi
    done
done

# salloc --nodes=1 --partition=p5en-odcr-queue --time=2:00:00 --exclusive

