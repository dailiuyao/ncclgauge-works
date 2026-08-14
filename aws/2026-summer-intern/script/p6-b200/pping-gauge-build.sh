#!/bin/bash
set -e

export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/opt/amazon/openmpi
export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

export NCCL_SRC=/home/liuyaod/software/nccl_profile
export NCCL_HOME=$NCCL_SRC/build
# P6 B200 uses Blackwell architecture (compute capability 10.0)
export NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"

export PATH=$CUDA_HOME/bin:$MPI_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$MPI_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export NCCL_GAUGE_HOME="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern"
export GET_CYCLES_HOME="${NCCL_GAUGE_HOME}/gauge"
export PROFILING_LIB_DIR="${NCCL_GAUGE_HOME}/gauge"
export GAUGE_OUT_DIR="${NCCL_GAUGE_HOME}/gauge/p6-b200"

# Create output directory for p6-b200 binaries
mkdir -p $GAUGE_OUT_DIR

n_messages=(1)

for i in "${n_messages[@]}"; do
    for mode in pping; do
        nvcc $NVCC_GENCODE -ccbin g++ \
            -I"${NCCL_HOME}/include" -I"${MPI_HOME}/include" -I"${GET_CYCLES_HOME}" \
            -L"${NCCL_HOME}/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -L"${PROFILING_LIB_DIR}" \
            -lnccl -lcudart -lmpi -lprofiling_arrays \
            -Xlinker --export-dynamic -Xlinker --allow-shlib-undefined \
            -Xlinker -rpath="${PROFILING_LIB_DIR}" \
            -D GAUGE_N_MESSAGES=${i} \
            "${GET_CYCLES_HOME}/get_clock.cu" "${GET_CYCLES_HOME}/${mode}_gauge.cu" \
            -o "${GAUGE_OUT_DIR}/${mode}_gauge_n_${i}.exe"
        if [ -f "${GAUGE_OUT_DIR}/${mode}_gauge_n_${i}.exe" ]; then
            echo "Compilation successful: ${GAUGE_OUT_DIR}/${mode}_gauge_n_${i}.exe"
        else
            echo "Compilation failed: ${mode}_gauge_n_${i}.exe"
            exit 1
        fi
    done
done

echo "=== Gauge build complete for P6 B200 ==="
echo "Binaries location: $GAUGE_OUT_DIR"
