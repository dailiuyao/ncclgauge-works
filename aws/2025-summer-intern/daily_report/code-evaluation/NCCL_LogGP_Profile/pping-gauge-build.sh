#!/bin/bash

set -e

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda

# Set MPI environment
export MPI_HOME=/opt/amazon/openmpi
export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

export NCCL_SRC=/home/liuyaod/software/nccl_p2p_profile

# Set NCCL environment
export NCCL_HOME=$NCCL_SRC/build

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
export NCCL_SRC_LOCATION=$NCCL_SRC

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

export NCCL_GAUGE_HOME="/home/liuyaod/software/gauge-test/ncclguage"

export GET_CYCLES_HOME="${NCCL_GAUGE_HOME}/gauge"

n_messages=(1 10)

# for i in "${n_messages[@]}"; do
#     for mode in pping pping_nchannels pping_msgs; do
#         # Use proper variable expansion and quoting in the command
#         nvcc "$NVCC_GENCODE" -ccbin g++ \
#             -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" -I"${GET_CYCLES_HOME}" \
#             -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
#             -D GAUGE_N_MESSAGES=${i} \
#             "${GET_CYCLES_HOME}/get_clock.cu" "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" \
#             -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}.exe"
#         # Verification of the output
#         if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}.exe" ]; then
#             echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}.exe"
#         else
#             echo "Compilation failed."
#         fi
#     done
# done

n_messages=(1 10)

for i in "${n_messages[@]}"; do
    for mode in pping_3ranks; do
        for k in 1 2; do
            # Use proper variable expansion and quoting in the command
            nvcc "$NVCC_GENCODE" -ccbin g++ \
                -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" -I"${GET_CYCLES_HOME}" \
                -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
                -D GAUGE_N_MESSAGES=${i} \
                -D P2P_GROUP_SIZE_LYD=${k} \
                "${GET_CYCLES_HOME}/get_clock.cu" "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" \
                -o "${NCCL_GAUGE_HOME}/gauge/${mode}_groupsz_${k}_gauge_n_${i}.exe"
            # Verification of the output
            if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_groupsz_${k}_gauge_n_${i}.exe" ]; then
                echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_groupsz_${k}_gauge_n_${i}.exe"
            else
                echo "Compilation failed."
            fi
        done
    done
done