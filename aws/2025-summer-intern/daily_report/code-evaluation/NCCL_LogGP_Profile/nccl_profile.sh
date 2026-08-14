#!/bin/bash

set -e

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda

# Set MPI environment
export MPI=1
export MPI_HOME=/opt/amazon/openmpi
export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

export NCCL_SRC=/home/liuyaod/software/nccl_p2p_profile
export AWS_OFI_NCCL_SRC=/home/liuyaod/software/aws-ofi-nccl
export NCCL_TEST_SRC=/home/liuyaod/software/nccl-tests

###### Build NCCL ######
pushd $NCCL_SRC

make clean
rm -rf build

make -j$(nproc) src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

popd

# export NCCL_SRC=/home/liuyaod/software/nccl

# # Set NCCL environment
# export NCCL_HOME=$NCCL_SRC/build

# export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

# # Install NCCL Tests
# pushd $NCCL_TEST_SRC
# make clean
# export  LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH
# make MPI=1 MPI_HOME=/opt/amazon/openmpi/ NCCL_HOME=$NCCL_HOME CUDA_HOME=/usr/local/cuda
# popd

# ###### Build AWS_OFI_NCCL ######

# pushd ${AWS_OFI_NCCL_SRC}

# # make distclean

# ./autogen.sh
# ./configure --prefix=${AWS_OFI_NCCL_SRC}/install --with-mpi=/opt/amazon/openmpi/ \
#             --with-libfabric=/opt/amazon/efa \
#             --with-nccl=$NCCL_SRC/build \
#             --with-cuda=/usr/local/cuda
# make && make install

# popd

