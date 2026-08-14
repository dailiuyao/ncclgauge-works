# Makefile for GIN P2P Tests on AWS P6 B200

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /home/liuyaod/software/nccl
MPI_HOME ?= /opt/amazon/openmpi5

NVCC := $(CUDA_HOME)/bin/nvcc
MPICC := $(MPI_HOME)/bin/mpicc

CUDA_INC := -I$(CUDA_HOME)/include
CUDA_LIB := -L$(CUDA_HOME)/lib64 -lcudart

NCCL_INC := -I$(NCCL_HOME)/build/include
NCCL_LIB := -L$(NCCL_HOME)/build/lib -lnccl

MPI_INC := -I$(MPI_HOME)/include
MPI_LIB := -L$(MPI_HOME)/lib -lmpi

CXXFLAGS := -O3 -std=c++11
NVCCFLAGS := -O3 -std=c++11 -arch=sm_90

# Working tests
TARGETS := gin_p2p_symmetric

all: $(TARGETS)

gin_p2p_symmetric: gin_p2p_symmetric.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) $(NCCL_INC) $(MPI_INC) \
		$< -o $@ \
		$(CUDA_LIB) $(NCCL_LIB) $(MPI_LIB)

clean:
	rm -f $(TARGETS)

test-symmetric: gin_p2p_symmetric
	@echo "Running GIN P2P symmetric memory test..."
	mpirun -np 2 --allow-run-as-root ./gin_p2p_symmetric

.PHONY: all clean test-symmetric
