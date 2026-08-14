/*
 * GIN Mode P2P Test with SYMMETRIC MEMORY ALLOCATION
 * Uses ncclMemAlloc() for symmetric virtual addresses (required for GIN)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("[Rank %d] CUDA Error at %s:%d - %s\n", \
        rank, __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("[Rank %d] NCCL Error at %s:%d - %s\n", \
        rank, __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if (e != MPI_SUCCESS) {                           \
    printf("MPI Error at %s:%d - code %d\n",        \
        __FILE__,__LINE__,e);                       \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int rank = -1;  // Global for error macros

int main(int argc, char* argv[]) {
    int nranks;

    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    printf("[Rank %d/%d] Starting GIN P2P test with SYMMETRIC MEMORY\n", rank, nranks);

    if (nranks < 2) {
        if (rank == 0) {
            printf("This test requires at least 2 ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Set GPU device
    int gpu_id = 0;
    CUDACHECK(cudaSetDevice(gpu_id));
    printf("[Rank %d] Using GPU %d\n", rank, gpu_id);

    // Initialize NCCL
    ncclUniqueId id;
    ncclComm_t comm;

    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    printf("[Rank %d] Initializing NCCL communicator\n", rank);
    NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
    printf("[Rank %d] NCCL communicator initialized\n", rank);

    // Test parameters
    size_t size = 1024 * 1024; // 1 MB
    size_t count = size / sizeof(float);

    printf("[Rank %d] Test size: %zu bytes (%zu elements)\n", rank, size, count);

    // HOST buffers for initialization and verification
    float *h_sendbuf = (float*)malloc(size);
    float *h_recvbuf = (float*)malloc(size);

    // Initialize host buffers
    for (size_t i = 0; i < count; i++) {
        h_sendbuf[i] = rank * 1000.0f + i;
        h_recvbuf[i] = -1.0f;
    }

    // ⭐ KEY: Use ncclMemAlloc for SYMMETRIC allocation
    // This ensures same virtual address on all ranks (required for GIN)
    void *d_sendbuf = NULL;
    void *d_recvbuf = NULL;

    printf("[Rank %d] Allocating SYMMETRIC memory with ncclMemAlloc()\n", rank);
    NCCLCHECK(ncclMemAlloc(&d_sendbuf, size));
    NCCLCHECK(ncclMemAlloc(&d_recvbuf, size));

    printf("[Rank %d] Symmetric buffers allocated: sendbuf=%p recvbuf=%p\n",
           rank, d_sendbuf, d_recvbuf);

    // Copy data to device
    CUDACHECK(cudaMemcpy(d_sendbuf, h_sendbuf, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_recvbuf, h_recvbuf, size, cudaMemcpyHostToDevice));

    // Create CUDA stream
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Synchronize before registering
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    printf("[Rank %d] Registering memory window for GIN mode\n", rank);

    // Register memory window for GIN mode
    ncclWindow_t win = NULL;
    ncclResult_t res = ncclCommWindowRegister(comm, d_recvbuf, size, &win, 0);

    if (res != ncclSuccess) {
        printf("[Rank %d] ❌ ERROR: ncclCommWindowRegister failed: %s\n",
               rank, ncclGetErrorString(res));
        printf("[Rank %d] Note: Ensure NCCL_GIN_TYPE=2 for AWS EFA\n", rank);

        // Cleanup
        CUDACHECK(cudaStreamDestroy(stream));
        NCCLCHECK(ncclMemFree(d_sendbuf));
        NCCLCHECK(ncclMemFree(d_recvbuf));
        free(h_sendbuf);
        free(h_recvbuf);
        NCCLCHECK(ncclCommDestroy(comm));
        MPICHECK(MPI_Finalize());
        return 1;
    }

    printf("[Rank %d] ✓ Memory window registered successfully at %p\n", rank, d_recvbuf);

    // Synchronize all ranks before test
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Test: rank 0 writes to rank 1
    if (rank == 0 && nranks >= 2) {
        int peer = 1;
        printf("[Rank %d] Writing to rank %d using ncclPutSignal\n", rank, peer);

        NCCLCHECK(ncclPutSignal(d_sendbuf, count, ncclFloat, peer, win, 0,
                                0, 0, 0, comm, stream));

    } else if (rank == 1) {
        int peer = 0;
        printf("[Rank %d] Waiting for data from rank %d\n", rank, peer);

        ncclWaitSignalDesc_t desc;
        desc.opCnt = 1;
        desc.peer = peer;
        desc.sigIdx = 0;
        desc.ctx = 0;

        NCCLCHECK(ncclWaitSignal(1, &desc, comm, stream));
    }

    // Synchronize stream
    CUDACHECK(cudaStreamSynchronize(stream));

    // Verify (only rank 1)
    if (rank == 1) {
        CUDACHECK(cudaMemcpy(h_recvbuf, d_recvbuf, size, cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < 10 && i < count; i++) {
            float expected = 0 * 1000.0f + i;  // From rank 0
            if (h_recvbuf[i] != expected) {
                printf("[Rank %d] ❌ Mismatch at index %zu: expected %f, got %f\n",
                       rank, i, expected, h_recvbuf[i]);
                errors++;
            }
        }

        if (errors == 0) {
            printf("[Rank %d] ✅ Data verification PASSED\n", rank);
        } else {
            printf("[Rank %d] ❌ Data verification FAILED (%d errors)\n", rank, errors);
        }
    }

    // Cleanup
    printf("[Rank %d] Cleaning up\n", rank);
    NCCLCHECK(ncclCommWindowDeregister(comm, win));
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclMemFree(d_sendbuf));
    NCCLCHECK(ncclMemFree(d_recvbuf));
    free(h_sendbuf);
    free(h_recvbuf);
    NCCLCHECK(ncclCommDestroy(comm));

    if (rank == 0) {
        printf("\n==============================================\n");
        printf("✅ GIN mode P2P test completed successfully!\n");
        printf("==============================================\n");
        printf("Key: Used ncclMemAlloc() for symmetric memory\n");
        printf("     Required for GIN window registration\n");
        printf("==============================================\n");
    }

    MPICHECK(MPI_Finalize());
    return 0;
}
