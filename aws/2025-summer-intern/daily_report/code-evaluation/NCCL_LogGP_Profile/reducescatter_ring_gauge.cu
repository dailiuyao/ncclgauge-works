#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <thread>
#include <chrono> 
#include <string>
#include "get_clock.h"

typedef std::chrono::duration<double, std::milli> Duration;  // Use 'typedef' for type definition

std::chrono::time_point<std::chrono::high_resolution_clock> net_transmitted_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> net_done_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> net_post_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> net_data_ready_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS];

std::chrono::time_point<std::chrono::high_resolution_clock> nccl_kernel_start_time;

Duration T1[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS];
Duration T2[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS];
Duration T9[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS];
Duration T8[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS];
Duration T0_total[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS];
Duration T7_total[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS];
Duration T10_total[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS];
Duration T0[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS];
Duration T7[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS];
Duration T10[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS];


int lyd_chunk_num;

int lyd_channels_num;

int lyd_protocol;

int lyd_algo;

#define N_GAUGE_GPUS 8

#define  MAX_PRINT_CHUNKS (MAX_GAUGE_CHUNKS - 1)

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

uint64_t rdtsc() {
    uint32_t lo, hi;
    // Inline assembly to read the TSC
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64_t)hi << 32 | lo;
}

void printNCCLInfo(std::string chosen_protocol,
                   std::string chosen_algo,
                   const char* env_gauge_heo_var,
                   const char* env_gauge_mode_var,
                   const char* env_gauge_size_var,
                   const char* env_gauge_nchannels_var,
                   const char* env_gauge_iteration_var,
                   double nccl_init_time) {
    printf("INFO: algo(%s)_protocol(%s)_mode(%s)_message size(%s)_"
           "nchannels(%d)_nmessages(%d)_"
           "iteration(%s)\n", 
           chosen_algo.c_str(), 
           chosen_protocol.c_str(), 
           env_gauge_mode_var, 
           env_gauge_size_var, 
           lyd_channels_num, 
           N_ITERS, 
           env_gauge_iteration_var);
    
    printf("-- [T init] nccl init time: %.6f ms\n", 
           nccl_init_time);
}

void AccumulateTimingMetrics(
  int myRank, 
  int channel_id, 
  int peer_id, 
  int lyd_chunk_num,
  const std::chrono::time_point<std::chrono::high_resolution_clock>& nccl_kernel_start_time,
  const std::chrono::time_point<std::chrono::high_resolution_clock>& nccl_func_end_time,
  const std::chrono::time_point<std::chrono::high_resolution_clock> (&net_post_time)[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS],
  const std::chrono::time_point<std::chrono::high_resolution_clock> (&net_data_ready_time)[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS],
  const std::chrono::time_point<std::chrono::high_resolution_clock> (&net_transmitted_time)[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS],
  const std::chrono::time_point<std::chrono::high_resolution_clock> (&net_done_time)[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS]) {

  auto t1_diff = net_post_time[channel_id][0][peer_id] - nccl_kernel_start_time;
  if (t1_diff.count() > 0) {
    T1[channel_id][peer_id] += t1_diff;
  }

  auto t2_diff = net_data_ready_time[channel_id][0][peer_id] - nccl_kernel_start_time;
  if (t2_diff.count() > 0) {
    T2[channel_id][peer_id] += t2_diff;
  }

  auto t9_diff = net_done_time[channel_id][0][peer_id] - nccl_kernel_start_time;
  if (t9_diff.count() > 0) {
    T9[channel_id][peer_id] += t9_diff;
  }

  auto t8_diff = nccl_func_end_time - net_done_time[channel_id][MAX_GAUGE_CHUNKS - 1][peer_id];
  if (t8_diff.count() > 0) {
    T8[channel_id][peer_id] += t8_diff;
  }

  for (size_t i = 0; i < std::min(static_cast<size_t>(lyd_chunk_num), static_cast<size_t>(MAX_PRINT_CHUNKS)); ++i) {
    auto t0_diff = net_post_time[channel_id][i][peer_id] - net_post_time[channel_id][0][peer_id];

    if (t0_diff.count() >= 0) {
      T0[channel_id][i][peer_id] += t0_diff;
    }

    auto t7_diff = net_done_time[channel_id][i][peer_id] - net_transmitted_time[channel_id][i][peer_id];
    if (t7_diff.count() > 0) {
      T7[channel_id][i][peer_id] += t7_diff;
    }

    auto t10_diff = net_done_time[channel_id][i][peer_id] - net_post_time[channel_id][0][peer_id];
    if (t10_diff.count() > 0) {
      T10[channel_id][i][peer_id] += t10_diff;
    }
  }
  
  if (lyd_chunk_num > MAX_PRINT_CHUNKS) {
    auto t0_total_diff = net_post_time[channel_id][MAX_PRINT_CHUNKS][peer_id] - net_post_time[channel_id][0][peer_id];
    if (t0_total_diff.count() > 0) {
      T0_total[channel_id][peer_id] += t0_total_diff;
    }

    auto t7_total_diff = net_done_time[channel_id][MAX_PRINT_CHUNKS][peer_id] - net_transmitted_time[channel_id][MAX_PRINT_CHUNKS][peer_id];
    if (t7_total_diff.count() > 0) {
      T7_total[channel_id][peer_id] += t7_total_diff;
    }

    auto t10_total_diff = net_done_time[channel_id][MAX_PRINT_CHUNKS][peer_id] - net_post_time[channel_id][0][peer_id];
    if (t10_total_diff.count() > 0) {
      T10_total[channel_id][peer_id] += t10_total_diff;
    }
  }
}

void printTimingMetrics(
  int myRank, 
  int channel_id, 
  int peer_id, 
  int lyd_chunk_num,
  const Duration (&T1)[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS],
  const Duration (&T2)[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS],
  const Duration (&T9)[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS],
  const Duration (&T8)[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS],
  const Duration (&T0_total)[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS],
  const Duration (&T7_total)[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS],
  const Duration (&T10_total)[MAX_GAUGE_CHANNELS][MAX_GAUGE_PEERS],
  const Duration (&T0)[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS],
  const Duration (&T7)[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS],
  const Duration (&T10)[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_GAUGE_PEERS]) {

  printf("-- [T1] kernel start to first post time: %.6f ms\n", 
          T1[channel_id][peer_id].count() / N_ITERS);

  printf("-- [T2] kernel start to first data ready time: %.6f ms\n", 
          T2[channel_id][peer_id].count() / N_ITERS);

  printf("-- [T9] kernel start to first done time: %.6f ms\n", 
          T9[channel_id][peer_id].count() / N_ITERS);

  printf("-- [T8] nccl last ncclIsend done to func end time: %.6f ms\n", 
          T8[channel_id][peer_id].count() / N_ITERS);

  for (size_t i = 0; i < std::min(static_cast<size_t>(lyd_chunk_num), static_cast<size_t>(MAX_PRINT_CHUNKS)); ++i) {    
    printf("-- [T0] chunk %zu posted from first chunk posted time (r%d to r%d) in channel %d: %.6f ms\n", 
            i, myRank, peer_id, channel_id, T0[channel_id][i][peer_id].count() / N_ITERS);

    // calc_time = net_data_ready_time[channel_id][i][peer_id] - net_post_time[channel_id][i][peer_id];
    // printf("-- [T3] chunk %zu posted to data ready time (r%d to r%d) in channel %d: %.6f ms\n", 
    //         i, myRank, peer_id, channel_id, calc_time.count());

    // calc_time = net_transmitted_time[channel_id][i][peer_id] - net_data_ready_time[channel_id][i][peer_id];
    // printf("-- [T4] chunk %zu data ready to transmitted time (r%d to r%d) in channel %d: %.6f ms\n", 
    //         i, myRank, peer_id, channel_id, calc_time.count());

    printf("-- [T7] chunk %zu transmitted to done time (r%d to r%d) in channel %d: %.6f ms\n", 
            i, myRank, peer_id, channel_id, T7[channel_id][i][peer_id].count() / N_ITERS);
    printf("-- [T10] chunk 0 posted to chunk %zu done time (r%d to r%d) in channel %d: %.6f ms\n", 
            i, myRank, peer_id, channel_id, T10[channel_id][i][peer_id].count() / N_ITERS);
  }
  
  if (lyd_chunk_num > MAX_PRINT_CHUNKS) {
    printf("-- [T0] chunk %zu posted from first chunk posted time (r%d to r%d) in channel %d: %.6f ms\n", 
          static_cast<size_t>(lyd_chunk_num), myRank, peer_id, channel_id, T0_total[channel_id][peer_id].count() / N_ITERS);

    // calc_time = net_data_ready_time[channel_id][MAX_PRINT_CHUNKS][peer_id] - net_post_time[channel_id][MAX_PRINT_CHUNKS][peer_id];
    // printf("-- [T3] chunk %zu posted to data ready time (r%d to r%d) in channel %d: %.6f ms\n", 
    //       static_cast<size_t>(lyd_chunk_num), myRank, peer_id, channel_id, calc_time.count());

    // calc_time = net_transmitted_time[channel_id][MAX_PRINT_CHUNKS][peer_id] - net_data_ready_time[channel_id][MAX_PRINT_CHUNKS][peer_id];
    // printf("-- [T4] chunk %zu data ready to transmitted time (r%d to r%d) in channel %d: %.6f ms\n", 
    //       static_cast<size_t>(lyd_chunk_num), myRank, peer_id, channel_id, calc_time.count());

    printf("-- [T7] chunk %zu transmitted to done time (r%d to r%d) in channel %d: %.6f ms\n", 
          static_cast<size_t>(lyd_chunk_num), myRank, peer_id, channel_id, T7_total[channel_id][peer_id].count() / N_ITERS);
    printf("-- [T10] chunk 0 posted to chunk %zu done time (r%d to r%d) in channel %d: %.6f ms\n", 
          static_cast<size_t>(lyd_chunk_num), myRank, peer_id, channel_id, T10_total[channel_id][peer_id].count() / N_ITERS);
  }
}

int main(int argc, char* argv[])
{
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

  TimePoint nccl_init_time_start;

  TimePoint nccl_init_time_end;

  for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
    for (int k = 0; k < MAX_GAUGE_PEERS; ++k) {
      for (int j = 0; j < MAX_GAUGE_CHUNKS; ++j) {
        net_transmitted_time[i][j][k] = TimePoint();
        net_done_time[i][j][k] = TimePoint();
        net_post_time[i][j][k] = TimePoint();
        net_data_ready_time[i][j][k] = TimePoint();
      }
    }
  }

  // Initialize arrays to zero
  for (int i = 0; i < MAX_GAUGE_CHANNELS; i++) {
      for (int j = 0; j < MAX_GAUGE_PEERS; j++) {
          T1[i][j] = Duration::zero();  
          T2[i][j] = Duration::zero();
          T9[i][j] = Duration::zero();
          T8[i][j] = Duration::zero();
          T0_total[i][j] = Duration::zero();
          T7_total[i][j] = Duration::zero();
          T10_total[i][j] = Duration::zero();
          
          for (int k = 0; k < MAX_GAUGE_CHUNKS; k++) {
              T0[i][k][j] = Duration::zero();
              T7[i][k][j] = Duration::zero();
              T10[i][k][j] = Duration::zero();
          }
      }
  }

  const char* env_gauge_heo_var = getenv("GAUGE_HEO");

  const char* env_gauge_mode_var = getenv("GAUGE_MODE");

  const char* env_gauge_iteration_var = getenv("GAUGE_ITERATION");

  const char* env_gauge_nchannels_var = getenv("GAUGE_NCHANNELS");

  const char* env_gauge_output_dir_var = getenv("GAUGE_OUT_DIRE");

  // Check if environment variables are set
  if (!env_gauge_heo_var) env_gauge_heo_var = "unknown_gauge_heo";
  if (!env_gauge_mode_var) env_gauge_mode_var = "unknown_gauge_mode";
  if (!env_gauge_iteration_var) env_gauge_iteration_var = "unknown_gauge_iteration";
  if (!env_gauge_nchannels_var) env_gauge_nchannels_var = "unknown_gauge_nchannels";
  if (!env_gauge_output_dir_var) {
    env_gauge_output_dir_var = "unknown_gauge_output_dir";
    printf("unknown gauge output dir\n");
  }

  long long size = 1;  // Default size
  const char* env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
  // if (env_gauge_size_var != nullptr) {
  //     size = atoll(env_gauge_size_var) * 1024 / 4;  // Convert from kilobytes to number of floats, assuming the environment variable is in kilobytes
  // }
  if (env_gauge_size_var != nullptr) {
    // Parse fraction if present
    if (strchr(env_gauge_size_var, '/') != nullptr) {
        double numerator, denominator;
        sscanf(env_gauge_size_var, "%lf/%lf", &numerator, &denominator);
        size = (long long)((numerator/denominator) * 1024 / 4);
    } else {
        size = atoll(env_gauge_size_var) * 1024 / 4;
    }
  }

  int gauge_nchannels = atoi(env_gauge_nchannels_var);

  int myRank, nRanks, localRank = 0;

  // Set the device scheduling flag before creating a device context
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device flags: %s\n", cudaGetErrorString(err));
        return 1;
    }

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  char filename[256];

  if (myRank == 0 || myRank == 1) {
      sprintf(filename, "%s/nccl_%s_heo-%s_r-%d.out",
              env_gauge_output_dir_var, env_gauge_mode_var, env_gauge_heo_var,
              myRank);

      FILE *file = freopen(filename, "a", stdout);
      if (file == NULL) {
          perror("freopen failed");
          MPI_Abort(MPI_COMM_WORLD, 1);
      }

      setbuf(stdout, NULL);  // Disable buffering
      fflush(stdout);
  } else {
      freopen("/dev/null", "w", stdout);
  }

  MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks proceed together

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);

  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, nRanks * size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float))); 
  CUDACHECK(cudaStreamCreate(&s));
  
  ////////////////////////////// PROFILE_LYD_ALLREDUCE_DEVICE: START //////////////////////////////
  
  nccl_init_time_start = std::chrono::high_resolution_clock::now();
  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  nccl_init_time_end = std::chrono::high_resolution_clock::now();

  //communicating using NCCL
  cudaEvent_t start, stop;
  float elapsed_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // // Warm up START 
  CUDACHECK(cudaStreamSynchronize(s));

  for (int i = 0 ; i < WARMUP_ITERATION; i++) {
    NCCLCHECK(ncclReduceScatter((const void*)((float*)sendbuff), 
                        (void*)((float*)recvbuff), 
                        size, 
                        ncclFloat, 
                        ncclSum, 
                        comm, 
                        s));
    CUDACHECK(cudaStreamSynchronize(s));
  }

  // // Warm up END

  ////////////////////////////// PROFILE_LYD_ALLREDUCE_DEVICE: START //////////////////////////////

  CUDACHECK(cudaStreamSynchronize(s));

  cudaEventRecord(start, s);

  for (int i = 0; i < N_ITERS; i++) {
    NCCLCHECK(ncclReduceScatter((const void*)((float*)sendbuff), 
                            (void*)((float*)recvbuff), 
                            size, 
                            ncclFloat, 
                            ncclSum, 
                            comm, 
                            s));
    // busyWaitMilliseconds(gauge_d); 
    CUDACHECK(cudaStreamSynchronize(s));

    std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_end_time = std::chrono::high_resolution_clock::now();

    if (myRank == 1) {
      int channel_id = 0;
      int peer_id = 8;
      AccumulateTimingMetrics(myRank, channel_id, peer_id, lyd_chunk_num,
                    nccl_kernel_start_time, nccl_func_end_time,
                    net_post_time,
                    net_data_ready_time, net_transmitted_time,
                    net_done_time);
    }
    CUDACHECK(cudaStreamSynchronize(s));
  }
  
  CUDACHECK(cudaStreamSynchronize(s));

  cudaEventRecord(stop, s);
  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  // Calculate elapsed time between events
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // Destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop); 

  std::chrono::duration<float, std::milli> nccl_init_time = nccl_init_time_end - nccl_init_time_start;

  ////////////////////////////// PROFILE_LYD_ALLREDUCE_DEVICE: END //////////////////////////////

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  std::string chosen_protocol;
  if (lyd_protocol == 0) {
      chosen_protocol = "LL";
  } else if (lyd_protocol == 1) { 
      chosen_protocol = "LL128";
  } else {
      chosen_protocol = "Simple";
  }

  std::string chosen_algo;
  if (lyd_algo == 0) {
      chosen_algo = "TREE";
  } else if (lyd_algo == 1) { 
      chosen_algo = "RING";
  } else if (lyd_algo == 5) {
      chosen_algo = "NVLS_TREE";
  } else if (lyd_algo == 6) {
      chosen_algo = "PAT";
  }

  MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks proceed together

  // Create a buffer to hold the string
  char protocol_buffer[32] = {0};  // Adjust size as needed
  char algo_buffer[32] = {0};  // Adjust size as needed

  if (myRank == 0) {
      // Copy the string to the buffer on rank 0
      strncpy(protocol_buffer, chosen_protocol.c_str(), sizeof(protocol_buffer) - 1);
      strncpy(algo_buffer, chosen_algo.c_str(), sizeof(algo_buffer) - 1);
  }

  // Broadcast the buffer from rank 0 to all ranks
  MPI_Bcast(protocol_buffer, sizeof(protocol_buffer), MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(algo_buffer, sizeof(algo_buffer), MPI_CHAR, 0, MPI_COMM_WORLD);

  // Broadcast lyd_channels_num from rank 0 to all ranks
  MPI_Bcast(&lyd_channels_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (myRank != 0) {
      // Other ranks copy from the buffer to their string
      chosen_protocol = std::string(protocol_buffer);
      chosen_algo = std::string(algo_buffer);
  }

  MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks proceed together

  printf("------ Reduce ------\n");
  if (myRank == 1) {
    printNCCLInfo(chosen_protocol, chosen_algo, env_gauge_heo_var, env_gauge_mode_var, env_gauge_size_var, env_gauge_nchannels_var, env_gauge_iteration_var, 
                nccl_init_time.count());
    int channel_id = 0;
    int peer_id = 8;
    printTimingMetrics(myRank, channel_id, peer_id, lyd_chunk_num,
                  T1, T2, T9, T8,
                  T0_total, T7_total, T10_total,
                  T0, T7, T10);
  }

  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}