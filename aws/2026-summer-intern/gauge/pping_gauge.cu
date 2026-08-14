#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <thread>
#include "get_clock.h"
#include <limits>  // for std::numeric_limits


#ifndef GAUGE_N_MESSAGES
#define GAUGE_N_MESSAGES 1  // fallback default
#endif

#define MAX_GAUGE_CHANNELS_PRINT 2

std::chrono::time_point<std::chrono::high_resolution_clock> net_first_post_time[MAX_GAUGE_CHANNELS][N_MESSAGES][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> netIsend_first_chunk_time[MAX_GAUGE_CHANNELS][N_MESSAGES][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> netIrecv_first_chunk_time[MAX_GAUGE_CHANNELS][N_MESSAGES][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> netIsend_last_chunk_time[MAX_GAUGE_CHANNELS][N_MESSAGES][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> netIrecv_last_chunk_time[MAX_GAUGE_CHANNELS][N_MESSAGES][MAX_PEERS];

std::chrono::time_point<std::chrono::high_resolution_clock> net_transmitted_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> send_net_done_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> recv_net_done_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> net_post_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> net_data_ready_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];

#include "profiling_interface.h"
// Plugin timing arrays are now defined in libprofiling_arrays.so (declared as extern in profiling_interface.h)
// No definitions here - we link against the shared library

std::chrono::time_point<std::chrono::high_resolution_clock> net_Recv_polled_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> net_Recv_done_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
std::chrono::time_point<std::chrono::high_resolution_clock> net_Recv_posted_time[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];

std::chrono::time_point<std::chrono::high_resolution_clock> proxy_init_time_start;
std::chrono::time_point<std::chrono::high_resolution_clock> proxy_init_time_end;
std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_start_time;
std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_end_time;

int host_isend_gauge_d;
int host_check_GPU_data_gauge_d;
int host_recv_gauge_d;
int gauge_send_message_itr;
int gauge_recv_message_itr;
int gauge_cpu_mhz;
int myRank, nRanks;

// v2.27.6 approach: gauge DEFINES these variables (owns the storage)
// NCCL library references them as undefined symbols, resolved at runtime
// gauge_send_message_itr and gauge_recv_message_itr already defined above
int gauge_chunk_num;
int gauge_channels_num;
int gauge_protocol;
// Bitmask of channels actually used by NCCL on the most recent ncclSend/Recv.
// Set by enqueue.cc when scheduling the P2P plan. For a fixed message size,
// every iteration in the test window picks the same channels, so the latest
// value of this mask is enough to know which channels were used.
uint64_t gauge_channels_used_mask = 0;

// Helper: returns true iff channel `ch` was actually used by NCCL.
// Reads gauge_channels_used_mask, which NCCL updates inside enqueue.cc on every
// ncclSend/Recv (cleared at the start of each plan, then bits get OR'd in for
// every channel actually picked). Since every iter under a fixed message size
// produces the same mask, reading it directly here is correct.
// If the mask is still 0 (NCCL didn't update it for some reason), fall back
// to printing every configured channel (no filtering).
static inline bool channel_was_used(size_t ch) {
    if (gauge_channels_used_mask == 0) return true;
    return (gauge_channels_used_mask & (uint64_t(1) << ch)) != 0;
}

// Real plugin chunk count: number of distinct msg_seq_num slots the plugin
// wrote in the most recent kept iteration. Updated in AccumulateTimingMetrics
// and printed as part of the per-test Plugin Timing Metrics block.
int g_plugin_real_chunk_count = 0;

int gauge_iterations;

// Environment variables
const char* env_gauge_heo_var;
const char* env_gauge_mode_var;
const char* env_gauge_size_var;
const char* env_gauge_nthreads_var;
const char* env_gauge_iteration_var;

#define DEFAULT_D 0
#define GAUGE_SUBGROUP_SIZE 1

int send_peer_idx;  // tpRemoteRank of the send target, set at runtime

int send_gauge_d = DEFAULT_D;
int recv_gauge_d = DEFAULT_D;

// --- Filtering controls ---
static const double kOutlierFactor = 1.3;   // threshold = 1.5 * running_mean (tunable)
static const int    kWarmupKeep    = 10;      // keep first W iterations unfiltered

// Count of iterations that actually contributed to sums/averages
static size_t g_kept_iters = 0;

// Baseline = min of first kWarmupKeep iterations (no std::vector)
static int    g_warmup_count   = 0;
static double g_baseline_min_ms = std::numeric_limits<double>::infinity();
static double g_baseline_ms     = 0.0;
static bool   g_baseline_ready  = false;

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
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64_t)hi << 32 | lo;
}

typedef std::chrono::duration<double, std::milli> Duration;

// Timing variables
Duration Sum_T_total = Duration(0.0);
Duration Sum_T1 = Duration(0.0);
Duration Sum_T2 = Duration(0.0);
Duration Sum_T8 = Duration(0.0);
Duration Sum_net_post_compl_time = Duration(0.0);
Duration Sum_netIrecv_netIsend_time = Duration(0.0);

// Dynamic timing arrays
Duration Sum_T0[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
Duration Sum_T3[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
Duration Sum_T4[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
Duration Sum_T7[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
// Per-channel: time from chunk-0 data-ready to last-chunk done (covers the
// data-transfer span of one ncclSend across however many chunks it produced).
Duration Sum_T_dataready_to_done[MAX_GAUGE_CHANNELS];
Duration Sum_T5[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
Duration Sum_T6[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][MAX_PEERS];
Duration Sum_T9_1[MAX_GAUGE_CHANNELS][N_MESSAGES];
Duration Sum_T10_1[MAX_GAUGE_CHANNELS][N_MESSAGES];

// New plugin timing metrics (per-chunk, indexed by chunk_id = msg_seq_num - 1)
Duration Sum_T_nccl_to_libfabric[MAX_GAUGE_CHUNKS];  // ncclSend launch (CPU)  → first rail post
Duration Sum_T_kernel_to_libfabric[MAX_GAUGE_CHUNKS]; // ncclSend kernel start (GPU) → first rail post
Duration Sum_T_first_rail[MAX_GAUGE_CHUNKS];         // First rail post → first rail completion
Duration Sum_T_all_rails[MAX_GAUGE_CHUNKS];          // First rail post → all rails completion
Duration Sum_T_rail_skew[MAX_GAUGE_CHUNKS];          // First rail compl → last rail compl
Duration Sum_T_libfabric_to_done[MAX_GAUGE_CHUNKS];  // Last rail compl (plugin) → NCCL test() done

// CUDA-event-based GPU↔CPU clock calibration for measuring kernel launch time.
// kernel_start_event is recorded on the stream right before each ncclSend; once
// the GPU executes that command, its timestamp marks the moment NCCL's send
// kernel is about to launch. We translate the GPU time back into CPU time using
// a one-shot calibration captured during init.
cudaEvent_t calib_event;
std::chrono::time_point<std::chrono::high_resolution_clock> calib_cpu_time;
cudaEvent_t kernel_start_event;
// Filled in each iteration body before AccumulateTimingMetrics is called.
std::chrono::time_point<std::chrono::high_resolution_clock> kernel_launch_cpu_time;
Duration Sum_netIrecv_time[MAX_GAUGE_CHANNELS][N_MESSAGES];
Duration Sum_netIrecv_total_time[MAX_GAUGE_CHANNELS];

int N_CHUNKS = MAX_GAUGE_CHUNKS - 1;

std::chrono::duration<float, std::milli> post_compl_time;
std::chrono::duration<float, std::milli> nccl_func_time;

void initializeTimingArrays() {
    for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
        for (int j = 0; j < MAX_GAUGE_CHUNKS; ++j) {
            for (int k = 0; k < MAX_PEERS; ++k) {
                Sum_T0[i][j][k] = Duration(0.0);
                Sum_T3[i][j][k] = Duration(0.0);
                Sum_T4[i][j][k] = Duration(0.0);
                Sum_T7[i][j][k] = Duration(0.0);
                Sum_T5[i][j][k] = Duration(0.0);
                Sum_T6[i][j][k] = Duration(0.0);
            }
        }
    }

    for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
        for (int j = 0; j < N_MESSAGES; ++j) {
            Sum_T9_1[i][j] = Duration(0.0);
            Sum_T10_1[i][j] = Duration(0.0);
            Sum_netIrecv_time[i][j] = Duration(0.0);
        }
        Sum_netIrecv_total_time[i] = Duration(0.0);
        Sum_T_dataready_to_done[i] = Duration(0.0);
    }

    // Initialize plugin timing arrays
    for (int i = 0; i < MAX_GAUGE_CHUNKS; ++i) {
        Sum_T_nccl_to_libfabric[i] = Duration(0.0);
        Sum_T_kernel_to_libfabric[i] = Duration(0.0);
        Sum_T_first_rail[i] = Duration(0.0);
        Sum_T_all_rails[i] = Duration(0.0);
        Sum_T_rail_skew[i] = Duration(0.0);
        Sum_T_libfabric_to_done[i] = Duration(0.0);
    }

    // DO NOT initialize plugin timestamp arrays here!
    // These arrays are shared with the plugin and written by the plugin.
    // Initializing them here would clear the plugin's data.
    // The arrays are initialized once at program startup in libprofiling_arrays.so
}

void AccumulateTimingMetrics() {
    static int accum_call_count = 0;
    accum_call_count++;

    std::chrono::duration<float, std::milli> netIrecv_netIsend_time;
    std::chrono::duration<float, std::milli> netIsend_func_time;
    std::chrono::duration<float, std::milli> proxy_init_time;
    std::chrono::duration<float, std::milli> func_start_netIsend_time;
    std::chrono::duration<float, std::milli> netIrecv_total_time(0.0f);
    std::chrono::duration<float, std::milli> netIbsend_time;
    std::chrono::duration<float, std::milli> netIbrecv_time;

    // Identify the first/last channel NCCL actually used this iteration.
    // With NCCL_MIN/MAX_NCHANNELS > 1, NCCL may still pick only a subset
    // (e.g. channel 2 out of 4 for a tiny msg). Reading channel 0 or
    // channel N-1 unconditionally yields an uninitialized timestamp
    // (epoch=0) and produces ±1.78e12 ms garbage in T2/T8/etc.
    int first_used_ch = 0;
    int last_used_ch  = (int)gauge_channels_num - 1;
    if (gauge_channels_used_mask != 0) {
        first_used_ch = __builtin_ctzll(gauge_channels_used_mask);
        last_used_ch  = 63 - __builtin_clzll(gauge_channels_used_mask);
    }

    if (myRank == 0) {
        Sum_T_total += Duration(nccl_func_time.count());
        proxy_init_time = proxy_init_time_end - proxy_init_time_start;
        Sum_T1 += Duration(proxy_init_time.count());

        // T2 / T8 used to hard-code channel 0 / gauge_channels_num-1, but NCCL
        // may pick a different channel; we use first_used_ch / last_used_ch
        // (computed above outside the if/else) instead.
        int p = send_peer_idx;
        func_start_netIsend_time = netIsend_first_chunk_time[first_used_ch][0][p] - nccl_func_start_time;
        Sum_T2 += Duration(func_start_netIsend_time.count());

        // Use send_net_done_time sentinel for send operations
        netIsend_func_time = nccl_func_end_time - send_net_done_time[last_used_ch][MAX_GAUGE_CHUNKS - 1][p];
        Sum_T8 += Duration(netIsend_func_time.count());

        for (size_t channelid = 0; channelid < gauge_channels_num; channelid++) {
            for (size_t i = 0; i < std::min(static_cast<size_t>(gauge_chunk_num), static_cast<size_t>(N_CHUNKS)); ++i) {
                netIbsend_time = net_post_time[channelid][i][p] - net_post_time[channelid][0][p];
                Sum_T0[channelid][i][0] += Duration(netIbsend_time.count());
                netIbsend_time = net_data_ready_time[channelid][i][p] - net_post_time[channelid][i][p];
                Sum_T3[channelid][i][0] += Duration(netIbsend_time.count());
                netIbsend_time = net_transmitted_time[channelid][i][p] - net_data_ready_time[channelid][i][p];
                Sum_T4[channelid][i][0] += Duration(netIbsend_time.count());
                netIbsend_time = send_net_done_time[channelid][i][p] - net_transmitted_time[channelid][i][p];
                Sum_T7[channelid][i][0] += Duration(netIbsend_time.count());
            }
        }

        for (size_t channelid = 0; channelid < gauge_channels_num; channelid++) {
            if (gauge_chunk_num < MAX_GAUGE_CHUNKS)
                net_data_ready_time[channelid][MAX_GAUGE_CHUNKS - 1][p] = net_data_ready_time[channelid][gauge_chunk_num - 1][p];

            netIbsend_time = net_post_time[channelid][MAX_GAUGE_CHUNKS - 1][p] - net_post_time[channelid][0][p];
            Sum_T0[channelid][MAX_GAUGE_CHUNKS - 1][0] += Duration(netIbsend_time.count());
            netIbsend_time = net_data_ready_time[channelid][MAX_GAUGE_CHUNKS - 1][p] - net_post_time[channelid][MAX_GAUGE_CHUNKS - 1][p];
            Sum_T3[channelid][MAX_GAUGE_CHUNKS - 1][0] += Duration(netIbsend_time.count());
            netIbsend_time = net_transmitted_time[channelid][MAX_GAUGE_CHUNKS - 1][p] - net_data_ready_time[channelid][MAX_GAUGE_CHUNKS - 1][p];
            Sum_T4[channelid][MAX_GAUGE_CHUNKS - 1][0] += Duration(netIbsend_time.count());
            netIbsend_time = send_net_done_time[channelid][MAX_GAUGE_CHUNKS - 1][p] - net_transmitted_time[channelid][MAX_GAUGE_CHUNKS - 1][p];
            Sum_T7[channelid][MAX_GAUGE_CHUNKS - 1][0] += Duration(netIbsend_time.count());

            netIbsend_time = send_net_done_time[channelid][MAX_GAUGE_CHUNKS - 1][p] - net_data_ready_time[channelid][0][p];
            Sum_T_dataready_to_done[channelid] += Duration(netIbsend_time.count());
        }

        // Calculate plugin timing metrics (per-chunk)
        // The plugin's msg_seq_num is monotonic across warmup + test, so chunk_id 0 of
        // the test maps to plugin slot[plugin_test_start_seq_num], chunk_id 1 to slot[start+1], etc.
        // plugin_test_start_seq_num is captured atomically by the plugin on its first post
        // after the test-window reset.
        int plugin_start = plugin_test_start_seq_num;

        // Count the real number of chunks the plugin wrote during this iteration.
        // The plugin only writes libfabric_first_post_time[seq] when the slot is
        // empty (epoch == 0), and the gauge clears all slots between iterations
        // (see end of this function). So the number of non-zero slots == the
        // number of distinct msg_seq_num values plugin saw == real chunks written
        // by NCCL into the network plugin.
        // We don't print per-iter; just record the value so it can be printed
        // once as part of the Plugin Timing Metrics block (see printTimingMetrics).
        int plugin_real_chunk_count = 0;
        for (int i = 0; i < MAX_PLUGIN_MSG_SEQ_NUM; i++) {
            if (libfabric_first_post_time[i].time_since_epoch().count() != 0) {
                plugin_real_chunk_count++;
            }
        }
        g_plugin_real_chunk_count = plugin_real_chunk_count;

        for (size_t chunk_id = 0; chunk_id < gauge_chunk_num && chunk_id < MAX_GAUGE_CHUNKS; chunk_id++) {
            size_t plugin_idx = (plugin_start >= 0)
                ? (size_t)((plugin_start + (int)chunk_id) % MAX_PLUGIN_MSG_SEQ_NUM)
                : chunk_id;
            // T_nccl_to_libfabric: time from CPU-side ncclSend launch (nccl-tests
            // style start point) to plugin's first fi_write for this chunk.
            // Captures the full launch overhead: ncclSend host call + kernel
            // launch + proxy wakeup + plugin post path.
            netIbsend_time = libfabric_first_post_time[plugin_idx] - nccl_func_start_time;
            Sum_T_nccl_to_libfabric[chunk_id] += Duration(netIbsend_time.count());

            // T_kernel_to_libfabric: time from GPU-side kernel launch start
            // (translated to CPU clock via cudaEvent calibration) to plugin's
            // first fi_write. Strips out CPU-side ncclSend host API + kernel
            // enqueue overhead, leaving: kernel exec + proxy wakeup + plugin.
            netIbsend_time = libfabric_first_post_time[plugin_idx] - kernel_launch_cpu_time;
            Sum_T_kernel_to_libfabric[chunk_id] += Duration(netIbsend_time.count());

            // T_first_rail: First rail post → first rail completion
            netIbsend_time = libfabric_first_compl_time[plugin_idx] - libfabric_first_post_time[plugin_idx];
            Sum_T_first_rail[chunk_id] += Duration(netIbsend_time.count());

            // T_all_rails: First rail post → all rails completion
            netIbsend_time = libfabric_last_compl_time[plugin_idx] - libfabric_first_post_time[plugin_idx];
            Sum_T_all_rails[chunk_id] += Duration(netIbsend_time.count());

            // T_rail_skew: First rail completion → last rail completion
            netIbsend_time = libfabric_last_compl_time[plugin_idx] - libfabric_first_compl_time[plugin_idx];
            Sum_T_rail_skew[chunk_id] += Duration(netIbsend_time.count());

            // T_libfabric_to_done: time from plugin's last-rail completion (in
            // inc_req_completion, when req->state is set to COMPLETED) to the
            // moment NCCL's send proxy sees test() return done=true.
            // Use last_used_ch because with multiple channels, the plugin's
            // libfabric_last_compl_time slot gets overwritten by whichever
            // channel completes last (all channels share the same msg_seq_num
            // index space). The last channel's send_net_done_time is the
            // correct pairing.
            netIbsend_time = send_net_done_time[last_used_ch][chunk_id][p] - libfabric_last_compl_time[plugin_idx];
            Sum_T_libfabric_to_done[chunk_id] += Duration(netIbsend_time.count());
        }

        // After accumulating this iteration's plugin timestamps, clear the plugin
        // arrays and reset the test-window start so the NEXT iteration's plugin
        // writes (gated by "slot is empty") win, instead of being skipped because
        // the slot still holds this iteration's value. Without this reset,
        // plugin timestamps would freeze at iteration 1 while NCCL timestamps
        // (e.g. net_data_ready_time) advance every iteration, causing
        // Sum_T_nccl_to_libfabric to grow into a large negative number.
        plugin_test_start_seq_num = -1;
        for (int i = 0; i < MAX_PLUGIN_MSG_SEQ_NUM; ++i) {
            libfabric_first_post_time[i] = std::chrono::time_point<std::chrono::high_resolution_clock>();
            libfabric_first_compl_time[i] = std::chrono::time_point<std::chrono::high_resolution_clock>();
            libfabric_last_compl_time[i] = std::chrono::time_point<std::chrono::high_resolution_clock>();
        }

        for (size_t channelid = 0; channelid < gauge_channels_num; channelid++) {
            for (size_t i = 0; i < GAUGE_N_MESSAGES; ++i) {
                netIbsend_time = net_first_post_time[channelid][i][p] - net_first_post_time[channelid][0][p];
                Sum_T9_1[channelid][i] += Duration(netIbsend_time.count());
            }
        }

        for (size_t channelid = 0; channelid < gauge_channels_num; channelid++) {
            for (size_t i = 0; i < GAUGE_N_MESSAGES; ++i) {
                netIbsend_time = netIsend_last_chunk_time[channelid][i][p] - netIsend_first_chunk_time[channelid][i][p];
                Sum_T10_1[channelid][i] += Duration(netIbsend_time.count());
            }
        }

        // Use last/first actually-used channel so we don't read uninitialized
        // timestamps when NCCL didn't pick channel 0 / channel N-1.
        post_compl_time = send_net_done_time[last_used_ch][MAX_GAUGE_CHUNKS - 1][p] - net_first_post_time[first_used_ch][0][p];
        Sum_net_post_compl_time += Duration(post_compl_time.count());
    } else {
        Sum_T_total += Duration(nccl_func_time.count());
        int p = send_peer_idx;
        netIrecv_netIsend_time = netIsend_first_chunk_time[first_used_ch][0][p] - netIrecv_last_chunk_time[first_used_ch][GAUGE_N_MESSAGES-1][p];
        Sum_netIrecv_netIsend_time += Duration(netIrecv_netIsend_time.count());

        netIrecv_total_time = netIrecv_last_chunk_time[first_used_ch][GAUGE_N_MESSAGES-1][p] - netIrecv_first_chunk_time[first_used_ch][0][p];
        Sum_netIrecv_total_time[0] += Duration(netIrecv_total_time.count());

        for (size_t channelid = 0; channelid < gauge_channels_num; channelid++) {
            for (size_t i = 0; i < std::min(static_cast<size_t>(gauge_chunk_num), static_cast<size_t>(N_CHUNKS)); ++i) {
                for (int k = 0; k < 1; ++k) {
                    netIbrecv_time = net_Recv_posted_time[channelid][i][k] - net_Recv_posted_time[channelid][0][k];
                    Sum_T0[channelid][i][k] += Duration(netIbrecv_time.count());
                    netIbrecv_time = net_Recv_polled_time[channelid][i][k] - net_Recv_posted_time[channelid][i][k];
                    Sum_T5[channelid][i][k] += Duration(netIbrecv_time.count());
                    netIbrecv_time = net_Recv_done_time[channelid][i][k] - net_Recv_polled_time[channelid][i][k];
                    Sum_T6[channelid][i][k] += Duration(netIbrecv_time.count());
                }
            }
        }

        size_t channelid = gauge_channels_num - 1;
        netIbrecv_time = net_Recv_posted_time[channelid][MAX_GAUGE_CHUNKS - 1][0] - net_Recv_posted_time[channelid][0][0];
        Sum_T0[channelid][MAX_GAUGE_CHUNKS - 1][0] += Duration(netIbrecv_time.count());
        netIbrecv_time = net_Recv_polled_time[channelid][MAX_GAUGE_CHUNKS - 1][0] - net_Recv_posted_time[channelid][MAX_GAUGE_CHUNKS - 1][0];
        Sum_T5[channelid][MAX_GAUGE_CHUNKS - 1][0] += Duration(netIbrecv_time.count());
        netIbrecv_time = net_Recv_done_time[channelid][MAX_GAUGE_CHUNKS - 1][0] - net_Recv_polled_time[channelid][MAX_GAUGE_CHUNKS - 1][0];
        Sum_T6[channelid][MAX_GAUGE_CHUNKS - 1][0] += Duration(netIbrecv_time.count());

        for (size_t i = 0; i < GAUGE_N_MESSAGES; ++i) {
            for (size_t channelid = 0; channelid < gauge_channels_num; channelid++) {
                std::chrono::duration<float, std::milli> netIrecv_time = netIrecv_last_chunk_time[channelid][i][p] - netIrecv_first_chunk_time[channelid][0][p];
                Sum_netIrecv_time[channelid][i] += Duration(netIrecv_time.count());
            }
        }   
    } 
}

void printTimingMetrics() {

    int denom = (g_kept_iters > 0) ? static_cast<int>(g_kept_iters) : 1;

    // Broadcast the profiling variables from rank 0
    MPICHECK(MPI_Bcast(&gauge_chunk_num, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&gauge_channels_num, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&gauge_protocol, 1, MPI_INT, 0, MPI_COMM_WORLD));

    long long gauge_pp_chunk_size = (gauge_chunk_num * gauge_channels_num > 0) ? 1024LL * atoi(env_gauge_size_var) / (gauge_chunk_num * gauge_channels_num) : 0;

    const char* print_protocol;
    switch(gauge_protocol) {
        case 0:
            print_protocol = "LL";
            break;
        case 1:
            print_protocol = "LL128";
            break;
        case 2:
            print_protocol = "simple";
            break;
        default:
            print_protocol = "unknown";
            break;
    }

    if (myRank == 0) {
        printf("INFO: heo(%s)_mode(%s)_message size(%s)_nchannels(%d)_nthreads(%s)_nmessages(%d)_chunksize(%lld)_protocol(%s)_send-d(%d)_recv-d(%d)_iteration(%d)\n",
               env_gauge_heo_var, env_gauge_mode_var, env_gauge_size_var, gauge_channels_num,
               env_gauge_nthreads_var, GAUGE_N_MESSAGES, gauge_pp_chunk_size, print_protocol, send_gauge_d, recv_gauge_d, denom);
        // Diagnostic: which channels NCCL actually picked. Under a fixed message
        // size every iter picks the same channels, so this mask reflects all iters.
        {
            int channels_used = __builtin_popcountll(gauge_channels_used_mask);
            printf("-- channels actually used: count=%d, mask=0x%lx (out of %d configured)\n",
                   channels_used, (unsigned long)gauge_channels_used_mask, gauge_channels_num);
        }
        printf("-- [T total] nccl pping elapsed time by clock: %.6f ms\n", Sum_T_total.count() / denom);
        printf("-- [T1] proxy init time: %.6f ms\n", Sum_T1.count() / denom);
        printf("-- [T2] func start to first netIsend of first message time: %.6f ms\n", Sum_T2.count() / denom);
        printf("-- [T8] nccl last ncclIsend of last message done to func end time: %.6f ms\n", Sum_T8.count() / denom);
        
        for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); channelid++) {
            if (!channel_was_used(channelid)) continue;   // skip channels NCCL didn't actually pick
            for (size_t i = 0; i < std::min(static_cast<size_t>(gauge_chunk_num), static_cast<size_t>(N_CHUNKS)); ++i) {
                for (int k = 0; k < GAUGE_SUBGROUP_SIZE; ++k) {
                    printf("-- [T0] chunk %zu posted from first chunk posted time (r0 to r%d) in channel %zu: %.6f ms\n", i, k+1, channelid, Sum_T0[channelid][i][k].count() / denom);
                    printf("-- [T3] chunk %zu posted to data ready time (r0 to r%d) in channel %zu: %.6f ms\n", i, k+1, channelid, Sum_T3[channelid][i][k].count() / denom);
                    printf("-- [T4] chunk %zu data ready to transmitted time (r0 to r%d) in channel %zu: %.6f ms\n", i, k+1, channelid, Sum_T4[channelid][i][k].count() / denom);
                    printf("-- [T7] chunk %zu transmitted to done time (r0 to r%d) in channel %zu: %.6f ms\n", i, k+1, channelid, Sum_T7[channelid][i][k].count() / denom);
                }
            }
        }

        // T_dataready_to_done: chunk-0 data-ready  →  last-chunk done.
        // Spans the entire data-transfer phase of one ncclSend (across however
        // many chunks NCCL produced for this message). This is independent of
        // chunk count: even when the message is split, this captures the
        // total wire-side time from the first byte being ready until the last
        // chunk's completion is observed by NCCL.
        for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); channelid++) {
            if (!channel_was_used(channelid)) continue;
            printf("-- [T_dataready_to_done] chunk 0 data ready → last chunk done (r0 to r1) in channel %zu: %.6f ms\n",
                   channelid, Sum_T_dataready_to_done[channelid].count() / denom);
        }

        // Print the sentinel slot (Sum_T*[ch][MAX_GAUGE_CHUNKS-1][0]) ONLY when
        // the message was split into more chunks than the per-chunk arrays can
        // hold (chunk_num >= MAX_GAUGE_CHUNKS). The sentinel is the "always
        // overwritten by the latest chunk" slot, used to capture the last
        // chunk's timing when individual chunk slots overflowed. For small
        // chunk counts this duplicates chunk 0 / chunk_num-1 with a tiny ns
        // offset, so we suppress it to keep the output clean.
        if (gauge_chunk_num >= MAX_GAUGE_CHUNKS) {
            for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); channelid++) {
                if (!channel_was_used(channelid)) continue;
                printf("-- [T0] chunk %d posted from first chunk posted time (r0 to r%d) in channel %zu: %.6f ms\n", gauge_chunk_num-1, 1, channelid, Sum_T0[channelid][MAX_GAUGE_CHUNKS - 1][0].count() / denom);
                printf("-- [T3] chunk %d posted to data ready time (r0 to r%d) in channel %zu: %.6f ms\n", gauge_chunk_num-1, 1, channelid, Sum_T3[channelid][MAX_GAUGE_CHUNKS - 1][0].count() / denom);
                printf("-- [T4] chunk %d data ready to transmitted time (r0 to r%d) in channel %zu: %.6f ms\n", gauge_chunk_num-1, 1, channelid, Sum_T4[channelid][MAX_GAUGE_CHUNKS - 1][0].count() / denom);
                printf("-- [T7] chunk %d transmitted to done time (r0 to r%d) in channel %zu: %.6f ms\n", gauge_chunk_num-1, 1, channelid, Sum_T7[channelid][MAX_GAUGE_CHUNKS - 1][0].count() / denom);
            }
        }

        // Print new plugin timing metrics (per-chunk analysis)
        printf("\n--- Plugin Timing Metrics (NCCL → Libfabric overhead, per chunk) ---\n");
        printf("-- chunk number: %d (NCCL view: gauge_chunk_num=%d, plugin view: distinct msg_seq_num written=%d)\n",
               g_plugin_real_chunk_count, gauge_chunk_num, g_plugin_real_chunk_count);
        for (size_t chunk_id = 0; chunk_id < std::min(gauge_chunk_num, MAX_GAUGE_CHUNKS); chunk_id++) {
            printf("-- [T_nccl_to_libfabric] chunk %zu ncclSend launch (CPU) → libfabric first rail post: %.6f ms\n",
                   chunk_id, Sum_T_nccl_to_libfabric[chunk_id].count() / denom);
            printf("-- [T_kernel_to_libfabric] chunk %zu ncclSend kernel start (GPU) → libfabric first rail post: %.6f ms\n",
                   chunk_id, Sum_T_kernel_to_libfabric[chunk_id].count() / denom);
            printf("-- [T_first_rail] chunk %zu first rail post → first rail completion: %.6f ms\n",
                   chunk_id, Sum_T_first_rail[chunk_id].count() / denom);
            printf("-- [T_all_rails] chunk %zu first rail post → all rails completion: %.6f ms\n",
                   chunk_id, Sum_T_all_rails[chunk_id].count() / denom);
            printf("-- [T_rail_skew] chunk %zu first rail compl → last rail compl (rail imbalance): %.6f ms\n",
                   chunk_id, Sum_T_rail_skew[chunk_id].count() / denom);
            printf("-- [T_libfabric_to_done] chunk %zu last rail compl → NCCL test() done: %.6f ms\n",
                   chunk_id, Sum_T_libfabric_to_done[chunk_id].count() / denom);
        }
        printf("\n");

        for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); channelid++) {
            if (!channel_was_used(channelid)) continue;
            for (size_t i = 0; i < GAUGE_N_MESSAGES; ++i) {
                printf("-- [T9-1] message %zu inter-message post gap (msg[i] first post - msg[0] first post) in channel %zu: %.6f ms\n", i, channelid, Sum_T9_1[channelid][i].count() / denom);
            }
        }

        if (gauge_chunk_num > 1) {
            for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); channelid++) {
                if (!channel_was_used(channelid)) continue;
                for (size_t i = 0; i < GAUGE_N_MESSAGES; ++i) {
                    printf("-- [T10-1] message %zu intra-message chunk span (last chunk isend - first chunk isend) in channel %zu: %.6f ms\n", i, channelid, Sum_T10_1[channelid][i].count() / denom);
                }
            }
        }

        printf("--posted to completed send time (nchunks per channel %d) in all channels: %.6f ms\n", gauge_chunk_num, Sum_net_post_compl_time.count() / denom);

    } else {
        printf("INFO: heo(%s)_mode(%s)_message size(%s)_nchannels(%d)_nthreads(%s)_nmessages(%d)_chunksize(%lld)_protocol(%s)_send-d(%d)_recv-d(%d)_iteration(%d)\n",
               env_gauge_heo_var, env_gauge_mode_var, env_gauge_size_var, gauge_channels_num,
               env_gauge_nthreads_var, GAUGE_N_MESSAGES, gauge_pp_chunk_size, print_protocol, send_gauge_d, recv_gauge_d, denom);    
        printf("--nccl pping elapsed time by clock: %.6f ms\n", Sum_T_total.count() / denom);
        printf("--nccl last ncclIrecv of last message to first netIsend of first message time: %.6f ms\n", Sum_netIrecv_netIsend_time.count() / denom);
        
        printf("--nccl total ncclIrecv time: %.6f ms\n", Sum_netIrecv_total_time[0].count() / denom); 

        for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); channelid++) {
            if (!channel_was_used(channelid)) continue;
            for (size_t i = 0; i < std::min(static_cast<size_t>(gauge_chunk_num), static_cast<size_t>(N_CHUNKS)); ++i) {
                for (int k = 0; k < 1; ++k) {
                    printf("-- [T0] chunk %zu posted (r0 to r%d) in channel %zu: %.6f ms\n", i, k+1, channelid, Sum_T0[channelid][i][k].count() / denom);
                    printf("-- [T5] chunk %zu orc_poll_comm (r0 to r%d) in channel %zu: %.6f ms\n", i, k+1, channelid, Sum_T5[channelid][i][k].count() / denom);
                    printf("-- [T6] chunk %zu orc_poll_sync (r0 to r%d) in channel %zu: %.6f ms\n", i, k+1, channelid, Sum_T6[channelid][i][k].count() / denom);
                }
            }
        }

        size_t channelid = gauge_channels_num - 1;
        printf("-- [T0] chunk %d polled (r0 to r%d) in channel %zu: %.6f ms\n", gauge_chunk_num-1, 1, channelid, Sum_T0[channelid][MAX_GAUGE_CHUNKS - 1][0].count() / denom);
        printf("-- [T5] chunk %d orc_poll_comm (r0 to r%d) in channel %zu: %.6f ms\n", gauge_chunk_num-1, 1, channelid, Sum_T5[channelid][MAX_GAUGE_CHUNKS - 1][0].count() / denom);
        printf("-- [T6] chunk %d orc_poll_sync (r0 to r%d) in channel %zu: %.6f ms\n", gauge_chunk_num-1, 1, channelid, Sum_T6[channelid][MAX_GAUGE_CHUNKS - 1][0].count() / denom);

        for (size_t i = 0; i < GAUGE_N_MESSAGES; ++i) {
            for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); channelid++) {
                printf("--message %zu netIrecv time in channel %zu: %.6f ms\n", i, channelid, Sum_netIrecv_time[channelid][i].count() / denom); 
            }
        }   
    } 
}

int main(int argc, char* argv[]) {

    // Initialize timing arrays
    initializeTimingArrays();

    // Initialize time points
    for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
        for (int j = 0; j < MAX_GAUGE_CHUNKS; ++j) {
            for (int k = 0; k < MAX_PEERS; ++k) {
                net_transmitted_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
                send_net_done_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
                recv_net_done_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
                net_post_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
                net_data_ready_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
            }
        }
    }

    for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
        for (int j = 0; j < N_MESSAGES; ++j) {
            for (int k = 0; k < MAX_PEERS; ++k) {
                net_first_post_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
                netIsend_first_chunk_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
                netIrecv_first_chunk_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
                netIsend_last_chunk_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
                netIrecv_last_chunk_time[i][j][k] = std::chrono::time_point<std::chrono::high_resolution_clock>();
            }
        }
    }

    // Get environment variables
    env_gauge_heo_var = getenv("GAUGE_HEO");
    env_gauge_mode_var = getenv("GAUGE_MODE");
    env_gauge_iteration_var = getenv("GAUGE_ITERATION");
    const char* env_gauge_output_dir_var = getenv("GAUGE_OUT_DIRE");
    env_gauge_nthreads_var = getenv("NCCL_NTHREADS");

    // Check if environment variables are set
    if (!env_gauge_heo_var) env_gauge_heo_var = "unknown_gauge_heo";
    if (!env_gauge_mode_var) env_gauge_mode_var = "unknown_gauge_mode";
    if (!env_gauge_iteration_var) env_gauge_iteration_var = "unknown_gauge_iteration";
    if (!env_gauge_nthreads_var) env_gauge_nthreads_var = "unknown_gauge_nthreads";  
    if (!env_gauge_output_dir_var) {
        env_gauge_output_dir_var = "unknown_gauge_output_dir";
        printf("unknown gauge output dir\n");
    }

    long long size = 1;  // Default size (in number of floats)
    env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
    if (env_gauge_size_var != nullptr) {
        // GAUGE_MESSAGE_SIZE is now in BYTES; convert to number of floats.
        // For sub-4-byte sizes we still allocate at least 1 float; the kernel will
        // use only the requested byte count.
        long long bytes = atoll(env_gauge_size_var);
        size = (bytes + 3) / 4;  // round up to floats
        if (size < 1) size = 1;
    }

    gauge_iterations = atoi(env_gauge_iteration_var);

    if (argc >= 3) {
        send_gauge_d = atoi(argv[1]);
        recv_gauge_d = atoi(argv[2]);
    }

    int localRank = 0;

    // Set the device scheduling flag before creating a device context
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device flags: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


    // The trailing "vary-XXX" tag in the output filename is selected by the
    // GAUGE_OUT_SUFFIX env var (e.g. "msg", "chunk"). Defaults to "msg" for
    // backwards compatibility with older scripts.
    const char* env_gauge_out_suffix = getenv("GAUGE_OUT_SUFFIX");
    if (!env_gauge_out_suffix || env_gauge_out_suffix[0] == '\0') {
        env_gauge_out_suffix = "msg";
    }

    char filename[256];
    if (myRank < 2) {
        sprintf(filename, "%s/nccl_pping_%s_r-%d-vary-%s.out",
                env_gauge_output_dir_var, env_gauge_heo_var,
                myRank, env_gauge_out_suffix);

        FILE *file = freopen(filename, "a", stdout);
        if (file == NULL) {
            perror("freopen failed");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        setbuf(stdout, NULL);
        fflush(stdout);
    } else {
        freopen("/dev/null", "w", stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);  

    gauge_cpu_mhz = get_cpu_mhz();
    host_isend_gauge_d = 0;
    host_check_GPU_data_gauge_d = 0;
    host_recv_gauge_d = 0;
    gauge_send_message_itr = 0;
    gauge_recv_message_itr = 0;

    // Calculate localRank based on hostname
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

    // Get NCCL unique ID at rank 0 and broadcast it to all others
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Pick a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, GAUGE_N_MESSAGES * size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, GAUGE_N_MESSAGES * size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    // Create CUDA events for measuring GPU-side kernel-launch timestamps
    CUDACHECK(cudaEventCreate(&calib_event));
    CUDACHECK(cudaEventCreate(&kernel_start_event));

    int localDev = -1;
    cudaGetDevice(&localDev);
    char host[128]; gethostname(host, sizeof(host));
    printf("[Rank %d @ %s] cudaDevice=%d (CUDA_VISIBLE_DEVICES=%s)\n",
        myRank, host, localDev, getenv("CUDA_VISIBLE_DEVICES"));
    fflush(stdout);

    // Initialize NCCL
    proxy_init_time_start = std::chrono::high_resolution_clock::now();
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
    proxy_init_time_end = std::chrono::high_resolution_clock::now();

    // Communicating using NCCL - P2P
    int recvPeer = (myRank-1+nRanks) % nRanks;
    int sendPeer = (myRank+1) % nRanks;
    send_peer_idx = sendPeer;

    // Warm up
    CUDACHECK(cudaStreamSynchronize(s));

    for (int i = 0 ; i < WARMUP_ITERATION; i++) {
        if (myRank == 0) {
            NCCLCHECK(ncclSend((const void*)((float*)sendbuff), size, ncclFloat, sendPeer, comm, s));
        } else {
            NCCLCHECK(ncclRecv((void*)((float*)recvbuff), size, ncclFloat, recvPeer, comm, s));
        }
    }
    CUDACHECK(cudaStreamSynchronize(s));

    for (int i = 0 ; i < WARMUP_ITERATION; i++) {
        if (myRank == 1) {
            NCCLCHECK(ncclSend((const void*)((float*)sendbuff), size, ncclFloat, sendPeer, comm, s));
        } else {
            NCCLCHECK(ncclRecv((void*)((float*)recvbuff), size, ncclFloat, recvPeer, comm, s));
        }
    }
    CUDACHECK(cudaStreamSynchronize(s));

    CUDACHECK(cudaStreamSynchronize(s));

    // === GPU/CPU clock calibration ===
    // Record an event on the now-idle stream and synchronize, then immediately
    // grab a CPU timestamp. This anchors GPU event-time 0 (== calib_event)
    // to the CPU timestamp captured here. Subsequent kernel_start_event times
    // (measured via cudaEventElapsedTime against calib_event) can be added to
    // calib_cpu_time to get a CPU-clock-equivalent timestamp.
    CUDACHECK(cudaEventRecord(calib_event, s));
    CUDACHECK(cudaEventSynchronize(calib_event));
    calib_cpu_time = std::chrono::high_resolution_clock::now();

    // Clear plugin timestamp arrays AFTER warmup, BEFORE actual measurement.
    // The plugin writes to a slot only if it is empty (epoch == 0), so clearing here
    // ensures the plugin's first write during the test wins (not overwritten by retries
    // or msg_seq_num wraparound).
    // Also reset plugin_test_start_seq_num so the plugin's first post during the test
    // captures its msg_seq_num as the start of the test window.
    plugin_test_start_seq_num = -1;
    for (int i = 0; i < MAX_PLUGIN_MSG_SEQ_NUM; ++i) {
        libfabric_first_post_time[i] = std::chrono::time_point<std::chrono::high_resolution_clock>();
        libfabric_first_compl_time[i] = std::chrono::time_point<std::chrono::high_resolution_clock>();
        libfabric_last_compl_time[i] = std::chrono::time_point<std::chrono::high_resolution_clock>();
    }

    for (int gauge_iter=0; gauge_iter < gauge_iterations; gauge_iter++) {
        if (myRank == 0) {
            gauge_send_message_itr = 1;
            gauge_recv_message_itr = 0;
        }
        if (myRank == 1) {
            gauge_send_message_itr = 0;
            gauge_recv_message_itr = 1;
        }
        if (myRank == 0) usleep(10);

        nccl_func_start_time = std::chrono::high_resolution_clock::now();

        host_isend_gauge_d = send_gauge_d;
        host_recv_gauge_d = recv_gauge_d;

        // Record an event on the stream right before the first ncclSend/ncclRecv.
        // The GPU will execute this command in stream order, so its timestamp
        // captures the moment NCCL's send/recv kernel is about to launch.
        CUDACHECK(cudaEventRecord(kernel_start_event, s));

        for (int i = 0; i < GAUGE_N_MESSAGES; i++) {
            if (myRank == 0) {
                NCCLCHECK(ncclSend((const void*)((float*)sendbuff + i * size), size, ncclFloat, sendPeer, comm, s));
            } else {
                NCCLCHECK(ncclRecv((void*)((float*)recvbuff + i * size), size, ncclFloat, recvPeer, comm, s));
            }
            CUDACHECK(cudaStreamSynchronize(s));
            MPI_Barrier(MPI_COMM_WORLD);
            if (myRank == 0) gauge_send_message_itr += 1;
            if (myRank == 1) gauge_recv_message_itr += 1;
        }

        CUDACHECK(cudaStreamSynchronize(s));
        MPI_Barrier(MPI_COMM_WORLD);

        nccl_func_end_time = std::chrono::high_resolution_clock::now();

        // Translate kernel_start_event (GPU clock) into CPU clock using calibration.
        // After cudaStreamSynchronize above, kernel_start_event is guaranteed to
        // have completed, so cudaEventElapsedTime is safe to call here.
        float gpu_delta_ms = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&gpu_delta_ms, calib_event, kernel_start_event));
        kernel_launch_cpu_time = calib_cpu_time +
            std::chrono::nanoseconds((long long)(gpu_delta_ms * 1e6));

        host_isend_gauge_d = 0;
        host_recv_gauge_d = 0;
        if (myRank == 0) {
            gauge_send_message_itr = 0;
            gauge_recv_message_itr = 1;
        }
        if (myRank == 1) {
            gauge_send_message_itr = 1;
            gauge_recv_message_itr = 0;
        }

        if (myRank == 1) {
            NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, sendPeer, comm, s));
        } else {
            NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, recvPeer, comm, s));
        }
        CUDACHECK(cudaStreamSynchronize(s));

        // nccl_func_time = nccl_func_end_time - nccl_func_start_time;
        // // --- Outlier filtering based on running mean of kept iterations ---
        // // Compute running mean from previous kept iters using Sum_T_total (which is only
        // // incremented inside AccumulateTimingMetrics on kept iterations).
        // double running_mean_ms = (g_kept_iters > 0) ? (Sum_T_total.count() / static_cast<double>(g_kept_iters))
        //                                             : 0.0;
        // double this_iter_ms    = nccl_func_time.count();

        // // Keep first kWarmupKeep iterations to stabilize the mean.
        // bool keep = (g_kept_iters < static_cast<size_t>(kWarmupKeep))
        //             || (this_iter_ms <= kOutlierFactor * running_mean_ms);

        // if (keep) {
        //     AccumulateTimingMetrics();  // this will add to Sum_T_total and other sums
        //     ++g_kept_iters;             // count only accepted iterations
        // } else {
        //     // Skip accumulation entirely for this iteration.
        //     // (Optionally log or count filtered iterations.)
        //     // fprintf(stderr, "[Rank %d] Filtered iteration: %.3f ms > %.3f ms (mean * %.2f)\n",
        //     //         myRank, this_iter_ms, running_mean_ms * kOutlierFactor, kOutlierFactor);
        // }

        nccl_func_time = nccl_func_end_time - nccl_func_start_time;
        double this_iter_ms = nccl_func_time.count();

        bool keep = false;
        if (!g_baseline_ready) {
            // Always keep the first kWarmupKeep iterations; update running min
            keep = true;
            AccumulateTimingMetrics();
            ++g_kept_iters;

            if (this_iter_ms < g_baseline_min_ms) g_baseline_min_ms = this_iter_ms;
            ++g_warmup_count;

            if (g_warmup_count >= kWarmupKeep) {
                g_baseline_ms    = g_baseline_min_ms;  // baseline = min of warmups
                g_baseline_ready = true;
            }
        } else {
            // Filter using factor * baseline
            const double threshold = kOutlierFactor * g_baseline_ms;
            keep = (this_iter_ms <= threshold);
            if (keep) {
                AccumulateTimingMetrics();
                ++g_kept_iters;

                // Capture profiling variables after first successful iteration
                // At this point, proxy has definitely run and set the variables
            }
        }
    }

    // Complete NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    printTimingMetrics();

    // Free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    // Finalize NCCL
    ncclCommDestroy(comm);

    // Finalize MPI
    MPICHECK(MPI_Finalize());

    printf("[MPI Rank %d] Success \n", myRank);
    return 0;
}