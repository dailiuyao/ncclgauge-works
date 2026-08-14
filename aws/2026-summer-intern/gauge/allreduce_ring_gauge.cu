#include <stdio.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <thread>
#include "get_clock.h"
#include <algorithm>
#include <cmath>

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

#ifndef GAUGE_N_MESSAGES
#define GAUGE_N_MESSAGES 1
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

int gauge_chunk_num;
int gauge_channels_num;
int gauge_protocol;
uint64_t gauge_channels_used_mask = 0;

// K = ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS. Derived once from NCCL_PROTO.
//   Simple  → CHUNKSTEPS=4, SLICESTEPS=2 → K=2
//   LL / LL128 → enqueue forces both to 1 (see enqueue.cc: gate on protocol==SIMPLE
//   && algorithm==RING) → K=1
// Overridable via GAUGE_RING_K for edge cases.
int g_ring_k = 2;

static inline bool channel_was_used(size_t ch) {
    if (gauge_channels_used_mask == 0) return true;
    return (gauge_channels_used_mask & (uint64_t(1) << ch)) != 0;
}

int gauge_iterations;

const char* env_gauge_heo_var;
const char* env_gauge_mode_var;
const char* env_gauge_size_var;
const char* env_gauge_nthreads_var;
const char* env_gauge_iteration_var;
const char* env_split_mask_var;

#define DEFAULT_D 0
#define GAUGE_SUBGROUP_SIZE MAX_PEERS

int send_gauge_d = DEFAULT_D;
int recv_gauge_d = DEFAULT_D;

static size_t g_kept_iters = 0;

using Duration = std::chrono::duration<double, std::milli>;
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
Duration nccl_func_time;

// Accumulated timing metrics
Duration Sum_T_total(0.0);
Duration Sum_PROXY_INIT(0.0);   // proxy init
Duration Sum_FSTART_TO_POST(0.0);   // func start → first netIsend
Duration Sum_DONE_TO_FEND(0.0);   // last done → func end

// Per-channel per-chunk metrics
Duration Sum_POST_GAP[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE];  // chunk post gap
Duration Sum_POST_TO_DR[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE];  // posted → data ready
// POST_TO_DR split: POST_TO_RECVDONE = post → recv-done (waiting for incoming chunk),
//                   RECVDONE_TO_DR   = recv-done → data ready (GPU reduce).
// Phase 0 (directSend, no recv dep) leaves the split unrecorded → printed as N/A.
Duration Sum_POST_TO_RECVDONE[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE];  // posted → recv done
Duration Sum_RECVDONE_TO_DR[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE];  // recv done → data ready
Duration Sum_DR_TO_TX[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE];  // data ready → transmitted
Duration Sum_TX_TO_DONE[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE];  // transmitted → done
Duration Sum_T_dataready_to_done[MAX_GAUGE_CHANNELS][MAX_PEERS];      // data ready → last chunk done
// Unified per-(channel, chunk, peer) valid-iter counter. All send-side metrics
// (POST_TO_DR/DR_TO_TX/TX_TO_DONE/ABS_TX/ABS_TX_DONE + rails + ctrl arrival) are
// accumulated atomically per iter — either every metric is recorded (iter valid) or none.
// For phase>0 chunks, POST_TO_RECVDONE / RECVDONE_TO_DR are gated on the SAME condition
// (so their denom equals Sum_valid_iters and the two means sum to mean(POST_TO_DR) exactly).
// For phase=0 chunks (directSend, no recv dep), the split is not accumulated
// (Sum_POST_TO_RECVDONE=Sum_RECVDONE_TO_DR=0 → print N/A). Sum_valid_iters still counts
// these iters because POST_TO_DR/DR_TO_TX/TX_TO_DONE/... are all valid there.
long long Sum_valid_iters[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE] = {{{0}}};
// Whether this (ch, i, k) is a phase>0 chunk (has the POST_TO_RECVDONE /
// RECVDONE_TO_DR split). Set on first accumulation; used at print time to
// decide N/A vs numeric.
bool has_recv_split[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE] = {{{false}}};

// Absolute send timestamp (relative to nccl_func_start_time on this rank) — enables cross-rank alignment
Duration Sum_ABS_TX[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE];   // net_transmitted_time[i] - func_start
Duration Sum_ABS_TX_DONE[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE]; // send_net_done_time[i] - func_start

// Recv-side per-channel per-chunk per-peer metrics
Duration Sum_R0[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE];    // Irecv post gap: posted[i] - posted[0]
Duration Sum_R_wait[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE]; // Irecv posted → Irecv done
Duration Sum_ABS_RX_POST[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE]; // Irecv posted - func_start
Duration Sum_ABS_RX_DONE[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE]; // Irecv done   - func_start
Duration Sum_R_posted_to_done[MAX_GAUGE_CHANNELS][MAX_PEERS];  // first Irecv posted → last chunk recv done
long long Sum_valid_recv_iters[MAX_GAUGE_CHANNELS][MAX_GAUGE_CHUNKS][GAUGE_SUBGROUP_SIZE] = {{{0}}}; // recv-side unified denom

// Plugin (aws-ofi-nccl) per-rail per-chunk timing, split by (channel, peer).
// The plugin indexes its slots by [s_comm_id][msg_seq_num][rail_id], and the
// gauge joins that with the (channel, peer) → s_comm map published by NCCL
// (nccl_channel_scomm) + plugin_scomm_by_id. This way nchannels>1 does not
// overwrite slots.
//   RAIL_START = post_fi_send  - nccl_func_start_time
//   RAIL_END   = per_rail_compl - nccl_func_start_time
//   RAIL_DUR   = RAIL_END - RAIL_START
// Rails: only accumulated when the same (ch, i, k) send-side gate passed this
// iter AND both post/complete stamps exist for the rail. Per-rail counter is
// kept because some iters may legitimately skip a rail (eager path etc.).
Duration  Sum_RAIL_START[MAX_GAUGE_CHANNELS][MAX_PEERS][MAX_GAUGE_CHUNKS][MAX_PLUGIN_RAILS];
Duration  Sum_RAIL_END  [MAX_GAUGE_CHANNELS][MAX_PEERS][MAX_GAUGE_CHUNKS][MAX_PLUGIN_RAILS];
Duration  Sum_RAIL_DUR  [MAX_GAUGE_CHANNELS][MAX_PEERS][MAX_GAUGE_CHUNKS][MAX_PLUGIN_RAILS];
long long Sum_RAIL_iters[MAX_GAUGE_CHANNELS][MAX_PEERS][MAX_GAUGE_CHUNKS][MAX_PLUGIN_RAILS] = {{{{0}}}};

// Ctrl-message arrival at sender, in this rank's func_start frame.
// Same "send-gate + stamp-present" atomicity as rails.
Duration  Sum_CTRL_ARRIVAL[MAX_GAUGE_CHANNELS][MAX_PEERS][MAX_GAUGE_CHUNKS];
long long Sum_CTRL_ARRIVAL_iters[MAX_GAUGE_CHANNELS][MAX_PEERS][MAX_GAUGE_CHUNKS] = {{{0}}};

// ---- Per-iter sample storage for percentile / distribution stats ----
// Independent, smaller dims than MAX_GAUGE_CHANNELS/CHUNKS so gauge memory
// stays bounded (float ms values; sub-us precision is enough for stats).
// Samples beyond STATS_MAX_* bounds are silently dropped from the stats
// stream — the Sum_/Sum_valid_iters means still see them.
#define STATS_MAX_CHANNELS 16
#define STATS_MAX_CHUNKS   64
#define STATS_MAX_PEERS    32
#define STATS_MAX_ITERS    128
#define STATS_MAX_RAILS    MAX_PLUGIN_RAILS

// Send-side per-iter samples (ms), one row per (ch, chunk, peer).
float S_POST_GAP        [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_POST_TO_DR        [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_DR_TO_TX        [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_POST_TO_RECVDONE        [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_RECVDONE_TO_DR        [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_TX_TO_DONE        [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_ABS_TX    [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_ABS_TX_DONE[STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
// Per-iter iterative O_net (ms), computed IN-ITER from same-source timestamps (see ComputeIterativeOnet).
// One value per (ch, chunk, peer) per iteration; final [O_NET] line = median over iters.
// This avoids the cross-iteration median-of-timestamps contamination of k_eff.
float S_ONET      [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
// o_net_eff = O_net / overlap_per_ch, overlap_per_ch = max(k_eff_onet / nch, 1); one division,
// k_eff_onet counted on the O_net segment [atx+W+L, +onet]. K_EFF stores that per-chunk k_eff.
float S_ONET_EFF  [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_KEFF      [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
// plugin/other split (per-iter, same-source): plugin = TX_TO_DONE - (rail0 UNION rail1);
// plugin_eff = plugin / overlap_per_ch ; other_eff = o_net_eff - plugin_eff.
float S_PLUGIN_EFF[STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_OTHER_EFF [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
int   S_onet_n    [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS] = {{{0}}};
int   S_send_n    [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS] = {{{0}}};
// Whether this (ch, i, k) recorded the POST_TO_RECVDONE / RECVDONE_TO_DR split in stats
// (mirrors has_recv_split gate).
bool  S_has_recv_split  [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS] = {{{false}}};

// Recv-side per-iter samples.
float S_R0        [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_R_wait    [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_ABS_RX_POST[STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
float S_ABS_RX_DONE[STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS][STATS_MAX_ITERS];
int   S_recv_n    [STATS_MAX_CHANNELS][STATS_MAX_CHUNKS][STATS_MAX_PEERS] = {{{0}}};

// Rail per-iter samples (indexed [ch][peer][chunk][rail][iter]).
float S_RAIL_START[STATS_MAX_CHANNELS][STATS_MAX_PEERS][STATS_MAX_CHUNKS][STATS_MAX_RAILS][STATS_MAX_ITERS];
float S_RAIL_END  [STATS_MAX_CHANNELS][STATS_MAX_PEERS][STATS_MAX_CHUNKS][STATS_MAX_RAILS][STATS_MAX_ITERS];
float S_RAIL_DUR  [STATS_MAX_CHANNELS][STATS_MAX_PEERS][STATS_MAX_CHUNKS][STATS_MAX_RAILS][STATS_MAX_ITERS];
int   S_rail_n    [STATS_MAX_CHANNELS][STATS_MAX_PEERS][STATS_MAX_CHUNKS][STATS_MAX_RAILS] = {{{{0}}}};

// Ctrl arrival per-iter samples.
float S_CTRL      [STATS_MAX_CHANNELS][STATS_MAX_PEERS][STATS_MAX_CHUNKS][STATS_MAX_ITERS];
int   S_ctrl_n    [STATS_MAX_CHANNELS][STATS_MAX_PEERS][STATS_MAX_CHUNKS] = {{{0}}};

// Percentile helper. Sorts a scratch copy of the samples in-place.
// p in [0,1]; linear interpolation between neighboring ranks.
static double stats_percentile(float* buf, int n, double p) {
    if (n <= 0) return 0.0;
    std::sort(buf, buf + n);
    if (n == 1) return buf[0];
    double idx = p * (n - 1);
    int lo = (int)idx;
    int hi = lo + 1 < n ? lo + 1 : lo;
    double frac = idx - lo;
    return (double)buf[lo] * (1.0 - frac) + (double)buf[hi] * frac;
}

// Prints one summary line: "med=X p95=Y min=Z max=W". Buf is scratch (mutated).
static void stats_print_summary(float* buf, int n) {
    if (n <= 0) { printf(" med=N/A p95=N/A min=N/A max=N/A"); return; }
    std::sort(buf, buf + n);
    double med = stats_percentile(buf, n, 0.5);
    double p95 = stats_percentile(buf, n, 0.95);
    double mn  = buf[0];
    double mx  = buf[n - 1];
    printf(" med=%.6f p95=%.6f min=%.6f max=%.6f", med, p95, mn, mx);
}

Duration Sum_net_post_compl_time(0.0);

// GPU/CPU calibration
static cudaEvent_t calib_event, kernel_start_event;
static TimePoint calib_cpu_time;
static TimePoint kernel_launch_cpu_time;

static void initializeTimingArrays() {
    for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
        for (int j = 0; j < MAX_GAUGE_CHUNKS; ++j) {
            for (int k = 0; k < MAX_PEERS; ++k) {
                Sum_POST_GAP[i][j][k] = Duration(0.0);
                Sum_POST_TO_DR[i][j][k] = Duration(0.0);
                Sum_POST_TO_RECVDONE[i][j][k] = Duration(0.0);
                Sum_RECVDONE_TO_DR[i][j][k] = Duration(0.0);
                Sum_DR_TO_TX[i][j][k] = Duration(0.0);
                Sum_TX_TO_DONE[i][j][k] = Duration(0.0);
                Sum_ABS_TX[i][j][k] = Duration(0.0);
                Sum_ABS_TX_DONE[i][j][k] = Duration(0.0);
                Sum_valid_iters[i][j][k] = 0;
                Sum_R0[i][j][k] = Duration(0.0);
                Sum_R_wait[i][j][k] = Duration(0.0);
                Sum_ABS_RX_POST[i][j][k] = Duration(0.0);
                Sum_ABS_RX_DONE[i][j][k] = Duration(0.0);
                Sum_valid_recv_iters[i][j][k] = 0;
            }
        }
        for (int k = 0; k < MAX_PEERS; ++k) {
            Sum_T_dataready_to_done[i][k] = Duration(0.0);
            Sum_R_posted_to_done[i][k] = Duration(0.0);
        }
    }
    for (int c = 0; c < MAX_GAUGE_CHANNELS; ++c) {
        for (int p = 0; p < MAX_PEERS; ++p) {
            for (int j = 0; j < MAX_GAUGE_CHUNKS; ++j) {
                for (int r = 0; r < MAX_PLUGIN_RAILS; ++r) {
                    Sum_RAIL_START[c][p][j][r] = Duration(0.0);
                    Sum_RAIL_END  [c][p][j][r] = Duration(0.0);
                    Sum_RAIL_DUR  [c][p][j][r] = Duration(0.0);
                    Sum_RAIL_iters[c][p][j][r] = 0;
                }
                Sum_CTRL_ARRIVAL[c][p][j] = Duration(0.0);
                Sum_CTRL_ARRIVAL_iters[c][p][j] = 0;
            }
        }
    }
}

static inline bool recv_peer_was_used(size_t ch, int peer) {
    return net_Recv_posted_time[ch][0][peer] != TimePoint();
}

static inline bool peer_was_used(size_t ch, int peer) {
    return net_post_time[ch][0][peer] != TimePoint();
}

// ============================================================================
// Per-iteration iterative O_net solver (recovered-bandwidth fixed point).
// Runs at the END of one physical iteration, using THIS iteration's raw
// net_transmitted_time / send_net_done_time (same source, same func_start
// frame -> chunk-to-chunk gaps are the true single-run gaps, NOT medians).
// This root-fixes the median-of-timestamps contamination of k_eff: median is
// applied only to the FINAL per-chunk O_net (over iterations), never to gaps.
//
// Model (ms units):
//   W_i(init) = (done_i - tx_i) - 2L         # TX_TO_DONE - 2L, O_net=0 assumption
//   window_i  = [atx_i, atx_i + W_i],  atx_i = tx_i - func_start
//   keff_i    = (sum_j overlap(i,j)) / W_i   # time-avg concurrency, ALL channels pooled
//   recBW_i   = (chunk_bytes / W_i) * keff_i
//   W_i      <- clip(W_i * recBW_i/BW, 1e-6, Wmax_i)   # Wmax_i = (done_i - tx_i) - 2L
//   iterate to fixed point;  O_net_i = (done_i - tx_i) - W_i - 2L
// This iteration's rail0 UNION rail1 interval length (ms) for one (channel,chunk,peer).
// Uses raw libfabric_post_fi_send_time / per_rail_compl_time via the scomm map (same as
// AccumulateTimingMetrics). Returns -1 if no rail present. union = sum(dur) - overlap.
static double _rail_union_ms_thisiter(size_t channelid, size_t i, int k) {
    if ((size_t)i >= static_cast<size_t>(MAX_PLUGIN_PROFILING_SLOTS)) return -1.0;
    void *target = nccl_channel_scomm[channelid][k].load(std::memory_order_acquire);
    if (target == nullptr) return -1.0;
    int cid = -1;
    for (int ii = 0; ii < MAX_PLUGIN_S_COMMS; ++ii) {
        if (plugin_scomm_by_id[ii].load(std::memory_order_acquire) == target) { cid = ii; break; }
    }
    if (cid < 0) return -1.0;
    double s[2], e[2]; int nr = 0;
    for (int rr = 0; rr < MAX_PLUGIN_RAILS && nr < 2; ++rr) {
        TimePoint post_r     = libfabric_post_fi_send_time [cid][i][rr];
        TimePoint rail_compl = libfabric_per_rail_compl_time[cid][i][rr];
        if (post_r.time_since_epoch().count() == 0) continue;
        if (rail_compl.time_since_epoch().count() == 0) continue;
        s[nr] = Duration(post_r     - nccl_func_start_time).count();
        e[nr] = Duration(rail_compl - nccl_func_start_time).count();
        ++nr;
    }
    if (nr == 0) return -1.0;
    if (nr == 1) return e[0] - s[0];
    double ov = (e[0] < e[1] ? e[0] : e[1]) - (s[0] > s[1] ? s[0] : s[1]);  // min(end)-max(start)
    if (ov < 0.0) ov = 0.0;
    return (e[0] - s[0]) + (e[1] - s[1]) - ov;                              // union length
}

static void ComputeIterativeOnet() {
    // ---- config-level constants ----
    // 2L: one-way latency doubled. Default 0.02492 ms (a=24.92us from pping); override via env.
    static double two_L_ms = -1.0;
    if (two_L_ms < 0.0) {
        const char* e = getenv("GAUGE_TWO_L_MS");
        two_L_ms = (e && atof(e) > 0.0) ? atof(e) : 0.02492;
    }
    // on-wire chunk bytes = msg_bytes * 2(n-1) / (n * nch * nchunks_per_channel)
    //   (verified numerically equal to the buffsize-based chunk_bytes convention).
    long long msg_bytes = 1;
    { const char* e = getenv("GAUGE_MESSAGE_SIZE"); if (e) msg_bytes = atoll(e); }
    int nch = gauge_channels_num;
    int ncpc = gauge_chunk_num;
    if (msg_bytes <= 0 || nch <= 0 || ncpc <= 0 || nRanks <= 1) return;
    double chunk_bytes = (double)msg_bytes * 2.0 * (nRanks - 1) / ((double)nRanks * nch * ncpc);
    // per-chunk on-wire proto factor: Simple = chunk_bytes (buffsize/4 already the send unit);
    // LL sends data+flag (x2 on wire relative to /8), LL128 15/16. The derivation above already
    // yields the SEND unit (matches notebook chunk_bytes for Simple). For LL/LL128 O_net~0 anyway.
    // NIC bandwidth pool (GB/s): dual EFA 48.75, single 24.375 (nch==1 & per-gpu<128KB).
    // Both overridable from the launch script: GAUGE_BW_DUAL_GBS / GAUGE_BW_SINGLE_GBS.
    static double bw_dual_GBs = -1.0, bw_single_GBs = -1.0, single_rail_thresh_B = -1.0;
    if (bw_dual_GBs < 0.0) {
        const char* e = getenv("GAUGE_BW_DUAL_GBS");   bw_dual_GBs   = (e && atof(e) > 0.0) ? atof(e) : 48.75;
        const char* s = getenv("GAUGE_BW_SINGLE_GBS"); bw_single_GBs = (s && atof(s) > 0.0) ? atof(s) : 24.375;
        // per-GPU byte threshold below which nch==1 uses a single EFA rail (default 128 KB).
        const char* t = getenv("GAUGE_SINGLE_RAIL_THRESH_KB");
        single_rail_thresh_B = ((t && atof(t) > 0.0) ? atof(t) : 128.0) * 1024.0;
    }
    double per_gpu = (double)msg_bytes / (double)(nch * nRanks);
    double BW = (((nch == 1) && (per_gpu <= single_rail_thresh_B)) ? bw_single_GBs : bw_dual_GBs) * 1e9 * 1e-3; // bytes per ms

    // ---- gather THIS iteration's chunks across ALL channels (pooled) ----
    // Flattened arrays; keep (ch, chunk, peer) index to write O_net back.
    static const int MAXN = STATS_MAX_CHANNELS * STATS_MAX_CHUNKS * STATS_MAX_PEERS;
    static double atx[MAXN], Wmax[MAXN], W[MAXN], span[MAXN];
    static int idx_ch[MAXN], idx_chunk[MAXN], idx_peer[MAXN];
    int N = 0;
    size_t chunk_end = std::min(static_cast<size_t>(gauge_chunk_num), static_cast<size_t>(MAX_GAUGE_CHUNKS));
    for (size_t ch = 0; ch < (size_t)gauge_channels_num && ch < (size_t)STATS_MAX_CHANNELS; ++ch) {
        if (!channel_was_used(ch)) continue;
        for (size_t i = 0; i < chunk_end && i < (size_t)STATS_MAX_CHUNKS; ++i) {
            for (int k = 0; k < MAX_PEERS && k < STATS_MAX_PEERS; ++k) {
                TimePoint tx = net_transmitted_time[ch][i][k];
                TimePoint dn = send_net_done_time  [ch][i][k];
                if (tx.time_since_epoch().count() == 0) continue;
                if (dn.time_since_epoch().count() == 0) continue;
                double sp = Duration(dn - tx).count();               // TX_TO_DONE (ms)
                double wm = sp - two_L_ms;                            // Wmax = TX_TO_DONE - 2L
                if (wm <= 0.0) continue;                              // no room for transfer window
                if (N >= MAXN) break;
                atx[N]  = Duration(tx - nccl_func_start_time).count();
                span[N] = sp; Wmax[N] = wm; W[N] = wm;
                idx_ch[N] = (int)ch; idx_chunk[N] = (int)i; idx_peer[N] = k;
                ++N;
            }
        }
    }
    if (N < 2) return;

    // ---- fixed-point iteration (recBW -> BW) ----
    const double TOL = 0.005; const int MAXIT = 500;
    for (int it = 0; it < MAXIT; ++it) {
        double maxrel = 0.0;
        // keff_i = sum_j overlap([atx_i,atx_i+W_i],[atx_j,atx_j+W_j]) / W_i
        for (int i = 0; i < N; ++i) {
            double ai = atx[i], bi = atx[i] + W[i], ov = 0.0;
            for (int j = 0; j < N; ++j) {
                double lo = ai > atx[j] ? ai : atx[j];
                double hj = atx[j] + W[j];
                double hi = bi < hj ? bi : hj;
                double d = hi - lo; if (d > 0.0) ov += d;
            }
            double keff = ov / (W[i] > 1e-9 ? W[i] : 1e-9);
            double recBW = (chunk_bytes / (W[i] > 1e-9 ? W[i] : 1e-9)) * keff; // bytes/ms
            double Wn = W[i] * (recBW / BW);
            if (Wn < 1e-6) Wn = 1e-6;
            if (Wn > Wmax[i]) Wn = Wmax[i];
            double rel = fabs(Wn - W[i]) / (W[i] > 1e-9 ? W[i] : 1e-9);
            if (rel > maxrel) maxrel = rel;
            span[i] = Wn;   // reuse span[] as next-W scratch (span no longer needed)
        }
        for (int i = 0; i < N; ++i) W[i] = span[i];
        if (maxrel < TOL) break;
    }

    // ---- per-chunk O_net = TX_TO_DONE - W - 2L ; then o_net_eff = O_net / overlap_per_ch ----
    // overlap_per_ch = max(k_eff_onet / nch, 1), where k_eff_onet is the time-avg concurrency
    // over the O_NET segment [os_i, os_i+onet_i], os_i = atx_i + W_i + L (L = 2L/2). One pass,
    // no iteration (the O_net values are already fixed from the converged W).
    const double L_ms = two_L_ms * 0.5;
    static double onet_arr[MAXN], os[MAXN];
    for (int i = 0; i < N; ++i) {
        double sp = Duration(send_net_done_time[idx_ch[i]][idx_chunk[i]][idx_peer[i]]
                             - net_transmitted_time[idx_ch[i]][idx_chunk[i]][idx_peer[i]]).count();
        onet_arr[i] = sp - W[i] - two_L_ms;                 // O_net_i (ms), may be <=0
        os[i] = atx[i] + W[i] + L_ms;                       // O_net segment start
    }
    for (int i = 0; i < N; ++i) {
        double oi = onet_arr[i] > 1e-9 ? onet_arr[i] : 1e-9; // guard tiny/neg windows
        double ai = os[i], bi = os[i] + oi, ov = 0.0;
        for (int j = 0; j < N; ++j) {
            double oj = onet_arr[j] > 1e-9 ? onet_arr[j] : 1e-9;
            double lo = ai > os[j] ? ai : os[j];
            double hj = os[j] + oj;
            double hi = bi < hj ? bi : hj;
            double d = hi - lo; if (d > 0.0) ov += d;
        }
        double keff_onet = ov / oi;                          // time-avg concurrency on O_net window
        double overlap_pc = keff_onet / (double)nch;
        if (overlap_pc < 1.0) overlap_pc = 1.0;
        double onet_eff = onet_arr[i] / overlap_pc;          // one division, no iterate
        int ch = idx_ch[i], ck = idx_chunk[i], pk = idx_peer[i];
        // plugin split (same iteration): plugin = TX_TO_DONE - (rail0 UNION rail1); may be <0 when
        // the two rails run serially (union > single-rail TX_TO_DONE). plugin_eff = plugin/overlap_pc.
        double sp_i = Duration(send_net_done_time[ch][ck][pk] - net_transmitted_time[ch][ck][pk]).count();
        double runion = _rail_union_ms_thisiter((size_t)ch, (size_t)ck, pk);
        int sn = S_onet_n[ch][ck][pk];
        if (sn < STATS_MAX_ITERS) {
            S_ONET    [ch][ck][pk][sn] = (float)onet_arr[i];
            S_ONET_EFF[ch][ck][pk][sn] = (float)onet_eff;
            S_KEFF    [ch][ck][pk][sn] = (float)keff_onet;
            if (runion >= 0.0) {
                double plugin_eff = (sp_i - runion) / overlap_pc;    // (TX_TO_DONE - union)/overlap
                S_PLUGIN_EFF[ch][ck][pk][sn] = (float)plugin_eff;
                S_OTHER_EFF [ch][ck][pk][sn] = (float)(onet_eff - plugin_eff);  // libfabric+EFA remainder
            } else {
                S_PLUGIN_EFF[ch][ck][pk][sn] = 0.0f;
                S_OTHER_EFF [ch][ck][pk][sn] = (float)onet_eff;      // no rail data -> all "other"
            }
            S_onet_n[ch][ck][pk] = sn + 1;
        }
    }
}

void AccumulateTimingMetrics() {
    Duration netIbsend_time;
    Duration proxy_init_time;


    Sum_T_total += Duration(nccl_func_time.count());
    proxy_init_time = proxy_init_time_end - proxy_init_time_start;
    Sum_PROXY_INIT += Duration(proxy_init_time.count());

    // Find first active peer across all channels for FSTART_TO_POST
    TimePoint earliest_post = TimePoint::max();
    TimePoint latest_done = TimePoint();
    for (size_t ch = 0; ch < (size_t)gauge_channels_num; ch++) {
        if (!channel_was_used(ch)) continue;
        for (int p = 0; p < MAX_PEERS; p++) {
            if (net_post_time[ch][0][p] != TimePoint() && net_post_time[ch][0][p] < earliest_post)
                earliest_post = net_post_time[ch][0][p];
            if (send_net_done_time[ch][MAX_GAUGE_CHUNKS-1][p] != TimePoint() && send_net_done_time[ch][MAX_GAUGE_CHUNKS-1][p] > latest_done)
                latest_done = send_net_done_time[ch][MAX_GAUGE_CHUNKS-1][p];
        }
    }

    // FSTART_TO_POST: func start → first chunk posted
    netIbsend_time = earliest_post - nccl_func_start_time;
    Sum_FSTART_TO_POST += Duration(netIbsend_time.count());

    // DONE_TO_FEND: last done → func end
    netIbsend_time = nccl_func_end_time - latest_done;
    Sum_DONE_TO_FEND += Duration(netIbsend_time.count());

    // Per-channel per-chunk per-peer send metrics.
    //
    // Atomic gating: for each (channelid, chunk i, peer k) we validate all
    // required timestamps up-front. If ANY is missing/invalid, this iter's
    // sample for this (ch, i, k) is dropped from EVERY send-side accumulator
    // (POST_TO_DR/DR_TO_TX/POST_TO_RECVDONE/RECVDONE_TO_DR/TX_TO_DONE/ABS_TX/ABS_TX_DONE
    // + rails + ctrl). Otherwise all get recorded together. This guarantees consistent
    // denominators so mean(POST_TO_RECVDONE)+mean(RECVDONE_TO_DR)==mean(POST_TO_DR),
    // and rail/ctrl times align with the send-side series.
    //
    // Ring AllReduce phase math (POST_TO_RECVDONE / RECVDONE_TO_DR slicing):
    //   NCCL runRing outer iter ships M = 2*(nRanks-1)*K slices, K = g_ring_k.
    //   phase r = i % M; phase 0 = directSend (no recv dep) → split undefined.
    //   otherwise send depends on recv slice (i - K) of the SAME outer iter.
    //
    // Recv peer discovery: under SPLIT_MASK=0x0 the ring is rail-scrambled and
    // this rank's NET recv peer is NOT (myRank-1+nRanks)%nRanks in general. Use
    // net_Recv_posted_time (via recv_peer_was_used) as ground truth — the ring
    // has exactly one recv peer per channel from the NET side. When the recv
    // side is intra-node (P2P/CUMEM, no NET stamps) no peer is discovered and
    // recv split is N/A for the whole channel, but send-side samples still
    // record: recv-split absence must NOT drop them.
    for (size_t channelid = 0; channelid < (size_t)gauge_channels_num; channelid++) {
        if (!channel_was_used(channelid)) continue;

        int actual_recv_peer = -1;
        for (int rk = 0; rk < MAX_PEERS; ++rk) {
            if (recv_peer_was_used(channelid, rk)) { actual_recv_peer = rk; break; }
        }

        for (size_t i = 0; i < std::min(static_cast<size_t>(gauge_chunk_num), static_cast<size_t>(MAX_GAUGE_CHUNKS)); ++i) {
            for (int k = 0; k < MAX_PEERS; ++k) {
                if (!peer_was_used(channelid, k)) continue;

                TimePoint post_i     = net_post_time       [channelid][i][k];
                TimePoint dr_i       = net_data_ready_time [channelid][i][k];
                TimePoint tx_i       = net_transmitted_time[channelid][i][k];
                TimePoint done_i     = send_net_done_time  [channelid][i][k];
                TimePoint post0      = net_post_time       [channelid][0][k];

                // All four timestamps must be present and monotonic.
                if (post_i.time_since_epoch().count() == 0) continue;
                if (dr_i  .time_since_epoch().count() == 0) continue;
                if (tx_i  .time_since_epoch().count() == 0) continue;
                if (done_i.time_since_epoch().count() == 0) continue;
                if (post0 .time_since_epoch().count() == 0) continue;
                // Raw stamps only: no monotonicity / ordering rejection. Negative
                // Negative POST_TO_RECVDONE / RECVDONE_TO_DR samples are possible from
                // proxy single-thread stamp jitter and pass through unchanged — use
                // median for a robust summary.

                // Phase 0 (directSend) has no recv dep → split undefined (print N/A).
                // If actual_recv_peer is not discovered (intra-node P2P recv) or the
                // recv_done stamp is missing for this iter, still record send-side
                // samples but leave the split unset.
                const int K = g_ring_k;
                const int M = 2 * (nRanks - 1) * K;
                const int r = (M > 0) ? ((int)i % M) : 0;
                const bool phase0 = (r < K);
                Duration post_to_recvdone(0.0), recvdone_to_dr(0.0);
                bool this_iter_has_recv_split = false;
                if (!phase0 && actual_recv_peer >= 0) {
                    size_t rj = i - K;
                    TimePoint recv_done = recv_net_done_time[channelid][rj][actual_recv_peer];
                    if (recv_done.time_since_epoch().count() != 0) {
                        // Raw split — no clamp. POST_TO_RECVDONE + RECVDONE_TO_DR == POST_TO_DR identically.
                        post_to_recvdone = Duration(recv_done - post_i);
                        recvdone_to_dr = Duration(dr_i - recv_done);
                        this_iter_has_recv_split = true;
                    }
                }

                // Accumulate every send-side metric atomically.
                Duration d_POST_GAP = Duration(post_i - post0);
                Duration d_POST_TO_DR = Duration(dr_i   - post_i);
                Duration d_DR_TO_TX = Duration(tx_i   - dr_i);
                Duration d_TX_TO_DONE = Duration(done_i - tx_i);
                Duration d_ABS_TX = Duration(tx_i   - nccl_func_start_time);
                Duration d_ABS_TX_DONE = Duration(done_i - nccl_func_start_time);
                Sum_POST_GAP        [channelid][i][k] += d_POST_GAP;
                Sum_POST_TO_DR        [channelid][i][k] += d_POST_TO_DR;
                if (this_iter_has_recv_split) {
                    Sum_POST_TO_RECVDONE[channelid][i][k] += post_to_recvdone;
                    Sum_RECVDONE_TO_DR[channelid][i][k] += recvdone_to_dr;
                    has_recv_split[channelid][i][k] = true;
                }
                Sum_DR_TO_TX        [channelid][i][k] += d_DR_TO_TX;
                Sum_TX_TO_DONE        [channelid][i][k] += d_TX_TO_DONE;
                Sum_ABS_TX    [channelid][i][k] += d_ABS_TX;
                Sum_ABS_TX_DONE[channelid][i][k] += d_ABS_TX_DONE;
                Sum_valid_iters[channelid][i][k] += 1;

                // Also record this iter's sample into the stats arrays (if in bounds).
                if (channelid < (size_t)STATS_MAX_CHANNELS && i < (size_t)STATS_MAX_CHUNKS && k < STATS_MAX_PEERS) {
                    int sn = S_send_n[channelid][i][k];
                    if (sn < STATS_MAX_ITERS) {
                        S_POST_GAP        [channelid][i][k][sn] = (float)d_POST_GAP.count();
                        S_POST_TO_DR        [channelid][i][k][sn] = (float)d_POST_TO_DR.count();
                        S_DR_TO_TX        [channelid][i][k][sn] = (float)d_DR_TO_TX.count();
                        S_TX_TO_DONE        [channelid][i][k][sn] = (float)d_TX_TO_DONE.count();
                        S_ABS_TX    [channelid][i][k][sn] = (float)d_ABS_TX.count();
                        S_ABS_TX_DONE[channelid][i][k][sn] = (float)d_ABS_TX_DONE.count();
                        if (this_iter_has_recv_split) {
                            S_POST_TO_RECVDONE[channelid][i][k][sn] = (float)post_to_recvdone.count();
                            S_RECVDONE_TO_DR[channelid][i][k][sn] = (float)recvdone_to_dr.count();
                            S_has_recv_split[channelid][i][k] = true;
                        } else {
                            S_POST_TO_RECVDONE[channelid][i][k][sn] = 0.0f;
                            S_RECVDONE_TO_DR[channelid][i][k][sn] = 0.0f;
                        }
                        S_send_n[channelid][i][k] = sn + 1;
                    }
                }

                // Plugin per-rail + ctrl arrival: only accumulate for this iter
                // because the send-side gate passed. Rails still keep their own
                // counter (a rail may be legitimately absent for some iters);
                // this ties rail/ctrl samples to the same iters that contributed
                // to POST_TO_DR/DR_TO_TX/POST_TO_RECVDONE/RECVDONE_TO_DR/TX_TO_DONE/ABS_TX above.
                if ((size_t)i < static_cast<size_t>(MAX_PLUGIN_PROFILING_SLOTS)) {
                    void *target = nccl_channel_scomm[channelid][k].load(std::memory_order_acquire);
                    if (target != nullptr) {
                        int cid = -1;
                        for (int ii = 0; ii < MAX_PLUGIN_S_COMMS; ++ii) {
                            if (plugin_scomm_by_id[ii].load(std::memory_order_acquire) == target) {
                                cid = ii;
                                break;
                            }
                        }
                        if (cid >= 0) {
                            for (int rr = 0; rr < MAX_PLUGIN_RAILS; ++rr) {
                                TimePoint post_r      = libfabric_post_fi_send_time [cid][i][rr];
                                TimePoint rail_compl  = libfabric_per_rail_compl_time[cid][i][rr];
                                if (post_r.time_since_epoch().count()     == 0) continue;
                                if (rail_compl.time_since_epoch().count() == 0) continue;
                                Duration d_rs = Duration(post_r     - nccl_func_start_time);
                                Duration d_re = Duration(rail_compl - nccl_func_start_time);
                                Duration d_rd = Duration(rail_compl - post_r);
                                Sum_RAIL_START[channelid][k][i][rr] += d_rs;
                                Sum_RAIL_END  [channelid][k][i][rr] += d_re;
                                Sum_RAIL_DUR  [channelid][k][i][rr] += d_rd;
                                Sum_RAIL_iters[channelid][k][i][rr] += 1;
                                if (channelid < (size_t)STATS_MAX_CHANNELS && i < (size_t)STATS_MAX_CHUNKS && k < STATS_MAX_PEERS && rr < STATS_MAX_RAILS) {
                                    int rn = S_rail_n[channelid][k][i][rr];
                                    if (rn < STATS_MAX_ITERS) {
                                        S_RAIL_START[channelid][k][i][rr][rn] = (float)d_rs.count();
                                        S_RAIL_END  [channelid][k][i][rr][rn] = (float)d_re.count();
                                        S_RAIL_DUR  [channelid][k][i][rr][rn] = (float)d_rd.count();
                                        S_rail_n[channelid][k][i][rr] = rn + 1;
                                    }
                                }
                            }
                            TimePoint ctrl = libfabric_ctrl_arrival_time[cid][i];
                            if (ctrl.time_since_epoch().count() != 0) {
                                Duration d_ctrl = Duration(ctrl - nccl_func_start_time);
                                Sum_CTRL_ARRIVAL      [channelid][k][i] += d_ctrl;
                                Sum_CTRL_ARRIVAL_iters[channelid][k][i] += 1;
                                if (channelid < (size_t)STATS_MAX_CHANNELS && i < (size_t)STATS_MAX_CHUNKS && k < STATS_MAX_PEERS) {
                                    int cn = S_ctrl_n[channelid][k][i];
                                    if (cn < STATS_MAX_ITERS) {
                                        S_CTRL[channelid][k][i][cn] = (float)d_ctrl.count();
                                        S_ctrl_n[channelid][k][i] = cn + 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Per-channel per-chunk per-peer recv metrics.
    // All times are in this rank's local func_start frame (no cross-rank alignment).
    // Atomic gating: R0 / R_wait / ABS_RX_POST / ABS_RX_DONE are recorded together
    // or dropped together for a given (ch, i, k) iter sample.
    for (size_t channelid = 0; channelid < (size_t)gauge_channels_num; channelid++) {
        if (!channel_was_used(channelid)) continue;
        for (size_t i = 0; i < std::min(static_cast<size_t>(gauge_chunk_num), static_cast<size_t>(MAX_GAUGE_CHUNKS)); ++i) {
            for (int k = 0; k < MAX_PEERS; ++k) {
                if (!recv_peer_was_used(channelid, k)) continue;

                TimePoint rp_i    = net_Recv_posted_time[channelid][i][k];
                TimePoint rp0     = net_Recv_posted_time[channelid][0][k];
                TimePoint rd_i    = recv_net_done_time  [channelid][i][k];
                if (rp_i.time_since_epoch().count() == 0) continue;
                if (rp0.time_since_epoch().count() == 0) continue;
                if (rd_i.time_since_epoch().count() == 0) continue;
                if (!(rp_i <= rd_i)) continue;

                Duration d_R0     = Duration(rp_i - rp0);
                Duration d_Rw     = Duration(rd_i - rp_i);
                Duration d_ABS_RP = Duration(rp_i - nccl_func_start_time);
                Duration d_ABS_RD = Duration(rd_i - nccl_func_start_time);
                Sum_R0         [channelid][i][k] += d_R0;
                Sum_R_wait     [channelid][i][k] += d_Rw;
                Sum_ABS_RX_POST[channelid][i][k] += d_ABS_RP;
                Sum_ABS_RX_DONE[channelid][i][k] += d_ABS_RD;
                Sum_valid_recv_iters[channelid][i][k] += 1;

                if (channelid < (size_t)STATS_MAX_CHANNELS && i < (size_t)STATS_MAX_CHUNKS && k < STATS_MAX_PEERS) {
                    int rn = S_recv_n[channelid][i][k];
                    if (rn < STATS_MAX_ITERS) {
                        S_R0        [channelid][i][k][rn] = (float)d_R0.count();
                        S_R_wait    [channelid][i][k][rn] = (float)d_Rw.count();
                        S_ABS_RX_POST[channelid][i][k][rn] = (float)d_ABS_RP.count();
                        S_ABS_RX_DONE[channelid][i][k][rn] = (float)d_ABS_RD.count();
                        S_recv_n[channelid][i][k] = rn + 1;
                    }
                }
            }
        }
    }

    // T_dataready_to_done per channel per peer
    for (size_t channelid = 0; channelid < (size_t)gauge_channels_num; channelid++) {
        if (!channel_was_used(channelid)) continue;
        for (int k = 0; k < MAX_PEERS; ++k) {
            if (peer_was_used(channelid, k)) {
                netIbsend_time = send_net_done_time[channelid][MAX_GAUGE_CHUNKS - 1][k] - net_data_ready_time[channelid][0][k];
                Sum_T_dataready_to_done[channelid][k] += Duration(netIbsend_time.count());
            }
            if (recv_peer_was_used(channelid, k)) {
                Duration d = recv_net_done_time[channelid][MAX_GAUGE_CHUNKS - 1][k] - net_Recv_posted_time[channelid][0][k];
                Sum_R_posted_to_done[channelid][k] += Duration(d.count());
            }
        }
    }

    // Total post→completion time (earliest post → latest done)
    Duration post_compl_time = latest_done - earliest_post;
    Sum_net_post_compl_time += Duration(post_compl_time.count());

    // NOTE: rails/ctrl only fire on iters where the
    // (channel, chunk, peer) send-side gate passed. This keeps rail/ctrl means
    // temporally consistent with POST_TO_DR/DR_TO_TX/POST_TO_RECVDONE/RECVDONE_TO_DR/TX_TO_DONE/ABS_TX above.

    // Clear plugin arrays so the NEXT iter's plugin writes land in fresh slots.
    plugin_test_start_seq_num = -1;
    for (int i = 0; i < MAX_PLUGIN_MSG_SEQ_NUM; ++i) {
        libfabric_first_post_time [i] = TimePoint();
        libfabric_first_compl_time[i] = TimePoint();
        libfabric_last_compl_time [i] = TimePoint();
    }
    for (int c = 0; c < MAX_PLUGIN_S_COMMS; ++c) {
        for (int i = 0; i < MAX_PLUGIN_PROFILING_SLOTS; ++i) {
            for (int r = 0; r < MAX_PLUGIN_RAILS; ++r) {
                libfabric_pre_fi_send_time   [c][i][r] = TimePoint();
                libfabric_post_fi_send_time  [c][i][r] = TimePoint();
                libfabric_per_rail_compl_time[c][i][r] = TimePoint();
            }
            libfabric_ctrl_arrival_time[c][i] = TimePoint();
        }
    }

    // Per-iteration iterative O_net: solve NOW, while this iteration's raw timestamps
    // are still in the net_*_time arrays (same-source, same func_start frame).
    ComputeIterativeOnet();
}

// Cap the per-chunk print loops to the first PRINT_CHUNK_LIMIT chunks.
// Accumulation still runs over MAX_GAUGE_CHUNKS; this only trims what
// gets emitted to stdout.
static constexpr size_t PRINT_CHUNK_LIMIT = 64;
static inline size_t print_chunk_end() {
    size_t hw = std::min(static_cast<size_t>(gauge_chunk_num),
                         static_cast<size_t>(MAX_GAUGE_CHUNKS));
    return std::min(hw, PRINT_CHUNK_LIMIT);
}

void printTimingMetrics() {
    if (g_kept_iters == 0) {
        printf("No iterations kept (all filtered). No metrics to print.\n");
        return;
    }
    double denom = (double)g_kept_iters;

    const char* proto_str = (gauge_protocol == 0) ? "LL" :
                            (gauge_protocol == 1) ? "LL128" :
                            (gauge_protocol == 2) ? "Simple" : "unknown";
    printf("INFO: allreduce_ring_%s_rank(%d)_nranks(%d)_message_size(%s)_nchannels(%d)_nthreads(%s)_protocol(%s)_iteration(%d)\n",
           env_split_mask_var, myRank, nRanks, env_gauge_size_var, gauge_channels_num,
           env_gauge_nthreads_var, proto_str, (int)g_kept_iters);
    printf("-- channels actually used: count=%d, mask=0x%lx (out of %d configured)\n",
           __builtin_popcountll(gauge_channels_used_mask), gauge_channels_used_mask, gauge_channels_num);
    printf("-- [TOTAL] ncclAllReduce elapsed time: %.6f ms\n", Sum_T_total.count() / denom);
    printf("-- [PROXY_INIT] proxy init time: %.6f ms\n", Sum_PROXY_INIT.count() / denom);
    printf("-- [FSTART_TO_POST] func start to first netIsend time: %.6f ms\n", Sum_FSTART_TO_POST.count() / denom);
    printf("-- [DONE_TO_FEND] last done to func end time: %.6f ms\n", Sum_DONE_TO_FEND.count() / denom);

    // NOTE: All ABS_TX / ABS_TX_DONE / ABS_RX_POST / ABS_RX_DONE / RAIL_* / CTRL_ARRIVAL
    // times are in THIS rank's local func_start frame. No cross-rank clock alignment
    // is applied — comparisons across ranks require external post-processing.

    for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); channelid++) {
        if (!channel_was_used(channelid)) continue;
        // Send-side blocks (one per send-peer).
        // All send-side metrics for a given (channel, chunk, peer) share one
        // denominator Sum_valid_iters[ch][chunk][peer]. Non-recording iters are
        // dropped consistently across every metric so
        // mean(POST_TO_RECVDONE)+mean(RECVDONE_TO_DR)==mean(POST_TO_DR).
        for (int k = 0; k < MAX_PEERS; ++k) {
            bool any_send = false;
            for (size_t i = 0; i < print_chunk_end() && !any_send; ++i) {
                if (Sum_valid_iters[channelid][i][k] > 0) any_send = true;
            }
            if (!any_send) continue;
            printf("-- --- send to peer %d in channel %zu ---\n", k, channelid);
            for (size_t i = 0; i < print_chunk_end(); ++i) {
                long long n = Sum_valid_iters[channelid][i][k];
                if (n <= 0) {
                    printf("-- [chunk %zu] no valid samples (all iters dropped by send-side gate)\n", i);
                    continue;
                }
                double d = (double)n;
                // Sample buffer for stats: only present when this (ch, i, k) is in STATS_* bounds.
                bool has_stats = (channelid < (size_t)STATS_MAX_CHANNELS
                                  && i < (size_t)STATS_MAX_CHUNKS
                                  && k < STATS_MAX_PEERS);
                int  sn = has_stats ? S_send_n[channelid][i][k] : 0;
                float scratch[STATS_MAX_ITERS];

                #define PRINT_STAT(tag, sum_val, samples) do { \
                    printf("-- [" tag "] chunk %zu %s: mean=%.6f",  i, "", (sum_val) / d); \
                    if (has_stats && sn > 0) { \
                        for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = (samples)[channelid][i][k][_ii]; \
                        stats_print_summary(scratch, sn); \
                    } \
                    printf(" ms (n=%lld)\n", n); \
                } while(0)

                // POST_GAP
                printf("-- [POST_GAP] chunk %zu post gap from chunk 0: mean=%.6f",   i, Sum_POST_GAP[channelid][i][k].count() / d);
                if (has_stats && sn > 0) {
                    for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = S_POST_GAP[channelid][i][k][_ii];
                    stats_print_summary(scratch, sn);
                }
                printf(" ms (n=%lld)\n", n);

                // POST_TO_DR
                printf("-- [POST_TO_DR] chunk %zu posted → data ready: mean=%.6f", i, Sum_POST_TO_DR[channelid][i][k].count() / d);
                if (has_stats && sn > 0) {
                    for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = S_POST_TO_DR[channelid][i][k][_ii];
                    stats_print_summary(scratch, sn);
                }
                printf(" ms (n=%lld)\n", n);

                // POST_TO_RECVDONE / RECVDONE_TO_DR (phase-gated)
                if (has_recv_split[channelid][i][k]) {
                    printf("-- [POST_TO_RECVDONE] chunk %zu posted → recv done       : mean=%.6f",
                           i, Sum_POST_TO_RECVDONE[channelid][i][k].count() / d);
                    if (has_stats && sn > 0 && S_has_recv_split[channelid][i][k]) {
                        for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = S_POST_TO_RECVDONE[channelid][i][k][_ii];
                        stats_print_summary(scratch, sn);
                    }
                    printf(" ms (n=%lld)\n", n);
                    printf("-- [RECVDONE_TO_DR] chunk %zu recv done → data ready   : mean=%.6f",
                           i, Sum_RECVDONE_TO_DR[channelid][i][k].count() / d);
                    if (has_stats && sn > 0 && S_has_recv_split[channelid][i][k]) {
                        for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = S_RECVDONE_TO_DR[channelid][i][k][_ii];
                        stats_print_summary(scratch, sn);
                    }
                    printf(" ms (n=%lld)\n", n);
                } else {
                    printf("-- [POST_TO_RECVDONE] chunk %zu posted → recv done       : N/A (phase 0 or intra-node recv)\n", i);
                    printf("-- [RECVDONE_TO_DR] chunk %zu recv done → data ready   : N/A (phase 0 or intra-node recv)\n", i);
                }

                // DR_TO_TX
                printf("-- [DR_TO_TX] chunk %zu data ready → transmitted: mean=%.6f", i, Sum_DR_TO_TX[channelid][i][k].count() / d);
                if (has_stats && sn > 0) {
                    for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = S_DR_TO_TX[channelid][i][k][_ii];
                    stats_print_summary(scratch, sn);
                }
                printf(" ms (n=%lld)\n", n);

                // TX_TO_DONE
                printf("-- [TX_TO_DONE] chunk %zu transmitted → done: mean=%.6f", i, Sum_TX_TO_DONE[channelid][i][k].count() / d);
                if (has_stats && sn > 0) {
                    for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = S_TX_TO_DONE[channelid][i][k][_ii];
                    stats_print_summary(scratch, sn);
                }
                printf(" ms (n=%lld)\n", n);

                // O_NET (iterative, per-iteration recovered-bandwidth fixed point; median over iters).
                // Solved in-iter from same-source timestamps -> free of median-of-gap contamination.
                {
                    int on_n = has_stats ? S_onet_n[channelid][i][k] : 0;
                    if (on_n > 0) {
                        double on_sum = 0.0;
                        for (int _ii = 0; _ii < on_n; ++_ii) { scratch[_ii] = S_ONET[channelid][i][k][_ii]; on_sum += scratch[_ii]; }
                        printf("-- [O_NET] chunk %zu iterative o_net (per-iter, median): mean=%.6f", i, on_sum / on_n);
                        stats_print_summary(scratch, on_n);
                        printf(" ms (n=%d)\n", on_n);
                        // O_NET_EFF = O_net / overlap_per_ch (one division, per-iter then median)
                        double eff_sum = 0.0;
                        for (int _ii = 0; _ii < on_n; ++_ii) { scratch[_ii] = S_ONET_EFF[channelid][i][k][_ii]; eff_sum += scratch[_ii]; }
                        printf("-- [O_NET_EFF] chunk %zu o_net / overlap_per_ch: mean=%.6f", i, eff_sum / on_n);
                        stats_print_summary(scratch, on_n);
                        printf(" ms (n=%d)\n", on_n);
                        // K_EFF = time-avg concurrency on the O_net segment (all channels pooled)
                        double kf_sum = 0.0;
                        for (int _ii = 0; _ii < on_n; ++_ii) { scratch[_ii] = S_KEFF[channelid][i][k][_ii]; kf_sum += scratch[_ii]; }
                        printf("-- [K_EFF] chunk %zu concurrency on o_net window: mean=%.6f", i, kf_sum / on_n);
                        stats_print_summary(scratch, on_n);
                        printf(" (n=%d)\n", on_n);
                        // PLUGIN_EFF = (TX_TO_DONE - rail0∪rail1) / overlap_per_ch  (per-iter, median)
                        double pe_sum = 0.0;
                        for (int _ii = 0; _ii < on_n; ++_ii) { scratch[_ii] = S_PLUGIN_EFF[channelid][i][k][_ii]; pe_sum += scratch[_ii]; }
                        printf("-- [PLUGIN_EFF] chunk %zu plugin (o_net - rail_union) / overlap: mean=%.6f", i, pe_sum / on_n);
                        stats_print_summary(scratch, on_n);
                        printf(" ms (n=%d)\n", on_n);
                        // OTHER_EFF = o_net_eff - plugin_eff  (libfabric + EFA remainder)
                        double ot_sum = 0.0;
                        for (int _ii = 0; _ii < on_n; ++_ii) { scratch[_ii] = S_OTHER_EFF[channelid][i][k][_ii]; ot_sum += scratch[_ii]; }
                        printf("-- [OTHER_EFF] chunk %zu libfabric+EFA (o_net_eff - plugin_eff): mean=%.6f", i, ot_sum / on_n);
                        stats_print_summary(scratch, on_n);
                        printf(" ms (n=%d)\n", on_n);
                    } else {
                        printf("-- [O_NET] chunk %zu iterative o_net: N/A (no valid iters)\n", i);
                        printf("-- [O_NET_EFF] chunk %zu o_net_eff: N/A (no valid iters)\n", i);
                        printf("-- [K_EFF] chunk %zu concurrency: N/A (no valid iters)\n", i);
                        printf("-- [PLUGIN_EFF] chunk %zu plugin_eff: N/A (no valid iters)\n", i);
                        printf("-- [OTHER_EFF] chunk %zu other_eff: N/A (no valid iters)\n", i);
                    }
                }

                // ABS_TX
                printf("-- [ABS_TX] chunk %zu transmitted @ func_start + mean=%.6f", i, Sum_ABS_TX[channelid][i][k].count() / d);
                if (has_stats && sn > 0) {
                    for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = S_ABS_TX[channelid][i][k][_ii];
                    stats_print_summary(scratch, sn);
                }
                printf(" ms (n=%lld)\n", n);

                // ABS_TX_DONE
                printf("-- [ABS_TX_DONE] chunk %zu send done @ func_start + mean=%.6f", i, Sum_ABS_TX_DONE[channelid][i][k].count() / d);
                if (has_stats && sn > 0) {
                    for (int _ii = 0; _ii < sn; ++_ii) scratch[_ii] = S_ABS_TX_DONE[channelid][i][k][_ii];
                    stats_print_summary(scratch, sn);
                }
                printf(" ms (n=%lld)\n", n);
                #undef PRINT_STAT
            }
            printf("-- [T_dataready_to_done] data ready → last chunk done: %.6f ms\n", Sum_T_dataready_to_done[channelid][k].count() / denom);
        }
        // Recv-side blocks (one per recv-peer). Uses Sum_valid_recv_iters as denom.
        for (int k = 0; k < MAX_PEERS; ++k) {
            bool any_recv = false;
            for (size_t i = 0; i < print_chunk_end() && !any_recv; ++i) {
                if (Sum_valid_recv_iters[channelid][i][k] > 0) any_recv = true;
            }
            if (!any_recv) continue;
            printf("-- --- recv from peer %d in channel %zu ---\n", k, channelid);
            for (size_t i = 0; i < print_chunk_end(); ++i) {
                long long n = Sum_valid_recv_iters[channelid][i][k];
                if (n <= 0) {
                    printf("-- [recv chunk %zu] no valid samples\n", i);
                    continue;
                }
                double d = (double)n;
                bool has_stats = (channelid < (size_t)STATS_MAX_CHANNELS
                                  && i < (size_t)STATS_MAX_CHUNKS
                                  && k < STATS_MAX_PEERS);
                int  rn = has_stats ? S_recv_n[channelid][i][k] : 0;
                float scratch[STATS_MAX_ITERS];

                printf("-- [R0] chunk %zu Irecv post gap from chunk 0: mean=%.6f", i, Sum_R0[channelid][i][k].count() / d);
                if (has_stats && rn > 0) {
                    for (int _ii = 0; _ii < rn; ++_ii) scratch[_ii] = S_R0[channelid][i][k][_ii];
                    stats_print_summary(scratch, rn);
                }
                printf(" ms (n=%lld)\n", n);

                printf("-- [R_wait] chunk %zu Irecv posted → done: mean=%.6f", i, Sum_R_wait[channelid][i][k].count() / d);
                if (has_stats && rn > 0) {
                    for (int _ii = 0; _ii < rn; ++_ii) scratch[_ii] = S_R_wait[channelid][i][k][_ii];
                    stats_print_summary(scratch, rn);
                }
                printf(" ms (n=%lld)\n", n);

                printf("-- [ABS_RX_POST] chunk %zu Irecv posted @ func_start + mean=%.6f", i, Sum_ABS_RX_POST[channelid][i][k].count() / d);
                if (has_stats && rn > 0) {
                    for (int _ii = 0; _ii < rn; ++_ii) scratch[_ii] = S_ABS_RX_POST[channelid][i][k][_ii];
                    stats_print_summary(scratch, rn);
                }
                printf(" ms (n=%lld)\n", n);

                printf("-- [ABS_RX_DONE] chunk %zu Irecv done   @ func_start + mean=%.6f", i, Sum_ABS_RX_DONE[channelid][i][k].count() / d);
                if (has_stats && rn > 0) {
                    for (int _ii = 0; _ii < rn; ++_ii) scratch[_ii] = S_ABS_RX_DONE[channelid][i][k][_ii];
                    stats_print_summary(scratch, rn);
                }
                printf(" ms (n=%lld)\n", n);
            }
            printf("-- [R_posted_to_done] first Irecv posted → last chunk done: %.6f ms\n", Sum_R_posted_to_done[channelid][k].count() / denom);
        }
    }

    printf("-- [T_post_to_done] posted to completed send time (nchunks per channel %d) in all channels: %.6f ms\n", gauge_chunk_num, Sum_net_post_compl_time.count() / denom);

    // ---- Plugin per-rail timing (per-channel, per-peer, per-chunk, per-rail) ----
    // For each (channel, peer) & each physical EFA rail:
    //   RAIL_START : post_fi_send    - nccl_func_start_time
    //   RAIL_END   : per_rail_compl  - nccl_func_start_time
    //   RAIL_DUR   : RAIL_END - RAIL_START
    // Frame: same as ABS_TX / ABS_TX_DONE (this rank's func_start).
    for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); ++channelid) {
        if (!channel_was_used(channelid)) continue;
        for (int k = 0; k < MAX_PEERS; ++k) {
            // Check if this (channel, peer) has any RAIL data at all
            bool has_any = false;
            for (size_t i = 0; i < print_chunk_end() && !has_any; ++i) {
                for (int r = 0; r < MAX_PLUGIN_RAILS; ++r) {
                    if (Sum_RAIL_iters[channelid][k][i][r] > 0) { has_any = true; break; }
                }
            }
            if (!has_any) continue;
            printf("-- --- plugin per-rail timing: channel %zu peer %d ---\n", channelid, k);
            for (size_t i = 0; i < print_chunk_end(); ++i) {
                for (int r = 0; r < MAX_PLUGIN_RAILS; ++r) {
                    long long nr = Sum_RAIL_iters[channelid][k][i][r];
                    if (nr <= 0) continue;
                    double dr = (double)nr;
                    bool has_stats_r = (channelid < (size_t)STATS_MAX_CHANNELS
                                        && i < (size_t)STATS_MAX_CHUNKS
                                        && k < STATS_MAX_PEERS
                                        && r < STATS_MAX_RAILS);
                    int  rsn = has_stats_r ? S_rail_n[channelid][k][i][r] : 0;
                    float scratch_r[STATS_MAX_ITERS];

                    printf("-- [RAIL_START] channel %zu peer %d chunk %zu rail %d start @ func_start + mean=%.6f",
                           channelid, k, i, r, Sum_RAIL_START[channelid][k][i][r].count() / dr);
                    if (has_stats_r && rsn > 0) {
                        for (int _ii = 0; _ii < rsn; ++_ii) scratch_r[_ii] = S_RAIL_START[channelid][k][i][r][_ii];
                        stats_print_summary(scratch_r, rsn);
                    }
                    printf(" ms (n=%lld)\n", nr);

                    printf("-- [RAIL_END]   channel %zu peer %d chunk %zu rail %d end   @ func_start + mean=%.6f",
                           channelid, k, i, r, Sum_RAIL_END[channelid][k][i][r].count() / dr);
                    if (has_stats_r && rsn > 0) {
                        for (int _ii = 0; _ii < rsn; ++_ii) scratch_r[_ii] = S_RAIL_END[channelid][k][i][r][_ii];
                        stats_print_summary(scratch_r, rsn);
                    }
                    printf(" ms (n=%lld)\n", nr);

                    printf("-- [RAIL_DUR]   channel %zu peer %d chunk %zu rail %d end - start           : mean=%.6f",
                           channelid, k, i, r, Sum_RAIL_DUR[channelid][k][i][r].count() / dr);
                    if (has_stats_r && rsn > 0) {
                        for (int _ii = 0; _ii < rsn; ++_ii) scratch_r[_ii] = S_RAIL_DUR[channelid][k][i][r][_ii];
                        stats_print_summary(scratch_r, rsn);
                    }
                    printf(" ms (n=%lld)\n", nr);
                }
            }
        }
    }

    // ---- Ctrl-message arrival at sender (per-channel, per-peer, per-chunk) ----
    // Time when sender's send() first observed the receiver-posted ctrl_msg for
    // this send in its ctrl_mailbox (has_ctrl_msg()==true). Stamped in plugin
    // send() inside the have_ctrl branch, non-eager path only (eager fires data
    // before ctrl and never needs to wait). Frame: this rank's func_start.
    //   CTRL_ARRIVAL - RAIL_START = time proxy stalled waiting for ctrl.
    for (size_t channelid = 0; channelid < static_cast<size_t>(gauge_channels_num); ++channelid) {
        if (!channel_was_used(channelid)) continue;
        for (int k = 0; k < MAX_PEERS; ++k) {
            bool has_any = false;
            for (size_t i = 0; i < print_chunk_end() && !has_any; ++i) {
                if (Sum_CTRL_ARRIVAL_iters[channelid][k][i] > 0) has_any = true;
            }
            if (!has_any) continue;
            printf("-- --- ctrl arrival: channel %zu peer %d ---\n", channelid, k);
            for (size_t i = 0; i < print_chunk_end(); ++i) {
                long long nc = Sum_CTRL_ARRIVAL_iters[channelid][k][i];
                if (nc <= 0) continue;
                double dc = (double)nc;
                bool has_stats_c = (channelid < (size_t)STATS_MAX_CHANNELS
                                    && i < (size_t)STATS_MAX_CHUNKS
                                    && k < STATS_MAX_PEERS);
                int  cn = has_stats_c ? S_ctrl_n[channelid][k][i] : 0;
                float scratch_c[STATS_MAX_ITERS];
                printf("-- [CTRL_ARRIVAL] channel %zu peer %d chunk %zu ctrl seen @ func_start + mean=%.6f",
                       channelid, k, i, Sum_CTRL_ARRIVAL[channelid][k][i].count() / dc);
                if (has_stats_c && cn > 0) {
                    for (int _ii = 0; _ii < cn; ++_ii) scratch_c[_ii] = S_CTRL[channelid][k][i][_ii];
                    stats_print_summary(scratch_c, cn);
                }
                printf(" ms (n=%lld)\n", nc);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    initializeTimingArrays();

    for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
        for (int j = 0; j < MAX_GAUGE_CHUNKS; ++j) {
            for (int k = 0; k < MAX_PEERS; ++k) {
                net_transmitted_time[i][j][k] = TimePoint();
                send_net_done_time[i][j][k] = TimePoint();
                recv_net_done_time[i][j][k] = TimePoint();
                net_post_time[i][j][k] = TimePoint();
                net_data_ready_time[i][j][k] = TimePoint();
                net_Recv_posted_time[i][j][k] = TimePoint();
            }
        }
    }

    for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
        for (int j = 0; j < N_MESSAGES; ++j) {
            for (int k = 0; k < MAX_PEERS; ++k) {
                net_first_post_time[i][j][k] = TimePoint();
                netIsend_first_chunk_time[i][j][k] = TimePoint();
                netIrecv_first_chunk_time[i][j][k] = TimePoint();
                netIsend_last_chunk_time[i][j][k] = TimePoint();
                netIrecv_last_chunk_time[i][j][k] = TimePoint();
            }
        }
    }

    // Plugin state (aws-ofi-nccl) — arm capture and clear slots before the
    // first iteration so the plugin's first post captures a fresh base seq_num.
    plugin_test_start_seq_num = -1;
    for (int i = 0; i < MAX_PLUGIN_MSG_SEQ_NUM; ++i) {
        libfabric_first_post_time [i] = TimePoint();
        libfabric_first_compl_time[i] = TimePoint();
        libfabric_last_compl_time [i] = TimePoint();
    }
    for (int c = 0; c < MAX_PLUGIN_S_COMMS; ++c) {
        for (int i = 0; i < MAX_PLUGIN_PROFILING_SLOTS; ++i) {
            for (int r = 0; r < MAX_PLUGIN_RAILS; ++r) {
                libfabric_pre_fi_send_time   [c][i][r] = TimePoint();
                libfabric_post_fi_send_time  [c][i][r] = TimePoint();
                libfabric_per_rail_compl_time[c][i][r] = TimePoint();
            }
            libfabric_ctrl_arrival_time[c][i] = TimePoint();
        }
    }

    env_gauge_heo_var = getenv("GAUGE_HEO");
    env_gauge_mode_var = getenv("GAUGE_MODE");
    env_gauge_iteration_var = getenv("GAUGE_ITERATION");
    const char* env_gauge_output_dir_var = getenv("GAUGE_OUT_DIRE");
    env_gauge_nthreads_var = getenv("NCCL_NTHREADS");
    env_split_mask_var = getenv("NCCL_TESTS_SPLIT_MASK");

    if (!env_gauge_heo_var) env_gauge_heo_var = "inter";
    if (!env_gauge_mode_var) env_gauge_mode_var = "allreduce_ring";
    if (!env_gauge_iteration_var) env_gauge_iteration_var = "100";
    if (!env_gauge_nthreads_var) env_gauge_nthreads_var = "512";
    if (!env_split_mask_var) env_split_mask_var = "0x0";
    if (!env_gauge_output_dir_var) {
        env_gauge_output_dir_var = ".";
        printf("WARNING: GAUGE_OUT_DIRE not set, using current dir\n");
    }

    // Derive ring K = ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS from NCCL_PROTO.
    // enqueue.cc gates chunkSteps/sliceSteps on (protocol==SIMPLE && algo==RING);
    // for LL / LL128 both drop to 1. Explicit GAUGE_RING_K overrides.
    {
        const char* env_k = getenv("GAUGE_RING_K");
        if (env_k && env_k[0] != '\0') {
            int k = atoi(env_k);
            if (k >= 1) g_ring_k = k;
        } else {
            const char* proto = getenv("NCCL_PROTO");
            if (proto && (strcasecmp(proto, "LL") == 0 || strcasecmp(proto, "LL128") == 0)) {
                g_ring_k = 1;
            } else {
                g_ring_k = 2;   // Simple, or unset (NCCL default may pick Simple for AR/Ring)
            }
        }
        if (myRank == 0) {
            const char* proto_show = getenv("NCCL_PROTO");
            printf("[GAUGE] Ring K = %d (NCCL_PROTO=%s%s)\n",
                   g_ring_k,
                   proto_show ? proto_show : "<unset>",
                   getenv("GAUGE_RING_K") ? ", override via GAUGE_RING_K" : "");
        }
    }

    long long size = 1;
    env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
    if (env_gauge_size_var != nullptr) {
        long long bytes = atoll(env_gauge_size_var);
        size = (bytes + 3) / 4;
        if (size < 1) size = 1;
    }

    gauge_iterations = atoi(env_gauge_iteration_var);

    if (argc >= 3) {
        send_gauge_d = atoi(argv[1]);
        recv_gauge_d = atoi(argv[2]);
    }

    int localRank = 0;

    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device flags: %s\n", cudaGetErrorString(err));
        return 1;
    }

    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    const char* env_gauge_out_suffix = getenv("GAUGE_OUT_SUFFIX");
    if (!env_gauge_out_suffix || env_gauge_out_suffix[0] == '\0') {
        env_gauge_out_suffix = "msg";
    }

    // GAUGE_STDOUT_ONLY_RANK0=1 -> only rank 0 writes a per-rank .out file;
    // other ranks redirect stdout to /dev/null so their gauge output does NOT
    // leak into mpirun's aggregated stream (i.e. the SLURM job stdout). Useful
    // when only rank-0 metrics are analyzed. Note this also drops non-rank-0
    // NCCL_DEBUG=INFO lines from the SLURM log — rank 0's is usually enough.
    const char* env_only_r0 = getenv("GAUGE_STDOUT_ONLY_RANK0");
    bool only_rank0 = (env_only_r0 && env_only_r0[0] == '1');

    if (only_rank0 && myRank != 0) {
        if (freopen("/dev/null", "w", stdout) == NULL) {
            perror("freopen /dev/null failed");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        char filename[256];
        sprintf(filename, "%s/nccl_allreduce_ring_%s_r-%d-%s.out",
                env_gauge_output_dir_var, env_gauge_heo_var,
                myRank, env_gauge_out_suffix);

        FILE *file = freopen(filename, "a", stdout);
        if (file == NULL) {
            perror("freopen failed");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        setbuf(stdout, NULL);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    gauge_cpu_mhz = get_cpu_mhz();
    host_isend_gauge_d = 0;
    host_check_GPU_data_gauge_d = 0;
    host_recv_gauge_d = 0;
    gauge_send_message_itr = 0;
    gauge_recv_message_itr = 0;

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

    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

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

    // In-place vs out-of-place: when GAUGE_INPLACE=1, sendbuff == recvbuff.
    const char* inplace_env = getenv("GAUGE_INPLACE");
    const bool gauge_inplace = (inplace_env != nullptr) && (atoi(inplace_env) != 0);
    const void* ar_sendbuff = gauge_inplace ? (const void*)recvbuff : (const void*)sendbuff;
    if (myRank == 0) {
        printf("[GAUGE_INPLACE] %s (sendbuff %s recvbuff)\n",
               gauge_inplace ? "1 (in-place)" : "0 (out-of-place)",
               gauge_inplace ? "==" : "!=");
        fflush(stdout);
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERATION; i++) {
        NCCLCHECK(ncclAllReduce(ar_sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));
    }
    CUDACHECK(cudaStreamSynchronize(s));
    MPI_Barrier(MPI_COMM_WORLD);

    // GPU/CPU clock calibration
    CUDACHECK(cudaEventRecord(calib_event, s));
    CUDACHECK(cudaEventSynchronize(calib_event));
    calib_cpu_time = std::chrono::high_resolution_clock::now();

    // Enable profiling
    gauge_send_message_itr = 1;
    gauge_recv_message_itr = 1;
    host_isend_gauge_d = send_gauge_d;
    host_recv_gauge_d = recv_gauge_d;

    for (int gauge_iter = 0; gauge_iter < gauge_iterations; gauge_iter++) {
        // Reset per-iteration timing arrays
        for (int i = 0; i < MAX_GAUGE_CHANNELS; ++i) {
            for (int j = 0; j < MAX_GAUGE_CHUNKS; ++j) {
                for (int k = 0; k < MAX_PEERS; ++k) {
                    net_transmitted_time[i][j][k] = TimePoint();
                    send_net_done_time[i][j][k] = TimePoint();
                    net_post_time[i][j][k] = TimePoint();
                    net_data_ready_time[i][j][k] = TimePoint();
                    recv_net_done_time[i][j][k] = TimePoint();
                    net_Recv_posted_time[i][j][k] = TimePoint();
                }
            }
        }
        // Re-arm plugin state (AccumulateTimingMetrics also does this
        // on kept iters; drop here for outlier / discarded iters).
        plugin_test_start_seq_num = -1;
        for (int i = 0; i < MAX_PLUGIN_MSG_SEQ_NUM; ++i) {
            libfabric_first_post_time [i] = TimePoint();
            libfabric_first_compl_time[i] = TimePoint();
            libfabric_last_compl_time [i] = TimePoint();
        }
        for (int c = 0; c < MAX_PLUGIN_S_COMMS; ++c) {
            for (int i = 0; i < MAX_PLUGIN_PROFILING_SLOTS; ++i) {
                for (int r = 0; r < MAX_PLUGIN_RAILS; ++r) {
                    libfabric_pre_fi_send_time   [c][i][r] = TimePoint();
                    libfabric_post_fi_send_time  [c][i][r] = TimePoint();
                    libfabric_per_rail_compl_time[c][i][r] = TimePoint();
                }
                libfabric_ctrl_arrival_time[c][i] = TimePoint();
            }
        }

        // Align entry across ranks so per-iter counter resets are close in time.
        MPI_Barrier(MPI_COMM_WORLD);

        // Reset each s_comm's per-iter send counter to 0. This must happen
        // after Barrier (so any leftover retry from the prior iter is drained
        // by then) and before ncclAllReduce (so this iter's first send lands
        // in slot 0). Uses a callback published by the plugin at dlopen.
        for (int cid = 0; cid < MAX_PLUGIN_S_COMMS; ++cid) {
            void *sc = plugin_scomm_by_id[cid].load(std::memory_order_acquire);
            plugin_reset_iter_seq(sc);
        }

        nccl_func_start_time = std::chrono::high_resolution_clock::now();

        CUDACHECK(cudaEventRecord(kernel_start_event, s));

        NCCLCHECK(ncclAllReduce(ar_sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));

        CUDACHECK(cudaStreamSynchronize(s));
        MPI_Barrier(MPI_COMM_WORLD);

        nccl_func_end_time = std::chrono::high_resolution_clock::now();

        float gpu_delta_ms = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&gpu_delta_ms, calib_event, kernel_start_event));
        kernel_launch_cpu_time = calib_cpu_time +
            std::chrono::nanoseconds((long long)(gpu_delta_ms * 1e6));

        nccl_func_time = nccl_func_end_time - nccl_func_start_time;

        AccumulateTimingMetrics();
        ++g_kept_iters;
    }

    CUDACHECK(cudaStreamSynchronize(s));

    printTimingMetrics();

    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    ncclCommDestroy(comm);

    MPICHECK(MPI_Finalize());

    printf("[MPI Rank %d] Success \n", myRank);
    return 0;
}
