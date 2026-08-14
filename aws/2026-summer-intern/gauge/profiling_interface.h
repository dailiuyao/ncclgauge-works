#ifndef NCCL_PLUGIN_PROFILING_H
#define NCCL_PLUGIN_PROFILING_H

#ifdef __cplusplus
#include <atomic>
#include <chrono>

#ifndef PROFILE_P2P
#define PROFILE_P2P 1
#endif

#define MAX_PLUGIN_MSG_SEQ_NUM 1024
// aws-ofi-nccl caps physical rails at 4 (see include/nccl_ofi_rdma.h MAX_NUM_RAILS).
// Keep this in sync — if the plugin's MAX_NUM_RAILS changes, bump this.
#define MAX_PLUGIN_RAILS 4
// Upper bound on distinct s_comms observed on one rank.
// nchannels × nPeers (worst case: nchannels=32, but per-rank fewer peers).
#define MAX_PLUGIN_S_COMMS 64
// Must match the value in nccl.h.in — kept as a small local mirror so that
// the plugin can compile without pulling nccl.h.
#define MAX_GAUGE_CHANNELS_LOCAL 32
#define MAX_PEERS_LOCAL          32

// Per-s_comm profiling slot count. Slot = iter-local send counter (0-based);
// posts beyond this many sends in one iter are NOT stamped. Gauge only cares
// about the first ~128 chunks per iter, so 128 is enough. Sized per s_comm
// (each NCCL channel/peer has its own s_comm and its own counter).
#define MAX_PLUGIN_PROFILING_SLOTS 128

#if PROFILE_P2P == 1
// Timing arrays for measuring per-chunk performance metrics
// Indexed by msg_seq_num (plugin's monotonic request sequence number)
// Storage owned by gauge executable, referenced by plugin library
extern std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_first_post_time[MAX_PLUGIN_MSG_SEQ_NUM];
extern std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_first_compl_time[MAX_PLUGIN_MSG_SEQ_NUM];
extern std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_last_compl_time[MAX_PLUGIN_MSG_SEQ_NUM];

// Per-(s_comm, iter-local slot, rail) enter/return timestamps for fi_write*.
// Slot = plugin's per-iter send counter, reset to 0 by the gauge at each
// iter boundary. Posts with slot >= MAX_PLUGIN_PROFILING_SLOTS are dropped.
extern std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_pre_fi_send_time [MAX_PLUGIN_S_COMMS][MAX_PLUGIN_PROFILING_SLOTS][MAX_PLUGIN_RAILS];
extern std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_post_fi_send_time[MAX_PLUGIN_S_COMMS][MAX_PLUGIN_PROFILING_SLOTS][MAX_PLUGIN_RAILS];

// Per-rail CQE arrival for SEND writes, stamped in handle_cq_entry
// (FI_WRITE + NCCL_OFI_RDMA_SEND). Indexed by [s_comm_id][slot][rail],
// where slot is the per-iter send counter captured at send() time.
extern std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_per_rail_compl_time[MAX_PLUGIN_S_COMMS][MAX_PLUGIN_PROFILING_SLOTS][MAX_PLUGIN_RAILS];

// Time when sender's send() first observed the ctrl_msg for this msg_seq_num
// in ctrl_mailbox (has_ctrl_msg()==true). Ctrl arrives via receiver→sender
// RDMA WRITE to the sender's mailbox; sender detects it by polling
// (no CQE). One entry per send (no rail dimension — ctrl is one arrival per
// msg). Indexed by [s_comm_id][slot].
extern std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_ctrl_arrival_time[MAX_PLUGIN_S_COMMS][MAX_PLUGIN_PROFILING_SLOTS];

// --- s_comm bridge tables --------------------------------------------------
// Plugin publishes: local_comm_id → s_comm pointer (set once in create_send_comm).
// NCCL publishes:   [channelId][peer] → s_comm pointer (set once in sendConnect).
// Gauge cross-references these two tables at print time to map each plugin
// slot back to its NCCL (channel, peer) identity.
extern std::atomic<void*> plugin_scomm_by_id [MAX_PLUGIN_S_COMMS];
extern std::atomic<void*> nccl_channel_scomm [MAX_GAUGE_CHANNELS_LOCAL][MAX_PEERS_LOCAL];

// Test-window start in plugin's msg_seq_num space (legacy single-anchor).
// Still used by first_post/first_compl/last_compl arrays which are not
// per-s_comm. Plugin CAS-writes the first msg_seq_num it sees after the
// gauge resets it to -1.
extern int plugin_test_start_seq_num;

// Callback pointer set by the plugin at load time. Zeros a given s_comm's
// per-iter profiling send counter so the next AllReduce's first send lands
// in slot 0. Gauge calls this every iter after MPI_Barrier and before
// ncclAllReduce. Storage lives in the gauge exe (profiling_arrays.cpp) so
// the gauge can call across the dlopen boundary; the plugin's static
// initializer publishes the pointer.
typedef void (*plugin_reset_iter_seq_fn)(void *scomm);
extern std::atomic<plugin_reset_iter_seq_fn> plugin_reset_iter_seq_ptr;

inline void plugin_reset_iter_seq(void *scomm) {
    auto fn = plugin_reset_iter_seq_ptr.load(std::memory_order_acquire);
    if (fn && scomm) fn(scomm);
}
#endif

#endif // __cplusplus
#endif // NCCL_PLUGIN_PROFILING_H
