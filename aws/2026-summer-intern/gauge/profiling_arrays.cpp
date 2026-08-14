#include "profiling_interface.h"

// Define the timing arrays that will be shared between gauge and plugin
#if PROFILE_P2P == 1
std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_first_post_time[MAX_PLUGIN_MSG_SEQ_NUM];
std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_first_compl_time[MAX_PLUGIN_MSG_SEQ_NUM];
std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_last_compl_time[MAX_PLUGIN_MSG_SEQ_NUM];

std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_pre_fi_send_time [MAX_PLUGIN_S_COMMS][MAX_PLUGIN_PROFILING_SLOTS][MAX_PLUGIN_RAILS];
std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_post_fi_send_time[MAX_PLUGIN_S_COMMS][MAX_PLUGIN_PROFILING_SLOTS][MAX_PLUGIN_RAILS];
std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_per_rail_compl_time[MAX_PLUGIN_S_COMMS][MAX_PLUGIN_PROFILING_SLOTS][MAX_PLUGIN_RAILS];
std::chrono::time_point<std::chrono::high_resolution_clock> libfabric_ctrl_arrival_time[MAX_PLUGIN_S_COMMS][MAX_PLUGIN_PROFILING_SLOTS];

std::atomic<void*> plugin_scomm_by_id [MAX_PLUGIN_S_COMMS];
std::atomic<void*> nccl_channel_scomm [MAX_GAUGE_CHANNELS_LOCAL][MAX_PEERS_LOCAL];

int plugin_test_start_seq_num = -1;

std::atomic<plugin_reset_iter_seq_fn> plugin_reset_iter_seq_ptr{nullptr};
#endif
