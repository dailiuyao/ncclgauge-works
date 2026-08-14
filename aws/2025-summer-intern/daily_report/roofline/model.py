import math
from dataclasses import dataclass
from typing import Optional, Callable, Any, List, Dict, Tuple

@dataclass
class TreeParams:
    o_rrs_small: float
    o_send_small: float
    o_rcs_small: float
    o_rrcs_small: float
    L_rrs_small: float
    L_send_small: float
    L_rcs_small: float
    L_rrcs_small: float
    L_rrs_large: float
    L_send_large: float
    L_rcs_large: float
    L_rrcs_large: float
    bw: float
    num_channels: Callable


@dataclass
class RingParams:
    G_reverse: float
    L_G_ss: float
    L_G: float
    max_nchannels_start: int
    slice_step: int
    FIFO_Depth: int
    isend_chunk_size: int
    num_channels: Callable

@dataclass
class ModelResult:
    best_algo: object
    busbw: float
    latency: float

class LogGP(object):
    KB = 1024
    MB = 1024 * 1024
    GB = 1024 * 1024 * 1024

    def __init__(self):
        self.tree_simple: TreeParams = None
        self.tree_ll128: TreeParams = None
        self.tree_ll: TreeParams = None
        self.ring_simple: RingParams = None
        self.ring_ll: RingParams = None
        self.ring_ll128: RingParams = None
        self.algos: Dict[str, List[Callable]] = {}

    def allr_tree(p: TreeParams, size: int, nodes: int, gpus_per_node: int, chunk_size: float):
        number_channels = p.num_channels(size, nodes)
        a = min(chunk_size * number_channels, size / LogGP.MB)
        log_n_nodes = math.log2(nodes)
        if a * LogGP.KB * p.o_send_small + p.L_send_small < p.L_send_large:
            latency = ((a * LogGP.KB * (
                        (p.o_rrs_small + p.o_rcs_small) * (log_n_nodes - 1) + p.o_send_small + p.o_rrcs_small))
                    + p.L_send_small + (log_n_nodes - 1) * (p.L_rrs_small + p.L_rcs_small) + p.L_rrcs_small
                    + max(size / LogGP.MB - number_channels * chunk_size, 0) * LogGP.KB / p.bw)
        else:
            latency = (p.L_send_large + (log_n_nodes - 1) * (p.L_rrs_large + p.L_rcs_large)
                + p.L_rrcs_large + max(size / LogGP.MB - number_channels * chunk_size, 0) * LogGP.KB / p.bw)

        total_gpus = nodes * gpus_per_node
        busbw = ((size / (LogGP.GB)) / (latency * 1e-6)) * 2 * (total_gpus - 1) / total_gpus
        return (latency, busbw)

    def allr_ring(p: RingParams, size: int, nodes: int, gpus_per_node:int):
        number_channels = p.num_channels(size, nodes)

        if size / LogGP.KB < p.max_nchannels_start / p.slice_step:
            latency = p.L_G_ss * (nodes * gpus_per_node * 2 - 2) * p.slice_step / p.FIFO_Depth
        elif size / LogGP.KB < nodes * gpus_per_node * number_channels * p.slice_step * p.isend_chunk_size:
            latency = (p.G_reverse * (
                        size / LogGP.KB / (nodes * gpus_per_node * number_channels * p.slice_step)) + p.L_G_ss) * (
                        nodes * gpus_per_node * 2 - 2) * p.slice_step / p.FIFO_Depth
        else:
            latency = ((p.G_reverse * p.isend_chunk_size + p.L_G_ss)
                * ((size / LogGP.KB / (nodes * gpus_per_node * number_channels * p.slice_step)) / p.isend_chunk_size)
                * (nodes * gpus_per_node * 2 - 2) * p.slice_step) / p.FIFO_Depth

        total_gpus = nodes * gpus_per_node
        busbw = ((size / (LogGP.GB)) / (latency * 1e-6)) * 2 * (total_gpus - 1) / total_gpus
        return (latency, busbw)

    def allg_ring(p: RingParams, size: int, nodes: int, gpus_per_node: int):
        number_channels = p.num_channels(size, nodes)

        if size / LogGP.KB < p.max_nchannels_start * nodes * gpus_per_node / p.slice_step:
            latency = p.L_G_ss * (nodes * gpus_per_node - 1) * p.slice_step / p.FIFO_Depth
        elif size / LogGP.KB < nodes * gpus_per_node * number_channels * p.slice_step * p.isend_chunk_size:
            latency = (p.G_reverse * (
                        size / LogGP.KB / (nodes * gpus_per_node * number_channels * p.slice_step)) + p.L_G_ss) * (
                        nodes * gpus_per_node - 1) * p.slice_step / p.FIFO_Depth
        else:
            latency = ((p.G_reverse * p.isend_chunk_size + p.L_G_ss)
                * ((size / LogGP.KB / (nodes * gpus_per_node * number_channels * p.slice_step)) / p.isend_chunk_size)
                * (nodes * gpus_per_node - 1) * p.slice_step) / p.FIFO_Depth

        total_gpus = nodes * gpus_per_node
        busbw = ((size / (LogGP.GB)) / (latency * 1e-6)) * (total_gpus - 1) / total_gpus
        return (latency, busbw)

    def reducescatter_ring(p: RingParams, size: int, nodes: int, gpus_per_node: int):
        number_channels = p.num_channels(size, nodes)

        if size / LogGP.KB < p.max_nchannels_start * nodes * gpus_per_node:
            latency = p.L_G_ss * (nodes * gpus_per_node - 1) * p.slice_step / p.FIFO_Depth
        elif size / LogGP.KB < nodes * gpus_per_node * number_channels * p.slice_step * p.isend_chunk_size:
            latency = (p.G_reverse * (
                        size / LogGP.KB / (nodes * gpus_per_node * number_channels * p.slice_step)) + p.L_G_ss) * (
                        nodes * gpus_per_node - 1) * p.slice_step / p.FIFO_Depth
        else:
            latency = ((p.G_reverse * p.isend_chunk_size + p.L_G_ss)
                * ((size / LogGP.KB / (nodes * gpus_per_node * number_channels * p.slice_step)) / p.isend_chunk_size)
                * (nodes * gpus_per_node - 1) * p.slice_step) / p.FIFO_Depth

        total_gpus = nodes * gpus_per_node
        busbw = ((size / (LogGP.GB)) / (latency * 1e-6)) * (total_gpus - 1) / total_gpus
        return (latency, busbw)

    def allr_tree_simple(self, size: int, nodes: int, gpus_per_node: int):
        chunk_size = 0.5
        return LogGP.allr_tree(self.tree_simple, size, nodes, gpus_per_node, chunk_size)

    def allr_tree_ll(self, size: int, nodes: int, gpus_per_node: int):
        chunk_size = 0.03125
        return LogGP.allr_tree(self.tree_ll, size, nodes, gpus_per_node, chunk_size)

    def allr_tree_ll128(self, size: int, nodes: int, gpus_per_node: int):
        chunk_size = 0.5487674169
        return LogGP.allr_tree(self.tree_ll128, size, nodes, gpus_per_node, chunk_size) # revised

    def allr_ring_simple(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.allr_ring(self.ring_simple, size, nodes,gpus_per_node)

    def allr_ring_ll(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.allr_ring(self.ring_ll, size, nodes, gpus_per_node)
    
    def allr_ring_ll128(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.allr_ring(self.ring_ll128, size, nodes, gpus_per_node)
    
    def allg_ring_simple(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.allg_ring(self.ag_ring_simple, size, nodes, gpus_per_node)

    def allg_ring_ll(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.allg_ring(self.ag_ring_ll, size, nodes, gpus_per_node)
    
    def allg_ring_ll128(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.allg_ring(self.ag_ring_ll128, size, nodes, gpus_per_node)
    
    def reducescatter_ring_simple(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.reducescatter_ring(self.rs_ring_simple, size, nodes, gpus_per_node)

    def reducescatter_ring_ll(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.reducescatter_ring(self.rs_ring_ll, size, nodes, gpus_per_node)
    
    def reducescatter_ring_ll128(self, size: int, nodes: int, gpus_per_node: int):
        return LogGP.reducescatter_ring(self.rs_ring_ll128, size, nodes, gpus_per_node)

    def results(self, operation: str, size: int, nodes: int, gpus_per_node: int):

        # Calculate latencies for each algorithm
        results = {}
        busbws = {}
        best_algo = None
        best_latency = float('inf')
        best_busbw = 0
    
        for algo in self.algos[operation]:
            current_latency, current_busbw = algo(size, nodes, gpus_per_node)
            results[algo] = (current_latency, current_busbw)
            
            # Update best algorithm if current is better
            if current_latency < best_latency:
                best_algo = algo
                best_latency = current_latency
                best_busbw = current_busbw

        #print(f"best_algo is {best_algo}")

        return ModelResult(best_algo=best_algo, latency=best_latency, busbw=best_busbw)


class P5en(LogGP):
    def tree_num_channels(self, size: int, nodes: int):
        assert(nodes == 16)
        if size <= 32768: return 1
        elif size <= 65536: return 2
        elif size <= 131072: return 4
        elif size <= 262144: return 8
        else: return 16

    def ring_num_channels(self, size: int, nodes: int):
        assert(nodes == 16)
        if size <= 32768: return 1
        elif size <= 65536: return 2
        elif size <= 131072: return 4
        elif size <= 524288: return 8
        else: return 16

    def AG_RS_simple_ring_num_channels(self, size: int, nodes: int):
        assert(nodes == 16)
        if size <= 1024: return 1
        elif size <= 2048: return 2
        elif size <= 4096: return 4
        elif size <= 8192: return 8
        else: return 16

    def AG_RS_LL_ring_num_channels(self, size: int, nodes: int):
        assert(nodes == 16)
        if size <= 4096: return 1
        elif size <= 8192: return 2
        elif size <= 16384: return 4
        elif size <= 32768: return 8
        else: return 16

    def AG_RS_LL128_ring_num_channels(self, size: int, nodes: int):
        assert(nodes == 16)
        if size <= 1024: return 2
        elif size <= 2048: return 4
        elif size <= 4096: return 8
        else: return 16

    def __init__(self):
        LogGP.__init__(self)
        self.algos['All Reduce'] = [
            self.allr_tree_simple, self.allr_tree_ll, self.allr_tree_ll128,
            self.allr_ring_simple, self.allr_ring_ll, self.allr_ring_ll128
        ]
        self.algos['All Gather'] = [
            self.allg_ring_simple, self.allg_ring_ll, self.allg_ring_ll128
        ]
        self.algos['Reduce Scatter'] = [
            self.reducescatter_ring_simple, self.reducescatter_ring_ll, self.reducescatter_ring_ll128
        ]

        self.tree_simple: TreeParams = TreeParams(
            bw=164.713445,
            num_channels=self.tree_num_channels,
            o_send_small=0.016, L_send_small=36.720,
            o_rrs_small=0.002, L_rrs_small=35.260,
            o_rrcs_small=0.002, L_rrcs_small=33.388,
            o_rcs_small=0.002, L_rcs_small=25.293,
            L_send_large=152.975,
            L_rrs_large=71.174,
            L_rrcs_large=63.394,
            L_rcs_large=58.319
        )
        self.tree_ll: TreeParams = TreeParams(
            bw=19.087940,
            num_channels=self.tree_num_channels,
            o_send_small=0.060, L_send_small=40.389,
            o_rrs_small=0.019, L_rrs_small=31.587,
            o_rrcs_small=0.015, L_rrcs_small=28.961,
            o_rcs_small=0.018, L_rcs_small=28.600,
            L_send_large=77.431,
            L_rrs_large=97.402,
            L_rrcs_large=42.302,
            L_rcs_large=67.187
        )
        self.tree_ll128: TreeParams = TreeParams(
            bw=116.402182,
            num_channels=self.tree_num_channels,
            o_send_small=0.003, L_send_small=45.886,
            o_rrs_small=0.001, L_rrs_small=30.867,
            o_rrcs_small=0.001, L_rrcs_small=29.630,
            o_rcs_small=0.001, L_rcs_small=30.014,
            L_send_large=120.397,
            L_rrs_large=67.440,
            L_rrcs_large=93.793,
            L_rcs_large=64.701
        )
        self.ring_simple: RingParams = RingParams(
            num_channels=self.ring_num_channels,
            G_reverse=0.154,
            L_G=39.727,
            L_G_ss=24.903,
            max_nchannels_start=1024,
            slice_step=2,
            FIFO_Depth=4, # revised
            isend_chunk_size=2048
        )
        self.ring_ll: RingParams = RingParams(
            num_channels=self.ring_num_channels,
            G_reverse=2.189,
            L_G=51.879,
            L_G_ss=28.076,
            max_nchannels_start=2048,
            slice_step=1,
            FIFO_Depth=8, # revised
            isend_chunk_size=32,
        )
        self.ring_ll128: RingParams = RingParams(
            num_channels=self.ring_num_channels,
            G_reverse=0.330,
            L_G=53.632,
            L_G_ss=26.644,
            max_nchannels_start=2048,
            slice_step=1,
            FIFO_Depth=8, # revised
            isend_chunk_size=562,
        )
        self.ag_ring_simple: RingParams = RingParams(
            num_channels=self.AG_RS_simple_ring_num_channels,
            G_reverse=0.157,
            L_G=44.199,
            L_G_ss=27.711,
            max_nchannels_start=16,
            slice_step=2,
            FIFO_Depth=4, # revised
            isend_chunk_size=2048
        )
        self.ag_ring_ll: RingParams = RingParams(
            num_channels=self.AG_RS_LL_ring_num_channels,
            G_reverse=2.503,
            L_G=52.666,
            L_G_ss=30.716,
            max_nchannels_start=64,
            slice_step=1,
            FIFO_Depth=8, # revised
            isend_chunk_size=32,
        )
        self.ag_ring_ll128: RingParams = RingParams(
            num_channels=self.AG_RS_LL128_ring_num_channels,
            G_reverse=0.405,
            L_G=57.163,
            L_G_ss=27.711,
            max_nchannels_start=8,
            slice_step=1,
            FIFO_Depth=8, # revised
            isend_chunk_size=562,
        )
        self.rs_ring_simple: RingParams = RingParams(
            num_channels=self.AG_RS_simple_ring_num_channels,
            G_reverse=0.155,
            L_G=38.214,
            L_G_ss=27.182,
            max_nchannels_start=16,
            slice_step=2,
            FIFO_Depth=4, # revised
            isend_chunk_size=2048
        )
        self.rs_ring_ll: RingParams = RingParams(
            num_channels=self.AG_RS_LL_ring_num_channels,
            G_reverse=2.668,
            L_G=47.813,
            L_G_ss=30.496,
            max_nchannels_start=64,
            slice_step=1,
            FIFO_Depth=8, # revised
            isend_chunk_size=32,
        )
        self.rs_ring_ll128: RingParams = RingParams(
            num_channels=self.AG_RS_LL128_ring_num_channels,
            G_reverse=0.334,
            L_G=61.165,
            L_G_ss=29.365,
            max_nchannels_start=8,
            slice_step=1,
            FIFO_Depth=8, # revised
            isend_chunk_size=562,
        )


LogGP.p5en = P5en()