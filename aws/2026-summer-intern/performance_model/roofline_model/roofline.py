"""Roofline (alpha-beta) AllReduce / AllGather time models.

The roofline uses only physical link latency and theoretical link bandwidth. It is
the idealized best case; comparing it to measured results exposes where and how large
the gap is.

Common P2P latency parameter (all families):
    RTT(m) = a + m*b   (from pping, inter-node)
    a = fixed latency, taken as the mean RTT of small messages (<= 32 B). a/2 is the
    one-way (per-step) startup latency. Only a is used here; NIC_BW is set from the
    theoretical link bandwidth (50 GB/s * 97.5%).

The formulas are PLATFORM-INDEPENDENT: hardware parameters (NIC bandwidth via cfg, node
counts, ``a``) are all passed in, so nothing here is tied to a specific machine. Functions
are grouped by the collective's TOPOLOGY / algorithm model, not by platform:

* ``ring0x7_*`` / ``tree0x7_*`` – 0x7 topology: the ring spans NODES (n = number of nodes),
  no channel cap.
* ``ring0x0_*`` / ``tree0x0_*`` – 0x0 topology: all GPUs on one ring (n_total = N*n_gpus),
  channel count capped at the NIC count; Ring/Simple has a small/large message branch;
  the LL/LL128 fixed term carries the per-protocol intra-node latency.
* ``allgather_ring_0x7`` – AllGather Ring, 0x7 topology (n-1 steps instead of 2(n-1)).
* ``gcp_*`` – thin wrappers used by the p6-b200 GCP-vs-AWS notebooks: they take a PER-SIZE
  ``nch`` array (nchannels read from a table) and return just the busBW column. AllReduce
  ``gcp_ring_*``/``gcp_tree_*`` delegate to the 0x0 topology; AllGather ``gcp_ag_*`` delegate to
  ``allgather_ring_0x7``.

The ``ring0x7_*`` / ``tree0x7_*`` / ``ring0x0_*`` / ``tree0x0_*`` helpers return a dict
of DataFrame columns so a notebook does::

    for col, val in ring0x7_simple(sizes, n, nch, a).items():
        dft[col] = val

The ``gcp_*`` helpers return a busBW (GB/s) array directly.

To run a specific platform, pass ``cfg=PlatformConfig.load(<name>)`` and ``cfg.a_us`` as
``a``; with no cfg the functions fall back to the default EFA bandwidth. The module is the
single source of truth for the roofline math (the notebooks only load data and plot).
"""

import json
import os

import numpy as np


# ==========================================================================
# Per-platform hardware parameters (loaded from config.json)
# ==========================================================================
# The roofline FORMULAS below are platform-independent; only the numeric hardware
# parameters change per platform. Those live in config.json and are loaded into a
# PlatformConfig, which every roofline function accepts as an optional ``cfg`` argument.
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


class PlatformConfig:
    """Hardware parameters for one platform, loaded from ``config.json``.

    Usage::

        cfg = PlatformConfig.load('p5en')
        cfg.nic_bw_dual                        # aggregate dual-rail NIC BW (bytes/us)
        cfg.a_us                               # P2P RTT of a small message (us) == alpha
        cfg.intra_node_latency('ring', 'll')   # NCCL intra-node latency (us)

    Pass the instance as ``cfg=...`` to the roofline functions (and ``cfg.a_us`` as ``a``).
    """

    def __init__(self, name, platform_dict):
        self.name = name
        self._p = platform_dict
        nic = platform_dict['nic']
        # aggregate NIC bandwidth (bytes/us) = link_bw_GBs * 1e3 * efficiency
        self.nic_bw_dual = nic['link_bw_GBs'] * 1e3 * nic['efficiency']
        self.n_gpus_per_node = platform_dict['n_gpus_per_node']
        # P2P RTT of a small message (us): fixed startup latency (== alpha). a/2 per step.
        self.a_us = platform_dict['a_us']

    @classmethod
    def load(cls, name, path=None):
        """Load platform ``name`` (e.g. 'p5en', 'p6-b200') from config.json."""
        with open(path or _CONFIG_PATH) as f:
            cfg = json.load(f)
        if name not in cfg['platforms']:
            raise KeyError(f"platform {name!r} not in config.json; "
                           f"available: {sorted(cfg['platforms'])}")
        return cls(name, cfg['platforms'][name])

    def intra_node_latency(self, algo, proto):
        """NCCL intra-node latency (us) = base + (n_gpus_per_node-1)*hop, per (algo, proto).
        algo in {'ring','tree'}, proto in {'simple','ll','ll128'} (case-insensitive)."""
        lat = self._p['nccl_intra_node_latency_us']
        base = lat['base'][algo.lower()][proto.lower()]
        hop = lat['hop'][algo.lower()][proto.lower()]
        return base + (self.n_gpus_per_node - 1) * hop


# Default NIC bandwidth (AWS EFA: 50 GB/s * 97.5%). Fallback when a function is called
# WITHOUT a PlatformConfig; to run another platform, pass cfg=PlatformConfig.load(<name>).
NIC_BW_DUAL_EFA = 50 * 1e3 * 0.975        # bytes/us (50 GB/s * 97.5%)

# On-wire chunk sizes and payload-inflation factors (NCCL constants, platform-independent).
#   LL   sends data + a flag word (2x on wire); default chunk 32 KB.
#   LL128 carries 1 flag slot per 16 (16/15 on wire); chunk 576000 B
#         (= BuffSize_LL128/NCCL_STEPS/16*15 = 4,915,200/8/16*15 ~ 562.5 KiB).
LL_CHUNK = 32 * 1024
LL128_CHUNK = 576000
LL_WIRE_FACTOR = 2.0
LL128_WIRE_FACTOR = 16.0 / 15.0


def _nic_bw_dual(cfg):
    """NIC bandwidth (bytes/us): from cfg if given, else the EFA default."""
    return NIC_BW_DUAL_EFA if cfg is None else cfg.nic_bw_dual


def busbw_factor(n):
    """AllReduce busBW reporting factor 2(n-1)/n."""
    return 2 * (n - 1) / n


def _bw_cols(sizes, T_roof, bfactor):
    """T/algBW/busBW columns from a time array (GB/s)."""
    algbw = sizes / (T_roof * 1e-6) / 1e9
    busbw = algbw * bfactor
    return {'T_roofline_us': T_roof, 'algbw_roofline_GBs': algbw,
            'busbw_roofline_GBs': busbw}




# ==========================================================================
# 0x7 layout: ring spans NODES (n = number of nodes), no channel cap
# ==========================================================================
def ring0x7_simple(sizes, n, nch, a, cfg=None):
    """AllReduce Ring/Simple roofline, 0x7 (n = number of nodes on the ring).

        chunk_size     = min(4 MB, m / (n * nch))
        fixed_overhead = (2n-2) * (a/2) * ceil(m / (n * nch * chunk_size))
        NIC_BW         = cfg dual-rail BW (default 50 * 1e3 * 97.5%)  (bytes/us)
        G              = 2(n-1) / (n * NIC_BW)
        T(m)           = fixed_overhead + m * G
    """
    sizes = np.asarray(sizes, dtype=float)
    nic_bw = _nic_bw_dual(cfg)
    fixed_per_step = (2 * n - 2) * (a / 2)
    G = 2 * (n - 1) / (n * nic_bw)
    msg_per_gpu_ch = sizes / (n * nch)
    chunk = np.minimum(4 * 1024 * 1024, msg_per_gpu_ch)
    n_chunks = np.ceil(msg_per_gpu_ch / chunk).astype(int)
    T_roof = fixed_per_step * n_chunks + sizes * G
    return _bw_cols(sizes, T_roof, busbw_factor(n))


def _ring0x7_ll_like(sizes, n, nch, a, chunk_size, wire_factor, cfg=None):
    sizes = np.asarray(sizes, dtype=float)
    n_pipeline_steps = np.ceil(sizes / (nch * chunk_size * n)).astype(int)
    fixed = 2 * (n - 1) * (a / 2) * n_pipeline_steps
    nic_bw = _nic_bw_dual(cfg)
    G = wire_factor * 2 * (n - 1) / (n * nic_bw)
    T_roof = fixed + sizes * G
    return _bw_cols(sizes, T_roof, busbw_factor(n))


def ring0x7_ll(sizes, n, nch, a, cfg=None):
    """AllReduce Ring/LL roofline, 0x7 (n = number of nodes).

        chunk_size     = 32 KB
        n_steps        = ceil(m / (nch * chunk_size * n))
        fixed_overhead = 2(n-1) * (a/2) * n_steps
        NIC_BW         = cfg dual-rail BW (default 50 * 1e3 * 97.5%)
        G              = 2 * 2(n-1) / (n * NIC_BW)   (LL is 2x on wire: data + flag)
        T(m)           = fixed_overhead + m * G
    """
    return _ring0x7_ll_like(sizes, n, nch, a, LL_CHUNK, LL_WIRE_FACTOR, cfg)


def ring0x7_ll128(sizes, n, nch, a, cfg=None):
    """AllReduce Ring/LL128 roofline, 0x7. Same shape as Ring/LL but:

        chunk_size = 576000 B (~562.5 KiB)
        G          = (16/15) * 2(n-1) / (n * NIC_BW)   (LL128 is 16/15 on wire)
    """
    return _ring0x7_ll_like(sizes, n, nch, a, LL128_CHUNK, LL128_WIRE_FACTOR, cfg)


def _tree_divisor(nch):
    """Tree bottleneck: middle node does 3 send+3 recv (/3) for nch<=2, /2 for nch>=4."""
    return 3 if nch <= 2 else 2


def tree0x7_simple(sizes, n, nch, a, cfg=None):
    """AllReduce Tree/Simple roofline, 0x7 (n = number of nodes on the tree).

        fixed_overhead = (2*log2(n)+1) * a/2
        NIC_BW_REAL    = NIC_BW/3  for nch in {1,2}  (middle node: 3 send + 3 recv)
                         NIC_BW/2  for nch in {4,8,16}  (contention averages out)
        T(m)           = fixed_overhead + m / NIC_BW_REAL
    """
    sizes = np.asarray(sizes, dtype=float)
    fixed = (2 * np.log2(n) + 1) * a / 2
    bw_real = _nic_bw_dual(cfg) / _tree_divisor(nch)
    T_roof = fixed + sizes * (1 / bw_real)
    return _bw_cols(sizes, T_roof, busbw_factor(n))


def tree0x7_ll(sizes, n, nch, a, cfg=None):
    """AllReduce Tree/LL roofline, 0x7.

        fixed_overhead = (2*log2(n)+1) * a/2
        NIC_BW         = single/dual EFA (as in Ring/LL); LL effective BW = NIC_BW/2
        NIC_BW_REAL    = (NIC_BW/2)/3 for nch<=2, (NIC_BW/2)/2 for nch>=4
        T(m)           = fixed_overhead + m / NIC_BW_REAL
    """
    sizes = np.asarray(sizes, dtype=float)
    fixed = (2 * np.log2(n) + 1) * a / 2
    bw_real = (_nic_bw_dual(cfg) / 2) / _tree_divisor(nch)
    T_roof = fixed + sizes * (1 / bw_real)
    return _bw_cols(sizes, T_roof, busbw_factor(n))


def tree0x7_ll128(sizes, n, nch, a, cfg=None):
    """AllReduce Tree/LL128 roofline, 0x7. Same as Tree/LL but LL128 effective BW =
    NIC_BW*15/16, so NIC_BW_REAL = (NIC_BW*15/16)/divisor (divisor 3 for nch<=2, else 2)."""
    sizes = np.asarray(sizes, dtype=float)
    fixed = (2 * np.log2(n) + 1) * a / 2
    bw_real = (_nic_bw_dual(cfg) * 15 / 16) / _tree_divisor(nch)
    T_roof = fixed + sizes * (1 / bw_real)
    return _bw_cols(sizes, T_roof, busbw_factor(n))


# ==========================================================================
# 0x0 layout: all GPUs on one ring (n_total = N*n_gpus), channel count capped at NIC count
# ==========================================================================
def ring0x0_simple(sizes, N_nodes, n_gpus, nch_raw, a, threshold_latency=700, half_lat=None,
                   cfg=None):
    """AllReduce Ring/Simple roofline, 0x0.

        n_total = N * n_gpus                    (all GPUs are on one ring)
        nch     = min(nch_raw, 8)               (capped at the NIC count)
        NIC_BW  = cfg dual-rail BW (default 50 * 1e3 * 97.5%)
        G_agg   = nch * NIC_BW                  (nch parallel NICs between adjacent nodes)
        transfer = m / G_agg
        threshold_latency: above this per-chunk transfer, each chunk's startup latency is
            pipeline-hidden, so only inter-node steps count; below it, every step's a/2
            is exposed.
          transfer <  threshold: T = (2*n_total - 1) * a/2 + (2*(n_total-1)/n_total) * m/G_agg
          transfer >= threshold: T = (2*N       - 1) * a/2 + (2*(n_total-1)/n_total) * m/G_agg
    """
    sizes = np.asarray(sizes, dtype=float)
    n_total = N_nodes * n_gpus
    bfactor = 2 * (n_total - 1) / n_total
    nch = np.minimum(nch_raw, 8)
    G_agg = nch * _nic_bw_dual(cfg)   # aggregate BW across nch NICs (full dual-rail)
    G_per_byte = bfactor / G_agg
    transfer = sizes / G_agg
    hl = (a / 2) if half_lat is None else half_lat   # half_lat overrides a/2 (alt roofline overlay)
    fixed_small = (2 * n_total - 1) * hl
    fixed_large = (2 * N_nodes - 1) * hl
    fixed = np.where(transfer < threshold_latency, fixed_small, fixed_large)
    T_roof = fixed + sizes * G_per_byte
    return _bw_cols(sizes, T_roof, bfactor)


def _ring0x0_ll_like(sizes, N_nodes, n_gpus, nch_raw, a, intra_node_lat,
                     chunk_size, wire_factor, cfg=None):
    sizes = np.asarray(sizes, dtype=float)
    n_total = N_nodes * n_gpus
    bfactor = 2 * (n_total - 1) / n_total
    fixed_per_step = (2 * N_nodes - 1) * (a / 2 + intra_node_lat)
    nch = np.minimum(nch_raw, 8)
    G_agg = nch * _nic_bw_dual(cfg)   # aggregate BW across nch NICs (full dual-rail)
    G = wire_factor * 2 * (n_total - 1) / (n_total * G_agg)
    denom = nch_raw * chunk_size * n_total
    n_steps = np.ceil(sizes / denom).astype(int)
    T_roof = fixed_per_step * n_steps + sizes * G
    # 0x0 LL/LL128 attaches only T_roofline_us + busbw_roofline_GBs
    busbw = sizes / (T_roof * 1e-6) / 1e9 * bfactor
    return {'T_roofline_us': T_roof, 'busbw_roofline_GBs': busbw}


def ring0x0_ll(sizes, N_nodes, n_gpus, nch_raw, a, intra_node_lat, cfg=None):
    """AllReduce Ring/LL roofline, 0x0.

        n_total        = N * n_gpus;  nch = min(nch_raw, 8);  chunk_size = 32 KB
        fixed_per_step = (2N - 1) * (a/2 + intra_node_lat)   (intra_node_lat is the LL
                         per-protocol intra-node latency, passed in by the caller)
        n_steps        = ceil(m / (nch_raw * chunk_size * n_total))
        G_agg          = nch * NIC_BW;  G = 2*2(n_total-1) / (n_total * G_agg)
        T(m)           = fixed_per_step * n_steps + m * G
    """
    return _ring0x0_ll_like(sizes, N_nodes, n_gpus, nch_raw, a, intra_node_lat,
                            LL_CHUNK, LL_WIRE_FACTOR, cfg)


def ring0x0_ll128(sizes, N_nodes, n_gpus, nch_raw, a, intra_node_lat, cfg=None):
    """AllReduce Ring/LL128 roofline, 0x0. Same as Ring/LL but chunk_size = 576000 B and
    G = (16/15) * 2(n_total-1) / (n_total * G_agg)."""
    return _ring0x0_ll_like(sizes, N_nodes, n_gpus, nch_raw, a, intra_node_lat,
                            LL128_CHUNK, LL128_WIRE_FACTOR, cfg)


def tree0x0_simple(sizes, N_nodes, n_gpus, nch, a, intra_node_tree_lat, cfg=None):
    """AllReduce Tree/Simple roofline, 0x0.

        nch_eff        = min(nch, 8)
        fixed_overhead = (2*log2(N)+1) * a/2 + 2 * intra_node_tree_lat
        NIC_BW_REAL    = nch_eff * NIC_BW / 3
        T(m)           = fixed_overhead + m / NIC_BW_REAL
    busBW uses the n_total-based factor 2(n_total-1)/n_total.
    """
    sizes = np.asarray(sizes, dtype=float)
    n_total = N_nodes * n_gpus
    bfactor = 2 * (n_total - 1) / n_total
    fixed = (2 * np.log2(N_nodes) + 1) * a / 2 + 2 * intra_node_tree_lat
    nic_bw_real = np.minimum(nch, 8) * _nic_bw_dual(cfg) / 3
    T_roof = fixed + sizes * (1 / nic_bw_real)
    return _bw_cols(sizes, T_roof, bfactor)


def _tree0x0_ll_like(sizes, N_nodes, n_gpus, nch, a, intra_node_tree_lat, wire_factor, cfg=None):
    sizes = np.asarray(sizes, dtype=float)
    n_total = N_nodes * n_gpus
    bfactor = 2 * (n_total - 1) / n_total   # n_total-based, consistent with tree0x0_simple / gcp_tree_*
    # LL effective data BW = NIC_BW/2; LL128 = NIC_BW*15/16.
    nic_bw_base = _nic_bw_dual(cfg) / wire_factor if wire_factor == 2.0 else _nic_bw_dual(cfg) * 15 / 16
    fixed = (2 * np.log2(N_nodes) + 1) * a / 2 + 2 * intra_node_tree_lat
    nic_bw_real = np.minimum(nch, 8) * nic_bw_base / 3
    T_roof = fixed + sizes * (1 / nic_bw_real)
    busbw = sizes / (T_roof * 1e-6) / 1e9 * bfactor
    return {'T_roofline_us': T_roof, 'busbw_roofline_GBs': busbw}


def tree0x0_ll(sizes, N_nodes, n_gpus, nch, a, intra_node_tree_lat, cfg=None):
    """AllReduce Tree/LL roofline, 0x0.

        nch_eff        = min(nch, 8)
        fixed_overhead = (2*log2(N)+1) * a/2 + 2 * intra_node_tree_lat  (LL per-protocol)
        LL effective BW = NIC_BW/2;  NIC_BW_REAL = nch_eff * (NIC_BW/2) / 3
        T(m)           = fixed_overhead + m / NIC_BW_REAL
    """
    return _tree0x0_ll_like(sizes, N_nodes, n_gpus, nch, a, intra_node_tree_lat,
                            LL_WIRE_FACTOR, cfg)


def tree0x0_ll128(sizes, N_nodes, n_gpus, nch, a, intra_node_tree_lat, cfg=None):
    """AllReduce Tree/LL128 roofline, 0x0. Same as Tree/LL but LL128 effective BW =
    NIC_BW*15/16, so NIC_BW_REAL = nch_eff * (NIC_BW*15/16) / 3."""
    return _tree0x0_ll_like(sizes, N_nodes, n_gpus, nch, a, intra_node_tree_lat,
                            LL128_WIRE_FACTOR, cfg)


# ==========================================================================
# GCP-vs-AWS AllReduce: same 0x0 math, caller passes a per-size nchannels array (from a table)
# ==========================================================================
def gcp_ring_simple_busbw(sizes, nch, N_nodes, n_gpus, a, half_lat=None,
                          threshold_latency=700, cfg=None):
    """AllReduce Ring/Simple busBW (GB/s), p6 GCP comparison.

    Thin wrapper over :func:`ring0x0_simple` (same formula) that takes a per-size ``nch``
    array and returns just the busBW column. ``half_lat`` selects the alternate
    (a/2 = const) roofline overlay.
    """
    return ring0x0_simple(sizes, N_nodes, n_gpus, np.asarray(nch, float), a,
                          threshold_latency=threshold_latency,
                          half_lat=half_lat, cfg=cfg)['busbw_roofline_GBs']


def gcp_ring_ll_busbw(sizes, nch_raw, nch, N_nodes, n_gpus, a, intra_node_lat, cfg=None):
    """AllReduce Ring/LL busBW (GB/s), p6 GCP comparison. Thin wrapper over
    :func:`ring0x0_ll` (same formula); ``nch`` is ignored (the core clips nch_raw)."""
    return ring0x0_ll(sizes, N_nodes, n_gpus, np.asarray(nch_raw, float), a,
                      intra_node_lat, cfg=cfg)['busbw_roofline_GBs']


def gcp_ring_ll128_busbw(sizes, nch_raw, nch, N_nodes, n_gpus, a, intra_node_lat, cfg=None):
    """AllReduce Ring/LL128 busBW (GB/s), p6 GCP comparison. Thin wrapper over
    :func:`ring0x0_ll128` (same formula)."""
    return ring0x0_ll128(sizes, N_nodes, n_gpus, np.asarray(nch_raw, float), a,
                         intra_node_lat, cfg=cfg)['busbw_roofline_GBs']


def gcp_tree_simple_busbw(sizes, nch_eff, N_nodes, n_gpus, a, intra_node_tree_lat, cfg=None):
    """AllReduce Tree/Simple busBW (GB/s), p6 GCP comparison. Thin wrapper over
    :func:`tree0x0_simple` (same formula) with a per-size ``nch_eff`` array."""
    return tree0x0_simple(sizes, N_nodes, n_gpus, np.asarray(nch_eff, float), a,
                          intra_node_tree_lat, cfg=cfg)['busbw_roofline_GBs']


def gcp_tree_ll_busbw(sizes, nch_eff, N_nodes, n_gpus, a, intra_node_tree_lat, cfg=None):
    """AllReduce Tree/LL busBW (GB/s), p6 GCP comparison. Thin wrapper over
    :func:`tree0x0_ll` (same formula) with a per-size ``nch_eff`` array."""
    return tree0x0_ll(sizes, N_nodes, n_gpus, np.asarray(nch_eff, float), a,
                      intra_node_tree_lat, cfg=cfg)['busbw_roofline_GBs']


def gcp_tree_ll128_busbw(sizes, nch_eff, N_nodes, n_gpus, a, intra_node_tree_lat, cfg=None):
    """AllReduce Tree/LL128 busBW (GB/s), p6 GCP comparison. Thin wrapper over
    :func:`tree0x0_ll128` (same formula) with a per-size ``nch_eff`` array."""
    return tree0x0_ll128(sizes, N_nodes, n_gpus, np.asarray(nch_eff, float), a,
                         intra_node_tree_lat, cfg=cfg)['busbw_roofline_GBs']


# ==========================================================================
# 0x7 AllGather Ring — generic core (n-1 steps instead of AllReduce's 2(n-1))
# ==========================================================================
def allgather_ring_0x7(sizes, n, nch, a, proto='simple', cfg=None):
    """AllGather Ring roofline busBW (GB/s), 0x7 layout — platform-independent core.

    ``n`` = number of ring participants; ``proto`` in {'simple','ll','ll128'}.

        chunk      = Simple: min(4 MB, m/(n*nch)); LL: 32 KB; LL128: 576000 B
        n_steps    = ceil((per-GPU-per-channel data) / chunk)
        fixed      = (n-1) * a/2 * n_steps                 (AllGather: n-1 steps)
        G          = wire_factor * (n-1) / (n * NIC_BW)    (wire_factor: LL=2, LL128=16/15)
        T(m)       = fixed + m * G ;  busBW = m/T * (n-1)/n
    """
    sizes = np.asarray(sizes, dtype=float)
    nch = np.asarray(nch, dtype=float)
    proto = proto.lower()
    nic_bw = _nic_bw_dual(cfg)
    bfactor = (n - 1) / n
    if proto == 'simple':
        wire_factor = 1.0
        msg_per_gpu_ch = sizes / (n * nch)
        chunk = np.minimum(4 * 1024 * 1024, msg_per_gpu_ch)
        n_steps = np.ceil(msg_per_gpu_ch / chunk).astype(int)
    else:
        chunk_size = LL_CHUNK if proto == 'll' else LL128_CHUNK
        wire_factor = LL_WIRE_FACTOR if proto == 'll' else LL128_WIRE_FACTOR
        n_steps = np.ceil(sizes / (nch * chunk_size * n)).astype(int)
    G = wire_factor * (n - 1) / (n * nic_bw)
    fixed = (n - 1) * a / 2 * n_steps
    T_roof = fixed + sizes * G
    return sizes / (T_roof * 1e-6) / 1e9 * bfactor


# --- thin wrappers used by the p6-b200 GCP AllGather notebook -------------------
def gcp_ag_ring_simple_busbw(sizes, nch, n_on_ring, a, busbw_factor, cfg=None):
    """AllGather Ring/Simple busBW (GB/s). Thin wrapper over :func:`allgather_ring_0x7`.
    ``busbw_factor`` is accepted for backward compat but the core derives (n-1)/n itself."""
    return allgather_ring_0x7(sizes, n_on_ring, nch, a, 'simple', cfg)


def gcp_ag_ring_ll_busbw(sizes, nch, n_on_ring, a, busbw_factor, cfg=None):
    """AllGather Ring/LL busBW (GB/s). Thin wrapper over :func:`allgather_ring_0x7`."""
    return allgather_ring_0x7(sizes, n_on_ring, nch, a, 'll', cfg)


def gcp_ag_ring_ll128_busbw(sizes, nch, n_on_ring, a, busbw_factor, cfg=None):
    """AllGather Ring/LL128 busBW (GB/s). Thin wrapper over :func:`allgather_ring_0x7`."""
    return allgather_ring_0x7(sizes, n_on_ring, nch, a, 'll128', cfg)
