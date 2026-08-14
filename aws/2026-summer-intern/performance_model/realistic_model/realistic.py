"""Realistic Ring-AllReduce time model for the Simple protocol (0x7).

``compute_ring_simple`` returns the modelled AllReduce time (us) for an array of
message sizes, using the LogGP-style fixed-overhead formula:

    fixed = 2L + nrounds*max(O_round - chunk/(G/nch), 0)
               + nrounds*nsteps*(L + max(max(O_net, O_mem) - chunk/(G/nch), 0))
    T     = fixed + msg * 2(n-1)/(n*G)

The two hardware inputs (``a`` = alpha, ``G_bw`` = NIC bandwidth) come from this
package's own :class:`PlatformConfig` (``config.json``), independent of
``roofline_model``. The calibrated overhead terms O_net / O_mem / O_round come from
:class:`~realistic_model.overheads.RealisticParams` (loaded from ``overhead.json``,
produced by ``ring_param_extraction.ipynb``). Protocol chunking constants live here.
"""

import json
import os

import numpy as np

PROTO_SIMPLE = 'Simple'

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


class PlatformConfig:
    """Per-platform hardware parameters for the realistic model, from ``config.json``.

    Independent of ``roofline_model``'s config. The realistic Ring/Simple 0x7 model
    needs only ``a_us`` (fixed startup latency == alpha) and the aggregate NIC
    bandwidth. Usage::

        cfg = PlatformConfig.load('p5en')
        cfg.a_us          # P2P RTT of a small message (us) == alpha
        cfg.nic_bw_dual   # aggregate NIC bandwidth (bytes/us)
    """

    def __init__(self, name, platform_dict):
        self.name = name
        nic = platform_dict['nic']
        # aggregate NIC bandwidth (bytes/us) = link_bw_GBs * 1e3 * efficiency
        self.nic_bw_dual = nic['link_bw_GBs'] * 1e3 * nic['efficiency']
        # P2P RTT of a small message (us): fixed startup latency (== alpha).
        self.a_us = platform_dict['a_us']

    @classmethod
    def load(cls, name, path=None):
        """Load platform ``name`` (e.g. 'p5en') from config.json."""
        with open(path or _CONFIG_PATH) as f:
            cfg = json.load(f)
        if name not in cfg['platforms']:
            raise KeyError(f"platform {name!r} not in config.json; "
                           f"available: {sorted(cfg['platforms'])}")
        return cls(name, cfg['platforms'][name])


def compute_ring_simple(params, sizes, n, nch, G_bw, L_val, proto=PROTO_SIMPLE):
    """Realistic AllReduce time (us) for Ring/Simple (0x7).

    chunk = min(4MB, msg/(n*nch)); slice = chunk/2; nsteps = 2(n-1);
    nrounds = ceil(msg/(n*nch*chunk)). O_net/O_mem looked up at chunk (fit keyed on chunk).
    """
    sizes = np.asarray(sizes, dtype=float)
    chunk_size = np.minimum(4 * 1024 * 1024, sizes / (n * nch))
    slice_size = chunk_size / 2
    nsteps = 2 * (n - 1)
    nrounds = np.maximum(np.ceil(sizes / (n * nch * chunk_size)).astype(int), 1)

    O_net_arr = params.O_net_eff_us(proto, chunk_size, nch)
    O_mem_arr = np.maximum(params.O_mem_us(proto, chunk_size, nch), 0.0)
    O_round_arr = params.O_round_us(proto, slice_size, nch)
    T_nic_chunk = chunk_size * nch / G_bw                       # chunk/(G/nch)

    round_excess = np.maximum(O_round_arr - T_nic_chunk, 0.0)
    step_excess = L_val + np.maximum(np.maximum(O_net_arr, O_mem_arr) - T_nic_chunk, 0.0)
    fixed_overhead = 2 * L_val + nrounds * round_excess + nrounds * nsteps * step_excess

    G_allreduce = 2 * (n - 1) / (n * G_bw)
    return fixed_overhead + sizes * G_allreduce
