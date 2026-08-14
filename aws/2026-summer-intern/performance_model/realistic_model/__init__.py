"""Realistic modelling package.

The realistic model introduces three profiling-calibrated overhead terms (O_net, O_mem,
O_round):

    from realistic_model import PlatformConfig, RealisticParams, compute_ring_simple
    cfg = PlatformConfig.load('p5en')                       # hardware params (a, NIC BW)
    params = RealisticParams.from_json('.../overhead.json')  # calibrated overheads
    T = compute_ring_simple(params, sizes, n, nch, cfg.nic_bw_dual, cfg.a_us / 2)

Two config sources, mirroring roofline_model but kept separate:

* ``config.json`` – per-platform HARDWARE params (a, NIC bandwidth), loaded via
  :class:`PlatformConfig`. Independent of roofline_model's config.
* ``overhead.json`` – profiling-calibrated O_net/O_mem/O_round. The parsing/fitting/stats
  that produce it live in the :mod:`~realistic_model.extraction` submodule.

Public surface:

* :class:`PlatformConfig` – loads config.json; exposes a_us / nic_bw_dual.
* :class:`RealisticParams` – loads overhead.json; exposes O_net_eff_us / O_mem_us /
  O_round_us lookups.
* :func:`compute_ring_simple` – Ring/Simple realistic AllReduce time.
* :mod:`extraction` – parse .out -> fit -> assemble overhead.json (used by the
  extraction notebook; ``from realistic_model import extraction``).
"""

from .realistic import PlatformConfig, compute_ring_simple, PROTO_SIMPLE
from .overheads import RealisticParams

__all__ = [
    'PlatformConfig',
    'RealisticParams',
    'compute_ring_simple',
    'PROTO_SIMPLE',
]
