"""Profiling-calibrated per-chunk overheads for the realistic model.

The realistic model augments the roofline with three measured overhead terms:

* ``O_net``  – network/firmware overhead beyond the ideal wire time
* ``O_mem``  – GPU data-preparation time
* ``O_round``– per-round drain cost

Their calibration is produced by the profiler (``ring_param_extraction`` notebook)
and serialized to ``overhead.json``. This module loads that JSON and exposes the
overhead lookups as methods on :class:`RealisticParams`, so the notebook no longer
carries the model code inline.
"""

import json
import os

import numpy as np


class RealisticParams:
    """Calibrated O_net / O_mem / O_round lookups loaded from ``overhead.json``.

    Construct via :meth:`from_json`. All lookups accept scalar or array ``bytes`` and
    return the overhead in microseconds, clipped to be non-negative.
    """

    def __init__(self, p):
        self._p = p
        self.O_NET_EFF_LINFIT = p.get('O_NET_EFF_LINFIT', {})
        self.G_MEM_GBS = p['G_MEM_GBS']
        self.O_MEM_FIT = p.get('O_MEM_FIT', {})
        self.O_ROUND_C = p['O_ROUND_C_us']
        self.O_ROUND_K = p['O_ROUND_K_us_per_MBch']
        self.NIC_BW_DUAL_BU = p.get('NIC_BW_DUAL_bytes_us', 50 * 1e3 * 0.975)
        self.STAT = p.get('STAT')
        self.n_configs = p.get('n_configs')

    @classmethod
    def from_json(cls, path):
        """Load calibrated parameters. Raises FileNotFoundError if the JSON is missing
        (there is no hardcoded fallback — regenerate it with the profiler notebook)."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Calibrated parameters not found: {path}\n"
                f"Run ring_param_extraction.ipynb first to generate overhead.json.")
        with open(path) as f:
            return cls(json.load(f))

    @property
    def source(self):
        return f"(STAT={self.STAT}, n_configs={self.n_configs})"

    # ------------------------------------------------------------------ #
    # O_net
    # ------------------------------------------------------------------ #
    def O_net_eff_us(self, proto, slice_bytes, nch):
        """Effective per-chunk exposed network overhead (us) from the TWO-SEGMENT o_net_eff
        fit (O_NET_EFF_LINFIT), CONTINUOUS at the breakpoint:
          chunk_KB <  breakpoint -> lo_const                          (flat plateau)
          chunk_KB >= breakpoint -> lo_const + hi_slope*(chunk_KB-bp)  (line from plateau)
        Already concurrency-divided. Nearest-nch; clipped >= 0."""
        tab = self.O_NET_EFF_LINFIT.get(proto, {})
        sb = np.asarray(slice_bytes, dtype=float)
        if not tab:
            return np.zeros_like(sb)
        key = str(nch) if str(nch) in tab else min(tab.keys(), key=lambda k: abs(int(k) - nch))
        r = tab[key]
        skb = sb / 1024.0
        bp = r['breakpoint_KB']
        lo = r['lo_const_us']
        hi = lo + r['hi_slope_us_per_KB'] * (skb - bp)
        return np.maximum(np.where(skb < bp, lo, hi), 0.0)

    # ------------------------------------------------------------------ #
    # O_mem / O_round
    # ------------------------------------------------------------------ #
    def O_mem_us(self, proto, slice_bytes, nch=1):
        """Mem-copy time per slice (us): recv FIFO -> send FIFO. LINEAR fit k*slice_KB + b
        (O_MEM_FIT). nch 8/16 have a low-chunk PLATEAU below nch_plateau_thr_KB (held at the
        line value at the threshold, continuous join). Clipped >= 0."""
        sb = np.asarray(slice_bytes, dtype=float)
        fit = self.O_MEM_FIT.get(proto)
        if not fit:   # fallback: old slope-only through-origin model
            return sb / (self.G_MEM_GBS[proto] * 1e3)
        k = fit['k_us_per_KB']
        b = fit['b_us']
        skb = sb / 1024.0
        val = k * skb + b
        thr = fit.get('nch_plateau_thr_KB', {}).get(str(nch),
                                                    fit.get('nch_plateau_thr_KB', {}).get(nch))
        if thr is not None:
            val = np.where(skb < thr, k * thr + b, val)
        return np.maximum(val, 0.0)

    def O_round_us(self, proto, slice_bytes, nch):
        """Per-round drain overhead (us): O_ROUND_C + O_ROUND_K*(slice_MB*nch)."""
        slice_bytes = np.asarray(slice_bytes, dtype=float)
        slice_MB = slice_bytes / (1024**2)
        return self.O_ROUND_C + self.O_ROUND_K * (slice_MB * nch)
