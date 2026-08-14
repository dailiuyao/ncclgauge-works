"""Roofline (alpha-beta) modelling package for NCCL AllReduce over EFA.

Public surface:

* I/O parsing   – :func:`parse_section_data`, :func:`extract_configs` (NCCL .out),
                  :func:`parse_size_str`, :func:`parse_bw_file`,
                  :func:`parse_nchannels_table` (GCP comparison files).
* Formatting    – :func:`fmt_size`, :func:`size_tick_formatter`, :func:`size_label`.
* Roofline math – platform-independent, grouped by topology: ``ring0x7_*`` / ``tree0x7_*``
                  (0x7, ring across nodes), ``ring0x0_*`` / ``tree0x0_*`` (0x0, all GPUs on
                  one ring), and ``gcp_*`` (same 0x0 math with a per-size nchannels array;
                  ``gcp_ag_*`` = AllGather). Pass ``cfg=PlatformConfig.load(...)`` to select
                  a platform. Plus :class:`PlatformConfig` and :func:`busbw_factor`.
"""

from .io_parsing import (
    parse_section_data,
    extract_configs,
    parse_size_str,
    parse_bw_file,
    parse_nchannels_table,
)
from .formatting import fmt_size, size_tick_formatter, size_label
from . import roofline
from .roofline import (
    PlatformConfig,
    busbw_factor,
    ring0x7_simple, ring0x7_ll, ring0x7_ll128,
    tree0x7_simple, tree0x7_ll, tree0x7_ll128,
    ring0x0_simple, ring0x0_ll, ring0x0_ll128,
    tree0x0_simple, tree0x0_ll, tree0x0_ll128,
    gcp_ring_simple_busbw, gcp_ring_ll_busbw, gcp_ring_ll128_busbw,
    gcp_tree_simple_busbw, gcp_tree_ll_busbw, gcp_tree_ll128_busbw,
    gcp_ag_ring_simple_busbw, gcp_ag_ring_ll_busbw, gcp_ag_ring_ll128_busbw,
)

__all__ = [
    'parse_section_data', 'extract_configs',
    'parse_size_str', 'parse_bw_file', 'parse_nchannels_table',
    'fmt_size', 'size_tick_formatter', 'size_label',
    'PlatformConfig',
    'busbw_factor', 'roofline',
    'ring0x7_simple', 'ring0x7_ll', 'ring0x7_ll128',
    'tree0x7_simple', 'tree0x7_ll', 'tree0x7_ll128',
    'ring0x0_simple', 'ring0x0_ll', 'ring0x0_ll128',
    'tree0x0_simple', 'tree0x0_ll', 'tree0x0_ll128',
    'gcp_ring_simple_busbw', 'gcp_ring_ll_busbw', 'gcp_ring_ll128_busbw',
    'gcp_tree_simple_busbw', 'gcp_tree_ll_busbw', 'gcp_tree_ll128_busbw',
    'gcp_ag_ring_simple_busbw', 'gcp_ag_ring_ll_busbw', 'gcp_ag_ring_ll128_busbw',
]
