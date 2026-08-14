"""Shared primitives for NCCL per-chunk timeline notebooks (any algorithm / platform).

Two independent submodules — import from whichever you need so parsing stays matplotlib-free:

* :mod:`common.parsing` – pure text parsing of the .out format (no matplotlib):
  ``STAT``, ``pick``, ``split_test_cases``, ``parse_msg_header``, ``parse_info_fields``,
  ``parse_nchunks``, ``parse_t_total``, ``split_sections``, ``findall_metric``, ``extract_mode``.
* :mod:`common.plotting` – generic matplotlib timeline helpers:
  ``fmt_bytes``, ``TimelineStyle``, ``draw_group_background``, ``draw_chunk_label``,
  ``draw_separator``, ``finalize_timeline_axes``, ``save_and_show``.

Each per-algorithm notebook keeps its own algorithm-specific parsing (topology, section
layout) and plot body (bar placement); these are the pieces they share.
"""

from . import parsing
from . import plotting

__all__ = ['parsing', 'plotting']
