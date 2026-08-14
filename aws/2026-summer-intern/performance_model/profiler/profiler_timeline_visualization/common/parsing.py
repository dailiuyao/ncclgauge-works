"""Shared parsing primitives for NCCL per-chunk timeline .out files.

Platform- and algorithm-independent building blocks. No matplotlib dependency — this
module is pure text parsing, reusable by any consumer (timeline plots, analysis, tests).

Each per-algorithm notebook keeps its own algorithm-specific parsing (which sections
exist, the topology) but shares the generic pieces here:

    STAT, pick(blob)             per-chunk statistic selector from a 'med=..' summary
    split_test_cases(content)    split a .out file into per-message-size test-case blocks
    parse_msg_header(tc)         message label / bytes (case-insensitive)
    parse_info_fields(tc)        nranks / nchannels / protocol / nthreads / buffsize
    parse_nchunks(tc)            nchunks per channel
    parse_t_total(tc)            [T total] elapsed ms
    split_sections(tc, regex)    split a test case on a 2-group section-header regex
    findall_metric(section, re)  [(chunk_id, pick(value)), ...] for one bracket-tag metric
    extract_mode(filepath)       NCCL_TESTS_SPLIT_MASK / mode suffix from a filename
"""

import os
import re

STAT = 'med'   # 'med' | 'mean' | 'p95' | 'min' | 'max' — per-chunk statistic to read


def pick(blob, stat=STAT):
    """Extract the chosen STAT from a per-iteration summary value string
    ('mean=.. med=.. p95=.. min=.. max=..'). Returns 0.0 if the field is absent."""
    m = re.search(rf'{stat}=([-\d.]+)', blob)
    return float(m.group(1)) if m else 0.0


def split_test_cases(content, min_hashes=1):
    """Split a .out file into per-test-case blocks on the '# Test case:' banner.

    The leading element (file preamble) is kept as-is; callers guard with a header match
    (:func:`parse_msg_header` returns None) so it is skipped naturally. ``min_hashes`` sets
    how many leading '#' the banner line must have (1 = the ``#+`` used by new-format files)."""
    return re.split(rf'#{{{min_hashes},}}\n# Test case:', content)


def parse_msg_header(tc):
    """(msg_label, msg_bytes) from 'Message size = 1 MB (1048576 bytes)', case-insensitive.
    Returns (None, None) if the block has no message header (e.g. the file preamble)."""
    m = re.search(r'message size = ([^,]+) \((\d+) bytes\)', tc, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def parse_info_fields(tc):
    """Generic per-config fields from the INFO line + nthreads/buffsize.

    Returns dict {nranks, nchannels, protocol, nthreads, buffsize} (values None if absent)."""
    info = re.search(r'nranks\((\d+)\)_message_size\(\d+\)_nchannels\((\d+)\).*?_protocol\((\w+)\)', tc)
    nt = re.search(r'nthreads=(\d+)', tc)
    bf = re.search(r'buffsize=([^,]+)', tc)
    return {
        'nranks': int(info.group(1)) if info else None,
        'nchannels': int(info.group(2)) if info else None,
        'protocol': info.group(3) if info else 'unknown',
        'nthreads': int(nt.group(1)) if nt else None,
        'buffsize': bf.group(1).strip() if bf else None,
    }


def parse_nchunks(tc):
    """'nchunks per channel N' -> int or None."""
    m = re.search(r'nchunks per channel (\d+)', tc)
    return int(m.group(1)) if m else None


def parse_t_total(tc):
    """'[T total] ncclAllReduce elapsed time: X ms' -> float or None."""
    m = re.search(r'\[T total\] ncclAllReduce elapsed time: ([\d.]+) ms', tc)
    return float(m.group(1)) if m else None


def split_sections(tc, header_regex):
    """Split a test case on a two-group section-header regex.

    ``header_regex`` must capture exactly two integer groups (e.g. peer, channel). Returns
    a list of (group1, group2, body) tuples — one per matched section."""
    parts = re.split(header_regex, tc)
    out = []
    for i in range(1, len(parts), 3):
        out.append((int(parts[i]), int(parts[i + 1]), parts[i + 2]))
    return out


def findall_metric(section, pattern, stat=STAT):
    """All (chunk_id:int, pick(value)) pairs for one bracket-tag metric in a section body.

    ``pattern`` must capture (chunk_id, value_string); the value string is passed through
    :func:`pick`, which selects the STAT field from the summary."""
    return [(int(cid), pick(v, stat)) for cid, v in re.findall(pattern, section)]


def extract_mode(filepath):
    """Extract the NCCL_TESTS_SPLIT_MASK / mode suffix (e.g. '0x7') from a filename.
    Returns '' when the filename has no explicit mode suffix."""
    base = os.path.basename(filepath)
    m = re.search(r'vary-nch-?([\w]*)', base)
    if m and m.group(1):
        return m.group(1)
    m = re.search(r'(0x[0-9a-fA-F]+)', base)
    return m.group(1) if m else ''
