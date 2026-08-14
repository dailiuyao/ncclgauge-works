"""Parsers for the raw measurement files used by the roofline/realistic notebooks.

Two input families:

1. NCCL ``nccl-tests`` ``.out`` dumps (p5en/p6 AllReduce runs). Each file contains
   several config sections, each headed by an ``NCCL_ALGO=..., NCCL_PROTO=...,
   NCCL_MIN_NCHANNELS=...`` line followed by the standard nccl-tests table.

2. GCP-vs-AWS comparison text files (``gcp_AR_*.txt`` / ``gcp_AG_*.txt``) plus the
   markdown ``P6-b200_nchannels_table.txt``.

"""

import re

import pandas as pd


# --------------------------------------------------------------------------- #
# NCCL nccl-tests .out parsing
# --------------------------------------------------------------------------- #
def parse_section_data(section_text):
    """Extract (sizes, times, algbws, busbws) from one nccl-tests config section.

    Returns four parallel lists (bytes, us, GB/s, GB/s).
    """
    sizes, times, algbws, busbws = [], [], [], []
    for line in section_text.split('\n'):
        m = re.match(r'\s+(\d+)\s+\d+\s+\w+\s+\w+\s+\S+\s+'
                     r'[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+'
                     r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+', line)
        if m:
            sizes.append(int(m.group(1)))
            times.append(float(m.group(2)))
            algbws.append(float(m.group(3)))
            busbws.append(float(m.group(4)))
    return sizes, times, algbws, busbws


def extract_configs(algo, proto, sections):
    """All channel configs for a given (algo, proto) from a list of text sections.

    ``sections`` is the list of config-section strings for one node scale (each the
    text between ``NCCL_ALGO=...`` headers). Returns ``{nch: DataFrame}`` with columns
    size_bytes, time_us, algbw_GBs, busbw_GBs, size_MB.
    """
    data = {}
    for sec in sections:
        pattern = rf'NCCL_ALGO={algo}, NCCL_PROTO={proto}, NCCL_MIN_NCHANNELS=(\d+)'
        cfg_match = re.search(pattern, sec)
        if not cfg_match:
            continue
        nch = int(cfg_match.group(1))
        sizes, times, algbws, busbws = parse_section_data(sec)
        if sizes:
            dft = pd.DataFrame({'size_bytes': sizes, 'time_us': times,
                                'algbw_GBs': algbws, 'busbw_GBs': busbws})
            dft['size_MB'] = dft['size_bytes'] / (1024**2)
            data[nch] = dft
    return data


# --------------------------------------------------------------------------- #
# GCP-vs-AWS comparison files
# --------------------------------------------------------------------------- #
def parse_size_str(s):
    """Convert a size string to bytes.

    Accepts both spaced/full-unit forms ('1 KB', '2 MB', '1 GB') AND the compact,
    single-letter form used in P6-b200_nchannels_table.txt ('1K', '128K', '1M').
    The optional space and the trailing 'B' are both tolerated. Returns None on no match.
    """
    s = s.strip()
    m = re.match(r'^([\d.]+)\s*([KMGT]?)B?$', s, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    multipliers = {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    return int(val * multipliers[m.group(2).upper()])


def parse_bw_file(filepath):
    """Parse a gcp_AR_*.txt / gcp_AG_*.txt file.

    Returns a DataFrame with size_bytes, aws_busbw, gcp_busbw (busbw None when '—'/'-').
    Rows are tab-separated: <size> <aws> <gcp>.
    """
    rows = []
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        parts = [p for p in parts if p.strip()]   # handle leading/blank tabs
        if len(parts) < 3:
            continue
        size_bytes = parse_size_str(parts[0].strip())
        if size_bytes is None:
            continue
        try:
            aws_bw = float(parts[1]) if parts[1].strip() not in ('—', '-', '') else None
        except ValueError:
            aws_bw = None
        try:
            gcp_bw = float(parts[2]) if parts[2].strip() not in ('—', '-', '') else None
        except ValueError:
            gcp_bw = None
        rows.append({'size_bytes': size_bytes, 'aws_busbw': aws_bw, 'gcp_busbw': gcp_bw})
    return pd.DataFrame(rows)


def parse_nchannels_table(filepath, section='AllReduce'):
    """Parse P6-b200_nchannels_table.txt for one section ('AllReduce' or 'AllGather').

    Returns ``{column_name: {size_bytes: nchannels}}`` for every protocol/config column
    (e.g. '0x7 Ring Simple'). The size column uses the compact '1K'/'1M' form handled
    by parse_size_str.
    """
    with open(filepath) as f:
        content = f.read()
    sections = re.split(r'###\s+', content)
    target = None
    for sec in sections:
        if sec.strip().startswith(section):
            target = sec
            break
    if target is None:
        return {}
    lines = target.strip().split('\n')
    table_lines = [l for l in lines if l.strip().startswith('|')]
    if len(table_lines) < 3:
        return {}
    header = [h.strip() for h in table_lines[0].split('|') if h.strip()]
    result = {h: {} for h in header[1:]}
    for line in table_lines[2:]:
        cols = [c.strip() for c in line.split('|') if c.strip()]
        if len(cols) < 2:
            continue
        size_bytes = parse_size_str(cols[0])
        if size_bytes is None:
            continue
        for i, h in enumerate(header[1:], start=1):
            if i < len(cols):
                try:
                    result[h][size_bytes] = int(cols[i])
                except ValueError:
                    pass
    return result
