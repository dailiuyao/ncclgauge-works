#!/usr/bin/env python3
"""Analyze the channel-verification sweep logs.

For each (mask, algo, proto) log, extracts:
  N      = comm->nChannels (from "Channel XX/YY" init lines; YY is the ceiling)
  nRanks = communicator size
  For each op:  (size_bytes, measured nChannels) from rank-0 TUNING line
      "<Coll>: <N> Bytes -> Algo <A> proto <P> channel{Lo..Hi}={lo..hi}"

Prints per-config table and compares against the prediction:
  cost = nt * threshold, where
    nt (maxThreads)  = Simple 512, LL 512, LL128 640   (tuning.cc:239-251)
    threshold        = Simple 64,  LL 8,   LL128 8     (comm.h:49-51)
    Ring+LL threshold multiplied by nRanks             (tuning.cc:556)
  Ring+Simple gets an extra +32 threads for sync (enqueue.cc:2039), but that
  only changes nt after the loop finishes — for the loop itself the base
  maxThreads applies. We use base values here.
  predicted_nc(S) = clamp(floor(S / cost), 1, N)
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# --- constants from source ---
MAX_THREADS = {"Simple": 512, "LL": 512, "LL128": 640}
THRESHOLDS  = {"Simple": 64,  "LL": 8,   "LL128": 8}

# --- regexes ---
# Init "Channel 00/08 : 0 1 2 3 ..." — the /YY is comm->nChannels.
RE_INIT_CHAN = re.compile(r"NCCL INFO Channel (\d+)/(\d+)\s*:")
# "nranks=32" or "comm 0x... rank X nranks Y ..."
RE_NRANKS = re.compile(r"nranks[=\s]+(\d+)")
# rank-0 TUNING per-op line — captures func, bytes, algo, proto, lo, hi.
RE_TUNING = re.compile(
    r"NCCL INFO (\w+):\s*(\d+)\s*Bytes\s*->\s*Algo\s+(\w+)\s+proto\s+(\w+)\s+channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}"
)

@dataclass
class LogSummary:
    tag: str
    path: str
    N: int | None = None
    nRanks: int | None = None
    ops: list[tuple[str, int, str, str, int]] = field(default_factory=list)  # (func, size, algo, proto, nc)

def parse_log(path: str) -> LogSummary:
    tag = os.path.basename(path).replace(".log", "")
    s = LogSummary(tag=tag, path=path)
    max_slash = 0
    with open(path, errors="ignore") as f:
        for line in f:
            if s.N is None or "Channel" in line:
                m = RE_INIT_CHAN.search(line)
                if m:
                    v = int(m.group(2))
                    if v > max_slash:
                        max_slash = v
            if s.nRanks is None:
                m = RE_NRANKS.search(line)
                if m:
                    s.nRanks = int(m.group(1))
            m = RE_TUNING.search(line)
            if m:
                func = m.group(1)
                size = int(m.group(2))
                algo = m.group(3).capitalize()   # RING -> Ring
                proto_raw = m.group(4)
                # SIMPLE -> Simple, LL -> LL, LL128 -> LL128
                if proto_raw.upper() == "SIMPLE":
                    proto = "Simple"
                elif proto_raw.upper() == "LL":
                    proto = "LL"
                elif proto_raw.upper() == "LL128":
                    proto = "LL128"
                else:
                    proto = proto_raw
                lo, hi = int(m.group(5)), int(m.group(6))
                nc = hi - lo + 1
                s.ops.append((func, size, algo, proto, nc))
    if max_slash:
        s.N = max_slash
    return s

def predict_layerA(size: int, algo: str, proto: str, N: int, nRanks: int) -> int:
    """Layer-A prediction from channel_selection_verification.md §2."""
    nt = MAX_THREADS[proto]
    thr = THRESHOLDS[proto]
    if algo == "Ring" and proto == "LL":
        thr = thr * nRanks
    cost = nt * thr
    return max(1, min(N, size // cost))

def predict_phase2_allreduce(size: int, proto: str, N: int) -> int:
    """Phase-2 traffic-cell rebuild in enqueue.cc:652-668.
    For AllReduce, trafficPerByte = 2. LL bumps it to 8.
    cellSize = MinTrafficPerChannel(32K) / trafficPerByte -> 16K for Simple/LL128, 4K for LL.
    Approx phase-2 cap = ceil(size / cellSize), clamped to N."""
    tpb = 2  # AllReduce
    if proto == "LL":
        tpb = 2 * 4
    # cellSize = divUp(divUp(32K, tpb), 16) * 16
    import math
    cell = math.ceil(math.ceil((32 << 10) / tpb) / 16) * 16
    cells = math.ceil(size / cell) if size > 0 else 1
    return max(1, min(N, cells))

def predict(size: int, algo: str, proto: str, N: int, nRanks: int) -> int:
    return min(
        predict_layerA(size, algo, proto, N, nRanks),
        predict_phase2_allreduce(size, proto, N),
    )

def human(n: int) -> str:
    for unit, div in (("G", 1<<30), ("M", 1<<20), ("K", 1<<10)):
        if n >= div:
            v = n / div
            return f"{v:g}{unit}"
    return f"{n}B"

def summarize_one(s: LogSummary, first_only: bool = True) -> None:
    """Print measured nChannels vs prediction, one row per unique message size.
    first_only: since -n 20 -w 5 means the same op logs ~20 times, dedupe to first."""
    print()
    print(f"=== {s.tag} ===")
    print(f"    N (comm->nChannels)     = {s.N}")
    print(f"    nRanks                  = {s.nRanks}")
    if not s.ops:
        print("    (no per-op TUNING lines found)")
        return

    # Dedupe by size, keeping first sighting per size for the AllReduce op.
    ar = [op for op in s.ops if op[0] == "AllReduce"]
    if not ar:
        # fall back to all ops if the log has no AllReduce func label (shouldn't happen)
        ar = s.ops
    seen: dict[int, tuple[str, str, int]] = {}
    for func, size, algo, proto, nc in ar:
        if first_only and size in seen:
            continue
        seen[size] = (algo, proto, nc)

    print(f"    {'size':>10} {'algo':>6} {'proto':>6} {'meas':>5} {'pred':>5} {'match':>6}")
    print(f"    {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")
    n_match = n_total = 0
    for size in sorted(seen.keys()):
        algo, proto, nc = seen[size]
        if s.N and s.nRanks and proto in MAX_THREADS:
            p = predict(size, algo, proto, s.N, s.nRanks)
            match = "OK" if p == nc else f"d={nc-p:+d}"
        else:
            p = -1
            match = "?"
        if isinstance(match, str) and match == "OK":
            n_match += 1
        n_total += 1
        print(f"    {human(size):>10} {algo:>6} {proto:>6} {nc:>5} {p:>5} {match:>6}")
    print(f"    -> {n_match}/{n_total} match")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p5-en/channel_verify_AR")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, "*.log")))
    if not paths:
        print(f"No .log files in {args.dir}", file=sys.stderr)
        sys.exit(1)

    summaries = [parse_log(p) for p in paths]

    print("=" * 70)
    print("Per-mask N summary")
    print("=" * 70)
    by_mask = defaultdict(list)
    for s in summaries:
        m = re.match(r"mask(0x[0-9a-fA-F]+)_", s.tag)
        if m:
            by_mask[m.group(1)].append(s)
    for mask, group in sorted(by_mask.items()):
        Ns = sorted({s.N for s in group if s.N is not None})
        nRs = sorted({s.nRanks for s in group if s.nRanks is not None})
        print(f"  {mask}: N={Ns}, nRanks={nRs}, configs={len(group)}")

    for s in summaries:
        summarize_one(s)

if __name__ == "__main__":
    main()
