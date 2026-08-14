#!/usr/bin/env python3
"""Analyze the AllGather channel-verification sweep logs.

Key differences from the AR analyzer:

1. Rank-0 NCCL_TUNING line at enqueue.cc:794-796 prints
       "AllGather: <task->count * eltSize> Bytes -> Algo <A> proto <P> channel{...}"
   For AG, task->count is the PER-RANK sendcount (nccl-tests divides -b S by
   nRanks at all_gather.cu:11 before calling ncclAllGather). So the "Bytes"
   value in the log = S / nRanks, NOT the user's -b argument.

   To reconstruct S from the log:  S = printed_bytes * nRanks.
   NCCL internally uses nBytes = ncclFuncMaxSendRecvCount(count, AG) =
   nRanks * task->count * eltSize = S — so predictions use S.

2. AG has NO Tree algorithm. All *_algoTree_* configs fail with
   'invalid usage'. Skip them.

3. Phase-2 trafficPerByte for AG = nRanks (not 2 as for AR). With
   nRanks=128 the cellSize collapses to 16 B, so Phase-2 essentially
   never constrains AG — Layer-A alone decides.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

MAX_THREADS = {"Simple": 512, "LL": 512, "LL128": 640}
THRESHOLDS  = {"Simple": 64,  "LL": 8,   "LL128": 8}

RE_INIT_CHAN = re.compile(r"NCCL INFO Channel (\d+)/(\d+)\s*:")
RE_NRANKS = re.compile(r"nranks[=\s]+(\d+)")
RE_TUNING = re.compile(
    r"NCCL INFO (\w+):\s*(\d+)\s*Bytes\s*->\s*Algo\s+(\w+)\s+proto\s+(\w+)\s+channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}"
)

@dataclass
class LogSummary:
    tag: str
    path: str
    N: int | None = None
    nRanks: int | None = None
    ops: list[tuple[str, int, str, str, int]] = field(default_factory=list)

def parse_log(path: str) -> LogSummary:
    tag = os.path.basename(path).replace(".log", "")
    s = LogSummary(tag=tag, path=path)
    max_slash = 0
    with open(path, errors="ignore") as f:
        for line in f:
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
                algo = m.group(3).capitalize()
                proto_raw = m.group(4).upper()
                proto = {"SIMPLE": "Simple", "LL": "LL", "LL128": "LL128"}.get(proto_raw, proto_raw)
                lo, hi = int(m.group(5)), int(m.group(6))
                nc = hi - lo + 1
                s.ops.append((func, size, algo, proto, nc))
    if max_slash:
        s.N = max_slash
    return s

def predict_layerA(nBytes: int, algo: str, proto: str, N: int, nRanks: int) -> int:
    if proto not in MAX_THREADS:
        return -1
    nt = MAX_THREADS[proto]
    thr = THRESHOLDS[proto]
    if algo == "Ring" and proto == "LL":
        thr *= nRanks
    cost = nt * thr
    return max(1, min(N, nBytes // cost))

def predict_phase2_AG(S: int, proto: str, N: int, nRanks: int) -> int:
    """Phase-2 (enqueue.cc:652-668) uses:
       cells = divUp(task->count * elementSize, cellSize)
    where task->count is the AG SEND count (per-rank), so we pass S/nRanks.
    trafficPerByte for AG = nRanks; LL multiplies by 4.
    cellSize = divUp(divUp(32K, tpb), 16) * 16.
    """
    perRankBytes = S // nRanks
    tpb = nRanks
    if proto == "LL":
        tpb *= 4
    cell = math.ceil(math.ceil((32 << 10) / tpb) / 16) * 16
    cells = math.ceil(perRankBytes / cell) if perRankBytes > 0 else 1
    return max(1, min(N, cells))

def predict(S: int, algo: str, proto: str, N: int, nRanks: int) -> int:
    """Layer-A takes nBytes = S (total); Phase-2 takes per-rank bytes = S/nRanks."""
    return min(predict_layerA(S, algo, proto, N, nRanks),
               predict_phase2_AG(S, proto, N, nRanks))

def human(n: int) -> str:
    for unit, div in (("G", 1<<30), ("M", 1<<20), ("K", 1<<10)):
        if n >= div:
            return f"{n/div:g}{unit}"
    return f"{n}B"

def _kb(n):  return n << 10
def _mb(n):  return n << 20
def _gb(n):  return n << 30

# Per-protocol size ranges requested for this sweep (0x7 mask only).
PROTO_SIZES = {
    "Simple": [_kb(1), _kb(2), _kb(4), _kb(8), _kb(16), _kb(32), _kb(64), _kb(128),
               _kb(256), _kb(512), _mb(1), _mb(2), _mb(4), _mb(8), _mb(16), _mb(32),
               _mb(64), _mb(128), _mb(256), _mb(512), _gb(1), _gb(2), _gb(4), _gb(8)],
    "LL":     [_kb(1), _kb(2), _kb(4), _kb(8), _kb(16), _kb(32), _kb(64), _kb(128),
               _kb(256), _kb(512), _mb(1), _mb(2), _mb(4), _mb(8), _mb(16), _mb(32),
               _mb(64), _mb(128)],
    "LL128":  [_kb(8), _kb(16), _kb(32), _kb(64), _kb(128), _kb(256), _kb(512),
               _mb(1), _mb(2), _mb(4), _mb(8), _mb(16), _mb(32), _mb(64), _mb(128)],
}

def summarize_one(s: LogSummary) -> None:
    # Only analyze 0x7 configs — 0x0 is skipped, Tree fails (AG has no Tree).
    if "mask0x7" not in s.tag:
        return

    print()
    print(f"=== {s.tag} ===")
    print(f"    N (comm->nChannels)     = {s.N}")
    print(f"    nRanks                  = {s.nRanks}")
    if not s.ops:
        print("    (no per-op TUNING lines found)")
        return

    ag = [op for op in s.ops if op[0] == "AllGather"]
    if not ag:
        ag = s.ops

    # log's "Bytes" = per-rank bytes; user-input S = per_rank * nRanks
    seen: dict[int, tuple[str, str, int]] = {}
    for func, per_rank_bytes, algo, proto, nc in ag:
        S = per_rank_bytes * (s.nRanks or 1)
        if S in seen:
            continue
        seen[S] = (algo, proto, nc)

    # Filter to the per-protocol size list requested by the user.
    proto_in_tag = None
    for p in ("LL128", "Simple", "LL"):   # order matters: LL128 before LL
        if f"proto{p}" in s.tag:
            proto_in_tag = p
            break
    wanted = set(PROTO_SIZES.get(proto_in_tag, []))

    print(f"    {'-b S':>10} {'algo':>6} {'proto':>6} {'meas':>5} {'pred':>5} {'match':>6}")
    print(f"    {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")
    n_match = n_total = 0
    for S in sorted(wanted):
        if S not in seen:
            print(f"    {human(S):>10} {'?':>6} {'?':>6} {'-':>5} {'-':>5} {'MISS':>6}")
            n_total += 1
            continue
        algo, proto, nc = seen[S]
        if s.N and s.nRanks and proto in MAX_THREADS:
            p = predict(S, algo, proto, s.N, s.nRanks)
            match = "OK" if p == nc else f"d={nc-p:+d}"
        else:
            p = -1
            match = "?"
        if match == "OK":
            n_match += 1
        n_total += 1
        print(f"    {human(S):>10} {algo:>6} {proto:>6} {nc:>5} {p:>5} {match:>6}")
    print(f"    -> {n_match}/{n_total} match")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/liuyaod/netgauge-test/ncclguage/aws/2026-summer-intern/out/p5-en/channel_verify_AG")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, "*.log")))
    if not paths:
        print(f"No .log files in {args.dir}", file=sys.stderr)
        sys.exit(1)

    summaries = [parse_log(p) for p in paths]

    print("=" * 70)
    print("0x7 sub-comm summary (AG, 0x7 Ring × {Simple, LL, LL128})")
    print("=" * 70)
    group = [s for s in summaries if "mask0x7" in s.tag]
    Ns = sorted({s.N for s in group if s.N is not None})
    nRs = sorted({s.nRanks for s in group if s.nRanks is not None})
    print(f"  N={Ns}, nRanks={nRs}, configs={len(group)}")

    for s in summaries:
        summarize_one(s)

if __name__ == "__main__":
    main()
