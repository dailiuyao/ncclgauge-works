"""Parameter extraction for the realistic model.

Turns the raw NCCL profiling dump into the
calibrated overhead parameters consumed by :class:`~realistic_model.overheads.RealisticParams`
(serialized to ``overhead.json``). All the parsing / fitting / statistics live here; the
extraction notebook only calls these functions, plots, and writes the JSON.

Scope: Ring / Simple only. For the Simple protocol every chunk-byte convention (on-wire,
slice, O_round) collapses to ``buffsize/4``, so a single :func:`chunk_bytes` suffices.

Pipeline:
    rank0     = parse_ring_file(path)                  # per-config per-chunk timings
    param_df  = build_param_df(rank0)                  # O_net / O_mem / O_round per config
    onet_fit  = fit_onet_eff(rank0)                    # two-segment linear o_net_eff fit per nch
    omem_fit, gmem = fit_omem(param_df)                # O_mem linear fit + plateau
    oround    = fit_oround(param_df)                   # O_round bilinear fit
    params    = assemble_params(param_df, onet_fit, omem_fit, gmem, oround)
    write_overhead_json(params, '.../overhead.json')  # serialize the calibration

Or the whole pipeline in one call (used by both notebooks)::

    res = build_overhead('.../nccl_..._0x7.out', '.../overhead.json')
    # res = {'params', 'param_df', 'onet_eff', 'omem_fit', 'gmem', 'oround'}
"""

import json
import re
from math import erf, sqrt

import numpy as np
import pandas as pd

from .realistic import PlatformConfig

# --------------------------------------------------------------------------- #
# Constants (Simple protocol, p5en dual-EFA)
# --------------------------------------------------------------------------- #
PROTO = 'Simple'
STAT_DEFAULT = 'med'                 # per-chunk statistic read from the summary lines
NIC_BW_DUAL = PlatformConfig.load('p5en').nic_bw_dual   # bytes/us, from config.json (single source)
MIN_RECVDONE_TO_DR_N = 50            # drop chunks with <50 iter samples (unreliable median)
NT_FIT = 512                         # nthreads config used for all fits
EXCLUDE_BUF = {'16MB'}               # largest buffsize (chunk > representative cap) excluded from fits
O_MEM_BIG_KB = 256.0                 # "large chunk" cutoff for the O_mem shared-slope pool


# --------------------------------------------------------------------------- #
# .out parsing
# --------------------------------------------------------------------------- #
def _pick(blob, stat=STAT_DEFAULT):
    """Extract chosen STAT from 'mean=.. med=.. p95=..', or a bare float (old format)."""
    m = re.search(rf'{stat}=([-\d.]+)', blob)
    if m:
        return float(m.group(1))
    m = re.search(r'([-\d.]+)', blob)
    return float(m.group(1)) if m else 0.0


def parse_ring_file(filepath, stat=STAT_DEFAULT):
    """Parse a ring timeline .out file into a list of config dicts (sorted by msg_bytes).

    Each config has metadata (protocol, nchannels, nthreads, buffsize, ...) and a per-send
    list of per-chunk timing dicts, including the C-gauge fields ``onet_gauge_ms`` /
    ``onet_eff_gauge_ms`` and the ``recvdone_to_dr`` mem-copy timings.
    """
    with open(filepath) as f:
        content = f.read()
    cases = []
    for tc in re.split(r'#+\n# Test case:', content):
        mm = re.search(r'Message size = ([^,]+) \((\d+) bytes\)', tc)
        if not mm:
            continue
        msg_label, msg_bytes = mm.group(1), int(mm.group(2))
        rh = re.search(r'\[Rank (\d+) @ ([^\]]+)\]', tc)
        rank_id = int(rh.group(1)) if rh else None
        info = re.search(r'nranks\((\d+)\)_message_size\(\d+\)_nchannels\((\d+)\).*?_protocol\((\w+)\)', tc)
        nranks = int(info.group(1)) if info else None
        nchannels = int(info.group(2)) if info else None
        protocol = info.group(3) if info else 'unknown'
        nt = re.search(r'nthreads=(\d+)', tc); nthreads = int(nt.group(1)) if nt else None
        bf = re.search(r'buffsize=([^,]+)', tc); buffsize = bf.group(1).strip() if bf else None
        nck = re.search(r'nchunks per channel (\d+)', tc); ncpc = int(nck.group(1)) if nck else None
        rxdone = {int(cc): _pick(vv, stat) for cc, vv in
                  re.findall(r'\[ABS_RX_DONE\] chunk (\d+) Irecv done\s+@ func_start \+\s*(.+?) ms', tc)}
        _rdur = {}
        for _ch, _ck, _rl, _v in re.findall(r'\[RAIL_DUR\]\s+channel (\d+) peer \d+ chunk (\d+) rail (\d+) end - start\s*:\s*(.+?) ms', tc):
            _rdur[(int(_ch), int(_ck), int(_rl))] = _pick(_v, stat)

        def _rail_maxdur_ms(ch, cid):
            durs = [_rdur[(ch, cid, _rl)] for _rl in (0, 1)
                    if _rdur.get((ch, cid, _rl)) is not None and _rdur[(ch, cid, _rl)] > 0]
            return max(durs) if durs else None

        secs = re.split(r'-- --- send to peer (\d+) in channel (\d+) ---', tc)
        sends = []
        for i in range(1, len(secs), 3):
            peer, channel, section = int(secs[i]), int(secs[i + 1]), secs[i + 2]
            post_gap = [(c, _pick(v, stat)) for c, v in re.findall(r'\[POST_GAP\] chunk (\d+) post gap from chunk 0:\s*(.+?) ms', section)]
            post_to_dr = [(c, _pick(v, stat)) for c, v in re.findall(r'\[POST_TO_DR\] chunk (\d+) posted .+? data ready:\s*(.+?) ms', section)]
            dr_to_tx = [(c, _pick(v, stat)) for c, v in re.findall(r'\[DR_TO_TX\] chunk (\d+) data ready .+? transmitted:\s*(.+?) ms', section)]
            post_to_recvdone = {int(c): _pick(v, stat) for c, v in re.findall(r'\[POST_TO_RECVDONE\] chunk (\d+) posted .+? recv done\s*:\s*(.+?) ms', section)}
            recvdone_to_dr = {int(c): _pick(v, stat) for c, v in re.findall(r'\[RECVDONE_TO_DR\] chunk (\d+) recv done .+? data ready\s*:\s*(.+?) ms', section)}
            recvdone_to_dr_n = {int(c): int(nn) for c, nn in re.findall(r'\[RECVDONE_TO_DR\] chunk (\d+) recv done .+? data ready\s*:.+? ms \(n=(\d+)\)', section)}
            tx_to_done = [(c, _pick(v, stat)) for c, v in re.findall(r'\[TX_TO_DONE\] chunk (\d+) transmitted .+? done:\s*(.+?) ms', section)]
            atx = [(c, _pick(v, stat)) for c, v in re.findall(r'\[ABS_TX\] chunk (\d+) transmitted @ func_start \+\s*(.+?) ms', section)]
            onet_g = {int(c): _pick(v, stat) for c, v in re.findall(r'\[O_NET\] chunk (\d+) fluid o_net .+?:\s*(.+?) ms', section)}
            oneteff_g = {int(c): _pick(v, stat) for c, v in re.findall(r'\[O_NET_EFF\] chunk (\d+) o_net / overlap_per_ch:\s*(.+?) ms', section)}
            keff_g = {int(c): _pick(v, stat) for c, v in re.findall(r'\[K_EFF\] chunk (\d+) concurrency on o_net window:\s*(.+?) \(n=', section)}
            chunks = []
            for j in range(len(post_gap)):
                cid = int(post_gap[j][0])
                chunks.append({
                    'chunk_id': cid,
                    'post_to_dr': post_to_dr[j][1] if j < len(post_to_dr) else 0,
                    'dr_to_tx': dr_to_tx[j][1] if j < len(dr_to_tx) else 0,
                    'tx_to_done': tx_to_done[j][1] if j < len(tx_to_done) else 0,
                    'post_to_recvdone': post_to_recvdone.get(cid),
                    'recvdone_to_dr': recvdone_to_dr.get(cid),
                    'recvdone_to_dr_n': recvdone_to_dr_n.get(cid),
                    'abs_tx': atx[j][1] if j < len(atx) else None,
                    'abs_rx_done': rxdone.get(cid),
                    'rail_maxdur_ms': _rail_maxdur_ms(channel, cid),
                    'onet_gauge_ms': onet_g.get(cid),
                    'onet_eff_gauge_ms': oneteff_g.get(cid),
                    'keff_gauge': keff_g.get(cid),
                })
            sends.append({'peer': peer, 'channel': channel, 'chunks': chunks})
        cases.append({'msg_label': msg_label, 'msg_bytes': msg_bytes, 'nranks': nranks,
                      'nchannels': nchannels, 'nchunks_per_channel': ncpc, 'protocol': protocol,
                      'nthreads': nthreads, 'buffsize': buffsize,
                      'rank_id': rank_id, 'sends': sends})
    cases.sort(key=lambda c: c['msg_bytes'])
    return cases


# --------------------------------------------------------------------------- #
# byte / chunk helpers (Simple: on-wire == slice == buffsize/4)
# --------------------------------------------------------------------------- #
def buf_bytes(bs):
    """Parse a buffsize string ('8MB', '512KB') to bytes."""
    m = re.match(r'([\d.]+)\s*([KMG]?B)', bs.strip())
    return float(m.group(1)) * {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}[m.group(2)]


def chunk_bytes(buffsize):
    """On-wire / slice bytes per chunk for Simple = buffsize/4."""
    return buf_bytes(buffsize) / 4


# --------------------------------------------------------------------------- #
# param_df: O_net (gauge median), O_mem (RECVDONE_TO_DR interval union), O_round
# --------------------------------------------------------------------------- #
def _steady(chunks):
    ids = [c['chunk_id'] for c in chunks]
    if not ids:
        return []
    nmax = max(ids)
    lo, hi = (12, 24) if nmax >= 24 else (max(3, nmax // 4), nmax - 1)
    return [c for c in chunks if lo <= c['chunk_id'] < hi]


def build_param_df(rank0, proto=PROTO):
    """Extract O_net / O_mem / O_round per config into a DataFrame (``proto`` only).

    * O_net_gauge_us = median of the O_net over all chunks/channels.
    * O_mem_model_us = union of RECVDONE_TO_DR intervals (overlap counted once) / #chunks,
      averaged over channels (drops chunks with <MIN_RECVDONE_TO_DR_N samples).
    * O_round_raw_us = atx[24]-atx[23] on channel 0 (round-boundary send interval).
    """
    rows = []
    for c in rank0:
        if c['protocol'] != proto or not c['sends']:
            continue
        ch = sorted(c['sends'], key=lambda s: s['channel'])[0]['chunks']
        if not _steady(ch):
            continue
        cbytes = chunk_bytes(c['buffsize'])
        twire_us = cbytes * c['nchannels'] / NIC_BW_DUAL

        atx = {x['chunk_id']: x['abs_tx'] for x in ch if x['abs_tx'] is not None}
        O_round_raw = (atx[24] - atx[23]) * 1e3 if (23 in atx and 24 in atx) else np.nan

        _omem_per_ch = []
        for _sg in c['sends']:
            ivs = []
            for x in _sg['chunks']:
                if (x['chunk_id'] < 24 and x.get('abs_rx_done') is not None
                        and x.get('recvdone_to_dr') is not None
                        and (x.get('recvdone_to_dr_n') or 0) >= MIN_RECVDONE_TO_DR_N):
                    st = x['abs_rx_done'] * 1e3
                    ivs.append((st, st + x['recvdone_to_dr'] * 1e3))
            if not ivs:
                continue
            ivs.sort()
            union = 0.0; cur_s, cur_e = ivs[0]
            for st, en in ivs[1:]:
                if st > cur_e:
                    union += cur_e - cur_s; cur_s, cur_e = st, en
                else:
                    cur_e = max(cur_e, en)
            union += cur_e - cur_s
            _omem_per_ch.append(union / len(ivs))
        O_mem_model = float(np.mean(_omem_per_ch)) if _omem_per_ch else np.nan

        _og = [x['onet_gauge_ms'] * 1e3 for sg in c['sends'] for x in sg['chunks']
               if x.get('onet_gauge_ms') is not None]
        O_net_gauge_us = float(np.median(_og)) if _og else np.nan
        O_net_gauge_std_us = float(np.std(_og)) if _og else np.nan

        rows.append({'protocol': c['protocol'], 'nch': c['nchannels'], 'nthreads': c['nthreads'],
                     'buffsize': c['buffsize'],
                     'O_net_gauge_us': O_net_gauge_us, 'O_net_gauge_std_us': O_net_gauge_std_us,
                     'O_mem_model_us': O_mem_model,
                     'O_round_raw_us': O_round_raw,
                     'O_round_excess_us': (O_round_raw - twire_us) if not np.isnan(O_round_raw) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# statistics (numpy-only, no scipy)
# --------------------------------------------------------------------------- #
def ols(x, y):
    """Ordinary least squares. Returns dict with intercept a, slope b, R2, std errors."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    b, a = np.polyfit(x, y, 1)
    yh = a + b * x
    ss_res = float(np.sum((y - yh)**2)); ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    sxx = float(np.sum((x - x.mean())**2)); s2 = ss_res / (n - 2) if n > 2 else 0.0
    se_b = sqrt(s2 / sxx) if sxx > 0 else 0.0
    se_a = sqrt(s2 * (1.0 / n + x.mean()**2 / sxx)) if sxx > 0 else 0.0
    return {'a': a, 'b': b, 'r2': r2, 'se_a': se_a, 'se_b': se_b, 'n': n}


def normal_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def one_sample_t(x, mu=0.0):
    """One-sample t-test vs mu. Returns (t, n, two-sided p via normal approx)."""
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float('nan'), n, float('nan')
    se = x.std(ddof=1) / sqrt(n)
    if se == 0:
        return float('inf'), n, 0.0
    t = (x.mean() - mu) / se
    p = 2 * (1 - normal_cdf(abs(t)))
    return t, n, p


# --------------------------------------------------------------------------- #
# per-config accessor (for plots and the o_net_eff fit)
# --------------------------------------------------------------------------- #
def onet_eff_per_chunk(cfg):
    """(chunk_ids, o_net_eff_us) pooled across channels from the o_net_eff field."""
    oe, cid = [], []
    for sg in cfg['sends']:
        for x in sg['chunks']:
            if x.get('onet_eff_gauge_ms') is not None:
                oe.append(x['onet_eff_gauge_ms'] * 1e3); cid.append(x['chunk_id'])
    if len(oe) < 2:
        return None, None
    return np.array(cid), np.array(oe)


# --------------------------------------------------------------------------- #
# fits (lifted out of the notebook plot loops)
# --------------------------------------------------------------------------- #
def _fit_onet_eff_twoseg(xf, yf):
    """Two-segment fit on sorted (xf, yf): flat plateau (mean) below an auto-searched
    breakpoint, OLS line above. Returns a fit-row dict, or None if too few points."""
    if len(xf) < 4:
        return None
    best = None   # (sse, bp, lo_const, ols_hi)
    for k in range(2, len(xf) - 1):
        bp = (xf[k - 1] + xf[k]) / 2.0
        c = float(np.mean(yf[:k]))
        sse_lo = float(np.sum((yf[:k] - c)**2))
        fh = ols(xf[k:], yf[k:])
        sse_hi = float(np.sum((yf[k:] - (fh['a'] + fh['b'] * xf[k:]))**2))
        s = sse_lo + sse_hi
        if best is None or s < best[0]:
            best = (s, bp, c, fh)
    _sse, bp, c, fh = best
    return {'breakpoint_KB': float(bp), 'lo_const_us': round(c, 4),
            'hi_intercept_us': round(fh['a'], 4), 'hi_slope_us_per_KB': round(fh['b'], 6),
            'hi_R2': round(fh['r2'], 4)}


def fit_onet_eff(rank0, nt=NT_FIT, exclude_buf=EXCLUDE_BUF, proto=PROTO):
    """o_net_eff scatter + two-segment fit per nch.

    Returns ``{nch: {'xs','ys','lo','hi','fit'}}`` where xs=chunk_KB, ys=steady-half mean,
    lo/hi=p10/p90 (all sorted by xs), and ``fit`` is the two-segment fit-row dict. The
    notebook plots xs/ys/lo/hi and collects the ``fit`` dicts into O_NET_EFF_LINFIT.
    """
    cfgs = [c for c in rank0 if c['protocol'] == proto and c['nthreads'] == nt
            and c['buffsize'] not in exclude_buf]
    bufs = sorted({c['buffsize'] for c in cfgs}, key=buf_bytes)
    nchs = sorted({c['nchannels'] for c in cfgs})
    out = {}
    for n in nchs:
        xs, ys, lo, hi = [], [], [], []
        for buf in bufs:
            cfg = next((c for c in cfgs if c['buffsize'] == buf and c['nchannels'] == n), None)
            if cfg is None:
                continue
            cid, eff = onet_eff_per_chunk(cfg)
            if cid is None:
                continue
            cids = np.array(sorted(set(cid)))
            curve = np.array([np.median(eff[cid == cc]) for cc in cids])
            plat = curve[len(curve) // 2:]            # steady half
            xs.append(chunk_bytes(buf) / 1024); ys.append(float(np.mean(plat)))
            lo.append(float(np.percentile(plat, 10))); hi.append(float(np.percentile(plat, 90)))
        if not xs:
            continue
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        lo = np.array(lo)[order]; hi = np.array(hi)[order]
        out[int(n)] = {'xs': xs, 'ys': ys, 'lo': lo, 'hi': hi,
                       'fit': _fit_onet_eff_twoseg(xs, ys)}
    return out


def fit_omem(param_df, proto=PROTO, nt=NT_FIT, big_kb=O_MEM_BIG_KB):
    """O_mem = k*slice_KB + b (shared slope from the large-chunk pool), with a low-chunk
    plateau for nch 8/16. Returns (O_MEM_FIT_dict, G_mem_GBs)."""
    src = param_df[param_df.nthreads == nt].copy()
    src['skb'] = src.apply(lambda r: chunk_bytes(r.buffsize) / 1024.0, axis=1)
    dp = src[src.protocol == proto].dropna(subset=['O_mem_model_us'])
    big = dp[dp.skb >= big_kb]
    f = ols(big.skb.values, big.O_mem_model_us.values) if len(big) >= 2 \
        else ols(dp.skb.values, dp.O_mem_model_us.values)
    k, b, r2 = f['b'], f['a'], f['r2']
    thr_by_nch = {}
    for nch in sorted(dp.nch.unique()):
        if int(nch) not in (8, 16):
            continue
        s = dp[dp.nch == nch].sort_values('skb'); xs = s.skb.values; ys = s.O_mem_model_us.values
        best = None
        for ki in range(2, len(xs)):
            thr = (xs[ki - 1] + xs[ki]) / 2.0; c = k * thr + b
            sse = float(np.sum((ys[:ki] - c)**2) + np.sum((ys[ki:] - (k * xs[ki:] + b))**2))
            if best is None or sse < best[0]:
                best = (sse, thr)
        if best is not None:
            thr_by_nch[int(nch)] = round(best[1], 1)
    fit = {'k_us_per_KB': round(k, 6), 'b_us': round(b, 4), 'R2': round(r2, 4),
           'big_chunk_KB': big_kb, 'nch_plateau_thr_KB': thr_by_nch}
    gmem = round(1024.0 / k / 1e3, 4) if k > 0 else float('nan')   # GB/s = (1024/k) bytes/us
    return fit, gmem


def fit_oround(param_df, proto=PROTO, nt=NT_FIT):
    """O_round = C + K*(slice_MB*nch), OLS on the raw round-boundary interval. Returns (C, K, R2)."""
    sim = param_df[(param_df.protocol == proto) & (param_df.nthreads == nt)].dropna(subset=['O_round_raw_us']).copy()
    sim['x'] = sim.apply(lambda r: chunk_bytes(r.buffsize) / (1024**2) * r.nch, axis=1)
    fr = ols(sim.x.values, sim.O_round_raw_us.values)
    return round(fr['a'], 4), round(fr['b'], 4), round(fr['r2'], 4)


# --------------------------------------------------------------------------- #
# assemble overhead.json params
# --------------------------------------------------------------------------- #
def assemble_params(param_df, onet_eff, omem_fit, gmem, oround, stat=STAT_DEFAULT):
    """Build the overhead.json ``params`` dict.

    ``onet_eff`` is the dict from :func:`fit_onet_eff`; ``omem_fit``/``gmem`` from
    :func:`fit_omem`; ``oround`` = (C, K, R2) from :func:`fit_oround`.
    """
    O_ROUND_C, O_ROUND_K, O_ROUND_R2 = oround
    O_NET_EFF_LINFIT = {PROTO: {str(n): d['fit'] for n, d in sorted(onet_eff.items()) if d['fit']}}
    return {
        'note': ('Calibrated from ring_param_extraction (Ring/Simple only). Overhead helpers '
                 'take SLICE_SIZE (bytes); Simple chunk = buffsize/4. '
                 'O_net = Iterative recovered-bandwidth gauge, median over chunks; O_NET_EFF_LINFIT = two-segment '
                 'fit per nch (flat plateau below breakpoint, line above).'),
        'STAT': stat,
        'n_configs': int(len(param_df)),
        'O_NET_EFF_LINFIT': O_NET_EFF_LINFIT,
        'NIC_BW_DUAL_bytes_us': NIC_BW_DUAL,
        'G_MEM_GBS': {PROTO: gmem},
        'O_MEM_FIT': {PROTO: omem_fit},
        'O_ROUND_C_us': O_ROUND_C,
        'O_ROUND_K_us_per_MBch': O_ROUND_K,
        'O_ROUND_R2': O_ROUND_R2,
    }


def write_overhead_json(params, path):
    """Serialize the assembled ``params`` dict to ``path`` as overhead.json (indent=2)."""
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)
    return path


def build_overhead(out_path, json_path):
    """End-to-end regeneration of overhead.json from the raw NCCL profiling dump (Ring/Simple).

    parse -> build_param_df -> fit_onet_eff / fit_omem / fit_oround -> assemble_params ->
    write_overhead_json. Returns a dict of every intermediate so a notebook can plot the
    evidence figures without recomputing::

        {'params', 'param_df', 'onet_eff', 'omem_fit', 'gmem', 'oround'}

    Callers that only want the JSON on disk can ignore the return value.
    """
    rank0 = parse_ring_file(out_path)
    param_df = build_param_df(rank0)
    onet_eff = fit_onet_eff(rank0)
    omem_fit, gmem = fit_omem(param_df)
    oround = fit_oround(param_df)
    params = assemble_params(param_df, onet_eff, omem_fit, gmem, oround)
    write_overhead_json(params, json_path)
    return {'params': params, 'param_df': param_df, 'onet_eff': onet_eff,
            'omem_fit': omem_fit, 'gmem': gmem, 'oround': oround}
