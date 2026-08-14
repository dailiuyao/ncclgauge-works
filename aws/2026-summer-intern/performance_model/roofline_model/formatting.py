"""Message-size formatting helpers shared across the roofline/realistic notebooks.

These are small, pure functions used for axis tick labels and table output. 
They live here so the notebooks import one canonical version.
"""


def fmt_size(nbytes):
    """Short human-readable byte size: 1536 -> '1K', 2*1024**2 -> '2M', etc.

    Used for compact table columns / labels (no fractional units).
    """
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.0f}G"
    if nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.0f}M"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.0f}K"
    return f"{nbytes:.0f}"


def size_tick_formatter(x, pos=None):
    """matplotlib tick formatter for a byte-valued (log) axis.

    Returns '' for non-power-of-1024 values so only clean K/M/G ticks are labelled.
    Signature matches matplotlib.ticker.FuncFormatter (value, position).
    """
    if x <= 0:
        return ''
    if x < 1024:
        return f'{int(x)}B'
    elif x < 1024**2:
        v = x / 1024
        return f'{int(v)}K' if v == int(v) else ''
    elif x < 1024**3:
        v = x / 1024**2
        return f'{int(v)}M' if v == int(v) else ''
    else:
        v = x / 1024**3
        return f'{int(v)}G' if v == int(v) else ''


def size_label(x, pos=None):
    """matplotlib tick formatter that always labels the value (used for endpoint ticks).

    Unlike size_tick_formatter this never blanks a tick — callers pick the ticks
    (e.g. only first/last message size) and want every chosen one labelled.
    """
    if x >= 1024**3:
        return f'{x / 1024**3:.0f}G'
    elif x >= 1024**2:
        return f'{x / 1024**2:.0f}M'
    elif x >= 1024:
        return f'{x / 1024:.0f}K'
    else:
        return f'{x:.0f}'
