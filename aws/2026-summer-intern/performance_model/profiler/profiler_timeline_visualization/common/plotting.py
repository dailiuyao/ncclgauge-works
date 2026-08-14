"""Shared plotting primitives for NCCL per-chunk timeline figures.

Platform- and algorithm-independent matplotlib helpers. The complex, algorithm-specific
bar layout (e.g. ring dual-rail + T_wire, tree reduce/broadcast sub-rows) stays in each
notebook; these are the generic pieces the timeline figures have in common:

    fmt_bytes(n)                       human-readable byte size (MB / KB / B)
    TimelineStyle                      fonts / colors / font size (overridable defaults)
    draw_group_background(...)         rounded rectangle behind a chunk group
    draw_chunk_label(...)              'c{id}' left-margin label
    draw_separator(...)                dashed inter-chunk separator
    finalize_timeline_axes(ax, ...)    hide spines, y-ticks off, x label, limits
    save_and_show(fig, out_dir, name)  savefig + show + print
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def fmt_bytes(n):
    """Human-readable byte size: '>=1MB -> MB', '>=1KB -> KB', else 'B' (1 decimal)."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n:.0f} B"


class TimelineStyle:
    """Fonts / colors / font size for a timeline figure. Override any field per notebook."""

    def __init__(self, font_family='serif', font_serif=('Times New Roman', 'Times', 'DejaVu Serif'),
                 mathtext='stix', fs=13, bg='white', text='#2C3E50',
                 group_edge='#E2E8F0', separator='#B0B8C0'):
        self.font_family = font_family
        self.font_serif = list(font_serif)
        self.mathtext = mathtext
        self.fs = fs
        self.bg = bg
        self.text = text
        self.group_edge = group_edge
        self.separator = separator

    def apply(self):
        """Push the font settings into matplotlib rcParams."""
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.serif'] = self.font_serif
        plt.rcParams['mathtext.fontset'] = self.mathtext


def draw_group_background(ax, x0, y0, w, h, style, pad=0.06):
    """Rounded rectangle behind a chunk group."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), w, h, boxstyle=f"round,pad={pad}",
        facecolor=style.bg, edgecolor=style.group_edge, linewidth=0.5, zorder=0))


def draw_chunk_label(ax, x, y, chunk_id, style):
    """Left-margin 'c{chunk_id}' label at a chunk row."""
    ax.text(x, y, f"c{chunk_id}", ha='right', va='center',
            fontsize=style.fs, color=style.text, fontweight='bold', zorder=6)


def draw_separator(ax, x0, x1, y, style):
    """Dashed horizontal separator between chunk rows."""
    ax.plot([x0, x1], [y, y], ls='--', color=style.separator, lw=0.7, alpha=0.6, zorder=1)


def finalize_timeline_axes(ax, xlim, ylim, style, xlabel='Time (ms)'):
    """Hide top/right/left spines, drop y-ticks, set x label + limits."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=style.fs, color=style.text)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=14)


def save_and_show(fig, out_dir, save_name, facecolor='white'):
    """savefig(out_dir/save_name) + show + 'Saved:' print."""
    fig.savefig(os.path.join(out_dir, save_name), dpi=150,
                bbox_inches='tight', pad_inches=0.05, facecolor=facecolor)
    plt.show()
    print(f"Saved: {save_name}")
