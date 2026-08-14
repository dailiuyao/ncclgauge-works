"""Circular-dependency figure (conference-paper style, causal-loop diagram).

Fixed-point loop between two quantities:
    T_wire (per-chunk wire time)  and  k (parallelism / #overlapping windows)
  Effect A (top arc):    longer T_wire -> windows overlap more -> k up
  Effect B (bottom arc): k up -> BW_eff = BW/k down -> T_wire up
"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))  # save outputs next to this script
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# --- Times New Roman everywhere, ONE font size ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
FS = 15   # single uniform font size

# --- muted, paper-friendly palette ---
C_TEXT  = '#22303C'
C_MUTE  = '#5B6B7A'
C_A     = '#B0413E'   # Effect A (terracotta)
C_B     = '#356899'   # Effect B (steel blue)
C_GREEN = '#6E9B5A'
C_NODE  = '#EEF2F6'
C_PANEL = '#F6F8FA'
C_PANEL_E = '#D4DBE2'

fig = plt.figure(figsize=(13, 5.6), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, 16); ax.set_ylim(0, 8.35)


def node(cx, cy, w, h, label, edge):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.04",
                 facecolor=C_NODE, edgecolor=edge, linewidth=2.2, zorder=6))
    ax.text(cx, cy, label, ha='center', va='center', fontsize=FS,
            color=C_TEXT, fontweight='bold', zorder=7)


# ============================================================
# (1) Model definition — top
# ============================================================
# Known quantities are green, unknown quantities are red; operators/plain math stay dark.
C_KNOWN = '#2E7D32'
C_UNK   = '#C0392B'
_EQ = r'$\;=\;$'; _OP = r'$\;-\;$'; _PL = r'$\;+\;$'; _CM = r'$,\;\;\;$'
# Each segment is either an inline token ('t', mathtext, color) or a colored fraction
# ('f', numerator_math, num_color, denominator_math, den_color).
_formula = [
    ('t', r'$o\_net$', C_UNK), ('t', _EQ, C_TEXT),
    ('t', r'$T\_send$', C_KNOWN), ('t', _OP, C_TEXT), ('t', r'$T\_wire$', C_UNK),
    ('t', _CM, C_TEXT),
    ('t', r'$T\_wire$', C_UNK), ('t', _EQ, C_TEXT),
    ('f', r'$Chunk\_Size$', C_KNOWN, r'$BW\_eff$', C_UNK),
    ('t', _PL, C_TEXT), ('t', r'$2L$', C_KNOWN),
    ('t', _CM, C_TEXT),
    ('t', r'$BW\_eff$', C_UNK), ('t', _EQ, C_TEXT),
    ('f', r'$EFA\ BW$', C_KNOWN, r'$k$', C_UNK),
]
def _mwidth(txt, fs, r, inv):
    t = ax.text(0, -50, txt, fontsize=fs); fig.canvas.draw()
    bb = t.get_window_extent(r)
    w = inv.transform((bb.x1, bb.y0))[0] - inv.transform((bb.x0, bb.y0))[0]
    t.remove(); return w
def _draw_colored_math(ax, segs, y, fs, cx=8.0):
    """Layout segments left→right, centered at cx. Fractions get a colored num/den + rule."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    widths = []
    for seg in segs:
        if seg[0] == 't':
            widths.append(_mwidth(seg[1], fs, r, inv))
        else:  # fraction: width = max(num,den) + small padding
            widths.append(max(_mwidth(seg[1], fs, r, inv), _mwidth(seg[3], fs, r, inv)) + 0.12)
    total = sum(widths); x = cx - total/2
    dy = 0.30  # vertical offset of numerator/denominator from the fraction bar
    for seg, w in zip(segs, widths):
        if seg[0] == 't':
            ax.text(x, y, seg[1], ha='left', va='center', fontsize=fs, color=seg[2])
        else:
            _, num, ncol, den, dcol = seg
            xc = x + w/2
            ax.text(xc, y + dy, num, ha='center', va='center', fontsize=fs, color=ncol)
            ax.text(xc, y - dy, den, ha='center', va='center', fontsize=fs, color=dcol)
            ax.plot([x+0.05, x+w-0.05], [y, y], color=C_TEXT, lw=1.3)
        x += w
_draw_colored_math(ax, _formula, 7.9, FS)
# known/unknown legend — to the RIGHT of the formula, on the same line
ax.add_patch(FancyBboxPatch((12.35, 7.82), 0.24, 0.18, boxstyle="round,pad=0.01", facecolor=C_KNOWN, edgecolor='none', zorder=4))
ax.text(12.66, 7.91, 'known', ha='left', va='center', fontsize=FS, color=C_TEXT)
ax.add_patch(FancyBboxPatch((12.35, 7.46), 0.24, 0.18, boxstyle="round,pad=0.01", facecolor=C_UNK, edgecolor='none', zorder=4))
ax.text(12.66, 7.55, 'unknown', ha='left', va='center', fontsize=FS, color=C_TEXT)

# ============================================================
# (2) Causal loop — two tight nodes + two arcs (centered about x=8.0)
# ============================================================
Lx, Rx, Cy = 4.6, 11.4, 5.3
nw, nh = 1.9, 0.8            # tight: box only a bit bigger than the label
node(Lx, Cy, nw, nh, r'$T\_wire$', C_A)
node(Rx, Cy, nw, nh, r'$k$', C_B)

# top arc: T_wire -> k  (Effect A)
ax.add_patch(FancyArrowPatch((Lx+nw/2, Cy+nh/2-0.05), (Rx-nw/2, Cy+nh/2-0.05),
             connectionstyle="arc3,rad=-0.32", arrowstyle='-|>', mutation_scale=22,
             color=C_A, lw=3.0, zorder=4))
# bottom arc: k -> T_wire  (Effect B)
ax.add_patch(FancyArrowPatch((Rx-nw/2, Cy-nh/2+0.05), (Lx+nw/2, Cy-nh/2+0.05),
             connectionstyle="arc3,rad=-0.32", arrowstyle='-|>', mutation_scale=22,
             color=C_B, lw=3.0, zorder=4))

# Measured arcs (Cy=5.3): top apex y≈6.52, bottom nadir y≈4.08.
# Place each effect block the SAME gap (0.40) from its arc so A and B are symmetric.
# Effect A text — ABOVE the top arc apex
ax.text(8.0, 7.02, 'Effect A', ha='center', va='center', fontsize=FS, color=C_A, fontweight='bold')
ax.text(8.0, 6.82, r'longer $T\_wire$ → more overlap → $k\!\uparrow$',
        ha='center', va='center', fontsize=FS, color=C_TEXT)

# Effect B text — BELOW the bottom arc nadir (mirror of A: nearest line 0.40 from the arc)
ax.text(8.0, 3.78, 'Effect B', ha='center', va='center', fontsize=FS, color=C_B, fontweight='bold')
ax.text(8.0, 3.48, r'$k\!\uparrow$ → $BW\_eff = BW/k\!\downarrow$ → $T\_wire\!\uparrow$',
        ha='center', va='center', fontsize=FS, color=C_TEXT)

# ============================================================
# (3) Two supporting mini-illustrations (tight panels)
# ============================================================
# -- Effect A panel: overlapping windows (left) -- narrower (content is shorter than B)
axp = (1.3, 0.35, 5.8, 2.4)
ax.add_patch(FancyBboxPatch((axp[0], axp[1]), axp[2], axp[3], boxstyle="round,pad=0.03",
             facecolor=C_PANEL, edgecolor=C_PANEL_E, linewidth=1.2, zorder=2))
ax.text(axp[0]+0.25, axp[1]+axp[3]-0.28, r'A:  longer $T\_wire$ → more overlap → higher $k$',
        ha='left', va='center', fontsize=FS, color=C_A, fontweight='bold')
ax.text(axp[0]+0.25, axp[1]+1.15, 'k=2', ha='left', va='center', fontsize=FS, color=C_MUTE)
for i in range(2):
    ax.add_patch(Rectangle((axp[0]+1.15+i*0.8, axp[1]+1.38-i*0.26), 1.15, 0.20,
                 facecolor=C_A, edgecolor='white', linewidth=1.2, zorder=3))
ax.text(axp[0]+0.25, axp[1]+0.42, 'k=4', ha='left', va='center', fontsize=FS, color=C_MUTE)
for i in range(4):
    ax.add_patch(Rectangle((axp[0]+1.15+i*0.55, axp[1]+0.66-i*0.17), 1.6, 0.15,
                 facecolor=C_A, edgecolor='white', linewidth=1.2, zorder=3))

# -- Effect B panel: bandwidth split (right) -- width trimmed to hug the content (no right gap)
bxp = (7.5, 0.35, 7.0, 2.4)
ax.add_patch(FancyBboxPatch((bxp[0], bxp[1]), bxp[2], bxp[3], boxstyle="round,pad=0.03",
             facecolor=C_PANEL, edgecolor=C_PANEL_E, linewidth=1.2, zorder=2))
ax.text(bxp[0]+0.25, bxp[1]+bxp[3]-0.28, r'B:  higher $k$ → each chunk gets less BW → longer $T\_wire$',
        ha='left', va='center', fontsize=FS, color=C_B, fontweight='bold')
ax.text(bxp[0]+0.25, bxp[1]+1.25, 'k=1', ha='left', va='center', fontsize=FS, color=C_MUTE)
ax.add_patch(Rectangle((bxp[0]+1.15, bxp[1]+1.08), 5.4, 0.36, facecolor=C_GREEN, edgecolor='white', linewidth=1.4, zorder=3))
ax.text(bxp[0]+1.15+5.4/2, bxp[1]+1.26, 'BW', ha='center', va='center', fontsize=FS, color='white', fontweight='bold', zorder=4)
ax.text(bxp[0]+0.25, bxp[1]+0.5, 'k=4', ha='left', va='center', fontsize=FS, color=C_MUTE)
seg = 5.4/4
for i in range(4):
    ax.add_patch(Rectangle((bxp[0]+1.15+i*seg, bxp[1]+0.32), seg-0.08, 0.36,
                 facecolor=C_GREEN, edgecolor='white', linewidth=1.4, zorder=3))
    ax.text(bxp[0]+1.15+i*seg+(seg-0.08)/2, bxp[1]+0.5, 'BW/4', ha='center', va='center',
            fontsize=FS, color='white', fontweight='bold', zorder=4)

plt.savefig(os.path.join(_HERE, 'circular-dependency.png'),
            dpi=200, bbox_inches='tight', pad_inches=0.06, facecolor='white')
plt.close()
print("Diagram saved: circular-dependency.png")
