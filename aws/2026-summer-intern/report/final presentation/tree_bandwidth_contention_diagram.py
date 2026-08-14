"""Combined 'Single Tree' (left) + 'Four Trees' (right) bandwidth-contention diagram.

Left  : one tree drawn 3 times, each highlighting node 2's role (leaf=1 send/1 recv,
        middle=3 send/3 recv), with a red dashed box calling out the send/recv count.
Right : the four trees of the 4-tree AllReduce, highlighting node 2 in each (green),
        showing it is the middle node in trees 0/1 and a leaf in trees 2/3.
"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))  # save outputs next to this script
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# All text: Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

FS = 20

C_BG      = '#FAFBFC'
C_NODE    = '#3B7DD8'   # normal node blue
C_NODE_HL = '#27AE60'   # highlighted (green) node
C_HL_EDGE = '#1E5E34'
C_ARROW   = '#F5A623'   # yellow bidirectional link
C_DASH    = '#D94F4F'   # red dashed callout box
C_TEXT    = '#2C3E50'
NR        = 0.42        # node radius

fig = plt.figure(figsize=(13, 7.6), facecolor='none')
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.05)
axT = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])   # Top=Single Tree, Bottom=Four Trees
for ax in (axT, axB):
    ax.set_facecolor('none'); ax.axis('off'); ax.set_aspect('equal')


def node(ax, xy, label, hl=False):
    fc = C_NODE_HL if hl else C_NODE
    ec = C_HL_EDGE if hl else C_NODE
    ax.add_patch(Circle(xy, NR, facecolor=fc, edgecolor=ec, linewidth=2.5 if hl else 0, zorder=5))
    ax.text(xy[0], xy[1], label, ha='center', va='center', fontsize=FS,
            color='white', fontweight='bold', zorder=6)


def bidir(ax, a, b):
    """Yellow double-headed link between two node centers (stops at node edges)."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    L = (dx*dx+dy*dy) ** 0.5
    ux, uy = dx/L, dy/L
    sa = (a[0]+ux*NR, a[1]+uy*NR)
    sb = (b[0]-ux*NR, b[1]-uy*NR)
    ax.add_patch(FancyArrowPatch(sa, sb, arrowstyle='<|-|>', mutation_scale=16,
                                 color=C_ARROW, lw=3.0, shrinkA=0, shrinkB=0, zorder=3))


def small_tree(ax, cx, cy, labels, hl_idx, s=1.0, vdrop=1.15):
    """Star tree: top-left leaf, middle, two bottom leaves.
    labels = [top_left, middle, bottom_left, bottom_right]; hl_idx highlights one node.
    vdrop = vertical distance of the bottom leaves below the middle (smaller = flatter tree)."""
    mid = (cx, cy)
    tl  = (cx - 1.9*s, cy)
    bl  = (cx - 0.7*s, cy - vdrop)
    br  = (cx + 0.7*s, cy - vdrop)
    pos = [tl, mid, bl, br]
    # links: middle <-> each of the other three
    bidir(ax, mid, tl); bidir(ax, mid, bl); bidir(ax, mid, br)
    for i, (p, lb) in enumerate(zip(pos, labels)):
        node(ax, p, lb, hl=(i == hl_idx))
    return pos


def dashed_box(ax, node_xy, w=1.15, h=1.15):
    """Red dashed box around a highlighted node."""
    ax.add_patch(FancyBboxPatch((node_xy[0]-w/2, node_xy[1]-h/2), w, h, boxstyle="round,pad=0.02",
                 facecolor='none', edgecolor=C_DASH, linewidth=2.2,
                 linestyle=(0, (5, 4)), zorder=8))


def role_label(ax, x, y, text):
    """Send/receive count label (below a tree)."""
    ax.text(x, y, text, ha='center', va='center',
            fontsize=FS-5, color=C_DASH, fontweight='bold', zorder=9)


# Both panels use IDENTICAL limits + equal aspect, so every node renders the same size.
# Trees are flattened (small vdrop) so the whole figure is compact without distortion.
XLIM = (-0.6, 13.0)
YLIM = (0.55, 4.35)

# ============================================================
# TOP : Single Tree — three copies highlighting node roles (row layout)
# ============================================================
axT.set_xlim(*XLIM); axT.set_ylim(*YLIM)
axT.text((XLIM[0]+XLIM[1])/2, 4.2, 'Single Tree', ha='center', va='center',
         fontsize=FS+4, fontweight='bold', color=C_TEXT)

# three trees side by side; cy chosen so the whole star fits in YLIM
cy_top = 3.15
label_y_top = 1.05
# labels [top_left, middle, bottom_left, bottom_right] — same numbering as the Four-Trees row
singles = [
    (['0', '2', '1', '3'], 2, '1 send, 1 receive'),   # highlight bottom-left leaf
    (['0', '2', '1', '3'], 1, '3 send, 3 receive'),   # highlight middle
    (['0', '2', '1', '3'], 3, '1 send, 1 receive'),   # highlight bottom-right leaf
]
# three trees centered on the figure (same centering as the Four-Trees row -> rows aligned)
_pitch_top = 3.1
_xc_top = (XLIM[0] + XLIM[1]) / 2
sx = [_xc_top + (i - 1) * _pitch_top for i in range(3)]
for (labels, hl, role), x in zip(singles, sx):
    pos = small_tree(axT, x, cy_top, labels, hl_idx=hl, s=0.85)
    dashed_box(axT, pos[hl])
    role_label(axT, x - 0.55, label_y_top, role)   # centered under the tree

# ============================================================
# BOTTOM : Four Trees — node 2's role per tree (row layout)
# ============================================================
axB.set_xlim(*XLIM); axB.set_ylim(*YLIM)
axB.text((XLIM[0]+XLIM[1])/2, 4.2, 'Four Trees', ha='center', va='center',
         fontsize=FS+4, fontweight='bold', color=C_TEXT)

cy_bot = 3.15
label_y_bot = 1.05
# Each tree: labels [top_left, middle, bottom_left, bottom_right]; highlight node "2".
# trees 0,1: node 2 is the middle;  trees 2,3: node 2 is a bottom leaf.
trees = [
    ('Tree 0', ['0', '2', '1', '3'], 1),
    ('Tree 1', ['0', '2', '1', '3'], 1),
    ('Tree 2', ['3', '1', '2', '0'], 2),
    ('Tree 3', ['3', '1', '2', '0'], 2),
]
# four trees centered on the figure so this row aligns with the (centered) Single-Tree row above
_pitch = 3.1
_xc = (XLIM[0] + XLIM[1]) / 2                      # figure center
tx = [_xc + (i - 1.5) * _pitch for i in range(4)]  # 4 trees centered about _xc
for (title, labels, hl), x in zip(trees, tx):
    small_tree(axB, x, cy_bot, labels, hl_idx=hl, s=0.85)
    axB.text(x - 0.55, label_y_bot, title, ha='center', va='center',
             fontsize=FS, fontweight='bold', color=C_TEXT)

plt.savefig(os.path.join(_HERE, 'tree-bandwidth-contention.png'),
            dpi=200, bbox_inches='tight', pad_inches=0.04, facecolor='none', transparent=True)
plt.close()
print("Diagram saved: tree-bandwidth-contention.png")
