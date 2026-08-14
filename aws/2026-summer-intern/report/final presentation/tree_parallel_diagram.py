import os
_HERE = os.path.dirname(os.path.abspath(__file__))  # save outputs next to this script
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# All text: Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

FS = 28                       # unified font size

C_BG = '#FAFBFC'
C_TEXT = '#2C3E50'
C_SUBTEXT = '#7F8C8D'
C_REDUCE_FC = '#F4C7C3'; C_REDUCE_EC = '#D94F4F'
C_BCAST_FC = '#C7E8CE';  C_BCAST_EC = '#27AE60'
C_OVERLAP = '#E8790C'

# rows: (label, fill, edge, first_step_index, n_steps)  — label on ONE line
rows = [
    ('GPU1,3→GPU2 (reduce)',    C_REDUCE_FC, C_REDUCE_EC, 0, 4),
    ('GPU2→GPU0 (reduce)',      C_REDUCE_FC, C_REDUCE_EC, 1, 4),
    ('GPU0→GPU2 (broadcast)',   C_BCAST_FC,  C_BCAST_EC,  2, 4),
    ('GPU2→GPU1,3 (broadcast)', C_BCAST_FC,  C_BCAST_EC,  3, 4),
]

step_w = 1.0
step_h = 0.18                  # box height — only slightly taller than the label text
row_pitch = 0.26               # row spacing — row is only slightly taller than the text
gap = 0.03

fig = plt.figure(figsize=(11, 4.2), facecolor='none')
ax = fig.add_axes([0, 0, 1, 1])   # axes fill the whole figure (no internal margins)
ax.set_facecolor('none')
ax.axis('off')

n_rows = len(rows)
y_top = 0.0
y_positions = [y_top - i * row_pitch for i in range(n_rows)]

label_x = -0.3                 # right edge of the text labels
x0 = 0.4                       # left edge of the first possible step column

for (label, fc, ec, first, nsteps), y in zip(rows, y_positions):
    ax.text(label_x, y, label, ha='right', va='center', fontsize=FS, color=C_TEXT)
    for k in range(nsteps):
        col = first + k
        x = x0 + col * step_w
        ax.add_patch(FancyBboxPatch((x + gap, y - step_h/2), step_w - 2*gap, step_h,
                     boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec,
                     linewidth=1.4, zorder=5))
        ax.text(x + step_w/2, y, f'c{k}', ha='center', va='center',
                fontsize=FS, color=C_TEXT, zorder=6)

# ---- vertical spacing knobs (all scale with the compact row height) ----
box_pad    = 0.06   # dashed overlap box padding above/below the boxes
gap_below  = 0.14   # gap from overlap-caption down to the Time axis
title_gap  = 0.14   # gap from the top row up to the title
edge_pad   = 0.10   # tiny top/bottom margin for the figure limits

# Overlap box: covers columns 2..5 (where reduce & broadcast coexist), spanning all rows
ov_x0 = x0 + 2 * step_w
ov_x1 = x0 + 6 * step_w
ov_top = y_positions[0] + step_h/2 + box_pad
ov_bot = y_positions[-1] - step_h/2 - box_pad
ax.add_patch(FancyBboxPatch((ov_x0, ov_bot), ov_x1 - ov_x0, ov_top - ov_bot,
             boxstyle="round,pad=0.0", facecolor='none', edgecolor=C_OVERLAP,
             linewidth=2.5, linestyle=(0, (6, 4)), zorder=8))
cap_y = ov_bot - 0.06
ax.text((ov_x0 + ov_x1)/2, cap_y, 'Overlap',
        ha='center', va='top', fontsize=FS, color=C_OVERLAP)

# Time axis (arrow spans the step columns; label centered under the arrow midpoint)
time_x0 = x0
time_x1 = x0 + 7.4 * step_w
time_y = cap_y - gap_below
ax.annotate('', xy=(time_x1, time_y), xytext=(time_x0, time_y),
            arrowprops=dict(arrowstyle='->', color=C_SUBTEXT, lw=1.8))
ax.text((time_x0 + time_x1) / 2, time_y - 0.04, 'Time →',
        ha='center', va='top', fontsize=FS, color=C_SUBTEXT)

# Title (close to the plot area)
ax.text((x0 + 1.7 * step_w), y_positions[0] + step_h/2 + title_gap,
        '(1) Reduce and Broadcast run in PARALLEL',
        ha='center', va='bottom', fontsize=FS + 3, fontweight='bold', color=C_TEXT)

# tight limits: hug the actual content so there is no surrounding whitespace
ax.set_xlim(label_x - 2.4, x0 + 7.7 * step_w)
ax.set_ylim(time_y - edge_pad, y_positions[0] + step_h/2 + title_gap + 0.28)

plt.savefig(os.path.join(_HERE, 'tree-parallel.png'),
            dpi=180, bbox_inches='tight', pad_inches=0.0, facecolor='none', transparent=True)
plt.close()
print("Diagram saved: tree-parallel.png")
