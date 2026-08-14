import os
_HERE = os.path.dirname(os.path.abspath(__file__))  # save outputs next to this script
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Color palette (same as tree diagram)
C_BG = '#FAFBFC'
C_REDUCE = '#D94F4F'
C_BROADCAST = '#27AE60'
C_NORMAL = '#F5A623'
C_NORMAL_BORDER = '#E08E0B'
C_NODE_ACTIVE = '#3B7DD8'
C_NODE_DIM = '#C8D6E5'
C_TEXT = '#2C3E50'
C_SUBTEXT = '#7F8C8D'
C_LATENCY = '#8E44AD'

fig = plt.figure(figsize=(11, 5.9), facecolor='none')

# ============================================================
# Pipeline / Step diagram - multiple GPU rows
# ============================================================
ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.9])
ax1.set_facecolor('none')
ax1.set_xlim(-1.5, 10.7)
ax1.set_ylim(1.55, 8.7)
ax1.axis('off')

# Parameters
n = 4  # number of ranks/chunks for illustration
n_rs_steps = n - 1    # ReduceScatter steps
n_ag_steps = n - 1    # AllGather steps
total_steps = n_rs_steps + n_ag_steps  # 2(n-1)

chunk_width = 1.4
chunk_height = 0.42   # box height — only slightly taller than the fontsize-14 text
gap = 0.06
row_pitch = 0.62      # vertical distance between adjacent GPU rows (tighter = more compact)
row_bg_pad = 0.04     # top/bottom padding of each row's background strip

# GPU rows - show all 4 GPUs, with GPU 0 as bottleneck (wider chunks)
gpu_labels = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']
row_top = 5.8         # y of the first (GPU 0) row
y_positions = [row_top - i * row_pitch for i in range(len(gpu_labels))]
bottleneck_gpu = 0  # GPU 0 is on the bottleneck link

# Chunk widths per GPU: bottleneck GPU has wider (slower) chunks
chunk_width_slow = 1.7
chunk_width_fast = 1.05

# Row backgrounds
for i, y in enumerate(y_positions):
    bg_color = '#EEF2F7' if i % 2 == 0 else C_BG
    rect = mpatches.FancyBboxPatch((-1.1, y - chunk_height/2 - row_bg_pad), 11.5, chunk_height + 2*row_bg_pad,
                                    boxstyle="round,pad=0.05", facecolor=bg_color,
                                    edgecolor='none', zorder=0)
    ax1.add_patch(rect)

# Draw GPU rows
for gpu_idx, (gpu_label, y) in enumerate(zip(gpu_labels, y_positions)):
    is_bottleneck = (gpu_idx == bottleneck_gpu)
    cw = chunk_width_slow if is_bottleneck else chunk_width_fast

    # GPU label
    weight = 'bold' if is_bottleneck else 'medium'
    color = C_REDUCE if is_bottleneck else C_TEXT
    ax1.text(-0.7, y, gpu_label, ha='center', va='center', fontsize=14,
             fontweight=weight, color=color)

    # Draw steps for this GPU
    for i in range(total_steps):
        x_start = i * chunk_width_slow  # all aligned to bottleneck timing
        if is_bottleneck:
            w = chunk_width_slow
        else:
            w = chunk_width_fast

        if i < n_rs_steps:
            fc = C_REDUCE if is_bottleneck else '#E88A8A'
            ec = '#B83A3A' if is_bottleneck else '#D06060'
        else:
            fc = C_BROADCAST if is_bottleneck else '#7DCEA0'
            ec = '#1E8449' if is_bottleneck else '#52BE80'

        alpha = 0.9 if is_bottleneck else 0.7

        rect = FancyBboxPatch((x_start + gap, y - chunk_height/2),
                              w - 2*gap, chunk_height,
                              boxstyle="round,pad=0.05",
                              facecolor=fc, edgecolor=ec, linewidth=1.0,
                              alpha=alpha, zorder=5)
        ax1.add_patch(rect)

        # Chunk label (only on bottleneck GPU)
        if is_bottleneck:
            if i < n_rs_steps:
                label = f'C{i+1}'
            else:
                label = f'C{i - n_rs_steps + 1}'
            ax1.text(x_start + w/2, y, label, ha='center', va='center',
                     fontsize=14, color='white', fontweight='bold', zorder=10)

# Bottleneck annotation
ax1.annotate('Bottleneck (slowest link)',
             xy=(chunk_width_slow * 2.5, y_positions[0] + chunk_height/2 + 0.02),
             xytext=(chunk_width_slow * 2.5, y_positions[0] + chunk_height/2 + 0.55),
             fontsize=14, ha='center', color=C_REDUCE, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_REDUCE, lw=1.5))

# Phase brackets (above GPU 0)
rs_start = 0
rs_end = n_rs_steps * chunk_width_slow
ag_start = rs_end
ag_end = total_steps * chunk_width_slow

# ReduceScatter bracket
bracket_y = 7.55
ax1.annotate('', xy=(rs_start, bracket_y), xytext=(rs_end, bracket_y),
             arrowprops=dict(arrowstyle='<->', color=C_REDUCE, lw=2))
ax1.text((rs_start + rs_end) / 2, bracket_y + 0.15,
         'Phase 1: ReduceScatter', ha='center', va='bottom',
         fontsize=14, fontweight='bold', color=C_REDUCE)
ax1.text((rs_start + rs_end) / 2, bracket_y - 0.25,
         r'($\lceil S/(N\times\mathrm{Chunk\_Max})\rceil\times(N-1)$ steps)', ha='center', va='top',
         fontsize=14, fontweight='bold', color=C_REDUCE)

# AllGather bracket
ax1.annotate('', xy=(ag_start, bracket_y), xytext=(ag_end, bracket_y),
             arrowprops=dict(arrowstyle='<->', color=C_BROADCAST, lw=2))
ax1.text((ag_start + ag_end) / 2, bracket_y + 0.15,
         'Phase 2: AllGather', ha='center', va='bottom',
         fontsize=14, fontweight='bold', color=C_BROADCAST)
ax1.text((ag_start + ag_end) / 2, bracket_y - 0.25,
         r'($\lceil S/(N\times\mathrm{Chunk\_Max})\rceil\times(N-1)$ steps)', ha='center', va='top',
         fontsize=14, fontweight='bold', color=C_BROADCAST)

# Total bracket
bracket_y_total = 8.2
ax1.annotate('', xy=(rs_start, bracket_y_total), xytext=(ag_end, bracket_y_total),
             arrowprops=dict(arrowstyle='<->', color=C_TEXT, lw=2.5))
ax1.text((rs_start + ag_end) / 2, bracket_y_total + 0.1,
         r'Total: $n = \lceil S/(N\times\mathrm{Chunk\_Max})\rceil \times 2 \times (N-1)$ steps', ha='center', va='bottom',
         fontsize=14, fontweight='bold', color=C_TEXT)

# Time axis
time_y = 3.15
ax1.annotate('', xy=(ag_end + 0.3, time_y), xytext=(rs_start - 0.3, time_y),
             arrowprops=dict(arrowstyle='->', color=C_SUBTEXT, lw=1.5))
ax1.text((rs_start + ag_end) / 2, time_y - 0.3, 'Time',
         ha='center', va='top', fontsize=14, color=C_SUBTEXT, style='italic')

# Per-step breakdown annotation (below time axis, in one row)
anno_y = 2.05

# "Each step =" label on left
ax1.text(1.2, anno_y, 'Each step =', ha='right', va='center',
         fontsize=14, fontweight='bold', color=C_TEXT)

# Zoomed-in view of one step (to the right of the label)
zoom_x = 1.6
zoom_w_bw = 2.4
zoom_w_lat = 1.0
zoom_h = 0.34   # box height — only slightly taller than the fontsize-14 text

# Background for zoom
ax1.add_patch(FancyBboxPatch((zoom_x - 0.08, anno_y - zoom_h/2 - 0.06),
              zoom_w_bw + zoom_w_lat + 0.16, zoom_h + 0.12,
              boxstyle="round,pad=0.06", facecolor='#F8F9FA',
              edgecolor='#CBD5E0', linewidth=1, zorder=2))

# Bandwidth component
ax1.add_patch(FancyBboxPatch((zoom_x, anno_y - zoom_h/2),
              zoom_w_bw, zoom_h,
              boxstyle="round,pad=0.04", facecolor=C_NORMAL,
              edgecolor=C_NORMAL_BORDER, linewidth=1.2, zorder=5))
ax1.text(zoom_x + zoom_w_bw/2, anno_y, 'Chunk Size / BW', ha='center', va='center',
         fontsize=14, color='white', fontweight='bold', zorder=10)

# Latency component
ax1.add_patch(FancyBboxPatch((zoom_x + zoom_w_bw, anno_y - zoom_h/2),
              zoom_w_lat, zoom_h,
              boxstyle="round,pad=0.04", facecolor=C_LATENCY,
              edgecolor='#6C3483', linewidth=1.2, zorder=5))
ax1.text(zoom_x + zoom_w_bw + zoom_w_lat/2, anno_y, 'HRTT', ha='center', va='center',
         fontsize=14, color='white', fontweight='bold', zorder=10)

# explanation
ax1.text(zoom_x + zoom_w_bw + zoom_w_lat + 0.3, anno_y,
         '(on bottleneck link)', ha='left', va='center',
         fontsize=14, color=C_SUBTEXT, style='italic')


plt.savefig(os.path.join(_HERE, 'ring_allreduce_time_breakdown_0x7.png'),
            dpi=180, bbox_inches='tight', pad_inches=0.02, facecolor='none', transparent=True)
plt.close()

print("Diagram saved: ring_allreduce_time_breakdown_0x7.png")
