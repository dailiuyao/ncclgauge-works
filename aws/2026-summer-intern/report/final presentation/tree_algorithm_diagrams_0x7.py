import os
_HERE = os.path.dirname(os.path.abspath(__file__))  # save outputs next to this script
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# All text: Times New Roman, one unified size (matches the ring breakdown figure).
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
FS = 20

# Color palette
C_BG = '#FAFBFC'
C_BOTTLENECK = '#D94F4F'
C_NORMAL = '#F5A623'
C_NORMAL_BORDER = '#E08E0B'
C_LATENCY = '#27AE60'
C_NODE_ACTIVE = '#3B7DD8'
C_NODE_DIM = '#C8D6E5'
C_TEXT = '#2C3E50'
C_SUBTEXT = '#7F8C8D'
C_REDUCE = '#D94F4F'
C_BROADCAST = '#27AE60'

fig = plt.figure(figsize=(13, 6.2), facecolor='none')

# ============================================================
# Pipeline / Gantt diagram (single panel)
# ============================================================
ax1 = fig.add_axes([0, 0, 1, 1])
ax1.set_facecolor('none')
ax1.set_xlim(-5.6, 14.5)
ax1.set_ylim(-3.5, 9)
ax1.axis('off')

# Links — reduce/broadcast now on the SAME line as the GPU arrow
links = ['GPU1/3 → GPU2 (reduce)', 'GPU2 → GPU0 (reduce)',
         'GPU0 → GPU2 (broadcast)', 'GPU2 → GPU1/3 (broadcast)']
# compressed row spacing (rows shorter than before)
row_pitch = 0.85
row_top = 5.8
y_positions = [row_top - i * row_pitch for i in range(4)]

n_chunks = 6
chunk_width_slow = 1.7
chunk_width_fast = 1.05
chunk_widths = [chunk_width_slow, chunk_width_fast, chunk_width_fast, chunk_width_fast]
chunk_height = 0.5
gap = 0.08

bottleneck_idx = 0

# Row backgrounds — left edge extended to cover the long single-line labels
bg_left = -5.4          # covers the widest label (~-5.1)
bg_right = 13.8
STRIP_PAD = 0.2         # grey strip height beyond the box (bigger = taller; stays centered)
STRIP_DY  = -0.1         # grey strip vertical shift (positive = up, negative = down)
for i, y in enumerate(y_positions):
    bg_color = '#EEF2F7' if i % 2 == 0 else C_BG
    rect = mpatches.FancyBboxPatch((bg_left, y - chunk_height/2 - STRIP_PAD + STRIP_DY),
                                    bg_right - bg_left, chunk_height + 2*STRIP_PAD,
                                    boxstyle="round,pad=0.05", facecolor=bg_color,
                                    edgecolor='none', zorder=0)
    ax1.add_patch(rect)

# Link labels (single line: arrow + reduce/broadcast together)
for i, (link, y) in enumerate(zip(links, y_positions)):
    weight = 'bold' if i == bottleneck_idx else 'medium'
    color = C_BOTTLENECK if i == bottleneck_idx else C_TEXT
    ax1.text(-0.5, y, link, ha='right', va='center', fontsize=FS,
             fontweight=weight, color=color)

# Chunk position calculation
def get_start(link_idx, chunk_i):
    return chunk_i * chunk_width_slow + sum(chunk_widths[:link_idx])

# Draw chunks
for link_idx in range(4):
    y = y_positions[link_idx]
    w = chunk_widths[link_idx]
    for chunk_i in range(n_chunks):
        start = get_start(link_idx, chunk_i)

        if link_idx == bottleneck_idx:
            fc = C_BOTTLENECK
            ec = '#B83A3A'
        else:
            fc = C_NORMAL
            ec = C_NORMAL_BORDER

        # Rectangle with NO pad expansion so every box (red & yellow) has the exact
        # same drawn height = chunk_height, centered on the row's y.
        rect = FancyBboxPatch((start + gap, y - chunk_height/2),
                              w - 2*gap, chunk_height,
                              boxstyle="round,pad=0",
                              facecolor=fc, edgecolor=ec, linewidth=1.2,
                              alpha=0.9, zorder=5)
        ax1.add_patch(rect)

        ax1.text(start + w/2, y, f'C{chunk_i+1}', ha='center', va='center',
                 fontsize=FS, color='white', fontweight='bold', zorder=10)

# Brackets
bw_start = get_start(0, 0)
bw_end = get_start(0, n_chunks - 1) + chunk_width_slow
lat_start = bw_end
lat_end = get_start(3, n_chunks - 1) + chunk_widths[3]
total_start = bw_start
total_end = lat_end

# --- bracket / annotation y-positions retuned for the compressed rows ---
row_top_edge = y_positions[0] + chunk_height/2      # ~6.1
row_bot_edge = y_positions[-1] - chunk_height/2     # ~2.3

# Total bracket
bracket_y_total = row_top_edge + 0.85
ax1.annotate('', xy=(total_start, bracket_y_total), xytext=(total_end, bracket_y_total),
             arrowprops=dict(arrowstyle='<->', color=C_TEXT, lw=2.5))
ax1.text((total_start + total_end) / 2, bracket_y_total + 0.12,
         'Total AllReduce Time', ha='center', va='bottom',
         fontsize=FS, fontweight='bold', color=C_TEXT)

# Bandwidth part bracket
bracket_y = row_top_edge + 0.28
ax1.annotate('', xy=(bw_start, bracket_y), xytext=(bw_end, bracket_y),
             arrowprops=dict(arrowstyle='<->', color=C_BOTTLENECK, lw=2))
ax1.text((bw_start + bw_end) / 2, bracket_y + 0.12,
         'Bandwidth Part', ha='center', va='bottom',
         fontsize=FS, fontweight='bold', color=C_BOTTLENECK)

# Latency part bracket
ax1.annotate('', xy=(lat_start, bracket_y), xytext=(lat_end, bracket_y),
             arrowprops=dict(arrowstyle='<->', color=C_LATENCY, lw=2))
ax1.text((lat_start + lat_end) / 2, bracket_y + 0.12,
         'Latency Part', ha='center', va='bottom',
         fontsize=FS, fontweight='bold', color=C_LATENCY)

# annotation text baseline below the rows
anno_y = row_bot_edge - 1.15

# Bottleneck annotation
ax1.annotate('Bottleneck link (determines throughput)',
             xy=(get_start(0, 3) + chunk_width_slow/2, y_positions[0] - chunk_height/2 - 0.05),
             xytext=(get_start(0, 3) + chunk_width_slow/2, anno_y),
             fontsize=FS, ha='center', color=C_BOTTLENECK, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_BOTTLENECK, lw=2,
                            connectionstyle='arc3,rad=0'))

# Latency annotation
ax1.annotate('Last chunk travels\nthrough entire tree',
             xy=(get_start(3, n_chunks-1) + chunk_widths[3]/2, y_positions[3] - chunk_height/2 - 0.05),
             xytext=(get_start(3, n_chunks-1) + chunk_widths[3]/2, anno_y),
             fontsize=FS, ha='center', color=C_LATENCY, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_LATENCY, lw=2,
                            connectionstyle='arc3,rad=0'))

# Time axis — raise closer to the rows by shrinking TIME_GAP
TIME_GAP = 0.4
time_y = anno_y - TIME_GAP
ax1.annotate('', xy=(total_end + 0.8, time_y), xytext=(bw_start - 0.3, time_y),
             arrowprops=dict(arrowstyle='->', color=C_SUBTEXT, lw=1.5))
ax1.text((total_start + total_end) / 2, time_y - 0.28, 'Time',
         ha='center', va='top', fontsize=FS, color=C_SUBTEXT, style='italic')

# tighten the panel vertical extent to the content
ax1.set_ylim(time_y - 0.7, bracket_y_total + 0.7)

plt.savefig(os.path.join(_HERE, 'tree_allreduce_time_breakdown_0x7.png'),
            dpi=180, bbox_inches='tight', pad_inches=0.02, facecolor='none', transparent=True)
plt.close()

print("Diagram saved: tree_allreduce_time_breakdown_0x7.png")
