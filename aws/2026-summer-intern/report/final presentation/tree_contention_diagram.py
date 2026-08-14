import os
_HERE = os.path.dirname(os.path.abspath(__file__))  # save outputs next to this script
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# All text: Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

FS = 15

C_BG = '#FAFBFC'
C_TEXT = '#2C3E50'
C_REDUCE = '#C0392B'
C_BCAST = '#27AE60'
C_GPU0_FC = '#EAF6EC'; C_GPU0_EC = '#27AE60'; C_GPU0_TX = '#1E8449'
C_GPU2_FC = '#EAF2FB'; C_GPU2_EC = '#3B7DD8'; C_GPU2_TX = '#2471A3'
C_GPU1_FC = '#FBEAEA'; C_GPU1_EC = '#D94F4F'; C_GPU1_TX = '#B83A3A'
C_GPU3_FC = '#FDEBD6'; C_GPU3_EC = '#E8790C'; C_GPU3_TX = '#B5650A'
C_BANNER_FC = '#FEF6E3'; C_BANNER_EC = '#E8790C'; C_BANNER_TX = '#B5650A'

fig = plt.figure(figsize=(6.6, 2.2), facecolor='none')
ax = fig.add_axes([0, 0, 1, 1])   # axes fill the whole figure (no internal margins)
ax.set_facecolor('none')
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.3)               # vertical span halved -> figure height halved (box size unchanged)

bw, bh = 1.15, 0.5   # node box size — only slightly larger than the label text
def node(cx, cy, text, fc, ec, tc):
    ax.add_patch(FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                 boxstyle="round,pad=0.03", facecolor=fc, edgecolor=ec,
                 linewidth=1.8, zorder=5))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=FS,
            color=tc, zorder=6)

# node centers — vertical layout compressed into the halved canvas
gpu0 = (5.0, 2.75)
gpu2 = (5.0, 1.95)
gpu1 = (2.7, 1.05)
gpu3 = (7.3, 1.05)

node(*gpu0, 'GPU0', C_GPU0_FC, C_GPU0_EC, C_GPU0_TX)
node(*gpu2, 'GPU2', C_GPU2_FC, C_GPU2_EC, C_GPU2_TX)
node(*gpu1, 'GPU1', C_GPU1_FC, C_GPU1_EC, C_GPU1_TX)
node(*gpu3, 'GPU3', C_GPU3_FC, C_GPU3_EC, C_GPU3_TX)

def arrow(p_from, p_to, color, rad=0.0, shrink=12):
    a = FancyArrowPatch(p_from, p_to, connectionstyle=f"arc3,rad={rad}",
                        arrowstyle='-|>', mutation_scale=15, color=color,
                        lw=2.2, shrinkA=shrink, shrinkB=shrink, zorder=3)
    ax.add_patch(a)

# vertical GPU2 <-> GPU0: reduce up (red), broadcast down (green), offset horizontally
arrow((gpu2[0]-0.2, gpu2[1]), (gpu0[0]-0.2, gpu0[1]), C_REDUCE)   # reduce up
arrow((gpu0[0]+0.2, gpu0[1]), (gpu2[0]+0.2, gpu2[1]), C_BCAST)    # broadcast down

# GPU1 <-> GPU2 : reduce (GPU1->GPU2) red, broadcast (GPU2->GPU1) green
arrow((gpu1[0]+0.1, gpu1[1]+0.1), (gpu2[0]-0.1, gpu2[1]-0.1), C_REDUCE, rad=0.12)
arrow((gpu2[0]-0.3, gpu2[1]-0.12), (gpu1[0]-0.05, gpu1[1]+0.12), C_BCAST, rad=0.12)

# GPU3 <-> GPU2 : reduce (GPU3->GPU2) red, broadcast (GPU2->GPU3) green
arrow((gpu3[0]-0.1, gpu3[1]+0.1), (gpu2[0]+0.1, gpu2[1]-0.1), C_REDUCE, rad=-0.12)
arrow((gpu2[0]+0.3, gpu2[1]-0.12), (gpu3[0]+0.05, gpu3[1]+0.12), C_BCAST, rad=-0.12)

# Legend (upper-right, kept inside the canvas)
lx = 7.9
ax.plot([lx, lx+0.5], [2.72, 2.72], color=C_REDUCE, lw=3, zorder=4)
ax.text(lx+0.62, 2.72, 'Reduce', ha='left', va='center', fontsize=FS, color=C_REDUCE)
ax.plot([lx, lx+0.5], [2.4, 2.4], color=C_BCAST, lw=3, zorder=4)
ax.text(lx+0.62, 2.4, 'Broadcast', ha='left', va='center', fontsize=FS, color=C_BCAST)

# Bottom banner
ax.add_patch(FancyBboxPatch((0.55, 0.06), 8.9, 0.5,
             boxstyle="round,pad=0.03", facecolor=C_BANNER_FC, edgecolor=C_BANNER_EC,
             linewidth=1.8, zorder=5))
ax.text(5.0, 0.31, r'3 recv + 3 send on one NIC  $\rightarrow\ \beta = ?$',
        ha='center', va='center', fontsize=FS, color=C_BANNER_TX, zorder=6)

# Title (close to the top)
ax.text(5.0, 3.18, '(2) Bandwidth contention on GPU2',
        ha='center', va='center', fontsize=FS + 3, fontweight='bold', color=C_TEXT)

plt.savefig(os.path.join(_HERE, 'tree-contention.png'),
            dpi=200, bbox_inches='tight', pad_inches=0.04, facecolor='none', transparent=True)
plt.close()
print("Diagram saved: tree-contention.png")
