"""Horizontal flowchart: iterative fixed-point solve for o_net.

  T_send = T_bw + o_net + 2L.  Start o_net=0 -> T_bw = T_send - 2L.
  Build chunk timeline -> parallelism k -> BW_eff = chunk_size / T_bw.
  While BW_eff * k < NIC_BW:  increase o_net, recompute T_bw, rebuild, repeat.
  Stop when BW_eff * k == NIC_BW  ->  that o_net is the answer.
"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))  # save outputs next to this script
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Times New Roman, ONE font size everywhere
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
FS = 26   # single uniform font size

C_TEXT   = '#22303C'
C_START  = '#5B6B7A'
C_STEP   = '#356899'; C_STEP_F = '#E8EEF5'
C_DEC    = '#B0413E'; C_DEC_F  = '#F6E4E3'
C_DONE   = '#2E7D32'; C_DONE_F = '#E5F0E4'
C_ARROW  = '#5B6B7A'
C_LOOP   = '#B0413E'

fig = plt.figure(figsize=(20, 8.2), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, 44); ax.set_ylim(0, 18)

CY = 9.0          # main row y-center
BH = 2.6          # box height
TOP = CY + 3.0    # top parallel row
BOT = CY - 3.0    # bottom parallel row


def box_at(cx, cy, w, text, fc, ec, h=BH, tc=C_TEXT):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.05",
                 facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=5))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=FS, color=tc, zorder=6)


def arrow(p0, p1, color=C_ARROW, lw=2.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=22,
                 color=color, lw=lw, zorder=4))


# ---- x positions (left -> right) ----
x_start, w_start = 4.0, 6.2
x_tbw,   w_tbw   = 13.0, 8.4
x_par,   w_par   = 24.0, 9.0     # the two parallel boxes share this x-center
x_dec,   w_dec   = 36.5, 8.6

# start
box_at(x_start, CY, w_start, r'Assume $o\_net = 0$', C_START, C_START, tc='white')
arrow((x_start+w_start/2, CY), (x_tbw-w_tbw/2, CY))

# T_bw
box_at(x_tbw, CY, w_tbw, r'$T\_bw = T\_send$' '\n' r'$-\, 2L - o\_net$', C_STEP_F, C_STEP)

# --- split: T_bw feeds BOTH parallel boxes ---
xr = x_tbw + w_tbw/2       # right edge of T_bw
xl = x_par - w_par/2       # left edge of parallel boxes
# top branch (timeline -> k)
ax.plot([xr, (xr+xl)/2], [CY, CY], color=C_ARROW, lw=2.4, zorder=3)
ax.plot([(xr+xl)/2, (xr+xl)/2], [CY, TOP], color=C_ARROW, lw=2.4, zorder=3)
arrow(((xr+xl)/2, TOP), (xl, TOP))
# bottom branch (BW_eff)
ax.plot([(xr+xl)/2, (xr+xl)/2], [CY, BOT], color=C_ARROW, lw=2.4, zorder=3)
arrow(((xr+xl)/2, BOT), (xl, BOT))

box_at(x_par, TOP, w_par, 'Build chunk timeline\n(with overlap)\n$\\Rightarrow$ parallelism  $k$', C_STEP_F, C_STEP)
box_at(x_par, BOT, w_par, r'$BW\_eff = Chunk\_Size / T\_bw$', C_STEP_F, C_STEP)

# --- merge: both parallel boxes feed the decision ---
xpr = x_par + w_par/2      # right edge of parallel boxes
xdl = x_dec - w_dec/2      # left edge of decision
xm = (xpr + xdl)/2
ax.plot([xpr, xm], [TOP, TOP], color=C_ARROW, lw=2.4, zorder=3)
ax.plot([xpr, xm], [BOT, BOT], color=C_ARROW, lw=2.4, zorder=3)
ax.plot([xm, xm], [TOP, BOT], color=C_ARROW, lw=2.4, zorder=3)
arrow((xm, CY), (xdl, CY))

# decision
box_at(x_dec, CY, w_dec, r'$BW\_eff \times k$' '\n' r'$=\; NIC\_BW$ ?', C_DEC_F, C_DEC, tc=C_DEC)

# YES: decision -> down -> done
done_y = CY - 5.2
arrow((x_dec, CY-BH/2), (x_dec, done_y+1.0), color=C_DONE, lw=2.2)
ax.text(x_dec+0.4, (CY-BH/2 + done_y+1.0)/2, 'yes', ha='left', va='center',
        fontsize=FS, color=C_DONE, style='italic')
box_at(x_dec, done_y, 6.8, r'$o\_net$  found', C_DONE_F, C_DONE, h=1.9, tc=C_DONE)

# NO: feedback loop over the top -> back into T_bw
loop_y = TOP + BH/2 + 2.0
ax.plot([x_dec, x_dec], [CY+BH/2, loop_y], color=C_LOOP, lw=2.6, zorder=3)
ax.plot([x_dec, x_tbw], [loop_y, loop_y], color=C_LOOP, lw=2.6, zorder=3)
arrow((x_tbw, loop_y), (x_tbw, CY+BH/2), color=C_LOOP, lw=2.6)
ax.text((x_tbw+x_dec)/2, loop_y+0.4, r'no:  increase $o\_net$', ha='center', va='bottom',
        fontsize=FS, color=C_LOOP, style='italic')

plt.savefig(os.path.join(_HERE, 'onet-iteration.png'),
            dpi=200, bbox_inches='tight', pad_inches=0.08, facecolor='white')
plt.close()
print("Diagram saved: onet-iteration.png")
