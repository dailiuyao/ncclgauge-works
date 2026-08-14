"""Recreate the 'modeling approach' diagram.

The original was a standalone PNG with no source. This script reproduces the
three-column layout (Stack -> Overhead -> Model) with curved connectors, and
renames the top overhead box 'Oss / Ors' -> 'Os (software)' and the middle
overhead box 'Osf / Orf' -> 'Of (firmware)'.
"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))  # save outputs next to this script
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

# All text: Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

fig, ax = plt.subplots(figsize=(13.02, 5.20), dpi=100)
ax.set_xlim(0, 13.02)
ax.set_ylim(0.10, 5.05)
ax.invert_yaxis()  # top-down like the image
ax.axis("off")

TEAL = "#2f8f83"
TEAL_FILL = "#e2f0ec"
SALMON = "#e07a5f"
SALMON_FILL = "#fbe6e1"
GREY = "#8a97a3"

# Stack column: dark navy -> light blue
stack_colors = ["#12304e", "#1c4a6e", "#2f6b93", "#3f86ab", "#68a7c8"]
stack_labels = ["NCCL", "Plugin", "Libfabric", "EFA", "Network"]

box_w = 2.05
box_h = 0.62


def box(x, y, w, h, facecolor, edgecolor, text, textcolor, fontsize=17, bold=True):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2.2, facecolor=facecolor, edgecolor=edgecolor, mutation_aspect=1,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=textcolor, fontsize=fontsize,
            fontweight="bold" if bold else "normal")
    return (x, y, w, h)


def curve(p0, p1, color, lw=4.5):
    """Smooth cubic bezier between two points, horizontal tangents."""
    x0, y0 = p0
    x1, y1 = p1
    mx = (x0 + x1) / 2
    path = Path([(x0, y0), (mx, y0), (mx, y1), (x1, y1)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, edgecolor=color, linewidth=lw,
                           capstyle="round", alpha=0.85))


# Column headers
ax.text(1.85, 0.45, "Stack", ha="center", color=GREY, fontsize=20, fontweight="bold")
ax.text(6.23, 0.45, "Overhead", ha="center", color=GREY, fontsize=20, fontweight="bold")
ax.text(10.95, 0.45, "Model", ha="center", color=GREY, fontsize=20, fontweight="bold")

# --- Stack boxes ---
stack_x = 0.82
stack_ys = [0.85, 1.62, 2.39, 3.16, 3.93]
stack_boxes = []
for y, c, lbl in zip(stack_ys, stack_colors, stack_labels):
    stack_boxes.append(box(stack_x, y, box_w, box_h, c, c, lbl, "white", fontsize=17))

# --- Overhead boxes (wider so labels have margin) ---
oh_w = 2.55
oh_x = 5.00
oh_top = box(oh_x, 1.02, oh_w, box_h + 0.15, TEAL_FILL, TEAL, "Os (software)", TEAL, fontsize=15)
oh_mid = box(oh_x, 2.60, oh_w, box_h + 0.15, TEAL_FILL, TEAL, "Of (firmware)", TEAL, fontsize=15)
oh_bot = box(oh_x, 4.18, oh_w, box_h + 0.15, TEAL_FILL, TEAL, "G, L", TEAL, fontsize=16)

# --- Model boxes (wider so labels have margin) ---
md_w = 3.30
md_x = 9.35
md_top = box(md_x, 1.35, md_w, 1.55, SALMON_FILL, SALMON, "Realistic (LogGP)", SALMON, fontsize=17)
md_bot = box(md_x, 4.05, md_w, 0.78, SALMON_FILL, SALMON, "Roofline (α-β)", SALMON, fontsize=17)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


# Stack -> Overhead connectors (teal)
# NCCL, Plugin -> Os (software)
curve(right(stack_boxes[0]), left(oh_top), TEAL)
curve(right(stack_boxes[1]), left(oh_top), TEAL)
# Libfabric -> Os (software) and Of (firmware)
curve(right(stack_boxes[2]), left(oh_top), TEAL)
curve(right(stack_boxes[2]), left(oh_mid), TEAL)
# EFA -> Of (firmware)
curve(right(stack_boxes[3]), left(oh_mid), TEAL)
# EFA, Network -> G, L
curve(right(stack_boxes[3]), left(oh_bot), TEAL)
curve(right(stack_boxes[4]), left(oh_bot), TEAL)

# Overhead -> Model connectors (salmon)
curve(right(oh_top), left(md_top), SALMON)
curve(right(oh_mid), left(md_top), SALMON)
curve(right(oh_bot), left(md_top), SALMON)
curve(right(oh_bot), left(md_bot), SALMON)

out = os.path.join(_HERE, 'modeling approach.png')
plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
print("saved", out)
