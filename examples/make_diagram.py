"""Render the system / logic diagram for the advisor (docs/system_diagram.png).

A clean top-to-bottom flow: geometry in, orientation, voxelization, the analyses
that run on the voxel model (build simulation, distortion FEA, DfAM) plus
inspection planning from the tolerances, all feeding the release gate, which
emits the digital-thread record and hands off to the runtime monitoring twin.

Run:  python examples/make_diagram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# LaTeX-style serif (Computer Modern) look, no usetex (robust + fast).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

_INK = "#1f2933"
_W = 14.0   # canvas width


def box(ax, cx, cy, w, h, title, body, fc, ec, tfs=11, bfs=9.0):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.10",
                                linewidth=1.4, edgecolor=ec, facecolor=fc, mutation_aspect=0.6))
    if body:
        ax.text(cx, cy + h * 0.20, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=_INK)
        ax.text(cx, cy - h * 0.24, body, ha="center", va="center",
                fontsize=bfs, color="#3b4754")
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=_INK)


def arrow(ax, x1, y1, x2, y2, color="#5b6b7b"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=1.4, color=color,
                                 shrinkA=1, shrinkB=1))


def main() -> int:
    fig, ax = plt.subplots(figsize=(11.5, 11.0))
    ax.set_xlim(0, _W)
    ax.set_ylim(0, 15.4)
    ax.axis("off")
    cx = _W / 2

    blue, blue_e = "#e9f1fb", "#2b6cb0"
    gray, gray_e = "#eef1f5", "#6b7682"
    green, green_e = "#e7f5ee", "#2f855a"
    amber, amber_e = "#fdf1da", "#b8860b"

    ax.text(cx, 14.9, "Additive Build Advisor — design-to-inspection digital thread",
            ha="center", va="center", fontsize=13, fontweight="bold", color=_INK)

    box(ax, cx, 14.0, 9.2, 0.85, "STL part geometry  +  tolerance spec (JSON)", "", gray, gray_e, 11.5)
    arrow(ax, cx, 13.55, cx, 13.15)
    box(ax, cx, 12.6, 9.6, 1.0, "Geometry kernel",
        "recompute normals · watertight check · volume / bbox", blue, blue_e)
    arrow(ax, cx, 12.05, cx, 11.65)
    box(ax, cx, 11.1, 9.6, 1.0, "Orientation screening  (rest-on-face)",
        "score: support volume · base contact · build height", blue, blue_e)
    arrow(ax, cx, 10.55, cx, 10.15)
    box(ax, cx, 9.65, 9.6, 0.9, "Voxelization", "ray-stabbing occupancy grid", blue, blue_e)

    # fan out to four analyses
    xs = [2.55, 5.55, 8.45, 11.45]
    for x in xs:
        arrow(ax, cx, 9.15, x, 8.05)
    box(ax, xs[0], 7.35, 2.7, 1.35, "Build sim", "layers · support\ntime · cost", blue, blue_e, 10, 8.5)
    box(ax, xs[1], 7.35, 2.7, 1.35, "Distortion FEA", "inherent strain\n(scikit-fem, LPBF)", amber, amber_e, 10, 8.5)
    box(ax, xs[2], 7.35, 2.7, 1.35, "DfAM checks", "walls · support\naspect · voids", blue, blue_e, 10, 8.5)
    box(ax, xs[3], 7.35, 2.7, 1.35, "Inspection plan", "tolerances to\nmethod + capability", blue, blue_e, 10, 8.5)

    for x in xs:
        arrow(ax, x, 6.65, cx, 5.55)
    box(ax, cx, 4.95, 10.8, 1.15, "Release gate  (verify before act)",
        "release_to_build   ·   needs_engineering_review   ·   redesign_required",
        green, green_e, 11.5, 9.5)
    arrow(ax, cx, 4.35, cx, 3.95)
    box(ax, cx, 3.45, 9.6, 0.9, "Digital-thread record", "machine-readable JSON  +  HTML report", gray, gray_e)
    arrow(ax, cx, 3.0, cx, 2.6)
    box(ax, cx, 2.05, 9.6, 1.0, "Hand-off to runtime monitoring twin",
        "mini-manufacturing-digital-twin · as-built monitoring", gray, gray_e)

    out = ROOT / "docs" / "system_diagram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
