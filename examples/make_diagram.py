"""Render the system / logic diagram for the advisor (docs/system_diagram.png).

A clean top-to-bottom flow: geometry in, orientation, voxelization, the three
analyses that run on the voxel model (build simulation, distortion FEA, DfAM)
plus inspection planning from the tolerances, all feeding the release gate, which
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

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})

_INK = "#1f2933"


def box(ax, x, y, w, h, text, fc, ec, fs=10, fw="normal", tc=_INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight=fw, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#5b6b7b"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.3, color=color,
                                 shrinkA=2, shrinkB=2))


def main() -> int:
    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 13)
    ax.axis("off")

    blue, blue_e = "#e8f0fb", "#2b6cb0"
    gray, gray_e = "#eef1f5", "#7b8794"
    green, green_e = "#e6f4ec", "#2f855a"
    amber, amber_e = "#fdf0d9", "#b8860b"

    # input
    box(ax, 6, 12.3, 4.2, 0.9, "STL part geometry  +  tolerance spec (JSON)", gray, gray_e, 10, "bold")
    arrow(ax, 6, 11.85, 6, 11.5)
    # geometry
    box(ax, 6, 11.0, 6.6, 0.9,
        "Geometry kernel\nrecompute normals · watertight check · volume / bbox", blue, blue_e, 9.5)
    arrow(ax, 6, 10.55, 6, 10.2)
    # orientation
    box(ax, 6, 9.7, 6.6, 0.9,
        "Orientation screening (rest-on-face)\nscore: support volume · base contact · height", blue, blue_e, 9.5)
    arrow(ax, 6, 9.25, 6, 8.9)
    # voxelize
    box(ax, 6, 8.4, 6.6, 0.8, "Voxelization  (ray-stabbing occupancy grid)", blue, blue_e, 9.5)

    # fan out to four analyses
    xs = [1.9, 4.6, 7.4, 10.1]
    for x in xs:
        arrow(ax, 6, 8.0, x, 7.25)
    box(ax, xs[0], 6.7, 2.5, 1.15, "Build simulation\nlayers · support\ntime · cost", blue, blue_e, 8.5)
    box(ax, xs[1], 6.7, 2.5, 1.15, "Distortion FEA\ninherent strain\n(metal LPBF)", amber, amber_e, 8.5, "bold")
    box(ax, xs[2], 6.7, 2.5, 1.15, "DfAM checks\nwalls · support\naspect · voids", blue, blue_e, 8.5)
    box(ax, xs[3], 6.7, 2.5, 1.15, "Inspection plan\ntolerances to\nmethod + capability", blue, blue_e, 8.5)

    # converge to gate
    for x in xs:
        arrow(ax, x, 6.1, 6, 5.25)
    box(ax, 6, 4.7, 8.4, 1.05,
        "Release gate  (verify before act)\nrelease_to_build   ·   needs_engineering_review   ·   redesign_required",
        green, green_e, 9.5, "bold")
    arrow(ax, 6, 4.15, 6, 3.8)
    # record
    box(ax, 6, 3.3, 6.6, 0.85, "Digital-thread record  (machine-readable JSON + HTML report)", gray, gray_e, 9.5)
    arrow(ax, 6, 2.85, 6, 2.5)
    # handoff
    box(ax, 6, 2.0, 7.4, 0.9,
        "hand-off to runtime monitoring twin\n(mini-manufacturing-digital-twin: as-built monitoring)", gray, gray_e, 9)

    ax.text(6, 0.7, "Additive Build Advisor — design-to-inspection digital thread",
            ha="center", va="center", fontsize=11, fontweight="bold", color=_INK)

    out = ROOT / "docs" / "system_diagram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
