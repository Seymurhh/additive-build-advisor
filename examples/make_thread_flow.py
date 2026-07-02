"""Render the end-to-end digital-thread flow figure (docs/thread_flow.png).

One image that reads left-to-right as the whole thread — **design -> build ->
inspect -> decide** — built from a *real* pipeline run (the sample bracket on
FFF/PLA), not a mock-up:

    DESIGN    part in its chosen build orientation (rest-on-face)
    BUILD     voxel build model  +  thermal-contraction warpage FEA
    INSPECT   DfAM severities  +  first-article inspection capability
    DECIDE    the release-gate verdict + hand-off

Each panel is captioned with the actual numbers from the run, and the four
phases are banded with the same colors the web app uses.

Run:  python examples/make_thread_flow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from abadvisor import shapes  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402
from abadvisor.voxelize import voxelize  # noqa: E402

# Modern sans-serif so the flow figure matches the web app's look.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})

INK, MUTED, LINE = "#1a2230", "#67728a", "#dfe5ee"
PART_FC, PART_EC = "#9bb7d4", "#33506e"
PHASES = [("DESIGN", "#2b6cb0"), ("BUILD", "#0f766e"),
          ("INSPECT", "#6d28d9"), ("DECIDE", "#b45309")]
SEV = {"ok": "#1b7f3b", "info": "#2c6fbb", "warning": "#b8860b", "critical": "#b22222"}
GATE = {"release_to_build": ("#1b7f3b", "RELEASE TO BUILD"),
        "needs_engineering_review": ("#b8860b", "NEEDS ENGINEERING REVIEW"),
        "redesign_required": ("#b22222", "REDESIGN REQUIRED")}

# five panels laid out left->right (design | build-a | build-b | inspect | decide)
_W, _GAP, _Y, _H = 0.164, 0.035, 0.135, 0.56
_LEFTS = [round(0.03 + i * (_W + _GAP), 4) for i in range(5)]


def _rect(i):
    return [_LEFTS[i], _Y, _W, _H]


def _equalize(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass
    ax.set_axis_off()


def _draw_oriented(ax, mesh):
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.92,
                                         facecolor=PART_FC, edgecolor=PART_EC, linewidths=0.25))
    _equalize(ax, mesh.triangles.reshape(-1, 3))
    ax.view_init(elev=20, azim=-58)


def _draw_voxels(ax, occ):
    ax.voxels(occ, facecolors="#9bb7d4dd", edgecolor="#33506e", linewidth=0.12)
    ax.set_box_aspect(occ.shape)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-58)


def _draw_fea(ax, fea):
    p, quads, u = fea.nodes, fea.quads, fea.u_nodal
    field, maxd = fea.mag_nodal, max(fea.max_displacement_mm, 1e-9)
    ext = p.max(axis=1) - p.min(axis=1)
    defp = p + u * (0.09 * float(ext.max()) / maxd)
    polys = [defp[:, quads[:, k]].T for k in range(quads.shape[1])]
    facevals = np.array([field[quads[:, k]].mean() for k in range(quads.shape[1])])
    norm, cmap = Normalize(0.0, maxd), plt.get_cmap("turbo")
    poly = Poly3DCollection(polys, linewidths=0.08, edgecolors=(0, 0, 0, 0.12))
    poly.set_facecolor(cmap(norm(facevals)))
    ax.add_collection3d(poly)
    _equalize(ax, defp.T)
    ax.view_init(elev=20, azim=-58)
    cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            orientation="horizontal", fraction=0.05, pad=0.02, aspect=22)
    cb.set_label(r"$|u|$ (mm)", fontsize=7.5, color=MUTED)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED)
    cb.outline.set_visible(False)


def _draw_scorecard(ax, rec):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    dfam = rec["manufacturability"]["findings"]
    ax.text(0.0, 0.97, "Manufacturability (DfAM)", fontsize=9.5, fontweight="bold", color=INK)
    y = 0.86
    for f in dfam[:5]:
        ax.add_patch(Rectangle((0.0, y - 0.028), 0.055, 0.055, facecolor=SEV.get(f["severity"], "#777"),
                               edgecolor="none", transform=ax.transAxes))
        ax.text(0.09, y, f["check"].replace("_", " "), fontsize=8.2, color="#3b4754", va="center")
        ax.text(1.0, y, f["severity"], fontsize=7.2, color=SEV.get(f["severity"], "#777"),
                fontweight="bold", ha="right", va="center")
        y -= 0.105
    insp = rec["inspection_plan"]
    ax.plot([0, 1], [y + 0.02, y + 0.02], color=LINE, lw=1.0)
    ax.text(0.0, y - 0.04, "Inspection plan", fontsize=9.5, fontweight="bold", color=INK)
    tol = insp.get("tightest_tolerance_mm")
    lines = [
        f"tightest tolerance   ±{tol} mm" if tol is not None else "tightest tolerance   n/a",
        f"steps / flagged        {insp.get('n_steps', 0)} / {insp.get('n_capability_flags', 0)}",
        f"requires CMM          {'yes' if insp.get('requires_cmm') else 'no'}",
    ]
    yy = y - 0.14
    for ln in lines:
        ax.text(0.0, yy, ln, fontsize=8.0, color="#3b4754", family="monospace")
        yy -= 0.095


def _draw_verdict(ax, rec):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    gate = rec["gate"]
    color, label = GATE.get(gate["decision"], ("#b8860b", gate["decision"].upper()))
    ax.add_patch(FancyBboxPatch((0.02, 0.30), 0.96, 0.66,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=color, edgecolor="none", transform=ax.transAxes))
    ax.text(0.5, 0.85, "RELEASE GATE", fontsize=8, color="#ffffffcc", fontweight="bold",
            ha="center", va="center")
    words = label.split()
    disp = label if len(words) <= 2 else " ".join(words[:2]) + "\n" + " ".join(words[2:])
    ax.text(0.5, 0.60, disp, fontsize=12.5, color="white", fontweight="bold",
            ha="center", va="center", linespacing=1.15)
    ax.text(0.5, 0.375, f"confidence {gate.get('confidence', 0):.2f}", fontsize=8.5,
            color="#ffffffdd", ha="center", va="center")
    # confidence meter
    conf = float(gate.get("confidence", 0))
    ax.add_patch(Rectangle((0.14, 0.34), 0.72, 0.018, facecolor="#ffffff55", edgecolor="none",
                           transform=ax.transAxes))
    ax.add_patch(Rectangle((0.14, 0.34), 0.72 * conf, 0.018, facecolor="white", edgecolor="none",
                           transform=ax.transAxes))
    n = len(gate.get("reasons", []))
    ax.text(0.5, 0.20, f"{n} reason{'s' if n != 1 else ''} on record", fontsize=8, color=MUTED,
            ha="center", va="center")
    ax.text(0.5, 0.09, "hand-off to runtime FFF print twin", fontsize=7.6, color=MUTED,
            ha="center", va="center", style="italic")


def _band(fig, x0, x1, name, color):
    fig.add_artist(FancyBboxPatch((x0, 0.80), x1 - x0, 0.075,
                                  boxstyle="round,pad=0.004,rounding_size=0.02",
                                  facecolor=color, edgecolor="none", transform=fig.transFigure))
    fig.text((x0 + x1) / 2, 0.8375, name, ha="center", va="center", color="white",
             fontsize=11, fontweight="bold")


def _caption(fig, i, text):
    x = _LEFTS[i] + _W / 2
    fig.text(x, 0.065, text, ha="center", va="top", fontsize=7.8, color="#3b4754",
             linespacing=1.5)


def main() -> int:
    tol = json.loads((ROOT / "examples" / "tolerances_bracket.json").read_text())
    result = advise(mesh=shapes.gantry_bracket(), process="fff_pla",
                    tolerance_spec=tol, grid_n=64, fea_grid_n=24)
    rec = result["record"]
    oriented = result["oriented_mesh"]
    fea = result["fea"]
    occ = voxelize(oriented, grid_n=18).occ
    sim = rec["simulation"]

    fig = plt.figure(figsize=(17.5, 5.4))

    ax0 = fig.add_axes(_rect(0), projection="3d")
    ax1 = fig.add_axes(_rect(1), projection="3d")
    ax2 = fig.add_axes(_rect(2), projection="3d")
    ax3 = fig.add_axes(_rect(3))
    ax4 = fig.add_axes(_rect(4))

    _draw_oriented(ax0, oriented)
    _draw_voxels(ax1, occ)
    _draw_fea(ax2, fea)
    _draw_scorecard(ax3, rec)
    _draw_verdict(ax4, rec)

    # phase bands
    _band(fig, _LEFTS[0], _LEFTS[0] + _W, "DESIGN", PHASES[0][1])
    _band(fig, _LEFTS[1], _LEFTS[2] + _W, "BUILD", PHASES[1][1])
    _band(fig, _LEFTS[3], _LEFTS[3] + _W, "INSPECT", PHASES[2][1])
    _band(fig, _LEFTS[4], _LEFTS[4] + _W, "DECIDE", PHASES[3][1])

    # flow arrows between panels
    rights = [_LEFTS[i] + _W for i in range(5)]
    for a, b in zip(rights[:-1], _LEFTS[1:]):
        xc = (a + b) / 2
        fig.add_artist(FancyArrowPatch((xc - 0.010, _Y + _H / 2), (xc + 0.010, _Y + _H / 2),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=20, lw=1.8, color="#94a3b8"))

    # captions with real numbers
    down = rec["design_decision"]["chosen_orientation"]
    _caption(fig, 0, f"watertight: yes · rests on a flat face\n{down.get('support_volume_mm3', 0):.0f} mm³ support")
    _caption(fig, 1, f"ray-stabbing voxel model\n{sim['n_layers']} layers · {sim['build_time_h']:.2f} h · ${sim['total_cost_usd']:.2f}")
    _caption(fig, 2, f"thermal-contraction FEA (scikit-fem)\npeak {rec['distortion_fea']['max_distortion_mm']:.3f} mm off the bed")
    _caption(fig, 3, f"severity-ranked checks +\nas-built capability vs. tolerances")
    _caption(fig, 4, f"release · review · redesign\nrecord out → runtime FFF print twin")

    fig.suptitle("Additive Build Advisor — the design-to-inspection digital thread",
                 x=0.5, y=0.965, fontsize=14.5, fontweight="bold", color=INK)

    out = ROOT / "docs" / "thread_flow.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
