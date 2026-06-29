"""Render the visual pipeline figure (docs/pipeline_filmstrip.png).

A filmstrip showing what the part looks like at each stage of the pipeline:
STL mesh -> chosen orientation -> voxelization -> build simulation (layer
cross-section) -> distortion FEA. Generated from the sample bracket so the
stages are directly comparable.

Run:  python examples/make_pipeline_figure.py
"""

from __future__ import annotations

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
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from abadvisor import report as _report  # noqa: E402,F401  applies LaTeX-style rcParams
from abadvisor import shapes  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402
from abadvisor.voxelize import voxelize  # noqa: E402

_MESH_FC = "#9bb7d4"
_MESH_EC = "#33506e"


def _equalize(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect(hi - lo)
    except Exception:
        pass
    ax.set_axis_off()


def _draw_mesh(ax, mesh, title):
    ax.add_collection3d(Poly3DCollection(mesh.triangles, alpha=0.9,
                                         facecolor=_MESH_FC, edgecolor=_MESH_EC, linewidths=0.2))
    _equalize(ax, mesh.triangles.reshape(-1, 3))
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title, fontsize=11, pad=-2)


def _draw_voxels(ax, occ, pitch, title):
    ax.voxels(occ, facecolors="#9bb7d4cc", edgecolor="#33506e", linewidth=0.15)
    ax.set_box_aspect(occ.shape)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title, fontsize=11, pad=-2)


def _draw_layers(ax, sim, title):
    z = sim.layer_z_mm
    ax.fill_betweenx(z, 0, sim.layer_area_mm2, color="#2b6cb0", alpha=0.85, lw=0)
    if sim.support_layer_area_mm2 is not None and float(np.sum(sim.support_layer_area_mm2)) > 0:
        ax.fill_betweenx(z, 0, sim.support_layer_area_mm2, color="#c05621", alpha=0.8, lw=0)
    ax.set_xlabel("area (mm$^2$)", fontsize=8)
    ax.set_ylabel("build height $z$ (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11)


def _draw_fea(ax, fea, title):
    p, quads, u = fea.nodes, fea.quads, fea.u_nodal
    field, maxd = fea.mag_nodal, max(fea.max_displacement_mm, 1e-9)
    ext = p.max(axis=1) - p.min(axis=1)
    scale = 0.08 * float(ext.max()) / maxd
    defp = p + u * scale
    polys = [defp[:, quads[:, k]].T for k in range(quads.shape[1])]
    facevals = np.array([field[quads[:, k]].mean() for k in range(quads.shape[1])])
    norm = Normalize(0.0, maxd)
    cmap = plt.get_cmap("turbo")
    poly = Poly3DCollection(polys, linewidths=0.1, edgecolors=(0, 0, 0, 0.15))
    poly.set_facecolor(cmap(norm(facevals)))
    ax.add_collection3d(poly)
    _equalize(ax, defp.T)
    ax.view_init(elev=22, azim=-58)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("$|u|$ (mm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11, pad=-2)


def main() -> int:
    mesh = shapes.gantry_bracket()
    result = advise(mesh=mesh, process="lpbf_ti64", grid_n=64, fea_grid_n=24)
    oriented = result["oriented_mesh"]
    sim = result["sim"]
    fea = result["fea"]
    occ_thumb = voxelize(oriented, grid_n=16).occ

    fig = plt.figure(figsize=(17.5, 4.3))
    ax1 = fig.add_subplot(1, 5, 1, projection="3d")
    ax2 = fig.add_subplot(1, 5, 2, projection="3d")
    ax3 = fig.add_subplot(1, 5, 3, projection="3d")
    ax4 = fig.add_subplot(1, 5, 4)
    ax5 = fig.add_subplot(1, 5, 5, projection="3d")

    _draw_mesh(ax1, mesh, "1. STL part")
    _draw_mesh(ax2, oriented, "2. Orientation")
    _draw_voxels(ax3, occ_thumb, voxelize(oriented, grid_n=16).pitch, "3. Voxelization")
    _draw_layers(ax4, sim, "4. Build simulation")
    _draw_fea(ax5, fea, "5. Distortion FEA")

    # arrows between panels (figure coordinates)
    for x in (0.205, 0.405, 0.605, 0.805):
        fig.add_artist(FancyArrowPatch((x - 0.012, 0.5), (x + 0.012, 0.5),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=18, lw=1.6, color="#5b6b7b"))

    fig.suptitle("Additive Build Advisor — the part through the pipeline",
                 fontsize=13, fontweight="bold", y=1.02)
    out = ROOT / "docs" / "pipeline_filmstrip.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
