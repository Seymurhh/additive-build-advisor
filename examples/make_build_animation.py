"""Render a layer-by-layer build-simulation animation (docs/build_animation.gif).

Shows the chosen part building up one layer at a time on the plate (voxels
stacking, coloured by height) alongside a live metrics panel: current layer,
build height, percent complete, and material deposited, with the per-layer
cross-section filling in as the build progresses. This is the additive analog of
the runtime CNC digital-twin dashboard.

Run:  python examples/make_build_animation.py
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
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from abadvisor import report as _report  # noqa: E402,F401  LaTeX-style rcParams
from abadvisor import shapes  # noqa: E402
from abadvisor.materials import get_profile  # noqa: E402
from abadvisor.orientation import optimize_orientation  # noqa: E402
from abadvisor.voxelize import voxelize  # noqa: E402

GRID_N = 22
PROCESS = "fff_pla"


def main() -> int:
    prof = get_profile(PROCESS)
    mesh = shapes.gantry_bracket()
    best = optimize_orientation(mesh, prof)["best"]
    oriented = mesh.transformed(best.rotation).dropped_to_plate()
    grid = voxelize(oriented, grid_n=GRID_N)
    occ = grid.occ
    nx, ny, nz = occ.shape
    pitch = grid.pitch
    cell_vol = pitch ** 3
    cell_area = pitch ** 2

    layer_counts = occ.sum(axis=(0, 1))
    cum_vox = np.cumsum(layer_counts)
    total_vox = int(occ.sum())
    z_mm = (np.arange(nz) + 1) * pitch
    area_mm2 = layer_counts * cell_area

    # rough build-time estimate for the metrics readout
    height = nz * pitch
    n_real_layers = max(1, int(np.ceil(height / prof.default_layer_height_mm)))
    est_total_h = (total_vox * cell_vol / 1000.0) / prof.nominal_volume_rate_cm3_per_h \
        + n_real_layers * prof.recoat_time_s_per_layer / 3600.0

    # per-voxel colour by height (viridis)
    cmap = plt.get_cmap("viridis")
    kz = np.arange(nz)
    layer_rgba = cmap(kz / max(nz - 1, 1))
    facecolors = np.zeros(occ.shape + (4,))
    for k in range(nz):
        facecolors[:, :, k, :] = layer_rgba[k]

    fig = plt.figure(figsize=(10.5, 4.8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axm = fig.add_subplot(1, 2, 2)

    def draw(k):
        ax3d.cla()
        filled = np.zeros_like(occ)
        filled[:, :, : k + 1] = occ[:, :, : k + 1]
        if filled.any():
            ax3d.voxels(filled, facecolors=facecolors, edgecolor=(0, 0, 0, 0.12), linewidth=0.1)
        ax3d.set_xlim(0, nx); ax3d.set_ylim(0, ny); ax3d.set_zlim(0, nz)
        try:
            ax3d.set_box_aspect((nx, ny, nz))
        except Exception:
            pass
        ax3d.view_init(elev=24, azim=-58)
        ax3d.set_axis_off()
        ax3d.set_title(f"Building — layer {k + 1}/{nz}", fontsize=12)

        axm.cla()
        axm.plot(area_mm2, z_mm, color="#b8c2cc", lw=1.2)
        axm.fill_betweenx(z_mm[: k + 1], 0, area_mm2[: k + 1], step="mid", color="#2b6cb0", alpha=0.85, lw=0)
        axm.axhline(z_mm[k], color="#c05621", lw=1.0, ls="--")
        axm.set_xlim(0, area_mm2.max() * 1.08)
        axm.set_ylim(0, z_mm[-1] * 1.02)
        axm.set_xlabel("cross-section area (mm$^2$)")
        axm.set_ylabel("build height $z$ (mm)")
        axm.set_title("Per-layer cross-section", fontsize=12)

        pct = 100.0 * (k + 1) / nz
        mat = cum_vox[k] * cell_vol / 1000.0
        txt = (f"process: {prof.name}\n"
               f"layer: {k + 1} / {nz}\n"
               f"height: {z_mm[k]:.1f} mm\n"
               f"complete: {pct:.0f}%\n"
               f"material: {mat:.2f} cm$^3$")
        axm.text(0.97, 0.05, txt, transform=axm.transAxes, ha="right", va="bottom",
                 fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f6f8fb", ec="#c8cdd4"))
        return []

    fig.suptitle("Additive build simulation — layer by layer", fontsize=13, fontweight="bold", y=0.99)
    anim = FuncAnimation(fig, draw, frames=nz, interval=220)
    out = ROOT / "docs" / "build_animation.gif"
    anim.save(str(out), writer=PillowWriter(fps=6))
    plt.close(fig)
    print(f"Wrote {out}  ({nz} frames, grid {occ.shape}, est build ~{est_total_h:.1f} h)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
