"""Validate the distortion FEA and render a validation figure for the docs.

Two checks, both saved to ``docs/fea_validation.png``:

1. **Analytical + mesh convergence.** A clamped prismatic bar under a uniform
   contraction eigenstrain has an analytical top displacement of |eps*| x height.
   We refine the mesh and show the FEA converging to it.
2. **Process sensitivity.** The same bracket solved across every process family,
   showing predicted warpage scaling with the per-process thermal-contraction
   strain (ABS > PLA > SLS > SLA) -- ABS warps more than PLA, as it does in
   practice.

Run:  python examples/validate_fea.py
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

from abadvisor import report as _report  # noqa: E402,F401  (applies publication style)
from abadvisor import shapes  # noqa: E402
from abadvisor.fea import solve_thermal_warp  # noqa: E402
from abadvisor.materials import list_profiles  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402


def main() -> int:
    # ---- 1. clamped-bar convergence -----------------------------------
    eps, H = -0.01, 40.0
    analytic = abs(eps) * H
    pitches, errs = [], []
    print("Clamped-bar convergence (analytic top displacement = %.3f mm):" % analytic)
    for nz in (10, 20, 40, 80):
        pitch = H / nz
        cxy = max(1, int(round(8.0 / pitch)))
        occ = np.ones((cxy, cxy, nz), dtype=bool)
        r = solve_thermal_warp(occ, pitch, E=70000.0, nu=0.33, eigenstrain=eps, tol=1e-9)
        pitches.append(pitch)
        errs.append(100.0 * (r.max_displacement_mm - analytic) / analytic)
        print(f"  pitch {pitch:4.1f} mm  ->  {r.max_displacement_mm:.4f} mm  ({errs[-1]:+.2f}%)")

    # ---- 2. process sensitivity on one part ---------------------------
    names, dist = [], []
    print("\nBracket warpage by process:")
    for prof in list_profiles():
        r = advise(mesh=shapes.gantry_bracket(), process=prof.key, grid_n=32, fea_grid_n=20)
        names.append(prof.material)
        dist.append(r["record"]["distortion_fea"]["max_distortion_mm"])
        print(f"  {prof.name:28s} eps*={prof.contraction_strain:+.3f}  ->  {dist[-1]:.3f} mm")

    # ---- 3. warp-prone geometry (long flat cantilever bar, ABS) ------
    # A flat slender bar is the worst case for cooling warpage; ABS warps most.
    cant = advise(mesh=shapes.cantilever_benchmark(), process="fff_abs", grid_n=40, fea_grid_n=22)
    cd = cant["record"]["distortion_fea"]
    print("\nWarp-prone flat cantilever bar (ABS, the FFF material that warps most):")
    print(f"  predicted ON-BED peak warpage {cd['max_distortion_mm']:.3f} mm "
          f"(part still bonded to the bed).")
    print("  NOTE: this is the on-bed field, a relative screen -- not the spring-back")
    print("  after the part is peeled off the bed, which needs a release step + a")
    print("  contraction strain calibrated to a measured cooling history.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.axhline(0.0, color="#888", lw=1, ls="--", label="analytical")
    a1.plot(pitches, errs, "o-", color="#2c6fbb")
    a1.set_xlabel("voxel pitch (mm)")
    a1.set_ylabel("error vs analytical (%)")
    a1.set_title("Clamped-bar convergence\n(FEA error vs analytical |eps*|*H as mesh refines)")
    a1.invert_xaxis()
    a1.grid(alpha=0.3)

    colors = ["#b22222" if "ABS" in n else "#9bb7d4" for n in names]
    a2.bar(range(len(names)), dist, color=colors)
    a2.set_xticks(range(len(names)))
    a2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a2.set_ylabel("FEA peak distortion (mm)")
    a2.set_title("Process sensitivity — same bracket\n(warpage tracks thermal-contraction strain)")
    fig.tight_layout()

    out = ROOT / "docs" / "fea_validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
