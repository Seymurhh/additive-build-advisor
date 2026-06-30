"""Render the cross-process comparison figure (docs/process_comparison.png).

Runs the *same part* (the gantry bracket) through three processes — FFF (PLA),
SLA (resin), and metal LPBF (AlSi10Mg) — and compares build time, cost, support
material, and predicted distortion. FFF is the home process; the metal LPBF bar
shows the same pipeline + thermal-contraction FEA (there, the inherent-strain
method) on metal, as a point of comparison.

Run:  python examples/make_comparison_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from abadvisor import report as _report  # noqa: E402,F401  applies LaTeX-style rcParams
from abadvisor import shapes  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402

PROCESSES = [
    ("FFF\n(PLA)", "fff_pla", "#2b6cb0"),
    ("SLA\n(resin)", "sla_resin", "#2f855a"),
    ("metal LPBF\n(AlSi10Mg)", "lpbf_alsi10mg", "#b22222"),
]


def main() -> int:
    rows = []
    for label, key, color in PROCESSES:
        r = advise(mesh=shapes.gantry_bracket(), process=key, grid_n=48, fea_grid_n=22)
        sim = r["record"]["simulation"]
        fea = r["record"]["distortion_fea"]
        rows.append({
            "label": label, "color": color,
            "time": sim["build_time_h"], "cost": sim["total_cost_usd"],
            "layers": sim["n_layers"], "distortion": fea["max_distortion_mm"],
        })
        print(f"  {key:16s} time={sim['build_time_h']:.2f}h cost=${sim['total_cost_usd']:.2f} "
              f"layers={sim['n_layers']} distortion={fea['max_distortion_mm']:.3f}mm")

    labels = [r["label"] for r in rows]
    colors = [r["color"] for r in rows]
    metrics = [
        ("Build time (h)", [r["time"] for r in rows]),
        ("Cost (USD)", [r["cost"] for r in rows]),
        ("Layers", [r["layers"] for r in rows]),
        ("FEA distortion (mm)", [r["distortion"] for r in rows]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7))
    for ax, (title, vals) in zip(axes, metrics):
        ax.bar(range(len(labels)), vals, color=colors, width=0.66)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10.5)
        ax.margins(y=0.18)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Same bracket across three processes", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    out = ROOT / "docs" / "process_comparison.png"
    fig.savefig(out, dpi=190, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
