"""Render the runtime FFF print-twin dashboard figure (docs/twin_dashboard.png).

The additive analog of a runtime CNC process monitor: where the advisor's release
gate hands off, this is what the companion print twin shows. It runs the twin's
*actual* simulator, anomaly detector, and recommender (no mock-up) on the sample
bracket under the multi-fault stress scenario, and lays out the dashboard:

    * a smoothed runtime health score,
    * hotend / bed temperature, extrusion flow, vibration, and corner-lift, each
      against its expected envelope with detected anomaly windows shaded,
    * the verify-before-act recommendation — including the sensor-dropout window
      where the twin explicitly *holds* rather than acting on incomplete data.

Run:  python examples/make_twin_dashboard.py
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
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from abadvisor import shapes  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402
from abadvisor.runtime_twin import simulate_runtime  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

INK, MUTED = "#1a2230", "#67728a"
SEV = {"ok": "#1b7f3b", "info": "#2c6fbb", "warning": "#b8860b", "critical": "#b22222"}
ORDER = ["hotend_c", "bed_c", "flow_pct", "vibration_g", "corner_lift_um"]
FAULT_LABEL = {"flow_pct": "under-extrusion", "vibration_g": "layer shift",
               "corner_lift_um": "warping / corner-lift", "hotend_c": "thermal drift"}


def main() -> int:
    import json
    tol = json.loads((ROOT / "examples" / "tolerances_bracket.json").read_text())
    rec = advise(mesh=shapes.gantry_bracket(), process="fff_pla",
                 tolerance_spec=tol, grid_n=64, fea_grid_n=24)["record"]
    twin = simulate_runtime(rec, "stress_demo")
    x = twin.layers

    fig, axes = plt.subplots(6, 1, figsize=(13, 11.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[0.75, 1, 1, 1, 1, 1], hspace=0.42))
    fig.subplots_adjust(top=0.90, bottom=0.20, left=0.085, right=0.975)

    # ---- health score ------------------------------------------------
    ax = axes[0]
    ax.plot(x, twin.health * 100, color="#0891b2", lw=2.0)
    ax.fill_between(x, 0, twin.health * 100, color="#0891b2", alpha=0.10)
    ax.set_ylim(0, 105); ax.set_ylabel("health %", fontsize=9)
    ax.set_title("Runtime health score", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.grid(True, color="#eef2f7")

    # ---- sensor channels --------------------------------------------
    for i, key in enumerate(ORDER):
        ax = axes[i + 1]
        s = twin.sensors[key]
        ax.fill_between(x, s["lo"], s["hi"], color="#94a3b8", alpha=0.16, lw=0)
        ax.plot(x, s["actual"], color="#1e4e8c", lw=1.35)
        ax.set_ylabel(s["unit"], fontsize=9)
        ax.set_title(s["label"], loc="left", fontsize=9.8, fontweight="bold", color=INK)
        ax.grid(True, color="#eef2f7")

    # ---- anomaly windows + fault labels -----------------------------
    labelled = set()
    for a in twin.anomalies:
        ai = ORDER.index(a["sensor"]) + 1
        axes[ai].axvspan(x[a["start"]], x[a["end"]], color=SEV[a["severity"]], alpha=0.16, lw=0)
        if a["sensor"] not in labelled and a["sensor"] in FAULT_LABEL:
            mid = (a["start"] + a["end"]) // 2
            axes[ai].annotate(FAULT_LABEL[a["sensor"]], xy=(x[mid], 0.86), xycoords=("data", "axes fraction"),
                              ha="center", fontsize=8, color=SEV[a["severity"]], fontweight="bold")
            labelled.add(a["sensor"])

    # ---- sensor-dropout window (held; not a band anomaly) -----------
    drop = np.nonzero(~twin.data_ok)[0]
    if drop.size:
        vi = ORDER.index("vibration_g") + 1
        axes[vi].axvspan(x[drop[0]], x[drop[-1]], facecolor="#2c6fbb", alpha=0.12, hatch="////",
                         edgecolor="#2c6fbb", lw=0)
        axes[vi].annotate("sensor dropout — twin holds", xy=(x[(drop[0] + drop[-1]) // 2], 0.86),
                          xycoords=("data", "axes fraction"), ha="center", fontsize=8,
                          color="#2c6fbb", fontweight="bold")

    axes[-1].set_xlabel("layer", fontsize=9.5)

    # ---- title -------------------------------------------------------
    fig.suptitle("Runtime FFF print twin — live process monitoring",
                 x=0.085, y=0.965, ha="left", fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.085, 0.925, "Where the thread continues: the advisor's released build, monitored on the "
             "printer. Each channel vs. its expected envelope; detected faults shaded; verify-before-act.",
             ha="left", fontsize=9.5, color=MUTED)

    # ---- recommendation banner --------------------------------------
    import textwrap
    end = twin.snapshot(twin.n_layers - 1)["rec"]
    rc = SEV.get(end.get("color", "critical"), "#b22222")
    fig.add_artist(FancyBboxPatch((0.085, 0.028), 0.55, 0.115,
                                  boxstyle="round,pad=0.006,rounding_size=0.015",
                                  facecolor=rc, edgecolor="none", transform=fig.transFigure))
    fig.text(0.103, 0.122, f"RECOMMENDATION · {end.get('title', '')}", fontsize=10.5,
             color="white", fontweight="bold")
    fig.text(0.103, 0.093, textwrap.fill(end.get("detail", ""), 82), fontsize=8.6,
             color="#ffffffee", va="top", linespacing=1.5)
    fig.add_artist(FancyBboxPatch((0.655, 0.028), 0.32, 0.115,
                                  boxstyle="round,pad=0.006,rounding_size=0.015",
                                  facecolor="#eef4fb", edgecolor="#2c6fbb", transform=fig.transFigure))
    fig.text(0.67, 0.122, "VERIFY-BEFORE-ACT", fontsize=10.5, color="#2c6fbb", fontweight="bold")
    fig.text(0.67, 0.093, "On the sensor dropout the twin held —\nno parameter change on incomplete data.",
             fontsize=8.6, color="#3b4754", va="top", linespacing=1.5)

    out = ROOT / "docs" / "twin_dashboard.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
