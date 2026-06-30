"""Run the advisor on the sample parts and write reports.

Demonstrates all three release-gate outcomes end to end:

  1. calibration_cube  + FFF  with loose tolerances     -> release_to_build
  2. gantry_bracket    + FFF  with tight tolerances      -> needs_engineering_review
  3. hollow_housing    + SLA  (enclosed cavity, must drain) -> redesign_required

Generates the sample STL files first if they are missing, then writes a report
(HTML + JSON) per scenario under ``output/<part>__<process>/``.

Run:  python examples/run_example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abadvisor import report  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "output"
EX = ROOT / "examples"

# Loose tolerances a cube can actually hold as-built on FFF -> clean release.
_CUBE_SPEC = {
    "part_name": "calibration_cube",
    "critical_dimensions": [
        {"name": "edge_x", "nominal_mm": 20.0, "tolerance_mm": 0.5, "type": "length"},
        {"name": "edge_y", "nominal_mm": 20.0, "tolerance_mm": 0.5, "type": "length"},
        {"name": "edge_z", "nominal_mm": 20.0, "tolerance_mm": 0.5, "type": "length"},
    ],
}

_CANTILEVER_SPEC = {
    "part_name": "cantilever_benchmark",
    "notes": "Warp-prone flat bar printed in ABS (the FFF material that warps most) -- "
             "the worst case for cooling warpage, so distortion is the point of interest.",
    "critical_dimensions": [
        {"name": "length", "nominal_mm": 75.0, "tolerance_mm": 0.2, "type": "length"},
        {"name": "thickness", "nominal_mm": 6.0, "tolerance_mm": 0.1, "type": "length"},
    ],
}

SCENARIOS = [
    ("calibration_cube", "fff_pla", _CUBE_SPEC),
    ("gantry_bracket", "fff_pla", json.loads((EX / "tolerances_bracket.json").read_text())),
    ("hollow_housing", "sla_resin", json.loads((EX / "tolerances_housing.json").read_text())),
    ("cantilever_benchmark", "fff_abs", _CANTILEVER_SPEC),
]


def _ensure_parts() -> None:
    if not (DATA / "gantry_bracket.stl").exists():
        import make_sample_parts
        make_sample_parts.main()


def main() -> int:
    _ensure_parts()
    rows = []
    for part, process, spec in SCENARIOS:
        stl = DATA / f"{part}.stl"
        result = advise(stl_path=str(stl), process=process, tolerance_spec=spec, grid_n=64)
        rec = result["record"]
        outdir = OUT / f"{part}__{process}"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "digital_thread.json").write_text(json.dumps(rec, indent=2))
        report.render_html(result, str(outdir))
        sim = rec["simulation"]
        rows.append((part, process, sim["build_time_h"], sim["total_cost_usd"],
                     rec["distortion_fea"]["max_distortion_mm"],
                     rec["manufacturability"]["worst_severity"], rec["gate"]["decision"]))
        print(f"  [{part} / {process}] -> {rec['gate']['decision']}   (report: {outdir/'report.html'})")

    print("\n  Summary")
    print(f"  {'part':18s} {'process':14s} {'time_h':>7s} {'cost$':>8s} {'distort_mm':>10s} {'dfam':>9s}  gate")
    for r in rows:
        print(f"  {r[0]:18s} {r[1]:14s} {r[2]:7.2f} {r[3]:8.2f} {r[4]:10.3f} {r[5]:>9s}  {r[6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
