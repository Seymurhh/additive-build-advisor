"""Command-line interface for the Additive Build Advisor.

    build-advisor PART.stl --process fff_pla --tolerances spec.json --out output/

Runs the full pipeline, writes ``report.html`` and ``digital_thread.json`` to the
output directory, and prints a short verdict to the console.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report
from .materials import list_profiles
from .pipeline import advise


def _load_tolerances(path):
    if not path:
        return None
    return json.loads(Path(path).read_text())


def _print_summary(rec) -> None:
    sim = rec["simulation"]
    fea = rec["distortion_fea"]
    gate = rec["gate"]
    print(f"\n  Part         : {rec['part']['name']}  (geometry id {rec['part']['geometry_hash']})")
    print(f"  Process      : {rec['process']['name']}")
    o = rec["design_decision"]["chosen_orientation"]
    print(f"  Orientation  : {o['label']}  height={o['height_mm']} mm  "
          f"base contact={o['base_contact_mm2']} mm^2  support={o['support_volume_mm3']} mm^3")
    print(f"  Simulation   : {sim['part_volume_cm3']} cm^3, {sim['n_layers']} layers, "
          f"{sim['build_time_h']} h, ${sim['total_cost_usd']}")
    print(f"  Distortion   : FEA peak {fea['max_distortion_mm']} mm "
          f"({fea['elements']} elems, {fea['solver_iterations']} CG iters), "
          f"peak von Mises {fea['peak_von_mises_mpa']} MPa")
    print(f"  Validation   : voxel/mesh volume error {sim['grid_validation']['volume_error_pct']}%")
    print(f"  DfAM         : worst={rec['manufacturability']['worst_severity']}  "
          f"(critical={rec['manufacturability']['n_critical']}, warning={rec['manufacturability']['n_warning']})")
    print(f"  Inspection   : worst={rec['inspection_plan']['worst_severity']}  "
          f"requires_cmm={rec['inspection_plan']['requires_cmm']}")
    print(f"\n  >> GATE: {gate['decision'].upper()}  (confidence {gate['confidence']})")
    for r in gate["reasons"]:
        print(f"       - {r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="build-advisor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stl", nargs="?", help="path to an STL file (binary or ASCII)")
    p.add_argument("--process", default="fff_pla", help="process profile key (see --list-processes)")
    p.add_argument("--tolerances", help="path to a tolerance-spec JSON file")
    p.add_argument("--grid", type=int, default=64, help="voxels along the longest axis (default 64)")
    p.add_argument("--out", default="output", help="output directory (default ./output)")
    p.add_argument("--no-embed", action="store_true", help="reference figures instead of embedding them")
    p.add_argument("--json-only", action="store_true", help="write only the JSON record, skip the HTML report")
    p.add_argument("--list-processes", action="store_true", help="list available process profiles and exit")
    args = p.parse_args(argv)

    if args.list_processes:
        for prof in list_profiles():
            print(f"  {prof.key:16s} {prof.name}  "
                  f"(layer {prof.default_layer_height_mm} mm, self-support {prof.self_support_angle_deg}°)")
        return 0

    if not args.stl:
        p.error("an STL path is required (or use --list-processes)")

    result = advise(
        stl_path=args.stl,
        process=args.process,
        tolerance_spec=_load_tolerances(args.tolerances),
        grid_n=args.grid,
    )
    rec = result["record"]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "digital_thread.json"
    json_path.write_text(json.dumps(rec, indent=2))

    _print_summary(rec)
    print(f"\n  Wrote {json_path}")
    if not args.json_only:
        html_path = report.render_html(result, str(outdir), embed=not args.no_embed)
        print(f"  Wrote {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
