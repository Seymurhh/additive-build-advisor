"""Generate the sample STL parts into ``data/``.

Each part is synthesized from scratch (see :mod:`abadvisor.shapes`) so the repo
is self-contained -- no external CAD files needed. The parts are chosen to
exercise the full advisor:

* ``calibration_cube``  -- analytic volume; validates the voxel engine
* ``gantry_bracket``    -- top-flange overhang; exercises the orientation DoE
* ``tall_standoff``     -- slender; exercises the aspect-ratio / stability check
* ``hollow_housing``    -- enclosed cavity; exercises trapped-volume / drain holes

Run:  python examples/make_sample_parts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abadvisor import shapes  # noqa: E402
from abadvisor.stl_io import write_stl_binary  # noqa: E402

PARTS = {
    "calibration_cube": lambda: shapes.cube(20.0),
    "gantry_bracket": lambda: shapes.gantry_bracket(),
    "tall_standoff": lambda: shapes.tall_standoff(),
    "hollow_housing": lambda: shapes.hollow_housing(),
}


def main() -> int:
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in PARTS.items():
        mesh = factory()
        path = data_dir / f"{name}.stl"
        write_stl_binary(path, mesh.triangles, mesh.face_normals, header=f"abadvisor {name}")
        print(f"  wrote {path}  ({mesh.n_facets} facets, {mesh.volume_mm3:.1f} mm^3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
