"""Smoke + validation tests.

Runs without matplotlib (the report renderer is not imported), so the engine
and pipeline can be validated headless. Runnable two ways:

    pytest
    python tests/test_smoke.py        # prints "Smoke test passed"

The most important test is the engine validation: the discretized voxel volume
must converge to the analytic mesh volume. If that drifts, every downstream
estimate is suspect, so it is asserted explicitly.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abadvisor import shapes  # noqa: E402
from abadvisor.geometry import Mesh, rotation_about_axis, watertight_report  # noqa: E402
from abadvisor.pipeline import advise  # noqa: E402
from abadvisor.stl_io import read_stl, write_stl_binary  # noqa: E402
from abadvisor.voxelize import trapped_volume, voxelize  # noqa: E402


def test_sample_parts_watertight():
    for factory in (shapes.cube, shapes.gantry_bracket, shapes.tall_standoff, shapes.hollow_housing):
        mesh = factory()
        assert watertight_report(mesh)["is_watertight"], factory.__name__


def test_stl_roundtrip(tmp_path=None):
    import tempfile
    mesh = shapes.gantry_bracket()
    path = os.path.join(tempfile.gettempdir(), "abadvisor_roundtrip.stl")
    write_stl_binary(path, mesh.triangles, mesh.face_normals)
    tris, _ = read_stl(path)
    back = Mesh(tris)
    assert back.n_facets == mesh.n_facets
    assert abs(back.volume_mm3 - mesh.volume_mm3) < 1e-6


def test_cube_volume_is_exact():
    # An axis-aligned cube lands exactly on the grid -> discretization is exact.
    mesh = shapes.cube(20.0)
    for n in (24, 48, 72):
        g = voxelize(mesh, grid_n=n)
        err = abs(g.occupied_volume_mm3 - mesh.volume_mm3) / mesh.volume_mm3
        assert err < 1e-6, (n, err)


def test_voxel_volume_converges_on_rotated_part():
    # A part rotated off the grid axes still converges to the analytic volume.
    mesh = shapes.gantry_bracket().transformed(
        rotation_about_axis([1, 1, 1], 33.0)
    )
    g = voxelize(mesh, grid_n=80)
    err = abs(g.occupied_volume_mm3 - mesh.volume_mm3) / mesh.volume_mm3
    assert err < 0.03, err


def test_hollow_housing_has_trapped_volume():
    g = voxelize(shapes.hollow_housing(), grid_n=48)
    assert trapped_volume(g)["trapped_volume_mm3"] > 100.0


def test_orientation_reduces_overhang():
    mesh = shapes.gantry_bracket()
    res = advise(mesh=mesh, process="fff_pla", grid_n=32)
    cands = res["orientation"]["candidates"]
    best = res["orientation"]["best"]
    worst = max(cands, key=lambda c: c.overhang_area_mm2)
    assert best.overhang_area_mm2 <= worst.overhang_area_mm2


def test_gate_outcomes():
    # release: a cube with tolerances it can hold as-built
    loose = {"part_name": "cube", "critical_dimensions": [
        {"name": "x", "nominal_mm": 20.0, "tolerance_mm": 0.6, "type": "length"}]}
    r = advise(mesh=shapes.cube(20.0), process="fff_pla", tolerance_spec=loose, grid_n=32)
    assert r["record"]["gate"]["decision"] == "release_to_build"

    # needs review: a tolerance below FFF as-built capability
    tight = {"part_name": "cube", "critical_dimensions": [
        {"name": "x", "nominal_mm": 20.0, "tolerance_mm": 0.02, "type": "length"}]}
    r = advise(mesh=shapes.cube(20.0), process="fff_pla", tolerance_spec=tight, grid_n=32)
    assert r["record"]["gate"]["decision"] == "needs_engineering_review"

    # redesign: an enclosed cavity on a process that must drain
    r = advise(mesh=shapes.hollow_housing(), process="sla_resin", grid_n=48)
    assert r["record"]["gate"]["decision"] == "redesign_required"
    assert r["record"]["manufacturability"]["n_critical"] >= 1


def test_record_is_json_serializable():
    r = advise(mesh=shapes.gantry_bracket(), process="lpbf_alsi10mg", grid_n=32)
    json.dumps(r["record"])  # raises if any numpy types leaked through


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print("Smoke test passed")


if __name__ == "__main__":
    _run_all()
