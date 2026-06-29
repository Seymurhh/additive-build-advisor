"""End-to-end pipeline: STL in, digital-thread record out.

This wires the stages together in the order a real build-prep workflow runs:

    load mesh -> check geometry -> choose orientation (DoE) -> orient & drop to
    plate -> voxelize -> simulate build -> DfAM checks -> inspection plan ->
    assemble digital-thread record + release gate

It returns both the rich objects (for the report renderer) and the serializable
record (for the JSON hand-off). This is the single function the CLI and the
example runner call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from . import digital_thread
from .am_sim import simulate_build
from .dfam import run_dfam
from .fea import solve_inherent_strain
from .geometry import Mesh, watertight_report
from .inspection import generate_inspection_plan
from .materials import get_profile
from .orientation import optimize_orientation
from .stl_io import read_stl
from .voxelize import voxelize


def advise(
    *,
    stl_path: Optional[str] = None,
    mesh: Optional[Mesh] = None,
    process: str = "fff_pla",
    tolerance_spec: Optional[Dict[str, object]] = None,
    grid_n: int = 64,
    fea_grid_n: int = 26,
    source_label: Optional[str] = None,
) -> Dict[str, object]:
    """Run the full advisory pipeline on a part.

    Provide either ``stl_path`` or an in-memory ``mesh``. ``process`` is a key
    from :mod:`abadvisor.materials`. Returns a dict of stage outputs plus the
    assembled ``record``.
    """
    if mesh is None:
        if stl_path is None:
            raise ValueError("Provide either stl_path or mesh.")
        triangles, _ = read_stl(stl_path)
        mesh = Mesh(triangles)
        source_label = source_label or Path(stl_path).name
    source_label = source_label or "in-memory mesh"

    profile = get_profile(process)

    watertight = watertight_report(mesh)
    orientation = optimize_orientation(mesh, profile)
    best = orientation["best"]

    oriented = mesh.transformed(best.rotation).dropped_to_plate()
    grid = voxelize(oriented, grid_n=grid_n)
    sim = simulate_build(oriented, grid, profile)

    # Distortion FEA (inherent-strain method) on a separate coarse grid.
    fea_grid = voxelize(oriented, grid_n=fea_grid_n)
    fea = solve_inherent_strain(
        fea_grid.occ, fea_grid.pitch,
        E=profile.youngs_modulus_mpa, nu=profile.poisson_ratio,
        eigenstrain=profile.inherent_strain, compute_stress=True,
    )

    dfam = run_dfam(oriented, grid, sim, profile, fea, orientation_fits=best.fits_build_volume)
    inspection = generate_inspection_plan(oriented, profile, tolerance_spec)

    record = digital_thread.build_record(
        source_file=source_label,
        mesh=oriented,
        watertight=watertight,
        profile=profile,
        orientation=orientation,
        sim=sim,
        fea=fea,
        dfam=dfam,
        inspection=inspection,
    )

    return {
        "profile": profile,
        "mesh": mesh,
        "oriented_mesh": oriented,
        "grid": grid,
        "fea_grid": fea_grid,
        "fea": fea,
        "watertight": watertight,
        "orientation": orientation,
        "sim": sim,
        "dfam": dfam,
        "inspection": inspection,
        "record": record,
    }
