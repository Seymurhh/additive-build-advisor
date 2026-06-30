"""Digital-thread record + release gate.

This is where the thread is tied together. It takes the geometry check, the
orientation decision, the build simulation, the DfAM verdict, and the inspection
plan, and assembles a single machine-readable record. Then it applies a
**release gate**: the advisor never silently "approves" a build. It returns one
of three decisions -- ``release_to_build``, ``needs_engineering_review``, or
``redesign_required`` -- with the reasons and the specific blocking findings
attached.

This is the same verify-before-act discipline used in the companion runtime
monitoring twin (``mini-manufacturing-digital-twin``): a model may recommend,
but a physical action is gated on evidence, on the confidence of the simulation
(is the mesh watertight? did the discretized volume validate?), and on a human
review path when anything is uncertain. The record also carries an explicit
hand-off to that monitoring twin, so design intent flows downstream to as-built
monitoring -- the front and back halves of one digital thread.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .am_sim import BuildSimulation
from .fea import FEAResult
from .geometry import Mesh
from .materials import ProcessProfile

SCHEMA = "abadvisor.digital_thread/v1"


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _simulation_confidence(watertight: bool, volume_error_pct: float) -> float:
    """How much should we trust the simulation? Mirrors 'do not act on bad data'.

    An unsealed mesh makes the inside/outside test unreliable, and a large gap
    between the discretized volume and the analytic mesh volume means the grid
    is too coarse to trust. Both pull confidence down.
    """
    conf = 1.0
    if not watertight:
        conf = min(conf, 0.40)
    conf -= min(0.5, abs(volume_error_pct) / 10.0)  # 10% volume error -> -0.5
    return float(max(0.0, round(conf, 3)))


def _apply_gate(
    watertight: bool,
    volume_error_pct: float,
    dfam: Dict[str, object],
    inspection: Dict[str, object],
) -> Dict[str, object]:
    reasons: List[str] = []
    blocking: List[Dict[str, object]] = []

    dfam_critical = [f for f in dfam["findings"] if f["severity"] == "critical"]
    dfam_warning = [f for f in dfam["findings"] if f["severity"] == "warning"]
    insp_critical = [s for s in inspection["steps"] if s["severity"] == "critical"]
    insp_warning = [s for s in inspection["steps"] if s["severity"] == "warning"]

    confidence = _simulation_confidence(watertight, volume_error_pct)

    if dfam_critical:
        decision = "redesign_required"
        for f in dfam_critical:
            reasons.append(f"DfAM critical: {f['message']}")
            blocking.append({"source": "dfam", **f})
    elif dfam_warning or insp_critical or insp_warning or not watertight or abs(volume_error_pct) > 5.0:
        decision = "needs_engineering_review"
        if not watertight:
            reasons.append("Mesh is not watertight; simulation confidence is reduced.")
        if abs(volume_error_pct) > 5.0:
            reasons.append(f"Voxel/mesh volume mismatch {volume_error_pct:.1f}% exceeds 5% -- refine the grid.")
        for f in dfam_warning:
            reasons.append(f"DfAM warning: {f['message']}")
            blocking.append({"source": "dfam", **f})
        for s in insp_critical:
            reasons.append(f"Inspection: {s['feature']} -- {s['note']}")
            blocking.append({"source": "inspection", **s})
        for s in insp_warning:
            reasons.append(f"Inspection: {s['feature']} -- {s['note']}")
    else:
        decision = "release_to_build"
        reasons.append("All DfAM checks pass, tolerances are within as-built capability, "
                       "and the simulation validated.")

    return {
        "decision": decision,
        "confidence": confidence,
        "reasons": reasons,
        "blocking_findings": blocking,
    }


def build_record(
    *,
    source_file: str,
    mesh: Mesh,
    watertight: Dict[str, object],
    profile: ProcessProfile,
    orientation: Dict[str, object],
    sim: BuildSimulation,
    fea: "FEAResult",
    dfam: Dict[str, object],
    inspection: Dict[str, object],
) -> Dict[str, object]:
    """Assemble the full digital-thread record (and run the release gate)."""
    lo, hi = mesh.bounds
    best = orientation["best"]
    is_watertight = bool(watertight["is_watertight"])

    gate = _apply_gate(is_watertight, sim.volume_error_pct, dfam, inspection)

    part_id = f"{inspection['part_name']}-{mesh.geometry_hash}"
    machine_id = f"SEAS-{profile.family}-01"

    record = {
        "schema": SCHEMA,
        "part": {
            "name": inspection["part_name"],
            "source_file": source_file,
            "geometry_hash": mesh.geometry_hash,
            "n_facets": mesh.n_facets,
            "watertight": watertight,
            "volume_mm3": round(mesh.volume_mm3, 3),
            "surface_area_mm2": round(mesh.surface_area_mm2, 3),
            "bbox_min_mm": [round(float(v), 3) for v in lo],
            "bbox_max_mm": [round(float(v), 3) for v in hi],
        },
        "process": {
            "key": profile.key,
            "name": profile.name,
            "family": profile.family,
            "material": profile.material,
            "layer_height_mm": profile.default_layer_height_mm,
            "build_volume_mm": list(profile.build_volume_mm),
            "self_support_angle_deg": profile.self_support_angle_deg,
            "post_processing": profile.post_processing,
        },
        "design_decision": {
            "chosen_orientation": best.summary(),
            "doe": orientation["design"],
            "alternatives": [c.summary() for c in orientation["candidates"][1:4]],
        },
        "simulation": {
            **sim.summary(),
            "build_time_breakdown_h": {
                "deposition": round(sim.deposition_time_h, 3),
                "layer_overhead": round(sim.overhead_time_h, 3),
            },
            "cost_breakdown_usd": {
                "material": round(sim.material_cost_usd, 2),
                "machine": round(sim.machine_cost_usd, 2),
            },
            "grid_validation": {
                "voxel_volume_mm3": round(sim.voxel_volume_mm3, 3),
                "mesh_volume_mm3": round(sim.part_volume_mm3, 3),
                "volume_error_pct": round(sim.volume_error_pct, 3),
            },
        },
        "distortion_fea": {
            "method": "thermal-contraction (eigenstrain) linear-elastic FEM (scikit-fem, hex elements)",
            "solver": fea.solver,
            "target_process": "FFF (polymer warpage)",
            "applicability": ("home regime (FFF / polymer cooling warpage)" if profile.family == "FFF"
                              else f"comparison run on {profile.family}; the same eigenstrain solve is "
                                   "the inherent-strain method on metal PBF"),
            "eigenstrain": fea.eigenstrain,
            "max_distortion_mm": round(fea.max_displacement_mm, 4),
            "mean_distortion_mm": round(fea.mean_displacement_mm, 4),
            "peak_von_mises_mpa": (round(fea.peak_von_mises_mpa, 1)
                                   if fea.peak_von_mises_mpa is not None else None),
            "elements": fea.n_elements,
            "dof": fea.n_dof,
            "converged": fea.converged,
        },
        "manufacturability": dfam,
        "inspection_plan": inspection,
        "gate": gate,
        "handoff": {
            "to": "mini-manufacturing-digital-twin",
            "note": "Release gate output becomes the as-built monitoring context "
                    "for the runtime digital twin once the part is on the machine.",
            "as_built_context": {
                "machine_id": machine_id,
                "part_id": part_id,
                "operation": f"additive_{profile.family.lower()}",
                "expected_layers": sim.n_layers,
                "expected_build_time_h": round(sim.total_time_h, 2),
                "watch": ([f["check"] for f in dfam["findings"]
                           if f["severity"] in ("warning", "critical")] or ["nominal"]),
            },
        },
    }
    return _jsonable(record)


def write_json(record: Dict[str, object], path: str) -> None:
    Path(path).write_text(json.dumps(record, indent=2))
