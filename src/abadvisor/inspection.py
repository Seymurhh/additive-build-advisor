"""Inspection-plan generation -- the design-intent end of the digital thread.

A part's tolerances are design intent. This module turns them into a concrete
first-article inspection plan: for each toleranced dimension, GD&T control, or
surface-finish requirement it selects a measurement method and equipment based
on how tight the tolerance is, states the pass/fail limits, and -- crucially --
checks the tolerance against the *as-built process capability*. A tolerance the
process cannot hold as-built is flagged as needing post-machining, because
inspecting to it would only confirm a guaranteed failure.

The tolerance spec is plain JSON so the design data stays CAD-neutral (a Fusion
or STEP exporter would populate the same fields). If no spec is supplied, a
default plan is generated from the bounding box so the thread is never empty.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .geometry import Mesh
from .materials import ProcessProfile

# As-built capability by process family: typical linear tolerance (+/- mm) and
# typical surface roughness (Ra, micrometers). Representative, not qualified.
_CAPABILITY = {
    "FFF": {"linear_mm": 0.40, "ra_um": 12.0},
    "SLA": {"linear_mm": 0.15, "ra_um": 4.0},
    "SLS": {"linear_mm": 0.30, "ra_um": 10.0},
    "LPBF": {"linear_mm": 0.15, "ra_um": 12.0},
}

_FORM_CONTROLS = {"flatness", "parallelism", "perpendicularity", "position", "runout", "profile", "concentricity"}
_SEVERITY_RANK = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


@dataclass
class InspectionStep:
    feature: str
    characteristic: str
    nominal_mm: Optional[float]
    tolerance_mm: float
    method: str
    equipment: str
    pass_if: str
    severity: str          # capability verdict: ok | warning | critical
    note: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _method_for(characteristic: str, tol: float, internal: bool) -> Dict[str, str]:
    c = characteristic.lower()
    if c == "surface_finish":
        return {"method": "Surface roughness measurement", "equipment": "Contact profilometer"}
    if internal:
        return {"method": "Volumetric / internal scan", "equipment": "Industrial CT scanner"}
    if c in _FORM_CONTROLS:
        if tol < 0.05:
            return {"method": "GD&T form/location on CMM", "equipment": "CMM (touch probe)"}
        return {"method": "Form check on surface plate", "equipment": "Surface plate + dial indicator"}
    if c == "diameter":
        if tol < 0.05:
            return {"method": "Diameter on CMM", "equipment": "CMM (touch probe)"}
        if tol < 0.2:
            return {"method": "Bore/pin gauge", "equipment": "Bore gauge or pin gauges"}
        return {"method": "Caliper diameter", "equipment": "Digital calipers"}
    # generic linear dimension
    if tol < 0.02:
        return {"method": "Precision dimensional on CMM", "equipment": "CMM (touch probe)"}
    if tol < 0.10:
        return {"method": "Micrometer / CMM", "equipment": "Micrometer or CMM"}
    if tol < 0.50:
        return {"method": "Caliper / height gauge", "equipment": "Digital calipers / height gauge"}
    return {"method": "Caliper", "equipment": "Digital calipers"}


def _capability_check(characteristic: str, tol: float, profile: ProcessProfile, ra_um: Optional[float]) -> Dict[str, str]:
    cap = _CAPABILITY.get(profile.family, {"linear_mm": 0.4, "ra_um": 12.0})
    c = characteristic.lower()
    if c == "surface_finish":
        target = ra_um if ra_um is not None else tol
        if target < cap["ra_um"] * 0.5:
            return {"severity": "critical", "note": f"Ra {target} um is well below as-built ~{cap['ra_um']} um; requires machining/polishing."}
        if target < cap["ra_um"]:
            return {"severity": "warning", "note": f"Ra {target} um below as-built ~{cap['ra_um']} um; plan a finishing step."}
        return {"severity": "ok", "note": "Achievable as-built."}
    # dimensional / form
    cap_lin = cap["linear_mm"]
    if tol < cap_lin * 0.5:
        return {"severity": "critical", "note": f"Tolerance +/-{tol} mm is far below as-built capability +/-{cap_lin} mm; requires post-machining."}
    if tol < cap_lin:
        return {"severity": "warning", "note": f"Tolerance +/-{tol} mm near/below as-built capability +/-{cap_lin} mm; verify or machine."}
    return {"severity": "ok", "note": "Within as-built capability."}


def _step(feature, characteristic, nominal, tol, profile, internal=False, ra_um=None) -> InspectionStep:
    m = _method_for(characteristic, tol, internal)
    cap = _capability_check(characteristic, tol, profile, ra_um)
    if characteristic.lower() == "surface_finish":
        pass_if = f"Ra <= {ra_um} um"
    elif characteristic.lower() in _FORM_CONTROLS:
        pass_if = f"{characteristic} within {tol} mm"
    else:
        pass_if = f"{nominal} +/- {tol} mm" if nominal is not None else f"within {tol} mm"
    return InspectionStep(
        feature=feature, characteristic=characteristic, nominal_mm=nominal,
        tolerance_mm=tol, method=m["method"], equipment=m["equipment"],
        pass_if=pass_if, severity=cap["severity"], note=cap["note"],
    )


def _default_spec(mesh: Mesh) -> Dict[str, object]:
    ext = mesh.extents
    return {
        "part_name": "part",
        "critical_dimensions": [
            {"name": "overall_x", "nominal_mm": round(float(ext[0]), 3), "tolerance_mm": 0.3, "type": "length"},
            {"name": "overall_y", "nominal_mm": round(float(ext[1]), 3), "tolerance_mm": 0.3, "type": "length"},
            {"name": "overall_z", "nominal_mm": round(float(ext[2]), 3), "tolerance_mm": 0.3, "type": "length"},
        ],
    }


def generate_inspection_plan(
    mesh: Mesh,
    profile: ProcessProfile,
    tolerance_spec: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build a first-article inspection plan from the part's tolerance spec."""
    spec = tolerance_spec or _default_spec(mesh)
    steps: List[InspectionStep] = []

    for d in spec.get("critical_dimensions", []):
        steps.append(_step(
            feature=d.get("name", "dimension"),
            characteristic=d.get("type", "length"),
            nominal=d.get("nominal_mm"),
            tol=float(d.get("tolerance_mm", 0.3)),
            profile=profile,
            internal=bool(d.get("internal", False)),
        ))
    for g in spec.get("gdt", []):
        steps.append(_step(
            feature=g.get("feature", "feature"),
            characteristic=g.get("control", "profile"),
            nominal=None,
            tol=float(g.get("tolerance_mm", 0.1)),
            profile=profile,
            internal=bool(g.get("internal", False)),
        ))
    for s in spec.get("surface_finish", []):
        steps.append(_step(
            feature=s.get("feature", "surface"),
            characteristic="surface_finish",
            nominal=None,
            tol=0.0,
            profile=profile,
            ra_um=float(s.get("Ra_um", s.get("ra_um", 6.3))),
        ))

    steps.sort(key=lambda s: -_SEVERITY_RANK[s.severity])
    worst = max((s.severity for s in steps), key=lambda s: _SEVERITY_RANK[s], default="ok")
    methods = sorted({s.equipment for s in steps})
    tolerances = [s.tolerance_mm for s in steps if s.characteristic.lower() != "surface_finish"]
    return {
        "part_name": spec.get("part_name", "part"),
        "steps": [s.as_dict() for s in steps],
        "n_steps": len(steps),
        "equipment_required": methods,
        "tightest_tolerance_mm": min(tolerances) if tolerances else None,
        "requires_cmm": any("CMM" in s.equipment for s in steps),
        "requires_ct": any("CT" in s.equipment for s in steps),
        "worst_severity": worst,
        "n_capability_flags": sum(1 for s in steps if s.severity in ("warning", "critical")),
    }
