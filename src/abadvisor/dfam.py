"""Design-for-Additive-Manufacturing (DfAM) checks.

These are the manufacturability checks a build-prep engineer runs before
committing a part: are the walls printable, is there an unreasonable amount of
support, is the part tall and tippy, will trapped powder or resin be stuck
inside, and does it even fit the machine. Each check returns a ``Finding`` with
a severity; the worst severity feeds the release gate in
:mod:`abadvisor.digital_thread`.

The checks read from the same voxel grid and build simulation that produced the
cost/time numbers, so the manufacturability verdict and the estimates are always
consistent with one another.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from .am_sim import BuildSimulation
from .geometry import Mesh
from .materials import ProcessProfile
from .voxelize import VoxelGrid, thin_wall_analysis, trapped_volume

_SEVERITY_RANK = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


@dataclass
class Finding:
    check: str
    severity: str           # ok | info | warning | critical
    message: str
    value: float
    limit: float
    recommendation: str

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["value"] = round(self.value, 3)
        d["limit"] = round(self.limit, 3)
        return d


def _worst(findings: List[Finding]) -> str:
    if not findings:
        return "ok"
    return max(findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity


def run_dfam(
    mesh: Mesh,
    grid: VoxelGrid,
    sim: BuildSimulation,
    profile: ProcessProfile,
    orientation_fits: bool = True,
) -> Dict[str, object]:
    findings: List[Finding] = []

    # 1) Build-volume fit -------------------------------------------------
    if not orientation_fits:
        bv = profile.build_volume_mm
        findings.append(Finding(
            "build_volume_fit", "critical",
            "Part does not fit the build volume in any evaluated orientation.",
            value=max(sim.height_mm, *sim.footprint_dims_mm), limit=min(bv),
            recommendation="Split the part, scale it down, or use a larger machine.",
        ))
    else:
        findings.append(Finding(
            "build_volume_fit", "ok", "Part fits the build volume.",
            value=sim.height_mm, limit=profile.build_volume_mm[2],
            recommendation="None.",
        ))

    # 2) Thin walls -------------------------------------------------------
    tw = thin_wall_analysis(grid, profile.min_wall_mm)
    frac = float(tw["thin_fraction"])
    if frac > 0.10:
        sev = "critical"
    elif frac > 0.02:
        sev = "warning"
    else:
        sev = "ok"
    findings.append(Finding(
        "thin_walls", sev,
        f"{frac*100:.1f}% of the volume is in features thinner than the "
        f"{profile.min_wall_mm} mm minimum wall.",
        value=frac, limit=0.02,
        recommendation="Thicken thin walls/ribs or accept reduced strength there."
        if sev != "ok" else "None.",
    ))

    # 3) Support burden ---------------------------------------------------
    if profile.support_infill_frac > 0 and sim.part_volume_mm3 > 0:
        ratio = sim.support_material_mm3 / sim.part_volume_mm3
        if ratio > 0.75:
            sev = "warning"
        elif ratio > 0.05:
            sev = "info"
        else:
            sev = "ok"
        findings.append(Finding(
            "support_burden", sev,
            f"Support material is {ratio*100:.0f}% of part volume "
            f"({sim.support_material_mm3/1000:.2f} cm3).",
            value=ratio, limit=0.75,
            recommendation="Re-orient, add chamfers under overhangs, or design self-supporting angles."
            if sev == "warning" else "None.",
        ))

    # 4) Aspect ratio / stability ----------------------------------------
    if sim.aspect_ratio > 12:
        sev = "critical"
    elif sim.aspect_ratio > 6:
        sev = "warning"
    else:
        sev = "ok"
    findings.append(Finding(
        "aspect_ratio", sev,
        f"Aspect ratio (height / min footprint) is {sim.aspect_ratio:.1f}.",
        value=sim.aspect_ratio, limit=6.0,
        recommendation="Add a raft/brim or re-orient to lower the aspect ratio."
        if sev != "ok" else "None.",
    ))

    # 5) Trapped volume / drainage ---------------------------------------
    tv = trapped_volume(grid)
    trapped = float(tv["trapped_volume_mm3"])
    trapped_frac = trapped / sim.part_volume_mm3 if sim.part_volume_mm3 else 0.0
    if profile.needs_drain_holes and trapped_frac > 0.005:
        sev = "critical"
        msg = (f"Enclosed void of {trapped/1000:.2f} cm3 will trap "
               f"{'resin' if profile.family == 'SLA' else 'powder'} -- needs drain holes.")
        rec = "Add drain/escape holes to every enclosed cavity."
    elif trapped_frac > 0.005:
        sev = "info"
        msg = f"Enclosed void of {trapped/1000:.2f} cm3 detected (no drainage needed for this process)."
        rec = "Confirm the internal cavity is intended."
    else:
        sev = "ok"
        msg = "No significant enclosed voids."
        rec = "None."
    findings.append(Finding(
        "trapped_volume", sev, msg, value=trapped_frac, limit=0.005, recommendation=rec,
    ))

    # 6) Warpage risk (from the build simulation) ------------------------
    if sim.warpage_index > 80:
        sev = "critical"
    elif sim.warpage_index > 60:
        sev = "warning"
    elif sim.warpage_index > 35:
        sev = "info"
    else:
        sev = "ok"
    findings.append(Finding(
        "warpage_risk", sev,
        f"Reduced-order warpage index is {sim.warpage_index:.0f}/100.",
        value=sim.warpage_index, limit=60.0,
        recommendation="Run a thermo-mechanical simulation and consider stress relief / orientation change."
        if sev in ("warning", "critical") else "None.",
    ))

    findings_sorted = sorted(findings, key=lambda f: -_SEVERITY_RANK[f.severity])
    worst = _worst(findings)
    return {
        "findings": [f.as_dict() for f in findings_sorted],
        "worst_severity": worst,
        "n_critical": sum(1 for f in findings if f.severity == "critical"),
        "n_warning": sum(1 for f in findings if f.severity == "warning"),
    }
