"""Build simulation: turn an oriented mesh + voxel grid into build estimates.

Given the winning orientation, this module estimates the things a process
engineer actually cares about before committing a build:

* layer count (from the real layer height) and per-layer cross-section area
* part and support material volume, mass, and cost
* build time (material deposition time + per-layer recoat/peel overhead)
* machine + material cost
* a reduced-order **warpage-risk index** with its contributors broken out

Every number is an estimate from a coarse model, and the most physics-heavy one
-- warpage risk -- is an explicitly heuristic index, not an FEA result. It
combines the recognized drivers of residual-stress distortion (abrupt
layer-to-layer area change, aspect ratio, down-facing/overhang fraction, and
large cross-sections) into one interpretable score. A production pipeline would
hand the geometry to a thermo-mechanical solver here; the index is a fast screen
that flags which parts deserve that solver. See REPORT.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .geometry import Mesh
from .materials import ProcessProfile
from .voxelize import VoxelGrid, support_estimate

# Relative warpage-susceptibility of each process family (heuristic).
_FAMILY_WARPAGE = {"LPBF": 1.0, "FFF": 0.7, "SLA": 0.4, "SLS": 0.3}

# Contributor weights inside the warpage index (sum to 1.0).
_WARP_W = {"area_gradient": 0.30, "aspect_ratio": 0.25, "overhang": 0.20, "cross_section": 0.25}


@dataclass
class BuildSimulation:
    # geometry / validation
    part_volume_mm3: float
    voxel_volume_mm3: float
    volume_error_pct: float
    surface_area_mm2: float
    height_mm: float
    footprint_dims_mm: tuple
    # material
    support_envelope_mm3: float
    support_material_mm3: float
    part_mass_g: float
    support_mass_g: float
    # layers / time
    layer_height_mm: float
    n_layers: int
    deposition_time_h: float
    overhead_time_h: float
    total_time_h: float
    # cost
    material_cost_usd: float
    machine_cost_usd: float
    total_cost_usd: float
    # risk
    warpage_index: float
    warpage_contributors: Dict[str, float]
    aspect_ratio: float
    max_cross_section_mm2: float
    support_interface_area_mm2: float
    # profiles for plots (kept out of repr)
    layer_z_mm: np.ndarray = field(repr=False, default=None)
    layer_area_mm2: np.ndarray = field(repr=False, default=None)
    support_layer_area_mm2: np.ndarray = field(repr=False, default=None)

    @property
    def total_mass_g(self) -> float:
        return self.part_mass_g + self.support_mass_g

    def summary(self) -> Dict[str, object]:
        return {
            "part_volume_cm3": round(self.part_volume_mm3 / 1000.0, 3),
            "volume_error_pct": round(self.volume_error_pct, 3),
            "support_material_cm3": round(self.support_material_mm3 / 1000.0, 3),
            "part_mass_g": round(self.part_mass_g, 2),
            "support_mass_g": round(self.support_mass_g, 2),
            "n_layers": self.n_layers,
            "build_time_h": round(self.total_time_h, 2),
            "total_cost_usd": round(self.total_cost_usd, 2),
            "warpage_index": round(self.warpage_index, 1),
            "aspect_ratio": round(self.aspect_ratio, 2),
        }


def _soft(value: float, scale: float) -> float:
    """Clip ``value / scale`` into [0, 1] -- a saturating normalization."""
    return float(min(1.0, max(0.0, value / scale)))


def _warpage_index(
    profile: ProcessProfile,
    layer_area: np.ndarray,
    height_mm: float,
    min_footprint_mm: float,
    overhang_fraction: float,
    plate_area_mm2: float,
) -> Dict[str, float]:
    nonzero = layer_area[layer_area > 0]
    if nonzero.size >= 2:
        mean_area = float(nonzero.mean())
        max_grad = float(np.abs(np.diff(nonzero)).max()) / (mean_area or 1.0)
        max_cross = float(nonzero.max())
    else:
        max_grad, max_cross = 0.0, float(nonzero.max()) if nonzero.size else 0.0

    aspect = height_mm / max(min_footprint_mm, 1e-6)

    contributors = {
        "area_gradient": _soft(max_grad, 0.5),       # 50%+ jump between layers = saturated
        "aspect_ratio": _soft(aspect, 8.0),          # aspect 8+ = saturated
        "overhang": _soft(overhang_fraction, 0.40),  # 40%+ down-facing = saturated
        "cross_section": _soft(max_cross, 0.5 * plate_area_mm2),
    }
    family = _FAMILY_WARPAGE.get(profile.family, 0.7)
    raw = sum(_WARP_W[k] * v for k, v in contributors.items())
    index = 100.0 * family * raw
    contributors = {k: round(v, 3) for k, v in contributors.items()}
    contributors["family_factor"] = family
    contributors["aspect_ratio_value"] = round(aspect, 3)
    contributors["max_area_gradient"] = round(max_grad, 3)
    return {"index": index, "contributors": contributors, "aspect_ratio": aspect, "max_cross": max_cross}


def simulate_build(mesh: Mesh, grid: VoxelGrid, profile: ProcessProfile) -> BuildSimulation:
    """Run the build simulation for ``mesh`` (oriented, on the plate) on ``grid``."""
    part_vol = mesh.volume_mm3
    voxel_vol = grid.occupied_volume_mm3
    vol_err = 100.0 * (voxel_vol - part_vol) / part_vol if part_vol else 0.0
    surface = mesh.surface_area_mm2

    ext = mesh.extents
    height = float(ext[2])
    footprint_dims = (float(ext[0]), float(ext[1]))
    min_foot = max(min(footprint_dims), 1e-6)

    sup = support_estimate(grid, profile.support_infill_frac)
    support_env = sup["support_envelope_mm3"]
    support_mat = sup["support_material_mm3"]
    interface = sup["support_interface_area_mm2"]

    # material
    rho = profile.density_g_mm3
    part_mass = part_vol * rho
    support_mass = support_mat * rho

    # layers + time
    layer_h = profile.default_layer_height_mm
    n_layers = max(1, int(math.ceil(height / layer_h)))
    deposited_cm3 = (part_vol + support_mat) / 1000.0
    deposition_h = deposited_cm3 / profile.nominal_volume_rate_cm3_per_h
    overhead_h = n_layers * profile.recoat_time_s_per_layer / 3600.0
    total_h = deposition_h + overhead_h

    # cost
    material_cost = (part_mass / 1000.0) * profile.material_cost_per_kg + (
        support_mass / 1000.0
    ) * profile.support_material_cost_per_kg
    machine_cost = total_h * profile.machine_rate_per_hour
    total_cost = material_cost + machine_cost

    # warpage risk
    layer_area = grid.layer_area_mm2()
    plate_area = profile.build_volume_mm[0] * profile.build_volume_mm[1]
    overhang_fraction = interface / surface if surface else 0.0
    warp = _warpage_index(profile, layer_area, height, min_foot, overhang_fraction, plate_area)

    return BuildSimulation(
        part_volume_mm3=part_vol,
        voxel_volume_mm3=voxel_vol,
        volume_error_pct=vol_err,
        surface_area_mm2=surface,
        height_mm=height,
        footprint_dims_mm=footprint_dims,
        support_envelope_mm3=support_env,
        support_material_mm3=support_mat,
        part_mass_g=part_mass,
        support_mass_g=support_mass,
        layer_height_mm=layer_h,
        n_layers=n_layers,
        deposition_time_h=deposition_h,
        overhead_time_h=overhead_h,
        total_time_h=total_h,
        material_cost_usd=material_cost,
        machine_cost_usd=machine_cost,
        total_cost_usd=total_cost,
        warpage_index=warp["index"],
        warpage_contributors=warp["contributors"],
        aspect_ratio=warp["aspect_ratio"],
        max_cross_section_mm2=warp["max_cross"],
        support_interface_area_mm2=interface,
        layer_z_mm=grid.z_centers(),
        layer_area_mm2=layer_area,
        support_layer_area_mm2=sup["support_layer_area_mm2"],
    )
