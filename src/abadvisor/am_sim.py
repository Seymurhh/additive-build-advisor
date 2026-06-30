"""Build simulation: turn an oriented mesh + voxel grid into build estimates.

Given the winning orientation, this estimates the things a process engineer
cares about before committing a build: layer count and per-layer cross-section,
support material volume, build time (deposition + per-layer overhead), and cost
(material + amortized machine time).

Distortion / warpage is deliberately *not* estimated here with a heuristic --
it is solved separately as a finite-element problem in :mod:`abadvisor.fea`
(the thermal-contraction / eigenstrain method) and attached downstream. This
module only keeps the geometric descriptors (aspect ratio, peak cross-section)
that the DfAM checks use directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .geometry import Mesh
from .materials import ProcessProfile
from .voxelize import VoxelGrid, support_estimate


@dataclass
class BuildSimulation:
    # geometry / validation
    part_volume_mm3: float
    voxel_volume_mm3: float
    volume_error_pct: float
    surface_area_mm2: float
    height_mm: float
    footprint_dims_mm: tuple
    aspect_ratio: float
    max_cross_section_mm2: float
    # material
    support_envelope_mm3: float
    support_material_mm3: float
    support_interface_area_mm2: float
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
            "aspect_ratio": round(self.aspect_ratio, 2),
        }


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
    aspect = height / min_foot

    layer_area = grid.layer_area_mm2()
    nonzero = layer_area[layer_area > 0]
    max_cross = float(nonzero.max()) if nonzero.size else 0.0

    sup = support_estimate(grid, profile.support_infill_frac)
    support_env = sup["support_envelope_mm3"]
    support_mat = sup["support_material_mm3"]
    interface = sup["support_interface_area_mm2"]

    rho = profile.density_g_mm3
    part_mass = part_vol * rho
    support_mass = support_mat * rho

    layer_h = profile.default_layer_height_mm
    n_layers = max(1, int(math.ceil(height / layer_h)))
    deposited_cm3 = (part_vol + support_mat) / 1000.0
    deposition_h = deposited_cm3 / profile.nominal_volume_rate_cm3_per_h
    overhead_h = n_layers * profile.recoat_time_s_per_layer / 3600.0
    total_h = deposition_h + overhead_h

    material_cost = (part_mass / 1000.0) * profile.material_cost_per_kg + (
        support_mass / 1000.0
    ) * profile.support_material_cost_per_kg
    machine_cost = total_h * profile.machine_rate_per_hour
    total_cost = material_cost + machine_cost

    return BuildSimulation(
        part_volume_mm3=part_vol,
        voxel_volume_mm3=voxel_vol,
        volume_error_pct=vol_err,
        surface_area_mm2=surface,
        height_mm=height,
        footprint_dims_mm=footprint_dims,
        aspect_ratio=aspect,
        max_cross_section_mm2=max_cross,
        support_envelope_mm3=support_env,
        support_material_mm3=support_mat,
        support_interface_area_mm2=interface,
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
        layer_z_mm=grid.z_centers(),
        layer_area_mm2=layer_area,
        support_layer_area_mm2=sup["support_layer_area_mm2"],
    )
