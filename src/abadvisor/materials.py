"""Process + material library for the build simulation.

A ``ProcessProfile`` bundles everything the simulator needs to reason about a
build: the material's density and cost, the machine's build envelope and
amortized rate, the layer height, the self-support overhang angle, the
Design-for-AM minimums, and a small build-time model.

The numbers are *representative* defaults for each process family, not
machine-qualified values for a specific OEM. They are deliberately easy to find
and override (see ``ProcessProfile`` fields). The point of the project is the
workflow and the engineering judgment around the numbers, not the numbers
themselves -- a production version would pull these from a qualified machine
profile. See REPORT.md, "Honest scope".

Self-support angle convention
-----------------------------
``self_support_angle_deg`` is the minimum inclination of a *downward-facing*
surface, measured from the horizontal build plate, that prints reliably without
support. A flat downward overhang is 0 deg (worst case); a vertical wall is
90 deg (always fine). Powder-bed processes (SLS) are self-supporting, so their
angle is 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ProcessProfile:
    key: str
    name: str
    family: str  # FFF | SLA | SLS | LPBF
    material: str

    # Material
    density_g_cm3: float
    material_cost_per_kg: float
    support_material_cost_per_kg: float

    # Machine
    machine_rate_per_hour: float           # amortized machine + labor, USD/h
    build_volume_mm: Tuple[float, float, float]
    default_layer_height_mm: float

    # Design-for-AM minimums / rules
    self_support_angle_deg: float          # see module docstring
    min_wall_mm: float
    min_hole_mm: float
    min_feature_mm: float
    support_infill_frac: float             # fraction of the support envelope actually solid

    # Build-time model
    nominal_volume_rate_cm3_per_h: float   # effective solid deposition / fusion rate
    recoat_time_s_per_layer: float         # per-layer overhead (recoat / peel / Z + travel)

    # Process flags
    needs_drain_holes: bool                # trapped fluid (resin) or powder must escape
    post_processing: List[str] = field(default_factory=list)

    # Mechanical properties for the distortion FEA (thermal-contraction method).
    # ``contraction_strain`` is the representative isotropic thermal-contraction
    # eigenstrain that drives the warpage solve (negative = the part shrinks as it
    # cools from the deposition temperature back toward the bed/chamber). It is of
    # order eps* ~ alpha * dT_eff -- e.g. PLA (alpha ~ 68e-6 /K, dT_eff ~ 90 K)
    # gives ~ -0.006; ABS contracts (and warps) more, so it carries a larger value.
    # These are illustrative per-process values, not constants fit to a measured
    # cooling history. The same eigenstrain machinery is the "inherent-strain
    # method" when it is applied to metal powder-bed fusion.
    youngs_modulus_mpa: float = 3000.0
    poisson_ratio: float = 0.35
    contraction_strain: float = -0.006

    @property
    def density_g_mm3(self) -> float:
        return self.density_g_cm3 / 1000.0


_PROFILES: Dict[str, ProcessProfile] = {
    p.key: p
    for p in [
        ProcessProfile(
            key="fff_pla",
            name="FFF / FDM (PLA)",
            family="FFF",
            material="PLA",
            density_g_cm3=1.24,
            material_cost_per_kg=25.0,
            support_material_cost_per_kg=25.0,
            machine_rate_per_hour=5.0,
            build_volume_mm=(256.0, 256.0, 256.0),
            default_layer_height_mm=0.20,
            self_support_angle_deg=45.0,
            min_wall_mm=0.8,
            min_hole_mm=2.0,
            min_feature_mm=0.5,
            support_infill_frac=0.15,
            nominal_volume_rate_cm3_per_h=12.0,
            recoat_time_s_per_layer=1.5,
            needs_drain_holes=False,
            post_processing=["support removal", "optional sanding"],
            youngs_modulus_mpa=3500.0,
            poisson_ratio=0.36,
            contraction_strain=-0.006,
        ),
        ProcessProfile(
            key="fff_abs",
            name="FFF / FDM (ABS)",
            family="FFF",
            material="ABS",
            density_g_cm3=1.04,
            material_cost_per_kg=28.0,
            support_material_cost_per_kg=28.0,
            machine_rate_per_hour=6.0,
            build_volume_mm=(256.0, 256.0, 256.0),
            default_layer_height_mm=0.20,
            self_support_angle_deg=45.0,
            min_wall_mm=1.0,
            min_hole_mm=2.0,
            min_feature_mm=0.6,
            support_infill_frac=0.15,
            nominal_volume_rate_cm3_per_h=12.0,
            recoat_time_s_per_layer=1.5,
            needs_drain_holes=False,
            post_processing=["support removal", "vapor smoothing (optional)"],
            youngs_modulus_mpa=2200.0,
            poisson_ratio=0.35,
            contraction_strain=-0.012,
        ),
        ProcessProfile(
            key="sls_pa12",
            name="SLS (PA12 Nylon)",
            family="SLS",
            material="PA12",
            density_g_cm3=1.01,
            material_cost_per_kg=60.0,
            support_material_cost_per_kg=0.0,  # powder bed is self-supporting
            machine_rate_per_hour=25.0,
            build_volume_mm=(300.0, 300.0, 300.0),
            default_layer_height_mm=0.10,
            self_support_angle_deg=0.0,  # self-supporting
            min_wall_mm=0.7,
            min_hole_mm=1.5,
            min_feature_mm=0.4,
            support_infill_frac=0.0,
            nominal_volume_rate_cm3_per_h=18.0,
            recoat_time_s_per_layer=9.0,
            needs_drain_holes=True,  # trapped powder must escape
            post_processing=["depowder", "bead blast", "optional dye"],
            youngs_modulus_mpa=1700.0,
            poisson_ratio=0.4,
            contraction_strain=-0.004,
        ),
        ProcessProfile(
            key="sla_resin",
            name="SLA / DLP (Standard Resin)",
            family="SLA",
            material="Photopolymer resin",
            density_g_cm3=1.15,
            material_cost_per_kg=120.0,
            support_material_cost_per_kg=120.0,
            machine_rate_per_hour=8.0,
            build_volume_mm=(145.0, 145.0, 185.0),
            default_layer_height_mm=0.05,
            self_support_angle_deg=35.0,
            min_wall_mm=0.5,
            min_hole_mm=0.5,
            min_feature_mm=0.3,
            support_infill_frac=0.08,
            nominal_volume_rate_cm3_per_h=12.0,
            recoat_time_s_per_layer=7.0,
            needs_drain_holes=True,  # trapped resin / suction cups
            post_processing=["support removal", "wash (IPA)", "UV cure"],
            youngs_modulus_mpa=2800.0,
            poisson_ratio=0.4,
            contraction_strain=-0.003,
        ),
        # Metal laser powder-bed fusion. FFF is this project's home turf (it is what
        # my ES 51 students print); these metal profiles are kept so the same
        # pipeline + thermal-contraction FEA can be run on metal, where the method is
        # known as the inherent-strain method, as a point of comparison.
        ProcessProfile(
            key="lpbf_alsi10mg",
            name="LPBF Metal (AlSi10Mg)",
            family="LPBF",
            material="AlSi10Mg",
            density_g_cm3=2.67,
            material_cost_per_kg=80.0,
            support_material_cost_per_kg=80.0,
            machine_rate_per_hour=80.0,
            build_volume_mm=(250.0, 250.0, 325.0),
            default_layer_height_mm=0.03,
            self_support_angle_deg=45.0,
            min_wall_mm=0.4,
            min_hole_mm=0.5,
            min_feature_mm=0.3,
            support_infill_frac=0.20,
            nominal_volume_rate_cm3_per_h=20.0,
            recoat_time_s_per_layer=9.0,
            needs_drain_holes=False,
            post_processing=["stress relief", "wire-EDM off plate", "support machining"],
            youngs_modulus_mpa=70000.0,
            poisson_ratio=0.33,
            contraction_strain=-0.008,
        ),
        ProcessProfile(
            key="lpbf_in625",
            name="LPBF Metal (Inconel 625)",
            family="LPBF",
            material="IN625",
            density_g_cm3=8.44,
            material_cost_per_kg=110.0,
            support_material_cost_per_kg=110.0,
            machine_rate_per_hour=90.0,
            build_volume_mm=(250.0, 250.0, 325.0),
            default_layer_height_mm=0.03,
            self_support_angle_deg=45.0,
            min_wall_mm=0.4,
            min_hole_mm=0.5,
            min_feature_mm=0.3,
            support_infill_frac=0.22,
            nominal_volume_rate_cm3_per_h=10.0,
            recoat_time_s_per_layer=10.0,
            needs_drain_holes=False,
            post_processing=["stress relief", "wire-EDM off plate", "support machining"],
            youngs_modulus_mpa=208000.0,
            poisson_ratio=0.30,
            contraction_strain=-0.008,
        ),
        ProcessProfile(
            key="lpbf_ti64",
            name="LPBF Metal (Ti-6Al-4V)",
            family="LPBF",
            material="Ti-6Al-4V",
            density_g_cm3=4.43,
            material_cost_per_kg=350.0,
            support_material_cost_per_kg=350.0,
            machine_rate_per_hour=90.0,
            build_volume_mm=(250.0, 250.0, 325.0),
            default_layer_height_mm=0.03,
            self_support_angle_deg=45.0,
            min_wall_mm=0.4,
            min_hole_mm=0.5,
            min_feature_mm=0.3,
            support_infill_frac=0.25,
            nominal_volume_rate_cm3_per_h=12.0,
            recoat_time_s_per_layer=10.0,
            needs_drain_holes=False,
            post_processing=["stress relief", "HIP (optional)", "wire-EDM off plate", "support machining"],
            youngs_modulus_mpa=110000.0,
            poisson_ratio=0.34,
            contraction_strain=-0.01,
        ),
    ]
}


def get_profile(key: str) -> ProcessProfile:
    try:
        return _PROFILES[key]
    except KeyError:
        raise KeyError(
            f"Unknown process profile {key!r}. "
            f"Available: {', '.join(sorted(_PROFILES))}"
        )


def list_profiles() -> List[ProcessProfile]:
    return list(_PROFILES.values())
