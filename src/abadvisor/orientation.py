"""Build-orientation selection as a design-of-experiments (DoE) sweep.

Orientation is the single highest-leverage decision in additive manufacturing:
it sets how much support is needed, how tall (and therefore how slow) the build
is, how stable the part is on the plate, and which surfaces end up rough. So we
treat it as a small DoE -- a full factorial over two rotation factors -- and
score every candidate against a transparent, weighted objective.

Scoring uses only *analytic facet metrics* (overhang area from face normals,
bounding-box height, footprint, center-of-mass height). These are cheap, so we
can evaluate the whole design space quickly; the expensive voxel simulation
then runs once, on the winning orientation. That two-tier structure -- screen
cheaply, simulate the survivor -- is the same DoE discipline used to keep
physical experiment counts down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .geometry import Mesh, rotation_about_axis
from .materials import ProcessProfile

_DEFAULT_LEVELS = (0.0, 45.0, 90.0, 135.0)

# Objective weights (lower score is better). Support dominates because it drives
# both cost and down-facing surface quality; height drives build time; stability
# guards against tall, tippy builds.
_W_SUPPORT = 0.50
_W_HEIGHT = 0.30
_W_STABILITY = 0.20


@dataclass
class OrientationCandidate:
    index: int
    rx_deg: float
    ry_deg: float
    rotation: np.ndarray = field(repr=False)
    height_mm: float
    footprint_mm2: float
    footprint_dims_mm: tuple
    overhang_area_mm2: float
    com_height_mm: float
    stability_ratio: float          # COM height / min footprint dim (higher = tippier)
    fits_build_volume: bool
    score: float = 0.0

    def summary(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "rx_deg": self.rx_deg,
            "ry_deg": self.ry_deg,
            "height_mm": round(self.height_mm, 3),
            "footprint_mm2": round(self.footprint_mm2, 2),
            "overhang_area_mm2": round(self.overhang_area_mm2, 2),
            "stability_ratio": round(self.stability_ratio, 3),
            "fits_build_volume": self.fits_build_volume,
            "score": round(self.score, 4),
        }


def _evaluate(mesh: Mesh, profile: ProcessProfile, rx: float, ry: float, index: int) -> OrientationCandidate:
    rot = rotation_about_axis([0, 1, 0], ry) @ rotation_about_axis([1, 0, 0], rx)
    oriented = mesh.transformed(rot).dropped_to_plate()

    ext = oriented.extents
    height = float(ext[2])
    footprint_dims = (float(ext[0]), float(ext[1]))
    footprint = footprint_dims[0] * footprint_dims[1]
    overhang = oriented.overhang_area_mm2(profile.self_support_angle_deg)
    com = oriented.center_of_mass
    com_height = float(com[2])
    min_foot = max(min(footprint_dims), 1e-6)
    stability = com_height / min_foot

    bv = profile.build_volume_mm
    fits = ext[0] <= bv[0] and ext[1] <= bv[1] and ext[2] <= bv[2]

    return OrientationCandidate(
        index=index, rx_deg=rx, ry_deg=ry, rotation=rot,
        height_mm=height, footprint_mm2=footprint, footprint_dims_mm=footprint_dims,
        overhang_area_mm2=overhang, com_height_mm=com_height,
        stability_ratio=stability, fits_build_volume=fits,
    )


def _normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def optimize_orientation(
    mesh: Mesh,
    profile: ProcessProfile,
    levels: Sequence[float] = _DEFAULT_LEVELS,
) -> Dict[str, object]:
    """Run the orientation DoE and return ranked candidates plus the winner."""
    candidates: List[OrientationCandidate] = []
    idx = 0
    for rx in levels:
        for ry in levels:
            candidates.append(_evaluate(mesh, profile, rx, ry, idx))
            idx += 1

    # de-duplicate orientations that are physically identical (same metrics)
    seen = {}
    distinct: List[OrientationCandidate] = []
    for c in candidates:
        key = (round(c.height_mm, 3), round(c.footprint_mm2, 1), round(c.overhang_area_mm2, 1))
        if key not in seen:
            seen[key] = True
            distinct.append(c)

    n_over = _normalize([c.overhang_area_mm2 for c in distinct])
    n_height = _normalize([c.height_mm for c in distinct])
    n_stab = _normalize([c.stability_ratio for c in distinct])
    for c, o, h, s in zip(distinct, n_over, n_height, n_stab):
        penalty = 0.0 if c.fits_build_volume else 1.0  # hard push to the bottom
        c.score = float(_W_SUPPORT * o + _W_HEIGHT * h + _W_STABILITY * s + penalty)

    distinct.sort(key=lambda c: c.score)
    best = distinct[0]

    return {
        "best": best,
        "candidates": distinct,
        "design": {
            "factors": ["rx_deg", "ry_deg"],
            "levels": list(levels),
            "n_evaluated": len(candidates),
            "n_distinct": len(distinct),
            "weights": {"support": _W_SUPPORT, "height": _W_HEIGHT, "stability": _W_STABILITY},
            "self_support_angle_deg": profile.self_support_angle_deg,
        },
    }
