"""Build-orientation selection by resting the part on candidate flat faces.

Orientation is the highest-leverage decision in additive manufacturing, so it
deserves a physically meaningful search rather than an arbitrary angle sweep. We
generate candidates the way an engineer does -- *which face goes down on the
plate* -- by clustering the mesh's facet normals into its significant flat faces
and adding the six bounding-box directions as a fallback. Each candidate rests a
real face on the plate, so there are no degenerate edge-balanced orientations.

Each candidate is then scored on quantities that actually matter, measured from
a coarse voxelization of that orientation:

* **support material volume** -- the dominant cost/quality driver (minimize);
* **base contact area** -- a large flat footprint means good adhesion and
  stability (maximize); a tiny contact patch is penalized;
* **build height** -- drives build time (minimize);

with a hard penalty for any orientation that does not fit the build volume. The
cheap coarse voxelization screens the whole candidate set; the full-resolution
simulation and FEA then run once, on the winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .geometry import Mesh, _rotation_between
from .materials import ProcessProfile
from .voxelize import support_estimate, voxelize

ORIENT_GRID_N = 28  # coarse grid for screening candidate orientations

# Objective weights (lower score is better).
_W_SUPPORT = 0.45
_W_CONTACT = 0.35
_W_HEIGHT = 0.20

_DOWN = np.array([0.0, 0.0, -1.0])


@dataclass
class OrientationCandidate:
    index: int
    down_dir: tuple                 # mesh direction (in original frame) now facing down
    label: str
    rotation: np.ndarray = field(repr=False)
    height_mm: float
    footprint_mm2: float
    base_contact_mm2: float
    contact_fraction: float
    support_volume_mm3: float
    fits_build_volume: bool
    score: float = 0.0

    def summary(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "down_dir": [round(float(v), 3) for v in self.down_dir],
            "label": self.label,
            "height_mm": round(self.height_mm, 3),
            "base_contact_mm2": round(self.base_contact_mm2, 1),
            "contact_fraction": round(self.contact_fraction, 3),
            "support_volume_mm3": round(self.support_volume_mm3, 1),
            "fits_build_volume": bool(self.fits_build_volume),
            "score": round(self.score, 4),
        }


def _candidate_down_directions(mesh: Mesh, max_faces: int = 8) -> List[np.ndarray]:
    """Down-direction candidates: significant flat-face normals + the 6 bbox axes."""
    normals = mesh.face_normals
    areas = mesh.face_areas
    # cluster facets by rounded normal, accumulate area
    keys = np.round(normals, 2)
    buckets: Dict[tuple, list] = {}
    for n, a, k in zip(normals, areas, map(tuple, keys)):
        if k not in buckets:
            buckets[k] = [a, n * a]
        else:
            buckets[k][0] += a
            buckets[k][1] = buckets[k][1] + n * a
    ranked = sorted(buckets.values(), key=lambda v: -v[0])
    max_area = ranked[0][0] if ranked else 1.0
    dirs: List[np.ndarray] = []
    for area, area_weighted_normal in ranked[:max_faces]:
        if area < 0.03 * max_area:
            break
        d = area_weighted_normal / (np.linalg.norm(area_weighted_normal) or 1.0)
        dirs.append(d)  # this face's outward normal -> point it DOWN
    # always include the 6 axis directions for coverage
    for ax in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
        dirs.append(np.array(ax, dtype=float))
    # dedupe by rounded direction
    seen = set()
    unique: List[np.ndarray] = []
    for d in dirs:
        key = tuple(np.round(d, 2))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _evaluate(mesh: Mesh, profile: ProcessProfile, down_dir: np.ndarray, index: int) -> OrientationCandidate:
    rot = _rotation_between(down_dir, _DOWN)
    oriented = mesh.transformed(rot).dropped_to_plate()
    ext = oriented.extents
    height = float(ext[2])
    footprint = float(ext[0] * ext[1])

    grid = voxelize(oriented, grid_n=ORIENT_GRID_N)
    base_contact = float(grid.occ[:, :, 0].sum()) * grid.cell_area_mm2
    support = support_estimate(grid, profile.support_infill_frac)["support_material_mm3"]
    if profile.support_infill_frac == 0.0:
        # self-supporting process: use the geometric overhang need so orientation
        # still distinguishes, but de-weighted (handled by small envelope numbers)
        support = profile.support_infill_frac  # 0; orientation driven by height+contact

    bv = profile.build_volume_mm
    fits = ext[0] <= bv[0] and ext[1] <= bv[1] and ext[2] <= bv[2]
    contact_frac = base_contact / footprint if footprint > 0 else 0.0

    label = f"face ⟂ ({down_dir[0]:+.2f}, {down_dir[1]:+.2f}, {down_dir[2]:+.2f})"
    return OrientationCandidate(
        index=index, down_dir=tuple(float(v) for v in down_dir), label=label,
        rotation=rot, height_mm=height, footprint_mm2=footprint,
        base_contact_mm2=base_contact, contact_fraction=contact_frac,
        support_volume_mm3=float(support), fits_build_volume=fits,
    )


def _normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def optimize_orientation(mesh: Mesh, profile: ProcessProfile, **_) -> Dict[str, object]:
    """Screen rest-on-face orientations; return ranked candidates and the winner."""
    dirs = _candidate_down_directions(mesh)
    candidates = [_evaluate(mesh, profile, d, i) for i, d in enumerate(dirs)]

    # dedupe physically identical orientations (same screened metrics)
    seen = {}
    distinct: List[OrientationCandidate] = []
    for c in candidates:
        key = (round(c.height_mm, 2), round(c.base_contact_mm2, 1), round(c.support_volume_mm3, 1))
        if key not in seen:
            seen[key] = True
            distinct.append(c)

    n_support = _normalize([c.support_volume_mm3 for c in distinct])
    n_height = _normalize([c.height_mm for c in distinct])
    contact_pen = [1.0 - c.contact_fraction for c in distinct]  # already in [0,1]
    for c, s, h, cp in zip(distinct, n_support, n_height, contact_pen):
        penalty = 0.0 if c.fits_build_volume else 1.0
        c.score = float(_W_SUPPORT * s + _W_HEIGHT * h + _W_CONTACT * cp + penalty)

    distinct.sort(key=lambda c: c.score)
    best = distinct[0]
    return {
        "best": best,
        "candidates": distinct,
        "design": {
            "method": "rest-on-face screening",
            "n_candidates": len(candidates),
            "n_distinct": len(distinct),
            "screen_grid_n": ORIENT_GRID_N,
            "weights": {"support": _W_SUPPORT, "contact": _W_CONTACT, "height": _W_HEIGHT},
            "self_support_angle_deg": profile.self_support_angle_deg,
        },
    }
