"""Voxelization engine -- the heart of the build simulation.

We discretize the (already oriented, plate-dropped) mesh into an occupancy grid
by *ray stabbing*: for every (x, y) column of the grid we shoot a vertical ray,
collect where it crosses the mesh, sort the crossings, and fill the voxels
between entry/exit pairs (the even-odd / crossing-number rule). This is a
standard, transparent way to convert a triangle soup into a solid model, and it
is fully vectorized per triangle on top of numpy.

The resulting grid drives several downstream estimates that are otherwise hard
to get from a raw mesh:

* occupied volume -- cross-checked against the analytic mesh volume (validation)
* per-layer cross-sectional area -- feeds the build-time and warpage models
* support volume -- empty space that sits under solid and must be filled
* thin-wall fraction -- via a morphological opening
* trapped volume -- enclosed voids found by flood-filling from outside

The grid is a *reduced-order* model. Its resolution is set by ``grid_n`` (voxels
along the longest axis), so every derived number is an estimate that converges
as the grid is refined. REPORT.md shows that convergence on an analytic cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .geometry import Mesh

# Deterministic, *per-axis-distinct* sub-voxel offset so grid sample points
# rarely land exactly on a triangle edge or face diagonal (which would
# double-count crossings). The offsets differ between axes on purpose: equal
# offsets would preserve the y=x diagonal that splits an axis-aligned face into
# two triangles, sending every diagonal column straight onto a shared edge.
# No RNG -> fully reproducible.
_GRID_JITTER = np.array([0.13579, 0.24681, 0.31415])


@dataclass
class VoxelGrid:
    occ: np.ndarray          # bool, shape (nx, ny, nz)
    pitch: float             # voxel edge length, mm
    origin: np.ndarray       # world coord of the lo corner of voxel (0,0,0)

    @property
    def cell_volume_mm3(self) -> float:
        return self.pitch ** 3

    @property
    def cell_area_mm2(self) -> float:
        return self.pitch ** 2

    @property
    def occupied_volume_mm3(self) -> float:
        return float(self.occ.sum()) * self.cell_volume_mm3

    @property
    def shape(self):
        return self.occ.shape

    def layer_area_mm2(self) -> np.ndarray:
        """Cross-sectional area per z-slice (voxel resolution)."""
        return self.occ.sum(axis=(0, 1)).astype(np.float64) * self.cell_area_mm2

    def z_centers(self) -> np.ndarray:
        nz = self.occ.shape[2]
        return self.origin[2] + (np.arange(nz) + 0.5) * self.pitch


def voxelize(mesh: Mesh, grid_n: int = 64) -> VoxelGrid:
    """Voxelize ``mesh`` into a grid with ~``grid_n`` voxels on the longest axis."""
    lo, hi = mesh.bounds
    extents = hi - lo
    pitch = float(extents.max()) / grid_n
    if pitch <= 0:
        raise ValueError("Degenerate mesh: zero extent.")

    nx, ny, nz = (np.ceil(extents / pitch).astype(int) + 1)
    nx, ny, nz = int(nx), int(ny), int(nz)
    origin = lo - _GRID_JITTER * pitch  # per-axis nudge so centers avoid exact edges
    dedupe_tol = 1e-6 * pitch           # crossings closer than this are one surface

    xs = origin[0] + (np.arange(nx) + 0.5) * pitch
    ys = origin[1] + (np.arange(ny) + 0.5) * pitch

    # Per-column list of z-crossings, gathered triangle by triangle.
    crossings = [[[] for _ in range(ny)] for _ in range(nx)]

    tris = mesh.triangles
    for tri in tris:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tri
        # column index range covered by this triangle's xy bounding box
        xmin, xmax = min(x0, x1, x2), max(x0, x1, x2)
        ymin, ymax = min(y0, y1, y2), max(y0, y1, y2)
        i0 = max(0, int(np.floor((xmin - origin[0]) / pitch)))
        i1 = min(nx - 1, int(np.ceil((xmax - origin[0]) / pitch)))
        j0 = max(0, int(np.floor((ymin - origin[1]) / pitch)))
        j1 = min(ny - 1, int(np.ceil((ymax - origin[1]) / pitch)))
        if i1 < i0 or j1 < j0:
            continue

        # barycentric test for the candidate column centers
        gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-14:
            continue  # triangle is edge-on (vertical) -> contributes no crossings
        a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        z_hit = a * z0 + b * z1 + c * z2  # plane z at each interior column center

        ii, jj = np.nonzero(inside)
        for di, dj in zip(ii, jj):
            crossings[i0 + di][j0 + dj].append(float(z_hit[di, dj]))

    occ = np.zeros((nx, ny, nz), dtype=bool)
    z0_world = origin[2]
    for i in range(nx):
        row = crossings[i]
        for j in range(ny):
            zs = row[j]
            if len(zs) < 2:
                continue
            zs.sort()
            # Collapse coincident crossings (a column landing on a shared edge or
            # vertex hits two facets at the same z). Without this the even-odd
            # pairing flips and the column fills incorrectly.
            deduped = [zs[0]]
            for z in zs[1:]:
                if z - deduped[-1] > dedupe_tol:
                    deduped.append(z)
            zs = deduped
            if len(zs) < 2:
                continue
            # pair consecutive crossings (drop a trailing unpaired grazing hit).
            # A voxel is solid iff its *center* lies within an entry/exit span --
            # the same center-sampling rule used in x and y, which makes the
            # discretized volume converge without bias.
            for k in range(0, len(zs) - 1, 2):
                z_enter, z_exit = zs[k], zs[k + 1]
                ka = int(np.ceil((z_enter - z0_world) / pitch - 0.5))
                kb = int(np.floor((z_exit - z0_world) / pitch - 0.5))
                ka = max(0, ka)
                kb = min(nz - 1, kb)
                if kb >= ka:
                    occ[i, j, ka:kb + 1] = True

    return VoxelGrid(occ=occ, pitch=pitch, origin=origin)


# ---------------------------------------------------------------------------
# Grid morphology helpers (6-connectivity, zero-fill at the boundary)
# ---------------------------------------------------------------------------
def _shift(a: np.ndarray, axis: int, step: int) -> np.ndarray:
    """Shift ``a`` along ``axis`` by ``step``, filling vacated cells with False."""
    out = np.zeros_like(a)
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if step > 0:
        dst[axis] = slice(step, None)
        src[axis] = slice(None, -step)
    elif step < 0:
        dst[axis] = slice(None, step)
        src[axis] = slice(-step, None)
    else:
        return a.copy()
    out[tuple(dst)] = a[tuple(src)]
    return out


def _erode1(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for axis in range(3):
        out &= _shift(a, axis, 1)
        out &= _shift(a, axis, -1)
    return out


def _dilate1(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for axis in range(3):
        out |= _shift(a, axis, 1)
        out |= _shift(a, axis, -1)
    return out


def support_estimate(grid: VoxelGrid, infill_frac: float) -> Dict[str, object]:
    """Estimate support material from the occupancy grid.

    Any empty voxel that has solid somewhere above it in the same column cannot
    be printed onto air, so it must be filled with support down to the plate or
    the surface below. The support *interface* is the set of solid voxels whose
    neighbor directly below is empty (and that are not resting on the plate) --
    this drives support-removal effort and down-facing surface quality.
    """
    occ = grid.occ
    # reverse cumulative-or along z, then shift down -> "solid strictly above"
    cum_or = np.flip(np.maximum.accumulate(np.flip(occ, axis=2), axis=2), axis=2)
    solid_above = _shift(cum_or, axis=2, step=-1)
    empty = ~occ
    support_mask = empty & solid_above

    interface = occ & ~_shift(occ, axis=2, step=1)
    interface[:, :, 0] = False  # voxels on the plate are not overhang interfaces

    cell_vol = grid.cell_volume_mm3
    support_cells = int(support_mask.sum())
    return {
        "support_envelope_mm3": support_cells * cell_vol,
        "support_material_mm3": support_cells * cell_vol * infill_frac,
        "support_interface_area_mm2": int(interface.sum()) * grid.cell_area_mm2,
        "support_layer_area_mm2": support_mask.sum(axis=(0, 1)).astype(np.float64)
        * grid.cell_area_mm2,
    }


def thin_wall_analysis(grid: VoxelGrid, min_wall_mm: float) -> Dict[str, object]:
    """Flag features thinner than ``min_wall_mm`` with a morphological opening.

    Eroding then dilating (an opening) removes any feature that a ball of radius
    ~``min_wall_mm/2`` cannot fit inside. Voxels present before the opening but
    gone after it belong to walls thinner than the printable minimum.
    """
    occ = grid.occ
    radius_vox = int(round((min_wall_mm / grid.pitch) / 2.0))
    if radius_vox < 1 or occ.sum() == 0:
        return {"thin_fraction": 0.0, "thin_volume_mm3": 0.0, "radius_voxels": radius_vox}

    eroded = occ.copy()
    for _ in range(radius_vox):
        eroded = _erode1(eroded)
    opened = eroded.copy()
    for _ in range(radius_vox):
        opened = _dilate1(opened)
    opened &= occ

    thin = occ & ~opened
    thin_cells = int(thin.sum())
    return {
        "thin_fraction": thin_cells / float(occ.sum()),
        "thin_volume_mm3": thin_cells * grid.cell_volume_mm3,
        "radius_voxels": radius_vox,
    }


def trapped_volume(grid: VoxelGrid) -> Dict[str, object]:
    """Find enclosed voids by flood-filling empty space from the grid boundary.

    Empty voxels reachable from the outside drain freely. Empty voxels that the
    flood cannot reach are enclosed cavities -- trapped powder or resin for
    processes that need drain holes.
    """
    occ = grid.occ
    empty = ~occ
    reachable = np.zeros_like(occ)
    # seed every empty voxel on the six faces of the grid
    reachable[0, :, :] |= empty[0, :, :]
    reachable[-1, :, :] |= empty[-1, :, :]
    reachable[:, 0, :] |= empty[:, 0, :]
    reachable[:, -1, :] |= empty[:, -1, :]
    reachable[:, :, 0] |= empty[:, :, 0]
    reachable[:, :, -1] |= empty[:, :, -1]

    # iterative flood fill within empty space until it stops growing
    while True:
        grown = _dilate1(reachable) & empty
        if grown.sum() == reachable.sum():
            break
        reachable = grown

    trapped = empty & ~reachable
    return {
        "trapped_volume_mm3": int(trapped.sum()) * grid.cell_volume_mm3,
        "has_trapped_volume": bool(trapped.any()),
    }
