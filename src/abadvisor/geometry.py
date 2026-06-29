"""Mesh geometry, computed from first principles on top of numpy.

A :class:`Mesh` is a triangle soup (the STL model). From it we derive the
quantities the rest of the pipeline needs:

* facet normals recomputed from vertex winding (the file's normals are ignored)
* facet areas and total surface area
* signed volume via the divergence theorem
* axis-aligned bounding box and the solid's center of mass
* a watertight / manifold check (every welded edge shared by exactly two facets)
* overhang area for a given build direction
* affine transforms (rotation) and a stable geometry hash for the digital thread

All angle conventions assume the build direction is +Z. The orientation search
rotates the mesh so that this holds, which keeps every downstream calculation
simple and direction-agnostic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict

import numpy as np

# Build direction is +Z everywhere downstream of the orientation step.
BUILD_DIR = np.array([0.0, 0.0, 1.0])


@dataclass
class Mesh:
    """A triangle-soup mesh. ``triangles`` has shape ``(N, 3, 3)``."""

    triangles: np.ndarray

    def __post_init__(self) -> None:
        self.triangles = np.asarray(self.triangles, dtype=np.float64)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape (N, 3, 3)")

    # ---- basic counts -------------------------------------------------
    @property
    def n_facets(self) -> int:
        return int(self.triangles.shape[0])

    # ---- per-facet quantities ----------------------------------------
    @property
    def _edge_vectors(self):
        v0, v1, v2 = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        return v0, v1 - v0, v2 - v0

    @property
    def face_normals(self) -> np.ndarray:
        """Unit normals recomputed from winding (right-hand rule)."""
        _, e1, e2 = self._edge_vectors
        n = np.cross(e1, e2)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        lengths[lengths == 0.0] = 1.0  # guard degenerate facets
        return n / lengths

    @property
    def face_areas(self) -> np.ndarray:
        _, e1, e2 = self._edge_vectors
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    @property
    def surface_area_mm2(self) -> float:
        return float(self.face_areas.sum())

    # ---- bulk quantities ---------------------------------------------
    @property
    def volume_mm3(self) -> float:
        """Signed volume via the divergence theorem; returns the magnitude."""
        v0, v1, v2 = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        signed = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
        return float(abs(signed.sum()))

    @property
    def signed_volume_mm3(self) -> float:
        v0, v1, v2 = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        signed = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
        return float(signed.sum())

    @property
    def center_of_mass(self) -> np.ndarray:
        """Volume centroid of the solid (assumes a closed, consistently wound mesh)."""
        v0, v1, v2 = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        tet_vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
        tet_centroid = (v0 + v1 + v2) / 4.0
        total = tet_vol.sum()
        if abs(total) < 1e-12:
            return self.triangles.reshape(-1, 3).mean(axis=0)
        return (tet_centroid * tet_vol[:, None]).sum(axis=0) / total

    @property
    def bounds(self):
        pts = self.triangles.reshape(-1, 3)
        return pts.min(axis=0), pts.max(axis=0)

    @property
    def extents(self) -> np.ndarray:
        lo, hi = self.bounds
        return hi - lo

    # ---- overhang / support -----------------------------------------
    def overhang_mask(self, self_support_angle_deg: float) -> np.ndarray:
        """Boolean per-facet mask: downward-facing facets that need support.

        A facet's inclination is the angle of its surface from the horizontal
        build plate. Downward-facing facets (normal has a -Z component) whose
        inclination is below the self-support angle cannot be printed unsupported.
        """
        nz = self.face_normals[:, 2]
        downward = nz < -1e-9
        # inclination from horizontal = angle between the (downward) normal and -Z
        inclination = np.degrees(np.arccos(np.clip(-nz, -1.0, 1.0)))
        return downward & (inclination < self_support_angle_deg - 1e-9)

    def overhang_area_mm2(self, self_support_angle_deg: float) -> float:
        mask = self.overhang_mask(self_support_angle_deg)
        return float(self.face_areas[mask].sum())

    # ---- transforms ---------------------------------------------------
    def transformed(self, rotation: np.ndarray) -> "Mesh":
        """Return a new mesh with ``rotation`` (3x3) applied to every vertex."""
        rot = np.asarray(rotation, dtype=np.float64)
        return Mesh(self.triangles @ rot.T)

    def dropped_to_plate(self) -> "Mesh":
        """Translate so the mesh sits on z=0 and is centered in x, y."""
        lo, hi = self.bounds
        shift = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
        return Mesh(self.triangles - shift)

    # ---- identity -----------------------------------------------------
    @property
    def geometry_hash(self) -> str:
        """Stable SHA-256 over quantized vertices -- the part's design ID."""
        lo, hi = self.bounds
        diag = float(np.linalg.norm(hi - lo)) or 1.0
        quantized = np.round(self.triangles / (diag * 1e-6)).astype(np.int64)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:16]


def weld_vertices(mesh: Mesh, tol_frac: float = 1e-6):
    """Weld coincident vertices to recover topology from the triangle soup.

    Returns ``(unique_vertices, faces)`` where ``faces`` is an ``(N, 3)`` array
    of indices into ``unique_vertices``. STL stores every triangle's vertices
    independently, so shared vertices must be re-identified by position before
    any topological reasoning (like the watertight check) is possible.
    """
    lo, hi = mesh.bounds
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    tol = diag * tol_frac
    pts = mesh.triangles.reshape(-1, 3)
    keys = np.round(pts / tol).astype(np.int64)
    _, first_idx, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    unique_vertices = pts[first_idx]
    faces = inverse.reshape(-1, 3)
    return unique_vertices, faces


def watertight_report(mesh: Mesh, tol_frac: float = 1e-6) -> Dict[str, object]:
    """Check manifoldness: every edge should be shared by exactly two facets."""
    _, faces = weld_vertices(mesh, tol_frac=tol_frac)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())      # open edges
    nonmanifold = int((counts > 2).sum())     # >2 facets share an edge
    return {
        "is_watertight": boundary == 0 and nonmanifold == 0,
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "total_edges": int(counts.shape[0]),
    }


def axis_aligned_rotation(axis_to_up: int, sign: int) -> np.ndarray:
    """Rotation matrix that maps the chosen +/- principal axis onto +Z.

    ``axis_to_up`` in {0, 1, 2} selects X, Y or Z; ``sign`` in {+1, -1} picks
    which face points up. This is the candidate set for the 6 axis-aligned
    "lay the part on a flat face" orientations used to seed the DoE search.
    """
    src = np.zeros(3)
    src[axis_to_up] = sign
    dst = np.array([0.0, 0.0, 1.0])
    return _rotation_between(src, dst)


def rotation_z(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation about an arbitrary unit axis."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    x, y, z = axis
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def _rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking unit vector ``src`` to unit vector ``dst``."""
    src = src / (np.linalg.norm(src) or 1.0)
    dst = dst / (np.linalg.norm(dst) or 1.0)
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: rotate 180 deg about any axis perpendicular to src
        perp = np.array([1.0, 0.0, 0.0])
        if abs(src[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        return rotation_about_axis(np.cross(src, perp), 180.0)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
