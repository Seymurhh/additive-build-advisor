"""Parametric mesh primitives for generating test/sample parts.

These build clean, watertight triangle meshes from scratch: a 2D profile is
triangulated by ear-clipping and extruded into a prism, and a few named helpers
compose recognizable parts (a calibration cube, a gantry bracket with a top-
flange overhang, a hollow housing with a trapped cavity, a tall standoff). They
exist so the repo is self-contained -- the example runner and the tests
synthesize their own STL inputs rather than depending on external CAD files.

The meshes are deliberately simple but exercise every DfAM check: overhangs,
thin walls, high aspect ratio, and enclosed voids.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .geometry import Mesh


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_tri(p, a, b, c) -> bool:
    v0, v1, v2 = c - a, b - a, p - a
    d00, d01, d02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    d11, d12 = np.dot(v1, v1), np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return False
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    return (u >= -1e-9) and (v >= -1e-9) and (u + v <= 1 + 1e-9)


def _triangulate(poly: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon (handles non-convex)."""
    poly = np.asarray(poly, dtype=np.float64)
    idx = list(range(len(poly)))
    if _signed_area(poly) < 0:          # work with CCW order
        idx.reverse()
    tris: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 100000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:              # reflex or collinear -> not an ear
                continue
            if any(_point_in_tri(poly[p], a, b, c) for p in idx if p not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            break
        else:
            break  # no ear found (degenerate); stop
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def fix_winding(mesh: Mesh) -> Mesh:
    """Flip all facets if the signed volume is negative, so normals face out."""
    if mesh.signed_volume_mm3 < 0:
        flipped = mesh.triangles[:, [0, 2, 1], :].copy()
        return Mesh(flipped)
    return mesh


def extrude_polygon(profile: Sequence[Tuple[float, float]], height: float, base_z: float = 0.0) -> Mesh:
    """Extrude a 2D profile (in the x-y plane) into a prism along +Z."""
    poly = np.asarray(profile, dtype=np.float64)
    if _signed_area(poly) < 0:
        poly = poly[::-1]
    tris2d = _triangulate(poly)
    n = len(poly)
    bottom = np.column_stack([poly, np.full(n, base_z)])
    top = np.column_stack([poly, np.full(n, base_z + height)])

    faces: List[np.ndarray] = []
    for (a, b, c) in tris2d:                      # bottom cap (normal -Z)
        faces.append(np.array([bottom[a], bottom[c], bottom[b]]))
    for (a, b, c) in tris2d:                      # top cap (normal +Z)
        faces.append(np.array([top[a], top[b], top[c]]))
    for i in range(n):                            # side walls
        j = (i + 1) % n
        faces.append(np.array([bottom[i], bottom[j], top[j]]))
        faces.append(np.array([bottom[i], top[j], top[i]]))
    return fix_winding(Mesh(np.array(faces)))


def cube(size: float = 20.0) -> Mesh:
    s = size
    return extrude_polygon([(0, 0), (s, 0), (s, s), (0, s)], s)


def cylinder(radius: float = 8.0, height: float = 40.0, segments: int = 48) -> Mesh:
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    profile = [(radius * np.cos(a), radius * np.sin(a)) for a in ang]
    return extrude_polygon(profile, height)


def gantry_bracket(wall: float = 4.0, height: float = 40.0, flange: float = 36.0, depth: float = 30.0) -> Mesh:
    """A Gamma-profile bracket: a vertical wall with a top flange.

    Modeled flange-up, so the flange underside is a flat overhang -- the
    orientation DoE should rotate it away. Profile is non-convex.
    """
    profile = [
        (0.0, 0.0), (wall, 0.0), (wall, height - wall),
        (flange, height - wall), (flange, height), (0.0, height),
    ]
    # extrude along Z gives the profile in x-z; rotate so depth runs along Y and
    # build height runs along Z by swapping axes after extrusion.
    m = extrude_polygon(profile, depth)
    # m currently: profile in x-y, extruded in z. Map (x,y,z)->(x,z,y) so the
    # profile's "height" axis becomes the build Z and the extrusion becomes Y.
    t = m.triangles.copy()
    t = t[:, :, [0, 2, 1]]
    return fix_winding(Mesh(t))


def tall_standoff(radius: float = 4.0, height: float = 70.0, segments: int = 32) -> Mesh:
    """A tall, slender cylinder -- exercises the aspect-ratio check."""
    return cylinder(radius=radius, height=height, segments=segments)


def cantilever_benchmark(length: float = 75.0, width: float = 12.0, height: float = 6.0) -> Mesh:
    """A long, thin, large-footprint bar -- the classic warp-prone geometry.

    A flat slender beam built on the plate is the worst case for cooling warpage:
    the corners lift as the part contracts. Used to exercise the distortion FEA on
    a geometry that warps a lot. Run on a metal profile it echoes the NIST AM-Bench
    2018 single-cantilever inherent-strain benchmark, which is why it keeps that
    name (see REPORT.md).
    """
    return extrude_polygon([(0, 0), (length, 0), (length, width), (0, width)], height)


def hollow_housing(outer: float = 30.0, wall: float = 3.0, height: float = 24.0) -> Mesh:
    """A closed box with a fully enclosed internal cavity (trapped volume)."""
    o = outer
    outer_box = extrude_polygon([(0, 0), (o, 0), (o, o), (0, o)], height)
    iw = wall
    inner_box = extrude_polygon(
        [(iw, iw), (o - iw, iw), (o - iw, o - iw), (iw, o - iw)],
        height - 2 * iw, base_z=iw,
    )
    # invert the inner shell so its normals face into the cavity
    inner_t = inner_box.triangles[:, [0, 2, 1], :]
    return Mesh(np.concatenate([outer_box.triangles, inner_t]))
