"""STL read/write, written from scratch (binary + ASCII).

STL stores a soup of independent triangles. We parse them into:

* ``triangles`` -- float array, shape ``(N, 3, 3)`` = N triangles x 3 vertices x (x, y, z)
* ``file_normals`` -- float array, shape ``(N, 3)`` = the normals stored in the file

The file normals are kept for reference only. Many exporters write zero or
inconsistent normals, so :mod:`abadvisor.geometry` recomputes them from the
vertex winding rather than trusting the file. Treating the file as untrusted
input is the realistic posture for a manufacturing pipeline.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# Binary STL record: 3 floats normal + 9 floats vertices + 1 uint16 attribute.
_BINARY_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
)
_BINARY_RECORD_BYTES = 50
_BINARY_HEADER_BYTES = 84  # 80-byte header + uint32 triangle count


def _looks_binary(raw: bytes) -> bool:
    """Decide binary vs ASCII by checking the declared triangle count.

    The ``solid`` keyword is not a reliable discriminator: many binary files
    begin with the ASCII bytes ``solid`` in their 80-byte header. The robust
    test is whether the file size matches the size implied by the triangle
    count stored at byte 80.
    """
    if len(raw) < _BINARY_HEADER_BYTES:
        return False
    (n_tri,) = struct.unpack_from("<I", raw, 80)
    expected = _BINARY_HEADER_BYTES + n_tri * _BINARY_RECORD_BYTES
    return len(raw) == expected


def _read_binary(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    (n_tri,) = struct.unpack_from("<I", raw, 80)
    records = np.frombuffer(
        raw, dtype=_BINARY_DTYPE, count=n_tri, offset=_BINARY_HEADER_BYTES
    )
    triangles = records["vertices"].astype(np.float64)
    normals = records["normal"].astype(np.float64)
    return triangles, normals


def _read_ascii(text: str) -> Tuple[np.ndarray, np.ndarray]:
    normals = []
    tris = []
    current = []
    current_normal = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0].lower()
        if tag == "facet" and len(parts) >= 5 and parts[1].lower() == "normal":
            current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
        elif tag == "vertex" and len(parts) >= 4:
            current.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "endfacet":
            if len(current) == 3:
                tris.append(current)
                normals.append(current_normal if current_normal else [0.0, 0.0, 0.0])
            current = []
            current_normal = None
    if not tris:
        raise ValueError("No triangles parsed from ASCII STL.")
    return np.asarray(tris, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def read_stl(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Read an STL file (binary or ASCII, auto-detected).

    Returns ``(triangles, file_normals)`` with shapes ``(N, 3, 3)`` and
    ``(N, 3)``.
    """
    raw = Path(path).read_bytes()
    if _looks_binary(raw):
        return _read_binary(raw)
    return _read_ascii(raw.decode("ascii", errors="replace"))


def write_stl_binary(
    path: PathLike, triangles: np.ndarray, normals: np.ndarray, header: str = ""
) -> None:
    """Write a binary STL. ``triangles`` is ``(N, 3, 3)``; ``normals`` is ``(N, 3)``."""
    triangles = np.asarray(triangles, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    n_tri = triangles.shape[0]
    records = np.zeros(n_tri, dtype=_BINARY_DTYPE)
    records["normal"] = normals
    records["vertices"] = triangles
    head = header.encode("ascii", errors="replace")[:80].ljust(80, b"\x00")
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(struct.pack("<I", n_tri))
        fh.write(records.tobytes())


def write_stl_ascii(
    path: PathLike, triangles: np.ndarray, normals: np.ndarray, name: str = "part"
) -> None:
    """Write an ASCII STL (handy for eyeballing small sample meshes)."""
    triangles = np.asarray(triangles, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    lines = [f"solid {name}"]
    for tri, nrm in zip(triangles, normals):
        lines.append(f"  facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}")
        lines.append("    outer loop")
        for v in tri:
            lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    Path(path).write_text("\n".join(lines) + "\n")
