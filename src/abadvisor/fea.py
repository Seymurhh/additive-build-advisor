"""Voxel finite-element analysis for build distortion (inherent-strain method).

This is the scientific core of the warpage prediction. It is a small but real
3D linear-elastic finite-element solver built directly on the voxel grid:

* each occupied voxel becomes an 8-node trilinear hexahedral element (24 DOF);
* the **inherent-strain method** models the accumulated thermal shrinkage of the
  build as a uniform eigenstrain applied to every element;
* the base layer is clamped to the build plate;
* the equilibrium system ``K u = f`` is solved matrix-free with a
  Jacobi-preconditioned conjugate gradient, so no global sparse matrix is ever
  assembled (the element operator is identical on a regular grid).

The resulting displacement field ``u`` is the predicted distortion. Its peak
magnitude is the warpage estimate; clamping the base while the bulk shrinks
reproduces the corner-lift that drives real additive distortion.

This is the standard reduced-order approach used by tools like Netfabb and ANSYS
Additive. It is *not* a calibrated transient thermo-mechanical solve: the
eigenstrain is a representative per-process value, not fit to melt-pool history.
But it is a genuine FEA — validated in ``tests`` against the analytical
clamped-bar solution and for mesh convergence. See REPORT.md.

Note: for a pure eigenstrain load with no external force, the *displacement*
field is independent of Young's modulus (it cancels between ``K`` and ``f``), so
distortion is governed by geometry, eigenstrain, and Poisson's ratio. Young's
modulus only enters the stress field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

_GAUSS = 1.0 / np.sqrt(3.0)
# Local node offsets (unit cube), standard hex ordering.
_NODE_OFFSETS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)


def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """Isotropic 6x6 constitutive matrix (Voigt: xx,yy,zz,xy,yz,zx)."""
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    g = (1.0 - 2.0 * nu) / 2.0
    D = np.zeros((6, 6))
    D[:3, :3] = c * np.array([[1 - nu, nu, nu], [nu, 1 - nu, nu], [nu, nu, 1 - nu]])
    D[3, 3] = D[4, 4] = D[5, 5] = c * g
    return D


def _B_at(xi: float, eta: float, zeta: float, h: float) -> np.ndarray:
    """Strain-displacement matrix B (6x24) for a cube element of side h."""
    s = 2.0 * _NODE_OFFSETS - 1.0  # natural-coord signs per node, in {-1,+1}
    nat = np.array([xi, eta, zeta])
    # shape-function derivatives wrt natural coords, then scale to physical (2/h)
    dN = np.zeros((8, 3))
    for a in range(8):
        sx, sy, sz = s[a]
        dN[a, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta)
        dN[a, 1] = 0.125 * sy * (1 + sx * xi) * (1 + sz * zeta)
        dN[a, 2] = 0.125 * sz * (1 + sx * xi) * (1 + sy * eta)
    dN *= 2.0 / h
    B = np.zeros((6, 24))
    for a in range(8):
        bx, by, bz = dN[a]
        c = 3 * a
        B[0, c + 0] = bx
        B[1, c + 1] = by
        B[2, c + 2] = bz
        B[3, c + 0] = by; B[3, c + 1] = bx
        B[4, c + 1] = bz; B[4, c + 2] = by
        B[5, c + 0] = bz; B[5, c + 2] = bx
    return B


def hex8_KE_and_eigenload(E: float, nu: float, h: float, eigenstrain: np.ndarray):
    """Return (KE 24x24, f_eigen 24) for a cube element under uniform eigenstrain."""
    D = elasticity_matrix(E, nu)
    detJ = (h / 2.0) ** 3
    KE = np.zeros((24, 24))
    fE = np.zeros(24)
    De = D @ eigenstrain
    for xi in (-_GAUSS, _GAUSS):
        for eta in (-_GAUSS, _GAUSS):
            for zeta in (-_GAUSS, _GAUSS):
                B = _B_at(xi, eta, zeta, h)
                KE += detJ * (B.T @ D @ B)
                fE += detJ * (B.T @ De)
    return KE, fE


@dataclass
class FEAResult:
    max_displacement_mm: float
    mean_displacement_mm: float
    converged: bool
    iterations: int
    n_elements: int
    n_dof: int
    eigenstrain: float
    # node-grid displacement magnitude, shape (nx+1, ny+1, nz+1); kept out of repr
    disp_grid: np.ndarray = field(repr=False, default=None)
    node_shape: Tuple[int, int, int] = (0, 0, 0)
    pitch: float = 0.0
    peak_von_mises_mpa: Optional[float] = None


def solve_inherent_strain(
    occ: np.ndarray,
    pitch: float,
    E: float,
    nu: float,
    eigenstrain: float,
    clamp_base: bool = True,
    max_iter: int = 4000,
    tol: float = 1e-6,
    compute_stress: bool = False,
) -> FEAResult:
    """Solve for build distortion of the occupancy grid under uniform eigenstrain.

    ``eigenstrain`` is the (negative) isotropic shrinkage applied to every
    element. Base nodes (z-index 0) are clamped to the plate. Solved matrix-free
    with Jacobi-preconditioned CG.
    """
    nx, ny, nz = occ.shape
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    n_nodes = nnx * nny * nnz
    n_dof = 3 * n_nodes

    eps_vec = np.array([eigenstrain, eigenstrain, eigenstrain, 0.0, 0.0, 0.0])
    KE, fE = hex8_KE_and_eigenload(E, nu, pitch, eps_vec)

    elem_idx = np.argwhere(occ)  # (n_el, 3) active voxel coords (i,j,k)
    n_el = elem_idx.shape[0]
    if n_el == 0:
        return FEAResult(0.0, 0.0, True, 0, 0, 0, eigenstrain,
                         np.zeros((nnx, nny, nnz)), (nnx, nny, nnz), pitch)

    # node id for each element's 8 local nodes -> edof (n_el, 24)
    def node_id(i, j, k):
        return i + j * nnx + k * nnx * nny

    edof = np.zeros((n_el, 24), dtype=np.int64)
    off = _NODE_OFFSETS.astype(int)
    for a in range(8):
        di, dj, dk = off[a]
        nid = node_id(elem_idx[:, 0] + di, elem_idx[:, 1] + dj, elem_idx[:, 2] + dk)
        edof[:, 3 * a] = 3 * nid
        edof[:, 3 * a + 1] = 3 * nid + 1
        edof[:, 3 * a + 2] = 3 * nid + 2

    # global load vector (assemble eigenstrain load)
    f = np.bincount(edof.ravel(), weights=np.tile(fE, n_el), minlength=n_dof)

    # which DOFs are active (touched by an element)
    active = np.zeros(n_dof, dtype=bool)
    active[np.unique(edof)] = True

    # clamp base nodes (k == 0) that are active
    fixed = np.zeros(n_dof, dtype=bool)
    if clamp_base:
        active_nodes = np.unique(edof) // 3
        base_nodes = active_nodes[(active_nodes // (nnx * nny)) == 0]
        for d in range(3):
            fixed[3 * base_nodes + d] = True
    free = active & ~fixed

    # matrix-free K*u over active elements
    KE_flat = KE

    def matvec(u):
        ue = u[edof]                       # (n_el, 24)
        fe = ue @ KE_flat.T                # (n_el, 24)
        out = np.bincount(edof.ravel(), weights=fe.ravel(), minlength=n_dof)
        out[~free] = 0.0
        return out

    # Jacobi preconditioner: diagonal of K
    diagKE = np.diag(KE)
    diagK = np.bincount(edof.ravel(), weights=np.tile(diagKE, n_el), minlength=n_dof)
    diagK[~free] = 1.0
    Minv = 1.0 / diagK

    # preconditioned CG on free DOFs
    b = f.copy()
    b[~free] = 0.0
    u = np.zeros(n_dof)
    r = b - matvec(u)
    r[~free] = 0.0
    z = Minv * r
    p = z.copy()
    rz = float(r @ z)
    bnorm = float(np.linalg.norm(b)) or 1.0
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        Ap = matvec(p)
        denom = float(p @ Ap) or 1e-30
        alpha = rz / denom
        u += alpha * p
        r -= alpha * Ap
        r[~free] = 0.0
        if np.linalg.norm(r) / bnorm < tol:
            converged = True
            break
        z = Minv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new

    disp = u.reshape(n_nodes, 3)
    mag = np.linalg.norm(disp, axis=1)
    # zero-out inactive nodes for clean stats/plots
    active_node_mask = np.zeros(n_nodes, dtype=bool)
    active_node_mask[np.unique(edof) // 3] = True
    mag_active = mag[active_node_mask]
    disp_grid = mag.reshape(nnx, nny, nnz)

    peak_vm = None
    if compute_stress:
        peak_vm = _peak_von_mises(u, edof, elem_idx, pitch, E, nu, eps_vec)

    return FEAResult(
        max_displacement_mm=float(mag_active.max()) if mag_active.size else 0.0,
        mean_displacement_mm=float(mag_active.mean()) if mag_active.size else 0.0,
        converged=converged,
        iterations=it,
        n_elements=n_el,
        n_dof=int(free.sum()),
        eigenstrain=eigenstrain,
        disp_grid=disp_grid,
        node_shape=(nnx, nny, nnz),
        pitch=pitch,
        peak_von_mises_mpa=peak_vm,
    )


def _peak_von_mises(u, edof, elem_idx, h, E, nu, eps_vec) -> float:
    """Peak element-centroid von Mises stress (mechanical strain = total - eigen)."""
    D = elasticity_matrix(E, nu)
    B0 = _B_at(0.0, 0.0, 0.0, h)
    ue = u[edof]                       # (n_el, 24)
    total = ue @ B0.T                  # (n_el, 6) strain at centroid
    mech = total - eps_vec             # subtract eigenstrain
    stress = mech @ D.T                # (n_el, 6) Voigt stress
    sx, sy, sz, txy, tyz, tzx = stress.T
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                 + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))
    return float(vm.max()) if vm.size else 0.0
