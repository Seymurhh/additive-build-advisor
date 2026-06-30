"""Build-distortion finite-element analysis (thermal-contraction warping).

This is the scientific core of the warpage prediction. It assembles and solves a
3D linear-elastic finite-element problem with an established FEM library
(`scikit-fem <https://scikit-fem.readthedocs.io>`_) rather than a hand-rolled
solver:

* the occupied voxels are turned into a hexahedral mesh (``MeshHex``), restricted
  to the part (trilinear ``ElementHex1`` vector elements — good for bending);
* linear elasticity is assembled with ``skfem.models.elasticity``;
* the part's accumulated cooling shrinkage is modeled as a uniform thermal
  **eigenstrain** ``eps* ~ -alpha*dT``, entering as the consistent load
  ``f = integral( sigma0 : sym_grad(v) )`` with ``sigma0 = (3*lam+2*mu)*eps*``;
* the first layer is clamped to the plate (in FFF this is the bed-adhesion
  constraint);
* the sparse system ``K u = f`` is solved with SciPy's sparse direct solver.

The resulting displacement field is the predicted distortion; we also recover the
element von Mises stress. Clamping the base while the bulk contracts reproduces
the **corner-lift / warping** that is the dominant geometric defect in FFF: as
each road cools from the extrusion temperature it shrinks, the already-solid
material below resists it, residual stress builds, and the part curls up off the
bed at its corners (worse for large flat footprints, and far worse for ABS than
PLA).

Process focus and prior art
---------------------------
The home process here is **fused filament fabrication (FFF)** — it is what my
ES 51 (Computer-Aided Machine Design) students print, and warping off the bed is
the failure mode they actually hit. The reduced-order recipe (lump the cooling
history into one effective contraction strain and apply it as a static eigenstrain
load to a part-scale elastic FEA) is the same machinery that, applied to metal
powder-bed fusion, is called the **inherent-strain method** (what Netfabb / ANSYS
Additive implement; review in *Int. J. Adv. Manuf. Technol.*, 2022). So the
solver runs unchanged on the metal profiles too, as a point of comparison.

Scope honesty: this is a *simplified* model — a representative isotropic
contraction strain (not a tensor fit to a measured cooling history), applied to
the whole part at once, with the base bonded to the bed. So the reported
distortion is the *on-bed* field (a relative warpage screen), not the
post-removal spring-back after the part is released from the bed. It is validated
against the analytical clamped-bar solution in ``tests``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skfem import (Basis, ElementHex1, ElementVector, LinearForm, MeshHex,
                   asm, condense, solve)
from skfem.helpers import div
from skfem.models.elasticity import lame_parameters, linear_elasticity


@dataclass
class FEAResult:
    max_displacement_mm: float
    mean_displacement_mm: float
    peak_von_mises_mpa: Optional[float]
    n_elements: int
    n_dof: int
    converged: bool
    solver: str
    eigenstrain: float
    # mesh + nodal fields for plotting (kept out of repr)
    nodes: np.ndarray = field(repr=False, default=None)        # (3, N)
    quads: np.ndarray = field(repr=False, default=None)        # (4, M) boundary facets
    u_nodal: np.ndarray = field(repr=False, default=None)      # (3, N)
    mag_nodal: np.ndarray = field(repr=False, default=None)    # (N,)
    vm_nodal: np.ndarray = field(repr=False, default=None)     # (N,)
    pitch: float = 0.0

    @property
    def iterations(self) -> int:  # back-compat with the record schema
        return 0


def _empty_result(eigenstrain: float, pitch: float) -> FEAResult:
    return FEAResult(0.0, 0.0, 0.0, 0, 0, True, "scipy sparse (direct)", eigenstrain,
                     np.zeros((3, 0)), np.zeros((4, 0), dtype=int),
                     np.zeros((3, 0)), np.zeros(0), np.zeros(0), pitch)


def _hex_mesh_from_voxels(occ: np.ndarray, pitch: float) -> MeshHex:
    """Build a hexahedral mesh of the occupied voxels (skfem node ordering)."""
    nx, ny, nz = occ.shape
    xs = np.arange(nx + 1) * pitch
    ys = np.arange(ny + 1) * pitch
    zs = np.arange(nz + 1) * pitch
    full = MeshHex.init_tensor(xs, ys, zs)
    centroids = full.p[:, full.t].mean(axis=1)          # (3, n_elem)
    vi = np.clip(np.floor(centroids[0] / pitch).astype(int), 0, nx - 1)
    vj = np.clip(np.floor(centroids[1] / pitch).astype(int), 0, ny - 1)
    vk = np.clip(np.floor(centroids[2] / pitch).astype(int), 0, nz - 1)
    keep = occ[vi, vj, vk]
    t = full.t[:, keep]
    used, inv = np.unique(t, return_inverse=True)
    t2 = np.ascontiguousarray(inv.reshape(t.shape))
    p2 = np.ascontiguousarray(full.p[:, used])
    return MeshHex(p2, t2)


def solve_thermal_warp(
    occ: np.ndarray,
    pitch: float,
    E: float,
    nu: float,
    eigenstrain: float,
    clamp_base: bool = True,
    **_ignored,
) -> FEAResult:
    """Solve for build distortion under a uniform thermal-contraction eigenstrain.

    ``eigenstrain`` is the (negative) isotropic cooling contraction applied to
    every element (``eps* ~ -alpha*dT``). Base nodes (z ~ 0) are clamped to the
    plate -- the bed-adhesion constraint in FFF. Assembled with scikit-fem,
    solved with SciPy's sparse direct solver. Applied to a metal profile the same
    solve is the inherent-strain method.
    """
    if occ.sum() == 0:
        return _empty_result(eigenstrain, pitch)

    mesh = _hex_mesh_from_voxels(occ, pitch)
    basis = Basis(mesh, ElementVector(ElementHex1()))
    lam, mu = lame_parameters(E, nu)

    K = asm(linear_elasticity(lam, mu), basis)
    sigma0 = (3.0 * lam + 2.0 * mu) * eigenstrain  # hydrostatic eigenstress

    @LinearForm
    def eigenload(v, w):
        return sigma0 * div(v)

    f = asm(eigenload, basis)

    if clamp_base:
        clamped = basis.get_dofs(lambda x: x[2] < pitch * 0.4)
    else:  # fall back to a single corner to remove rigid-body modes
        clamped = basis.get_dofs(lambda x: (x[0] < pitch * 0.4) & (x[1] < pitch * 0.4) & (x[2] < pitch * 0.4))

    u = solve(*condense(K, f, D=clamped))

    nodes = mesh.p
    n_nodes = nodes.shape[1]
    u_nodal = u[basis.nodal_dofs]                       # (3, N)
    mag = np.linalg.norm(u_nodal, axis=0)               # (N,)

    # element von Mises at quadrature points, mechanical strain = total - eigen
    wu = basis.interpolate(u)
    G = wu.grad                                         # (3, 3, n_elem, n_qp)
    eps = 0.5 * (G + G.transpose(1, 0, 2, 3))
    epsm = eps.copy()
    for d in range(3):
        epsm[d, d] = epsm[d, d] - eigenstrain
    trm = epsm[0, 0] + epsm[1, 1] + epsm[2, 2]
    sig = 2.0 * mu * epsm
    for d in range(3):
        sig[d, d] = sig[d, d] + lam * trm
    sx, sy, sz = sig[0, 0], sig[1, 1], sig[2, 2]
    txy, tyz, tzx = sig[0, 1], sig[1, 2], sig[2, 0]
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                 + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))
    vm_elem = vm.mean(axis=1)                           # (n_elem,)

    # scatter element von Mises to nodes (average) for a smooth contour
    t = mesh.t                                          # (8, n_elem)
    vm_nodal = np.zeros(n_nodes)
    counts = np.zeros(n_nodes)
    np.add.at(vm_nodal, t.T.ravel(), np.repeat(vm_elem, t.shape[0]))
    np.add.at(counts, t.T.ravel(), 1.0)
    vm_nodal /= np.maximum(counts, 1.0)

    quads = mesh.facets[:, mesh.boundary_facets()]      # (4, M)

    return FEAResult(
        max_displacement_mm=float(mag.max()),
        mean_displacement_mm=float(mag.mean()),
        peak_von_mises_mpa=float(vm_elem.max()),
        n_elements=int(mesh.t.shape[1]),
        n_dof=int(basis.N),
        converged=True,
        solver="scikit-fem assembly + SciPy sparse direct",
        eigenstrain=eigenstrain,
        nodes=nodes,
        quads=quads,
        u_nodal=u_nodal,
        mag_nodal=mag,
        vm_nodal=vm_nodal,
        pitch=pitch,
    )


# Back-compat alias: the same eigenstrain solve is the "inherent-strain method"
# when it is run on a metal powder-bed-fusion profile.
solve_inherent_strain = solve_thermal_warp
