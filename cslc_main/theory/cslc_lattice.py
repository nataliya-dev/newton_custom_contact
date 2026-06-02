# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-sphere lattice extension of ``cslc_main.theory.cslc_theory``.

Adds the lateral springs that couple a sphere to its neighbours, the
lattice-scale quadratic form (the lattice stiffness matrix ``K``), and
the unified contact solver :func:`solve_lattice_contact` for a pad
:class:`Lattice` against a
:class:`cslc_main.theory.cslc_targets.PointSetTarget` (theory.md §6).

Lateral law: graph-Laplacian (theory.md §6.2)::

    f_lat(i, j) = −k_l · (δ_i − δ_j)

Isotropic in 3D; equivalent quadratic energy
``E = ½·k_l·Σ_edges ||δ_i − δ_j||²``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from .cslc_theory import EPS_ALIGN_DEFAULT, INACTIVE_RAW_EPS_FACTOR


LateralLaw = Literal["graph_laplacian"]


# ────────────────────────────────────────────────────────────────────────
#  Lattice description
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Lattice:
    """A connected lattice of CSLC spheres.

    All quantities are body-local (or world; rigid-body transforms
    preserve them).

    Attributes:
        p: (N, 3) float64 — rest positions [m].
        n: (N, 3) float64 — rest outward unit normals.
        edges: (E, 2) int64 — undirected edge list ``(i, j)`` with
            ``i < j``.  Each unordered edge appears once; force
            assembly handles both endpoints.
        ka: anchor stiffness [N/m].
        kl: lateral stiffness [N/m].

    The neighbour count of sphere ``i`` is the number of edges
    containing ``i``; computed lazily by :meth:`neighbour_counts`.
    """

    p: np.ndarray
    n: np.ndarray
    edges: np.ndarray
    ka: float
    kl: float

    def __post_init__(self):
        if self.p.ndim != 2 or self.p.shape[1] != 3:
            raise ValueError(f"p must be (N, 3), got {self.p.shape}")
        if self.n.shape != self.p.shape:
            raise ValueError(f"n must match p shape, got {self.n.shape}")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must be (E, 2), got {self.edges.shape}")
        if self.edges.size and self.edges.max() >= len(self.p):
            raise ValueError(
                f"edge index out of range: max(edges)={self.edges.max()} "
                f"vs N={len(self.p)}")

    @property
    def N(self) -> int:
        return len(self.p)

    @property
    def E(self) -> int:
        return len(self.edges)

    def neighbour_counts(self) -> np.ndarray:
        """Number of neighbours per sphere (degree in the edge graph).

        ``K_ii = ka + kl * |N(i)|`` uses this directly.
        """
        counts = np.zeros(self.N, dtype=np.int64)
        for (i, j) in self.edges:
            counts[i] += 1
            counts[j] += 1
        return counts

    def rest_lengths(self) -> np.ndarray:
        """L_ij = ||p_j - p_i|| for each edge."""
        L = np.zeros(self.E, dtype=np.float64)
        for k, (i, j) in enumerate(self.edges):
            L[k] = float(np.linalg.norm(self.p[j] - self.p[i]))
        return L

    def rest_directions(self) -> np.ndarray:
        """Unit vectors e_hat_ij^rest = (p_j - p_i) / L_ij, shape (E, 3)."""
        e = np.zeros((self.E, 3), dtype=np.float64)
        for k, (i, j) in enumerate(self.edges):
            d = self.p[j] - self.p[i]
            L = float(np.linalg.norm(d))
            if L < 1e-15:
                raise ValueError(
                    f"degenerate edge {(int(i), int(j))} with L = 0")
            e[k] = d / L
        return e


def make_chain(N: int, h: float, ka: float, kl: float,
               n: np.ndarray | None = None) -> Lattice:
    """Build a 1D chain of N spheres along x-axis with spacing h.

    Outward normal defaults to +y for every sphere (pad-like).  Edges
    are (i, i+1) for i in 0..N-2.
    """
    if n is None:
        n = np.array([0.0, 1.0, 0.0])
    p = np.zeros((N, 3))
    p[:, 0] = np.arange(N) * h
    n_arr = np.broadcast_to(n, (N, 3)).copy()
    edges = np.stack([np.arange(N - 1), np.arange(1, N)],
                     axis=1).astype(np.int64)
    return Lattice(p=p, n=n_arr, edges=edges, ka=ka, kl=kl)


def make_arc(N: int, R_pad: float, arc_length_spacing: float,
             ka: float, kl: float) -> Lattice:
    """Build a curved 1D chain on a circular arc of radius R_pad.

    N spheres laid out symmetrically about the apex (theta = pi/2,
    i.e. along +y).  Angular spacing is set so the arc-length spacing
    between adjacent spheres is ``arc_length_spacing``:
        delta_theta = arc_length_spacing / R_pad.

    Sphere positions (in the x-y plane, z = 0):
        theta_i = pi/2 + (i - i_centre) * delta_theta
        p_i     = R_pad * (cos theta_i, sin theta_i, 0)
        n_i     = (cos theta_i, sin theta_i, 0)        (radially outward)

    Edges connect nearest neighbours along the arc, same topology as
    :func:`make_chain`.
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    if R_pad <= 0.0:
        raise ValueError(f"R_pad must be > 0, got {R_pad}")
    i_centre = N // 2
    delta_theta = arc_length_spacing / R_pad
    thetas = np.pi / 2.0 + (np.arange(N) - i_centre) * delta_theta
    p = np.stack([R_pad * np.cos(thetas),
                  R_pad * np.sin(thetas),
                  np.zeros(N)], axis=1)
    n_arr = np.stack([np.cos(thetas),
                      np.sin(thetas),
                      np.zeros(N)], axis=1)
    edges = np.stack([np.arange(N - 1), np.arange(1, N)],
                     axis=1).astype(np.int64)
    return Lattice(p=p, n=n_arr, edges=edges, ka=ka, kl=kl)


def make_flat_grid(
    n_u: int, n_v: int, spacing: float, ka: float, kl: float,
    *, normal: np.ndarray | None = None,
    diagonals: bool = False,
) -> Lattice:
    """Build a flat 2D rectangular lattice in the x-y plane.

    Used for visualising and stress-testing the lattice equilibrium on
    a planar pad (the paper's ``test3_surface_deflection`` configuration
    and the canonical demo of the lateral-coupling Green's function).

    Sphere positions on a regular ``n_u × n_v`` grid::

        p_{i,j} = (i · spacing − (n_u−1)·spacing/2,
                   j · spacing − (n_v−1)·spacing/2,
                   0)

    centred on the origin.  All outward normals share the same direction
    ``normal`` (default ``+ẑ``).  Edges are 4-connected nearest
    neighbours along (u, v); pass ``diagonals=True`` to add the two
    diagonal edges per cell (8-connected).

    Args:
        n_u, n_v: grid dimensions (number of spheres along each axis).
            Both ≥ 2.
        spacing: nearest-neighbour distance [m].  Set ``r_pad = spacing/2``
            downstream to make the discs of radius r_pad tile the pad
            face exactly once (theory.md §3.5 tiling identity).
        ka: anchor stiffness [N/m].
        kl: lateral stiffness [N/m].
        normal: shared outward unit normal for every sphere.  Default +z.
        diagonals: include diagonal edges per cell.  Default False
            (4-connected); True gives 8-connected which weakly stiffens
            the lateral response but reduces the angular anisotropy of
            the Green's function.

    Returns:
        ``Lattice`` with ``N = n_u · n_v`` spheres.  Sphere index ordering
        is row-major in (u, v): ``idx = i·n_v + j``.
    """
    if n_u < 2 or n_v < 2:
        raise ValueError(f"n_u, n_v >= 2 required, got {n_u}, {n_v}")
    if spacing <= 0.0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / float(np.linalg.norm(normal))

    N = n_u * n_v
    i_idx, j_idx = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing="ij")
    x = (i_idx - (n_u - 1) / 2.0) * spacing
    y = (j_idx - (n_v - 1) / 2.0) * spacing
    p = np.stack([x.ravel(), y.ravel(), np.zeros(N)], axis=1)
    n_arr = np.broadcast_to(normal, (N, 3)).copy()

    edge_list: list[tuple[int, int]] = []
    def lin(i: int, j: int) -> int:
        return i * n_v + j

    for i in range(n_u):
        for j in range(n_v):
            if i + 1 < n_u:
                edge_list.append((lin(i, j), lin(i + 1, j)))
            if j + 1 < n_v:
                edge_list.append((lin(i, j), lin(i, j + 1)))
            if diagonals:
                if i + 1 < n_u and j + 1 < n_v:
                    edge_list.append((lin(i, j), lin(i + 1, j + 1)))
                if i + 1 < n_u and j - 1 >= 0:
                    edge_list.append((lin(i, j), lin(i + 1, j - 1)))
    edges = np.array(sorted({(min(a, b), max(a, b)) for (a, b) in edge_list}),
                     dtype=np.int64)
    return Lattice(p=p, n=n_arr, edges=edges, ka=ka, kl=kl)


def make_dome(N: int, R_pad: float, half_angle: float,
              ka: float, kl: float,
              k_neighbors: int = 6,
              ) -> tuple[Lattice, float, float]:
    """Build a 3D spherical-cap lattice via the Fibonacci-spiral sampler.

    Cap layout::

        z_min   = cos(half_angle)                       (cap base)
        z_i     = z_min + (1 − z_min)·(i + 0.5)/N        (equal area)
        r_xy_i  = sqrt(1 − z_i²)
        phi_i   = i · golden_angle    (golden_angle = π·(3 − sqrt(5)))
        p_i     = R_pad · (r_xy_i·cos(phi_i), r_xy_i·sin(phi_i), z_i)
        n_i     = p_i / R_pad                            (radial outward)

    The Fibonacci spiral with equal area per sample is quasi-uniform on
    the cap (variance of nearest-neighbour distance ≈ 5% of the mean
    on the cap interior).  Edges are unordered k-NN in 3D, which on a
    quasi-uniform 2D manifold approximates the surface Delaunay graph.
    Apex sits at theta = 0, i.e. ``(0, 0, R_pad)``.

    Args:
        N: number of lattice spheres.
        R_pad: dome radius [m].
        half_angle: cap half-angle [rad].
        ka: anchor stiffness [N/m].
        kl: lateral stiffness [N/m].
        k_neighbors: k for the k-NN neighbour graph (typical 6).

    Returns:
        ``(lat, spacing, cap_area)`` where ``spacing`` is the mean
        nearest-neighbour distance [m] and ``cap_area`` [m²] is the
        analytic spherical-cap area.
    """
    if R_pad <= 0.0 or half_angle <= 0.0 or half_angle >= np.pi:
        raise ValueError(
            f"R_pad > 0 and 0 < half_angle < pi required; got "
            f"R_pad={R_pad}, half_angle={half_angle}")
    if N < 7:
        raise ValueError(
            f"N >= 7 required for a meaningful 2D cap, got N={N}")

    z_min = float(np.cos(half_angle))
    indices = np.arange(N)
    z = z_min + (1.0 - z_min) * (indices + 0.5) / N
    r_xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    golden_angle = float(np.pi * (3.0 - np.sqrt(5.0)))
    phi = indices * golden_angle
    unit = np.stack([r_xy * np.cos(phi),
                     r_xy * np.sin(phi),
                     z], axis=1)
    # Move the densest sample (largest z, last index) to be the apex --
    # so that test drivers can target the apex sphere deterministically.
    # The Fibonacci-spiral apex is the LAST sample (i = N-1, z closest
    # to 1).  Roll it to index 0 for convenience.
    apex_idx = int(np.argmax(unit[:, 2]))
    perm = np.concatenate(([apex_idx], np.delete(np.arange(N), apex_idx)))
    unit = unit[perm]
    p = R_pad * unit
    n_arr = unit  # outward = radial; ||unit||_2 == 1 by construction

    # k-NN edge graph (undirected) on the 3D positions.  For a
    # quasi-uniform 2D manifold this is a close approximation of the
    # geodesic k-NN; we don't bother with the geodesic projection.
    from scipy.spatial import cKDTree
    tree = cKDTree(p)
    _, idx = tree.query(p, k=k_neighbors + 1)  # +1 to drop self
    edge_set: set[tuple[int, int]] = set()
    for i in range(N):
        for j in idx[i, 1:]:
            a, b = sorted((int(i), int(j)))
            if a != b:
                edge_set.add((a, b))
    edges = np.array(sorted(edge_set), dtype=np.int64)

    # Mean NN distance = lattice spacing (first neighbour column,
    # excluding self at column 0).
    nn_dists = np.linalg.norm(p[idx[:, 1]] - p, axis=1)
    spacing = float(np.mean(nn_dists))

    # Spherical-cap area: A = 2·π·R²·(1 − cos θ_max).
    cap_area = float(2.0 * np.pi * R_pad * R_pad * (1.0 - z_min))

    return Lattice(p=p, n=n_arr, edges=edges, ka=ka, kl=kl), spacing, cap_area


# ────────────────────────────────────────────────────────────────────────
#  Lattice stiffness matrix K (scalar form)
# ────────────────────────────────────────────────────────────────────────


def build_K_matrix(lat: Lattice) -> np.ndarray:
    """Assemble the scalar lattice stiffness ``K ∈ ℝ^{N×N}``.

    ``K_ii = ka + kl·|N(i)|``, ``K_ij = −kl`` if ``(i, j)`` is an edge,
    else 0.  In the vector setting the graph-Laplacian system is the
    Kronecker lift ``K_3D = K ⊗ I_3``, so each axis decouples and
    solves with the same scalar ``K``.

    Returns:
        Symmetric positive-definite array of shape ``(N, N)``.
    """
    N = lat.N
    K = np.zeros((N, N), dtype=np.float64)
    counts = lat.neighbour_counts()
    for i in range(N):
        K[i, i] = lat.ka + lat.kl * float(counts[i])
    for (i, j) in lat.edges:
        K[i, j] -= lat.kl
        K[j, i] -= lat.kl
    return K


def chain_discrete_decay_length(ka: float, kl: float) -> float:
    """Decay length of the chain's discrete Green's function, in spacings.

    For an infinite 1D chain with stiffness K (anchor ka, lateral kl),
    the Green's function g_i for a point source at i=0 is

        g_i = (g_0) * z_-^|i|

    where z_- is the smaller positive root of the characteristic
    equation k_l z^2 - (k_a + 2k_l) z + k_l = 0:

        z_- = (1 + alpha) - sqrt(alpha*(alpha + 2)),   alpha = k_a / (2*k_l).

    The decay length is

        l_c_discrete = -1 / ln(z_-).

    Limits:
        k_l >> k_a:  z_- -> 1 - sqrt(k_a/k_l), so l_c -> sqrt(k_l/k_a)
                     (the CONTINUUM Helmholtz decay length).
        k_l << k_a:  z_- -> k_l/k_a,            so l_c -> 1/ln(k_a/k_l).
                     The lattice cannot resolve a continuum decay
                     shorter than ~1 spacing, so it localises further.

    Use this — not the continuum approximation — as the reference for
    Green's-function fits on lattices where ``k_l/k_a`` is not ≫ 1.
    """
    if ka <= 0.0:
        raise ValueError(f"ka must be > 0, got {ka}")
    if kl <= 0.0:
        return 0.0
    alpha = ka / (2.0 * kl)
    z_minus = (1.0 + alpha) - np.sqrt(alpha * (alpha + 2.0))
    return float(-1.0 / np.log(z_minus))


def chain_analytical_eigenvalues(N: int, ka: float, kl: float) -> np.ndarray:
    """Eigenvalues of the chain's K, in ascending order.

    For a 1D chain of N spheres with free endpoints (degree-1 at the
    ends, degree-2 elsewhere), K is tridiagonal with the Neumann modes
    of a discrete Laplacian::

        λ_k = ka + 2·kl·(1 − cos(k·π/N)),    k = 0, 1, ..., N−1

    Reference: Strang, "Computational Science and Engineering" eq. 2.34
    (free-free 1D Laplacian DCT-II spectrum).  At ``k = 0`` we recover
    the uniform-translation mode ``λ_0 = ka`` — the only mode the
    lateral spring cannot resist.
    """
    k = np.arange(N, dtype=np.float64)
    return ka + 2.0 * kl * (1.0 - np.cos(k * np.pi / N))


# ────────────────────────────────────────────────────────────────────────
#  Forces and energies
# ────────────────────────────────────────────────────────────────────────


def anchor_force_all(lat: Lattice, deltas: np.ndarray) -> np.ndarray:
    """Per-sphere isotropic anchor force ``f_anchor = +ka · δ``.

    Args:
        deltas: ``(N, 3)`` displacements.
    Returns:
        ``(N, 3)`` anchor forces.
    """
    return lat.ka * deltas


def anchor_energy(lat: Lattice, deltas: np.ndarray) -> float:
    """``E_anchor = ½·ka·Σ_i ||δ_i||²``."""
    return 0.5 * lat.ka * float(np.sum(deltas * deltas))


def lateral_force_graph_laplacian(lat: Lattice,
                                  deltas: np.ndarray) -> np.ndarray:
    """Graph-Laplacian lateral force per sphere (theory.md §6.2)::

        f_lat_i = −k_l · Σ_{j ∈ N(i)} (δ_i − δ_j)

    Equivalent to ``-(kl · L) @ δ`` where ``L = D − A`` is the graph
    Laplacian (``D`` = degree, ``A`` = adjacency).  Quadratic and convex
    in ``δ``; with anchor it is the solution of
    ``K · δ_axis = f_ext_axis`` per axis independently (axes decouple).
    """
    f = np.zeros_like(deltas)
    for (i, j) in lat.edges:
        diff = deltas[i] - deltas[j]
        f[i] -= lat.kl * diff
        f[j] += lat.kl * diff
    return f


def lateral_energy_graph_laplacian(lat: Lattice, deltas: np.ndarray) -> float:
    """E_lat^GL = (1/2) * k_l * sum_edges ||delta_i - delta_j||^2."""
    e = 0.0
    for (i, j) in lat.edges:
        diff = deltas[i] - deltas[j]
        e += float(np.dot(diff, diff))
    return 0.5 * lat.kl * e


# ────────────────────────────────────────────────────────────────────────
#  Lattice contact solver  (theory.md §6)
#
#  Pad ``Lattice`` vs ``PointSetTarget``.  Half-space overlap, graph-
#  Laplacian lateral, anisotropic anchor via ``ka_t_ratio``.
#
#  :func:`solve_lattice_contact` warm-starts with L-BFGS-B on a
#  heuristic Lyapunov-like energy whose Jacobian is the force-form
#  contact term, then refines with damped Jacobi (theory.md §6.5–§6.6)
#  to reach the kernel's force fixed point.
# ────────────────────────────────────────────────────────────────────────


def solve_lattice_contact(
    lat: Lattice,
    target,                       # PointSetTarget (from cslc_targets)
    kc: float,
    *,
    r_pad: np.ndarray | float,
    delta0: np.ndarray | None = None,
    eps: float = 1.0e-9,
    eps_align: float = EPS_ALIGN_DEFAULT,
    ka_t_ratio: float = 1.0,
    tol: float = 1.0e-12,
    maxiter: int = 5000,
    kernel_half_width: np.ndarray | float | None = None,
) -> tuple[np.ndarray, dict]:
    """Lattice contact equilibrium for ``Lattice`` vs ``PointSetTarget``.

    Reaches the force-balance fixed point of theory.md §6.5::

        0 = ka·δ_n·n̂_pad + (ka·ρ)·δ_t                         (anchor §6.1)
            + kl·Σ_{j∈N(i)}(δ_i − δ_j)                         (lateral §6.2)
            + Σ_j kc·A_j·w_t_ij·a_ij·φ_eff_ij·gate_ij·n̂_face_j (contact §5)

    with the Hertz-like coupling::

        φ_eff = σ_ε(raw) · √(σ_ε(raw) + ε)        (theory.md eq:phi-eff)
        gate  = Σ_ε(raw)                          (theory.md eq:gate)
        w_t   = Σ_ε(r_i − d_t)                    (theory.md §3.5)

    Under the Hertz lift the kernel writes the force law directly
    (theory.md §4 "Note on energy form") — there is no globally-defined
    potential whose ``∂E/∂δ`` equals this force.  This solver
    warm-starts with L-BFGS-B on the heuristic Lyapunov-like energy
    ``½·kc·A·w·a·φ_eff²`` paired with the force-form Jacobian, then
    refines via damped Jacobi (theory.md §6.6) to the kernel's
    fixed point.  The Jacobi sweep matches
    :func:`cslc_kernels.jacobi_step` directly.

    **Anisotropic anchor (theory.md §6.1).**  In each pad sphere's
    local ``{n̂_pad_i, n̂_pad_i^⊥}`` frame::

        δ_n_i = δ_i · n̂_pad_i,   δ_t_i = δ_i − δ_n_i·n̂_pad_i
        E_anchor_i = ½·k_a·δ_n_i² + ½·(k_a·ρ)·||δ_t_i||²

    with ``ρ = ka_t_ratio``.  ``ρ = 1`` recovers the isotropic anchor.
    Since ``n̂_pad_i`` is rest body-local (δ-independent), the gradient
    picks up no extra ``∂n̂_pad/∂δ`` term::

        ∂E_anchor_i/∂δ = k_a·δ_n_i·n̂_pad_i + (k_a·ρ)·δ_t_i

    **w_t treated as δ-independent.**  The ``∂w_t/∂δ`` term is dropped
    so the theory-side gradient matches the kernel's truncated
    gradient exactly.  This makes ``jac`` an ``O(eps²/r³)`` inconsistent
    approximation of ``fun``, but the converged ``∂E/∂δ = 0`` point is
    the same smooth-surrogate equilibrium the kernel iterates to.

    **Smooth alignment gate ``a_ij``** (theory.md §3.6)::

        align_arg_ij = −(n̂_face_j · n̂_pad_i)      (>0 = opposing)
        a_ij         = smoothstep(align_arg_ij; 0, eps_align)

    C¹ cubic smoothstep on the one-sided band ``[0, +eps_align]``:
    exactly 1 for ``align_arg ≥ +eps_align`` (face-on), exactly 0 for
    ``align_arg ≤ 0`` (perpendicular or back-to-back, both hard-culled).
    δ-independent (``n_pad`` and ``n_face`` are rest body-local), so it
    enters the gradient as a constant multiplier.  Compact support
    means back-side and perpendicular samples contribute exactly zero,
    avoiding the polynomial tail that an open band would produce on
    closed convex targets.

    Args:
        lat: pad lattice (uses ``lat.p``, ``lat.n``, ``lat.edges``,
            ``lat.ka``, ``lat.kl``).
        target: a :class:`cslc_main.theory.cslc_targets.PointSetTarget`
            with ``positions`` ``(M, 3)``, ``normals`` ``(M, 3)``, and
            optional ``areas`` ``(M,)``.
        kc: contact stiffness in kernel units ``[Pa·m^(−1/2)]`` — the
            per-volume form ``kc = kc_per_sphere / (π·r_pad²)``
            (theory.md §10).  The CSLC handler does this rescale
            upstream; theory-side callers pass per-volume kc directly to
            match the kernel API.  Single-pair primitives in
            :mod:`cslc_theory` take per-sphere kc since they have no
            area or locality weights.
        r_pad: per-pad-sphere radius [m].  Scalar or shape ``(N,)``.
        delta0: warm start, shape ``(N, 3)``.  Default zeros.
        eps: smoothing width [m] for the half-space surrogate.  Default
             1e-9 (theory-side precision); production kernel uses 5e-4.
        eps_align: smoothing half-width for the alignment gate
            (dimensionless cosine units, in ``[0, 1]``).  Default 0.05
            (≈ 2.87° angular half-transition, one-sided from
            ``α = 0`` to ``α = +eps_align``).  Set to 0 for the binary
            hard step at ``n_face·n_pad = 0`` (ablation only — the
            hard step produces a force discontinuity).
        ka_t_ratio: anisotropic anchor ratio ``ρ = k_at / k_a``.  Default
            1.0 (isotropic).  ``Lattice`` does not carry this; passed
            here as a scalar so callers can sweep ``ρ`` without
            rebuilding the lattice.
        tol: L-BFGS-B ``gtol`` / ``ftol``.
        maxiter: L-BFGS-B iteration cap.
        kernel_half_width: tangential locality kernel half-width [m]
            (the ``r_i`` in theory.md eq:w_t §3.5).  Scalar or ``(N,)``
            or None.  None ⇒ ``r_pad`` — the spec value, matching the
            production kernel.  This makes the disc-of-radius-r tiling
            cover the contact patch exactly once when pads are
            CVT-sampled at spacing ``2·r_pad`` (theory.md §3.5 tiling
            identity).

    Returns:
        ``(deltas, info)`` — ``deltas`` is ``(N, 3)``; ``info`` is the
        scipy diagnostic dict augmented with ``n_active_pairs`` and
        ``jacobi_refine_iters``.
    """
    from scipy.optimize import minimize

    N = lat.N
    M = int(target.M)
    p_rest = np.asarray(lat.p, dtype=np.float64)              # (N, 3)
    # (N, 3) outward normals
    n_pad = np.asarray(lat.n, dtype=np.float64)
    edges = np.asarray(lat.edges, dtype=np.int64)             # (E, 2)
    ka = float(lat.ka)
    # anisotropic tangent
    ka_t = ka * float(ka_t_ratio)
    kl = float(lat.kl)
    t_pos = np.asarray(target.positions, dtype=np.float64)    # (M, 3)
    n_face = np.asarray(target.normals, dtype=np.float64)     # (M, 3)
    areas = (np.asarray(target.areas, dtype=np.float64)
             if target.areas is not None
             else np.ones(M, dtype=np.float64))

    # Smooth alignment gate (theory.md §3.6).  align_arg = -(n_face · n_pad)
    # is +1 face-on, -1 back-to-back, 0 perpendicular.  C¹ cubic
    # smoothstep on the one-sided band [0, +eps_align].  δ-independent
    # (n_pad and n_face are rest-frame); cached once.
    n_face_dot_n_pad = np.einsum("nj,mj->nm", n_pad, n_face)  # (N, M)
    align_arg = -n_face_dot_n_pad                             # (N, M)
    if eps_align > 0.0:
        t = np.clip(align_arg / eps_align, 0.0, 1.0)
        align_gate = t * t * (3.0 - 2.0 * t)                  # (N, M)
    else:
        # Hard binary step (perpendicular hard-culled).
        align_gate = (align_arg > 0.0).astype(np.float64)
    # Performance cull: align_gate == 0 contributes nothing.
    align_active = align_arg > 0.0                            # (N, M) bool

    if np.isscalar(r_pad):
        r_arr = np.full(N, float(r_pad), dtype=np.float64)
    else:
        r_arr = np.asarray(r_pad, dtype=np.float64)
        if r_arr.shape != (N,):
            raise ValueError(
                f"r_pad must be scalar or shape ({N},), got {r_arr.shape}")

    if kernel_half_width is None:
        # theory.md §3.5: w_t half-width = r_i (pad sphere radius itself).
        # Matches cslc_kernels.jacobi_step (line ~735: kernel_h = r_i).
        kh_arr = r_arr
    elif np.isscalar(kernel_half_width):
        kh_arr = np.full(N, float(kernel_half_width), dtype=np.float64)
    else:
        kh_arr = np.asarray(kernel_half_width, dtype=np.float64)
        if kh_arr.shape != (N,):
            raise ValueError(
                f"kernel_half_width must be scalar or shape ({N},), "
                f"got {kh_arr.shape}")

    if delta0 is None:
        delta0_arr = np.zeros((N, 3), dtype=np.float64)
    else:
        delta0_arr = np.asarray(delta0, dtype=np.float64)
        if delta0_arr.shape != (N, 3):
            raise ValueError(
                f"delta0 must be ({N}, 3), got {delta0_arr.shape}")

    inactive_threshold = INACTIVE_RAW_EPS_FACTOR * eps   # contract eq:inactive
    edge_i = edges[:, 0] if edges.size else np.array([], dtype=np.int64)
    edge_j = edges[:, 1] if edges.size else np.array([], dtype=np.int64)

    def _state(d_flat: np.ndarray):
        """Compute everything needed by fun() and jac() at delta = d_flat.

        Returns a dict with raws, sigma, phi_eff, gate, w_t — all
        ``(N, M)``-shaped.  ``phi_eff = σ_ε(raw)·√(σ_ε(raw)+ε)`` is the
        Hertz-like coupling (theory.md eq:phi-eff).  ``w_t`` is treated
        as δ-independent (matches the kernel's truncated gradient).
        """
        d = d_flat.reshape(N, 3)
        q = p_rest - d                                          # (N, 3)
        diff = q[:, None, :] - t_pos[None, :, :]                # (N, M, 3)
        # Half-space depth along n_face: raw = r − n_face · (q − t).
        proj = np.einsum("nmj,mj->nm", diff, n_face)            # (N, M)
        raws = r_arr[:, None] - proj                            # (N, M)
        # Active set: raw ≥ −50·eps AND align_gate > 0 (theory.md §3.6).
        active = (raws >= inactive_threshold) & align_active    # (N, M)
        # Tangential magnitude (vector dropped — only |d_t| enters w_t).
        proj_vec = proj[:, :, None] * n_face[None, :, :]        # (N, M, 3)
        z_vec = diff - proj_vec
        d_t = np.linalg.norm(z_vec, axis=2)                     # (N, M)
        # Smooth quantities (theory.md §3.4–§3.5).
        #   sigma   = σ_ε(raw)       smooth ReLU
        #   phi_eff = σ_ε(raw)·√(σ_ε(raw)+ε)  Hertz lift (eq:phi-eff)
        #   gate    = Σ_ε(raw)       smooth step (eq:gate)
        #   w_t     = Σ_ε(kh − d_t)  tangential locality (eq:w_t)
        if eps > 0.0:
            r2 = raws * raws + eps * eps
            sqr2 = np.sqrt(r2)
            sigma = 0.5 * (raws + sqr2)
            phi_eff = sigma * np.sqrt(sigma + eps)
            gate = 0.5 * (1.0 + raws / sqr2)
            w_t = 0.5 * (1.0 + (kh_arr[:, None] - d_t)
                         / np.sqrt((kh_arr[:, None] - d_t) ** 2 + eps * eps))
        else:
            sigma = np.maximum(0.0, raws)
            phi_eff = sigma * np.sqrt(sigma)  # raw^1.5 for raw≥0, else 0
            gate = np.where(raws > 0, 1.0,
                            np.where(raws == 0, 0.5, 0.0))
            arg = kh_arr[:, None] - d_t
            w_t = np.where(arg > 0, 1.0,
                           np.where(arg == 0, 0.5, 0.0))
        # Mask inactive pairs to zero.
        sigma = np.where(active, sigma, 0.0)
        phi_eff = np.where(active, phi_eff, 0.0)
        gate = np.where(active, gate, 0.0)
        w_t = np.where(active, w_t, 0.0)
        return {
            "d": d, "raws": raws, "active": active, "sigma": sigma,
            "phi_eff": phi_eff, "gate": gate, "w_t": w_t,
        }

    def fun(d_flat: np.ndarray) -> float:
        s = _state(d_flat)
        d = s["d"]
        # Anisotropic anchor (theory.md §6.1): split δ into pad-normal
        # and tangent components using rest n̂_pad (δ-independent).
        delta_n = np.einsum("nj,nj->n", d, n_pad)               # (N,)
        delta_t = d - delta_n[:, None] * n_pad                  # (N, 3)
        E_anchor = (0.5 * ka * float(np.sum(delta_n * delta_n))
                    + 0.5 * ka_t * float(np.sum(delta_t * delta_t)))
        if edges.size:
            ed = d[edge_i] - d[edge_j]
            E_lateral = 0.5 * kl * float(np.sum(ed * ed))
        else:
            E_lateral = 0.0
        # Heuristic contact energy: ½·kc·A_j·w_t·a·φ_eff² (Lyapunov
        # surrogate; not the antiderivative of the force-form Jacobian
        # under the Hertz lift — see theory.md §4).
        E_contact = 0.5 * kc * float(
            np.sum(areas[None, :] * s["w_t"] * align_gate
                   * s["phi_eff"] * s["phi_eff"]))
        return E_anchor + E_lateral + E_contact

    def jac(d_flat: np.ndarray) -> np.ndarray:
        s = _state(d_flat)
        d = s["d"]
        # Anisotropic anchor gradient (theory.md §6.1):
        #   ∂E_anchor_i/∂δ = ka·δ_n·n̂_pad + ka·ρ·δ_t
        delta_n = np.einsum("nj,nj->n", d, n_pad)               # (N,)
        delta_t = d - delta_n[:, None] * n_pad                  # (N, 3)
        grad = ka * delta_n[:, None] * n_pad + ka_t * delta_t
        if edges.size:
            ed = d[edge_i] - d[edge_j]
            # ∂E_lat/∂δ_i = +kl · Σ_{j ∈ N(i)} (δ_i − δ_j)
            np.add.at(grad, edge_i, kl * ed)
            np.add.at(grad, edge_j, -kl * ed)
        # Force-form contact term (theory.md eq:f-phys):
        #   +kc · A_j · w_t · a · φ_eff · gate · n_face_j
        # w_t is treated as δ-independent (matches the kernel's truncated
        # gradient).  This is the kernel's force law, not the gradient
        # of the heuristic ``fun``; convergence at jac = 0 is the
        # force-balance fixed point.
        weights = (kc * areas[None, :] * s["w_t"] * align_gate
                   * s["phi_eff"] * s["gate"])
        grad = grad + np.einsum("nm,mj->nj", weights, n_face)
        return grad.reshape(-1)

    res = minimize(
        fun, delta0_arr.reshape(-1), jac=jac, method="L-BFGS-B",
        options={"gtol": tol, "ftol": tol, "maxiter": maxiter},
    )
    # Damped-Jacobi refinement to the force fixed point (theory.md §6.5).
    # L-BFGS-B uses an inconsistent fun/jac pair under the Hertz lift,
    # so its convergence stalls before reaching the true force-balance
    # zero; the Jacobi sweep matches the kernel's iteration and pulls
    # δ to the same fixed point as the kernel.
    delta_lbfgs = np.asarray(res.x, dtype=np.float64).reshape(N, 3)
    delta_refined, jacobi_iters = _jacobi_refine(
        delta_lbfgs,
        N=N, p_rest=p_rest, n_pad=n_pad, t_pos=t_pos, n_face=n_face,
        areas=areas, edges=edges, edge_i=edge_i, edge_j=edge_j,
        r_arr=r_arr, kh_arr=kh_arr,
        ka=ka, ka_t=ka_t, kl=kl, kc=kc,
        eps=eps, eps_align=eps_align,
        align_gate=align_gate, align_active=align_active,
        inactive_threshold=inactive_threshold,
        max_iter=maxiter, tol=1.0e-10, alpha=0.3,
    )
    deltas = delta_refined
    final_state = _state(deltas.reshape(-1))
    n_active = int(np.sum(final_state["active"]))
    info = {
        "success": bool(res.success),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "final_grad_norm": float(np.linalg.norm(jac(deltas.reshape(-1)))),
        "energy": float(fun(deltas.reshape(-1))),
        "message": str(res.message),
        "n_active_pairs": n_active,
        "jacobi_refine_iters": int(jacobi_iters),
    }
    return deltas, info


def _jacobi_refine(
    delta_in: np.ndarray,
    *,
    N: int,
    p_rest: np.ndarray, n_pad: np.ndarray,
    t_pos: np.ndarray, n_face: np.ndarray, areas: np.ndarray,
    edges: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray,
    r_arr: np.ndarray, kh_arr: np.ndarray,
    ka: float, ka_t: float, kl: float, kc: float,
    eps: float, eps_align: float,
    align_gate: np.ndarray, align_active: np.ndarray,
    inactive_threshold: float,
    max_iter: int, tol: float, alpha: float,
) -> tuple[np.ndarray, int]:
    """Damped block-Jacobi mirror of the Warp kernel's ``jacobi_step``.

    Iterates the force-balance fixed-point equation of theory.md §6.5
    in each pad sphere's local ``{n̂_pad, n̂_pad^⊥}`` frame::

        δ_jacobi_n = (rhs_n + S_n · δ_old_n) / (ka + S_n)
        S_n        = kl·|N(i)| + kc·Σ_j A_j · w_t · a · gate

    (analogous block on the tangent axis).  Stops when
    ``max|δ_new − δ_old| < tol`` or after ``max_iter`` sweeps.

    Pure numpy; anchor + lateral + contact only (friction is handled by
    the smooth-friction primitives in :mod:`cslc_main.theory.cslc_theory`).
    """
    delta = np.asarray(delta_in, dtype=np.float64).copy()
    if N == 0:
        return delta, 0
    # Per-sphere neighbour count for the lateral stabiliser S_n contribution.
    nbr_count = np.zeros(N, dtype=np.float64)
    if edges.size:
        np.add.at(nbr_count, edge_i, 1.0)
        np.add.at(nbr_count, edge_j, 1.0)

    iters_run = 0
    for _ in range(int(max_iter)):
        q = p_rest - delta                                    # (N, 3)
        diff = q[:, None, :] - t_pos[None, :, :]              # (N, M, 3)
        proj = np.einsum("nmj,mj->nm", diff, n_face)          # (N, M)
        raws = r_arr[:, None] - proj                          # (N, M)
        active = (raws >= inactive_threshold) & align_active  # (N, M)
        proj_vec = proj[:, :, None] * n_face[None, :, :]      # (N, M, 3)
        z_vec = diff - proj_vec
        d_t = np.linalg.norm(z_vec, axis=2)                   # (N, M)
        # Hertz-like phi_eff = σ_ε(raw)·√(σ_ε(raw)+ε) (theory.md eq:phi-eff);
        # gate = Σ_ε(raw); w_t = Σ_ε(kh - d_t) with kh = r_pad.
        if eps > 0.0:
            r2 = raws * raws + eps * eps
            sqr2 = np.sqrt(r2)
            sigma = 0.5 * (raws + sqr2)
            phi_eff = sigma * np.sqrt(sigma + eps)
            gate = 0.5 * (1.0 + raws / sqr2)
            arg_t = kh_arr[:, None] - d_t
            w_t = 0.5 * (1.0 + arg_t / np.sqrt(arg_t * arg_t + eps * eps))
        else:
            sigma = np.maximum(0.0, raws)
            phi_eff = sigma * np.sqrt(sigma)
            gate = np.where(raws > 0, 1.0,
                            np.where(raws == 0, 0.5, 0.0))
            arg_t = kh_arr[:, None] - d_t
            w_t = np.where(arg_t > 0, 1.0,
                           np.where(arg_t == 0, 0.5, 0.0))
        phi_eff = np.where(active, phi_eff, 0.0)
        gate = np.where(active, gate, 0.0)
        w_t = np.where(active, w_t, 0.0)
        # Contact load on δ_i = −kc · A · w_t · a · φ_eff · gate · n_face
        # (theory.md eq:f-load).  At the fixed point this load balances
        # anchor + lateral via ka·δ_n = rhs_n_scalar below.
        load_weights = (kc * areas[None, :] * w_t * align_gate
                        * phi_eff * gate)                     # (N, M)
        f_contact = -np.einsum("nm,mj->nj", load_weights, n_face)  # (N, 3)
        # Lateral LOAD on δ_i = -kl Σ_{j∈N(i)} (δ_i − δ_j)
        f_lateral = np.zeros((N, 3), dtype=np.float64)
        if edges.size:
            ed = delta[edge_i] - delta[edge_j]
            np.add.at(f_lateral, edge_i, -kl * ed)
            np.add.at(f_lateral, edge_j, +kl * ed)
        rhs_explicit = f_contact + f_lateral                  # (N, 3)
        delta_n_old = np.einsum("nj,nj->n", delta, n_pad)     # (N,)
        delta_t_old = delta - delta_n_old[:, None] * n_pad    # (N, 3)
        rhs_n_scalar = np.einsum("nj,nj->n", rhs_explicit, n_pad)  # (N,)
        rhs_t_vec = rhs_explicit - rhs_n_scalar[:, None] * n_pad
        # Diagonal stabiliser: S_n = kl·|N(i)| + kc·Σ_j A·w_t·a·gate
        sum_gate = np.einsum("nm->n",
                             areas[None, :] * w_t * align_gate * gate)
        S_n = kl * nbr_count + kc * sum_gate
        S_t = kl * nbr_count
        k_diag_n = ka + S_n
        k_diag_t = ka_t + S_t
        rhs_n_total = rhs_n_scalar + S_n * delta_n_old
        rhs_t_total = rhs_t_vec + S_t[:, None] * delta_t_old
        delta_jac_n = rhs_n_total / k_diag_n
        delta_jac_t = rhs_t_total / k_diag_t[:, None]
        delta_jac = delta_jac_n[:, None] * n_pad + delta_jac_t
        delta_new = (1.0 - alpha) * delta + alpha * delta_jac
        iters_run += 1
        if float(np.max(np.abs(delta_new - delta))) < tol:
            delta = delta_new
            break
        delta = delta_new
    return delta, iters_run


# ────────────────────────────────────────────────────────────────────────
#  Per-sphere contact force readout
# ────────────────────────────────────────────────────────────────────────


def lattice_contact_normal_forces(
    lat: Lattice,
    target,                       # PointSetTarget
    deltas: np.ndarray,
    kc: float,
    *,
    r_pad: np.ndarray | float,
    eps: float = 1.0e-9,
    eps_align: float = EPS_ALIGN_DEFAULT,
    kernel_half_width: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sphere contact-force vector and pad-normal magnitude.

    Given an equilibrium ``deltas`` from :func:`solve_lattice_contact`,
    reconstruct the per-sphere aggregate contact force and project on
    each sphere's outward normal (theory.md §6.4, eq:F-contact-i)::

        F_contact_i = Σ_j  k_c · A_j · w_t_ij · a_ij
                            · phi_eff_ij · gate_ij · n_face_j   (3-vector)
        f_n_i       = |F_contact_i · n̂_pad_i|                  (scalar)

    using the same smooth quantities and gates as
    :func:`solve_lattice_contact`.  Inactive pairs
    (``raw < −50·eps`` or ``align_arg ≤ 0``) contribute zero.

    Args:
        lat: pad lattice; ``lat.n`` provides per-sphere outward normals.
        target: :class:`cslc_main.theory.cslc_targets.PointSetTarget` —
            the same instance used in the original
            :func:`solve_lattice_contact` call.
        deltas: equilibrium displacement, shape ``(N, 3)``.
        kc, r_pad, eps, eps_align, kernel_half_width: must match the
            values passed to :func:`solve_lattice_contact` — this
            helper recomputes the contact-force gradient from
            ``deltas``, so any kwarg drift silently desynchronises
            ``f_n`` from the equilibrium state.

    Returns:
        ``(F_contact, f_n)`` — ``F_contact`` is ``(N, 3)`` per-sphere
        force (= physical force on ``q_i``, theory.md §2);
        ``f_n`` is ``(N,)`` the magnitude of the projection on the
        pad's own outward normal.  Inactive spheres get the zero vector.
    """
    N = lat.N
    M = int(target.M)
    p_rest = np.asarray(lat.p, dtype=np.float64)
    n_pad = np.asarray(lat.n, dtype=np.float64)
    t_pos = np.asarray(target.positions, dtype=np.float64)
    n_face = np.asarray(target.normals, dtype=np.float64)
    areas = (np.asarray(target.areas, dtype=np.float64)
             if target.areas is not None
             else np.ones(M, dtype=np.float64))
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.shape != (N, 3):
        raise ValueError(f"deltas must be ({N}, 3), got {deltas.shape}")

    if np.isscalar(r_pad):
        r_arr = np.full(N, float(r_pad), dtype=np.float64)
    else:
        r_arr = np.asarray(r_pad, dtype=np.float64)
        if r_arr.shape != (N,):
            raise ValueError(
                f"r_pad must be scalar or shape ({N},), got {r_arr.shape}")
    if kernel_half_width is None:
        # Matches solve_lattice_contact: kh = r_pad (theory.md §3.5).
        kh_arr = r_arr
    elif np.isscalar(kernel_half_width):
        kh_arr = np.full(N, float(kernel_half_width), dtype=np.float64)
    else:
        kh_arr = np.asarray(kernel_half_width, dtype=np.float64)
        if kh_arr.shape != (N,):
            raise ValueError(
                f"kernel_half_width must be scalar or shape ({N},), "
                f"got {kh_arr.shape}")

    # Alignment gate (theory.md §3.6) — δ-independent.  One-sided cubic
    # smoothstep on [0, +eps_align]; perpendicular faces (α = 0) are
    # hard-culled to align_w = 0.
    n_face_dot_n_pad = np.einsum("nj,mj->nm", n_pad, n_face)
    align_arg = -n_face_dot_n_pad
    if eps_align > 0.0:
        t = np.clip(align_arg / eps_align, 0.0, 1.0)
        align_gate = t * t * (3.0 - 2.0 * t)
    else:
        align_gate = (align_arg > 0.0).astype(np.float64)
    align_active = align_arg > 0.0

    inactive_threshold = INACTIVE_RAW_EPS_FACTOR * eps

    # Half-space raws.
    q = p_rest - deltas
    diff = q[:, None, :] - t_pos[None, :, :]                  # (N, M, 3)
    proj = np.einsum("nmj,mj->nm", diff, n_face)              # (N, M)
    raws = r_arr[:, None] - proj
    active = (raws >= inactive_threshold) & align_active
    # Tangential.
    proj_vec = proj[:, :, None] * n_face[None, :, :]
    z_vec = diff - proj_vec
    d_t = np.linalg.norm(z_vec, axis=2)
    # Smooth surrogates.  Hertz-like phi_eff = σ_ε(raw)·√(σ_ε(raw)+ε)
    # (theory.md eq:phi-eff); matches the kernel exactly.
    if eps > 0.0:
        r2 = raws * raws + eps * eps
        sqr2 = np.sqrt(r2)
        sigma = 0.5 * (raws + sqr2)
        phi_eff = sigma * np.sqrt(sigma + eps)
        gate = 0.5 * (1.0 + raws / sqr2)
        arg = kh_arr[:, None] - d_t
        arg2 = arg * arg + eps * eps
        sqa = np.sqrt(arg2)
        w_t = 0.5 * (1.0 + arg / sqa)
    else:
        sigma = np.maximum(0.0, raws)
        phi_eff = sigma * np.sqrt(sigma)
        gate = np.where(raws > 0, 1.0, np.where(raws == 0, 0.5, 0.0))
        arg = kh_arr[:, None] - d_t
        w_t = np.where(arg > 0, 1.0, np.where(arg == 0, 0.5, 0.0))
    phi_eff = np.where(active, phi_eff, 0.0)
    gate = np.where(active, gate, 0.0)
    w_t = np.where(active, w_t, 0.0)

    # Per-pair force magnitude along n_face_j.
    weights = (kc * areas[None, :] * w_t * align_gate
               * phi_eff * gate)                              # (N, M)
    # F_contact_i = Σ_j weights_ij · n_face_j  (3-vector per sphere).
    F_contact = np.einsum("nm,mj->nj", weights, n_face)       # (N, 3)
    # Pad-normal magnitude (theory.md §6.4 projection).
    f_n = np.abs(np.einsum("nj,nj->n", F_contact, n_pad))     # (N,)
    return F_contact, f_n


# ────────────────────────────────────────────────────────────────────────
#  Equilibrium solvers without contact (anchor + lateral + external load)
# ────────────────────────────────────────────────────────────────────────


def solve_equilibrium_graph_laplacian(lat: Lattice,
                                      f_ext: np.ndarray) -> np.ndarray:
    """Linear solve ``(K ⊗ I_3) · δ = f_ext``.

    Because the graph-Laplacian + anchor system decouples by axis, each
    of x, y, z is solved with the same scalar ``K``.

    Args:
        f_ext: ``(N, 3)`` external force vector per sphere.

    Returns:
        ``deltas`` of shape ``(N, 3)``.
    """
    K = build_K_matrix(lat)
    deltas = np.zeros_like(f_ext)
    for axis in range(3):
        deltas[:, axis] = np.linalg.solve(K, f_ext[:, axis])
    return deltas


def total_energy(lat: Lattice, deltas: np.ndarray,
                 f_ext: np.ndarray,
                 lateral: LateralLaw = "graph_laplacian") -> float:
    """``E_total(δ) = E_anchor + E_lateral − f_ext · δ``.

    The ``-f_ext · δ`` term is the external potential whose gradient is
    ``-f_ext``; minimising ``E_total`` at fixed ``f_ext`` yields the
    equilibrium displacement.
    """
    if lateral != "graph_laplacian":
        raise ValueError(f"unknown lateral law: {lateral!r}")
    E = anchor_energy(lat, deltas)
    E += lateral_energy_graph_laplacian(lat, deltas)
    E -= float(np.sum(f_ext * deltas))
    return E


def total_force(lat: Lattice, deltas: np.ndarray,
                f_ext: np.ndarray,
                lateral: LateralLaw = "graph_laplacian") -> np.ndarray:
    """Residual force per sphere = ``f_anchor − f_ext − f_lateral``.

    Signs reflect the energy gradient::

        ∇ E_anchor      = +ka·δ           (= f_anchor)
        ∇ E_lateral     = −f_lateral      (quadratic in (δ_i − δ_j),
                                           gradient pushes δ apart —
                                           opposite to the restoring
                                           lateral force)
        ∇ (−f_ext·δ)    = −f_ext

    At equilibrium the residual is zero.
    """
    if lateral != "graph_laplacian":
        raise ValueError(f"unknown lateral law: {lateral!r}")
    f = anchor_force_all(lat, deltas)
    f_lat = lateral_force_graph_laplacian(lat, deltas)
    # f_anchor pulls toward rest (positive sign in our convention).
    # f_lat is the lateral force ON each sphere (sign convention above).
    # External force f_ext is applied; at equilibrium anchor + lateral + ext = 0.
    return f - f_lat - f_ext


def solve_equilibrium_numerical(lat: Lattice,
                                f_ext: np.ndarray,
                                *,
                                lateral: LateralLaw = "graph_laplacian",
                                delta0: np.ndarray | None = None,
                                tol: float = 1.0e-10,
                                ) -> tuple[np.ndarray, dict]:
    """General equilibrium by L-BFGS-B on ``E_total``.

    Numerical witness against the linear
    :func:`solve_equilibrium_graph_laplacian`; the two should agree to
    L-BFGS-B precision.  Only the ``graph_laplacian`` lateral law is
    supported.
    """
    if delta0 is None:
        delta0 = np.zeros((lat.N, 3))

    def fun(x: np.ndarray) -> float:
        return total_energy(lat, x.reshape(lat.N, 3), f_ext, lateral=lateral)

    def jac(x: np.ndarray) -> np.ndarray:
        # Gradient of E w.r.t. delta = anchor - lateral - f_ext.
        return total_force(lat, x.reshape(lat.N, 3), f_ext,
                           lateral=lateral).reshape(-1)

    res = minimize(
        fun, delta0.reshape(-1), jac=jac, method="L-BFGS-B",
        options={"gtol": tol, "ftol": tol, "maxiter": 2000},
    )
    info = {
        "success": bool(res.success),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "final_grad_norm": float(np.linalg.norm(res.jac)),
        "energy": float(res.fun),
        "message": str(res.message),
    }
    return np.asarray(res.x, dtype=np.float64).reshape(lat.N, 3), info


__all__ = [
    "Lattice",
    "LateralLaw",
    "make_chain",
    "make_arc",
    "make_dome",
    "make_flat_grid",
    "build_K_matrix",
    "chain_analytical_eigenvalues",
    "chain_discrete_decay_length",
    # Lattice contact (theory.md §6).
    "solve_lattice_contact",
    "lattice_contact_normal_forces",
    # Per-sphere primitives (anchor + graph-Laplacian lateral).
    "anchor_force_all",
    "anchor_energy",
    "lateral_force_graph_laplacian",
    "lateral_energy_graph_laplacian",
    # Linear / numerical equilibrium witnesses (no contact term).
    "solve_equilibrium_graph_laplacian",
    "solve_equilibrium_numerical",
    "total_energy",
    "total_force",
]
