# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-sphere lattice extension of ``cslc_main.theory.cslc_theory``.

The single-sphere module covers anchor + contact + friction on ONE
lattice sphere.  This module adds the **lateral springs** that couple
a sphere to its neighbours, the lattice-scale quadratic form (the
lattice stiffness matrix ``K``), and the v2 unified contact solver
``solve_lattice_contact`` (contract §6).

Lateral law: graph-Laplacian only.

    f_lat(i, j) = -k_l * (delta_i - delta_j)            (paper eq. 9)

Isotropic in 3D; equivalent quadratic energy
``E = (1/2) k_l Σ_edges ||delta_i - delta_j||^2``.  The nonlinear
distance-preserving lateral was deleted in Phase 5 along with the v1
sphere-target solver chain (``ContactTarget`` / ``SphereIndenter`` /
``PointSetIndenter``).
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
    preserve them).  No contact target -- this module is the lateral
    coupling reference; contact is layered on in step 3.

    Attributes:
        p: (N, 3) float64 -- rest positions.
        n: (N, 3) float64 -- rest outward unit normals.
        edges: (E, 2) int64 -- undirected edge list (i, j) with i < j.
            We treat each unordered edge once; force assembly handles
            both endpoints.
        ka: float -- anchor stiffness [N/m] (isotropic in step 2).
        kl: float -- lateral stiffness [N/m].

    The neighbour count of sphere i is the number of edges that contain
    i; it is computed lazily by ``neighbour_counts``.
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
                raise ValueError(f"degenerate edge {(int(i), int(j))} with L = 0")
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
    edges = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1).astype(np.int64)
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
    a chain.  The IDEAL lateral-coupling story changes vs the chain
    because rest edges no longer point along a single global axis -- in
    particular, distance-preserving can produce outward bulging at
    finite curvature (see test_05_arc_contact step-5 math).
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
    edges = np.stack([np.arange(N - 1), np.arange(1, N)], axis=1).astype(np.int64)
    return Lattice(p=p, n=n_arr, edges=edges, ka=ka, kl=kl)


def make_dome(N: int, R_pad: float, half_angle: float,
              ka: float, kl: float,
              k_neighbors: int = 6,
              ) -> tuple[Lattice, float, float]:
    """Build a 3D spherical-cap lattice via the Fibonacci-spiral sampler.

    Step-5's ``make_arc`` proved the 1D bulge window analytically; this
    function lifts the same idea to a 2D manifold (the dome) so we can
    study sphere-vs-sphere contact at production geometry.  Cap layout:

        z_min   = cos(half_angle)              (cap base)
        z_i     = z_min + (1 - z_min) * (i + 0.5) / N      (equal area)
        r_xy_i  = sqrt(1 - z_i^2)
        phi_i   = i * golden_angle             (golden_angle = pi*(3 - sqrt(5)))
        p_i     = R_pad * (r_xy_i cos phi_i, r_xy_i sin phi_i, z_i)
        n_i     = p_i / R_pad                  (radial outward unit)

    The Fibonacci spiral with equal-area-per-sample is *quasi-uniform*
    on the cap (variance of nearest-neighbour distance ~ 5% of the mean
    on the cap interior), and is the production-equivalent of
    Lloyd/CVT sampling on an analytic cap (deterministic, no scipy
    optimiser in the loop).  Edges = unordered k-NN in 3D, which on a
    quasi-uniform 2D manifold is a close approximation of the surface
    Delaunay graph and reproduces the production
    ``make_cslc_pad_from_samples`` topology.

    Apex sits at theta = 0 (top of cap), i.e. ``(0, 0, R_pad)``.

    Args:
        N: number of lattice spheres.
        R_pad: dome radius [m] (the *pad* sphere radius, NOT the held
            object's).  Production fingertip dome: R_pad = 10 mm.
        half_angle: cap half-angle [rad].  Production fingertip dome
            spans theta_max ~ 1.26 rad (~72 deg, cos = 0.31).
        ka: anchor stiffness [N/m].
        kl: lateral stiffness [N/m].
        k_neighbors: k for the k-NN neighbour graph (production: 6).

    Returns:
        ``(lat, spacing, cap_area)`` where ``spacing`` is the mean
        nearest-neighbour distance [m] (a Fibonacci-spiral invariant
        analogous to Lloyd's CVT spacing) and ``cap_area`` [m^2] is the
        analytic spherical-cap area used downstream for Hertz-patch
        density predictions.
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

    # Mean NN distance = lattice spacing (matches the production
    # ``make_cslc_pad_from_samples`` convention).  Use the first
    # neighbour column (excluding self at column 0).
    nn_dists = np.linalg.norm(p[idx[:, 1]] - p, axis=1)
    spacing = float(np.mean(nn_dists))

    # Spherical-cap area for diagnostics: A = 2 pi R^2 (1 - cos theta_max).
    cap_area = float(2.0 * np.pi * R_pad * R_pad * (1.0 - z_min))

    return Lattice(p=p, n=n_arr, edges=edges, ka=ka, kl=kl), spacing, cap_area


# ────────────────────────────────────────────────────────────────────────
#  Lattice stiffness matrix K (paper eq. 10, scalar form)
# ────────────────────────────────────────────────────────────────────────


def build_K_matrix(lat: Lattice) -> np.ndarray:
    """Assemble the scalar lattice stiffness K in R^{N x N} (paper eq. 10).

    K_ii = ka + kl * |N(i)|,    K_ij = -kl if (i, j) in edges, else 0.

    This is the matrix the paper uses for the scalar normal-projection
    formulation; in the vec3 setting (the ideal we want), the
    graph-Laplacian system is the Kronecker lift  K_3D = K x I_3  so
    each axis decouples and solves with the same K.  We expose the
    scalar K here so the test can inspect, eigen-decompose, and
    hand-check it.

    Returns:
        Symmetric PD numpy array of shape (N, N).
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

    Use this -- not the continuum approximation -- as the reference for
    Green's-function fits on lattices where k_l/k_a is not >> 1.
    """
    if ka <= 0.0:
        raise ValueError(f"ka must be > 0, got {ka}")
    if kl <= 0.0:
        return 0.0
    alpha = ka / (2.0 * kl)
    z_minus = (1.0 + alpha) - np.sqrt(alpha * (alpha + 2.0))
    return float(-1.0 / np.log(z_minus))


def chain_analytical_eigenvalues(N: int, ka: float, kl: float) -> np.ndarray:
    """Eigenvalues of the chain's K, in ASCENDING order.

    For a 1D chain of N spheres with free endpoints (degree-1 at the
    ends, degree-2 elsewhere), K is a tridiagonal matrix whose
    eigenvalues are the Neumann modes of a discrete Laplacian:

        lambda_k = ka + 2 * kl * (1 - cos(k * pi / N)),   k = 0, 1, ..., N-1.

    Reference: Strang, "Computational Science and Engineering" eq.
    2.34 (free-free 1D Laplacian DCT-II spectrum).  At k=0 we recover
    the uniform-translation mode lambda_0 = ka (the only mode the
    lateral spring cannot resist).

    This is the analytical witness the eigenvalue test in
    ``test_02_chain.py`` compares against.
    """
    k = np.arange(N, dtype=np.float64)
    return ka + 2.0 * kl * (1.0 - np.cos(k * np.pi / N))


# ────────────────────────────────────────────────────────────────────────
#  Forces and energies
# ────────────────────────────────────────────────────────────────────────


def anchor_force_all(lat: Lattice, deltas: np.ndarray) -> np.ndarray:
    """f_anchor = +ka * delta, per sphere.  (Isotropic for step 2.)

    Args:
        deltas: (N, 3) displacements.
    Returns:
        (N, 3) anchor forces.
    """
    return lat.ka * deltas


def anchor_energy(lat: Lattice, deltas: np.ndarray) -> float:
    """E_anchor = (1/2) * ka * sum_i ||delta_i||^2."""
    return 0.5 * lat.ka * float(np.sum(deltas * deltas))


def lateral_force_graph_laplacian(lat: Lattice,
                                  deltas: np.ndarray) -> np.ndarray:
    """Paper eq. 9.  Graph-Laplacian force per sphere.

        f_lat_i = -k_l * sum_{j in N(i)} (delta_i - delta_j)

    Equivalent to ``-(kl * L) @ deltas`` where ``L = D - A`` is the
    graph Laplacian (D = degree, A = adjacency).  Quadratic and convex
    in deltas; the equilibrium with anchor + this lateral is the
    solution of ``K @ delta_axis = f_ext_axis`` for each axis
    independently (axes decouple).
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
#  v2 unified lattice contact solver  (contract_v2.md §6)
#
#  Pad ``Lattice`` vs ``PointSetTargetV2``.  Half-space overlap, graph-
#  Laplacian lateral, anisotropic anchor via ``ka_t_ratio``.  The v1
#  ``ContactTarget`` / ``SphereIndenter`` / ``PointSetIndenter`` solvers
#  and the distance-preserving lateral law were deleted in Phase 5.
#
#  ``solve_lattice_contact`` uses L-BFGS-B on the per-pair contact
#  energy summed over the full (N, M) pad×target grid (gradient form
#  per contract §6.5) plus ``_jacobi_refine`` post-step to reach the
#  kernel's damped-Jacobi fixed point on sphere targets (Phase 4b
#  finding #14).
# ────────────────────────────────────────────────────────────────────────



def solve_lattice_contact(
    lat: Lattice,
    target,                       # PointSetTargetV2 (from cslc_targets)
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
    """v2 unified contact equilibrium: ``Lattice`` vs ``PointSetTargetV2``.

    Minimises (per contract_v2.md §6, all gradient terms with + sign)::

        E_total  =  E_anchor_aniso + E_lateral_GL
                  + Σ_i Σ_j (½ k_c A_j w_t_ij a_ij φ_eff_ij²)

    over per-pad-sphere δ ∈ ℝ^(N×3).

    **Anisotropic anchor (Phase 4 extension, contract §6.1).**  In each
    pad sphere's local ``{n̂_pad_i, n̂_pad_i^⊥}`` frame::

        δ_n_i = δ_i · n̂_pad_i,   δ_t_i = δ_i − δ_n_i n̂_pad_i
        E_anchor_i = ½ k_a δ_n_i² + ½ (k_a ρ) ||δ_t_i||²

    with ``ρ = ka_t_ratio``.  ``ρ = 1`` recovers the isotropic anchor.
    Because ``n̂_pad_i`` is rest body-local (δ-independent), the
    gradient picks up no extra ``∂n̂_pad/∂δ`` term::

        ∂E_anchor_i/∂δ = k_a δ_n_i n̂_pad_i + (k_a ρ) δ_t_i

    The Phase 4 bridge harness (T-K scene E) requires this.

    **w_t treated as δ-independent (Phase 4 commit, finding #4).**
    The ∂w_t/∂δ term ``∂E_contact/∂δ_i = ... + ½ k_c A_j a φ_eff² ·
    Σ'_ε(3r − d_t) · ẑ_ij`` is **dropped** so theory's gradient matches
    the kernel's truncated gradient (`jacobi_step_point_set` ignores
    this term by convention; cf. contract §4 deriv).  Removing it
    means ``jac`` is an O(ε²/r³) inconsistent approximation of ``fun``
    — L-BFGS-B's line search may oscillate marginally — but the
    converged ``∂E/∂δ = 0`` point is the SAME smooth-surrogate
    equilibrium the kernel iterates to.  This is the Phase 4 commit
    (cf. contract §17 finding #9): **theory = kernel, never both ways**.

    ``a_ij`` is the **smooth alignment gate** (contract §3.6, amended
    Phase 4b after finding #11):

        align_arg_ij  =  -(n̂_face_j · n̂_pad_i)               (>0 = opposing)
        a_ij          =  smoothstep(align_arg_ij; 0, eps_align)

    a C¹ cubic smoothstep on the **one-sided** band ``[0, +eps_align]``:
    exactly 1 for ``align_arg ≥ +eps_align`` (face-on), exactly 0 for
    ``align_arg ≤ 0`` (perpendicular OR back-to-back — both HARD-culled).
    It is δ-independent (n_pad and n_face are rest body-local geometry),
    so it enters the gradient as a constant multiplier — no extra
    ``∂a/∂δ`` term.  Compact support means back-side AND perpendicular
    samples contribute *exactly* zero (no polynomial tail; cf. the
    alternative Σ_ε form which has a 1/x² tail).

    **Why one-sided.**  The earlier symmetric band ``[-eps_align,
    +eps_align]`` gave ``a = 0.5`` at perpendicular, which over-coupled
    side-face samples on closed convex targets (box, mesh).  At
    production eps a single pad vs a full box failed to converge under
    L-BFGS-B because corner samples of side faces sat inside the
    locality kernel with half-strength asymmetric in-plane forces.
    Contract amendment after Phase 4a finding #11; see contract §17.

    Args:
        lat: pad lattice (uses ``lat.p``, ``lat.n``, ``lat.edges``,
            ``lat.ka``, ``lat.kl``).
        target: a ``PointSetTargetV2`` instance from
            :mod:`cslc_main.theory.cslc_targets`.  Has ``positions``
            (M, 3), ``normals`` (M, 3), and ``areas`` (M,) or None.
        kc: contact stiffness [N/m] (per-pair; same for all pairs in v2).
        r_pad: per-pad-sphere radius [m].  Scalar or shape (N,).
        delta0: warm start, shape (N, 3).  Default zeros.
        eps: smoothing width [m] for the half-space surrogate.  Default
             1e-9 for theory-side precision; production kernel uses 5e-4.
        eps_align: smoothing half-width for the alignment gate
            (dimensionless cosine units in [0, 1]).  Default 0.05
            (≈ 2.87° angular half-transition, one-sided from α = 0 to
            α = +eps_align).  Set to 0 for the legacy binary cull
            (recovers the hard step at n_face·n_pad = 0, with the
            documented force-discontinuity bug — only use for ablation
            testing).
        ka_t_ratio: anisotropic anchor ratio ``ρ = k_at / k_a``.  Default
            1.0 (isotropic).  Lattice does not carry this — the kernel
            takes it as a scalar parameter (``ka_tangent_ratio``), and
            this signature follows the same convention so bridge scenes
            can sweep ρ without rebuilding the lattice.
        tol: L-BFGS-B ``gtol`` / ``ftol``.
        maxiter: L-BFGS-B iteration cap.
        kernel_half_width: tangential locality kernel half-width [m]
            (the ``3 r_i`` in contract eq:w_t).  Scalar or (N,) or
            None.  None ⇒ ``3 · r_pad`` (the contract default).  Pass a
            smaller value to restrict contact locality (useful for
            chain-vs-single-sample tests where you want only one pad
            sphere to engage).

    Returns:
        ``(deltas, info)`` — ``deltas`` is (N, 3); ``info`` is the
        scipy diagnostic dict augmented with ``n_active_pairs``.
    """
    from scipy.optimize import minimize

    N = lat.N
    M = int(target.M)
    p_rest = np.asarray(lat.p, dtype=np.float64)              # (N, 3)
    n_pad = np.asarray(lat.n, dtype=np.float64)               # (N, 3) outward normals
    edges = np.asarray(lat.edges, dtype=np.int64)             # (E, 2)
    ka = float(lat.ka)
    ka_t = ka * float(ka_t_ratio)                             # anisotropic tangent
    kl = float(lat.kl)
    t_pos = np.asarray(target.positions, dtype=np.float64)    # (M, 3)
    n_face = np.asarray(target.normals, dtype=np.float64)     # (M, 3)
    areas = (np.asarray(target.areas, dtype=np.float64)
             if target.areas is not None
             else np.ones(M, dtype=np.float64))

    # ──────────────────────────────────────────────────────────────────
    # Smooth alignment gate (contract §3.6, amended Phase 4b after
    # finding #11).
    # ``align_arg = -(n_face · n_pad)`` is +1 at perfect face-on and -1
    # at back-to-back; perpendicular faces give align_arg = 0.  The C¹
    # cubic smoothstep on the ONE-SIDED band [0, +eps_align] is exactly
    # 1 for align_arg ≥ +eps_align, exactly 0 for align_arg ≤ 0
    # (perpendicular OR back-to-back), and smooth in between.
    # Compactly supported ⇒ back-side AND perpendicular samples
    # contribute exactly zero; no polynomial tail to over-couple side
    # faces of closed convex targets (boxes, meshes).  Cached once
    # since (n_pad, n_face) are δ-independent.
    # ──────────────────────────────────────────────────────────────────
    n_face_dot_n_pad = np.einsum("nj,mj->nm", n_pad, n_face)  # (N, M)
    align_arg = -n_face_dot_n_pad                             # (N, M)
    if eps_align > 0.0:
        # Cubic smoothstep s(t) = 3t² − 2t³ on t = clip(α/ε, 0, 1).
        # Perpendicular (α = 0) ⇒ t = 0 ⇒ s = 0 (HARD-cull).
        t = np.clip(align_arg / eps_align, 0.0, 1.0)
        align_gate = t * t * (3.0 - 2.0 * t)                  # (N, M)
    else:
        # Hard binary step (recovered limit; perpendicular hard-culled).
        align_gate = (align_arg > 0.0).astype(np.float64)
    # Performance cull: pairs with align_gate == 0 contribute nothing.
    # Perpendicular (α = 0) is hard-culled under the amended gate.
    align_active = align_arg > 0.0                            # (N, M) bool

    if np.isscalar(r_pad):
        r_arr = np.full(N, float(r_pad), dtype=np.float64)
    else:
        r_arr = np.asarray(r_pad, dtype=np.float64)
        if r_arr.shape != (N,):
            raise ValueError(
                f"r_pad must be scalar or shape ({N},), got {r_arr.shape}")

    if kernel_half_width is None:
        kh_arr = 3.0 * r_arr
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

        Returns a dict with raws, phi_eff, gate, w_t, diff, d_t — all
        (N, M)-shaped where applicable.  w_t is treated as δ-independent
        (its derivative is NOT carried — see solve_lattice_contact
        docstring for the Phase 4 commit and contract §17 finding #9).
        """
        d = d_flat.reshape(N, 3)
        q = p_rest - d                                          # (N, 3)
        diff = q[:, None, :] - t_pos[None, :, :]                # (N, M, 3)
        # Half-space depth along n_face: raw = r - n_face · (q - t).
        proj = np.einsum("nmj,mj->nm", diff, n_face)            # (N, M)
        raws = r_arr[:, None] - proj                            # (N, M)
        # Active set: raw >= -50·eps AND align_gate > 0 (contract §3.6).
        active = (raws >= inactive_threshold) & align_active    # (N, M)
        # Tangential magnitude (vector dropped — only |d_t| enters w_t).
        proj_vec = proj[:, :, None] * n_face[None, :, :]        # (N, M, 3)
        z_vec = diff - proj_vec
        d_t = np.linalg.norm(z_vec, axis=2)                     # (N, M)
        # Smooth quantities.
        if eps > 0.0:
            r2 = raws * raws + eps * eps
            sqr2 = np.sqrt(r2)
            phi_eff = 0.5 * (raws + sqr2)
            gate = 0.5 * (1.0 + raws / sqr2)
            w_t = 0.5 * (1.0 + (kh_arr[:, None] - d_t)
                         / np.sqrt((kh_arr[:, None] - d_t) ** 2 + eps * eps))
        else:
            phi_eff = np.maximum(0.0, raws)
            gate = np.where(raws > 0, 1.0,
                            np.where(raws == 0, 0.5, 0.0))
            arg = kh_arr[:, None] - d_t
            w_t = np.where(arg > 0, 1.0,
                           np.where(arg == 0, 0.5, 0.0))
        # Mask inactive pairs to zero.
        phi_eff = np.where(active, phi_eff, 0.0)
        gate = np.where(active, gate, 0.0)
        w_t = np.where(active, w_t, 0.0)
        return {
            "d": d, "raws": raws, "active": active,
            "phi_eff": phi_eff, "gate": gate, "w_t": w_t,
        }

    def fun(d_flat: np.ndarray) -> float:
        s = _state(d_flat)
        d = s["d"]
        # Anisotropic anchor (contract §6.1): split each δ_i into its
        # pad-normal component δ_n and tangent δ_t, using rest n̂_pad
        # (δ-independent, so no extra Jacobian term).
        delta_n = np.einsum("nj,nj->n", d, n_pad)               # (N,)
        delta_t = d - delta_n[:, None] * n_pad                  # (N, 3)
        E_anchor = (0.5 * ka * float(np.sum(delta_n * delta_n))
                    + 0.5 * ka_t * float(np.sum(delta_t * delta_t)))
        if edges.size:
            ed = d[edge_i] - d[edge_j]
            E_lateral = 0.5 * kl * float(np.sum(ed * ed))
        else:
            E_lateral = 0.0
        # E_contact = sum_ij ½ kc A_j w_t_ij a_ij phi_eff_ij²
        # (a_ij = align_gate; δ-independent, just a multiplier.)
        E_contact = 0.5 * kc * float(
            np.sum(areas[None, :] * s["w_t"] * align_gate
                   * s["phi_eff"] * s["phi_eff"]))
        return E_anchor + E_lateral + E_contact

    def jac(d_flat: np.ndarray) -> np.ndarray:
        s = _state(d_flat)
        d = s["d"]
        # Anisotropic anchor gradient (contract §6.1):
        #   ∂E_anchor_i/∂δ = ka·δ_n·n̂_pad + ka·ρ·δ_t
        delta_n = np.einsum("nj,nj->n", d, n_pad)               # (N,)
        delta_t = d - delta_n[:, None] * n_pad                  # (N, 3)
        grad = ka * delta_n[:, None] * n_pad + ka_t * delta_t
        if edges.size:
            ed = d[edge_i] - d[edge_j]
            # ∂E_lat/∂δ_i = +kl Σ_j∈N(i) (δ_i − δ_j)
            np.add.at(grad, edge_i, kl * ed)
            np.add.at(grad, edge_j, -kl * ed)
        # Contact gradient — main term only.  ∂w_t/∂δ DROPPED to match
        # the kernel's truncated gradient (Phase 4 policy, contract
        # finding #9).  The converged δ is the point where the
        # TRUNCATED gradient is zero — NOT a stationary point of the
        # full smooth-surrogate energy in ``fun``.  Both theory and
        # kernel see the same offset, so bridge parity holds;
        # ``info["energy"]`` is off from the true minimum by O(1%).
        #
        #   ∂/∂δ_i (½ kc A_j w_t a phi_eff²) at fixed w_t, a
        #     =  kc A_j w_t a phi_eff gate · n_face_j
        # ∂(raw)/∂δ_i = +n_face_j (contract §4)
        weights = (kc * areas[None, :] * s["w_t"] * align_gate
                   * s["phi_eff"] * s["gate"])
        grad = grad + np.einsum("nm,mj->nj", weights, n_face)
        return grad.reshape(-1)

    res = minimize(
        fun, delta0_arr.reshape(-1), jac=jac, method="L-BFGS-B",
        options={"gtol": tol, "ftol": tol, "maxiter": maxiter},
    )
    # Post-Jacobi refinement (Phase 4b, finding #14).  L-BFGS-B's
    # internal Wolfe line search compares ``fun`` (full E, including
    # w_t) against the Phase 4a truncated ``jac`` (no ∂w_t/∂δ).  For
    # flat / box targets the inconsistency is O(ε²/r³) per pair and
    # L-BFGS-B converges fine.  For sphere targets (high curvature,
    # many off-axis samples simultaneously in w_t transition zone)
    # the inconsistency stalls L-BFGS-B short of the truncated-
    # gradient zero — the kernel's damped Jacobi reaches a noticeably
    # different fixed point.  Adding a Jacobi post-refinement mirrors
    # the kernel's iteration (contract §6.5) and pulls theory to the
    # same truncated-gradient zero as the kernel.  On scenes where
    # L-BFGS-B already converged (flat / box / chain), Jacobi exits
    # in O(1) iterations; on scenes where it stalled, Jacobi cleans
    # up.  Tolerance ``jacobi_tol`` chosen to match the kernel's
    # JACOBI_TOL (1e-10) for direct bridge parity.
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
    """Damped block-Jacobi mirror of ``jacobi_step_point_set``.

    Iterates the same fixed-point equation as the Warp kernel
    (contract §6.5): in each pad sphere's local ``{n̂_pad, n̂_pad^⊥}``
    frame, ``δ_jacobi_n = (rhs_n + S_n · δ_old_n) / (ka + S_n)`` with
    ``S_n = kl·|N(i)| + kc·Σ_j A_j · w_t · a · gate``.  Stops when
    ``max|δ_new − δ_old| < tol`` or after ``max_iter`` sweeps.

    Pure numpy; no friction (anchor + lateral + contact only — friction
    is added separately by the smooth-friction path in
    :mod:`cslc_main.theory.cslc_theory`).
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
        if eps > 0.0:
            r2 = raws * raws + eps * eps
            sqr2 = np.sqrt(r2)
            phi_eff = 0.5 * (raws + sqr2)
            gate = 0.5 * (1.0 + raws / sqr2)
            arg_t = kh_arr[:, None] - d_t
            w_t = 0.5 * (1.0 + arg_t / np.sqrt(arg_t * arg_t + eps * eps))
        else:
            phi_eff = np.maximum(0.0, raws)
            gate = np.where(raws > 0, 1.0,
                            np.where(raws == 0, 0.5, 0.0))
            arg_t = kh_arr[:, None] - d_t
            w_t = np.where(arg_t > 0, 1.0,
                           np.where(arg_t == 0, 0.5, 0.0))
        phi_eff = np.where(active, phi_eff, 0.0)
        gate = np.where(active, gate, 0.0)
        w_t = np.where(active, w_t, 0.0)
        # Contact LOAD on δ_i = -kc · A · w_t · a · phi · gate · n_face
        # (= -∂E_contact/∂δ_i, per contract §4 eq:f-load).  Mirrors
        # the kernel's ``f_contact_vec = kc · area_kernel · align_w
        # · phi_eff · gate · n_eff`` where ``n_eff = -n_face_world``
        # (the negative sign on n_eff IS the load-vs-grad sign flip).
        # The equilibrium ka·δ_n = rhs_n_scalar derived below picks
        # up this load with a + sign so that the standard form
        # ka·δ_n = -∂E_contact·n_pad/∂δ holds at the fixed point.
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
#  Per-sphere normal force readout  (v2 contract — Phase 3)
# ────────────────────────────────────────────────────────────────────────


def lattice_contact_normal_forces(
    lat: Lattice,
    target,                       # PointSetTargetV2
    deltas: np.ndarray,
    kc: float,
    *,
    r_pad: np.ndarray | float,
    eps: float = 1.0e-9,
    eps_align: float = EPS_ALIGN_DEFAULT,
    kernel_half_width: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sphere contact-force vector and normal-axis magnitude (v2).

    Given an equilibrium ``deltas`` from :func:`solve_lattice_contact`,
    reconstruct the per-sphere aggregate contact force and project on
    each sphere's outward normal — the inputs T-J's grip aggregation
    needs (contract §6.4, eq:F-contact-i).

    Computes (gradient form per contract §6.5; positive sign on each
    term)::

        F_contact_i  =  Σ_j  k_c · A_j · w_t_ij · a_ij
                              · phi_eff_ij · gate_ij · n_face_j        (3-vec)
        f_n_i        =  |F_contact_i · n̂_pad_i|                       (scalar)

    using the **same** smooth quantities and gates as
    :func:`solve_lattice_contact` — half-space ``raw_ij``, smoothed
    ``phi_eff_ij = σ_ε(raw)``, ``gate_ij = Σ_ε(raw)``, locality
    ``w_t_ij = Σ_ε(kernel_half_width − d_t_ij)``, and the one-sided
    cubic smoothstep alignment gate ``a_ij`` from contract §3.6
    (amended Phase 4b: band ``[0, +eps_align]``; perpendicular faces
    HARD-culled).  Inactive pairs (raw < −50ε or align_arg ≤ 0)
    contribute zero.

    Args:
        lat: pad lattice; ``lat.n`` provides per-sphere outward normals.
        target: ``PointSetTargetV2`` — the same target instance used in
            the original :func:`solve_lattice_contact` call.
        deltas: equilibrium displacement, shape (N, 3).
        kc, r_pad, eps, eps_align, kernel_half_width: **must match
            the values passed to** :func:`solve_lattice_contact` —
            this helper recomputes the contact-force gradient from
            ``deltas``, so any kwarg drift silently desynchronises
            ``f_n`` from the equilibrium state.  Pass identical values
            (e.g. as a shared kwargs dict) at both call sites.

    Returns:
        ``(F_contact, f_n)`` — ``F_contact`` is (N, 3) per-sphere force
        in the gradient form (= physical force on q_i, contract §2);
        ``f_n`` is (N,) the magnitude of the projection on the pad's
        own outward normal.  Inactive spheres get the zero vector.
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
        kh_arr = 3.0 * r_arr
    elif np.isscalar(kernel_half_width):
        kh_arr = np.full(N, float(kernel_half_width), dtype=np.float64)
    else:
        kh_arr = np.asarray(kernel_half_width, dtype=np.float64)
        if kh_arr.shape != (N,):
            raise ValueError(
                f"kernel_half_width must be scalar or shape ({N},), "
                f"got {kh_arr.shape}")

    # Alignment gate (contract §3.6, amended Phase 4b) — δ-independent.
    # One-sided cubic smoothstep on [0, +eps_align]; perpendicular
    # faces (α = 0) are HARD-culled to align_w = 0.
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
    # Smooth surrogates.
    if eps > 0.0:
        r2 = raws * raws + eps * eps
        sqr2 = np.sqrt(r2)
        phi_eff = 0.5 * (raws + sqr2)
        gate = 0.5 * (1.0 + raws / sqr2)
        arg = kh_arr[:, None] - d_t
        arg2 = arg * arg + eps * eps
        sqa = np.sqrt(arg2)
        w_t = 0.5 * (1.0 + arg / sqa)
    else:
        phi_eff = np.maximum(0.0, raws)
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
    # Pad-normal magnitude (contract §6.4 chooses this projection).
    f_n = np.abs(np.einsum("nj,nj->n", F_contact, n_pad))     # (N,)
    return F_contact, f_n


# ────────────────────────────────────────────────────────────────────────
#  Equilibrium solvers (no contact in step 2 -- just anchor + lateral
#  + external load)
# ────────────────────────────────────────────────────────────────────────


def solve_equilibrium_graph_laplacian(lat: Lattice,
                                      f_ext: np.ndarray) -> np.ndarray:
    """Linear solve (K x I_3) @ delta = f_ext.

    Because the graph-Laplacian + anchor system separates by axis, we
    can solve each of x, y, z with the same scalar K and stack the
    results.  This is the cheapest possible "ground truth" reference.

    Args:
        f_ext: (N, 3) external force vector per sphere.

    Returns:
        deltas: (N, 3) equilibrium displacements.
    """
    K = build_K_matrix(lat)
    deltas = np.zeros_like(f_ext)
    for axis in range(3):
        deltas[:, axis] = np.linalg.solve(K, f_ext[:, axis])
    return deltas


def total_energy(lat: Lattice, deltas: np.ndarray,
                 f_ext: np.ndarray,
                 lateral: LateralLaw = "graph_laplacian") -> float:
    """E_total(delta) = E_anchor + E_lateral - f_ext . delta.

    The last term is the work the external force does against delta;
    negating it makes the equilibrium the minimiser of E_total at
    fixed f_ext.
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
    """Residual force per sphere = f_anchor - f_ext - f_lateral.

    The signs reflect the energy gradient:
        grad E_anchor = +ka * delta = +f_anchor
        grad E_lateral = -(force on sphere from lateral) [confusing but
            true: lateral energy is quadratic in (delta_i - delta_j),
            so its gradient pushes deltas APART, opposite to the
            restoring force].
        grad of -f_ext . delta = -f_ext.

    At equilibrium, the residual is zero.  The numerical solvers below
    drive ``||residual||`` below tol.
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
    """General equilibrium by L-BFGS-B on E_total.

    Retained as a numerical witness against the linear
    :func:`solve_equilibrium_graph_laplacian`.  Phase 5 dropped the
    distance-preserving lateral law, so the only supported choice is
    ``lateral = "graph_laplacian"``; the numerical and linear solvers
    should agree to L-BFGS-B precision.
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
    "build_K_matrix",
    "chain_analytical_eigenvalues",
    "chain_discrete_decay_length",
    # v2 unified lattice contact path (contract §6).
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
