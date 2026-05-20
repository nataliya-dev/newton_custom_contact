# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-sphere lattice extension of ``cslc_main.theory.cslc_theory``.

The single-sphere module covers anchor + contact + friction on ONE
lattice sphere.  This module adds the missing piece: the **lateral
springs** that couple a sphere to its neighbours, and the lattice-scale
quadratic form (the lattice stiffness matrix ``K``).

Two lateral laws are implemented side by side so the test suite can
compare them:

* ``lateral_force_graph_laplacian`` -- the paper's eq. 9
      f_lat(i, j) = -k_l * (delta_i - delta_j)
  Isotropic in 3D.  Equivalent quadratic energy:
      E = (1/2) * k_l * sum_edges ||delta_i - delta_j||^2.
  Easy to assemble as a sparse scalar matrix K via the scalar
  ``build_K_matrix`` below (paper eq. 10), with a Kronecker-with-I_3
  lift to the full 3N x 3N system.

* ``lateral_force_distance_preserving`` -- the IDEAL non-linear law
      f_lat(i, j) = -k_l * (||q_j - q_i|| - L_ij) * e_hat_ij
  where q_i = p_i - delta_i.  Linearises around delta = 0 to a rank-1
  projector onto the rest edge direction (see notes in
  ``test_02_chain.py``); reduces to the graph Laplacian for loads
  along the rest edges, diverges from it for transverse loads.

The pair of laws share anchor energy and the test harness so we never
build incompatible scenes for the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize


LateralLaw = Literal["graph_laplacian", "distance_preserving"]


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


def lateral_force_distance_preserving(lat: Lattice,
                                      deltas: np.ndarray) -> np.ndarray:
    """IDEAL distance-preserving lateral.  Hookean spring on physical
    distance between deformed centres q_i = p_i - delta_i.

        f_lat_i->j = -k_l * (||q_j - q_i|| - L_ij) * e_hat_ij,
        e_hat_ij = (q_j - q_i) / ||q_j - q_i||,
        L_ij = ||p_j - p_i||.

    Force on sphere i:  f_lat_i = sum_{j in N(i)} f_lat_i->j.
    Force on sphere j:  f_lat_j = -f_lat_i->j (action-reaction).

    Linearises to the rank-1 projector law
        f_lat_i->j_lin = -k_l * (e_hat^rest * e_hat^rest^T) * (delta_i - delta_j)
    at delta = 0; on a 1D chain (all edges along the same axis) this
    reduces to the graph-Laplacian on that axis only.
    """
    f = np.zeros_like(deltas)
    for k, (i, j) in enumerate(lat.edges):
        q_i = lat.p[i] - deltas[i]
        q_j = lat.p[j] - deltas[j]
        d = q_j - q_i
        L_def = float(np.linalg.norm(d))
        if L_def < 1e-15:
            continue  # degenerate -- ignore force contribution
        L_rest = float(np.linalg.norm(lat.p[j] - lat.p[i]))
        e_hat = d / L_def
        force_ij = lat.kl * (L_def - L_rest) * e_hat  # along e_hat_ij from i toward j
        f[i] += force_ij      # pulls i toward j when stretched
        f[j] -= force_ij      # action-reaction
    return f


def lateral_energy_distance_preserving(lat: Lattice,
                                       deltas: np.ndarray) -> float:
    """E_lat^DP = (1/2) * k_l * sum_edges (||q_j - q_i|| - L_ij)^2."""
    e = 0.0
    for (i, j) in lat.edges:
        q_i = lat.p[i] - deltas[i]
        q_j = lat.p[j] - deltas[j]
        L_def = float(np.linalg.norm(q_j - q_i))
        L_rest = float(np.linalg.norm(lat.p[j] - lat.p[i]))
        dL = L_def - L_rest
        e += dL * dL
    return 0.5 * lat.kl * e


# ────────────────────────────────────────────────────────────────────────
#  Contact target (step 3 onwards)
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContactTarget:
    """A rigid sphere we contact against, paired with which lattice sphere
    it overlaps and with a stiffness.

    For step 3 we put one of these on a chain.  Future steps will allow
    a list of targets, each one overlapping a different lattice sphere.

    Attributes:
        sphere_idx: Lattice index that this target overlaps at rest.
        t: Target centre, world frame.
        R: Target radius.
        kc: Contact stiffness for this target.
    """

    sphere_idx: int
    t: np.ndarray
    R: float
    kc: float

    def __post_init__(self):
        if self.t.shape != (3,):
            raise ValueError(f"t must be (3,), got {self.t.shape}")


def contact_energy_face_on(lat: Lattice, target: ContactTarget,
                           deltas: np.ndarray, phi_rest: float,
                           *, eps: float = 0.0) -> float:
    """E_contact = (1/2) kc * [phi_rest - dot(delta_k, n_k)]_+^2  (face-on).

    Convenience wrapper for the step-3 chain test where the target sits
    on the rest outward normal of sphere ``target.sphere_idx`` and the
    test driver carries phi_rest = (r_lat + R) - ||t - p_k|| separately.

    Args:
        eps: smoothing width [m].  eps = 0 gives the hard max(0, .).
    """
    k = target.sphere_idx
    delta_n = float(np.dot(deltas[k], lat.n[k]))
    raw = phi_rest - delta_n
    if eps <= 0.0:
        phi_eff = max(0.0, raw)
    else:
        phi_eff = 0.5 * (raw + np.sqrt(raw * raw + eps * eps))
    return 0.5 * target.kc * phi_eff * phi_eff


def contact_force_face_on(lat: Lattice, target: ContactTarget,
                          deltas: np.ndarray, phi_rest: float,
                          *, eps: float = 0.0) -> np.ndarray:
    """Force on each sphere from the face-on contact.

    Only sphere ``target.sphere_idx`` receives a non-zero force; it
    points OPPOSITE to its outward normal (contact pushes the sphere
    INWARD toward the body interior, i.e. in -n_k direction, while
    ``delta_k`` points OUTWARD as defined in cslc_theory.py).  Wait --
    revisit the sign convention.

    The chain convention here: delta_n_k > 0 means q_k has moved
    INWARD relative to the body.  Equivalent to compressing the sphere
    INWARD along its OUTWARD normal direction (because q = p - delta,
    delta_n>0 puts q on the body side of p).

    Anchor energy E_anchor = (1/2) ka ||delta||^2 -> anchor force on
    sphere = grad_delta E_anchor = +ka * delta (in our sign convention
    where delta = displacement of q from p in body-coord, positive
    when compressed inward).

    Contact energy E_contact = (1/2) kc (phi_rest - dot(delta, n))^2_+
    -> contact contribution to grad_delta E_contact at sphere k:
        grad = -kc * phi_eff * n_k   (note the minus sign)
    which is the FORCE THAT GROWS phi_eff (the "applied" load, pushing
    delta along +n).  At equilibrium grad_delta E_total = 0, i.e.
    anchor + lateral = +kc * phi_eff * n_k on sphere k.

    To match the convention used by the test driver
    (``solve_chain_contact_linear``), this function returns the
    "contact source" vector  +kc * phi_eff * n_k  on sphere k and zero
    elsewhere -- the right-hand side of the linear equilibrium
    K delta = source.
    """
    k = target.sphere_idx
    delta_n = float(np.dot(deltas[k], lat.n[k]))
    raw = phi_rest - delta_n
    if eps <= 0.0:
        phi_eff = max(0.0, raw)
    else:
        phi_eff = 0.5 * (raw + np.sqrt(raw * raw + eps * eps))
    f = np.zeros_like(deltas)
    f[k] = target.kc * phi_eff * lat.n[k]
    return f


def solve_lattice_contact_linear(lat: Lattice,
                                 target: ContactTarget,
                                 phi_rest: float) -> np.ndarray:
    """Linear graph-Laplacian + face-on contact solve for general lattices.

    Assembles the full 3N x 3N system

        [(K (x) I_3) + kc * (e_k e_k^T) (x) (n_k n_k^T)] delta_flat
        = kc * phi_rest * (e_k (x) n_k)

    where (x) is the Kronecker product, K is the scalar lattice
    stiffness matrix, e_k is the indicator vector at the contact
    sphere, and n_k is its outward normal.  The block-diagonal
    K (x) I_3 decouples lateral coupling per Cartesian axis; the
    contact term is rank-1 in 3D at sphere k only.

    Reduces to ``solve_chain_contact_linear`` when all outward normals
    coincide (face-on contact on a flat lattice).  Used for arcs and
    other curved lattices where the contact sphere's normal is not
    parallel to its neighbours' normals.

    Args:
        lat: Lattice (any topology, any per-sphere outward normals).
        target: ContactTarget (sphere_idx, kc).
        phi_rest: rest overlap [m].

    Returns:
        (N, 3) deltas array.
    """
    N = lat.N
    K = build_K_matrix(lat)
    A = np.kron(K, np.eye(3))  # 3N x 3N block-diag of K on each axis
    k = target.sphere_idx
    n_k = lat.n[k]
    A[3 * k:3 * k + 3, 3 * k:3 * k + 3] += target.kc * np.outer(n_k, n_k)
    rhs = np.zeros(3 * N)
    rhs[3 * k:3 * k + 3] = target.kc * phi_rest * n_k
    delta_flat = np.linalg.solve(A, rhs)
    return delta_flat.reshape(N, 3)


def solve_chain_contact_linear(lat: Lattice,
                               target: ContactTarget,
                               phi_rest: float) -> np.ndarray:
    """Closed-form equilibrium for one face-on contact (graph-Laplacian).

    Solves (K + kc * e_k e_k^T) * delta_n = kc * phi_rest * e_k for the
    scalar delta_n field (component along sphere k's outward normal).
    Lifts to vec3 via delta_i = delta_n_i * n_i.

    Assumes all spheres share the SAME outward normal (true on our
    chain by construction); for general lattices use the numerical
    minimiser with the contact_energy_face_on / contact_force_face_on
    helpers.

    Args:
        lat: chain Lattice from make_chain.
        target: ContactTarget (specifies sphere_idx and kc).
        phi_rest: rest overlap of the target with sphere
            target.sphere_idx, in [m].

    Returns:
        deltas: (N, 3) equilibrium displacements.
    """
    K = build_K_matrix(lat)
    K_aug = K.copy()
    K_aug[target.sphere_idx, target.sphere_idx] += target.kc
    rhs = np.zeros(lat.N)
    rhs[target.sphere_idx] = target.kc * phi_rest
    delta_n = np.linalg.solve(K_aug, rhs)
    n_global = lat.n[target.sphere_idx]
    return np.einsum("i,j->ij", delta_n, n_global)


def solve_lattice_contact_numerical(lat: Lattice,
                                    target: ContactTarget,
                                    phi_rest: float,
                                    *,
                                    lateral: LateralLaw = "distance_preserving",
                                    delta0: np.ndarray | None = None,
                                    eps: float = 1.0e-7,
                                    tol: float = 1.0e-12,
                                    ) -> tuple[np.ndarray, dict]:
    """L-BFGS-B minimisation of E_total = anchor + lateral + contact_face_on.

    Used for distance-preserving and for any non-face-on geometry that
    graph-Laplacian's closed-form cannot handle directly.

    The Jacobian is computed inline with explicit signs (rather than
    reusing the ``*_force`` helpers) to avoid sign-convention drift
    between the graph-Laplacian and distance-preserving helpers.  Each
    contribution below corresponds to one term of dE/d delta:

        dE_anchor/d delta_i               = +ka * delta_i
        dE_lat_GL/d delta_i               = +kl * Sum_j (delta_i - delta_j)
        dE_lat_DP/d delta_i (edge (i,j))  = +kl * (l - L) * e_hat
                                            where l = ||q_j - q_i||, L = ||p_j - p_i||,
                                            e_hat = (q_j - q_i) / l
        dE_contact/d delta_k              = -kc * phi_eff * smooth_step(raw, eps) * n_k
                                            (face-on contact at sphere k;
                                             smooth_step is the C^infinity
                                             Heaviside surrogate, equal to
                                             sigma_eps'(raw); forgetting it
                                             gives a gradient up to 2x
                                             wrong at raw ~ eps)
    """
    if delta0 is None:
        delta0 = np.zeros((lat.N, 3))

    def fun(x: np.ndarray) -> float:
        d = x.reshape(lat.N, 3)
        E = anchor_energy(lat, d)
        if lateral == "graph_laplacian":
            E += lateral_energy_graph_laplacian(lat, d)
        elif lateral == "distance_preserving":
            E += lateral_energy_distance_preserving(lat, d)
        else:
            raise ValueError(f"unknown lateral: {lateral!r}")
        E += contact_energy_face_on(lat, target, d, phi_rest, eps=eps)
        return E

    def jac(x: np.ndarray) -> np.ndarray:
        d = x.reshape(lat.N, 3)
        g = np.zeros_like(d)
        # Anchor.
        g += lat.ka * d
        # Lateral.
        if lateral == "graph_laplacian":
            for (i, j) in lat.edges:
                diff = d[i] - d[j]
                g[i] += lat.kl * diff
                g[j] -= lat.kl * diff
        elif lateral == "distance_preserving":
            for (i, j) in lat.edges:
                q_i = lat.p[i] - d[i]
                q_j = lat.p[j] - d[j]
                v = q_j - q_i
                l = float(np.linalg.norm(v))
                if l < 1e-15:
                    continue
                L_rest = float(np.linalg.norm(lat.p[j] - lat.p[i]))
                e_hat = v / l
                contrib = lat.kl * (l - L_rest) * e_hat
                g[i] += contrib
                g[j] -= contrib
        else:
            raise ValueError(f"unknown lateral: {lateral!r}")
        # Contact (face-on).  raw = phi_rest - delta_n.  Smooth surrogate:
        #   phi_eff = sigma_eps(raw) = 0.5 (raw + sqrt(raw^2 + eps^2))
        # E_contact = 0.5 kc phi_eff^2, dE/d delta_k = kc phi_eff sigma_eps'(raw) (-n_k).
        # The smooth_step factor is essential at raw ~ eps.
        k = target.sphere_idx
        delta_n = float(np.dot(d[k], lat.n[k]))
        raw = phi_rest - delta_n
        if eps <= 0.0:
            phi_eff = max(0.0, raw)
            step = 1.0 if raw > 0.0 else (0.5 if raw == 0.0 else 0.0)
        else:
            r2 = raw * raw + eps * eps
            phi_eff = 0.5 * (raw + np.sqrt(r2))
            step = 0.5 * (1.0 + raw / np.sqrt(r2))
        g[k] -= target.kc * phi_eff * step * lat.n[k]
        return g.reshape(-1)

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
    E = anchor_energy(lat, deltas)
    if lateral == "graph_laplacian":
        E += lateral_energy_graph_laplacian(lat, deltas)
    elif lateral == "distance_preserving":
        E += lateral_energy_distance_preserving(lat, deltas)
    else:
        raise ValueError(f"unknown lateral law: {lateral!r}")
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
    f = anchor_force_all(lat, deltas)
    if lateral == "graph_laplacian":
        f_lat = lateral_force_graph_laplacian(lat, deltas)
    elif lateral == "distance_preserving":
        f_lat = lateral_force_distance_preserving(lat, deltas)
    else:
        raise ValueError(f"unknown lateral law: {lateral!r}")
    # f_anchor pulls toward rest (positive sign in our convention).
    # f_lat is the lateral force ON each sphere (sign convention above).
    # External force f_ext is applied; at equilibrium anchor + lateral + ext = 0.
    return f - f_lat - f_ext


def solve_equilibrium_numerical(lat: Lattice,
                                f_ext: np.ndarray,
                                *,
                                lateral: LateralLaw = "distance_preserving",
                                delta0: np.ndarray | None = None,
                                tol: float = 1.0e-10,
                                ) -> tuple[np.ndarray, dict]:
    """General equilibrium by L-BFGS-B on E_total.

    Used for distance-preserving lateral (nonlinear) where a direct
    linear solve does not apply.  For graph-Laplacian it would just
    rediscover the linear solve, with overhead.
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
    "ContactTarget",
    "make_chain",
    "make_arc",
    "build_K_matrix",
    "chain_analytical_eigenvalues",
    "chain_discrete_decay_length",
    "contact_energy_face_on",
    "contact_force_face_on",
    "solve_chain_contact_linear",
    "solve_lattice_contact_linear",
    "solve_lattice_contact_numerical",
    "anchor_force_all",
    "anchor_energy",
    "lateral_force_graph_laplacian",
    "lateral_energy_graph_laplacian",
    "lateral_force_distance_preserving",
    "lateral_energy_distance_preserving",
    "solve_equilibrium_graph_laplacian",
    "solve_equilibrium_numerical",
    "total_energy",
    "total_force",
]
