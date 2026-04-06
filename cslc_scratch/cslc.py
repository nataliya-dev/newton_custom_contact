"""
cslc.py — Compliant Sphere Lattice Contact: Core Module

The sphere lattice models the ROBOT FINGER PAD (soft, deformable).
The object is rigid — its surface spheres have no springs.

Building blocks:
  1. Pad creation  — 1-D chain or 2-D grid of spring-connected spheres
  2. Graph Laplacian — encodes which spheres share lateral springs
  3. Stiffness matrix — K = ka·I + kl·L, precomputed once per finger
  4. Solvers — linear (applied force) and nonlinear (contact)

All spatial quantities are in SI (metres, Newtons).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ═══════════════════════════════════════════════════════════════════════
#  PAD CREATION
#
#  A "pad" is a collection of spheres with a neighbor graph.
#  The neighbor graph determines the lateral spring topology.
#  For a 1-D chain: 2-connectivity (left/right).
#  For a 2-D grid:  4-connectivity (up/down/left/right).
#
#  These live on the finger's inner face.  The object is separate.
# ═══════════════════════════════════════════════════════════════════════

def create_pad_1d(n, spacing=0.002):
    """
    1-D chain of n spheres for pedagogical demonstrations.

    Spheres are placed along the y-axis at y = 0, dy, 2dy, ...
    Each sphere connects to its immediate left/right neighbor.
    """
    radius = spacing * 0.45
    positions = np.arange(n, dtype=float) * spacing
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        if i > 0:
            neighbors[i].append(i - 1)
        if i < n - 1:
            neighbors[i].append(i + 1)
    degrees = np.array([len(nb) for nb in neighbors], dtype=float)
    return dict(
        n_spheres=n, positions=positions, neighbors=neighbors,
        degrees=degrees, spacing=spacing, radius=radius,
        kind='1d', shape=(n,),
    )


def create_pad_2d(ny, nz, spacing=0.002):
    """
    2-D grid finger pad: ny × nz spheres with 4-connectivity.

    This is the standard CSLC finger pad.  Spheres sit on the
    finger's inner face (the YZ plane).  Contact happens along x.

    4-connectivity: each sphere connects to grid neighbors
    (up, down, left, right) — NOT diagonals, NOT depth.
    """
    radius = spacing * 0.45
    ns = ny * nz
    positions = np.zeros((ns, 2))
    indices = np.zeros((ns, 2), dtype=int)
    neighbors = [[] for _ in range(ns)]

    def idx(iy, iz):
        return iy * nz + iz

    for iy in range(ny):
        for iz in range(nz):
            i = idx(iy, iz)
            positions[i] = [iy * spacing, iz * spacing]
            indices[i] = [iy, iz]
            if iy > 0:
                neighbors[i].append(idx(iy - 1, iz))
            if iy < ny - 1:
                neighbors[i].append(idx(iy + 1, iz))
            if iz > 0:
                neighbors[i].append(idx(iy, iz - 1))
            if iz < nz - 1:
                neighbors[i].append(idx(iy, iz + 1))

    degrees = np.array([len(nb) for nb in neighbors], dtype=float)
    return dict(
        n_spheres=ns, positions=positions, indices=indices,
        neighbors=neighbors, degrees=degrees,
        ny=ny, nz=nz, spacing=spacing, radius=radius,
        kind='2d', shape=(ny, nz),
    )


# ═══════════════════════════════════════════════════════════════════════
#  GRAPH LAPLACIAN  &  STIFFNESS MATRIX
#
#  L encodes the lateral spring topology:
#    L_ii = degree(i)     — how many neighbors sphere i has
#    L_ij = -1            — if i and j are connected
#
#  Applied to displacements:  [L·δ]_i = Σ_{j∈N(i)} (δ_i − δ_j)
#
#  The stiffness matrix  K = k_a·I + k_ℓ·L  combines:
#    - Anchor springs (pull each sphere to its rest position)
#    - Lateral springs (couple each sphere to its neighbors)
#
#  K depends ONLY on finger pad geometry and spring constants.
#  It is PRECOMPUTED ONCE and reused every timestep.
#  For any finger shape, you build the neighbor graph, then K.
# ═══════════════════════════════════════════════════════════════════════

def graph_laplacian(pad):
    """Sparse graph Laplacian L ∈ ℝ^{n_s × n_s}."""
    ns = pad['n_spheres']
    rows, cols, vals = [], [], []
    for i in range(ns):
        nb = pad['neighbors'][i]
        rows.append(i)
        cols.append(i)
        vals.append(float(len(nb)))
        for j in nb:
            rows.append(i)
            cols.append(j)
            vals.append(-1.0)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(ns, ns))


def stiffness_matrix(pad, ka, kl):
    """Precomputed stiffness  K = k_a·I + k_ℓ·L  (sparse)."""
    L = graph_laplacian(pad)
    return ka * sparse.eye(pad['n_spheres'], format='csr') + kl * L


# ═══════════════════════════════════════════════════════════════════════
#  SOLVERS
# ═══════════════════════════════════════════════════════════════════════

def solve_applied_force(pad, ka, kl, f_applied):
    """
    Direct linear solve:  K · δ = f_applied.

    Use when the external force is KNOWN and FIXED (not
    dependent on displacement).  Pedagogical case:
    "push one sphere and see how the lattice responds."

    Parameters
    ----------
    f_applied : (n_s,) force on each sphere (one component)

    Returns  δ : (n_s,) displacement of each sphere
    """
    K = stiffness_matrix(pad, ka, kl)
    return spsolve(K, f_applied)


def solve_contact_flat(pad, ka, kl, kc, nominal_pen, n_iter=80,
                       alpha=0.25):
    """
    Quasi-static solve: pad pressing a flat rigid surface.

    nominal_pen : how far the finger body has advanced past
                  the surface (metres).  Positive = overlap.

    Each sphere's penetration depends on its displacement:
        pen_i = max(0, nominal_pen − δ_x_i)
    Contact force pushes the sphere in +x (away from surface):
        f_contact_i = k_c · pen_i

    At equilibrium:  K · δ_x = f_contact(δ_x)   [nonlinear]
    Solved by damped Jacobi, vectorised via sparse L.
    """
    ns = pad['n_spheres']
    L = graph_laplacian(pad)
    degrees = pad['degrees']
    delta_x = np.zeros(ns)
    res_hist = []

    for _ in range(n_iter):
        pen = np.maximum(nominal_pen - delta_x, 0.0)
        f_contact = kc * pen
        L_delta = L @ delta_x
        residual = f_contact - ka * delta_x - kl * L_delta
        eff_denom = ka + kl * degrees + kc * (pen > 0).astype(float)
        delta_x += alpha * residual / eff_denom
        res_hist.append(float(np.max(np.abs(residual))))

    pen = np.maximum(nominal_pen - delta_x, 0.0)
    forces = kc * pen
    active = forces > 1e-8
    return dict(
        delta_x=delta_x, forces=forces, pen=pen,
        n_contacts=int(np.sum(active)),
        total_force=float(np.sum(forces)),
        max_force=float(np.max(forces)) if np.any(active) else 0.0,
        residual_history=res_hist,
    )


def solve_contact_local(pad, ka, kl, kc, pen_per_sphere,
                        n_iter=80, alpha=0.25, warm_start=None):
    """
    Like solve_contact_flat but with PER-SPHERE nominal penetration.

    Use for non-uniform contact: curved indenter touching only the
    pad centre, or a tilted surface.

    pen_per_sphere : (n_s,) — undeformed penetration per sphere.
    warm_start     : (n_s,) or None — previous delta for faster convergence.
    """
    ns = pad['n_spheres']
    L = graph_laplacian(pad)
    degrees = pad['degrees']
    delta_x = np.zeros(ns) if warm_start is None else warm_start.copy()
    res_hist = []

    for _ in range(n_iter):
        pen = np.maximum(pen_per_sphere - delta_x, 0.0)
        f_contact = kc * pen
        L_delta = L @ delta_x
        residual = f_contact - ka * delta_x - kl * L_delta
        eff_denom = ka + kl * degrees + kc * (pen > 0).astype(float)
        delta_x += alpha * residual / eff_denom
        res_hist.append(float(np.max(np.abs(residual))))

    pen = np.maximum(pen_per_sphere - delta_x, 0.0)
    forces = kc * pen
    active = forces > 1e-8
    return dict(
        delta_x=delta_x, forces=forces, pen=pen,
        n_contacts=int(np.sum(active)),
        total_force=float(np.sum(forces)),
        max_force=float(np.max(forces)) if np.any(active) else 0.0,
        residual_history=res_hist,
    )


# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════

def row_profile_2d(pad, values, fix_axis, fix_index):
    """
    Extract values along one row of a 2-D pad.

    fix_axis  : 0 (fix iy, vary iz) or 1 (fix iz, vary iy)
    fix_index : which row/column to extract
    Returns (coords, vals) along the free axis.
    """
    vary_axis = 1 - fix_axis
    n_vary = pad['shape'][vary_axis]
    coords = np.zeros(n_vary)
    vals = np.zeros(n_vary)
    for i in range(pad['n_spheres']):
        ix = pad['indices'][i]
        if ix[fix_axis] == fix_index:
            j = ix[vary_axis]
            coords[j] = pad['positions'][i, vary_axis]
            vals[j] = values[i]
    return coords, vals
