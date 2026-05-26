# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC data structures: lattice geometry, topology, and GPU upload.

CSLCLattice: CPU-side geometry and neighbor topology for one rigid body's
worth of compliant-skin lattice spheres.  Generic in geometry: the caller
provides positions / normals / edges from whatever sampling pipeline
they choose (e.g. Poisson-disc Lloyd CVT on a mesh in
``cslc_main/grasp``); no shape-specific helpers live here.

CSLCData: merged GPU arrays for all lattices in a simulation.

File location: newton/_src/geometry/cslc_data.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from ..core.types import Devicelike


# ═══════════════════════════════════════════════════════════════════════════
#  CSLCLattice — CPU-side lattice geometry for one rigid body
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CSLCLattice:
    """Compliant-skin lattice for one rigid body shape.

    Generic in geometry: callers construct ``CSLCLattice`` from whatever
    sampling pipeline they use (e.g. Poisson-disc on a mesh).  This
    module ships no shape-specific generators -- the box-grid helpers
    that used to live here were removed in step 7 (see
    ``cslc_main/theory/notes.md``).

    Attributes:
        positions: (N, 3) float32 -- sphere centres in shape-local frame.
        radii: (N,) float32 -- sphere radii.
        is_surface: (N,) bool -- True for spheres that participate in contact.
        outward_normals: (N, 3) float32 -- outward-pointing normal for surface
            spheres. Interior spheres have (0, 0, 0). Used as the displacement
            direction in the Jacobi solve and as the local-frame basis for
            the anisotropic-anchor decomposition in contact writing.
        neighbor_indices: list of arrays -- per-sphere neighbour index lists
            (one entry per sphere; each entry is an int32 numpy array of
            neighbour sphere indices, local to this lattice).
        shape_index: int -- which shape in the Model this lattice belongs to.
        spacing: float -- characteristic distance between sphere centres [m].
            Bookkeeping field used by stiffness calibration; not required by
            the kernels.
        sphere_radius: float -- radius of each lattice sphere [m].
    """

    positions: np.ndarray
    radii: np.ndarray
    is_surface: np.ndarray
    outward_normals: np.ndarray
    neighbor_indices: list[np.ndarray]
    shape_index: int
    spacing: float
    sphere_radius: float

    @property
    def n_spheres(self) -> int:
        return len(self.positions)

    @property
    def n_surface(self) -> int:
        return int(self.is_surface.sum())


# ═══════════════════════════════════════════════════════════════════════════
#  Stiffness calibration
# ═══════════════════════════════════════════════════════════════════════════


def calibrate_kc(
    ke_bulk: float,
    lattices: list[CSLCLattice],
    *,
    ka: float,
    contact_fraction: float = 0.3,
    per_lattice: bool = True,
    ke_target: float | None = None,
) -> float:
    """Derive per-sphere contact stiffness kc from bulk ke.

    The fair invariant is the per-lattice aggregate normal stiffness at the
    operating penetration, composed across all springs in series:

    - **Two-spring chain (rigid target, ``ke_target=None``):** anchor
      ``ka`` in series with contact ``kc``.  Per-sphere effective
      stiffness ``keff = ka*kc/(ka+kc)``.  Inverting the aggregate
      identity ``N_contact * keff = ke_bulk`` for ``kc`` gives
      ``kc = ke_bulk * ka / (N_contact * ka - ke_bulk)``.

    - **Three-spring chain (compliant target, ``ke_target`` provided):**
      anchor ``ka`` in series with contact ``kc`` in series with target
      modulus ``ke_target``.  The harmonic-mean composition of contact
      and target moduli is exactly the series-spring law (Masterjohn
      et al. 2021, PFC-V eq. 23; Castro et al. 2022, SAP) that the
      ``write_cslc_contacts`` kernel emits as ``k_series``.  Per-sphere
      effective stiffness becomes ``1/keff = 1/ka + 1/kc + 1/ke_target``.
      Solving for ``kc`` against the same aggregate identity:
      ``1/kc = N_contact/ke_bulk - 1/ka - 1/ke_target``.

    .. note:: Step 7 calibration alignment

       The series-spring identity above is now an EXACT match for the
       kernel's contact law (post-D4 deformed-centre emission +
       smooth_step gradient factor in ``jacobi_step``).  At saturated
       contact the kernel emits per-contact force
       ``F = keff * phi_rest = ka*kc/(ka+kc) * phi_rest``, which is what
       this formula already targets.  No kc migration is needed for the
       auto-calibrated production path -- the calibration formula was
       always solving the series-spring system; the pre-D4 kernel was
       the broken side, applying ``kc * phi_rest`` (constant load) and
       agreeing numerically only in the ``kc << ka`` regime.  Hand-tuned
       scenes that set ``kc`` to match a specific force under the old
       law would need ``kc_new = kc_old * (ka + kc_old) / ka`` to
       approximately restore that force; this helper is exempt.

    Args:
        ke_bulk: target per-lattice aggregate stiffness [N/m].
        lattices: list of CSLCLattices (used to count surface spheres).
        ka: anchor stiffness [N/m].
        contact_fraction: estimated active-contact fraction.
        per_lattice: if True, each lattice must independently aggregate to
            ke_bulk; if False, the calibration matches the sum across
            lattices.
        ke_target: if provided, include the target body's contact
            stiffness in the series chain.  Use ``None`` (default) for
            the legacy rigid-target calibration.

    Returns:
        Per-sphere contact stiffness ``kc`` [N/m].  Falls back to
        ``ke_bulk / N_contact`` when the analytic formula has no
        positive solution (under-stiff anchor or under-stiff target).
    """
    if per_lattice:
        # Average n_surface across lattices — assumes lattices are roughly uniform
        # in size. For mixed lattice sizes, promote to per-shape kc storage
        # in CSLCData.
        n_surface = int(np.mean([p.n_surface for p in lattices]))
    else:
        n_surface = sum(p.n_surface for p in lattices)
    n_contact = max(int(n_surface * contact_fraction), 1)

    # Per-sphere effective stiffness target.
    inv_keff_target = float(n_contact) / ke_bulk

    # Subtract the anchor's contribution (1/ka) and, if provided, the
    # target's contribution (1/ke_target). Remaining budget is 1/kc.
    inv_kc = inv_keff_target - 1.0 / ka
    if ke_target is not None and ke_target > 0.0:
        inv_kc -= 1.0 / ke_target

    if inv_kc <= 0.0:
        # The anchor (and/or target) alone is too soft to reach the
        # per-sphere effective stiffness with any kc; fall back to a
        # conservative per-sphere stiffness.
        ka_min = ke_bulk / float(n_contact)
        if ke_target is not None and ke_target > 0.0:
            inv_term = 1.0 / ka_min - 1.0 / ke_target
            ka_min = (1.0 / inv_term) if inv_term > 0.0 else float("inf")
        import warnings as _warnings
        _warnings.warn(
            f"calibrate_kc: anchor ka={ka:.3g} too soft for "
            f"ke_bulk={ke_bulk:.3g} at N_contact={n_contact} "
            + (f"(ke_target={ke_target:.3g}) " if ke_target else "")
            + f"-- analytic series-spring has no positive kc solution. "
            f"Falling back to kc=ke_bulk/N_contact={ke_bulk / n_contact:.3g}. "
            f"To use the analytic formula, raise ka to > {ka_min:.3g}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ke_bulk / max(n_contact, 1)
    return 1.0 / inv_kc


# ═══════════════════════════════════════════════════════════════════════════
#  CSLCData — merged GPU arrays
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CSLCData:
    """GPU-resident CSLC lattice data, merged from one or more CSLCLattices.

    Neighbor lookups use CSR format:
        sphere i's neighbors are
        neighbor_list[neighbor_start[i] : neighbor_start[i] + neighbor_count[i]]
    """

    n_spheres: int
    n_surface: int
    positions: wp.array       # (n_spheres,) vec3 — shape-local rest positions
    radii: wp.array           # (n_spheres,) float32
    is_surface: wp.array      # (n_spheres,) int32 — 1 = surface, 0 = interior
    outward_normals: wp.array # (n_spheres,) vec3 — outward normal for surface spheres
    sphere_shape: wp.array    # (n_spheres,) int32 — shape index per sphere
    # (n_spheres,) vec3 — converged displacement of each lattice sphere
    # from its rest position in world frame.  Deformed sphere centre is
    # q_i = p_i_world - sphere_delta[i].  Carries the full 3-D
    # displacement -- no scalar projection / outward-normal shim -- so
    # tangential components survive across iterations (required for
    # off-axis contact and stick-slip friction).  Persisted across
    # steps to warm-start the next collide().
    sphere_delta: wp.array
    ka: float
    kl: float
    kc: float
    dc: float
    neighbor_start: wp.array  # (n_spheres,) int32 — CSR row pointer
    neighbor_count: wp.array  # (n_spheres,) int32
    neighbor_list: wp.array   # (n_edges,) int32
    # Smoothing width [m] for the differentiable surrogates of `[·]_+` and
    # the contact-active gates in cslc_kernels.py.  eps → 0 recovers the
    # original non-smooth behaviour; default 1e-5 m is essentially binary
    # above 0.1 mm penetration with C^∞ derivatives at the threshold so
    # wp.Tape can backprop through CSLC contact dynamics.
    smoothing_eps: float = 1.0e-5
    # Anisotropic anchor stiffness (theory step 6).  `ka` is the NORMAL
    # (out-of-plane) anchor stiffness -- resistance to compression
    # along the rest outward normal.  `ka_tangent_ratio` scales it for
    # the two in-plane (tangential) axes: ka_t = ka_tangent_ratio · ka.
    # The flesh-like default (1/3) follows from a nearly-incompressible
    # isotropic elastic medium (Poisson ν → 0.5), where shear modulus
    # G = E / (2(1+ν)) → E/3.  Set to 1.0 for isotropic anchor.  Set
    # to 0.0 for "frictionless tangent" (lattice spheres free to slide
    # laterally, restrained only by lateral coupling).  Any positive
    # value is supported by both the closed-form warm-start
    # (lattice_solve_equilibrium uses A_inv_t for tangent solves) and
    # the iterative jacobi_step.
    ka_tangent_ratio: float = 1.0
    # Stick-slip friction via tangential δ (theory step 4 + 6).  The
    # compliant skin develops a tangential displacement
    # δ_t = δ − dot(δ, n_eff)·n_eff under shear loading; this generates
    # a friction-like force f_t = -k_stick · δ_t clamped at the
    # Coulomb cone ‖f_t‖ ≤ μ·f_n.  In stick mode (proposed ≤ cone) the
    # force pulls the contact patch back toward zero shear; in slip
    # mode (proposed > cone) the magnitude saturates at the cone
    # boundary.  Models real flesh: static friction via shear
    # deformation BEFORE macroscopic slip.
    #
    # Default k_stick = ka so the tangent shear-stiffness matches the
    # anchor's normal stiffness (reasonable starting point for
    # flesh-like calibration; with ka_tangent_ratio = 1/3 the geometric
    # tangent anchor is 3× softer than k_stick, so friction dominates
    # static restraint over anchor in the tangent plane).  Set
    # k_stick = 0 (or mu_friction = 0) to disable friction entirely.
    k_stick: float = 25000.0
    # Coulomb friction coefficient applied to the cone clamp.  Read at
    # construction from the active CSLC shape's material friction in
    # `_from_model`; falls back to the dataclass default for tests that
    # construct CSLCData directly.  This is the friction USED BY THE
    # LATTICE SOLVER for the stick-slip restraint — it does NOT replace
    # MuJoCo's rigid-body friction (which still acts at the macroscopic
    # contact, with the geom-pair μ from `shape_material_mu`).
    mu_friction: float = 0.3
    # Dense inverses of the lattice system matrices, applied per-axis in
    # each sphere's local rest-normal frame.  Built only when
    # `build_A_inv=True` is passed to `from_lattices`; consumed by
    # `lattice_solve_equilibrium` as the closed-form linear warm-start
    # before damped Jacobi refines.
    #
    # A_inv_n = (K_n + kc·I)^-1
    #     where K_n is the SPD lattice Laplacian assembled with the NORMAL
    #     anchor stiffness on the diagonal: K_n_ii = ka + kl·|N(i)|,
    #     K_n_ij = -kl if j ∈ N(i).  The +kc·I term captures the contact
    #     spring in the saturated-active limit (paper eq. 12); inactive
    #     spheres have phi ≈ 0 so their contribution stays near zero.
    #     Applied to the scalar normal-axis force field
    #     f_n_j = phi[j] · dot(n_eff_j, n_outward_j).
    #
    # A_inv_t = (K_t)^-1
    #     where K_t = ka·ratio·I + kl·L (no kc -- the contact spring
    #     acts along n_eff and is absorbed into the normal axis).
    #     Applied to the tangent-axis force vec3 field
    #     f_t_j = phi[j] · (n_eff_j − dot(n_eff_j, n_outward_j)·n_outward_j).
    #
    # Both are SPD for ka > 0 and ka·ratio > 0; np.linalg.inv is fine at
    # the n we use.  For n ≳ 1000 the right follow-up is a sparse
    # Cholesky factorisation + triangular tri-solve kernels (one per
    # axis class), which would also bound the memory growth from O(n²)
    # to O(n·avg_neigh).  ``A_inv`` is the public attribute name for the
    # normal-axis matrix; ``A_inv_t`` for the tangent.
    A_inv: wp.array | None = None
    A_inv_t: wp.array | None = None
    device: str | None = None

    @classmethod
    def from_lattices(
        cls, lattices: list[CSLCLattice], *, ka: float, kl: float, kc: float,
        dc: float, smoothing_eps: float = 1.0e-5,
        ka_tangent_ratio: float = 1.0,
        k_stick: float | None = None,
        mu_friction: float | None = None,
        build_A_inv: bool = False,
        kl_physical: float | None = None,
        device: Devicelike | None = None,
    ) -> CSLCData:
        """Merge CSLCLattices into GPU-resident CSLCData with global indexing.

        Args:
            lattices: one or more CSLCLattices to merge (assumed uniform material).
            ka, kl, kc, dc: spring constants; see CSLCData docstring.
            smoothing_eps: differentiability width for kernel gates.
            build_A_inv: if True, precompute the dense inverse of the
                lattice system matrix A = K + kc·I and store in A_inv
                + A_inv_t for use by the tape-compatible
                ``lattice_solve_equilibrium`` kernel (the linear
                warm-start that runs before ``jacobi_step`` refines).
                Default ``False`` keeps the signature lean for unit
                tests; production callers (``cslc_main/grasp``) pass
                ``True`` so the closed-form warm-start absorbs most
                of the per-step displacement and the iterative
                refinement converges in a handful of iterations.
            kl_physical: optional resolution-independent lateral stiffness
                in continuous-PDE units [N·m].  When provided, the kernel-
                level ``kl`` is replaced with ``kl_physical / spacing²``
                so the discrete Helmholtz operator
                ``ka·I − (kl_physical/h²)·L_graph``  approximates the
                continuous operator ``ka·I − kl_physical·∇²`` consistently
                across refinement.  The induced lateral correlation length
                in physical units is ``ℓ_c = √(kl_physical / ka)``,
                invariant to grid spacing.  When ``None`` (default),
                ``kl`` is used directly and the correlation length in
                lattice-spacing units is ``√(kl/ka)``, which shrinks under
                refinement (theory step 2 documents the discrete vs
                continuum decay-length subtlety; see
                ``cslc_main/theory/notes.md``).  Assumes uniform spacing
                across all lattices (uses ``lattices[0].spacing``); a
                future per-lattice extension would index by lattice.
        """
        # Resolution-independent lateral coupling (Fix 1.1).  When
        # ``kl_physical`` is provided, override the kernel-facing ``kl``
        # so that the same continuous PDE is consistently discretised at
        # any spacing.
        if kl_physical is not None:
            if not lattices:
                raise ValueError(
                    "kl_physical requires at least one lattice to read spacing from"
                )
            spacing = float(lattices[0].spacing)
            if spacing <= 0.0:
                raise ValueError(
                    f"kl_physical requires positive spacing; got {spacing}"
                )
            kl = float(kl_physical) / (spacing * spacing)
        if device is None:
            device = wp.get_device()

        offsets = []
        offset = 0
        for lattice in lattices:
            offsets.append(offset)
            offset += lattice.n_spheres
        n_total = offset

        all_pos = np.zeros((n_total, 3), dtype=np.float32)
        all_radii = np.zeros(n_total, dtype=np.float32)
        all_surface = np.zeros(n_total, dtype=np.int32)
        all_normals = np.zeros((n_total, 3), dtype=np.float32)
        all_shape = np.zeros(n_total, dtype=np.int32)

        for lattice, off in zip(lattices, offsets):
            sl = slice(off, off + lattice.n_spheres)
            all_pos[sl] = lattice.positions
            all_radii[sl] = lattice.radii
            all_surface[sl] = lattice.is_surface.astype(np.int32)
            all_normals[sl] = lattice.outward_normals
            all_shape[sl] = lattice.shape_index

        # CSR neighbor structure.  Per-edge rest-lengths were removed
        # (2026-05-24) when the production kernel switched from the
        # distance-preserving lateral spring to the linear graph-
        # Laplacian; see ``jacobi_step`` in cslc_kernels.py.
        all_start = np.zeros(n_total, dtype=np.int32)
        all_count = np.zeros(n_total, dtype=np.int32)
        neighbor_lists = []
        edge_offset = 0

        for lattice, glob_off in zip(lattices, offsets):
            for local_i, neighbors in enumerate(lattice.neighbor_indices):
                global_i = glob_off + local_i
                all_start[global_i] = edge_offset
                all_count[global_i] = len(neighbors)
                neighbor_lists.append(neighbors + glob_off)
                edge_offset += len(neighbors)

        all_neighbor_list = (
            np.concatenate(neighbor_lists).astype(np.int32)
            if neighbor_lists else np.zeros(0, dtype=np.int32)
        )

        # Build TWO dense inverses for the closed-form lattice solve,
        # one per axis class in the per-sphere local rest-normal frame:
        #   A_inv_n = (K_n + kc·I)^-1   with K_n_ii = ka + kl·|N(i)|
        #   A_inv_t = (K_t)^-1          with K_t_ii = ka·ratio + kl·|N(i)|
        # The off-diagonal Laplacian coupling (-kl on edges) is
        # identical for both axes -- only the diagonal anchor stiffness
        # differs.  Splitting into per-axis matrices supports anisotropic
        # anchors (ka_tangent_ratio ≠ 1) in closed form without falling
        # back to iterative Jacobi for the warm-start, and keeps the
        # matvec tape-compatible (the iterative path's src/dst aliasing
        # breaks wp.Tape backward).
        #
        # When ratio == 1.0 (isotropic default) the two matrices differ
        # only by the +kc·I term: A_inv_n absorbs the contact spring;
        # A_inv_t does not (the contact force is along n_eff, projected
        # entirely onto the normal axis in the decomposition kernel --
        # see `lattice_solve_equilibrium`).
        A_inv_wp = None
        A_inv_t_wp = None
        if build_A_inv:
            if ka_tangent_ratio <= 0.0:
                raise ValueError(
                    "ka_tangent_ratio must be > 0 for the closed-form solve "
                    "(K_t = ka·ratio·I + kl·L would be singular when "
                    "ratio = 0 because kl·L has a constant nullspace). "
                    f"Got ka_tangent_ratio={ka_tangent_ratio}."
                )
            # Assemble the graph Laplacian L once; both K_n and K_t share
            # the same off-diagonal structure with diagonals = |N(i)|.
            L = np.zeros((n_total, n_total), dtype=np.float64)
            for lattice, glob_off in zip(lattices, offsets):
                for local_i, neighbors in enumerate(lattice.neighbor_indices):
                    gi = int(glob_off + local_i)
                    L[gi, gi] = float(len(neighbors))
                    for nb in neighbors:
                        gj = int(glob_off + int(nb))
                        L[gi, gj] = -1.0

            I_n = np.eye(n_total, dtype=np.float64)
            # Normal axis: ka·I + kl·L + kc·I.
            A_n = ka * I_n + kl * L + kc * I_n
            A_inv_np = np.linalg.inv(A_n).astype(np.float32)
            A_inv_wp = wp.array(A_inv_np, dtype=wp.float32, device=device)
            # Tangent axes: ka·ratio·I + kl·L (no contact spring).
            A_t = (ka * ka_tangent_ratio) * I_n + kl * L
            A_inv_t_np = np.linalg.inv(A_t).astype(np.float32)
            A_inv_t_wp = wp.array(A_inv_t_np, dtype=wp.float32, device=device)

        return cls(
            n_spheres=n_total,
            n_surface=int(all_surface.sum()),
            positions=wp.array(all_pos, dtype=wp.vec3, device=device),
            radii=wp.array(all_radii, dtype=wp.float32, device=device),
            is_surface=wp.array(all_surface, dtype=wp.int32, device=device),
            outward_normals=wp.array(all_normals, dtype=wp.vec3, device=device),
            sphere_shape=wp.array(all_shape, dtype=wp.int32, device=device),
            sphere_delta=wp.zeros(n_total, dtype=wp.vec3, device=device),
            ka=ka, kl=kl, kc=kc, dc=dc,
            neighbor_start=wp.array(all_start, dtype=wp.int32, device=device),
            neighbor_count=wp.array(all_count, dtype=wp.int32, device=device),
            neighbor_list=wp.array(all_neighbor_list, dtype=wp.int32, device=device),
            smoothing_eps=smoothing_eps,
            ka_tangent_ratio=ka_tangent_ratio,
            k_stick=k_stick if k_stick is not None else ka,
            mu_friction=mu_friction if mu_friction is not None else 0.3,
            A_inv=A_inv_wp,
            A_inv_t=A_inv_t_wp,
            device=device,
        )