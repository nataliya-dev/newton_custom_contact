# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Theory-side bridge shim for the kernel-vs-theory verification.

This module is the **gold reference** the Warp kernels in
``newton/_src/geometry/cslc_kernels.py`` will be measured against in
step 7.  It takes a Newton-shaped scene description (CSLC lattice
sphere positions, edges, normals, target sphere, material params) and
solves the same quasi-static equilibrium the kernel should converge
to -- using the theory's pure-numpy primitives.

Two solver paths are exposed, dispatched on lattice size:

* **Single sphere (N == 1)** -- ``cslc_theory.equilibrium_numerical``
  and friends.  Optional tangential external load wires through to
  ``equilibrium_with_friction_smooth_numerical``.

* **Multi-sphere (N >= 2)** -- ``cslc_lattice.solve_lattice_contact_numerical``
  with ``lateral='distance_preserving'``.  The contact target must
  overlap exactly one lattice sphere at rest (the bridge tests are
  designed that way; we error out if multiple spheres are in contact).

Output: per-sphere equilibrium ``delta`` plus per-sphere contact
force, in the SAME world frame the kernel reports.  Both are the
direct theory values; the kernel will be compared against them in
``test_07_kernel_bridge.py``.

Sign convention (matching the theory module): ``q = p - delta``,
``f_contact = kc * phi_eff * e_hat_def`` with ``e_hat_def = (q - t) /
||q - t||``.  Forgetting the ``smooth_step`` factor on the gradient
side is the silent bug fixed in step 4 audit pass 1 -- the bridge
inherits the correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scipy.optimize import minimize

from cslc_main.theory.cslc_lattice import (
    ContactTarget,
    Lattice,
    anchor_energy,
    lateral_energy_distance_preserving,
    solve_lattice_contact_numerical,
)
from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    RigidTarget,
    contact_raw_overlap,
    deformed_centre,
    effective_penetration,
    equilibrium_face_on_analytical,
    equilibrium_numerical,
    equilibrium_with_friction_smooth_numerical,
    rest_overlap,
    smooth_step,
)


# ─────────────────────────────────────────────────────────────────────────
#  Scene + solution dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class KernelScene:
    """Newton-shaped scene description shared by the kernel and the theory.

    All vectors are in the SAME world frame the kernel will report (no
    body transforms, no shape transforms -- the bridge tests are built
    at identity transforms so this is unambiguous).

    Attributes:
        positions: (N, 3) lattice sphere centres in world frame.
        radii: (N,) sphere radii.
        outward_normals: (N, 3) per-sphere outward unit normals (the
            local frame for the anisotropic anchor decomposition).
        neighbor_indices: list of length N; entry i is an int array of
            neighbour indices of sphere i.  Undirected edges (so j in
            N(i) iff i in N(j)).
        is_surface: (N,) bool; True if the sphere participates in
            contact.  Non-surface spheres are reserved for future
            volumetric-lattice scenes; the kernel zeros their contact
            force.  For step-7 bridge tests every sphere is a surface
            sphere.
        target_position: (3,) target sphere centre in world frame.
        target_radius: target sphere radius [m].
        ka, kl, kc: anchor, lateral, contact stiffnesses [N/m].
        ka_t_ratio: tangent / normal anchor ratio (step 6).
            Defaults to 1.0 (isotropic).
        k_stick: stick-slip friction stiffness [N/m].  0 disables
            friction.
        mu_friction: Coulomb friction coefficient.  0 disables.
        eps: smoothing width [m] for the differentiable surrogates.
            Match the kernel's eps (default 1e-5) when comparing.
        f_ext_tangent: optional external tangential force [N] applied
            to the lattice sphere (single-sphere scenes only -- the
            multi-sphere solver doesn't currently accept an external
            tangential load).
    """

    positions: np.ndarray
    radii: np.ndarray
    outward_normals: np.ndarray
    neighbor_indices: list[np.ndarray]
    is_surface: np.ndarray
    target_position: np.ndarray
    target_radius: float
    ka: float
    kl: float
    kc: float
    ka_t_ratio: float = 1.0
    k_stick: float = 0.0
    mu_friction: float = 0.0
    eps: float = 1.0e-5
    f_ext_tangent: np.ndarray | None = None

    @property
    def n_spheres(self) -> int:
        return int(self.positions.shape[0])

    def __post_init__(self) -> None:
        if self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (N, 3), got {self.positions.shape}")
        if self.outward_normals.shape != self.positions.shape:
            raise ValueError(
                "outward_normals must match positions shape "
                f"({self.positions.shape}); got {self.outward_normals.shape}")
        if self.radii.shape != (self.n_spheres,):
            raise ValueError(
                f"radii must be (N,), got {self.radii.shape}")
        if self.is_surface.shape != (self.n_spheres,):
            raise ValueError(
                f"is_surface must be (N,), got {self.is_surface.shape}")
        if self.target_position.shape != (3,):
            raise ValueError(
                f"target_position must be (3,), got {self.target_position.shape}")
        if len(self.neighbor_indices) != self.n_spheres:
            raise ValueError(
                f"neighbor_indices must have length N={self.n_spheres}, "
                f"got {len(self.neighbor_indices)}")
        if self.f_ext_tangent is not None and self.n_spheres != 1:
            raise ValueError(
                "f_ext_tangent is only supported for single-sphere scenes; "
                "the multi-sphere theory solver doesn't accept an external "
                "tangential load yet.")


@dataclass
class KernelSolution:
    """Per-sphere equilibrium delta + per-sphere contact force.

    Both are the THEORY values evaluated in the same world frame the
    scene was built in.  Used as the gold reference in the kernel
    bridge tests.

    Attributes:
        delta: (N, 3) equilibrium displacements (sign matches q = p - delta).
        contact_force: (N, 3) per-sphere contact force vec3 in world
            frame, evaluated AT the returned equilibrium.  Non-contact
            spheres carry zero.  This is the theory's
            ``kc * phi_eff(delta) * e_hat_def(delta)``; the kernel's
            output is compared against it in the bridge tests.
        contact_sphere_idx: which sphere is in contact, or -1 if none.
            Used by the bridge tests to assert force is concentrated.
        solver_info: misc per-solver diagnostics (regime, iteration
            count, converged flag, residual norm).
    """

    delta: np.ndarray
    contact_force: np.ndarray
    contact_sphere_idx: int
    solver_info: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
#  Public solvers
# ─────────────────────────────────────────────────────────────────────────


def solve_theory(scene: KernelScene) -> KernelSolution:
    """Solve the scene's quasi-static equilibrium via the theory primitives.

    Dispatches to the single-sphere solver for ``N == 1`` and to the
    lattice solver for ``N >= 2``.  Returns the gold-reference
    ``KernelSolution`` for kernel comparison.
    """
    if scene.n_spheres == 1:
        return _solve_single_sphere(scene)
    return _solve_lattice(scene)


def compute_contact_force(scene: KernelScene,
                          deltas: np.ndarray) -> np.ndarray:
    """Evaluate the theory's per-sphere contact force at a given delta.

    Independent of how ``deltas`` was obtained -- this is the function
    the kernel-bridge tests use to convert kernel-output delta into a
    per-contact force vector for vector comparison against the theory's
    contact force at the theory's own delta.

    Per sphere ``i``:

        e_hat_def = (q_i - t) / ||q_i - t||,  q_i = p_i - delta_i
        phi_eff   = smooth_relu(phi_rest - dot(delta_i, e_hat_def), eps)
        f_i       = kc * phi_eff * e_hat_def   if is_surface[i]
                  = 0                          otherwise

    The smooth_relu and the e_hat_def's dependence on delta together
    encode the deformed-centre / series-spring law that step 7's
    kernel-side rewrite is converging to.  Zero phi_eff means no
    contact (smoothed); the kernel should report the same.

    Args:
        scene: scene description.
        deltas: (N, 3) per-sphere displacements.

    Returns:
        (N, 3) per-sphere contact force in world frame.
    """
    if deltas.shape != (scene.n_spheres, 3):
        raise ValueError(
            f"deltas must be ({scene.n_spheres}, 3), got {deltas.shape}")
    f = np.zeros((scene.n_spheres, 3), dtype=np.float64)
    for i in range(scene.n_spheres):
        if not bool(scene.is_surface[i]):
            continue
        q = scene.positions[i] - deltas[i]
        diff = q - scene.target_position
        L = float(np.linalg.norm(diff))
        if L < 1.0e-15:
            continue
        e_hat = diff / L
        r_lat = float(scene.radii[i])
        raw = (r_lat + scene.target_radius) - L
        # Smooth positive part (matches phi_eff in cslc_theory).
        if scene.eps <= 0.0:
            phi_eff = max(0.0, raw)
        else:
            phi_eff = 0.5 * (raw + np.sqrt(raw * raw + scene.eps * scene.eps))
        f[i] = scene.kc * phi_eff * e_hat
    return f


# ─────────────────────────────────────────────────────────────────────────
#  Single-sphere dispatch
# ─────────────────────────────────────────────────────────────────────────


def _solve_single_sphere(scene: KernelScene) -> KernelSolution:
    p = scene.positions[0]
    n_hat = scene.outward_normals[0]
    r = float(scene.radii[0])
    # Local-frame sphere (origin at p, normal along n_hat).
    sphere = LatticeSphere(
        p=np.zeros(3), r=r, n=n_hat.astype(np.float64),
        ka=scene.ka, ka_t_ratio=scene.ka_t_ratio,
    )
    target = RigidTarget(
        t=(scene.target_position - p).astype(np.float64),
        R=float(scene.target_radius),
    )

    info: dict = {}
    if scene.f_ext_tangent is not None and scene.k_stick > 0.0 and scene.mu_friction > 0.0:
        # Stick-slip friction path (step 4 / step 6).
        delta_local, info_smooth = equilibrium_with_friction_smooth_numerical(
            sphere, target, scene.kc,
            scene.f_ext_tangent.astype(np.float64),
            k_stick=scene.k_stick, mu=scene.mu_friction,
            eps_contact=scene.eps, eps_friction=1.0e-12,
            tol=1.0e-12,
        )
        info.update(info_smooth)
        info["regime"] = "with_friction_smooth"
    else:
        # Pure normal contact: pick face-on closed form when applicable
        # (zero tangential offset of the target from the rest normal),
        # otherwise fall back to L-BFGS-B numerical.
        offset = target.t - sphere.p
        offset_n = float(np.dot(offset, sphere.n))
        offset_t = offset - offset_n * sphere.n
        face_on = float(np.linalg.norm(offset_t)) < 1.0e-9
        if face_on and scene.ka_t_ratio == 1.0 and scene.f_ext_tangent is None:
            delta_local, F = equilibrium_face_on_analytical(sphere, target, scene.kc)
            info["regime"] = "face_on_analytical"
            info["face_on_F"] = float(F)
        else:
            # Warm-start: project the analytic series-spring magnitude
            # onto the contact-line direction.  Falls back to zero if
            # the target is far from contact.
            phi_rest = rest_overlap(sphere, target)
            if phi_rest > 0:
                line = offset / max(float(np.linalg.norm(offset)), 1e-15)
                delta0 = (scene.kc / (scene.ka + scene.kc)) * phi_rest * line
            else:
                delta0 = np.zeros(3)
            delta_local, info_num = equilibrium_numerical(
                sphere, target, scene.kc,
                eps=scene.eps, delta0=delta0, tol=1.0e-12,
            )
            info.update(info_num)
            info["regime"] = "numerical"

    # Promote to (1, 3) array, transform back to world.  Both p and the
    # solver work in the same translated frame, so delta is invariant.
    deltas = delta_local[np.newaxis, :].astype(np.float64)
    f_contact = compute_contact_force(scene, deltas)
    contact_idx = 0 if bool(scene.is_surface[0]) and np.linalg.norm(f_contact[0]) > 0 else -1
    return KernelSolution(
        delta=deltas,
        contact_force=f_contact,
        contact_sphere_idx=contact_idx,
        solver_info=info,
    )


# ─────────────────────────────────────────────────────────────────────────
#  Multi-sphere dispatch
# ─────────────────────────────────────────────────────────────────────────


def _build_lattice(scene: KernelScene) -> Lattice:
    """Translate ``KernelScene`` into the theory's ``Lattice`` form.

    Edges are reconstructed from the per-sphere neighbour lists: for
    every ordered pair (i, j in N(i)) with i < j we emit one
    undirected edge.  This matches the theory's edge convention.
    """
    N = scene.n_spheres
    edge_set: set[tuple[int, int]] = set()
    for i in range(N):
        for j in scene.neighbor_indices[i]:
            jj = int(j)
            if jj == i:
                continue
            a, b = (i, jj) if i < jj else (jj, i)
            edge_set.add((a, b))
    edges = np.array(sorted(edge_set), dtype=np.int64) if edge_set \
        else np.zeros((0, 2), dtype=np.int64)
    return Lattice(
        p=scene.positions.astype(np.float64),
        n=scene.outward_normals.astype(np.float64),
        edges=edges,
        ka=scene.ka, kl=scene.kl,
    )


def _contact_overlaps(scene: KernelScene) -> list[tuple[int, float]]:
    """Return ``[(i, phi_rest_i), ...]`` for every surface sphere whose
    rest overlap with the target is positive.

    Unlike the validated theory steps 3/5 (which design the scene so
    only one sphere overlaps the target), realistic kernel scenes have
    a CONTACT PATCH of several lattice spheres all overlapping the
    same large target.  The bridge supports both cases:

    * single-overlap -> we can use the theory's
      ``solve_lattice_contact_numerical`` directly (matches steps 3/5).
    * multi-overlap -> we solve the full multi-contact energy
      ourselves via L-BFGS-B, reusing the theory's per-component
      energies (anchor + distance-preserving lateral) plus a
      per-sphere series-spring contact contribution.
    """
    matches: list[tuple[int, float]] = []
    R = float(scene.target_radius)
    for i in range(scene.n_spheres):
        if not bool(scene.is_surface[i]):
            continue
        r = float(scene.radii[i])
        dist = float(np.linalg.norm(scene.positions[i] - scene.target_position))
        phi = (r + R) - dist
        if phi > 0.0:
            matches.append((i, phi))
    return matches


def _solve_lattice(scene: KernelScene) -> KernelSolution:
    lat = _build_lattice(scene)
    overlaps = _contact_overlaps(scene)
    if not overlaps:
        return KernelSolution(
            delta=np.zeros((scene.n_spheres, 3), dtype=np.float64),
            contact_force=np.zeros((scene.n_spheres, 3), dtype=np.float64),
            contact_sphere_idx=-1,
            solver_info={"regime": "no_contact"},
        )

    # Step 7 / S7-4 finding: for kernel verification, route every lattice
    # scene through the multi-contact path so the theory applies the
    # SAME smooth gate at EVERY surface sphere that the kernel does.
    # The previous single-overlap path (`solve_lattice_contact_numerical`)
    # places a contact-energy term only at the apex sphere; the kernel
    # instead applies `kc * smooth_relu(phi_rest - dot(δ, n_eff), eps)`
    # at every surface sphere, which produces a `kc * eps / (2 * √(ka * (ka + kc)))`
    # leak at non-contact spheres.  Matching the theory's gate to the
    # kernel's removes that mismatch -- both still recover the bulge /
    # series-spring physics in the eps → 0 limit.  The bulge results
    # from step 5 are preserved because they use the same energy
    # functional in the inactive-gate limit (raw < 0, smooth_relu → 0).
    return _solve_lattice_multi_contact(scene, lat)


def _solve_lattice_single_overlap_legacy(scene: KernelScene,
                                         lat: Lattice,
                                         overlaps: list[tuple[int, float]]
                                         ) -> KernelSolution:
    """Original step 3 / step 5 single-overlap path, kept for archaeology.

    NOT used by the kernel bridge.  See `_solve_lattice` for why
    we always route through the multi-contact path now.
    """
    contact_idx, phi_rest = overlaps[0]
    target = ContactTarget(
        sphere_idx=contact_idx,
        t=scene.target_position.astype(np.float64),
        R=float(scene.target_radius),
        kc=scene.kc,
    )
    delta_lat, info_num = solve_lattice_contact_numerical(
        lat, target, phi_rest,
        lateral="distance_preserving",
        eps=scene.eps, tol=1.0e-12,
    )
    f_contact = compute_contact_force(scene, delta_lat)
    return KernelSolution(
        delta=delta_lat,
        contact_force=f_contact,
        contact_sphere_idx=contact_idx,
        solver_info={
            "regime": "lattice_single_contact_legacy",
            "contact_sphere": contact_idx,
            "phi_rest": phi_rest,
            **info_num,
        },
    )


def _solve_lattice_multi_contact(scene: KernelScene,
                                 lat: Lattice) -> KernelSolution:
    """L-BFGS-B on E_anchor + E_lateral + sum_i E_contact_i.

    Bridge-only helper; the theory module's
    ``solve_lattice_contact_numerical`` is restricted to a single
    contact sphere, but realistic kernel scenes have multi-sphere
    contact patches.  We assemble the same energies the theory uses
    elsewhere plus per-sphere series-spring contact contributions, and
    minimise over delta in R^{3N}.

    Anisotropic anchor (``ka_t_ratio != 1``) is NOT supported here --
    the theory's ``anchor_energy`` and the multi-contact gradient
    below are isotropic.  Anisotropic-lattice scenes are deferred to
    step 7b alongside the lattice-level ``ka_t_ratio`` support flagged
    in the step-6 scope note.
    """
    if scene.ka_t_ratio != 1.0:
        raise NotImplementedError(
            "Multi-contact lattice solver does not yet support anisotropic "
            "anchor (ka_t_ratio != 1) -- the theory module's lattice "
            "primitives are isotropic, see step-6 scope note.  Use a "
            "single-sphere scene for anisotropic anchor verification."
        )
    N = scene.n_spheres
    kc = scene.kc
    eps = scene.eps
    R = float(scene.target_radius)
    t = scene.target_position.astype(np.float64)
    surface_mask = scene.is_surface.astype(bool)
    radii = scene.radii.astype(np.float64)
    positions = lat.p  # body-local; we built lat from scene.positions

    def _contact_terms(d: np.ndarray):
        """Yield (i, phi_eff, smooth_step_val, e_hat_def) for each
        surface sphere whose smoothed contact contribution is active.
        """
        for i in range(N):
            if not surface_mask[i]:
                continue
            q_i = positions[i] - d[i]
            diff = q_i - t
            L = float(np.linalg.norm(diff))
            if L < 1.0e-15:
                continue
            e_hat = diff / L
            raw = (radii[i] + R) - L
            # Activation window: smooth_relu(raw, eps) is non-trivial
            # for raw > -O(eps).  For deep separation we skip to save
            # time; the smooth_relu and smooth_step terms are both
            # < 1e-9 there.
            if eps <= 0.0:
                if raw <= 0.0:
                    continue
                phi_eff = raw
                step = 1.0
            else:
                if raw < -50.0 * eps:
                    continue
                r2 = raw * raw + eps * eps
                sqr2 = np.sqrt(r2)
                phi_eff = 0.5 * (raw + sqr2)
                step = 0.5 * (1.0 + raw / sqr2)
            yield i, phi_eff, step, e_hat

    def fun(x: np.ndarray) -> float:
        d = x.reshape(N, 3)
        E = anchor_energy(lat, d) + lateral_energy_distance_preserving(lat, d)
        for _, phi_eff, _, _ in _contact_terms(d):
            E += 0.5 * kc * phi_eff * phi_eff
        return E

    def jac(x: np.ndarray) -> np.ndarray:
        d = x.reshape(N, 3)
        g = np.zeros_like(d)
        # Anchor (isotropic; ka_t_ratio asserted == 1.0 above).
        g += lat.ka * d
        # Lateral (distance-preserving) gradient, inlined to match
        # the form in cslc_lattice.solve_lattice_contact_numerical.
        for (i, j) in lat.edges:
            q_i = lat.p[i] - d[i]
            q_j = lat.p[j] - d[j]
            v = q_j - q_i
            l = float(np.linalg.norm(v))
            if l < 1.0e-15:
                continue
            L_rest = float(np.linalg.norm(lat.p[j] - lat.p[i]))
            e_hat = v / l
            contrib = lat.kl * (l - L_rest) * e_hat
            g[i] += contrib
            g[j] -= contrib
        # Contact (per-sphere series-spring with deformed direction).
        # dE_i/d delta_i = kc * phi_eff * smooth_step(raw) * e_hat_def.
        for i, phi_eff, step, e_hat in _contact_terms(d):
            g[i] += kc * phi_eff * step * e_hat
        return g.reshape(-1)

    res = minimize(
        fun, np.zeros(3 * N), jac=jac, method="L-BFGS-B",
        options={"gtol": 1.0e-12, "ftol": 1.0e-12, "maxiter": 5000},
    )
    delta = res.x.reshape(N, 3)
    f_contact = compute_contact_force(scene, delta)
    return KernelSolution(
        delta=delta,
        contact_force=f_contact,
        contact_sphere_idx=-1,
        solver_info={
            "regime": "lattice_multi_contact",
            "success": bool(res.success),
            "nit": int(res.nit),
            "final_grad_norm": float(np.linalg.norm(res.jac)),
            "energy": float(res.fun),
            "n_overlaps": len(_contact_overlaps(scene)),
        },
    )


__all__ = [
    "KernelScene",
    "KernelSolution",
    "solve_theory",
    "compute_contact_force",
]
