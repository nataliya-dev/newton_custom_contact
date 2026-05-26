# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""T-M: CSLC handler integration test (contract_v2.md §12 Phase 6).

Drives the full :class:`CSLCHandler` pipeline end-to-end:

    compute_cslc_penetration → compute_outward_normals_world →
    lattice_solve_equilibrium → jacobi_step × n_iter →
    cslc_copy_active → write_cslc_contacts.

The bridge harness (``cslc_main/theory/test_07_kernel_bridge.py``) verifies
``jacobi_step`` against theory's ``solve_lattice_contact`` at 12/12 scenes.
T-M closes the gap that the bridge harness leaves open: warm-start,
emission, and handler wiring.

Two assertion stages per contract §12 T-M:

  **Stage 1 — Newton's 3rd law internal consistency (fp precision).**
    For each scene, sum the emitted contact wrench
    ``Σ_slots stiffness · solver_pen · (-normal_ab)``  (force on the
    CSLC pad shape0 from MuJoCo's reconstructed solver_pen) and compare
    to the anchor reaction  ``-Σ_i (ka·δ_n_i·n_i + ka·ratio·δ_t_i)``
    aggregated from the converged ``sphere_delta``.  These are the
    physical force the contact applies to the pad body and the
    physical force the lattice applies to the pad body via the anchor
    springs; at static equilibrium of the lattice (anchor + lateral +
    contact = 0 per sphere, lateral sums to zero across the lattice)
    they must agree to floating-point precision.  Catches emission
    algebra bugs (wrong margin0/margin1, wrong normal_ab sign, wrong
    kc_series composition, missing area/locality weight).

  **Stage 2 — §10 series-spring identity (≤1% rel).**
    For the *single pad sphere vs flat-face point set* scene (a
    geometry where the §10 calibration assumption — N_contact engaged
    spheres each at depth d — is exact), assert
    ``F_emission ≈ k_eff · d_rest`` with
    ``k_eff = ka·kc/(ka + kc + small ke_target correction)``.  ``kc``
    is the per-volume stiffness the handler now produces
    automatically from a per-sphere calibration (see the
    Phase 6 update to :meth:`CSLCHandler.from_model_with_lattices`).

The sphere-target scene (scene J geometry from the bridge harness:
tennis-ball-scale R = 33.5 mm, 1500 Fibonacci samples) is run for
Stage 1 only — at production geometry the half-space approximation
adds an O(r_pad²/R) ≈ 7% per-edge-pair curvature error to the §10
identity (cf. contract §7.2), so a 1% analytical assertion against
``F = k_eff · d`` would force the test to fail on the geometry, not on
the code.  Stage 1 (handler-internal consistency) still catches every
bug T-M is supposed to surface; Stage 2 on a flat target nails the
analytical identity that §10 is built on.

File location: newton/_src/geometry/tests/test_cslc_handler_sphere_target.py
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.geometry.cslc_data import CSLCLattice
from newton._src.geometry.cslc_handler import CSLCHandler

# Theory-side fixtures (reused so T-M's geometry is the same as the
# bridge harness's verified scene J, modulo target sampling).
from cslc_main.theory.cslc_lattice import make_dome
from cslc_main.theory.cslc_targets import (
    PointSetTargetV2,
    make_flat_face_target,
    make_sphere_target,
)

# ── Constants matching contract v2 (production defaults) ────────────────
EPS_PROD = 5.0e-4           # smoothing width, matches kernel + theory
N_ITER = 10000              # Jacobi iters per pair.  Production grasp
                            # uses 40 because each collide is preceded
                            # by a warm start, but T-M cold-starts every
                            # assertion.  Bridge harness saturates at
                            # ~5000 iters for kc/ka=10; bump to 10000
                            # for headroom on the unwarmed cold start.
ALPHA = 0.3                 # damped-Jacobi mixing
EPS_ALIGN = 0.05            # one-sided alignment band literal


def _build_single_pad_lattice(
    *, r_pad: float, shape_index: int,
) -> CSLCLattice:
    """One pad sphere at the origin, outward normal +z, no neighbours."""
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    radii = np.array([r_pad], dtype=np.float32)
    is_surface = np.ones(1, dtype=bool)
    outward_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    neighbor_indices = [np.zeros(0, dtype=np.int32)]
    return CSLCLattice(
        positions=positions, radii=radii, is_surface=is_surface,
        outward_normals=outward_normals,
        neighbor_indices=neighbor_indices,
        shape_index=shape_index,
        spacing=2.0 * r_pad,      # bookkeeping; not consumed by kernels
        sphere_radius=r_pad,
    )


def _build_dome_lattice(
    *, N: int, R_dome: float, half_angle: float, ka: float, kl: float,
    k_neighbors: int, shape_index: int,
) -> tuple[CSLCLattice, float, float]:
    """Dome pad via make_dome; r_pad = spacing / 2 (matches grasp pipeline).

    The theory ``Lattice`` already places the dome apex at +z and
    radial outward normals; we convert to the kernel-side
    :class:`CSLCLattice` with the per-sphere radius set to half the
    mean nearest-neighbour distance (the production convention used by
    :func:`cslc_main.grasp.contact_models.make_cslc_pad_from_samples`).
    """
    lat, spacing, cap_area = make_dome(
        N=N, R_pad=R_dome, half_angle=half_angle,
        ka=ka, kl=kl, k_neighbors=k_neighbors,
    )
    r_pad = float(spacing * 0.5)
    n = lat.N
    positions = lat.p.astype(np.float32)
    radii = np.full(n, r_pad, dtype=np.float32)
    is_surface = np.ones(n, dtype=bool)
    outward_normals = lat.n.astype(np.float32)
    # Build per-sphere neighbour index lists from the edge list.  The
    # theory ``Lattice`` stores edges with i < j; the kernel CSR wants
    # each sphere's full neighbour list, so we add both endpoints.
    neighbor_indices: list[np.ndarray] = [[] for _ in range(n)]
    for (i, j) in lat.edges:
        neighbor_indices[int(i)].append(int(j))
        neighbor_indices[int(j)].append(int(i))
    neighbor_indices_np = [
        np.array(sorted(nbs), dtype=np.int32) for nbs in neighbor_indices
    ]
    return (
        CSLCLattice(
            positions=positions, radii=radii, is_surface=is_surface,
            outward_normals=outward_normals,
            neighbor_indices=neighbor_indices_np,
            shape_index=shape_index, spacing=spacing,
            sphere_radius=r_pad,
        ),
        spacing,
        cap_area,
    )


def _build_model(
    *,
    pad_lattice: CSLCLattice,
    target_body_xform: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    target_geo_type: GeoType,
    target_radius: float | None,
    target_extents: tuple[float, float, float] | None,
    ka: float, kl: float, ke_pad: float, ke_target: float,
    mu: float = 0.0,
) -> tuple[newton.Model, int, int, int, int]:
    """Construct a minimal two-body Newton model for the CSLC handler.

    Body 0 = pad (CSLC-flagged sphere shape, lattice attached).
    Body 1 = target (rigid sphere or rigid box; geo type per arg).

    The pad's mesh-typed visualisation is irrelevant for the CSLC path
    (kernels read the lattice, not the mesh); we use a SPHERE shape with
    ``is_cslc=True`` for the pad to keep the builder happy without
    needing a trimesh.

    Returns ``(model, pad_shape, target_shape, pad_body, target_body)``.
    """
    b = newton.ModelBuilder()
    pad_cfg = newton.ModelBuilder.ShapeConfig(
        ke=ke_pad, kd=0.0, kf=0.0, mu=mu, gap=0.0,
        density=1000.0,
        is_cslc=True,
        cslc_spacing=float(pad_lattice.spacing),
        cslc_ka=ka, cslc_kl=kl, cslc_dc=0.0,
        cslc_n_iter=N_ITER, cslc_alpha=ALPHA,
    )
    target_cfg = newton.ModelBuilder.ShapeConfig(
        ke=ke_target, kd=0.0, kf=0.0, mu=mu, gap=0.0,
        density=1000.0,
    )

    pad_body = b.add_body(mass=1.0, label="pad_body")
    # Pad mesh shape (CSLC-flagged).  Pad lattice rest spheres live in
    # this shape's local frame; the shape itself is a unit-radius sphere
    # at the body origin (the visualisation is irrelevant — kernels read
    # the lattice).  We make r small enough that the pad sphere shape
    # itself wouldn't satisfy a half-space gate against the target.
    pad_shape = b.add_shape_sphere(
        pad_body, xform=wp.transform(),
        radius=float(pad_lattice.radii.max()) * 0.5,
        cfg=pad_cfg,
    )

    target_pos, target_rot = target_body_xform
    target_xform = wp.transform(wp.vec3(*target_pos), wp.quat(*target_rot))
    target_body = b.add_body(
        xform=target_xform, mass=1.0, label="target_body",
    )
    if target_geo_type == GeoType.SPHERE:
        assert target_radius is not None
        target_shape = b.add_shape_sphere(
            target_body, xform=wp.transform(),
            radius=float(target_radius), cfg=target_cfg,
        )
    elif target_geo_type == GeoType.BOX:
        assert target_extents is not None
        hx, hy, hz = target_extents
        target_shape = b.add_shape_box(
            target_body, xform=wp.transform(),
            hx=float(hx), hy=float(hy), hz=float(hz), cfg=target_cfg,
        )
    else:
        raise ValueError(f"Unsupported target geo type: {target_geo_type}")

    model = b.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model, pad_shape, target_shape, pad_body, target_body


def _populate_target_arrays(
    pair, samples, device, K_max: int,
) -> None:
    """Upload the sampler's ``(positions, normals, areas)`` to the pair."""
    pair.target_positions_local = wp.array(
        samples.positions.astype(np.float32), dtype=wp.vec3, device=device,
    )
    pair.target_normals_local = wp.array(
        samples.normals.astype(np.float32), dtype=wp.vec3, device=device,
    )
    if samples.areas is None:
        # ``areas = None`` is the bridge-convention per-pair Hookean
        # form.  T-M uses physical Voronoi areas (the grasp convention)
        # so the handler's per-volume kc rescale lands in the right
        # units.  Refuse the bridge form here to keep the kc units
        # unambiguous; T-M's whole point is to test the production
        # handler convention.
        raise ValueError(
            "T-M requires target.areas to be populated (Voronoi areas "
            "in [m²]); the per-pair Hookean form (areas=None) is "
            "bridge-only.",
        )
    pair.target_areas_local = wp.array(
        samples.areas.astype(np.float32),
        dtype=wp.float32, device=device,
    )
    pair.target_count = int(samples.positions.shape[0])
    pair.K_max = int(K_max)


def _launch_and_extract(
    handler: CSLCHandler, model: newton.Model, state: newton.State,
    contacts: newton.Contacts,
) -> dict:
    """Drive ``handler.launch`` once and pull out the diagnostics T-M needs.

    Returns
    -------
    dict with keys
        ``sphere_delta``        (N_lattice, 3) world-frame displacement
        ``outward_normals_world`` (N_lattice, 3) world-frame outward n̂_i
        ``radii``               (N_lattice,)
        ``is_surface``          (N_lattice,) bool
        ``stiffness``           per-slot emission stiffness
        ``solver_pen``          per-slot reconstructed MuJoCo pen
        ``normal``              per-slot contact normal (world frame —
                                we transform per body's xform once here)
        ``active_mask``         which slots actually got written
        ``handler_truncations`` count of pad spheres that hit K_max
    """
    handler.launch(model, state, contacts, contact_offset=0)
    # Force the launch's GPU work to retire before we read back.  All
    # subsequent ``.numpy()`` calls would do this implicitly, but
    # collecting all sync points here makes the test's flow obvious.
    wp.synchronize_device()

    data = handler.cslc_data
    delta_np = data.sphere_delta.numpy()
    out_n_world_np = handler.out_normal_world_scratch.numpy()
    radii_np = data.radii.numpy()
    is_surface_np = data.is_surface.numpy().astype(bool)

    # Emission slots written by ``write_cslc_contacts``.  Slots that
    # remained sentinel (out_shape0 = -1) are inactive.
    shape0_np = contacts.rigid_contact_shape0.numpy()
    shape1_np = contacts.rigid_contact_shape1.numpy()
    point0_np = contacts.rigid_contact_point0.numpy()
    point1_np = contacts.rigid_contact_point1.numpy()
    normal_np = contacts.rigid_contact_normal.numpy()
    margin0_np = contacts.rigid_contact_margin0.numpy()
    margin1_np = contacts.rigid_contact_margin1.numpy()
    stiffness_np = contacts.rigid_contact_stiffness.numpy()

    active = shape0_np >= 0

    # Reconstruct world-space point0 / point1 / normal using the
    # contacts' shape_a / shape_b body transforms.  The emission kernel
    # writes p0/p1 in their respective bodies' local frames and the
    # normal as a world-frame vector that has already been transformed
    # by the target body's transform (it's ``-n_face_world``); see the
    # kernel docstring's "Body-frame contact geometry" block.
    body_q_np = state.body_q.numpy()
    shape_body_np = model.shape_body.numpy()
    shape_transform_np = model.shape_transform.numpy()

    def _xform_apply(xform: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Apply a (pos, quat_xyzw) Newton transform to a 3-vec."""
        p = xform[:3]
        q = xform[3:7]
        qx, qy, qz, qw = q
        # Rotate v by quat (qx, qy, qz, qw).
        # Using the Hamilton convention rot(v) = q v q*.
        x, y, z = v
        tx = 2 * (qy * z - qz * y)
        ty = 2 * (qz * x - qx * z)
        tz = 2 * (qx * y - qy * x)
        vx = x + qw * tx + (qy * tz - qz * ty)
        vy = y + qw * ty + (qz * tx - qx * tz)
        vz = z + qw * tz + (qx * ty - qy * tx)
        return np.array([p[0] + vx, p[1] + vy, p[2] + vz])

    def _xform_apply_vec(xform: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate-only variant — used for direction vectors (normals)."""
        q = xform[3:7]
        qx, qy, qz, qw = q
        x, y, z = v
        tx = 2 * (qy * z - qz * y)
        ty = 2 * (qz * x - qx * z)
        tz = 2 * (qx * y - qy * x)
        return np.array([
            x + qw * tx + (qy * tz - qz * ty),
            y + qw * ty + (qz * tx - qx * tz),
            z + qw * tz + (qx * ty - qy * tx),
        ])

    # The emission kernel writes:
    #     p0_body = X_wb^-1 ·(q_world_def)         -- pad body-local
    #     p1_body = X_tb^-1 ·(t_world)             -- target body-local
    #     normal  = -n_face_world                  -- world frame
    # Reconstruct the world-frame versions for the solver_pen formula.
    n_slots = shape0_np.shape[0]
    p0_world = np.zeros_like(point0_np)
    p1_world = np.zeros_like(point1_np)
    n_world_unrot = normal_np.copy()  # normal is already world frame
    for k in range(n_slots):
        if not active[k]:
            continue
        sa = int(shape0_np[k])
        sb = int(shape1_np[k])
        ba = int(shape_body_np[sa])
        bb = int(shape_body_np[sb])
        Xa = body_q_np[ba]
        Xb = body_q_np[bb]
        # body-frame -> world (multiply by shape_transform? no — emission
        # kernel writes points in BODY frame already; the shape_transform
        # is for the shape relative to its body, but the kernel uses
        # body_q^-1 · q_world directly, so the body-frame point is the
        # body-relative point.  Apply body_q.
        p0_world[k] = _xform_apply(Xa, point0_np[k])
        p1_world[k] = _xform_apply(Xb, point1_np[k])

    diff = p1_world - p0_world
    # ``solver_pen = margin0 + margin1 - (p1 - p0) · normal``  (MuJoCo
    # reconstruction; cf. write_cslc_contacts docstring).
    solver_pen = (margin0_np + margin1_np -
                  np.einsum("ij,ij->i", diff, n_world_unrot))

    truncations = sum(
        int(handler.truncation_count_pairs[idx].numpy()[0])
        for idx in range(len(handler.shape_pairs))
    )

    return {
        "sphere_delta": delta_np,
        "outward_normals_world": out_n_world_np,
        "radii": radii_np,
        "is_surface": is_surface_np,
        "stiffness": stiffness_np,
        "solver_pen": solver_pen,
        "normal_world": n_world_unrot,
        "active": active,
        "p0_world": p0_world,
        "p1_world": p1_world,
        "truncations": truncations,
        "kc_handler": float(handler.cslc_data.kc),
        "ka_handler": float(handler.cslc_data.ka),
        "kl_handler": float(handler.cslc_data.kl),
    }


def _aggregate_forces(
    diag: dict, approach_axis: np.ndarray, *,
    ka_t_ratio: float = 1.0,
) -> dict:
    """Project anchor + emission wrenches onto the approach axis.

    Sign convention: ``approach_axis`` points from pad → target (the
    direction along which the test compresses the contact).  The
    *force on the pad body* along ``approach_axis`` is negative at
    compressive contact (target pushes pad away from itself, opposite
    to the approach axis).  Both ``F_anchor`` and ``F_emission`` are
    returned as the SIGNED projection along ``approach_axis``.

    The Newton's-3rd-law identity asserted in T-M Stage 1 is then::

        F_anchor_z ≈ F_emission_z

    where ``z`` denotes the approach-axis projection (both forces on
    the pad body, expected negative at compressive contact).
    """
    delta = diag["sphere_delta"]
    n_world = diag["outward_normals_world"]
    is_surface = diag["is_surface"]
    surface_mask = is_surface.astype(bool)

    ka = diag["ka_handler"]
    # Per-sphere anchor force on q_i (PHYSICAL — pulls q back toward p):
    #   F_anchor_on_q_i = ka·δ_n_i·n_i + ka·ratio·δ_t_i
    # By Newton's 3rd law, the lattice exerts -F_anchor_on_q_i on the
    # pad RIGID body at the same point.  The total force the lattice
    # transmits to the pad body, summed across all spheres, is the
    # negative aggregate.  That is, ultimately, the same force as the
    # CONTACT load (Newton's 3rd law on the lattice itself: the
    # contact force on each q_i balances the anchor + lateral force,
    # and the lateral sums to zero across the lattice).
    delta_surf = delta[surface_mask]
    n_surf = n_world[surface_mask]
    delta_n = np.einsum("ij,ij->i", delta_surf, n_surf)            # (Ns,)
    delta_t = delta_surf - delta_n[:, None] * n_surf                # (Ns, 3)
    # Physical anchor force on q_i  (kernel "physical" form, +∂E/∂δ).
    F_anchor_on_q = ka * (delta_n[:, None] * n_surf + ka_t_ratio * delta_t)
    # The lattice transmits ``F_lattice_on_pad = -F_anchor_on_q`` to the
    # pad rigid body at each anchor.  Equivalently, the contact applies
    # ``-F_anchor_on_q`` to each q_i.  Summed (lateral cancels):
    F_anchor_total_on_pad = -F_anchor_on_q.sum(axis=0)
    F_anchor_proj = float(F_anchor_total_on_pad @ approach_axis)

    # Emission wrench on shape0 (pad).  MuJoCo's constraint convention:
    # force on shape0 = -stiffness · solver_pen · normal_ab
    # where normal_ab points shape0 → shape1.
    active = diag["active"]
    stiffness = diag["stiffness"][active]
    solver_pen = diag["solver_pen"][active]
    normal = diag["normal_world"][active]
    F_on_pad_world = -(stiffness * solver_pen)[:, None] * normal
    F_emission_total = F_on_pad_world.sum(axis=0)
    F_emission_proj = float(F_emission_total @ approach_axis)

    # Diagnostic per-sphere ladder.
    n_engaged = int(np.sum(stiffness > 0.0))
    apex_idx = (
        int(np.argmax(delta_surf[:, 2])) if delta_surf.size > 0 else -1
    )
    apex_delta_n = (
        float(delta_n[apex_idx]) if apex_idx >= 0 else 0.0
    )

    return {
        "F_anchor_proj": F_anchor_proj,
        "F_emission_proj": F_emission_proj,
        "F_anchor_vec": F_anchor_total_on_pad,
        "F_emission_vec": F_emission_total,
        "n_active_contacts": n_engaged,
        "apex_delta_n": apex_delta_n,
        "n_surface_spheres": int(surface_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Test class
# ─────────────────────────────────────────────────────────────────────────


class TestCSLCHandlerSphereTarget(unittest.TestCase):
    """Phase 6 / T-M handler integration test (contract §12)."""

    def setUp(self) -> None:
        wp.set_device("cuda:0" if wp.is_cuda_available() else "cpu")

    # ─── Helpers ──────────────────────────────────────────────────────

    def _run_single_pad_vs_flat(
        self, *,
        kc_ratio: float, depth: float,
        ka: float = 25000.0, kl: float = 5000.0,
        # Production grasp ke_target ~ 1e9 [N/m].  The handler
        # pre-composes kc with ke_target up front (contract §11
        # Phase 6 amendment), so the lattice solver and emission see
        # the same effective kc — no kc_series re-composition gap.
        ke_target: float = 1.0e9,
        sampling: str = "single",        # "single" | "dense"
    ) -> dict:
        """See class docstring for layout; ``sampling`` switches between
        a single A_j=A_patch contact point and a dense 60×60 grid."""
        """Single pad sphere vs sampled flat face along the +z approach.

        Geometry: pad sphere at origin with n_pad = +ẑ; flat face at
        z = r_pad - depth (so the rest half-space raw at any face-on
        sample is exactly ``depth``).

        Two sampling modes:

        * ``"single"`` — one target point at the contact axis, with
          ``A_j`` set to the kernel patch area ``A_patch = π·(3·r_pad)²``.
          This eliminates the surface-integral discretisation error
          entirely (``Σ_j A_j · w_t = A_patch · 1 = A_patch``), so the
          §10 identity ``F = k_eff · d`` is exact to floating point
          modulo the smooth-gate factor at finite ``raw_eq / eps``.

        * ``"dense"`` — 30 mm × 30 mm regular grid at 0.5 mm pitch
          (~225 samples in the kernel disc).  Exercises the surface
          integral; precision floor ≈ 2-3% from the regular-grid
          truncation of the kernel disc.
        """
        r_pad = 1.5e-3
        # Calibration: we want ``calibrate_kc`` to return a specific
        # ``kc_per_sphere = kc_ratio · ka`` so the §10 series-spring
        # ``F = k_eff · d`` identity has a clean analytical prediction.
        # ``calibrate_kc`` solves ``1/kc = N_contact/ke_bulk − 1/ka``
        # (rigid-target default, no ke_target term).  For
        # ``N_contact=1`` this gives
        # ``ke_bulk = ka·kc/(ka+kc) = k_eff_two_spring``.  So we set
        # ``ke_pad = ke_bulk`` and let the handler's calibration
        # round-trip back to our target kc, with A_inv built around
        # the correct kc and no post-hoc overrides needed.
        kc_per_sphere = kc_ratio * ka
        ke_bulk = ka * kc_per_sphere / (ka + kc_per_sphere)
        # k_eff for the §10 prediction (three-spring chain with the
        # target's compliance included).  With rigid ke_target this
        # collapses to ke_bulk.
        k_eff_per_sphere = 1.0 / (1.0 / ka + 1.0 / kc_per_sphere +
                                  1.0 / ke_target)

        # Pad lattice: single sphere.
        # We assign shape_index = 0 here as a placeholder; the actual
        # shape index is fixed up after the Newton model is built (the
        # lattice's shape_index must match what ModelBuilder assigned to
        # the pad's CSLC-flagged shape).
        pad_lat = _build_single_pad_lattice(r_pad=r_pad, shape_index=0)

        # Flat target along +z (face_z = r_pad - depth ⇒ raw_rest = depth).
        face_z_world = r_pad - depth
        # Target body sits at origin (identity body_q), so target-local
        # = world for our purposes.
        if sampling == "single":
            # Single sample at the apex with A_j = A_patch.  The
            # locality kernel ``w_t = smooth_step(3·r_pad - 0, eps)
            # ≈ 1`` at the apex, so ``Σ A_j · w_t = A_patch`` exactly
            # by construction.
            kernel_h = 3.0 * r_pad
            A_patch = float(math.pi * kernel_h * kernel_h)
            target = PointSetTargetV2(
                positions=np.array([[0.0, 0.0, face_z_world]],
                                   dtype=np.float64),
                normals=np.array([[0.0, 0.0, -1.0]], dtype=np.float64),
                areas=np.array([A_patch], dtype=np.float64),
            )
            face_span = 1.0  # bookkeeping for the target shape extents
        elif sampling == "dense":
            face_span = 0.030
            target = make_flat_face_target(
                centre=np.array([0.0, 0.0, face_z_world]),
                normal=np.array([0.0, 0.0, -1.0]),
                span_u=face_span, span_v=face_span,
                pitch=0.5e-3,
            )
        else:
            raise ValueError(f"Unknown sampling mode: {sampling}")

        # ── Build Newton model ──
        # Target body's position is the world centre of the flat face.
        model, pad_shape, target_shape, pad_body, target_body = _build_model(
            pad_lattice=pad_lat,
            target_body_xform=(
                (0.0, 0.0, 0.0),     # target body at origin
                (0.0, 0.0, 0.0, 1.0),
            ),
            target_geo_type=GeoType.BOX,
            target_radius=None,
            target_extents=(face_span, face_span, 1.0e-3),
            ka=ka, kl=kl,
            ke_pad=ke_bulk,           # used by calibrate_kc as "ke_bulk"
            ke_target=ke_target,      # used by kernel for kc_series
        )
        # Fix the lattice's shape_index to point at the right CSLC shape.
        pad_lat.shape_index = pad_shape

        # ── Build CSLC handler (uses the handler's per-volume rescale) ──
        handler = CSLCHandler.from_model_with_lattices(
            model, lattices_by_shape={pad_shape: pad_lat},
            contact_fraction=1.0,  # single pad sphere ⇒ N_contact = 1.
        )
        self.assertIsNotNone(
            handler, "CSLCHandler.from_model_with_lattices returned None",
        )
        # Override the CSLCData smoothing width to the production
        # value the grasp pipeline uses (cslc_main/grasp/params.py
        # ``CSLCParams.smoothing_eps = 5e-4``).  This catches any
        # kernel-vs-emission mismatch that hides at the
        # ``CSLCData`` default ``1e-5`` (essentially-binary gates):
        # at production eps and 1 mm overlap, ``raw/eps = 2`` and
        # the phi_eff–raw smooth-zone gap is significant.
        handler.cslc_data.smoothing_eps = EPS_PROD
        # Sanity check: the handler should have round-tripped our
        # target kc through calibrate_kc, then composed it with
        # ke_target (Phase 6 fix), and finally rescaled per-volume.
        kernel_h = 3.0 * r_pad
        A_patch = float(math.pi * kernel_h * kernel_h)
        kc_per_sphere_eff = (
            kc_per_sphere * ke_target / (kc_per_sphere + ke_target)
        )
        expected_kc_per_volume = kc_per_sphere_eff / A_patch
        actual_kc_per_volume = float(handler.cslc_data.kc)
        kc_rel = abs(actual_kc_per_volume - expected_kc_per_volume) / max(
            expected_kc_per_volume, 1.0e-30
        )
        self.assertLess(
            kc_rel, 1.0e-3,
            f"Handler kc mismatch: got {actual_kc_per_volume:.4e} "
            f"vs predicted {expected_kc_per_volume:.4e} "
            f"(kc_per_sphere_eff={kc_per_sphere_eff:.4e}, "
            f"ke_bulk={ke_bulk}, ke_target={ke_target})",
        )

        # Populate target arrays on the single pair.
        self.assertEqual(len(handler.shape_pairs), 1)
        pair = handler.shape_pairs[0]
        # K_max sized for the dense flat grid: ~225 samples inside the
        # kernel disc + slack for the smooth w_t tail past 3·r_pad.
        # Single-sample mode needs K_max ≥ 1; we keep the buffer
        # generous either way.
        _populate_target_arrays(
            pair, target, device=handler.device,
            K_max=4 if sampling == "single" else 512,
        )

        # ── Build State + Contacts ──
        state = model.state()
        # Pad body at origin, identity rotation.  Target already placed
        # at origin in body_q (the target-local face is at face_z_world
        # along +z).
        # Sanity-check pad body world-frame position.
        body_q_np = state.body_q.numpy()
        # Force the pad body to be at the identity transform so the
        # lattice rest sphere lands at the world origin.  (Add_body
        # auto-creates a free joint and sets body_q from xform; the
        # default xform is identity, which is what we want.)
        contacts = newton.Contacts(
            rigid_contact_max=handler.contact_count + 16,
            soft_contact_max=0,
            per_contact_shape_properties=True,
            device=handler.device,
        )

        diag = _launch_and_extract(handler, model, state, contacts)
        diag["k_eff_per_sphere_predicted"] = k_eff_per_sphere
        diag["kc_per_sphere_predicted"] = kc_per_sphere
        diag["depth_rest"] = depth
        diag["r_pad"] = r_pad
        diag["approach_axis"] = np.array([0.0, 0.0, +1.0])

        agg = _aggregate_forces(
            diag, diag["approach_axis"], ka_t_ratio=1.0,
        )
        diag.update(agg)
        return diag

    def _run_single_pad_vs_sphere(
        self, *,
        kc_ratio: float, depth: float,
        ka: float = 25000.0, kl: float = 5000.0,
        ke_target: float = 1.0e9,           # production scale
        R_sphere: float = 33.5e-3,          # tennis-ball scale (scene J)
        n_sphere_samples: int = 1500,
    ) -> dict:
        """Scene J replicated through the CSLCHandler.

        Single pad sphere at origin; sphere target above it along +z
        (sphere centre at z = R_sphere + r_pad - depth so the apex
        rest raw equals ``depth``).  Stage 1 only — the curvature
        approximation breaks 1% at production R/r_pad ≈ 22.
        """
        r_pad = 1.5e-3
        kc_per_sphere = kc_ratio * ka
        ke_bulk = ka * kc_per_sphere / (ka + kc_per_sphere)

        pad_lat = _build_single_pad_lattice(r_pad=r_pad, shape_index=0)

        sphere_centre_world = np.array(
            [0.0, 0.0, (r_pad - depth) + R_sphere], dtype=np.float64,
        )

        model, pad_shape, target_shape, pad_body, target_body = _build_model(
            pad_lattice=pad_lat,
            target_body_xform=(
                tuple(float(x) for x in sphere_centre_world),
                (0.0, 0.0, 0.0, 1.0),
            ),
            target_geo_type=GeoType.SPHERE,
            target_radius=R_sphere,
            target_extents=None,
            ka=ka, kl=kl,
            ke_pad=ke_bulk,
            ke_target=ke_target,
        )
        pad_lat.shape_index = pad_shape

        handler = CSLCHandler.from_model_with_lattices(
            model, lattices_by_shape={pad_shape: pad_lat},
            contact_fraction=1.0,
        )
        self.assertIsNotNone(handler)
        handler.cslc_data.smoothing_eps = EPS_PROD

        # Sphere target sampled in target-body-local frame (sphere
        # centred at body origin).
        target = make_sphere_target(
            t=np.zeros(3),
            R=R_sphere,
            n_samples=n_sphere_samples,
        )

        pair = handler.shape_pairs[0]
        _populate_target_arrays(
            pair, target, device=handler.device, K_max=256,
        )

        state = model.state()
        contacts = newton.Contacts(
            rigid_contact_max=handler.contact_count + 16,
            soft_contact_max=0,
            per_contact_shape_properties=True,
            device=handler.device,
        )

        diag = _launch_and_extract(handler, model, state, contacts)
        diag["kc_per_sphere_predicted"] = kc_per_sphere
        diag["depth_rest"] = depth
        diag["r_pad"] = r_pad
        diag["approach_axis"] = np.array([0.0, 0.0, +1.0])
        agg = _aggregate_forces(
            diag, diag["approach_axis"], ka_t_ratio=1.0,
        )
        diag.update(agg)
        return diag

    # ─── Stage 1: Newton's 3rd law internal consistency ───────────────

    def test_single_pad_flat_emission_matches_anchor(self) -> None:
        """Single-pad vs single-sample flat target: F_emission ≈ F_anchor.

        Stage 1 of T-M.  The kernel emits per-pair (stiffness,
        solver_pen, normal) where MuJoCo's reconstructed
        ``stiffness · solver_pen = kc · A · w · α · gate · raw``;
        the lattice solver applies ``kc · A · w · α · phi_eff · gate``
        per sphere.  Contract §8 calls out the two are "algebraically
        the same in the deep-saturated limit (φ_eff ≈ raw when
        raw ≫ ε)", but at raw ~ ε there is an inherent
        ``O((ε/raw)²)`` gap because phi_eff = smooth_relu(raw, ε)
        slightly exceeds raw near the boundary.

        Newton's 3rd law on the lattice (anchor + lateral + contact
        = 0 per sphere, lateral sums to zero across the lattice) ties
        the anchor reaction to the contact load; the emission then
        differs by the phi_eff–raw gap.  We test depths deep enough
        that ``raw_eq = d/(1+kc/ka) ≫ ε`` for every ``kc/ka`` to
        keep the gap below the 1% contract tolerance.  Production
        ``ε_data = 1e-5`` (CSLCData default), so 10·ε = 0.1 mm at the
        toughest case (``kc/ka = 10``) needs ``d ≥ 1.1 mm``.  We use
        ``d = 15 mm`` for headroom: at kc/ka=10 the raw_eq is ~1.4 mm
        ~140·ε, deeply saturated.

        Single-sample mode (``A_j = A_patch``) eliminates the
        surface-integral discretisation error so the only residual is
        the phi_eff–raw gap.
        """
        for kc_ratio in [0.1, 1.0, 10.0]:
            with self.subTest(kc_ratio=kc_ratio):
                diag = self._run_single_pad_vs_flat(
                    kc_ratio=kc_ratio, depth=15.0e-3, sampling="single",
                )
                Fa = diag["F_anchor_proj"]
                Fe = diag["F_emission_proj"]
                rel = abs(Fa - Fe) / max(abs(Fa), 1.0e-15)
                print(
                    f"  [Stage 1 flat-1pt kc/ka={kc_ratio:>5}] "
                    f"F_anchor={Fa:+.6e} N, F_emission={Fe:+.6e} N, "
                    f"rel={rel:.3e}, "
                    f"n_active_pairs={diag['n_active_contacts']}, "
                    f"truncations={diag['truncations']}",
                )
                self.assertEqual(diag["truncations"], 0,
                                 "K_max budget under-sized")
                self.assertGreater(diag["n_active_contacts"], 0,
                                   "no contacts emitted at 15 mm depth")
                self.assertLess(
                    rel, 1.0e-4,
                    f"Stage 1 (emission↔anchor) failed at kc/ka={kc_ratio}: "
                    f"rel={rel:.2e}",
                )

    def test_single_pad_sphere_emission_matches_anchor(self) -> None:
        """Single-pad vs sphere target (scene J geometry): F_e ≈ F_a.

        Same kernel internal consistency check as the flat case, but
        with a curved sphere target sampled via the Fibonacci spiral.
        Exercises the multi-sample contribution path: off-axis target
        samples within the locality kernel each carry their own raw,
        w_t, and align contributions.  A bug in any one of those
        (e.g. wrong target-body transform of the sample normals,
        miscomputed d_t, wrong align_arg) would break Stage 1.

        Tolerance ``1e-2`` instead of ``1e-4`` because the multi-sample
        surface-integral discretisation adds an additional residual
        (``O(r_pad²/R)`` per pair from the half-space approximation of
        the sphere; cf. contract §7.2 — at production
        ``R/r_pad ≈ 22`` this is ~6.7% per *edge* pair, averaged down
        to a few percent in aggregate).  Tight bug-detection
        coverage is provided by the flat-single-point case at ``1e-4``;
        this case is the extension to a curved target.
        """
        for kc_ratio in [0.1, 1.0, 10.0]:
            with self.subTest(kc_ratio=kc_ratio):
                diag = self._run_single_pad_vs_sphere(
                    kc_ratio=kc_ratio, depth=15.0e-3,
                )
                Fa = diag["F_anchor_proj"]
                Fe = diag["F_emission_proj"]
                rel = abs(Fa - Fe) / max(abs(Fa), 1.0e-15)
                print(
                    f"  [Stage 1 sphere kc/ka={kc_ratio:>5}] "
                    f"F_anchor={Fa:+.6e} N, F_emission={Fe:+.6e} N, "
                    f"rel={rel:.3e}, "
                    f"n_active_pairs={diag['n_active_contacts']}, "
                    f"truncations={diag['truncations']}",
                )
                self.assertEqual(diag["truncations"], 0,
                                 "K_max budget under-sized")
                self.assertGreater(diag["n_active_contacts"], 0,
                                   "no contacts emitted at 15 mm depth")
                self.assertLess(
                    rel, 1.0e-2,
                    f"Stage 1 (emission↔anchor, sphere) failed at "
                    f"kc/ka={kc_ratio}: rel={rel:.2e}",
                )

    # ─── Stage 2: §10 calibration identity (flat target) ──────────────

    def test_single_pad_flat_keff_times_depth(self) -> None:
        """F_emission ≈ k_eff · depth within 1% for single-pad-vs-flat.

        Stage 2 of T-M.  On a flat target (zero curvature error in the
        half-space approximation), the §10 series-spring identity
        ``F = k_eff · d`` is exact for a single engaged pad sphere
        whose contact patch is fully sampled (face span ≫ 3·r_pad
        locality kernel).  ``k_eff = 1 / (1/ka + 1/kc + 1/ke_target)``.

        Precision floor (contract T-C Part D).  At production
        ``eps = 5×10⁻⁴`` the smooth-gate factor ``smooth_step(raw_eq,
        eps)`` is < 1 unless ``raw_eq ≥ 10·eps = 5 mm``.  In a
        series-spring with ratio ``kc/ka``, raw_eq at equilibrium is
        ``d · ka/(ka + kc) = d/(1 + kc/ka)``.  For the assertion
        ``F = k_eff · d`` to hold at 1%, we need
        ``d/(1 + kc/ka) ≥ 10·eps``.  This sweep picks per-ratio
        depths that satisfy the floor.
        """
        # depth = d such that d / (1 + kc/ka) ≥ 10·eps  ⇒  d ≥ 10·eps·(1 + kc/ka).
        # eps = 5e-4 m  ⇒  10·eps = 5 mm.
        # For kc/ka=10: d ≥ 55 mm — geometrically unphysical for a
        # 1.5 mm pad sphere.  Skip; Stage 1 still exercises the
        # high-kc regime for handler internal consistency, and the
        # bridge harness's T-K scene A at kc/ka=10 already verifies
        # the kernel matches theory there to 1.7e-6 rel.
        for (kc_ratio, depth_mm) in (
            (0.1, 6.0), (0.1, 10.0),
            (1.0, 11.0), (1.0, 15.0),
        ):
            with self.subTest(kc_ratio=kc_ratio, depth_mm=depth_mm):
                diag = self._run_single_pad_vs_flat(
                    kc_ratio=kc_ratio, depth=depth_mm * 1.0e-3,
                    sampling="single",
                )
                k_eff = diag["k_eff_per_sphere_predicted"]
                d = diag["depth_rest"]
                F_pred = k_eff * d
                F_emit = abs(diag["F_emission_proj"])
                rel = abs(F_emit - F_pred) / max(F_pred, 1.0e-15)
                # raw_eq at the smooth-gate floor of T-C Part D.
                raw_eq_pred = d / (1.0 + kc_ratio)
                print(
                    f"  [Stage 2 flat-1pt kc/ka={kc_ratio:>5} "
                    f"d={depth_mm:.1f}mm raw_eq≈{raw_eq_pred*1e3:.2f}mm] "
                    f"F_predicted={F_pred:.4e} N, "
                    f"F_emission={F_emit:.4e} N, rel={rel:.3e}",
                )
                self.assertLess(
                    rel, 1.0e-2,
                    f"Stage 2 (F=k_eff·d) failed at kc/ka={kc_ratio}, "
                    f"d={depth_mm} mm: rel={rel:.2e}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
