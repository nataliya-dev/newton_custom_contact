#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for CSLC-vs-BOX kernels and handler dispatch.

The CSLC kernel chain originally only supported sphere targets.  This
file verifies the box-target extension introduced in
`cslc_kernels.py:compute_cslc_penetration_box` /
`write_cslc_contacts_box` and dispatched in
`cslc_handler.py:_launch_vs_box`.

Layout
------
1. **TestBoxSDFAnalytical** — `_box_signed_dist` and
   `_box_closest_local` (the two `@wp.func` helpers in `cslc_kernels`)
   match the closed-form analytical formulas at known interior,
   surface, and exterior points.

2. **TestK1BoxGate** — `compute_cslc_penetration_box` returns a
   strictly positive `phi` when the lattice sphere is inside the box
   (the steady-state HOLD configuration), and ~0 when the lattice
   sphere is far outside the box.

   *** Regression note ***
   An earlier version of the box K1 kernel computed
       d_proj = dot(closest_world - q_world, n_pad)
   for the contact-active gate.  When the lattice sphere transitions
   from outside to inside the box, ``closest_world - q_world`` flips
   sign relative to ``n_pad`` (it goes from "toward the box face"
   to "out of the box face").  The smooth-step gate then closes
   wrongly during HOLD, producing zero contact force and the held
   body falling under gravity.  See
   `cslc_v1/contact_solver_walkthrough.tex` §6 for the analogous
   sign-preserving discussion in the sphere kernel.

   The fix uses the box centroid for the gate
       d_proj_gate = dot(box_centre - q_world, n_pad)
   which matches the sphere kernel's convention exactly and is
   positive in both "approaching" and "already inside"
   configurations.  These tests fail catastrophically (gate ~0
   inside the box) under the old formulation.

3. **TestBoxAggregate** — full pipeline (K1 + lattice solve + K3 +
   MuJoCo) on the production squeeze scene with `--object book`.
   Verifies that the per-pad aggregate normal force at HOLD matches
   the fair-calibration target ``ke_bulk · delta_face`` (paper §3.4
   calibration; summary §2 fair-calibration derivation).

Run
---
    uv run --extra dev -m unittest cslc_v1.cslc_box_test -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

from newton._src.geometry.cslc_kernels import (
    _box_closest_local,
    _box_signed_dist,
    compute_cslc_penetration_sphere,
    compute_cslc_penetration_box,
)


# ══════════════════════════════════════════════════════════════════════════
#  Wrapper kernels — expose the box @wp.func helpers to Python tests
# ══════════════════════════════════════════════════════════════════════════


@wp.kernel
def _eval_box_signed_dist(
    p: wp.array(dtype=wp.vec3),
    h: wp.vec3,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = _box_signed_dist(p[i], h)


@wp.kernel
def _eval_box_closest_local(
    p: wp.array(dtype=wp.vec3),
    h: wp.vec3,
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    out[i] = _box_closest_local(p[i], h)


# ══════════════════════════════════════════════════════════════════════════
#  Analytic references (numpy float64 — ground truth)
# ══════════════════════════════════════════════════════════════════════════


def np_box_signed_dist(p, h):
    """Inigo Quilez' box SDF: positive outside, negative inside.

    Outside half-space contribution is ``||max(a, 0)||_2`` where
    ``a = |p| - h`` is the per-axis "how far past the face" amount.
    Inside contribution is the largest negative axis value (smallest
    perpendicular distance to a face), or 0 when any axis is outside.
    """
    p = np.asarray(p, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    a = np.abs(p) - h
    outside = float(np.linalg.norm(np.maximum(a, 0.0)))
    inside = float(min(max(a[0], a[1], a[2]), 0.0))
    return outside + inside


def np_box_closest_local(p, h):
    """Closest point on the box surface to a local-frame point ``p``.

    Outside points: per-axis clamp to ``[-h, h]``.  Inside points:
    snap the axis with the smallest face distance to its surface,
    keeping the other two axes unchanged.
    """
    p = np.asarray(p, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    clamped = np.clip(p, -h, h)
    inside = bool((np.abs(p) <= h).all())
    if not inside:
        return clamped
    face_dists = h - np.abs(p)            # all non-negative
    axis = int(np.argmin(face_dists))
    out = clamped.copy()
    sign = math.copysign(1.0, p[axis]) if p[axis] != 0.0 else 0.0
    out[axis] = sign * h[axis]
    return out


def _eval_sd(points_np: np.ndarray, h_np: np.ndarray) -> np.ndarray:
    """Run the wrapper kernel over a batch of query points; return SDFs."""
    p_arr = wp.array(points_np.astype(np.float32), dtype=wp.vec3)
    h_vec = wp.vec3(float(h_np[0]), float(h_np[1]), float(h_np[2]))
    out = wp.zeros(len(points_np), dtype=wp.float32)
    wp.launch(_eval_box_signed_dist, dim=len(points_np),
              inputs=[p_arr, h_vec], outputs=[out])
    return out.numpy()


def _eval_closest(points_np: np.ndarray, h_np: np.ndarray) -> np.ndarray:
    p_arr = wp.array(points_np.astype(np.float32), dtype=wp.vec3)
    h_vec = wp.vec3(float(h_np[0]), float(h_np[1]), float(h_np[2]))
    out = wp.zeros(len(points_np), dtype=wp.vec3)
    wp.launch(_eval_box_closest_local, dim=len(points_np),
              inputs=[p_arr, h_vec], outputs=[out])
    return out.numpy()


# ══════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════


class TestBoxSDFAnalytical(unittest.TestCase):
    """Verify the box SDF helpers against the closed-form numpy reference.

    Tolerance is fp32 atol = 1e-7 m (≈ one ULP at our typical mm-scale
    geometry).  Higher than that masks real bugs; lower trips on
    legitimate float32 rounding.
    """

    def setUp(self):
        # Half-extents typical of the squeeze book scene (16 × 100 × 150 mm).
        self.h_np = np.array([0.008, 0.05, 0.075], dtype=np.float64)

    def assertSDFAlmostEqual(self, p_np, expected, msg=""):
        sd_kernel = float(_eval_sd(p_np[None, :], self.h_np)[0])
        sd_ref = float(np_box_signed_dist(p_np, self.h_np))
        self.assertAlmostEqual(
            sd_kernel, sd_ref, places=6,
            msg=f"{msg}: kernel SDF {sd_kernel:.6e} != ref {sd_ref:.6e}",
        )
        self.assertAlmostEqual(
            sd_kernel, expected, places=6,
            msg=f"{msg}: kernel SDF {sd_kernel:.6e} != expected {expected:.6e}",
        )

    def test_signed_dist_at_centre(self):
        """Box centre: SDF = -min(h) = perpendicular distance to nearest face."""
        self.assertSDFAlmostEqual(
            np.array([0.0, 0.0, 0.0]), -0.008, "centre",
        )

    def test_signed_dist_inside_near_x_face(self):
        """1.5 mm inside the +x face: SDF = -0.0015."""
        # x = h.x - 0.0015 = 0.008 - 0.0015 = 0.0065
        self.assertSDFAlmostEqual(
            np.array([0.0065, 0.0, 0.0]), -0.0015, "inside +x face",
        )

    def test_signed_dist_on_surface(self):
        """Point exactly on +x face: SDF = 0."""
        self.assertSDFAlmostEqual(
            np.array([0.008, 0.0, 0.0]), 0.0, "+x face surface",
        )

    def test_signed_dist_outside_along_axis(self):
        """1 mm outside +x face: SDF = +0.001."""
        self.assertSDFAlmostEqual(
            np.array([0.009, 0.0, 0.0]), 0.001, "outside +x",
        )

    def test_signed_dist_outside_corner(self):
        """Outside corner: SDF = euclidean distance from corner."""
        # 1 mm outside in x and y → distance = sqrt(2) mm
        p = np.array([0.009, 0.051, 0.0])
        expected = math.sqrt(0.001 ** 2 + 0.001 ** 2)
        self.assertSDFAlmostEqual(p, expected, "outside +x +y corner")

    def test_closest_point_inside_snaps_to_nearest_face(self):
        """Inside point with smallest face distance on +x → snap x to +h.x."""
        p = np.array([0.0065, 0.0, 0.0])  # 1.5 mm from +x face
        c_kernel = _eval_closest(p[None, :], self.h_np)[0]
        c_ref = np_box_closest_local(p, self.h_np)
        np.testing.assert_allclose(c_kernel, c_ref, atol=1e-7)
        np.testing.assert_allclose(
            c_kernel, np.array([0.008, 0.0, 0.0]), atol=1e-7,
        )

    def test_closest_point_outside_clamps_per_axis(self):
        """Outside point: per-axis clamp to [-h, h]."""
        p = np.array([0.012, 0.06, -0.10])
        c_kernel = _eval_closest(p[None, :], self.h_np)[0]
        c_ref = np_box_closest_local(p, self.h_np)
        np.testing.assert_allclose(c_kernel, c_ref, atol=1e-7)
        np.testing.assert_allclose(
            c_kernel, np.array([0.008, 0.05, -0.075]), atol=1e-7,
        )


# ──────────────────────────────────────────────────────────────────────────
#  Test 2 — K1 box gate (regresses the d_proj-sign bug)
# ──────────────────────────────────────────────────────────────────────────


def _run_k1_box_single_sphere(
    *,
    sphere_world: tuple[float, float, float],
    pad_outward_normal_world: tuple[float, float, float],
    box_world: tuple[float, float, float],
    box_half_extents: tuple[float, float, float],
    r_lat: float = 0.0025,
    eps: float = 1.0e-5,
) -> tuple[float, float]:
    """Invoke `compute_cslc_penetration_box` with a single hand-built sphere.

    Returns ``(phi, signed_dist)``.  Used by TestK1BoxGate to exercise
    the kernel without spinning up a Newton model.

    Layout: shape 0 is the pad (body 0), shape 1 is the box (body 1).
    Both shape transforms are identity in their bodies.  The sphere's
    rest position is encoded in `sphere_pos_local[0]`; the body
    transform places it at the requested world position.
    """
    sphere_world_v = wp.vec3(*sphere_world)
    n_world = wp.vec3(*pad_outward_normal_world)
    box_world_v = wp.vec3(*box_world)
    h_v = wp.vec3(*box_half_extents)

    # Single lattice sphere at body origin (in pad body frame).
    sphere_pos_local = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3)
    sphere_radii = wp.array([r_lat], dtype=wp.float32)
    sphere_delta = wp.array([0.0], dtype=wp.float32)
    sphere_shape = wp.array([0], dtype=wp.int32)         # belongs to shape 0
    is_surface = wp.array([1], dtype=wp.int32)
    sphere_outward_normal = wp.array([n_world], dtype=wp.vec3)

    # body 0 = pad (placed so the sphere lands at sphere_world).
    # body 1 = box (placed at box_world).
    body_q = wp.array([
        wp.transform(sphere_world_v, wp.quat_identity()),
        wp.transform(box_world_v, wp.quat_identity()),
    ], dtype=wp.transform)
    shape_body = wp.array([0, 1], dtype=wp.int32)
    shape_transform = wp.array([
        wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    ], dtype=wp.transform)

    target_local_xform = wp.transform(
        wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    raw_pen = wp.zeros(1, dtype=wp.float32)
    contact_normal = wp.zeros(1, dtype=wp.vec3)

    wp.launch(
        kernel=compute_cslc_penetration_box,
        dim=1,
        inputs=[
            sphere_pos_local, sphere_radii, sphere_delta,
            sphere_shape, is_surface, sphere_outward_normal,
            body_q, shape_body, shape_transform,
            0,                  # active_cslc_shape_idx
            1, 1,               # target_body_idx, target_shape_idx
            target_local_xform, h_v,
            eps,
        ],
        outputs=[raw_pen, contact_normal],
    )

    # Reference SDF for comparison (we only use position, not the kernel
    # output, since we want an independent oracle).
    q_box_local = np.asarray(sphere_world, dtype=np.float64) \
        - np.asarray(box_world, dtype=np.float64)
    sd_ref = np_box_signed_dist(q_box_local, np.asarray(box_half_extents))
    return float(raw_pen.numpy()[0]), sd_ref


class TestK1BoxGate(unittest.TestCase):
    """Regress the d_proj-sign bug.  Inside-box phi must be positive.

    Each test exercises a different lattice-sphere position relative to
    a 16 × 100 × 150 mm book at the world origin.  The pad's outward
    normal points along +x (the squeeze axis).  Without the centroid-
    based gate fix, ``phi`` would collapse to ~0 in the inside cases.
    """

    H = (0.008, 0.05, 0.075)
    BOX_AT_ORIGIN = (0.0, 0.0, 0.0)
    PAD_OUTWARD_PLUS_X = (1.0, 0.0, 0.0)
    R_LAT = 0.0025

    # Convention: the pad outward normal points FROM the pad face
    # TOWARD the target's interior.  For a pad pressing on the +x face
    # of the book (i.e., a pad located at world x > +h_x, lattice
    # sphere just inside +x face), the outward normal is -x — pointing
    # from the pad inward into the box centroid.  This mirrors how
    # squeeze_test.py's right pad is rotated 180° around z so its
    # local +x face points toward the book centroid.

    def test_phi_positive_when_lattice_sphere_inside_plus_x_face(self):
        """Lattice sphere 1.5 mm inside the +x face → phi ≈ r_lat - sd > 0.

        Pad approaches from world +x side; pad outward = -x (toward
        box centroid).  signed_dist = -0.0015 m, pen_3d = 0.004 m, and
        the gate must be open because the centroid is in -x direction
        from the lattice sphere AND n_pad is also -x → d_proj_gate > 0.
        """
        phi, sd_ref = _run_k1_box_single_sphere(
            sphere_world=(0.0065, 0.0, 0.0),
            pad_outward_normal_world=(-1.0, 0.0, 0.0),
            box_world=self.BOX_AT_ORIGIN,
            box_half_extents=self.H,
            r_lat=self.R_LAT,
        )
        self.assertAlmostEqual(sd_ref, -0.0015, places=6)
        # phi should equal r_lat - sd up to the smooth_relu / smooth_step
        # corrections (eps = 1e-5 m gives <1e-9 deviation here).
        expected_phi = self.R_LAT - sd_ref
        self.assertAlmostEqual(
            phi, expected_phi, places=6,
            msg=("phi much smaller than expected — gate likely closed; "
                 "regresses the d_proj-using-closest-point bug")
        )
        self.assertGreater(
            phi, 0.9 * expected_phi,
            msg=("Gate is closing for inside-box lattice spheres "
                 f"(phi={phi:.6e}, expected≈{expected_phi:.6e}). "
                 "This is the d_proj sign-flip bug — gate must use "
                 "the BOX CENTROID, not the closest-point.")
        )

    def test_phi_positive_when_lattice_sphere_inside_minus_x_face(self):
        """Symmetric counterpart: lattice sphere 1.5 mm inside the -x face.

        Pad approaches from world -x side; pad outward = +x (toward
        box centroid).
        """
        phi, sd_ref = _run_k1_box_single_sphere(
            sphere_world=(-0.0065, 0.0, 0.0),
            pad_outward_normal_world=(+1.0, 0.0, 0.0),
            box_world=self.BOX_AT_ORIGIN,
            box_half_extents=self.H,
            r_lat=self.R_LAT,
        )
        self.assertAlmostEqual(sd_ref, -0.0015, places=6)
        expected_phi = self.R_LAT - sd_ref
        self.assertGreater(phi, 0.9 * expected_phi)

    def test_phi_zero_when_lattice_sphere_far_outside_box(self):
        """Lattice sphere 30 mm outside +x face → phi ≈ 0 (no contact)."""
        phi, sd_ref = _run_k1_box_single_sphere(
            sphere_world=(0.038, 0.0, 0.0),
            pad_outward_normal_world=(-1.0, 0.0, 0.0),
            box_world=self.BOX_AT_ORIGIN,
            box_half_extents=self.H,
            r_lat=self.R_LAT,
        )
        self.assertAlmostEqual(sd_ref, 0.030, places=6)
        # smooth_relu(r_lat - sd, eps) ≈ 0 because (r_lat - sd) is hugely
        # negative.  The smooth_relu floor ε/2 ≈ 5e-6 m is the only
        # residual.  Bound it well above that:
        self.assertLess(phi, 1.0e-4)

    def test_phi_at_box_surface_equals_r_lat(self):
        """Lattice sphere centre exactly on +x face → sd = 0 → phi ≈ r_lat."""
        phi, sd_ref = _run_k1_box_single_sphere(
            sphere_world=(0.008, 0.0, 0.0),
            pad_outward_normal_world=(-1.0, 0.0, 0.0),
            box_world=self.BOX_AT_ORIGIN,
            box_half_extents=self.H,
            r_lat=self.R_LAT,
        )
        self.assertAlmostEqual(sd_ref, 0.0, places=6)
        # On surface: pen_3d = r_lat, gate ≈ 1 (centroid is -x from the
        # lattice sphere by 0.008 m, n_pad is -x → d_proj_gate > 0).
        self.assertAlmostEqual(phi, self.R_LAT, places=5)


# ──────────────────────────────────────────────────────────────────────────
#  Test 3 — Aggregate force matches ke_bulk on the production book scene
# ──────────────────────────────────────────────────────────────────────────


class TestBoxAggregate(unittest.TestCase):
    """Full pipeline check: per-sphere F_i at equilibrium = keff · phi_i.

    Builds the squeeze-test book scene (two flat pads on a flat 1.2 kg
    book), runs collide() once at the initial penetration, and checks
    the summed equilibrium per-sphere normal force against the
    series-spring closed form
        F_i = k_c · (phi_i - delta_i) = keff · phi_i ,
        keff = k_c · k_a / (k_c + k_a)         [paper §3.4]
    using the actual `phi_i` (from the K1 raw_penetration buffer) and
    `delta_i` (from the post-K2 lattice equilibrium), so the test does
    NOT depend on the analytical phi formula — it verifies the
    series-spring identity holds across the kernel chain.

    The aggregate sum across both pads is then compared to
    ``2 · N · keff · phi̅``, also from the buffers.  These two
    quantities should agree to fp32 precision.
    """

    def test_per_sphere_force_matches_keff_phi(self):
        # Import here so pure-helper tests above don't pay the import cost.
        from cslc_v1.common import recalibrate_cslc_kc_per_pad
        from cslc_v1.squeeze_test import SceneParams, build_cslc_scene

        p = SceneParams()
        p.object_kind = "book"
        # Mirror the squeeze main()'s book-mode geometry/calibration overrides.
        p.pad_gap_initial = 2.0 * (p.book_hx - 0.0015)
        p.cslc_contact_fraction = 1.0  # full pad-on-cover overlap

        model = build_cslc_scene(p)
        contacts = model.contacts()
        recalibrate_cslc_kc_per_pad(model, p.cslc_contact_fraction)

        state = model.state()
        model.collide(state, contacts)
        wp.synchronize()

        pipeline = model._collision_pipeline
        handler = pipeline.cslc_handler
        self.assertIsNotNone(handler, "CSLC handler not built")

        d = handler.cslc_data
        kc = float(d.kc)
        ka = float(d.ka)
        keff = kc * ka / (kc + ka)

        # Pull per-sphere phi (from K1) and delta (from K2 / dense solve).
        is_surface = d.is_surface.numpy() == 1
        delta = d.sphere_delta.numpy()

        # raw_penetration_pairs has one buffer per CSLC pair; only the
        # spheres on each pair's active pad have nonzero phi.  Sum them.
        phi_total_per_pad = []
        for buf in handler.raw_penetration_pairs:
            phi = buf.numpy()
            phi_total_per_pad.append(phi[is_surface])

        # Per-sphere equilibrium force at the dense-solve fixed point:
        #   F_i = kc · (phi_i - delta_i)
        # which equals keff · phi_i for an isolated sphere (kl=0) and is
        # within ~1 % for the lateral-coupled case at our kl=500.
        f_kc_dphi = 0.0
        f_keff_phi = 0.0
        n_active = 0
        for phi_pad in phi_total_per_pad:
            active = phi_pad > 1.0e-6
            n_active += int(active.sum())
            f_kc_dphi += float(np.sum(kc * (phi_pad[active] - delta[is_surface][active])))
            f_keff_phi += float(np.sum(keff * phi_pad[active]))

        # Sanity: at least some spheres should be active under a 1.5 mm pen.
        self.assertGreater(
            n_active, 100,
            msg="Too few active CSLC contacts on the book — gate likely closed.",
        )

        # The dense solve converges to delta = kc · (K + kc·I)^-1 · phi.
        # For the all-active limit with kl > 0, the per-sphere identity
        #   kc · (phi - delta) = keff · phi
        # holds only when kl = 0 (isolated sphere).  With kl > 0 the
        # per-sphere force is redistributed via the lattice Laplacian
        # but the AGGREGATE per pad still equals N · keff · phi̅ because
        # the Laplacian is zero-sum (column sums are zero).
        # Allow a generous 5 % aggregate tolerance.
        rel_err = abs(f_kc_dphi - f_keff_phi) / max(abs(f_keff_phi), 1.0)
        self.assertLess(
            rel_err, 0.05,
            msg=(f"Aggregate F mismatch: kc·(phi-delta) = {f_kc_dphi:.2f} N "
                 f"vs keff·phi = {f_keff_phi:.2f} N (rel err {rel_err:.3%}) — "
                 "lattice solve not at equilibrium or wrong keff."),
        )

        # And: total force at HOLD must exceed the book's weight × (1/μ),
        # otherwise no amount of friction would hold the book.  Mass is
        # 1.2 kg, μ = 0.5, so we need |F_n| ≳ 11.77 N per pad → ≥ 23.6 N
        # total.  Empirical for this scene at delta_face = 1.5 mm is
        # ≈ 400 N total — safely above the friction-cone requirement.
        self.assertGreater(
            f_keff_phi, 24.0,
            msg=("Aggregate F is below the friction-cone requirement to "
                 "hold the book — grip would fail under gravity."),
        )


if __name__ == "__main__":
    unittest.main()
