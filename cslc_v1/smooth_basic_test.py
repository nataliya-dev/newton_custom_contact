#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigorous smoothness and differentiability tests for CSLC kernels.

This file verifies (and, where it fails, *documents*) the differentiability
claims made by `newton/_src/geometry/cslc_kernels.py`. The checks are
hermetic — no Newton Model, no SceneParams — so failures point at the
kernel math, not at integration logic.

Layout
------
1. **TestSmoothPrimitives** — `smooth_relu` and `smooth_step` are C^∞ for
   eps > 0, have the expected invariants (non-negativity, [0,1] range,
   symmetry), the correct analytic derivatives, and correct `wp.Tape`
   backward.

2. **TestKernel1PhiSmoothness** — `compute_cslc_penetration_sphere`
   returns phi that is a continuous (Lipschitz) function of the target
   position when sweeping through the contact-active boundary.

3. **TestLatticeSolveEquilibrium** — `lattice_solve_equilibrium` is a
   single linear matvec delta = kc · A_inv · phi, which is tape-safe
   (no Python-side buffer aliasing). Value, linearity, tape-backward,
   and Jacobian vs central-FD all verified.

4. **TestJacobiConvergesToAnalytic** — iterated `jacobi_step` (with the
   src/dst swap pattern the handler uses) converges to the same fixed
   point as `lattice_solve_equilibrium` in the all-contact-active limit.

5. **TestKernel3WriteCullGap** — DOCUMENTS a known remaining
   non-smoothness: `write_cslc_contacts` retains hard `if d_proj <= 0`
   and `if pen_3d <= 0` early-returns. When the target crosses either
   boundary, `out_shape0` flips between a valid index and -1, and
   `out_stiffness` stops being written. This creates a discrete jump in
   the force delivered to the downstream rigid-body solver. The test
   quantifies the jump so the fix (route the write through
   `smooth_step(d_proj, eps) * smooth_step(pen_3d, eps)`) can be
   validated later.

Run
---
    uv run --extra dev -m unittest cslc_v1.smooth_basic_test -v
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.cslc_kernels import (
    compute_cslc_penetration_sphere,
    jacobi_step,
    lattice_solve_equilibrium,
    smooth_relu,
    smooth_step,
    write_cslc_contacts,
)


# ══════════════════════════════════════════════════════════════════════════
#  Wrapper kernels — expose @wp.func primitives to Python
# ══════════════════════════════════════════════════════════════════════════

@wp.kernel
def _eval_smooth_relu(
    x: wp.array(dtype=wp.float32),
    eps: float,
    y: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    y[i] = smooth_relu(x[i], eps)


@wp.kernel
def _eval_smooth_step(
    x: wp.array(dtype=wp.float32),
    eps: float,
    y: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    y[i] = smooth_step(x[i], eps)


# ══════════════════════════════════════════════════════════════════════════
#  Analytic references (float64 numpy — ground truth)
# ══════════════════════════════════════════════════════════════════════════

def np_smooth_relu(x, eps):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (x + np.sqrt(x * x + eps * eps))


def np_smooth_step(x, eps):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + x / np.sqrt(x * x + eps * eps))


def np_smooth_step_deriv(x, eps):
    # d/dx [0.5·(1 + x/sqrt(x² + eps²))] = 0.5·eps² / (x² + eps²)^{3/2}
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * eps * eps / np.power(x * x + eps * eps, 1.5)


# ══════════════════════════════════════════════════════════════════════════
#  Device selection
# ══════════════════════════════════════════════════════════════════════════

def _pick_device():
    try:
        wp.get_device("cuda:0")
        return "cuda:0"
    except Exception:
        return "cpu"


# ══════════════════════════════════════════════════════════════════════════
#  Test 1 — smooth primitives (smooth_relu, smooth_step)
# ══════════════════════════════════════════════════════════════════════════

class TestSmoothPrimitives(unittest.TestCase):
    """Verify the @wp.func surrogates are correct C^∞ smooth replacements
    for ReLU and Heaviside, with matching analytic derivatives and
    working wp.Tape backward."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = _pick_device()

    def setUp(self):
        self.eps = 1.0e-5
        # Dense near the transition x=0 (width ~10·eps), coarse elsewhere.
        self.xs = np.concatenate([
            np.linspace(-1.0, -1e-4, 200),
            np.linspace(-1e-4, 1e-4, 2000),
            np.linspace(1e-4, 1.0, 200),
        ]).astype(np.float32)

    def _eval(self, kernel, xs=None):
        xs = self.xs if xs is None else np.asarray(xs, dtype=np.float32)
        n = len(xs)
        x_wp = wp.array(xs, dtype=wp.float32, device=self.device)
        y_wp = wp.zeros(n, dtype=wp.float32, device=self.device)
        wp.launch(kernel=kernel, dim=n,
                  inputs=[x_wp, self.eps], outputs=[y_wp], device=self.device)
        return y_wp.numpy()

    # ── value correctness ────────────────────────────────────────────────
    def test_smooth_relu_matches_analytic(self):
        y = self._eval(_eval_smooth_relu)
        np.testing.assert_allclose(y, np_smooth_relu(self.xs, self.eps),
                                   rtol=1e-4, atol=1e-8)

    def test_smooth_step_matches_analytic(self):
        """fp32 atol = 1e-6 ≈ one fp32 ULP around 1.  The float64 reference
        returns values down to ~1e-10 in the |x| ≫ eps tail, which fp32
        can't resolve (sqrt(x² + eps²) rounds to |x| exactly) — not a
        kernel bug, just machine precision."""
        y = self._eval(_eval_smooth_step)
        np.testing.assert_allclose(y, np_smooth_step(self.xs, self.eps),
                                   rtol=1e-4, atol=1e-6)

    # ── invariants ───────────────────────────────────────────────────────
    def test_smooth_relu_is_non_negative(self):
        """δ ≥ 0 invariant in `jacobi_step` relies on smooth_relu ≥ 0 for all x."""
        y = self._eval(_eval_smooth_relu)
        self.assertGreaterEqual(float(y.min()), 0.0,
                                "smooth_relu produced a negative value")

    def test_smooth_relu_minimum_is_eps_over_two(self):
        """smooth_relu(0, eps) = eps/2 exactly (smallest value it can take)."""
        y = self._eval(_eval_smooth_relu, xs=[0.0])
        self.assertAlmostEqual(float(y[0]), 0.5 * self.eps, delta=1e-9)

    def test_smooth_step_is_in_unit_interval(self):
        """Contact-active gate must lie in [0, 1] for any x."""
        y = self._eval(_eval_smooth_step)
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_smooth_step_complementary(self):
        """smooth_step(-x) + smooth_step(x) = 1 (algebraic identity)."""
        xs = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
        y_pos = self._eval(_eval_smooth_step, xs)
        y_neg = self._eval(_eval_smooth_step, -xs)
        np.testing.assert_allclose(y_pos + y_neg, np.ones_like(xs),
                                   rtol=1e-5, atol=1e-7)

    # ── Lipschitz continuity ─────────────────────────────────────────────
    def test_smooth_relu_lipschitz_bound_one(self):
        """d/dx smooth_relu = smooth_step ∈ [0,1] ⇒ Lipschitz const ≤ 1."""
        y = self._eval(_eval_smooth_relu)
        dy = np.abs(np.diff(y))
        dx = np.abs(np.diff(self.xs))
        # fp32 slack grows with |y|
        slack = 1e-5 * np.maximum(np.abs(y[:-1]), np.abs(y[1:])) + 1e-7
        self.assertTrue(np.all(dy <= dx + slack),
                        f"worst Lipschitz ratio: "
                        f"{(dy / np.maximum(dx, 1e-12)).max():.3f}")

    def test_smooth_step_lipschitz_bound(self):
        """|d/dx smooth_step| ≤ 1/(2·eps). With eps=1e-5 → L ≤ 5e4."""
        y = self._eval(_eval_smooth_step)
        dy = np.abs(np.diff(y))
        dx = np.abs(np.diff(self.xs))
        L = 1.0 / (2.0 * self.eps)
        self.assertTrue(np.all(dy <= L * dx + 1e-5),
                        f"worst ratio: {(dy / np.maximum(dx, 1e-12)).max():.1f}")

    # ── derivatives (FD vs analytic) ─────────────────────────────────────
    def test_smooth_relu_derivative_matches_smooth_step(self):
        """d/dx smooth_relu(x, eps) = smooth_step(x, eps) — verified by central FD.

        Central-FD truncation error is |h² · f'''(x) / 6|.  Near the
        transition (|x| ~ eps), |f'''| peaks at ~1/eps², so we need
        h ≪ eps to keep the FD truncation below rtol·|f'|.  Picked
        h = eps/50 = 2e-7; in float64 the subtraction yp − ym still has
        ≫ 10 digits of agreement with the analytic derivative.
        """
        xs = np.linspace(-0.01, 0.01, 101)
        h = self.eps / 50.0  # 2e-7
        fd = (np_smooth_relu(xs + h, self.eps) -
              np_smooth_relu(xs - h, self.eps)) / (2.0 * h)
        analytic = np_smooth_step(xs, self.eps)
        np.testing.assert_allclose(fd, analytic, rtol=1e-3, atol=1e-4)

    def test_smooth_step_derivative_matches_closed_form(self):
        """d/dx smooth_step = 0.5·eps²/(x²+eps²)^{3/2} — verified by central FD.

        Picked x sample spacing so FD step h = eps/10 is safe: small enough to
        resolve the peak (height 1/(2·eps) at x=0), large enough to avoid
        catastrophic cancellation in float64."""
        xs = np.linspace(-5 * self.eps, 5 * self.eps, 101)
        h = self.eps / 10.0
        fd = (np_smooth_step(xs + h, self.eps) -
              np_smooth_step(xs - h, self.eps)) / (2.0 * h)
        analytic = np_smooth_step_deriv(xs, self.eps)
        np.testing.assert_allclose(fd, analytic, rtol=1e-2, atol=1.0)

    # ── eps → 0 recovers hard ReLU / Heaviside ──────────────────────────
    def test_smooth_relu_recovers_hard_relu_small_eps(self):
        """|smooth_relu(x, eps) - max(x, 0)| ≤ eps/2 for all x, any eps."""
        xs = np.array([-1.0, -0.01, 0.01, 1.0])
        for eps in [1e-3, 1e-5, 1e-7]:
            y = np_smooth_relu(xs, eps)
            expected = np.maximum(xs, 0.0)
            self.assertTrue(np.all(np.abs(y - expected) <= 0.5 * eps + 1e-15),
                            f"eps={eps}: y={y}, expected={expected}")

    def test_smooth_step_recovers_heaviside_far_from_zero(self):
        """For |x| ≫ eps: smooth_step(x>0) ≈ 1, smooth_step(x<0) ≈ 0."""
        eps = 1e-5
        xs_pos = np.array([100 * eps, 1.0])
        xs_neg = np.array([-100 * eps, -1.0])
        self.assertTrue(np.all(np_smooth_step(xs_pos, eps) > 0.9999))
        self.assertTrue(np.all(np_smooth_step(xs_neg, eps) < 0.0001))

    # ── wp.Tape backward ─────────────────────────────────────────────────
    def test_smooth_relu_tape_backward_matches_smooth_step(self):
        """Σ smooth_relu(x_i) backward — gradient = smooth_step pointwise."""
        xs = np.array([-1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-2], dtype=np.float32)
        n = len(xs)
        x_wp = wp.array(xs, dtype=wp.float32, device=self.device,
                        requires_grad=True)
        y_wp = wp.zeros(n, dtype=wp.float32, device=self.device,
                        requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel=_eval_smooth_relu, dim=n,
                      inputs=[x_wp, self.eps], outputs=[y_wp],
                      device=self.device)
        y_wp.grad = wp.array(np.ones(n, dtype=np.float32),
                             dtype=wp.float32, device=self.device)
        tape.backward()
        np.testing.assert_allclose(x_wp.grad.numpy(),
                                   np_smooth_step(xs, self.eps),
                                   rtol=1e-3, atol=1e-5)

    def test_smooth_step_tape_backward_matches_analytic(self):
        """Σ smooth_step(x_i) backward — gradient = 0.5·eps²/(x²+eps²)^{3/2}."""
        xs = np.array([-5e-5, -1e-5, 0.0, 1e-5, 5e-5], dtype=np.float32)
        n = len(xs)
        x_wp = wp.array(xs, dtype=wp.float32, device=self.device,
                        requires_grad=True)
        y_wp = wp.zeros(n, dtype=wp.float32, device=self.device,
                        requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel=_eval_smooth_step, dim=n,
                      inputs=[x_wp, self.eps], outputs=[y_wp],
                      device=self.device)
        y_wp.grad = wp.array(np.ones(n, dtype=np.float32),
                             dtype=wp.float32, device=self.device)
        tape.backward()
        got = x_wp.grad.numpy()
        analytic = np_smooth_step_deriv(xs, self.eps)
        # Peak magnitude is 1/(2·eps) = 5e4 for eps=1e-5.  Use loose atol.
        np.testing.assert_allclose(got, analytic, rtol=1e-2, atol=5.0)


# ══════════════════════════════════════════════════════════════════════════
#  Test 2 — Kernel 1 phi is a smooth function of target position
# ══════════════════════════════════════════════════════════════════════════

class TestKernel1PhiSmoothness(unittest.TestCase):
    """`compute_cslc_penetration_sphere` should be continuous in the target
    sphere's position, because its contact-active gate and ReLU are both
    smoothed. We sweep the target across the contact boundary and verify
    Lipschitz behaviour."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = _pick_device()

    def setUp(self):
        self.r_lat = 0.005        # 5 mm lattice sphere
        self.r_target = 0.010     # 10 mm target sphere
        self.eps = 1.0e-5

    def _phi(self, target_x, target_y=0.0, target_z=0.0, delta=0.0):
        """Launch Kernel 1 for one lattice sphere at world origin (normal
        +x) and one target at (target_x, target_y, target_z). Returns phi."""
        d = self.device
        pad_tx = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        tgt_tx = wp.transform(
            wp.vec3(float(target_x), float(target_y), float(target_z)),
            wp.quat_identity())

        raw_pen = wp.zeros(1, dtype=wp.float32, device=d)
        contact_normal = wp.zeros(1, dtype=wp.vec3, device=d)

        wp.launch(
            kernel=compute_cslc_penetration_sphere,
            dim=1,
            inputs=[
                wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=d),  # pos
                wp.array([self.r_lat], dtype=wp.float32, device=d),    # radii
                wp.array([float(delta)], dtype=wp.float32, device=d),  # delta
                wp.array([0], dtype=wp.int32, device=d),               # shape
                wp.array([1], dtype=wp.int32, device=d),               # is_surface
                wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=d),  # normal
                wp.array([pad_tx, tgt_tx], dtype=wp.transform, device=d),  # body_q
                wp.array([0], dtype=wp.int32, device=d),               # shape_body
                wp.array([wp.transform_identity()], dtype=wp.transform, device=d),
                0,   # active_cslc_shape_idx
                1,   # target_body_idx
                -1,  # target_shape_idx (unused in Kernel 1)
                wp.vec3(0.0, 0.0, 0.0),  # target_local_pos
                self.r_target, self.eps,
            ],
            outputs=[raw_pen, contact_normal],
            device=d,
        )
        return float(raw_pen.numpy()[0])

    def test_phi_is_zero_when_target_is_far_behind_face(self):
        """d_proj ≪ -eps ⇒ smooth_step(d_proj, eps) ≈ 0 ⇒ phi ≈ 0."""
        self.assertLess(self._phi(-0.05), self.eps,
                        "phi should vanish behind the face")

    def test_phi_equals_pen_3d_well_inside_contact(self):
        """d_proj ≫ eps and pen_3d ≫ eps ⇒ phi ≈ pen_3d.

        target at (0.005, 0, 0): dist = 0.005, pen_3d = 0.015 - 0.005 = 0.010,
        d_proj = 0.005 (≫ eps). Expected phi ≈ 0.010."""
        phi = self._phi(0.005)
        expected_pen_3d = (self.r_lat + self.r_target) - 0.005
        np.testing.assert_allclose(phi, expected_pen_3d,
                                   rtol=1e-3, atol=1e-6)

    def test_phi_is_zero_when_spheres_not_overlapping(self):
        """pen_3d < 0 ⇒ smooth_relu(pen_3d, eps) ≈ 0 ⇒ phi ≈ 0 (even if
        d_proj > 0)."""
        # target far away: dist = 1 m >> r_lat + r_target
        self.assertLess(self._phi(1.0), self.eps)

    def test_phi_lipschitz_across_d_proj_zero(self):
        """Sweeping target through x=0 (d_proj crosses zero): phi is
        continuous.  Adjacent-sample |Δphi| must be bounded by a finite
        Lipschitz constant — discrete jumps would indicate a hard branch."""
        # Sweep x from behind the face to full-contact, dense near 0.
        xs = np.concatenate([
            np.linspace(-0.01, -1e-5, 50),
            np.linspace(-1e-5, 1e-5, 500),
            np.linspace(1e-5, 0.01, 50),
        ])
        phis = np.array([self._phi(float(x)) for x in xs])
        dphi = np.abs(np.diff(phis))
        dx = np.abs(np.diff(xs))
        # Theoretical: near d_proj=0, phi ≈ pen_3d · smooth_step(x, eps), so
        # dphi/dx ≤ pen_3d / (2·eps) ≈ 0.015/(2e-5) ≈ 750.  Allow 100× slack.
        L_bound = (self.r_lat + self.r_target) / (2.0 * self.eps) * 100.0
        worst = (dphi / np.maximum(dx, 1e-12)).max()
        self.assertLess(worst, L_bound,
                        f"Kernel 1 phi not smooth across d_proj=0: "
                        f"max(dphi/dx) = {worst:.1f}, bound = {L_bound:.1f}")

    def test_phi_lipschitz_across_pen_3d_zero(self):
        """Sweep target along +x radially away from lattice sphere,
        crossing pen_3d = 0 at x = r_lat + r_target = 0.015."""
        xs = np.linspace(0.010, 0.025, 1001)
        phis = np.array([self._phi(float(x)) for x in xs])
        dphi = np.abs(np.diff(phis))
        dx = np.abs(np.diff(xs))
        # Near pen_3d=0, smooth_relu has slope = smooth_step ∈ [0, 1], so
        # dphi/dx ≤ 1 (plus a tiny gate-derivative contribution that is ≪
        # 1 here because d_proj is ~ x ≫ eps).
        self.assertLess((dphi / np.maximum(dx, 1e-12)).max(), 2.0,
                        "phi not smooth across pen_3d=0")

    def test_phi_ignores_delta(self):
        """Kernel 1 does NOT subtract delta from r_lat — that happens in
        Kernel 3.  Verify phi is independent of the delta input."""
        phi0 = self._phi(0.005, delta=0.0)
        phi1 = self._phi(0.005, delta=0.002)
        np.testing.assert_allclose(phi0, phi1, rtol=1e-6, atol=1e-9)


# ══════════════════════════════════════════════════════════════════════════
#  Test 3 — lattice_solve_equilibrium (one-shot tape-safe lattice solve)
# ══════════════════════════════════════════════════════════════════════════

class TestLatticeSolveEquilibrium(unittest.TestCase):
    """`lattice_solve_equilibrium` is the tape-safe replacement for iterated
    `jacobi_step` (which cannot backprop because of Python-side buffer
    aliasing).  It computes delta = kc · A_inv · phi exactly.

    We use a 3-node 1D chain lattice so we can build K by hand and invert
    it in numpy.  Node 0 — 1 — 2 with degrees (1, 2, 1).
    """

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = _pick_device()

    def setUp(self):
        self.n = 3
        self.ka = 1000.0
        self.kl = 100.0
        self.kc = 500.0
        # K: K_ii = ka + kl·deg(i); K_ij = -kl if j ∈ N(i).
        self.K = np.array([
            [self.ka + self.kl,         -self.kl,                 0.0],
            [-self.kl,           self.ka + 2 * self.kl,     -self.kl],
            [0.0,                       -self.kl,      self.ka + self.kl],
        ], dtype=np.float64)
        self.A = self.K + self.kc * np.eye(self.n)
        self.A_inv = np.linalg.inv(self.A).astype(np.float32)

    def _solve(self, phi_np):
        d = self.device
        A_inv_wp = wp.array(self.A_inv, dtype=wp.float32, device=d)
        phi_wp = wp.array(phi_np.astype(np.float32), dtype=wp.float32, device=d)
        delta_wp = wp.zeros(len(phi_np), dtype=wp.float32, device=d)
        wp.launch(kernel=lattice_solve_equilibrium, dim=len(phi_np),
                  inputs=[A_inv_wp, phi_wp, self.kc], outputs=[delta_wp],
                  device=d)
        return delta_wp.numpy()

    def test_value_matches_closed_form(self):
        phi = np.array([0.010, 0.005, 0.000], dtype=np.float32)
        expected = self.kc * (self.A_inv.astype(np.float64) @
                              phi.astype(np.float64))
        np.testing.assert_allclose(self._solve(phi), expected,
                                   rtol=1e-4, atol=1e-7)

    def test_linear_in_phi(self):
        phi = np.array([0.003, 0.005, 0.001], dtype=np.float32)
        d1 = self._solve(phi)
        d2 = self._solve(2.0 * phi)
        np.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-4, atol=1e-7)

    def test_zero_phi_gives_zero_delta(self):
        np.testing.assert_allclose(self._solve(np.zeros(self.n, dtype=np.float32)),
                                   0.0, atol=1e-7)

    def test_tape_backward_matches_transpose(self):
        """d(Σ δ_i)/dφ_j = kc · Σ_i A_inv[i, j]."""
        phi_np = np.array([0.005, 0.010, 0.002], dtype=np.float32)
        d = self.device
        A_inv_wp = wp.array(self.A_inv, dtype=wp.float32, device=d)
        phi_wp = wp.array(phi_np, dtype=wp.float32, device=d, requires_grad=True)
        delta_wp = wp.zeros(self.n, dtype=wp.float32, device=d, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(kernel=lattice_solve_equilibrium, dim=self.n,
                      inputs=[A_inv_wp, phi_wp, self.kc], outputs=[delta_wp],
                      device=d)
        delta_wp.grad = wp.array(np.ones(self.n, dtype=np.float32),
                                 dtype=wp.float32, device=d)
        tape.backward()
        analytic = self.kc * self.A_inv.astype(np.float64).sum(axis=0)
        np.testing.assert_allclose(phi_wp.grad.numpy(), analytic,
                                   rtol=1e-4, atol=1e-6)

    def test_full_jacobian_matches_central_fd(self):
        """Full J = ∂δ/∂φ = kc · A_inv recovered by central-FD."""
        phi = np.array([0.005, 0.010, 0.002], dtype=np.float32)
        h = 1e-4
        J_fd = np.zeros((self.n, self.n), dtype=np.float64)
        for j in range(self.n):
            phi_p = phi.copy(); phi_p[j] += h
            phi_m = phi.copy(); phi_m[j] -= h
            J_fd[:, j] = (self._solve(phi_p) - self._solve(phi_m)) / (2.0 * h)
        J_analytic = self.kc * self.A_inv.astype(np.float64)
        np.testing.assert_allclose(J_fd, J_analytic, rtol=1e-3, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════
#  Test 4 — iterated jacobi_step converges to the analytic solve
# ══════════════════════════════════════════════════════════════════════════

class TestJacobiConvergesToAnalytic(unittest.TestCase):
    """A single `jacobi_step` kernel launch is a smooth function of its
    inputs.  With the Python-side src/dst swap the handler uses, after
    enough iterations the result converges to `lattice_solve_equilibrium`
    — in the limit where every surface sphere is contact-active, so the
    smooth_step gate is ≈ 1 everywhere."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = _pick_device()

    def setUp(self):
        self.n = 3
        self.ka = 1000.0
        self.kl = 100.0
        self.kc = 500.0
        self.alpha = 0.6
        self.eps = 1e-5
        # phi_i chosen ≫ eps for all i so the smooth gate is ≈ 1 everywhere
        # and the Jacobi fixed point ≈ linear system solution.
        self.phi_np = np.array([0.010, 0.005, 0.002], dtype=np.float32)

        # Chain lattice 0-1-2
        self.nbr_list = np.array([1, 0, 2, 1], dtype=np.int32)
        self.nbr_start = np.array([0, 1, 3], dtype=np.int32)
        self.nbr_count = np.array([1, 2, 1], dtype=np.int32)

        K = np.array([
            [self.ka + self.kl,         -self.kl,               0.0],
            [-self.kl,           self.ka + 2 * self.kl,   -self.kl],
            [0.0,                       -self.kl,    self.ka + self.kl],
        ], dtype=np.float64)
        self.A_inv = np.linalg.inv(K + self.kc * np.eye(self.n))

    def _run_jacobi(self, n_iter):
        d = self.device
        src = wp.zeros(self.n, dtype=wp.float32, device=d)
        dst = wp.zeros(self.n, dtype=wp.float32, device=d)
        phi_wp = wp.array(self.phi_np, dtype=wp.float32, device=d)
        is_surf = wp.array([1, 1, 1], dtype=wp.int32, device=d)
        sphere_shape = wp.array([0, 0, 0], dtype=wp.int32, device=d)
        nbr_start = wp.array(self.nbr_start, dtype=wp.int32, device=d)
        nbr_count = wp.array(self.nbr_count, dtype=wp.int32, device=d)
        nbr_list = wp.array(self.nbr_list, dtype=wp.int32, device=d)
        for _ in range(n_iter):
            wp.launch(kernel=jacobi_step, dim=self.n,
                      inputs=[src, dst, phi_wp, is_surf,
                              nbr_start, nbr_count, nbr_list,
                              self.ka, self.kl, self.kc, self.alpha,
                              sphere_shape, 0, self.eps],
                      device=d)
            src, dst = dst, src
        return src.numpy()

    def test_iterated_jacobi_matches_lattice_solve(self):
        """After 500 damped Jacobi sweeps, δ equals kc · A_inv · φ."""
        target = (self.kc * self.A_inv @
                  self.phi_np.astype(np.float64))
        got = self._run_jacobi(500).astype(np.float64)
        np.testing.assert_allclose(got, target, rtol=5e-3, atol=1e-5)

    def test_jacobi_error_monotonically_decreases(self):
        """Damped Jacobi on an SPD system with alpha ≤ 2/ρ_max is
        monotonically convergent.  Error norm must be non-increasing."""
        target = self.kc * self.A_inv @ self.phi_np.astype(np.float64)
        errs = []
        for n_iter in [1, 5, 20, 100, 500]:
            errs.append(float(np.linalg.norm(
                self._run_jacobi(n_iter).astype(np.float64) - target)))
        for prev, curr in zip(errs[:-1], errs[1:]):
            # Allow a tiny increase from fp32 rounding.
            self.assertLessEqual(curr, prev * 1.001 + 1e-9,
                                 f"Jacobi error increased: {errs}")


# ══════════════════════════════════════════════════════════════════════════
#  Test 5 — Kernel 3 smooth-in-practice write path
# ══════════════════════════════════════════════════════════════════════════

class TestKernel3WriteCullGap(unittest.TestCase):
    """`write_cslc_contacts` uses a hybrid emission policy:

    * Inside the physically meaningful transition region (where
      `smooth_step(d_proj, eps) * smooth_step(pen_3d, eps) ≥ 1e-4`), the
      kernel writes a valid contact with smoothly-gated stiffness.  The
      stiffness is C^∞ in the target pose across this whole band, so
      `wp.Tape` backward can flow through contact-onset transitions.

    * Deep in the tail (gate < 1e-4, i.e. |d_proj| ≳ 30·eps outside the
      boundary), the kernel hard-culls the write (shape0 = −1).  In this
      regime the smooth force is already sub-nanoNewton and its gradient
      is machine-zero, so the discrete cull costs nothing for gradient-
      based optimisation.  The cull is needed because MuJoCo's
      per-constraint compliance leak grows with the number of emitted
      constraints — see kernel comment for the lift-test regression that
      motivated this policy.

    Tests verify:

    1. Inside the transition region, `out_shape0` is always valid and
       stiffness is Lipschitz continuous across both boundaries.
    2. Deep in the tail, `out_shape0 = −1` (expected hard cull).
    3. Stiffness recovers `cslc_kc · pen_scale` exactly in fully-active
       contact and is non-negative everywhere.
    """

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = _pick_device()

    def setUp(self):
        self.r_lat = 0.005
        self.r_target = 0.010
        self.eps = 1e-5
        self.kc = 500.0
        self.dc = 0.0

    def _write_contact(self, target_x):
        """Launch Kernel 3 for a single sphere pair at (target_x, 0, 0).
        Returns (shape0, stiffness)."""
        d = self.device
        # Lattice pad: shape index 0, body 0, at world origin, +x normal.
        pad_tx = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        tgt_tx = wp.transform(wp.vec3(float(target_x), 0.0, 0.0),
                              wp.quat_identity())

        # Minimal inputs.
        out_shape0 = wp.array([-42], dtype=wp.int32, device=d)
        out_shape1 = wp.zeros(1, dtype=wp.int32, device=d)
        out_point0 = wp.zeros(1, dtype=wp.vec3, device=d)
        out_point1 = wp.zeros(1, dtype=wp.vec3, device=d)
        out_offset0 = wp.zeros(1, dtype=wp.vec3, device=d)
        out_offset1 = wp.zeros(1, dtype=wp.vec3, device=d)
        out_normal = wp.zeros(1, dtype=wp.vec3, device=d)
        out_margin0 = wp.zeros(1, dtype=wp.float32, device=d)
        out_margin1 = wp.zeros(1, dtype=wp.float32, device=d)
        out_tids = wp.zeros(1, dtype=wp.int32, device=d)
        out_stiffness = wp.array([-42.0], dtype=wp.float32, device=d)
        out_damping = wp.zeros(1, dtype=wp.float32, device=d)
        out_friction = wp.zeros(1, dtype=wp.float32, device=d)
        debug_reason = wp.zeros(1, dtype=wp.int32, device=d)
        dbg = [wp.zeros(1, dtype=wp.float32, device=d) for _ in range(5)]

        wp.launch(
            kernel=write_cslc_contacts, dim=1,
            inputs=[
                wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=d),
                wp.array([self.r_lat], dtype=wp.float32, device=d),
                wp.zeros(1, dtype=wp.float32, device=d),
                wp.array([0], dtype=wp.int32, device=d),
                wp.array([1], dtype=wp.int32, device=d),
                wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=d),
                wp.array([pad_tx, tgt_tx], dtype=wp.transform, device=d),
                wp.array([0], dtype=wp.int32, device=d),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=d),
                0, 1, 99,
                wp.vec3(0.0, 0.0, 0.0),
                self.r_target,
                0,                                      # contact_offset
                wp.array([0], dtype=wp.int32, device=d),  # surface_slot_map
                wp.zeros(1, dtype=wp.float32, device=d),  # raw_penetration (unused here)
                out_shape0, out_shape1, out_point0, out_point1,
                out_offset0, out_offset1, out_normal,
                out_margin0, out_margin1, out_tids,
                wp.array([0.5], dtype=wp.float32, device=d),  # mu (unused)
                self.kc, self.dc, self.eps,
                out_stiffness, out_damping, out_friction, debug_reason,
                0,                                      # diag_offset
                dbg[0], dbg[1], dbg[2], dbg[3], dbg[4],
            ],
            device=d,
        )
        return int(out_shape0.numpy()[0]), float(out_stiffness.numpy()[0])

    def test_shape0_valid_inside_transition_region(self):
        """Inside the smooth transition region (gate ≥ 1e-4, i.e. roughly
        |d_proj| ≤ 30·eps = 300 µm outside the boundary), shape0 is valid
        — no ±1 flips as the target crosses the face plane."""
        # Span ±20·eps across the d_proj=0 boundary.
        for target_x in np.linspace(-2e-4, 2e-4, 11):
            s0, _ = self._write_contact(float(target_x))
            self.assertEqual(s0, 0,
                             f"shape0 must stay valid in transition region; "
                             f"at target_x={target_x} got {s0}")

    def test_shape0_hard_culled_deep_in_tail(self):
        """Deep in the tail (gate ≪ 1e-4), shape0 = −1 is the intended
        hard cull — this is what prevents MuJoCo's per-constraint
        compliance leak from degrading static friction when all 378 lattice
        slots would otherwise be live."""
        # Far behind face and far separated — both gates essentially zero.
        for target_x in [-0.05, 1.0]:
            s0, _ = self._write_contact(float(target_x))
            self.assertEqual(s0, -1,
                             f"deep-tail cull expected; at target_x={target_x} got {s0}")

    def test_stiffness_matches_original_inside_contact(self):
        """Well inside the contact region (gate ≈ 1), stiffness recovers
        the ungated formula `cslc_kc * pen_scale` within fp32 tolerance."""
        target_x = 0.005  # d_proj = 0.005 ≫ eps, pen_3d = 0.010 ≫ eps
        _, k = self._write_contact(target_x)
        # Analytic: dist = 0.005, pen_3d = 0.010, d_proj = 0.005,
        # solver_pen = effective_r + R − d_proj = 0.005 + 0.010 − 0.005 = 0.010.
        # pen_scale = 0.010 / 0.010 = 1.0. Expected stiffness = cslc_kc = 500.
        self.assertAlmostEqual(k, self.kc, delta=self.kc * 1e-3,
                               msg=f"Expected ~{self.kc}, got {k}")

    def _assert_smooth_sweep(self, xs, label):
        """Common smoothness assertion: inside the emission band, no single
        adjacent-sample jump in stiffness exceeds 10 % of the peak stiffness.
        A hard discontinuity would produce a jump ≳ 50 %; a smooth C^0
        function stays well under that with dense sampling."""
        results = [self._write_contact(float(x)) for x in xs]
        shape0s = np.array([r[0] for r in results])
        stiffnesses = np.array([r[1] for r in results])
        self.assertTrue(np.all(shape0s == 0),
                        f"{label}: sweep exited the smooth-emission band "
                        f"(tighten the x range). shape0 values: "
                        f"{np.unique(shape0s)}")
        dk = np.abs(np.diff(stiffnesses))
        k_peak = float(stiffnesses.max())
        # Relative-jump criterion: smooth sweep ⇒ adjacent samples
        # vary by a small fraction of the peak value.
        tol = 0.10 * k_peak
        worst = float(dk.max())
        self.assertLess(worst, tol,
                        f"{label}: worst adjacent-sample jump {worst:.3e} "
                        f"exceeds 10 % of peak stiffness {k_peak:.3e}. "
                        f"A non-smooth operator may have been introduced.")

    def test_stiffness_lipschitz_across_d_proj_zero(self):
        """Sweep target through x=0 (the d_proj boundary) within the
        smooth-emission band.  Stiffness must be continuous here — this is
        the region where MPC/RL gradients flow through contact onset."""
        self._assert_smooth_sweep(
            np.linspace(-15 * self.eps, 15 * self.eps, 1001),
            "d_proj=0 sweep")

    def test_stiffness_lipschitz_across_pen_3d_zero(self):
        """Sweep target radially across pen_3d=0 (at x = r_lat + r_target),
        within the smooth-emission band."""
        centre = self.r_lat + self.r_target  # 0.015
        self._assert_smooth_sweep(
            np.linspace(centre - 15 * self.eps, centre + 15 * self.eps, 1001),
            "pen_3d=0 sweep")

    def test_stiffness_non_negative_everywhere(self):
        """smooth_relu floor + zero-on-cull guarantees stiffness ≥ 0 for
        any target pose, preserving the MuJoCo conversion kernel's
        positivity requirement."""
        xs = np.linspace(-0.1, 0.1, 201)
        for x in xs:
            _, k = self._write_contact(float(x))
            self.assertGreaterEqual(k, 0.0,
                                    f"negative stiffness at x={x}: {k}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
