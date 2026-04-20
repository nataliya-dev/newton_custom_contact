#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""diff_test — Differentiable CSLC proof of concept for grip control.

Demonstrates that the smoothed CSLC kernels (smooth_relu / smooth_step in
newton/_src/geometry/cslc_kernels.py) are sufficient for gradient-based
grip control: backprop ∂(grip force)/∂(pad position) through the three
CSLC kernels under wp.Tape, then use that gradient to drive a closed-loop
force controller.

This is the differentiable analogue of squeeze_test: instead of position-
controlled pads, the "controller" reads the CSLC contact force and adjusts
pad_x via gradient descent to track a target normal force.

Pipeline (all five phases run from `main()`):
    Phase 1 — Forward pass:  pad_x → CSLC kernels → total_F
    Phase 2 — Gradient check: ∂F/∂pad_x via tape, vs central FD
    Phase 3 — Inverse design: gradient descent finds pad_x for F_target
    Phase 4 — Closed-loop:    track a time-varying F_target trajectory
    Phase 5 — Viewer:         visual playback of the optimization

Usage
-----
    uv run cslc_v1/diff_test.py                 # run all phases (no viewer)
    uv run cslc_v1/diff_test.py --viewer gl     # interactive viewer
    uv run cslc_v1/diff_test.py --phase 1       # run a single phase
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.cslc_data import (
    CSLCData,
    create_pad_for_box_face,
)
from newton._src.geometry.cslc_kernels import (
    compute_cslc_penetration_sphere,
    jacobi_step,
    lattice_solve_equilibrium,
    smooth_step,
)


# ── Output helpers ────────────────────────────────────────────────────────

_HSEP = "━" * 68
_SEP  = "─" * 68


def _section(title: str) -> None:
    print(f"\n{_HSEP}\n  {title}\n{_HSEP}")


def _sub(title: str) -> None:
    print(f"\n  {_SEP}\n  {title}\n  {_SEP}")


def _log(msg: str, indent: int = 0) -> None:
    print(f"  {'  ' * indent}│ {msg}")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1 helpers — minimal differentiable CSLC forward pass
# ═══════════════════════════════════════════════════════════════════════════

# Scene constants — small enough to keep gradients interpretable.
PAD_HX, PAD_HY, PAD_HZ   = 0.005, 0.015, 0.015   # 10×30×30 mm pad
PAD_SPACING              = 0.005                  # 5 mm lattice → 7×7 face
SPHERE_RADIUS            = 0.030                  # 30 mm grasped sphere
SPHERE_Z                 = 0.0                    # sphere centre on z=0

# Material parameters (matched to squeeze_test for consistency)
KA, KL, KC, DC = 5000.0, 500.0, 1087.0, 0.0
SMOOTHING_EPS  = 1.0e-3   # 1 mm — generous smoothing so gradient is visibly
                          # smooth at threshold; production runs use 1e-5

JACOBI_ITERS   = 30
JACOBI_ALPHA   = 0.6


def _build_scene(device: str = "cuda:0"):
    """Construct one CSLC pad on the LEFT (its +x face faces the sphere) and
    a sphere body on the right.  We use just one pad (not two) so the gradient
    interpretation stays simple — F_grip is the normal force from the left
    pad on the sphere; doubling it gives the symmetric two-pad analogue.

    build_A_inv=True so CSLCData precomputes the dense (K + kc·I)^-1 used
    by the tape-compatible `lattice_solve_equilibrium` kernel — the one-shot
    closed-form replacement for the iterative jacobi_step.  This preserves
    full lattice physics (ka, kl, kc) in the differentiable path.

    Body indices:
        0 — pad body (CSLC lattice)
        1 — sphere body (target)
    Shape indices:
        0 — pad box (CSLC-flagged)
        1 — sphere
    """
    pad = create_pad_for_box_face(
        PAD_HX, PAD_HY, PAD_HZ,
        face_axis=0, face_sign=+1, spacing=PAD_SPACING, shape_index=0,
    )
    data = CSLCData.from_pads(
        [pad], ka=KA, kl=KL, kc=KC, dc=DC,
        smoothing_eps=SMOOTHING_EPS, build_A_inv=True, device=device,
    )

    # Static (non-grad) auxiliary arrays.
    shape_body = wp.array([0, 1], dtype=wp.int32, device=device)
    shape_xform = wp.array(
        [wp.transform_identity(), wp.transform_identity()],
        dtype=wp.transform, device=device,
    )
    return data, pad, shape_body, shape_xform


@wp.kernel
def _build_body_q(
    pad_x: wp.array(dtype=wp.float32),
    sphere_x: float,
    sphere_z: float,
    body_q_out: wp.array(dtype=wp.transform),
):
    """Build a (2,) transform array from the scalar pad_x.
    Body 0 (pad) at (-pad_x, 0, sphere_z) with identity rotation.
    Body 1 (sphere) at (sphere_x, 0, sphere_z).

    Tape captures ∂body_q/∂pad_x so gradients can flow back to pad_x.
    """
    # Pad body: position depends on pad_x, identity rotation.
    body_q_out[0] = wp.transform(
        wp.vec3(-pad_x[0], 0.0, sphere_z),
        wp.quat_identity(),
    )
    # Sphere body: fixed.
    body_q_out[1] = wp.transform(
        wp.vec3(sphere_x, 0.0, sphere_z),
        wp.quat_identity(),
    )


@wp.kernel
def _force_per_sphere(
    phi: wp.array(dtype=wp.float32),
    delta: wp.array(dtype=wp.float32),
    is_surface: wp.array(dtype=wp.int32),
    kc: float,
    eps: float,
    F_per: wp.array(dtype=wp.float32),
):
    """F_per_sphere = kc * effective_pen * smooth_step(eff_pen, eps).

    Used by the iterative-Jacobi forward path (NOT differentiable end-to-end
    through the iteration — see _force_closed_form below for the diff path).
    """
    tid = wp.tid()
    if is_surface[tid] != 1:
        F_per[tid] = 0.0
        return
    eff_pen = phi[tid] - delta[tid]
    gate = smooth_step(eff_pen, eps)
    F_per[tid] = kc * eff_pen * gate


@wp.kernel
def _force_closed_form(
    phi: wp.array(dtype=wp.float32),
    is_surface: wp.array(dtype=wp.int32),
    keff: float,
    F_per: wp.array(dtype=wp.float32),
):
    """Closed-form CSLC per-sphere force at equilibrium (NO neighbor coupling).

    For an isolated lattice sphere with kl=0, the quasistatic equilibrium is
        δ*  = kc · φ / (ka + kc)
        F*  = kc · (φ − δ*) = kc · ka · φ / (ka + kc) = keff · φ
    where keff = kc·ka/(ka+kc) is the series-spring effective stiffness.

    This is the form we use for the differentiable forward pass:
      - Avoids the iterative Jacobi swap (which breaks wp.Tape backward
        because src/dst aliasing prevents per-iteration value recovery).
      - Drops the lateral coupling kl (analytic closed form would require
        a sparse linear solve, K^-1, against the full lattice Laplacian).
      - The signal we care about for control — F sensitive to pad position
        — is captured because phi already encodes the smoothed contact gate.

    For production simulation use the iterative path; for gradient-based
    control use this closed form.
    """
    tid = wp.tid()
    if is_surface[tid] != 1:
        F_per[tid] = 0.0
        return
    F_per[tid] = keff * phi[tid]


@wp.kernel
def _sum_total(F_per: wp.array(dtype=wp.float32),
               total: wp.array(dtype=wp.float32)):
    """Atomically reduce the per-sphere forces into a single scalar."""
    tid = wp.tid()
    wp.atomic_add(total, 0, F_per[tid])


def differentiable_forward(
    pad_x: wp.array,                      # (1,) float32, requires_grad
    body_q: wp.array,                     # (2,) transform, requires_grad
    delta_a: wp.array,                    # (n,) δ output, requires_grad
    delta_b: wp.array,                    # unused (kept for signature compat)
    raw_pen: wp.array,                    # (n,) phi buffer, requires_grad
    n_world: wp.array,                    # (n,) normals scratch, requires_grad
    F_per: wp.array,                      # (n,) per-sphere force, requires_grad
    total_F: wp.array,                    # (1,) total force, requires_grad
    data: CSLCData,
    shape_body: wp.array,
    shape_xform: wp.array,
    sphere_x: float,
    sphere_z: float,
    target_radius: float,
):
    """Differentiable CSLC forward pass: pad_x → total grip force.

    Pipeline:
        1. body_q  ← build_body_q(pad_x)              (tape ∂body_q/∂pad_x)
        2. raw_pen ← compute_cslc_penetration_sphere  (smoothed phi(pos))
        3. delta   ← lattice_solve_equilibrium(A_inv, phi, kc)
                     ↑ one-shot tape-compatible matvec: δ = kc · A_inv · φ
                     where A = K + kc·I and K is the CSLC Laplacian.
                     Restores lateral coupling (kl) in the diff path.
        4. F_per   ← _force_per_sphere(phi, delta, eps) = kc · (φ−δ) · gate
        5. total_F ← _sum_total(F_per)

    Why not the iterative jacobi_step?  The N=30 iteration with src/dst
    ping-pong buffers aliases memory in-place; wp.Tape backward can't
    reconstruct per-iteration values (Phase-2 diagnostic: no-Jacobi tape
    matched FD to 0.00%, with-Jacobi tape disagreed by 24000%).  The
    one-shot matvec via A_inv is mathematically equivalent at equilibrium
    (for the ungated linearisation) and is a pure linear op that tape
    handles correctly.  Small residual error from the smooth-relu floor
    on inactive-sphere φ — decays as smoothing_eps → 0.
    """
    n = data.n_spheres
    device = data.device

    # 1. Build body_q from pad_x.
    wp.launch(_build_body_q, dim=1,
              inputs=[pad_x, sphere_x, sphere_z],
              outputs=[body_q],
              device=device)

    # 2. Penetration kernel (smoothed gate inside).
    wp.launch(
        kernel=compute_cslc_penetration_sphere,
        dim=n,
        inputs=[
            data.positions, data.radii, data.sphere_delta,
            data.sphere_shape, data.is_surface, data.outward_normals,
            body_q, shape_body, shape_xform,
            0,                                    # active_cslc_shape_idx
            1, 1,                                 # target body / shape idx
            wp.vec3(0.0, 0.0, 0.0),               # target_local_pos
            target_radius,
            float(data.smoothing_eps),
        ],
        outputs=[raw_pen, n_world],
        device=device,
    )

    # 3. One-shot lattice equilibrium solve: δ = kc · A_inv · φ.
    # A_inv = (K + kc·I)^-1 was precomputed in CSLCData.from_pads.
    wp.launch(
        kernel=lattice_solve_equilibrium,
        dim=n,
        inputs=[data.A_inv, raw_pen, float(data.kc)],
        outputs=[delta_a],
        device=device,
    )

    # 4. Per-sphere force with smooth contact-active gate:
    #    F_i = kc · (φ_i − δ_i) · smooth_step(φ_i − δ_i, eps)
    wp.launch(
        kernel=_force_per_sphere,
        dim=n,
        inputs=[raw_pen, delta_a, data.is_surface,
                data.kc, float(data.smoothing_eps)],
        outputs=[F_per],
        device=device,
    )

    # 5. Reduce.
    total_F.zero_()
    wp.launch(_sum_total, dim=n, inputs=[F_per], outputs=[total_F],
              device=device)

    return delta_a   # converged δ available for diagnostics


# ═══════════════════════════════════════════════════════════════════════════
#  Phase scaffold — allocate persistent buffers shared across phases
# ═══════════════════════════════════════════════════════════════════════════


class DiffScene:
    """Container for the differentiable scene state and grad-tracked buffers.

    All wp arrays that are read or written inside the differentiable forward
    pass have `requires_grad=True` so that wp.Tape can backprop through them.

    Reuse across phases: each phase resets pad_x and re-runs forward(); the
    buffers persist for efficiency.
    """

    def __init__(self, sphere_x: float = 0.025, device: str = "cuda:0"):
        self.device = device
        self.sphere_x = float(sphere_x)
        self.sphere_z = float(SPHERE_Z)
        self.target_radius = SPHERE_RADIUS

        self.data, self.pad, self.shape_body, self.shape_xform = _build_scene(device)
        n = self.data.n_spheres
        self.n = n

        # All these are tape-tracked.
        self.pad_x   = wp.array([0.0], dtype=wp.float32, device=device, requires_grad=True)
        self.body_q  = wp.zeros(2, dtype=wp.transform, device=device, requires_grad=True)
        self.delta_a = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
        self.delta_b = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
        self.raw_pen = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
        self.n_world = wp.zeros(n, dtype=wp.vec3,    device=device, requires_grad=True)
        self.F_per   = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
        self.total_F = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

    def set_pad_x(self, x: float) -> None:
        self.pad_x.assign(np.array([float(x)], dtype=np.float32))

    def zero_all_grads(self) -> None:
        """Zero gradients on every tape-tracked array.

        wp.Tape.backward() ACCUMULATES gradients into the .grad fields of
        input arrays.  When the tape is run in a loop (e.g. Phase 3 / 4),
        these accumulators must be reset between iterations or gradients
        explode (we observed g_F growing 1e4 → 4e7 over 30 steps without
        this zero).
        """
        for arr in (self.pad_x, self.body_q, self.delta_a, self.delta_b,
                    self.raw_pen, self.n_world, self.F_per, self.total_F):
            if arr.grad is not None:
                arr.grad.zero_()

    def forward(self):
        """Run the differentiable pipeline.  Caller should wrap in wp.Tape()."""
        return differentiable_forward(
            self.pad_x, self.body_q,
            self.delta_a, self.delta_b,
            self.raw_pen, self.n_world,
            self.F_per, self.total_F,
            self.data, self.shape_body, self.shape_xform,
            self.sphere_x, self.sphere_z, self.target_radius,
        )

    def total_force(self) -> float:
        return float(self.total_F.numpy()[0])

    def per_sphere_force(self) -> np.ndarray:
        return self.F_per.numpy().copy()

    def per_sphere_pen(self) -> np.ndarray:
        return self.raw_pen.numpy().copy()


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1 — sanity check the forward pass
# ═══════════════════════════════════════════════════════════════════════════


def phase_1_forward(scene: DiffScene) -> None:
    _section("PHASE 1 — Differentiable forward pass")

    # Pad face at (−pad_x + PAD_HX) when pad centre = (−pad_x, 0, 0) and the
    # CSLC face is at +x relative to the body.  So the pad face's world x is:
    #     face_x = -pad_x + PAD_HX
    # The sphere surface (left edge) is at sphere_x - SPHERE_RADIUS.
    # Penetration of the face into the sphere = (-pad_x + PAD_HX) - (sphere_x - SPHERE_RADIUS).
    # We want a small positive face penetration (~2 mm) at our nominal pad_x.
    #
    # Solve:  -pad_x + PAD_HX - (sphere_x - SPHERE_RADIUS) = +0.002
    #         -pad_x = 0.002 - PAD_HX + sphere_x - SPHERE_RADIUS
    #         pad_x = -0.002 + PAD_HX - sphere_x + SPHERE_RADIUS
    nominal_pad_x = -0.002 + PAD_HX - scene.sphere_x + SPHERE_RADIUS
    scene.set_pad_x(nominal_pad_x)
    scene.forward()
    F = scene.total_force()
    F_per = scene.per_sphere_force()
    pen = scene.per_sphere_pen()

    n_active = int((F_per > 0.05).sum())   # spheres carrying meaningful force
    _log(f"pad_x = {nominal_pad_x*1e3:.2f} mm  (face_pen ≈ 2 mm)")
    _log(f"total_F  = {F:.3f} N  (single-pad normal force)")
    _log(f"max F_per = {F_per.max():.4f} N over {scene.n} spheres")
    _log(f"n active (F > 0.05 N) = {n_active}")
    _log(f"max pen (mm) = {pen.max()*1e3:.3f}")
    assert F > 0.0, f"forward returned non-positive force: {F}"
    assert math.isfinite(F)
    _log("✓ Phase 1 passed (forward returns finite positive force)")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2 — gradient check (tape vs central FD)
# ═══════════════════════════════════════════════════════════════════════════


def _tape_grad(scene: DiffScene) -> float:
    """Tape ∂total_F/∂pad_x.  Caller must scene.set_pad_x() first."""
    scene.pad_x.grad.zero_()
    tape = wp.Tape()
    with tape:
        scene.forward()
    tape.backward(loss=scene.total_F)
    return float(scene.pad_x.grad.numpy()[0])


def _fd_grad(scene: DiffScene, h: float) -> tuple[float, float, float]:
    """Central FD ∂total_F/∂pad_x at the current pad_x."""
    x0 = float(scene.pad_x.numpy()[0])
    scene.set_pad_x(x0 + h)
    scene.forward()
    Fp = scene.total_force()
    scene.set_pad_x(x0 - h)
    scene.forward()
    Fm = scene.total_force()
    scene.set_pad_x(x0)
    return Fp, Fm, (Fp - Fm) / (2.0 * h)


def phase_2_gradient_check(scene: DiffScene) -> None:
    _section("PHASE 2 — Gradient check (tape vs FD)")

    nominal_pad_x = -0.002 + PAD_HX - scene.sphere_x + SPHERE_RADIUS
    scene.set_pad_x(nominal_pad_x)

    g_tape = _tape_grad(scene)
    _log(f"tape gradient ∂F/∂pad_x = {g_tape:+.4f} N/m")

    _sub("Central FD sweep")
    _log(f"{'step [m]':>12s}  {'F+':>10s}  {'F-':>10s}  {'g_FD':>12s}  {'rel err':>9s}")
    best_err = float('inf')
    g_fd_best = 0.0
    for h in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        scene.set_pad_x(nominal_pad_x)
        Fp, Fm, g_fd = _fd_grad(scene, h)
        rel_err = abs(g_tape - g_fd) / max(abs(g_tape), 1e-12)
        if rel_err < best_err:
            best_err = rel_err
            g_fd_best = g_fd
        print(f"  │  {h:12.1e}  {Fp:10.4f}  {Fm:10.4f}  {g_fd:+12.4f}  {rel_err:9.4%}")

    _log(f"Best vs FD: tape={g_tape:+.2f}  FD={g_fd_best:+.2f}  rel err={best_err:.3%}")
    if best_err > 0.01:
        _log("⚠ Phase 2 WARNING: tape and FD disagree by >1%")
    else:
        _log("✓ Phase 2 passed (tape and FD agree within 1%)")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 3 — inverse design via gradient descent
# ═══════════════════════════════════════════════════════════════════════════


def phase_3_inverse_design(scene: DiffScene, F_target: float = 8.0,
                            n_steps: int = 30, relax: float = 0.5,
                            verbose: bool = True) -> list[tuple]:
    """Find pad_x such that total_F ≈ F_target via gradient-based Newton step.

    For linear-ish F(pad_x), one Newton iteration suffices:
        Δx = -(F - F_target) / (∂F/∂pad_x)
    where ∂F/∂pad_x is computed via wp.Tape backward through the closed-form
    differentiable forward.  Under-relaxation factor `relax` damps the step
    so the squared-loss isn't overshot near saturation (smooth_relu corner).

    Sign convention (verified in Phase 2): pad_x is the magnitude of pad
    retraction (pad world position = -pad_x).  Increasing pad_x → pad
    further from sphere → less penetration → less force.  So ∂F/∂pad_x < 0,
    and to INCREASE F we DECREASE pad_x.

    Returns trajectory: list of (step, pad_x, F, loss) tuples.
    """
    _section(f"PHASE 3 — Inverse design (F_target = {F_target:.1f} N)")

    # Start RETRACTED so initial F < F_target.  Adding to nominal_pad_x
    # moves pad further from sphere (since pad world x = -pad_x).
    nominal_pad_x = -0.002 + PAD_HX - scene.sphere_x + SPHERE_RADIUS
    pad_x_init = nominal_pad_x + 0.003   # 3 mm too retracted → F ≈ 0
    scene.set_pad_x(pad_x_init)

    if verbose:
        _log(f"start:  pad_x = {pad_x_init*1e3:+.3f} mm  (retracted)")
        _log(f"target: F = {F_target:.2f} N    |    relax = {relax}")
        print(f"  │  {'step':>4s}  {'pad_x[mm]':>10s}  {'F[N]':>9s}  "
              f"{'err':>9s}  {'dF/dx':>12s}  {'Δx[mm]':>10s}")

    trajectory = []
    for step in range(n_steps):
        # Tape gradient of F w.r.t. pad_x (Phase-2-verified).
        scene.zero_all_grads()
        tape = wp.Tape()
        with tape:
            scene.forward()
        tape.backward(loss=scene.total_F)
        g_F = float(scene.pad_x.grad.numpy()[0])

        F = scene.total_force()
        err = F - F_target

        # Newton step:  pad_x ← pad_x − relax · err / (∂F/∂pad_x).
        # Guard against tiny gradient (no contact yet → no info).
        if abs(g_F) < 1.0:
            # No contact, no gradient signal — take a fixed nudge inward.
            dx = -1e-3   # 1 mm closer per step
        else:
            dx = -relax * err / g_F
            # Clip step to ≤2 mm for stability when F is very nonlinear.
            dx = max(min(dx, 2e-3), -2e-3)

        x = float(scene.pad_x.numpy()[0])
        loss = 0.5 * err * err
        trajectory.append((step, x, F, loss))

        if verbose and (step < 5 or step % 3 == 0 or step == n_steps - 1):
            print(f"  │  {step:4d}  {x*1e3:+10.4f}  {F:9.4f}  {err:+9.4f}  "
                  f"{g_F:+12.2e}  {dx*1e3:+10.4f}")

        scene.set_pad_x(x + dx)
        # Early stop if converged.
        if abs(err) < 0.05 and step > 2:
            _log(f"converged at step {step}")
            break

    final_F = trajectory[-1][2]
    final_x = trajectory[-1][1]
    _log(f"final:  pad_x = {final_x*1e3:+.3f} mm  F = {final_F:.3f} N  "
         f"err = {final_F - F_target:+.4f} N")
    if abs(final_F - F_target) < 0.5:
        _log("✓ Phase 3 passed (gradient-based Newton converged to target force)")
    else:
        _log("⚠ Phase 3 NOT converged — increase n_steps or reduce relax")
    return trajectory


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 4 — closed-loop force tracking with a time-varying target
# ═══════════════════════════════════════════════════════════════════════════


def phase_4_closed_loop(scene: DiffScene, n_steps: int = 80,
                         relax: float = 0.4, verbose: bool = True
                         ) -> list[tuple]:
    """Track a step-up F_target trajectory: 5 N for n/2 steps, then 15 N.

    Mimics a real-world grip task: gripper holds light, then needs to grip
    harder (e.g. tilt detected).  At each control step we use the same
    Newton update from Phase 3, but the setpoint is time-varying.

    Returns trajectory: list of (step, pad_x, F, F_target).
    """
    _section("PHASE 4 — Closed-loop tracking (5 N → 15 N step at t=N/2)")

    nominal_pad_x = -0.002 + PAD_HX - scene.sphere_x + SPHERE_RADIUS
    scene.set_pad_x(nominal_pad_x + 0.003)   # start retracted

    if verbose:
        _log(f"{'step':>5s}  {'pad_x[mm]':>10s}  {'F[N]':>9s}  {'F_target':>9s}  "
             f"{'err':>9s}  {'Δx[mm]':>10s}")

    trajectory = []
    for step in range(n_steps):
        F_target = 5.0 if step < n_steps // 2 else 15.0

        scene.zero_all_grads()
        tape = wp.Tape()
        with tape:
            scene.forward()
        tape.backward(loss=scene.total_F)
        g_F = float(scene.pad_x.grad.numpy()[0])

        F = scene.total_force()
        err = F - F_target
        if abs(g_F) < 1.0:
            dx = -1e-3
        else:
            dx = -relax * err / g_F
            dx = max(min(dx, 2e-3), -2e-3)

        x = float(scene.pad_x.numpy()[0])
        trajectory.append((step, x, F, F_target))
        if verbose and (step < 5 or step % 10 == 0
                        or step == n_steps // 2
                        or step == n_steps // 2 - 1
                        or step == n_steps - 1):
            print(f"  │  {step:5d}  {x*1e3:+10.4f}  {F:9.4f}  {F_target:9.4f}  "
                  f"{err:+9.4f}  {dx*1e3:+10.4f}")

        scene.set_pad_x(x + dx)

    F_first = trajectory[n_steps // 2 - 1][2]
    F_last  = trajectory[-1][2]
    _log(f"end of phase A (F_target=5):  F = {F_first:.3f} N "
         f"(err {F_first - 5.0:+.3f})")
    _log(f"end of phase B (F_target=15): F = {F_last:.3f} N "
         f"(err {F_last - 15.0:+.3f})")
    if abs(F_first - 5.0) < 0.5 and abs(F_last - 15.0) < 1.5:
        _log("✓ Phase 4 passed (controller tracked both setpoints)")
    else:
        _log("⚠ Phase 4 incomplete — controller did not reach setpoints")
    return trajectory


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 5 — Newton viewer (interactive playback of the optimization)
# ═══════════════════════════════════════════════════════════════════════════


def _quat_rotate(q, v):
    xyz = np.array([q[0], q[1], q[2]])
    t = 2.0 * np.cross(xyz, v)
    return v + q[3] * t + np.cross(xyz, t)


class Example:
    """Newton viewer for the diff_test.

    Plays back the closed-loop force-tracking session live, showing:
      - the pad moving inward/outward as gradient descent adjusts pad_x
      - the grasped sphere
      - per-sphere CSLC contact spheres color-coded by force magnitude

    Use ↑/↓ arrows or rerun with different `--target-N1`/`--target-N2`
    targets to change the force step.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.test_mode = getattr(args, "test", False)
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_step = 0
        self.sim_time = 0.0

        self.target_a = float(getattr(args, "target_n1", 5.0))
        self.target_b = float(getattr(args, "target_n2", 15.0))
        # Newton-step under-relaxation (Phase 3/4 uses 0.4–0.5).
        self.relax = float(getattr(args, "relax", 0.4))
        self.switch_step = 60          # step where target changes A → B
        self.total_steps = 120

        self.scene = DiffScene(sphere_x=0.025, device="cuda:0")
        # Start retracted so the viewer shows convergence dynamics.
        # pad_x = magnitude of retraction; pad world x = -pad_x.
        # Larger pad_x = more retracted = less force.
        nominal_pad_x = (-0.002 + PAD_HX - self.scene.sphere_x + SPHERE_RADIUS)
        self.scene.set_pad_x(nominal_pad_x + 0.003)
        # Run forward once so total_F has a sensible value at frame 0.
        self.scene.forward()

        # Build a tiny Newton model for the viewer to render: the pad as a
        # box body and the sphere as a sphere body.  The viewer only renders
        # state — it doesn't run any solver — so we update body_q manually
        # each frame from the differentiable scene.
        b = newton.ModelBuilder()
        pad_body = b.add_body(xform=wp.transform_identity(), label="pad")
        b.add_shape_box(pad_body, hx=PAD_HX, hy=PAD_HY, hz=PAD_HZ,
                        cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        sphere_body = b.add_body(xform=wp.transform_identity(), label="sphere")
        b.add_shape_sphere(sphere_body, radius=SPHERE_RADIUS,
                           cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        self.model = b.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.15, -0.18, 0.10),
            pitch=-15.0, yaw=125.0,
        )

        # Lattice-sphere visualisation buffers.
        n_surf = int(self.scene.data.is_surface.numpy().sum())
        self._lat_n = n_surf
        self._lat_xforms = np.zeros((n_surf, 7), np.float32)
        self._lat_xforms[:, 6] = 1.0
        self._lat_colors = np.zeros((n_surf, 3), np.float32)
        self._lat_mats = np.tile([0.5, 0.3, 0.0, 0.0],
                                  (n_surf, 1)).astype(np.float32)
        self._lat_radius = float(PAD_SPACING * 0.2)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

        _log(f"Viewer ready: {n_surf} surface CSLC spheres on the pad")
        _log(f"Schedule: {self.switch_step} steps @ {self.target_a:.1f} N, "
             f"then {self.total_steps - self.switch_step} steps @ {self.target_b:.1f} N")

    def _control_step(self):
        """One closed-loop control step: tape backward → Newton update.

        Same update rule as phase_3_inverse_design and phase_4_closed_loop:
        compute ∂F/∂pad_x via tape, then take a damped Newton step
            Δx = −relax · (F − F_target) / (∂F/∂pad_x)
        clipped to ±2 mm/step for stability near the smoothing corner.
        """
        F_target = self.target_a if self.sim_step < self.switch_step else self.target_b

        self.scene.zero_all_grads()
        tape = wp.Tape()
        with tape:
            self.scene.forward()
        tape.backward(loss=self.scene.total_F)
        g_F = float(self.scene.pad_x.grad.numpy()[0])

        F = self.scene.total_force()
        err = F - F_target
        if abs(g_F) < 1.0:
            dx = -1e-3   # no-contact nudge inward
        else:
            dx = -self.relax * err / g_F
            dx = max(min(dx, 2e-3), -2e-3)

        x = float(self.scene.pad_x.numpy()[0])
        new_x = x + dx
        self.scene.set_pad_x(new_x)

        self.current_F = F
        self.current_F_target = F_target
        self.current_pad_x = new_x
        self.current_grad = g_F

    def step(self):
        if self.sim_step < self.total_steps:
            self._control_step()
        self.sim_step += 1
        self.sim_time += self.frame_dt

        # Push body_q into the renderer's state.
        x = float(self.scene.pad_x.numpy()[0])
        q = self.state.body_q.numpy()
        q[0] = np.array([-x, 0.0, SPHERE_Z, 0, 0, 0, 1], dtype=np.float32)
        q[1] = np.array([self.scene.sphere_x, 0.0, SPHERE_Z, 0, 0, 0, 1],
                         dtype=np.float32)
        self.state.body_q.assign(
            wp.array(q, dtype=wp.transform, device=self.state.body_q.device))

    def _update_lattice_viz(self):
        d = self.scene.data
        pos_local = d.positions.numpy()
        is_surf   = d.is_surface.numpy()
        normals   = d.outward_normals.numpy()
        F_per     = self.scene.per_sphere_force()
        x = float(self.scene.pad_x.numpy()[0])

        # Pad body world transform = (-x, 0, SPHERE_Z), identity rotation.
        pad_pos = np.array([-x, 0.0, SPHERE_Z])

        # Color scale: white at F=0, red at F=F_max.
        F_max = max(F_per.max(), 1e-3)
        idx = 0
        for i in range(len(is_surf)):
            if is_surf[i] != 1:
                continue
            world = pos_local[i] + pad_pos    # identity body rotation
            self._lat_xforms[idx, :3] = world
            t = min(F_per[i] / F_max, 1.0)
            self._lat_colors[idx] = [1.0, 1.0 - t, 1.0 - t]
            idx += 1

        if idx == 0:
            return
        self.viewer.log_shapes(
            "/cslc_lattice", newton.GeoType.SPHERE, self._lat_radius,
            wp.array(self._lat_xforms[:idx], dtype=wp.transform),
            wp.array(self._lat_colors[:idx], dtype=wp.vec3),
            wp.array(self._lat_mats[:idx], dtype=wp.vec4),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self._update_lattice_viz()
        self.viewer.end_frame()

    def _render_ui(self, imgui):
        F_target = getattr(self, "current_F_target", self.target_a)
        F = getattr(self, "current_F", 0.0)
        x = getattr(self, "current_pad_x", 0.0)
        g = getattr(self, "current_grad", 0.0)
        imgui.text(f"Step:   {self.sim_step}/{self.total_steps}")
        imgui.text(f"Target: {F_target:.2f} N")
        imgui.text(f"F:      {F:.3f} N")
        imgui.text(f"Err:    {F - F_target:+.3f} N")
        imgui.text(f"Pad x:  {x*1e3:+.3f} mm")
        imgui.text(f"∂L/∂x:  {g:+.2e}")

    def test_final(self):
        # Lenient pass — viewer is for inspection, not strict regression.
        assert self.current_F > 0.5, f"viewer ended with F={self.current_F}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--target-n1", type=float, default=5.0,
                             help="Target grip force (N) for the first half of the run.")
        parser.add_argument("--target-n2", type=float, default=15.0,
                             help="Target grip force (N) after the step change.")
        parser.add_argument("--relax", type=float, default=0.4,
                             help="Newton-step under-relaxation factor (0..1).")
        return parser


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def _make_arg_parser():
    parser = argparse.ArgumentParser(
        description="Differentiable CSLC proof of concept.")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run only one phase (1-4); 0 = all phases.")
    parser.add_argument("--mode", type=str, default="headless",
                        choices=["headless", "viewer"],
                        help="headless = run phases 1-4 + print; "
                             "viewer = phase 5 (interactive playback).")
    return parser


def main():
    args, _ = _make_arg_parser().parse_known_args()
    wp.init()

    if args.mode == "viewer" or "--viewer" in sys.argv:
        viewer_parser = Example.create_parser()
        viewer, viewer_args = newton.examples.init(viewer_parser)
        newton.examples.run(Example(viewer, viewer_args), viewer_args)
        return

    print(f"\n{_HSEP}\n  CSLC DIFFERENTIABLE TEST — phases {args.phase or 'all'}\n{_HSEP}")
    scene = DiffScene(sphere_x=0.025, device="cuda:0")
    if args.phase in (0, 1):
        phase_1_forward(scene)
    if args.phase in (0, 2):
        phase_2_gradient_check(scene)
    if args.phase in (0, 3):
        phase_3_inverse_design(scene)
    if args.phase in (0, 4):
        phase_4_closed_loop(scene)

    print(f"\n{_HSEP}\n  Done — for visual playback use:\n"
          f"    uv run cslc_v1/diff_test.py --mode viewer --viewer gl\n{_HSEP}")


if __name__ == "__main__":
    main()
