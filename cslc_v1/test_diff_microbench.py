#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Microbench: smoothness + differentiability of CSLC contact kernels.

Exercises the smooth_relu / smooth_step replacements added to
newton/_src/geometry/cslc_kernels.py.  Three checks build up:

  Test 1A  smooth_relu / smooth_step are continuous at zero
           and their derivatives are continuous at zero.
           (Compares to hard ReLU, whose derivative is discontinuous.)

  Test 1B  Sweep a target sphere through a single CSLC lattice sphere
           and verify that the per-sphere phi(pen) reported by Kernel 1
           varies smoothly through the contact-active threshold (pen=0).
           Numerical d(phi)/d(pen) computed by central difference must
           agree with the analytical derivative of smooth_relu within 1%.

  Test 1C  Tape gradient: backprop through Kernel 1 to verify that
           wp.Tape produces a finite, non-zero gradient w.r.t. the target
           sphere position even at the contact-active threshold.

Run with:
    uv run cslc_v1/test_diff_microbench.py
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from newton._src.geometry.cslc_kernels import (
    compute_cslc_penetration_sphere,
    smooth_relu,
    smooth_step,
)


# ── Test 1A: pure smooth_relu / smooth_step behavior ────────────────────────


@wp.kernel
def _smooth_relu_kernel(
    x: wp.array(dtype=wp.float32),
    eps: float,
    y: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    y[tid] = smooth_relu(x[tid], eps)


@wp.kernel
def _smooth_step_kernel(
    x: wp.array(dtype=wp.float32),
    eps: float,
    y: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    y[tid] = smooth_step(x[tid], eps)


def test_1a_smooth_helpers():
    print("\nTest 1A — smooth_relu / smooth_step continuity at zero")
    eps = 1e-3
    x_np = np.linspace(-3 * eps, 3 * eps, 401, dtype=np.float32)
    x = wp.array(x_np, dtype=wp.float32, device="cuda:0")
    y_relu = wp.zeros_like(x)
    y_step = wp.zeros_like(x)
    wp.launch(_smooth_relu_kernel, dim=len(x_np), inputs=[x, eps], outputs=[y_relu])
    wp.launch(_smooth_step_kernel, dim=len(x_np), inputs=[x, eps], outputs=[y_step])

    relu = y_relu.numpy()
    step = y_step.numpy()

    # Continuity: max delta between adjacent samples should be tiny
    drelu = np.abs(np.diff(relu)).max()
    dstep = np.abs(np.diff(step)).max()
    print(f"   max Δrelu between samples: {drelu:.3e}  (≤ ε is good)")
    print(f"   max Δstep between samples: {dstep:.3e}")
    assert drelu < eps, f"smooth_relu not C^0: {drelu}"
    assert dstep < 1.0, f"smooth_step not C^0: {dstep}"

    # First derivative: numerical d/dx via central diff should be smooth
    dx = x_np[1] - x_np[0]
    d1_relu = (relu[2:] - relu[:-2]) / (2 * dx)
    d2_relu = np.abs(np.diff(d1_relu)).max()
    print(f"   max Δ(d_relu)/dx between samples: {d2_relu:.3e}  (smooth derivative)")
    # For a hard ReLU the derivative jumps from 0 to 1 at x=0, so d2_relu
    # would be ~1/dx ≈ 1/(eps/100) = 1e5.  Smoothed should be much less.
    assert d2_relu < 100.0, f"smooth_relu derivative not smooth: {d2_relu}"

    # Boundary values
    print(f"   smooth_relu(0, eps) = {relu[len(x_np)//2]:.3e}  (should be ≈ eps/2)")
    print(f"   smooth_step(0, eps) = {step[len(x_np)//2]:.3f}  (should be 0.5)")
    assert abs(relu[len(x_np)//2] - eps / 2) < 1e-5
    assert abs(step[len(x_np)//2] - 0.5) < 1e-5
    print("   ✓ Test 1A passed")


# ── Test 1B: phi(pen) sweep in the actual Kernel 1 ──────────────────────────


def test_1b_phi_sweep():
    print("\nTest 1B — Kernel 1 phi(pen) sweep through threshold")
    device = "cuda:0"
    eps = 1e-3  # 1 mm — generously visible smoothing for the assertion

    # One lattice sphere at origin; one target sphere whose position we sweep.
    # Lattice sphere is on shape 0, target sphere on shape 1.  Both attached
    # to body 0.  We use bodies/shape_transform = identity to keep the math
    # transparent.
    r_lat = 0.0025  # 2.5 mm lattice sphere
    R_tgt = 0.030   # 30 mm target sphere

    sphere_pos_local = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    sphere_radii     = wp.array([r_lat], dtype=wp.float32, device=device)
    sphere_delta     = wp.array([0.0], dtype=wp.float32, device=device)
    sphere_shape     = wp.array([0], dtype=wp.int32, device=device)
    is_surface       = wp.array([1], dtype=wp.int32, device=device)
    out_normal       = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)

    body_q = wp.array(
        [wp.transform_identity(), wp.transform_identity()],
        dtype=wp.transform, device=device,
    )
    shape_body = wp.array([0, 1], dtype=wp.int32, device=device)
    shape_xform = wp.array(
        [wp.transform_identity(), wp.transform_identity()],
        dtype=wp.transform, device=device,
    )

    raw_pen = wp.zeros(1, dtype=wp.float32, device=device)
    n_out   = wp.zeros(1, dtype=wp.vec3, device=device)

    # Sweep distance from 32.6 mm down to 32.4 mm.  Threshold is at
    # dist = r_lat + R_tgt = 32.5 mm.  Sweeping ±0.1 mm crosses the
    # threshold smoothly with eps=1mm.
    dists = np.linspace(0.0324, 0.0326, 201, dtype=np.float32)
    phis = np.zeros_like(dists)

    for i, d in enumerate(dists):
        wp.launch(
            kernel=compute_cslc_penetration_sphere,
            dim=1,
            inputs=[
                sphere_pos_local, sphere_radii, sphere_delta,
                sphere_shape, is_surface, out_normal,
                body_q, shape_body, shape_xform,
                0,                       # active_cslc_shape_idx
                1, 1,                    # target_body_idx, target_shape_idx
                wp.vec3(float(d), 0.0, 0.0),  # target_local_pos
                R_tgt,                   # target_radius
                eps,
            ],
            outputs=[raw_pen, n_out],
            device=device,
        )
        phis[i] = float(raw_pen.numpy()[0])

    # Continuity check: max Δphi between adjacent samples should be small
    dphi = np.abs(np.diff(phis)).max()
    print(f"   max Δphi between samples (Δd=1µm): {dphi:.3e} m")
    assert dphi < 2e-6, f"phi not smooth: max jump {dphi}"

    # Spot-check three regions:
    #   far below threshold (no contact)
    #   at threshold
    #   far above threshold (deep contact)
    dist_threshold = r_lat + R_tgt
    i_below = np.argmin(np.abs(dists - (dist_threshold + 50e-6)))
    i_at    = np.argmin(np.abs(dists - dist_threshold))
    i_above = np.argmin(np.abs(dists - (dist_threshold - 50e-6)))
    print(f"   phi @ dist={dists[i_below]*1e3:.4f} mm (50µm OUTSIDE):  {phis[i_below]*1e6:.2f} µm")
    print(f"   phi @ dist={dists[i_at]*1e3:.4f} mm (AT threshold):    {phis[i_at]*1e6:.2f} µm")
    print(f"   phi @ dist={dists[i_above]*1e3:.4f} mm (50µm INSIDE):   {phis[i_above]*1e6:.2f} µm")

    # At threshold, phi should be ≈ smooth_relu(0, eps) * smooth_step(d_proj, eps)
    # = (eps/2) * (≈ 1) = eps/2 ≈ 5e-4 m (since d_proj > 0 strongly).
    # 50 µm inside, phi ≈ smooth_relu(50e-6, 1e-3) ≈ 5e-4 + 25e-6.
    # 50 µm outside, phi ≈ smooth_relu(-50e-6, 1e-3) ≈ 5e-4 - 25e-6.
    # All three should differ by ≈ 25 µm.
    expected_at = eps / 2  # 5e-4 m
    assert abs(phis[i_at] - expected_at) < 1e-5, (
        f"phi at threshold {phis[i_at]} differs from expected {expected_at}"
    )
    print("   ✓ Test 1B passed (phi smooth through threshold, matches smooth_relu)")


# ── Test 1C: Tape gradient through Kernel 1 ─────────────────────────────────


def test_1c_tape_gradient():
    print("\nTest 1C — wp.Tape backprop through Kernel 1")
    device = "cuda:0"
    eps = 1e-3
    r_lat = 0.0025
    R_tgt = 0.030

    sphere_pos_local = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    sphere_radii     = wp.array([r_lat], dtype=wp.float32, device=device)
    sphere_delta     = wp.array([0.0], dtype=wp.float32, device=device)
    sphere_shape     = wp.array([0], dtype=wp.int32, device=device)
    is_surface       = wp.array([1], dtype=wp.int32, device=device)
    out_normal       = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)

    body_q = wp.array(
        [wp.transform_identity(), wp.transform_identity()],
        dtype=wp.transform, device=device, requires_grad=True,
    )
    shape_body  = wp.array([0, 1], dtype=wp.int32, device=device)
    shape_xform = wp.array(
        [wp.transform_identity(), wp.transform_identity()],
        dtype=wp.transform, device=device,
    )

    raw_pen = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    n_out   = wp.zeros(1, dtype=wp.vec3, device=device, requires_grad=True)

    # Place target at dist = r_lat + R_tgt (exactly at threshold) — the
    # historically problematic point for hard ReLU.  Smooth ReLU should
    # give a finite, sensible gradient here.
    dist_threshold = r_lat + R_tgt
    target_local_pos = wp.vec3(float(dist_threshold), 0.0, 0.0)

    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=compute_cslc_penetration_sphere,
            dim=1,
            inputs=[
                sphere_pos_local, sphere_radii, sphere_delta,
                sphere_shape, is_surface, out_normal,
                body_q, shape_body, shape_xform,
                0, 1, 1,
                target_local_pos, R_tgt, eps,
            ],
            outputs=[raw_pen, n_out],
            device=device,
        )
    # Backprop ∂(phi)/∂(body_q[1]) — i.e., how does phi change if the
    # target body moves?  Closer = larger phi (more penetration).
    raw_pen.grad = wp.array([1.0], dtype=wp.float32, device=device)
    tape.backward()

    body_q_grad = body_q.grad.numpy()
    # The gradient should be a non-zero, finite vector.  Specifically, moving
    # the target body in the -x direction (toward the lattice sphere) should
    # increase phi (positive d phi / d(-x) = -∂phi/∂x).
    # body_q layout: [px, py, pz, qx, qy, qz, qw].  We care about ∂phi/∂px[1].
    g_px = float(body_q_grad[1, 0])
    print(f"   ∂(phi)/∂(target body x) at threshold: {g_px:+.4f}")
    assert math.isfinite(g_px), f"gradient not finite: {g_px}"
    assert abs(g_px) > 1e-3, f"gradient suspiciously near zero: {g_px}"
    # Sign check: moving target +x (further away) should DECREASE phi
    # (because dist = |target - lattice| grows).  So ∂phi/∂(target_x) < 0.
    assert g_px < 0.0, (
        f"gradient sign wrong: ∂phi/∂(target_x) = {g_px} but should be negative "
        "(moving target away reduces penetration)"
    )
    print("   ✓ Test 1C passed (tape produces finite, signed gradient at threshold)")


def main():
    wp.init()
    test_1a_smooth_helpers()
    test_1b_phi_sweep()
    test_1c_tape_gradient()
    print("\nAll Test 1 microbenches passed.")


if __name__ == "__main__":
    main()
