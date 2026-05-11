#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC Warp GPU kernels for contact generation.


File location: newton/_src/geometry/cslc_kernels.py
"""

import warp as wp


# ═══════════════════════════════════════════════════════════════════════════
#  Smooth differentiable surrogates for ReLU and step
#
#  Replace hard `if x > 0` ops with smooth analogues so wp.Tape can backprop
#  through CSLC contact dynamics for MPC and RL workflows.  All four kernel
#  branches that gate on continuous physical quantities (pen_3d, d_proj,
#  effective_pen, delta clamps) are smoothed with eps as the transition
#  width [m].  eps → 0 recovers the original non-smooth behavior; default
#  eps = 1e-5 m gives essentially-binary forces above 0.1 mm and a smooth
#  C^1 transition through the threshold.
#
#  smooth_relu(x, eps) = 0.5 * (x + sqrt(x² + eps²))
#      → max(x, 0) as eps → 0
#      derivative is smooth_step(x, eps), well-defined at x=0
#
#  smooth_step(x, eps) = 0.5 * (1 + x / sqrt(x² + eps²))
#      → 1 if x >> eps, → 0 if x << -eps, smooth sigmoid-like through 0
#
#  Both are C^∞ for eps > 0 and have bounded gradients.
# ═══════════════════════════════════════════════════════════════════════════


@wp.func
def smooth_relu(x: float, eps: float) -> float:
    return 0.5 * (x + wp.sqrt(x * x + eps * eps))


@wp.func
def smooth_step(x: float, eps: float) -> float:
    return 0.5 * (1.0 + x / wp.sqrt(x * x + eps * eps))


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 1: Penetration (lattice sphere vs target sphere)
# ═══════════════════════════════════════════════════════════════════════════




@wp.kernel
def compute_cslc_penetration_sphere(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    target_shape_idx: int,
    target_local_pos: wp.vec3,
    target_radius: float,
    eps: float,
    raw_penetration: wp.array(dtype=wp.float32),
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """raw 3-D sphere-sphere overlap per lattice sphere (differentiable).

    Smoothed:
        phi = smooth_relu(pen_3d, eps) * smooth_step(d_proj, eps)
    so that wp.Tape can backprop through the contact-active gate.
    Recovers the hard `if pen_3d > 0 and d_proj > 0: phi = pen_3d` rule as
    eps → 0.
    """
    tid = wp.tid()
    phi     = 0.0
    n_world = wp.vec3(0.0, 0.0, 0.0)

    # Active-pad filter.  Discrete index branch — not differentiable, but
    # the parameter being indexed (sphere_shape) is integer-valued and
    # never a learning target.
    if sphere_shape[tid] != active_cslc_shape_idx:
        raw_penetration[tid] = 0.0
        return

    if is_surface[tid] == 1:
        s_idx = sphere_shape[tid]
        b_idx = shape_body[s_idx]
        X_ws  = shape_transform[s_idx]
        X_wb  = body_q[b_idx]

        p_local = sphere_pos_local[tid]
        r_lat   = sphere_radii[tid]
        out_n   = sphere_outward_normal[tid]

        q_body  = wp.transform_point(X_ws, p_local)
        q_world = wp.transform_point(X_wb, q_body)

        X_tb    = body_q[target_body_idx]
        t_world = wp.transform_point(X_tb, target_local_pos)

        diff = t_world - q_world
        dist = wp.length(diff)

        n_body  = wp.transform_vector(X_ws, out_n)
        n_world = wp.transform_vector(X_wb, n_body)
        d_proj  = wp.dot(diff, n_world)

        pen_3d = (r_lat + target_radius) - dist

        # Smooth contact-active gate:
        #   phi ≈ pen_3d when pen_3d > 0 AND d_proj > 0
        #   phi ≈ 0 otherwise
        # Continuous and C^∞ for eps > 0.
        phi = smooth_relu(pen_3d, eps) * smooth_step(d_proj, eps)

    raw_penetration[tid] = phi
    contact_normal_out[tid] = n_world



# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2a: Tape-compatible one-shot lattice solve
#
#  Closed-form equivalent of the iterative damped Jacobi: solves
#      (K + kc·I) δ = kc · φ
#  where K is the CSLC Laplacian (anchor ka + lateral kl).  Uses a
#  precomputed dense A_inv = (K + kc·I)^-1 (built once in CSLCData.from_pads
#  when build_A_inv=True) to compute
#      δ_i = kc · Σ_j  A_inv[i, j] · φ[j]
#  as a pure matvec.  This is a tape-differentiable drop-in replacement for
#  jacobi_step + src/dst swap — the Python-side ping-pong buffer alias
#  breaks wp.Tape backward (see cslc_v1/diff_test.py Phase-2 diagnostic),
#  but a single matvec is linear and backprops correctly.
#
#  Differences vs the iterative Jacobi:
#    – Ungated: treats all surface spheres as contributing.  For inactive
#      ones, φ ≈ 0 (smooth_relu(negative, eps) ≈ 0 already applied in
#      Kernel 1), so their δ stays near zero.  Small residual error from
#      the smooth-relu's eps/2 floor; decays as eps → 0.
#    – Lateral kl coupling is preserved (baked into K and therefore A_inv).
#    – O(n²) time per solve, O(n²) memory for A_inv; for n ≳ 1000 prefer
#      a sparse Cholesky factorisation plus two triangular solve kernels.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def lattice_solve_equilibrium(
    A_inv: wp.array2d(dtype=wp.float32),        # (n_spheres, n_spheres)
    phi: wp.array(dtype=wp.float32),            # (n_spheres,)
    kc: float,
    delta_out: wp.array(dtype=wp.float32),      # (n_spheres,)
):
    """δ_i = kc · Σ_j  A_inv[i,j] · φ[j]  — tape-compatible CSLC equilibrium."""
    i = wp.tid()
    n = A_inv.shape[1]
    acc = float(0.0)
    for j in range(n):
        acc = acc + A_inv[i, j] * phi[j]
    delta_out[i] = kc * acc


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2b: Active-pad-selective copy
#
#  When `lattice_solve_equilibrium` runs for one pair (active pad P), it
#  writes a per-sphere δ for *every* sphere — including spheres on other
#  pads, where φ was zeroed in Kernel 1 and so δ ends up at zero.
#  Unconditionally copying that full δ buffer back into
#  `CSLCData.sphere_delta` would wipe the other pad's warm-start every
#  step, leaving only the *last* pair's pad with non-zero compression in
#  `sphere_delta`.  That breaks visualisation (lattice viz reads
#  `sphere_delta` and would render only one pad as compressed) and warm
#  starts (next step's φ for the wiped pad starts from δ=0 again).
#
#  The iterative jacobi path doesn't have this problem because
#  `jacobi_step` has an active-pad branch that pass-throughs δ for
#  non-active pads.  This selective-copy kernel adds the same guard to
#  the dense-solve path's writeback: only update sphere_delta[i] when
#  sphere i belongs to the active pad.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def cslc_copy_active(
    src: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    active_cslc_shape_idx: int,
    dst: wp.array(dtype=wp.float32),
):
    """dst[i] ← src[i] only for spheres on the active CSLC pad.

    Used by `cslc_handler._launch_vs_sphere` to merge the dense-solve
    output back into `CSLCData.sphere_delta` without clobbering other
    pads' warm-starts.
    """
    i = wp.tid()
    if sphere_shape[i] == active_cslc_shape_idx:
        dst[i] = src[i]


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2: Damped Jacobi iteration
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def jacobi_step(
    delta_src: wp.array(dtype=wp.float32),
    delta_dst: wp.array(dtype=wp.float32),
    raw_penetration: wp.array(dtype=wp.float32),
    is_surface: wp.array(dtype=wp.int32),
    neighbor_start: wp.array(dtype=wp.int32),
    neighbor_count: wp.array(dtype=wp.int32),
    neighbor_list: wp.array(dtype=wp.int32),
    ka: float,
    kl: float,
    kc: float,
    alpha: float,
    sphere_shape: wp.array(dtype=wp.int32),
    active_cslc_shape_idx: int,
    eps: float,
):

    """one damped Jacobi sweep for the ACTIVE pad only (differentiable).

    Non-active-pad spheres copy delta through unchanged, preserving the
    warm-start from the previous step.  Without this, each pair launch
    used to drive the OTHER pad's deltas toward zero through the Laplacian
    coupling, destroying the warm-start for the next pair.

    Smoothed differences vs. the original:
      `if effective_pen > 0` is now smoothed via smooth_step(effective_pen,
        eps).  At equilibrium with effective_pen >> eps, behaviour matches
        the hard-gated version.  Within ±eps of zero, contact contribution
        ramps smoothly between 0 and full.
      `if delta_new < 0: delta_new = 0` is now smooth_relu(delta_new, eps).
        The δ ≥ 0 invariant is preserved up to O(eps).

    Formulation (smoothed):
      gate(eff_pen) = smooth_step(eff_pen, eps)
      (ka + kl|N(i)| + kc·gate) · δ_i = kc · gate · φ_i + kl · Σ_j δ_j
    """
    tid = wp.tid()

    # Pad filter: non-active pads preserve their delta.  Discrete index
    # branch — does not flow gradient (sphere_shape is integer-valued).
    if sphere_shape[tid] != active_cslc_shape_idx:
        delta_dst[tid] = delta_src[tid]
        return

    delta_old   = delta_src[tid]
    n_neighbors = neighbor_count[tid]

    neighbor_sum = float(0.0)
    start = neighbor_start[tid]
    for n in range(n_neighbors):
        j = neighbor_list[start + n]
        neighbor_sum = neighbor_sum + delta_src[j]

    f_contact = float(0.0)
    k_diag    = ka + kl * float(n_neighbors)

    if is_surface[tid] == 1:
        phi = raw_penetration[tid]
        effective_pen = phi - delta_old
        # Smooth contact-active gate (replaces `if effective_pen > 0:`).
        gate = smooth_step(effective_pen, eps)
        f_contact = kc * phi * gate
        k_diag    = k_diag + kc * gate

    delta_jacobi = (f_contact + kl * neighbor_sum) / k_diag
    delta_new    = (1.0 - alpha) * delta_old + alpha * delta_jacobi
    # Smooth lower clamp δ ≥ 0 (replaces `if delta_new < 0: delta_new = 0`).
    delta_new    = smooth_relu(delta_new, eps)
    delta_dst[tid] = delta_new


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 3: Write contacts to Newton's Contacts buffer (v3)
#
#  v3: No point0 shift. point0 = CSLC sphere center directly.
#      Solver computes pen = (effective_r + target_radius) - d_proj
#      where d_proj = dot(diff, outward_normal). For inner-face contacts
#      on a flat pad, d_proj is UNIFORM across the patch, matching the
#      "uniform flat contact" assumption in the kc calibration.
# ═══════════════════════════════════════════════════════════════════════════

@wp.kernel
def write_cslc_contacts(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    target_shape_idx: int,
    target_local_pos: wp.vec3,
    target_radius: float,
    contact_offset: int,
    surface_slot_map: wp.array(dtype=wp.int32),
    raw_penetration: wp.array(dtype=wp.float32),
    out_shape0: wp.array(dtype=wp.int32),
    out_shape1: wp.array(dtype=wp.int32),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_margin0: wp.array(dtype=wp.float32),
    out_margin1: wp.array(dtype=wp.float32),
    out_tids: wp.array(dtype=wp.int32),
    # shape_material_mu: no longer read inside this kernel (2026-04-19).
    # Previously used as out_friction = mu (the pad shape's friction coefficient),
    # which caused a double-count: the MuJoCo conversion kernel multiplies
    # rigid_contact_friction onto the geom pair base friction (already = mu),
    # giving effective_mu = mu * mu = mu^2 instead of mu.
    # The fix writes out_friction = 1.0 (no scale), so geom friction is used as-is.
    # This parameter is kept in the signature to avoid breaking the handler call;
    # remove it from both here and cslc_handler.py in a future cleanup.
    shape_material_mu: wp.array(dtype=wp.float32),
    cslc_kc: float,
    target_ke: float,
    cslc_dc: float,
    sim_dt: float,
    eps: float,
    out_stiffness: wp.array(dtype=wp.float32),
    out_damping: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
    debug_reason: wp.array(dtype=wp.int32),
    # ── Diagnostic outputs (physics-neutral; read back by the handler) ──
    # Indexed by (diag_offset + slot) where diag_offset =
    # pair_idx * n_surface_contacts.  This lays each pair's diagnostics
    # out in its own block (same layout as the contacts buffer), so
    # pair_1's launch can't overwrite pair_0's diagnostic writes.
    diag_offset: int,
    dbg_pen_scale: wp.array(dtype=wp.float32),
    dbg_solver_pen: wp.array(dtype=wp.float32),
    dbg_effective_r: wp.array(dtype=wp.float32),
    dbg_d_proj: wp.array(dtype=wp.float32),
    dbg_radial: wp.array(dtype=wp.float32),
):
    """Stage 1: write one rigid contact per lattice sphere in real 3-D overlap.

    Convention (unchanged from v3):
      point0  = lattice sphere centre (body frame, no shift)
      point1  = target  sphere centre (body frame)
      normal  = face outward normal (world)
      margin0 = effective_r = r_lat − δ
      margin1 = target_radius
      offset0 =  effective_r   · normal  (body frame)
      offset1 = −target_radius · normal  (body frame)

    The rigid-body solver computes
        solver_pen = margin0 + margin1 − dot(point1_w − point0_w, normal)
                   = (effective_r + R) − d_proj
    which OVER-estimates the true 3-D overlap whenever the target is
    off-axis (d_proj < dist).  We fix the force magnitude by scaling the
    per-contact stiffness:
        kc_eff = kc · pen_3d / solver_pen       ⇒  F = kc · pen_3d
    Friction stays in the face's tangent plane (paper's flat-patch story).

    debug_reason codes:
        0 = wrote successfully
        1 = culled on pen_3d <= 0  (no real 3-D overlap)
        3 = culled on d_proj <= 0  (target is behind the face)
        4 = wrong pad for this pair
    """
    tid = wp.tid()

    slot = surface_slot_map[tid]
    if slot < 0:
        return
    buf_idx = contact_offset + slot
    # dslot: where THIS pair's diagnostic for THIS slot lives in the
    # handler's per-pair-per-slot diagnostic arrays.  Mirrors the layout
    # of the contacts buffer: pair_idx * n_surface_contacts + slot.
    dslot = diag_offset + slot

    # Pair filter.
    if sphere_shape[tid] != active_cslc_shape_idx:
        out_shape0[buf_idx] = -1
        debug_reason[slot]  = 4
        # Sentinel: negative pen_scale signals "no contact this slot".
        dbg_pen_scale[dslot]   = -1.0
        dbg_solver_pen[dslot]  = 0.0
        dbg_effective_r[dslot] = 0.0
        dbg_d_proj[dslot]      = 0.0
        dbg_radial[dslot]      = 0.0
        return

    # Shape A: lattice sphere.
    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]

    p_local  = sphere_pos_local[tid]
    out_n    = sphere_outward_normal[tid]
    delta_val = sphere_delta[tid]
    r_lat    = sphere_radii[tid]

    q_body  = wp.transform_point(X_ws, p_local)
    q_world = wp.transform_point(X_wb, q_body)

    # Smooth lower clamp on effective radius (replaces `if effective_r < 0`).
    # Recovers max(r_lat - δ, 0) as eps → 0 with a finite derivative at 0.
    effective_r = smooth_relu(r_lat - delta_val, eps)

    # Shape B: target sphere centre in world.
    X_tb    = body_q[target_body_idx]
    t_world = wp.transform_point(X_tb, target_local_pos)

    diff = t_world - q_world
    dist = wp.length(diff)

    # Face normal in world.
    n_body    = wp.transform_vector(X_ws, out_n)
    normal_ab = wp.transform_vector(X_wb, n_body)
    d_proj    = wp.dot(diff, normal_ab)

    # True 3-D overlap and solver-side projected penetration.  Computed
    # regardless of sign — the smooth contact-active gate below drives the
    # output stiffness to ~0 outside the contact region, replacing the hard
    # `if d_proj <= 0: return` and `if pen_3d <= 0: return` culls.  This
    # makes the kernel-to-solver interface C^∞ in the target pose, so
    # wp.Tape backward can flow through contact-onset transitions.
    pen_3d     = (effective_r + target_radius) - dist
    solver_pen = (effective_r + target_radius) - d_proj
    # Sign-preserving smooth reciprocal for pen_3d / solver_pen.  The
    # naive `pen_3d / sqrt(solver_pen² + δ²)` divides by |solver_pen|,
    # which flips the sign of pen_scale when both pen_3d and solver_pen
    # cross zero together (e.g. as the target passes through x = r_lat +
    # r_target on the face normal).  Replacing with
    # `pen_scale = pen_3d · solver_pen / (solver_pen² + δ²)`
    # is C^∞, preserves sign in the physically-correct same-sign regime
    # (pen_3d and solver_pen agree along the normal axis), and recovers
    # `pen_3d / solver_pen` for |solver_pen| ≫ δ.
    #
    # δ = eps sets the half-width of the resulting "notch" at
    # solver_pen = 0.  With eps = 1e-5 m this is ~10 µm wide — narrow
    # enough that physical forces are unaffected outside the singular
    # point, but wide enough that wp.Tape gradient samples resolve the
    # transition smoothly rather than as a discrete step.
    pen_scale  = pen_3d * solver_pen / (solver_pen * solver_pen + eps * eps)

    # Smooth contact-active gate:
    #   smooth_step(d_proj, eps) ≈ 1 in front of the face, ≈ 0 behind;
    #   smooth_step(pen_3d, eps) ≈ 1 when spheres overlap, ≈ 0 when separated.
    # Product is in [0, 1] and recovers the hard `d_proj > 0 AND pen_3d > 0`
    # rule as eps → 0, with C^∞ behaviour across both boundaries.
    contact_gate = smooth_step(d_proj, eps) * smooth_step(pen_3d, eps)

    # Body-frame contact geometry.
    X_wb_inv = wp.transform_inverse(X_wb)
    X_tb_inv = wp.transform_inverse(X_tb)

    p0_body      = wp.transform_point(X_wb_inv, q_world)
    p1_body      = wp.transform_point(X_tb_inv, t_world)
    offset0_body = wp.transform_vector(X_wb_inv,  effective_r   * normal_ab)
    offset1_body = wp.transform_vector(X_tb_inv, -target_radius * normal_ab)

    # Diagnostic radial for optional prints (cheap — already have the pieces).
    radial_sq = dist * dist - d_proj * d_proj
    if radial_sq < 0.0:
        radial_sq = 0.0
    radial = wp.sqrt(radial_sq)

    # ── Record per-contact diagnostics (physics-neutral) ──
    # Now always written: the smooth gate replaces the hard cull, so every
    # surface-sphere slot carries valid diagnostics every step.  Multiply
    # pen_scale by the gate so the "active force magnitude" reading matches
    # the stiffness actually handed to MuJoCo.
    dbg_pen_scale[dslot]   = pen_scale * contact_gate
    dbg_solver_pen[dslot]  = solver_pen
    dbg_effective_r[dslot] = effective_r
    dbg_d_proj[dslot]      = d_proj
    dbg_radial[dslot]      = radial

    # Hybrid emission policy: emit the contact to the downstream solver
    # ONLY when the smooth gate is non-negligible (> 1e-4).  This keeps
    # the kernel-to-solver interface C^∞ across the physically meaningful
    # transition region (|d_proj| or |pen_3d| ≲ 30·eps, where smooth_step
    # varies between ~2.5e-4 and ~0.99975) while hard-culling the deep
    # tail where gate ≲ 1e-4 — a regime in which the smooth force is
    # already sub-nanoNewton *and* its gradient is machine-zero, so the
    # discrete cull costs nothing for gradient-based optimisation.
    #
    # Why the cull exists: MuJoCo's soft-constraint solver carries a
    # per-contact compliance term (c · f_n with c = 1/k).  Every live slot
    # contributes some constraint leak per step, so writing ALL 378
    # surface-sphere slots — even with near-zero stiffness — measurably
    # degrades static friction during HOLD (verified against the lift
    # test: 4 mm → 20 mm creep regression without this cull).
    #
    # gate_threshold is set so the cull activates at |d_proj| ≈ 30 · eps
    # ≈ 300 µm for eps = 1e-5 m — several times the transition width of
    # the smooth step, so gradient flow through contact onset is
    # unaffected.  This is a smooth-in-practice, hard-in-the-tail hybrid.
    gate_threshold = float(1.0e-4)
    if contact_gate < gate_threshold:
        out_shape0[buf_idx]    = -1
        out_stiffness[buf_idx] = 0.0
        debug_reason[slot]     = 3
        return

    out_shape0[buf_idx]    = s_idx
    out_shape1[buf_idx]    = target_shape_idx
    out_point0[buf_idx]    = p0_body
    out_point1[buf_idx]    = p1_body
    out_offset0[buf_idx]   = offset0_body
    out_offset1[buf_idx]   = offset1_body
    out_normal[buf_idx]    = normal_ab
    out_margin0[buf_idx]   = effective_r
    out_margin1[buf_idx]   = target_radius
    out_tids[buf_idx]      = 0

    # H1 (2026-05-10): harmonic-mean per-constraint compliance composition.
    # Masterjohn21 eq 23 / Castro22: per-constraint emitted stiffness is the
    # series composition of lattice-side and target-side material moduli,
    # which under MuJoCo's `F = ke · pen` constraint correctly enforces the
    # physical series-spring force.  Geometric correction (pen_scale ·
    # contact_gate) is applied to the FORCE (i.e. to `kc_series`), not to
    # the modulus composition itself, so the gate continues to ramp the
    # emitted stiffness smoothly to zero outside the contact region.
    # Rigid limit: as target_ke → ∞, kc_series → cslc_kc (current behavior).
    # Soft limit: as target_ke → 0, kc_series → 0 (no contact force, correct).
    # The eps² floor in the denominator guards against 0/0 when both are
    # zero; not expected at runtime.
    kc_series = (cslc_kc * target_ke) / (cslc_kc + target_ke + eps * eps)
    out_stiffness[buf_idx] = smooth_relu(
        kc_series * pen_scale * contact_gate, 1.0e-9)
    # H2: SAP-R damping emission (Castro 2022 eq 19).
    # When sim_dt > 0, emit kd = 2·sqrt(R_n_inv / imp) where
    #   R_n_inv = dt² · kc_series   (τ_d = 0 simplification)
    #   imp = 0.9                   (matches solimp max impedance)
    # Newton's MuJoCo conversion (kd > 0 branch) then gives:
    #   tc  = 2/kd = sqrt(imp/R_n_inv) = sqrt(0.9 / (dt² · kc_series))
    #   dr  = sqrt(imp / (tc² · ke))  = sqrt(R_n_inv / kc_series)
    # which preserves ke_mujoco = kc_series exactly (F = kc_series · pen).
    # When sim_dt == 0: falls back to kd=0 branch (tc = sqrt(imp/ke), dr=1).
    # THEORETICAL NOTE: both encodings give R = tc²·dr²/imp = 1/kc_series,
    # so the Anitescu velocity gap is identical — H2 is expected to produce
    # zero HoldCreep change vs the kd=0 baseline (empirical Rung 3 confirms).
    _IMP_SAP = float(0.9)
    R_n_inv = sim_dt * sim_dt * kc_series
    kd_sap = 2.0 * wp.sqrt(R_n_inv / _IMP_SAP) if sim_dt > 0.0 else 0.0
    out_damping[buf_idx] = kd_sap * contact_gate
    out_friction[buf_idx]  = 1.0

    debug_reason[slot] = 0


# ═══════════════════════════════════════════════════════════════════════════
#  CSLC vs BOX target — penetration and contact writing
#
#  The sphere variant above assumed the target is a SPHERE primitive
#  (centre + radius).  For a flat, rectangular target (a "book") the
#  contact geometry is different: the closest point on the target lies
#  on a face of the box, not at the centre.  These two box kernels
#  mirror the sphere chain (kernel 1 + kernel 3) but compute the
#  closest point on a box surface to each lattice sphere.
#
#  Convention vs the sphere variant:
#    – `point1` is the closest point on the box surface (in box body
#      frame), NOT the box centre.
#    – `margin1 = 0` (the contact point already sits on the surface).
#    – The d_proj smooth gate is computed against the
#      lattice-centre-to-box-surface direction along the pad's outward
#      normal, so it stays positive when the pad is pressing into the
#      box (matching the sphere formulation).
#
#  Both kernels handle the case where the lattice centre is INSIDE the
#  box (snap to nearest face) — important for safety even though it's
#  rare in steady-state operation: the lattice centre normally sits
#  just outside the box surface during HOLD.
# ═══════════════════════════════════════════════════════════════════════════


@wp.func
def _box_closest_local(p: wp.vec3, h: wp.vec3) -> wp.vec3:
    """Closest point on a box of half-extents h to a local-frame point p.

    For an outside point this is the per-axis clamp.  For an inside
    point we snap the axis with the smallest face distance to its
    surface, leaving the other two axes unchanged.
    """
    cx = wp.clamp(p[0], -h[0], h[0])
    cy = wp.clamp(p[1], -h[1], h[1])
    cz = wp.clamp(p[2], -h[2], h[2])

    inside_x = wp.abs(p[0]) <= h[0]
    inside_y = wp.abs(p[1]) <= h[1]
    inside_z = wp.abs(p[2]) <= h[2]
    is_inside = inside_x and inside_y and inside_z

    if is_inside:
        dx = h[0] - wp.abs(p[0])
        dy = h[1] - wp.abs(p[1])
        dz = h[2] - wp.abs(p[2])
        if dx <= dy and dx <= dz:
            cx = wp.sign(p[0]) * h[0]
        else:
            if dy <= dz:
                cy = wp.sign(p[1]) * h[1]
            else:
                cz = wp.sign(p[2]) * h[2]

    return wp.vec3(cx, cy, cz)


@wp.func
def _box_signed_dist(p: wp.vec3, h: wp.vec3) -> float:
    """Signed distance from local-frame point p to the box surface.

    Positive when p is outside, negative when inside (negative the
    perpendicular distance to the nearest face).
    """
    cx = wp.clamp(p[0], -h[0], h[0])
    cy = wp.clamp(p[1], -h[1], h[1])
    cz = wp.clamp(p[2], -h[2], h[2])
    diff = p - wp.vec3(cx, cy, cz)
    outside_dist = wp.length(diff)

    inside_x = wp.abs(p[0]) <= h[0]
    inside_y = wp.abs(p[1]) <= h[1]
    inside_z = wp.abs(p[2]) <= h[2]
    is_inside = inside_x and inside_y and inside_z

    sd = outside_dist
    if is_inside:
        dx = h[0] - wp.abs(p[0])
        dy = h[1] - wp.abs(p[1])
        dz = h[2] - wp.abs(p[2])
        sd = -wp.min(wp.min(dx, dy), dz)
    return sd


@wp.kernel
def compute_cslc_penetration_box(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    target_shape_idx: int,
    target_local_xform: wp.transform,
    target_half_extents: wp.vec3,
    eps: float,
    raw_penetration: wp.array(dtype=wp.float32),
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """Per-lattice-sphere overlap with a BOX target.

    Mirrors `compute_cslc_penetration_sphere` but with the box-surface
    signed-distance computation in place of sphere-sphere overlap.
        phi = smooth_relu(pen_3d, eps) * smooth_step(d_proj_gate, eps)
    where pen_3d = r_lat - signed_dist_to_box_surface and d_proj_gate
    is the projection of (box_centre - lattice_centre) on the pad's
    outward normal.

    *** Why the gate uses the BOX CENTROID, not the closest-point ***
    A naive `d_proj = dot(closest - q_world, n_pad)` flips sign when
    the lattice sphere transitions from outside the box (closest is
    in the +n_pad direction → gate opens correctly) to inside the box
    (closest is on the entry face, in the -n_pad direction → gate
    closes wrongly during the steady-state HOLD configuration where
    the lattice sphere is intentionally inside the box by ~1.5 mm).
    Replacing `closest` with the box centroid mirrors the sphere-vs-
    sphere convention exactly: ``diff = target_centre - q_world`` and
    ``d_proj = dot(diff, n_pad)``.  The centroid is on the +n_pad side
    of the lattice sphere whenever the pad is pressing into the box,
    regardless of whether the sphere has crossed the box surface.
    """
    tid = wp.tid()
    phi = 0.0
    n_world = wp.vec3(0.0, 0.0, 0.0)

    if sphere_shape[tid] != active_cslc_shape_idx:
        raw_penetration[tid] = 0.0
        return

    if is_surface[tid] == 1:
        s_idx = sphere_shape[tid]
        b_idx = shape_body[s_idx]
        X_ws = shape_transform[s_idx]
        X_wb = body_q[b_idx]

        p_local = sphere_pos_local[tid]
        r_lat = sphere_radii[tid]
        out_n = sphere_outward_normal[tid]

        q_body = wp.transform_point(X_ws, p_local)
        q_world = wp.transform_point(X_wb, q_body)

        # Box's world transform = target body transform composed with
        # target shape's body-local transform.
        X_tb = body_q[target_body_idx]
        X_tw = wp.transform_multiply(X_tb, target_local_xform)
        X_tw_inv = wp.transform_inverse(X_tw)

        q_target_local = wp.transform_point(X_tw_inv, q_world)
        h = target_half_extents
        signed_dist = _box_signed_dist(q_target_local, h)

        # Pad outward normal in world frame — the gate direction.
        n_body = wp.transform_vector(X_ws, out_n)
        n_world = wp.transform_vector(X_wb, n_body)

        # Sphere-convention gate: project the lattice-centre-to-box-
        # centroid vector onto the pad's outward normal.  Positive
        # whenever the pad is pressing toward the box bulk (both
        # "approaching" and "already inside" configurations) and
        # negative when the pad has been pulled past the box.
        box_centre_world = wp.transform_get_translation(X_tw)
        d_proj_gate = wp.dot(box_centre_world - q_world, n_world)

        pen_3d = r_lat - signed_dist
        phi = smooth_relu(pen_3d, eps) * smooth_step(d_proj_gate, eps)

    raw_penetration[tid] = phi
    contact_normal_out[tid] = n_world


@wp.kernel
def write_cslc_contacts_box(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    target_shape_idx: int,
    target_local_xform: wp.transform,
    target_half_extents: wp.vec3,
    contact_offset: int,
    surface_slot_map: wp.array(dtype=wp.int32),
    raw_penetration: wp.array(dtype=wp.float32),
    out_shape0: wp.array(dtype=wp.int32),
    out_shape1: wp.array(dtype=wp.int32),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_margin0: wp.array(dtype=wp.float32),
    out_margin1: wp.array(dtype=wp.float32),
    out_tids: wp.array(dtype=wp.int32),
    cslc_kc: float,
    target_ke: float,
    cslc_dc: float,
    sim_dt: float,
    eps: float,
    out_stiffness: wp.array(dtype=wp.float32),
    out_damping: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
    debug_reason: wp.array(dtype=wp.int32),
    # ── Diagnostic outputs (physics-neutral; read back by the handler) ──
    # Same layout as the sphere variant: indexed by
    # (diag_offset + slot) where diag_offset = pair_idx * n_surface_contacts.
    # `dbg_d_proj` here records `d_proj_solver` (the projection that enters
    # solver_pen), and `dbg_radial` is the in-face-plane component of
    # (closest − q) — the analog of the sphere kernel's
    # sqrt(dist² − d_proj²) lateral offset.
    diag_offset: int,
    dbg_pen_scale: wp.array(dtype=wp.float32),
    dbg_solver_pen: wp.array(dtype=wp.float32),
    dbg_effective_r: wp.array(dtype=wp.float32),
    dbg_d_proj: wp.array(dtype=wp.float32),
    dbg_radial: wp.array(dtype=wp.float32),
):
    """Write one Newton contact per active surface lattice sphere vs a BOX target.

    Mirrors `write_cslc_contacts` but uses the closest point on the box
    as `point1` (margin1 = 0) instead of a sphere centre + radius pair.
    Same hybrid emission policy: contacts whose smooth gate is below
    1e-4 are hard-culled (shape0 = -1) to avoid swamping MuJoCo's
    per-constraint compliance leak in the deep tail.

    Two distinct projections of the lattice-centre-to-box vector onto
    the pad's outward normal are used:

      * ``d_proj_solver = dot(closest_world - q_world, n_pad)`` enters
        ``solver_pen = effective_r - d_proj_solver``, which exactly
        matches what MuJoCo reconstructs from
        ``margin0 + margin1 - dot(point1_world - point0_world, normal)``
        with margin0 = effective_r, margin1 = 0,
        point1_world = closest_world.  Required for the solver-side
        penetration to be physically correct.

      * ``d_proj_gate = dot(box_centre_world - q_world, n_pad)`` drives
        the smooth contact-active gate.  Uses the box centroid (sphere-
        kernel convention) rather than ``closest_world`` because the
        latter flips sign when the lattice centre crosses the box
        surface from outside to inside.  See the analogous discussion
        in ``compute_cslc_penetration_box``.
    """
    tid = wp.tid()

    slot = surface_slot_map[tid]
    if slot < 0:
        return
    buf_idx = contact_offset + slot
    dslot = diag_offset + slot

    if sphere_shape[tid] != active_cslc_shape_idx:
        out_shape0[buf_idx] = -1
        debug_reason[slot] = 4
        # Sentinel: negative pen_scale signals "no contact this slot".
        dbg_pen_scale[dslot]   = -1.0
        dbg_solver_pen[dslot]  = 0.0
        dbg_effective_r[dslot] = 0.0
        dbg_d_proj[dslot]      = 0.0
        dbg_radial[dslot]      = 0.0
        return

    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws = shape_transform[s_idx]
    X_wb = body_q[b_idx]

    p_local = sphere_pos_local[tid]
    out_n = sphere_outward_normal[tid]
    delta_val = sphere_delta[tid]
    r_lat = sphere_radii[tid]

    q_body = wp.transform_point(X_ws, p_local)
    q_world = wp.transform_point(X_wb, q_body)

    effective_r = smooth_relu(r_lat - delta_val, eps)

    # Pad outward normal in world frame.
    n_body_normal = wp.transform_vector(X_ws, out_n)
    normal_ab = wp.transform_vector(X_wb, n_body_normal)

    # Box closest-point in box body frame.
    X_tb = body_q[target_body_idx]
    X_tw = wp.transform_multiply(X_tb, target_local_xform)
    X_tw_inv = wp.transform_inverse(X_tw)

    q_target_local = wp.transform_point(X_tw_inv, q_world)
    h = target_half_extents
    closest_local = _box_closest_local(q_target_local, h)
    signed_dist = _box_signed_dist(q_target_local, h)
    closest_world = wp.transform_point(X_tw, closest_local)

    # Solver-side projection (matches MuJoCo's reconstruction).
    diff_world = closest_world - q_world
    d_proj_solver = wp.dot(diff_world, normal_ab)

    # Gate-side projection (centroid-based; sphere-kernel convention).
    box_centre_world = wp.transform_get_translation(X_tw)
    d_proj_gate = wp.dot(box_centre_world - q_world, normal_ab)

    # Post-equilibrium effective 3-D overlap (paper eq. 5 analog for box).
    # MUST use effective_r = r_lat - δ, not r_lat: the kernel-3 force the
    # solver applies is F ≈ kc · pen_3d · gate (see pen_scale below), and
    # paper eq. 6 says F = kc · (φ_rest − δ) · gate at equilibrium.  For
    # a box target, φ_rest − δ = (r_lat − δ) − signed_dist = effective_r
    # − signed_dist.  Using r_lat here (i.e. the K1 rest overlap) instead
    # would over-stiffen each sphere by a factor (kc+ka)/ka — invisible
    # under kinematic pads (squeeze book) but wrong for dynamic pads or
    # for the per-sphere pressure-distribution claim.  Mirrors the sphere
    # variant which uses (effective_r + target_radius) − dist.  margin1
    # is 0 below because the contact point already lies on the box
    # surface.
    pen_3d = effective_r - signed_dist
    solver_pen = effective_r - d_proj_solver
    pen_scale = pen_3d * solver_pen / (solver_pen * solver_pen + eps * eps)

    contact_gate = smooth_step(d_proj_gate, eps) * smooth_step(pen_3d, eps)

    # Body-frame transforms for point0 / offset0 (lattice side) and
    # point1 / offset1 (box side).  point1 is the closest-point in the
    # target BODY frame (box's body, not its shape frame), since
    # rigid_contact_point* are body-relative.
    X_wb_inv = wp.transform_inverse(X_wb)
    X_tb_inv = wp.transform_inverse(X_tb)

    p0_body = wp.transform_point(X_wb_inv, q_world)
    p1_body = wp.transform_point(X_tb_inv, closest_world)

    # offset1 is zero because margin1 is zero (the contact point IS on
    # the surface).  offset0 still pushes the lattice sphere's surface
    # contribution along the normal by effective_r, exactly as in the
    # sphere variant.
    offset0_body = wp.transform_vector(X_wb_inv, effective_r * normal_ab)
    offset1_body = wp.vec3(0.0, 0.0, 0.0)

    # In-face-plane lateral offset of (closest − q) — analog of the
    # sphere kernel's sqrt(dist² − d_proj²) "radial" diagnostic.
    radial_vec = diff_world - d_proj_solver * normal_ab
    radial = wp.length(radial_vec)

    # ── Record per-contact diagnostics (physics-neutral) ──
    # Always written so the hard-cull below preserves the physical
    # values for inspection.  pen_scale * gate matches the stiffness
    # actually handed to MuJoCo when the gate is non-negligible.
    dbg_pen_scale[dslot]   = pen_scale * contact_gate
    dbg_solver_pen[dslot]  = solver_pen
    dbg_effective_r[dslot] = effective_r
    dbg_d_proj[dslot]      = d_proj_solver
    dbg_radial[dslot]      = radial

    # Hybrid emission: hard-cull when the smooth gate has decayed deep
    # into its tail to limit MuJoCo's per-constraint compliance leak.
    gate_threshold = float(1.0e-4)
    if contact_gate < gate_threshold:
        out_shape0[buf_idx] = -1
        out_stiffness[buf_idx] = 0.0
        debug_reason[slot] = 3
        return

    out_shape0[buf_idx] = s_idx
    out_shape1[buf_idx] = target_shape_idx
    out_point0[buf_idx] = p0_body
    out_point1[buf_idx] = p1_body
    out_offset0[buf_idx] = offset0_body
    out_offset1[buf_idx] = offset1_body
    out_normal[buf_idx] = normal_ab
    out_margin0[buf_idx] = effective_r
    out_margin1[buf_idx] = 0.0
    out_tids[buf_idx] = 0

    # H1: harmonic-mean compliance composition; see sphere-variant kernel
    # for the full derivation comment.
    kc_series = (cslc_kc * target_ke) / (cslc_kc + target_ke + eps * eps)
    out_stiffness[buf_idx] = smooth_relu(
        kc_series * pen_scale * contact_gate, 1.0e-9)
    # H2: SAP-R damping — same formula as sphere-variant kernel (see above).
    _IMP_SAP = float(0.9)
    R_n_inv = sim_dt * sim_dt * kc_series
    kd_sap = 2.0 * wp.sqrt(R_n_inv / _IMP_SAP) if sim_dt > 0.0 else 0.0
    out_damping[buf_idx] = kd_sap * contact_gate
    out_friction[buf_idx] = 1.0

    debug_reason[slot] = 0


# ═══════════════════════════════════════════════════════════════════════════
#  H3: Predictive Contact Wrench Compensation (PCWC)
#
#  Generalized predictive Coulomb friction kernel.  For each active surface
#  sphere on the current CSLC pad:
#    1. Computes the contact point in world frame from the sphere's shape-local
#       position and the pad body transform.
#    2. Computes the target body's velocity at that contact point (v_com + ω × r).
#    3. Projects the velocity error (actual − desired) onto the tangential plane
#       of the contact normal (from Kernel 1's contact_normal_scratch).
#    4. Applies a predictive Coulomb force: the force needed to eliminate the
#       tangential velocity error in one semi-implicit Euler step, clamped to
#       the per-sphere Coulomb bound μ · kc · δ.
#    5. Accumulates the force and its moment arm (r × f) into body_f as a
#       full wrench (force + torque).
#
#  Generalizations vs. the original scalar z-only implementation:
#    - v_desired: arbitrary desired CoM velocity (not hardcoded to zero)
#    - Contact normal from Kernel 1: any orientation, not just z-aligned
#    - Full wrench (force + torque): correct for off-center contact points
#
#  Runs as a single thread (dim=1) to avoid atomics; n_spheres ≤ ~400 so
#  the serial loop cost is negligible compared to the Jacobi solve.
#  Must be called per-pair inside _launch_vs_sphere, after cslc_copy_active
#  has written converged deltas and while contact_normal_scratch is still
#  valid for the current pair.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def h3_pcwc_friction(
    # CSLC geometry (all pairs, already merged into sphere_delta)
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_delta: wp.array(dtype=wp.float32),
    is_surface: wp.array(dtype=wp.int32),
    contact_normals: wp.array(dtype=wp.vec3),   # world-frame normals (any pair)
    n_spheres: int,
    kc: float,
    # Pad body transform (for contact-point positions → torque arms)
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    cslc_shape: int,
    cslc_body: int,
    # Target body dynamics
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    target_body: int,
    # Control parameters
    mass: float,
    gravity: wp.vec3,    # full 3-D gravity vector, e.g. (0, 0, -9.81)
    v_desired: wp.vec3,  # desired CoM velocity; (0,0,0) for static hold
    mu: float,
    dt: float,
):
    """Predictive Contact Wrench Compensation — runs as a single thread (dim=1).

    Computes the aggregate Coulomb friction wrench needed to arrest the target
    body's velocity error in one semi-implicit Euler step, across all active
    CSLC contacts from the current pair.

    Algorithm:
      Pass 1: sum total normal force (fn_total) and fn-weighted contact normals
              (to find the aggregate contact-normal direction n_agg).
      Pass 2: project gravity and velocity error onto the tangential plane of
              n_agg; compute a single predictive wrench for the body.
      Pass 3: distribute the wrench across contact points proportional to fn_i
              to obtain per-sphere torque arms.

    Generalisations over the original scalar z-only kernel:
      - gravity: arbitrary 3-D gravity, not hardcoded to z
      - v_desired: arbitrary desired CoM velocity (not fixed to zero)
      - n_agg: friction is in the tangential plane of the actual contact normal
      - Wrench: force + torque at the body CoM (not just force in z)

    Must run AFTER Kernel 2 writes sphere_delta for this pair, and BEFORE
    solver.step() consumes body_f.
    """
    # ── Pass 1: aggregate normal force and fn-weighted normal direction ──
    fn_total = float(0.0)
    n_fn_weighted = wp.vec3(0.0, 0.0, 0.0)

    for i in range(n_spheres):
        if is_surface[i] == 0:
            continue
        d = sphere_delta[i]
        if d <= 0.0:
            continue
        fn_i = kc * d
        fn_total = fn_total + fn_i
        n_i = contact_normals[i]
        n_fn_weighted = n_fn_weighted + fn_i * n_i

    if fn_total <= 0.0:
        return

    f_coulomb = mu * fn_total

    # Aggregate contact normal direction (fn-weighted mean, then normalize).
    # For symmetric contact (e.g. two opposing pads) the normals cancel; fall
    # back to zero-tangential-projection (treat all directions as tangential).
    n_agg_len = wp.length(n_fn_weighted)
    has_agg_normal = n_agg_len > fn_total * 1.0e-4  # relative threshold
    if has_agg_normal:
        n_agg_hat = n_fn_weighted / n_agg_len
    else:
        n_agg_hat = wp.vec3(0.0, 0.0, 0.0)   # treat all directions as tangential

    # ── Aggregate predictive force at body CoM ──
    qd = body_qd[target_body]
    v_com = wp.vec3(qd[3], qd[4], qd[5])   # body_qd = [ω_xyz, v_xyz]
    omega = wp.vec3(qd[0], qd[1], qd[2])

    # Velocity error and gravity projected onto tangential plane
    dv = v_com - v_desired
    dv_t = dv - wp.dot(dv, n_agg_hat) * n_agg_hat
    g_t = gravity - wp.dot(gravity, n_agg_hat) * n_agg_hat

    # Predictive force: eliminate velocity error + compensate tangential gravity
    f_pred = -mass * (dv_t / dt + g_t)
    f_pred_mag = wp.length(f_pred)

    if f_pred_mag < 1.0e-8:
        return

    # Clamp to aggregate Coulomb bound
    f_applied_mag = wp.min(f_pred_mag, f_coulomb)
    f_applied = (f_applied_mag / f_pred_mag) * f_pred

    # ── Pass 2: distribute force across contact points to compute torque ──
    # Each sphere i carries fraction fn_i/fn_total of the total force.
    # The resulting torque is τ = Σ_i r_i × (fn_i/fn_total · f_applied).
    X_pad_body = body_q[cslc_body]
    X_pad_shape = shape_transform[cslc_shape]
    X_pad_world = wp.transform_multiply(X_pad_body, X_pad_shape)
    X_target = body_q[target_body]
    p_com = wp.transform_get_translation(X_target)

    tau_total = wp.vec3(0.0, 0.0, 0.0)
    for i in range(n_spheres):
        if is_surface[i] == 0:
            continue
        d = sphere_delta[i]
        if d <= 0.0:
            continue
        fn_i = kc * d
        f_i = (fn_i / fn_total) * f_applied
        p_world = wp.transform_point(X_pad_world, sphere_pos_local[i])
        r_i = p_world - p_com
        tau_total = tau_total + wp.cross(r_i, f_i)

    # Accumulate wrench into body_f  [force_xyz, torque_xyz]
    bf = body_f[target_body]
    body_f[target_body] = wp.spatial_vector(
        bf[0] + f_applied[0], bf[1] + f_applied[1], bf[2] + f_applied[2],
        bf[3] + tau_total[0], bf[4] + tau_total[1], bf[5] + tau_total[2],
    )