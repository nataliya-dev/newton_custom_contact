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
    cslc_dc: float,
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

    # Gated stiffness with a smooth lower floor so MuJoCo's
    # timeconst = sqrt(imp / ke) stays finite when the gate drives ke → 0.
    # For fully active contacts (gate ≈ 1), the floor is invisible and the
    # stiffness recovers `cslc_kc · pen_scale` exactly.
    out_stiffness[buf_idx] = smooth_relu(
        cslc_kc * pen_scale * contact_gate, 1.0e-9)
    # DAMPING BUG (2026-04-19):
    # cslc_dc=2.0 N·s/m is calibrated for Newton's semi-implicit solver.
    # In the MuJoCo conversion kernel, kd>0 triggers timeconst = 2/kd = 1.0s,
    # making both normal AND friction constraints 250× softer than standard
    # contacts (timeconst=0.004s for ke=50000, kd=500). This soft friction
    # timeconst causes excessive Coulomb creep in the HOLD phase.
    # FIX: write 0.0 → uses kd=0 branch → timeconst = sqrt(imp/ke) ≈ 0.030s.
    # Normal force is unchanged by design of the stiffness fix; only friction
    # stiffness improves (33× stiffer timeconst). cslc_dc retained in signature.
    out_damping[buf_idx]   = 0.0
    # FRICTION BUG FIX (2026-04-19):
    # The MuJoCo conversion kernel (kernels.py) treats rigid_contact_friction as a
    # SCALE FACTOR multiplied onto the geom pair's base friction:
    #   effective_mu = geom_friction_max × rigid_contact_friction
    # The geom pair base friction is max(mu_pad, mu_sphere) = mu (from shape materials).
    # ORIGINAL: out_friction = shape_material_mu → effective_mu = mu × mu = mu² (WRONG!)
    # FIX:      out_friction = 1.0              → effective_mu = mu × 1.0 = mu (CORRECT)
    out_friction[buf_idx]  = 1.0

    debug_reason[slot] = 0