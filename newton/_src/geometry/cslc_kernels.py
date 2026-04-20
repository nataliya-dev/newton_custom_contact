#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC Warp GPU kernels for contact generation.


File location: newton/_src/geometry/cslc_kernels.py
"""

import warp as wp


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
    raw_penetration: wp.array(dtype=wp.float32),
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """raw 3-D sphere-sphere overlap per lattice sphere.

    For surface spheres belonging to the active pad:
        phi = (r_lat + R_target) − ||t_world − q_world||     if d_proj > 0
        phi = 0                                              otherwise
    For everything else: phi = 0.
    """
    tid = wp.tid()
    phi     = 0.0
    n_world = wp.vec3(0.0, 0.0, 0.0)

    # Active-pad filter.  Other pads get phi=0 so the Jacobi solve sees
    # no contact force for them during this pair's launches.
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

        # Only accept the contact if (a) there's real 3-D overlap AND
        # (b) the target is on the outward side of the face.  No radial gate.
        if pen_3d > 0.0 and d_proj > 0.0:
            phi = pen_3d

    raw_penetration[tid] = phi
    contact_normal_out[tid] = n_world



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
    active_cslc_shape_idx: int
):

    """one damped Jacobi sweep for the ACTIVE pad only.

    Non-active-pad spheres copy delta through unchanged, preserving the
    warm-start from the previous step.  Without this, each pair launch
    used to drive the OTHER pad's deltas toward zero through the Laplacian
    coupling, destroying the warm-start for the next pair.

    Formulation (unchanged):
      (ka + kl|N(i)| + kc) · δ_i = kc · φ_i + kl · Σ_j δ_j    (contact)
      (ka + kl|N(i)|)     · δ_i =            kl · Σ_j δ_j    (no contact)
    """
    tid = wp.tid()

    # Pad filter: non-active pads preserve their delta.  This keeps the
    # warm-start intact across multi-pair launches in the same step.
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
        if effective_pen > 0.0:
            f_contact = kc * phi
            k_diag    = k_diag + kc

    delta_jacobi = (f_contact + kl * neighbor_sum) / k_diag
    delta_new    = (1.0 - alpha) * delta_old + alpha * delta_jacobi
    if delta_new < 0.0:
        delta_new = 0.0
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

    effective_r = r_lat - delta_val
    if effective_r < 0.0:
        effective_r = 0.0

    # Shape B: target sphere centre in world.
    X_tb    = body_q[target_body_idx]
    t_world = wp.transform_point(X_tb, target_local_pos)

    diff = t_world - q_world
    dist = wp.length(diff)

    # Face normal in world.
    n_body    = wp.transform_vector(X_ws, out_n)
    normal_ab = wp.transform_vector(X_wb, n_body)
    d_proj    = wp.dot(diff, normal_ab)

    # Cull: target on the wrong side of the face.
    if d_proj <= 0.0:
        out_shape0[buf_idx] = -1
        debug_reason[slot]  = 3
        dbg_pen_scale[dslot]   = -1.0
        dbg_solver_pen[dslot]  = 0.0
        dbg_effective_r[dslot] = effective_r
        dbg_d_proj[dslot]      = d_proj
        dbg_radial[dslot]      = 0.0
        return

    # True 3-D overlap.
    pen_3d = (effective_r + target_radius) - dist
    if pen_3d <= 0.0:
        out_shape0[buf_idx] = -1
        debug_reason[slot]  = 1
        dbg_pen_scale[dslot]   = -1.0
        dbg_solver_pen[dslot]  = 0.0
        dbg_effective_r[dslot] = effective_r
        dbg_d_proj[dslot]      = d_proj
        dbg_radial[dslot]      = 0.0
        return

    # Solver-side projected penetration.  Already > 0 because dist >= d_proj
    # and pen_3d > 0 implies (effective_r + R) > dist >= d_proj.
    solver_pen = (effective_r + target_radius) - d_proj
    pen_scale  = pen_3d / wp.max(solver_pen, 1.0e-8)

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
    # These arrays are read back by the handler for telemetry.  They're
    # written only on the success path; cull paths wrote sentinels above.
    dbg_pen_scale[dslot]   = pen_scale
    dbg_solver_pen[dslot]  = solver_pen
    dbg_effective_r[dslot] = effective_r
    dbg_d_proj[dslot]      = d_proj
    dbg_radial[dslot]      = radial

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

    out_stiffness[buf_idx] = cslc_kc * pen_scale
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