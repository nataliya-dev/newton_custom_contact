#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC Warp GPU kernels for contact generation.

Three-kernel pipeline:
  1. compute_cslc_penetration_sphere — raw penetration per lattice sphere.
  2. jacobi_step — one damped Jacobi iteration for lattice equilibrium.
  3. write_cslc_contacts — write equilibrium contacts to Newton's Contacts buffer.

All kernels are differentiable (no atomic_add). CSLC uses pre-allocated
contact slots so the entire pipeline is smooth through Warp's tape.

File location: newton/_src/geometry/cslc_kernels.py
"""

import warp as wp


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 1: Penetration (lattice sphere vs target sphere)
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_cslc_penetration_sphere(
    # Lattice geometry
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    # Transforms
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    # Target sphere
    target_body_idx: int,
    target_shape_idx: int,
    target_local_pos: wp.vec3,
    target_radius: float,
    # Outputs
    raw_penetration: wp.array(dtype=wp.float32),
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """Compute raw penetration of each surface lattice sphere against target.

    Non-surface spheres get penetration = 0 (they don't make contact).

    The contact normal is world-frame, pointing from the lattice sphere
    toward the target (A-to-B convention matching Newton).
    """
    tid = wp.tid()

    pen = float(0.0)
    n_world = wp.vec3(0.0, 0.0, 0.0)

    if is_surface[tid] == 1:
        s_idx = sphere_shape[tid]
        b_idx = shape_body[s_idx]
        X_ws = shape_transform[s_idx]
        X_wb = body_q[b_idx]

        # Displaced lattice sphere position (used for Jacobi equilibrium)
        p_local = sphere_pos_local[tid]
        out_n = sphere_outward_normal[tid]
        delta_val = sphere_delta[tid]
        q_local = p_local + delta_val * out_n
        q_body = wp.transform_point(X_ws, q_local)
        q_world = wp.transform_point(X_wb, q_body)

        r_lat = sphere_radii[tid]

        # Target sphere world position
        X_tb = body_q[target_body_idx]
        t_world = wp.transform_point(X_tb, target_local_pos)

        diff = t_world - q_world  # A->B direction
        dist = wp.length(diff)
        pen = (r_lat + target_radius) - dist

        if dist > 1.0e-8:
            n_world = diff / dist
        else:
            # Bug #4 fix: transform outward normal to world frame for fallback.
            n_body = wp.transform_vector(X_ws, out_n)
            n_world = wp.transform_vector(X_wb, n_body)

    raw_penetration[tid] = pen
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
):
    """One damped Jacobi iteration for quasistatic lattice equilibrium.

    The equilibrium per sphere i (when contact is active, phi > delta):
        (ka + kl*|N(i)| + kc) * delta_i = kc * phi_i + kl * sum_j delta_j

    When contact is inactive (phi <= delta):
        (ka + kl*|N(i)|) * delta_i = kl * sum_j delta_j

    Under-relaxed: delta_new = (1-alpha)*delta_old + alpha*delta_jacobi
    Clamped to non-negative (sphere cannot retract into the body).
    """
    tid = wp.tid()

    delta_old = delta_src[tid]
    n_neighbors = neighbor_count[tid]

    # Neighbor coupling sum
    neighbor_sum = float(0.0)
    start = neighbor_start[tid]
    for n in range(n_neighbors):
        j = neighbor_list[start + n]
        neighbor_sum = neighbor_sum + delta_src[j]

    f_contact = float(0.0)
    k_diag = ka + kl * float(n_neighbors)

    if is_surface[tid] == 1:
        phi = raw_penetration[tid]
        effective_pen = phi - delta_old
        if effective_pen > 0.0:
            f_contact = kc * phi
            k_diag = k_diag + kc

    # Jacobi update + under-relaxation
    delta_jacobi = (f_contact + kl * neighbor_sum) / k_diag
    delta_new = (1.0 - alpha) * delta_old + alpha * delta_jacobi

    if delta_new < 0.0:
        delta_new = 0.0

    delta_dst[tid] = delta_new


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 3: Write contacts to Newton's Contacts buffer
#
#  FIX: Use REST position (not displaced) and reduce margin by delta.
#
#  The Jacobi solve determines equilibrium delta for each surface sphere.
#  The effective penetration the solver should see is (phi_raw - delta),
#  NOT (phi_raw + delta) which is what the displaced position produces.
#
#  By using the REST position and setting margin0 = max(r_lat - delta, 0),
#  the solver computes:
#    pen = margin0 + margin1 - dist_rest = (r - delta) + R - D = phi - delta
#  which is the correct equilibrium contact force.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def write_cslc_contacts(
    # Lattice data
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    # Transforms
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    # Target info
    target_body_idx: int,
    target_shape_idx: int,
    target_local_pos: wp.vec3,
    target_radius: float,
    # Pre-allocated slot mapping (no atomic_add -> differentiable)
    contact_offset: int,
    surface_slot_map: wp.array(dtype=wp.int32),
    # ── Newton Contacts buffer arrays ──
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
    # ── Per-contact material properties ──
    shape_material_mu: wp.array(dtype=wp.float32),
    cslc_kc: float,
    cslc_dc: float,
    out_stiffness: wp.array(dtype=wp.float32),
    out_damping: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
):
    """Write one contact per surface sphere into Newton's Contacts buffer.

    Uses the REST position (no delta displacement) and reduces margin0
    by delta to encode the equilibrium deformation. This makes the solver
    compute pen = phi - delta (correct) instead of phi + delta (wrong).
    """
    tid = wp.tid()

    slot = surface_slot_map[tid]
    if slot < 0:
        return

    buf_idx = contact_offset + slot

    # ── Shape A: lattice sphere ──
    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws = shape_transform[s_idx]
    X_wb = body_q[b_idx]

    p_local = sphere_pos_local[tid]
    out_n = sphere_outward_normal[tid]
    delta_val = sphere_delta[tid]
    r_lat = sphere_radii[tid]

    # FIX: Use REST position (no delta displacement).
    # The delta is encoded in the reduced margin instead.
    q_local = p_local  # was: p_local + delta_val * out_n
    q_body = wp.transform_point(X_ws, q_local)
    q_world = wp.transform_point(X_wb, q_body)

    # Effective radius: reduced by lattice deformation.
    # This makes the solver see pen = (r - delta) + R - D = phi - delta.
    effective_r = r_lat - delta_val
    if effective_r < 0.0:
        effective_r = 0.0

    # ── Shape B: target sphere ──
    X_tb = body_q[target_body_idx]
    t_world = wp.transform_point(X_tb, target_local_pos)

    # ── Normal A->B (world frame) ──
    diff = t_world - q_world
    dist = wp.length(diff)

    # Check penetration using effective radius
    pen = (effective_r + target_radius) - dist
    if pen <= 0.0:
        out_shape0[buf_idx] = -1
        return

    if dist > 1.0e-8:
        normal_ab = diff / dist
    else:
        n_body = wp.transform_vector(X_ws, out_n)
        normal_ab = wp.transform_vector(X_wb, n_body)

    # ── Body-frame contact points and offsets ──
    X_wb_inv = wp.transform_inverse(X_wb)
    X_tb_inv = wp.transform_inverse(X_tb)

    p0_body = wp.transform_point(X_wb_inv, q_world)
    p1_body = wp.transform_point(X_tb_inv, t_world)

    # Offsets use effective_r (matching Newton's convention: offset_mag = margin)
    offset0_body = wp.transform_vector(X_wb_inv, effective_r * normal_ab)
    offset1_body = wp.transform_vector(X_tb_inv, -target_radius * normal_ab)

    # ── Write ──
    out_shape0[buf_idx] = s_idx
    out_shape1[buf_idx] = target_shape_idx
    out_point0[buf_idx] = p0_body
    out_point1[buf_idx] = p1_body
    out_offset0[buf_idx] = offset0_body
    out_offset1[buf_idx] = offset1_body
    out_normal[buf_idx] = normal_ab
    out_margin0[buf_idx] = effective_r     # was: r_lat
    out_margin1[buf_idx] = target_radius
    out_tids[buf_idx] = 0

    out_stiffness[buf_idx] = cslc_kc
    out_damping[buf_idx] = cslc_dc
    out_friction[buf_idx] = shape_material_mu[s_idx]