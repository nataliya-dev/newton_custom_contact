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

        # Displaced lattice sphere position
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
            n_world = out_n

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

    Equilibrium per sphere i:
        (ka + kl*|N(i)|) * delta_i = kc * max(phi_i - delta_old_i, 0) + kl * sum_j delta_j

    Under-relaxed: delta_new = (1-alpha)*delta_old + alpha*delta_jacobi
    Clamped to non-negative (sphere cannot retract into the body).
    """
    tid = wp.tid()

    delta_old = delta_src[tid]
    n_neighbors = neighbor_count[tid]
    k_diag = ka + kl * float(n_neighbors)

    # Neighbor coupling sum
    neighbor_sum = float(0.0)
    start = neighbor_start[tid]
    for n in range(n_neighbors):
        j = neighbor_list[start + n]
        neighbor_sum = neighbor_sum + delta_src[j]

    # Contact force (surface spheres only)
    f_contact = float(0.0)
    if is_surface[tid] == 1:
        phi = raw_penetration[tid]
        effective_pen = phi - delta_old
        if effective_pen > 0.0:
            f_contact = kc * effective_pen

    # Jacobi update + under-relaxation
    delta_jacobi = (f_contact + kl * neighbor_sum) / k_diag
    delta_new = (1.0 - alpha) * delta_old + alpha * delta_jacobi

    if delta_new < 0.0:
        delta_new = 0.0

    delta_dst[tid] = delta_new


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 3: Write contacts to Newton's Contacts buffer
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
    # Field names match contacts.py / ContactWriterData exactly:
    out_shape0: wp.array(dtype=wp.int32),
    out_shape1: wp.array(dtype=wp.int32),
    out_point0: wp.array(dtype=wp.vec3),      # body-frame contact point on A
    out_point1: wp.array(dtype=wp.vec3),      # body-frame contact point on B
    out_offset0: wp.array(dtype=wp.vec3),     # body-frame friction anchor offset A
    out_offset1: wp.array(dtype=wp.vec3),     # body-frame friction anchor offset B
    out_normal: wp.array(dtype=wp.vec3),       # world-frame normal (A->B)
    out_margin0: wp.array(dtype=wp.float32),   # surface thickness A
    out_margin1: wp.array(dtype=wp.float32),   # surface thickness B
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

    Each surface sphere writes to a pre-allocated slot:
        buf_idx = contact_offset + surface_slot_map[tid]

    Non-surface spheres skip (slot == -1).

    Convention (matching Newton's write_contact in collide.py):
      - normal: world-frame, A->B (from CSLC lattice toward target)
      - point0: lattice sphere center in CSLC body frame
      - point1: target sphere center in target body frame
      - offset0: margin0 * normal in CSLC body frame
      - offset1: -margin1 * normal in target body frame
      - margin0: lattice sphere radius (effective surface thickness)
      - margin1: target sphere radius (effective surface thickness)

    The solver computes signed separation as:
      d = dot(p1_world - p0_world, normal) - (margin0 + margin1)
    Force is applied when d < 0 (penetrating).

    No atomic_add anywhere -> fully differentiable via wp.Tape.
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
    q_local = p_local + delta_val * out_n

    # Lattice sphere center -> world
    q_body = wp.transform_point(X_ws, q_local)
    q_world = wp.transform_point(X_wb, q_body)
    r_lat = sphere_radii[tid]

    # ── Shape B: target sphere ──
    X_tb = body_q[target_body_idx]
    t_world = wp.transform_point(X_tb, target_local_pos)

    # ── Normal A->B (world frame) ──
    diff = t_world - q_world
    dist = wp.length(diff)
    if dist > 1.0e-8:
        normal_ab = diff / dist
    else:
        normal_ab = out_n

    # ── Body-frame transforms ──
    X_wb_inv = wp.transform_inverse(X_wb)
    X_tb_inv = wp.transform_inverse(X_tb)

    # Contact points: sphere centers in body frames
    p0_body = wp.transform_point(X_wb_inv, q_world)
    p1_body = wp.transform_point(X_tb_inv, t_world)

    # Friction anchor offsets in body frames
    # offset0 points from center toward contact surface (along +normal in world)
    # offset1 points from center toward contact surface (along -normal in world)
    offset0_body = wp.transform_vector(X_wb_inv, r_lat * normal_ab)
    offset1_body = wp.transform_vector(X_tb_inv, -target_radius * normal_ab)

    # ── Write ──
    out_shape0[buf_idx] = s_idx
    out_shape1[buf_idx] = target_shape_idx
    out_point0[buf_idx] = p0_body
    out_point1[buf_idx] = p1_body
    out_offset0[buf_idx] = offset0_body
    out_offset1[buf_idx] = offset1_body
    out_normal[buf_idx] = normal_ab
    out_margin0[buf_idx] = r_lat
    out_margin1[buf_idx] = target_radius
    out_tids[buf_idx] = 0

    # ── Per-contact material properties ──
    # Stiffness: calibrated per-sphere kc so N contacts sum to bulk ke.
    # Damping: CSLC-specific dc (Hunt-Crossley coefficient). The solver
    #   applies f_n = kc * max(-d,0) * (1 + dc * max(-v_n,0)), so dc
    #   controls energy dissipation during approach. Using CSLC's own dc
    #   rather than averaging shape kd gives correct per-sphere damping
    #   consistent with the distributed stiffness calibration.
    # Friction: average of shape-pair mu (surface property, not structural).
    out_stiffness[buf_idx] = cslc_kc
    out_damping[buf_idx] = cslc_dc
    # rigid_contact_friction is a SCALE FACTOR (see eval_body_contact):
    # the solver already computes mu = avg(shape_mu_a, shape_mu_b),
    # then multiplies by this value. Write 1.0 for neutral scaling.
    out_friction[buf_idx] = 1.0