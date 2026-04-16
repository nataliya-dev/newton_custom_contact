#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC Warp GPU kernels for contact generation (v3).

v3 changes from v2:
  - Removed point0 shift. Without it, point0 stays at the CSLC sphere
    center (geometrically clean), and each contact sees a uniform
    solver penetration = effective_r + target_radius - d_proj, where
    d_proj is the horizontal distance from the CSLC sphere center to
    the target center. This matches the "uniform flat contact patch"
    assumption used in the kc calibration derivation.

Three-kernel pipeline:
  1. compute_cslc_penetration_sphere — raw penetration per lattice sphere.
  2. jacobi_step — one damped Jacobi iteration for lattice equilibrium.
  3. write_cslc_contacts — write equilibrium contacts to Newton's Contacts buffer.

All kernels are differentiable (no atomic_add). CSLC uses pre-allocated
contact slots so the entire pipeline is smooth through Warp's tape.

TUNING NOTES:
  For gripper geometries where only a fraction of surface spheres are in
  contact, the default contact_fraction=0.3 in calibrate_kc overestimates
  active contacts and produces too-soft kc. Recommend:
    - cslc_spacing = 10mm (not 5mm) for typical gripper pads
    - contact_fraction = 0.1 in calibrate_kc
  These produce kc ~ 100-200 N/m, which MuJoCo handles well at 500Hz.

File location: newton/_src/geometry/cslc_kernels.py
"""

import warp as wp

CSLC_RADIAL_CUTOFF = 2.0e-3

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



    tid = wp.tid()
    phi = 0.0
    n_world = wp.vec3(0.0, 0.0, 0.0)


    if sphere_shape[tid] != active_cslc_shape_idx:
        raw_penetration[tid] = 0.0   # penetration kernel only
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

        X_tb = body_q[target_body_idx]
        t_world = wp.transform_point(X_tb, target_local_pos)

        diff = t_world - q_world
        dist = wp.length(diff)

        n_body = wp.transform_vector(X_ws, out_n)
        n_world = wp.transform_vector(X_wb, n_body)

        d_proj = wp.dot(diff, n_world)
        pen_3d = (r_lat + target_radius) - dist

        radial_sq = dist * dist - d_proj * d_proj
        if radial_sq < 0.0:
            radial_sq = 0.0
        radial = wp.sqrt(radial_sq)



        if pen_3d > 0.0 and d_proj > 0.0 and radial <= CSLC_RADIAL_CUTOFF:
            phi = (r_lat + target_radius) - d_proj




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
):
    """Damped Jacobi step for quasistatic lattice equilibrium.

    (ka + kl*|N(i)| + kc) * delta_i = kc * phi_i + kl * sum_j delta_j  (if contact)
    (ka + kl*|N(i)|) * delta_i = kl * sum_j delta_j                    (no contact)
    """
    tid = wp.tid()

    delta_old = delta_src[tid]
    n_neighbors = neighbor_count[tid]

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

    # if is_surface[tid] == 1 and (tid == 136 or tid == 191 or tid == 192):
    #     phi = raw_penetration[tid]
    #     effective_pen = phi - delta_old
    #     wp.printf(
    #         "CSLC_JAC tid=%d phi=%.6f delta_old=%.6f effective_pen=%.6f fcontact_used=%.6f\n",
    #         tid, phi, delta_old, effective_pen,
    #         kc * phi if effective_pen > 0.0 else 0.0
    #     )


    delta_jacobi = (f_contact + kl * neighbor_sum) / k_diag
    delta_new = (1.0 - alpha) * delta_old + alpha * delta_jacobi

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
    # Lattice data
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
    # Pre-allocated slot mapping (no atomic_add -> differentiable)
    contact_offset: int,
    surface_slot_map: wp.array(dtype=wp.int32),
    raw_penetration: wp.array(dtype=wp.float32),
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
    # debug
    debug_reason: wp.array(dtype=wp.int32)
):
    """Write one contact per active surface sphere (v3, no shift).

    Convention:
      point0 = CSLC sphere center (body frame)    — NO shift
      point1 = target sphere center (body frame)
      normal = outward surface normal (world)     — horizontal for flat faces
      margin0 = effective_r = r_lat - delta
      margin1 = target_radius
      offset0 = effective_r * normal              (body frame)
      offset1 = -target_radius * normal           (body frame)

    Solver sees: pen = (effective_r + target_radius) - dot(diff, normal)
                     = (effective_r + target_radius) - d_proj
    Force is horizontal; friction lies in the vertical+lateral tangent plane.

    We cull contacts where the 3D overlap is zero/negative — an off-axis
    sphere may have d_proj > 0 but actually not overlap in 3D.
    """

    tid = wp.tid()


    slot = surface_slot_map[tid]
    if slot < 0:
        return

    buf_idx = contact_offset + slot




    if sphere_shape[tid] != active_cslc_shape_idx:
        out_shape0[buf_idx] = -1
        debug_reason[slot] = 4   # wrong pad for this pair
        return

    # ── Shape A: lattice sphere ──
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

    effective_r = r_lat - delta_val
    if effective_r < 0.0:
        effective_r = 0.0

    # ── Shape B: target sphere ──
    X_tb = body_q[target_body_idx]
    t_world = wp.transform_point(X_tb, target_local_pos)

    diff = t_world - q_world
    dist = wp.length(diff)


    # Outward-normal contact direction
    n_body = wp.transform_vector(X_ws, out_n)
    normal_ab = wp.transform_vector(X_wb, n_body)

    # Target must be on the outward side of the surface
    d_proj = wp.dot(diff, normal_ab)

    radial_sq = dist * dist - d_proj * d_proj
    if radial_sq < 0.0:
        radial_sq = 0.0
    radial = wp.sqrt(radial_sq)

    if radial > CSLC_RADIAL_CUTOFF:
        out_shape0[buf_idx] = -1
        # if tid == 136 or tid == 148 or tid == 191 or tid == 192:
        #     wp.printf(
        #         "CSLC_RADIAL_CULL tid=%d slot=%d radial=%.6f dist3d=%.6f dproj=%.6f pen3d=%.6f\n",
        #         tid, slot, radial, dist, d_proj, pen_3d
        #     )
        return


    # if tid == 136 or tid == 148 or tid == 191 or tid == 192:
    #     wp.printf(
    #         "CSLC_GEOM tid=%d slot=%d diff=(%.6f, %.6f, %.6f) normal=(%.6f, %.6f, %.6f) dist3d=%.6f dproj=%.6f pen3d=%.6f pensolver=%.6f\n",
    #         tid, slot,
    #         diff[0], diff[1], diff[2],
    #         normal_ab[0], normal_ab[1], normal_ab[2],
    #         dist, d_proj, pen_3d, (effective_r + target_radius) - d_proj
    #     )


    # if tid == 136 or tid == 148 or tid == 191 or tid == 192:
    #     wp.printf(
    #         "CSLC_WRITE tid=%d slot=%d delta=%.6f dist3d=%.6f dproj=%.6f pen3d=%.6f pensolver=%.6f qz=%.6f tz=%.6f\n",
    #         tid, slot, delta_val, dist, d_proj, pen_3d, pen_solver, q_world[2], t_world[2]
    #     )

    pen_solver = (effective_r + target_radius) - d_proj

    raw_phi = raw_penetration[tid]
    expected_pen = raw_phi - delta_val
    mismatch = pen_solver - expected_pen

    # if tid == 192:
    #     wp.printf(
    #         "CSLC_MATCH tid=%d raw_phi=%.6f delta=%.6f expected_pen=%.6f pen_solver=%.6f mismatch=%.6f\n",
    #         tid, raw_phi, delta_val, expected_pen, pen_solver, mismatch
    #     )



    if d_proj <= 0.0:
        out_shape0[buf_idx] = -1
        return

    # Body-frame contact points (NO SHIFT)
    X_wb_inv = wp.transform_inverse(X_wb)
    X_tb_inv = wp.transform_inverse(X_tb)

    p0_body = wp.transform_point(X_wb_inv, q_world)
    p1_body = wp.transform_point(X_tb_inv, t_world)

    offset0_body = wp.transform_vector(X_wb_inv, effective_r * normal_ab)
    offset1_body = wp.transform_vector(X_tb_inv, -target_radius * normal_ab)


    pen_3d = (effective_r + target_radius) - dist

    if tid == 136 or tid == 148 or tid == 191 or tid == 192:
        wp.printf(
            "CSLC_STATE tid=%d phi=%.6f delta=%.6f effpen=%.6f dproj=%.6f pen3d=%.6f radial=%.6f active=%d\n",
            tid,
            raw_phi,              # or phi in kernel 1
            delta_val,            # use 0.0 in kernel 1 if easier
            raw_phi - delta_val,  # or phi - delta_old in Jacobi
            d_proj,
            pen_3d,
            radial,
            1 if (pen_3d > 0.0 and d_proj > 0.0 and radial <=CSLC_RADIAL_CUTOFF) else 0
        )



    active = 1 if (pen_3d > 0.0 and d_proj > 0.0 and radial <= CSLC_RADIAL_CUTOFF) else 0
    near_edge = radial > (CSLC_RADIAL_CUTOFF - 3.0e-4)

    if (tid == 136 or tid == 191 or tid == 192 or tid == 148) and (near_edge or active == 0):
        wp.printf(
            "CSLC_EDGE tid=%d dproj=%.6f pen3d=%.6f radial=%.6f cutoff=%.6f active=%d delta=%.6f\n",
            tid, d_proj, pen_3d, radial, CSLC_RADIAL_CUTOFF, active, delta_val
        )

    debug_reason[slot] = 99

    # 3D overlap check — cull contacts with no actual 3D intersection

    if pen_3d <= 0.0:
        out_shape0[buf_idx] = -1
        debug_reason[slot] = 1
        return


    radial_sq = dist * dist - d_proj * d_proj
    if radial_sq < 0.0:
        radial_sq = 0.0

    if radial > CSLC_RADIAL_CUTOFF:
        out_shape0[buf_idx] = -1
        debug_reason[slot] = 2
        return


    if d_proj <= 0.0:
        out_shape0[buf_idx] = -1
        debug_reason[slot] = 3   # dproj fail
        return

    debug_reason[slot] = 0       # wrote successfully



    out_shape0[buf_idx] = s_idx
    out_shape1[buf_idx] = target_shape_idx
    out_point0[buf_idx] = p0_body
    out_point1[buf_idx] = p1_body
    out_offset0[buf_idx] = offset0_body
    out_offset1[buf_idx] = offset1_body
    out_normal[buf_idx] = normal_ab
    out_margin0[buf_idx] = effective_r
    out_margin1[buf_idx] = target_radius
    out_tids[buf_idx] = 0

    out_stiffness[buf_idx] = cslc_kc
    out_damping[buf_idx] = cslc_dc
    out_friction[buf_idx] = shape_material_mu[s_idx]