# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels specific to UXPBD: lattice projection and lattice-aware contact.

The remaining kernels used by the solver (joint resolution, body integration,
restitution, body_parent_f reporting) live in
:mod:`newton._src.solvers.xpbd.kernels` and are imported there.
"""

import warp as wp

from ...geometry import ParticleFlags


@wp.kernel
def update_lattice_world_positions(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    lattice_link: wp.array[wp.int32],
    lattice_p_rest: wp.array[wp.vec3],
    lattice_delta: wp.array[float],
    lattice_r: wp.array[float],
    lattice_particle_index: wp.array[wp.int32],
    # outputs
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
):
    """Project body_q onto lattice particles.

    For lattice sphere ``i`` with body-frame offset ``p_rest`` hosted by
    ``link``, set:

    ``particle_q[pidx] = body_q[link] x p_rest``,
    ``particle_qd[pidx] = v_lin + omega x (R . (p_rest - com))``,
    ``particle_radius[pidx] = lattice_r[i] - lattice_delta[i]``  (delta == 0 in v1).

    The ``particle_radius`` write is the load-bearing CSLC v2 seam: v2 writes
    nonzero ``lattice_delta`` and downstream contact kernels automatically
    see the compressed effective radius.
    """
    sid = wp.tid()
    link = lattice_link[sid]
    p_local = lattice_p_rest[sid]
    tf = body_q[link]
    pidx = lattice_particle_index[sid]

    # World position
    particle_q[pidx] = wp.transform_point(tf, p_local)

    # World velocity at offset
    rot = wp.transform_get_rotation(tf)
    r_world = wp.quat_rotate(rot, p_local - body_com[link])
    twist = body_qd[link]
    v_lin = wp.spatial_top(twist)
    omega = wp.spatial_bottom(twist)
    particle_qd[pidx] = v_lin + wp.cross(omega, r_world)

    # Effective contact radius. v1: delta == 0 so radius == rest radius.
    particle_radius[pidx] = lattice_r[sid] - lattice_delta[sid]


@wp.func
def lattice_sphere_w_eff(
    body_inv_mass: float,
    body_inv_inertia: wp.mat33,
    body_rot: wp.quat,
    r_world: wp.vec3,
    n: wp.vec3,
) -> float:
    """Effective inverse mass at a lattice sphere along contact normal ``n``.

    Implements ``w_eff = w_body + (r x n)^T . W_world . (r x n)``, where
    ``W_world = R . I^{-1} . R^T``. Matches the inverse-mass term used in
    XPBD's ``solve_body_contact_positions``.
    """
    angular = wp.cross(r_world, n)
    rot_angular = wp.quat_rotate_inv(body_rot, angular)
    return body_inv_mass + wp.dot(rot_angular, body_inv_inertia * rot_angular)


@wp.kernel
def solve_lattice_shape_contacts(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    lattice_particle_index: wp.array[wp.int32],
    lattice_link: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    shape_body: wp.array[int],
    shape_material_mu: wp.array[float],
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    particle_to_lattice: wp.array[wp.int32],
    dt: float,
    relaxation: float,
    # outputs
    body_delta: wp.array[wp.spatial_vector],
):
    """Lattice-aware variant of XPBD's solve_particle_shape_contacts.

    Routes the position correction for each lattice-sphere contact into the
    host link's spatial wrench accumulator. Non-lattice particles are
    ignored by this kernel; XPBD's solve_particle_shape_contacts handles
    those in later phases.
    """
    tid = wp.tid()
    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    particle_index = contact_particle[tid]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    # Only handle lattice particles in this kernel.
    lat_idx = particle_to_lattice[particle_index]
    if lat_idx < 0:
        return

    host_link = lattice_link[lat_idx]
    shape_index = contact_shape[tid]
    shape_link = shape_body[shape_index]

    # Skip self-contacts. When a link carries both an analytical collision shape
    # (e.g. add_shape_box) and a MorphIt lattice, the collision pipeline finds
    # particle-shape overlaps between the embedded lattice spheres and the host
    # body's own analytical shape. Resolving these as contacts produces spurious
    # bidirectional impulses that diverge within ~50 frames. Skipping when the
    # contact's shape body equals the lattice sphere's host link is the
    # correct fix: the lattice spheres ARE the body's particle-world
    # representation, so the analytical shape on the same body is logically
    # redundant for physics (it remains for rendering / debug visualization).
    if shape_link == host_link:
        return

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    # Build the world-space contact frame from the shape side.
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if shape_link >= 0:
        X_wb = body_q[shape_link]
        X_com = body_com[shape_link]

    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r_shape = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]
    if c > particle_ka:
        return

    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])

    # Shape-side body velocity at contact.
    body_v_s = wp.spatial_vector()
    if shape_link >= 0:
        body_v_s = body_qd[shape_link]
    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)
    bv = body_v + wp.cross(body_w, r_shape) + wp.transform_vector(X_wb, contact_body_vel[tid])

    # Relative velocity at contact.
    v = pv - bv

    # Effective inverse mass on the lattice side using the host link's inertia.
    host_q = body_q[host_link]
    host_com_world = wp.transform_point(host_q, body_com[host_link])
    r_lat = px - host_com_world
    w_lat = lattice_sphere_w_eff(
        body_m_inv[host_link],
        body_I_inv[host_link],
        wp.transform_get_rotation(host_q),
        r_lat,
        n,
    )

    # Effective inverse mass on the shape side.
    w_shape = float(0.0)
    if shape_link >= 0:
        angular = wp.cross(r_shape, n)
        q_shape = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q_shape, angular)
        w_shape = body_m_inv[shape_link] + wp.dot(rot_angular, body_I_inv[shape_link] * rot_angular)

    denom = w_lat + w_shape
    if denom == 0.0:
        return

    # Normal correction in velocity domain: lambda = c / (dt * denom).
    # Matching the convention of solve_body_contact_positions (compute_contact_constraint_delta),
    # apply_body_deltas interprets body_delta linear part as velocity-domain, so
    # dp = body_delta * inv_m [m/s], p += dp * dt [m], v += dp [m/s].
    # Dividing c by dt gives impulse-scale correction sufficient to stop a fast body.
    lambda_n = c / dt
    delta_n = n * lambda_n

    # Friction (Coulomb, velocity-level).
    vn = wp.dot(n, v)
    vt = v - n * vn
    lambda_f = wp.max(mu * lambda_n, -wp.length(vt))
    delta_f = wp.normalize(vt) * lambda_f

    delta_total = (delta_f - delta_n) / denom * relaxation

    # Route the position correction into the host body.
    # The lattice sphere is rigidly attached to the body, so the body receives
    # the same correction direction as the sphere (unlike XPBD particle-shape
    # where the shape body receives the Newton's-3rd-law reaction).
    # apply_body_deltas: dp = spatial_top(body_delta) * inv_m => p += dp * dt,
    # so a positive linear contribution moves the body in that direction.
    t_lat = wp.cross(r_lat, delta_total)
    wp.atomic_add(body_delta, host_link, wp.spatial_vector(delta_total, t_lat))

    if shape_link >= 0:
        # Shape is dynamic: also push back its body.
        t_shape = wp.cross(r_shape, delta_total)
        wp.atomic_sub(body_delta, shape_link, wp.spatial_vector(delta_total, t_shape))
