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
def solve_particle_shape_contacts_uxpbd(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[wp.float32],
    particle_radius: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_substrate: wp.array[wp.uint8],
    particle_to_lattice: wp.array[wp.int32],
    lattice_link: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[wp.float32],
    body_I_inv: wp.array[wp.mat33],
    shape_body: wp.array[wp.int32],
    shape_material_mu: wp.array[wp.float32],
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array[wp.int32],
    contact_particle: wp.array[wp.int32],
    contact_shape: wp.array[wp.int32],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    body_delta: wp.array[wp.spatial_vector],
    body_contact_count: wp.array[wp.float32],
    particle_deltas: wp.array[wp.vec3],
):
    """Phase 2 cross-substrate particle-shape contact.

    Substrate 0 (lattice): routes Δx into the host link's spatial wrench
        (Newton's 3rd law on the shape side; self-contact with host shape is
        skipped, same as Phase 1).
    Substrate 1 (SM-rigid): routes Δx into particle_deltas; the SRXPBD
        shape-matching pass at the end of the iteration re-enforces rigidity.

    For each body that receives a wrench contribution, ``body_contact_count``
    is atomically incremented by 1. ``apply_body_deltas`` reads this as
    ``constraint_inv_weights`` and divides the accumulated wrench by the
    count, implementing UPPFRTA §4.2 constraint averaging at the body
    level. Without this, N synchronized lattice-particle penetrations
    against a single shape produce N× the per-particle wrench on the
    host body, which launches it off the ground (lattice drop regression).
    """
    tid = wp.tid()
    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    particle_index = contact_particle[tid]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    sub = particle_substrate[particle_index]
    is_lattice = sub == wp.uint8(0)

    shape_index = contact_shape[tid]
    shape_link = shape_body[shape_index]

    # Self-contact guard for the lattice case.
    if is_lattice:
        host_link = lattice_link[particle_to_lattice[particle_index]]
        if shape_link == host_link:
            return

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

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

    body_v_s = wp.spatial_vector()
    if shape_link >= 0:
        body_v_s = body_qd[shape_link]
    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)
    bv = body_v + wp.cross(body_w, r_shape) + wp.transform_vector(X_wb, contact_body_vel[tid])
    v = pv - bv

    # Position-level PBD contact: lambda_n is a signed normal *position* delta
    # (negative when penetrating). apply_particle_deltas downstream expects a
    # position delta and reconstructs v via v_new = vp + d/dt, so dividing by
    # dt here would scale the response by 1/dt and explode the velocity.
    lambda_n = c
    delta_n = n * lambda_n
    vn = wp.dot(n, v)
    vt = v - n * vn
    # Coulomb cap is on the position-level normal magnitude; the tangential
    # term is the position-level relative motion this step, vt * dt.
    # Zero-guard: at pure-normal contact, vt is numerically ~0; wp.normalize(vt)
    # would produce a noise-driven unit vector multiplied by the (non-zero)
    # mu*lambda_n cap and inject an arbitrary-direction tangential kick. Skip
    # the friction branch entirely when |vt| is below a numerical floor.
    vt_mag = wp.length(vt)
    delta_f = wp.vec3(0.0)
    if vt_mag > 1.0e-6:
        lambda_f = wp.max(mu * lambda_n, -vt_mag * dt)
        delta_f = vt * (lambda_f / vt_mag)

    # Particle-side effective inverse mass.
    if is_lattice:
        host_link = lattice_link[particle_to_lattice[particle_index]]
        host_q = body_q[host_link]
        host_com_world = wp.transform_point(host_q, body_com[host_link])
        r_lat = px - host_com_world
        angular = wp.cross(r_lat, n)
        rot_angular = wp.quat_rotate_inv(wp.transform_get_rotation(host_q), angular)
        w_particle = body_m_inv[host_link] + wp.dot(rot_angular, body_I_inv[host_link] * rot_angular)
    else:
        w_particle = particle_invmass[particle_index]

    # Shape-side effective inverse mass.
    w_shape = wp.float32(0.0)
    if shape_link >= 0:
        angular = wp.cross(r_shape, n)
        rot_angular = wp.quat_rotate_inv(wp.transform_get_rotation(X_wb), angular)
        w_shape = body_m_inv[shape_link] + wp.dot(rot_angular, body_I_inv[shape_link] * rot_angular)

    denom = w_particle + w_shape
    if denom == 0.0:
        return

    delta_total = (delta_f - delta_n) / denom * relaxation

    # Route particle-side correction.
    # SM-rigid (particle_deltas) consumer treats deltas as POSITION (m).
    # Body-side (body_delta) consumer treats spatial_top as IMPULSE (kg*m/s),
    # so divide the position-scale delta_total by dt when routing to bodies.
    if is_lattice:
        host_link = lattice_link[particle_to_lattice[particle_index]]
        host_q = body_q[host_link]
        host_com_world = wp.transform_point(host_q, body_com[host_link])
        r_lat = px - host_com_world
        body_impulse = delta_total / dt
        t_lat = wp.cross(r_lat, body_impulse)
        wp.atomic_add(body_delta, host_link, wp.spatial_vector(body_impulse, t_lat))
        wp.atomic_add(body_contact_count, host_link, 1.0)
    else:
        wp.atomic_add(particle_deltas, particle_index, delta_total * w_particle)

    # Newton's 3rd law on the shape side (impulse-scale).
    if shape_link >= 0:
        body_impulse = delta_total / dt
        t_shape = wp.cross(r_shape, body_impulse)
        wp.atomic_sub(body_delta, shape_link, wp.spatial_vector(body_impulse, t_shape))
        wp.atomic_add(body_contact_count, shape_link, 1.0)


@wp.kernel
def solve_particle_particle_contacts_uxpbd(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[wp.float32],
    particle_radius: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_group: wp.array[wp.int32],
    particle_substrate: wp.array[wp.uint8],
    particle_to_lattice: wp.array[wp.int32],
    lattice_link: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[wp.float32],
    body_I_inv: wp.array[wp.mat33],
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    # outputs
    particle_deltas: wp.array[wp.vec3],
    body_delta: wp.array[wp.spatial_vector],
    body_contact_count: wp.array[wp.float32],
):
    """Cross-substrate particle-particle contact.

    For each cross-phase pair: lattice particles route corrections into body
    wrenches, SM-rigid particles route into particle_deltas. Same-group and
    same-lattice-host pairs are skipped.

    Per UPPFRTA §4.2 constraint averaging: each contact that contributes to
    a body's wrench also increments ``body_contact_count`` by 1. The count
    is consumed by ``apply_body_deltas`` via its ``constraint_inv_weights``
    parameter to divide the accumulated wrench, preventing redundant
    co-located lattice contacts from compounding into a launch impulse.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        return
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return

    sub_i = particle_substrate[i]
    is_lat_i = sub_i == wp.uint8(0)

    x_i = particle_x[i]
    v_i = particle_v[i]
    r_i = particle_radius[i]

    query = wp.hash_grid_query(grid, x_i, r_i + max_radius + k_cohesion)
    index = int(0)
    delta_acc = wp.vec3(0.0)
    body_delta_lin = wp.vec3(0.0)
    body_delta_ang = wp.vec3(0.0)
    body_contact_n = float(0.0)

    while wp.hash_grid_query_next(query, index):
        if index == i:
            continue
        if (particle_flags[index] & ParticleFlags.ACTIVE) == 0:
            continue
        # Same particle group -> skip (handled by shape matching).
        if particle_group[i] >= 0 and particle_group[i] == particle_group[index]:
            continue
        # Skip fluid-fluid pairs (handled by PBF density constraint in step()).
        if particle_substrate[i] == wp.uint8(3) and particle_substrate[index] == wp.uint8(3):
            continue
        sub_j = particle_substrate[index]
        is_lat_j = sub_j == wp.uint8(0)
        # Same lattice host -> skip.
        if is_lat_i and is_lat_j:
            host_i = lattice_link[particle_to_lattice[i]]
            host_j = lattice_link[particle_to_lattice[index]]
            if host_i == host_j:
                continue

        n = x_i - particle_x[index]
        d = wp.length(n)
        err = d - r_i - particle_radius[index]
        if err > k_cohesion:
            continue
        if d < 1e-12:
            continue
        n_unit = n / d

        # Effective inverse mass for each side.
        if is_lat_i:
            host_i = lattice_link[particle_to_lattice[i]]
            host_q = body_q[host_i]
            r_lat_i = x_i - wp.transform_point(host_q, body_com[host_i])
            angular = wp.cross(r_lat_i, n_unit)
            rot_a = wp.quat_rotate_inv(wp.transform_get_rotation(host_q), angular)
            w_i = body_m_inv[host_i] + wp.dot(rot_a, body_I_inv[host_i] * rot_a)
        else:
            w_i = particle_invmass[i]

        if is_lat_j:
            host_j = lattice_link[particle_to_lattice[index]]
            host_q = body_q[host_j]
            r_lat_j = particle_x[index] - wp.transform_point(host_q, body_com[host_j])
            angular = wp.cross(r_lat_j, n_unit)
            rot_a = wp.quat_rotate_inv(wp.transform_get_rotation(host_q), angular)
            w_j = body_m_inv[host_j] + wp.dot(rot_a, body_I_inv[host_j] * rot_a)
        else:
            w_j = particle_invmass[index]

        denom = w_i + w_j
        if denom == 0.0:
            continue

        vrel = v_i - particle_v[index]
        # Position-level constraint (same convention as the particle-shape kernel
        # above). See comments there for the unit/consumer rationale.
        lambda_n = err
        delta_n = n_unit * lambda_n
        vn = wp.dot(n_unit, vrel)
        vt = vrel - n_unit * vn
        # Same zero-guard as solve_particle_shape_contacts_uxpbd: avoid
        # wp.normalize(vt) on a numerically-zero tangential velocity.
        vt_mag = wp.length(vt)
        delta_f = wp.vec3(0.0)
        if vt_mag > 1.0e-6:
            lambda_f = wp.max(k_mu * lambda_n, -vt_mag * dt)
            delta_f = vt * (lambda_f / vt_mag)
        d_total = (delta_f - delta_n) / denom * relaxation

        if is_lat_i:
            host_i = lattice_link[particle_to_lattice[i]]
            host_q = body_q[host_i]
            r_lat_i = x_i - wp.transform_point(host_q, body_com[host_i])
            # Convert position-scale d_total to impulse-scale for body_delta.
            body_impulse = d_total / dt
            body_delta_lin += body_impulse
            body_delta_ang += wp.cross(r_lat_i, body_impulse)
            body_contact_n += 1.0
        else:
            delta_acc += d_total * w_i

    if is_lat_i:
        host_i = lattice_link[particle_to_lattice[i]]
        wp.atomic_add(body_delta, host_i, wp.spatial_vector(body_delta_lin, body_delta_ang))
        wp.atomic_add(body_contact_count, host_i, body_contact_n)
    else:
        wp.atomic_add(particle_deltas, i, delta_acc)


@wp.kernel
def apply_particle_deltas_position_only(
    x_pred: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[wp.float32],
    delta: wp.array[wp.vec3],
    # output
    x_out: wp.array[wp.vec3],
):
    """UPPFRTA section 4.4 stabilization sub-loop position-only update.

    Updates ``x`` only -- velocity is left untouched. Used during the
    pre-main-loop stabilization pass to fix initial penetration without
    injecting kinetic energy. Per UPPFRTA Algorithm 1 lines 10-15:

        while iter < stabilizationIterations do
            Delta_x <- 0
            solve contact constraints for Delta_x
            update x_i  <- x_i + Delta_x
            update x*   <- x* + Delta_x      <-- this kernel
        end while

    The paper updates BOTH the original x_i AND the predicted x* so that
    the eventual v = (x* - x_i)/dt naturally cancels the stabilization
    correction. UXPBD's per-iteration apply_particle_deltas instead does
    v_new = vp + d/dt incrementally, so the equivalent is to apply only
    the position component during stabilization (vp is left unchanged,
    so future iterations' v_new computes from the unmodified vp).

    Static (mass=0) and inactive particles are passthrough.
    """
    tid = wp.tid()
    if particle_mass[tid] == 0.0:
        x_out[tid] = x_pred[tid]
        return
    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        x_out[tid] = x_pred[tid]
        return
    x_out[tid] = x_pred[tid] + delta[tid]


@wp.kernel
def compute_mass_scale(
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    up_axis: int,
    k_factor: float,
    # output
    scaled_inv_mass: wp.array[wp.float32],
):
    """UPPFRTA §5.2 stack-height mass scaling: m* = m * exp(-k * h).

    Returns scaled inverse mass (1 / m*) so contact kernels can read it
    directly. h is the particle's coordinate along ``up_axis`` (0=X, 1=Y, 2=Z).
    Negative h is clamped to 0.
    """
    tid = wp.tid()
    m = particle_mass[tid]
    if m <= 0.0:
        scaled_inv_mass[tid] = wp.float32(0.0)
        return
    h = particle_q[tid][up_axis]
    if h < 0.0:
        h = wp.float32(0.0)
    scale = wp.exp(-k_factor * h)
    m_eff = m * scale
    if m_eff <= 0.0:
        scaled_inv_mass[tid] = wp.float32(0.0)
    else:
        scaled_inv_mass[tid] = wp.float32(1.0) / m_eff
