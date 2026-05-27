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
    lattice_normal: wp.array[wp.vec3],
    lattice_delta: wp.array[wp.vec3],
    lattice_delta_prev: wp.array[wp.vec3],
    lattice_r: wp.array[float],
    lattice_particle_index: wp.array[wp.int32],
    dt: float,
    clamp_delta_dot_max: float,  # < 0 disables (uses sentinel)
    # outputs
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
):
    """Project body_q onto lattice particles, with CSLC compliance applied.

    For lattice sphere ``i`` with body-frame offset ``p_rest`` hosted by
    ``link``:

    - ``p_pin = body_q[link] x p_rest``  (rigid-lattice "kinematic pin")
    - ``particle_q[pidx] = p_pin - lattice_delta[sid]``  (CSLC contract
      sign convention :math:`q_i = p_i - \\delta_i`; ``δ`` along
      ``+n_outward`` corresponds to compression inward)
    - ``particle_qd[pidx] = v_lin + omega x r_world  -  delta_dot``
      where ``delta_dot = (lattice_delta - lattice_delta_prev) / dt``
      is the per-substep vec3 finite difference (Hunt-Crossley rate
      coupling).
    - ``particle_radius[pidx] = lattice_r[sid]``  (rest radius unchanged;
      compliance is in the position shift, not a radius shrink, so
      downstream contact sees the same sphere geometry just translated)

    ``δ`` is a vec3 so it can carry tangential shear (used by the
    stick-slip friction term in the Jacobi solver) in addition to normal
    compression.  The v1 closed-form solver writes ``δ_n · n_outward``;
    the Jacobi solver writes the full vec3.

    ``clamp_delta_dot_max`` (in [m/s]) bounds ``|delta_dot|`` per-axis
    against the warmup spike on first contact when ``lattice_delta_prev
    = 0`` and ``lattice_delta`` jumps to a finite value in one substep.
    Pass a negative sentinel to disable.  ``dt <= 0`` suppresses the
    rate term entirely (used for the pre-step projection where
    ``lattice_delta_prev`` is stale).
    """
    sid = wp.tid()
    link = lattice_link[sid]
    p_local = lattice_p_rest[sid]
    tf = body_q[link]
    pidx = lattice_particle_index[sid]

    # CSLC seam displacement for this substep (already populated by
    # compute_compliant_contact_response; v1 anchor-only closed-form or
    # v2 Jacobi solve, indistinguishable here).
    delta = lattice_delta[sid]

    # Rigid-lattice pin + inward CSLC displacement.
    p_pin = wp.transform_point(tf, p_local)
    particle_q[pidx] = p_pin - delta

    # World velocity at offset (rigid-lattice term).
    rot = wp.transform_get_rotation(tf)
    r_world = wp.quat_rotate(rot, p_local - body_com[link])
    twist = body_qd[link]
    v_lin = wp.spatial_top(twist)
    omega = wp.spatial_bottom(twist)
    v_rigid = v_lin + wp.cross(omega, r_world)

    # Hunt-Crossley velocity coupling: per-substep finite difference of
    # δ contributes to the particle's world velocity.  Skip when dt<=0
    # (caller signals "no rate term", e.g. pre-step initialization).
    delta_dot = wp.vec3(0.0, 0.0, 0.0)
    if dt > 0.0:
        delta_dot = (delta - lattice_delta_prev[sid]) / dt
        if clamp_delta_dot_max >= 0.0:
            # Per-axis symmetric clamp.  A magnitude-based clamp would
            # change the direction at the threshold; per-axis preserves
            # it (each component independently saturated).
            dx = delta_dot[0]
            dy = delta_dot[1]
            dz = delta_dot[2]
            if dx > clamp_delta_dot_max:
                dx = clamp_delta_dot_max
            elif dx < -clamp_delta_dot_max:
                dx = -clamp_delta_dot_max
            if dy > clamp_delta_dot_max:
                dy = clamp_delta_dot_max
            elif dy < -clamp_delta_dot_max:
                dy = -clamp_delta_dot_max
            if dz > clamp_delta_dot_max:
                dz = clamp_delta_dot_max
            elif dz < -clamp_delta_dot_max:
                dz = -clamp_delta_dot_max
            delta_dot = wp.vec3(dx, dy, dz)
    particle_qd[pidx] = v_rigid - delta_dot

    # Geometric radius is preserved; compliance is expressed through the
    # particle position shift, not a radius shrink.
    particle_radius[pidx] = lattice_r[sid]


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
    body_articulation: wp.array[wp.int32],
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

    # Self-contact guard for the lattice case. Skip same host, and also
    # skip cross-link pairs on the same articulation: adjacent URDF links
    # carry placeholder collision shapes that are physically enclosed by
    # neighbours' lattice spheres, and treating those as contacts launches
    # the kinematic chain. -1 (free body) is treated as a unique
    # articulation so two free bodies still collide.
    if is_lattice:
        host_link = lattice_link[particle_to_lattice[particle_index]]
        if shape_link == host_link:
            return
        if shape_link >= 0:
            art_host = body_articulation[host_link]
            art_shape = body_articulation[shape_link]
            if art_host >= 0 and art_host == art_shape:
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
    body_articulation: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[wp.float32],
    body_I_inv: wp.array[wp.mat33],
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    cslc_owns_lattice_wrench: int,
    # outputs
    particle_deltas: wp.array[wp.vec3],
    body_delta: wp.array[wp.spatial_vector],
    body_contact_count: wp.array[wp.float32],
):
    """Cross-substrate particle-particle contact.

    For each cross-phase pair: lattice particles route corrections into body
    wrenches, SM-rigid particles route into particle_deltas. Same-group,
    same-lattice-host, and same-articulation lattice pairs are skipped — the
    last guard prevents adjacent links of a URDF-loaded robot (whose lattice
    spheres overlap by design at every joint) from generating contact
    wrenches that tear the kinematic chain apart. ``body_articulation``
    carries per-body articulation IDs (-1 for free bodies); -1 bodies are
    treated as independent, so two free bodies still collide.

    Per UPPFRTA §4.2 constraint averaging: each contact that contributes to
    a body's wrench also increments ``body_contact_count`` by 1. The count
    is consumed by ``apply_body_deltas`` via its ``constraint_inv_weights``
    parameter to divide the accumulated wrench, preventing redundant
    co-located lattice contacts from compounding into a launch impulse.

    When ``cslc_owns_lattice_wrench != 0``, the lattice-side ``body_delta``
    write is suppressed: the body wrench from pad/object contact is
    instead produced by the CSLC anchor reactions
    (:func:`accumulate_cslc_body_wrench` in
    ``newton._src.solvers.uxpbd.compliant_lattice``). The SM-rigid /
    object-side ``particle_delta`` write is unaffected -- the ball
    still gets pushed away from the (compressed) lattice spheres. The
    pad gets pushed back by integrated CSLC anchor reactions instead
    of by raw position-constraint resolution; this is what makes the
    spheres visibly deform under load instead of the body absorbing
    the entire overlap.
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
        # Same lattice host or same articulation -> skip.
        # The articulation guard suppresses cross-link lattice overlaps for
        # URDF-loaded robots; -1 (free body) is treated as a unique
        # articulation so two free lattice hosts still collide.
        if is_lat_i and is_lat_j:
            host_i = lattice_link[particle_to_lattice[i]]
            host_j = lattice_link[particle_to_lattice[index]]
            if host_i == host_j:
                continue
            art_i = body_articulation[host_i]
            art_j = body_articulation[host_j]
            if art_i >= 0 and art_i == art_j:
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
        if cslc_owns_lattice_wrench == 0:
            host_i = lattice_link[particle_to_lattice[i]]
            wp.atomic_add(body_delta, host_i, wp.spatial_vector(body_delta_lin, body_delta_ang))
            wp.atomic_add(body_contact_count, host_i, body_contact_n)
        # else: CSLC anchor-reaction kernel writes the lattice body wrench.
    else:
        wp.atomic_add(particle_deltas, i, delta_acc)


@wp.kernel
def apply_particle_deltas_uxpbd(
    x_orig: wp.array[wp.vec3],
    x_pred: wp.array[wp.vec3],
    v_pred: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[wp.float32],
    delta: wp.array[wp.vec3],
    dt: float,
    v_max: float,
    # output
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
):
    """UXPBD particle-delta apply that preserves v for mass-0 particles.

    Like :func:`newton._src.solvers.srxpbd.kernels.apply_particle_deltas` but
    passes through ``v_pred`` (instead of zeroing it) for mass==0 particles.

    Rationale: in UXPBD, lattice spheres have ``particle_mass == 0`` because
    their inertia is carried by the host body, not the particle. Their
    ``particle_qd`` is the projected body-frame point velocity
    ``v_lin + omega x r`` (computed by :func:`update_lattice_world_positions`).
    The SRXPBD apply kernel writes ``v_out = 0`` for mass-0 particles, which
    destroys this body-projected velocity and forces the solver to re-run
    :func:`update_lattice_world_positions` after every particle-delta apply
    just to restore qd. By passing through ``v_pred`` here, lattice qd is
    preserved across apply calls and the post-apply projection is no longer
    needed (only post-apply-body-deltas projections remain).

    For mass==0 particles whose v_pred is genuinely zero (true statics),
    behavior is unchanged: v_out == v_pred == 0.

    See :mod:`newton._src.solvers.uxpbd` design notes on lattice-qd
    preservation.
    """
    tid = wp.tid()

    if particle_mass[tid] == 0.0:
        # Mass-0 particle: preserve position and velocity from prediction.
        # For lattice particles, v_pred is the body-projected velocity from
        # update_lattice_world_positions and must not be overwritten.
        x_out[tid] = x_pred[tid]
        v_out[tid] = v_pred[tid]
        return

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return

    x0 = x_orig[tid]
    xp = x_pred[tid]
    vp = v_pred[tid]
    d = delta[tid]

    # See srxpbd.apply_particle_deltas docstring for the v_new = vp + d/dt
    # rationale (long-horizon accuracy vs (x_new - x0)/dt).
    v_new = vp + d / dt
    x_new = xp + d

    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    x_out[tid] = x_new
    v_out[tid] = v_new


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
