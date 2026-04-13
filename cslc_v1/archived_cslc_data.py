"""
kernels_cslc.py — CSLC contact force evaluation kernels.

Ground-plane pipeline (per timestep):
  1. compute_cslc_penetration       — lattice spheres vs ground plane
  2. cslc_jacobi_iteration          — one damped Jacobi step (launched N times)
  3. accumulate_cslc_wrench         — forces on lattice parent body only

Sphere-contact pipeline (for grasp scenarios):
  1. compute_cslc_penetration_sphere — lattice spheres vs target sphere
  2. cslc_jacobi_iteration           — same Jacobi kernel (reused)
  3. accumulate_cslc_wrench_sphere   — forces on finger + reaction on object

Drop-in: newton/_src/solvers/semi_implicit/kernels_cslc.py
"""

import warp as wp


# ═══════════════════════════════════════════════════════════════════════
#  Kernel 1: Penetration query (lattice spheres vs ground plane)
# ═══════════════════════════════════════════════════════════════════════

@wp.kernel
def compute_cslc_penetration(
    # body state
    body_q: wp.array[wp.transform],
    # shape → body mapping
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    # CSLC pad data
    pad_shape_index: wp.array[int],
    sphere_pad_index: wp.array[int],
    sphere_positions: wp.array[wp.vec3],   # rest pos, body frame
    sphere_radii: wp.array[float],
    # ground plane (hardcoded z=0, normal=[0,0,1] for now)
    ground_height: float,
    # output
    pen_per_sphere: wp.array[float],       # nominal penetration (scalar along face normal)
):
    """Compute nominal penetration of each lattice sphere against the ground plane."""
    tid = wp.tid()

    pad_id = sphere_pad_index[tid]
    shape_id = pad_shape_index[pad_id]
    body_id = shape_body[shape_id]

    # Transform sphere rest position: body frame → world frame
    # sphere is in body frame (shape xform already baked in during pad creation)
    pos_body = sphere_positions[tid]
    if body_id >= 0:
        X_wb = body_q[body_id]
        pos_world = wp.transform_point(X_wb, pos_body)
    else:
        pos_world = pos_body

    r = sphere_radii[tid]

    # Penetration against ground plane z = ground_height
    # Sphere bottom at pos_world.z - r; penetration = ground_height - (pos_world.z - r)
    pen = ground_height - (pos_world[2] - r)
    pen_per_sphere[tid] = wp.max(pen, 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Kernel 2: One damped Jacobi iteration
# ═══════════════════════════════════════════════════════════════════════

@wp.kernel
def cslc_jacobi_iteration(
    # lattice topology (CSR)
    neighbor_start: wp.array[int],
    neighbor_count: wp.array[int],
    neighbor_indices: wp.array[int],
    degrees: wp.array[float],
    # current state
    delta: wp.array[float],            # scalar displacement along normal
    pen_nominal: wp.array[float],      # from kernel 1
    # spring constants
    ka: float,
    kl: float,
    kc: float,
    alpha: float,                      # damping factor
    # output (double-buffered)
    delta_out: wp.array[float],
):
    """One Jacobi iteration of the CSLC quasistatic solver.

    Matches cslc.py solve_contact_local exactly:
      pen = max(pen_nominal - delta, 0)
      f_contact = kc * pen
      L_delta = sum_{j in N(i)} (delta_i - delta_j)
      residual = f_contact - ka*delta - kl*L_delta
      delta += alpha * residual / (ka + kl*deg + kc*(pen>0))
    """
    i = wp.tid()

    d_i = delta[i]
    pen = wp.max(pen_nominal[i] - d_i, 0.0)
    f_contact = kc * pen

    # Lateral coupling: L @ delta for row i
    L_delta_i = float(0.0)
    nb_start = neighbor_start[i]
    nb_count = neighbor_count[i]
    for k in range(nb_count):
        j = neighbor_indices[nb_start + k]
        L_delta_i += d_i - delta[j]

    residual = f_contact - ka * d_i - kl * L_delta_i

    # Effective denominator (diagonal preconditioner)
    in_contact = float(0.0)
    if pen > 0.0:
        in_contact = 1.0
    eff_denom = ka + kl * degrees[i] + kc * in_contact

    delta_out[i] = d_i + alpha * residual / eff_denom


# ═══════════════════════════════════════════════════════════════════════
#  Kernel 3: Accumulate wrench from equilibrium lattice forces
# ═══════════════════════════════════════════════════════════════════════

@wp.kernel
def accumulate_cslc_wrench(
    # body state
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # shape mapping
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    # CSLC data
    pad_shape_index: wp.array[int],
    pad_face_normal: wp.array[wp.vec3],
    sphere_pad_index: wp.array[int],
    sphere_positions: wp.array[wp.vec3],
    sphere_radii: wp.array[float],
    delta: wp.array[float],
    pen_nominal: wp.array[float],
    kc: float,
    dc: float,       # Hunt-Crossley damping coefficient
    mu: float,       # friction coefficient
    friction_eps: float,
    # body velocities (for damping + friction)
    body_qd: wp.array[wp.spatial_vector],
    # ground plane
    ground_height: float,
    # output
    body_f: wp.array[wp.spatial_vector],
):
    """Convert per-sphere equilibrium forces into spatial wrench on the parent body."""
    tid = wp.tid()

    d_i = delta[tid]
    pen = wp.max(pen_nominal[tid] - d_i, 0.0)
    f_n_mag = kc * pen

    if f_n_mag < 1.0e-10:
        return

    pad_id = sphere_pad_index[tid]
    shape_id = pad_shape_index[pad_id]
    body_id = shape_body[shape_id]
    if body_id < 0:
        return  # static body, no forces to accumulate

    # Face normal in body frame → world frame
    face_n_body = pad_face_normal[pad_id]
    X_wb = body_q[body_id]
    # The face normal points OUTWARD from the shape.
    # The contact force on the object pushes it AWAY from the ground (opposite to face normal).
    # For the bottom face, face_n_body = [0,0,-1], so ground pushes in +z.
    face_n_world = wp.transform_vector(X_wb, face_n_body)
    # Contact normal: from ground toward body = -face_n_world (ground pushes body up)
    contact_normal = -face_n_world  # points from ground into body = upward

    # Sphere world position (displaced)
    pos_body = sphere_positions[tid]
    pos_world = wp.transform_point(X_wb, pos_body)
    # Contact point on the ground plane (projection)
    cp_world = wp.vec3(
        pos_world[0],
        pos_world[1],
        ground_height,
    )

    # ── Hunt & Crossley damping ──
    com_world = wp.transform_point(X_wb, body_com[body_id])
    r_cp = cp_world - com_world

    body_v_s = body_qd[body_id]
    body_v = wp.spatial_top(body_v_s)
    body_w = wp.spatial_bottom(body_v_s)
    v_body_at_cp = body_v + wp.cross(body_w, r_cp)

    v_n = wp.dot(v_body_at_cp, contact_normal)
    hc = wp.max(1.0 + dc * wp.max(0.0, -v_n), 0.0)  # approaching = negative v_n
    fn_damped = f_n_mag * hc

    # Normal force on body (pushes up = contact_normal direction)
    f_normal = contact_normal * fn_damped

    # ── Regularised Coulomb friction ──
    v_t = v_body_at_cp - contact_normal * v_n
    v_t_mag = wp.length(v_t)
    f_friction = wp.vec3(0.0)
    if mu > 0.0 and v_t_mag > 1.0e-12:
        t_hat = v_t / v_t_mag
        fric_scale = v_t_mag / (v_t_mag + friction_eps)
        f_friction = -mu * fn_damped * fric_scale * t_hat

    f_total = f_normal + f_friction

    # Accumulate wrench (body frame convention: sub for consistency with eval_body_contact)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(f_total, wp.cross(r_cp, f_total)))


# ═══════════════════════════════════════════════════════════════════════
#  Python orchestrator
# ═══════════════════════════════════════════════════════════════════════

def eval_cslc_contact_forces(
    model,
    state,
    cslc_data,
    body_f_out=None,
    ground_height: float = 0.0,
    dc: float = 2.0,
    mu: float = 0.0,
    friction_eps: float = 1e-3,
    n_iter: int = 40,
    alpha: float = 0.3,
):
    """
    Evaluate CSLC distributed contact forces.

    Args:
        model: Newton Model (need shape_body, shape_transform, body_com).
        state: Newton State (need body_q, body_qd, body_f).
        cslc_data: CSLCData with lattice pad info.
        body_f_out: Output force array. If None, uses state.body_f.
        ground_height: Z-height of the ground plane.
        dc: Hunt-Crossley damping coefficient.
        mu: Coulomb friction coefficient.
        friction_eps: Friction regularization epsilon.
        n_iter: Number of Jacobi iterations.
        alpha: Jacobi damping factor.
    """
    if cslc_data.n_spheres_total == 0:
        return

    if body_f_out is None:
        body_f_out = state.body_f

    device = model.device
    ns = cslc_data.n_spheres_total

    # Allocate scratch buffers; warm-start from previous timestep's solution
    pen_nominal = wp.zeros(ns, dtype=wp.float32, device=device)
    delta_a = wp.clone(cslc_data.sphere_delta)
    delta_b = wp.zeros(ns, dtype=wp.float32, device=device)

    # ── Stage 1: penetration query ──
    wp.launch(
        compute_cslc_penetration,
        dim=ns,
        inputs=[
            state.body_q,
            model.shape_body,
            model.shape_transform,
            cslc_data.pad_shape_index,
            cslc_data.sphere_pad_index,
            cslc_data.sphere_positions,
            cslc_data.sphere_radii,
            ground_height,
        ],
        outputs=[pen_nominal],
        device=device,
    )

    # ── Stage 2: Jacobi iterations (double-buffered) ──
    src, dst = delta_a, delta_b
    for _ in range(n_iter):
        wp.launch(
            cslc_jacobi_iteration,
            dim=ns,
            inputs=[
                cslc_data.neighbor_start,
                cslc_data.neighbor_count,
                cslc_data.neighbor_indices,
                cslc_data.degrees,
                src,
                pen_nominal,
                cslc_data.ka,
                cslc_data.kl,
                cslc_data.kc,
                alpha,
            ],
            outputs=[dst],
            device=device,
        )
        src, dst = dst, src
    # After loop, `src` holds the latest delta
    # Persist for warm-starting the next timestep
    wp.copy(cslc_data.sphere_delta, src)

    # ── Stage 3: wrench accumulation ──
    wp.launch(
        accumulate_cslc_wrench,
        dim=ns,
        inputs=[
            state.body_q,
            model.body_com,
            model.shape_body,
            model.shape_transform,
            cslc_data.pad_shape_index,
            cslc_data.pad_face_normal,
            cslc_data.sphere_pad_index,
            cslc_data.sphere_positions,
            cslc_data.sphere_radii,
            src,           # final delta
            pen_nominal,
            cslc_data.kc,
            dc,
            mu,
            friction_eps,
            state.body_qd,
            ground_height,
        ],
        outputs=[body_f_out],
        device=device,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Sphere-vs-sphere contact kernels (for grasp scenarios)
# ═══════════════════════════════════════════════════════════════════════

@wp.kernel
def compute_cslc_penetration_sphere(
    # body state
    body_q: wp.array[wp.transform],
    # shape → body mapping
    shape_body: wp.array[int],
    # CSLC pad data
    pad_shape_index: wp.array[int],
    sphere_pad_index: wp.array[int],
    sphere_positions: wp.array[wp.vec3],
    sphere_radii: wp.array[float],
    # target sphere (looked up from model arrays)
    target_body_id: int,
    target_shape_pos: wp.vec3,   # shape-local position (usually origin)
    target_radius: float,
    # output
    pen_per_sphere: wp.array[float],
    contact_normals: wp.array[wp.vec3],
):
    """Compute penetration of each lattice sphere against a target sphere.

    Contact normal points from target center toward lattice sphere (outward from object).
    """
    tid = wp.tid()

    pad_id = sphere_pad_index[tid]
    shape_id = pad_shape_index[pad_id]
    body_id = shape_body[shape_id]

    # Lattice sphere → world
    pos_body = sphere_positions[tid]
    if body_id >= 0:
        X_wb = body_q[body_id]
        pos_world = wp.transform_point(X_wb, pos_body)
    else:
        pos_world = pos_body

    r_lattice = sphere_radii[tid]

    # Target sphere → world
    if target_body_id >= 0:
        X_wb_target = body_q[target_body_id]
        target_pos = wp.transform_point(X_wb_target, target_shape_pos)
    else:
        target_pos = target_shape_pos

    # Sphere-sphere penetration: φ = (r_i + r_j) - ||q_i - p_j||
    diff = pos_world - target_pos
    dist = wp.length(diff)
    pen = (r_lattice + target_radius) - dist
    pen_per_sphere[tid] = wp.max(pen, 0.0)

    # Contact normal: from target toward lattice sphere
    if dist > 1.0e-10:
        contact_normals[tid] = diff / dist
    else:
        contact_normals[tid] = wp.vec3(0.0, 0.0, 1.0)


@wp.kernel
def accumulate_cslc_wrench_sphere(
    # body state
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # shape mapping
    shape_body: wp.array[int],
    # CSLC data
    pad_shape_index: wp.array[int],
    sphere_pad_index: wp.array[int],
    sphere_positions: wp.array[wp.vec3],
    sphere_radii: wp.array[float],
    delta: wp.array[float],
    pen_nominal: wp.array[float],
    contact_normals: wp.array[wp.vec3],
    kc: float,
    dc: float,
    mu: float,
    friction_eps: float,
    # body velocities
    body_qd: wp.array[wp.spatial_vector],
    # target sphere
    target_body_id: int,
    target_shape_pos: wp.vec3,
    target_radius: float,
    # output
    body_f: wp.array[wp.spatial_vector],
):
    """Convert equilibrium forces to wrench on finger AND reaction on object.

    Unlike the ground-plane version, this kernel:
      - Uses per-sphere contact normals (radial, not uniform)
      - Uses relative velocity (finger - object) for damping
      - Applies equal-opposite wrench on the target (object) body
    """
    tid = wp.tid()

    d_i = delta[tid]
    pen = wp.max(pen_nominal[tid] - d_i, 0.0)
    f_n_mag = kc * pen

    if f_n_mag < 1.0e-10:
        return

    pad_id = sphere_pad_index[tid]
    shape_id = pad_shape_index[pad_id]
    finger_body_id = shape_body[shape_id]

    contact_normal = contact_normals[tid]  # from object toward finger

    # Lattice sphere world position
    pos_world = sphere_positions[tid]  # body frame
    if finger_body_id >= 0:
        X_wb_finger = body_q[finger_body_id]
        pos_world = wp.transform_point(X_wb_finger, sphere_positions[tid])

    # Target sphere world position
    if target_body_id >= 0:
        X_wb_target = body_q[target_body_id]
        target_pos = wp.transform_point(X_wb_target, target_shape_pos)
    else:
        target_pos = target_shape_pos

    # Contact point: on the surface of the target sphere
    cp_world = target_pos + contact_normal * target_radius

    # ── Relative velocity at contact point ──
    # Finger velocity at contact point
    v_finger_at_cp = wp.vec3(0.0)
    if finger_body_id >= 0:
        com_finger = wp.transform_point(body_q[finger_body_id], body_com[finger_body_id])
        r_cp_finger = cp_world - com_finger
        finger_v_s = body_qd[finger_body_id]
        finger_v = wp.spatial_top(finger_v_s)
        finger_w = wp.spatial_bottom(finger_v_s)
        v_finger_at_cp = finger_v + wp.cross(finger_w, r_cp_finger)

    # Object velocity at contact point
    v_object_at_cp = wp.vec3(0.0)
    if target_body_id >= 0:
        com_object = wp.transform_point(body_q[target_body_id], body_com[target_body_id])
        r_cp_object = cp_world - com_object
        object_v_s = body_qd[target_body_id]
        object_v = wp.spatial_top(object_v_s)
        object_w = wp.spatial_bottom(object_v_s)
        v_object_at_cp = object_v + wp.cross(object_w, r_cp_object)

    # Relative velocity (finger - object), projected onto normal
    v_rel = v_finger_at_cp - v_object_at_cp
    v_n = wp.dot(v_rel, contact_normal)

    # ── Hunt & Crossley damping ──
    # v_n < 0 means approaching (finger moving toward object)
    hc = wp.max(1.0 + dc * wp.max(0.0, -v_n), 0.0)
    fn_damped = f_n_mag * hc

    f_normal = contact_normal * fn_damped

    # ── Regularised Coulomb friction ──
    v_t = v_rel - contact_normal * v_n
    v_t_mag = wp.length(v_t)
    f_friction = wp.vec3(0.0)
    if mu > 0.0 and v_t_mag > 1.0e-12:
        t_hat = v_t / v_t_mag
        fric_scale = v_t_mag / (v_t_mag + friction_eps)
        f_friction = -mu * fn_damped * fric_scale * t_hat

    # Total force on finger (pushes finger away from object)
    f_total = f_normal + f_friction

    # ── Accumulate wrench on FINGER body ──
    if finger_body_id >= 0:
        com_finger = wp.transform_point(body_q[finger_body_id], body_com[finger_body_id])
        r_f = cp_world - com_finger
        wp.atomic_add(body_f, finger_body_id,
                      wp.spatial_vector(f_total, wp.cross(r_f, f_total)))

    # ── Reaction wrench on OBJECT body (Newton's 3rd law) ──
    if target_body_id >= 0:
        com_object = wp.transform_point(body_q[target_body_id], body_com[target_body_id])
        r_o = cp_world - com_object
        f_reaction = -f_total
        wp.atomic_add(body_f, target_body_id,
                      wp.spatial_vector(f_reaction, wp.cross(r_o, f_reaction)))


# ═══════════════════════════════════════════════════════════════════════
#  Sphere-contact orchestrator
# ═══════════════════════════════════════════════════════════════════════

def eval_cslc_sphere_contact_forces(
    model,
    state,
    cslc_data,
    target_body_id: int,
    target_shape_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_radius: float = 0.05,
    body_f_out=None,
    dc: float = 2.0,
    mu: float = 0.0,
    friction_eps: float = 1e-3,
    n_iter: int = 40,
    alpha: float = 0.3,
):
    """
    Evaluate CSLC distributed contact forces against a target sphere.

    Lattice spheres on finger pads contact a single target sphere (the grasped
    object).  Forces are accumulated on both finger bodies and the object body.

    Args:
        model: Newton Model.
        state: Newton State.
        cslc_data: CSLCData with lattice pad info.
        target_body_id: Body index of the target sphere.
        target_shape_pos: Shape-local position of the target sphere center
            (usually (0,0,0) if shape is centered on body origin).
        target_radius: Radius of the target sphere [m].
        body_f_out: Output force array. If None, uses state.body_f.
        dc: Hunt-Crossley damping coefficient.
        mu: Coulomb friction coefficient.
        friction_eps: Friction regularization epsilon.
        n_iter: Number of Jacobi iterations.
        alpha: Jacobi damping factor.
    """
    if cslc_data.n_spheres_total == 0:
        return

    if body_f_out is None:
        body_f_out = state.body_f

    device = model.device
    ns = cslc_data.n_spheres_total

    # Scratch buffers; warm-start delta from previous timestep
    pen_nominal = wp.zeros(ns, dtype=wp.float32, device=device)
    contact_normals = wp.zeros(ns, dtype=wp.vec3, device=device)
    delta_a = wp.clone(cslc_data.sphere_delta)
    delta_b = wp.zeros(ns, dtype=wp.float32, device=device)

    target_pos_wp = wp.vec3(*target_shape_pos)

    # ── Stage 1: penetration query (sphere vs sphere) ──
    wp.launch(
        compute_cslc_penetration_sphere,
        dim=ns,
        inputs=[
            state.body_q,
            model.shape_body,
            cslc_data.pad_shape_index,
            cslc_data.sphere_pad_index,
            cslc_data.sphere_positions,
            cslc_data.sphere_radii,
            target_body_id,
            target_pos_wp,
            target_radius,
        ],
        outputs=[pen_nominal, contact_normals],
        device=device,
    )

    # ── Stage 2: Jacobi iterations (unchanged — works on scalars) ──
    src, dst = delta_a, delta_b
    for _ in range(n_iter):
        wp.launch(
            cslc_jacobi_iteration,
            dim=ns,
            inputs=[
                cslc_data.neighbor_start,
                cslc_data.neighbor_count,
                cslc_data.neighbor_indices,
                cslc_data.degrees,
                src,
                pen_nominal,
                cslc_data.ka,
                cslc_data.kl,
                cslc_data.kc,
                alpha,
            ],
            outputs=[dst],
            device=device,
        )
        src, dst = dst, src
    # Persist for warm-starting
    wp.copy(cslc_data.sphere_delta, src)

    # ── Stage 3: wrench accumulation (sphere version) ──
    wp.launch(
        accumulate_cslc_wrench_sphere,
        dim=ns,
        inputs=[
            state.body_q,
            model.body_com,
            model.shape_body,
            cslc_data.pad_shape_index,
            cslc_data.sphere_pad_index,
            cslc_data.sphere_positions,
            cslc_data.sphere_radii,
            src,
            pen_nominal,
            contact_normals,
            cslc_data.kc,
            dc,
            mu,
            friction_eps,
            state.body_qd,
            target_body_id,
            target_pos_wp,
            target_radius,
        ],
        outputs=[body_f_out],
        device=device,
    )