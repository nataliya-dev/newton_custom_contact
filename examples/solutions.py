import numpy as np
import warp as wp


def push_object_on_ground_solution(x0, rot0, m, mu, F, t):
    # Assumes no tipping and no rotation, so the force is purely planar and the box does not rotate.
    assert np.count_nonzero(F) == 1 and F[2] == 0.0
    x0_np = np.asarray(x0, dtype=np.float32)
    planar_F = np.asarray(F, dtype=np.float32)
    F_planar_mag = np.linalg.norm(planar_F)
    g = 9.81
    f_fric_max = float(mu) * m * g

    if F_planar_mag <= f_fric_max:
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        dir_planar = planar_F / F_planar_mag
        net_planar = planar_F - f_fric_max * dir_planar
        a = (net_planar / m).astype(np.float32)
    pos = x0_np + 0.5 * a * (t * t)
    vel = a * t
    omega_analytic = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    L_analytic = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p_analytic = (float(m) * vel).astype(np.float32)
    return np.asarray(pos), np.asarray(vel), np.asarray(rot0), omega_analytic, L_analytic, p_analytic


def obj_on_slope_solution(x0, rot0, mu, t, slope_rot, m):
    # Assumes no tipping and no rotation, so the force is purely planar and the box does not rotate.
    x0_np = np.asarray(x0, dtype=np.float32)
    rot0_np = np.asarray(rot0, dtype=np.float32)
    g = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    local_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    q = np.asarray(slope_rot, dtype=np.float32).ravel()
    q_vec = q[:3]
    w = q[3]
    t_vec = 2.0 * np.cross(q_vec, local_z)
    n = local_z + w * t_vec + np.cross(q_vec, t_vec)

    g_dot_n = float(np.dot(g, n))
    a_t = g - n * g_dot_n

    a_t_mag = float(np.linalg.norm(a_t))
    friction_acc = float(mu) * abs(g_dot_n)

    if a_t_mag == 0.0 or friction_acc >= a_t_mag:
        a_net = np.zeros(3, dtype=np.float32)
    else:
        tan_hat = a_t / a_t_mag
        a_net = (a_t - tan_hat * friction_acc).astype(np.float32)

    pos = x0_np + 0.5 * a_net * (t * t)
    vel = a_net * t

    omega_analytic = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    L_analytic = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    p_analytic = (float(m) * vel).astype(np.float32)

    return np.asarray(pos), np.asarray(vel), np.asarray(rot0_np), omega_analytic, L_analytic, p_analytic


def translate_and_rotate_on_ground_special_I_iso(x0, rot0, I, mu, m, F, tau, t):
    v0 = wp.vec3(0.0, 0.0, 0.0)
    w0 = wp.vec3(0.0, 0.0, 0.0)

    g = 9.81
    N = m * g

    # -------------------------------------------------
    # --- TRANSLATION (with Coulomb friction)
    # -------------------------------------------------

    # Remove vertical motion (ground constraint)
    F_planar = wp.vec3(F[0], F[1], 0.0)

    F_mag = wp.length(F_planar)
    f_fric_max = float(mu) * m * g

    if F_mag <= f_fric_max:
        a = wp.vec3(0.0, 0.0, 0.0)
    else:
        F_dir = F_planar * (1.0 / F_mag)

        # kinetic friction
        F_fric = -mu * N * F_dir

        F_net = F_planar + F_fric

        a = F_net * (1.0 / m)

    pos = x0 + v0 * t + 0.5 * a * (t * t)
    vel = v0 + a * t

    # constrain to ground
    pos = wp.vec3(pos[0], pos[1], x0[2])
    vel = wp.vec3(vel[0], vel[1], 0.0)

    # -------------------------------------------------
    # --- ROTATION (isotropic inertia)
    # -------------------------------------------------

    lam = (I[0, 0] + I[1, 1] + I[2, 2]) / 3.0

    alpha = tau * (1.0 / lam)
    w = w0 + alpha * t

    L = I @ w

    alpha_mag = wp.length(alpha)

    if alpha_mag < 1.0e-12:
        rot = rot0
    else:
        axis = alpha * (1.0 / alpha_mag)

        # theta(t) = 0.5 |alpha| t^2
        angle = 0.5 * alpha_mag * (t * t)

        half = 0.5 * angle
        s = wp.sin(half)

        dq = wp.quat(
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            wp.cos(half)
        )

        rot = wp.normalize(dq * rot0)

    p = np.asarray(m * vel)

    return (
        np.asarray(pos),
        np.asarray(vel),
        np.asarray(rot),
        np.asarray(w),
        np.asarray(L),
        p,
    )


def translate_and_rotate_on_ground_general_I_constrained_ground(x0, rot0, I, m, mu, F, tau, t):
    v0 = wp.vec3(0.0, 0.0, 0.0)
    w0 = wp.vec3(0.0, 0.0, 0.0)

    g = 9.81
    N = m * g

    # -------------------------------------------------
    # --- TRANSLATION (planar, with Coulomb friction)
    # -------------------------------------------------

    # Remove vertical component (ground constraint)
    F_planar = wp.vec3(F[0], F[1], 0.0)

    F_mag = wp.length(F_planar)

    f_fric_max = float(mu) * m * g

    if F_mag <= f_fric_max:
        # No horizontal force
        a = wp.vec3(0.0, 0.0, 0.0)
    else:
        # kinetic friction opposes motion direction
        F_dir = F_planar * (1.0 / F_mag)

        F_fric = -mu * N * F_dir

        F_net = F_planar + F_fric

        a = F_net * (1.0 / m)

    # Position and velocity
    pos = x0 + v0 * t + 0.5 * a * (t * t)
    vel = v0 + a * t

    # Constrain to ground
    pos = wp.vec3(pos[0], pos[1], x0[2])
    vel = wp.vec3(vel[0], vel[1], 0.0)

    # -------------------------------------------------
    # --- ROTATION (yaw only)
    # -------------------------------------------------

    # Only z torque contributes (roll/pitch reacted by ground)
    tau_z = tau[2]

    Izz = I[2, 2]

    alpha_z = tau_z / Izz

    w = wp.vec3(0.0, 0.0, alpha_z * t)

    L = I @ w

    alpha_mag = wp.abs(alpha_z)

    if alpha_mag < 1.0e-12:
        rot = rot0
    else:
        # rotation about world z
        axis = wp.vec3(0.0, 0.0, 1.0)

        # theta(t) = 0.5 * alpha * t^2
        angle = 0.5 * alpha_z * (t * t)

        half = 0.5 * angle
        s = wp.sin(half)

        dq = wp.quat(
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            wp.cos(half)
        )

        rot = wp.normalize(dq * rot0)

    p = np.asarray(m * vel)  # Linear momentum

    return (
        np.asarray(pos),
        np.asarray(vel),
        np.asarray(rot),
        np.asarray(w),
        np.asarray(L),
        p,
    )


def push_t_rod_solution(x0_rod, x0_tblock, m_rod, m_tblock, rod_force, t):
    raise NotImplementedError()



class QuasistaticPushSimulator:
    """
    Quasistatic planar push based on Lynch & Mason (1996).
    Ellipsoidal limit surface approximation.
    """

    def __init__(
        self,
        half_width,
        half_height,
        mu_push,
        l_squared,
        contact_normal,
        com_offset_x=0.0,
    ):
        self.half_width = half_width
        self.half_height = half_height
        self.friction_half_angle = np.arctan(mu_push)
        self.l_squared = l_squared
        self.contact_normal = contact_normal
        self.com_offset_x = com_offset_x

    def step(self, slider_config, pusher_pos, pusher_vel, dt):
        """Returns (new_config, in_contact). All 2D: config=(x,y,theta), pos/vel=(x,y)."""
        theta = slider_config[2]
        c, s = np.cos(-theta), np.sin(-theta)

        # Contact point in slider frame
        dx = pusher_pos[0] - slider_config[0]
        dy = pusher_pos[1] - slider_config[1]
        cx = c * dx - s * dy
        cy = s * dx + c * dy

        if abs(cx) > self.half_width or abs(cy) > self.half_height:
            return slider_config.copy(), False

        # Friction cone boundary forces (rotate contact normal by ±alpha)
        alpha = self.friction_half_angle
        n = self.contact_normal[:2]
        ca, sa = np.cos(alpha), np.sin(alpha)
        fb1 = np.array([ca * n[0] - sa * n[1], sa * n[0] + ca * n[1]])
        fb2 = np.array([ca * n[0] + sa * n[1], -sa * n[0] + ca * n[1]])

        # Boundary contact-point velocities via limit surface
        m1 = cx * fb1[1] - cy * fb1[0]
        m2 = cx * fb2[1] - cy * fb2[0]
        vb1 = np.array(
            [self.l_squared * fb1[0] - m1 * cy, self.l_squared * fb1[1] + m1 * cx]
        )
        vb2 = np.array(
            [self.l_squared * fb2[0] - m2 * cy, self.l_squared * fb2[1] + m2 * cx]
        )

        # Push direction in slider frame
        ux = c * pusher_vel[0] - s * pusher_vel[1]
        uy = s * pusher_vel[0] + c * pusher_vel[1]
        norm_u = np.sqrt(ux**2 + uy**2)
        if norm_u < 1e-12:
            return slider_config.copy(), True

        # Sticking vs sliding (angle comparison)
        nb1, nb2 = np.linalg.norm(vb1), np.linalg.norm(vb2)
        cos_pb1 = (vb1[0] * ux + vb1[1] * uy) / (nb1 * norm_u)
        cos_pb2 = (vb2[0] * ux + vb2[1] * uy) / (nb2 * norm_u)
        cos_b12 = (vb1[0] * vb2[0] + vb1[1] * vb2[1]) / (nb1 * nb2)

        if cos_b12 < cos_pb1 and cos_b12 < cos_pb2:
            vp_x, vp_y = ux, uy  # sticking
        else:
            vb = vb1 if cos_pb1 <= cos_pb2 else vb2  # sliding along exceeded boundary
            n2d = self.contact_normal[:2]
            kappa = (n2d[0] * ux + n2d[1] * uy) / (n2d[0] * vb[0] + n2d[1] * vb[1])
            vp_x, vp_y = kappa * vb[0], kappa * vb[1]

        # Inverse mapping: contact-point velocity → slider velocity
        D = self.l_squared + cx**2 + cy**2
        tx = ((self.l_squared + cx**2) * vp_x + cx * cy * vp_y) / D
        ty = (cx * cy * vp_x + (self.l_squared + cy**2) * vp_y) / D
        omega = (cx * ty - cy * tx) / self.l_squared
        ty_adj = ty - omega * self.com_offset_x

        # Semi-implicit Euler integration
        theta_new = theta + omega * dt
        ct, st = np.cos(theta_new), np.sin(theta_new)
        x_new = slider_config[0] + (tx * ct - ty_adj * st) * dt
        y_new = slider_config[1] + (tx * st + ty_adj * ct) * dt

        return np.array([x_new, y_new, theta_new]), True


class PushBoxRodSolution:
    def __init__(self, x0_rod, x0_box, m_rod, rod_force, box_hx, box_hy, mu_push):
        l_squared = (box_hx**2 + box_hy**2) / 3.0
        self.sim = QuasistaticPushSimulator(
            half_width=box_hx,
            half_height=box_hy,
            mu_push=mu_push,
            l_squared=l_squared,
            contact_normal=np.array([0.0, -1.0]),
        )
        self.box_config = np.array([float(x0_box[0]), float(x0_box[1]), 0.0])
        self.x0_box_z = float(x0_box[2])
        self.rod_pos_2d = np.array([float(x0_rod[0]), float(x0_rod[1])])
        self.a_rod = np.array([float(rod_force[0]), float(rod_force[1])]) / m_rod
        self.current_t = 0.0

    def step(self, dt):
        rod_vel_2d = self.a_rod * self.current_t
        rod_pos_now = self.rod_pos_2d + 0.5 * self.a_rod * self.current_t**2
        new_config, in_contact = self.sim.step(
            self.box_config, rod_pos_now, rod_vel_2d, dt
        )

        prev_config = self.box_config.copy()
        if in_contact:
            self.box_config = new_config
        self.current_t += dt

        theta = self.box_config[2]
        pos_box = np.array(
            [self.box_config[0], self.box_config[1], self.x0_box_z], dtype=np.float32
        )
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(theta))

        # Instantaneous velocity from finite difference
        vel_2d = (self.box_config[:2] - prev_config[:2]) / dt
        vel_box = np.array([vel_2d[0], vel_2d[1], 0.0], dtype=np.float32)

        # Instantaneous omega from finite difference
        omega_z = (self.box_config[2] - prev_config[2]) / dt
        omega = np.array([0.0, 0.0, omega_z], dtype=np.float32)

        # Momentum (not used, quasistatic assumption)
        # For now, leave as zero
        L_analytic = np.zeros(3, dtype=np.float32)
        p_analytic = np.zeros(3, dtype=np.float32)

        return pos_box, vel_box, np.asarray(rot), omega, L_analytic, p_analytic
