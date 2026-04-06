"""
compare_tilt.py — Side-by-side comparison: Baseline (Newton MuJoCo) vs CSLC.

Same scene, same perturbation, same solver. Only difference: how
finger–box contact is computed.

Produces: tilt_comparison.png
"""

import sys, os
import numpy as np
import warp as wp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import newton
import newton.examples

sys.path.insert(0, os.path.dirname(__file__))
from cslc import solve_contact_local
from sphere_body import create_finger_pad
from contact_pipeline import penetration_against_flat_face, project_onto_face
from utilities import quat_to_rotmat
from config import K_ANCHOR, K_LATERAL, K_CONTACT


# ═══════════════════════════════════════════════════════════════════════
#  CSLC contact model (same as before, cleaned up)
# ═══════════════════════════════════════════════════════════════════════

class CSLCContact:
    def __init__(self, pad_ny, pad_nz, spacing, radius_factor,
                 ka, kl, kc, dc, mu, friction_eps, n_iter, alpha):
        self.pad = create_finger_pad(pad_ny, pad_nz, spacing, radius_factor)
        self.ka, self.kl, self.kc = ka, kl, kc
        self.dc, self.mu, self.friction_eps = dc, mu, friction_eps
        self.n_iter, self.alpha = n_iter, alpha
        self.ns = self.pad['n_spheres']
        self.radii = self.pad['radii']
        self.pad_local = self.pad['positions_3d'].copy()
        self.prev_delta = None

    def compute_wrench(self, finger_pos_3, box_com, box_quat_wxyz,
                       box_vel, box_omega, face_axis, face_sign,
                       box_halfs, finger_radius):
        local = self.pad_local
        sphere_pos = np.zeros_like(local)
        if face_axis == 1:
            sphere_pos[:, 0] = local[:, 1] + finger_pos_3[0]
            sphere_pos[:, 2] = local[:, 2] + finger_pos_3[2]
            inner_y = finger_pos_3[1] - face_sign * finger_radius
            sphere_pos[:, 1] = inner_y
        else:
            raise NotImplementedError

        R = quat_to_rotmat(box_quat_wxyz)
        local_normal = np.zeros(3)
        local_normal[face_axis] = float(face_sign)
        face_normal = R @ local_normal
        local_face_pt = np.zeros(3)
        local_face_pt[face_axis] = face_sign * box_halfs[face_axis]
        face_point = R @ local_face_pt + box_com

        pen, signed_dist = penetration_against_flat_face(
            sphere_pos, self.radii, face_point, face_normal)

        sol = solve_contact_local(
            self.pad, self.ka, self.kl, self.kc, pen,
            n_iter=self.n_iter, alpha=self.alpha,
            warm_start=self.prev_delta)
        self.prev_delta = sol['delta_x']

        contact_pts = project_onto_face(sphere_pos, signed_dist, face_normal)
        r_from_com = contact_pts - box_com
        lattice_forces = sol['forces']

        v_obj_at_cp = box_vel + np.cross(box_omega, r_from_com)
        v_rel = -v_obj_at_cp
        v_n_scalar = np.einsum('ij,j->i', v_rel, face_normal)
        hc = np.maximum(1.0 + self.dc * np.maximum(0.0, -v_n_scalar), 0.0)
        fn_damped = lattice_forces * hc

        f_normal = -fn_damped[:, None] * face_normal

        v_t = v_rel - v_n_scalar[:, None] * face_normal
        v_t_mag = np.linalg.norm(v_t, axis=1)
        v_t_safe = np.maximum(v_t_mag, 1e-12)
        t_hat = v_t / v_t_safe[:, None]
        fric_scale = v_t_mag / (v_t_mag + self.friction_eps)
        f_friction = self.mu * fn_damped[:, None] * fric_scale[:, None] * t_hat

        f_per = f_normal + f_friction
        active = lattice_forces > 1e-10
        f_per[~active] = 0.0

        f_total = np.sum(f_per, axis=0)
        tau_total = np.sum(np.cross(r_from_com, f_per), axis=0)

        n_active = int(np.sum(active))
        return f_total, tau_total, {
            'n_contacts': n_active,
            'total_normal': float(np.sum(fn_damped[active])) if n_active else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Shared scene parameters
# ═══════════════════════════════════════════════════════════════════════

BOX_HX, BOX_HY, BOX_HZ = 0.05, 0.05, 0.1
BOX_HALFS = np.array([BOX_HX, BOX_HY, BOX_HZ])
BOX_MASS = 1.0
BOX_POS = wp.vec3(0.0, 0.0, 1.0)
SPHERE_R = 0.04
GRASP_FORCE = 30.0
MU = 1.0
DC = 2.0

# Timing
FPS = 100
SUBSTEPS = 10
FRAME_DT = 1.0 / FPS
SIM_DT = FRAME_DT / SUBSTEPS

# Perturbation
PERTURB_TIME = 0.1
PERTURB_AXIS = 0
PERTURB_OMEGA = 3.0
MAX_TIME = 2.0

# ── Equilibrium penetration for CSLC ──
# At equilibrium: grasp_force ≈ n_spheres * kc * pen_eq
# Solve: pen_eq = grasp_force / (n_spheres * kc)
# With lateral coupling this is approximate, but close enough for init.
N_SPHERES = 121
PEN_EQ = GRASP_FORCE / (N_SPHERES * K_CONTACT)  # ≈ 0.124mm


# ═══════════════════════════════════════════════════════════════════════
#  Build scene
# ═══════════════════════════════════════════════════════════════════════

def build_scene(use_cslc):
    """Build the grasp scene. If use_cslc, filter finger-box contacts."""
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_ground_plane()

    body_box = builder.add_body(
        xform=wp.transform(p=BOX_POS, q=wp.quat_identity()),
        mass=BOX_MASS, label="box",
    )
    shape_box = builder.add_shape_box(body_box, hx=BOX_HX, hy=BOX_HY, hz=BOX_HZ)

    # Finger offset: for CSLC, place fingers so initial penetration ≈ PEN_EQ
    # For baseline, place at just-touching (Newton's contact handles the rest)
    if use_cslc:
        # Pad inner face should be PEN_EQ past the box face
        # inner_face_y = finger_y - SPHERE_R
        # We want inner_face_y = BOX_HY - PEN_EQ (slightly inside box)
        # → finger_y = BOX_HY - PEN_EQ + SPHERE_R
        # But pad sphere radius (1.1mm) differs from Newton sphere (40mm)
        # Pad centre is at inner_y = finger_y - SPHERE_R = BOX_HY - PEN_EQ
        # Pad sphere surface at inner_y - pad_radius = BOX_HY - PEN_EQ - 0.0011
        # Box face at BOX_HY, so pen = BOX_HY - (BOX_HY - PEN_EQ - 0.0011) = PEN_EQ + pad_r
        # Hmm, need to be more careful.
        # signed_dist = (sphere_pos - face_point) · face_normal
        # For left finger: face_normal = +y, face_point.y = BOX_HY + box_com.y
        # sphere_pos.y = inner_y = finger_y - SPHERE_R
        # signed_dist = inner_y - (BOX_HY + 0) = finger_y - SPHERE_R - BOX_HY
        # pen = pad_radius - signed_dist = pad_radius - finger_y + SPHERE_R + BOX_HY
        # Want pen = PEN_EQ → finger_y = pad_radius + SPHERE_R + BOX_HY - PEN_EQ
        pad_radius = 0.002 * 0.55  # 1.1mm
        finger_y = pad_radius + SPHERE_R + BOX_HY - PEN_EQ
    else:
        finger_y = BOX_HY + SPHERE_R  # just touching

    left_pos = BOX_POS + wp.vec3(0.0, finger_y, 0.0)
    body_left = builder.add_link(
        xform=wp.transform(p=left_pos, q=wp.quat_identity()),
        mass=1.0, label="left_finger",
    )
    shape_left = builder.add_shape_sphere(body_left, radius=SPHERE_R)
    lj = builder.add_joint_prismatic(
        parent=-1, child=body_left,
        parent_xform=wp.transform(p=left_pos, q=wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    builder.add_articulation([lj])

    right_pos = BOX_POS + wp.vec3(0.0, -finger_y, 0.0)
    body_right = builder.add_link(
        xform=wp.transform(p=right_pos, q=wp.quat_identity()),
        mass=1.0, label="right_finger",
    )
    shape_right = builder.add_shape_sphere(body_right, radius=SPHERE_R)
    rj = builder.add_joint_prismatic(
        parent=-1, child=body_right,
        parent_xform=wp.transform(p=right_pos, q=wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    builder.add_articulation([rj])

    for si in range(builder.shape_count):
        builder.shape_material_mu[si] = MU
        builder.shape_material_mu_torsional[si] = MU
        builder.shape_material_mu_rolling[si] = MU

    if use_cslc:
        builder.add_shape_collision_filter_pair(shape_left, shape_box)
        builder.add_shape_collision_filter_pair(shape_right, shape_box)

    return builder, body_box, body_left, body_right


# ═══════════════════════════════════════════════════════════════════════
#  Run simulation
# ═══════════════════════════════════════════════════════════════════════

def run_sim(use_cslc, label):
    print(f"\n  Running {label}...")
    wp.init()

    builder, body_box, body_left, body_right = build_scene(use_cslc)
    model = builder.finalize()

    solver = newton.solvers.SolverMuJoCo(
        model, iterations=200, njmax=50, cone="elliptic", impratio=50.0)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    # CSLC setup
    cslc_L = cslc_R = None
    if use_cslc:
        params = dict(
            pad_ny=11, pad_nz=11, spacing=0.002, radius_factor=0.55,
            ka=K_ANCHOR, kl=K_LATERAL, kc=K_CONTACT,
            dc=DC, mu=MU, friction_eps=0.005, n_iter=60, alpha=0.30)
        cslc_L = CSLCContact(**params)
        cslc_R = CSLCContact(**params)

    history = {"t": [], "tilt_deg": [], "box_z": []}
    perturb_applied = False
    sim_time = 0.0
    n_frames = int(MAX_TIME / FRAME_DT)

    for frame in range(n_frames):
        for sub in range(SUBSTEPS):
            state_0.clear_forces()

            n_bodies = state_0.body_f.shape[0]
            f_ext = np.zeros((n_bodies, 6), dtype=np.float32)

            # Gravity on box
            f_ext[body_box, 2] = -9.81 * BOX_MASS

            if use_cslc:
                body_q = state_0.body_q.numpy()
                body_qd = state_0.body_qd.numpy()
                bq = body_q[body_box]
                box_com = bq[:3]
                box_quat = np.array([bq[6], bq[3], bq[4], bq[5]])
                bqd = body_qd[body_box]
                box_vel, box_omega = bqd[:3], bqd[3:]
                lp = body_q[body_left][:3]
                rp = body_q[body_right][:3]

                fL, tL, infoL = cslc_L.compute_wrench(
                    lp, box_com, box_quat, box_vel, box_omega,
                    1, +1, BOX_HALFS, SPHERE_R)
                fR, tR, infoR = cslc_R.compute_wrench(
                    rp, box_com, box_quat, box_vel, box_omega,
                    1, -1, BOX_HALFS, SPHERE_R)

                f_ext[body_box, :3] += (fL + fR).astype(np.float32)
                f_ext[body_box, 3:] += (tL + tR).astype(np.float32)
                f_ext[body_left, :3] = (np.array([0, -GRASP_FORCE, 0]) - fL).astype(np.float32)
                f_ext[body_right, :3] = (np.array([0, GRASP_FORCE, 0]) - fR).astype(np.float32)
            else:
                f_ext[body_left, :3] = [0, -GRASP_FORCE, 0]
                f_ext[body_right, :3] = [0, GRASP_FORCE, 0]

            # Perturbation
            if not perturb_applied and sim_time >= PERTURB_TIME:
                I_approx = BOX_MASS / 3.0 * (BOX_HY**2 + BOX_HZ**2)
                impulse = PERTURB_OMEGA * I_approx / SIM_DT
                f_ext[body_box, 3 + PERTURB_AXIS] += impulse
                perturb_applied = True
                print(f"    Perturbation at t={sim_time:.3f}s")

            state_0.body_f.assign(
                wp.array(f_ext, dtype=wp.spatial_vectorf, device="cpu"))

            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, SIM_DT)
            state_0, state_1 = state_1, state_0

        sim_time += FRAME_DT
        q = state_0.body_q.numpy()[body_box]
        tilt = np.degrees(2.0 * q[3 + PERTURB_AXIS])
        history["t"].append(sim_time)
        history["tilt_deg"].append(tilt)
        history["box_z"].append(float(q[2]))

    z = np.array(history["box_z"])
    tilt = np.array(history["tilt_deg"])
    print(f"    Peak tilt: {np.max(np.abs(tilt)):.3f}°")
    print(f"    Final tilt: {tilt[-1]:.3f}°")
    print(f"    Z drift: {abs(z[-1]-z[0])*1000:.2f} mm")

    return history


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rec_baseline = run_sim(use_cslc=False, label="Baseline (MuJoCo point contact)")
    rec_cslc = run_sim(use_cslc=True, label="CSLC (11×11 lattice)")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    t_b = np.array(rec_baseline["t"])
    t_c = np.array(rec_cslc["t"])

    ax1.plot(t_b * 1000, rec_baseline["tilt_deg"], '-', color='#d62728',
             lw=1.5, label='Point contact (MuJoCo)')
    ax1.plot(t_c * 1000, rec_cslc["tilt_deg"], '-', color='#1f77b4',
             lw=1.5, label='CSLC (11×11)')
    ax1.axhline(0, color='grey', ls=':', lw=0.5)
    ax1.axvline(PERTURB_TIME * 1000, color='grey', ls='--', lw=0.5, alpha=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Tilt angle (deg)')
    ax1.set_title('Tilt response after perturbation')
    ax1.legend()

    ax2.plot(t_b * 1000, np.array(rec_baseline["box_z"]) * 1000,
             '-', color='#d62728', lw=1.5, label='Point contact')
    ax2.plot(t_c * 1000, np.array(rec_cslc["box_z"]) * 1000,
             '-', color='#1f77b4', lw=1.5, label='CSLC')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Box z (mm)')
    ax2.set_title('Vertical position (grip stability)')
    ax2.legend()

    fig.tight_layout()
    fig.savefig('tilt_comparison.png', dpi=200, bbox_inches='tight')
    print(f"\n  → tilt_comparison.png saved")