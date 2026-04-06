"""
example_grasp_cslc.py — CSLC grasp: lattice contact replaces Newton's contact.

Same scene as the baseline, except:
  1. Finger–box contacts are DISABLED via collision filter pairs
  2. Each substep, we run the CSLC lattice solver to compute finger–box forces
  3. The CSLC wrench is injected into body_f on the box

The fingers are still Newton bodies (spheres on prismatic joints) so they
appear in the viewer and respond to the grasp force. But the contact
between finger and box is entirely handled by CSLC.

Usage:
  uv run python example_grasp_cslc.py --viewer gl --perturb 3.0
"""

import sys
import os
import numpy as np
import warp as wp
import newton
import newton.examples

# ── CSLC imports (local modules) ──
sys.path.insert(0, os.path.dirname(__file__))
from cslc import create_pad_2d, solve_contact_local
from sphere_body import create_finger_pad
from config import K_ANCHOR, K_LATERAL, K_CONTACT
from contact_pipeline import penetration_against_flat_face, project_onto_face
from utilities import quat_to_rotmat

# ═══════════════════════════════════════════════════════════════════════
#  CSLC wrench computation (CPU, NumPy)
#
#  This is called each substep. It:
#    1. Reads box pose from Newton state
#    2. Computes geometric penetration of each pad sphere against box face
#    3. Runs the CSLC lattice solver
#    4. Converts per-sphere forces to a 3D wrench on the box
#
#  The finger pads are "virtual" — their positions are computed from
#  the Newton finger body positions, but the pad spheres themselves
#  are not Newton bodies. They exist only in this computation.
# ═══════════════════════════════════════════════════════════════════════

class CSLCContact:
    """Manages CSLC contact computation for one finger pad."""

    def __init__(self, pad_ny, pad_nz, spacing, radius_factor,
                 ka, kl, kc, dc, mu, friction_eps, n_iter, alpha):
        self.pad = create_finger_pad(pad_ny, pad_nz, spacing, radius_factor)
        self.ka = ka
        self.kl = kl
        self.kc = kc
        self.dc = dc
        self.mu = mu
        self.friction_eps = friction_eps
        self.n_iter = n_iter
        self.alpha = alpha

        self.ns = self.pad['n_spheres']
        self.radii = self.pad['radii']
        self.pad_local = self.pad['positions_3d'].copy()  # (ns, 3), centred at origin

        # Warm-start state
        self.prev_delta = None

    def compute_wrench(self, finger_pos_3, box_com, box_quat_wxyz,
                        box_vel, box_omega, face_axis, face_sign,
                        box_halfs, finger_radius):
            """
            Compute the CSLC wrench on the box from this finger pad.

            The pad is created in the YZ plane (x=0, contact along x).
            For y-axis contact, we rotate: pad_x→world_y, pad_y→world_x, pad_z→world_z.
            The pad surface is placed at the finger's inner face (toward the box).
            """
            # ── Rotate pad from x-axis contact to the actual contact axis ──
            # pad['positions_3d'] has x=0 (contact dir), y,z = pad surface
            local = self.pad_local  # (ns, 3)

            if face_axis == 1:
                # y-axis contact: pad_x→y, pad_y→x, pad_z→z
                sphere_pos = np.zeros_like(local)
                sphere_pos[:, 0] = local[:, 1] + finger_pos_3[0]  # pad_y → x
                sphere_pos[:, 2] = local[:, 2] + finger_pos_3[2]  # pad_z → z
                # y: place at inner face of finger (surface facing the box)
                inner_y = finger_pos_3[1] - face_sign * finger_radius
                sphere_pos[:, 1] = inner_y
            elif face_axis == 0:
                # x-axis contact (original layout): pad_x→x, pad_y→y, pad_z→z
                sphere_pos = local.copy()
                inner_x = finger_pos_3[0] - face_sign * finger_radius
                sphere_pos[:, 0] = inner_x
                sphere_pos[:, 1] += finger_pos_3[1]
                sphere_pos[:, 2] += finger_pos_3[2]
            else:
                raise NotImplementedError(f"face_axis={face_axis} not yet supported")

            # ── Get box face geometry in world frame ──
            R = quat_to_rotmat(box_quat_wxyz)

            local_normal = np.zeros(3)
            local_normal[face_axis] = float(face_sign)
            face_normal = R @ local_normal

            local_face_pt = np.zeros(3)
            local_face_pt[face_axis] = face_sign * box_halfs[face_axis]
            face_point = R @ local_face_pt + box_com

            # ── Penetration ──
            pen, signed_dist = penetration_against_flat_face(
                sphere_pos, self.radii, face_point, face_normal)

            # ── CSLC lattice solve ──
            sol = solve_contact_local(
                self.pad, self.ka, self.kl, self.kc, pen,
                n_iter=self.n_iter, alpha=self.alpha,
                warm_start=self.prev_delta)
            self.prev_delta = sol['delta_x']

            # ── Convert to wrench (rest of method unchanged) ──
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
            info = {
                'n_contacts': n_active,
                'total_normal': float(np.sum(fn_damped[active])) if n_active else 0.0,
                'delta': sol['delta_x'],
            }
            return f_total, tau_total, info


class Example:
    """Newton example: CSLC grasp with tilt perturbation."""

    def __init__(self, viewer, args):
        # ── Timing ──
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer

        # ── Experiment parameters ──
        self.grasp_force = args.grasp_force
        self.mu = args.mu
        self.box_mass = args.mass
        self.max_time = args.max_time
        self.perturb_omega = args.perturb
        self.perturb_time = 0.1
        self.perturb_axis = 0  # tilt about x
        self._perturb_applied = False

        # ── Box geometry ──
        self.box_hx = 0.05
        self.box_hy = 0.05
        self.box_hz = 0.1
        self.box_halfs = np.array([self.box_hx, self.box_hy, self.box_hz])

        # ── Build scene ──
        builder = self._build_scene()
        self.model = builder.finalize()

        # ── Solver ──
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=200,
            njmax=50,
            cone="elliptic",
            impratio=50.0,
        )

        # ── State buffers ──
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        # ── CSLC contact models (one per finger) ──
        cslc_params = dict(
            pad_ny=11, pad_nz=11, spacing=0.002, radius_factor=0.55,
            ka=K_ANCHOR, kl=K_LATERAL, kc=K_CONTACT,
            dc=2.0, mu=self.mu, friction_eps=0.005,
            n_iter=60, alpha=0.30,
        )
        self.cslc_left = CSLCContact(**cslc_params)
        self.cslc_right = CSLCContact(**cslc_params)

        # ── Data collection ──
        self.history = {"t": [], "box_z": [], "box_quat": [],
                        "n_contacts_L": [], "n_contacts_R": [],
                        "normal_force_L": [], "normal_force_R": []}

    def _build_scene(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_ground_plane()

        # ── Box ──
        self.box_pos = wp.vec3(0.0, 0.0, 1.0)
        self.body_box = builder.add_body(
            xform=wp.transform(p=self.box_pos, q=wp.quat_identity()),
            mass=self.box_mass, label="box",
        )
        self.shape_box = builder.add_shape_box(
            self.body_box, hx=self.box_hx, hy=self.box_hy, hz=self.box_hz,
        )

        # ── Fingers ──
        sphere_r = 0.04
        finger_y = self.box_hy + sphere_r

        left_pos = self.box_pos + wp.vec3(0.0, finger_y, 0.0)
        self.body_left = builder.add_link(
            xform=wp.transform(p=left_pos, q=wp.quat_identity()),
            mass=1.0, label="left_finger",
        )
        self.shape_left = builder.add_shape_sphere(self.body_left, radius=sphere_r)
        left_joint = builder.add_joint_prismatic(
            parent=-1, child=self.body_left,
            parent_xform=wp.transform(p=left_pos, q=wp.quat_identity()),
            axis=newton.Axis.Y,
        )
        builder.add_articulation([left_joint])

        right_pos = self.box_pos + wp.vec3(0.0, -finger_y, 0.0)
        self.body_right = builder.add_link(
            xform=wp.transform(p=right_pos, q=wp.quat_identity()),
            mass=1.0, label="right_finger",
        )
        self.shape_right = builder.add_shape_sphere(self.body_right, radius=sphere_r)
        right_joint = builder.add_joint_prismatic(
            parent=-1, child=self.body_right,
            parent_xform=wp.transform(p=right_pos, q=wp.quat_identity()),
            axis=newton.Axis.Y,
        )
        builder.add_articulation([right_joint])

        # ── Friction on remaining contacts (box–ground, etc.) ──
        for shape_idx in range(builder.shape_count):
            builder.shape_material_mu[shape_idx] = self.mu
            builder.shape_material_mu_torsional[shape_idx] = self.mu
            builder.shape_material_mu_rolling[shape_idx] = self.mu

        # ── DISABLE finger–box contact in Newton's pipeline ──
        # CSLC will handle these contacts instead
        builder.add_shape_collision_filter_pair(self.shape_left, self.shape_box)
        builder.add_shape_collision_filter_pair(self.shape_right, self.shape_box)
        print(f"  Collision filters: left({self.shape_left})–box({self.shape_box}), "
              f"right({self.shape_right})–box({self.shape_box})")

        return builder

    def _compute_cslc_wrench(self):
            """
            Run CSLC for both fingers.
            Returns (f_L, tau_L, f_R, tau_R) — forces/torques ON THE BOX.
            """
            body_q = self.state_0.body_q.numpy()
            body_qd = self.state_0.body_qd.numpy()

            box_q = body_q[self.body_box]
            box_com = box_q[:3]
            qx, qy, qz, qw = box_q[3], box_q[4], box_q[5], box_q[6]
            box_quat_wxyz = np.array([qw, qx, qy, qz])

            box_qd = body_qd[self.body_box]
            box_vel = box_qd[:3]
            box_omega = box_qd[3:]

            left_pos = body_q[self.body_left][:3]
            right_pos = body_q[self.body_right][:3]

            f_L, tau_L, self._last_info_L = self.cslc_left.compute_wrench(
                finger_pos_3=left_pos,
                box_com=box_com, box_quat_wxyz=box_quat_wxyz,
                box_vel=box_vel, box_omega=box_omega,
                face_axis=1, face_sign=+1,
                box_halfs=self.box_halfs,
                finger_radius=0.04,
            )

            f_R, tau_R, self._last_info_R = self.cslc_right.compute_wrench(
                finger_pos_3=right_pos,
                box_com=box_com, box_quat_wxyz=box_quat_wxyz,
                box_vel=box_vel, box_omega=box_omega,
                face_axis=1, face_sign=-1,
                box_halfs=self.box_halfs,
                finger_radius=0.04,
            )

            return f_L, tau_L, f_R, tau_R

    def _apply_forces(self):
            """Apply grasp forces, gravity, and CSLC wrench (with reaction on fingers)."""
            # ── CSLC wrench on box ──
            cslc_f_L, cslc_tau_L, cslc_f_R, cslc_tau_R = self._compute_cslc_wrench()

            # ── Build force array on CPU ──
            n_bodies = self.state_0.body_f.shape[0]
            f_ext = np.zeros((n_bodies, 6), dtype=np.float32)

            # Box: gravity + CSLC from both fingers
            box_force = cslc_f_L + cslc_f_R + np.array([0, 0, -9.81 * self.box_mass])
            box_torque = cslc_tau_L + cslc_tau_R

            # Perturbation (one substep only)
            if not self._perturb_applied and self.sim_time >= self.perturb_time:
                I_approx = self.box_mass / 3.0 * (self.box_hy**2 + self.box_hz**2)
                impulse = self.perturb_omega * I_approx / self.sim_dt
                box_torque[self.perturb_axis] += impulse
                self._perturb_applied = True
                print(f"  *** Torque impulse: {impulse:.1f} N·m at t={self.sim_time:.3f}s")

            f_ext[self.body_box, :3] = box_force.astype(np.float32)
            f_ext[self.body_box, 3:] = box_torque.astype(np.float32)

            # Left finger: grasp force (toward box) + CSLC reaction (away from box)
            left_grasp = np.array([0.0, -self.grasp_force, 0.0])
            left_reaction = -cslc_f_L  # Newton's 3rd law
            f_ext[self.body_left, :3] = (left_grasp + left_reaction).astype(np.float32)

            # Right finger: grasp force (toward box) + CSLC reaction (away from box)
            right_grasp = np.array([0.0, self.grasp_force, 0.0])
            right_reaction = -cslc_f_R
            f_ext[self.body_right, :3] = (right_grasp + right_reaction).astype(np.float32)

            self.state_0.body_f.assign(
                wp.array(f_ext, dtype=wp.spatial_vectorf, device="cpu"))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._apply_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1,
                self.control, self.contacts, self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

        # Record
        q = self.state_0.body_q.numpy()[self.body_box]
        self.history["t"].append(self.sim_time)
        self.history["box_z"].append(float(q[2]))
        self.history["box_quat"].append(q[3:].copy())
        self.history["n_contacts_L"].append(self._last_info_L['n_contacts'])
        self.history["n_contacts_R"].append(self._last_info_R['n_contacts'])
        self.history["normal_force_L"].append(self._last_info_L['total_normal'])
        self.history["normal_force_R"].append(self._last_info_R['total_normal'])

        if len(self.history["t"]) % 20 == 0:
            tilt = 2.0 * q[3]  # small-angle: tilt ≈ 2*qx
            print(f"  t={self.sim_time:.3f}s  z={q[2]:.5f}  "
                  f"tilt={np.degrees(tilt):.3f}°  "
                  f"contacts={self._last_info_L['n_contacts']}+"
                  f"{self._last_info_R['n_contacts']}")

        if self.sim_time >= self.max_time:
            self._print_summary()
            self.viewer.close()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _print_summary(self):
        z = np.array(self.history["box_z"])
        quats = np.array(self.history["box_quat"])
        tilt_deg = np.degrees(2.0 * quats[:, self.perturb_axis])
        print(f"\n{'='*50}")
        print(f"  CSLC grasp with tilt perturbation")
        print(f"  Grasp force: {self.grasp_force} N, μ={self.mu}")
        print(f"  Pad: 11×11 = 121 spheres per finger")
        print(f"  Perturbation: {self.perturb_omega} rad/s at t={self.perturb_time}s")
        print(f"  Box z drift: {abs(z[-1]-z[0])*1000:.3f} mm")
        print(f"  Peak tilt: {np.max(np.abs(tilt_deg)):.3f} deg")
        print(f"  Final tilt: {tilt_deg[-1]:.3f} deg")
        print(f"  Avg contacts: {np.mean(self.history['n_contacts_L']):.0f}+"
              f"{np.mean(self.history['n_contacts_R']):.0f}")
        print(f"  Avg normal force: {np.mean(self.history['normal_force_L']):.1f}+"
              f"{np.mean(self.history['normal_force_R']):.1f} N")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--grasp-force", type=float, default=30.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=2.0)
    parser.add_argument("--perturb", type=float, default=3.0)

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)