"""
example_grasp_baseline.py — Clean baseline grasp for CSLC comparison.

Scene:
  - A box at z=1, free to move in 6-DOF
  - Two spheres on prismatic joints (along y), one on each side of the box
  - Constant inward force applied to each sphere → squeeze
  - Gravity applied as a manual force on the box (not global, for control)

Purpose:
  Establish the Newton-native contact baseline that CSLC will later replace.
  The finger–box contact is handled by Newton's MuJoCo solver here.
  In the CSLC version, we will:
    1. Call add_shape_collision_filter_pair() to disable finger–box contact
    2. Run the CSLC lattice solve to compute the finger–box wrench
    3. Inject that wrench into state.body_f on the box

Architecture decisions (from working reference by colleague):
  - Prismatic joints, not kinematic bodies: gives force-controlled fingers
    with finite mass, which is more stable than infinite-mass kinematic walls.
  - MuJoCo solver: proper complementarity contact with friction cones,
    much more robust than penalty-based SemiImplicit at small scales.
  - Manual gravity: lets us control exactly what forces act on the box,
    which is essential when we later replace contact forces with CSLC.

Usage:
  uv run python cslc_scratch/example_grasp_baseline.py --viewer gl
  uv run python cslc_scratch/example_grasp_baseline.py --viewer null
"""

import os
import numpy as np
import warp as wp
import newton
import newton.examples


# ═══════════════════════════════════════════════════════════════════════
#  Warp kernel: apply per-body forces each substep
#
#  This runs on all bodies in parallel. Each body checks its index
#  and applies the appropriate force. Warp kernels are the standard
#  way to write to GPU arrays in Newton.
#
#  body_f layout (confirmed empirically): [fx, fy, fz, tx, ty, tz]
#  spatial_vector constructor: wp.spatial_vector(linear, angular)
#    → BUT Warp spatial_vector is actually (angular, linear) internally
#    so we write: wp.spatial_vector(force_vec, torque_vec)
#    Let's verify by looking at the colleague's code — they write:
#      wp.spatial_vector(left_f, wp.vec3(0.0))
#    where left_f is the force. So the first arg IS the linear force.
#
#  UPDATE: from our Step 2 diagnostics, body_f = [fx,fy,fz, tx,ty,tz].
#  The spatial_vector(a, b) maps to [a, b], so first arg = linear force.
# ═══════════════════════════════════════════════════════════════════════

@wp.kernel
def apply_forces_with_torque_kernel(
    body_f: wp.array(dtype=wp.spatial_vector),
    box_idx: int,
    box_force: wp.vec3,
    box_torque: wp.vec3,
    left_idx: int,
    left_force: wp.vec3,
    right_idx: int,
    right_force: wp.vec3,
):
    tid = wp.tid()
    if tid == box_idx:
        body_f[tid] = wp.spatial_vector(box_force, box_torque)
    elif tid == left_idx:
        body_f[tid] = wp.spatial_vector(left_force, wp.vec3(0.0, 0.0, 0.0))
    elif tid == right_idx:
        body_f[tid] = wp.spatial_vector(right_force, wp.vec3(0.0, 0.0, 0.0))


class Example:
    """Newton example: two-finger grasp baseline."""

    def __init__(self, viewer, args):
        # ── Timing ──
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0


        # ── Perturbation parameters ──
        self.perturb_time = 0.1           # apply after grip settles
        self.perturb_axis = 0             # 0=x tilt, 1=y tilt, 2=z twist
        self.perturb_omega = args.perturb  # rad/s angular impulse
        self._perturb_applied = False

        self.viewer = viewer

        # ── Experiment parameters ──
        self.grasp_force = args.grasp_force   # N, inward on each finger
        self.mu = args.mu                     # friction coefficient
        self.box_mass = args.mass             # kg
        self.max_time = args.max_time         # seconds to run

        # ── Build scene ──
        builder = self._build_scene()
        self.model = builder.finalize()

        # ── Solver: MuJoCo with tuned contact parameters ──
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=200,      # contact solver iterations
            njmax=50,            # max contact points
            cone="elliptic",     # elliptic friction cone (more accurate)
            impratio=50.0,       # impedance ratio for contacts
        )

        # ── Allocate state buffers ──
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        # ── Data collection ──
        self.history = {"t": [], "box_z": [], "box_quat": []}

    def _build_scene(self):
        """
        Construct the grasp scene.

        Geometry (y is grip axis, z is up):

            left sphere ←→ [  box  ] ←→ right sphere
                  ↑ grasp_force    grasp_force ↑
                           ↓ gravity

        Returns the ModelBuilder with all bodies, shapes, joints.
        """
        builder = newton.ModelBuilder(gravity=0.0)  # manual gravity
        builder.add_ground_plane()

        # ── Box: free-floating object ──
        box_hx, box_hy, box_hz = 0.05, 0.05, 0.1
        self.box_pos = wp.vec3(0.0, 0.0, 1.0)

        self.body_box = builder.add_body(
            xform=wp.transform(p=self.box_pos, q=wp.quat_identity()),
            mass=self.box_mass,
            label="box",
        )
        self.shape_box = builder.add_shape_box(
            self.body_box, hx=box_hx, hy=box_hy, hz=box_hz,
        )

        # ── Fingers: spheres on prismatic joints along y-axis ──
        sphere_r = 0.04
        finger_y = box_hy + sphere_r  # just touching

        # Left finger (+y side)
        left_pos = self.box_pos + wp.vec3(0.0, finger_y, 0.0)
        self.body_left = builder.add_link(
            xform=wp.transform(p=left_pos, q=wp.quat_identity()),
            mass=1.0,
            label="left_finger",
        )
        self.shape_left = builder.add_shape_sphere(
            self.body_left, radius=sphere_r,
        )
        left_joint = builder.add_joint_prismatic(
            parent=-1,  # world
            child=self.body_left,
            parent_xform=wp.transform(p=left_pos, q=wp.quat_identity()),
            axis=newton.Axis.Y,
        )
        builder.add_articulation([left_joint])

        # Right finger (-y side)
        right_pos = self.box_pos + wp.vec3(0.0, -finger_y, 0.0)
        self.body_right = builder.add_link(
            xform=wp.transform(p=right_pos, q=wp.quat_identity()),
            mass=1.0,
            label="right_finger",
        )
        self.shape_right = builder.add_shape_sphere(
            self.body_right, radius=sphere_r,
        )
        right_joint = builder.add_joint_prismatic(
            parent=-1,
            child=self.body_right,
            parent_xform=wp.transform(p=right_pos, q=wp.quat_identity()),
            axis=newton.Axis.Y,
        )
        builder.add_articulation([right_joint])

        # ── Friction: set on all shapes ──
        for shape_idx in range(builder.shape_count):
            builder.shape_material_mu[shape_idx] = self.mu
            builder.shape_material_mu_torsional[shape_idx] = self.mu
            builder.shape_material_mu_rolling[shape_idx] = self.mu

        # ── Store shape indices for future collision filtering ──
        # When we add CSLC, we will call:
        #   builder.add_shape_collision_filter_pair(self.shape_left, self.shape_box)
        #   builder.add_shape_collision_filter_pair(self.shape_right, self.shape_box)
        # to disable Newton's finger–box contact and replace it with CSLC.
        print(f"  Shape indices: box={self.shape_box}, "
              f"left={self.shape_left}, right={self.shape_right}")

        return builder


    def _apply_forces(self):
            """Apply grasp forces on fingers, gravity + optional perturbation on box."""
            # Gravity + perturbation torque
            box_f = wp.vec3(0.0, 0.0, -9.81 * self.box_mass)

            # Torque impulse: τ = I·Δω / Δt_frame
            # For a box, Ix ≈ m/12*(hy²+hz²)*4. But simpler: just apply
            # a big torque for one frame and measure what happens.
            box_torque = wp.vec3(0.0, 0.0, 0.0)
            if not self._perturb_applied and self.sim_time >= self.perturb_time:
                box_torque = wp.vec3(0.0, 0.0, 0.0)  # placeholder, set below
                # Approximate: torque = large value for one substep
                # I_xx for box ≈ m/3*(hy²+hz²) ≈ 1.0/3*(0.05²+0.1²) ≈ 0.0042
                # τ·dt = I·ω  →  τ = I·ω/dt = 0.0042 * 3.0 / 0.001 ≈ 12.5 N·m
                impulse_torque = self.perturb_omega * 0.0042 / self.sim_dt
                if self.perturb_axis == 0:
                    box_torque = wp.vec3(impulse_torque, 0.0, 0.0)
                elif self.perturb_axis == 1:
                    box_torque = wp.vec3(0.0, impulse_torque, 0.0)
                else:
                    box_torque = wp.vec3(0.0, 0.0, impulse_torque)
                self._perturb_applied = True
                print(f"  *** Torque impulse: {impulse_torque:.1f} N·m "
                    f"for 1 substep at t={self.sim_time:.3f}s")

            left_force = wp.vec3(0.0, -self.grasp_force, 0.0)
            right_force = wp.vec3(0.0, self.grasp_force, 0.0)

            wp.launch(
                kernel=apply_forces_with_torque_kernel,
                dim=self.state_0.body_f.shape[0],
                inputs=[
                    self.state_0.body_f,
                    int(self.body_box), box_f, box_torque,
                    int(self.body_left), left_force,
                    int(self.body_right), right_force,
                ],
                device=wp.get_device(),
            )


    def simulate(self):
        """Run one frame (multiple substeps)."""
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
        """Called each frame by newton.examples.run()."""
        self.simulate()
        self.sim_time += self.frame_dt

        # ── Record data ──
        q = self.state_0.body_q.numpy()[self.body_box]
        self.history["t"].append(self.sim_time)
        self.history["box_z"].append(float(q[2]))
        self.history["box_quat"].append(q[3:].copy())

        # ── Print progress ──
        if len(self.history["t"]) % 20 == 0:
            print(f"  t={self.sim_time:.3f}s  box_z={q[2]:.5f}  "
                  f"quat=[{q[3]:.4f} {q[4]:.4f} {q[5]:.4f} {q[6]:.4f}]")

        # ── Stop condition ──
        if self.sim_time >= self.max_time:
            self._print_summary()
            self.viewer.close()

    def render(self):
        """Called each frame for visualization."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _print_summary(self):
            z = np.array(self.history["box_z"])
            quats = np.array(self.history["box_quat"])
            # Small-angle: tilt ≈ 2*qx or 2*qy (depending on axis)
            tilt = 2.0 * quats[:, self.perturb_axis]
            tilt_deg = np.degrees(tilt)
            print(f"\n{'='*50}")
            print(f"  Baseline grasp with tilt perturbation")
            print(f"  Grasp force: {self.grasp_force} N, μ={self.mu}")
            print(f"  Perturbation: {self.perturb_omega} rad/s at t={self.perturb_time}s")
            print(f"  Box z drift: {abs(z[-1]-z[0])*1000:.3f} mm")
            print(f"  Peak tilt: {np.max(np.abs(tilt_deg)):.3f} deg")
            print(f"  Final tilt: {tilt_deg[-1]:.3f} deg")
            print(f"{'='*50}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--grasp-force", type=float, default=30.0,
                        help="Inward force on each finger (N)")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Friction coefficient")
    parser.add_argument("--mass", type=float, default=1.0,
                        help="Box mass (kg)")
    parser.add_argument("--max-time", type=float, default=2.0,
                        help="Simulation duration (s)")

    parser.add_argument("--perturb", type=float, default=3.0,
                        help="Angular impulse magnitude (rad/s)")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)