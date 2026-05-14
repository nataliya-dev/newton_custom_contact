# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD 3-Link Arm Push
#
# A 3-link planar revolute arm with MorphIt-style lattices on each link
# is driven by constant joint torques to push a static (kinematic) box on
# the ground. Closest Phase 1 analog to the planned scenario A (Franka
# rigid pick and place), which requires Phase 2 (free shape-matched rigid)
# for the pickable object to be dynamic.
#
# The target box uses is_kinematic=True so it does not respond to forces
# and acts as a fixed obstacle for the arm to push against. mass=0.0
# alone is insufficient because it only zeros the inverse mass without
# setting the KINEMATIC body flag that prevents position updates.
#
# Command: python -m newton.examples uxpbd_arm_push
###########################################################################

import os

import warp as wp

import newton
import newton.examples

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "uxpbd")

# Link geometry [m].
LINK_LENGTH = 0.4
LINK_HALF_W = 0.05


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # Static target box -- kinematic so it does not move when the arm pushes
        # it. Phase 2 will make this a free rigid body that can be grasped.
        target_body = builder.add_body(
            is_kinematic=True,
            xform=wp.transform(p=wp.vec3(1.3, 0.0, 0.1), q=wp.quat_identity()),
        )
        builder.add_shape_box(target_body, hx=0.1, hy=0.1, hz=0.1)

        # Reuse the existing link_box.json lattice asset for each arm link.
        json_path = os.path.normpath(os.path.join(_ASSET_DIR, "link_box.json"))

        # Build a 3-link planar arm. All joints rotate around Y. The base pivot
        # is at (0, 0, 0.5); each subsequent pivot is at the distal end of the
        # previous link.
        prev_body = -1  # -1 = world for the first joint
        # parent_xform for the first joint: world position of the base pivot.
        prev_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity())
        joints = []

        for _i in range(3):
            link = builder.add_link(
                xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
                mass=0.5,
            )
            builder.add_shape_box(link, hx=LINK_LENGTH, hy=LINK_HALF_W, hz=LINK_HALF_W)

            j = builder.add_joint_revolute(
                parent=prev_body,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=prev_xform,
                child_xform=wp.transform(p=wp.vec3(-LINK_LENGTH, 0.0, 0.0), q=wp.quat_identity()),
                target_ke=0.0,
                target_kd=0.0,
            )

            # Attach a lattice so this link participates in particle-based contact.
            # pos offsets the lattice to match the link's resting world position,
            # but at t=0 all links start at the origin so we leave pos as default.
            builder.add_lattice(
                link=link,
                morphit_json=json_path,
                total_mass=0.0,
            )

            joints.append(j)
            prev_body = link
            # Distal end of this link becomes the parent anchor for the next joint.
            prev_xform = wp.transform(p=wp.vec3(LINK_LENGTH, 0.0, 0.0), q=wp.quat_identity())

        builder.add_articulation(joints, label="arm")

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=8)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Apply constant torques to the three arm revolute joints to swing the arm
        # toward the target box. The target box's free joint (index 0, added by
        # add_body) has one DOF; skip it. The arm joints follow.
        if self.control.joint_f is not None and self.control.joint_f.shape[0] > 0:
            joint_f_np = self.control.joint_f.numpy().copy()
            # joint 0 = target box free joint (6 DOF); arm revolute joints follow.
            # We apply a moderate torque to each of the 3 arm revolute DOFs.
            n_dof = joint_f_np.shape[0]
            # Target box free joint uses 6 DOFs (indices 0..5). Arm revolute
            # joints start at index 6 (one DOF each).
            arm_start = 6
            for k in range(arm_start, min(arm_start + 3, n_dof)):
                joint_f_np[k] = 5.0
            self.control.joint_f.assign(joint_f_np)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # The arm should have moved. We read joint_q from the state (body_q
        # encodes the full pose); check that at least one arm link has rotated.
        body_q = self.state_0.body_q.numpy()
        # body 0 = target box (kinematic, should not have moved significantly).
        # bodies 1, 2, 3 = arm links.
        n_bodies = body_q.shape[0]
        arm_moved = False
        for i in range(1, min(4, n_bodies)):
            qw = float(body_q[i, 6])
            # Any rotation gives qw < 1.
            if qw < 0.999:
                arm_moved = True
            # Divergence check: qw must be a valid quaternion component.
            if abs(qw) > 1.01:
                raise RuntimeError(f"Arm body {i} has invalid quaternion qw={qw}")
        if not arm_moved:
            raise RuntimeError("Arm did not rotate under applied joint torques.")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
