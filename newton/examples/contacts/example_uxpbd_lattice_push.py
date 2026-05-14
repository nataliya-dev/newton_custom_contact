# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Lattice Push
#
# A single revolute link with a MorphIt-packed kinematic lattice rotates
# under a constant torque about the world Z axis. Phase 1 demo for
# SolverUXPBD: validates lattice-to-shape contact via the body wrench
# accumulation pipeline, with joint constraints in the iteration loop.
#
# Command: python -m newton.examples uxpbd_lattice_push
###########################################################################

import os

import warp as wp

import newton
import newton.examples


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

        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.06, hz=0.06)
        j = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.3), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=0.0,
            target_kd=0.0,
        )
        builder.add_articulation([j], label="pusher")

        json_path = os.path.join(os.path.dirname(__file__), "..", "assets", "uxpbd", "link_box.json")
        json_path = os.path.normpath(json_path)
        builder.add_lattice(
            link=link,
            morphit_json=json_path,
            total_mass=0.0,
            pos=wp.vec3(0.0, 0.0, 0.3),
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=4)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Constant torque about Z to drive rotation [N*m].
        if self.control.joint_f is not None and self.control.joint_f.shape[0] > 0:
            joint_f_np = self.control.joint_f.numpy().copy()
            joint_f_np[0] = 5.0
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
        bq = self.state_0.body_q.numpy()[0]
        body_z = float(bq[2])
        qw = float(bq[6])
        if abs(qw) >= 0.999:
            raise RuntimeError(f"Pusher did not rotate: qw={qw}")
        if body_z < 0.2:
            raise RuntimeError(f"Pusher fell through revolute constraint: z={body_z}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
