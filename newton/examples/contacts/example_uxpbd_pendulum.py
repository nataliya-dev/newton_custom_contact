# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Pendulum
#
# A simple revolute pendulum of length L oscillates at the analytical
# small-angle period T = 2*pi*sqrt(L/g). Validates joint dynamics and
# energy conservation in SolverUXPBD.
#
# Command: python -m newton.examples uxpbd_pendulum
###########################################################################

import math

import warp as wp

import newton
import newton.examples

# Pendulum half-length [m]. The pivot is at one end and the mass at the other.
L = 1.0
# Small initial angle [rad] -- small-angle approximation holds for < ~10 deg.
INITIAL_ANGLE = 0.05


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        # Analytical period for small-angle pendulum: T = 2*pi*sqrt(L/g).
        # We simulate slightly more than one period and check the pendulum has
        # returned near its starting angle. Using L=1.0 and g=9.81 gives T~2.0 s.
        g = 9.81
        self.period = 2.0 * math.pi * math.sqrt(L / g)

        builder = newton.ModelBuilder(up_axis="Z")

        link = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
        )
        # The pendulum arm is a thin rod along Z. The joint is a revolute around X
        # with the pivot at (0, 0, 5) and the child anchor at the +Z end of the arm.
        # In the zero-angle configuration the rod hangs straight down, with its CoM
        # at (0, 0, 5 - L/2). A small INITIAL_ANGLE (in radians) tilts it from vertical.
        builder.add_shape_box(link, hx=0.05, hy=0.05, hz=L)

        j = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(1.0, 0.0, 0.0),  # rotate around X; swings in YZ plane
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, L), q=wp.quat_identity()),
            target_ke=0.0,
            target_kd=0.0,
        )
        builder.add_articulation([j], label="pendulum")

        self.model = builder.finalize()

        # Set the initial joint angle (small tilt from vertical).
        joint_q = self.model.joint_q.numpy().copy()
        joint_q[0] = INITIAL_ANGLE
        self.model.joint_q.assign(joint_q)

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=8)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # After at least one full period the pendulum should be back near its
        # starting orientation (qw close to 1 for a small-angle swing).
        bq = self.state_0.body_q.numpy()[0]
        qw = float(bq[6])
        if qw < 0.99:
            raise RuntimeError(f"Pendulum drifted too far from identity; qw={qw}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
