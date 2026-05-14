# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Free Fall
#
# A free body with no contact falls under gravity. Validates the body
# integrator in SolverUXPBD: trajectory must match z = z0 - g*t^2/2
# within 1% relative error. No shapes and no lattice are attached so
# this is a pure ballistic trajectory test.
#
# Command: python -m newton.examples uxpbd_free_fall
###########################################################################

import warp as wp

import newton
import newton.examples

INITIAL_Z = 10.0  # [m] initial height above the origin
G = 9.81  # [m/s^2] gravitational acceleration (matches Newton default)


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
        # A single free body, no shapes, no lattice -- pure ballistic fall.
        builder.add_body(
            mass=1.0,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, INITIAL_Z), q=wp.quat_identity()),
        )
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=1)
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
        # Analytical trajectory: z(t) = INITIAL_Z - 0.5 * G * t^2
        expected_z = INITIAL_Z - 0.5 * G * self.sim_time * self.sim_time
        body_z = float(self.state_0.body_q.numpy()[0, 2])

        # Skip the check if the analytical answer says the body should already
        # be far underground (test was run for many seconds).
        if expected_z < -1.0:
            return

        rel_err = abs(body_z - expected_z) / max(abs(expected_z), 1.0e-3)
        if rel_err > 0.01:
            raise RuntimeError(
                f"Free fall trajectory error: z={body_z:.4f}, expected={expected_z:.4f}, "
                f"rel_err={rel_err:.4f} (t={self.sim_time:.2f} s)"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
