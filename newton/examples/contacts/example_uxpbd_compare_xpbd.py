# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD vs XPBD Side-by-Side
#
# Two identical revolute pendulums run with SolverXPBD (left) and
# SolverUXPBD (right) in lockstep. Visually confirms that the
# articulated rigid path of SolverUXPBD matches SolverXPBD when no
# lattice contact is involved.
#
# Phase 1 limitation: the viewer renders only one model at a time. The
# UXPBD pendulum is rendered; the XPBD pendulum runs in lockstep but is
# not displayed. test_final verifies they stayed in sync. Phase 2's
# multi-model viewer composition would allow both to be visible.
#
# Command: python -m newton.examples uxpbd_compare_xpbd
###########################################################################

import warp as wp

import newton
import newton.examples

# Pendulum geometry
L = 1.0
# Larger angle to make divergence (if any) more visible.
INITIAL_ANGLE = 0.3


def _build_pendulum(x_offset: float) -> newton.Model:
    """Build a single revolute pendulum offset along X for side-by-side viewing."""
    builder = newton.ModelBuilder(up_axis="Z")
    link = builder.add_link(
        xform=wp.transform(p=wp.vec3(x_offset, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(link, hx=L, hy=0.05, hz=0.05)
    j = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(x_offset, 0.0, 5.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-L, 0.0, 0.0), q=wp.quat_identity()),
        target_ke=0.0,
        target_kd=0.0,
    )
    builder.add_articulation([j], label="pendulum")
    model = builder.finalize()

    joint_q = model.joint_q.numpy().copy()
    joint_q[0] = INITIAL_ANGLE
    model.joint_q.assign(joint_q)
    return model


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        # Build two identical pendulums; one per solver.
        self.xpbd_model = _build_pendulum(x_offset=-2.0)
        self.uxpbd_model = _build_pendulum(x_offset=+2.0)

        self.xpbd_solver = newton.solvers.SolverXPBD(self.xpbd_model, iterations=8)
        self.uxpbd_solver = newton.solvers.SolverUXPBD(self.uxpbd_model, iterations=8)

        # XPBD state
        self.xpbd_state_0 = self.xpbd_model.state()
        self.xpbd_state_1 = self.xpbd_model.state()
        self.xpbd_control = self.xpbd_model.control()
        newton.eval_fk(
            self.xpbd_model,
            self.xpbd_model.joint_q,
            self.xpbd_model.joint_qd,
            self.xpbd_state_0,
        )
        self.xpbd_contacts = self.xpbd_model.contacts()

        # UXPBD state
        self.uxpbd_state_0 = self.uxpbd_model.state()
        self.uxpbd_state_1 = self.uxpbd_model.state()
        self.uxpbd_control = self.uxpbd_model.control()
        newton.eval_fk(
            self.uxpbd_model,
            self.uxpbd_model.joint_q,
            self.uxpbd_model.joint_qd,
            self.uxpbd_state_0,
        )
        self.uxpbd_contacts = self.uxpbd_model.contacts()

        # The viewer shows the UXPBD model. The XPBD model runs in the background
        # and is validated via test_final.
        self.viewer.set_model(self.uxpbd_model)

        # Expose the attributes that newton.examples.run inspects for NaN checks.
        # These must point to the state being actively rendered.
        self.model = self.uxpbd_model
        self.state_0 = self.uxpbd_state_0
        self.state_1 = self.uxpbd_state_1
        self.control = self.uxpbd_control
        self.contacts = self.uxpbd_contacts

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Step XPBD
            self.xpbd_state_0.clear_forces()
            self.xpbd_solver.step(
                self.xpbd_state_0,
                self.xpbd_state_1,
                self.xpbd_control,
                self.xpbd_contacts,
                self.sim_dt,
            )
            self.xpbd_state_0, self.xpbd_state_1 = self.xpbd_state_1, self.xpbd_state_0

            # Step UXPBD
            self.uxpbd_state_0.clear_forces()
            self.viewer.apply_forces(self.uxpbd_state_0)
            self.uxpbd_solver.step(
                self.uxpbd_state_0,
                self.uxpbd_state_1,
                self.uxpbd_control,
                self.uxpbd_contacts,
                self.sim_dt,
            )
            self.uxpbd_state_0, self.uxpbd_state_1 = self.uxpbd_state_1, self.uxpbd_state_0

        # Keep the aliased state_0 in sync so NaN checks in run() see the current
        # buffer after the swap.
        self.state_0 = self.uxpbd_state_0
        self.state_1 = self.uxpbd_state_1

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # After the simulation the two pendulums should have similar body poses.
        # We compare the quaternion w-component; large deviations indicate one
        # solver is drifting relative to the other.
        xpbd_bq = self.xpbd_state_0.body_q.numpy()[0]
        uxpbd_bq = self.uxpbd_state_0.body_q.numpy()[0]

        xpbd_qw = float(xpbd_bq[6])
        uxpbd_qw = float(uxpbd_bq[6])

        diff = abs(xpbd_qw - uxpbd_qw)
        if diff > 0.05:
            raise RuntimeError(
                f"XPBD vs UXPBD pendulum diverged: xpbd_qw={xpbd_qw:.4f}, uxpbd_qw={uxpbd_qw:.4f}, diff={diff:.4f}"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.uxpbd_state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
