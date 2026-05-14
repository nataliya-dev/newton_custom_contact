# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Drop to Ground
#
# A free body with a MorphIt-style 16-sphere surface lattice (link_box.json)
# falls under gravity and settles on the ground plane. Validates the
# lattice-to-body wrench accumulation pipeline under SolverUXPBD, including
# off-COM contacts: the contact correction on each lattice particle must be
# routed into a wrench on the host body.
#
# A collision box shape (hx=hy=hz=0.05 m) is added to give the body a
# physically realistic inertia tensor. Without it, validate_inertia clamps
# the inertia to ~1e-6 kg*m^2 (inv_I ~ 1e6), and torques from off-COM
# contacts (tau = r x F, |r| up to 0.14 m) blow up angular velocity within
# ~1500 substeps. The self-contact guard in solve_particle_shape_contacts_uxpbd
# prevents the body's own box from colliding with its own lattice spheres.
#
# Resting geometry:
#   - Bottom lattice spheres have body-frame z = -0.05 m, radius 0.04 m
#   - Lowest world point: body_z - 0.05 - 0.04 = body_z - 0.09
#   - Ground contact when body_z = 0.09 m
#
# Command: python -m newton.examples uxpbd_drop_to_ground
###########################################################################

import os

import warp as wp

import newton
import newton.examples

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "uxpbd")


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

        # A free floating body starting 0.5 m above the ground.
        body = builder.add_body(
            mass=1.0,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()),
        )
        # Add a collision box shape so the body has a physically realistic
        # inertia tensor (I ~ 1/6 * m * side^2 per axis). Without a shape,
        # validate_inertia clamps to ~1e-6 kg*m^2, giving inv_I ~ 1e6, which
        # causes torque blow-up from off-COM lattice contacts.
        # The self-contact guard in solve_particle_shape_contacts_uxpbd (shape_link ==
        # host_link) prevents this box from colliding with the body's own lattice.
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)

        # Attach a lattice so the body can interact with the ground plane via
        # particle-based contact. link_box.json is the existing asset used in
        # example_uxpbd_lattice_push; radii ~0.04 m, 16 spheres in a box pattern.
        json_path = os.path.normpath(os.path.join(_ASSET_DIR, "link_box.json"))
        builder.add_lattice(
            link=body,
            morphit_json=json_path,
            total_mass=0.0,
            pos=wp.vec3(0.0, 0.0, 0.5),
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=4)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
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
        # The bottom lattice spheres (body-frame z = -0.05, radius 0.04) rest on
        # the ground when body_z = 0.09. Assert settling within 5 mm tolerance.
        body_z = float(self.state_0.body_q.numpy()[0, 2])
        expected_z = 0.09
        tol = 5.0e-3
        if abs(body_z - expected_z) > tol:
            raise RuntimeError(
                f"Body did not settle at lattice resting height; "
                f"z={body_z:.4f}, expected {expected_z:.4f} +/- {tol:.4f}"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
