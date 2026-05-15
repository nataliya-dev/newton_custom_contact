# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Drop to Ground
#
# A free body with a kinematic lattice falls under gravity and settles on
# the ground plane. Validates the lattice-to-body wrench accumulation
# pipeline under SolverUXPBD, including off-COM contacts: the contact
# correction on each lattice particle must be routed into a wrench on the
# host body.
#
# The lattice mirrors the configuration used by example_uxpbd_lattice_stack:
# a 4x4x4 sphere packing whose centers are inscribed in a CUBE_HALF_EXTENT
# cube (matching the cube particles used by the stacked SM-rigid demo), but
# with a LARGER per-sphere radius (LATTICE_SPHERE_R) than the SM cube's
# particles so that the lattice spheres protrude past the host body's
# collision box face and act as a contact shell.
#
# A collision box of the same half-extent gives the body a realistic
# body_inertia tensor via the shape's default density (1000 kg/m^3). Using
# `add_body(mass=0.0)` lets that density carry both mass and inertia
# consistently; mixing a nonzero add_body mass with shape density produces
# a body where body_inv_mass and body_inv_inertia correspond to different
# masses (see the NOTE in example_uxpbd_lattice_stack.py).
#
# Resting geometry:
#   - Bottom lattice spheres: body-frame z = -(CUBE_HALF_EXTENT - CUBE_SPHERE_R)
#     = -0.028, radius LATTICE_SPHERE_R = 0.02
#   - Lowest world point: body_z - 0.028 - 0.02 = body_z - 0.048
#   - Ground contact when body_z = 0.048 m
#
# Command: python -m newton.examples uxpbd_drop_to_ground
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    CUBE_HALF_EXTENT = 0.04
    CUBE_SPHERE_R = 0.012        # cube particle radius (used only for the
                                 # shared packing center layout)
    LATTICE_SPHERE_R = 0.02      # protrudes past the box face for contact

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
        # mass=0.0 + add_shape_box density carries both mass and inertia
        # consistently. See example_uxpbd_lattice_stack.py for the full
        # explanation of why mixing add_body(mass=...) with shape density
        # produces inconsistent body_inv_mass / body_inv_inertia.
        body = builder.add_body(
            mass=0.0,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()),
        )
        builder.add_shape_box(
            body,
            hx=self.CUBE_HALF_EXTENT,
            hy=self.CUBE_HALF_EXTENT,
            hz=self.CUBE_HALF_EXTENT,
        )

        # Same 4x4x4 packing center layout as example_uxpbd_lattice_stack,
        # with the larger LATTICE_SPHERE_R so the lattice spheres protrude
        # past the box face and form a contact shell.
        coords = np.linspace(
            -self.CUBE_HALF_EXTENT + self.CUBE_SPHERE_R,
            self.CUBE_HALF_EXTENT - self.CUBE_SPHERE_R,
            4,
        )
        xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
        packing_centers = np.stack(
            [xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
        lattice_radii = np.full(
            packing_centers.shape[0], self.LATTICE_SPHERE_R, dtype=np.float32)
        builder.add_lattice(
            link=body,
            morphit_json={"centers": packing_centers, "radii": lattice_radii},
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
        # The bottom lattice spheres (body-frame z = -0.028, radius 0.02)
        # rest on the ground when body_z = 0.048. Assert settling within
        # 5 mm tolerance.
        body_z = float(self.state_0.body_q.numpy()[0, 2])
        expected_z = 0.048
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
