# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Fluid Drop
#
# A 5x5x5 grid of PBF fluid particles (125 total) falls from 0.20 m onto
# the ground plane and spreads into a puddle. Adapted from the unit test
# `test_uxpbd_fluid_on_ground_no_penetration` (and `test_uxpbd_fluid_block_
# settles`) for visual inspection of the Phase 4 PBF path.
#
# Command: python -m newton.examples uxpbd_fluid_drop
###########################################################################

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

        # 5^3 = 125 particles, radius 0.008 m, cell = 2*r = 0.016 m, so a
        # 0.08 m side cube of fluid dropped from 0.20 m. The smoothing
        # radius is set to 3*r (= 1.5*cell) so each particle's PBF kernel
        # reaches its face+edge neighbors. With the default factor of 2.0,
        # cell == h and neighbors land exactly on the Poly6 cutoff
        # (W = 0), which leaves PBF unable to generate inter-particle
        # corrections and lets the block collapse to a flat plate.
        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()
        builder.add_fluid_grid(
            pos=wp.vec3(-0.032, -0.032, 0.20),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=5, dim_y=5, dim_z=5,
            cell_x=0.016, cell_y=0.016, cell_z=0.016,
            particle_radius=0.008,
            rest_density=1000.0,
            smoothing_radius_factor=3.0,
            viscosity=0.05,
            cohesion=0.0,
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.0
        self.model.soft_contact_mu = 0.0

        self.solver = newton.solvers.SolverUXPBD(
            self.model, iterations=2, fluid_iterations=2)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(0.35, -0.35, 0.20),
            pitch=-25.0, yaw=135.0,
        )

        # Snapshot initial x-extent so test_final can verify spreading.
        pos = self.state_0.particle_q.numpy()
        self._x_extent_0 = float(pos[:, 0].max() - pos[:, 0].min())

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pos = self.state_0.particle_q.numpy()
        z_min = float(pos[:, 2].min())
        x_extent = float(pos[:, 0].max() - pos[:, 0].min())

        # 1. Fluid did not pass through the ground.
        assert z_min > -0.02, f"fluid penetrated ground: z_min={z_min:.4f}"
        # 2. Fluid spread out under PBF + gravity (puddle wider than initial cube).
        assert x_extent > self._x_extent_0 * 1.2, (
            f"fluid did not spread: x_extent={x_extent:.4f} vs initial {self._x_extent_0:.4f}"
        )

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
