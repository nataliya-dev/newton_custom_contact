# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Fluid Cohesive Blob
#
# A 5x5x5 block of PBF fluid particles with a high Akinci cohesion
# coefficient. The blob falls under gravity and stays together as a
# coherent mass instead of spreading out (compare with
# example_uxpbd_fluid_drop, which uses zero cohesion). Demonstrates the
# `cohesion` knob exposed by `add_fluid_grid` (Akinci et al. 2013).
#
# Command: python -m newton.examples uxpbd_fluid_cohesive_blob
###########################################################################

import numpy as np
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

        # Same packing as example_uxpbd_fluid_drop, but with strong cohesion.
        # smoothing_radius_factor=3.0 so PBF kernel reaches neighbors; see
        # fluid_drop for the cell == h degeneracy this avoids.
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
            cohesion=50.0,  # large vs. fluid_drop (0.0): blob stays compact
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.0
        self.model.soft_contact_mu = 0.0

        self.solver = newton.solvers.SolverUXPBD(
            self.model, iterations=4, fluid_iterations=4)
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

        # Snapshot initial max-pairwise-distance proxy: bounding diagonal.
        pos = self.state_0.particle_q.numpy()
        bbox = pos.max(axis=0) - pos.min(axis=0)
        self._diag_0 = float(np.linalg.norm(bbox))

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
        bbox = pos.max(axis=0) - pos.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        z_min = float(pos[:, 2].min())

        # 1. Blob stayed compact (bounding-box diagonal did not blow up).
        # A loose, no-cohesion drop spreads to ~2-3x its initial diagonal;
        # the cohesive blob should stay close to its initial extent.
        assert diag < 2.0 * self._diag_0, (
            f"blob lost cohesion: diag={diag:.4f} vs initial {self._diag_0:.4f}"
        )
        # 2. No ground penetration.
        assert z_min > -0.02, f"fluid penetrated ground: z_min={z_min:.4f}"

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
