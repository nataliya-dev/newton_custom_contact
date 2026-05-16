# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Fluid Column Collapse
#
# A tall narrow column of PBF fluid particles (3x3x12 = 108) collapses
# onto the ground plane under gravity, demonstrating PBF
# incompressibility: the column gets shorter and the puddle gets wider,
# but particles do not stack to a point. This is the "dam-break lite"
# scenario adapted from the Phase 4 unit-test setup.
#
# Command: python -m newton.examples uxpbd_fluid_column_collapse
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

        # Tall column: 3x3 cross-section, 12 layers high. Particle radius
        # 0.008 m, cell = 2*r = 0.016 m. Column footprint 0.048 m,
        # height 0.192 m. Spawned with its base touching the ground. The
        # smoothing radius is set to 3*r so PBF actually couples
        # neighbors (see fluid_drop example for the cell == h pitfall);
        # without that, the column has no compressibility resistance and
        # ground contact launches the top of the stack upward.
        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()
        builder.add_fluid_grid(
            pos=wp.vec3(-0.024, -0.024, 0.012),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=3, dim_y=3, dim_z=12,
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
            pos=wp.vec3(0.30, -0.30, 0.15),
            pitch=-15.0, yaw=135.0,
        )

        pos = self.state_0.particle_q.numpy()
        self._z_max_0 = float(pos[:, 2].max())
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
        z_max = float(pos[:, 2].max())
        z_min = float(pos[:, 2].min())
        x_extent = float(pos[:, 0].max() - pos[:, 0].min())

        # 1. Column got shorter (collapsed).
        assert z_max < 0.5 * self._z_max_0, (
            f"column did not collapse: z_max={z_max:.4f} vs initial {self._z_max_0:.4f}"
        )
        # 2. Footprint widened.
        assert x_extent > self._x_extent_0 * 1.5, (
            f"fluid did not spread: x_extent={x_extent:.4f} vs initial {self._x_extent_0:.4f}"
        )
        # 3. No ground penetration.
        assert z_min > -0.02, f"fluid penetrated ground: z_min={z_min:.4f}"

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
