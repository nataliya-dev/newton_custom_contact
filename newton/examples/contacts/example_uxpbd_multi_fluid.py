# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Multi-Fluid (Buoyancy / Density Stratification)
#
# Two fluid blocks of DIFFERENT REST DENSITIES are spawned with the
# heavier fluid above the lighter fluid. Per UPPFRTA section 7.1.1
# (mass-weighted PBF: "by using the mass-weighted version of position-
# based dynamics we can simulate fluids with differing densities"), the
# heavy fluid should sink through the light fluid until they invert
# (Rayleigh-Taylor instability resolved into stratification).
#
# Each fluid is registered as a SEPARATE phase via add_fluid_grid, so
# they don't contribute to each other's PBF density estimate (the
# kernel filters on `particle_fluid_phase[j] == phase`). The two
# fluids interact only through the cross-substrate particle-particle
# contact pass (solve_particle_particle_contacts_uxpbd; fluid-fluid
# pairs are skipped, so the actual interaction here goes through the
# unilateral contact constraint when particles overlap).
#
# Layout (Z up):
#
#         ::::::::       heavy fluid (rho_0 = 1500 kg/m^3) at top
#         ::::::::
#         ........       light fluid (rho_0 = 1000 kg/m^3) at bottom
#         ........
#         ============== ground (z=0)
#
# Expected: heavy fluid migrates down, light fluid migrates up.
#
# Command: python -m newton.examples uxpbd_multi_fluid
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    PARTICLE_R = 0.008
    CELL = 0.016
    DIMS = (5, 5, 4)            # 100 particles per phase

    LIGHT_BASE_Z = 0.01          # bottom layer
    HEAVY_BASE_Z = 0.078         # 4 layers up; lands directly on the light layer

    LIGHT_DENSITY = 1000.0
    HEAVY_DENSITY = 1500.0

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        x_min = -(self.DIMS[0] - 1) * self.CELL / 2
        y_min = -(self.DIMS[1] - 1) * self.CELL / 2

        # Light fluid (bottom layer).
        self.light_phase = builder.add_fluid_grid(
            pos=wp.vec3(x_min, y_min, self.LIGHT_BASE_Z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.DIMS[0], dim_y=self.DIMS[1], dim_z=self.DIMS[2],
            cell_x=self.CELL, cell_y=self.CELL, cell_z=self.CELL,
            particle_radius=self.PARTICLE_R,
            rest_density=self.LIGHT_DENSITY,
            smoothing_radius_factor=3.0,
            viscosity=0.05,
            cohesion=0.0,
        )
        # Heavy fluid (top layer).
        self.heavy_phase = builder.add_fluid_grid(
            pos=wp.vec3(x_min, y_min, self.HEAVY_BASE_Z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.DIMS[0], dim_y=self.DIMS[1], dim_z=self.DIMS[2],
            cell_x=self.CELL, cell_y=self.CELL, cell_z=self.CELL,
            particle_radius=self.PARTICLE_R,
            rest_density=self.HEAVY_DENSITY,
            smoothing_radius_factor=3.0,
            viscosity=0.05,
            cohesion=0.0,
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.0
        self.model.soft_contact_mu = 0.0

        self.solver = newton.solvers.SolverUXPBD(
            self.model,
            iterations=4,
            fluid_iterations=4,
            stabilization_iterations=2,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Cache per-phase indices for test_final.
        phase = self.model.particle_fluid_phase.numpy()
        self._light_idx = np.where(phase == self.light_phase)[0].astype(np.int32)
        self._heavy_idx = np.where(phase == self.heavy_phase)[0].astype(np.int32)

        # Snapshot initial mean z for each phase.
        pos = self.state_0.particle_q.numpy()
        self._z_light_0 = float(pos[self._light_idx, 2].mean())
        self._z_heavy_0 = float(pos[self._heavy_idx, 2].mean())

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.4, -0.4, 0.20),
                               pitch=-20.0, yaw=135.0)

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
        z_light = float(pos[self._light_idx, 2].mean())
        z_heavy = float(pos[self._heavy_idx, 2].mean())

        # 1. No phase exploded upward.
        z_max = float(pos[:, 2].max())
        assert z_max < self.HEAVY_BASE_Z + (self.DIMS[2] * self.CELL) * 2, (
            f"A fluid launched: z_max={z_max:.4f}"
        )
        # 2. No fluid penetrated the ground.
        z_min = float(pos[:, 2].min())
        assert z_min > -0.02, f"Fluid penetrated ground: z_min={z_min:.4f}"
        # 3. The heavy fluid moved down relative to its initial mean z, OR
        #    the heavy and light fluids show some intermixing (heavy mean
        #    is no longer cleanly above light mean by the original gap).
        #    Either is evidence of buoyancy / stratification dynamics.
        initial_gap = self._z_heavy_0 - self._z_light_0
        final_gap = z_heavy - z_light
        moved = (z_heavy < self._z_heavy_0 - 0.005) or (final_gap < 0.5 * initial_gap)
        assert moved, (
            f"No buoyancy effect: heavy z {self._z_heavy_0:.3f}->{z_heavy:.3f}, "
            f"light z {self._z_light_0:.3f}->{z_light:.3f}, "
            f"gap {initial_gap:.3f}->{final_gap:.3f}"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
