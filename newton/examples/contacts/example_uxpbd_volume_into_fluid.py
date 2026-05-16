# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Volume Into Fluid
#
# A free shape-matched (SM-rigid) ball, built from add_particle_volume,
# is dropped into a pool of PBF fluid. Tests SM-rigid <-> fluid
# coupling: PBF density estimation includes the ball's particles as
# "solid" contributors via the UPPFRTA section 7.1 eq. 27 coupling
# factor s, while the SM-rigid shape-matching pass keeps the ball
# rigid (PBD-R momentum-conservation post-pass).
#
# Layout (Z up):
#
#         SM ball
#         (####)         drop from z=0.30
#         ~~~~~~~~~      fluid pool surface ~ z=0.10
#         ::::::::
#         ::::::::       fluid block at z=[0.01, 0.10]
#         ::::::::
#         ============== ground (z=0)
#
# Command: python -m newton.examples uxpbd_volume_into_fluid
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    BALL_RADIUS = 0.04
    BALL_SPHERE_R = 0.012
    BALL_SPAWN_Z = 0.30

    FLUID_PARTICLE_R = 0.008
    FLUID_CELL = 0.016
    FLUID_DIMS = (8, 8, 5)
    FLUID_BASE_Z = 0.01

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

        # ----- Fluid pool -----
        x_min = -(self.FLUID_DIMS[0] - 1) * self.FLUID_CELL / 2
        y_min = -(self.FLUID_DIMS[1] - 1) * self.FLUID_CELL / 2
        builder.add_fluid_grid(
            pos=wp.vec3(x_min, y_min, self.FLUID_BASE_Z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.FLUID_DIMS[0],
            dim_y=self.FLUID_DIMS[1],
            dim_z=self.FLUID_DIMS[2],
            cell_x=self.FLUID_CELL,
            cell_y=self.FLUID_CELL,
            cell_z=self.FLUID_CELL,
            particle_radius=self.FLUID_PARTICLE_R,
            rest_density=1000.0,
            smoothing_radius_factor=3.0,
            viscosity=0.05,
            cohesion=0.0,
        )

        # ----- SM-rigid ball (sphere-masked particle volume) -----
        n = 4
        coords = np.linspace(-self.BALL_RADIUS + self.BALL_SPHERE_R,
                             self.BALL_RADIUS - self.BALL_SPHERE_R, n)
        bxs, bys, bzs = np.meshgrid(coords, coords, coords, indexing="ij")
        all_centers = np.stack(
            [bxs.flatten(), bys.flatten(), bzs.flatten()], axis=1).astype(np.float32)
        keep = np.linalg.norm(all_centers, axis=1) <= self.BALL_RADIUS - 0.5 * self.BALL_SPHERE_R
        ball_centers = all_centers[keep]
        ball_radii = np.full(ball_centers.shape[0], self.BALL_SPHERE_R, dtype=np.float32)
        # Total mass calibrated so the ball is denser than water ->
        # sinks. Volume ~ (4/3)pi*R^3 = 2.68e-4 m^3, density ~ 2000
        # kg/m^3 -> ~0.54 kg.
        ball_total_mass = 0.54
        self.ball_group = builder.add_particle_volume(
            volume_data={"centers": ball_centers.tolist(),
                         "radii": ball_radii.tolist()},
            total_mass=ball_total_mass,
            pos=wp.vec3(0.0, 0.0, self.BALL_SPAWN_Z),
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.0
        self.model.soft_contact_mu = 0.0

        self.solver = newton.solvers.SolverUXPBD(
            self.model,
            iterations=6,
            fluid_iterations=4,
            stabilization_iterations=2,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        ball_idx = self.model.particle_groups[self.ball_group]
        if hasattr(ball_idx, "numpy"):
            ball_idx = ball_idx.numpy()
        self._ball_idx = np.asarray(list(ball_idx), dtype=np.int32)

        substrate = self.model.particle_substrate.numpy()
        self._fluid_idx = np.where(substrate == 3)[0].astype(np.int32)

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.5, -0.5, 0.30),
                               pitch=-25.0, yaw=135.0)

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
        # Ball: settled (sank through fluid), near rest.
        ball_q = self.state_0.particle_q.numpy()[self._ball_idx]
        ball_v = self.state_0.particle_qd.numpy()[self._ball_idx]
        ball_com_z = float(ball_q[:, 2].mean())
        ball_v_mag = float(np.linalg.norm(ball_v.mean(axis=0)))
        assert ball_com_z < self.BALL_SPAWN_Z - 0.10, (
            f"Ball did not fall significantly: com_z={ball_com_z:.4f}"
        )
        assert ball_com_z > 0.02, f"Ball penetrated ground: com_z={ball_com_z:.4f}"
        assert ball_v_mag < 1.0, f"Ball still moving: |v|={ball_v_mag:.3f} m/s"

        # Fluid: didn't launch, didn't penetrate.
        fluid_q = self.state_0.particle_q.numpy()[self._fluid_idx]
        z_max_fluid = float(fluid_q[:, 2].max())
        assert z_max_fluid < self.BALL_SPAWN_Z + 0.10, (
            f"Fluid splashed too high: z_max={z_max_fluid:.4f}"
        )
        z_min_fluid = float(fluid_q[:, 2].min())
        assert z_min_fluid > -0.02, f"Fluid penetrated ground: z_min={z_min_fluid:.4f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
