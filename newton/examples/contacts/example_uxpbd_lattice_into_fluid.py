# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Lattice Into Fluid
#
# A lattice-clad articulated rigid body is dropped from above into a
# pool of PBF fluid sitting on the ground plane. Tests the cross-substrate
# pipeline:
#
#   - lattice particles (substrate 0) collide with fluid particles
#     (substrate 3) via solve_particle_particle_contacts_uxpbd
#   - lattice contact deltas route into the host body wrench (design
#     spec sect 5.4) while fluid particle deltas update particle_q
#   - fluid density estimation includes the lattice spheres as "solid"
#     contributors per UPPFRTA eq. 27 (fluid_solid_coupling_s)
#
# Layout (Z up):
#
#         lattice cube
#         ____ ____
#         |####|         drop from z=0.30
#         |####|
#         ~~~~~~~~~      fluid pool surface ~ z=0.10
#         ::::::::
#         ::::::::       fluid block at z=[0.01, 0.10]
#         ::::::::
#         ============== ground (z=0)
#
# Command: python -m newton.examples uxpbd_lattice_into_fluid
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    CUBE_HALF_EXTENT = 0.04
    CUBE_SPHERE_R = 0.012
    CUBE_SPAWN_Z = 0.30

    FLUID_PARTICLE_R = 0.008
    FLUID_CELL = 0.016
    FLUID_DIMS = (8, 8, 5)              # 320 fluid particles
    FLUID_BASE_Z = 0.01

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # 16 substeps + stabilization keeps the cube from launching off
        # the fluid surface during the impact frame.
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # ----- Fluid pool (8x8x5 = 320 particles) -----
        # Sized so the cube's footprint (~8 cm) is a small fraction of
        # the pool footprint (~13 cm); avoids edge effects at impact.
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
        fluid_top_z = (self.FLUID_BASE_Z
                       + (self.FLUID_DIMS[2] - 1) * self.FLUID_CELL)

        # ----- Lattice-clad cube above the pool -----
        coords = np.linspace(
            -self.CUBE_HALF_EXTENT + self.CUBE_SPHERE_R,
            self.CUBE_HALF_EXTENT - self.CUBE_SPHERE_R,
            4,
        )
        xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
        cube_centers = np.stack(
            [xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
        cube_radii = np.full(
            cube_centers.shape[0], self.CUBE_SPHERE_R, dtype=np.float32)

        self.lattice_body = builder.add_body(
            mass=0.0,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.CUBE_SPAWN_Z),
                q=wp.quat_identity(),
            ),
        )
        builder.add_shape_box(
            self.lattice_body,
            hx=self.CUBE_HALF_EXTENT,
            hy=self.CUBE_HALF_EXTENT,
            hz=self.CUBE_HALF_EXTENT,
        )
        builder.add_lattice(
            link=self.lattice_body,
            morphit_json={"centers": cube_centers, "radii": cube_radii},
            total_mass=0.0,
            pos=wp.vec3(0.0, 0.0, self.CUBE_SPAWN_Z),
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.0
        self.model.soft_contact_mu = 0.0

        self.solver = newton.solvers.SolverUXPBD(
            self.model,
            iterations=6,
            fluid_iterations=4,
            stabilization_iterations=2,   # double pass for fluid-on-ground
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        substrate = self.model.particle_substrate.numpy()
        self._fluid_idx = np.where(substrate == 3)[0].astype(np.int32)

        self._fluid_top_z = fluid_top_z
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
        # 1. Cube settled below its spawn height (it sank/displaced fluid).
        body_z = float(self.state_0.body_q.numpy()[0, 2])
        assert body_z < self.CUBE_SPAWN_Z - 0.10, (
            f"Cube did not fall significantly: z={body_z:.4f} "
            f"vs spawn {self.CUBE_SPAWN_Z}"
        )
        # 2. Cube did not pierce the ground.
        assert body_z > 0.02, f"Cube penetrated ground: z={body_z:.4f}"
        # 3. Cube near rest.
        body_v = float(np.linalg.norm(self.state_0.body_qd.numpy()[0, :3]))
        assert body_v < 1.0, f"Cube still moving: |v|={body_v:.3f} m/s"

        # 4. Fluid did not launch (no particle above 2x spawn pool top).
        fluid_q = self.state_0.particle_q.numpy()[self._fluid_idx]
        z_max_fluid = float(fluid_q[:, 2].max())
        assert z_max_fluid < self.CUBE_SPAWN_Z + 0.10, (
            f"Fluid splashed too high: z_max={z_max_fluid:.4f}"
        )
        # 5. Fluid did not penetrate ground.
        z_min_fluid = float(fluid_q[:, 2].min())
        assert z_min_fluid > -0.02, (
            f"Fluid penetrated ground: z_min={z_min_fluid:.4f}"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
