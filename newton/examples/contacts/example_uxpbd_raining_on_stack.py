# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Raining On Stack
#
# A grid of loose particles ("rain") is dropped onto a lattice-clad
# articulated rigid body sitting on the ground. Inspired by the
# UPPFRTA-style "particles on cloth" combo (Macklin 2014 fig. 1) and
# the C++ reference's Key_6 scene; UXPBD lacks cloth in v1, so the
# rigid lattice substitutes.
#
# Tests three substrates simultaneously:
#   - lattice particles (substrate 0) on the host body
#   - loose particles (substrate 1, no shape-matched group) raining
#     down via add_particle_grid
#   - the cross-substrate particle-particle contact path
#     (solve_particle_particle_contacts_uxpbd) routing the rain hits
#     into the lattice host body's wrench (per design spec sect 5.4)
#
# Layout (Z up):
#
#         . . . . .       loose particles (rain) at z=0.40
#         . . . . .
#                        (free fall under gravity)
#         . . . . .
#         |#####|         lattice cube at z=0.04 (resting on ground)
#         |#####|
#         ============== ground (z=0)
#
# Command: python -m newton.examples uxpbd_raining_on_stack
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    CUBE_HALF_EXTENT = 0.04
    CUBE_SPHERE_R = 0.012

    RAIN_DIM = 5                  # 5x5 columns
    RAIN_LAYERS = 3               # 3 layers tall = 75 rain particles total
    RAIN_R = 0.012
    RAIN_CELL = 0.024             # 2x rain particle diameter -> spaced apart
    RAIN_BASE_Z = 0.40
    RAIN_PARTICLE_MASS = 0.01     # 10 g per rain particle

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

        # ----- Lattice cube on the ground -----
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

        # Spawn slightly above the rest height (0.04) so the body
        # actively settles in the first few frames before the rain
        # arrives -- exercises the stabilization sub-loop on the lattice
        # path even though rain hits dominate later.
        self.lattice_body = builder.add_body(
            mass=0.0,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, 0.06),
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
            pos=wp.vec3(0.0, 0.0, 0.06),
        )

        # ----- Rain: a grid of loose particles falling from above -----
        x_min = -(self.RAIN_DIM - 1) * self.RAIN_CELL / 2
        y_min = -(self.RAIN_DIM - 1) * self.RAIN_CELL / 2
        builder.add_particle_grid(
            pos=wp.vec3(x_min, y_min, self.RAIN_BASE_Z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.RAIN_DIM, dim_y=self.RAIN_DIM, dim_z=self.RAIN_LAYERS,
            cell_x=self.RAIN_CELL, cell_y=self.RAIN_CELL, cell_z=self.RAIN_CELL,
            mass=self.RAIN_PARTICLE_MASS,
            jitter=0.0,
            radius_mean=self.RAIN_R,
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.0
        self.model.soft_contact_mu = 0.0

        self.solver = newton.solvers.SolverUXPBD(
            self.model,
            iterations=6,
            fluid_iterations=0,           # no fluid in this scene
            stabilization_iterations=2,   # helps the lattice settle on the ground
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        # Rain particles are everything that's not on the lattice.
        substrate = self.model.particle_substrate.numpy()
        # Substrate 0 = lattice; rain particles default to substrate 1
        # (loose / SM-rigid path). add_particle_grid does not register
        # a shape-matching group, so they behave as ungrouped solids.
        self._rain_idx = np.where(substrate != 0)[0].astype(np.int32)

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
        # 1. Lattice body is still on the ground (rain didn't launch it).
        body_z = float(self.state_0.body_q.numpy()[0, 2])
        # The cube's lattice rest height is z = 0.04. Allow tolerance for
        # rain pile pressing it slightly down (numerical compression in
        # the contact-PBF chain) -- it should NOT have flown off the
        # ground or sunk through it.
        assert 0.02 < body_z < 0.10, (
            f"Lattice body off ground or sunk: z={body_z:.4f}"
        )
        # 2. Rain particles all came down (no rain still floating above
        #    its initial spawn height, which would mean a launch event).
        rain_q = self.state_0.particle_q.numpy()[self._rain_idx]
        z_max_rain = float(rain_q[:, 2].max())
        spawn_top = self.RAIN_BASE_Z + self.RAIN_LAYERS * self.RAIN_CELL
        assert z_max_rain < spawn_top, (
            f"Rain particle launched above spawn: z_max={z_max_rain:.4f} "
            f"vs spawn_top={spawn_top:.4f}"
        )
        # 3. Rain accumulated above ground (some rest on lattice top
        #    or ground around it).
        z_min_rain = float(rain_q[:, 2].min())
        assert z_min_rain > -0.02, (
            f"Rain particle penetrated ground: z_min={z_min_rain:.4f}"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
