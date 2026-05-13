# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import ModelBuilder, State, eval_fk


class Example:

    def add_tet_box(self, scene):
        drop_z = 0.3
        N = 1
        particles_per_cell = 2
        dim_x = dim_y = dim_z = N * particles_per_cell
        voxel_size = 0.1
        particle_spacing = voxel_size / particles_per_cell
        radius = 0.05
        half_extent = 0.5 * ((dim_x - 1) * particle_spacing + 2.0 * radius)
        box_center = wp.vec3(-0.2, -0.7, 0.3)

        mass = 4.0
        body_box = scene.add_body(
            mass=mass,
            xform=wp.transform(p=box_center, q=wp.quat_identity()),
            key="box",
        )

        scene.add_shape_box(
            body_box,
            hx=half_extent,
            hy=half_extent,
            hz=half_extent,
    )
        return scene
    
    def __init__(self, viewer):
        # simulation params
        self.sim_substeps = 15
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.scene = ModelBuilder()

        # ---------- Robot ----------
        franka = ModelBuilder()
        self.create_articulation(franka)
        self.scene.add_builder(franka)

        # ---------- Table ----------
        self.scene.add_shape_box(
            -1,
            wp.transform(
                wp.vec3(0.0, -0.5, 0.1),
                wp.quat_identity(),
            ),
            hx=0.4,
            hy=0.4,
            hz=0.1,
        )

        # ---------- Particle Box ----------
        self.scene = self.add_tet_box(self.scene)

        # ---------- Ground ----------
        self.scene.add_ground_plane()

        # ---------- Finalize model ----------
        self.model = self.scene.finalize(requires_grad=False)

        self.state_0: State = self.model.state()
        self.state_1: State = self.model.state()

        self.control = self.model.control()

        # TODO
        # self.model.particle_ke = 1.0e4
        # self.model.particle_kd = 0.0
        # self.model.particle_kf = 0.0
        # self.model.particle_mu = 0.5
        # self.model.particle_adhesion = 0.0
        # self.model.soft_contact_ke = 1.0e2
        # self.model.soft_contact_kd = 1.0
        # self.model.soft_contact_kf = 0.0
        # self.model.soft_contact_mu = 0.5

        # Initialize solvers
        self.robot_solver = newton.solvers.SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.particle_solver = newton.solvers.SolverXPBD(
            self.model, 
            iterations=self.sim_substeps,
        )

        self.viewer.set_model(self.model)

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)
        
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def create_articulation(self, builder: ModelBuilder):
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (-0.5, -0.5, -0.1),
                wp.quat_identity(),
            ),
            floating=False,
            scale=1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:9] = [
            0.0,
            -0.785398,
            -1.0,
            -2.356194,
            0.0,
            1.570796,
            0.785398,
            0.0,  # finger 1
            0.0,  # finger 2
        ]

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            # -------- Manually set joint velocities --------
            qd = self.state_0.joint_qd.numpy()
            t = self.sim_time
            qd[:] = 0.0
            qd[0] = 0.5 * np.sin(t)
            self.state_0.joint_qd.assign(qd)
            # -----------------------------------------------

            particle_count = self.model.particle_count
            self.model.particle_count = 0

            self.model.gravity.assign(self.gravity_zero)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            # self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            self.contacts = self.model.collide(self.state_0) # TODO: why state_0?
            self.particle_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 2.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    # wp.config.debug = True
    example = Example(viewer)
    newton.examples.run(example, args)

# WARP_BACKTRACE=1 PYTHONFAULTHANDLER=1 uv run -m newton.examples fr_sp_box