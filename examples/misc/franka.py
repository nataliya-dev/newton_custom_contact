# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Simple Franka
#
# This simulation just loads a Franka arm and drives it by manually
# setting joint_qd every step (no cloth, no Jacobian control).
#
# Command: python -m newton.examples simple_franka
#
###########################################################################

from __future__ import annotations

import numpy as np
from newton._src.solvers.style3d import builder
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone


class Example:
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

        # ---------- Ground ----------
        self.scene.add_ground_plane()

        # ---------- Finalize model ----------
        self.model = self.scene.finalize(requires_grad=False)

        self.state_0: State = self.model.state()
        self.state_1: State = self.model.state()

        self.control = self.model.control()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.viewer.set_model(self.model)
        
        # gravity in m/s^2
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
            scale=1,  # unit: meters
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        # builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
        builder.joint_q[:9] = [
            0.0,
        -0.785398,
            -1.0,
        -2.356194,
            0.0,
            1.570796,
            0.785398,
            0.04,       # finger 1
            0.04,       # finger 2
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
            qd[1] = 0.5 * np.sin(t)
            self.state_0.joint_qd.assign(qd)
            # -----------------------------------------------

            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)

            self.contacts = self.model.collide(self.state_0)

            self.robot_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test(self):
        # Just a simple sanity check on body velocities
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 2.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer)
    newton.examples.run(example, args)
