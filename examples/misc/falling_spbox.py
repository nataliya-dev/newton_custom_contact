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
# Example Basic Shapes
#
# Shows how to programmatically creates a variety of
# collision shapes using the newton.ModelBuilder() API.
#
# Command: python -m newton.examples basic_shapes
#
###########################################################################
import warp as wp
import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        self.sim_time = 0.0
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        drop_z = 1.0

        N = 2
        particles_per_cell = 3
        dim_x = N * particles_per_cell
        dim_y = N * particles_per_cell
        dim_z = N * particles_per_cell
        voxel_size = 0.5
        particle_spacing = voxel_size / particles_per_cell
        total_mass = 216.0
        mass_per_particle = total_mass / (dim_x * dim_y * dim_z)

        builder.add_particle_grid(
            pos=wp.vec3(0, 0, drop_z),
            rot = wp.quat(0.1830127, 0.1830127, 0.6830127, 0.6830127),
            vel=wp.vec3(0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            cell_z=particle_spacing,
            mass=mass_per_particle,
            jitter=0.0,
        )

        self.model = builder.finalize()

        # TODO
        # self.model.particle_ke = 1.0e15
        # self.model.particle_mu = 0.6
        # self.model.soft_contact_ke = 1.0e2
        # self.model.soft_contact_kd = 1.0e0
        # self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverSRXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    viewer.show_particles = True
    example = Example(viewer)
    newton.examples.run(example, args)
