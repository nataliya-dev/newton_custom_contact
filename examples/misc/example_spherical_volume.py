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
# Command: uv run -m newton.examples spherical_volume
# if having issues with OpenGL on WSL2, try setting the environment variable PYOPENGL_PLATFORM=glx
###########################################################################
import math

import numpy as np
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

        # Initial height (defines the origin of the volume)
        drop_z = 1.0
        # Total mass of the volume
        total_mass = 15.0

        # Orientation of the volume wrt world
        orientation = wp.quat(0.1830127, 0.1830127, 0.6830127, 0.6830127)

        # Test using example from MorphIt/src/main.py using pink_box.obj as the input and num_spheres=100
        # morphit_spheres = "/home/joe/workspace/Particles/MorphIt-1/src/results/output/morphit_results.json"
        # Alternatively, can test by manually defining sphere data
        morphit_spheres = {
            "centers": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0.5]],
            "radii": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]
        }

        builder.add_particle_volume(
            volume_data=morphit_spheres,
            pos=wp.vec3(0, 0, drop_z),
            rot = orientation,
            vel=wp.vec3(0.0),
            total_mass=total_mass
        )

        # Second body: the bunny mesh as a single rigid body. The upstream variant loaded a
        # bunny URDF that isn't shipped in this repo; load the .obj directly instead.
        import trimesh as _trimesh
        _bunny_obj = "assets/bunny-lowpoly/Bunny-LowPoly-ws.obj"
        _tm = _trimesh.load(_bunny_obj, force="mesh")
        bunny_mesh = newton.Mesh(
            np.asarray(_tm.vertices, dtype=np.float32),
            np.asarray(_tm.faces, dtype=np.int32).flatten(),
        )
        _bunny_xform = wp.transform(
            p=wp.vec3(0.0, 0.0, 1.0),
            q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0),
        )
        _bunny_body = builder.add_body(xform=_bunny_xform)
        builder.add_shape_mesh(_bunny_body, mesh=bunny_mesh)

        # Third body: same bunny mesh, sphere-packed for the SRXPBD solver.
        builder.manual_sphere_packing(
            _bunny_obj,
            radius=0.005,
            spacing=0.010,
            total_mass=total_mass,
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0),
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverSRXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

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
