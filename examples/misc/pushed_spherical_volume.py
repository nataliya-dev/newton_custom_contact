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
# Example of spherical volume being "pushed" by a constant horizontal force
#
#
# Command: uv run -m examples.pushed_spherical_volume
# if having issues with OpenGL on WSL2, try setting the environment variable PYOPENGL_PLATFORM=glx
###########################################################################
import warp as wp
import newton
import newton.examples


@wp.kernel
def apply_constant_force(
    particle_force_array: wp.array(dtype=wp.vec3),
    const_f_total: wp.vec3,
    particle_mass_array: wp.array(dtype=float),
    total_mass: float,
):
    """
    Kernel fucntion to apply forces to particles
    
    Args:
        particle_force_array: Array of force vectors (output)
        constant_f_total: Total force being applied to the volume
        particle_mass_array: Array of particle masses
        total_mass: Total mass of the volume
    """
    i = wp.tid()

    # Apply force to each particle proportional to its mass fraction
    mass_ratio = particle_mass_array[i] / total_mass

    # Using += ensures gravity is not overwritten
    particle_force_array[i] += mass_ratio * const_f_total


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
        # Should correspond to the radii of the sphere that you want resting on the plane
        drop_z = 0.1
        # Total mass of the volume
        self.total_mass = 10.0

        # Orientation of the volume wrt world
        orientation = wp.quat_identity()

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
            total_mass=self.total_mass
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverSRXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Total constant force to apply to the volume
        self.const_force = wp.vec3(50.0, 0.0, 0.0)

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
            self.viewer.apply_forces(self.state_0)

            n = self.state_0.particle_f.shape[0]
            wp.launch(
                kernel=apply_constant_force,
                dim=n,
                inputs=[self.state_0.particle_f, self.const_force, self.model.particle_mass, self.total_mass],
                device=wp.get_device(),
            )

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
