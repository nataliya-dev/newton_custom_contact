# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Box Push
#
# Visualizes the PBD-R Test 1 box-push benchmark: a 4x4x4 sphere-packed
# rigid box on a frictional ground (mu=0.4) is pushed horizontally by a
# constant F=17 N force at its center of mass. The analytical trajectory is
#
#     x(t) = 0.5 * (F - mu * M * g) / M * t^2
#
# Mirrors newton/tests/test_solver_uxpbd_phase2.py::test_pbdr_t1_pushed_box.
# Exercises the Phase 2 cross-substrate (SM-rigid <-> static shape) contact
# path with Coulomb friction.
#
# Command: python -m newton.examples uxpbd_box_push
###########################################################################


import numpy as np
import warp as wp

import newton
import newton.examples


def _build_pbdr_box(builder, pos=(0.0, 0.0, 0.10), rot=None):
    """PBD-R reference box: 4x4x4 = 64 spheres, m=4 kg, edge=0.20 m.

    Matches the helper in test_solver_uxpbd_phase2._build_pbdr_box so that the
    visualization sees exactly the geometry the analytical test asserts on.
    """
    if rot is None:
        rot = wp.quat_identity()
    half_extent = 0.10
    num_spheres = 4
    radius_mean = half_extent / num_spheres        # 0.025
    cell_x = (2.0 * half_extent - 2.0 * radius_mean) / (num_spheres - 1)  # 0.05
    total_mass = 4.0
    mass_per_particle = total_mass / (num_spheres ** 3)
    pos_corner = wp.vec3(*pos) + wp.quat_rotate(
        rot,
        wp.vec3(
            -half_extent + radius_mean,
            -half_extent + radius_mean,
            -half_extent + radius_mean,
        ),
    )
    start_idx = len(builder.particle_q)
    builder.add_particle_grid(
        pos=pos_corner,
        rot=rot,
        vel=wp.vec3(0.0),
        dim_x=num_spheres,
        dim_y=num_spheres,
        dim_z=num_spheres,
        cell_x=cell_x,
        cell_y=cell_x,
        cell_z=cell_x,
        mass=mass_per_particle,
        jitter=0.0,
        radius_mean=radius_mean,
        radius_std=0.0,
    )
    end_idx = len(builder.particle_q)
    # add_particle_grid does not register a particle group; shape matching needs one.
    group_id = builder.particle_group_count
    builder.particle_group_count += 1
    for i in range(start_idx, end_idx):
        builder.particle_group[i] = group_id
    builder.particle_groups[group_id] = list(range(start_idx, end_idx))
    return group_id


class Example:
    # Analytical-trajectory parameters, mirroring test_pbdr_t1_pushed_box.
    F = 17.0    # horizontal push force [N]
    M = 4.0     # total cube mass [kg]
    MU = 0.4    # ground friction
    G = 9.81    # gravity magnitude [m/s^2]

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps  # 1e-3 s, matches the test
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()
        self.cube_group = _build_pbdr_box(builder)

        self.model = builder.finalize()
        self.model.particle_mu = self.MU
        self.model.soft_contact_mu = self.MU
        # Kernel uses mu = 0.5 * (particle_mu + shape_material_mu[shape]); the
        # default shape mu is 0.5, so override every shape to get a true MU.
        self.model.shape_material_mu.assign(
            np.full(self.model.shape_count, self.MU, dtype=np.float32)
        )

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        # Constant horizontal force, F split evenly across the 64 particles so
        # the net wrench at the COM is (F, 0, 0).
        force_np = np.zeros((self.model.particle_count, 3), dtype=np.float32)
        force_np[:, 0] = self.F / self.model.particle_count
        self._force_np = force_np

        # COM x at t=0, used by test_final to check x(t) - x(0) matches theory.
        self._initial_com_x = float(self.state_0.particle_q.numpy().mean(axis=0)[0])

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.5, -1.5, 1.2), pitch=-25.0, yaw=135.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_0.particle_f.assign(self._force_np)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
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
        a = (self.F - self.MU * self.M * self.G) / self.M
        expected = self._initial_com_x + 0.5 * a * self.sim_time ** 2
        com_x = float(self.state_0.particle_q.numpy().mean(axis=0)[0])
        rel_err = abs(com_x - expected) / max(abs(expected), 1e-6)
        assert rel_err < 0.05, (
            f"box-push COM x={com_x:.4f}, expected={expected:.4f} after "
            f"{self.sim_time:.2f}s (rel_err={rel_err:.4f})"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
