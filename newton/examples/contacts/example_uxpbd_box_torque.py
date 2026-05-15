# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Box Torque
#
# Visualizes the PBD-R Test 2 box-torque benchmark: a 4x4x4 sphere-packed
# rigid box floats in zero gravity with mu=0 while a tangential force
# pattern applies a net torque tau=0.01 N*m about the +z axis. The cube
# should spin about z with the analytical free-body trajectory
#
#     theta(t) = 0.5 * (tau / I_zz) * t^2
#
# Mirrors newton/tests/test_solver_uxpbd_phase2.py::test_pbdr_t2_box_torque.
# Note: that test approximates I_zz as (2/3)*M*h^2 with h=0.11, which is the
# uniform-solid-cube formula. The actual sphere-packed cube's I_zz computed
# directly from the 64 particle centers is 0.0250 kg*m^2, ~29% smaller, so
# the test's expected angle is off by ~29%. This example uses the directly
# computed I_zz (matching _compute_principal_inertia_zz in the t5 bunny
# torque test) and is therefore the analytically faithful reference.
#
# Command: python -m newton.examples uxpbd_box_torque
###########################################################################


import numpy as np
import warp as wp

import newton
import newton.examples


def _build_pbdr_box(builder, pos=(0.0, 0.0, 0.10), rot=None):
    """PBD-R reference box: 4x4x4 = 64 spheres, m=4 kg, edge=0.20 m.

    Matches the helper in test_solver_uxpbd_phase2._build_pbdr_box so the
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


def _compute_principal_inertia_zz(particle_q_np, particle_mass_np):
    """I_zz = sum_i m_i * (x_i^2 + y_i^2) about the COM."""
    M = particle_mass_np.sum()
    com = (particle_mass_np[:, None] * particle_q_np).sum(axis=0) / M
    r = particle_q_np - com
    return float(np.sum(particle_mass_np * (r[:, 0] ** 2 + r[:, 1] ** 2)))


class Example:
    # Analytical-trajectory parameters, mirroring test_pbdr_t2_box_torque.
    TAU = 0.01  # net torque about +z [N*m]

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps  # 1e-3 s, matches the test
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        # Float the cube high above the origin so gravity-off and no-ground are
        # visually unambiguous; pure-rotation dynamics are translation-invariant.
        self.cube_group = _build_pbdr_box(builder, pos=(0.0, 0.0, 5.0))

        self.model = builder.finalize()
        # Zero gravity, mu=0 -> free rotation, no friction dissipation.
        self.model.gravity.assign(np.zeros_like(self.model.gravity.numpy()))
        self.model.particle_mu = 0.0

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()  # unused (no shapes), kept for renderer

        # ----- Force pattern that produces a pure-z torque of magnitude TAU -----
        # For each particle: f_i = (TAU / sum_j ||r_j,xy||) * t_hat_i, where
        # t_hat_i = (-r_iy, r_ix, 0) / ||r_i,xy||. Then
        #   tau_z = sum_i (r_i x f_i)_z = (TAU / sum_j ||r_j,xy||) * sum_i ||r_i,xy|| = TAU.
        # Net force is zero by the (x,y) -> (-y,x) symmetry of the pattern.
        initial_pos = self.state_0.particle_q.numpy()
        com = initial_pos.mean(axis=0)
        r = initial_pos - com
        tangent = np.stack([-r[:, 1], r[:, 0], np.zeros_like(r[:, 0])], axis=1)
        tan_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tan_norm = np.where(tan_norm > 1e-9, tan_norm, 1.0)
        tangent = tangent / tan_norm
        r_xy = np.linalg.norm(r[:, :2], axis=1)
        f_mag = self.TAU / float(np.sum(r_xy))
        self._force_np = (tangent * f_mag).astype(np.float32)

        # ----- True principal inertia, computed directly from particle layout -----
        self._I_zz = _compute_principal_inertia_zz(
            initial_pos, self.model.particle_mass.numpy()
        )
        self._alpha = self.TAU / self._I_zz  # rad/s^2

        # Reference particle (index 0, a corner sphere) tracked for angle readout.
        self._p0_init_rel = (initial_pos[0] - com).astype(np.float64)
        self._theta_init = float(np.arctan2(self._p0_init_rel[1], self._p0_init_rel[0]))

        print(
            f"[uxpbd_box_torque] I_zz (from particles) = {self._I_zz:.6f} kg*m^2, "
            f"alpha = {self._alpha:.4f} rad/s^2 "
            f"(test approximation I_zz=2/3*M*0.11^2 would give "
            f"{(2.0/3.0)*4.0*0.11**2:.6f} -> alpha={self.TAU/((2.0/3.0)*4.0*0.11**2):.4f})"
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        # Look down the +z axis from a tilted angle so the rotation is obvious.
        self.viewer.set_camera(pos=wp.vec3(1.2, -1.2, 5.8), pitch=-25.0, yaw=135.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_0.particle_f.assign(self._force_np)
            # No model.collide(): no ground, no shapes, so contacts are unused.
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _measured_theta(self):
        """Unwrap the corner-particle angle to the branch nearest the expected."""
        final_pos = self.state_0.particle_q.numpy()
        com_f = final_pos.mean(axis=0)
        p_now = final_pos[0] - com_f
        theta_now = float(np.arctan2(p_now[1], p_now[0]))
        theta = theta_now - self._theta_init
        expected = 0.5 * self._alpha * self.sim_time ** 2
        while theta < expected - np.pi:
            theta += 2.0 * np.pi
        while theta > expected + np.pi:
            theta -= 2.0 * np.pi
        return theta, expected

    def test_final(self):
        theta, expected = self._measured_theta()
        rel_err = abs(theta - expected) / max(abs(expected), 1e-6)
        assert rel_err < 0.05, (
            f"box-torque theta={theta:.4f} rad, expected={expected:.4f} rad after "
            f"{self.sim_time:.2f}s (rel_err={rel_err:.4f}, I_zz={self._I_zz:.6f})"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
