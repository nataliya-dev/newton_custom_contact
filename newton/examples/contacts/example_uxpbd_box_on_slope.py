# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Box on Slope
#
# Visualizes the PBD-R Test 3 box-on-slope benchmark: a 4x4x4 sphere-packed
# rigid box rests on a frictionless inclined plane (theta=pi/8, mu=0.0) and
# slides downhill under gravity alone. Analytical free-body trajectory of
# the COM along the slope tangent is
#
#     x_slope(t) = 0.5 * g sin(theta) * t^2
#
# Mirrors newton/tests/test_solver_uxpbd_phase2.py::test_pbdr_t3_box_on_slope.
# Frictionless because mu>0 + this slope geometry makes the analytical
# acceleration a = g(sin theta - mu cos theta) a small difference of large
# numbers, so 1% friction-model imprecision amplifies to ~30% trajectory
# error. The friction code is exercised by t1 (pushed_box); this test
# isolates the gravity-driven slide.
#
# Command: python -m newton.examples uxpbd_box_on_slope
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
    # Analytical-trajectory parameters, mirroring test_pbdr_t3_box_on_slope.
    SLOPE_ANGLE = np.pi / 8.0  # 22.5 deg about +Y
    MU = 0.0                   # frictionless slope (see header comment)
    G = 9.81                   # gravity magnitude [m/s^2]
    BOX_HE = 0.10              # box half-extent [m]
    TOL = 0.05                 # 5% relative tolerance, same as the test
    # Slide goes as 0.5 * g sin(theta) * t^2 -> ~188 m at 10 s. Make the slope
    # large enough that the box never falls off during a typical session.
    SLOPE_WIDTH = 500.0        # along x (downhill direction)
    SLOPE_LENGTH = 10.0        # along y

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps  # 1e-3 s, matches the test
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        # Slope rotated about +Y -> surface descends in +x. Slope passes through
        # the origin with width/length 10 m so the box can slide ~7 m without
        # falling off (10 s of simulation reaches that distance analytically).
        slope_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(self.SLOPE_ANGLE))
        builder.add_shape_plane(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=slope_rot),
            width=self.SLOPE_WIDTH,
            length=self.SLOPE_LENGTH,
        )

        # Seat the cube on the slope: COM at perpendicular distance BOX_HE from
        # the slope plane, kept at x=y=0 so the perpendicular drop projects to
        # world z = BOX_HE / cos(theta). Apply the slope rotation to the box
        # so its bottom face lies flush against the slope (no initial drop).
        z_pos = self.BOX_HE / float(np.cos(self.SLOPE_ANGLE))
        self.cube_group = _build_pbdr_box(builder, pos=(0.0, 0.0, z_pos), rot=slope_rot)

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

        # Downhill direction in world frame. For a plane rotated by SLOPE_ANGLE
        # about +Y, the slope normal is (sin, 0, cos) and the downhill tangent
        # (gradient direction of decreasing z on the slope) is (cos, 0, -sin).
        c, s = float(np.cos(self.SLOPE_ANGLE)), float(np.sin(self.SLOPE_ANGLE))
        self._slope_dir = np.array([c, 0.0, -s], dtype=np.float64)

        # Initial COM in world frame, used by test_final as the reference origin
        # for the slide-distance check.
        self._initial_com = self.state_0.particle_q.numpy().mean(axis=0).astype(np.float64)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        # Camera off-axis so the slope tilt and downhill direction are obvious.
        self.viewer.set_camera(pos=wp.vec3(4.0, -3.5, 2.5), pitch=-25.0, yaw=130.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            # No applied force: gravity alone drives the slide.
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
        # Frictionless: a = g sin(theta). The MU term is included so the
        # formula stays valid if someone bumps MU > 0.
        a = self.G * np.sin(self.SLOPE_ANGLE) - self.MU * self.G * np.cos(self.SLOPE_ANGLE)
        expected = 0.5 * a * self.sim_time ** 2
        final_com = self.state_0.particle_q.numpy().mean(axis=0).astype(np.float64)
        x_slide = float(np.dot(final_com - self._initial_com, self._slope_dir))
        rel_err = abs(x_slide - expected) / max(abs(expected), 1e-6)
        assert rel_err < self.TOL, (
            f"box-on-slope slide={x_slide:.4f} m, expected={expected:.4f} m after "
            f"{self.sim_time:.2f}s (rel_err={rel_err:.4f})"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
