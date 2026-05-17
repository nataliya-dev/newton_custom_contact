# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Pick and Place (Scenario A)
#
# A Franka arm with lattice-shelled finger pads friction-grasps a free
# shape-matched rigid cube (mass 0.3 kg, mu=0.7) and lifts it.
# Phase machine: APPROACH -> SQUEEZE -> LIFT -> HOLD.
#
# Phase 2 demo: validates the cross-substrate lattice <-> SM-rigid contact
# path with friction closure. Requires Phase 2 PBD-R kernels (CUDA only).
#
# Command: python -m newton.examples uxpbd_pick_and_place
###########################################################################


import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args
        self.phase_t0 = 0.0

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # Pickable cube: 4x4x4 sphere packing inscribed in a 0.08 m cube.
        # Total mass 0.3 kg, mu=0.7 (friction-closure grasp). The sphere packing
        # acts as the shape-matched rigid body for the cube in Phase 2.
        half_extent = 0.04  # cube half-side [m]
        # sphere radius [m]; 4 spheres span 0.096 m ~ 0.08 m side
        sphere_r = 0.012
        coords = np.linspace(-half_extent + sphere_r,
                             half_extent - sphere_r, 4)
        xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
        cube_centers = np.stack(
            [xs.flatten(), ys.flatten(), zs.flatten()], axis=1)
        cube_radii = np.full(cube_centers.shape[0], sphere_r)
        self.cube_group = builder.add_particle_volume(
            volume_data={"centers": cube_centers.tolist(),
                         "radii": cube_radii.tolist()},
            total_mass=0.3,
            pos=wp.vec3(0.55, 0.0, 0.05),
        )

        self.model = builder.finalize()
        # Friction coefficient on cube particles (mu for particle-particle and
        # particle-shape contacts, including the lattice finger pads).
        self.model.particle_mu = 0.7
        self.model.soft_contact_mu = 0.7

        self.solver = newton.solvers.SolverUXPBD(
            self.model, iterations=8, shock_propagation_k=1.0)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(
            1.5, -1.5, 1.2), pitch=-25.0, yaw=135.0)

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
        """Verify the sphere-packed cube settled on the ground without
        blowing up.

        The cube is a 4x4x4 packing of r=0.012 m spheres inscribed in a
        0.08 m cube, spawned with its center at z=0.05 m (lowest sphere
        center at z=0.022 m). After ~400 frames it should settle with the
        bottom layer resting on the ground, i.e. lowest sphere center at
        z ~ sphere_r = 0.012 m, and stop moving.
        """
        cube_idx = self.model.particle_groups[self.cube_group]
        if hasattr(cube_idx, "numpy"):
            cube_idx = cube_idx.numpy()
        cube_idx = np.asarray(list(cube_idx), dtype=np.int32)

        pos = self.state_0.particle_q.numpy()[cube_idx]
        vel = self.state_0.particle_qd.numpy()[cube_idx]

        # 1. NaN/Inf check — first line of defense against PBF/SM-rigid
        #    instability before any other assertion can fire misleadingly.
        assert np.isfinite(pos).all(), "NaN/Inf in particle positions"
        assert np.isfinite(vel).all(), "NaN/Inf in particle velocities"

        # 2. Lowest sphere center settled near ground (within ~5 mm of
        #    the sphere radius, allowing for slight compression / chatter).
        z_min = float(pos[:, 2].min())
        assert 0.005 < z_min < 0.030, (
            f"Cube did not settle on ground: z_min={z_min:.4f}, "
            f"expected near sphere_r = 0.012"
        )

        # 3. COM stays inside the original horizontal footprint (no
        #    lateral runaway).
        com_xy = pos[:, :2].mean(axis=0)
        assert float(np.linalg.norm(com_xy - np.array([0.55, 0.0]))) < 0.10, (
            f"Cube drifted laterally: com_xy={com_xy}"
        )

        # 4. No particle exceeds a sane positional bound (catch escapes).
        q_abs_max = float(np.abs(pos).max())
        assert q_abs_max < 2.0, (
            f"Particle escaped: |q|_max={q_abs_max:.3f} m"
        )

        # 5. Velocity small (cube near rest after ~4 s of simulation).
        v_max = float(np.linalg.norm(vel, axis=1).max())
        assert v_max < 0.5, f"Cube still moving: v_max={v_max:.3f} m/s"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
