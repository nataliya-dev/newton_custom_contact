# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Bunny Push
#
# Visualizes the PBD-R Test 4 bunny-push benchmark: the Stanford Bunny
# (sphere-packed, m=2.18 kg) on a frictional ground (mu=0.4) is pushed
# horizontally by a constant F=10 N force at its center of mass. Analytical
# free-body trajectory of the COM is
#
#     x(t) = x(0) + 0.5 * (F - mu * M * g) / M * t^2
#
# Mirrors newton/tests/test_solver_uxpbd_phase2.py::test_pbdr_t4_pushed_bunny.
# Exercises the Phase 2 cross-substrate (SM-rigid <-> static shape) contact
# path with Coulomb friction on a non-axis-aligned, non-uniform shape.
#
# Command: python -m newton.examples uxpbd_bunny_push
###########################################################################


import numpy as np
import warp as wp

import newton
import newton.examples


def _build_pbdr_bunny(builder, pos=(0.0, 0.0, 0.15)):
    """PBD-R reference bunny (Stanford Bunny, m=2.18 kg).

    Matches the helper in test_solver_uxpbd_phase2._build_pbdr_bunny so the
    visualization sees exactly the geometry the analytical test asserts on.
    """
    return builder.add_particle_volume(
        volume_data="assets/bunny-lowpoly/morphit_results.json",
        total_mass=2.18,
        pos=wp.vec3(*pos),
    )


class Example:
    # Analytical-trajectory parameters, mirroring test_pbdr_t4_pushed_bunny.
    F = 10.0          # horizontal push force [N]
    M = 2.18          # total bunny mass [kg]
    MU = 0.4          # ground friction
    G = 9.81          # gravity magnitude [m/s^2]
    TOL = 0.10        # 10% relative tolerance, same as the test
    CLEARANCE = 0.0   # initial gap between bunny's lowest sphere and the ground [m]

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
        self.bunny_group = _build_pbdr_bunny(builder)

        self.model = builder.finalize()
        self.model.particle_mu = self.MU
        self.model.soft_contact_mu = self.MU
        # Kernel uses mu = 0.5 * (particle_mu + shape_material_mu[shape]); the
        # default shape mu is 0.5, so override every shape to get a true MU.
        self.model.shape_material_mu.assign(
            np.full(self.model.shape_count, self.MU, dtype=np.float32)
        )

        # Seat the bunny so its lowest sphere sits CLEARANCE above the ground.
        # The asset spawns ~15 cm up because of the test's pos=(0,0,0.15); this
        # shift undoes that offset. Must happen BEFORE solver construction so
        # SolverUXPBD captures particle_q_rest at the seated pose.
        pq = self.model.particle_q.numpy()
        pr = self.model.particle_radius.numpy()
        lowest_z_before = float(np.min(pq[:, 2] - pr))
        pq[:, 2] += self.CLEARANCE - lowest_z_before
        self.model.particle_q.assign(pq)
        lowest_z_after = float(np.min(self.model.particle_q.numpy()[:, 2] - pr))
        print(
            f"[uxpbd_bunny_push] bunny lowest-point z: "
            f"{lowest_z_before:.4f} -> {lowest_z_after:.4f} m "
            f"(target CLEARANCE={self.CLEARANCE:.4f})"
        )

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()

        # Constant horizontal force, F split evenly across the bunny particles
        # so the net force on the COM is (F, 0, 0).
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
        assert rel_err < self.TOL, (
            f"bunny-push COM x={com_x:.4f}, expected={expected:.4f} after "
            f"{self.sim_time:.2f}s (rel_err={rel_err:.4f})"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
