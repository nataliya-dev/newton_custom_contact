# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Combo
#
# A side-by-side stress test of every substrate SolverUXPBD currently
# supports, all in a single scene with a single ground plane:
#
#   substrate 0 (lattice)     -- a lattice-clad rigid body falls and rests
#   substrate 1 (SM-rigid)    -- a free shape-matched cube falls and rests
#   substrate 1 (SM-rigid)    -- a particle-volume "ball" falls and rests
#   substrate 3 (fluid)       -- a PBF fluid block falls and spreads to a
#                                puddle without launching upward
#
# Layout (Z up, looking from +Y):
#
#         lattice       SM cube        SM ball         fluid
#         (x=-0.30)     (x=-0.10)      (x=+0.10)       (x=+0.30)
#         |####|        |####|         (####)          ::::::
#         |####|        |####|         (####)          ::::::
#         ============================================ ground (z=0)
#
# Inspired by the combo scene in the UPPFRTA-style reference particle
# solver (cslc_xpbd/papers/uppfrta_preprint.pdf, fig. 1; also the PBD-R
# Stanford-bunny pile in fig. 9). Demonstrates that the lattice contact
# pipeline (per-body wrench averaging, design spec sect 5.7), the
# SM-rigid shape-match + momentum-conservation pass (PBD-R sect 5),
# and the PBF fluid pipeline (Macklin & Muller 2013, with the s_corr
# gate and per-iteration position-delta clamp from fluid.py) all
# coexist without the substrates fighting each other.
#
# Command: python -m newton.examples uxpbd_combo
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    # ---- Shared geometry for the lattice body and the SM-rigid cube ----
    CUBE_HALF_EXTENT = 0.04
    CUBE_SPHERE_R = 0.012
    CUBE_TOTAL_MASS = 0.512   # = 1000 kg/m^3 * (2 * 0.04)^3, matches lattice body

    # ---- Spawn positions (Z up) ----
    LATTICE_X = -0.30
    SM_CUBE_X = -0.10
    SM_BALL_X = 0.10
    FLUID_X = 0.30

    LATTICE_SPAWN_Z = 0.20    # lattice body falls a small distance to rest
    SM_CUBE_SPAWN_Z = 0.20    # SM cube falls similarly
    SM_BALL_SPAWN_Z = 0.20
    FLUID_SPAWN_Z = 0.20      # fluid block falls into a puddle

    # ---- Expected resting heights (for test_final assertions) ----
    LATTICE_REST_Z = 0.04     # body_z when bottom lattice sphere just touches ground
    SM_CUBE_REST_Z = 0.04     # SM cube COM at rest with same geometry as lattice
    SM_BALL_REST_Z = 0.04     # SM ball COM at rest

    # ---- Fluid parameters ----
    FLUID_PARTICLE_R = 0.008
    FLUID_CELL = 0.016        # = 2 * particle_radius (touching)

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # 16 substeps: stacked-contact stiffness (lattice + SM-rigid pile)
        # benefits from extra constraint sweeps; PBF fluid path also needs
        # multiple sub-iterations to converge against the ground.
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # ----- Shared 4x4x4 sphere packing for lattice and SM cube -----
        coords = np.linspace(
            -self.CUBE_HALF_EXTENT + self.CUBE_SPHERE_R,
            self.CUBE_HALF_EXTENT - self.CUBE_SPHERE_R,
            4,
        )
        xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
        cube_centers = np.stack(
            [xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
        cube_radii = np.full(
            cube_centers.shape[0], self.CUBE_SPHERE_R, dtype=np.float32)

        # ----- Substrate 0: lattice-clad rigid body -----
        # `add_body(mass=0.0)` is intentional: see example_uxpbd_lattice_drop
        # for the full explanation of why mixing add_body(mass=...) with
        # shape density produces inconsistent body_inv_mass / body_inv_inertia.
        self.lattice_body = builder.add_body(
            mass=0.0,
            xform=wp.transform(
                p=wp.vec3(self.LATTICE_X, 0.0, self.LATTICE_SPAWN_Z),
                q=wp.quat_identity(),
            ),
        )
        builder.add_shape_box(
            self.lattice_body,
            hx=self.CUBE_HALF_EXTENT,
            hy=self.CUBE_HALF_EXTENT,
            hz=self.CUBE_HALF_EXTENT,
        )
        builder.add_lattice(
            link=self.lattice_body,
            morphit_json={"centers": cube_centers, "radii": cube_radii},
            total_mass=0.0,
            pos=wp.vec3(self.LATTICE_X, 0.0, self.LATTICE_SPAWN_Z),
        )

        # ----- Substrate 1: free SM-rigid cube -----
        self.sm_cube_group = builder.add_particle_volume(
            volume_data={"centers": cube_centers.tolist(),
                         "radii": cube_radii.tolist()},
            total_mass=self.CUBE_TOTAL_MASS,
            pos=wp.vec3(self.SM_CUBE_X, 0.0, self.SM_CUBE_SPAWN_Z),
        )

        # ----- Substrate 1: SM-rigid ball (sphere packing) -----
        # 32-particle sphere (~3-shell icosphere style packing). Demonstrates
        # the shape-matching kernel handles non-cubic rest configurations.
        ball_radius = self.CUBE_HALF_EXTENT
        n_ball = 4
        ball_coords = np.linspace(-ball_radius + self.CUBE_SPHERE_R,
                                   ball_radius - self.CUBE_SPHERE_R, n_ball)
        bxs, bys, bzs = np.meshgrid(ball_coords, ball_coords, ball_coords, indexing="ij")
        ball_centers_full = np.stack(
            [bxs.flatten(), bys.flatten(), bzs.flatten()], axis=1).astype(np.float32)
        # Keep only those within ball_radius (sphere mask).
        keep = np.linalg.norm(ball_centers_full, axis=1) <= ball_radius - self.CUBE_SPHERE_R * 0.5
        ball_centers = ball_centers_full[keep]
        ball_radii = np.full(ball_centers.shape[0], self.CUBE_SPHERE_R, dtype=np.float32)
        self.sm_ball_group = builder.add_particle_volume(
            volume_data={"centers": ball_centers.tolist(),
                         "radii": ball_radii.tolist()},
            total_mass=self.CUBE_TOTAL_MASS * (len(ball_centers) / len(cube_centers)),
            pos=wp.vec3(self.SM_BALL_X, 0.0, self.SM_BALL_SPAWN_Z),
        )

        # ----- Substrate 3: PBF fluid block -----
        # 4x4x4 = 64 fluid particles. smoothing_radius_factor=3.0 because at
        # 2.0 the cubic-grid neighbours land on the Poly6 cutoff (W = 0)
        # and PBF cannot generate inter-particle corrections; see
        # example_uxpbd_fluid_drop for the cell == h pitfall write-up.
        builder.add_fluid_grid(
            pos=wp.vec3(self.FLUID_X - 0.024, -0.024, self.FLUID_SPAWN_Z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4, dim_y=4, dim_z=4,
            cell_x=self.FLUID_CELL, cell_y=self.FLUID_CELL, cell_z=self.FLUID_CELL,
            particle_radius=self.FLUID_PARTICLE_R,
            rest_density=1000.0,
            smoothing_radius_factor=3.0,
            viscosity=0.05,
            cohesion=0.0,
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.0
        self.model.soft_contact_mu = 0.0

        # iterations=6: stacked-contact convergence (lattice ground contact,
        # SM-rigid recovery, PBF density solve) all benefit from extra sweeps.
        self.solver = newton.solvers.SolverUXPBD(
            self.model, iterations=6, fluid_iterations=4)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Cache per-substrate particle indices for test_final.
        sm_cube_idx = self.model.particle_groups[self.sm_cube_group]
        if hasattr(sm_cube_idx, "numpy"):
            sm_cube_idx = sm_cube_idx.numpy()
        self._sm_cube_idx = np.asarray(list(sm_cube_idx), dtype=np.int32)

        sm_ball_idx = self.model.particle_groups[self.sm_ball_group]
        if hasattr(sm_ball_idx, "numpy"):
            sm_ball_idx = sm_ball_idx.numpy()
        self._sm_ball_idx = np.asarray(list(sm_ball_idx), dtype=np.int32)

        substrate = self.model.particle_substrate.numpy()
        self._fluid_idx = np.where(substrate == 3)[0].astype(np.int32)

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        # Pull camera back so all four piles are framed.
        self.viewer.set_camera(pos=wp.vec3(0.0, -1.2, 0.5),
                               pitch=-25.0, yaw=90.0)

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
        # Each substrate must reach a physically sensible final state.

        # Substrate 0: lattice body at rest near LATTICE_REST_Z.
        body_q = self.state_0.body_q.numpy()[0]
        body_qd = self.state_0.body_qd.numpy()[0]
        body_z = float(body_q[2])
        assert abs(body_z - self.LATTICE_REST_Z) < 5.0e-2, (
            f"Lattice body did not settle: z={body_z:.4f}, "
            f"expected ~{self.LATTICE_REST_Z:.4f}"
        )
        body_v = float(np.linalg.norm(body_qd[:3]))
        assert body_v < 0.5, f"Lattice body still moving: |v|={body_v:.3f} m/s"

        # Substrate 1: SM cube COM near rest height, near-stationary.
        sm_cube_q = self.state_0.particle_q.numpy()[self._sm_cube_idx]
        sm_cube_v = self.state_0.particle_qd.numpy()[self._sm_cube_idx]
        sm_cube_com_z = float(sm_cube_q[:, 2].mean())
        sm_cube_v_mag = float(np.linalg.norm(sm_cube_v.mean(axis=0)))
        assert abs(sm_cube_com_z - self.SM_CUBE_REST_Z) < 5.0e-2, (
            f"SM cube did not settle: com_z={sm_cube_com_z:.4f}, "
            f"expected ~{self.SM_CUBE_REST_Z:.4f}"
        )
        assert sm_cube_v_mag < 0.5, (
            f"SM cube still moving: |v|={sm_cube_v_mag:.3f} m/s"
        )

        # Substrate 1 (sphere packing): SM ball COM near rest height.
        sm_ball_q = self.state_0.particle_q.numpy()[self._sm_ball_idx]
        sm_ball_v = self.state_0.particle_qd.numpy()[self._sm_ball_idx]
        sm_ball_com_z = float(sm_ball_q[:, 2].mean())
        sm_ball_v_mag = float(np.linalg.norm(sm_ball_v.mean(axis=0)))
        assert abs(sm_ball_com_z - self.SM_BALL_REST_Z) < 5.0e-2, (
            f"SM ball did not settle: com_z={sm_ball_com_z:.4f}, "
            f"expected ~{self.SM_BALL_REST_Z:.4f}"
        )
        assert sm_ball_v_mag < 0.5, (
            f"SM ball still moving: |v|={sm_ball_v_mag:.3f} m/s"
        )

        # Substrate 3: fluid puddled near the ground, did not launch upward.
        # No particle should be above its spawn height (FLUID_SPAWN_Z + 3*cell).
        fluid_q = self.state_0.particle_q.numpy()[self._fluid_idx]
        fluid_v = self.state_0.particle_qd.numpy()[self._fluid_idx]
        z_max_fluid = float(fluid_q[:, 2].max())
        spawn_top = self.FLUID_SPAWN_Z + 3 * self.FLUID_CELL
        assert z_max_fluid < spawn_top, (
            f"Fluid launched upward: z_max={z_max_fluid:.4f}, "
            f"spawn_top={spawn_top:.4f}"
        )
        z_min_fluid = float(fluid_q[:, 2].min())
        assert z_min_fluid > -0.02, (
            f"Fluid penetrated ground: z_min={z_min_fluid:.4f}"
        )
        v_max_fluid = float(np.linalg.norm(fluid_v, axis=1).max())
        assert v_max_fluid < 5.0, (
            f"Fluid still moving fast: v_max={v_max_fluid:.3f} m/s"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
