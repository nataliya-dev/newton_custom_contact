# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Lattice Stack
#
# Verifies the UXPBD cross-substrate contact pipeline by dropping a free
# shape-matched (SM-rigid) cube directly on top of an articulated rigid
# body that wears a kinematic lattice. The lattice particles AND the
# SM-rigid cube particles use the exact same 4x4x4 sphere packing
# inscribed in a half-extent CUBE_HALF_EXTENT cube; the lattice body also
# carries a collision box of the same half-extent so it has mass and a
# realistic inertia tensor.
#
# Layout (Z up):
#
#       +------+   <-- SM-rigid cube (spawns at z = 0.6)
#       | cube |
#       +------+
#       +------+   <-- lattice-clad rigid body (pre-positioned near rest)
#       |latt  |
#       +------+
#       ========   <-- ground plane
#
# Expected behaviour after ~5 s of simulated time:
#   - Lattice body settles near LATTICE_REST_Z.
#   - Cube settles on top of the lattice body. The lattice spheres are
#     inscribed in the box (sphere tops coplanar with the box face), so
#     the cube contacts both the box face (shape-particle) and the top
#     lattice spheres (particle-particle) roughly simultaneously.
#   - Both objects come to a near-stationary equilibrium (no continuous
#     drift or bouncing).
#
# Command: python -m newton.examples uxpbd_lattice_stack
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    # Both objects spawn directly above the origin. The lattice and the cube
    # use IDENTICAL 4x4x4 sphere packings (same centers, same radii); the
    # UPPFRTA-style per-body contact-count averaging in SolverUXPBD's
    # apply_body_deltas keeps redundant co-located contacts from compounding
    # into a launch impulse, so we don't need an XY offset or a larger
    # lattice radius to break degenerate head-on alignment.
    #
    # TODO(uxpbd Phase 2+): SolverUXPBD currently runs lattice contacts at
    # zero compliance (α=0, infinite per-iteration stiffness). Layer 1
    # (constraint averaging) alone is sufficient for this scene and all
    # current Phase 1 tests, but layer 2 (XPBD compliance with accumulated
    # λ_n, ~1e-7 m/N) will become necessary when:
    #   - convert_lattice_contact_impulse_to_force is wired up
    #     (design doc §4 step 7; needs accumulated λ to produce force traces);
    #   - the Phase 2 restitution test from the PBD-R benchmark
    #     (design doc §8.2) is added with controlled coefficient e;
    #   - deeper stacks (3+ bodies) are exercised, where a single contact
    #     supports cumulative weight from multiple bodies and Layer 1
    #     averaging only bounds the within-body contribution.
    # See cslc_xpbd/papers/XPBD.pdf Eq. (18) for the formula and
    # docs/superpowers/specs/2026-05-13-uxpbd-design.md §9.4 for the
    # runaway-launch incident that motivated Layer 1.
    SPAWN_X = 0.0
    LATTICE_SPAWN_Z = 0.20   # well above the rest height; falls a short distance
    # Lattice top face at spawn = LATTICE_SPAWN_Z + 0.04 = 0.24. Cube
    # bottom must clear that, so cube_com >= 0.24 + CUBE_HALF_EXTENT = 0.28.
    # Use 0.30 for a small 0.18 m drop above CUBE_REST_Z_NOMINAL = 0.12,
    # giving impact velocity ~1.9 m/s.
    CUBE_SPAWN_Z = 0.30

    # Shared geometry: the lattice body and the SM-rigid cube use the
    # EXACT same 4x4x4 sphere packing inscribed in a CUBE_HALF_EXTENT cube
    # (same centers, same radii). The lattice body also wraps a collision
    # box of the same half-extent so it has mass and a realistic
    # body_inertia tensor. Mass for the lattice body is set by the shape
    # density: volume = (2h)^3 at h = CUBE_HALF_EXTENT, default density
    # = 1000 kg/m^3 -> body mass ~ 0.512 kg.
    #
    # With CUBE_SPHERE_R = 0.012 and centers at body-frame z in
    # [-0.028, 0.028], the bottom lattice sphere bottoms at body_z - 0.04
    # which equals the bottom face of the collision box. SolverUXPBD does
    # not handle shape-vs-shape contact (the box is invisible to gravity-
    # side contact); ground support comes from the lattice particles via
    # particle-vs-shape contact.
    CUBE_HALF_EXTENT = 0.04
    CUBE_SPHERE_R = 0.012
    # Same density-volume math as the lattice body: 1000 kg/m^3 * (2*0.04)^3
    # = 0.512 kg. Matching the masses keeps the recoil ratio symmetric so
    # the first impact doesn't punt the lighter body off the ground.
    CUBE_TOTAL_MASS = 0.512

    # Body z when the bottom lattice sphere just touches the ground.
    # Bottom sphere center in body frame: -(CUBE_HALF_EXTENT - CUBE_SPHERE_R)
    # = -0.028. Ground contact => body_z - 0.028 = CUBE_SPHERE_R = 0.012,
    # so body_z = 0.04.
    LATTICE_REST_Z = 0.04

    # ----- Expected resting geometry -----
    # When the lattice body settles at LATTICE_REST_Z = 0.04, its top
    # lattice sphere centers are at world z = body_z + 0.028 = 0.068.
    # Sphere-sphere contact between cube particle and lattice particle
    # separates centers by 2 * CUBE_SPHERE_R = 0.024.
    # Cube bottom sphere centers at cube body-frame z = -0.028.
    # Ideal cube_com_z = 0.068 + 0.024 + 0.028 = 0.12.
    CUBE_REST_Z_NOMINAL = 0.12
    CUBE_REST_Z_TOLERANCE = 0.04   # +/- 4 cm around the nominal value

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # More substeps than the side-by-side demo because the stacked
        # geometry produces stiffer effective contact: the cube has to
        # transmit its weight through the lattice into the body wrench,
        # and that takes a couple more constraint passes per frame to
        # converge cleanly.
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # Shared 4x4x4 sphere packing used for BOTH the lattice body's
        # kinematic particles AND the SM-rigid cube's dynamic particles.
        # Centers AND radii are identical (cf. class-level comment).
        coords = np.linspace(
            -self.CUBE_HALF_EXTENT + self.CUBE_SPHERE_R,
            self.CUBE_HALF_EXTENT - self.CUBE_SPHERE_R,
            4,
        )
        xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
        packing_centers = np.stack(
            [xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
        packing_radii = np.full(
            packing_centers.shape[0], self.CUBE_SPHERE_R, dtype=np.float32)

        # ----- Lattice body (bottom of the stack) -----
        # NOTE: `add_body(mass=0.0)` is intentional. `add_body(mass=m)` would
        # add `m` kg of *point* mass with no inertia tensor (only an
        # `armature * I` term, default 0). The subsequent `add_shape_box`
        # call ALSO adds the shape's `density * volume` worth of mass AND
        # the corresponding cuboid inertia (see
        # `ModelBuilder._update_body_mass` in newton/_src/sim/builder.py).
        # Mixing both produces an internally inconsistent body: the total
        # mass is `m + density * volume` but the inertia tensor only
        # reflects the shape's `density * volume` contribution, so
        # `body_inv_mass` and `body_inv_inertia` correspond to different
        # masses. By using `mass=0.0` we let the shape's density carry both
        # mass and inertia consistently.
        self.body = builder.add_body(
            mass=0.0,
            xform=wp.transform(
                p=wp.vec3(self.SPAWN_X, 0.0, self.LATTICE_SPAWN_Z),
                q=wp.quat_identity(),
            ),
        )
        builder.add_shape_box(
            self.body,
            hx=self.CUBE_HALF_EXTENT,
            hy=self.CUBE_HALF_EXTENT,
            hz=self.CUBE_HALF_EXTENT,
        )
        builder.add_lattice(
            link=self.body,
            morphit_json={"centers": packing_centers, "radii": packing_radii},
            total_mass=0.0,
            pos=wp.vec3(self.SPAWN_X, 0.0, self.LATTICE_SPAWN_Z),
        )

        # ----- SM-rigid cube (top of the stack, falls onto the body) -----
        self.cube_group = builder.add_particle_volume(
            volume_data={"centers": packing_centers.tolist(),
                         "radii": packing_radii.tolist()},
            total_mass=self.CUBE_TOTAL_MASS,
            pos=wp.vec3(self.SPAWN_X, 0.0, self.CUBE_SPAWN_Z),
        )

        self.model = builder.finalize()

        # iterations=6 (vs 4 in the side-by-side demo) because the stacked
        # contact chain (cube particle <-> lattice particle <-> body wrench)
        # has two coupled constraint surfaces per step and benefits from
        # the extra Jacobi sweeps.
        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=6)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        cube_idx = self.model.particle_groups[self.cube_group]
        if hasattr(cube_idx, "numpy"):
            cube_idx = cube_idx.numpy()
        self._cube_idx = np.asarray(list(cube_idx), dtype=np.int32)

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        # Camera pulled back and angled down so both the body and the cube
        # are framed when stacked.
        self.viewer.set_camera(pos=wp.vec3(
            1.0, -1.0, 0.5), pitch=-20.0, yaw=135.0)

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
        # By the time the example finishes, both objects must be:
        #   1. roughly at rest (small velocity),
        #   2. stacked vertically (cube COM above lattice body z),
        #   3. not interpenetrating (cube COM well above the lattice body z,
        #      i.e. above the top of the lattice shell).
        body_q = self.state_0.body_q.numpy()[0]
        body_qd = self.state_0.body_qd.numpy()[0]
        cube_q = self.state_0.particle_q.numpy()[self._cube_idx]
        cube_com_z = float(cube_q[:, 2].mean())
        cube_v = self.state_0.particle_qd.numpy()[self._cube_idx]
        cube_v_mag = float(np.linalg.norm(cube_v.mean(axis=0)))

        # 1. Lattice body settles at the rest height (small downward
        #    correction allowed because the cube's weight is on top).
        body_z = float(body_q[2])
        assert abs(body_z - self.LATTICE_REST_Z) < 2.0e-2, (
            f"Lattice body did not settle: z={body_z:.4f}, "
            f"expected ~{self.LATTICE_REST_Z:.4f}"
        )

        # 2. Lattice body is near-stationary (under SI: m/s) at the end.
        body_v_lin = float(np.linalg.norm(body_qd[:3]))
        body_v_ang = float(np.linalg.norm(body_qd[3:]))
        assert body_v_lin < 0.10, (
            f"Lattice body still translating: |v|={body_v_lin:.3f} m/s")
        assert body_v_ang < 1.0, (
            f"Lattice body still rotating: |w|={body_v_ang:.3f} rad/s")

        # 3. Cube is above the body's COM and above the lattice top sphere
        #    centers (no interpenetration).
        lattice_top_z_in_world = (
            body_z + (self.CUBE_HALF_EXTENT - self.CUBE_SPHERE_R)
        )
        assert cube_com_z > lattice_top_z_in_world, (
            f"Cube COM ({cube_com_z:.4f}) below lattice top sphere centers "
            f"({lattice_top_z_in_world:.4f}) -- interpenetration."
        )

        # 4. Cube rests within tolerance of the geometric expectation.
        assert abs(cube_com_z - self.CUBE_REST_Z_NOMINAL) < self.CUBE_REST_Z_TOLERANCE, (
            f"Cube did not settle on top of body: com_z={cube_com_z:.4f}, "
            f"expected {self.CUBE_REST_Z_NOMINAL:.4f} "
            f"+/- {self.CUBE_REST_Z_TOLERANCE:.2f}"
        )

        # 5. Cube is near-stationary.
        assert cube_v_mag < 0.20, (
            f"Cube COM still moving: |v|={cube_v_mag:.3f} m/s")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
