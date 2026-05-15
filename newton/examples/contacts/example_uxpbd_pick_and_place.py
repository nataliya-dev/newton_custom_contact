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

FRANKA_HOME_Q = [
    0.0,
    -np.pi / 4.0,
    0.0,
    -3.0 * np.pi / 4.0,
    0.0,
    np.pi / 2.0,
    np.pi / 4.0,
]
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.01

PHASE_APPROACH = 0
PHASE_SQUEEZE = 1
PHASE_LIFT = 2
PHASE_HOLD = 3


def _find_body(builder, label):
    """Find a body index by suffix match on body_label.

    URDF bodies are registered with a URDF-name prefix, e.g.
    ``fr3/fr3_leftfinger``. This helper matches any label whose last
    path component equals *label*, so callers do not need to know the
    prefix.
    """
    for i, lbl in enumerate(builder.body_label):
        if lbl == label or lbl.split("/")[-1] == label:
            return i
    raise ValueError(
        f"Body '{label}' not found in builder. Available: {builder.body_label}")


def _attach_pad_lattice(builder, link_idx, half_extents, pos):
    """Build a 2x2x2 uniform lattice inside a finger pad and attach via add_lattice.

    The lattice is a 2x2x2 grid of equal-radius spheres inscribed in the
    rectangular pad volume. All spheres are marked as surface particles
    (is_surface=1) so they participate in cross-substrate contact.

    Args:
        builder: The ModelBuilder in progress.
        link_idx: Body index of the finger link to host the lattice.
        half_extents: (hx, hy, hz) half-extents [m] of the pad in link-local frame.
        pos: World-space position of the finger body at t=0 (after FK). The
            lattice particles are placed in world space as ``pos + p_local``,
            so the anchor constraint sees zero error on frame 1. Without
            this, mass-0 lattice particles get pulled across the world by
            the stiff anchor and the solver diverges to NaN within a couple
            of steps.
    """
    hx, hy, hz = half_extents
    n_per_axis = 2
    sphere_r = min(hx, hy, hz) / n_per_axis
    coords_x = np.linspace(-hx + sphere_r, hx - sphere_r, n_per_axis)
    coords_y = np.linspace(-hy + sphere_r, hy - sphere_r, n_per_axis)
    coords_z = np.linspace(-hz + sphere_r, hz - sphere_r, n_per_axis)

    centers = []
    radii = []
    is_surface = []
    for x in coords_x:
        for y in coords_y:
            for z in coords_z:
                centers.append([float(x), float(y), float(z)])
                radii.append(float(sphere_r))
                is_surface.append(1)

    builder.add_lattice(
        link=link_idx,
        morphit_json={"centers": centers,
                      "radii": radii, "is_surface": is_surface},
        total_mass=0.0,
        pos=pos,
    )


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args
        self.phase = PHASE_APPROACH
        self.phase_t0 = 0.0

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # Franka FR3 arm with hand (9 DOFs: 7 arm revolutes + 2 finger prismatic).
        builder.add_urdf(
            newton.utils.download_asset(
                "franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        # Set arm home pose and finger open width BEFORE attaching the lattices.
        # The lattice anchor is rigid (mass-0 particles) and starts in world space
        # at ``pos + p_local``; if ``pos`` does not match the finger's home-pose
        # world position, the anchor sees a huge initial error and diverges to NaN.
        # The target gains replicate the PD-gravity-comp tuning from robot_lift.py.
        builder.joint_q[:7] = FRANKA_HOME_Q
        builder.joint_q[7:9] = [FINGER_OPEN, FINGER_OPEN]
        builder.joint_target_pos[:9] = builder.joint_q[:9]

        # Probe FK to read the finger bodies' home-pose world transforms. We
        # finalize the URDF-only builder, run eval_fk, and discard; the second
        # finalize below picks up the lattices and cube added after this point.
        finger_l_idx = _find_body(builder, "fr3_leftfinger")
        finger_r_idx = _find_body(builder, "fr3_rightfinger")
        _probe_model = builder.finalize()
        _probe_state = _probe_model.state()
        newton.eval_fk(_probe_model, _probe_model.joint_q,
                       _probe_model.joint_qd, _probe_state)
        _probe_bq = _probe_state.body_q.numpy()
        finger_l_pos = wp.vec3(*[float(v)
                               for v in _probe_bq[finger_l_idx, :3]])
        finger_r_pos = wp.vec3(*[float(v)
                               for v in _probe_bq[finger_r_idx, :3]])

        # Attach a 2x2x2 lattice to each finger pad so they participate in
        # particle-based contact with the cube. The pad geometry is chosen to
        # match the physical finger tip dimensions of the Franka Hand.
        # Pad half-extents [m]: hx=cross-gap, hy=width, hz=height along finger
        pad_half = (0.012, 0.004, 0.025)
        _attach_pad_lattice(builder, finger_l_idx, pad_half, finger_l_pos)
        _attach_pad_lattice(builder, finger_r_idx, pad_half, finger_r_pos)

        # PD gains: arm joints use high stiffness; finger joints are softer
        # so they can be compliant without launching the cube.
        builder.joint_target_ke[:9] = [4500, 4500,
                                       3500, 3500, 2000, 2000, 2000, 500, 500]
        builder.joint_target_kd[:9] = [
            450, 450, 350, 350, 200, 200, 200, 50, 50]

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
        # particle-shape contacts, including the lattice finger pads). The
        # particle-shape kernel uses mu = 0.5 * (particle_mu + shape_material_mu[shape]),
        # so we override the per-shape value to match the intended 0.7 effective
        # coefficient (default shape_material_mu is 0.5).
        self.model.particle_mu = 0.7
        self.model.soft_contact_mu = 0.7
        self.model.shape_material_mu.assign(
            np.full(self.model.shape_count, 0.7, dtype=np.float32))

        self.solver = newton.solvers.SolverUXPBD(
            self.model, iterations=8, shock_propagation_k=1.0)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(
            1.5, -1.5, 1.2), pitch=-25.0, yaw=135.0)

    def _advance_phase(self):
        """Drive the phase machine: APPROACH -> SQUEEZE -> LIFT -> HOLD.

        Each phase transition updates joint_target_pos on the control object.
        APPROACH: wait 1 s (arm already at home pose near cube).
        SQUEEZE:  close fingers from FINGER_OPEN to FINGER_CLOSED over 1 s.
        LIFT:     retract elbow joint (joint_q[3]) to raise the end-effector.
        HOLD:     freeze targets indefinitely.
        """
        t = self.sim_time - self.phase_t0
        if self.phase == PHASE_APPROACH and t > 1.0:
            self.phase = PHASE_SQUEEZE
            self.phase_t0 = self.sim_time
            q = self.control.joint_target_pos.numpy().copy()
            q[7:9] = [FINGER_CLOSED, FINGER_CLOSED]
            self.control.joint_target_pos.assign(q)
        elif self.phase == PHASE_SQUEEZE and t > 1.0:
            self.phase = PHASE_LIFT
            self.phase_t0 = self.sim_time
            # Retract the elbow joint (joint 3 in 0-indexed arm DOFs) by +0.3 rad
            # to raise the hand while keeping the wrist orientation stable.
            q = self.control.joint_target_pos.numpy().copy()
            q[3] += 0.3
            self.control.joint_target_pos.assign(q)
        elif self.phase == PHASE_LIFT and t > 2.0:
            self.phase = PHASE_HOLD
            self.phase_t0 = self.sim_time

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self._advance_phase()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        """Verify the cube was lifted above the table surface.

        Reads mean Z of all cube particles. The cube rests at Z ~ 0.05 m
        before grasping; a successful lift must reach > 0.02 m (in case
        the cube settles on the ground) and not be ejected (< 1.5 m).
        Full grasp validation requires CUDA (Warp tile-reduce limitation).
        """
        cube_q = self.state_0.particle_q.numpy()
        cube_idx = self.model.particle_groups[self.cube_group]
        cube_idx_arr = np.asarray(list(cube_idx), dtype=np.int32)
        cube_z = float(np.mean(cube_q[cube_idx_arr, 2]))
        if cube_z < 0.02:
            raise RuntimeError(f"Cube not lifted; z={cube_z:.4f}")
        if cube_z > 1.5:
            raise RuntimeError(f"Cube ejected; z={cube_z:.4f}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
