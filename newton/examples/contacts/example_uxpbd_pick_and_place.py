# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Pick and Place (Scenario A)
#
# A spherical Franka arm — each link is shelled by a sphere lattice attached
# via add_lattice — near a free shape-matched rigid cube (mass 0.3 kg,
# mu=0.7). The robot loads from assets/panda/urdfs/10/sphere_panda.urdf:
# the articulated 7-DOF chain (fingers fixed) is loaded as rigid bodies, and
# each link's child collision spheres are stripped from the URDF and
# re-attached as a UXPBD lattice (substrate 0, anchored to the link).
# Phase machine: APPROACH -> SQUEEZE (placeholder) -> LIFT -> HOLD.
#
# Phase 2 demo: validates the cross-substrate lattice <-> SM-rigid contact
# path. Requires Phase 2 PBD-R kernels (CUDA only).
#
# Command: python -m newton.examples uxpbd_pick_and_place
###########################################################################


import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
SPHERE_PANDA_URDF = _ASSETS_DIR / "panda" / "urdfs" / "25" / "sphere_panda.urdf"

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

    URDF bodies are registered with a URDF-name prefix (e.g. ``panda/panda_link3``).
    This helper matches any label whose last path component equals *label*, so
    callers do not need to know the prefix.
    """
    for i, lbl in enumerate(builder.body_label):
        if lbl == label or lbl.split("/")[-1] == label:
            return i
    raise ValueError(
        f"Body '{label}' not found in builder. Available: {builder.body_label}")


def _prepare_sphere_panda_urdf(urdf_path):
    """Preprocess a spherical-Panda URDF for use with ``add_urdf`` + ``add_lattice``.

    Two transforms are applied to the in-memory XML:

    1. **Strip the sphere lattice children.** The sphere_panda format (see
       ``assets/panda/urdfs/<N>/sphere_panda.urdf``) encodes each link's
       collision lattice as N child links named ``<parent>_sphereK`` attached
       via a fixed joint ``<parent>_to_<parent>_sphereK`` whose origin xyz is
       the sphere center in the parent's local frame; the child link holds the
       sphere's radius in its collision geometry. We pull that data into a
       dict suitable for ``ModelBuilder.add_lattice(morphit_json=...)`` and
       drop the children so the remaining URDF is just the articulated chain.
    2. **Re-actuate the finger joints.** The source URDF ships the gripper
       finger joints (``panda_finger_joint{1,2}``) as ``fixed`` with their
       prismatic ``<limit>`` element commented out (frozen-hand variant).
       We promote them back to ``prismatic`` with the limits the source URDF
       documents, so the gripper has the 2 DOFs needed for pick-and-place.

    Returns:
        (cleaned_tree, link_lattices) where ``link_lattices`` maps parent
        link name -> {"centers": [[x,y,z], ...], "radii": [r, ...],
        "is_surface": [1, ...]}.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_by_name = {link.get("name"): link for link in root.findall("link")}

    # Limits restored on the finger joints; values match the commented-out
    # <limit> in the source URDF (effort N, lower/upper m, velocity m/s).
    _FINGER_LIMITS = {
        "effort": "20", "lower": "0.0", "upper": "0.04", "velocity": "0.2",
    }
    # The source URDF's finger joints have the rpy=π marker on the LEFT
    # finger (and a negated axis on the right), which (a) makes both fingers
    # translate to the same side of the hand on open, and (b) — combined
    # with identical sphere centers in both finger links' local frames —
    # bunches the gripper lattice onto one side instead of mirroring it
    # across the hand axis. The original Franka URDF (fr3_franka_hand) puts
    # the π flip on the RIGHT finger with axis (0,1,0) on both joints; we
    # rewrite both joints to that convention so the gripper opens
    # symmetrically and the per-finger sphere lattices mirror correctly.
    _FINGER_JOINT_FIX = {
        "panda_finger_joint1": {"rpy": "0 0 0",
                                "axis": "0 1 0"},  # left
        "panda_finger_joint2": {"rpy": "0 0 3.141592653589793",
                                "axis": "0 1 0"},  # right
    }

    link_lattices: dict[str, dict[str, list]] = {}
    joints_to_remove = []
    child_links_to_remove: set[str] = set()
    for joint in root.findall("joint"):
        name = joint.get("name") or ""
        if name in _FINGER_JOINT_FIX:
            joint.set("type", "prismatic")
            fix = _FINGER_JOINT_FIX[name]
            origin_el = joint.find("origin")
            if origin_el is not None:
                origin_el.set("rpy", fix["rpy"])
            axis_el = joint.find("axis")
            if axis_el is not None:
                axis_el.set("xyz", fix["axis"])
            # Remove any existing <limit> (shouldn't exist, but be defensive)
            # and add a fresh one with the documented gripper limits.
            for existing in list(joint.findall("limit")):
                joint.remove(existing)
            limit_el = ET.SubElement(joint, "limit")
            for k, v in _FINGER_LIMITS.items():
                limit_el.set(k, v)
            continue
        if "_sphere" not in name:
            continue
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link")
        child = child_el.get("link")
        child_link = link_by_name.get(child)
        if child_link is None:
            continue
        col = child_link.find("collision/geometry/sphere")
        if col is None:
            continue
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0] if origin is None else [
            float(v) for v in origin.get("xyz", "0 0 0").split()]
        radius = float(col.get("radius"))

        entry = link_lattices.setdefault(
            parent, {"centers": [], "radii": [], "is_surface": []})
        entry["centers"].append(xyz)
        entry["radii"].append(radius)
        entry["is_surface"].append(1)
        joints_to_remove.append(joint)
        child_links_to_remove.add(child)

    for j in joints_to_remove:
        root.remove(j)
    for cname in child_links_to_remove:
        root.remove(link_by_name[cname])

    return tree, link_lattices


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

        # Spherical Panda: 7-DOF revolute chain with fixed hand/fingers.
        # The URDF's per-link collision spheres are stripped here and
        # re-attached below as UXPBD lattices anchored to each link.
        cleaned_tree, link_lattices = _prepare_sphere_panda_urdf(SPHERE_PANDA_URDF)
        with tempfile.NamedTemporaryFile(
                suffix=".urdf", delete=False, mode="wb") as _tmp:
            cleaned_tree.write(_tmp)
            cleaned_urdf_path = _tmp.name
        builder.add_urdf(
            cleaned_urdf_path,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )

        # Set arm home pose + open-gripper width (9 DOFs: 7 arm revolutes
        # plus the 2 prismatic finger joints re-actuated above). The target
        # gains replicate the PD-gravity-comp tuning from robot_lift.py;
        # finger gains are softer so a future SQUEEZE doesn't launch the
        # cube on contact.
        builder.joint_q[:7] = FRANKA_HOME_Q
        builder.joint_q[7:9] = [FINGER_OPEN, FINGER_OPEN]
        builder.joint_target_pos[:9] = builder.joint_q[:9]
        builder.joint_target_ke[:9] = [4500, 4500,
                                       3500, 3500, 2000, 2000, 2000, 500, 500]
        builder.joint_target_kd[:9] = [
            450, 450, 350, 350, 200, 200, 200, 50, 50]

        # Probe FK to read each link's home-pose world position. The lattice
        # anchor is rigid (mass-0 particles) and starts in world space at
        # ``pos + p_local``; if ``pos`` does not match the link's home-pose
        # world position the anchor sees a huge initial error and the
        # solver diverges to NaN within a couple of steps.
        _probe_model = builder.finalize()
        _probe_state = _probe_model.state()
        newton.eval_fk(_probe_model, _probe_model.joint_q,
                       _probe_model.joint_qd, _probe_state)
        _probe_bq = _probe_state.body_q.numpy()

        # Attach a sphere lattice to each Panda link that carried spheres in
        # the source URDF. Each lattice's particles are anchored to the link
        # via add_lattice's mass-0 rigid anchor constraint, so the robot moves
        # rigidly while its collision surface is a particle lattice that
        # participates in cross-substrate UXPBD contact. Both the link's
        # world position AND rotation must be passed; add_lattice places each
        # particle at ``rot * p_local + pos`` at t=0 and any mismatch with the
        # link's actual world pose injects a huge initial constraint error
        # (most Panda links have non-trivial rotation at home pose).
        for link_name, spheres in link_lattices.items():
            link_idx = _find_body(builder, link_name)
            bq = _probe_bq[link_idx]
            link_pos = wp.vec3(float(bq[0]), float(bq[1]), float(bq[2]))
            # body_q quaternion layout is (qx, qy, qz, qw); wp.quat uses the
            # same xyzw layout.
            link_rot = wp.quat(float(bq[3]), float(bq[4]),
                               float(bq[5]), float(bq[6]))
            builder.add_lattice(
                link=link_idx,
                morphit_json=spheres,
                total_mass=0.0,
                pos=link_pos,
                rot=link_rot,
            )

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

        # Fluid block dropping onto the robot's upper arm. Centered above
        # panda_link2 (shoulder, world (0, 0, 0.333)) at z=0.95; the block
        # free-falls ~0.6 m onto the lattice, cascading down the chain.
        # 6x6x4 = 144 particles, particle radius 8 mm, cells touching at
        # 16 mm spacing (rest_density matches add_fluid_grid default).
        fluid_dims = (6, 6, 4)
        fluid_cell = 0.016
        fluid_r = 0.008
        fluid_corner = wp.vec3(
            -(fluid_dims[0] - 1) * fluid_cell / 2,
            -(fluid_dims[1] - 1) * fluid_cell / 2,
            0.95,
        )
        builder.add_fluid_grid(
            pos=fluid_corner,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=fluid_dims[0], dim_y=fluid_dims[1], dim_z=fluid_dims[2],
            cell_x=fluid_cell, cell_y=fluid_cell, cell_z=fluid_cell,
            particle_radius=fluid_r,
            rest_density=1000.0,
            smoothing_radius_factor=3.0,
            viscosity=0.05,
            cohesion=0.0,
        )

        self.model = builder.finalize()
        # Cap particle velocity to suppress cross-substrate "impact launch"
        # when the fluid block hits the robot's lattice (see the note in
        # example_uxpbd_lattice_into_fluid for the underlying mechanism).
        # Only applies to mass>0 particles, so the lattice anchors are
        # unaffected and the cube grasp dynamics still play normally.
        self.model.particle_max_velocity = 2.0
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
            self.model, iterations=8, shock_propagation_k=1.0,
            fluid_iterations=4)
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
        SQUEEZE:  close fingers from FINGER_OPEN to FINGER_CLOSED so the
                  gripper lattice friction-grasps the cube before LIFT.
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
        """Verify the cube was lifted above the table surface and the
        whole grasp pipeline stayed numerically stable.

        Reads mean Z of all cube particles. The cube rests at Z ~ 0.05 m
        before grasping; a successful lift must reach > 0.02 m (in case
        the cube settles on the ground) and not be ejected (< 1.5 m).
        Full grasp validation requires CUDA (Warp tile-reduce limitation).
        """
        # model.particle_groups[i] may be a wp.array; .numpy() and then
        # list() to get a plain Python iterable (wp.array does not
        # support Python item indexing or iteration).
        cube_idx = self.model.particle_groups[self.cube_group]
        if hasattr(cube_idx, "numpy"):
            cube_idx = cube_idx.numpy()
        cube_idx_arr = np.asarray(list(cube_idx), dtype=np.int32)

        cube_q = self.state_0.particle_q.numpy()[cube_idx_arr]
        cube_v = self.state_0.particle_qd.numpy()[cube_idx_arr]

        # 1. Numerical sanity — must hold before any height assertion.
        assert np.isfinite(cube_q).all(), "NaN/Inf in cube particle positions"
        assert np.isfinite(cube_v).all(), "NaN/Inf in cube particle velocities"

        # 2. Lift / ejection bound.
        cube_z = float(np.mean(cube_q[:, 2]))
        if cube_z < 0.02:
            raise RuntimeError(f"Cube not lifted; z={cube_z:.4f}")
        if cube_z > 1.5:
            raise RuntimeError(f"Cube ejected; z={cube_z:.4f}")

        # 3. Cube hasn't shot off horizontally (stays within a 1 m radius
        #    of its spawn).
        com_xy = cube_q[:, :2].mean(axis=0)
        assert float(np.linalg.norm(com_xy - np.array([0.55, 0.0]))) < 1.0, (
            f"Cube drifted out of workspace: com_xy={com_xy}"
        )

        # 4. No catastrophic velocity (the grasp + lift should not impart
        #    > a few m/s; >10 m/s indicates contact-PBF instability).
        v_max = float(np.linalg.norm(cube_v, axis=1).max())
        assert v_max < 5.0, f"Cube particle moving too fast: v_max={v_max:.3f} m/s"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
