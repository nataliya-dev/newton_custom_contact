# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Pick and Place (CSLC gripper grasp)
#
# A spherical Franka arm reaches a small free object, closes its two
# fingers around it, and lifts it. The two fingertips
# (panda_leftfinger / panda_rightfinger) are shelled with a COMPLIANT
# sphere lattice (CSLC) so the gripper "skin" physically deforms around
# the object during the grasp; nothing else carries a collision lattice,
# so the only contact -- and the only lattice compression we measure --
# is the fingertip grip.
#
# Pipeline:
#   1. Load assets/panda/urdfs/25/sphere_panda.urdf, strip the per-link
#      collision-sphere children, and re-actuate the two finger prismatic
#      joints (see _prepare_sphere_panda_urdf). That URDF stores the robot's
#      whole visible shell in those children, so the stripped arm/hand spheres
#      are re-added as VISUAL-ONLY shapes (no collision, zero mass) -- without
#      this only the fingertip lattice would render and the arm is invisible.
#   2. INVERSE KINEMATICS (newton.ik): solve the 7-DOF arm for two hand
#      poses -- the grasp pose at the object and a lifted pose -- both with
#      the hand pointing straight down. FK then (a) verifies the solved
#      configs put the hand on target (self._ik_max_err), (b) locates the
#      finger midpoint so the object spawns exactly between the fingers,
#      and (c) checks the fingertip lattice clears the ground at the grasp.
#   3. Attach a COMPLIANT lattice (CSLCParams on the solver + per-sphere
#      k_anchor / k_bulk on add_lattice) to each fingertip.
#   4. Phase machine SETTLE -> GRASP -> LIFT -> HOLD holds the arm at the
#      IK grasp pose (PD position control) while the object settles between
#      the open fingers, then closes the fingers (the CSLC skin compresses
#      around the object -- model.lattice_delta), then raises the arm to
#      the lifted pose so the friction grip carries the object up.
#
# The grasped object is a TALL sphere-packed column resting on the ground:
# the Franka fingers are ~8 cm long, so to grip a ground object without
# the fingertips jamming into the floor the object must extend up into the
# fingers' contact band. A short ball would force the fingertips below the
# ground plane (a violent ground-vs-fingertip contact that destabilises
# the arm), so we use a column that the fingers grip on its sides at
# mid-height while the fingertips stay well above z=0.
#
# NOTE on telemetry: SolverUXPBD is position-based -- it integrates body
# poses (state.body_q), not generalized joint coordinates, and does NOT
# write joint_q back after a step. So state.joint_q for the finger DOFs
# stays at its initial value even as the finger BODIES close. We therefore
# report grip state from the finger body poses, the lattice<->object
# surface gap, and model.lattice_delta -- never from joint_q.
#
# Requires the Phase 2 CSLC kernels (CUDA only).
#
# Command:  python -m newton.examples uxpbd_pick_and_place
#   --no-compliant   run the rigid-lattice baseline for A/B comparison
###########################################################################

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton import JointTargetMode
from newton.solvers import CSLCParams

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
FINGER_OPEN = 0.04    # finger prismatic joint at full open [m]
# Commanded grip target. NOT 0 (full closure): position-driving the fingers
# fully shut makes them crush straight through the compliant skin, so we
# command a target that lets the pads balance the PD drive at a firm,
# bounded squeeze. The grasp is BISTABLE in finger_closed (mapped by sweep,
# k_anchor=1.5e3, 0.08 kg / 28 mm-thick column; self.finger_closed overrides
# it for re-sweeping):
#   * LOOSE  (~0.020-0.023 m): the fingers settle against the lattice at
#     fsep ~47 mm and gently but STABLY hold the faces. The column lifts in
#     near-lockstep (climbs only ~1 cm in the grip), the pads stay engaged
#     through HOLD, and the hold is ROBUST + near-deterministic (6/6 runs).
#     Skin compression is gentle (~0.3 mm) but sustained.
#   * dead zone (~0.014-0.018 m): the column drops out during the lift.
#   * TIGHT  (~0.012-0.013 m): the fingers race toward full closure and
#     either seat the column high (climbs ~5 cm) with a large ~2 mm
#     compression, OR -- nondeterministically -- crush through and EJECT it
#     mid-lift. Flaky (~1 in 5 runs drops), so unfit for a reliable demo.
# We use the LOOSE regime: a reliable grasp matters more than a dramatic
# squeeze. (For a deliberately large, controlled deformation, force/impedance
# control or softer pads + a heavier object would be the right lever -- the
# gentle ~0.3 mm here is the honest cost of a position-controlled stable hold.)
FINGER_CLOSED = 0.0205

# ----- Grasped object: sphere-packed box column ------------------------
# A vertical box column, rotated about Z at build time so a flat FACE meets
# each finger pad (perpendicular to the gripper's actual closing axis, which
# the ~45 deg IK wrist tilts off the world axes). Flat-pad-on-flat-face is
# FORM CLOSURE: the object cannot squirt out sideways the way a round object
# does when squeezed between flat pads.
#   OBJ_HX = half-thickness along the closing axis (gripped faces 2*HX apart)
#   OBJ_HY = half-width along the face
#   OBJ_HZ = half-height
# HEIGHT is a tight trade-off set by the gripper geometry. The 8 cm fingers
# put the pinch zone high (~z=0.05), so the ground-resting column must be
# TALL enough to stand up into it -- too short and it sits below the pads and
# slips out (verified: OBJ_HZ<=0.032 never grips). 2*OBJ_HZ ~ 76 mm gives
# enough face to grip robustly (OBJ_HZ=0.038, 4/4 holds) while keeping the
# lifted top as low as possible. NOTE: because the hand is rendered in full
# (its sphere shell fills the inter-finger volume) and is collision-free, the
# gripped column's top still visually overlaps the palm -- inherent to this
# gripper, not a grasp failure.
OBJ_HX = 0.014
OBJ_HY = 0.018
OBJ_HZ = 0.038
OBJ_GRID = (3, 3, 7)  # spheres per axis (63 total), 3D-distributed so the
#                       shape-matching covariance is well-conditioned
#                       (collinear/planar packings make the SVD rotation
#                       extraction unstable; see the SceneParams.obj_radius
#                       note in example_uxpbd_lift_test).
OBJ_MASS = 0.08       # [kg]  (weight ~0.78 N)
MU = 1.0              # Coulomb friction (kernel mu_eff = 0.5*(particle+shape))

# Hand grasp target, world frame. X/Y put the object in front of the robot.
# Z is high enough that the downward-pointing fingers grip the column's
# upper-middle with the fingertip lattice band staying above the ground
# (verified by an assertion in __init__).
GRASP_X = 0.45
GRASP_Y = 0.0
GRASP_HAND_Z = 0.13
LIFT_DZ = 0.10        # how far the lifted hand pose sits above the grasp [m]

# Hand-down orientation: quaternion (x,y,z,w) = (1,0,0,0) is a 180 deg
# rotation about world X, sending the hand's body +Z (palm/approach axis,
# pointing out toward the fingers) to world -Z so the fingers point down.
DOWN_QUAT_XYZW = (1.0, 0.0, 0.0, 0.0)

# ----- Compliant fingertip lattice (CSLC) ------------------------------
# Per-sphere stiffnesses passed to add_lattice. The fingertip spheres are
# small (r ~ 4-11 mm) so k_anchor is much softer than the lift-test pad
# (1e5 N/m, driven by a 5e4-stiff prismatic) to keep the visible
# compression in the sub-mm-to-mm band: equilibrium delta ~
# F_grip / (N_active * k_anchor). k_bulk is the per-volume Jacobi contact
# stiffness k_c.
PAD_K_ANCHOR = 1.5e3   # N/m per sphere (anchor spring) -- soft enough that
#                        the gentle (position-controlled, loose-regime) lifting
#                        grip shows a sub-mm skin compression (peak delta
#                        ~0.3 mm, all pads lightly engaged) rather than microns.
#                        Softening it further mostly recruits MORE spheres
#                        rather than deepening any one, so deformation stays
#                        sub-mm; a dramatic squeeze needs force control or a
#                        heavier object, not just softer pads (see FINGER_CLOSED).
PAD_K_BULK = 1.0e8     # Pa.m^-1/2 (Jacobi per-volume contact stiffness)
PAD_K_LATERAL = 5.0e2  # N/m (reserved lateral coupling)
PAD_DAMPING = 2.0      # s/m (reserved Hunt-Crossley)

# Finger drive: firm enough to load the grip, soft enough not to crush the
# compliant skin past the sphere radius (which would pop the object out).
FINGER_KE = 800.0
FINGER_KD = 40.0

# Phase durations [s].
SETTLE_T = 0.5
GRASP_T = 1.0
LIFT_T = 2.0
HOLD_T = 0.5


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
            if axis_el is None:
                axis_el = ET.SubElement(joint, "axis")
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


def _build_box_column(builder, *, hx, hy, hz, grid, theta, mass, pos_xy,
                      ground_z=0.0, drop=0.005):
    """Add a sphere-packed SM-rigid box column standing on the ground at
    ``pos_xy``, yaw-rotated by ``theta`` [rad] about Z so its X-faces face the
    gripper.

    A ``grid = (nx, ny, nz)`` lattice of equal-radius spheres fills the box of
    half-extents (hx, hy, hz); the sphere radius is the largest per-axis
    half-spacing scaled up slightly so the emitted faces are gap-free.

    The column is spawned so its LOWEST sphere SURFACE sits ``drop`` above
    ``ground_z`` (a short free-fall to settle; a tangent-to-ground spawn is
    unstable for SM-rigid bodies). The lowest sphere centre is ``hz`` below the
    centroid and its surface another ``r`` below that, so the spawn centroid is
    ``ground_z + hz + r + drop`` -- NOT ``ground_z + hz`` (forgetting the +r
    spawns the bottom sphere ~r below the ground, and the resulting penetration
    launches short columns sideways during settle).

    Returns ``(group_id, rest_centroid_z)`` where ``rest_centroid_z =
    ground_z + hz + r`` is the settled centroid height (bottom surface on the
    ground).
    """
    nx, ny, nz = grid

    def _axis(h, n):
        return np.array([0.0]) if n <= 1 else np.linspace(-h, h, n)

    sp = max((2 * hx) / max(nx - 1, 1),
             (2 * hy) / max(ny - 1, 1),
             (2 * hz) / max(nz - 1, 1))
    r = float(0.5 * sp * 1.2)
    cx, cy, cz = _axis(hx, nx), _axis(hy, ny), _axis(hz, nz)
    xs, ys, zs = np.meshgrid(cx, cy, cz, indexing="ij")
    centers = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    # Yaw the box so its X-axis (thickness / gripped faces) aligns with the
    # gripper closing direction.
    c, s = np.cos(theta), np.sin(theta)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centers = (centers @ rot_z.T).astype(np.float32)
    radii = np.full(centers.shape[0], r, dtype=np.float32)
    rest_centroid_z = ground_z + hz + r
    spawn_z = rest_centroid_z + drop
    group = builder.add_particle_volume(
        volume_data={"centers": centers.tolist(), "radii": radii.tolist()},
        total_mass=mass,
        pos=wp.vec3(float(pos_xy[0]), float(pos_xy[1]), spawn_z),
    )
    return group, rest_centroid_z


def _smoothstep(s: float) -> float:
    """Cubic smoothstep 3s^2 - 2s^3 clamped to [0, 1]. Eases the arm and
    fingers between targets so the PD drive never sees a target-velocity
    step (which would impulse-load the compliant fingertip lattice)."""
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args
        self.use_cslc = not getattr(args, "no_compliant", False)
        # Phase durations [s] (instance attrs so a debug runner can shorten
        # the sequence for fast iteration).
        self.settle_t = SETTLE_T
        self.grasp_t = GRASP_T
        self.lift_t = LIFT_T
        self.hold_t = HOLD_T
        # Grip-close target (tunable; the runner sweeps this to land the grip).
        self.finger_closed = getattr(args, "finger_closed", None) or FINGER_CLOSED

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # Spherical Panda: 7-DOF revolute chain + 2 prismatic fingers. The
        # URDF's per-link collision spheres are stripped here; we re-attach
        # only the fingertip lattices below (the arm/hand stay collision
        # free -- nothing touches them in this grasp).
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

        # Arm/finger PD gains. Position mode on every DOF: the arm joints
        # arrive from add_urdf already actuated, but the two finger joints
        # were re-actuated from <fixed> and need ke/kd set explicitly.
        builder.joint_target_ke[:9] = [
            4500, 4500, 3500, 3500, 2000, 2000, 2000, FINGER_KE, FINGER_KE]
        builder.joint_target_kd[:9] = [
            450, 450, 350, 350, 200, 200, 200, FINGER_KD, FINGER_KD]
        for d in range(9):
            builder.joint_target_mode[d] = int(JointTargetMode.POSITION)

        # ----- Probe model + INVERSE KINEMATICS -----------------------
        # Finalize an arm-only probe (no lattices/object added yet -- those
        # only add particles, not bodies, so the panda_hand/finger body
        # indices are identical in the final model). Solve IK on it for the
        # hand poses. eval_fk then verifies the solutions and measures the
        # finger geometry for object placement + ground clearance.
        probe_model = builder.finalize()
        probe_state = probe_model.state()
        self.hand_idx = _find_body(builder, "panda_hand")
        self.lf_idx = _find_body(builder, "panda_leftfinger")
        self.rf_idx = _find_body(builder, "panda_rightfinger")

        def solve_ik(target_xyz):
            """Solve the 7-DOF arm so panda_hand reaches target_xyz with the
            hand pointing straight down. Returns (q_arm[7], pos_err)."""
            jq = probe_model.joint_q.numpy().copy().reshape(
                (1, probe_model.joint_coord_count))
            jq[0, :7] = FRANKA_HOME_Q  # warm start from home each solve
            jq_wp = wp.array(jq, dtype=wp.float32)
            pos_obj = ik.IKObjectivePosition(
                link_index=self.hand_idx, link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([wp.vec3(*target_xyz)], dtype=wp.vec3))
            rot_obj = ik.IKObjectiveRotation(
                link_index=self.hand_idx,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array(
                    [wp.vec4(*DOWN_QUAT_XYZW)], dtype=wp.vec4))
            lim_obj = ik.IKObjectiveJointLimit(
                joint_limit_lower=probe_model.joint_limit_lower,
                joint_limit_upper=probe_model.joint_limit_upper, weight=10.0)
            solver = ik.IKSolver(
                model=probe_model, n_problems=1,
                objectives=[pos_obj, rot_obj, lim_obj],
                lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
            solver.step(jq_wp, jq_wp, iterations=64)
            q_arm = jq_wp.numpy()[0, :7].copy()
            probe_model.joint_q.assign(
                np.concatenate([q_arm, [FINGER_OPEN, FINGER_OPEN]]).astype(np.float32))
            newton.eval_fk(probe_model, probe_model.joint_q,
                           probe_model.joint_qd, probe_state)
            hand_pos = probe_state.body_q.numpy()[self.hand_idx, :3]
            err = float(np.linalg.norm(hand_pos - np.asarray(target_xyz)))
            return q_arm, err

        grasp_xyz = (GRASP_X, GRASP_Y, GRASP_HAND_Z)
        lift_xyz = (GRASP_X, GRASP_Y, GRASP_HAND_Z + LIFT_DZ)
        self.q_grasp, e_g = solve_ik(grasp_xyz)
        self.q_lift, e_l = solve_ik(lift_xyz)
        self._ik_max_err = max(e_g, e_l)
        print(f"[IK] reach errors (mm): grasp={e_g*1e3:.2f} lift={e_l*1e3:.2f}")

        # ----- Start the arm AT the grasp config (fingers open) --------
        # The arm holds this IK-reached pose while the object settles
        # between the open fingers; then the fingers close and the arm
        # lifts. The lattice anchors are seeded in world space at t=0 from
        # the link pose at this config, so the builder joint_q MUST match.
        builder.joint_q[:7] = self.q_grasp
        builder.joint_q[7:9] = [FINGER_OPEN, FINGER_OPEN]
        builder.joint_target_pos[:7] = self.q_grasp
        builder.joint_target_pos[7:9] = [FINGER_OPEN, FINGER_OPEN]

        # FK at the grasp config (fingers open): finger lattice world poses
        # for the anchor placement, the finger midpoint for object spawn,
        # and the lowest fingertip-lattice sphere z for the ground-clearance
        # check. We use the actual lattice sphere world positions (rigidly
        # bound to each finger body) rather than the body origins, because
        # the lattice hangs ~p_local.z below the body origin along the
        # downward hand axis.
        probe_model.joint_q.assign(
            np.concatenate([self.q_grasp, [FINGER_OPEN, FINGER_OPEN]]).astype(np.float32))
        newton.eval_fk(probe_model, probe_model.joint_q,
                       probe_model.joint_qd, probe_state)
        bq_grasp = probe_state.body_q.numpy()

        def _lattice_world(link_name, link_idx):
            spheres = link_lattices[link_name]
            c = np.asarray(spheres["centers"], dtype=np.float64)
            rr = np.asarray(spheres["radii"], dtype=np.float64)
            p = bq_grasp[link_idx, :3].astype(np.float64)
            q = bq_grasp[link_idx, 3:7].astype(np.float64)  # xyzw
            qv = q[:3]
            world = np.array([ci + 2.0 * q[3] * np.cross(qv, ci)
                              + 2.0 * np.cross(qv, np.cross(qv, ci)) + p
                              for ci in c])
            return world, rr

        lfw, lfr = _lattice_world("panda_leftfinger", self.lf_idx)
        rfw, rfr = _lattice_world("panda_rightfinger", self.rf_idx)
        finger_mid = 0.5 * (lfw.mean(axis=0) + rfw.mean(axis=0))
        obj_xy = (float(finger_mid[0]), float(finger_mid[1]))
        lattice_bottom_z = float(min((lfw[:, 2] - lfr).min(),
                                     (rfw[:, 2] - rfr).min()))
        print(f"[place] finger-lattice midpoint at grasp = {np.round(finger_mid, 4)} "
              f"-> object spawn xy = {np.round(obj_xy, 4)}")
        print(f"[clearance] lowest fingertip-lattice sphere z = "
              f"{lattice_bottom_z*1e3:.1f} mm (must be > 0)")
        # Hard guard: if the fingertips dip below the ground at the grasp
        # pose, the ground-vs-fingertip contact would explode the arm.
        assert lattice_bottom_z > 0.002, (
            f"Fingertip lattice penetrates the ground at the grasp pose "
            f"(lowest sphere z={lattice_bottom_z*1e3:.1f} mm); raise "
            f"GRASP_HAND_Z or shorten the fingers.")

        # ----- Compliant fingertip lattices (the gripper "skin") -------
        # Only the two fingers carry a lattice. CSLC compliance is applied
        # uniformly (CSLCParams is a global solver flag), so the per-sphere
        # k_anchor here is what makes the fingertips the soft, deforming
        # surface; nothing else has a lattice, so model.lattice_delta is a
        # clean readout of the fingertip grip compression.
        for link_name, link_idx in (("panda_leftfinger", self.lf_idx),
                                     ("panda_rightfinger", self.rf_idx)):
            spheres = link_lattices[link_name]
            bq = bq_grasp[link_idx]
            link_pos = wp.vec3(float(bq[0]), float(bq[1]), float(bq[2]))
            link_rot = wp.quat(float(bq[3]), float(bq[4]),
                               float(bq[5]), float(bq[6]))
            builder.add_lattice(
                link=link_idx,
                morphit_json=spheres,
                total_mass=0.0,
                pos=link_pos,
                rot=link_rot,
                k_anchor=PAD_K_ANCHOR,
                k_lateral=PAD_K_LATERAL,
                k_bulk=PAD_K_BULK,
                damping=PAD_DAMPING,
            )

        # ----- Make the arm/hand visible (visual-only spheres) ---------
        # The sphere_panda URDF stores the robot's ENTIRE visible geometry
        # in its per-link collision-sphere children (panda_link*_sphereK);
        # the real chain links carry only an invisible 1 mm collision sphere
        # and no <visual>. _prepare_sphere_panda_urdf strips all those
        # children, so without re-adding them the arm/hand render as nothing
        # and only the fingertip particle lattice is visible. Re-add the
        # stripped arm/hand spheres as VISUAL-ONLY shapes (no shape/particle
        # collision) on their parent bodies, reusing the centres/radii the
        # prepare step already harvested into link_lattices. The two fingers
        # are intentionally skipped: their "skin" is the compliant PARTICLE
        # lattice added above -- that is the surface we want to watch deform,
        # so overlaying rigid spheres there would hide the compression.
        arm_visual_cfg = builder.default_shape_cfg.copy()
        arm_visual_cfg.has_shape_collision = False
        arm_visual_cfg.has_particle_collision = False
        arm_visual_cfg.is_visible = True
        # density=0: these are pure decoration. add_urdf already set each
        # link's mass/inertia from its <inertial> tag, and the builder
        # ACCUMULATES shape-derived mass on top -- so leaving the default
        # 1000 kg/m^3 here would dump >1 kg onto each link (the r~7 cm
        # spheres alone), wrecking the arm dynamics. Zero keeps them inert.
        arm_visual_cfg.density = 0.0
        for link_name, spheres in link_lattices.items():
            if link_name in ("panda_leftfinger", "panda_rightfinger"):
                continue
            link_idx = _find_body(builder, link_name)
            for c, r in zip(spheres["centers"], spheres["radii"]):
                builder.add_shape_sphere(
                    body=link_idx,
                    xform=wp.transform(wp.vec3(*c), wp.quat_identity()),
                    radius=float(r),
                    cfg=arm_visual_cfg,
                    color=wp.vec3(0.6, 0.6, 0.62),
                )

        # ----- Grasped object: tall SM-rigid box column ---------------
        # Rests on the ground at the finger midpoint and stands up into the
        # fingers' contact band. Yaw-aligned so a flat face meets each finger
        # pad (form closure -> no squirt-out). Spawned ~5 mm above its
        # resting height so it free-falls a short way and settles (a
        # tangent-to-ground spawn is unstable for SM-rigid bodies). The
        # fingers grip its faces at mid-height; the fingertips clear the
        # ground (asserted above).
        closing_dir = rfw.mean(axis=0)[:2] - lfw.mean(axis=0)[:2]
        obj_yaw = float(np.arctan2(closing_dir[1], closing_dir[0]))
        print(f"[place] gripper closing yaw = {np.degrees(obj_yaw):.1f} deg "
              f"-> box X-faces aligned to the fingers")
        self.obj_group, obj_rest_z = _build_box_column(
            builder, hx=OBJ_HX, hy=OBJ_HY, hz=OBJ_HZ, grid=OBJ_GRID,
            theta=obj_yaw, mass=OBJ_MASS, pos_xy=obj_xy, drop=0.005)
        print(f"[place] column rest centroid z = {obj_rest_z*1e3:.1f} mm")

        self.model = builder.finalize()
        # Friction: kernel mu_eff = 0.5*(particle_mu + shape_material_mu),
        # so set every channel to MU to get exactly MU at the grip.
        self.model.particle_mu = MU
        self.model.soft_contact_mu = MU
        self.model.shape_material_mu.assign(
            np.full(self.model.shape_count, MU, dtype=np.float32))
        # Cap particle velocity: a safety net against the SM-rigid +
        # first-contact "impact launch" at this small object scale. 2 m/s is
        # far above the mm/s grasp/lift dynamics, so the grip is unaffected.
        self.model.particle_max_velocity = 2.0

        # CSLC is enabled by passing a CSLCParams instance; None leaves the
        # fingertip lattice rigid (the --no-compliant A/B baseline).
        cslc_params = CSLCParams(
            clamp_delta_dot_max=1.0,
            ka_tangent_ratio=1.0,
            # Recover the friction-drag wrench the fingers feel from the
            # gripped object (the anchor reaction alone only encodes normal
            # compression) -- required to carry the object during LIFT.
            enable_lattice_pp_body_wrench=True,
        ) if self.use_cslc else None
        self.solver = newton.solvers.SolverUXPBD(
            self.model, iterations=8, stabilization_iterations=2,
            shock_propagation_k=1.0, cslc_params=cslc_params)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()

        # Object particle indices + resting baseline. The nominal rest
        # centroid is OBJ_HALF_H (cylinder stands from z=0 to z=2*OBJ_HALF_H);
        # the actual settled value is captured live during SETTLE so the
        # lift check is robust to small settling offsets.
        obj_idx = self.model.particle_groups[self.obj_group]
        if hasattr(obj_idx, "numpy"):
            obj_idx = obj_idx.numpy()
        self._obj_idx = np.asarray(list(obj_idx), dtype=np.int32)
        # Baselines captured live during SETTLE (see _record): the settled
        # object centroid and the PD-drooped grasp-pose hand height, both
        # used by test_final's lift / slip checks. Seeded with their nominal
        # values (box rest centroid from _build_box_column; hand target).
        self._obj_z_settled = obj_rest_z
        self._hand_z_grasp = GRASP_HAND_Z

        # Lattice (fingertip) particle indices = every particle not in the
        # object group, for the lattice<->object surface-gap telemetry.
        n_part = self.model.particle_count
        self._lat_pidx = np.setdiff1d(np.arange(n_part), self._obj_idx)
        self._part_r = self.model.particle_radius.numpy()

        # Peak compliance trackers (verified in test_final).
        self._peak_delta = 0.0
        self._peak_n_active = 0
        self.history: list[dict] = []

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(1.0, -1.0, 0.55),
                               pitch=-18.0, yaw=130.0)

    # ----- Phase machine ----------------------------------------------
    def _phase_targets(self, t: float):
        """Return (phase_name, q_arm[7], finger_target) at sim time ``t``.

        SETTLE : hold the grasp pose, fingers open (object settles between
                 the already-positioned open fingers).
        GRASP  : hold grasp pose, ease fingers open -> closed (CSLC skin
                 compresses around the column).
        LIFT   : ease the arm grasp -> lift, fingers held closed.
        HOLD   : hold lift pose, fingers closed.
        """
        if t < self.settle_t:
            return "settle", self.q_grasp, FINGER_OPEN
        t -= self.settle_t
        if t < self.grasp_t:
            w = _smoothstep(t / self.grasp_t)
            finger = FINGER_OPEN + w * (self.finger_closed - FINGER_OPEN)
            return "grasp", self.q_grasp, finger
        t -= self.grasp_t
        if t < self.lift_t:
            w = _smoothstep(t / self.lift_t)
            return "lift", self.q_grasp + w * (self.q_lift - self.q_grasp), self.finger_closed
        return "hold", self.q_lift, self.finger_closed

    def _apply_targets(self):
        phase, q_arm, finger = self._phase_targets(self.sim_time)
        self._phase = phase
        target = self.control.joint_target_pos.numpy()
        target[:7] = q_arm
        target[7:9] = finger
        self.control.joint_target_pos.assign(
            wp.array(target, dtype=wp.float32,
                     device=self.control.joint_target_pos.device))

    def simulate(self):
        self._apply_targets()
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
        self._record()

    # ----- Telemetry ---------------------------------------------------
    def _record(self):
        pq = self.state_0.particle_q.numpy()
        q = pq[self._obj_idx]
        v = self.state_0.particle_qd.numpy()[self._obj_idx]
        obj = q.mean(axis=0)
        # Track the settled centroid height during SETTLE so the lift check
        # is anchored to where the object actually came to rest.
        if self._phase == "settle":
            self._obj_z_settled = float(obj[2])
        v_max = float(np.linalg.norm(v, axis=1).max())
        bq = self.state_0.body_q.numpy()
        hand_z = float(bq[self.hand_idx, 2])
        # Capture the actual (PD-drooped) hand height while holding the grasp
        # pose, so the lift comparison uses the real grasp baseline.
        if self._phase == "settle":
            self._hand_z_grasp = hand_z
        # Finger closure from the finger BODY poses (joint_q is NOT updated
        # by the position-based solver, so it would read stale).
        finger_sep = float(np.linalg.norm(
            bq[self.lf_idx, :3] - bq[self.rf_idx, :3]))
        # Closest fingertip-sphere <-> object-sphere surface gap (negative =
        # interpenetrating, so CSLC should be active).
        lat = pq[self._lat_pidx]
        lr = self._part_r[self._lat_pidx][:, None]
        orr = self._part_r[self._obj_idx][None, :]
        d = np.linalg.norm(lat[:, None, :] - q[None, :, :], axis=2) - lr - orr
        min_gap = float(d.min())

        delta_max = delta_mean = 0.0
        n_active = 0
        if self.model.lattice_sphere_count > 0:
            ld = self.model.lattice_delta.numpy()
            mag = np.linalg.norm(ld, axis=1)
            delta_max = float(mag.max())
            delta_mean = float(mag.mean())
            n_active = int((mag > 1.0e-6).sum())
        self._peak_delta = max(self._peak_delta, delta_max)
        self._peak_n_active = max(self._peak_n_active, n_active)

        frame = int(round(self.sim_time * self.fps))
        row = {
            "frame": frame, "t": self.sim_time, "phase": self._phase,
            "obj_x": float(obj[0]), "obj_y": float(obj[1]),
            "obj_z": float(obj[2]), "obj_v_max": v_max, "hand_z": hand_z,
            "finger_sep": finger_sep, "min_gap": min_gap,
            "n_active": n_active, "delta_max": delta_max,
            "delta_mean": delta_mean,
        }
        self.history.append(row)
        if frame < 5 or frame % 20 == 0:
            print(f"[f={frame:03d} t={self.sim_time:4.2f} {self._phase:>6s}] "
                  f"obj=({obj[0]:+.3f},{obj[1]:+.3f},{obj[2]:+.3f}) "
                  f"hand_z={hand_z:+.3f} fsep={finger_sep*1e3:5.1f}mm "
                  f"min_gap={min_gap*1e3:+6.1f}mm |v|={v_max:4.2f} "
                  f"delta_max={delta_max*1e3:6.3f}mm "
                  f"n_act={n_active}/{self.model.lattice_sphere_count}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ----- Verification ------------------------------------------------
    def test_final(self):
        """Verify the full pipeline: IK reached the object, the fingertip
        CSLC skin compressed around it, and the column was lifted clear of
        the ground and held without slipping out or blowing up."""
        obj_q = self.state_0.particle_q.numpy()[self._obj_idx]
        obj_v = self.state_0.particle_qd.numpy()[self._obj_idx]
        assert np.isfinite(obj_q).all(), "NaN/Inf in object positions"
        assert np.isfinite(obj_v).all(), "NaN/Inf in object velocities"

        # 1. IK reached the targets (sub-mm hand pose error).
        assert self._ik_max_err < 1.0e-3, (
            f"IK did not reach the grasp poses: max err={self._ik_max_err*1e3:.2f} mm")

        obj_z = float(obj_q[:, 2].mean())
        hand_z = float(self.state_0.body_q.numpy()[self.hand_idx, 2])

        # 2. Cylinder was lifted clear of the ground (centroid rose well
        #    above its settled height) and not ejected.
        if obj_z < self._obj_z_settled + 0.04:
            raise RuntimeError(
                f"Object not lifted: obj_z={obj_z:.4f} "
                f"(settled~{self._obj_z_settled:.4f})")
        if obj_z > 0.6:
            raise RuntimeError(f"Object ejected: obj_z={obj_z:.4f}")

        # 3. Object stayed in the grip during the lift. The check is
        #    DIRECTIONAL: the failure mode is the object LAGGING the hand --
        #    i.e. rising less than the hand because it slipped down/out of
        #    the pads (the rigid --no-compliant baseline drops the column
        #    entirely). Seating the OTHER way -- the object climbing a few cm
        #    UP into the grip as the compliant pads draw it in during the
        #    lift -- is benign (it ends firmly held), so we do not penalise
        #    obj_rise > hand_rise beyond a generous sanity bound that still
        #    catches the object being flung.
        hand_rise = hand_z - self._hand_z_grasp
        obj_rise = obj_z - self._obj_z_settled
        assert hand_rise - obj_rise < 0.05, (
            f"Object lagged/slipped out of the grip: hand_rise={hand_rise*1e3:.1f} mm "
            f"obj_rise={obj_rise*1e3:.1f} mm (object rose far less than the hand)")
        assert obj_rise - hand_rise < 0.10, (
            f"Object climbed implausibly far in the grip (possible fling): "
            f"hand_rise={hand_rise*1e3:.1f} mm obj_rise={obj_rise*1e3:.1f} mm")

        # 4. The fingertip CSLC skin actually deformed (only when compliant).
        if self.use_cslc:
            assert self._peak_n_active > 0, "No fingertip lattice spheres ever engaged"
            assert self._peak_delta > 5.0e-5, (
                f"Fingertip compliance never engaged: "
                f"peak delta={self._peak_delta*1e3:.4f} mm (<0.05 mm)")

        # 5. No catastrophic velocity.
        v_max = float(np.linalg.norm(obj_v, axis=1).max())
        assert v_max < 5.0, f"Object moving too fast: v_max={v_max:.3f} m/s"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--no-compliant", action="store_true",
            help=("Disable the CSLC compliant fingertip lattice and run the "
                  "rigid-lattice baseline, for A/B comparison of the grip."))
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
