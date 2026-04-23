'''
    Phases
    ──────
    APPROACH   Pads move inward from free space toward the sphere.
    SQUEEZE    Pads press inward to build grip force.
    LIFT       Pads rise together; friction must overcome gravity.
    HOLD       Pads stationary in the air; sphere should stay gripped.

    Modes
    ─────
    viewer     Interactive OpenGL visualization (default).
    headless   Headless point vs CSLC comparison with printed summary.

    Flags
    ─────
    --contact-model {point,cslc}   Which contact model to visualise (viewer mode).
    --solver {mujoco,semi}         Physics solver backend (default: mujoco).
    --start-gripped                Skip APPROACH+SQUEEZE; spawn sphere already
                                    gripped at lift height. Useful for isolating
                                    the LIFT phase without ground-contact transients.
    --warm-start-sphere-vz         In --start-gripped mode, give the sphere an
                                    initial vz matching the lift speed so there is
                                    no velocity mismatch at t=0.
    --no-ground                    Remove the ground plane (reduces contact noise
                                    during the approach phase).
    
    uv run cslc_v1/robot_examples/robot_lift.py --viewer gl --contact-model point
'''

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.solvers import SolverSemiImplicit
from newton.solvers import SolverMuJoCo

from utils import _log, _section, _SEP, CSLC_FLAG, find_body_in_builder,\
                  inspect_model, count_active_contacts, read_cslc_state, recalibrate_cslc_kc_per_pad
from motion_planner import MotionPlan



def make_solver(model, solver_name, p: SceneParams):
    if solver_name == "mujoco":
        ncon = 5000
        if model.shape_cslc_spacing is not None:
            spacing = model.shape_cslc_spacing.numpy()
            flags = model.shape_flags.numpy()
            scale = model.shape_scale.numpy()
            for i in range(model.shape_count):
                if not (flags[i] & CSLC_FLAG):
                    continue
                sp = float(spacing[i])
                if sp <= 0:
                    continue
                hx, hy, hz = (float(scale[i][j]) for j in range(3))
                nx, ny, nz = (max(int(round(2*h/sp))+1, 2)
                              for h in [hx, hy, hz])
                interior = max(nx-2, 0)*max(ny-2, 0)*max(nz-2, 0)
                ncon += nx*ny*nz - interior
        return SolverMuJoCo(model, use_mujoco_contacts=False,
                            solver="cg", integrator="implicitfast",
                            cone="elliptic",
                            iterations=100, ls_iterations=10,
                            njmax=ncon, nconmax=ncon)
    elif solver_name == "semi":
        return SolverSemiImplicit(model)
    raise ValueError(f"Unknown solver: {solver_name}")



def create_articulation(builder: ModelBuilder, robot_base_pos: tuple):
    asset_path = newton.utils.download_asset("franka_emika_panda")
    builder.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=wp.transform(
            robot_base_pos,
            wp.quat_identity(),
        ),
        floating=False,
        scale=1,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        force_show_colliders=False,
    )
    builder.joint_q[7:] = [
        # 0.0,
        # -0.785398,
        # -1.0,
        # -2.356194,
        # 0.0,
        # 1.570796,
        # 0.785398,
        0.04,  # finger 1 (fully open)
        0.04,  # finger 2 (fully open)
    ]
    return builder



def _build_scene(scene_parameters: SceneParams, pad_cfg, sphere_cfg=None):
    """Build (optional) ground + robot + pads attached to the robot end effector + 1 free sphere.

    In start_gripped mode the sphere and pads both spawn at gripped_spawn_z,
    well above where the ground would be.  The prismatic-X joints are zeroed
    at the pad's world position, but the pads START at their gripped offset
    from each pad's joint origin — this is handled by positioning the joint
    parent_xform at the ungripped zero and setting the initial joint_q in
    set_initial_joint_q below.
    """
    scene = newton.ModelBuilder()
    
    robot = newton.ModelBuilder()
    robot = create_articulation(robot, scene_parameters.robot_base_pos)
    
    
    left_finger_idx = find_body_in_builder(robot, "fr3_leftfinger")
    right_finger_idx = find_body_in_builder(robot, "fr3_rightfinger")
    
    # scene.add_shape_box(pad, xform=box_xform,
    #                     hx=scene_parameters.pad_hx, hy=scene_parameters.pad_hy, hz=scene_parameters.pad_hz, cfg=pad_cfg)

    # Pad local transform: back face flush with the finger's inner gripping face (y=0),
    # extending inward (−y_local toward the sphere). Both fingers share the same local
    # transform because fr3_rightfinger is rotated 180° around Z relative to the hand,
    # so (0, −pad_hy, z_local) maps to the correct mirror position on each side.
    # Z is centred on the FR3 rubber tip (≈45.25 mm from the finger joint origin).
    pad_xform = wp.transform((0.0, -scene_parameters.pad_hy, scene_parameters.pad_local_z))

    robot.add_shape_box(body=left_finger_idx,
                        xform=pad_xform,
                        hx=scene_parameters.pad_hx,
                        hy=scene_parameters.pad_hy,
                        hz=scene_parameters.pad_hz,
                        cfg=pad_cfg)

    robot.add_shape_box(body=right_finger_idx,
                        xform=pad_xform,
                        hx=scene_parameters.pad_hx,
                        hy=scene_parameters.pad_hy,
                        hz=scene_parameters.pad_hz,
                        cfg=pad_cfg)
    
    scene.add_builder(robot)
    
    # Table — position and size driven by SceneParams; sphere rests on top.
    scene.add_shape_box(
        -1,
        wp.transform(
            wp.vec3(scene_parameters.table_center_x,
                    scene_parameters.table_center_y,
                    scene_parameters.table_surface_z - scene_parameters.table_hz),
            wp.quat_identity(),
        ),
        hx=scene_parameters.table_hx,
        hy=scene_parameters.table_hy,
        hz=scene_parameters.table_hz,
    )
    
    # Ground: optional.  Removing it eliminates the ground-release transient
    # as a possible launch trigger for CSLC diagnosis.
    if not scene_parameters.no_ground:
        ground_shape = scene.add_ground_plane()
    else:
        ground_shape = None
        

    # lx0 = -(scene_parameters.approach_gap / 2 + scene_parameters.pad_hx)
    # rx0 = +(scene_parameters.approach_gap / 2 + scene_parameters.pad_hx)
    # # In start_gripped mode we lift everything well above z=0 so any
    # # ground that WOULD exist (if add_ground_plane were called) stays out
    # # of reach for the duration of the test.
    # z0 = scene_parameters.gripped_spawn_z if scene_parameters.start_gripped else scene_parameters.sphere_start_z

    # # Ghost config for intermediate slider bodies (no collision)
    # ghost_cfg = newton.ModelBuilder.ShapeConfig(
    #     has_shape_collision=False, has_particle_collision=False, density=0.0)

    dof_map = {}
    # pad_joints = []

    # for label, x0 in [("left", lx0), ("right", rx0)]:
    #     # Slider body — intermediate link for X translation
    #     slider = scene.add_link(
    #         xform=wp.transform((x0, 0, z0), wp.quat_identity()),
    #         mass=0.1, label=f"{label}_slider")
    #     slider_shape = scene.add_shape_sphere(slider, radius=0.002, cfg=ghost_cfg)

    #     # Pad body — the gripping surface
    #     pad = scene.add_link(
    #         xform=wp.transform((x0, 0, z0), wp.quat_identity()),
    #         label=f"{label}_pad")

    #     # Orient the BOX SHAPE so its local +x face points toward the sphere.
    #     # CSLC uses the +x face as its 2-D contact lattice (see handler), so
    #     # both pads must present that face inward.
    #     #   Left pad  (at world -x): body's local +x already points toward
    #     #     the sphere at the origin — no rotation.
    #     #   Right pad (at world +x): rotate the SHAPE 180° around z so that
    #     #     its local +x maps to world -x (= inward).
    #     # We rotate the shape rather than the body so the prismatic joints
    #     # (which reference the body frame) are not affected.
    #     if label == "right":
    #         box_xform = wp.transform(
    #             wp.vec3(0.0, 0.0, 0.0),
    #             wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
    #         )
    #     else:
    #         box_xform = wp.transform_identity()
    #     scene.add_shape_box(pad, xform=box_xform,
    #                     hx=scene_parameters.pad_hx, hy=scene_parameters.pad_hy, hz=scene_parameters.pad_hz, cfg=pad_cfg)

    #     # Prismatic X: world → slider
    #     j_x = scene.add_joint_prismatic(
    #         parent=-1, child=slider,
    #         axis=wp.vec3(1, 0, 0),
    #         parent_xform=wp.transform((x0, 0, z0), wp.quat_identity()),
    #         child_xform=wp.transform((0, 0, 0), wp.quat_identity()),
    #         label=f"{label}_x")

    #     # Prismatic Z: slider → pad
    #     j_z = scene.add_joint_prismatic(
    #         parent=slider, child=pad,
    #         axis=wp.vec3(0, 0, 1),
    #         parent_xform=wp.transform((0, 0, 0), wp.quat_identity()),
    #         child_xform=wp.transform((0, 0, 0), wp.quat_identity()),
    #         label=f"{label}_z")

    #     scene.add_articulation([j_x, j_z], label=f"{label}_arm")
    #     pad_joints.extend([j_x, j_z])

    #     dof_map[f"{label}_x"] = scene.joint_qd_start[j_x]
    #     dof_map[f"{label}_z"] = scene.joint_qd_start[j_z]

    #     # Filter: ghost slider vs ground (avoid spurious contacts)
    #     if ground_shape is not None:
    #         scene.add_shape_collision_filter_pair(slider_shape, ground_shape)

    # # Set joint drives for pad joints (stiff PD position tracking)
    # for ji in pad_joints:
    #     dof = scene.joint_qd_start[ji]
    #     scene.joint_target_ke[dof] = scene_parameters.drive_ke
    #     scene.joint_target_kd[dof] = scene_parameters.drive_kd
    #     scene.joint_target_mode[dof] = int(JointTargetMode.POSITION)
    #     scene.joint_armature[dof] = 0.01

    # Dynamic sphere with explicit free joint (required for MuJoCo solver)
    sphere_z = (scene_parameters.gripped_spawn_z if scene_parameters.start_gripped
                else scene_parameters.sphere_start_z)
    sphere = scene.add_link(
        xform=wp.transform((scene_parameters.table_center_x,
                            scene_parameters.table_center_y,
                            sphere_z), wp.quat_identity()),
        label="sphere")
    s_cfg = sphere_cfg if sphere_cfg is not None else _sphere_cfg(scene_parameters)
    scene.add_shape_sphere(sphere, radius=scene_parameters.sphere_radius, cfg=s_cfg)
    j_free = scene.add_joint_free(sphere, label="sphere_free")
    scene.add_articulation([j_free], label="sphere")

    # Request the per-contact `force` extended attribute so we can read
    # actual solver-applied forces (for diagnostics) in Example.step.
    # Without this the contacts.force array is not allocated.
    scene.request_contact_attributes("force")

    # ── start_gripped mode: pre-position the pads at their squeeze-end X ──
    # In normal mode the pads start at dx=0 (outside the sphere) and the
    # APPROACH phase drives them inward.  In start_gripped mode we jump
    # the pads' prismatic-X joint_q directly to the gripped position so
    # there is no approach-phase transient, no ground contact, nothing
    # but the LIFT dynamics under test.
    if scene_parameters.start_gripped:
        dx_gripped = (
            scene_parameters.approach_speed * scene_parameters.approach_duration
            + scene_parameters.squeeze_speed * scene_parameters.squeeze_duration
        )
        # joint_q is indexed by DOF.  left_x → +dx_gripped (pad moves +x,
        # i.e. inward from lx0); right_x → -dx_gripped (inward from rx0).
        scene.joint_q[dof_map["left_x"]] = +dx_gripped
        scene.joint_q[dof_map["right_x"]] = -dx_gripped

        # ── Warm-start sphere velocity to match LIFT speed ──
        # Without this, at t=0 the sphere has vz=0 while the pads ramp up
        # to vz=lift_speed over lift_ramp_duration.  That velocity mismatch
        # drives regularized Coulomb friction into sustained slip, producing
        # a ~25mm upward overshoot (classical regularized-friction artifact,
        # which cannot model true static friction).
        #
        # If warm_start_sphere_vz is True, initialize the sphere's free-joint
        # vz to lift_speed so there is NO relative velocity at t=0, no slip,
        # and therefore no friction-driven overshoot.  This is a diagnostic
        # intervention to confirm the overshoot origin, not a production fix
        # — in real tasks we can't warm-start grasped objects.
        if scene_parameters.warm_start_sphere_vz:
            # Free joint qd is [wx, wy, wz, vx, vy, vz] (Featherstone order).
            # Sphere body is the last link added; its free joint DOFs are the
            # last 6 entries of joint_qd.  We only set vz (index 5 of the 6).
            # Alternative explicit lookup: find the free-joint DOF via joints
            # metadata, but for this fixed scene layout the last-6 works.
            sphere_vz_idx = len(scene.joint_qd) - 1  # last qd entry == sphere vz
            scene.joint_qd[sphere_vz_idx] = scene_parameters.lift_speed

    m = scene.finalize()
    m.set_gravity(scene_parameters.gravity)

    _log(f"DOF map: {dof_map}")
    _log(f"bodies={m.body_count}  shapes={m.shape_count}  "
         f"joints={m.joint_count}  DOFs={m.joint_dof_count}")
    if scene_parameters.start_gripped:
        _log(f"start_gripped=True  dx_initial={dx_gripped*1e3:.2f}mm  "
             f"spawn_z={sphere_z:.3f}m  no_ground={scene_parameters.no_ground}")

    return m, dof_map


def build_point_scene(p: SceneParams):
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density)
    return _build_scene(p, cfg)


def build_cslc_scene(p: SceneParams):
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density,
        is_cslc=True,
        cslc_spacing=p.cslc_spacing, cslc_ka=p.cslc_ka, cslc_kl=p.cslc_kl,
        cslc_dc=p.cslc_dc, cslc_n_iter=p.cslc_n_iter, cslc_alpha=p.cslc_alpha)
    return _build_scene(p, cfg)

def _sphere_cfg(p: SceneParams):
    return newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.sphere_density)


def build_hydro_scene(p: SceneParams):
    """Build the same articulated-pad scene with hydroelastic contact.

    Both pads AND the sphere need is_hydroelastic=True (PFC requires both
    bodies to carry pressure fields).  kh is the hydroelastic modulus [Pa];
    see SceneParams.kh docstring and section 9 of convo_april_19.md.
    """
    pad_cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density,
        kh=p.kh, is_hydroelastic=True, sdf_max_resolution=p.sdf_resolution)
    sphere_cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.sphere_density,
        kh=p.kh, is_hydroelastic=True, sdf_max_resolution=p.sdf_resolution)
    return _build_scene(p, pad_cfg, sphere_cfg=sphere_cfg)

BUILDERS = {
    "point": build_point_scene,
    "cslc": build_cslc_scene,
    "hydro": build_hydro_scene,
}


@dataclass
class SceneParams:
    """All knobs for the gripper lift scene."""

    # Sphere
    sphere_radius: float = 0.03
    sphere_density: float = 4421.0

    # Table geometry and world position
    table_center_x: float = 0.0
    table_center_y: float = 2.0
    table_surface_z: float = 0.1   # z of the top surface [m]
    table_hx: float = 0.1
    table_hy: float = 0.1
    table_hz: float = 0.05

    # Robot mount (position derived from table)
    robot_standoff: float = 0.1    # gap from robot base to table front edge [m]
    robot_mount_z: float = -0.1    # floor-mount height offset [m]

    # Pads (box half-extents and local finger-frame z-offset along finger axis)
    pad_hx: float = 0.01
    pad_hy: float = 0.005
    pad_hz: float = 0.01
    pad_local_z: float = 0.04525   # local z along finger axis — rubber-tip centre [m]
    pad_density: float = 1000.0

    # Phase timing
    #
    # Geometry at dx=0: pad inner face sits at x = ±(approach_gap/2) = ±50 mm.
    # Sphere surface at ±sphere_radius = ±30 mm.
    # So the pad-face-to-sphere-surface gap at APPROACH start = 50 − 30 = 20 mm.
    #
    # Target:
    #   APPROACH end  — pad face at the sphere surface (dx_app = 20 mm, face-pen = 0)
    #   SQUEEZE  end  — pad face 1 mm inside the sphere  (+1 mm face-pen)
    #
    # Why 1 mm: matches the paper's weight-capacity experiment (μ=0.5, pen=1 mm,
    # 2.5 kg object). At that compression the lattice is in its small-deformation
    # regime (δ << r_lat) and ke_bulk means what you expect it to mean.
    # Time for the arm to move from its initial pose to the grasp position above the sphere.
    reach_duration: float = 3.0

    approach_gap: float = 0.10
    approach_speed: float = 20e-3 / 1.5   # 13.33 mm/s → travels 20 mm over 1.5 s
    approach_duration: float = 1.5

    squeeze_speed: float = 1e-3 / 0.5     # 2 mm/s → travels 1 mm over 0.5 s
    squeeze_duration: float = 0.5

    lift_speed: float = 0.015
    lift_duration: float = 1.5

    # Smooth the target-velocity transition between SQUEEZE and LIFT over
    # this many seconds.  Without ramping, the target velocity jumps 0 →
    # lift_speed in a single timestep; the high-gain position PD drive
    # turns that into an ~impulsive pad velocity, which saturates friction
    # (μ·Fn) against the stationary sphere and launches it ballistically.
    # A 250 ms smoothstep puts the transition well above the drive's own
    # time constant (√(m/ke) ≈ 1 ms) so the pad follows the target faithfully
    # and the sphere accelerates gradually.
    lift_ramp_duration: float = 0.25

    hold_duration: float = 1.0

    # Material (shared by point + CSLC) — matched to squeeze_test.py.
    #
    # With spacing=5mm, the contact face (40×100 mm) has 9×21=189 surface
    # spheres per pad.  calibrate_kc targets contact_fraction=0.15 (≈28 per
    # pad), giving kc=1087 N/m and keff=892.9 N/m per sphere.
    #
    # Friction budget at face-pen ≈ 2 mm with ~10 active spheres per side:
    #   Fn_total = 2 × 10 × 892.9 × 0.002 = 35.7 N
    #   F_friction = μ × Fn = 0.5 × 35.7 = 17.9 N  vs  weight = 4.91 N  (3.6×)
    #
    # NOTE: pre-2026-04-19 comment said "ke=5000 WILL launch sphere" — that
    # was written before out_damping=0 fix (friction timeconst was 1.0s then,
    # now ≈0.030s). With stiff friction the sphere follows the pads smoothly.
    ke: float = 5.0e4     # matched to squeeze_test
    kd: float = 5.0e2     # matched to squeeze_test
    kf: float = 100.0
    mu: float = 0.5

    # Joint drive stiffness
    drive_ke: float = 5.0e4
    drive_kd: float = 1.0e3

    # CSLC tuning — matched to squeeze_test.py
    cslc_spacing: float = 0.005
    cslc_ka: float = 5000.0
    cslc_kl: float = 500.0
    cslc_dc: float = 2.0
    cslc_n_iter: int = 20
    cslc_alpha: float = 0.6
    # Per-pad contact fraction for kc recalibration after handler build.
    # The lift scene uses face_pen ≈ 1 mm vs squeeze_test's 15 mm, so the
    # active patch contains only ~5 spheres per pad (vs 87 in squeeze).
    # The handler's default cf=0.3 over-estimates the active count by ~11×,
    # leaving CSLC's per-pad aggregate stiffness an order of magnitude below
    # ke_bulk.  Setting cf to the empirical fraction restores the calibration
    # invariant: per-pad aggregate stiffness = ke_bulk.
    # Set None to keep the handler's default kc.
    cslc_contact_fraction: float | None = 0.025

    # Hydroelastic modulus [Pa] for the hydro contact model.  See section 9
    # in cslc_v1/convo_april_19.md for the kh stability sweep — 1e8 is
    # silicone-rubber-stiff and stable; 1e10 ejects the sphere.
    kh: float = 1.0e8
    sdf_resolution: int = 64

    # Integration
    dt: float = 1.0 / 500.0
    gravity: tuple = (0.0, 0.0, -9.81)

    # ── Diagnostic / decoupling options ────────────────────────────────────
    # no_ground:     skip add_ground_plane().  Eliminates the ground-contact
    #                elastic-rebound confounder (previous agent's hypothesis C).
    # start_gripped: place the sphere in the air with pads already at
    #                squeeze-end position, touching it.  Skips APPROACH
    #                and SQUEEZE; test begins at LIFT t=0.
    # The two together isolate "can friction carry the sphere?" from "is
    # the launch triggered by ground release?".
    no_ground: bool = False
    start_gripped: bool = False
    # Warm-start the sphere's vz to lift_speed at t=0 in start_gripped mode.
    # Diagnostic for the friction-overshoot hypothesis: if v_rel=0 at t=0,
    # regularized Coulomb can't drive slip, and the overshoot should not
    # appear.  If overshoot still appears with this flag, the hypothesis
    # is wrong and something else is going on.
    warm_start_sphere_vz: bool = False
    # Height at which to spawn the sphere in start_gripped mode.  Chosen so
    # the sphere sits well above where the ground would be even if no_ground
    # is False, so ground contact never becomes relevant during the test.
    gripped_spawn_z: float = 0.20

    @property
    def sphere_start_z(self) -> float:
        """Sphere centre z [m] — rests on the table surface."""
        return self.table_surface_z + self.sphere_radius

    @property
    def robot_base_pos(self) -> tuple[float, float, float]:
        """Robot base world position — directly in front of the table."""
        return (
            self.table_center_x,
            self.table_center_y - self.table_hy - self.robot_standoff,
            self.robot_mount_z,
        )

    @property
    def sphere_mass(self):
        return self.sphere_density * (4 / 3) * math.pi * self.sphere_radius ** 3

    @property
    def reach_steps(self):
        return int(self.reach_duration / self.dt)

    @property
    def approach_steps(self):
        return int(self.approach_duration / self.dt)

    @property
    def squeeze_steps(self):
        return int(self.squeeze_duration / self.dt)

    @property
    def lift_steps(self):
        return int(self.lift_duration / self.dt)

    @property
    def hold_steps(self):
        return int(self.hold_duration / self.dt)

    @property
    def total_steps(self):
        # In start_gripped mode APPROACH + SQUEEZE are skipped: the pads
        # start at their squeeze-end position via joint_q in _build_scene,
        # and the test begins directly at LIFT t=0.  Including those phases
        # in total_steps would just hold the gripped configuration stationary
        # for 2 s before lifting, which is harmless but wastes wall time.
        if self.start_gripped:
            return self.lift_steps + self.hold_steps
        return (self.reach_steps + self.approach_steps + self.squeeze_steps
                + self.lift_steps + self.hold_steps)

    def phase_of(self, step):
        s = step
        # When start_gripped, virtually fast-forward past APPROACH+SQUEEZE so
        # _pad_state sees LIFT from step 0.  The integer shift is exactly the
        # number of (virtual) steps we skipped; _pad_state's t = s * dt then
        # starts from t=0 within the LIFT phase, which is what we want.
        phases = [("LIFT", self.lift_steps), ("HOLD", self.hold_steps)]
        if not self.start_gripped:
            phases = [("REACH", self.reach_steps),
                      ("APPROACH", self.approach_steps),
                      ("SQUEEZE", self.squeeze_steps)] + phases
        for name, dur in phases:
            if s < dur:
                return name, s
            s -= dur
        return "HOLD", 0

    def dump(self):
        _section("SCENE PARAMETERS")
        m = self.sphere_mass
        bx, by, bz = self.robot_base_pos
        _log(f"Table:  center=({self.table_center_x:.2f},{self.table_center_y:.2f})  "
             f"surface_z={self.table_surface_z*1e3:.0f}mm  "
             f"hx={self.table_hx*1e3:.0f}mm  hy={self.table_hy*1e3:.0f}mm  hz={self.table_hz*1e3:.0f}mm")
        _log(f"Sphere: r={self.sphere_radius*1e3:.1f}mm  mass={m*1e3:.1f}g  weight={m*9.81:.3f}N  "
             f"start_z={self.sphere_start_z*1e3:.1f}mm")
        _log(f"Robot:  base=({bx:.2f},{by:.2f},{bz:.2f})  standoff={self.robot_standoff*1e3:.0f}mm")
        _log(f"Pads:   hx={self.pad_hx*1e3:.0f}mm  hy={self.pad_hy*1e3:.0f}mm  "
             f"hz={self.pad_hz*1e3:.0f}mm  local_z={self.pad_local_z*1e3:.0f}mm  density={self.pad_density:.0f}")
        _log(f"Phases: reach={self.reach_duration}s  approach={self.approach_duration}s  "
             f"squeeze={self.squeeze_duration}s  lift={self.lift_duration}s  hold={self.hold_duration}s")
        _log(f"Material: ke={self.ke:.0f}  kd={self.kd:.0f}  μ={self.mu:.2f}")
        _log(f"Drive: ke={self.drive_ke:.0f}  kd={self.drive_kd:.0f}")
        _log(f"Steps: {self.total_steps} total  dt={self.dt*1e3:.2f}ms")




SPHERE_BODY = 4
LEFT_PAD = 1
RIGHT_PAD = 3

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.test_mode = args.test
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(self.frame_dt / 0.002))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_step = 0

        self.contact_model = getattr(args, "contact_model", "cslc")
        self.solver_name = getattr(args, "solver", "mujoco")

        self.p = SceneParams(
            dt=self.sim_dt,
            no_ground=getattr(args, "no_ground", False),
            start_gripped=getattr(args, "start_gripped", False),
            warm_start_sphere_vz=getattr(args, "warm_start_sphere_vz", False),
        )
        self.p.dump()

        self.model, self.dof_map = BUILDERS[self.contact_model](self.p)
        inspect_model(self.model, self.contact_model)

        self.solver = make_solver(self.model, self.solver_name, self.p)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        self.current_z = self.p.sphere_start_z
        self.current_contacts = 0
        self.current_phase = "LIFT" if self.p.start_gripped else "REACH"

        # FD telemetry state (body_qd stale under MuJoCo GPU — use position FD).
        self._prev_sphere_z = None
        self._prev_pad_z = None
        self._prev_vz_sphere = None
        self._prev_vz_pad = None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.3, -0.3, self.p.sphere_start_z + 0.15),
            pitch=-15.0, yaw=135.0)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

        q = self.state_0.body_q.numpy()
        _log(f"INITIAL STATE ({self.model.body_count} bodies):")
        for bi in range(self.model.body_count):
            _log(
                f"  body {bi}: pos=({q[bi, 0]:.4f},{q[bi, 1]:.4f},{q[bi, 2]:.4f})", 1)
        _log(
            f"sim_substeps={self.sim_substeps}  sim_dt={self.sim_dt*1e3:.3f}ms")

        self._build_motion_plan()

    # ------------------------------------------------------------------
    # Motion plan
    # ------------------------------------------------------------------

    def _build_motion_plan(self):
        """Construct a joint-space waypoint trajectory for the grasp-and-lift
        task.  The simulate() loop plays this trajectory back kinematically by
        overriding state_0.joint_q / joint_qd every substep.

        Waypoint layout (9 DOFs: 7 arm + 2 fingers):
            0  initial pose                            (start)
            1  arm above sphere, fingers open           → end of REACH
            2  arm above sphere, fingers at sphere      → end of APPROACH
            3  arm above sphere, fingers fully closed   → end of SQUEEZE
            4  arm raised by lift_speed·lift_duration   → end of LIFT
            5  same as 4 (stationary)                   → end of HOLD
        """
        # ── Initial configuration (arm + fingers) ──────────────────────
        q0_full = self.model.joint_q.numpy().copy()
        q_initial = q0_full[:9].copy()

        # ── 3-D FK Jacobian at the initial configuration ──────────────
        # Rows = world X, Y, Z at the left-pad body; columns = arm DOFs.
        N_ARM = 7
        eps = 1e-5
        scratch = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, scratch)
        ee0 = scratch.body_q.numpy()[LEFT_PAD, :3].copy()

        J = np.zeros((3, N_ARM))
        for i in range(N_ARM):
            q_p = q0_full.copy()
            q_p[i] += eps
            newton.eval_fk(self.model, wp.array(q_p, dtype=wp.float32),
                           self.model.joint_qd, scratch)
            J[:, i] = (scratch.body_q.numpy()[LEFT_PAD, :3] - ee0) / eps
        J_pinv = np.linalg.pinv(J)

        # ── Arm pose that places the EE at the sphere centre ───────────
        ee_sphere = np.array([self.p.table_center_x,
                              self.p.table_center_y,
                              self.p.sphere_start_z])
        dq_reach = J_pinv @ (ee_sphere - ee0)
        q_arm_grasp = q_initial[:N_ARM] + dq_reach

        # ── Arm pose after LIFT: grasp pose raised by lift_distance ───
        lift_distance = self.p.lift_speed * self.p.lift_duration
        Jz = J[2, :]
        denom = float(Jz @ Jz)
        dq_lift = (Jz / denom) * lift_distance if denom > 1e-12 else np.zeros(N_ARM)
        q_arm_lifted = q_arm_grasp + dq_lift

        # ── Finger values per phase (empirical; tune if the pads miss) ──
        FINGER_OPEN      = 0.04   # fully open at start / end of REACH
        FINGER_CONTACT   = 0.015  # closed enough to touch the sphere surface
        FINGER_GRIP      = 0.0    # fully closed — builds grip normal force

        def make_wp(arm, finger_val):
            q = np.empty(9, dtype=float)
            q[:N_ARM] = arm
            q[N_ARM] = finger_val      # left finger
            q[N_ARM + 1] = finger_val  # right finger
            return q

        wp_reach    = make_wp(q_arm_grasp,   FINGER_OPEN)
        wp_approach = make_wp(q_arm_grasp,   FINGER_CONTACT)
        wp_squeeze  = make_wp(q_arm_grasp,   FINGER_GRIP)
        wp_lift     = make_wp(q_arm_lifted,  FINGER_GRIP)
        wp_hold     = make_wp(q_arm_lifted,  FINGER_GRIP)

        if self.p.start_gripped:
            # Skip REACH + APPROACH + SQUEEZE — start already gripped.
            waypoints = np.array([wp_squeeze, wp_lift, wp_hold])
            segment_times = np.array([self.p.lift_duration, self.p.hold_duration])
        else:
            waypoints = np.array([q_initial, wp_reach, wp_approach,
                                  wp_squeeze, wp_lift, wp_hold])
            segment_times = np.array([self.p.reach_duration,
                                      self.p.approach_duration,
                                      self.p.squeeze_duration,
                                      self.p.lift_duration,
                                      self.p.hold_duration])

        self.motion_plan = MotionPlan(waypoints, segment_times)

        _log(f"Motion plan: {len(waypoints)} waypoints, T={self.motion_plan.traj_time:.2f}s")
        _log(f"  ee0={np.round(ee0, 3)} → grasp={np.round(ee_sphere, 3)}  "
             f"Δq_reach={np.round(dq_reach, 3)}")
        _log(f"  lift Δz={lift_distance:.3f}m  Δq_lift={np.round(dq_lift, 3)}")

    # ------------------------------------------------------------------

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Kinematic trajectory playback: override arm + finger joint state
            # (DOFs 0:9) from the motion plan at the current substep time.
            # Sphere DOFs (9:) are left untouched so free dynamics apply.
            t = self.sim_step * self.sim_dt
            q_des, qd_des = self.motion_plan.state_at(t)

            jq = self.state_0.joint_q.numpy()
            jq[:9] = q_des
            self.state_0.joint_q.assign(jq)

            jqd = self.state_0.joint_qd.numpy()
            jqd[:9] = qd_des
            self.state_0.joint_qd.assign(jqd)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_step += 1
            

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

        q = self.state_0.body_q.numpy()
        self.current_z = float(q[SPHERE_BODY, 2])
        self.current_contacts = count_active_contacts(self.contacts)
        self.current_phase, _ = self.p.phase_of(self.sim_step)

        # Print every ~100 ms.
        if self.sim_step % (self.sim_substeps * 6) < self.sim_substeps:
            phase, _ = self.p.phase_of(self.sim_step)
            cslc = read_cslc_state(
                self.model) if self.contact_model == "cslc" else None
            cslc_str = f"  cslc={cslc['n_active']}/{cslc['n_surface']}" if cslc else ""
            sphere_z = float(q[SPHERE_BODY, 2])
            pad_z = float(q[LEFT_PAD, 2])
            delta_z = sphere_z - pad_z

            # FD velocity/acceleration from position history.
            # body_qd is stale under MuJoCo GPU solver (always reads ≈0).
            dt_print = 6.0 * self.frame_dt
            if self._prev_sphere_z is None:
                vz_sphere = 0.0
                vz_pad = 0.0
            else:
                vz_sphere = (sphere_z - self._prev_sphere_z) / dt_print
                vz_pad = (pad_z - self._prev_pad_z) / dt_print

            if self._prev_vz_sphere is None:
                az_sphere = 0.0
            else:
                az_sphere = (vz_sphere - self._prev_vz_sphere) / dt_print

            self._prev_sphere_z = sphere_z
            self._prev_pad_z = pad_z
            self._prev_vz_sphere = vz_sphere
            self._prev_vz_pad = vz_pad

            # ── Inferred vertical contact force on the sphere ──
            # Newton's second law:  m*a_z = Σ F_z = F_contact_z + F_gravity_z
            # so F_contact_z = m*a_z - m*g_z = m*(a_z + g_mag) because
            # g_z = -g_mag (gravity points down).  If the sphere is sitting
            # stationary under gravity in a perfect grip, a_z=0 and this
            # equals m*g = weight (contacts support the weight).  If the
            # sphere is being launched upward, F_contact_z >> weight.
            g_mag = abs(self.p.gravity[2])
            m_sphere = self.p.sphere_mass
            F_contact_z = m_sphere * (az_sphere + g_mag)
            weight_N = m_sphere * g_mag

            _log(f"[{phase:8s}] step={self.sim_step:5d}  "
                 f"sz={sphere_z:+.4f} pz={pad_z:+.4f} Δz={delta_z*1e3:+6.2f}mm  "
                 f"vz_s={vz_sphere:+.4f} vz_p={vz_pad:+.4f} "
                 f"az_s={az_sphere:+6.2f}m/s²  "
                 f"F_c={F_contact_z:+6.2f}N (W={weight_N:.2f}N)  "
                 f"n={self.current_contacts}{cslc_str}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _render_ui(self, imgui):
        imgui.text(f"Contact: {self.contact_model}")
        imgui.text(f"Phase:   {self.current_phase}")
        imgui.text(f"Sphere Z: {self.current_z:.4f}")
        imgui.text(f"Contacts: {self.current_contacts}")
        imgui.text(f"Step:    {self.sim_step}")

    def test_final(self):
        if self.contact_model == "cslc":
            assert self.current_z > 0.01, f"Sphere fell: z={self.current_z:.4f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--contact-model", type=str, default="cslc",
                            choices=["point", "cslc", "hydro"])
        parser.add_argument("--contact-models", type=str, nargs='+', default=[],
                            help="Comma-separated list for headless mode "
                                 "(e.g. 'point,cslc,hydro').")
        parser.add_argument("--solver", type=str, default="mujoco",
                            choices=["mujoco", "semi"])
        parser.add_argument("--mode", type=str, default="viewer",
                            choices=["viewer", "headless"])
        parser.add_argument("--start-gripped", action="store_true",
                            help="Skip APPROACH+SQUEEZE; start at squeeze-end position.")
        parser.add_argument("--no-ground", action="store_true",
                            help="Skip ground plane.")
        parser.add_argument("--warm-start-sphere-vz", action="store_true",
                            help="Init sphere vz = lift_speed at t=0.")
        parser.add_argument("--cslc-ka", type=float, default=None,
                            help="Override CSLC anchor stiffness ka [N/m].")
        parser.add_argument("--cslc-contact-fraction", type=float, default=None,
                            help="Override CSLC contact fraction for kc recalibration.")
        parser.add_argument("--kh", type=float, default=None,
                            help="Override hydroelastic modulus [Pa].")
        return parser


def main():
    parser = Example.create_parser()
    args, _ = parser.parse_known_args()

    wp.init()
    print(f"\n{'━' * 60}\n  ROBOT LIFT TEST — mode: {args.mode}\n{'━' * 60}")

    if args.mode == "viewer":
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
    else:
        args = parser.parse_args()
        sn = args.solver
        cm_list = args.contact_models 
        scene_kwargs = dict(
            no_ground=getattr(args, "no_ground", False),
            start_gripped=getattr(args, "start_gripped", False),
            warm_start_sphere_vz=getattr(args, "warm_start_sphere_vz", False),
        )
        if getattr(args, "cslc_ka", None) is not None:
            scene_kwargs["cslc_ka"] = float(args.cslc_ka)
        if getattr(args, "cslc_contact_fraction", None) is not None:
            scene_kwargs["cslc_contact_fraction"] = float(args.cslc_contact_fraction)
        if getattr(args, "kh", None) is not None:
            scene_kwargs["kh"] = float(args.kh)
            
        test_headless(SceneParams(**scene_kwargs), sn, contact_models=cm_list)


if __name__ == "__main__":
    main()
