#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gripper lift test — point contact vs CSLC.

A sphere sits on a ground plane.  Two articulated pads (driven by prismatic
joints) approach, squeeze, lift, and hold the sphere in the air.

Because the pads are DYNAMIC bodies driven by joint motors (not kinematic),
the solver correctly computes relative velocity at the contact → friction
can drag the sphere upward during the lift phase.

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

Usage
─────
  # Live viewer — full cycle (approach → squeeze → lift → hold):
  uv run cslc_v1/lift_test.py --viewer gl --contact-model cslc
  uv run cslc_v1/lift_test.py --viewer gl --contact-model point

  # Skip to lift phase immediately (no approach/squeeze transient):
  uv run cslc_v1/lift_test.py --viewer gl --contact-model cslc --start-gripped
  uv run cslc_v1/lift_test.py --viewer gl --contact-model cslc --start-gripped --no-ground

  # Headless batch comparison (both models, prints summary):
  uv run cslc_v1/lift_test.py --mode headless
  uv run cslc_v1/lift_test.py --mode headless --solver semi
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode

from cslc_v1.common import (
    CSLC_FLAG,
    HAS_MUJOCO,
    _SEP,
    _log,
    _section,
    count_active_contacts,
    get_cslc_lattice_viz_data,
    inspect_model,
    make_solver,
    read_cslc_state,
    recalibrate_cslc_kc_per_pad,
)


# ── Scene parameters ─────────────────────────────────────────────────────


@dataclass
class SceneParams:
    """All knobs for the gripper lift scene."""

    # Sphere
    sphere_radius: float = 0.03
    sphere_density: float = 4421.0
    sphere_start_z: float = 0.05

    # Pads (box half-extents)
    pad_hx: float = 0.01
    pad_hy: float = 0.02
    pad_hz: float = 0.05
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

    # CSLC tuning — matched to squeeze_test.py and the fair-calibration
    # derivation in §2 of cslc_v1/summary.md.  Under the H1 three-spring
    # series (anchor, contact, target), the recalibration formula
    # `1/kc = N/ke_bulk − 1/ka − 1/ke_target` requires ka above the
    # threshold ke_bulk/(N − ke_bulk/ke_target) ≈ 16667 to admit a
    # positive kc.  ka=25000 leaves a comfortable margin and solves to
    # kc=50000 → keff=12500 → aggregate per pad = 50000 N/m = ke_bulk ✓.
    cslc_spacing: float = 0.005
    cslc_ka: float = 25000.0
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

    # Hydroelastic modulus [Pa] for the hydro contact model.
    # Fair-calibrated default: kh_eff · A_patch(1 mm pen) = ke_bulk,
    # where kh_eff = kh_pad·kh_sphere/(kh_pad+kh_sphere) is the harmonic-
    # mean composition of pad and sphere moduli (Newton paper §II-A,
    # get_effective_stiffness in sdf_hydroelastic.py).  With both bodies
    # at the same kh, kh_eff = kh/2, so kh = 2·ke_bulk/A_patch.
    # At r=30 mm sphere on flat pad, A_patch ≈ π·(2·r·pen) = 188 mm²,
    # so kh = 2·5e4 / 1.88e-4 = 5.3e8 Pa.  See §2 in cslc_v1/summary.md.
    kh: float = 5.3e8
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
    def sphere_mass(self):
        return self.sphere_density * (4 / 3) * math.pi * self.sphere_radius ** 3

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
        return self.approach_steps + self.squeeze_steps + self.lift_steps + self.hold_steps

    def phase_of(self, step):
        s = step
        # When start_gripped, virtually fast-forward past APPROACH+SQUEEZE so
        # _pad_state sees LIFT from step 0.  The integer shift is exactly the
        # number of (virtual) steps we skipped; _pad_state's t = s * dt then
        # starts from t=0 within the LIFT phase, which is what we want.
        phases = [("LIFT", self.lift_steps), ("HOLD", self.hold_steps)]
        if not self.start_gripped:
            phases = [("APPROACH", self.approach_steps),
                      ("SQUEEZE", self.squeeze_steps)] + phases
        for name, dur in phases:
            if s < dur:
                return name, s
            s -= dur
        return "HOLD", 0

    def dump(self):
        _section("SCENE PARAMETERS")
        m = self.sphere_mass
        _log(
            f"Sphere: r={self.sphere_radius*1e3:.1f}mm  mass={m*1e3:.1f}g  weight={m*9.81:.3f}N")
        _log(f"Pads:   hx={self.pad_hx*1e3:.0f}mm  hy={self.pad_hy*1e3:.0f}mm  "
             f"hz={self.pad_hz*1e3:.0f}mm  density={self.pad_density:.0f}")
        _log(f"Phases: approach={self.approach_duration}s  squeeze={self.squeeze_duration}s  "
             f"lift={self.lift_duration}s  hold={self.hold_duration}s")
        _log(f"Material: ke={self.ke:.0f}  kd={self.kd:.0f}  μ={self.mu:.2f}")
        _log(f"Drive: ke={self.drive_ke:.0f}  kd={self.drive_kd:.0f}")
        _log(f"Steps: {self.total_steps} total  dt={self.dt*1e3:.2f}ms")


# ── Metrics ───────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    name: str = ""
    sphere_z: list[float] = field(default_factory=list)
    contacts: list[int] = field(default_factory=list)

    @property
    def max_z(self):
        return max(self.sphere_z) if self.sphere_z else 0.0

    @property
    def final_z(self):
        return self.sphere_z[-1] if self.sphere_z else 0.0

    @property
    def lifted(self):
        # True if sphere rose more than 5 mm above its resting position.
        # Uses min(sphere_z) as reference (settled ground height ≈ sphere_radius),
        # not sphere_z[0] which is the spawn height before the sphere drops.
        if len(self.sphere_z) < 2:
            return False
        return self.max_z > min(self.sphere_z) + 0.005

    @property
    def held(self):
        # True if sphere bottom is off the ground at end of simulation.
        # sphere bottom = final_z - sphere_radius; > 2 mm → clearly airborne.
        return self.final_z > 0.032 if self.sphere_z else False


# ── Scene builders ────────────────────────────────────────────────────────
#
#  Each pad is a 2-DOF articulated arm:
#    world ──[prismatic X]──> slider ──[prismatic Z]──> pad
#
#  The sphere has an explicit free joint so MuJoCo tracks it properly.
#
#  Body layout (ground plane creates no body):
#    0: left_slider    2: right_slider    4: sphere
#    1: left_pad       3: right_pad
# ─────────────────────────────────────────────────────────────────────────


def _sphere_cfg(p: SceneParams):
    return newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.sphere_density)


def _build_scene(p: SceneParams, pad_cfg, sphere_cfg=None):
    """Build (optional) ground + 2 articulated pads + 1 free sphere.

    In start_gripped mode the sphere and pads both spawn at gripped_spawn_z,
    well above where the ground would be.  The prismatic-X joints are zeroed
    at the pad's world position, but the pads START at their gripped offset
    from each pad's joint origin — this is handled by positioning the joint
    parent_xform at the ungripped zero and setting the initial joint_q in
    set_initial_joint_q below.
    """
    b = newton.ModelBuilder()

    # Ground: optional.  Removing it eliminates the ground-release transient
    # as a possible launch trigger for CSLC diagnosis.
    if not p.no_ground:
        ground_shape = b.add_ground_plane()
    else:
        ground_shape = None

    lx0 = -(p.approach_gap / 2 + p.pad_hx)
    rx0 = +(p.approach_gap / 2 + p.pad_hx)
    # In start_gripped mode we lift everything well above z=0 so any
    # ground that WOULD exist (if add_ground_plane were called) stays out
    # of reach for the duration of the test.
    z0 = p.gripped_spawn_z if p.start_gripped else p.sphere_start_z

    # Ghost config for intermediate slider bodies (no collision)
    ghost_cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False, has_particle_collision=False, density=0.0)

    dof_map = {}
    pad_joints = []

    for label, x0 in [("left", lx0), ("right", rx0)]:
        # Slider body — intermediate link for X translation
        slider = b.add_link(
            xform=wp.transform((x0, 0, z0), wp.quat_identity()),
            mass=0.1, label=f"{label}_slider")
        slider_shape = b.add_shape_sphere(slider, radius=0.002, cfg=ghost_cfg)

        # Pad body — the gripping surface
        pad = b.add_link(
            xform=wp.transform((x0, 0, z0), wp.quat_identity()),
            label=f"{label}_pad")

        # Orient the BOX SHAPE so its local +x face points toward the sphere.
        # CSLC uses the +x face as its 2-D contact lattice (see handler), so
        # both pads must present that face inward.
        #   Left pad  (at world -x): body's local +x already points toward
        #     the sphere at the origin — no rotation.
        #   Right pad (at world +x): rotate the SHAPE 180° around z so that
        #     its local +x maps to world -x (= inward).
        # We rotate the shape rather than the body so the prismatic joints
        # (which reference the body frame) are not affected.
        if label == "right":
            box_xform = wp.transform(
                wp.vec3(0.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
            )
        else:
            box_xform = wp.transform_identity()
        b.add_shape_box(pad, xform=box_xform,
                        hx=p.pad_hx, hy=p.pad_hy, hz=p.pad_hz, cfg=pad_cfg)

        # Prismatic X: world → slider
        j_x = b.add_joint_prismatic(
            parent=-1, child=slider,
            axis=wp.vec3(1, 0, 0),
            parent_xform=wp.transform((x0, 0, z0), wp.quat_identity()),
            child_xform=wp.transform((0, 0, 0), wp.quat_identity()),
            label=f"{label}_x")

        # Prismatic Z: slider → pad
        j_z = b.add_joint_prismatic(
            parent=slider, child=pad,
            axis=wp.vec3(0, 0, 1),
            parent_xform=wp.transform((0, 0, 0), wp.quat_identity()),
            child_xform=wp.transform((0, 0, 0), wp.quat_identity()),
            label=f"{label}_z")

        b.add_articulation([j_x, j_z], label=f"{label}_arm")
        pad_joints.extend([j_x, j_z])

        dof_map[f"{label}_x"] = b.joint_qd_start[j_x]
        dof_map[f"{label}_z"] = b.joint_qd_start[j_z]

        # Filter: ghost slider vs ground (avoid spurious contacts)
        if ground_shape is not None:
            b.add_shape_collision_filter_pair(slider_shape, ground_shape)

    # Set joint drives for pad joints (stiff PD position tracking)
    for ji in pad_joints:
        dof = b.joint_qd_start[ji]
        b.joint_target_ke[dof] = p.drive_ke
        b.joint_target_kd[dof] = p.drive_kd
        b.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        b.joint_armature[dof] = 0.01

    # Dynamic sphere with explicit free joint (required for MuJoCo solver)
    sphere = b.add_link(
        xform=wp.transform((0, 0, z0), wp.quat_identity()),
        label="sphere")
    s_cfg = sphere_cfg if sphere_cfg is not None else _sphere_cfg(p)
    b.add_shape_sphere(sphere, radius=p.sphere_radius, cfg=s_cfg)
    j_free = b.add_joint_free(sphere, label="sphere_free")
    b.add_articulation([j_free], label="sphere")

    # Request the per-contact `force` extended attribute so we can read
    # actual solver-applied forces (for diagnostics) in Example.step.
    # Without this the contacts.force array is not allocated.
    b.request_contact_attributes("force")

    # ── start_gripped mode: pre-position the pads at their squeeze-end X ──
    # In normal mode the pads start at dx=0 (outside the sphere) and the
    # APPROACH phase drives them inward.  In start_gripped mode we jump
    # the pads' prismatic-X joint_q directly to the gripped position so
    # there is no approach-phase transient, no ground contact, nothing
    # but the LIFT dynamics under test.
    if p.start_gripped:
        dx_gripped = (
            p.approach_speed * p.approach_duration
            + p.squeeze_speed * p.squeeze_duration
        )
        # joint_q is indexed by DOF.  left_x → +dx_gripped (pad moves +x,
        # i.e. inward from lx0); right_x → -dx_gripped (inward from rx0).
        b.joint_q[dof_map["left_x"]] = +dx_gripped
        b.joint_q[dof_map["right_x"]] = -dx_gripped

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
        if p.warm_start_sphere_vz:
            # Free joint qd is [wx, wy, wz, vx, vy, vz] (Featherstone order).
            # Sphere body is the last link added; its free joint DOFs are the
            # last 6 entries of joint_qd.  We only set vz (index 5 of the 6).
            # Alternative explicit lookup: find the free-joint DOF via joints
            # metadata, but for this fixed scene layout the last-6 works.
            sphere_vz_idx = len(b.joint_qd) - 1  # last qd entry == sphere vz
            b.joint_qd[sphere_vz_idx] = p.lift_speed

    m = b.finalize()
    m.set_gravity(p.gravity)

    _log(f"DOF map: {dof_map}")
    _log(f"bodies={m.body_count}  shapes={m.shape_count}  "
         f"joints={m.joint_count}  DOFs={m.joint_dof_count}")
    if p.start_gripped:
        _log(f"start_gripped=True  dx_initial={dx_gripped*1e3:.2f}mm  "
             f"spawn_z={z0:.3f}m  no_ground={p.no_ground}")

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

SPHERE_BODY = 4
LEFT_PAD = 1
RIGHT_PAD = 3


# ── Pad trajectory ────────────────────────────────────────────────────────

def _pad_state(step, p: SceneParams):
    """Compute (dx_inward, dz_up) at a given step.

    The dz profile during LIFT uses a C¹-smooth velocity ramp over the first
    `lift_ramp_duration` seconds so the pad's commanded velocity eases from 0
    to `lift_speed` instead of stepping discontinuously.  This prevents the
    impulsive friction kick that otherwise launches the sphere upward.

    Velocity profile (s = t / ramp):
        v(t) = lift_speed · smoothstep(s)   for t ≤ ramp
        v(t) = lift_speed                   for t  > ramp
    where smoothstep(s) = 3s² − 2s³ (C¹ at both endpoints).

    Integrating gives position:
        dz(t) = lift_speed · ramp · (s³ − s⁴/2)         for t ≤ ramp
        dz(t) = lift_speed · (t − ramp/2)               for t  > ramp
    The `− ramp/2` is the constant of integration that makes dz continuous
    at the handoff; intuitively, the ramp's average velocity is lift_speed/2,
    so after `ramp` seconds you are `ramp/2` behind the no-ramp trajectory.
    """
    phase, s = p.phase_of(step)
    t = s * p.dt

    if phase == "APPROACH":
        return p.approach_speed * t, 0.0

    dx_app = p.approach_speed * p.approach_duration
    if phase == "SQUEEZE":
        return dx_app + p.squeeze_speed * t, 0.0

    dx_total = dx_app + p.squeeze_speed * p.squeeze_duration

    def _lift_dz(t_lift: float) -> float:
        """C¹-smooth velocity profile with ramps at BOTH endpoints.

        Symmetric end-ramp eliminates the inertial overshoot at LIFT→HOLD:
        if the pad's commanded velocity drops 0 → lift_speed → 0 with smooth
        transitions, the sphere never accelerates abruptly and so does not
        carry momentum past the pad's stop position.

        Velocity: smoothstep up over [0, ramp], constant over
        [ramp, lift_duration-ramp], smoothstep down over the last ramp.
        Position: integrate.  Total integral over [0, lift_duration] equals
        lift_speed * (lift_duration - ramp), i.e. the original no-ramp distance
        minus one full ramp.  We renormalise lift_speed to keep the same total
        travel as a single-end-ramp profile would give.
        """
        ramp = p.lift_ramp_duration
        T = p.lift_duration
        if ramp <= 0.0:
            return p.lift_speed * t_lift
        # Renormalise so total travel matches the single-end-ramp design:
        #   travel_single_end = lift_speed * (T - 0.5*ramp)
        #   travel_two_end    = lift_speed * (T - ramp)        (one less ramp)
        # Multiply by (T - 0.5*ramp) / (T - ramp) so the two-ramp profile
        # delivers the same lift distance as the original.
        v_eff = p.lift_speed * (T - 0.5 * ramp) / max(T - ramp, 1e-9)
        # Phase 1: smoothstep up over [0, ramp].
        if t_lift < ramp:
            sn = t_lift / ramp
            return v_eff * ramp * (sn ** 3 - 0.5 * sn ** 4)
        # Phase 2: constant velocity over [ramp, T-ramp].
        if t_lift < T - ramp:
            return v_eff * ramp * (1.0 - 0.5) + v_eff * (t_lift - ramp)
        # Phase 3: smoothstep down over [T-ramp, T].
        s_back = (T - t_lift) / ramp  # goes 1 → 0
        s_back = max(s_back, 0.0)
        # Travelled at constant velocity through end of phase 2:
        z_phase2_end = v_eff * (ramp * 0.5) + v_eff * (T - 2 * ramp)
        # Phase 3 contribution: integrate v_eff * smoothstep(1 - s_back²·...).
        # Using mirror of phase 1's integral:
        z_phase3 = v_eff * ramp * (0.5 - (s_back ** 3 - 0.5 * s_back ** 4))
        return z_phase2_end + z_phase3

    if phase == "LIFT":
        return dx_total, _lift_dz(t)

    # HOLD: freeze at the position reached at the end of LIFT.
    return dx_total, _lift_dz(p.lift_duration)


def set_pad_targets(control, step, p: SceneParams, dof_map, debug=False):
    """Write joint position targets for both pads."""
    dx, dz = _pad_state(step, p)

    target = control.joint_target_pos.numpy()
    target[dof_map["left_x"]] = +dx
    target[dof_map["left_z"]] = +dz
    target[dof_map["right_x"]] = -dx
    target[dof_map["right_z"]] = +dz
    control.joint_target_pos.assign(
        wp.array(target, dtype=wp.float32, device=control.joint_target_pos.device))

    if debug:
        phase, _ = p.phase_of(step)
        gap = p.approach_gap - 2 * dx
        # face_pen > 0 means pad face is INSIDE the sphere surface (squeezing).
        # face_pen < 0 means pad hasn't reached the sphere yet (approaching).
        face_pen = p.sphere_radius * 2 - gap       # sphere_diameter − gap
        _log(f"[{phase:8s}] step={step:5d}  dx={dx*1e3:+6.2f}mm  dz={dz*1e3:+5.2f}mm  "
             f"gap={gap*1e3:+6.1f}mm  face_pen={face_pen*1e3:+5.2f}mm")


# ── Helpers ───────────────────────────────────────────────────────────────

# `count_active_contacts`, `recalibrate_cslc_kc_per_pad`,
# `read_cslc_state`, `_quat_rotate`, `get_cslc_lattice_viz_data`,
# `inspect_model`, and `make_solver` all live in cslc_v1.common.


# ── Headless runner ───────────────────────────────────────────────────────

def run_headless(name, model, solver, p, dof_map, verbose=True):
    import time as _time
    met = Metrics(name=name)
    s0, s1 = model.state(), model.state()
    ctrl, con = model.control(), model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    is_cslc = "cslc" in name.lower()

    # Warm-up pass: compile kernels + prime the CG solver so the timed
    # loop measures steady-state compute cost, not one-off JIT overhead.
    set_pad_targets(ctrl, 0, p, dof_map)
    s0.clear_forces()
    model.collide(s0, con)
    solver.step(s0, s1, ctrl, con, p.dt)
    wp.synchronize()

    _time_start = _time.perf_counter()
    _time_collide = 0.0
    _time_step = 0.0

    for step in range(p.total_steps):
        set_pad_targets(ctrl, step, p, dof_map)
        s0.clear_forces()
        _t0 = _time.perf_counter()
        model.collide(s0, con)
        wp.synchronize()
        _t1 = _time.perf_counter()
        solver.step(s0, s1, ctrl, con, p.dt)
        wp.synchronize()
        _t2 = _time.perf_counter()
        _time_collide += _t1 - _t0
        _time_step += _t2 - _t1

        q = s1.body_q.numpy()
        sz = float(q[SPHERE_BODY, 2])
        nc = count_active_contacts(con)
        met.sphere_z.append(sz)
        met.contacts.append(nc)
        s0, s1 = s1, s0

        if verbose and ((step + 1) % 200 == 0 or step == p.total_steps - 1):
            phase, _ = p.phase_of(step)
            pad_z = float(q[LEFT_PAD, 2])
            line = (f"  {name:16s} {step+1:5d}/{p.total_steps}  [{phase:8s}]  "
                    f"sphere_z={sz:.5f}  pad_z={pad_z:.4f}  contacts={nc}")
            if is_cslc:
                ci = read_cslc_state(model)
                if ci:
                    line += f"  cslc={ci['n_active']}/{ci['n_surface']}"
            print(line)

    _wall = _time.perf_counter() - _time_start
    per_step_ms = 1000.0 * _wall / p.total_steps
    collide_ms = 1000.0 * _time_collide / p.total_steps
    step_ms = 1000.0 * _time_step / p.total_steps
    rtx = p.total_steps * p.dt / _wall
    print(
        f"  TIMING {name:16s}  wall={_wall:.3f}s  "
        f"per-step={per_step_ms:.3f}ms  "
        f"(collide={collide_ms:.3f}ms + step={step_ms:.3f}ms)  "
        f"realtime×={rtx:.2f}"
    )

    return met


def test_headless(p, solver_name, contact_models=None):
    if contact_models is None:
        contact_models = ["point", "cslc"]
    p.dump()
    results = []
    for cm in contact_models:
        label = f"{cm}_{solver_name}"
        _section(label.upper())
        model, dof_map = BUILDERS[cm](p)
        _ = model.contacts()
        inspect_model(model, label)
        if cm == "cslc" and p.cslc_contact_fraction is not None:
            recalibrate_cslc_kc_per_pad(model, p.cslc_contact_fraction)
        solver = make_solver(model, solver_name)
        met = run_headless(label, model, solver, p, dof_map)
        results.append(met)
        _log(f"RESULT: max_z={met.max_z:.4f}  final_z={met.final_z:.4f}  "
             f"lifted={'YES' if met.lifted else 'NO'}  held={'YES' if met.held else 'NO'}")

    _section("SUMMARY")
    print(f"  {'Config':<24} {'Max Z':>8} {'Final Z':>8} {'Lifted':>8} {'Held':>8}")
    print(f"  {_SEP}")
    for m in results:
        lifted_s = 'YES' if m.lifted else 'NO'
        held_s   = 'YES' if m.held   else 'NO'
        print(f"  {m.name:<24} {m.max_z:8.4f} {m.final_z:8.4f} {lifted_s:>8} {held_s:>8}")


# ── Viewer mode ───────────────────────────────────────────────────────────

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
        self.show_lattice = getattr(args, "show_lattice", True)

        self.p = SceneParams(
            dt=self.sim_dt,
            no_ground=getattr(args, "no_ground", False),
            start_gripped=getattr(args, "start_gripped", False),
            warm_start_sphere_vz=getattr(args, "warm_start_sphere_vz", False),
        )
        self.p.dump()

        self.model, self.dof_map = BUILDERS[self.contact_model](self.p)
        inspect_model(self.model, self.contact_model)

        self.solver = make_solver(self.model, self.solver_name)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        self.current_z = self.p.sphere_start_z
        self.current_contacts = 0
        self.current_phase = "APPROACH"

        # FD telemetry state (body_qd stale under MuJoCo GPU — use position FD).
        self._prev_sphere_z = None
        self._prev_pad_z = None
        self._prev_vz_sphere = None
        self._prev_vz_pad = None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.3, -0.3, self.p.sphere_start_z + 0.15),
            pitch=-15.0, yaw=135.0)

        self._init_lattice_rendering()

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

        q = self.state_0.body_q.numpy()
        _log(f"INITIAL STATE ({self.model.body_count} bodies):")
        for bi in range(self.model.body_count):
            _log(
                f"  body {bi}: pos=({q[bi, 0]:.4f},{q[bi, 1]:.4f},{q[bi, 2]:.4f})", 1)
        _log(
            f"sim_substeps={self.sim_substeps}  sim_dt={self.sim_dt*1e3:.3f}ms")

    def simulate(self):
        for _ in range(self.sim_substeps):
            set_pad_targets(self.control, self.sim_step, self.p, self.dof_map,
                            debug=(self.sim_step % 500 == 0))
            self.state_0.clear_forces()
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
        if self.show_lattice and self.contact_model == "cslc":
            self._render_lattice()
        self.viewer.end_frame()

    def _init_lattice_rendering(self):
        # Pre-allocate the per-surface-sphere viewer buffers once.  Sized
        # to the total surface-sphere count across all pads, so both the
        # left and right pad's lattice render in the same `log_shapes`
        # call each frame.
        self._lattice_n = 0
        pipeline = getattr(self.model, "_collision_pipeline", None)
        handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
        if handler is None:
            return
        d = handler.cslc_data
        n = d.n_surface
        self._lattice_n = n
        # Render lattice spheres at 40% of their lattice radius so the
        # underlying pad geometry stays visible.
        self._lattice_radius = float(d.radii.numpy()[0]) * 0.4
        self._lattice_xforms = np.zeros((n, 7), np.float32)
        self._lattice_xforms[:, 6] = 1.0  # identity quat
        self._lattice_colors = np.zeros((n, 3), np.float32)
        self._lattice_mats = np.tile(
            [0.5, 0.3, 0.0, 0.0], (n, 1)).astype(np.float32)

    def _render_lattice(self):
        viz = get_cslc_lattice_viz_data(self.model, self.state_0)
        if viz is None:
            return
        pw, dl, rd, sf = viz
        idx = 0
        # Normalise compression depth to the current step's max so colour
        # tracks relative compression even when penetration grows or shrinks.
        mx = max(float(np.max(np.abs(dl))), 1e-6)
        for i in range(len(dl)):
            if sf[i] == 0 or idx >= self._lattice_n:
                continue
            self._lattice_xforms[idx, :3] = pw[i]
            t = min(abs(dl[i]) / mx, 1.0)
            # Warm red for compressed spheres, cool gray for uncompressed.
            self._lattice_colors[idx] = [
                t, 0.2 * (1 - t), 1 - t] if dl[i] > 1e-8 else [0.3, 0.3, 0.35]
            idx += 1
        if idx == 0:
            return
        self.viewer.log_shapes(
            "/cslc_lattice", newton.GeoType.SPHERE, self._lattice_radius,
            wp.array(self._lattice_xforms[:idx], dtype=wp.transform),
            wp.array(self._lattice_colors[:idx], dtype=wp.vec3),
            wp.array(self._lattice_mats[:idx], dtype=wp.vec4))

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
        parser.add_argument("--contact-models", type=str, default=None,
                            help="Comma-separated list for headless mode "
                                 "(e.g. 'point,cslc,hydro').")
        parser.add_argument("--solver", type=str, default="mujoco",
                            choices=["mujoco", "semi"])
        parser.add_argument("--mode", type=str, default="viewer",
                            choices=["viewer", "headless"])
        parser.add_argument("--no-ground", action="store_true",
                            help="Skip ground plane.")
        parser.add_argument("--start-gripped", action="store_true",
                            help="Skip APPROACH+SQUEEZE; start at squeeze-end position.")
        parser.add_argument("--warm-start-sphere-vz", action="store_true",
                            help="Init sphere vz = lift_speed at t=0.")
        parser.add_argument(
            "--show-lattice", action="store_true", default=True,
            help="Render the CSLC lattice spheres in the viewer (default).")
        parser.add_argument(
            "--no-lattice", dest="show_lattice", action="store_false",
            help="Hide the CSLC lattice viz.")
        parser.add_argument("--cslc-ka", type=float, default=None,
                            help="Override CSLC anchor stiffness ka [N/m].")
        parser.add_argument("--cslc-contact-fraction", type=float, default=None,
                            help="Override CSLC contact fraction for kc recalibration.")
        parser.add_argument("--kh", type=float, default=None,
                            help="Override hydroelastic modulus [Pa].")
        return parser


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = Example.create_parser()
    args, _ = parser.parse_known_args()

    wp.init()
    print(f"\n{'━' * 60}\n  GRIPPER LIFT TEST — mode: {args.mode}\n{'━' * 60}")

    if args.mode == "viewer":
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
    else:
        args = parser.parse_args()
        sn = args.solver if HAS_MUJOCO or args.solver != "mujoco" else "semi"
        cm_arg = getattr(args, "contact_models", None)
        cm_list = [c.strip() for c in cm_arg.split(",") if c.strip()] if cm_arg else None
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
