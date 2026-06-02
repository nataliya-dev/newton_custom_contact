# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Lift Test
#
# Two articulated gripper pads close on a free shape-matched (SM-rigid)
# object, squeeze, then lift it together against gravity. Exercises the
# friction-driven grasp path under SolverUXPBD:
#
#   - Each pad is a rigid body driven by two prismatic joints
#     (X = approach axis, Z = lift axis), with a thin collision box for
#     inertia and a kinematic lattice that does the actual contact work.
#   - The object is a free SM-rigid sphere-packed body added via
#     add_particle_volume; its rigidity is maintained by the SRXPBD
#     shape-matching pass each iteration.
#   - Contact between pad lattice particles and object particles is
#     handled by solve_particle_particle_contacts_uxpbd, which applies
#     position-level Coulomb friction. Slip resistance during the LIFT
#     phase is the test's load-bearing physical behaviour.
#
# Phases (Z up):
#
#       ┌─┐         ┌─┐
#       │L│ ◄────── │R│      APPROACH    pads move inward
#       └─┘         └─┘
#       ┌─┐ ┌───┐ ┌─┐
#       │L│ │obj│ │R│        SQUEEZE     pads press a few mm in
#       └─┘ └───┘ └─┘
#         ↑       ↑
#       ┌─┐ ┌───┐ ┌─┐        LIFT        pads (and the gripped object)
#       │L│ │obj│ │R│                    rise together
#       └─┘ └───┘ └─┘
#       ════════════         HOLD        pads stationary in the air
#
# Adapted from cslc_mujoco/lift_test.py without the MuJoCo / CSLC /
# hydroelastic comparisons. UXPBD-only, single pipeline.
#
# Command: python -m newton.examples uxpbd_lift_test
###########################################################################

from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.solvers import CSLCParams

# Curved-pad asset bundle (a convex "scoop" mesh + a MorphIt sphere packing
# baked from that mesh) and a unit-sphere asset (mesh + MorphIt packing) for
# the grasped object. Both live at the repo root, not in
# newton/examples/assets, because they're specific to this contact study.
# pad_5x.obj + pad.json are at native 5x scale (~10 cm wide, 5 cm curve
# depth). sphere.json packs a unit sphere; we rescale at load time to
# match SceneParams.obj_radius.
_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
_PAD_ASSET_DIR = _ASSETS_DIR / "pad"
_SPHERE_ASSET_DIR = _ASSETS_DIR / "sphere"


# ----- Warp kernel: add a uniform force to an indexed particle subset --
# Used by the DISTURB phase to deliver a transient transverse force
# pulse to the bottom band of book particles. Atomic-add is overkill
# for a serial-launch-per-particle pattern but keeps the kernel safe
# if the index list ever contains duplicates.
@wp.kernel
def _disturb_add_force_kernel(
    particle_f: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    force_per_particle: wp.vec3,
):
    tid = wp.tid()
    idx = indices[tid]
    wp.atomic_add(particle_f, idx, force_per_particle)


@dataclass
class SceneParams:
    """All knobs for the gripper lift scene."""

    # --- Object (SM-rigid ball: MorphIt sphere packing from sphere.json) ---
    # A spherical sphere-packed SM-rigid body. Loads the 125-sphere
    # MorphIt packing of a unit sphere from assets/sphere/sphere.json and
    # rescales centers + radii at load time so the outer envelope has
    # radius `obj_radius`.
    #
    # Size is set to 4 cm radius (8 cm diameter) -- the same scale as
    # example_uxpbd_particle_drop, which is the proven-stable regime for
    # SM-rigid + ground contact. A smaller ball (1.2 cm) was tried
    # first to fit a tighter pad gap, but SVD-based rotation extraction
    # in the shape-matching kernel is numerically unstable at cm-scale:
    # covariance entries scale as r^2 while SVD noise is roughly
    # absolute, so a symmetric rest pose produces a spurious R != I that
    # drives m/s-scale internal velocity dispersion during pure free-
    # fall. At r=4 cm the SM signal dominates the noise and free-fall
    # stays perfectly rigid.
    obj_radius: float = 0.04  # outer envelope radius [m]
    # total mass distributed over 125 spheres [kg]
    obj_mass: float = 1.0

    # --- Pads (curved scoop mesh + baked sphere packing) ---
    # Geometry comes from assets/pad/pad_5x.obj and assets/pad/pad.json.
    # In the pad's body frame the mesh is a half-disk: flat back at
    # z = 0, convex peak at z = +pad_curve_depth, extending +/- 5 cm
    # in the orthogonal X and Y axes. The MorphIt sphere centers in
    # pad.json live in the same body frame.
    #
    # Each pad is pre-rotated (mesh vertices and lattice centers, in
    # NumPy at build time) so its convex face points toward the grasped
    # object: body +Z -> world +X for the left pad, body +Z -> world -X
    # for the right pad. The articulated body itself stays at identity
    # orientation through the prismatic joint chain, which keeps the
    # joint axes (world X for approach, world Z for lift) aligned with
    # the world axes.
    pad_curve_depth: float = 0.05  # max body-z of pad_5x.obj == distance
    # from body origin to the curved peak
    # Pad body spawn height. The pre-rotated lattice spans z ∈ [-0.051,
    # +0.052] relative to the body origin, so the body must sit at
    # z >= ~0.06 for the lowest sphere surface to clear the ground.
    # At z=0.06 the bottom sphere is ~9 mm above z=0 (no dragging) and
    # the lattice still brackets the ball, which settles with its center
    # at z=obj_radius=0.04 and surface at z ∈ [0, 0.08]; contact happens
    # 2 cm above the ball's equator, slightly above the maximum-moment-
    # arm position but well inside the lattice's vertical band.
    pad_z0: float = 0.06

    # --- Phase timing ---
    # SETTLE:   pads stationary, ball free-falls and settles on the ground.
    # APPROACH: pads move inward from `approach_gap` to the object surface.
    # SQUEEZE:  pads press an additional `squeeze_depth` past the surface.
    # LIFT:     pads rise together with a smooth velocity ramp.
    # HOLD:     pads stationary at the lifted height.
    # The SETTLE phase exists because a direct spawn-tangent-to-ground
    # SM-rigid initialisation is unstable: with ~3 g per particle, even
    # a sub-mm shape-matching correction maps to ~1 m/s velocity, and
    # only 4 particles in the sphere-masked bottom layer means contact
    # bias gets amplified into lateral runaway. Free-falling from a few
    # cm lets the SM-rigid cluster settle into ground contact with the
    # solver iterating over the impact instead of starting fused.
    settle_duration: float = 0.5
    # Gap between the two pads' inner faces at t=0. Needs to clear the
    # ball diameter (2 * obj_radius = 8 cm) with margin so the curved
    # pad faces start well clear of the ball.
    approach_gap: float = 0.20
    approach_duration: float = 1.0
    squeeze_depth: float = 0.015
    squeeze_duration: float = 0.5
    lift_speed: float = 0.015
    lift_duration: float = 3.5
    # Smooth the LIFT velocity transition over this duration. Without
    # ramping, the target velocity jumps 0 -> lift_speed in a single
    # timestep; the high-gain PD drive turns that into a near-impulsive
    # pad velocity which saturates friction against the static object
    # and shoots it upward.
    lift_ramp_duration: float = 0.25
    hold_duration: float = 0.5

    # --- "Book" object (tall rectangular sphere-packing) ---
    # Selected via ``--object book``. Sphere-packed at ``obj_book_sphere_r``
    # pitch into a 2*(hx, hy, hz) box. Default is roughly a paperback
    # (2 x 10 x 20 cm) so the pad-to-book ratio with the default
    # fingertip-sized box pad (2 x 3 x 3 cm) is realistic -- the pad
    # covers ~15% of the book's lateral face area, like a fingertip
    # pinching a book. The pads come in along +/- X, so the long axis
    # (Z, "up") is free to tilt about X under a +Y force pulse applied
    # at the book's lower particles -- the same kinematic setup as the
    # "Stable Patch Contact" figure.
    obj_book_hx: float = 0.010  # X half-extent [m] (pad approach axis)
    obj_book_hy: float = 0.050  # Y half-extent [m] (tilt response axis)
    obj_book_hz: float = 0.100  # Z half-extent [m] (long / vertical axis)
    # 10 mm sphere radius gives a single sphere across the book's
    # narrow X axis (surface tangent to both X faces -- no gap) and
    # ~72 spheres total at the default 1 x 5 x 10 cm half-extents.
    obj_book_sphere_r: float = 0.010

    # --- DISTURB phase: transient force pulse + ring-down measurement ---
    # Phase order: SETTLE -> APPROACH -> SQUEEZE -> LIFT -> HOLD -> DISTURB.
    # HOLD is kept as a short stabilisation buffer so any LIFT transients
    # have decayed before the disturbance fires. During DISTURB the pad
    # joint targets stay frozen at the end-of-LIFT pose; the disturbance
    # is delivered as a transverse force on the bottom-most book
    # particles for ``disturb_force_duration`` seconds, after which the
    # remaining ``disturb_duration - disturb_force_duration`` seconds is
    # the ring-down window plotted as tilt(t) and pad-torque(t).
    disturb_duration: float = 1.5
    # Peak transverse force [N], distributed over the selected bottom
    # particles. 5 N at 0.5 kg is roughly a "kick" that produces a few
    # degrees of tilt for the CSLC default stiffnesses -- adjust along
    # with obj_mass to stay in the "stays held, measurable tilt" zone.
    disturb_force_amplitude: float = 5.0
    disturb_force_duration: float = 0.05  # pulse width [s]
    # Direction vector for the disturbance. Default +X so the force is
    # applied perpendicular to the book's broad flat face (the book
    # cover, hy x hz = 10 x 20 cm). The book pitches about the Y axis;
    # see ``tilt_about_y_deg`` in the telemetry. A +Y direction would
    # roll the book about X instead -- captured by ``tilt_about_x_deg``.
    disturb_force_dir_x: float = 1.0
    disturb_force_dir_y: float = 0.0
    disturb_force_dir_z: float = 0.0
    # Fraction of the book (lowest z) that receives the force. Picking
    # the bottom band maximises the moment arm about the grip line.
    disturb_force_bottom_fraction: float = 0.3

    # --- Friction ---
    # mu_eff in the kernels = 0.5 * (particle_mu + shape_material_mu).
    # Setting both ends to the same value yields exactly this mu_eff.
    # The curved pad presents only a thin band of lattice spheres to
    # the small ball -- few contact pairs, so each needs a high friction
    # coefficient to carry its share of the load.
    mu: float = 1.0

    # --- Compliant lattice (CSLC) ---
    # When True the solver is constructed with a CSLCParams instance and
    # the per-pad lattices carry the per-sphere stiffness arrays defined
    # below. The pipeline: compute_compliant_contact_response writes
    # lattice_delta from the per-pair overlap on the particle grid,
    # update_lattice_world_positions physically displaces each lattice
    # particle inward by delta_n along its rest outward normal
    # (contract_v2.md sign convention q_i = p_i - delta_i), and the
    # accumulate_cslc_body_wrench kernel routes the per-sphere anchor
    # reaction F = -k_a * delta_n * n into body_delta (replacing the
    # lattice-side PBD body wrench that the pp contact kernel writes in
    # the rigid Phase 1 path).
    #
    # Per-sphere stiffnesses follow the series-spring identity
    # (contract §10, eq:calibration): per-sphere effective normal
    # stiffness k_eff = k_a * k_c / (k_a + k_c), compression ratio
    # k_c/(k_a+k_c). With k_a << k_c, the anchor takes the squeeze and
    # the skin compresses visibly; k_a >> k_c gives nearly rigid
    # behaviour (delta -> 0).
    #
    # Body equilibrium with CSLC owning the wrench:
    #   ke_drive * pos_error  ~  N_active * k_a * delta_n
    # so given a target visible delta_n, pick k_a = ke_drive * pos_error
    # / (N_active * delta_n). For the box pad at default geometry
    # (N_active ~ 4-5, ke_drive = 5e4, pos_error ~ squeeze_depth) this
    # lands k_a in the 1e4 range for a 0.5-1 mm equilibrium delta. Drop
    # k_a too low and the CSLC can't balance the drive (pad slides
    # through, delta blows up past sphere radius); raise it too high
    # and the deformation becomes invisible (microns).
    use_compliant_lattice: bool = True
    # CSLCParams.use_jacobi defaults to True (flipped 2026-05-28). The
    # Jacobi kernel consumes ``pad_k_bulk`` as the PER-VOLUME contact
    # stiffness ``k_c`` [Pa·m^(−1/2)], not as a per-sphere N/m linear
    # spring. Production CSLC (theory.md §10) uses ``k_a = 3.5×10⁴ N/m``
    # and ``k_c ≈ 10¹⁰ Pa·m^(−1/2)``; we use slightly softer values here
    # so the lattice compresses visibly (δ ~ 0.3-1 mm) at the grasp
    # depths typical of this scene.
    #
    # Force-chain sanity check at raw=2 mm, A_j ≈ 4·(10 mm)² = 4e-4 m²:
    #     F_pair = k_c · A_j · raw·√raw = 1e8 · 4e-4 · 2e-3 · √2e-3
    #            ≈ 3.6 N per pair  (saturated regime)
    # Anchor reaction per sphere at δ=0.5 mm:
    #     F_body = k_a · δ = 3e4 · 5e-4 = 15 N per sphere
    # Aggregate over ~5 active spheres ⇒ ~75 N capacity, plenty to
    # carry the book's ~3 N weight via Coulomb friction at μ=1.0.
    #
    # The earlier values (k_anchor=1e2, k_bulk=1e2) were inherited from
    # the v1 closed-form solver, which used k_c as a dimensionless ratio
    # in δ = k_c/(k_a+k_c)·overlap. Under the Jacobi kernel those values
    # produce δ ≈ 0.15 μm and body wrenches in the 10⁻⁵ N range — small
    # enough that the pad has no contact reaction in Z and the book
    # cannot be held against gravity.
    # CSLCParams.use_jacobi defaults to True (flipped 2026-05-28). The
    # Jacobi kernel consumes ``pad_k_bulk`` as the PER-VOLUME contact
    # stiffness ``k_c`` [Pa·m^(−1/2)] in ``F_pair = k_c · A_j · raw^1.5``,
    # NOT as a per-sphere N/m linear spring. Production CSLC
    # (theory.md §10) uses ``k_a = 3.5×10⁴ N/m`` and ``k_c ≈ 10¹⁰
    # Pa·m^(−1/2)``; we sit slightly softer so the lattice compresses
    # visibly (δ ~ 0.05-0.5 mm) at the operating depths of this scene.
    # The pre-fix values (k_anchor=1e2, k_bulk=1e2) were inherited from
    # the v1 closed-form solver where k_c was a dimensionless ratio in
    # δ = k_c/(k_a+k_c)·overlap; under the Jacobi kernel those values
    # give per-pair forces of ~10⁻⁵ N and the pad cannot grip.
    pad_k_anchor: float = 1.0e5  # N/m per sphere, anchor spring
    pad_k_bulk: float = 1.0e8  # Pa·m^(−1/2), per-volume kernel-side k_c
    pad_k_lateral: float = 5.0e3  # N/m, reserved for v2 jacobi
    pad_damping: float = 2.0  # s/m, reserved for v2 Hunt-Crossley force
    # Numerical guard against the first-substep delta jump when the
    # object first touches a pad and lattice_delta_prev = 0 leaves the
    # finite-difference rate huge. SETTLE phase usually keeps the pads
    # away from the object so this rarely fires; leave at 1.0 m/s for
    # safety. Pass None into CSLCParams to disable entirely.
    pad_clamp_delta_dot: float | None = 1.0
    # Anisotropic anchor tangent-to-normal stiffness ratio (CSLC theory
    # §6.1 / theory.md table).  1.0 = isotropic anchor; 1/3 =
    # "incompressible skin" limit.  Lower ratios make the lattice
    # tangentially compliant -- under a transverse disturbance pulse
    # the book can slip sideways within the lattice (its δ_t grows)
    # without yanking the pad body, isolating the pad from sudden
    # lateral impulses.  Crucially, this still leaves the NORMAL
    # anchor at full ``k_a`` so the lift grip (which is friction =
    # μ·F_normal) is unchanged.  Required to keep the disturbance
    # response bounded; otherwise the 30 N pulse on the book yanks
    # the pads off-target.
    pad_ka_tangent_ratio: float = 1.0
    # Implicit-Euler lattice velocity damping rate [N·s/m].  When > 0
    # the Jacobi kernel adds ``c_lattice / dt`` to each per-axis
    # diagonal AND ``c_lattice/dt · δ_prev`` to the RHS (theory §11).
    # Damps the lattice's natural oscillatory modes which otherwise
    # ring after the end-of-LIFT ramp and the disturbance pulse.
    pad_c_lattice: float = 0.0
    # When True (the default for this scene), the pp contact kernel
    # ALSO writes the lattice-side body wrench in addition to the CSLC
    # anchor reaction.  Recovers the tangential friction drag on the
    # pad body that the anchor reaction alone cannot capture -- the
    # anchor encodes compression along the rest normal, not friction
    # along the contact tangent.  Without this, the lattice friction
    # on the book is one-sided (book pulled by friction from many
    # spheres, pad feels no reaction torque), and the pad cannot
    # resist book rotation under disturbance.  Slightly over-counts
    # the normal direction (anchor + pp both push the body) but the
    # trade-off favours grasp scenes where distributed-friction
    # torque resistance is the load-bearing behaviour.
    pad_enable_lattice_pp_body_wrench: bool = True

    # --- Joint drive ---
    drive_ke: float = 5.0e4  # position stiffness
    drive_kd: float = 1.0e3  # velocity damping

    # --- Integration ---
    # Matches example_uxpbd_particle_drop's working SM-rigid + ground
    # configuration (8 iterations, shock_propagation_k=1.0). Fewer
    # iterations leak SM corrections into velocity faster than friction
    # can clamp them; lower shock propagation means contact impulses
    # don't propagate to the rest of the cluster in one substep.
    fps: int = 100
    sim_substeps: int = 16
    solver_iterations: int = 8
    shock_propagation_k: float = 1.0

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    @property
    def sim_dt(self) -> float:
        return self.frame_dt / self.sim_substeps

    @property
    def approach_speed(self) -> float:
        # Travel = (approach_gap/2) - obj_radius in `approach_duration`.
        travel = (self.approach_gap / 2.0) - self.obj_radius
        return travel / self.approach_duration

    @property
    def squeeze_speed(self) -> float:
        return self.squeeze_depth / self.squeeze_duration

    @property
    def total_frames(self) -> int:
        return int(
            (
                self.settle_duration
                + self.approach_duration
                + self.squeeze_duration
                + self.disturb_duration
                + self.lift_duration
                + self.hold_duration
            )
            * self.fps
        )


def _pad_target_xz(step: int, p: SceneParams) -> tuple[float, float]:
    """Compute (dx_inward, dz_up) at a given (sub)step index.

    dx_inward and dz_up are signed offsets that get mirrored onto each
    pad (left pad uses +dx, right pad uses -dx; both use +dz).
    """
    t = step * p.sim_dt

    # SETTLE: pads frozen at spawn while the ball free-falls onto the
    # ground. Shifts every later phase by `settle_duration`.
    if t < p.settle_duration:
        return 0.0, 0.0
    t -= p.settle_duration

    if t < p.approach_duration:
        return p.approach_speed * t, 0.0
    t -= p.approach_duration

    dx_app = p.approach_speed * p.approach_duration
    if t < p.squeeze_duration:
        return dx_app + p.squeeze_speed * t, 0.0
    t -= p.squeeze_duration

    dx_total = dx_app + p.squeeze_speed * p.squeeze_duration

    def _lift_dz(t_lift: float) -> float:
        # Velocity profile: smooth ramp UP (3s^2 - 2s^3) -> cruise at
        # ``lift_speed`` -> smooth ramp DOWN. Both ramps are
        # ``lift_ramp_duration`` long; the cruise phase fills the
        # middle. Without an end ramp the pad velocity jumps from
        # ``lift_speed`` to 0 at the LIFT->HOLD boundary, which
        # impulse-loads the compliant lattice and shows up as a
        # noticeable overshoot/dip in book z(t) right at the LIFT end.
        # An end ramp removes that visual artefact.
        ramp = p.lift_ramp_duration
        if ramp <= 0.0:
            return p.lift_speed * max(t_lift, 0.0)

        if t_lift <= 0.0:
            return 0.0

        cruise_end = p.lift_duration - ramp
        pos_start_ramp = p.lift_speed * ramp * 0.5  # integral of start ramp

        if t_lift < ramp:
            # Start ramp: velocity goes 0 -> lift_speed via smoothstep.
            # Position integral: lift_speed * ramp * (s^3 - 0.5 s^4).
            s = t_lift / ramp
            return p.lift_speed * ramp * (s**3 - 0.5 * s**4)

        if t_lift < cruise_end:
            # Cruise at constant ``lift_speed``.
            return pos_start_ramp + p.lift_speed * (t_lift - ramp)

        if t_lift < p.lift_duration:
            # End ramp: velocity goes lift_speed -> 0 via smoothstep.
            # Position integral over end ramp: lift_speed * ramp *
            # (s - s^3 + 0.5 s^4) starting from pos at cruise_end.
            pos_cruise_end = (pos_start_ramp
                              + p.lift_speed * (cruise_end - ramp))
            s = (t_lift - cruise_end) / ramp
            return (pos_cruise_end
                    + p.lift_speed * ramp * (s - s**3 + 0.5 * s**4))

        # Past lift_duration: hold at end-of-lift position.
        # Total lift distance = lift_speed * (lift_duration - ramp).
        return p.lift_speed * (p.lift_duration - ramp)

    if t < p.lift_duration:
        return dx_total, _lift_dz(t)
    # HOLD and DISTURB: pad joint targets stay frozen at end-of-LIFT
    # pose. The disturbance is delivered to the *object* (a force pulse
    # on its lower particles), not by moving the pads, so the pad-side
    # trajectory has nothing more to do after LIFT ends.
    return dx_total, _lift_dz(p.lift_duration)


def _phase_at(t: float, p: SceneParams) -> str:
    """Phase label at simulation time ``t`` [s].

    Mirrors the if-ladder in :func:`_pad_target_xz` so a single source of
    truth determines what each frame is doing; downstream telemetry tags
    rows with the phase, which makes "did it slip during HOLD?" or
    "what was the peak tilt during DISTURB?" simple CSV filters rather
    than a time-arithmetic exercise. ``hold`` is kept as a short
    stabilisation buffer before the disturbance fires.
    """
    if t < p.settle_duration:
        return "settle"
    t -= p.settle_duration
    if t < p.approach_duration:
        return "approach"
    t -= p.approach_duration
    if t < p.squeeze_duration:
        return "squeeze"
    t -= p.squeeze_duration
    if t < p.lift_duration:
        return "lift"
    t -= p.lift_duration
    if t < p.hold_duration:
        return "hold"
    return "disturb"


def _disturb_phase_start(p: SceneParams) -> float:
    """World time [s] at which the DISTURB phase begins.

    Single source of truth so the force-pulse window inside
    ``Example.simulate`` and the telemetry plot annotations agree on
    when t=0 of "ring-down" is.
    """
    return (p.settle_duration + p.approach_duration
            + p.squeeze_duration + p.lift_duration
            + p.hold_duration)


# ----- Pluggable object builders ---------------------------------------
# Each builder takes the open ModelBuilder, SceneParams, and a spawn
# position, and returns the particle-group id. Register additional
# objects with :func:`register_object_builder` from outside this file:
#
#     from newton.examples.contacts.example_uxpbd_lift_test import \
#         register_object_builder
#     def _build_object_my_shape(builder, params, pos):
#         ...
#         return group_id
#     register_object_builder("my_shape", _build_object_my_shape)
#
# Then pass --object my_shape (or set args.object on a Namespace passed
# to Example) to use it. The sweep script in
# example_uxpbd_lift_test_sweep.py drives the same registry, so any new
# object kind is immediately sweep-compatible.


def _build_object_sphere(builder, params: SceneParams, pos: tuple[float, float, float]) -> int:
    """MorphIt-packed sphere object (default). Rescales the unit
    packing in sphere.json so the outer envelope equals
    ``params.obj_radius`` and overrides volume-weighted masses with the
    JSON's physics-optimised values (see the inline comment in the
    monolithic version for the why)."""
    with open(_SPHERE_ASSET_DIR / "sphere.json") as f:
        sphere_data = json.load(f)
    native_centers = np.asarray(sphere_data["centers"], dtype=np.float32)
    native_radii = np.asarray(sphere_data["radii"], dtype=np.float32)
    native_masses = np.asarray(sphere_data["masses"], dtype=np.float32)
    native_envelope = float(
        (np.linalg.norm(native_centers, axis=1) + native_radii).max())
    obj_scale = params.obj_radius / native_envelope
    obj_centers = (native_centers * obj_scale).astype(np.float32)
    obj_radii = (native_radii * obj_scale).astype(np.float32)
    group = builder.add_particle_volume(
        volume_data={"centers": obj_centers.tolist(),
                     "radii": obj_radii.tolist()},
        total_mass=params.obj_mass,
        pos=wp.vec3(*pos),
    )
    mass_scale = params.obj_mass / float(native_masses.sum())
    obj_masses = (native_masses * mass_scale).astype(np.float32)
    for idx, m in zip(builder.particle_groups[group], obj_masses):
        builder.particle_mass[idx] = float(m)
    return group


def _build_object_cube(builder, params: SceneParams, pos: tuple[float, float, float]) -> int:
    """Uniform 4x4x4 (=64) particle cube with half-side ``obj_radius``.

    Used by --object cube for faster iteration; settled centroid lands
    at the same height as the sphere variant so test_final's rest-height
    check is unchanged.
    """
    n = 4
    sphere_r = params.obj_radius / n
    coords = np.linspace(
        -params.obj_radius + sphere_r,
        params.obj_radius - sphere_r,
        n,
    )
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    obj_centers = np.stack(
        [xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
    obj_radii = np.full(
        obj_centers.shape[0], sphere_r, dtype=np.float32)
    return builder.add_particle_volume(
        volume_data={"centers": obj_centers.tolist(),
                     "radii": obj_radii.tolist()},
        total_mass=params.obj_mass,
        pos=wp.vec3(*pos),
    )


def _build_object_book(builder, params: SceneParams, pos: tuple[float, float, float]) -> int:
    """Tall rectangular sphere-packed "book" used by the DISTURB phase.

    Sphere packing strategy:
      * The X axis (book thickness) is the thinnest dimension, so the
        sphere radius is chosen equal to ``obj_book_hx`` -- a single
        sphere spans the full thickness, surface tangent to both X
        faces, no gap.
      * The Y and Z axes use overlapping spheres via
        :func:`_axis_sphere_count` with ``OBJ_BOOK_OVERLAP_FACTOR``.
        Adjacent sphere surfaces compress into each other so the
        emitted contact face has no inter-sphere gaps -- a "smooth
        flat face" approximation in a sphere-packed model.

    At default dimensions (1, 5, 10 cm half-extents = 2 x 10 x 20 cm
    paperback) and r=10 mm the packing is 1 x 6 x 12 = 72 spheres,
    with ~3-4 mm of overlap per neighbour pair in Y / Z.
    """
    hx, hy, hz = params.obj_book_hx, params.obj_book_hy, params.obj_book_hz
    # Choose r so that:
    #   - X axis fits a single sphere whose surface reaches both faces
    #     (r = hx, n_x = 1, no gap),
    #   - Y / Z axes use the user-supplied radius if it allows overlap;
    #     otherwise fall back to ``hx`` so the radii stay consistent.
    r = float(min(hx, max(hx, params.obj_book_sphere_r)))
    nx = _axis_sphere_count(hx, r, OBJ_BOOK_OVERLAP_FACTOR)
    ny = _axis_sphere_count(hy, r, OBJ_BOOK_OVERLAP_FACTOR)
    nz = _axis_sphere_count(hz, r, OBJ_BOOK_OVERLAP_FACTOR)

    cx = _axis_centres(hx, nx, r)
    cy = _axis_centres(hy, ny, r)
    cz = _axis_centres(hz, nz, r)
    xs, ys, zs = np.meshgrid(cx, cy, cz, indexing="ij")
    centers = np.stack(
        [xs.flatten(), ys.flatten(), zs.flatten()], axis=1,
    ).astype(np.float32)
    radii = np.full(centers.shape[0], r, dtype=np.float32)
    return builder.add_particle_volume(
        volume_data={"centers": centers.tolist(),
                     "radii": radii.tolist()},
        total_mass=params.obj_mass,
        pos=wp.vec3(*pos),
    )


OBJECT_BUILDERS: dict = {
    "sphere": _build_object_sphere,
    "cube": _build_object_cube,
    "book": _build_object_book,
}


def _object_spawn_height(kind: str, params: SceneParams) -> float:
    """Centre-of-mass Z at spawn time (before SETTLE gravity drop), so
    each object lands with its lowest particle surface tangent to z=0
    after the SETTLE phase. The +4 cm free-fall margin keeps SRXPBD
    from grinding into the ground at t=0 (see the long inline comment
    in SceneParams about why SETTLE exists).
    """
    free_fall_margin = 0.04  # [m] above resting height at spawn
    if kind == "book":
        return params.obj_book_hz + free_fall_margin
    return params.obj_radius + free_fall_margin


def register_object_builder(name: str, fn) -> None:
    """Register a new object builder under the given name.

    The builder must accept ``(builder, params, pos)`` and return the
    particle-group id from ``builder.add_particle_volume``. After
    registration the new name is accepted by ``--object`` and by the
    sweep runner without any further plumbing.
    """
    OBJECT_BUILDERS[name] = fn


# ----- Pluggable pad shape + lattice builders --------------------------
# Box pad geometry constants (module-level so a custom pad builder can
# reuse them, and so the sweep runner can introspect them).
#
# Defaults sized to a fingertip pad (~2 x 3 x 3 cm) -- realistic
# proportions for gripping a paperback-sized book object (~2 x 10 x 20
# cm). This is roughly a 1:7 pad-to-book ratio along the lateral and
# vertical axes, matching the "few-finger pinch" scenario that the
# DISTURB phase rotational-stiffness test is set up to probe. Earlier
# sphere/cube experiments used (0.025, 0.05, 0.05) with (2, 4, 4) and
# were tuned to a 4 cm sphere; if you re-run those, override
# pad_hx/pad_hy/pad_hz (currently still read from the constants below;
# moving them to SceneParams is the natural next step if multi-scene
# coexistence becomes important).
BOX_HX: float = 0.010
BOX_HY: float = 0.015
BOX_HZ: float = 0.015
BOX_LATTICE_N: tuple[int, int, int] = (4, 6, 6)  # 144 spheres

# Sphere overlap factors -- ratio of sphere diameter to centre-to-centre
# spacing along each axis. >1 means neighbouring spheres overlap, which
# yields a smoother emitted contact surface (no inter-sphere gaps) and
# reduces asymmetric-contact effects during squeeze (the per-pair load
# is averaged across more contacts, so transient one-sided pinches do
# not impart a net twist on the gripped object). The pad uses the same
# factor on all three axes; the book uses it on the Y / Z axes (the X
# axis is single-sphere so the cover face is one big sphere touching
# both X faces).
PAD_SPHERE_OVERLAP_FACTOR: float = 1.5
OBJ_BOOK_OVERLAP_FACTOR: float = 2.0

# Distance [m] the outermost lattice sphere SURFACE protrudes past the
# rigid box face, on each axis. Required for CSLC mode because the
# pp-contact kernel's ``cslc_owns_lattice_wrench=1`` flag SKIPS the
# pad-side body wrench for lattice-vs-object pairs (expecting the CSLC
# anchor reaction to replace it), while the rigid box's shape-vs-
# particle contact does NOT skip its body wrench. If the lattice outer
# surface is coplanar with the box face (the default tiling
# ``centres = linspace(-h+r, h-r)``), the box face engages the object
# every time the lattice does, producing a double-counted contact
# force chain: book gets pushed by both box AND lattice, pad only
# feels the box reaction. The result is the "book slides off
# sideways" failure mode observed at moderate masses.
#
# Protruding the lattice by ``PAD_SPHERE_PROTRUSION`` past the box
# face makes the lattice the SOLE contact surface under normal grasp
# depths: the lattice tangent-touches the object first, the box stays
# recessed by ``PAD_SPHERE_PROTRUSION - δ_resolved`` behind the lattice
# outer surface, and the box only re-engages if the lattice fails
# (e.g. k_a too low to balance the drive) by more than the
# protrusion margin -- making the box a true safety net rather than a
# competing contact path.
#
# Default 5 mm: enough to clear the ``squeeze_depth`` (5 mm for the
# book) so the box never reaches the object at quasistatic grasp, even
# if the lattice compresses fully. Set to 0 to recover the pre-fix
# coplanar tiling.
PAD_SPHERE_PROTRUSION: float = 5.0e-3


def _axis_sphere_count(h: float, r: float, overlap: float) -> int:
    """Number of sphere centres along an axis of half-extent ``h`` such
    that adjacent spheres of radius ``r`` overlap by the requested
    factor (diameter / spacing). Returns 1 when the body half-extent
    is smaller than the sphere radius (single sphere spans the axis).
    """
    if h <= r:
        return 1
    # Spacing constraint: (2h - 2r) / (n - 1) <= 2r / overlap
    # so n >= 1 + overlap * (h - r) / r.
    return int(np.ceil(overlap * (h - r) / r)) + 1


def _axis_centres(h: float, n: int, r: float) -> np.ndarray:
    """Sphere centres along one axis: first/last placed at distance ``r``
    from the face so the outer sphere surfaces sit exactly on ``±h``.
    """
    if n <= 1:
        return np.array([0.0])
    return np.linspace(-h + r, h - r, n)


def _build_pad_box(builder, *, pad_link: int, x0: float, z0: float,
                   params: SceneParams, use_cslc: bool) -> None:
    """Adds the box shape with collision (the rigid "bone" of the
    fingertip) and -- only in CSLC mode -- a uniform overlapping
    lattice of compliant spheres wrapped around it (the "skin").

    The rigid baseline uses *only* the flat box face as the contact
    surface, with no extra kinematic sphere lattice on top. Adding a
    rigid sphere lattice in rigid mode would dump 288 extra
    kinematic contact points into the patch and artificially inflate
    the rotational stiffness -- the comparison would then be
    "compliant lattice" vs "rigid lattice" rather than "compliant
    lattice" vs the standard simulator-default "rigid plate". The
    latter is the apples-to-apples baseline against which CSLC's
    distributed-compliance restoring couple is meant to be measured.
    """
    builder.add_shape_box(pad_link, hx=BOX_HX, hy=BOX_HY, hz=BOX_HZ)
    if not use_cslc:
        # Rigid baseline: just the box face. No compliant skin, no
        # bonus kinematic spheres. The DISTURB response then reveals
        # the box face's "raw" rotational stiffness, which is what
        # off-the-shelf rigid-contact simulators give you.
        return

    # CSLC: overlapping lattice spheres wrapped around the box. Sphere
    # radius is the tangent-cell maximum across axes, scaled up by
    # PAD_SPHERE_OVERLAP_FACTOR so neighbouring spheres compress into
    # each other and the emitted contact face has no inter-sphere
    # gaps. Sphere surfaces protrude past every body face by
    # ``PAD_SPHERE_PROTRUSION`` so the lattice -- not the rigid box --
    # is the contact surface (see the constant's docstring above).
    nx, ny, nz = BOX_LATTICE_N
    tangent_r = max(BOX_HX / nx, BOX_HY / ny, BOX_HZ / nz)
    sphere_r = float(min(tangent_r * PAD_SPHERE_OVERLAP_FACTOR,
                         BOX_HX, BOX_HY, BOX_HZ))
    # Protrude every axis by the same amount. Y/Z protrusions are
    # harmless (those faces don't contact the object); the X
    # protrusion is what de-conflicts box-vs-lattice along the
    # approach axis. Per-axis envelopes:
    #   outer sphere SURFACE at ±(h + PAD_SPHERE_PROTRUSION)
    #   outer sphere CENTRE  at ±(h + PAD_SPHERE_PROTRUSION − sphere_r)
    pe = PAD_SPHERE_PROTRUSION
    cx = _axis_centres(BOX_HX + pe, nx, sphere_r)
    cy = _axis_centres(BOX_HY + pe, ny, sphere_r)
    cz = _axis_centres(BOX_HZ + pe, nz, sphere_r)
    xs, ys, zs = np.meshgrid(cx, cy, cz, indexing="ij")
    pad_centers = np.stack(
        [xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
    pad_radii = np.full(
        pad_centers.shape[0], sphere_r, dtype=np.float32)
    builder.add_lattice(
        link=pad_link,
        morphit_json={"centers": pad_centers, "radii": pad_radii},
        total_mass=0.0,
        pos=wp.vec3(x0, 0.0, z0),
        k_anchor=params.pad_k_anchor,
        k_lateral=params.pad_k_lateral,
        k_bulk=params.pad_k_bulk,
        damping=params.pad_damping,
    )


def _build_pad_curved(builder, *, pad_link: int, x0: float, z0: float,
                      params: SceneParams, use_cslc: bool,
                      assets: dict, rotation: np.ndarray) -> None:
    """Adds the curved scoop pad, pre-rotated by ``rotation`` so the
    convex face points inward.

    Mirrors the box pad's CSLC/rigid split (see :func:`_build_pad_box`)
    so the comparison is "compliant patch" vs the standard rigid
    point-contact baseline:

    * **Rigid baseline**: ONLY the convex mesh is collidable. A convex
      face on the flat book makes an apex line/point contact -- the
      sphere-based point-contact strawman with near-zero rotational
      stiffness. No lattice (a rigid sphere lattice would dump bonus
      contact points and artificially inflate the rotational stiffness).
    * **CSLC**: the compliant lattice is the SOLE contact surface. The
      mesh is kept for inertia (via its density) and visualisation but
      made non-collidable, so the rigid scoop cannot resolve the
      penetration first and shield the lattice. (With the mesh
      collidable the lattice never engages -- observed n_active ~3/250,
      delta ~9 um.) A convex lattice pressed onto the flat book then
      forms a distributed patch by dome flattening -- the mechanism
      CSLC is meant to recover.
    """
    verts_rot = (assets["mesh_verts"] @ rotation.T).astype(np.float32, copy=True)
    pad_mesh = newton.Mesh(verts_rot, assets["mesh_indices"])

    if not use_cslc:
        # Rigid baseline: bare convex mesh = apex point/line contact.
        builder.add_shape_mesh(pad_link, mesh=pad_mesh)
        return

    # CSLC: mesh is inertial-only (non-collidable AND non-visible); the
    # lattice is the contact surface AND what gets rendered. Collision
    # flags / visibility do not gate inertia, so the mesh's density still
    # carries the pad inertia identically to the rigid build (matched
    # inertia across the comparison). ``is_visible=False`` is essential
    # for *seeing* the compliance: the apex lattice spheres displace
    # inward by delta (mm-scale at soft k_anchor) and recede BEHIND the
    # rigid scoop surface, so a visible mesh occludes the deformation --
    # the dome looks rigid even though the lattice is flattening.
    mesh_cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False, has_particle_collision=False,
        is_visible=False)
    builder.add_shape_mesh(pad_link, mesh=pad_mesh, cfg=mesh_cfg)
    centers_rot = (assets["lattice_centers"] @ rotation.T).astype(np.float32, copy=True)
    builder.add_lattice(
        link=pad_link,
        morphit_json={"centers": centers_rot, "radii": assets["lattice_radii"]},
        total_mass=0.0,
        pos=wp.vec3(x0, 0.0, z0),
        k_anchor=params.pad_k_anchor,
        k_lateral=params.pad_k_lateral,
        k_bulk=params.pad_k_bulk,
        damping=params.pad_damping,
    )


PAD_BUILDERS: dict = {
    "box": _build_pad_box,
    "curved": _build_pad_curved,
}


def _pad_thickness(kind: str, params: SceneParams,
                   *, use_cslc: bool = False) -> float:
    """Effective pad half-thickness along the approach axis. Spawn
    position uses this so the inward face starts at +/- approach_gap/2
    regardless of pad geometry.

    In CSLC mode the box pad's effective inward face is the LATTICE
    outer surface, which protrudes ``PAD_SPHERE_PROTRUSION`` past the
    rigid box face. Use the lattice envelope so the APPROACH kinematics
    still bring the *contact-active* surface to the object face at
    end of approach (otherwise the lattice would contact the object
    PAD_SPHERE_PROTRUSION mm early, and ``squeeze_depth`` would no
    longer correspond to the contact-onset overlap).
    """
    if kind == "box":
        if use_cslc:
            return BOX_HX + PAD_SPHERE_PROTRUSION
        return BOX_HX
    if kind == "curved":
        return params.pad_curve_depth
    raise ValueError(f"unknown pad kind: {kind}")


def _load_curved_pad_assets() -> dict:
    """Load the curved-pad mesh and baked sphere packing (once per
    Example build). Both arrays live in the same body frame at native
    5x scale; the per-pad rotation is applied at lattice/mesh assembly
    time inside :func:`_build_pad_curved`."""
    raw_mesh = trimesh.load(_PAD_ASSET_DIR / "pad_5x.obj", force="mesh")
    with open(_PAD_ASSET_DIR / "pad.json") as f:
        pad_lattice_data = json.load(f)
    return {
        "mesh_verts": np.asarray(raw_mesh.vertices, dtype=np.float32),
        "mesh_indices": np.asarray(raw_mesh.faces.flatten(), dtype=np.int32),
        "lattice_centers": np.asarray(pad_lattice_data["centers"], dtype=np.float32),
        "lattice_radii": np.asarray(pad_lattice_data["radii"], dtype=np.float32),
    }


class Example:
    def __init__(self, viewer, args, params: SceneParams | None = None):
        # Allow callers (notably the sweep runner) to inject a customised
        # SceneParams without going through the CLI. Default preserves
        # the original ``Example(viewer, args)`` behaviour.
        self.p = params if params is not None else SceneParams()
        # Per-object setup overrides. Sphere/cube use SceneParams as-is;
        # the book has very different proportions (much taller than wide
        # in Z, very thin in X) and is gripped by fingertip-sized pads,
        # so several scene knobs that were tuned for the sphere/cube
        # case need substitution:
        #
        #   * ``obj_radius`` is the pad approach axis (X) half-extent
        #     for approach_speed. Sub in the book's X half-extent so
        #     the pads land on the book face instead of overshooting.
        #
        #   * ``pad_z0`` was set so the pads grip the equator of a
        #     4 cm sphere. The book is 20 cm tall and sits on the
        #     ground after SETTLE with its centre at obj_book_hz.
        #     Aim the pads at the book centre so the LIFT phase has
        #     symmetric moment arm and the DISTURB tilt response
        #     pivots about the middle of the book.
        #
        #   * ``squeeze_depth`` of 15 mm was sized to a 2.5 cm-thick
        #     box pad pressing into a 4 cm-wide ball. The fingertip
        #     pad is only 2 cm thick total, and 15 mm of penetration
        #     would push the pad face through the book centre. Drop
        #     to 5 mm: deep enough to recruit lattice spheres but
        #     shallow enough to avoid through-penetration.
        if args.object == "book":
            self.p = dataclasses.replace(
                self.p,
                obj_radius=self.p.obj_book_hx,
                pad_z0=self.p.obj_book_hz,
                squeeze_depth=0.005,
                # Spawn the pads close to the book so the APPROACH
                # phase doesn't leave the tall, top-heavy book sitting
                # unsupported long enough to drift on the ground. 6 cm
                # gap = pads at +/- 4 cm (vs +/- 11 cm at the default
                # 20 cm gap), and a shorter SETTLE means the book has
                # less unsupported time before the pads engage.
                approach_gap=0.06,
                settle_duration=0.3,
                # LIFT trajectory tuned for the paper figure:
                #   * lift_speed = 30 mm/s, lift_duration = 1.5 s,
                #     lift_ramp = 0.5 s (start AND end)
                #   * Net lift = lift_speed * (lift_duration - ramp)
                #     = 0.03 * 1.0 = 30 mm  (plenty of clearance, the
                #     book hangs ~3 cm above the ground at HOLD)
                #   * The 0.5 s end ramp decelerates the pads
                #     gradually instead of stopping them instantly --
                #     the compliant CSLC lattice would otherwise see a
                #     velocity impulse at the LIFT->HOLD boundary and
                #     ring at its natural frequency, polluting the
                #     pre-disturbance baseline.
                lift_duration=1.5,
                lift_speed=0.030,
                lift_ramp_duration=0.5,
            )
        # Optional CLI overrides (interactive viewer only): a single
        # command can then reproduce a specific paper-figure config.
        # ``params is not None`` (headless sweep / figure runners) sets
        # these through SceneParams directly and never carries the attrs,
        # so getattr(..., None) leaves that path untouched.
        if getattr(args, "obj_mass", None) is not None:
            self.p = dataclasses.replace(self.p, obj_mass=float(args.obj_mass))
        if getattr(args, "force", None) is not None:
            self.p = dataclasses.replace(
                self.p, disturb_force_amplitude=float(args.force))
        # Pad-stiffness overrides for dialing visible dome compliance.
        if getattr(args, "pad_k_anchor", None) is not None:
            self.p = dataclasses.replace(
                self.p, pad_k_anchor=float(args.pad_k_anchor))
        if getattr(args, "pad_k_bulk", None) is not None:
            self.p = dataclasses.replace(
                self.p, pad_k_bulk=float(args.pad_k_bulk))
        if getattr(args, "pad_k_lateral", None) is not None:
            self.p = dataclasses.replace(
                self.p, pad_k_lateral=float(args.pad_k_lateral))
        if getattr(args, "squeeze_depth", None) is not None:
            self.p = dataclasses.replace(
                self.p, squeeze_depth=float(args.squeeze_depth))
        self.frame_dt = self.p.frame_dt
        self.sim_substeps = self.p.sim_substeps
        self.sim_dt = self.p.sim_dt
        self.sim_time = 0.0
        self.sim_step = 0  # counts substeps, not frames
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # ----- Object: SM-rigid sphere-packed ball OR uniform cube ------
        # Both variants are spawned ~4 cm above the ground so they free-
        # fall during SETTLE; a near-ground spawn is unstable for SM-rigid
        # (shape matching fights ground contact at t=0 and the cluster
        # develops m/s-scale internal velocity dispersion -- working
        # reference: example_uxpbd_particle_drop).
        obj_z = _object_spawn_height(args.object, self.p)
        if args.object not in OBJECT_BUILDERS:
            raise ValueError(
                f"unknown object kind {args.object!r}; "
                f"registered: {sorted(OBJECT_BUILDERS)}")
        self.obj_group = OBJECT_BUILDERS[args.object](
            builder, self.p, (0.0, 0.0, obj_z))

        # ----- Two articulated pads -----------------------------------
        # Each pad: world --[prismatic X]--> slider --[prismatic Z]--> pad.
        # Slider is a massless intermediate link (no collision).
        # The pad body carries the curved scoop mesh AND the kinematic
        # lattice; both are pre-rotated so the convex face points inward.
        #
        # Pads start at x = +/- (approach_gap/2 + pad_curve_depth) so the
        # peak of the convex face is at +/- approach_gap/2 at t=0, then
        # prismatic-X moves them inward by `dx` from _pad_target_xz.
        # Pad spawn position uses the pad-kind-dependent half-thickness
        # so the inward face starts at +/- approach_gap/2 regardless of
        # which pad geometry is in use.
        if args.pad not in PAD_BUILDERS:
            raise ValueError(
                f"unknown pad kind {args.pad!r}; "
                f"registered: {sorted(PAD_BUILDERS)}")

        # Resolve the effective CSLC flag once: SceneParams toggle AND
        # the args.no_compliant override must both not block it.  Pad
        # thickness depends on this (CSLC mode uses the protruded
        # lattice envelope as the effective inward face).
        use_cslc_flag = self.p.use_compliant_lattice and not getattr(
            args, "no_compliant", False)

        pad_thickness = _pad_thickness(args.pad, self.p, use_cslc=use_cslc_flag)
        lx0 = -(self.p.approach_gap / 2.0 + pad_thickness)
        rx0 = +(self.p.approach_gap / 2.0 + pad_thickness)
        pad_z0 = self.p.pad_z0

        # Curved-pad-specific assets and per-side rotation: loaded only
        # when needed. Each pad picks up its rotation by label below.
        curved_assets = _load_curved_pad_assets() if args.pad == "curved" else None
        # Body-to-world rotations baked into the mesh and lattice. A
        # rotation of +/- 90 deg about the world Y axis sends the
        # convex-face normal (body +Z) to world +/- X. Keeping the
        # articulation at identity orientation preserves the prismatic
        # joint axes (world X and Z).
        curved_rotations = {
            "left": np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                dtype=np.float32),
            "right": np.array(
                [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=np.float32),
        } if args.pad == "curved" else {}

        # Ghost config for the slider's stub geometry (no collision, no mass).
        ghost_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=False,
            has_particle_collision=False,
            density=0.0,
        )

        self.dof = {}  # label -> qd index
        self.pad_bodies = []  # [left_pad_body, right_pad_body]

        # Debug staging: --num-pads lets us isolate the SM-rigid ball,
        # then a single inward-pressing pad, then the full grasp. Slice
        # the spec list rather than wrapping the build loop in an if.
        pad_specs = [("left", lx0), ("right", rx0)][: args.num_pads]
        for label, x0 in pad_specs:
            slider = builder.add_link(
                xform=wp.transform((x0, 0.0, pad_z0), wp.quat_identity()),
                mass=0.01,
                label=f"{label}_slider",
            )
            builder.add_shape_sphere(slider, radius=0.001, cfg=ghost_cfg)

            # mass=0.0: the pad's collision shape carries inertia via its
            # own density, avoiding the add_body(mass=m) + add_shape_*
            # double-counting gotcha (see example_uxpbd_lattice_stack.py).
            pad = builder.add_link(
                xform=wp.transform((x0, 0.0, pad_z0), wp.quat_identity()),
                mass=0.0,
                label=f"{label}_pad",
            )
            # Dispatch to the registered pad builder. Each variant gets
            # its own kwargs: curved needs assets + rotation, box needs
            # nothing extra. Adding a new pad kind = add a function with
            # this signature and an entry in PAD_BUILDERS.
            pad_kwargs: dict = {
                "pad_link": pad,
                "x0": x0,
                "z0": pad_z0,
                "params": self.p,
                "use_cslc": use_cslc_flag,
            }
            if args.pad == "curved":
                pad_kwargs["assets"] = curved_assets
                pad_kwargs["rotation"] = curved_rotations[label]
            PAD_BUILDERS[args.pad](builder, **pad_kwargs)

            j_x = builder.add_joint_prismatic(
                parent=-1,
                child=slider,
                axis=wp.vec3(1.0, 0.0, 0.0),
                parent_xform=wp.transform(
                    (x0, 0.0, pad_z0), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                label=f"{label}_x",
            )
            j_z = builder.add_joint_prismatic(
                parent=slider,
                child=pad,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform_identity(),
                child_xform=wp.transform_identity(),
                label=f"{label}_z",
            )
            builder.add_articulation([j_x, j_z], label=f"{label}_arm")

            self.dof[f"{label}_x"] = builder.joint_qd_start[j_x]
            self.dof[f"{label}_z"] = builder.joint_qd_start[j_z]
            self.pad_bodies.append(pad)

            for ji in (j_x, j_z):
                dof = builder.joint_qd_start[ji]
                builder.joint_target_ke[dof] = self.p.drive_ke
                builder.joint_target_kd[dof] = self.p.drive_kd
                builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()
        # Override friction params (kernel uses
        # mu_eff = 0.5 * (particle_mu + shape_material_mu)).
        self.model.particle_mu = self.p.mu
        self.model.soft_contact_mu = self.p.mu
        self.model.shape_material_mu.assign(
            np.full(self.model.shape_count, self.p.mu, dtype=np.float32))

        # --no-compliant disables CSLC for A/B comparison against the
        # rigid-lattice baseline. The args attribute is set by the CLI
        # parser added in create_parser() below.
        use_cslc = self.p.use_compliant_lattice and not getattr(
            args, "no_compliant", False)
        cslc_params = CSLCParams(
            clamp_delta_dot_max=self.p.pad_clamp_delta_dot,
            ka_tangent_ratio=self.p.pad_ka_tangent_ratio,
            c_lattice=self.p.pad_c_lattice,
            enable_lattice_pp_body_wrench=self.p.pad_enable_lattice_pp_body_wrench,
        ) if use_cslc else None
        self.solver = newton.solvers.SolverUXPBD(
            self.model,
            iterations=self.p.solver_iterations,
            stabilization_iterations=2,
            shock_propagation_k=self.p.shock_propagation_k,
            cslc_params=cslc_params,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        # Snapshot the SM-rigid ball's particle indices so test_final can
        # average their z position to track the ball's height.
        obj_idx = self.model.particle_groups[self.obj_group]
        if hasattr(obj_idx, "numpy"):
            obj_idx = obj_idx.numpy()
        self._obj_idx = np.asarray(list(obj_idx), dtype=np.int32)

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            # Front view: camera on -Y axis looking toward +Y so the pads
            # (at +/- X) and the lift direction (+Z) both lie in the
            # viewing plane. Slight downward pitch keeps the grasp and
            # the ground both in frame.
            pos=wp.vec3(0.0, -0.5, pad_z0 + 0.05),
            pitch=-5.0,
            yaw=90.0,
        )

        # Initial state snapshot (used by test_final to measure slip and lift).
        self._obj_z0 = float(self.state_0.particle_q.numpy()[
                             self._obj_idx, 2].mean())
        self._pad_z0 = float(self.state_0.body_q.numpy()[
                             self.pad_bodies[0], 2]) if self.pad_bodies else 0.0

        # Per-frame telemetry log. Populated by step(); consumed by the
        # sweep runner in example_uxpbd_lift_test_sweep.py. Keeping this
        # always-on (rather than gated behind a debug flag) keeps the
        # sweep and interactive code paths identical -- the printed
        # cadence stays the same; the dict is just appended every frame.
        self.history: list[dict] = []

        # Pad-lattice render overlay: the default particle render colours
        # EVERY particle the same tan, so the compliant pad lattice is
        # indistinguishable from the object's packed spheres and its
        # deformation can't be seen. Cache the lattice particle indices and
        # slightly-enlarged radii so render() can redraw just the pad in a
        # distinct colour on top. None when there is no lattice.
        self._lat_idx = None
        self._lat_radii_wp = None
        if self.model.lattice_sphere_count > 0:
            self._lat_idx = self.model.lattice_particle_index.numpy()
            self._lat_radii_wp = wp.array(
                self.model.lattice_r.numpy() * 1.08,
                dtype=wp.float32, device=self.model.device)

        # ---- DISTURB phase setup ----
        # Identify the bottom band of object particles (lowest z). The
        # disturbance force is split evenly across these so the moment
        # arm about the grip line is maximised. The selection is done
        # at spawn time (before SETTLE drops the object) because the
        # particle indices are fixed for the run; the z values shift
        # but the identity of the "bottom-most particles in the rest
        # layout" doesn't. Stored as a Warp int32 array on the same
        # device as ``particle_f`` so the kernel launch is zero-copy.
        obj_q0 = self.state_0.particle_q.numpy()[self._obj_idx]
        z_min = float(obj_q0[:, 2].min())
        z_max = float(obj_q0[:, 2].max())
        z_thresh = z_min + (z_max - z_min) * float(
            self.p.disturb_force_bottom_fraction)
        bottom_local = np.nonzero(obj_q0[:, 2] <= z_thresh)[0]
        bottom_global = self._obj_idx[bottom_local].astype(np.int32)
        # Cache the per-particle force vector base direction and the
        # pre-allocated wp.array of indices. The amplitude scales with
        # mass at call time so changing ``obj_mass`` between runs
        # doesn't require touching this setup.
        self._disturb_particles_wp = wp.array(
            bottom_global,
            dtype=wp.int32,
            device=self.state_0.particle_q.device,
        )
        self._disturb_n_particles = int(bottom_global.size)
        self._disturb_dir = np.array(
            [self.p.disturb_force_dir_x,
             self.p.disturb_force_dir_y,
             self.p.disturb_force_dir_z],
            dtype=np.float32,
        )
        # Normalise so amplitude has the units of [N] regardless of
        # whether the user supplied a unit vector.
        _dnorm = float(np.linalg.norm(self._disturb_dir))
        if _dnorm > 1.0e-12:
            self._disturb_dir = self._disturb_dir / _dnorm
        # Snapshot rest-layout particle positions for the tilt SVD.
        # Anchored at spawn (not end-of-SETTLE) because what we want
        # to measure is rotation of the rigid cluster's frame of
        # reference, and SETTLE keeps the orientation identity to
        # numerical noise.
        self._obj_rest = obj_q0 - obj_q0.mean(axis=0)
        # Frame-by-frame disturbance accounting (for plot annotations).
        self._disturb_t_start = _disturb_phase_start(self.p)
        self._disturb_t_end = (self._disturb_t_start
                                + self.p.disturb_force_duration)

    def _set_pad_targets(self):
        """Write the prismatic-joint position targets for active pads."""
        if not self.dof:
            return
        dx, dz = _pad_target_xz(self.sim_step, self.p)
        target = self.control.joint_target_pos.numpy()
        if "left_x" in self.dof:
            target[self.dof["left_x"]] = +dx
            target[self.dof["left_z"]] = +dz
        if "right_x" in self.dof:
            target[self.dof["right_x"]] = -dx
            target[self.dof["right_z"]] = +dz
        self.control.joint_target_pos.assign(
            wp.array(target, dtype=wp.float32,
                     device=self.control.joint_target_pos.device)
        )

    def _apply_disturb_force(self) -> bool:
        """If the current substep lies inside the DISTURB pulse window,
        launch the Warp kernel that adds the per-particle transverse
        force to the bottom band. Must run AFTER ``clear_forces`` (which
        zeroes ``particle_f``) and BEFORE ``solver.step`` so the force
        is integrated into velocity in this substep. Returns True iff
        the force was actually applied (used for telemetry annotation).
        """
        if (self._disturb_n_particles <= 0
                or self.p.disturb_force_amplitude <= 0.0
                or self.p.disturb_force_duration <= 0.0):
            return False
        t = self.sim_step * self.sim_dt
        if t < self._disturb_t_start or t >= self._disturb_t_end:
            return False
        # Split the amplitude across the bottom band so the *total*
        # applied force equals ``disturb_force_amplitude``.
        f_per = (float(self.p.disturb_force_amplitude)
                 / float(self._disturb_n_particles))
        fvec = wp.vec3(
            float(self._disturb_dir[0] * f_per),
            float(self._disturb_dir[1] * f_per),
            float(self._disturb_dir[2] * f_per),
        )
        wp.launch(
            _disturb_add_force_kernel,
            dim=self._disturb_n_particles,
            inputs=[self.state_0.particle_f,
                    self._disturb_particles_wp, fvec],
            device=self.state_0.particle_f.device,
        )
        return True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._set_pad_targets()
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            # DISTURB force pulse goes between clear_forces and the
            # solver step so it's integrated into velocity this substep.
            self._apply_disturb_force()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_step += 1

    def _record_frame(self) -> dict:
        """Snapshot the current sim state into one telemetry row.

        One row per call (post-substep). Numpy reads are batched: one
        ``particle_q``, one ``particle_qd``, optionally one ``body_q``
        and one ``lattice_delta``. At default fps=100 and ~6 s
        scenarios this is ~600 rows -- cheap enough to keep always-on.
        """
        frame = int(round(self.sim_time * self.p.fps))
        phase = _phase_at(self.sim_time, self.p)
        q = self.state_0.particle_q.numpy()[self._obj_idx]
        v = self.state_0.particle_qd.numpy()[self._obj_idx]
        obj = q.mean(axis=0)
        objv = v.mean(axis=0)
        # |v|_max reveals per-particle blow-up even when the mean
        # cancels out (e.g. symmetric SM correction).
        v_max = float(np.linalg.norm(v, axis=1).max())
        # Pad pose: left pad is the "leader" the test_final slip metric
        # uses. Both pads' z follows the same target so one is enough.
        left_pad_x = left_pad_z = float("nan")
        if self.pad_bodies:
            pad_q = self.state_0.body_q.numpy()[self.pad_bodies[0]]
            left_pad_x = float(pad_q[0])
            left_pad_z = float(pad_q[2])
        # Lattice compression magnitudes. delta is vec3 so we take the
        # vector norm per sphere; this collapses to |δ_n| for the
        # face-on regime and remains meaningful for tilted contacts.
        delta_max = 0.0
        delta_mean = 0.0
        n_active = 0
        if self.model.lattice_sphere_count > 0:
            ld = self.model.lattice_delta.numpy()
            ld_mag = np.linalg.norm(ld, axis=1)
            delta_max = float(ld_mag.max())
            delta_mean = float(ld_mag.mean())
            n_active = int((ld_mag > 1.0e-6).sum())
        # Derived "did the grasp work?" quantities: object lift, pad
        # lift, and slip = pad_lift - obj_lift (positive means the
        # object lagged the pad, i.e. slipped down through the grip).
        obj_lift = float(obj[2]) - self._obj_z0
        pad_lift = (left_pad_z - self._pad_z0) if self.pad_bodies else 0.0
        slip = pad_lift - obj_lift

        # Tilt angle of the object's local +Z axis, relative to world +Z.
        # Recovered by a Procrustes (Kabsch) fit between rest-layout
        # particle positions and the current ones -- the same algorithm
        # SRXPBD uses internally for shape matching. The reflection-free
        # variant (det-sign correction on V*Uᵀ) keeps R in SO(3) so the
        # extracted "book up" vector is sensible even if SVD picks a
        # mirrored basis. tilt_axis_x is the dominant tilt axis if the
        # disturbance was along Y (book tips in the YZ plane → tilt
        # about +X). For runs with no measurable tilt the Procrustes is
        # near-identity and tilt_deg ~ 0.
        # Subtract centroid each side so translation doesn't enter R.
        q_cent = q - obj
        # Cross-covariance H = rest.T @ current.
        H = self._obj_rest.T.astype(np.float64) @ q_cent.astype(np.float64)
        try:
            U, _S, Vt = np.linalg.svd(H)
            # Reflection guard.
            d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
            D = np.diag([1.0, 1.0, d])
            R = Vt.T @ D @ U.T
            up_now = R @ np.array([0.0, 0.0, 1.0])
            cos_tilt = float(np.clip(up_now[2], -1.0, 1.0))
            tilt_rad = float(np.arccos(cos_tilt))
            tilt_deg = float(np.degrees(tilt_rad))
            # Signed tilts decomposed onto the two horizontal axes:
            #   tilt_about_x_deg : positive when book top tips toward +Y
            #                      (used when the disturbance is along Y)
            #   tilt_about_y_deg : positive when book top tips toward +X
            #                      (used when the disturbance is along X,
            #                      i.e. perpendicular to the book's broad
            #                      flat face — the current default)
            tilt_about_x_deg = float(np.degrees(np.arctan2(
                float(up_now[1]), float(up_now[2]))))
            tilt_about_y_deg = float(np.degrees(np.arctan2(
                float(up_now[0]), float(up_now[2]))))
        except np.linalg.LinAlgError:
            tilt_deg = float("nan")
            tilt_about_x_deg = float("nan")
            tilt_about_y_deg = float("nan")

        # Distinguish whether the disturbance pulse is currently active
        # so the time-series plot can shade the pulse window.
        pulse_active = int(self._disturb_t_start
                           <= self.sim_time
                           < self._disturb_t_end)

        return {
            "frame": frame,
            "t": float(self.sim_time),
            "phase": phase,
            "pulse_active": pulse_active,
            "obj_x": float(obj[0]),
            "obj_y": float(obj[1]),
            "obj_z": float(obj[2]),
            "obj_vx": float(objv[0]),
            "obj_vy": float(objv[1]),
            "obj_vz": float(objv[2]),
            "obj_v_max": v_max,
            "left_pad_x": left_pad_x,
            "left_pad_z": left_pad_z,
            "obj_lift": obj_lift,
            "pad_lift": pad_lift,
            "slip": slip,
            "tilt_deg": tilt_deg,
            "tilt_about_x_deg": tilt_about_x_deg,
            "tilt_about_y_deg": tilt_about_y_deg,
            "n_active": n_active,
            "lattice_sphere_count": int(self.model.lattice_sphere_count),
            "delta_max": delta_max,
            "delta_mean": delta_mean,
        }

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        # Always record telemetry; print on the existing cadence so
        # interactive output stays readable. Dense sampling in the
        # first 20 frames catches the t~0 SRXPBD spike (shape-matching
        # vs ground at the spawn); sparser cadence afterwards.
        row = self._record_frame()
        self.history.append(row)
        if row["frame"] < 20 or row["frame"] % 25 == 0:
            msg = (
                f"[f={row['frame']:03d} t={row['t']:.3f} {row['phase']:>7s}] "
                f"obj=({row['obj_x']:+.4f},{row['obj_y']:+.4f},{row['obj_z']:+.4f}) "
                f"obj_v=({row['obj_vx']:+.4f},{row['obj_vy']:+.4f},{row['obj_vz']:+.4f}) "
                f"|v|_max={row['obj_v_max']:.3f}"
            )
            if self.pad_bodies:
                msg += (
                    f" left_pad_x={row['left_pad_x']:+.4f}"
                    f" left_pad_z={row['left_pad_z']:+.4f}"
                )
            if row["lattice_sphere_count"] > 0:
                msg += (
                    f" delta_max={row['delta_max'] * 1e3:+.3f}mm"
                    f" delta_mean={row['delta_mean'] * 1e3:+.4f}mm"
                    f" n_active={row['n_active']}/{row['lattice_sphere_count']}"
                )
            print(msg)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        # Redraw the pad lattice spheres coloured by their compression
        # |delta| (blue = undeformed, red = most compressed), on top of the
        # uniform-tan default particle render. A single flat colour only
        # distinguishes the pad from the object; colouring by delta shows
        # WHERE the dome is deforming -- the contact patch lights up red as
        # it conforms, the rest stays blue. Slightly enlarged radii cover
        # the tan copy beneath.
        if self._lat_idx is not None:
            dev = self.state_0.particle_q.device
            lat_xyz = self.state_0.particle_q.numpy()[self._lat_idx]
            lat_q = wp.array(lat_xyz, dtype=wp.vec3, device=dev)
            dmag = np.linalg.norm(self.model.lattice_delta.numpy(), axis=1)
            # Normalise to the current frame max, floored at 1 mm so small
            # deformations stay cool instead of saturating the colour map.
            scale = max(float(dmag.max()), 1.0e-3)
            t = np.clip(dmag / scale, 0.0, 1.0)
            cols = np.stack([t, np.full_like(t, 0.15), 1.0 - t], axis=1)
            lat_colors = wp.array(cols.astype(np.float32), dtype=wp.vec3, device=dev)
            self.viewer.log_points(
                "/pad_lattice", lat_q, self._lat_radii_wp, lat_colors)
        self.viewer.end_frame()

    def test_final(self):
        obj_q = self.state_0.particle_q.numpy()[self._obj_idx]
        obj_z = float(obj_q[:, 2].mean())
        obj_lift = obj_z - self._obj_z0

        # Numerical sanity applies to every stage: catches SM-rigid
        # shape-matching blow-ups before we look at the geometry.
        assert np.isfinite(obj_q).all(
        ), f"NaN/Inf in object positions (num_pads={self.args.num_pads})"

        if self.args.num_pads == 0:
            # Stage A: ball alone, free-falls onto the ground and rests.
            # After SETTLE, the lowest sphere surface should be tangent
            # to z=0, putting the cluster centroid at world z ~ obj_radius
            # (lowest body-frame center = -(obj_radius - obj_sphere_r),
            # plus obj_sphere_r to put its surface on the ground, gives
            # cluster center z = obj_radius).
            expected_rest_z = self.p.obj_radius
            obj_xy = obj_q[:, :2].mean(axis=0)
            v_max = float(np.linalg.norm(
                self.state_0.particle_qd.numpy()[self._obj_idx], axis=1).max())

            assert obj_z > 0.0, f"Ball penetrated ground: obj_z={obj_z:.4f}"
            assert abs(obj_z - expected_rest_z) < 0.005, (
                f"Ball did not settle to rest height: obj_z={obj_z:.4f}, expected ~{expected_rest_z:.4f}"
            )
            assert np.linalg.norm(
                obj_xy) < 0.005, f"Ball drifted in XY: |xy|={np.linalg.norm(obj_xy) * 1e3:.2f} mm"
            # Loose velocity bound: per-particle |v|_max stays ~0.3-0.5
            # m/s in steady-state observed runs (residual SM shake while
            # the cluster is in contact with the ground). >1 m/s means
            # the cluster isn't actually at rest.
            assert v_max < 1.0, f"Ball still moving after SETTLE: |v|_max={v_max:.3f} m/s"
            return

        if self.args.num_pads == 1:
            # Stage B: a single inward-pressing pad has nothing to
            # squeeze against, so we expect the ball to be pushed
            # sideways. Just check we did not blow up or eject.
            assert np.linalg.norm(obj_q.mean(
                axis=0)) < 0.5, "Ball escaped under single-pad push"
            return

        # Stage C (default): full two-pad grasp. We check:
        #   1. The object did not fall (its z is well above its spawn).
        #   2. The pads did rise (sanity check on the drive).
        #   3. Slip between object and pads is bounded (the object
        #      tracked the pad's vertical motion within a tolerance).
        pad_z = float(self.state_0.body_q.numpy()[self.pad_bodies[0], 2])
        pad_lift = pad_z - self._pad_z0
        # Slip = pad-frame z drift of the object. Positive means the
        # object lagged behind the pad (slipped down through the grip).
        slip = pad_lift - obj_lift

        assert obj_z > self._obj_z0 - \
            0.005, f"Object dropped: obj_z={obj_z:.4f}, started at {self._obj_z0:.4f}"
        assert pad_lift > 0.5 * self.p.lift_speed * self.p.lift_duration, (
            f"Pads did not lift: pad_lift={pad_lift:.4f}, expected ~{self.p.lift_speed * self.p.lift_duration:.4f}"
        )
        # Loose tolerance: position-based friction in UXPBD will leak
        # some tangential motion per iteration.
        assert abs(slip) < 0.01, (
            f"Object slipped too much: slip={slip * 1e3:.2f} mm "
            f"(pad_lift={pad_lift * 1e3:.2f} mm, obj_lift={obj_lift * 1e3:.2f} mm)"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--num-pads",
            type=int,
            default=2,
            choices=[0, 1, 2],
            help=(
                "Number of pads to build. Debug helper: 0 = ball only "
                "(check the SM-rigid object stays put on the ground), "
                "1 = single inward-pressing pad, 2 = full grasp (default)."
            ),
        )
        parser.add_argument(
            "--object",
            choices=tuple(OBJECT_BUILDERS),
            default="sphere",
            help=(
                "Grasped object geometry. 'sphere' (default) loads the "
                "125-particle MorphIt packing of a unit sphere. 'cube' "
                "swaps in a uniform 4x4x4 (=64) particle cube of matching "
                "outer extent for faster iteration. 'book' is a tall thin "
                "rectangular sphere-packing (see obj_book_* in SceneParams) "
                "used by the DISTURB phase to measure rotational stiffness "
                "/ pressure-gradient response under a transverse force pulse."
            ),
        )
        parser.add_argument(
            "--pad",
            choices=("curved", "box"),
            default="curved",
            help=(
                "Pad geometry. 'curved' (default) loads the MorphIt scoop "
                "mesh + 125-sphere lattice from assets/pad/. 'box' replaces "
                "each pad with a rectangular collision box + uniform 2x4x4 "
                "(=32) lattice. The box variant has the same inward face "
                "position at t=0 as the curved peak, so approach/squeeze "
                "timing is unchanged, but mesh contact and large lattice "
                "counts are both eliminated -- expect ~2-3x faster substeps."
            ),
        )
        parser.add_argument(
            "--no-compliant",
            action="store_true",
            help=(
                "Disable the CSLC compliant-lattice path and run on the "
                "rigid-lattice baseline (Phase 1 behaviour). Use to "
                "A/B-test the same scene with and without compliance: "
                "compliant should produce smaller slip in the LIFT phase "
                "because the pad spheres displace inward to conform to "
                "the object surface and recruit more contact pairs."
            ),
        )
        parser.add_argument(
            "--obj-mass",
            type=float,
            default=None,
            help=(
                "Override the grasped object mass [kg]. When unset the "
                "SceneParams default is used. Set to 0.3 to reproduce the "
                "disturbance-figure config in the interactive viewer."
            ),
        )
        parser.add_argument(
            "--force",
            type=float,
            default=None,
            help=(
                "Override the DISTURB transverse pulse amplitude [N]. When "
                "unset the SceneParams default is used. Set to 3 to "
                "reproduce the disturbance figures."
            ),
        )
        parser.add_argument(
            "--pad-k-anchor",
            type=float,
            default=None,
            help=(
                "Override the lattice anchor stiffness [N/m] (default "
                "1e5). DOMINANT lever for visible compliance: skin "
                "compression scales as delta ~ F_contact / k_anchor, so "
                "lowering this makes the domes flatten visibly. Try "
                "~1e3 to see mm-scale deformation; too low (<~1e2) and "
                "the grip can no longer carry the object."
            ),
        )
        parser.add_argument(
            "--pad-k-bulk",
            type=float,
            default=None,
            help=(
                "Override the lattice contact (bulk/Hertz) stiffness "
                "(default 1e8). Keep high so the sphere displaces "
                "(grows delta) rather than penetrating the object."
            ),
        )
        parser.add_argument(
            "--pad-k-lateral",
            type=float,
            default=None,
            help=(
                "Override the lateral graph-Laplacian stiffness [N/m] "
                "(default 5e3). Spreads the apex deflection to "
                "neighbours -- higher = broader, smoother dome "
                "flattening (see plot_dome_kl_sweep)."
            ),
        )
        parser.add_argument(
            "--squeeze-depth",
            type=float,
            default=None,
            help=(
                "Override the squeeze penetration depth [m] (book "
                "default 5mm, sphere/cube 15mm). The dome flattens by "
                "~this depth, so increase it for more dramatic "
                "flattening -- bounded by the object thickness (pressing "
                "past it pushes the pads through / destabilises)."
            ),
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
