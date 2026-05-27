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
    pad_k_anchor: float = 1.0e2  # N/m per sphere, anchor spring
    # N/m per sphere, bulk contact spring (ratio ~0.95)
    pad_k_bulk: float = 1.0e2
    pad_k_lateral: float = 5.0e3  # N/m, reserved for v2 jacobi
    pad_damping: float = 2.0  # s/m, reserved for v2 Hunt-Crossley force
    # Numerical guard against the first-substep delta jump when the
    # object first touches a pad and lattice_delta_prev = 0 leaves the
    # finite-difference rate huge. SETTLE phase usually keeps the pads
    # away from the object so this rarely fires; leave at 1.0 m/s for
    # safety. Pass None into CSLCParams to disable entirely.
    pad_clamp_delta_dot: float | None = 1.0

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
        # C1-smooth velocity ramp at start of LIFT: v(s) = lift_speed * (3s^2 - 2s^3),
        # integrated to z(s) = lift_speed * ramp * (s^3 - s^4/2).
        ramp = p.lift_ramp_duration
        if ramp <= 0.0 or t_lift >= ramp:
            return p.lift_speed * max(t_lift - 0.5 * ramp, 0.0) if t_lift >= ramp else 0.0
        s = t_lift / ramp
        return p.lift_speed * ramp * (s**3 - 0.5 * s**4)

    if t < p.lift_duration:
        return dx_total, _lift_dz(t)
    # HOLD: freeze at end-of-LIFT pose.
    return dx_total, _lift_dz(p.lift_duration)


class Example:
    def __init__(self, viewer, args):
        self.p = SceneParams()
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
        obj_z = self.p.obj_radius + 0.04
        if args.object == "cube":
            # Uniform 4x4x4 = 64 particle cube of half-side obj_radius.
            # Settled centroid lands at z = obj_radius (bottom sphere
            # surface tangent to z=0), matching the sphere variant so
            # test_final's rest-height check is unchanged.
            n = 4
            sphere_r = self.p.obj_radius / n  # 0.01 m at obj_radius=0.04
            coords = np.linspace(
                -self.p.obj_radius + sphere_r,
                self.p.obj_radius - sphere_r,
                n,
            )
            xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
            obj_centers = np.stack(
                [xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
            obj_radii = np.full(
                obj_centers.shape[0], sphere_r, dtype=np.float32)
            self.obj_group = builder.add_particle_volume(
                volume_data={"centers": obj_centers.tolist(),
                             "radii": obj_radii.tolist()},
                total_mass=self.p.obj_mass,
                pos=wp.vec3(0.0, 0.0, obj_z),
            )
        else:
            # MorphIt-packed sphere (current default). Rescale the unit
            # packing so the outer envelope has radius obj_radius;
            # native_envelope = max_i(|c_i| + r_i) is the furthest sub-
            # sphere surface from the body origin.
            with open(_SPHERE_ASSET_DIR / "sphere.json") as f:
                sphere_data = json.load(f)
            native_centers = np.asarray(
                sphere_data["centers"], dtype=np.float32)
            native_radii = np.asarray(sphere_data["radii"], dtype=np.float32)
            native_masses = np.asarray(sphere_data["masses"], dtype=np.float32)
            native_envelope = float(
                (np.linalg.norm(native_centers, axis=1) + native_radii).max())
            obj_scale = self.p.obj_radius / native_envelope
            obj_centers = (native_centers * obj_scale).astype(np.float32)
            obj_radii = (native_radii * obj_scale).astype(np.float32)
            self.obj_group = builder.add_particle_volume(
                volume_data={"centers": obj_centers.tolist(),
                             "radii": obj_radii.tolist()},
                total_mass=self.p.obj_mass,
                pos=wp.vec3(0.0, 0.0, obj_z),
            )
            # Override add_particle_volume's volume-weighted mass
            # distribution with MorphIt's physics-optimised per-particle
            # masses from sphere.json. MorphIt jointly tunes per-sphere
            # mass + position to minimise the discrepancy between the
            # packing and the true sphere's mass / COM / inertia; the
            # JSON's masses array carries that optimisation. Volume
            # weighting (m_i ~ r_i^3) discards it. Rescale so the total
            # still equals obj_mass.
            mass_scale = self.p.obj_mass / float(native_masses.sum())
            obj_masses = (native_masses * mass_scale).astype(np.float32)
            for idx, m in zip(builder.particle_groups[self.obj_group], obj_masses):
                builder.particle_mass[idx] = float(m)

        # ----- Two articulated pads -----------------------------------
        # Each pad: world --[prismatic X]--> slider --[prismatic Z]--> pad.
        # Slider is a massless intermediate link (no collision).
        # The pad body carries the curved scoop mesh AND the kinematic
        # lattice; both are pre-rotated so the convex face points inward.
        #
        # Pads start at x = +/- (approach_gap/2 + pad_curve_depth) so the
        # peak of the convex face is at +/- approach_gap/2 at t=0, then
        # prismatic-X moves them inward by `dx` from _pad_target_xz.
        # Box-pad geometry (used when args.pad == "box"). Sized so the
        # inward (+X for left) face matches the curved pad's peak position
        # at t=0; this lets approach_speed / approach_duration carry over
        # unchanged. hz is matched to the curved lattice's vertical extent
        # so pad_z0 = 0.06 still gives ground clearance.
        BOX_HX, BOX_HY, BOX_HZ = 0.025, 0.05, 0.05
        BOX_LATTICE_N = (2, 4, 4)  # 32 spheres per pad
        # Uniform cell, so sphere_r = half-cell = half_extent / n_axis on
        # the limiting axis; equal-cell layout requires hy == hz and
        # hx == hy * n_x / n_yz, which our chosen sizes satisfy.
        pad_thickness = BOX_HX if args.pad == "box" else self.p.pad_curve_depth
        lx0 = -(self.p.approach_gap / 2.0 + pad_thickness)
        rx0 = +(self.p.approach_gap / 2.0 + pad_thickness)
        pad_z0 = self.p.pad_z0

        # Curved-pad assets: only loaded if we're actually using them.
        pad_mesh_verts = pad_mesh_indices = None
        pad_lattice_centers = pad_lattice_radii = None
        pad_rotations: dict[str, np.ndarray] = {}
        if args.pad == "curved":
            # Load the pad scoop mesh and the MorphIt sphere packing baked
            # from that same mesh. Both arrays live in the same body frame
            # at native 5x scale (~10 cm wide, 5 cm curve depth); see
            # pad.json's metadata ("mesh_path": ".../pad_5x.obj").
            raw_mesh = trimesh.load(
                _PAD_ASSET_DIR / "pad_5x.obj", force="mesh")
            pad_mesh_verts = np.asarray(raw_mesh.vertices, dtype=np.float32)
            pad_mesh_indices = np.asarray(
                raw_mesh.faces.flatten(), dtype=np.int32)
            with open(_PAD_ASSET_DIR / "pad.json") as f:
                pad_lattice_data = json.load(f)
            pad_lattice_centers = np.asarray(
                pad_lattice_data["centers"], dtype=np.float32)
            pad_lattice_radii = np.asarray(
                pad_lattice_data["radii"], dtype=np.float32)

            # Body-to-world rotations baked into the mesh and lattice. A
            # rotation of +/- 90 deg about the world Y axis sends the
            # convex-face normal (body +Z) to world +/- X. Applying the
            # rotation in NumPy keeps the articulated body itself at
            # identity orientation, which preserves the prismatic joint
            # axes (world X and Z).
            #
            #   R(Y, +90) . (0,0,1) = (+1,0,0)   left pad faces +X (center)
            #   R(Y, -90) . (0,0,1) = (-1,0,0)   right pad faces -X (center)
            R_left = np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
            R_right = np.array(
                [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
            pad_rotations = {"left": R_left, "right": R_right}

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
            if args.pad == "box":
                # Rectangular pad: axis-aligned box centered at the body
                # origin extends from -BOX_HX (back) to +BOX_HX (inward
                # face); +x_local always points "inward" because the
                # left-pad body spawns at -|x| and the right at +|x|, so
                # the inward face naturally lands on opposite sides of the
                # ball without per-side rotation.
                #
                # Collision is disabled on the box itself when the
                # compliant lattice is on: the lattice spheres are the
                # contact surface (so we can see them deform), and the
                # box is kept only for inertia. The box visual is also
                # hidden because the inscribed lattice spheres live
                # exactly inside it -- a rendered box would occlude the
                # very particles we want to watch deform. When CSLC is
                # off we keep the box as the rigid contact surface and
                # visual (lattice is kinematic-only in that case).
                if self.p.use_compliant_lattice and not getattr(args, "no_compliant", False):
                    pad_box_cfg = newton.ModelBuilder.ShapeConfig(
                        has_shape_collision=False,
                        has_particle_collision=False,
                        is_visible=False,
                    )
                    builder.add_shape_box(
                        pad, hx=BOX_HX, hy=BOX_HY, hz=BOX_HZ, cfg=pad_box_cfg)
                else:
                    builder.add_shape_box(pad, hx=BOX_HX, hy=BOX_HY, hz=BOX_HZ)
                # Uniform sphere lattice filling the box. Equal cell
                # spacing on each axis (sphere_r = half-cell) so the
                # spheres tile without overlap on the same host link.
                nx, ny, nz = BOX_LATTICE_N
                sphere_r = BOX_HX / nx
                cx = np.linspace(-BOX_HX + sphere_r, BOX_HX - sphere_r, nx)
                cy = np.linspace(-BOX_HY + sphere_r, BOX_HY - sphere_r, ny)
                cz = np.linspace(-BOX_HZ + sphere_r, BOX_HZ - sphere_r, nz)
                xs, ys, zs = np.meshgrid(cx, cy, cz, indexing="ij")
                pad_centers = np.stack(
                    [xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
                pad_radii = np.full(
                    pad_centers.shape[0], sphere_r, dtype=np.float32)
                builder.add_lattice(
                    link=pad,
                    morphit_json={
                        "centers": pad_centers,
                        "radii": pad_radii,
                    },
                    total_mass=0.0,
                    pos=wp.vec3(x0, 0.0, pad_z0),
                    k_anchor=self.p.pad_k_anchor,
                    k_lateral=self.p.pad_k_lateral,
                    k_bulk=self.p.pad_k_bulk,
                    damping=self.p.pad_damping,
                )
            else:
                R = pad_rotations[label]
                # Pre-rotate mesh vertices and lattice centers into the
                # body frame the pad will actually use at runtime. .copy()
                # is required by newton.Mesh / add_lattice to get
                # contiguous float32 buffers from the transposed view.
                verts_rot = (pad_mesh_verts @
                             R.T).astype(np.float32, copy=True)
                centers_rot = (pad_lattice_centers @
                               R.T).astype(np.float32, copy=True)
                pad_mesh = newton.Mesh(verts_rot, pad_mesh_indices)
                builder.add_shape_mesh(pad, mesh=pad_mesh)
                builder.add_lattice(
                    link=pad,
                    morphit_json={
                        "centers": centers_rot,
                        "radii": pad_lattice_radii,
                    },
                    total_mass=0.0,
                    pos=wp.vec3(x0, 0.0, pad_z0),
                    k_anchor=self.p.pad_k_anchor,
                    k_lateral=self.p.pad_k_lateral,
                    k_bulk=self.p.pad_k_bulk,
                    damping=self.p.pad_damping,
                )

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
            clamp_delta_dot_max=self.p.pad_clamp_delta_dot) if use_cslc else None
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

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._set_pad_targets()
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_step += 1

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        # DEBUG: object + pad telemetry (ball centroid, averaged over the
        # SM-rigid sphere packing)
        frame = int(round(self.sim_time * self.p.fps))
        # Dense sampling in the first 20 frames catches the t~0 SRXPBD
        # spike (shape-matching vs ground at the spawn); sparser cadence
        # afterwards keeps the log readable.
        if frame < 20 or frame % 25 == 0:
            q = self.state_0.particle_q.numpy()[self._obj_idx]
            v = self.state_0.particle_qd.numpy()[self._obj_idx]
            obj = q.mean(axis=0)
            objv = v.mean(axis=0)
            # |v|_max reveals per-particle blow-up even when the mean
            # cancels out (e.g. symmetric SM correction).
            v_max = float(np.linalg.norm(v, axis=1).max())
            msg = (
                f"[f={frame:03d} t={self.sim_time:.3f}] "
                f"obj=({obj[0]:+.4f},{obj[1]:+.4f},{obj[2]:+.4f}) "
                f"obj_v=({objv[0]:+.4f},{objv[1]:+.4f},{objv[2]:+.4f}) "
                f"|v|_max={v_max:.3f}"
            )
            if self.pad_bodies:
                pad = self.state_0.body_q.numpy()[self.pad_bodies[0]]
                msg += f" left_pad_x={pad[0]:+.4f} left_pad_z={pad[2]:+.4f}"
            # CSLC diagnostic: max/mean per-sphere compression. Zero when
            # cslc_params is None or no lattice spheres are in contact.
            # n_active counts spheres with delta > 1 um so we can see the
            # contact patch grow / shrink during APPROACH / SQUEEZE / LIFT.
            if self.model.lattice_sphere_count > 0:
                # lattice_delta is vec3 — magnitude per sphere is the
                # physically meaningful "compression amount" (collapses
                # to δ_n for normal-axis-only deformation).
                ld = self.model.lattice_delta.numpy()
                ld_mag = np.linalg.norm(ld, axis=1)
                n_active = int((ld_mag > 1.0e-6).sum())
                msg += (
                    f" delta_max={ld_mag.max() * 1e3:+.3f}mm"
                    f" delta_mean={ld_mag.mean() * 1e3:+.4f}mm"
                    f" n_active={n_active}/{self.model.lattice_sphere_count}"
                )
            print(msg)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
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
            choices=("sphere", "cube"),
            default="sphere",
            help=(
                "Grasped object geometry. 'sphere' (default) loads the "
                "125-particle MorphIt packing of a unit sphere. 'cube' "
                "swaps in a uniform 4x4x4 (=64) particle cube of matching "
                "outer extent for faster iteration -- ~30%% fewer particles, "
                "no MorphIt asset load, otherwise identical add_particle_volume "
                "path."
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
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
