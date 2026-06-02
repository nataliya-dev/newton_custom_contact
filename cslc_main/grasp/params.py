# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grasp-test configuration — every tuning knob lives here.

Top-level ``GraspConfig`` holds a tree of nested dataclasses, one per
concern (object, pad, material, CSLC, hydroelastic, timing, solver,
drive, logging).  Every parameter is documented inline with its physical
meaning and reasonable bounds; defaults are calibrated to the production
``may_18_summary.md`` settings on a tennis-ball-sized sphere held by box
pads.

The CLI in :mod:`cslc_main.grasp.main` reads a few "hot" overrides
(``--pad-kind``, ``--contact-model``, ``--solver``, ``--cslc-kl``,
etc.) into this config, but everything else stays at the defaults
defined below.  To run a new ablation, edit a default here or
construct a ``GraspConfig`` programmatically — never sprinkle constants
across the scene-builder modules.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Repository root: cslc_main/grasp/params.py → ../..  is the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DOME_OBJ = _REPO_ROOT / "assets" / "pad" / "pad.obj"
_DEFAULT_BUNNY_OBJ = _REPO_ROOT / "assets" / "bunny" / "bunny.obj"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "grasp"


# ── Object (held body) ───────────────────────────────────────────────────


@dataclass
class ObjectParams:
    """The body the pads grasp.

    Default sphere matches the ITF tennis-ball specification: 67 mm
    diameter (radius 33.5 mm), mass ≈ 58 g.  Density chosen so the
    analytic sphere primitive recovers that mass:
        m = (4/3) · π · r³ · ρ  →  ρ = m / V ≈ 368 kg/m³.
    """

    # "sphere" (default, tennis ball) or "box" (dome-vs-flat-face
    # falsification).  The box's surface is uniformly point-set sampled
    # at construction; the CSLC handler dispatches to its single v2
    # unified ``_launch`` path automatically.
    kind: str = "sphere"

    # Sphere radius [m].  Tennis ball: 0.0335.  Only used when
    # ``kind == "sphere"``.
    radius: float = 0.0335

    # Fibonacci-spiral sample count on the sphere surface (Phase 7).
    # Default 300 ⇒ mean spacing ≈ sqrt(4π·R² / N) ≈ 7 mm on a 33.5 mm
    # sphere.  This must be smaller than the pad's tangential kernel
    # half-width (= r_pad) so each pad sphere can see at least one
    # target sample inside its contact kernel; without that, the
    # alignment+gate+w_tangent hard-culls in jacobi_step yield ZERO
    # active samples and the apex registers no contact force at all.

    sphere_n_samples: int = 300

    # Box half-extents [m] in body-local frame.  Default 33.5 mm
    # half-side = 67 mm full side, matching the sphere's bounding box
    # (radius 33.5 mm) so the two object kinds spawn at the same
    # ``settled_z_center`` (= 33.5 mm) and pads spawn at the same
    # ``grasp_axis_half`` (= 33.5 mm).  This makes sphere-vs-box a
    # head-to-head comparison with identical pad placement and
    # identical SQUEEZE/LIFT trajectories.
    #
    # The dome-vs-box geometric constraint
    # (``2*hx < 2*patch_radius + 6mm``) is still enforced by the
    # scene builder warning -- with R_pad = 10 mm, half-angle = 72°,
    # patch_radius ≈ 9.5 mm, so 2*hx = 67 mm comfortably exceeds the
    # 25 mm minimum.
    box_half_extents: tuple[float, float, float] = (0.0335, 0.0335, 0.0335)

    # Target-point pitch [m] on each box face.  Default 5 mm gives
    # ~169 samples per 67 mm face, ~338 total across the two approach
    # faces

    # Pitch must be ≤ the pad sphere's tangential-locality kernel
    # half-width ``3·r_pad`` (default r_pad ≈ 2.8 mm → kernel half-
    # width ≈ 8.5 mm) so every active pad sphere sees at least one
    # target sample inside its kernel disc.  Below ~2 mm pitch the
    # box scene gets very slow (target_count → 800+) without a
    # meaningful change in contact-patch resolution; above ~8 mm
    # pitch the locality kernel starts missing samples.
    box_face_pitch: float = 0.005

    # ── Bunny mesh object (kind="bunny") ──
    # Stanford-bunny demo object.  Loaded from an OBJ, scaled so its
    # vertical (Z) extent equals ``bunny_height``, and centred at its
    # bounding-box centre so the body origin sits at the geometric
    # centre (same convention as sphere/box).  The CSLC contact path
    # samples its surface into a point-set via Lloyd/CVT (see
    # :func:`cslc_main.grasp.objects.make_mesh_target`), exactly like the
    # pad contact face.
    bunny_obj: Path = _DEFAULT_BUNNY_OBJ
    # Upright height [m] the bunny is scaled to (Z extent).  100 mm by
    # default — comparable bulk to the 67 mm tennis ball / cube.
    bunny_height: float = 0.10
    # Lloyd surface-sample count for the CSLC target.  0 → auto: match
    # the sphere's surface point density (samples / m²) so the spacing
    # reads consistently across object kinds.
    bunny_n_samples: int = 0

    # Material density [kg/m³].  368 → tennis-ball mass at r=33.5 mm.
    density: float = 368.0

    # Spawn height [m] of the object's centre at t=0.  Chosen so a
    # free-falling sphere settles on the ground (z ≈ radius) before the
    # pads make contact.
    start_z: float = 0.05

    # Lateral spawn jitter [m] along Y, used to drive multi-seed
    # statistics for the C2 ke-sweep falsification (3 seeds at
    # {-1mm, 0, +1mm}).  Small enough to keep the grasp geometry
    # valid; large enough to break the perfect symmetry of the
    # default centred spawn and exercise asymmetric pad-vs-object
    # contact patches.  Default 0.0 preserves the legacy
    # centre-of-pad spawn.
    spawn_y_offset: float = 0.0

    def _bunny_trimesh(self):
        """The scaled, centred bunny mesh (cached).

        Lazy import of :mod:`cslc_main.grasp.objects` avoids a circular
        import at module load (``objects`` imports this module).
        """
        from . import objects
        return objects.make_bunny_trimesh(self)

    @property
    def mass(self) -> float:
        """Mass [kg], derived from density × volume for the active kind."""
        if self.kind == "sphere":
            return self.density * (4.0 / 3.0) * math.pi * self.radius**3
        if self.kind == "box":
            hx, hy, hz = self.box_half_extents
            return self.density * (2.0 * hx) * (2.0 * hy) * (2.0 * hz)
        if self.kind == "bunny":
            # abs(): a non-watertight OBJ can report signed volume.
            return self.density * abs(float(self._bunny_trimesh().volume))
        raise ValueError(f"Unknown object kind: {self.kind!r}")

    @property
    def grasp_axis_half(self) -> float:
        """Half-extent along the X (grasp) axis [m].

        Used by the scene builder to position pads laterally.  Pads
        spawn at ``x = +/-(grasp_axis_half + pad_thickness + approach_gap)``
        so the pad's inner face starts ``approach_gap`` clear of the
        object's side surface.
        """
        if self.kind == "sphere":
            return self.radius
        if self.kind == "box":
            return self.box_half_extents[0]
        if self.kind == "bunny":
            # Width at the grasp height (central band), NOT the global max
            # X-extent — see objects.bunny_grasp_half_width for why.
            from . import objects
            return objects.bunny_grasp_half_width(self)
        raise ValueError(f"Unknown object kind: {self.kind!r}")

    @property
    def settled_z_center(self) -> float:
        """Z position of the object's centre once it settles on the ground [m].

        Pads use this to position their vertical centre on the
        object's equator.  Sphere: ``radius`` (ball rolls to rest with
        centre at +radius).  Box: ``box_half_extents[2]`` (axis-aligned
        cube rests with bottom face on z=0).
        """
        if self.kind == "sphere":
            return self.radius
        if self.kind == "box":
            return self.box_half_extents[2]
        if self.kind == "bunny":
            # COM-centred mesh: the body origin (= centre of mass) rests
            # this far above the ground, i.e. the COM-to-base distance.
            return -float(self._bunny_trimesh().bounds[0][2])
        raise ValueError(f"Unknown object kind: {self.kind!r}")

    @property
    def weight(self) -> float:
        """Weight [N] under standard gravity."""
        return self.mass * 9.81


# ── Pad geometry + lattice sampling ──────────────────────────────────────


@dataclass
class PadParams:
    """Two-pad finger geometry and lattice sampling.

    The contact face of each pad is Lloyd/CVT-sampled (centroidal
    Voronoi tessellation, via ``point_cloud_utils.sample_mesh_lloyd``)
    to produce the CSLC lattice — exactly ``n_samples`` points,
    converging to hexagonal close-packing on flat patches and to
    as-uniform-as-possible spacing on curved patches.  The sampler's
    intrinsic spacing is ``mean nearest-neighbour distance``, used both
    as the lattice ``spacing`` (for kc calibration) and as the sphere
    radius (``spacing / 2``, matching Newton's box convention).
    """

    # "box" (default, flat pads), "dome" (curved pad from a pre-baked
    # OBJ asset), or "dome_param" (curved pad generated in-code from
    # ``dome_param_R_pad`` and ``dome_param_half_angle`` parameters)
    kind: str = "box"

    # Box pad half-extents [m] (only used when kind="box").  Defaults:
    # 16 mm thick × 40 mm wide × 40 mm tall.
    box_hx: float = 0.008
    box_hy: float = 0.020
    box_hz: float = 0.020

    # Dome OBJ path (only used when kind="dome").  The shipped asset is
    # fingertip-scale: ~20 mm wide × 10 mm thick with a curved face.
    dome_obj: Path = _DEFAULT_DOME_OBJ
    # When sampling the dome's curved face, only triangles whose face
    # normal has z-component above this threshold are sampled (i.e. the
    # outward-curving cap, not the back face).
    dome_nz_threshold: float = 0.3

    # Parametric-dome geometry (only used when kind="dome_param").
    dome_param_R_pad: float = 0.010
    dome_param_half_angle: float = 72.0 * math.pi / 180.0
    dome_param_back_height: float = 0.003
    # Tessellation -- enough to make the cap visually smooth and the
    # outward-normal direction estimate stable for CSLC sampling.
    dome_param_n_theta: int = 24
    dome_param_n_phi: int = 48

    # Z-coordinate [m] at which each pad BODY's centre sits at t=0.
    # ``None`` (default) → auto-derived in the scene builder:
    #   box  pads: max(object.radius, box_hz + 0.005)  — lift the pad
    #              enough that its bottom clears the ground by 5 mm and
    #              its centre is at least at the settled-object equator.
    #   dome pads: object.radius                       — the dome is
    #              short, so centring it on the object's equator works.
    pad_center_z: float | None = None

    # Material density [kg/m³] of the pad body (affects only joint inertia).
    density: float = 1000.0

    # Number of Lloyd/CVT samples drawn per pad contact face.  Determines
    # the resolution of the lattice; ``sample_mesh_lloyd`` returns
    # exactly this count.
    #
    # Note: 2026-05-27 we tried n_samples=50 thinking fewer pad spheres
    # would lower CSLC's shelf-dominated force on flat-pad geometries.
    # It went the OTHER way (F=110→134N on box/box at δ=1mm) because
    # fewer pad spheres → larger per-sphere locality kernel
    # (r_pad = spacing/2) → each captures MORE target samples within
    # its disc.  Total contact pair count dropped but per-contact
    # force grew faster.  The shelf is target-sample-driven, not pad-
    # sphere-driven; thinning the pad doesn't thin the shelf.
    # Reverted to n=100.
    n_samples: int = 100

    # k for the k-NN neighbour graph used to wire each lattice sphere to
    # its lateral-spring neighbours.  6 ≈ Delaunay valency in 2-D, which
    # the Poisson scatter on the contact face approximates.
    k_neighbors: int = 6

    # Initial pad-face-to-object-surface clearance [m] at t=0.  At dx=0
    # the pad inner face sits at this distance from the object surface;
    # APPROACH closes it.  Must match the APPROACH travel distance
    # (``timing.approach_speed * timing.approach_duration``) so the
    # face just touches the object at end of APPROACH; SQUEEZE then
    # drives the calibrated face_pen.
    approach_gap: float = 0.02

    # Random seed — unused by ``sample_mesh_lloyd`` (which is
    # deterministic given (v, f, n)) but retained for any future
    # stochastic sampling paths and for backward compatibility with
    # saved configs.
    seed: int = 0


# ── Material (shared baseline for Hunt-Crossley + Coulomb) ───────────────


@dataclass
class MaterialParams:
    """Hunt-Crossley + Coulomb material parameters shared across models.

    These set the elastic response (``ke``), dissipative response
    (``kd``), tangential stiffness (``kf``), and Coulomb friction
    coefficient (``mu``) at the SHAPE level.

    ke is split by role (two physical knobs, one per body of the
    contact pair):

    * ``ke_pad_physical`` -- the CSLC pad's bulk Young's-modulus-
      equivalent.  Feeds ``calibrate_kc(ke_bulk=...)`` and sets the
      pad's per-sphere contact stiffness ``kc``.
    * ``ke_target_physical`` -- the OBJECT's effective contact stiffness.
      Enters the series-spring composition ``1/kc_eff = 1/kc + 1/ke_target``
      in ``calibrate_kc`` (Newton III), so increasing ``ke_target_physical``
      makes the object behave more rigidly in contact.
    * ``kh`` -- hydroelastic physical-compliance modulus [Pa/m].  Used
      only under ``contact_model="hydro"``.

    Apples-to-apples comparison with hydroelastic: hold
    ``ke_target_physical`` fixed and sweep ``ke_pad_physical`` / ``kh``
    as the parallel "physical material" axis.

    .. note:: MuJoCo regularization coupling

       Under CSLC, ``ke_target_physical`` co-determines MuJoCo's
       constraint regularization timeconst because MuJoCo derives the
       timeconst from the emitted per-contact stiffness (one stiffness
       slot per contact in MuJoCo's API).  Truly decoupling physical
       compliance from numerical regularization would require wiring a
       separate solref override into the contact emission, which is
       deferred to the physics-cleanup phase.  Today, tuning
       ``ke_target_physical`` changes BOTH force-per-penetration AND
       solver stability characteristics.

    """

    # ── Fields ──

    # CSLC pad physical bulk-modulus equivalent [N/m].  Drives
    # ``calibrate_kc(ke_bulk=...)`` on the pad lattice.  Under
    # ``contact_model="hydro"`` this field is unused (hydro reads
    # ``kh`` for physical compliance instead).
    ke_pad_physical: float = 5.0e4

    # Object's physical contact stiffness [N/m].  Enters the
    # series-spring composition ``1/kc_eff = 1/kc + 1/ke_target`` in
    # ``calibrate_kc``; also flows to MuJoCo as the rigid-contact
    # stiffness (which is intrinsically coupled to the regularisation
    # timeconst by MuJoCo's API -- see class docstring's MuJoCo
    # regularization coupling note).
    ke_target_physical: float = 5.0e4

    # Hydroelastic physical-compliance modulus [Pa/m] split by role,
    # parallel to the ``ke_pad_physical`` / ``ke_target_physical`` split
    # above.  Used ONLY when ``contact_model="hydro"``; both fields are
    # silently ignored under CSLC or point.  Asymmetric setups (soft pad
    # / rigid object) set the two independently; the legacy single-knob
    # API (``material.kh = X``) is preserved as a property that writes
    # both at once (see getter/setter below).
    kh_pad: float = 5.0e9
    kh_object: float = 5.0e9

    # Hunt-Crossley damping coefficient [N·s/m].  Used by ``point``
    # contact.  Under ``cslc`` the emission kernel writes
    # ``rigid_contact_damping = 0`` to use MuJoCo's stiffness-derived
    # timeconst branch; under ``hydro`` MuJoCo uses its own kd.  Tuning
    # this knob has NO effect under cslc.
    kd: float = 5.0e2

    # Tangential / friction-spring stiffness [N/m] for the regularised
    # Coulomb model.  Used ONLY by ``point`` contact (cslc handles
    # tangential stiffness via :attr:`CSLCParams.k_stick`; hydro doesn't
    # use a tangential spring).
    kf: float = 100.0

    # Coulomb friction coefficient [-].  Single source of truth for
    # friction across all contact models.  ``point`` and ``hydro`` use
    # it directly on the geom pair; ``cslc`` reads it for the lattice's
    # stick-slip block AND lets MuJoCo apply it on emitted contacts.
    mu: float = 0.5

    # Proximity gap [m] for narrow-phase early-out.  Used by ``point``
    # and ``hydro`` collision narrow-phase.  ``cslc`` bypasses
    # narrow-phase (it samples target surfaces directly), so this knob
    # has NO effect under cslc.
    gap: float = 0.002

    # ── Legacy property (pre-split back-compat) ──

    @property
    def ke(self) -> float:
        """Legacy alias for ``ke_pad_physical`` (read-only getter).

        Pre-split code reads ``material.ke``; the split keeps this
        as a property returning the physical knob so no external
        consumer breaks.  New code should use ``ke_pad_physical``
        and ``ke_target_physical`` explicitly.
        """
        return self.ke_pad_physical

    @ke.setter
    def ke(self, value: float) -> None:
        """Legacy setter -- sets BOTH ke fields to ``value``.

        Pre-split semantics: ``material.ke = X`` set both pad and
        target stiffness to the same number.  Preserved here so
        existing scripts and the ``--material-ke`` CLI alias keep
        working bit-identically.  Explicit per-role assignment
        should set ``ke_pad_physical`` and ``ke_target_physical``
        directly.
        """
        self.ke_pad_physical = value
        self.ke_target_physical = value

    @property
    def kh(self) -> float:
        """Legacy alias for ``kh_pad`` (read-only getter).

        Pre-split code reads ``material.kh``; the split keeps this as
        a property returning the pad knob so existing call sites
        (e.g. the ``--kh`` CLI flag, the ``HydroParams`` docstring
        examples) keep working.  Asymmetric setups should set
        ``kh_pad`` and ``kh_object`` explicitly.
        """
        return self.kh_pad

    @kh.setter
    def kh(self, value: float) -> None:
        """Legacy setter -- sets BOTH kh fields to ``value``.

        Mirrors the ``ke`` setter above.  Use for symmetric hydro
        runs (both bodies share the same hydroelastic modulus); use
        ``kh_pad`` / ``kh_object`` directly for asymmetric soft-vs-
        rigid contact studies.
        """
        self.kh_pad = value
        self.kh_object = value


# ── CSLC compliant-skin tuning ───────────────────────────────────────────


@dataclass
class CSLCParams:
    """Knobs for the CSLC (Compliant Sphere Lattice Contact) model.

    Contact force law (Hertz-like ``raw^(3/2)``):
        F_per_contact = kc · A_j · w_tangent · α · gate · raw^1.5
    Per-MuJoCo-contact LOCAL stiffness is the derivative:
        dF/d(raw) = 1.5 · kc · A_j · w_tangent · α · gate · √raw
    which vanishes at first touch (raw → 0) and grows as √raw at depth
    — same scaling as Hertz's local stiffness ``2·E*·√(R·δ)``.  This
    eliminates the constant-stiffness impulse-on-engagement that broke
    the previous linear-law model across most material stiffnesses.

    Sphere-on-flat integration: ``∫ raw^1.5 dA ≈ (4πR/5)·δ^2.5``, so the
    aggregate ``F_total ∝ δ^2.5`` — one power stiffer than Hertz's
    ``δ^1.5`` because the contact patch grows with δ.  ``kc`` is
    therefore not literally a Young's modulus; the kc-to-E mapping
    picks a representative operating depth δ_op and matches local
    stiffness at that depth (see field comment below).

    Units: ``kc`` has units N / (m² · m^1.5) = Pa · m^(−1/2).

    Tune intentionally:
      - ``kc_per_volume`` -- primary contact stiffness, derive from
        material via ``kc ≈ (5/3π) · E* / (√R · δ_op)`` at a chosen
        operating depth.
      - ``ka`` has a threshold (~10·weight, scene-dependent); above it
        the value is irrelevant.  Below it, lattice goes liquid and
        the ball squirts sideways.
      - ``kl``, ``k_stick``, ``ka_tangent_ratio`` don't move lift
        outcome on the standard tennis-ball test; reserve for shear-
        dominated or Poisson-bulging studies.
      - ``smoothing_eps``, ``n_iter``, ``alpha`` are numerical and
        calibrated together; change one, you may need to change another.
    """

    # Per-volume contact stiffness [Pa · m^(−1/2)].  PRIMARY contact knob.
    # Hertz-like force: F = kc · A_j · w_tangent · α · gate · raw^1.5.
    #
    # Rebalanced 2026-05-27 from 3e10 → 1e10 so the default-knob CSLC
    # force levels at δ=1mm sit in the same regime as hydro across the
    # geometry matrix (cross-model comparison study).  Previous 3e10
    # produced 117 N on box-pad × box-object vs hydro's 33 N (3.5×
    # over) and 7.5 N on dome×sphere vs hydro's 11.7 N (under). At
    # 1e10 the per-cell ratio CSLC:hydro tightens to ~1.2× across the
    # matrix.  Production grasp scripts that hard-coded grip
    # behaviour around the old default may need a CLI override
    # (``--cslc-kc 3e10``) to recover the old force level.
    kc_per_volume: float = 1.0e10

    # Anchor stiffness [N/m] — pulls each lattice sphere back toward
    # its rest position relative to the pad body.  Threshold knob:
    #     3.5e3 → lattice liquefies, ball slides 230 mm sideways
    #   ≥3.5e4 → identical outcome (sweep: 35k, 350k both held cleanly)
    # Set to comfortably exceed the threshold for your scene.  Raising
    # past ~10× the object weight buys nothing.
    ka: float = 35_000.0

    # Lateral / distance-preservation stiffness [N/m] connecting each
    # sphere to its k-NN neighbours.  Skin elasticity knob:
    #         0 → spheres act alone, slip rises to ~6.7 mm
    # *    1000 → held cleanly, slip ~2.5 mm (BEST for tennis ball)
    #     20000 → lattice over-coupled, can't conform to curvature,
    #             lift truncated (final_z 49 vs 60 mm)
    kl: float = 1_000.0

    # Per-step Jacobi refinement iterations.  Each is one
    # ``wp.launch(jacobi_step)``; cost scales linearly.
    #
    # Calibration: with the Phase 5a alignment-aware Jacobi diagonal
    # and Phase 6 w_tangent hard-cull both active, the dome's small
    # contact patch (r_pad ≈ 1.4 mm, ~10 mm cap on a 67 mm ball)
    # needs ~40 iterations to converge the lattice δ before MuJoCo
    # emits constraints.  The pre-Phase-6 box default 20 left the
    # dome lattice non-converged enough that the emitted contact
    # pattern flickered per step and the held object oscillated.
    #
    #     3 → under-converged on every pad kind
    #     15 → converged on flat box (legacy default)
    # *   40 → converged on both box and dome (production default)
    #    200 → no measurable improvement over 40 (verified via
    #          step-0 max|δ| sweep on dome SQUEEZE end)
    #
    # Coupled with ``alpha``: low alpha needs more iters to converge.
    # Box-only scenes can drop back to 20 to save ~50% step time;
    # the dome path needs 40 for stable HOLD-phase behavior.
    n_iter: int = 40

    # Damping factor in the Jacobi step.  Coupled with ``n_iter``.
    #   0.2 → too damped, lattice can't reach equilibrium in n_iter=15
    # * 0.6 → converges cleanly (BEST)
    #   0.9 → aggressive but stable on this scene
    alpha: float = 0.6

    # Anisotropy of the anchor: tangent_axis_ka = ka × ratio.
    # No measurable effect on the symmetric squeeze-and-lift grasp
    # (sweep 0.333 / 1.0 / 3.0 all within noise).  Matters for shear-
    # dominated motion; 1/3 (≈0.333) matches incompressible-flesh
    # Poisson ν → 0.5.
    ka_tangent_ratio: float = 1.0

    # Differentiability width [m] for the kernel smooth-step gates.
    #   1e-4 → sharper gates, effective contact stiffens, lift truncated
    # * 5e-4 → calibrated sweet spot
    #   2e-3 → tail extends past surface, back-side samples leak in,
    #          ball slips out (12.5 mm slip)
    # Bigger eps breaks worse than smaller.  Couples with ``kc_per_volume``
    # (sharper gates ≈ effectively stiffer kc).
    smoothing_eps: float = 5.0e-4

    # Stick-slip friction stiffness [N/m] on the tangential δ_t.  For
    # the symmetric squeeze-and-lift test this knob is a NO-OP: 0,
    # 25000, and 250000 all give identical lift (within 0.2 mm) and
    # identical slip (within 0.2 mm).  Grip on this scene is dominated
    # by normal force + MuJoCo-level Coulomb friction; the lattice's
    # tangential δ stays small enough that the stick-slip term is
    # negligible.
    #
    # Become relevant only when lattice slip > a few mm (shear-heavy
    # motions, low-friction objects, asymmetric grasps).  Friction
    # COEFFICIENT lives on :attr:`MaterialParams.mu` -- single source
    # of truth across both the stick-slip block and MuJoCo's cone.
    k_stick: float = 25_000.0

    # If True, build the dense A_inv (= (K + kc·I)^-1) for the
    # closed-form linear warm-start before the Jacobi refinement.
    build_A_inv: bool = True

    # Lattice-level velocity damping coefficient [N·s/m].
    # Adds an explicit ``-c_lattice · (δ - δ_prev_step) / dt`` term to
    # each pad sphere's force balance in jacobi_step (alongside anchor,
    # lateral, contact, friction).  Default 0.0 disables the term
    # entirely (multiply by zero); pass ``--cslc-c-lat <value>`` to
    # enable.  Dissipates energy from oscillatory lattice modes BEFORE
    # the contact emission, complementary to MuJoCo-side damping
    # ``dc`` which acts AFTER emission.
    #
    # Easy-removal: every B3 site is tagged with ``# B3`` -- grep for
    # that comment and delete those blocks to revert.  The default
    # value 0.0 makes the change a no-op when disabled.
    c_lattice: float = 0.0

    # Per-contact damping coefficient written to MuJoCo's
    # ``rigid_contact_damping`` slot.  MuJoCo's parametrisation:
    #   dc = 0      → timeconst ≈ √(imp/ke) ≈ 30 ms (stiffness-derived)
    #   dc > 0      → timeconst = 2 / dc  (explicit)
    #
    # **TRAP**: there is a "dead zone" 0 < dc < ~67 where the explicit
    # branch gives a LARGER timeconst than the default (e.g. dc = 10 →
    # timeconst = 200 ms, dc = 50 → 40 ms).  In this region the
    # contact becomes SOFTER than dc=0 and the dome sim diverges
    # catastrophically (ball flies away at 100+ m, max_z literally
    # hundreds of metres).  Never use dc values below ~67.
    #
    # Sweep results on the production dome scene (lift+hold, 6.5 s):
    #     dc =    0  → tail σz =  1.16 mm, slip =  6.1 mm (oscillates)
    #     dc =   10  → DIVERGES (timeconst 200 ms is softer than default)
    #     dc =   50  → DIVERGES on some seeds
    #     dc =  200  → tail σz =  0.12 mm, slip =  3.6 mm (3× better)
    # *   dc = 1000  → tail σz = 0.011 mm, slip = 0.18 mm (60× better,
    #                  final_z lands within 0.1 mm of pad target)
    #     dc = 5000  → tail σz =  0.14 mm, slip =  7.7 mm (over-damped,
    #                  friction starts creeping during HOLD)
    #
    # The friction-timeconst coupling (cslc_kernels.py:1170) warns
    # that high dc softens Coulomb friction — visible at dc=5000.
    # ``dc = 1000`` (timeconst = 2 ms) is the production-default
    # recommended value: stiff contacts that don't bounce, while the
    # friction is still tight enough that the held object doesn't
    # creep.  Default kept at 0 for legacy-compat; pass
    # ``--cslc-dc 1000`` from the CLI to enable.
    dc: float = 1000.0


# ── Hydroelastic (PFC) parameters ────────────────────────────────────────


@dataclass
class HydroParams:
    """Hydroelastic contact-model SDF parameters.

    Only used when ``GraspConfig.contact_model == "hydro"``.

    ``kh`` (the hydroelastic physical-compliance modulus) moved to
    :class:`MaterialParams.kh` as part of the C2 split so that
    CSLC ``ke_pad_physical`` and hydro ``kh`` live on the same
    config object as the symmetric "physical material" axis.  This
    class now holds only the SDF resolution.
    """

    # Voxel grid resolution for the analytic SDF generated by Newton.
    sdf_resolution: int = 64


# ── Phase timing ─────────────────────────────────────────────────────────


@dataclass
class TimingParams:
    """Per-phase durations and pad speeds.

    Phase sequence is APPROACH → SQUEEZE → LIFT → HOLD.  Speeds are
    chosen to be slow enough that the PD drive's response time
    (≈ 1 ms) is well below the trajectory's time scale.
    """

    # Integration timestep [s].  500 Hz matches MuJoCo's typical sweet
    # spot for CSLC's ~170 simultaneous constraints.
    dt: float = 1.0 / 500.0

    # APPROACH: each pad moves inward at this speed [m/s] for this
    # duration [s].  Total inward travel = approach_speed × duration.
    approach_speed: float = 20.0e-3 / 1.5  # 13.33 mm/s → 20 mm in 1.5 s
    approach_duration: float = 1.5

    # SQUEEZE: each pad continues inward at this slower speed [m/s] to
    # build a controlled penetration into the object.  Trajectory commands
    # 1 mm of dx beyond the closed approach_gap.

    squeeze_speed: float = 1.0e-3 / 0.5  # 2 mm/s → 1 mm in 0.5 s
    squeeze_duration: float = 0.5

    # LIFT: each pad rises at this speed [m/s].
    lift_speed: float = 0.015
    lift_duration: float = 1.5

    # Smooths the velocity profile at BOTH ends of LIFT.  Without this
    # the 0 → lift_speed step at SQUEEZE→LIFT (and lift_speed → 0 at
    # LIFT→HOLD) drives the PD into ~impulsive responses that fling the
    # held object.  C¹-smooth ramps of this duration eliminate the
    # kicks.  See ``trajectory._lift_dz``.
    lift_ramp_duration: float = 0.25

    # HOLD: pads stationary at the final lifted position for this
    # duration [s].  Long enough to read steady-state slip/creep of
    # the object under gravity once the lift transient has decayed —
    # 3 s gives ~1500 steps at 500 Hz, plenty to fit a linear creep
    # rate and to expose any oscillatory grip loss.
    hold_duration: float = 3.0


# ── Solver ───────────────────────────────────────────────────────────────


@dataclass
class SolverParams:
    """Physics-solver selection and tuning.

    The factory in :mod:`cslc_main.grasp.solvers` auto-sizes MuJoCo's
    contact-slot budget (``njmax``/``nconmax``) to fit CSLC's
    one-slot-per-surface-sphere-per-pair allocation; the user shouldn't
    need to touch ``extra_ncon`` unless adding many more pads or pairs.
    """

    # "mujoco" or "semi" (semi-implicit fallback for diff'ability).
    name: str = "mujoco"

    # MuJoCo CG iterations.  ``None`` = auto: 100 with CSLC, 20 otherwise.
    iterations: int | None = None

    # MuJoCo line-search iterations per CG step.
    ls_iterations: int = 10

    # MuJoCo cone formulation: "elliptic" or "pyramidal".
    cone: str = "elliptic"

    # MuJoCo integrator: "implicitfast", "implicit", "Euler", "RK4".
    integrator: str = "implicitfast"

    # Underlying constraint solver: "cg" (default), "pgs", "newton".
    solver: str = "cg"

    # Extra contact-slot headroom above CSLC's reserved count.
    extra_ncon: int = 5_000


# ── Joint drive (pad PD position tracking) ──────────────────────────────


@dataclass
class DriveParams:
    """Stiff PD on each pad prismatic joint.

    Both joints (X for APPROACH/SQUEEZE, Z for LIFT) share the same
    gains.  The drive's natural time-constant is √(m/ke) ≈ 1 ms; both
    the squeeze and lift trajectories are much slower than that, so the
    pads follow their commanded positions faithfully.
    """

    # Joint position-tracking stiffness [N/m].
    ke: float = 5.0e4

    # Joint velocity-tracking damping [N·s/m].
    kd: float = 1.0e3


# ── Logging / output ────────────────────────────────────────────────────


@dataclass
class LoggingParams:
    """Per-run output configuration.

    Each run writes to ``output_root / <run_dir_name>/``.  ``run_label``
    defaults to a human-readable triple
    ``{pad_kind}_{object_kind}_{contact_model}`` if not set
    explicitly.  Set ``use_timestamp=False`` to overwrite the same
    directory on each re-run — convenient for fast tuning iteration.
    """

    # Root directory for all grasp-test outputs.
    output_root: Path = _DEFAULT_OUTPUT_ROOT

    # If None, auto-generate from
    # ``{pad_kind}_{object_kind}_{contact_model}``.
    run_label: str | None = None

    # If True, prefix the run directory name with a timestamp so every
    # run gets its own folder.  False to overwrite the previous run.
    use_timestamp: bool = True

    # Skip CSV rows for steps that aren't a multiple of this.  Set to 1
    # to log every step.
    log_every: int = 1

    # If True, save ``pad_lattice.png`` (3-D scatter + normal quivers)
    # to the run directory before the sim starts.
    save_lattice_preview: bool = True

    # If True, render post-sim plots (sphere trajectory, contact count,
    # CSLC δ statistics) from the logged CSVs after the sim ends.
    save_postsim_plots: bool = True


# ── Top-level config ─────────────────────────────────────────────────────


@dataclass
class GraspConfig:
    """The single configuration object for a grasp test run.

    All other modules in ``cslc_main.grasp`` accept this object (or a
    relevant sub-dataclass) and read knobs from it — they never define
    their own defaults or read environment variables.
    """

    # "cslc" (default, compliant-skin lattice), "point" (Hunt-Crossley
    # closest-point), or "hydro" (hydroelastic / PFC).  Point and hydro
    # paths are wired but uncalibrated for the new mesh-pad geometry —
    # CSLC is the only mode that is paper-grade in this PR.
    contact_model: str = "cslc"

    # World gravity vector [m/s²].
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # ── Sub-configs ──
    object: ObjectParams = field(default_factory=ObjectParams)
    pad: PadParams = field(default_factory=PadParams)
    material: MaterialParams = field(default_factory=MaterialParams)
    cslc: CSLCParams = field(default_factory=CSLCParams)
    hydro: HydroParams = field(default_factory=HydroParams)
    timing: TimingParams = field(default_factory=TimingParams)
    solver: SolverParams = field(default_factory=SolverParams)
    drive: DriveParams = field(default_factory=DriveParams)
    logging: LoggingParams = field(default_factory=LoggingParams)

    # ── Derived helpers ──

    @property
    def phase_sequence(self) -> list[tuple[str, int]]:
        """List of ``(phase_name, n_steps)`` tuples for APPROACH →
        SQUEEZE → LIFT → HOLD."""
        t = self.timing
        return [
            ("APPROACH", int(t.approach_duration / t.dt)),
            ("SQUEEZE", int(t.squeeze_duration / t.dt)),
            ("LIFT", int(t.lift_duration / t.dt)),
            ("HOLD", int(t.hold_duration / t.dt)),
        ]

    @property
    def total_steps(self) -> int:
        return sum(n for _, n in self.phase_sequence)

    def phase_of(self, step: int) -> tuple[str, int]:
        """Map a global step index to (phase_name, local_step_within_phase)."""
        s = step
        for name, n in self.phase_sequence:
            if s < n:
                return name, s
            s -= n
        # Past the end of HOLD: clamp to the last phase.
        last_name = self.phase_sequence[-1][0]
        return last_name, 0

    def run_dir_name(self) -> str:
        """Compute the run directory name from the logging config.

        The contact model is *always* appended to the final directory
        name so two runs that differ only in ``--contact-model`` end up
        in distinct directories under ``--no-timestamp``.  Users can
        keep their scene label clean (``--run-label dome_curved_flat``)
        and not worry about manually disambiguating CSLC vs hydro vs
        point runs.
        """
        lp = self.logging
        scene_label = lp.run_label or f"{self.pad.kind}_{self.object.kind}"
        label = f"{scene_label}_{self.contact_model}"
        if lp.use_timestamp:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return f"{ts}_{label}"
        return label

    def run_dir(self) -> Path:
        """Resolve the absolute path of this run's output directory."""
        return Path(self.logging.output_root) / self.run_dir_name()

    def to_json_safe_dict(self) -> dict:
        """Return a JSON-serialisable dict snapshot of the config.

        ``Path`` objects become strings; everything else passes through
        ``dataclasses.asdict``.
        """

        def _coerce(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_coerce(v) for v in obj]
            return obj

        return _coerce(asdict(self))

    def freeze_to_json(self, path: Path) -> None:
        """Write the config snapshot to ``path`` as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json_safe_dict(), f, indent=2)

    def dump_summary(self) -> str:
        """Compact one-screen summary, suitable for console echo."""
        o = self.object
        p = self.pad
        m = self.material
        c = self.cslc
        t = self.timing
        lines = [
            f"GraspConfig  contact={self.contact_model}  "
            f"solver={self.solver.name}",
            f"  object   : {o.kind}  r={o.radius * 1e3:.1f} mm  "
            f"m={o.mass * 1e3:.1f} g  W={o.weight:.3f} N",
            f"  pad      : {p.kind}  n_samples={p.n_samples}  "
            f"k_neighbors={p.k_neighbors}  approach_gap={p.approach_gap * 1e3:.0f} mm",
            f"  material : ke={m.ke:.0f}  kd={m.kd:.0f}  mu={m.mu:.2f}",
            f"  cslc     : kc={c.kc_per_volume:.2e}  ka={c.ka:.0f}  "
            f"kl={c.kl:.0f}  k_stick={c.k_stick:.0f}",
            f"  timing   : dt={t.dt * 1e3:.2f} ms  total_steps={self.total_steps}  "
            f"phases={'+'.join(name for name, _ in self.phase_sequence)}",
            f"  run_dir  : {self.run_dir()}",
        ]
        return "\n".join(lines)
