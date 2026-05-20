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

    # "sphere" is the only kind implemented for this PR; the field is
    # exposed so future extensions (cube, cylinder) drop in without
    # changing the surrounding code.
    kind: str = "sphere"

    # Sphere radius [m].  Tennis ball: 0.0335.
    radius: float = 0.0335

    # Material density [kg/m³].  368 → tennis-ball mass at r=33.5 mm.
    density: float = 368.0

    # Spawn height [m] of the object's centre at t=0.  Chosen so a
    # free-falling sphere settles on the ground (z ≈ radius) before the
    # pads make contact.
    start_z: float = 0.10

    @property
    def mass(self) -> float:
        """Mass [kg], derived from density × volume for the sphere kind."""
        if self.kind == "sphere":
            return self.density * (4.0 / 3.0) * math.pi * self.radius**3
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

    # "box" (default, simple flat pads) or "dome" (curved pad from OBJ).
    kind: str = "box"

    # Box pad half-extents [m] (only used when kind="box").  Defaults:
    # 16 mm thick × 40 mm wide × 80 mm tall.  The 80 mm height is
    # taller than the default tennis-ball diameter (67 mm) so the
    # contact face fully spans the sphere vertically — without this,
    # the pads contact only the upper or lower hemisphere and squeeze
    # the sphere out of the grip.  Make ``box_hz`` ≥ object.radius
    # for any new held object.
    box_hx: float = 0.008
    box_hy: float = 0.020
    box_hz: float = 0.040

    # Dome OBJ path (only used when kind="dome").  The shipped asset is
    # fingertip-scale: ~20 mm wide × 10 mm thick with a curved face.
    dome_obj: Path = _DEFAULT_DOME_OBJ
    # When sampling the dome's curved face, only triangles whose face
    # normal has z-component above this threshold are sampled (i.e. the
    # outward-curving cap, not the back face).
    dome_nz_threshold: float = 0.3

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
    n_samples: int = 150

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
    coefficient (``mu``) at the SHAPE level.  CSLC layers its own
    spring network on top via :class:`CSLCParams`; ``ke`` here doubles
    as the "bulk" target stiffness for CSLC's kc calibration.
    """

    # Hunt-Crossley elastic stiffness [N/m] of the pad-vs-object contact.
    ke: float = 5.0e4

    # Hunt-Crossley damping coefficient [N·s/m].
    kd: float = 5.0e2

    # Tangential / friction-spring stiffness [N/m] for the regularised
    # Coulomb model.  Only matters under point-contact mode.
    kf: float = 100.0

    # Coulomb friction coefficient [-].  May be overridden per-shape for
    # CSLC (see :attr:`CSLCParams.mu_friction`).
    mu: float = 0.5

    # Proximity gap [m] for narrow-phase early-out.  Should comfortably
    # exceed the maximum penetration expected during SQUEEZE.
    gap: float = 0.002


# ── CSLC compliant-skin tuning ───────────────────────────────────────────


@dataclass
class CSLCParams:
    """Knobs for the CSLC (Compliant Sphere Lattice Contact) model.

    Defaults are the may_18 production values that ship 4-6 mm XY slip
    on the dome pad_lift test (see ``cslc_mujoco/may_18_summary.md``
    §4).  ``contact_fraction`` is the only knob that needs scene-
    dependent tuning — it scales the kc recalibration to the empirical
    active-sphere fraction at the operating face_pen.
    """

    # Anchor stiffness [N/m] — pulls each lattice sphere back toward its
    # rest position relative to the pad body.  Above the threshold
    # ke_bulk/(N − ke_bulk/ke_target) ≈ 16667 to admit a positive kc; see
    # ``cslc_data.calibrate_kc``.
    ka: float = 25_000.0

    # Lateral / distance-preservation stiffness [N/m] connecting each
    # sphere to its k-NN neighbours.  may_18 default is 5000 (Micro-3
    # production value); raise to ~25000 to surface geometric Poisson
    # bulging at the cost of grip strength on dome pads.
    kl: float = 5_000.0

    # Contact damping coefficient [-].
    dc: float = 2.0

    # Per-step Jacobi refinement iterations on top of the closed-form
    # warm-start.  20 is sufficient for ~mm-scale δ; raise for stiffer
    # geometries.
    n_iter: int = 20

    # Damping factor in the Jacobi step.  0.6 balances stability and
    # convergence rate.
    alpha: float = 0.6

    # Fraction of surface spheres that should be considered "active"
    # under the operating penetration.  Drives the kc recalibration so
    # the per-pad aggregate stiffness equals ke_bulk.  0.025 matches
    # the working ``cslc_mujoco/pad_lift_test`` calibration at
    # face_pen=1 mm on dome pads — bigger pads (box, large contact
    # patch) engage more spheres and should override upward via
    # ``--cslc-contact-fraction`` (~0.15 is a reasonable starting
    # point for a flat box pad on a tennis-ball-sized sphere).
    contact_fraction: float = 0.025

    # Anisotropy of the anchor: tangent_axis_ka = ka × ratio.  1.0 =
    # isotropic; 1/3 (≈0.333) matches incompressible-flesh Poisson
    # ν → 0.5.
    ka_tangent_ratio: float = 1.0

    # Differentiability width [m] for the kernel smooth-step gates.
    # Tighter ε ≈ stiffer contact, less differentiability.  5e-4 is
    # the production default from ``cslc_mujoco/pad_lift_test``,
    # empirically calibrated on dome pads: tighter values (1e-4)
    # produce bimodal slip behaviour ("always cascades" or
    # "sometimes 7 mm, sometimes 360 mm").
    smoothing_eps: float = 5.0e-4

    # Stick-slip friction stiffness [N/m] on the tangential δ_t.  Set
    # to 0 to disable friction entirely.
    k_stick: float = 25_000.0

    # Tangential friction coefficient [-] in the stick-slip model.
    mu_friction: float = 0.3

    # If True, build the dense A_inv (= (K + kc·I)^-1) for the
    # closed-form linear warm-start before the Jacobi refinement.
    # Required for the may_18 hybrid solver path.
    build_A_inv: bool = True


# ── Hydroelastic (PFC) parameters ────────────────────────────────────────


@dataclass
class HydroParams:
    """Hydroelastic contact-model parameters.

    Only used when ``GraspConfig.contact_model == "hydro"``.  The default
    ``kh`` is fair-calibrated against ``MaterialParams.ke`` at the
    expected contact patch area (see ``cslc_mujoco/summary.md`` §2):
        kh_eff · A_patch(1 mm pen) = ke
    """

    # Hydroelastic modulus [Pa].
    kh: float = 5.3e8

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
    # build a controlled penetration into the object.
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
        """Compute the run directory name from the logging config."""
        lp = self.logging
        label = lp.run_label or (
            f"{self.pad.kind}_{self.object.kind}_{self.contact_model}"
        )
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
            f"  cslc     : ka={c.ka:.0f}  kl={c.kl:.0f}  "
            f"cf={c.contact_fraction:.3f}  k_stick={c.k_stick:.0f}",
            f"  timing   : dt={t.dt * 1e3:.2f} ms  total_steps={self.total_steps}  "
            f"phases={'+'.join(name for name, _ in self.phase_sequence)}",
            f"  run_dir  : {self.run_dir()}",
        ]
        return "\n".join(lines)
