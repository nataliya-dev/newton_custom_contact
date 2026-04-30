#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC squeeze diagnostic — compares point contact vs CSLC hold quality.

Two kinematic pads squeeze a sphere, then hold it under gravity.
Focus: WHY does the sphere slip, and how fast?

Usage
-----
  # Both models (point + cslc), MuJoCo solver, 1000 steps (default):
  uv run cslc_v1/squeeze_test.py --solver mujoco --mode squeeze --steps 1000

  # Interactive viewer (CSLC model, renders in real time):
  uv run cslc_v1/squeeze_test.py --contact-model cslc --solver mujoco
  # NOTE: --mode squeeze is headless. Pass --viewer to auto-switch to viewer mode.

  # Calibration diagnostic (no simulation — just geometry + stiffness):
  uv run cslc_v1/squeeze_test.py --mode calibrate

  # Step-by-step contact/body dump (N steps):
  uv run cslc_v1/squeeze_test.py --mode inspect --steps 10

Modes
-----
  viewer     — real-time rendering via newton.examples (uses --contact-model)
  squeeze    — runs point + cslc back-to-back and prints per-step diagnostics
  calibrate  — prints CSLC stiffness/calibration without simulating
  inspect    — dumps every contact and body state for N steps

Key diagnostics (printed every 50 steps in SQUEEZE/HOLD phases)
  z={m}  drop={mm}  rate={mm/s}   — sphere position, cumulative drop, FD velocity
  Fz≈{N}(W={N})                   — FD-inferred net Fz vs weight
                                    (NOTE: body_qd stale under MuJoCo GPU;
                                     velocity derived from position history)
  contacts={n}  δ={mm}  pen={mm}  — active contacts, max lattice delta, max pen
  [MJW_DIST ...]                   — MuJoCo dist summary at selected steps
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
import newton.examples

from cslc_v1.common import (
    CSLC_FLAG,
    GEO_NAMES,
    HAS_MUJOCO,
    _log,
    _section,
    apply_external_wrench,
    count_active_contacts,
    get_cslc_lattice_viz_data,
    make_solver,
    read_cslc_state,
    recalibrate_cslc_kc_per_pad,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Local formatting helpers (squeeze-test-specific section breaks)
# ═══════════════════════════════════════════════════════════════════════════

_HSEP = "━" * 72
_DSEP = "═" * 72


def _sub(title):
    print(f"\n  {'─' * 72}\n  {title}\n  {'─' * 72}")


def _parse_xyz(s: str, label: str) -> tuple[float, float, float]:
    """Parse 'x,y,z' into a 3-tuple, or raise a helpful error."""
    try:
        parts = [float(t.strip()) for t in s.split(",")]
    except ValueError as e:
        raise ValueError(f"--{label} expects 'x,y,z' floats, got {s!r}") from e
    if len(parts) != 3:
        raise ValueError(f"--{label} expects exactly 3 values, got {len(parts)} ({s!r})")
    return parts[0], parts[1], parts[2]


# ═══════════════════════════════════════════════════════════════════════════
#  Scene parameters
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SceneParams:
    """All tunable knobs for the squeeze scene.

    Geometry
    ────────
    Two kinematic box pads sit on either side of a dynamic sphere.
    The inner faces start closer than the sphere diameter so there is
    initial penetration.  During the squeeze phase each pad moves inward,
    increasing grip.  During the hold phase pads stop and gravity pulls down.

    Material
    ────────
    ke, kd, mu are shared by point contact and CSLC for a fair comparison.
    CSLC additionally needs: cslc_spacing, ka, kl, dc, n_iter, alpha.
    """

    # ── held object: sphere or book ──
    #
    # `object_kind` selects the geometry the pads squeeze.  With "sphere"
    # the test runs the legacy sphere-vs-pad scenario.  With "book" the
    # held body is a flat rectangular box (matching the paper's
    # rotational-stability experiment) — this exercises the new
    # CSLC-vs-BOX kernel chain in cslc_kernels.py.
    object_kind: str = "sphere"

    # ── sphere ──
    sphere_radius: float = 0.03
    sphere_density: float = 4421.0       # → ~0.5 kg for r=0.03
    sphere_start_z: float = 0.15

    # ── book (flat box held on its widest faces) ──
    #
    # Defaults model a typical trade paperback (152 × 229 × 25 mm, 0.45 kg
    # — roughly a 6"×9" softcover).  Pads (40 × 100 mm face) cover ~11 %
    # of the cover, which keeps the gripper-vs-object proportion realistic
    # for a Franka- or UR5-class arm.  `book_hx` is the *thin* dimension
    # along the squeeze axis; book_hy/book_hz are the cover dimensions.
    #
    # The paper's original "Rotational Grasp Stability" experiment used
    # a 16×300×400 mm, 1.2 kg synthetic panel — closer to a sheet of
    # plywood than a book.  To reproduce that scene, set
    # `--initial-pen` and (via SceneParams) book_hx/y/z/density manually,
    # or restore the paper-spec values in this dataclass.
    #
    # Reference real-book sizes for tuning:
    #   mass-market paperback:  108×175×17 mm,  ~0.18 kg
    #   trade paperback:        152×229×25 mm,  ~0.45 kg   ← default
    #   hardcover novel:        165×242×32 mm,  ~0.65 kg
    #   coffee-table book:      280×320×25 mm,  ~1.5  kg
    book_hx: float = 0.0125              # half-thickness, 25 mm total
    book_hy: float = 0.076               # half-width, 152 mm total (~6")
    book_hz: float = 0.115               # half-height, 230 mm total (~9")
    book_density: float = 515.0          # → ~0.45 kg trade paperback
    book_start_z: float = 0.30           # spawn high enough to hold under gravity

    # ── pads ──
    pad_hx: float = 0.01
    pad_hy: float = 0.02
    pad_hz: float = 0.05
    # Penetration regime is now aligned with lift_test.py: hold pen ≈ 1 mm
    # per side after the active squeeze.  Sphere diameter = 60 mm, so
    # gap_initial=59mm ⇒ initial pen = 0.5 mm/side; squeeze adds 0.5 mm
    # over 0.5 s → 1 mm/side at HOLD start.  This puts both tests in the
    # same operating point as the §2 fair-calibration derivation in
    # cslc_v1/summary.md, where CSLC's distributed-constraint advantage
    # is unambiguous (vs the previous 12.5 mm "deep-deformation" regime
    # in which hydro happened to win on creep).
    pad_gap_initial: float = 0.059       # 0.5 mm initial pen / side
    pad_squeeze_speed: float = 0.001     # 1 mm/s → 0.5 mm extra in 0.5 s
    pad_squeeze_duration: float = 0.5    # seconds of active squeeze
    pad_hold_duration: float = 1.5       # seconds of holding under gravity

    # ── material (shared by all contact models) ──
    ke: float = 5.0e4
    kd: float = 5.0e2
    kf: float = 100.0
    mu: float = 0.5

    # ── CSLC tuning (matched to lift_test.py fair calibration) ──
    cslc_spacing: float = 0.005
    cslc_ka: float = 15000.0  # fair-cal target: N·ka > ke_bulk so the
                              # exact branch of recalibrate_cslc_kc_per_pad
                              # solves for kc instead of falling back to
                              # the kc = ke/N approximation.
    cslc_kl: float = 500.0
    cslc_dc: float = 2.0
    cslc_n_iter: int = 20
    cslc_alpha: float = 0.6
    # Contact fraction used for per-pad kc calibration (see
    # `recalibrate_cslc_kc_per_pad`).  At 1 mm face_pen on this pad
    # geometry only ~5 of 189 surface spheres per pad lie inside the
    # active patch — cf=0.025 reflects that.  With ka=15000 and
    # ke_bulk=5e4 the calibration solves to kc=75000, keff=12500,
    # aggregate per pad = 50000 N/m = ke_bulk ✓.  See §2 in
    # cslc_v1/summary.md for the full derivation.
    # Use None to keep the built-in default from `calibrate_kc` (0.15).
    cslc_contact_fraction: float | None = 0.025

    # ── hydroelastic (for optional comparison) ──
    # kh = hydroelastic modulus [Pa], fair-calibrated so that
    # kh · A_patch(1 mm pen) = ke_bulk.  At r=30 mm the patch area
    # A_patch = π·(2·r·pen) ≈ 188 mm², so kh = 5e4 / 1.88e-4 = 2.65e8 Pa.
    # See §2 in cslc_v1/summary.md.  At kh=1e10 the solver ejects the
    # sphere; the §1 stability sweep documents the safe range.
    kh: float = 2.65e8
    sdf_resolution: int = 64

    # ── integration ──
    dt: float = 1.0 / 500.0
    gravity: tuple = (0.0, 0.0, -9.81)

    # ── external perturbation on the held object ──
    #
    # World-frame force [N] and torque [N·m] applied to the sphere body
    # during the HOLD phase only.  Lets the user probe CSLC's
    # distributed-contact response under a known disturbance — e.g.
    # `--external-force 0,0,-5` adds an extra 5 N pulling the sphere
    # downward (≈1× the sphere weight) so the friction constraints have
    # to fight harder; a non-zero torque exercises the rotational
    # stiffness paper claim (§4.2 in cslc_v1/overleaf_theory_cslc_icra.txt).
    # Default zero → no perturbation.
    external_force: tuple = (0.0, 0.0, 0.0)
    external_torque: tuple = (0.0, 0.0, 0.0)

    # ── derived ──

    @property
    def sphere_mass(self):
        return self.sphere_density * (4 / 3) * math.pi * self.sphere_radius ** 3

    @property
    def n_squeeze_steps(self):
        return int(self.pad_squeeze_duration / self.dt)

    @property
    def n_hold_steps(self):
        return int(self.pad_hold_duration / self.dt)

    @property
    def n_total_steps(self):
        return self.n_squeeze_steps + self.n_hold_steps

    def initial_penetration(self):
        """Penetration on each side at t=0 before any squeezing."""
        half_gap = self.pad_gap_initial / 2.0
        return max(self.sphere_radius - half_gap, 0.0)

    def final_penetration(self):
        """Penetration on each side after full squeeze."""
        return self.initial_penetration() + self.pad_squeeze_speed * self.pad_squeeze_duration

    def dump(self):
        """Print a full parameter summary with derived geometry analysis."""
        _section("SCENE PARAMETERS")
        m = self.sphere_mass
        w = m * 9.81
        pen0 = self.initial_penetration()
        pen1 = self.final_penetration()

        _log(f"Sphere:  r={self.sphere_radius*1e3:.1f}mm  "
             f"mass={m:.3f}kg ({m*1e3:.1f}g)  weight={w:.3f}N")
        _log(f"Pads:    hx={self.pad_hx*1e3:.1f}mm  hy={self.pad_hy*1e3:.1f}mm  "
             f"hz={self.pad_hz*1e3:.1f}mm")
        _log(f"Gap:     initial={self.pad_gap_initial*1e3:.1f}mm  "
             f"sphere_diameter={self.sphere_radius*2e3:.1f}mm")
        _log(f"Squeeze: speed={self.pad_squeeze_speed*1e3:.2f}mm/s/pad  "
             f"dur={self.pad_squeeze_duration:.2f}s  "
             f"travel={self.pad_squeeze_speed * self.pad_squeeze_duration * 1e3:.2f}mm/pad")
        _log(f"Penetration: initial={pen0*1e3:.2f}mm/side  "
             f"final={pen1*1e3:.2f}mm/side")
        _log(f"Material: ke={self.ke:.0f}  kd={self.kd:.0f}  μ={self.mu:.2f}")
        _log(f"CSLC:    spacing={self.cslc_spacing*1e3:.1f}mm  ka={self.cslc_ka:.0f}  "
             f"kl={self.cslc_kl:.0f}  dc={self.cslc_dc:.1f}")
        _log(f"Solver:  n_iter={self.cslc_n_iter}  alpha={self.cslc_alpha}  "
             f"dt={self.dt*1e3:.2f}ms ({1/self.dt:.0f}Hz)")
        _log(f"Steps:   {self.n_total_steps} total "
             f"({self.n_squeeze_steps} squeeze + {self.n_hold_steps} hold)")

        # Friction budget: can two point contacts hold the weight?
        fn_point = self.ke * pen0
        ff_point = self.mu * fn_point
        _log("")
        _log("Friction budget (point contact, initial penetration):")
        _log(
            f"  F_n ≈ ke × pen = {fn_point:.1f}N  →  F_friction = μ × F_n = {ff_point:.1f}N", 1)
        _log(f"  2 contacts × {ff_point:.1f}N = {2*ff_point:.1f}N  vs  weight = {w:.3f}N  "
             f"({'OK' if 2*ff_point > w else 'INSUFFICIENT'})", 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    name: str = ""
    sphere_z: list[float] = field(default_factory=list)
    sphere_x: list[float] = field(default_factory=list)
    # Per-step (qx, qy, qz, qw) of the held body — populated for both
    # the sphere and book objects.  Used to compute `max_tilt_deg` for
    # the rotational-stability claim (book tests measure how far the
    # body rotated from rest under an external torque).
    sphere_quat: list[tuple[float, float, float, float]] = field(default_factory=list)
    active_contacts: list[int] = field(default_factory=list)
    cslc_max_delta: list[float] = field(default_factory=list)
    cslc_active_surface: list[int] = field(default_factory=list)
    cslc_max_pen: list[float] = field(default_factory=list)
    n_squeeze_steps: int = 0
    dt: float = 0.0

    @property
    def z_drop_mm(self):
        return (self.sphere_z[0] - min(self.sphere_z)) * 1e3 if len(self.sphere_z) >= 2 else 0.0

    @property
    def hold_drop_mm(self):
        """z position at HOLD start minus z at end. Positive = sphere fell during HOLD.

        This metric isolates the compliance creep from the squeeze transient.
        Hydroelastic pushes the sphere UP during SQUEEZE (asymmetric 15 mm pressure
        field), so `z_drop_mm` based on global min is confounded — `hold_drop_mm`
        strips out the transient and measures only what matters for grip quality.
        """
        if self.n_squeeze_steps >= len(self.sphere_z):
            return 0.0
        return (self.sphere_z[self.n_squeeze_steps] - self.sphere_z[-1]) * 1e3

    @property
    def hold_creep_rate_mm_per_s(self):
        """Mean downward velocity during the last half of HOLD (steady state).

        Uses the second half of the HOLD phase to avoid the squeeze→hold
        transient. Positive = falling.
        """
        if self.dt <= 0 or len(self.sphere_z) <= self.n_squeeze_steps + 2:
            return 0.0
        hold = self.sphere_z[self.n_squeeze_steps:]
        mid = len(hold) // 2
        if len(hold) - mid < 2:
            return 0.0
        delta_z = hold[mid] - hold[-1]
        delta_t = (len(hold) - 1 - mid) * self.dt
        return (delta_z / delta_t) * 1e3 if delta_t > 0 else 0.0

    @property
    def peak_contacts(self):
        return max(self.active_contacts) if self.active_contacts else 0

    @property
    def max_tilt_deg(self):
        """Maximum angular deviation of the held body from its rest pose [deg].

        Computed from the body quaternion `(qx, qy, qz, qw)` as the
        unsigned rotation angle 2·acos(|qw|).  When the body is at rest
        with identity rotation this is zero; under an external torque
        it grows.  Useful for the paper's rotational-stability metric.
        """
        if not self.sphere_quat:
            return 0.0
        max_a = 0.0
        for q in self.sphere_quat:
            qw = abs(q[3])
            if qw >= 1.0:
                continue
            a = 2.0 * math.degrees(math.acos(qw))
            if a > max_a:
                max_a = a
        return max_a

    @property
    def final_tilt_deg(self):
        """Tilt at end of run [deg] — companion to max_tilt_deg."""
        if not self.sphere_quat:
            return 0.0
        qw = abs(self.sphere_quat[-1][3])
        if qw >= 1.0:
            return 0.0
        return 2.0 * math.degrees(math.acos(qw))


# ═══════════════════════════════════════════════════════════════════════════
#  Scene builders
# ═══════════════════════════════════════════════════════════════════════════

def _pad_cfg(p: SceneParams, **kw) -> newton.ModelBuilder.ShapeConfig:
    d = dict(ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=0.0)
    d.update(kw)
    return newton.ModelBuilder.ShapeConfig(**d)


def _sphere_cfg(p: SceneParams, **kw) -> newton.ModelBuilder.ShapeConfig:
    d = dict(ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu,
             gap=0.002, density=p.sphere_density)
    d.update(kw)
    return newton.ModelBuilder.ShapeConfig(**d)


def _book_cfg(p, **kw):
    d = dict(ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu,
             gap=0.002, density=p.book_density)
    d.update(kw)
    return newton.ModelBuilder.ShapeConfig(**d)


def _target_z(p):
    return p.book_start_z if p.object_kind == "book" else p.sphere_start_z


def _add_pads_and_target(b, p, pad_cfg, target_cfg):
    """Add two kinematic pads + the chosen dynamic target.  Returns body indices.

    Both pads must present their local +x face INWARD (toward the
    target) because cslc_handler hardcodes face_axis=0, face_sign=+1 —
    the CSLC lattice lives on the box's local +x face.  The left pad at
    world -x naturally satisfies this; the right pad at world +x needs
    a 180° rotation around z, applied to the SHAPE so kinematic control
    via body_q stays unchanged.

    The held target is either a sphere (`p.object_kind == "sphere"`) or
    a flat box (`"book"`).  Pad positions track `pad_gap_initial`,
    which the caller is expected to size correctly for the chosen
    object (the squeeze main override does this for book mode).
    """
    target_z = _target_z(p)

    lx = -(p.pad_gap_initial / 2 + p.pad_hx)
    rx = (p.pad_gap_initial / 2 + p.pad_hx)

    left = b.add_body(xform=wp.transform((lx, 0, target_z), wp.quat_identity()),
                      is_kinematic=True, label="left_pad")
    b.add_shape_box(left, hx=p.pad_hx, hy=p.pad_hy, hz=p.pad_hz, cfg=pad_cfg)

    right = b.add_body(xform=wp.transform((rx, 0, target_z), wp.quat_identity()),
                       is_kinematic=True, label="right_pad")
    right_box_xform = wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
    )
    b.add_shape_box(right, xform=right_box_xform,
                    hx=p.pad_hx, hy=p.pad_hy, hz=p.pad_hz, cfg=pad_cfg)

    target = b.add_body(
        xform=wp.transform((0, 0, target_z), wp.quat_identity()),
        label=p.object_kind,
    )
    if p.object_kind == "sphere":
        b.add_shape_sphere(target, radius=p.sphere_radius, cfg=target_cfg)
    elif p.object_kind == "book":
        b.add_shape_box(target,
                        hx=p.book_hx, hy=p.book_hy, hz=p.book_hz,
                        cfg=target_cfg)
    else:
        raise ValueError(f"Unknown object_kind: {p.object_kind!r}")

    return left, right, target


def _target_cfg(p, contact_model: str):
    """Material config for the held target depending on object + model.

    Hydro point/cslc paths use the basic config; the hydro path
    additionally tags the target with hydroelastic params so the
    handler emits pressure-field contacts.
    """
    base_kwargs = {}
    if contact_model == "hydro":
        base_kwargs = dict(kh=p.kh, is_hydroelastic=True,
                           sdf_max_resolution=p.sdf_resolution)
    if p.object_kind == "sphere":
        return _sphere_cfg(p, **base_kwargs)
    elif p.object_kind == "book":
        return _book_cfg(p, **base_kwargs)
    raise ValueError(f"Unknown object_kind: {p.object_kind!r}")


def build_point_scene(p):
    b = newton.ModelBuilder()
    _add_pads_and_target(b, p, _pad_cfg(p), _target_cfg(p, "point"))
    m = b.finalize()
    m.set_gravity(p.gravity)
    return m


def build_cslc_scene(p):
    b = newton.ModelBuilder()
    cslc_pad = _pad_cfg(p, is_cslc=True,
                        cslc_spacing=p.cslc_spacing, cslc_ka=p.cslc_ka, cslc_kl=p.cslc_kl,
                        cslc_dc=p.cslc_dc, cslc_n_iter=p.cslc_n_iter, cslc_alpha=p.cslc_alpha)
    _add_pads_and_target(b, p, cslc_pad, _target_cfg(p, "cslc"))
    m = b.finalize()
    m.set_gravity(p.gravity)
    return m


def build_hydro_scene(p):
    b = newton.ModelBuilder()
    hp = _pad_cfg(p, kh=p.kh, is_hydroelastic=True,
                  sdf_max_resolution=p.sdf_resolution)
    _add_pads_and_target(b, p, hp, _target_cfg(p, "hydro"))
    m = b.finalize()
    m.set_gravity(p.gravity)
    return m


BUILDERS = {"point": build_point_scene,
            "cslc": build_cslc_scene, "hydro": build_hydro_scene}


# ═══════════════════════════════════════════════════════════════════════════
#  Kinematic pad control
#
#  Warp spatial_vector layout: [wx, wy, wz, vx, vy, vz]
#    angular velocity → indices 0, 1, 2
#    linear  velocity → indices 3, 4, 5
# ═══════════════════════════════════════════════════════════════════════════

def compute_pad_x(step, p, sign):
    """World-frame x position for a pad at the given step."""
    initial_x = sign * (p.pad_gap_initial / 2 + p.pad_hx)
    t = step * p.dt
    return initial_x - sign * p.pad_squeeze_speed * min(t, p.pad_squeeze_duration)


def set_kinematic_pads(state, step, p, debug=False):
    """Set pad positions + linear velocities each step.

    Positions are set directly (kinematic bodies).  We also tell the solver
    the pads' linear velocity so that relative-velocity-based friction is
    computed correctly during the active squeeze phase.

    spatial_vector layout: [angular(3), linear(3)]
    """
    q = state.body_q.numpy()
    q[0, 0] = compute_pad_x(step, p, sign=-1.0)   # left pad
    q[1, 0] = compute_pad_x(step, p, sign=+1.0)    # right pad
    state.body_q.assign(wp.array(q, dtype=wp.transform,
                        device=state.body_q.device))

    qd = state.body_qd.numpy()
    qd[0] = 0.0
    qd[1] = 0.0
    t = step * p.dt
    if t < p.pad_squeeze_duration:
        # Linear velocity along x — indices [3] in spatial_vector
        qd[0, 3] = +p.pad_squeeze_speed   # left pad → +x (inward)
        qd[1, 3] = -p.pad_squeeze_speed   # right pad → -x (inward)
    state.body_qd.assign(
        wp.array(qd, dtype=wp.spatial_vector, device=state.body_qd.device))

    if debug:
        gap = (q[1, 0] - p.pad_hx) - (q[0, 0] + p.pad_hx)
        phase = "SQUEEZE" if t < p.pad_squeeze_duration else "HOLD"
        _log(f"Pads [{phase}] left_x={q[0, 0]:+.5f}  right_x={q[1, 0]:+.5f}  "
             f"gap={gap*1e3:.2f}mm  vx=({qd[0, 3]:+.4f}, {qd[1, 3]:+.4f})")




# ═══════════════════════════════════════════════════════════════════════════
#  Contact inspection
# ═══════════════════════════════════════════════════════════════════════════

# `count_active_contacts` lives in cslc_v1.common.

def dump_contacts(contacts, label="", max_show=20):
    """Print the contact buffer in detail."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    s0 = contacts.rigid_contact_shape0.numpy()
    s1 = contacts.rigid_contact_shape1.numpy()
    nrm = contacts.rigid_contact_normal.numpy()
    m0 = contacts.rigid_contact_margin0.numpy()
    m1 = contacts.rigid_contact_margin1.numpy()
    active = int(np.sum(s0[:n] >= 0)) if n > 0 else 0

    has_props = contacts.rigid_contact_stiffness is not None
    if has_props:
        ke = contacts.rigid_contact_stiffness.numpy()
        kd = contacts.rigid_contact_damping.numpy()
        mu = contacts.rigid_contact_friction.numpy()

    _sub(f"CONTACTS: {label}")
    _log(f"count={n}  active={active}  max={contacts.rigid_contact_max}")

    shown = 0
    for i in range(min(n, contacts.rigid_contact_max)):
        if s0[i] < 0:
            continue
        line = (f"[{i:5d}] s0={s0[i]} s1={s1[i]}  "
                f"n=({nrm[i, 0]:+.3f},{nrm[i, 1]:+.3f},{nrm[i, 2]:+.3f})  "
                f"margin=({m0[i]:.4f},{m1[i]:.4f})")
        if has_props:
            line += f"  ke={ke[i]:.1f} kd={kd[i]:.2f} μ={mu[i]:.3f}"
        _log(line)
        shown += 1
        if shown >= max_show:
            if active - shown > 0:
                _log(f"... {active - shown} more active contacts")
            break


# ═══════════════════════════════════════════════════════════════════════════
#  CSLC lattice inspection
# ═══════════════════════════════════════════════════════════════════════════

def inspect_cslc_handler(model, label=""):
    """Print a full dump of the CSLC handler: lattice geometry, stiffness,
    shape pairs, neighbor topology, surface normal distribution."""
    _sub(f"CSLC HANDLER: {label}")

    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        _log("No CSLC handler (not a CSLC scene)")
        return None

    d = handler.cslc_data
    _log(f"Spheres:        {d.n_spheres} total, {d.n_surface} surface")
    _log(
        f"Spring stiffness:  ka={d.ka:.0f} (anchor)  kl={d.kl:.0f} (lateral)  kc={d.kc:.1f} (contact)")
    _log(f"Damping:        dc={d.dc:.2f}")
    _log(f"Solver:         {handler.n_iter} iterations, α={handler.alpha}")
    _log(
        f"Contact slots:  {handler.n_surface_contacts}  (offset={pipeline.cslc_contact_offset})")

    # Effective per-sphere stiffness at equilibrium (uniform delta, all neighbors equal)
    eff = d.kc * d.ka / (d.ka + d.kc) if (d.ka + d.kc) > 0 else 0
    _log(f"Effective per-sphere stiffness (uniform contact): {eff:.1f}")
    _log(
        f"Aggregate (all surface): {d.n_surface * eff:.0f}  (vs shape ke={model.shape_material_ke.numpy()[0]:.0f})")

    for pair in handler.shape_pairs:
        _log(f"Pair: CSLC shape {pair.cslc_shape} vs shape {pair.other_shape}  "
             f"(geo={GEO_NAMES.get(pair.other_geo_type, '?')} body={pair.other_body} "
             f"r={pair.other_radius:.4f})")

    # Geometry summary per shape
    pos = d.positions.numpy()
    is_surf = d.is_surface.numpy()
    normals = d.outward_normals.numpy()
    shape_ids = d.sphere_shape.numpy()
    nbr_count = d.neighbor_count.numpy()

    for sid in np.unique(shape_ids):
        mask = shape_ids == sid
        smask = mask & (is_surf == 1)
        sp = pos[mask]
        sn = normals[smask]
        _log(f"Shape {sid}: {mask.sum()} spheres ({smask.sum()} surface)  "
             f"pos X=[{sp[:, 0].min():.4f},{sp[:, 0].max():.4f}]  "
             f"Y=[{sp[:, 1].min():.4f},{sp[:, 1].max():.4f}]  "
             f"Z=[{sp[:, 2].min():.4f},{sp[:, 2].max():.4f}]")
        for ax, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            npos = (sn[:, ax] > 0.9).sum()
            nneg = (sn[:, ax] < -0.9).sum()
            if npos or nneg:
                _log(f"  normals {name}: +{npos}  -{nneg}", 1)

    _log(
        f"Neighbor count: min={nbr_count.min()} max={nbr_count.max()} mean={nbr_count.mean():.1f}")

    # Convergence estimate
    avg_nbr = nbr_count.mean()
    rho = d.kl * avg_nbr / (d.ka + d.kl * avg_nbr +
                            d.kc) if (d.ka + d.kl * avg_nbr + d.kc) > 0 else 1
    _log(
        f"Spectral radius ρ ≈ {rho:.4f}  → ~{-1/math.log10(max(rho, 0.01)):.0f} iters per decade")

    # Confirm friction comes from material
    mu_arr = model.shape_material_mu.numpy()
    _log(f"Shape μ values: {mu_arr}")

    return handler


# `read_cslc_state` lives in cslc_v1.common.

def print_cslc_state(model, step=-1, inline=False):
    info = read_cslc_state(model)
    if not info:
        return ""
    s = (f"CSLC: {info['n_active_surface']}/{info['n_total_surface']} active  "
         f"δ_max={info['max_delta']*1e3:.3f}mm  pen_max={info['max_pen']*1e3:.3f}mm")
    if not inline:
        _log(s)
    return s


# ═══════════════════════════════════════════════════════════════════════════
#  Model inspection
# ═══════════════════════════════════════════════════════════════════════════

def inspect_model(model, label=""):
    _sub(f"MODEL: {label}")
    _log(f"bodies={model.body_count}  shapes={model.shape_count}")

    st = model.shape_type.numpy()
    sf = model.shape_flags.numpy()
    sc = model.shape_scale.numpy()
    sb = model.shape_body.numpy()
    ske = model.shape_material_ke.numpy()
    smu = model.shape_material_mu.numpy()
    sg = model.shape_gap.numpy()

    for i in range(model.shape_count):
        gn = GEO_NAMES.get(int(st[i]), f"?{st[i]}")
        cslc = " CSLC" if sf[i] & CSLC_FLAG else ""
        _log(f"Shape {i}: {gn}{cslc}  body={sb[i]}  "
             f"scale=({sc[i, 0]:.4f},{sc[i, 1]:.4f},{sc[i, 2]:.4f})  "
             f"ke={ske[i]:.0f}  μ={smu[i]:.2f}  gap={sg[i]:.4f}")

    pairs = model.shape_contact_pairs
    if pairs is not None:
        pnp = pairs.numpy()
        _log(f"Contact pairs: {len(pnp)}")
        for a, b in pnp:
            _log(f"  ({a}, {b})", 1)

    fp = getattr(model, "shape_collision_filter_pairs", set())
    if fp:
        _log(f"Filter pairs: {sorted(fp)}")

    _log(f"rigid_contact_max: {getattr(model, 'rigid_contact_max', '?')}")


# Solver factory and per-pad CSLC calibration both live in cslc_v1.common.
# Lattice visualisation helpers (`_quat_rotate`, `get_cslc_lattice_viz_data`)
# also live in cslc_v1.common.


def _reset_cslc(model):
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler:
        handler.cslc_data.sphere_delta.zero_()
        _log("CSLC warm-start reset")


# ═══════════════════════════════════════════════════════════════════════════
#  Headless test runners
# ═══════════════════════════════════════════════════════════════════════════

def _dump_mujoco_dist_diag(solver, label=""):
    """Print MuJoCo contact dist values after Newton→MuJoCo conversion.

    dist < 0 = penetrating (contact active).  dist > 0 = separated (no force).
    For CSLC with ~170 contacts, we want to see ~170 penetrating dist values.
    If most are > 0, MuJoCo is not seeing the contacts as penetrating, explaining
    why no friction force is applied.
    """
    mjw_data = getattr(solver, "mjw_data", None)
    if mjw_data is None:
        return  # CPU MuJoCo or non-MuJoCo solver — skip
    nacon_np = mjw_data.nacon.numpy()
    nacon = int(nacon_np.flat[0])
    if nacon == 0:
        print(
            f"    [MJW_DIST {label}] nacon=0 (no contacts visible to MuJoCo)")
        return
    dist_np = mjw_data.contact.dist.numpy().flatten()[:nacon]
    n_pen = int((dist_np < 0).sum())
    n_sep = int((dist_np >= 0).sum())
    d_min = float(dist_np.min())
    d_max = float(dist_np.max())
    d_mean = float(dist_np.mean())
    # includemargin is the contact activation distance (margin+gap); contacts
    # with dist > includemargin are culled by MuJoCo.
    incl_np = mjw_data.contact.includemargin.numpy().flatten()[:nacon]
    n_active = int((dist_np < incl_np).sum())
    print(f"    [MJW_DIST {label}] nacon={nacon}  penetrating={n_pen}  separated={n_sep}  "
          f"active(dist<incl)={n_active}  dist[mm]: min={d_min*1e3:.3f} mean={d_mean*1e3:.3f} max={d_max*1e3:.3f}")


def run_squeeze(name, model, solver, p, verbose=1):
    """Run the full squeeze→hold sequence with diagnostic output.

    verbose: 0=silent  1=periodic  2=every step  3=full contact dump

    NOTE: body_qd is stale under MuJoCo GPU solver. Velocity and force are
    derived from position history (met.sphere_z) using finite differences.
    """
    met = Metrics(name=name, n_squeeze_steps=p.n_squeeze_steps, dt=p.dt)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    con = model.contacts()
    _reset_cslc(model)
    is_cslc = "cslc" in name.lower()

    m_sphere = float(model.body_mass.numpy()[2])
    g_accel  = float(np.linalg.norm(model.gravity.numpy()))
    weight_N = m_sphere * g_accel
    PRINT_INTERVAL = 50   # print every N HOLD steps

    # Per-step external wrench on the sphere body (idx 2) — applied during
    # HOLD only, so squeeze dynamics are unperturbed.  Skipped entirely
    # when both vectors are zero (the default).
    has_external = any(p.external_force) or any(p.external_torque)
    sphere_body_idx = 2

    for step in range(p.n_total_steps):
        set_kinematic_pads(s0, step, p, debug=(verbose >= 3 and step % 100 == 0))
        s0.clear_forces()

        if has_external and step >= p.n_squeeze_steps:
            apply_external_wrench(
                s0, sphere_body_idx,
                force=p.external_force, torque=p.external_torque,
            )

        model.collide(s0, con)
        solver.step(s0, s1, ctrl, con, p.dt)
        wp.synchronize()

        q  = s1.body_q.numpy()
        sz = float(q[2, 2])
        sx = float(q[2, 0])
        # body_q layout: (px, py, pz, qx, qy, qz, qw) — pull the four
        # quaternion components for the held body so Metrics can compute
        # `max_tilt_deg` for the rotational-stability metric.
        squat = (float(q[2, 3]), float(q[2, 4]),
                 float(q[2, 5]), float(q[2, 6]))
        nc = count_active_contacts(con)

        met.sphere_z.append(sz)
        met.sphere_x.append(sx)
        met.sphere_quat.append(squat)
        met.active_contacts.append(nc)

        if is_cslc:
            ci = read_cslc_state(model)
            if ci:
                met.cslc_max_delta.append(ci["max_delta"])
                met.cslc_active_surface.append(ci["n_active_surface"])
                met.cslc_max_pen.append(ci["max_pen"])

        # Position-FD diagnostics (robust: doesn't use stale body_qd).
        # vz_fd  = position first-difference  [m/s], negative = falling
        # az_fd  = velocity first-difference  [m/s²]
        # Fz_est = net upward contact force   [N],  should ≈ weight_N
        n_hist = len(met.sphere_z)
        if n_hist >= 2:
            vz_fd = (met.sphere_z[-1] - met.sphere_z[-2]) / p.dt
        else:
            vz_fd = 0.0
        if n_hist >= 3:
            vz_prev = (met.sphere_z[-2] - met.sphere_z[-3]) / p.dt
            az_fd   = (vz_fd - vz_prev) / p.dt
            Fz_est  = m_sphere * (az_fd + g_accel)
        else:
            Fz_est = float("nan")

        in_hold   = step >= p.n_squeeze_steps
        hold_step = step - p.n_squeeze_steps

        # MuJoCo dist readout at hold start and periodic intervals.
        if in_hold and hold_step in (0, 1, 2, 50, 100, 200, 500, 1000):
            _dump_mujoco_dist_diag(solver, name)

        s0, s1 = s1, s0

        phase = "SQUEEZE" if not in_hold else "HOLD  "
        zd = (met.sphere_z[0] - sz) * 1e3
        rate_mms = -vz_fd * 1e3   # positive = falling

        do_print = (
            verbose >= 2
            or (verbose >= 1 and (step + 1) % 100 == 0)
            or step == p.n_total_steps - 1
            or (in_hold and hold_step % PRINT_INTERVAL == 0)
        )
        if do_print:
            line = (f"  {name:20s} {step+1:5d}/{p.n_total_steps}  [{phase}]  "
                    f"z={sz:+.6f}m  drop={zd:+.3f}mm  rate={rate_mms:+.3f}mm/s  "
                    f"Fz≈{Fz_est:+.2f}N(W={weight_N:.2f}N)  contacts={nc:4d}")
            if is_cslc and met.cslc_max_delta:
                line += (f"  δ={met.cslc_max_delta[-1]*1e3:.3f}mm"
                         f"  pen={met.cslc_max_pen[-1]*1e3:.3f}mm"
                         f"  surf={met.cslc_active_surface[-1]}")
            print(line)

        if verbose >= 3 and step % 200 == 0:
            dump_contacts(con, f"step {step+1}", max_show=10)

    return met




# ═══════════════════════════════════════════════════════════════════════════
#  Test orchestrators
# ═══════════════════════════════════════════════════════════════════════════

def test_squeeze(p, solver_name="mujoco", contact_models=None):
    if contact_models is None:
        contact_models = ["point", "cslc"]
    results = []
    for cm in contact_models:
        label = f"{cm}_{solver_name}"
        _section(label.upper())
        model = BUILDERS[cm](p)
        _ = model.contacts()  # forces _init_collision_pipeline() so we can inspect it
        inspect_model(model, label)
        if cm == "cslc":
            inspect_cslc_handler(model, label)
            if p.cslc_contact_fraction is not None:
                recalibrate_cslc_kc_per_pad(model, p.cslc_contact_fraction)
                # Re-inspect so the printed handler reflects the new kc.
                inspect_cslc_handler(model, f"{label} (recalibrated)")
        solver = make_solver(model, solver_name)
        m = run_squeeze(label, model, solver, p, verbose=1)
        results.append(m)
        _log(
            f"RESULT: z-drop(full)={m.z_drop_mm:.3f}mm  "
            f"hold-drop={m.hold_drop_mm:+.3f}mm  "
            f"creep-rate={m.hold_creep_rate_mm_per_s:+.3f}mm/s  "
            f"peak_contacts={m.peak_contacts}")
    return results



def test_calibrate(p):
    """Deep diagnostic of CSLC calibration — no simulation, just geometry + stiffness."""
    _section("CSLC CALIBRATION DIAGNOSTIC")
    p.dump()

    model = build_cslc_scene(p)
    inspect_model(model, "cslc")
    handler = inspect_cslc_handler(model, "cslc")
    if handler is None:
        _log("ERROR: no CSLC handler was created")
        return

    d = handler.cslc_data

    _sub("STIFFNESS ANALYSIS")
    _log(f"ka={d.ka:.0f}  kl={d.kl:.0f}  kc={d.kc:.1f}  dc={d.dc:.2f}")
    eff = d.kc * d.ka / (d.ka + d.kc)
    _log(
        f"Effective per-sphere stiffness (uniform case): kc·ka/(ka+kc) = {eff:.1f}")
    _log(
        f"Total aggregate: {d.n_surface} × {eff:.1f} = {d.n_surface * eff:.0f}")

    _sub("SINGLE-STEP CONVERGENCE TEST")
    state = model.state()
    contacts = model.contacts()
    set_kinematic_pads(state, 0, p, debug=True)
    state.clear_forces()
    model.collide(state, contacts)
    info = read_cslc_state(model)
    for k, v in info.items():
        fmt = f"{v*1e3:.4f}mm" if isinstance(v, float) and (
            "delta" in k or "pen" in k) else str(v)
        _log(f"  {k}: {fmt}")


def test_inspect(p, solver_name, n_steps=10):
    """Step-by-step dump: every contact, every body state, every CSLC delta."""
    _section(f"STEP-BY-STEP INSPECTION ({n_steps} steps)")
    p.dump()

    for cm in ["point", "cslc"]:
        label = f"{cm}_{solver_name}"
        _sub(f"Building {label}")
        model = BUILDERS[cm](p)
        inspect_model(model, label)
        if cm == "cslc":
            inspect_cslc_handler(model, label)

        solver = make_solver(model, solver_name)
        s0 = model.state()
        s1 = model.state()
        ctrl = model.control()
        con = model.contacts()
        _reset_cslc(model)

        for step in range(n_steps):
            _sub(f"{label} — STEP {step+1}/{n_steps}")
            set_kinematic_pads(s0, step, p, debug=True)
            s0.clear_forces()

            q = s0.body_q.numpy()
            qd = s0.body_qd.numpy()
            _log(f"Sphere PRE:  pos=({q[2, 0]:.5f},{q[2, 1]:.5f},{q[2, 2]:.5f})  "
                 f"ω=({qd[2, 0]:.5f},{qd[2, 1]:.5f},{qd[2, 2]:.5f})  "
                 f"v=({qd[2, 3]:.5f},{qd[2, 4]:.5f},{qd[2, 5]:.5f})")

            model.collide(s0, con)
            dump_contacts(con, f"{label} step {step+1}", max_show=30)
            if cm == "cslc":
                print_cslc_state(model, step)

            solver.step(s0, s1, ctrl, con, p.dt)
            wp.synchronize()

            q2 = s1.body_q.numpy()
            qd2 = s1.body_qd.numpy()
            _log(f"Sphere POST: pos=({q2[2, 0]:.5f},{q2[2, 1]:.5f},{q2[2, 2]:.5f})  "
                 f"ω=({qd2[2, 0]:.5f},{qd2[2, 1]:.5f},{qd2[2, 2]:.5f})  "
                 f"v=({qd2[2, 3]:.5f},{qd2[2, 4]:.5f},{qd2[2, 5]:.5f})")
            _log(f"Δz = {(q2[2, 2]-q[2, 2])*1e3:+.4f}mm")

            s0, s1 = s1, s0


# ═══════════════════════════════════════════════════════════════════════════
#  CSV output
# ═══════════════════════════════════════════════════════════════════════════

def save_results(all_metrics, path):
    with open(path, "w") as f:
        f.write("config,z_drop_mm,max_angle_deg,settled,peak_contacts\n")
        for m in all_metrics:
            s = "yes" if m.settled else ("no" if m.sphere_angle else "n/a")
            f.write(
                f"{m.name},{m.z_drop_mm:.4f},{m.max_angle_deg:.2f},{s},{m.peak_contacts}\n")
    print(f"\nResults saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Viewer mode
# ═══════════════════════════════════════════════════════════════════════════

class Example:
    """Interactive viewer following Newton's Example protocol."""

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

        self.p = SceneParams(dt=self.sim_dt)
        if getattr(args, "object", "sphere") == "book":
            self.p.object_kind = "book"
            self.p.pad_gap_initial = 2.0 * (self.p.book_hx - 0.0015)
            self.p.cslc_contact_fraction = 1.0
            pad_face_area = (2.0 * self.p.pad_hy) * (2.0 * self.p.pad_hz)
            self.p.kh = self.p.ke / pad_face_area
        if getattr(args, "external_force", None):
            self.p.external_force = _parse_xyz(
                args.external_force, "external-force")
        if getattr(args, "external_torque", None):
            self.p.external_torque = _parse_xyz(
                args.external_torque, "external-torque")
        # `--initial-pen` / `--mu` overrides — see main() for rationale.
        if getattr(args, "initial_pen", None) is not None:
            target_half = (self.p.book_hx if self.p.object_kind == "book"
                           else self.p.sphere_radius)
            self.p.pad_gap_initial = 2.0 * (target_half - float(args.initial_pen))
        if getattr(args, "mu", None) is not None:
            self.p.mu = float(args.mu)
        self.p.dump()

        self.model = BUILDERS[self.contact_model](self.p)
        inspect_model(self.model, self.contact_model)
        if self.contact_model == "cslc":
            inspect_cslc_handler(self.model, self.contact_model)

        self.solver = make_solver(self.model, self.solver_name)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.current_z_drop_mm = 0.0
        self.current_contacts = 0
        self.initial_z = self.p.sphere_start_z

        # FD telemetry (body_qd is stale under MuJoCo GPU — derive vz
        # from position history sampled every print interval).
        self._prev_sphere_z = None
        self._prev_pad_lx = None
        self._prev_pad_rx = None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.4, -0.4, self.p.sphere_start_z + 0.1),
            pitch=-10.0, yaw=135.0)

        self._init_lattice_rendering()

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

        _log(f"Contact model: {self.contact_model}")
        _log(f"Solver: {self.solver_name}")
        _log(f"Sphere mass: {self.p.sphere_mass:.3f} kg")

    def _init_lattice_rendering(self):
        self._lattice_n = 0
        pipeline = getattr(self.model, "_collision_pipeline", None)
        handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
        if handler is None:
            return
        d = handler.cslc_data
        n = d.n_surface
        self._lattice_n = n
        self._lattice_radius = float(d.radii.numpy()[0]) * 0.4
        self._lattice_xforms = np.zeros((n, 7), np.float32)
        self._lattice_xforms[:, 6] = 1.0
        self._lattice_colors = np.zeros((n, 3), np.float32)
        self._lattice_mats = np.tile(
            [0.5, 0.3, 0.0, 0.0], (n, 1)).astype(np.float32)

    def simulate(self):
        has_external = (any(self.p.external_force)
                        or any(self.p.external_torque))
        n_squeeze_steps = self.p.n_squeeze_steps
        for _ in range(self.sim_substeps):
            set_kinematic_pads(self.state_0, self.sim_step, self.p)
            self.state_0.clear_forces()
            if has_external and self.sim_step >= n_squeeze_steps:
                apply_external_wrench(
                    self.state_0, 2,
                    force=self.p.external_force,
                    torque=self.p.external_torque,
                )
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_step += 1

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        q = self.state_0.body_q.numpy()
        sphere_z = float(q[2, 2])
        pad_lx = float(q[0, 0])
        pad_rx = float(q[1, 0])
        self.current_z_drop_mm = (self.initial_z - sphere_z) * 1e3
        self.current_contacts = count_active_contacts(self.contacts)

        # Periodic console diagnostics (every ~100 ms of sim time).  The
        # mujoco GPU solver doesn't expose a usable sphere body_qd, so
        # vz is derived from position history at the print stride.
        if self.sim_step % (self.sim_substeps * 6) < self.sim_substeps:
            t = self.sim_step * self.p.dt
            phase = "SQUEEZE" if t < self.p.pad_squeeze_duration else "HOLD"
            gap = (pad_rx - self.p.pad_hx) - (pad_lx + self.p.pad_hx)
            face_pen = self.p.sphere_radius * 2.0 - gap

            dt_print = 6.0 * self.frame_dt
            if self._prev_sphere_z is None:
                vz_sphere = 0.0
            else:
                vz_sphere = (sphere_z - self._prev_sphere_z) / dt_print
            self._prev_sphere_z = sphere_z

            cslc_str = ""
            if self.contact_model == "cslc":
                ci = read_cslc_state(self.model)
                if ci:
                    cslc_str = (f"  cslc={ci['n_active_surface']}/{ci['n_total_surface']}"
                                f"  δ_max={ci['max_delta']*1e3:.2f}mm")

            _log(f"[{phase:7s}] step={self.sim_step:5d}  "
                 f"sz={sphere_z:+.4f}m  drop={self.current_z_drop_mm:+6.2f}mm  "
                 f"vz={vz_sphere*1e3:+6.2f}mm/s  "
                 f"face_pen={face_pen*1e3:+5.2f}mm  "
                 f"n={self.current_contacts}{cslc_str}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.show_lattice and self.contact_model == "cslc":
            self._render_lattice()
        self.viewer.end_frame()

    def _render_lattice(self):
        viz = get_cslc_lattice_viz_data(self.model, self.state_0)
        if viz is None:
            return
        pw, dl, rd, sf = viz
        idx = 0
        mx = max(float(np.max(np.abs(dl))), 1e-6)
        for i in range(len(dl)):
            if sf[i] == 0 or idx >= self._lattice_n:
                continue
            self._lattice_xforms[idx, :3] = pw[i]
            t = min(abs(dl[i]) / mx, 1.0)
            self._lattice_colors[idx] = [
                t, 0.2*(1-t), 1-t] if dl[i] > 1e-8 else [0.3, 0.3, 0.35]
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
        imgui.text(f"Z-drop: {self.current_z_drop_mm:.2f} mm")
        imgui.text(f"Contacts: {self.current_contacts}")
        imgui.text(f"Step: {self.sim_step}")
        t = self.sim_step * self.p.dt
        imgui.text(
            f"Phase: {'SQUEEZE' if t < self.p.pad_squeeze_duration else 'HOLD'}")
        if self.contact_model == "cslc":
            _, self.show_lattice = imgui.checkbox(
                "Show lattice", self.show_lattice)

    def test_final(self):
        if self.contact_model == "cslc":
            assert self.current_z_drop_mm < 5.0, f"z-drop too large: {self.current_z_drop_mm:.2f}mm"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--contact-model", type=str, default="cslc",
                            choices=["point", "cslc", "hydro"])
        parser.add_argument("--solver", type=str, default="mujoco",
                            choices=["mujoco", "semi"])
        parser.add_argument(
            "--show-lattice", action="store_true", default=True)
        parser.add_argument(
            "--no-lattice", dest="show_lattice", action="store_false")
        parser.add_argument("--mode", type=str, default="viewer",
                            choices=["viewer", "squeeze", "calibrate", "inspect"])
        parser.add_argument("--steps", type=int, default=10,
                            help="Steps for inspect mode")
        parser.add_argument(
            "--contact-fraction", type=float, default=None,
            help="Per-pad contact fraction for CSLC kc recalibration. "
                 "Pass the empirical observed fraction (e.g. 0.46 for the "
                 "default squeeze scene). Pass a negative value to keep the "
                 "default 0.15 in calibrate_kc.")
        parser.add_argument(
            "--cslc-spacing", type=float, default=None,
            help="Override the CSLC lattice spacing [m]. Default 5e-3.")
        parser.add_argument(
            "--contact-models", type=str, default=None,
            help="Comma-separated list of contact models to run in squeeze "
                 "mode (e.g. 'point,cslc,hydro'). Default: 'point,cslc'.")
        parser.add_argument(
            "--external-force", type=str, default=None,
            help="World-frame force [N] applied to the sphere during HOLD, "
                 "as 'fx,fy,fz' (e.g. '0,0,-5' for an extra 5N pulling down). "
                 "Default: no external force.")
        parser.add_argument(
            "--external-torque", type=str, default=None,
            help="World-frame torque [N·m] on the sphere during HOLD, as "
                 "'tx,ty,tz' (e.g. '0,0,0.05' for a 0.05 N·m twist about z).")
        parser.add_argument(
            "--object", type=str, default="sphere", choices=["sphere", "book"],
            help="Held object geometry. 'sphere' is the legacy r=30mm test "
                 "object; 'book' is a flat 16×300×400 mm box (paper §4.2 "
                 "rotational-stability scene). Defaults to sphere.")
        parser.add_argument(
            "--book-mass", type=float, default=None,
            help="Override the book's total mass [kg] by adjusting its "
                 "density.  Lighter books are easier for CSLC to hold "
                 "under MuJoCo's regularised friction (less per-contact "
                 "compliance leak f/k).  Default ≈1.2 kg (paper-spec).")
        parser.add_argument(
            "--initial-pen", type=float, default=None,
            help="Override the initial pad-face penetration [m].  Sets "
                 "pad_gap_initial = 2 · (target_half_extent − initial_pen) "
                 "so the pad face starts `initial_pen` inside the target "
                 "surface at t=0.  Default: 1.5 mm for book, 10 mm for "
                 "sphere.  Used by the compass-needle disturbance "
                 "experiment to bring the friction-cone budget close to "
                 "the operating torque.")
        parser.add_argument(
            "--mu", type=float, default=None,
            help="Override the shared Coulomb friction coefficient on "
                 "both pads and target.  Default 0.5.  Lower values "
                 "shrink the friction-cone budget and expose model-"
                 "differentiation under torque disturbance.")
        return parser


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = Example.create_parser()
    args, _ = parser.parse_known_args()
    mode = args.mode

    # --viewer is a newton.examples flag; it is only processed inside
    # newton.examples.init(), which is only called in viewer mode.
    # Auto-redirect so the user doesn't get a silent headless run.
    if mode != "viewer" and "--viewer" in sys.argv:
        print(f"  NOTE: --viewer is only supported with --mode viewer.\n"
              f"  Switching to viewer mode (use --contact-model to pick the model).\n"
              f"  For headless batch comparison: drop --viewer.")
        mode = "viewer"

    wp.init()
    print(f"\n{_HSEP}\n  CSLC TEST SUITE — mode: {mode}\n{_HSEP}")

    if mode == "viewer":
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
        return

    args = parser.parse_args()
    device = getattr(args, "device", "cuda:0")
    try:
        wp.set_device(device)
    except Exception:
        print(f"  Device '{device}' unavailable → CPU")
        wp.set_device("cpu")

    p = SceneParams()
    if getattr(args, "object", "sphere") == "book":
        p.object_kind = "book"
        # ── pad_gap_initial: geometry override ──
        # 2 * (half_extent_along_pad_axis - initial_pen).  Different
        # objects have different sizes, so the gap that puts the pad
        # face 1.5 mm inside the target is different.  Sphere uses
        # 0.059 m (2 * (0.030 - 0.0005)); book needs 0.013 m
        # (2 * (0.008 - 0.0015)).  Heavier book → bigger initial pen
        # so the t=0 friction has enough headroom to catch the body
        # before it slides out of the pad's z-range.
        p.pad_gap_initial = 2.0 * (p.book_hx - 0.0015)
        # ── cslc_contact_fraction: calibration prior override ──
        # Sphere target: only ~5/189 surface spheres engage (small
        # Hertzian patch) → cf ≈ 0.025.
        # Book target: pad face fully overlaps book cover → all 189
        # engage → cf ≈ 1.0.
        # `recalibrate_cslc_kc_per_pad` uses cf to solve for kc such
        # that N · keff = ke_bulk.  Wrong cf gives wrong kc — at
        # cf=0.025 with 189 active spheres the aggregate stiffness
        # becomes 47× ke_bulk and MuJoCo ejects the book.
        p.cslc_contact_fraction = 1.0
        # ── kh: hydroelastic-modulus override ──
        # Same fair-calibration principle as CSLC: kh · A_patch =
        # ke_bulk.  Sphere target's patch area is the Hertzian
        # contact (≈π·2r·pen = 188 mm² @ 1 mm pen) → kh = 2.65e8 Pa.
        # Book target's patch area is the pad face fully inside the
        # book (≈ 2hy · 2hz = 4000 mm²) → kh ≈ 1.25e7 Pa, 21× softer.
        # Without this override hydro is 21× too stiff for the book
        # and the body is ejected the same way as miscalibrated CSLC.
        pad_face_area = (2.0 * p.pad_hy) * (2.0 * p.pad_hz)
        p.kh = p.ke / pad_face_area
        # Optional: override book mass via CLI for stability sweeps.
        if getattr(args, "book_mass", None) is not None:
            book_volume = 8.0 * p.book_hx * p.book_hy * p.book_hz
            p.book_density = float(args.book_mass) / book_volume
    if getattr(args, "contact_fraction", None) is not None:
        cf = float(args.contact_fraction)
        p.cslc_contact_fraction = None if cf < 0 else cf
    if getattr(args, "cslc_spacing", None) is not None:
        p.cslc_spacing = float(args.cslc_spacing)
    if getattr(args, "external_force", None):
        p.external_force = _parse_xyz(args.external_force, "external-force")
    if getattr(args, "external_torque", None):
        p.external_torque = _parse_xyz(args.external_torque, "external-torque")
    # ── --initial-pen / --mu overrides ──
    # Applied AFTER the object-mode block so the user can override the
    # default pen/mu values that mode would otherwise pick.
    # `--initial-pen` recomputes pad_gap_initial from the target's
    # squeeze-axis half-extent (book_hx for book, sphere_radius for
    # sphere) so the geometric meaning ("pad face starts initial_pen
    # past the target surface at t=0") is preserved across object kinds.
    if getattr(args, "initial_pen", None) is not None:
        target_half = p.book_hx if p.object_kind == "book" else p.sphere_radius
        p.pad_gap_initial = 2.0 * (target_half - float(args.initial_pen))
    if getattr(args, "mu", None) is not None:
        p.mu = float(args.mu)
    sn = args.solver if hasattr(args, "solver") else (
        "mujoco" if HAS_MUJOCO else "semi")
    all_res = []

    if mode == "calibrate":
        test_calibrate(p)
        return
    if mode == "inspect":
        test_inspect(p, sn, args.steps)
        return

    if mode == "squeeze":
        p.dump()
        models = getattr(args, "contact_models", None)
        if models:
            cm_list = [c.strip() for c in models.split(",") if c.strip()]
        else:
            cm_list = ["point", "cslc"]
        all_res.extend(test_squeeze(p, sn, contact_models=cm_list))

    if all_res:
        _section("SUMMARY")
        print(f"  {'Config':<30} {'FullDrop':>10} {'HoldDrop':>10} "
              f"{'Creep':>10} {'MaxTilt':>9} {'EndTilt':>9} {'Contacts':>10}")
        print(f"  {'':<30} {'[mm]':>10} {'[mm]':>10} {'[mm/s]':>10} "
              f"{'[deg]':>9} {'[deg]':>9}")
        for m in all_res:
            print(f"  {m.name:<30} {m.z_drop_mm:10.3f} "
                  f"{m.hold_drop_mm:+10.3f} {m.hold_creep_rate_mm_per_s:+10.3f} "
                  f"{m.max_tilt_deg:9.2f} {m.final_tilt_deg:9.2f} "
                  f"{m.peak_contacts:10d}")
        print(f"{_DSEP}")


if __name__ == "__main__":
    main()
