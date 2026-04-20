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

import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSemiImplicit

try:
    from newton.solvers import SolverMuJoCo
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    warnings.warn("SolverMuJoCo not available. MuJoCo configs disabled.")


# ═══════════════════════════════════════════════════════════════════════════
#  Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════

_SEP = "─" * 72
_HSEP = "━" * 72
_DSEP = "═" * 72


def _section(title):
    print(f"\n{_DSEP}\n  {title}\n{_DSEP}")


def _sub(title):
    print(f"\n  {_SEP}\n  {title}\n  {_SEP}")


def _log(msg, indent=0):
    print(f"  {'  ' * indent}│ {msg}")


# ═══════════════════════════════════════════════════════════════════════════
#  Scene parameters
# ═══════════════════════════════════════════════════════════════════════════

GEO_NAMES = {
    0: "PLANE", 1: "MESH", 2: "HFIELD", 3: "SPHERE", 4: "CAPSULE",
    5: "CYLINDER", 6: "CONE", 7: "BOX", 8: "CONVEX_MESH", 9: "ELLIPSOID",
}
CSLC_FLAG = 1 << 5


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

    # ── sphere ──
    sphere_radius: float = 0.03
    sphere_density: float = 4421.0       # → ~0.5 kg for r=0.03
    sphere_start_z: float = 0.15

    # ── pads ──
    pad_hx: float = 0.01
    pad_hy: float = 0.02
    pad_hz: float = 0.05
    pad_gap_initial: float = 0.04        # distance between inner pad faces
    pad_squeeze_speed: float = 0.005     # m/s inward per pad
    pad_squeeze_duration: float = 0.5    # seconds of active squeeze
    pad_hold_duration: float = 1.5       # seconds of holding under gravity

    # ── material (shared by all contact models) ──
    ke: float = 5.0e4
    kd: float = 5.0e2
    kf: float = 100.0
    mu: float = 0.5

    # ── CSLC tuning ──
    cslc_spacing: float = 0.005
    cslc_ka: float = 5000.0
    cslc_kl: float = 500.0
    cslc_dc: float = 2.0
    cslc_n_iter: int = 20
    cslc_alpha: float = 0.6
    # Contact fraction used for per-pad kc calibration (see
    # `recalibrate_cslc_kc_per_pad`).  Default 0.46 matches the empirically
    # observed active fraction in this squeeze scene (174 of 378 surface
    # spheres active under 15 mm total penetration).  Setting this equal
    # to the observed fraction makes each pad's aggregate contact stiffness
    # at uniform contact equal to `ke_bulk`, which is the fair per-shape
    # analogue of a point-contact spring at the same material stiffness.
    # Use None to keep the built-in default from `calibrate_kc` (0.15, all
    # pads sharing one budget).
    cslc_contact_fraction: float | None = 0.46

    # ── hydroelastic (for optional comparison) ──
    # kh = hydroelastic modulus [Pa].  1e10 is steel-stiff and triggers a
    # MuJoCo solver instability (sphere ejected from grip after squeeze).
    # 1e8 is a stable starting point for silicone-rubber-stiff pads and
    # gives a contact patch comparable in size to CSLC's.  Sweep cf
    # SECTION 6 of cslc_v1/convo_april_19.md for the kh-stability curve.
    kh: float = 1.0e8
    sdf_resolution: int = 64

    # ── integration ──
    dt: float = 1.0 / 500.0
    gravity: tuple = (0.0, 0.0, -9.81)

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
    active_contacts: list[int] = field(default_factory=list)
    cslc_max_delta: list[float] = field(default_factory=list)
    cslc_active_surface: list[int] = field(default_factory=list)
    cslc_max_pen: list[float] = field(default_factory=list)

    @property
    def z_drop_mm(self):
        return (self.sphere_z[0] - min(self.sphere_z)) * 1e3 if len(self.sphere_z) >= 2 else 0.0

    @property
    def peak_contacts(self):
        return max(self.active_contacts) if self.active_contacts else 0


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


def _add_pads_and_sphere(b, p, pad_cfg, sphere_cfg):
    """Add two kinematic pads + one dynamic sphere.  Returns body indices.

    Both pads must present their local +x face INWARD (toward the sphere)
    because cslc_handler hardcodes face_axis=0, face_sign=+1 — the CSLC
    lattice lives on the box's local +x face.  The left pad at world -x
    naturally satisfies this.  The right pad at world +x needs a 180°
    rotation around z, applied to the SHAPE (not the body) so kinematic
    control via body_q still works unchanged.

    Without this rotation the right pad's lattice points away from the
    sphere, every surface sphere fails the d_proj>0 cull in kernel 1,
    and — because CSLC pairs are filtered from the standard narrow phase
    — the right pad produces zero contacts with the sphere.  Result:
    unilateral grip, sphere slides out of the gripper under gravity.
    """
    lx = -(p.pad_gap_initial / 2 + p.pad_hx)
    rx = (p.pad_gap_initial / 2 + p.pad_hx)

    left = b.add_body(xform=wp.transform((lx, 0, p.sphere_start_z), wp.quat_identity()),
                      is_kinematic=True, label="left_pad")
    b.add_shape_box(left, hx=p.pad_hx, hy=p.pad_hy, hz=p.pad_hz, cfg=pad_cfg)

    right = b.add_body(xform=wp.transform((rx, 0, p.sphere_start_z), wp.quat_identity()),
                       is_kinematic=True, label="right_pad")
    right_box_xform = wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
    )
    b.add_shape_box(right, xform=right_box_xform,
                    hx=p.pad_hx, hy=p.pad_hy, hz=p.pad_hz, cfg=pad_cfg)

    sphere = b.add_body(xform=wp.transform((0, 0, p.sphere_start_z), wp.quat_identity()),
                        label="sphere")
    b.add_shape_sphere(sphere, radius=p.sphere_radius, cfg=sphere_cfg)
    return left, right, sphere


def build_point_scene(p):
    b = newton.ModelBuilder()
    _add_pads_and_sphere(b, p, _pad_cfg(p), _sphere_cfg(p))
    m = b.finalize()
    m.set_gravity(p.gravity)
    return m


def build_cslc_scene(p):
    b = newton.ModelBuilder()
    cslc_pad = _pad_cfg(p, is_cslc=True,
                        cslc_spacing=p.cslc_spacing, cslc_ka=p.cslc_ka, cslc_kl=p.cslc_kl,
                        cslc_dc=p.cslc_dc, cslc_n_iter=p.cslc_n_iter, cslc_alpha=p.cslc_alpha)
    _add_pads_and_sphere(b, p, cslc_pad, _sphere_cfg(p))
    m = b.finalize()
    m.set_gravity(p.gravity)
    return m


def build_hydro_scene(p):
    b = newton.ModelBuilder()
    hp = _pad_cfg(p, kh=p.kh, is_hydroelastic=True,
                  sdf_max_resolution=p.sdf_resolution)
    hs = _sphere_cfg(p, kh=p.kh, is_hydroelastic=True,
                     sdf_max_resolution=p.sdf_resolution)
    _add_pads_and_sphere(b, p, hp, hs)
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

def count_active_contacts(contacts):
    """Count contacts where shape0 >= 0 (valid, non-ghost)."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0
    return int(np.sum(contacts.rigid_contact_shape0.numpy()[:n] >= 0))


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


def read_cslc_state(model, state=None):
    """Read CSLC delta / penetration arrays.  Returns dict or empty."""
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return {}
    d = handler.cslc_data
    deltas = d.sphere_delta.numpy()
    is_surf = d.is_surface.numpy()
    pen = handler.raw_penetration.numpy()

    sm = is_surf == 1
    sd = deltas[sm]
    sp = pen[sm]
    act = sp > 0

    return dict(
        max_delta=float(sd.max()) if len(sd) else 0,
        mean_delta=float(sd.mean()) if len(sd) else 0,
        max_pen=float(sp.max()) if len(sp) else 0,
        mean_pen_active=float(sp[act].mean()) if act.sum() else 0,
        n_active_surface=int(act.sum()),
        n_total_surface=int(sm.sum()),
    )


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


# ═══════════════════════════════════════════════════════════════════════════
#  Solver creation
# ═══════════════════════════════════════════════════════════════════════════

def _make_solver(model, solver_name, p):
    if solver_name == "mujoco":
        if not HAS_MUJOCO:
            raise RuntimeError("MuJoCo solver not available")

        # Estimate contact buffer size (must cover narrow phase + CSLC slots)
        ncon = 5000
        if model.shape_cslc_spacing is not None:
            spacing_np = model.shape_cslc_spacing.numpy()
            flags_np = model.shape_flags.numpy()
            scale_np = model.shape_scale.numpy()
            total_surface = 0
            for i in range(model.shape_count):
                if not (flags_np[i] & CSLC_FLAG):
                    continue
                sp = float(spacing_np[i])
                if sp <= 0:
                    continue
                hx, hy, hz = float(scale_np[i][0]), float(
                    scale_np[i][1]), float(scale_np[i][2])
                nx = max(int(round(2*hx/sp))+1, 2)
                ny = max(int(round(2*hy/sp))+1, 2)
                nz = max(int(round(2*hz/sp))+1, 2)
                nt = nx*ny*nz
                ni = max(nx-2, 0)*max(ny-2, 0)*max(nz-2, 0)
                total_surface += nt - ni
                _log(
                    f"Shape {i}: CSLC grid ({nx},{ny},{nz})={nt} spheres, {nt-ni} surface")
            if total_surface > 0:
                ncon = total_surface + 5000
                _log(f"MuJoCo buffer: nconmax={ncon}")

        # Increased from 20 → 100 iterations (2026-04-19).
        # CSLC generates ~170 contacts vs point contact's ~2.  With only 20
        # CG iterations the solver was not converging on the full constraint
        # set, causing the sphere to creep downward under gravity even when
        # theoretical friction >> weight.  Point contact held fine at 20 iters
        # because 2 constraints converge almost instantly.
        n_iters = 100 if model.shape_cslc_spacing is not None else 20
        return SolverMuJoCo(model, use_mujoco_contacts=False,
                            solver="cg", integrator="implicitfast",
                            cone="elliptic",
                            iterations=n_iters, ls_iterations=10,
                            njmax=ncon, nconmax=ncon)
    elif solver_name == "semi":
        return SolverSemiImplicit(model)
    raise ValueError(f"Unknown solver: {solver_name}")


def _reset_cslc(model):
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler:
        handler.cslc_data.sphere_delta.zero_()
        _log("CSLC warm-start reset")


def recalibrate_cslc_kc_per_pad(model, contact_fraction):
    """Override per-sphere contact stiffness `kc` so that EACH PAD's aggregate
    stiffness at uniform contact equals the bulk material stiffness `ke_bulk`.

    Background
    ----------
    The default `calibrate_kc` in `cslc_data.py` sums `n_surface` across all
    pads and equates the total to a single `ke_bulk`, which (a) splits the
    material budget across pads in multi-pad grasps, and (b) hardcodes a
    `contact_fraction=0.15` that under-estimates the actual active count
    for a flat squeeze (observed ~0.46 in this scene).

    Per-pad calibration (this function)
    -----------------------------------
        N_contact_per_pad * (kc * ka) / (ka + kc) = ke_bulk
    Solving for `kc` (with positive denominator):
        kc = ke_bulk * ka / (N_per_pad * ka - ke_bulk)
    Otherwise fall back to a single-spring approximation
    `kc = ke_bulk / N_per_pad`.

    This makes a CSLC pad's aggregate stiffness at saturated uniform contact
    equal to a single point contact at `ke_bulk`, isolating the
    distributed-area effect from the calibration mismatch effect.

    Args:
        model: A finalized model whose `_collision_pipeline` has been
            initialized (call `model.contacts()` first).
        contact_fraction: Fraction of surface spheres expected to be active
            in contact for this scene's geometry/penetration combination.
            Pass the observed empirical fraction (e.g. 0.46 for the squeeze
            scene with 0.4 mm/side penetration).

    Returns:
        New kc value if the handler exists, else None.
    """
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None

    d = handler.cslc_data
    ke_bulk = float(model.shape_material_ke.numpy()[0])

    shape_ids = d.sphere_shape.numpy()
    is_surface = d.is_surface.numpy()
    n_pads = int(len(np.unique(shape_ids)))
    n_surface_per_pad = int(is_surface.sum()) // max(n_pads, 1)
    n_contact_per_pad = max(int(n_surface_per_pad * contact_fraction), 1)

    ka = float(d.ka)
    denom = n_contact_per_pad * ka - ke_bulk
    if denom <= 0.0:
        new_kc = ke_bulk / max(n_contact_per_pad, 1)
        derivation = "fallback (denom<=0): kc = ke/N"
    else:
        new_kc = ke_bulk * ka / denom
        derivation = "exact: kc = ke*ka/(N*ka - ke)"

    old_kc = float(d.kc)
    d.kc = new_kc
    keff = new_kc * ka / (ka + new_kc)
    aggregate_per_pad = n_contact_per_pad * keff

    _sub("CSLC RECALIBRATION (per-pad)")
    _log(f"pads={n_pads}  surface/pad={n_surface_per_pad}  "
         f"contact_fraction={contact_fraction:.3f}  "
         f"N_contact_per_pad={n_contact_per_pad}")
    _log(f"ka={ka:.0f}  ke_bulk={ke_bulk:.0f}  ({derivation})")
    _log(f"kc: {old_kc:.2f}  →  {new_kc:.2f}  N/m")
    _log(f"keff_per_sphere = kc*ka/(ka+kc) = {keff:.2f} N/m")
    _log(f"aggregate per-pad = N * keff = {aggregate_per_pad:.1f} N/m  "
         f"(target ke_bulk={ke_bulk:.0f})")
    return new_kc


# ═══════════════════════════════════════════════════════════════════════════
#  Lattice visualisation helper
# ═══════════════════════════════════════════════════════════════════════════

def _quat_rotate(q, v):
    xyz = np.array([q[0], q[1], q[2]])
    t = 2.0 * np.cross(xyz, v)
    return v + q[3] * t + np.cross(xyz, t)


def get_cslc_lattice_viz_data(model, state):
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None
    d = handler.cslc_data
    n = d.n_spheres
    pl = d.positions.numpy()
    nm = d.outward_normals.numpy()
    dl = d.sphere_delta.numpy()
    rd = d.radii.numpy()
    sf = d.is_surface.numpy()
    si = d.sphere_shape.numpy()
    bq = state.body_q.numpy()
    sb = model.shape_body.numpy()
    sx = model.shape_transform.numpy()

    pw = np.zeros((n, 3), np.float32)
    for i in range(n):
        if sf[i] == 0:
            continue
        s = si[i]
        b = sb[s]
        ql = pl[i] + dl[i] * nm[i]
        qb = _quat_rotate(sx[s, 3:7], ql) + sx[s, :3]
        pw[i] = _quat_rotate(bq[b, 3:7], qb) + bq[b, :3]
    return pw, dl, rd, sf


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
    met = Metrics(name=name)
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

    for step in range(p.n_total_steps):
        set_kinematic_pads(s0, step, p, debug=(verbose >= 3 and step % 100 == 0))
        s0.clear_forces()

        model.collide(s0, con)
        solver.step(s0, s1, ctrl, con, p.dt)
        wp.synchronize()

        q  = s1.body_q.numpy()
        sz = float(q[2, 2])
        sx = float(q[2, 0])
        nc = count_active_contacts(con)

        met.sphere_z.append(sz)
        met.sphere_x.append(sx)
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
        solver = _make_solver(model, solver_name, p)
        m = run_squeeze(label, model, solver, p, verbose=1)
        results.append(m)
        _log(
            f"RESULT: z-drop={m.z_drop_mm:.3f}mm  peak_contacts={m.peak_contacts}")
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

        solver = _make_solver(model, solver_name, p)
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
        self.p.dump()

        self.model = BUILDERS[self.contact_model](self.p)
        inspect_model(self.model, self.contact_model)
        if self.contact_model == "cslc":
            inspect_cslc_handler(self.model, self.contact_model)

        self.solver = _make_solver(self.model, self.solver_name, self.p)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.current_z_drop_mm = 0.0
        self.current_contacts = 0
        self.initial_z = self.p.sphere_start_z

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
        for _ in range(self.sim_substeps):
            set_kinematic_pads(self.state_0, self.sim_step, self.p)
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_step += 1

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        q = self.state_0.body_q.numpy()
        self.current_z_drop_mm = (self.initial_z - float(q[2, 2])) * 1e3
        self.current_contacts = count_active_contacts(self.contacts)

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
    if getattr(args, "contact_fraction", None) is not None:
        cf = float(args.contact_fraction)
        p.cslc_contact_fraction = None if cf < 0 else cf
    if getattr(args, "cslc_spacing", None) is not None:
        p.cslc_spacing = float(args.cslc_spacing)
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
        save_results(all_res, "cslc_results.csv")
        _section("SUMMARY")
        print(f"  {'Config':<30} {'Z-drop':>8} {'Contacts':>10}")
        for m in all_res:
            print(f"  {m.name:<30} {m.z_drop_mm:8.2f} {m.peak_contacts:10d}")
        print(f"{_DSEP}")


if __name__ == "__main__":
    main()
