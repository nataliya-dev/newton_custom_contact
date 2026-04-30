# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for cslc_v1's squeeze, lift, and robot examples.

Everything here is CSLC-specific scaffolding that was being duplicated
across `squeeze_test.py`, `lift_test.py`, and `robot_example/utils.py`
in slightly inconsistent forms.  Lifting it into one module gives the
three tests a single source of truth for: stiffness recalibration,
state read-back, lattice visualisation, shape-config factories, and
the MuJoCo solver factory.

Tests still own their own scene construction and trajectory logic —
this module deliberately does NOT impose a common Example or
SceneParams class, because the three tests' driving mechanics
(kinematic pads / articulated PD pads / Franka URDF + IK) are
genuinely different and trying to unify them would create more
confusion than it removes.
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSemiImplicit

try:
    from newton.solvers import SolverMuJoCo
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    warnings.warn(
        "SolverMuJoCo not available — falling back to semi-implicit.")


# ── Logging ──────────────────────────────────────────────────────────────

_SEP = "─" * 60


def _log(msg, indent: int = 0) -> None:
    print(f"  {'  ' * indent}│ {msg}")


def _section(title: str) -> None:
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")


# ── CSLC bookkeeping ─────────────────────────────────────────────────────

# Must match the value in `newton/_src/geometry/types.py` (ShapeFlags.CSLC).
CSLC_FLAG = 1 << 5

# Geometry-type integer → name lookup, mirrors `newton.GeoType`.
GEO_NAMES = {
    0: "PLANE", 1: "MESH", 2: "HFIELD", 3: "SPHERE",
    4: "CAPSULE", 5: "CYLINDER", 6: "CONE", 7: "BOX",
    8: "CONVEX_MESH", 9: "ELLIPSOID",
}


def count_active_contacts(contacts) -> int:
    """Count contacts that survived narrow-phase culls.

    A "valid" contact has shape0 ≥ 0; CSLC's hybrid emission policy
    writes shape0 = -1 for slots whose smooth gate fell below 1e-4
    (see `cslc_kernels.write_cslc_contacts`).
    """
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0
    return int(np.sum(contacts.rigid_contact_shape0.numpy()[:n] >= 0))


def read_cslc_state(model) -> dict | None:
    """Snapshot the CSLC handler's per-step lattice state.

    Returns a dict with both the long ("n_active_surface", "max_delta")
    and short ("n_active", "max_delta_mm") key forms because the three
    tests historically used different conventions.  Returns None if no
    CSLC handler is attached to the model.
    """
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None
    d = handler.cslc_data
    is_surf = d.is_surface.numpy() == 1
    deltas = d.sphere_delta.numpy()[is_surf]
    pen = handler.raw_penetration.numpy()[is_surf]
    active = pen > 0
    n_active = int(active.sum())
    n_surface = int(is_surf.sum())
    max_delta = float(deltas.max()) if len(deltas) else 0.0
    max_pen = float(pen.max()) if len(pen) else 0.0
    mean_delta = float(deltas.mean()) if len(deltas) else 0.0
    mean_pen_active = float(pen[active].mean()) if n_active else 0.0
    return {
        # Short form (lift/robot convention).
        "n_active": n_active,
        "n_surface": n_surface,
        "max_delta_mm": max_delta * 1e3,
        "max_pen_mm": max_pen * 1e3,
        # Long form (squeeze convention).
        "n_active_surface": n_active,
        "n_total_surface": n_surface,
        "max_delta": max_delta,
        "max_pen": max_pen,
        "mean_delta": mean_delta,
        "mean_pen_active": mean_pen_active,
    }


def recalibrate_cslc_kc_per_pad(model, contact_fraction: float):
    """Override per-sphere kc so each pad's aggregate stiffness equals ke_bulk.

    Reads ke from the FIRST CSLC-flagged shape (so we don't read the ground
    plane's ke when shape 0 happens to be a plane in lift_test) and applies
    the §2 fair-calibration derivation:

        N_contact_per_pad · kc·ka/(ka+kc) = ke_bulk
        ⇒ kc = ke_bulk · ka / (N · ka − ke_bulk)        (if N·ka > ke_bulk)
        ⇒ kc = ke_bulk / N                              (fallback)

    Returns the new kc value, or None if no CSLC handler is attached.
    """
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None

    d = handler.cslc_data
    shape_flags = model.shape_flags.numpy()
    cslc_shape_idx = next(
        (i for i in range(model.shape_count) if (shape_flags[i] & CSLC_FLAG)),
        0,
    )
    ke_bulk = float(model.shape_material_ke.numpy()[cslc_shape_idx])

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
    _log(f"CSLC RECAL: pads={n_pads}  surface/pad={n_surface_per_pad}  "
         f"N_contact_per_pad={n_contact_per_pad}  ({derivation})")
    _log(f"            kc: {old_kc:.1f}  →  {new_kc:.1f} N/m  "
         f"keff={keff:.1f}  agg/pad={aggregate_per_pad:.0f} (target={ke_bulk:.0f})")
    return new_kc


def inspect_model(model, label: str = "") -> None:
    """Print a body / shape / joint summary, marking [CSLC] shapes."""
    _log(f"Model '{label}': {model.body_count} bodies, "
         f"{model.shape_count} shapes, "
         f"{model.joint_count} joints, {model.joint_dof_count} DOFs")
    st = model.shape_type.numpy()
    sf = model.shape_flags.numpy()
    sb = model.shape_body.numpy()
    for i in range(model.shape_count):
        cslc = " [CSLC]" if sf[i] & CSLC_FLAG else ""
        _log(f"  shape {i}: {GEO_NAMES.get(int(st[i]), '?')}  "
             f"body={sb[i]}{cslc}", 1)


# ── CSLC lattice visualisation ───────────────────────────────────────────

def _quat_rotate(q, v):
    xyz = np.array([q[0], q[1], q[2]])
    t = 2.0 * np.cross(xyz, v)
    return v + q[3] * t + np.cross(xyz, t)


def get_cslc_lattice_viz_data(model, state):
    """Compute world-space positions of every surface lattice sphere.

    Returns (pw, dl, rd, sf) where pw is (n_spheres, 3) world positions
    after applying per-sphere compression along outward normal, dl is
    the raw delta array, rd the radii, sf the surface flag.  Returns
    None if no CSLC handler is attached.
    """
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


# ── Shape config factories ───────────────────────────────────────────────
#
# These return a `newton.ModelBuilder.ShapeConfig` that the test scene
# builder can apply to either pad or held-object shapes.  Pulling the
# factories here keeps each test's scene-building code focused on
# geometry and topology, not material book-keeping.

def shape_cfg_point(*, ke, kd, kf, mu, density, gap=0.002):
    return newton.ModelBuilder.ShapeConfig(
        ke=ke, kd=kd, kf=kf, mu=mu, gap=gap, density=density)


def shape_cfg_cslc_pad(*, ke, kd, kf, mu, density,
                       cslc_spacing, cslc_ka, cslc_kl,
                       cslc_dc, cslc_n_iter, cslc_alpha, gap=0.002):
    return newton.ModelBuilder.ShapeConfig(
        ke=ke, kd=kd, kf=kf, mu=mu, gap=gap, density=density,
        is_cslc=True,
        cslc_spacing=cslc_spacing, cslc_ka=cslc_ka, cslc_kl=cslc_kl,
        cslc_dc=cslc_dc, cslc_n_iter=cslc_n_iter, cslc_alpha=cslc_alpha)


def shape_cfg_hydro(*, ke, kd, kf, mu, density, kh, sdf_resolution=64,
                    gap=0.002):
    return newton.ModelBuilder.ShapeConfig(
        ke=ke, kd=kd, kf=kf, mu=mu, gap=gap, density=density,
        kh=kh, is_hydroelastic=True, sdf_max_resolution=sdf_resolution)


# ── External wrench application ──────────────────────────────────────────
#
# Newton's `body_f` is a per-body spatial vector laid out
# [tx, ty, tz, fx, fy, fz] (Featherstone — torque first, then force).
# `state.clear_forces()` zeros it; setting it before `solver.step()`
# adds an external wrench in the world frame.  Used by squeeze_test's
# `--external-force` to demonstrate CSLC's distributed-contact response
# to a perturbation force during HOLD.

def apply_external_wrench(state, body_idx: int,
                          force: tuple[float, float, float] = (0.0, 0.0, 0.0),
                          torque: tuple[float, float, float] = (0.0, 0.0, 0.0)
                          ) -> None:
    """Set the spatial wrench on ``body_idx`` to (torque, force) for one step.

    Must be called AFTER ``state.clear_forces()`` and BEFORE
    ``solver.step()``.  Overwrites whatever was previously in
    ``body_f[body_idx]``; other bodies are untouched.
    """
    bf = state.body_f.numpy()
    bf[body_idx, 0] = float(torque[0])
    bf[body_idx, 1] = float(torque[1])
    bf[body_idx, 2] = float(torque[2])
    bf[body_idx, 3] = float(force[0])
    bf[body_idx, 4] = float(force[1])
    bf[body_idx, 5] = float(force[2])
    state.body_f.assign(
        wp.array(bf, dtype=wp.spatial_vector, device=state.body_f.device))


# ── Solver factory ───────────────────────────────────────────────────────

def make_mujoco_solver(model, *,
                       iterations: int | None = None,
                       ls_iterations: int = 10,
                       cone: str = "elliptic",
                       integrator: str = "implicitfast",
                       solver: str = "cg",
                       extra_ncon: int = 5000):
    """Build a MuJoCo solver sized to fit the model's CSLC contact slots.

    The CSLC handler reserves one contact slot per surface lattice sphere
    per pair, so a scene with 2 pads × 189 surface spheres × 1 target
    needs 2·189 = 378 extra slots above whatever the narrow phase emits.
    This helper walks the CSLC-flagged shapes, sums their surface-sphere
    counts, and grows ``njmax`` / ``nconmax`` accordingly.

    ``iterations`` defaults to 100 when CSLC is in the model, 20 otherwise
    — CSLC's ~170 simultaneous constraints don't converge under MuJoCo CG
    at the 20-iter setting that suffices for point contact (see
    summary §1).
    """
    if not HAS_MUJOCO:
        raise RuntimeError("SolverMuJoCo not available")

    has_cslc = model.shape_cslc_spacing is not None

    ncon = extra_ncon
    if has_cslc:
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
            nx, ny, nz = (max(int(round(2 * h / sp)) + 1, 2)
                          for h in (hx, hy, hz))
            interior = max(nx - 2, 0) * max(ny - 2, 0) * max(nz - 2, 0)
            ncon += nx * ny * nz - interior

    if iterations is None:
        iterations = 100 if has_cslc else 20

    return SolverMuJoCo(
        model, use_mujoco_contacts=False,
        solver=solver, integrator=integrator, cone=cone,
        iterations=iterations, ls_iterations=ls_iterations,
        njmax=ncon, nconmax=ncon)


def make_solver(model, solver_name: str, *, iterations: int | None = None):
    """Build a `mujoco` or `semi` solver — convenience wrapper used by tests."""
    if solver_name == "mujoco":
        return make_mujoco_solver(model, iterations=iterations)
    elif solver_name == "semi":
        return SolverSemiImplicit(model)
    raise ValueError(f"Unknown solver: {solver_name!r}")
