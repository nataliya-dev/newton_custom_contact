# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Empirical anchor measurement for the contact-model benchmark spec.

Drives a grasp pipeline run to HOLD, then computes the four
calibration anchors (benchmark_spec.md §7.1) over a HOLD-averaging
window:

  L                — proxy for mean penetration depth, measured as
                     the time-averaged projection of sphere_delta onto
                     the pad's outward normal [m].  See "L protocol"
                     section below for why this is a PROXY for
                     phi_eff and how to convert.
  A_patch          — convex hull area of active-sphere world positions
                     projected onto the box face plane [m²], per pad
  contact_fraction — N_active / N_surface, per pad (time-averaged)
  F_per_pad        — mean |qfrc_actuator| at left/right pad x-DOF [N]

L protocol (delta_n proxy, NOT raw phi_eff)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CSLC handler's ``raw_penetration`` array is a PER-PAIR scratch
buffer: the kernel zeros it for every sphere whose pad is not the
currently-active pair (cslc_kernels.py:103).  After ``collide()``
finishes, only the LAST-processed pair's penetrations survive in the
buffer; reading it from the runner loop after the step gives biased
per-pad statistics (only one pad's spheres look "active").

This script uses ``sphere_delta`` instead (persistent lattice state,
updated by all pair-kernels during collide).  Per sphere i, the
penetration proxy is

    delta_n_i = dot(sphere_delta[i], outward_normal_world[i])

which at face-on contact equilibrium relates to phi_eff via

    phi_eff_i ≈ delta_n_i · (ka + kc_series) / kc_series

(three-spring series; at v0.7 calibration the factor is ≈ 2.7).  The
output JSON stores ``L`` directly as delta_n_mean; downstream
calibration code should multiply by the factor before using as an
"L" for E·A/L calibration.

The pre-v0.8 §7.1 anchor measurement used a different ad-hoc
protocol; v0.8 numbers from this script are NOT bit-comparable to
v0.7's.  Treat v0.8 anchors as a new baseline.

Uses the qfrc_actuator wrench-readout pattern documented in
notes.md §7.1 / benchmark_spec.md Appendix A: pre-allocate
``state.mujoco.qfrc_actuator`` on both states before stepping, then
read after each ``solver.step``.  The conventional
``Contacts.rigid_contact_force`` is allocated by Newton but never
populated by MuJoCo's solver path, so it returns zeros — the
qfrc_actuator route is the only working one.

JSON output captures per-pad AND aggregated anchors plus the
provenance fields needed to reproduce a calibration table from the
result.  Designed to be the permanent re-anchor instrument — re-run
on any material / cube / pad-geometry change.

Examples::

    # v0.8 re-anchor at the corrected R_pad = 10 mm.
    uv run --extra importers -m cslc_main.grasp.scripts.exp_anchors \\
        --pad-r-pad 0.010 --pad-half-angle 72 \\
        --output /tmp/anchors_v08.json

    # v0.7-style anchor at R_pad = 20 mm (regression check).
    uv run --extra importers -m cslc_main.grasp.scripts.exp_anchors \\
        --pad-r-pad 0.020 --pad-half-angle 72 \\
        --output /tmp/anchors_v07_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import warp as wp

import newton

from cslc_main.grasp.params import GraspConfig
from cslc_main.grasp.runner import _simulate_one_step
from cslc_main.grasp.scene import build_scene
from cslc_main.grasp.solvers import make_solver


# ── Geometry helpers ────────────────────────────────────────────────────


def _xform_rotate(xform: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the rotation part of Newton's transform to a vector v."""
    qx, qy, qz, qw = float(xform[3]), float(xform[4]), float(xform[5]), float(xform[6])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, xy, xz = qx * x2, qx * y2, qx * z2
    yy, yz, zz = qy * y2, qy * z2, qz * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    rx = (1.0 - (yy + zz)) * vx + (xy - wz) * vy + (xz + wy) * vz
    ry = (xy + wz) * vx + (1.0 - (xx + zz)) * vy + (yz - wx) * vz
    rz = (xz - wy) * vx + (yz + wx) * vy + (1.0 - (xx + yy)) * vz
    return np.array([rx, ry, rz], dtype=np.float64)


def _xform_apply(xform: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply Newton's full (px,py,pz,qx,qy,qz,qw) transform to point p."""
    return _xform_rotate(xform, p) + xform[:3].astype(np.float64)


def _convex_hull_area_2d(pts: np.ndarray) -> float:
    """Area of the convex hull of 2D points, in same units squared.

    Returns 0 for fewer than 3 distinct points.  Uses scipy's qhull
    via ``ConvexHull.volume`` (which is the AREA in 2D — scipy's
    naming convention).
    """
    if pts.shape[0] < 3:
        return 0.0
    # De-duplicate to avoid qhull degeneracies.
    pts = np.unique(np.round(pts, decimals=10), axis=0)
    if pts.shape[0] < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull  # local import to avoid hard dep
        hull = ConvexHull(pts)
        return float(hull.volume)  # 2D ConvexHull.volume == area
    except Exception:
        # Fallback: bounding-box area (over-estimate).  Should not trip
        # on well-formed scenes.
        bbox = pts.max(axis=0) - pts.min(axis=0)
        return float(bbox[0] * bbox[1])


# ── Anchor-measurement protocol ─────────────────────────────────────────


def measure_anchors(config: GraspConfig, *,
                    hold_skip_s: float = 0.5,
                    phi_active_threshold: float = 1.0e-6,
                    verbose: bool = True) -> dict:
    """Run one grasp pipeline cycle and return the four anchors.

    Args:
        config: GraspConfig (must have contact_model == "cslc").
        hold_skip_s: seconds to skip after HOLD begins before sampling
            (default 0.5 s — lets squeeze transients settle).
        phi_active_threshold: per-sphere mean phi_eff threshold [m]
            below which a sphere is counted as inactive in the patch /
            contact_fraction computation (default 1e-6 m matches
            cslc_state's existing convention).
        verbose: print per-step progress.

    Returns:
        Dict with keys ``L``, ``A_patch_left``, ``A_patch_right``,
        ``A_patch_avg``, ``contact_fraction_left``,
        ``contact_fraction_right``, ``F_left``, ``F_right``,
        ``F_per_pad`` (= mean of left + right), plus diagnostic
        fields (``hold_samples``, ``n_surface_per_pad``,
        ``newton_iii_residual``, etc.).
    """
    if config.contact_model != "cslc":
        raise ValueError(
            f"measure_anchors requires CSLC contact model, got "
            f"{config.contact_model!r}")

    artifacts = build_scene(config)
    model = artifacts.model
    solver = make_solver(model, config.solver)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Pre-allocate the qfrc_actuator wrench buffer on BOTH states.  See
    # benchmark_spec.md Appendix A: without this pre-allocation MuJoCo's
    # solver path silently skips writing the joint-actuator force back
    # into the state, and qfrc_actuator stays at zero.
    for s in (state_0, state_1):
        s.mujoco = types.SimpleNamespace()
        s.mujoco.qfrc_actuator = wp.zeros(
            model.joint_dof_count, dtype=wp.float32, device=model.device)

    # Sphere → pad-body map from the CSLC handler, computed once.
    handler = model._collision_pipeline.cslc_handler
    cslc_data = handler.cslc_data
    is_surface_arr = cslc_data.is_surface.numpy().astype(bool)
    sphere_shape_arr = cslc_data.sphere_shape.numpy()
    # CSLCData.positions = (n_spheres, 3) vec3 in SHAPE-LOCAL rest frame.
    # World position is reconstructed via X_wb * X_ws * positions[i]
    # (same chain the kernels use; see _world_positions_for below).
    sphere_pos_local_arr = cslc_data.positions.numpy()
    outward_normals_local_arr = cslc_data.outward_normals.numpy()
    shape_body_arr = model.shape_body.numpy()
    shape_transform_arr = model.shape_transform.numpy()
    left_body = artifacts.pad_body_indices["left"]
    right_body = artifacts.pad_body_indices["right"]
    pad_body_of_sphere = np.array(
        [shape_body_arr[sphere_shape_arr[i]] for i in range(cslc_data.n_spheres)],
        dtype=np.int64,
    )
    is_left_pad = is_surface_arr & (pad_body_of_sphere == left_body)
    is_right_pad = is_surface_arr & (pad_body_of_sphere == right_body)
    n_surface_left = int(is_left_pad.sum())
    n_surface_right = int(is_right_pad.sum())

    if verbose:
        print(f"  scene: n_surface_left = {n_surface_left}, "
              f"n_surface_right = {n_surface_right}, "
              f"left_body = {left_body}, right_body = {right_body}")

    # Phase boundaries.
    t = config.timing
    hold_start_step = int(
        (t.approach_duration + t.squeeze_duration + t.lift_duration) / t.dt)
    hold_sample_start = hold_start_step + int(hold_skip_s / t.dt)
    total_steps = config.total_steps

    # Warm-up step (JIT) — discarded.
    _simulate_one_step(
        model, solver, control, contacts,
        state_0, state_1, 0, config, artifacts.dof_map)
    wp.synchronize()

    # Accumulators for the HOLD window.
    #
    # NOTE on data sources: we use ``sphere_delta`` (persistent lattice
    # state, valid for ALL pads simultaneously) rather than
    # ``handler.raw_penetration`` (per-pair scratch buffer that gets
    # zeroed for non-active spheres at each pair launch).  After the
    # full collide() finishes, ``raw_penetration`` retains only the
    # last-processed pair's values, so reading it from the runner loop
    # gives biased per-pad statistics (only one pad's spheres look
    # "active").  ``sphere_delta`` is the IDEAL CSLC state variable
    # itself — q_i = p_i_world − δ_i — and it's updated by every pair
    # of jacobi_step iterations during collide(), so the final delta
    # field reflects equilibrium across ALL pads.
    #
    # The penetration proxy is the projection of delta along the
    # sphere's outward normal: at face-on contact equilibrium,
    # delta_n ≈ phi_eff · keff / ka where keff is the three-spring
    # series stiffness, so delta_n is monotone in phi_eff and is a
    # valid activity / depth proxy.  The exact factor depends on the
    # calibration; report delta_n as L and document the conversion in
    # the script output.
    delta_n_sum = np.zeros(cslc_data.n_spheres, dtype=np.float64)
    active_count = np.zeros(cslc_data.n_spheres, dtype=np.int64)
    f_left_samples: list[float] = []
    f_right_samples: list[float] = []
    f_left_signed_samples: list[float] = []  # for Newton III residual
    f_right_signed_samples: list[float] = []
    n_hold_samples = 0
    final_body_q = None
    final_sphere_delta = None
    # Per-pad outward normals in WORLD frame — recomputed at HOLD start
    # since body_q changes with pad close (pads are in steady state at
    # HOLD so we can fix this once at sample start).
    outward_normals_world: np.ndarray | None = None

    t0 = time.perf_counter()
    for step in range(total_steps):
        state_0, state_1, _, _ = _simulate_one_step(
            model, solver, control, contacts,
            state_0, state_1, step, config, artifacts.dof_map,
        )
        phase, _ = config.phase_of(step)

        if phase == "HOLD" and step >= hold_sample_start:
            q_act = state_0.mujoco.qfrc_actuator.numpy()
            f_l = float(q_act[artifacts.dof_map["left_x"]])
            f_r = float(q_act[artifacts.dof_map["right_x"]])
            f_left_samples.append(abs(f_l))
            f_right_samples.append(abs(f_r))
            f_left_signed_samples.append(f_l)
            f_right_signed_samples.append(f_r)

            # Compute outward normals in WORLD frame once at the start
            # of the HOLD sampling window.  X_wb is constant during HOLD
            # (pads are stationary), so this matches every subsequent
            # step's geometry.
            if outward_normals_world is None:
                body_q_now = state_0.body_q.numpy()
                outward_normals_world = np.zeros(
                    (cslc_data.n_spheres, 3), dtype=np.float64)
                for i in range(cslc_data.n_spheres):
                    if not is_surface_arr[i]:
                        continue
                    s_idx = int(sphere_shape_arr[i])
                    b_idx = int(shape_body_arr[s_idx])
                    X_ws = shape_transform_arr[s_idx]
                    X_wb = body_q_now[b_idx]
                    # Outward normal is a unit vector — transform as a
                    # vector (rotation only, ignore translation).
                    n_local = outward_normals_local_arr[i]
                    n_body = _xform_rotate(X_ws, n_local)
                    outward_normals_world[i] = _xform_rotate(X_wb, n_body)

            # delta projection along outward normal (per-sphere
            # penetration proxy).  Positive = sphere compressed INTO
            # the body (toward the cube) = active contact.
            delta_vec = cslc_data.sphere_delta.numpy()  # (n_spheres, 3)
            delta_n_per_sphere = np.einsum(
                "ij,ij->i", delta_vec, outward_normals_world)
            delta_n_sum += delta_n_per_sphere
            active_count += (delta_n_per_sphere > phi_active_threshold).astype(
                np.int64)
            n_hold_samples += 1
            # Capture final state for post-loop world-position reconstruction.
            final_body_q = state_0.body_q.numpy().copy()
            final_sphere_delta = delta_vec.copy()

        if verbose and ((step + 1) % 500 == 0 or step == total_steps - 1):
            print(f"  step={step + 1:5d}/{total_steps}  [{phase:8s}]")

    wall = time.perf_counter() - t0
    if verbose:
        print(f"  TIMING wall={wall:.2f}s  hold_samples={n_hold_samples}")

    if n_hold_samples == 0:
        raise RuntimeError(
            f"No HOLD samples collected.  hold_sample_start = "
            f"{hold_sample_start} >= total_steps = {total_steps}.  "
            f"Reduce --hold-skip-s or extend hold_duration.")

    # Per-sphere time-averaged delta-along-normal.  Use only spheres
    # that were active in at least half the HOLD samples — guards
    # against spheres that briefly engage during transient HOLD
    # oscillations.
    n_min_active = max(1, n_hold_samples // 2)
    is_persistently_active = active_count >= n_min_active

    # L_delta_n: mean delta_n over persistently-active surface spheres.
    delta_n_mean = np.where(active_count > 0,
                            delta_n_sum / np.maximum(active_count, 1), 0.0)
    active_surface = is_persistently_active & is_surface_arr
    L_delta_n = (float(np.mean(delta_n_mean[active_surface]))
                 if active_surface.any() else 0.0)

    # L_phi_eff: contact penetration depth derived from L_delta_n via
    # the three-spring series identity at face-on contact equilibrium.
    # See script docstring "L protocol" section: at equilibrium
    #     delta_n = phi_eff · kc_series / (ka + kc_series)
    # so phi_eff = delta_n · (ka + kc_series) / kc_series.
    # The §3.3 calibration formula `ke_pad_physical = E·A/L` and Drake
    # `kh = E/L` use L = phi_eff (contact penetration depth, not
    # lattice sphere displacement), so downstream calibration should
    # consume L_phi_eff, not L_delta_n.  The script reports both so
    # the protocol choice is transparent.
    kc = float(cslc_data.kc)
    ka = float(cslc_data.ka)
    ke_target = float(config.material.ke_target_constraint)
    kc_series = (kc * ke_target) / (kc + ke_target) if (kc + ke_target) > 0 else 0.0
    if kc_series > 0:
        phi_eff_factor = (ka + kc_series) / kc_series
    else:
        phi_eff_factor = 1.0  # degenerate; fall back to delta_n
    L_phi_eff = L_delta_n * phi_eff_factor

    # Per-pad active surface masks.
    active_left = is_persistently_active & is_left_pad
    active_right = is_persistently_active & is_right_pad

    # A_patch: convex hull of active sphere world positions (from
    # FINAL HOLD frame) projected onto the box face plane.  For the
    # box-target scene the held object is axis-aligned, the face normals
    # are ±x, so projecting onto the y-z plane recovers the patch area.
    def _world_positions_for(mask: np.ndarray) -> np.ndarray:
        idxs = np.where(mask)[0]
        pts = np.zeros((len(idxs), 3), dtype=np.float64)
        for k, i in enumerate(idxs):
            s_idx = int(sphere_shape_arr[i])
            b_idx = int(shape_body_arr[s_idx])
            X_ws = shape_transform_arr[s_idx]
            X_wb = final_body_q[b_idx]
            p_local = sphere_pos_local_arr[i]
            pts[k] = _xform_apply(X_wb, _xform_apply(X_ws, p_local))
        return pts

    pts_left = _world_positions_for(active_left)
    pts_right = _world_positions_for(active_right)
    A_patch_left = _convex_hull_area_2d(pts_left[:, 1:3])  # project to (y, z)
    A_patch_right = _convex_hull_area_2d(pts_right[:, 1:3])

    # Contact fractions (time-averaged ratio of active per-frame to
    # n_surface per pad).
    active_left_per_frame = float(active_count[is_left_pad].sum() / n_hold_samples)
    active_right_per_frame = float(active_count[is_right_pad].sum() / n_hold_samples)
    contact_fraction_left = (active_left_per_frame / n_surface_left
                             if n_surface_left else 0.0)
    contact_fraction_right = (active_right_per_frame / n_surface_right
                              if n_surface_right else 0.0)

    # Forces.
    F_left = float(np.mean(f_left_samples))
    F_right = float(np.mean(f_right_samples))
    F_per_pad = 0.5 * (F_left + F_right)
    # Newton III check: signed sum should be ≈ 0 at static equilibrium.
    f_l_signed = float(np.mean(f_left_signed_samples))
    f_r_signed = float(np.mean(f_right_signed_samples))
    newton_iii_residual = abs(f_l_signed + f_r_signed) / max(F_per_pad, 1e-30)

    return {
        # Headline anchors (matches §7.1 schema).  L is the contact
        # penetration depth = phi_eff (use this in the §3.3 E·A/L
        # calibration formula).
        "L": L_phi_eff,
        "L_delta_n": L_delta_n,
        "L_phi_eff": L_phi_eff,
        "phi_eff_conversion_factor": phi_eff_factor,
        "phi_eff_conversion_provenance": (
            f"(ka + kc_series) / kc_series  with  ka = {ka:.3e}, "
            f"kc = {kc:.3e}, ke_target = {ke_target:.3e}, "
            f"kc_series = {kc_series:.3e}"),
        "A_patch_left": A_patch_left,
        "A_patch_right": A_patch_right,
        "A_patch_avg": 0.5 * (A_patch_left + A_patch_right),
        "contact_fraction_left": contact_fraction_left,
        "contact_fraction_right": contact_fraction_right,
        "F_left": F_left,
        "F_right": F_right,
        "F_per_pad": F_per_pad,
        # Per-pad active surface counts (in the final HOLD frame).
        "n_active_left_final": int(active_left.sum()),
        "n_active_right_final": int(active_right.sum()),
        "n_surface_left": n_surface_left,
        "n_surface_right": n_surface_right,
        # Protocol provenance — see script docstring for full
        # explanation.  Briefly: ``L_delta_n`` is the measured proxy
        # (mean projection of sphere_delta onto outward normal across
        # HOLD samples); ``L_phi_eff`` is the derived contact
        # penetration depth, scaled by the three-spring series factor
        # ``phi_eff_conversion_factor``.  The §3.3 calibration formula
        # consumes ``L_phi_eff`` (contact penetration depth, Drake
        # convention).
        "L_method": "delta_n_mean_HOLD_window_with_series_correction",
        # Sampling / quality diagnostics.
        "n_hold_samples": n_hold_samples,
        "hold_start_step": int(hold_start_step),
        "hold_sample_start": int(hold_sample_start),
        "newton_iii_residual_fraction": newton_iii_residual,
        # Provenance.
        "config_pad_kind": config.pad.kind,
        "config_pad_r_pad": float(config.pad.dome_param_R_pad),
        "config_pad_half_angle_deg": float(
            config.pad.dome_param_half_angle * 180.0 / math.pi),
        "config_box_side": float(2.0 * min(config.object.box_half_extents)),
        "config_object_density": float(config.object.density),
        "config_pad_close_offset": float(
            config.timing.squeeze_speed * config.timing.squeeze_duration),
        "config_ke_pad_physical": float(config.material.ke_pad_physical),
        "config_ke_target_constraint": float(config.material.ke_target_constraint),
        "config_mu": float(config.material.mu),
        "config_cslc_alpha": float(config.cslc.alpha),
        "config_cslc_n_iter": int(config.cslc.n_iter),
        "wall_seconds": wall,
    }


# ── CLI ─────────────────────────────────────────────────────────────────


def _build_config_from_args(args: argparse.Namespace) -> GraspConfig:
    """Apply anchor-script CLI args onto a fresh GraspConfig."""
    cfg = GraspConfig()
    cfg.contact_model = "cslc"
    cfg.pad.kind = "dome_param"
    cfg.pad.dome_param_R_pad = float(args.pad_r_pad)
    cfg.pad.dome_param_half_angle = float(args.pad_half_angle) * math.pi / 180.0
    cfg.object.kind = "box"
    half = float(args.box_side) * 0.5
    cfg.object.box_half_extents = (half, half, half)
    cfg.object.density = float(args.object_density)
    cfg.object.spawn_y_offset = float(args.spawn_y_offset)
    if args.ke_physical is not None:
        cfg.material.ke_pad_physical = float(args.ke_physical)
    if args.ke_constraint is not None:
        cfg.material.ke_target_constraint = float(args.ke_constraint)
    if args.mu is not None:
        cfg.material.mu = float(args.mu)
    # CSLC solver tuning -- benchmark §3.6 recommends alpha=0.3 and
    # n_iter=40 for the box-target scene (post-Bug-B, notes.md C2
    # closure).  GraspConfig defaults are the production CSLC values
    # (alpha=0.6, n_iter=20) which work for dome+sphere but are
    # marginal at the box-target ke regime; override here unless the
    # user explicitly passes them.
    cfg.cslc.alpha = (float(args.cslc_alpha) if args.cslc_alpha is not None
                      else 0.3)
    cfg.cslc.n_iter = (int(args.cslc_n_iter) if args.cslc_n_iter is not None
                       else 40)
    if args.cslc_contact_fraction is not None:
        cfg.cslc.contact_fraction = float(args.cslc_contact_fraction)
    # Pad-close-offset override -- maps to squeeze_speed so the
    # commanded SQUEEZE depth matches the requested offset (with the
    # default 0.5 s squeeze_duration).  Used by the §7.4 F(d) sweep.
    if args.pad_close_offset is not None:
        cfg.timing.squeeze_speed = (
            float(args.pad_close_offset) / cfg.timing.squeeze_duration)
    cfg.logging.use_timestamp = False
    cfg.logging.run_label = args.run_label or (
        f"_anchors_R{int(round(args.pad_r_pad * 1000))}mm"
        f"_a{int(round(args.pad_half_angle))}deg"
        f"_box{int(round(args.box_side * 1000))}mm"
        f"_rho{int(round(args.object_density))}"
    )
    # The anchor measurement script doesn't need plots; keep the run dir
    # lean.
    cfg.logging.save_postsim_plots = False
    cfg.logging.save_lattice_preview = False
    return cfg


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure empirical anchors for the CSLC benchmark spec.")
    p.add_argument("--pad-r-pad", type=float, default=0.010,
                   help="Dome cap radius [m] (default 0.010 = v0.8 corrected).")
    p.add_argument("--pad-half-angle", type=float, default=72.0,
                   help="Dome cap half-angle [deg] (default 72).")
    p.add_argument("--box-side", type=float, default=0.025,
                   help="Cube full side length [m] (default 0.025).")
    p.add_argument("--object-density", type=float, default=7800.0,
                   help="Cube density [kg/m^3] (default 7800 = steel).")
    p.add_argument("--spawn-y-offset", type=float, default=0.0,
                   help="Lateral spawn jitter [m] for seed-sweep "
                        "statistics (default 0 = centred).")
    p.add_argument("--ke-physical", type=float, default=None,
                   help="Override material.ke_pad_physical [N/m].  Default: "
                        "GraspConfig default.  For v0.7-anchored runs pass "
                        "1.38e6; for v0.8 re-anchor pass an a-priori estimate "
                        "and re-run with the measured ke_pad_physical = E*A/L.")
    p.add_argument("--ke-constraint", type=float, default=None,
                   help="Override material.ke_target_constraint [N/m].  "
                        "Default: GraspConfig default (5e5 in v0.7).")
    p.add_argument("--mu", type=float, default=None,
                   help="Override material.mu (Coulomb friction).  "
                        "Default: GraspConfig default (0.5 in v0.7).")
    p.add_argument("--cslc-alpha", type=float, default=None,
                   help="Override damped-Jacobi damping factor alpha.  "
                        "Default: 0.3 (benchmark §3.6 / Bug-B closure).")
    p.add_argument("--cslc-n-iter", type=int, default=None,
                   help="Override damped-Jacobi iteration count.  "
                        "Default: 40 (benchmark §3.6 / Bug-B closure).")
    p.add_argument("--cslc-contact-fraction", type=float, default=None,
                   help="Override CSLCParams.contact_fraction (the prior "
                        "used by calibrate_kc to derive per-sphere kc).  "
                        "Default: GraspConfig default (0.025).  For v0.9 "
                        "self-consistency iteration set this to the "
                        "empirical contact_fraction measured in the prior "
                        "exp_anchors run (~0.7 at silicone target on "
                        "R_pad=10mm box-target scene) and iterate "
                        "ke_pad_physical until both stabilize.")
    p.add_argument("--pad-close-offset", type=float, default=None,
                   help="Override the commanded squeeze depth [m].  "
                        "Maps to ``squeeze_speed = offset / "
                        "squeeze_duration`` (squeeze_duration stays at "
                        "the GraspConfig default of 0.5 s).  Default: "
                        "1 mm (GraspConfig default speed × duration).  "
                        "Use this for the §7.4 F(d) sweep at "
                        "{0.2, 0.5, 1.0, 2.0, 5.0} mm depths.")
    p.add_argument("--hold-skip-s", type=float, default=0.5,
                   help="Seconds to skip after HOLD start before sampling "
                        "(default 0.5 — lets squeeze transients settle).")
    p.add_argument("--phi-active-threshold", type=float, default=1.0e-6,
                   help="Minimum mean phi_eff [m] to count a sphere as "
                        "active (default 1e-6).")
    p.add_argument("--output", required=True,
                   help="Path to write the anchors JSON.")
    p.add_argument("--run-label", default=None,
                   help="Override the auto-generated run-directory label.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-step progress prints.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    wp.init()
    print(f"\n{'━' * 60}\n  cslc_main.grasp.scripts.exp_anchors\n{'━' * 60}")
    print(f"  pad_r_pad           : {args.pad_r_pad * 1000:.1f} mm")
    print(f"  pad_half_angle      : {args.pad_half_angle:.1f} deg")
    print(f"  box_side            : {args.box_side * 1000:.1f} mm")
    print(f"  object_density      : {args.object_density:.0f} kg/m^3")
    print(f"  hold_skip_s         : {args.hold_skip_s:.2f} s")
    print(f"  phi_active_threshold: {args.phi_active_threshold:.1e} m")
    print(f"  output              : {args.output}")
    config = _build_config_from_args(args)
    anchors = measure_anchors(
        config,
        hold_skip_s=args.hold_skip_s,
        phi_active_threshold=args.phi_active_threshold,
        verbose=not args.quiet,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(anchors, indent=2, sort_keys=True))

    print()
    print(f"  ─── ANCHORS ({out_path}) ───")
    print(f"  L_phi_eff (used in §3.3 calibration) = "
          f"{anchors['L_phi_eff'] * 1000:8.3f} mm")
    print(f"  L_delta_n (raw measurement proxy)    = "
          f"{anchors['L_delta_n'] * 1000:8.3f} mm")
    print(f"    [phi_eff_conversion_factor = "
          f"{anchors['phi_eff_conversion_factor']:.3f}]")
    print(f"  A_patch_left     = {anchors['A_patch_left'] * 1e6:8.1f} mm^2")
    print(f"  A_patch_right    = {anchors['A_patch_right'] * 1e6:8.1f} mm^2")
    print(f"  A_patch_avg      = {anchors['A_patch_avg'] * 1e6:8.1f} mm^2")
    print(f"  contact_fraction = {anchors['contact_fraction_left']:.3f} (L) / "
          f"{anchors['contact_fraction_right']:.3f} (R)")
    print(f"  F_left           = {anchors['F_left']:8.2f} N")
    print(f"  F_right          = {anchors['F_right']:8.2f} N")
    print(f"  F_per_pad        = {anchors['F_per_pad']:8.2f} N")
    print(f"  Newton-III resid = {anchors['newton_iii_residual_fraction'] * 100:6.2f} %")
    print(f"  HOLD samples     = {anchors['n_hold_samples']}")
    print()
    # If E and L are both known, derive ke_pad_physical for the
    # calibration table — convenience output.  Uses L_phi_eff (the
    # contact penetration depth), matching the §3.3 Drake convention
    # for elastic-foundation depth.
    E_guess = 5.0e5  # silicone-target Young's modulus from §3.1
    A = anchors["A_patch_avg"]
    L_calib = anchors["L_phi_eff"]
    if A > 0 and L_calib > 0:
        ke_phys_derived = E_guess * A / L_calib
        kh_derived = E_guess / L_calib
        print(f"  ─── DERIVED (with E = {E_guess:.1e} Pa silicone target, "
              f"L = L_phi_eff) ───")
        print(f"  ke_pad_physical = E*A/L = {ke_phys_derived:9.3e} N/m")
        print(f"  kh              = E/L   = {kh_derived:9.3e} Pa/m")
        print(f"  -> Re-run with --ke-physical {ke_phys_derived:.3e} "
              f"if this differs from input by more than ~20%.")


if __name__ == "__main__":
    main()
