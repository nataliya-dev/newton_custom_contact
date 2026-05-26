# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""F(d) pilot driver across {CSLC, hydro, point} contact models.

Drives the grasp pipeline at a configurable pad_close_offset for one
of the three benchmark contact models and reports per-pad steady-
state force + grip outcome.  Used by benchmark §7 Block B to bracket
the grip-transition zone of each model so the headline sweep range
covers all three transitions.

Outputs one JSON per (model, depth) cell with:
  - F_per_pad     (mean |qfrc_actuator| at left/right pad x-DOF over
                   the HOLD sampling window, post-skip)
  - F_left, F_right
  - newton_iii_residual_fraction
  - lifted, held  (binary, computed from final object z)
  - max_z, final_z, xy_slip_max
  - per-step n_contacts (mean over HOLD)

A CSLC-specific extension would re-add per-sphere L/A_patch/etc, but
those are exp_anchors.py's job; this script is intentionally narrow.

Examples::

    # Single (model, depth) cell.
    uv run --extra importers -m cslc_main.grasp.scripts.exp_fd_pilot \\
        --contact-model hydro --pad-close-offset 0.001 \\
        --output /tmp/fd_pilot_hydro_1mm.json

    # Full sweep (5 depths x 3 models = 15 cells; ~7.5 min wall).
    for model in cslc hydro point; do
      for d in 0.0001 0.0002 0.0005 0.001 0.002; do
        uv run --extra importers -m cslc_main.grasp.scripts.exp_fd_pilot \\
            --contact-model $model --pad-close-offset $d \\
            --output /tmp/fd_pilot_${model}_${d}.json --quiet
      done
    done
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

from cslc_main.grasp.params import GraspConfig
from cslc_main.grasp.runner import _attach_qfrc_actuator, _simulate_one_step
from cslc_main.grasp.scene import build_scene
from cslc_main.grasp.solvers import make_solver


# ── Pilot measurement protocol ──────────────────────────────────────────


def run_pilot(config: GraspConfig, *,
              hold_skip_s: float = 0.5,
              verbose: bool = True) -> dict:
    """Run one grasp pipeline cycle; return F + grip outcome.

    Generic across CSLC / hydro / point — does NOT touch any
    contact-model-specific handler.  Reads wrench from
    ``state.mujoco.qfrc_actuator`` (pre-allocated by runner's
    ``_attach_qfrc_actuator``) and metrics from final object z.
    """
    artifacts = build_scene(config)
    model = artifacts.model
    solver = make_solver(model, config.solver)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    _attach_qfrc_actuator((state_0, state_1), model)

    t = config.timing
    hold_start_step = int(
        (t.approach_duration + t.squeeze_duration + t.lift_duration) / t.dt)
    hold_sample_start = hold_start_step + int(hold_skip_s / t.dt)
    total_steps = config.total_steps

    # Warm-up (JIT).
    _simulate_one_step(
        model, solver, control, contacts,
        state_0, state_1, 0, config, artifacts.dof_map)
    wp.synchronize()

    obj_body = artifacts.object_body_index
    z_settled: float | None = None
    obj_xyz_history: list[tuple[float, float, float]] = []
    f_left_samples: list[float] = []
    f_right_samples: list[float] = []
    f_left_signed: list[float] = []
    f_right_signed: list[float] = []
    n_contacts_samples: list[int] = []

    t0 = time.perf_counter()
    for step in range(total_steps):
        state_0, state_1, _, _ = _simulate_one_step(
            model, solver, control, contacts,
            state_0, state_1, step, config, artifacts.dof_map,
        )
        phase, _ = config.phase_of(step)

        if phase == "APPROACH" and step == 200:
            # Capture settled-on-ground z as the "before grasp" reference.
            q = state_0.body_q.numpy()
            z_settled = float(q[obj_body, 2])

        q = state_0.body_q.numpy()
        obj_xyz_history.append(
            (float(q[obj_body, 0]), float(q[obj_body, 1]), float(q[obj_body, 2])))

        if phase == "HOLD" and step >= hold_sample_start:
            q_act = state_0.mujoco.qfrc_actuator.numpy()
            f_l = float(q_act[artifacts.dof_map["left_x"]])
            f_r = float(q_act[artifacts.dof_map["right_x"]])
            f_left_samples.append(abs(f_l))
            f_right_samples.append(abs(f_r))
            f_left_signed.append(f_l)
            f_right_signed.append(f_r)
            n = int(contacts.rigid_contact_count.numpy()[0])
            n_contacts_samples.append(n)

        if verbose and ((step + 1) % 500 == 0 or step == total_steps - 1):
            print(f"  step={step + 1:5d}/{total_steps}  [{phase:8s}]")

    wall = time.perf_counter() - t0

    obj_xyz_arr = np.array(obj_xyz_history, dtype=np.float64)
    final_xyz = obj_xyz_arr[-1]
    max_z = float(obj_xyz_arr[:, 2].max())
    final_z = float(final_xyz[2])
    z_settled = z_settled if z_settled is not None else float(obj_xyz_arr[0, 2])
    # Same lifted/held criteria as Metrics (~50mm threshold per
    # production regression).
    lifted = max_z > z_settled + 0.04
    held = lifted and final_z > z_settled + 0.04
    # XY slip from spawn position.
    xy0 = obj_xyz_arr[0, :2]
    xy_slip = np.linalg.norm(obj_xyz_arr[:, :2] - xy0, axis=1)
    xy_slip_max = float(xy_slip.max())

    F_left = float(np.mean(f_left_samples)) if f_left_samples else 0.0
    F_right = float(np.mean(f_right_samples)) if f_right_samples else 0.0
    F_per_pad = 0.5 * (F_left + F_right)
    f_l_signed = (float(np.mean(f_left_signed)) if f_left_signed else 0.0)
    f_r_signed = (float(np.mean(f_right_signed)) if f_right_signed else 0.0)
    newton_iii_residual = (abs(f_l_signed + f_r_signed) / max(F_per_pad, 1e-30)
                           if F_per_pad > 0 else 0.0)
    n_contacts_mean = (float(np.mean(n_contacts_samples))
                       if n_contacts_samples else 0.0)

    return {
        "contact_model": config.contact_model,
        "pad_close_offset": float(
            config.timing.squeeze_speed * config.timing.squeeze_duration),
        "F_left": F_left,
        "F_right": F_right,
        "F_per_pad": F_per_pad,
        "newton_iii_residual_fraction": newton_iii_residual,
        "n_contacts_mean": n_contacts_mean,
        "lifted": bool(lifted),
        "held": bool(held),
        "max_z": max_z,
        "final_z": final_z,
        "z_settled": z_settled,
        "xy_slip_max": xy_slip_max,
        "n_hold_samples": len(f_left_samples),
        "hold_start_step": int(hold_start_step),
        "hold_sample_start": int(hold_sample_start),
        # Provenance.
        "config_pad_kind": config.pad.kind,
        "config_pad_r_pad": float(config.pad.dome_param_R_pad),
        "config_pad_half_angle_deg": float(
            config.pad.dome_param_half_angle * 180.0 / math.pi),
        "config_box_side": float(2.0 * min(config.object.box_half_extents)),
        "config_object_density": float(config.object.density),
        "config_ke_pad_physical": float(config.material.ke_pad_physical),
        "config_ke_target_physical": float(config.material.ke_target_physical),
        "config_kh": float(config.material.kh),
        "config_mu": float(config.material.mu),
        "wall_seconds": wall,
    }


# ── CLI ─────────────────────────────────────────────────────────────────


def _build_config_from_args(args: argparse.Namespace) -> GraspConfig:
    cfg = GraspConfig()
    cfg.contact_model = args.contact_model
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
        cfg.material.ke_target_physical = float(args.ke_constraint)
    if args.kh is not None:
        cfg.material.kh = float(args.kh)
    if args.mu is not None:
        cfg.material.mu = float(args.mu)
    cfg.cslc.alpha = (float(args.cslc_alpha) if args.cslc_alpha is not None
                      else 0.3)
    cfg.cslc.n_iter = (int(args.cslc_n_iter) if args.cslc_n_iter is not None
                       else 40)
    # MuJoCo solver settings.  The v0.9 spec §7.3 *initially* picked
    # (15, 100) from the nut_bolt_hydro example, but that example has
    # ~12 contacts at HOLD; our box-grasp scene has 40+ contacts on
    # hydro and 100+ on CSLC, and the MuJoCo CG outer loop doesn't
    # converge in 15 iterations at that density — hydro/point read
    # F=0 because the constraint forces never propagate to the joint
    # actuator.  Defaults are now CLI-overridable; when not passed,
    # GraspConfig defaults apply (auto: 100 with CSLC, 20 otherwise;
    # ls_iterations=10).  Pass --solver-iterations / --solver-ls-iterations
    # to explicitly compare.
    if args.solver_iterations is not None:
        cfg.solver.iterations = int(args.solver_iterations)
    if args.solver_ls_iterations is not None:
        cfg.solver.ls_iterations = int(args.solver_ls_iterations)
    cfg.solver.cone = args.solver_cone
    cfg.solver.integrator = args.solver_integrator
    if args.pad_close_offset is not None:
        cfg.timing.squeeze_speed = (
            float(args.pad_close_offset) / cfg.timing.squeeze_duration)
    cfg.logging.use_timestamp = False
    cfg.logging.save_postsim_plots = False
    cfg.logging.save_lattice_preview = False
    cfg.logging.run_label = args.run_label or (
        f"_fd_pilot_{args.contact_model}_"
        f"d{int(round(args.pad_close_offset * 1e6))}um")
    return cfg


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F(d) pilot driver across CSLC / hydro / point.")
    p.add_argument("--contact-model", choices=["cslc", "hydro", "point"],
                   required=True)
    p.add_argument("--pad-close-offset", type=float, required=True,
                   help="Commanded squeeze depth [m].")
    p.add_argument("--pad-r-pad", type=float, default=0.010)
    p.add_argument("--pad-half-angle", type=float, default=72.0)
    p.add_argument("--box-side", type=float, default=0.025)
    p.add_argument("--object-density", type=float, default=7800.0)
    p.add_argument("--spawn-y-offset", type=float, default=0.0)
    p.add_argument("--ke-physical", type=float, default=5.0e5,
                   help="material.ke_pad_physical [N/m]; default 5e5 (v0.9 anchor).")
    p.add_argument("--ke-constraint", type=float, default=5.0e5,
                   help="material.ke_target_physical [N/m]; default 5e5.")
    p.add_argument("--kh", type=float, default=1.8e9,
                   help="material.kh [Pa/m] for hydro; default 1.8e9 (v0.9 anchor).")
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--cslc-alpha", type=float, default=None)
    p.add_argument("--cslc-n-iter", type=int, default=None)
    p.add_argument("--solver-iterations", type=int, default=None,
                   help="MuJoCo CG outer iterations.  Default: GraspConfig auto "
                        "(100 with CSLC, 20 otherwise).")
    p.add_argument("--solver-ls-iterations", type=int, default=None,
                   help="MuJoCo CG line-search iterations per step.  "
                        "Default: GraspConfig default (10).")
    p.add_argument("--solver-cone", choices=["elliptic", "pyramidal"],
                   default="elliptic")
    p.add_argument("--solver-integrator", default="implicitfast")
    p.add_argument("--hold-skip-s", type=float, default=0.5)
    p.add_argument("--output", required=True)
    p.add_argument("--run-label", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    wp.init()
    print(f"\n{'━' * 60}")
    print(f"  exp_fd_pilot  {args.contact_model:5s}  "
          f"pad_close_offset = {args.pad_close_offset * 1000:.3f} mm")
    print(f"{'━' * 60}")
    config = _build_config_from_args(args)
    result = run_pilot(config, hold_skip_s=args.hold_skip_s, verbose=not args.quiet)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    grip_str = "HELD" if result["held"] else (
        "LIFTED-then-DROPPED" if result["lifted"] else "NEVER-LIFTED")
    print()
    print(f"  ─── PILOT ({out_path}) ───")
    print(f"  model            = {result['contact_model']}")
    print(f"  pad_close_offset = {result['pad_close_offset'] * 1000:.3f} mm")
    print(f"  F_per_pad        = {result['F_per_pad']:8.3f} N "
          f"(L={result['F_left']:.2f}, R={result['F_right']:.2f})")
    print(f"  Newton-III resid = {result['newton_iii_residual_fraction'] * 100:6.2f} %")
    print(f"  n_contacts (avg) = {result['n_contacts_mean']:.1f}")
    print(f"  outcome          = {grip_str}  "
          f"(max_z={result['max_z'] * 1000:.1f}mm, "
          f"final_z={result['final_z'] * 1000:.1f}mm)")
    print(f"  xy_slip_max      = {result['xy_slip_max'] * 1000:.2f} mm")


if __name__ == "__main__":
    main()
