# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quasi-static F-vs-δ sweep across {cslc, hydro, point} contact models.

Step 1 of the two-step model-comparison framework: characterise each
contact model's normal-force-vs-penetration curve so we can later pick
an operating point F* and tune all three models to match it.

Per-run procedure
-----------------
For each (contact_model, pad_kind, object_kind, face_pen) cell:

  1. Build a GraspConfig with a stripped trajectory:
       APPROACH (0.5 s @ 10 mm/s) closes a 5 mm gap.
       SQUEEZE  (face_pen / 2 mm·s⁻¹) pushes to commanded depth.
       LIFT skipped (lift_speed = 0).
       HOLD     (200 ms) lets the PD + contact dynamics settle.
  2. Run the sim step-by-step, reading ``state.mujoco.qfrc_actuator``
     on the left-pad X-DOF over the last 50 ms.
  3. Average that as F_n (Newton's 3rd law: the actuator must supply
     the same magnitude as the contact reaction along X).

The two pads are symmetric, so the object stays on the centreline and
the commanded face_pen equals the geometric depth (modulo PD compliance,
which is negligible at the default 5e4 N/m gain × ~25 g pad mass —
overdamped, settles in a few ms).

Output: one CSV row per (model, pad, object, face_pen) with F_n_mean,
F_n_std, n_contacts_mean, wall_s, and the primary-stiffness value used.
Downstream consumers plot F vs δ overlays and pick F* for step 2.

Usage::

    uv run -m cslc_main.grasp.scripts.sweep_force_curve \\
        --models cslc,hydro,point \\
        --face-pens-mm 0.2,0.5,1.0,1.5,2.0 \\
        --output outputs/sweep/force_curve_default.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

from ..logger import count_active_contacts
from ..params import GraspConfig
from ..runner import _attach_qfrc_actuator, _simulate_one_step
from ..scene import build_scene
from ..solvers import make_solver


@contextlib.contextmanager
def _maybe_silence(quiet: bool):
    """Suppress stdout from the scene-build path when ``quiet`` is True.

    ``build_scene`` and ``build_cslc_handler_with_mesh_pads`` print
    diagnostic lines unconditionally; the sweep loop runs hundreds of
    builds and the chatter drowns the per-cell summary.
    """
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


# ── Per-run trajectory + config builder ────────────────────────────────


def _build_sweep_config(
    contact_model: str,
    pad_kind: str,
    object_kind: str,
    face_pen_mm: float,
    *,
    primary_stiffness: float | None,
    settling_ms: float,
    smoothing_eps: float | None = None,
    ka: float | None = None,
    pad_n_samples: int | None = None,
    box_face_pitch: float | None = None,
) -> GraspConfig:
    """GraspConfig with a stripped APPROACH→SQUEEZE→HOLD trajectory.

    LIFT is reduced to a single-step no-op (``lift_speed=0``,
    ``lift_duration=dt``) so the pad stays at the equator after SQUEEZE
    and HOLD runs immediately.  All logging side effects (PNG previews,
    post-sim plots) are disabled so per-run cost is dominated by the
    physics step, not by matplotlib.
    """
    config = GraspConfig()
    config.contact_model = contact_model
    config.pad.kind = pad_kind
    config.object.kind = object_kind

    # ── Primary-stiffness override per model ──
    if primary_stiffness is not None:
        if contact_model == "cslc":
            config.cslc.kc_per_volume = primary_stiffness
        elif contact_model == "hydro":
            # Override only the pad side; the object's kh_object stays
            # at its default (5e9 Pa/m).  Caller wanting an asymmetric
            # rigid-object setup overrides kh_object via a separate
            # config tweak before calling.
            config.material.kh_pad = primary_stiffness
        elif contact_model == "point":
            config.material.ke_pad_physical = primary_stiffness
        else:
            raise ValueError(f"Unknown contact_model={contact_model!r}")

    # ── CSLC-only smoothing_eps override (Task B: characterize shelf) ──
    if smoothing_eps is not None and contact_model == "cslc":
        config.cslc.smoothing_eps = smoothing_eps

    # ── CSLC-only anchor-stiffness override (secondary force lever) ──
    if ka is not None and contact_model == "cslc":
        config.cslc.ka = ka

    # ── Resolution overrides for the pair-count study ──
    if pad_n_samples is not None:
        config.pad.n_samples = pad_n_samples
    if box_face_pitch is not None:
        config.object.box_face_pitch = box_face_pitch

    # ── Trajectory shape ──
    # 5 mm gap closed by a brief APPROACH; depth-dependent SQUEEZE;
    # LIFT skipped; brief HOLD for settling + averaging.
    dt = config.timing.dt
    config.pad.approach_gap = 0.005          # 5 mm
    config.timing.approach_speed = 0.010     # 10 mm/s → 5 mm in 0.5 s
    config.timing.approach_duration = 0.5
    config.timing.squeeze_speed = 0.002      # 2 mm/s (matches production default)
    config.timing.squeeze_duration = max(face_pen_mm * 1e-3 / 0.002, dt)
    config.timing.lift_speed = 0.0
    config.timing.lift_ramp_duration = 0.0
    config.timing.lift_duration = dt         # 1 step at dt — effectively skipped
    config.timing.hold_duration = settling_ms / 1000.0

    # ── Logging side effects off ──
    config.logging.save_lattice_preview = False
    config.logging.save_postsim_plots = False
    config.logging.use_timestamp = False
    config.logging.run_label = (
        f"sweep_{contact_model}_{pad_kind}_{object_kind}_{face_pen_mm:.2f}mm"
    )

    return config


# ── Per-run measurement ────────────────────────────────────────────────


def measure_F_at_depth(
    contact_model: str,
    pad_kind: str,
    object_kind: str,
    face_pen_mm: float,
    *,
    primary_stiffness: float | None = None,
    smoothing_eps: float | None = None,
    ka: float | None = None,
    pad_n_samples: int | None = None,
    box_face_pitch: float | None = None,
    settling_ms: float = 200.0,
    avg_window_ms: float = 50.0,
    quiet: bool = True,
) -> dict:
    """Run one short sim, return steady-state normal force on the pad.

    Args:
        contact_model: ``cslc`` | ``hydro`` | ``point``.
        pad_kind: ``box`` | ``dome`` | ``dome_param``.
        object_kind: ``sphere`` | ``box``.
        face_pen_mm: commanded SQUEEZE-end face penetration [mm].
        primary_stiffness: override for the model's primary stiffness
            knob (``kc_per_volume`` / ``kh_pad`` / ``ke_pad_physical``).
            ``None`` uses the GraspConfig default.
        settling_ms: HOLD duration to let PD + contact dynamics settle.
        avg_window_ms: trailing window over which F_n is averaged.
        quiet: suppress the per-run scene-build prints.

    Returns:
        dict with model identifiers, F_n_mean / F_n_std [N],
        n_contacts_mean, the achieved pad-X position, the object-X
        drift (sanity check for symmetric stay-put), and wall_s.
    """
    config = _build_sweep_config(
        contact_model, pad_kind, object_kind, face_pen_mm,
        primary_stiffness=primary_stiffness, settling_ms=settling_ms,
        smoothing_eps=smoothing_eps, ka=ka,
        pad_n_samples=pad_n_samples, box_face_pitch=box_face_pitch,
    )

    with _maybe_silence(quiet):
        artifacts = build_scene(config)
        model = artifacts.model
        solver = make_solver(model, config.solver)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        _attach_qfrc_actuator((state_0, state_1), model)

        dt = config.timing.dt
        n_total = config.total_steps
        n_window = max(int(round(avg_window_ms / 1000.0 / dt)), 1)

        left_x_dof = artifacts.dof_map["left_x"]
        right_x_dof = artifacts.dof_map["right_x"]
        object_idx = artifacts.object_body_index
        left_pad_idx = artifacts.pad_body_indices["left"]

        F_left_window: list[float] = []
        F_right_window: list[float] = []
        n_contacts_window: list[int] = []

        # Warm-up step (kernel JIT + solver setup) — discarded.
        _simulate_one_step(
            model, solver, control, contacts,
            state_0, state_1, 0, config, artifacts.dof_map,
        )
        wp.synchronize()

        t0 = time.perf_counter()
        for step in range(n_total):
            state_0, state_1, _, _ = _simulate_one_step(
                model, solver, control, contacts,
                state_0, state_1, step, config, artifacts.dof_map,
            )
            if step >= n_total - n_window:
                qfrc = state_0.mujoco.qfrc_actuator.numpy()
                F_left_window.append(float(qfrc[left_x_dof]))
                F_right_window.append(float(qfrc[right_x_dof]))
                n_contacts_window.append(count_active_contacts(contacts))
        wp.synchronize()
        wall_s = time.perf_counter() - t0

        # Sanity-check positions (post-loop, single read).
        q = state_0.body_q.numpy()
        object_x_final = float(q[object_idx, 0])
        object_y_final = float(q[object_idx, 1])
        # Left pad's world X — diagnostic for PD tracking.  The pad
        # joint commands +dx_inward (positive X for the left pad,
        # which spawns at -x_spawn); world-X = -x_spawn + dx.
        pad_x_world_left = float(q[left_pad_idx, 0])

    # By symmetry, +F_left should equal -F_right (both pads pushing
    # inward; left pushes in +x, right pushes in -x; reaction on each
    # is opposite, so the actuator torques the two sides oppositely
    # in sign).  Use the magnitude average as F_n.
    F_left = np.array(F_left_window)
    F_right = np.array(F_right_window)
    F_n_mag = 0.5 * (np.abs(F_left) + np.abs(F_right))
    F_n_mean = float(np.mean(F_n_mag))
    F_n_std = float(np.std(F_n_mag))
    F_asymmetry = float(np.mean(np.abs(F_left + F_right)))  # ideally 0
    n_contacts_mean = float(np.mean(n_contacts_window))

    return {
        "contact_model": contact_model,
        "pad_kind": pad_kind,
        "object_kind": object_kind,
        "face_pen_mm": face_pen_mm,
        "F_n_mean": F_n_mean,
        "F_n_std": F_n_std,
        "F_lr_asymmetry": F_asymmetry,
        "n_contacts_mean": n_contacts_mean,
        "object_x_final": object_x_final,
        "object_y_final": object_y_final,
        "pad_x_world_left": pad_x_world_left,
        "wall_s": wall_s,
        "primary_stiffness": primary_stiffness if primary_stiffness is not None else float("nan"),
        "smoothing_eps": smoothing_eps if smoothing_eps is not None else float("nan"),
        "ka": ka if ka is not None else float("nan"),
    }


# ── CLI loop ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="cslc,hydro,point",
                        help="Comma-separated contact models")
    parser.add_argument("--pad-kinds", default="box",
                        help="Comma-separated pad kinds")
    parser.add_argument("--object-kinds", default="sphere",
                        help="Comma-separated object kinds")
    parser.add_argument("--face-pens-mm", default="0.2,0.5,1.0,1.5,2.0",
                        help="Comma-separated penetration depths [mm]")
    parser.add_argument("--pad-n-samples", default=None,
                        help="Comma-separated pad.n_samples values to sweep "
                             "(partition-of-unity invariance test).")
    parser.add_argument("--primary-stiffness", default=None,
                        help="Comma-separated primary-stiffness values for the "
                             "selected model(s).  CSLC: kc_per_volume [N/m^3.5]; "
                             "hydro: kh_pad [Pa/m]; point: ke_pad_physical [N/m]. "
                             "Sweeps as an outer-most loop (one cell per value).")
    parser.add_argument("--smoothing-eps-m", default=None,
                        help="Comma-separated CSLC smoothing_eps values [m] "
                             "to sweep (e.g. '1e-5,1e-4,5e-4').  Inner-most "
                             "loop; ignored for hydro/point cells.  If "
                             "omitted, uses the CSLCParams default.")
    parser.add_argument("--ka-list", default=None,
                        help="Comma-separated CSLC anchor-stiffness ``ka`` "
                             "values [N/m] to sweep (e.g. '3500,10000,35000'). "
                             "Inner-most loop; ignored for hydro/point cells. "
                             "If omitted, uses the CSLCParams default (35000).")
    parser.add_argument("--settling-ms", type=float, default=200.0,
                        help="HOLD duration for settling [ms]")
    parser.add_argument("--avg-window-ms", type=float, default=50.0,
                        help="Trailing window over which F_n is averaged [ms]")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV path")
    parser.add_argument("--verbose", action="store_true",
                        help="Leave per-run scene-build prints on")
    args = parser.parse_args()

    wp.init()

    models = [m.strip() for m in args.models.split(",")]
    pad_kinds = [p.strip() for p in args.pad_kinds.split(",")]
    object_kinds = [o.strip() for o in args.object_kinds.split(",")]
    face_pens = [float(x) for x in args.face_pens_mm.split(",")]
    primary_stiffness_list: list[float | None]
    if args.primary_stiffness is None:
        primary_stiffness_list = [None]
    else:
        primary_stiffness_list = [float(x) for x in args.primary_stiffness.split(",")]
    pad_n_samples_list: list[int | None]
    if args.pad_n_samples is None:
        pad_n_samples_list = [None]
    else:
        pad_n_samples_list = [int(x) for x in args.pad_n_samples.split(",")]
    smoothing_eps_list: list[float | None]
    if args.smoothing_eps_m is None:
        smoothing_eps_list = [None]
    else:
        smoothing_eps_list = [float(x) for x in args.smoothing_eps_m.split(",")]
    ka_list: list[float | None]
    if args.ka_list is None:
        ka_list = [None]
    else:
        ka_list = [float(x) for x in args.ka_list.split(",")]

    cells = list(itertools.product(
        models, pad_kinds, object_kinds, face_pens,
        primary_stiffness_list, smoothing_eps_list, ka_list,
        pad_n_samples_list,
    ))
    print(f"Sweeping {len(cells)} cells "
          f"({len(models)} models × {len(pad_kinds)} pads × "
          f"{len(object_kinds)} objects × {len(face_pens)} depths × "
          f"{len(primary_stiffness_list)} kc × "
          f"{len(smoothing_eps_list)} smoothing_eps × {len(ka_list)} ka × "
          f"{len(pad_n_samples_list)} pad_n_samples)")

    results: list[dict] = []
    sweep_t0 = time.perf_counter()
    for i, (model, pad, obj, pen, kc, eps, ka, pad_n) in enumerate(cells, 1):
        kc_label = f" kc={kc:.0e}" if kc is not None else ""
        eps_label = f" eps={eps:.0e}" if eps is not None else ""
        ka_label = f" ka={ka:.0f}" if ka is not None else ""
        pad_n_label = f" pad_n={pad_n}" if pad_n is not None else ""
        print(f"  [{i}/{len(cells)}] {model:5s} pad={pad:10s} "
              f"object={obj:6s} pen={pen:.2f}mm{kc_label}{eps_label}{ka_label}{pad_n_label}", flush=True)
        try:
            result = measure_F_at_depth(
                model, pad, obj, pen,
                primary_stiffness=kc,
                smoothing_eps=eps,
                ka=ka,
                pad_n_samples=pad_n,
                settling_ms=args.settling_ms,
                avg_window_ms=args.avg_window_ms,
                quiet=not args.verbose,
            )
        except Exception as e:
            print(f"     !! FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            result = {
                "contact_model": model,
                "pad_kind": pad,
                "object_kind": obj,
                "face_pen_mm": pen,
                "F_n_mean": float("nan"),
                "F_n_std": float("nan"),
                "F_lr_asymmetry": float("nan"),
                "n_contacts_mean": float("nan"),
                "object_x_final": float("nan"),
                "object_y_final": float("nan"),
                "pad_x_world_left": float("nan"),
                "wall_s": float("nan"),
                "primary_stiffness": kc if kc is not None else float("nan"),
                "smoothing_eps": eps if eps is not None else float("nan"),
                "ka": ka if ka is not None else float("nan"),
                "pad_n_samples": pad_n if pad_n is not None else float("nan"),
                "error": f"{type(e).__name__}: {e}",
            }
        else:
            print(f"     F_n={result['F_n_mean']:.3f} ± {result['F_n_std']:.3f} N  "
                  f"n_c={result['n_contacts_mean']:.1f}  "
                  f"obj_drift_x={result['object_x_final']*1e3:+.2f}mm  "
                  f"wall={result['wall_s']:.2f}s")
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Union of all keys across results (some may have 'error', others not).
    fieldnames: list[str] = []
    for r in results:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    total_wall = time.perf_counter() - sweep_t0
    print(f"\nWrote {len(results)} rows to {args.output} "
          f"({total_wall:.1f}s total)")


if __name__ == "__main__":
    main()
