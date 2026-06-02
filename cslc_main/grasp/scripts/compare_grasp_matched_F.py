# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 2 of the model-comparison framework: matched-F grasp dynamics.

Reads the calibrated stiffness CSV from :mod:`calibrate_stiffness` and
runs the full APPROACH→SQUEEZE→LIFT→HOLD pipeline for each
(contact_model, pad, object) cell at the calibrated primary stiffness.
Two calibration modes per cell are run:

  * ``symmetric``: pad and object both at the calibrated stiffness.
  * ``rigid_object``: object pinned at 1e12 (effectively rigid); pad
    at the calibrated value.  This is the "soft pad on rigid object"
    comparison.

Metrics extracted per run (the ones the user asked for):

  * ``held``        — did the grasp succeed?  (Metrics.held)
  * ``lifted``      — did the object actually leave the ground?
  * ``max_z``       — peak overshoot during LIFT
  * ``final_z``     — settling position
  * ``xy_slip_max`` — maximum XY drift during the run
  * ``tail_sigma_z`` — std of object Z over the last half of HOLD
                       (residual oscillation; lower = more stable)
  * ``n_active_hold`` — mean active contacts during HOLD
  * ``wall_s``      — per-run wall-clock time

Usage::

    uv run -m cslc_main.grasp.scripts.compare_grasp_matched_F \\
        --calibration outputs/sweep/calibrated_stiffness.csv \\
        --output outputs/sweep/matched_F_grasp_results.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import time
from pathlib import Path

import numpy as np
import warp as wp

from ..params import GraspConfig
from ..runner import run_headless


# Match calibrate_stiffness.py's rigid-object pin value.  If you change
# one, change the other.
_RIGID_OBJECT_STIFFNESS = {
    "cslc":  1.0e12,
    "hydro": 1.0e12,
    "point": 1.0e12,
}


@contextlib.contextmanager
def _maybe_silence(quiet: bool):
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


def _apply_stiffness(
    config: GraspConfig,
    contact_model: str,
    k_pad: float,
    k_obj: float,
) -> None:
    """Write the (pad, object) primary-stiffness pair into the config."""
    if contact_model == "cslc":
        config.cslc.kc_per_volume = k_pad
        config.material.ke_target_physical = k_obj
    elif contact_model == "hydro":
        config.material.kh_pad = k_pad
        config.material.kh_object = k_obj
    elif contact_model == "point":
        config.material.ke_pad_physical = k_pad
        config.material.ke_target_physical = k_obj
    else:
        raise ValueError(f"Unknown contact_model={contact_model!r}")


def run_matched_F_cell(
    contact_model: str,
    pad_kind: str,
    object_kind: str,
    mode: str,
    primary_stiffness_pad: float,
    primary_stiffness_obj: float,
    *,
    output_root: Path,
    face_pen_mm: float | None = None,
    quiet: bool = True,
) -> dict:
    """Run one matched-F grasp test, return its metrics row."""
    config = GraspConfig()
    config.contact_model = contact_model
    config.pad.kind = pad_kind
    config.object.kind = object_kind
    _apply_stiffness(config, contact_model, primary_stiffness_pad,
                     primary_stiffness_obj)

    # Optional commanded overlap override: keep squeeze_speed at default
    # and adjust squeeze_duration so the SQUEEZE phase ends with the
    # commanded penetration depth.
    if face_pen_mm is not None:
        sq_speed = config.timing.squeeze_speed
        config.timing.squeeze_duration = max(face_pen_mm * 1e-3 / sq_speed,
                                             config.timing.dt)

    # One isolated subdir per run; no timestamp so re-runs overwrite.
    config.logging.output_root = output_root
    pen_label = f"_pen{face_pen_mm:.2f}mm" if face_pen_mm is not None else ""
    config.logging.run_label = (
        f"{contact_model}_{pad_kind}_{object_kind}_{mode}{pen_label}"
    )
    config.logging.use_timestamp = False
    config.logging.save_lattice_preview = False
    config.logging.save_postsim_plots = False

    t0 = time.perf_counter()
    with _maybe_silence(quiet):
        metrics = run_headless(config)
    wall_s = time.perf_counter() - t0

    # ── Augmented metrics ──
    z_arr = np.array(metrics.object_z) if metrics.object_z else np.zeros(1)
    contacts_arr = np.array(metrics.contacts) if metrics.contacts else np.zeros(1)
    F_left_arr = np.array(metrics.F_n_left) if metrics.F_n_left else np.zeros(1)
    F_right_arr = np.array(metrics.F_n_right) if metrics.F_n_right else np.zeros(1)
    dx_arr = np.array(metrics.dx_left) if metrics.dx_left else np.zeros(1)

    dt = config.timing.dt
    n_hold_steps = max(int(round(config.timing.hold_duration / dt)), 1)
    # Use the LAST HALF of HOLD for steady-state averages — the first
    # half can still carry the lift-end transient.
    n_avg = max(n_hold_steps // 2, 1)
    z_tail = z_arr[-n_avg:] if len(z_arr) >= n_avg else z_arr
    contacts_tail = contacts_arr[-n_avg:] if len(contacts_arr) >= n_avg else contacts_arr
    F_left_tail = F_left_arr[-n_avg:] if len(F_left_arr) >= n_avg else F_left_arr
    F_right_tail = F_right_arr[-n_avg:] if len(F_right_arr) >= n_avg else F_right_arr
    dx_tail = dx_arr[-n_avg:] if len(dx_arr) >= n_avg else dx_arr

    tail_sigma_z = float(np.std(z_tail))
    n_active_hold = float(np.mean(contacts_tail))
    # Magnitude average of |F_left| and |F_right| (3rd-law symmetric).
    F_n_hold = float(0.5 * (np.mean(np.abs(F_left_tail)) + np.mean(np.abs(F_right_tail))))
    # Penetration δ = dx − approach_gap (SQUEEZE drives dx beyond the
    # initial gap; remainder is penetration into the object face).
    dx_hold = float(np.mean(dx_tail))
    delta_hold_mm = max(0.0, (dx_hold - config.pad.approach_gap) * 1e3)

    return {
        "contact_model": contact_model,
        "pad_kind": pad_kind,
        "object_kind": object_kind,
        "mode": mode,
        "face_pen_mm": face_pen_mm if face_pen_mm is not None
                       else config.timing.squeeze_duration * config.timing.squeeze_speed * 1e3,
        "k_pad": primary_stiffness_pad,
        "k_obj": primary_stiffness_obj,
        "held": int(metrics.held),
        "lifted": int(metrics.lifted),
        "max_z": metrics.max_z,
        "final_z": metrics.final_z,
        "min_z": metrics.min_z,
        "xy_slip_max": metrics.xy_slip_max,
        "tail_sigma_z": tail_sigma_z,
        "n_active_hold": n_active_hold,
        "F_n_hold": F_n_hold,
        "delta_hold_mm": delta_hold_mm,
        "wall_s": wall_s,
    }


def _load_calibration(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path,
                        help="Calibrated-stiffness CSV (from calibrate_stiffness)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV path")
    parser.add_argument("--output-root", type=Path,
                        default=Path("/home/nataliya/newton_custom_contact/outputs/matched_F"),
                        help="Per-run output dir root")
    parser.add_argument("--modes", default="symmetric,rigid_object",
                        help="Comma-separated calibration modes to run")
    parser.add_argument("--face-pens-mm", default=None,
                        help="Comma-separated commanded face penetration "
                             "values [mm] to sweep.  If omitted, the SQUEEZE "
                             "phase uses GraspConfig defaults (~1 mm).")
    parser.add_argument("--verbose", action="store_true",
                        help="Leave per-run prints on")
    args = parser.parse_args()

    wp.init()
    calib_rows = _load_calibration(args.calibration)
    modes = [m.strip() for m in args.modes.split(",")]
    if not calib_rows:
        raise SystemExit(f"No rows in {args.calibration}")

    face_pens: list[float | None]
    if args.face_pens_mm is None:
        face_pens = [None]
    else:
        face_pens = [float(x) for x in args.face_pens_mm.split(",")]

    cells = []
    for row in calib_rows:
        for mode in modes:
            if mode == "symmetric":
                k_pad = float(row["k_symmetric"])
                # Object's stiffness defaults to k_pad (both bodies same).
                # This matches the "both soft" case used to calibrate
                # F(δ_op) on the symmetric column of the sweep CSV.
                k_obj = k_pad
            elif mode == "rigid_object":
                k_pad = float(row["k_rigid_object_pad"])
                k_obj = float(row["k_rigid_object_obj"])
            else:
                raise ValueError(f"Unknown mode={mode!r}")
            for face_pen in face_pens:
                cells.append({
                    "contact_model": row["contact_model"],
                    "pad_kind": row["pad_kind"],
                    "object_kind": row["object_kind"],
                    "mode": mode,
                    "k_pad": k_pad,
                    "k_obj": k_obj,
                    "face_pen_mm": face_pen,
                })

    print(f"Running matched-F grasp comparison on {len(cells)} cells "
          f"({len(calib_rows)} geometries × {len(modes)} modes)")

    results: list[dict] = []
    sweep_t0 = time.perf_counter()
    for i, cell in enumerate(cells, 1):
        pen_label = (f" pen={cell['face_pen_mm']:.2f}mm"
                     if cell['face_pen_mm'] is not None else "")
        print(f"  [{i}/{len(cells)}] {cell['contact_model']:5s} "
              f"pad={cell['pad_kind']:10s} object={cell['object_kind']:6s} "
              f"mode={cell['mode']:13s}{pen_label} "
              f"k_pad={cell['k_pad']:.2e} k_obj={cell['k_obj']:.2e}",
              flush=True)
        try:
            r = run_matched_F_cell(
                cell["contact_model"], cell["pad_kind"], cell["object_kind"],
                cell["mode"], cell["k_pad"], cell["k_obj"],
                output_root=args.output_root,
                face_pen_mm=cell['face_pen_mm'],
                quiet=not args.verbose,
            )
        except Exception as e:
            print(f"     !! FAILED: {type(e).__name__}: {e}")
            r = {**cell, "held": -1, "error": f"{type(e).__name__}: {e}"}
        else:
            print(f"     held={r['held']}  F_n={r.get('F_n_hold', 0):.2f}N  "
                  f"δ={r.get('delta_hold_mm', 0):.2f}mm  "
                  f"final_z={r['final_z'] * 1e3:.1f}mm  "
                  f"slip={r['xy_slip_max'] * 1e3:.2f}mm  "
                  f"tail_σ={r['tail_sigma_z'] * 1e3:.3f}mm  "
                  f"n_act={r['n_active_hold']:.1f}  "
                  f"wall={r['wall_s']:.1f}s")
        results.append(r)

    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    total = time.perf_counter() - sweep_t0
    print(f"\nWrote {len(results)} rows to {args.output} ({total:.1f}s total)")


if __name__ == "__main__":
    main()
