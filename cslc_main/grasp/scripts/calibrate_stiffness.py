# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrate primary stiffness per (model, pad, object) for matched-F runs.

Step 2 setup of the two-step comparison framework.  Reads a sweep CSV
produced by :mod:`sweep_force_curve`, interpolates each curve at the
operating depth ``δ_op``, and computes the primary-stiffness value that
would shift each curve so it passes through ``(δ_op, F*)``.

First-pass calibration assumes ``F(δ_op) ∝ primary_stiffness`` at fixed
δ_op.  This holds exactly for the linear models (point: ``F = ke·δ``;
hydro: ``p = kh·φ`` per the paper's eq 1).  For CSLC the relation is
also approximately linear at fixed depth because the kernel multiplies
``kc_per_volume`` by per-sample area + locality kernel + smooth gate +
``raw^1.5`` — kc_per_volume enters as a uniform scalar factor on the
per-sample force, so the aggregate F also scales linearly in kc.

Post-condition: the matched-F grasp runner re-measures F(δ_op) at the
calibrated stiffness and confirms it lands within ±5%.  If not, iterate
this helper with the re-measured F as input.

Two calibration modes per (model, pad, object) cell:

  * ``symmetric``: scale the pad-side primary-stiffness so F(δ_op)=F*.
    Object stiffness left at its default (``ke_target_physical`` /
    ``kh_object``).
  * ``rigid_object``: pin the object-side to a "rigid" stiffness
    (huge value; defaults: ``ke_target_physical=1e12``,
    ``kh_object=1e12``).  Then scale the pad-side so F(δ_op)=F*.  This
    is the "soft pad on rigid object" comparison.

Usage::

    uv run -m cslc_main.grasp.scripts.calibrate_stiffness \\
        --input outputs/sweep/force_curve_matrix.csv \\
        --output outputs/sweep/calibrated_stiffness.csv \\
        --delta-op-mm 1.0 --F-target-N 6.0
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


# Defaults that the sweep harness uses when ``primary_stiffness`` is
# None.  Must stay in sync with the GraspConfig defaults; if those
# drift, update here.  (We could read these from the live GraspConfig
# at runtime, but doing so requires importing the whole grasp package
# which is heavyweight for a small post-processing helper.)
_DEFAULT_PRIMARY_STIFFNESS = {
    "cslc":  3.0e10,   # CSLCParams.kc_per_volume  [Pa · m^-1/2]
    "hydro": 5.0e9,    # MaterialParams.kh_pad     [Pa/m]
    "point": 5.0e4,    # MaterialParams.ke_pad_physical [N/m]
}

# "Rigid" stiffness for the object side under the rigid-object mode.
# 1e12 is several orders of magnitude above any plausible material
# modulus; the series-spring composition treats this as effectively
# infinite (1/k_rigid ≈ 0).
_RIGID_OBJECT_STIFFNESS = {
    "cslc":  1.0e12,   # MaterialParams.ke_target_physical (CSLC's pair.other_ke)
    "hydro": 1.0e12,   # MaterialParams.kh_object
    "point": 1.0e12,   # MaterialParams.ke_target_physical
}


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _group_by_geometry(
    rows: list[dict],
) -> dict[tuple[str, str, str], list[dict]]:
    """``(model, pad, object) -> rows sorted by depth``.

    Rows with a non-default ``smoothing_eps`` value are SKIPPED — the
    eps characterisation sweep is a separate tool ([[plot_eps_sweep]])
    and should not propagate variants into matched-F dynamics.  The
    eps=1e-4 CSLC variant was tried and produced divergent calibrated
    kc values for low-F dome geometries (kc ≈ 4.88e11 on dome+sphere
    at F*=6N), so it's been removed from the pipeline.
    """
    out: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        eps_val = r.get("smoothing_eps", "")
        if eps_val and eps_val.lower() != "nan":
            try:
                if not np.isnan(float(eps_val)):
                    continue  # Skip non-default-eps rows.
            except ValueError:
                continue
        key = (r["contact_model"], r["pad_kind"], r["object_kind"])
        out[key].append(r)
    for k in out:
        out[k].sort(key=lambda r: float(r["face_pen_mm"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Force-curve CSV from sweep_force_curve")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV of calibrated stiffness per cell")
    parser.add_argument("--delta-op-mm", type=float, default=1.0,
                        help="Operating depth at which F should equal F* [mm]")
    parser.add_argument("--F-target-N", type=float, default=6.0,
                        help="Target normal force at δ_op [N]")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    groups = _group_by_geometry(rows)
    if not groups:
        raise SystemExit(f"No usable rows in {args.input}")

    out_rows: list[dict] = []
    print(f"Calibrating to F* = {args.F_target_N:.2f} N at δ_op = "
          f"{args.delta_op_mm:.2f} mm:")
    print(f"{'model':5s} {'pad':10s} {'object':6s}  {'F_default':>9}  "
          f"{'k_default':>10}  {'k_sym':>10}  {'k_rigid_obj':>11}")
    for (model, pad, obj), cell_rows in sorted(groups.items()):
        depths = np.array([float(r["face_pen_mm"]) for r in cell_rows])
        F = np.array([float(r["F_n_mean"]) for r in cell_rows])
        if args.delta_op_mm < depths.min() or args.delta_op_mm > depths.max():
            print(f"  warning: δ_op={args.delta_op_mm}mm outside swept range "
                  f"[{depths.min():.2f}, {depths.max():.2f}]mm for "
                  f"({model}, {pad}, {obj}); extrapolating", flush=True)
        F_at_op = float(np.interp(args.delta_op_mm, depths, F))
        k_default = _DEFAULT_PRIMARY_STIFFNESS[model]

        # Linear scaling: F ∝ k at fixed depth.  Verified true at the
        # F-law level for point + hydro; approximately true for CSLC
        # (kc enters as a global scalar factor on every per-sample F).
        if F_at_op <= 0.0:
            print(f"  skip: F({pad},{obj}) = {F_at_op:.3f} N — no engagement",
                  flush=True)
            continue
        scale = args.F_target_N / F_at_op
        k_symmetric = k_default * scale

        # Rigid-object mode: same linear-scaling assumption.  The
        # symmetric-default F_at_op was measured with the object's
        # stiffness at its own default; under rigid-object mode the
        # object's contribution to the series-spring drops out, which
        # for hydro/point shifts the effective F slightly upward.  We
        # approximate by scaling from the same baseline; matched-F
        # grasp runner verifies and iterates if needed.
        k_rigid_object = k_default * scale

        out_rows.append({
            "contact_model": model,
            "pad_kind": pad,
            "object_kind": obj,
            "delta_op_mm": args.delta_op_mm,
            "F_target_N": args.F_target_N,
            "F_at_default": F_at_op,
            "k_default": k_default,
            "k_symmetric": k_symmetric,
            "k_rigid_object_pad": k_rigid_object,
            "k_rigid_object_obj": _RIGID_OBJECT_STIFFNESS[model],
        })
        print(f"  {model:5s} {pad:10s} {obj:6s}  {F_at_op:>8.2f}N  "
              f"{k_default:>10.2e}  {k_symmetric:>10.2e}  "
              f"{k_rigid_object:>11.2e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        raise SystemExit("No calibrated rows produced")
    fieldnames = list(out_rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} calibrated rows to {args.output}")


if __name__ == "__main__":
    main()
