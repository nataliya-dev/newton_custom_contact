# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep primary stiffness ``k`` at fixed depth ``δ_op``; measure F(k).

The :mod:`sweep_force_curve` tool maps F vs δ at default k.  This one
maps F vs k at default δ, so calibration to a target F* is read off
the actual measured curve rather than extrapolated linearly from a
single point.

Per cell ``(model, pad, object)``: run the quasi-static measurement at
each k in the sweep grid, log F at δ_op = 1mm.  Output CSV is consumed
by :mod:`plot_stiffness_curve` (one panel per model, one curve per
geometry, horizontal F* marker shows where each curve crosses).

The per-model k grids are model-specific because the natural scales
differ by 6 orders of magnitude:

  * CSLC ``kc_per_volume``:  default 3e10  (Pa · m^-1/2)
  * Hydro ``kh_pad``:        default 5e9   (Pa/m)
  * Point ``ke_pad_physical``: default 5e4 (N/m)

Default multipliers ``0.1, 0.3, 1.0, 3.0, 10.0`` give a 100× span
around each model's default, which brackets every calibration target
we've seen so far (F* ∈ [6, 30] N).

Usage::

    uv run -m cslc_main.grasp.scripts.sweep_stiffness_curve \\
        --models cslc,hydro,point \\
        --pad-kinds box,dome --object-kinds sphere,box \\
        --k-mults 0.1,0.3,1.0,3.0,10.0 \\
        --delta-op-mm 1.0 \\
        --output outputs/sweep/stiffness_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import warp as wp

from .sweep_force_curve import measure_F_at_depth


# Per-model default primary stiffness — must match GraspConfig.
_DEFAULT_K = {
    "cslc":  1.0e10,   # CSLCParams.kc_per_volume   [Pa · m^-1/2]
    "hydro": 5.0e9,    # MaterialParams.kh_pad      [Pa/m]
    "point": 5.0e4,    # MaterialParams.ke_pad_physical [N/m]
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="cslc,hydro,point",
                        help="Comma-separated contact models")
    parser.add_argument("--pad-kinds", default="box,dome",
                        help="Comma-separated pad kinds")
    parser.add_argument("--object-kinds", default="sphere,box",
                        help="Comma-separated object kinds")
    parser.add_argument("--delta-op-mm", type=float, default=1.0,
                        help="Penetration depth at which F is measured [mm]")
    parser.add_argument("--k-mults", default="0.1,0.3,1.0,3.0,10.0",
                        help="Comma-separated multipliers on each model's "
                             "default primary stiffness (see _DEFAULT_K).")
    parser.add_argument("--settling-ms", type=float, default=200.0)
    parser.add_argument("--avg-window-ms", type=float, default=50.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    wp.init()

    models = [m.strip() for m in args.models.split(",")]
    pad_kinds = [p.strip() for p in args.pad_kinds.split(",")]
    object_kinds = [o.strip() for o in args.object_kinds.split(",")]
    k_mults = [float(x) for x in args.k_mults.split(",")]

    cells = list(itertools.product(models, pad_kinds, object_kinds, k_mults))
    print(f"Sweeping {len(cells)} cells "
          f"({len(models)} models × {len(pad_kinds)} pads × "
          f"{len(object_kinds)} objects × {len(k_mults)} k-multipliers) "
          f"at δ={args.delta_op_mm:.2f} mm")

    results: list[dict] = []
    sweep_t0 = time.perf_counter()
    for i, (model, pad, obj, k_mult) in enumerate(cells, 1):
        k_default = _DEFAULT_K[model]
        k = k_default * k_mult
        print(f"  [{i}/{len(cells)}] {model:5s} pad={pad:10s} "
              f"object={obj:6s} k={k:.2e} (×{k_mult:g} default)",
              flush=True)
        try:
            r = measure_F_at_depth(
                model, pad, obj, args.delta_op_mm,
                primary_stiffness=k,
                settling_ms=args.settling_ms,
                avg_window_ms=args.avg_window_ms,
                quiet=not args.verbose,
            )
            r["k_default"] = k_default
            r["k_mult"] = k_mult
        except Exception as e:
            print(f"     !! FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            r = {
                "contact_model": model, "pad_kind": pad, "object_kind": obj,
                "face_pen_mm": args.delta_op_mm,
                "primary_stiffness": k, "k_default": k_default, "k_mult": k_mult,
                "F_n_mean": float("nan"), "F_n_std": float("nan"),
                "error": f"{type(e).__name__}: {e}",
            }
        else:
            print(f"     F_n={r['F_n_mean']:.2f} ± {r['F_n_std']:.2f} N  "
                  f"n_c={r['n_contacts_mean']:.1f}  wall={r['wall_s']:.2f}s")
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
