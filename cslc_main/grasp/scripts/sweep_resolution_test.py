# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep CSLC pad + target resolution to test the contact-pair-count theory.

Hypothesis: CSLC's force on flat-pad-on-flat-object configurations is
proportional to the total ``(pad sphere, target sample)`` contact-pair
count.  At default n_samples=100 + box_face_pitch=5mm the pair count
is ~488 and F at δ=1mm is ~110 N (vs hydro's 33 N).  If we reduce BOTH
pad density and target density PROPORTIONALLY, each pad sphere should
still capture roughly the same number of target samples on average,
but the total sphere count drops — so total pairs should drop roughly
linearly with the proportional reduction.

Quantitative prediction:
  ``total_pairs ≈ N_pad × (disc_area × target_density)``
  where ``disc_area ∝ 1/N_pad`` and ``target_density ∝ 1/pitch²``.
  Setting ``pitch² ∝ 1/N_pad`` keeps captures-per-sphere constant, so
  ``total_pairs ∝ N_pad``.

Test grid (4 variants × 4 cells × 3 depths = 48 runs):

  variant       pad n  box pitch  expected pairs
  baseline      100    5 mm       1× (488)
  halve both    50     10 mm      ~1/2 (~250)
  quarter both  25     20 mm      ~1/4 (~120)
  eighth both   13     40 mm      ~1/8 (~60)

If the theory holds, box/box F at δ=1mm should approximately halve at
each step → 110 → 55 → 28 → 14 N.  Hydro's 33 N target lands near the
"quarter both" variant.

Usage::

    uv run -m cslc_main.grasp.scripts.sweep_resolution_test \\
        --output outputs/sweep/resolution_test.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import warp as wp

from .sweep_force_curve import measure_F_at_depth


# Four variants, picked so each step halves the per-side density.
# Naming: (variant_label, pad_n_samples, box_face_pitch_m)
_VARIANTS: list[tuple[str, int, float]] = [
    ("baseline",    100, 0.005),   # current defaults
    ("halve_both",   50, 0.010),
    ("quarter_both", 25, 0.020),
    ("eighth_both",  13, 0.040),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pad-kinds", default="box,dome")
    parser.add_argument("--object-kinds", default="sphere,box")
    parser.add_argument("--depths-mm", default="0.2,1.0,2.0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    wp.init()

    pad_kinds = [p.strip() for p in args.pad_kinds.split(",")]
    object_kinds = [o.strip() for o in args.object_kinds.split(",")]
    depths = [float(x) for x in args.depths_mm.split(",")]

    results: list[dict] = []
    n_total = len(_VARIANTS) * len(pad_kinds) * len(object_kinds) * len(depths)
    print(f"Sweeping {n_total} cells "
          f"({len(_VARIANTS)} variants × {len(pad_kinds)} pads × "
          f"{len(object_kinds)} objects × {len(depths)} depths)")

    sweep_t0 = time.perf_counter()
    i = 0
    for label, pad_n, pitch_m in _VARIANTS:
        for pad in pad_kinds:
            for obj in object_kinds:
                for d in depths:
                    i += 1
                    # box_face_pitch is only meaningful when object is box.
                    pitch_apply = pitch_m if obj == "box" else None
                    print(f"  [{i}/{n_total}] {label:<14s}  "
                          f"pad={pad:5s}/obj={obj:6s}  pad_n={pad_n:>3d}  "
                          f"pitch={pitch_m*1e3:>5.1f}mm  δ={d:.2f}mm",
                          flush=True)
                    try:
                        r = measure_F_at_depth(
                            "cslc", pad, obj, d,
                            pad_n_samples=pad_n,
                            box_face_pitch=pitch_apply,
                            quiet=True,
                        )
                        r["variant"] = label
                        r["pad_n_samples"] = pad_n
                        r["box_face_pitch_mm"] = pitch_m * 1e3
                    except Exception as e:
                        print(f"     !! FAILED: {type(e).__name__}: {e}",
                              file=sys.stderr)
                        r = {
                            "contact_model": "cslc",
                            "pad_kind": pad,
                            "object_kind": obj,
                            "face_pen_mm": d,
                            "variant": label,
                            "pad_n_samples": pad_n,
                            "box_face_pitch_mm": pitch_m * 1e3,
                            "F_n_mean": float("nan"),
                            "F_n_std": float("nan"),
                            "n_contacts_mean": float("nan"),
                            "error": f"{type(e).__name__}: {e}",
                        }
                    else:
                        print(f"     F={r['F_n_mean']:.2f} ± {r['F_n_std']:.2f} N  "
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
