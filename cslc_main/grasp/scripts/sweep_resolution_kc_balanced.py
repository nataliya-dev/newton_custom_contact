# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep CSLC resolution with kc rescaled to keep ``kc · A_target`` constant.

Companion to :mod:`sweep_resolution_test` which showed that reducing
pad+target sampling proportionally does NOT reduce total CSLC force,
because per-contact F = kc · A_target · phi^1.5 scales UP with
A_target ∝ pitch² when pitch grows.  This sweep neutralises that boost
by setting ``kc = kc_base · (pitch_base / pitch)²``, so per-contact F
stays constant and total F should scale only with n_contacts (which
DOES drop with proportional density reduction).

Prediction: at constant per-contact F, total F should track n_contacts.
If baseline gives F=110N on box/box with n_c=488 → per-contact F =
0.225 N.  Lower-resolution variants should hold per-contact F ≈ 0.225
but with smaller n_c → smaller total F.

Test grid (4 variants × 4 cells × 3 depths = 48 runs):

  variant       pad n  pitch  kc          A_target ratio  expected F floor
  baseline      100    5 mm   1.00e10     1×              (matches earlier)
  halve_both    50     10 mm  2.50e9      4×              n_c × ~0.2
  quarter_both  25     20 mm  6.25e8      16×             ditto
  eighth_both   13     40 mm  1.56e8      64×             ditto
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import warp as wp

from .sweep_force_curve import measure_F_at_depth


# (label, pad_n, pitch_m, kc) — kc rescaled to neutralise A_target boost.
_VARIANTS: list[tuple[str, int, float, float]] = [
    ("baseline",       100, 0.005, 1.00e10),
    ("halve_both",      50, 0.010, 2.50e9),
    ("quarter_both",    25, 0.020, 6.25e8),
    ("eighth_both",     13, 0.040, 1.5625e8),
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
    print(f"Sweeping {n_total} cells (kc rescaled per variant)")

    sweep_t0 = time.perf_counter()
    i = 0
    for label, pad_n, pitch_m, kc in _VARIANTS:
        for pad in pad_kinds:
            for obj in object_kinds:
                for d in depths:
                    i += 1
                    pitch_apply = pitch_m if obj == "box" else None
                    print(f"  [{i}/{n_total}] {label:<14s}  "
                          f"pad={pad:5s}/obj={obj:6s}  pad_n={pad_n:>3d}  "
                          f"pitch={pitch_m*1e3:>5.1f}mm  kc={kc:.2e}  δ={d:.2f}mm",
                          flush=True)
                    try:
                        r = measure_F_at_depth(
                            "cslc", pad, obj, d,
                            primary_stiffness=kc,
                            pad_n_samples=pad_n,
                            box_face_pitch=pitch_apply,
                            quiet=True,
                        )
                        r["variant"] = label
                        r["pad_n_samples"] = pad_n
                        r["box_face_pitch_mm"] = pitch_m * 1e3
                        r["kc_used"] = kc
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
                            "kc_used": kc,
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
