# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-cell CSLC calibration to match Hydro F at δ=1mm, kc held fixed.

The contract: ``kc_per_volume`` is CSLC's primary contact stiffness
(the analog of Hydro's ``kh_pad`` and Point's ``ke_pad_physical``).
For a fair cross-model comparison we hold kc at the GraspConfig
default and tune the OTHER CSLC knobs to bring per-cell F into
agreement with Hydro.

Which auxiliary knobs actually move F at fixed δ?  Based on params.py
docstrings + the n_samples=50 ablation (which INCREASED F by 22%):

  * ``smoothing_eps``  primary lever.  Controls the smooth-step
    width of the contact gate; larger eps = more pad spheres pulled
    into engagement through the tail → bigger shelf.  Empirically
    drives F by 50-80% across [1e-5, 2e-3] on box-pad geometries.
  * ``ka`` (anchor stiffness)  threshold knob per docstring; above
    ~10× object weight it's invariant.  Tuneable secondary lever if
    eps alone can't reach the target.
  * ``pad.n_samples``  tested 100→50; F went UP (fewer spheres →
    larger per-sphere kernel → more target samples captured per
    sphere → more force).  Not useful for downward calibration.
  * ``kl``, ``ka_tangent_ratio``, ``k_stick``  redistribution /
    tangential only; do not change normal F magnitude.  Not used.

Per-cell algorithm:
  1. Measure CSLC at default (eps=5e-4) to anchor.
  2. Coarse eps sweep over {1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3}.
  3. Pick the eps with F closest to Hydro target.
  4. If best is still > ±15% off, do one secondary refinement with ka.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import warp as wp

from .sweep_force_curve import measure_F_at_depth


_TOLERANCE = 0.15
_DEFAULT_KC = 1.0e10        # held fixed across all cells
_DEFAULT_KA = 35_000.0
_DEFAULT_EPS = 5.0e-4

# Eps grid: 8 values spanning 1e-5 to 5e-3 (2.5 decades).
_EPS_GRID = [1.0e-5, 5.0e-5, 1.0e-4, 3.0e-4, 5.0e-4, 1.0e-3, 3.0e-3, 5.0e-3]

# Fallback ka grid (only consulted if eps sweep can't hit the target
# within tolerance).
_KA_GRID = [3_000.0, 10_000.0, 35_000.0, 100_000.0, 350_000.0]


def _read_hydro_target(matrix_csv: Path, delta_op_mm: float) -> dict[tuple[str, str], float]:
    rows: list[dict] = []
    with open(matrix_csv) as f:
        rows = list(csv.DictReader(f))
    hydro_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["contact_model"] == "hydro":
            hydro_groups[(r["pad_kind"], r["object_kind"])].append(r)
    out: dict[tuple[str, str], float] = {}
    for key, rs in hydro_groups.items():
        rs.sort(key=lambda r: float(r["face_pen_mm"]))
        depths = np.array([float(r["face_pen_mm"]) for r in rs])
        F = np.array([float(r["F_n_mean"]) for r in rs])
        out[key] = float(np.interp(delta_op_mm, depths, F))
    return out


def _measure_cslc(
    pad_kind: str, object_kind: str, delta_op_mm: float,
    *, eps: float, ka: float = _DEFAULT_KA,
) -> float:
    """Measure CSLC F at given depth with auxiliary-knob overrides."""
    r = measure_F_at_depth(
        "cslc", pad_kind, object_kind, delta_op_mm,
        primary_stiffness=_DEFAULT_KC,
        smoothing_eps=eps,
        ka=ka,
        quiet=True,
    )
    return r["F_n_mean"]


def _eps_grid_search(
    pad_kind: str, object_kind: str, target_F: float, delta_op_mm: float,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Try every eps in the grid, return (best_eps, best_F, all_probes)."""
    probes: list[tuple[float, float]] = []
    for eps in _EPS_GRID:
        F = _measure_cslc(pad_kind, object_kind, delta_op_mm, eps=eps)
        probes.append((eps, F))
        print(f"     eps={eps:.0e}  F={F:.2f}N  "
              f"(target {target_F:.2f}, ratio={F/target_F:.3f})", flush=True)
    # Pick the probe whose F is closest to target in log space (so
    # over- and under-shoot are weighted equally).
    log_target = np.log(target_F + 1e-9)
    best = min(probes, key=lambda p: abs(np.log(p[1] + 1e-9) - log_target))
    return best[0], best[1], probes


def _ka_refinement(
    pad_kind: str, object_kind: str, target_F: float, delta_op_mm: float,
    *, fixed_eps: float,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Secondary refinement: sweep ka at the chosen eps."""
    probes: list[tuple[float, float]] = []
    for ka in _KA_GRID:
        F = _measure_cslc(pad_kind, object_kind, delta_op_mm,
                          eps=fixed_eps, ka=ka)
        probes.append((ka, F))
        print(f"     ka={ka:>9.0f}  eps={fixed_eps:.0e}  F={F:.2f}N  "
              f"(target {target_F:.2f}, ratio={F/target_F:.3f})", flush=True)
    log_target = np.log(target_F + 1e-9)
    best = min(probes, key=lambda p: abs(np.log(p[1] + 1e-9) - log_target))
    return best[0], best[1], probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-matrix", required=True, type=Path)
    parser.add_argument("--output-matrix", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--delta-op-mm", type=float, default=1.0)
    parser.add_argument("--depths-mm", default="0.2,0.5,1.0,1.5,2.0")
    parser.add_argument("--pad-kinds", default="box,dome")
    parser.add_argument("--object-kinds", default="sphere,box")
    args = parser.parse_args()

    wp.init()

    hydro_targets = _read_hydro_target(args.reference_matrix, args.delta_op_mm)
    if not hydro_targets:
        raise SystemExit(f"No hydro rows found in {args.reference_matrix}")

    pad_kinds = [p.strip() for p in args.pad_kinds.split(",")]
    object_kinds = [o.strip() for o in args.object_kinds.split(",")]
    depths = [float(x) for x in args.depths_mm.split(",")]
    cells = [(p, o) for p in pad_kinds for o in object_kinds]

    print(f"\n=== Per-cell CSLC calibration to hydro F at δ={args.delta_op_mm:.2f} mm ===")
    print(f"   kc held fixed at {_DEFAULT_KC:.2e}; tuning eps (and ka if needed)\n")
    print("Hydro targets:")
    for cell in cells:
        if cell in hydro_targets:
            print(f"  {cell[0]+'/'+cell[1]:<14s}  F_target = {hydro_targets[cell]:.2f} N")

    cell_params: dict[tuple[str, str], dict] = {}
    for pad, obj in cells:
        if (pad, obj) not in hydro_targets:
            continue
        target_F = hydro_targets[(pad, obj)]
        print(f"\n--- Calibrating {pad}/{obj} (target F = {target_F:.2f} N) ---")
        print(f"  eps sweep at ka={_DEFAULT_KA:.0f}:")
        best_eps, best_F, eps_probes = _eps_grid_search(
            pad, obj, target_F, args.delta_op_mm,
        )
        best_ka = _DEFAULT_KA
        ratio = best_F / target_F if target_F > 0 else float("nan")
        used_ka_refine = False
        if abs(ratio - 1.0) > _TOLERANCE:
            print(f"  eps alone: ratio={ratio:.3f} (out of ±{int(_TOLERANCE*100)}%); "
                  f"refining with ka at best eps={best_eps:.0e}:")
            best_ka, best_F, _ = _ka_refinement(
                pad, obj, target_F, args.delta_op_mm, fixed_eps=best_eps,
            )
            ratio = best_F / target_F if target_F > 0 else float("nan")
            used_ka_refine = True
        within = abs(ratio - 1.0) <= _TOLERANCE
        print(f"  → eps={best_eps:.0e}  ka={best_ka:.0f}  F={best_F:.2f}N  "
              f"target={target_F:.2f}N  ratio={ratio:.3f}  "
              f"{'WITHIN' if within else 'OUTSIDE'} ±{int(_TOLERANCE*100)}%")
        cell_params[(pad, obj)] = {
            "pad_kind": pad, "object_kind": obj,
            "target_F": target_F,
            "kc": _DEFAULT_KC,            # held fixed
            "eps": best_eps,
            "ka": best_ka,
            "F_calibrated": best_F,
            "ratio": ratio,
            "within_tolerance": int(within),
            "used_ka_refine": int(used_ka_refine),
        }

    # ── Re-run full δ sweep with per-cell params ──
    print(f"\n=== Re-running CSLC F-vs-δ at calibrated per-cell params ===")
    cslc_rows: list[dict] = []
    for (pad, obj), p in cell_params.items():
        for d in depths:
            print(f"  CSLC {pad:5s}/{obj:6s}  δ={d:.2f}mm  "
                  f"eps={p['eps']:.0e}  ka={p['ka']:.0f}", flush=True)
            r = measure_F_at_depth(
                "cslc", pad, obj, d,
                primary_stiffness=p["kc"], smoothing_eps=p["eps"], ka=p["ka"],
                quiet=True,
            )
            print(f"    F={r['F_n_mean']:.2f} ± {r['F_n_std']:.2f} N  "
                  f"n_c={r['n_contacts_mean']:.1f}")
            cslc_rows.append(r)

    # ── Merge with existing hydro/point rows ──
    with open(args.reference_matrix) as f:
        ref_rows = list(csv.DictReader(f))
    non_cslc = [r for r in ref_rows if r["contact_model"] != "cslc"]
    fieldnames: list[str] = []
    for r in cslc_rows + non_cslc:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_matrix, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in cslc_rows + non_cslc:
            writer.writerow(r)
    print(f"\nWrote merged matrix → {args.output_matrix}")

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    cfg_fields = ["pad_kind", "object_kind", "target_F", "kc", "eps", "ka",
                  "F_calibrated", "ratio", "within_tolerance", "used_ka_refine"]
    with open(args.output_config, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cfg_fields)
        writer.writeheader()
        for p in cell_params.values():
            writer.writerow(p)
    print(f"Wrote per-cell calibration → {args.output_config}")


if __name__ == "__main__":
    main()
