# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot the CSLC smoothing_eps sub-sweep.

Reads a sweep CSV with a ``smoothing_eps`` column populated (i.e. one
produced by ``sweep_force_curve --smoothing-eps-m a,b,c,...``) and
emits one F-vs-δ curve per eps value on a single panel.  Used to
characterise the CSLC contact "shelf" identified in step 1: how much
of the low-δ engagement force comes from the smooth-step gate's
tail width versus the underlying lattice geometry.

Also prints a small table of n_contacts vs (depth, eps) — if eps
controls the shelf entirely, n_contacts should rise sharply with eps
at small depths (more pad spheres "leaking" engagement through the
smoothing tail).

Usage::

    uv run -m cslc_main.grasp.scripts.plot_eps_sweep \\
        --input outputs/sweep/force_curve_eps.csv \\
        --output outputs/sweep/force_curve_eps.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _load_rows(args.input)
    # Group by smoothing_eps value.  Skip rows without eps populated.
    by_eps: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        eps_str = r.get("smoothing_eps", "")
        if not eps_str or eps_str.lower() == "nan":
            continue
        try:
            eps = float(eps_str)
        except ValueError:
            continue
        if np.isnan(eps):
            continue
        by_eps[eps].append(r)

    if not by_eps:
        raise SystemExit(f"No rows with smoothing_eps in {args.input}")

    # Sort within each curve by depth.
    for k in by_eps:
        by_eps[k].sort(key=lambda r: float(r["face_pen_mm"]))

    # ── F-vs-δ panel ──
    fig, (ax_F, ax_n) = plt.subplots(1, 2, figsize=(13, 5))
    eps_values = sorted(by_eps.keys())
    cmap = plt.get_cmap("viridis")
    colors = {eps: cmap(i / max(len(eps_values) - 1, 1))
              for i, eps in enumerate(eps_values)}

    for eps in eps_values:
        rows_eps = by_eps[eps]
        depths = np.array([float(r["face_pen_mm"]) for r in rows_eps])
        F = np.array([float(r["F_n_mean"]) for r in rows_eps])
        F_std = np.array([float(r["F_n_std"]) for r in rows_eps])
        n_c = np.array([float(r["n_contacts_mean"]) for r in rows_eps])
        label = f"eps = {eps:.0e} m"
        ax_F.plot(depths, F, "-o", color=colors[eps], label=label)
        ax_F.fill_between(depths, F - F_std, F + F_std,
                          color=colors[eps], alpha=0.15)
        ax_n.plot(depths, n_c, "-o", color=colors[eps], label=label)

    ax_F.set_xlabel("Commanded face penetration δ [mm]")
    ax_F.set_ylabel("Steady-state normal force F_n [N]")
    ax_F.set_title("CSLC F_n vs δ — smoothing_eps sweep\n"
                   "(box pad + sphere object, kc_per_volume = default)")
    ax_F.legend(loc="upper left", fontsize=9)
    ax_F.grid(True, alpha=0.3)

    ax_n.set_xlabel("Commanded face penetration δ [mm]")
    ax_n.set_ylabel("Mean active contacts during HOLD")
    ax_n.set_title("Contact count vs δ — smoothing_eps sweep")
    ax_n.legend(loc="upper left", fontsize=9)
    ax_n.grid(True, alpha=0.3)
    ax_n.set_yscale("log")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)

    # Console summary.
    print(f"F_n vs (depth, eps) [N]:")
    depths_all = sorted({float(r["face_pen_mm"]) for rs in by_eps.values() for r in rs})
    header = "  depth(mm) " + "  ".join(f"{eps:>10.0e}" for eps in eps_values)
    print(header)
    for d in depths_all:
        cells = []
        for eps in eps_values:
            match = [r for r in by_eps[eps] if abs(float(r["face_pen_mm"]) - d) < 1e-6]
            cells.append(f"{float(match[0]['F_n_mean']):>10.2f}" if match else f"{'-':>10}")
        print(f"  {d:>8.2f}  " + "  ".join(cells))

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
