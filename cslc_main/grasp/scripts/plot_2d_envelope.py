# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""2D phase diagram: F_n / lift outcome over (overlap × stiffness) grid.

One panel per contact model.  Each panel:
  - x-axis: commanded face_pen [mm]
  - y-axis: stiffness multiplier (relative to model default)
  - Cell color: F_n during HOLD (log-scale colormap)
  - Cell annotation: F_n value plus ✓ (held=1) or ✗ (held=0)
  - Cell border: green = held, red = not held

A horizontal dotted line marks the Coulomb friction floor F_n_min = W/(2μ).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Tennis-ball sphere: m=58g → W=0.569 N → F_n_floor = 0.569 N (μ=0.5)
F_FLOOR_SPHERE = 0.569
F_FLOOR_BOX = 1.086


# Reference stiffness for computing the mult label per cell.
# CSLC: our project default (no Newton baseline for CSLC).
# Hydro/Point: Newton library defaults (newton/_src/sim/builder.py).
_MODEL_DEFAULT_K = {
    "cslc":  1.0e10,
    "hydro": 1.0e10,
    "point": 2.5e3,
}


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="2D envelope: F_n & lift over (overlap × stiffness)")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    pad_kind = rows[0]["pad_kind"]
    obj_kind = rows[0]["object_kind"]
    f_floor = F_FLOOR_SPHERE if obj_kind == "sphere" else F_FLOOR_BOX

    # Group by model
    by_model: dict = defaultdict(list)
    for r in rows:
        by_model[r["contact_model"]].append(r)

    # Explicit panel order: point, hydro, cslc (left to right)
    _PANEL_ORDER = ["point", "hydro", "cslc"]
    models = [m for m in _PANEL_ORDER if m in by_model]
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.6 * n_models, 5.8),
                             squeeze=False)
    axes = axes[0]

    # Build unified F_n range for colormap (log scale)
    all_F = [float(r["F_n_hold"]) for r in rows if float(r["F_n_hold"]) > 0]
    Fmin = max(min(all_F), 0.01) if all_F else 0.01
    Fmax = max(all_F) if all_F else 100.0
    norm = mcolors.LogNorm(vmin=Fmin, vmax=Fmax)
    # Green (low F_n, efficient) → red (high F_n, over-gripping)
    cmap = plt.cm.RdYlGn_r

    for ax, model in zip(axes, models):
        rs = by_model[model]
        k_default = _MODEL_DEFAULT_K[model]
        # Unique overlaps and stiffness mults
        overlaps = sorted({float(r["face_pen_mm"]) for r in rs})
        mults = sorted({float(r["k_pad"]) / k_default for r in rs})
        n_pen = len(overlaps); n_k = len(mults)

        # Build grids
        F_grid = np.full((n_k, n_pen), np.nan)
        held_grid = np.zeros((n_k, n_pen), dtype=int)
        slip_grid = np.full((n_k, n_pen), np.nan)
        for r in rs:
            i = mults.index(float(r["k_pad"]) / k_default)
            j = overlaps.index(float(r["face_pen_mm"]))
            F_grid[i, j] = float(r["F_n_hold"])
            held_grid[i, j] = int(r["held"])
            slip_grid[i, j] = float(r["xy_slip_max"]) * 1e3  # mm

        # Heatmap
        im = ax.imshow(F_grid, origin="lower", aspect="auto", cmap=cmap,
                       norm=norm, extent=[-0.5, n_pen - 0.5, -0.5, n_k - 0.5])
        ax.set_xticks(range(n_pen))
        ax.set_xticklabels([f"{d:.2f}" for d in overlaps])
        ax.set_yticks(range(n_k))
        ax.set_yticklabels([f"{m:.2f}×" for m in mults])
        ax.set_xlabel("Commanded overlap δ [mm]")
        ax.set_ylabel(f"Stiffness mult (default = {k_default:.0e})")
        ax.set_title(f"{model}")

        # Annotate each cell with F_n value + lift outcome marker
        for i in range(n_k):
            for j in range(n_pen):
                F = F_grid[i, j]
                if np.isnan(F):
                    continue
                held = held_grid[i, j]
                marker = "✓" if held else "✗"
                # Black text on RdYlGn_r reads well across the whole range
                ax.text(j, i + 0.18, f"{F:.2f} N", ha="center", va="center",
                        fontsize=10, color="black", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  alpha=0.7, edgecolor="none"))
                ax.text(j, i - 0.22, marker, ha="center", va="center",
                        fontsize=22,
                        color="darkgreen" if held else "darkred",
                        fontweight="bold")

                # Thin black border for cell separation
                rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                     fill=False, edgecolor="black", linewidth=0.5)
                ax.add_patch(rect)

        # Friction-floor reference (text in title bar)
        ax.text(0.02, 0.97,
                f"Coulomb floor: F_n_min = {f_floor:.2f} N",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle(f"{args.title}\npad={pad_kind}, object={obj_kind}", fontsize=13)
    # Reserve right margin for the colorbar so it doesn't overlap the
    # rightmost panel.  Then place the colorbar in the reserved strip.
    fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.10, wspace=0.30)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("F_n during HOLD [N] (log scale, green=low/efficient, red=high)",
                   fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: do NOT use bbox_inches="tight" here — it ignores subplots_adjust
    # and re-tightens around content, which puts the colorbar back over the
    # rightmost panel.  Save at the manually-set layout instead.
    fig.savefig(args.output, dpi=140)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
