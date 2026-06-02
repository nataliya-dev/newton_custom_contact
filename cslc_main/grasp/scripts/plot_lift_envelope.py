# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot lift envelope across (face_pen, model) for each (pad, object) cell.

For each cell, shows the F_n during HOLD as a function of commanded
overlap, with hollow markers for failed grasps (held=0) and filled
markers for successful holds.  This is the "what does it take to lift"
phase diagram the user asked for.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_MODEL_COLOR = {
    "cslc":  "#1f77b4",
    "hydro": "#2ca02c",
    "point": "#d62728",
}


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Lift envelope: F_n during HOLD vs commanded overlap")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    # Group by (pad, object) and contact_model
    panels: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["pad_kind"], r["object_kind"])
        panels[key][r["contact_model"]].append(r)
    for cell in panels:
        for m in panels[cell]:
            panels[cell][m].sort(key=lambda r: float(r["face_pen_mm"]))

    cells = sorted(panels.keys())
    n = len(cells)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for ax, cell in zip(axes.flat, cells):
        pad, obj = cell
        ax.set_title(f"pad = {pad}, object = {obj}")
        for model, rs in panels[cell].items():
            x = np.array([float(r["face_pen_mm"]) for r in rs])
            F = np.array([float(r.get("F_n_hold", 0)) for r in rs])
            held = np.array([int(r["held"]) for r in rs])
            color = _MODEL_COLOR.get(model, "0.5")
            # Continuous line
            ax.plot(x, F, "-", color=color, alpha=0.5, label=model)
            # Filled markers for held=1
            mask_held = held == 1
            mask_fail = ~mask_held
            if mask_held.any():
                ax.plot(x[mask_held], F[mask_held], "o", color=color,
                        markersize=10, markeredgecolor="black",
                        markeredgewidth=1, label="_nolegend_")
            if mask_fail.any():
                ax.plot(x[mask_fail], F[mask_fail], "X", color=color,
                        markersize=10, markeredgecolor="black",
                        markeredgewidth=1, alpha=0.6, label="_nolegend_")
            # Find lift threshold = smallest face_pen with held=1
            if mask_held.any():
                pen_thresh = float(x[mask_held].min())
                F_thresh = float(F[mask_held][x[mask_held].argmin()])
                ax.annotate(
                    f"lift @ {pen_thresh:.2f}mm\n  → {F_thresh:.1f} N",
                    xy=(pen_thresh, F_thresh),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color=color,
                )
        ax.set_xlabel("Commanded face penetration δ [mm]")
        ax.set_ylabel("F_n during HOLD [N]")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    # Build a unified legend at figure level
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_MODEL_COLOR["cslc"],
               markeredgecolor="black", markersize=10, label="cslc held=1"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=_MODEL_COLOR["cslc"],
               markeredgecolor="black", markersize=10, alpha=0.6, label="cslc held=0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_MODEL_COLOR["hydro"],
               markeredgecolor="black", markersize=10, label="hydro held=1"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=_MODEL_COLOR["hydro"],
               markeredgecolor="black", markersize=10, alpha=0.6, label="hydro held=0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_MODEL_COLOR["point"],
               markeredgecolor="black", markersize=10, label="point held=1"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=_MODEL_COLOR["point"],
               markeredgecolor="black", markersize=10, alpha=0.6, label="point held=0"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=10, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
