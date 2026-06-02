# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bar-chart grasp-dynamics stability metrics for CSLC vs Hydro.

Reads the CSV from :mod:`compare_grasp_matched_F` and plots a 2x2 grid:
held/lifted flags, final_z, xy_slip_max, and tail_sigma_z, with one
bar per (model, cell).
"""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--title", default="Grasp dynamics stability: CSLC vs Hydro")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    cells = sorted({(r["pad_kind"], r["object_kind"]) for r in rows})
    models = sorted({r["contact_model"] for r in rows})
    cell_labels = [f"{p}/{o}" for p, o in cells]
    n_cells = len(cells)
    x = np.arange(n_cells)
    bar_w = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    def get(model, pad, obj, key):
        for r in rows:
            if (r["contact_model"] == model and r["pad_kind"] == pad
                    and r["object_kind"] == obj):
                try:
                    return float(r[key])
                except (ValueError, KeyError):
                    return float("nan")
        return float("nan")

    # Panel 1: held (binary)
    ax = axes[0, 0]
    for i, model in enumerate(models):
        vals = [get(model, p, o, "held") for p, o in cells]
        ax.bar(x + (i - (len(models) - 1) / 2) * bar_w, vals, bar_w,
               label=model, color=_MODEL_COLOR.get(model, "0.5"))
    ax.set_title("held (1 = success)")
    ax.set_xticks(x); ax.set_xticklabels(cell_labels, rotation=15)
    ax.set_ylim(0, 1.15); ax.set_ylabel("held flag")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: final_z (mm) — settled object height
    ax = axes[0, 1]
    for i, model in enumerate(models):
        vals = [get(model, p, o, "final_z") * 1e3 for p, o in cells]
        ax.bar(x + (i - (len(models) - 1) / 2) * bar_w, vals, bar_w,
               label=model, color=_MODEL_COLOR.get(model, "0.5"))
    ax.set_title("final_z [mm] — settled object height")
    ax.set_xticks(x); ax.set_xticklabels(cell_labels, rotation=15)
    ax.set_ylabel("z [mm]")
    ax.axhline(33.5, color="0.5", linestyle=":", alpha=0.5,
               label="initial (object radius)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: xy_slip_max (log scale — slips span 4 decades)
    ax = axes[1, 0]
    for i, model in enumerate(models):
        vals = [max(get(model, p, o, "xy_slip_max") * 1e3, 1e-3) for p, o in cells]
        ax.bar(x + (i - (len(models) - 1) / 2) * bar_w, vals, bar_w,
               label=model, color=_MODEL_COLOR.get(model, "0.5"))
    ax.set_title("xy_slip_max [mm] — peak XY drift")
    ax.set_xticks(x); ax.set_xticklabels(cell_labels, rotation=15)
    ax.set_ylabel("slip [mm]")
    ax.set_yscale("log")
    ax.axhline(5.0, color="r", linestyle="--", alpha=0.3, label="5 mm threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both", axis="y")

    # Panel 4: tail_sigma_z (residual oscillation during HOLD)
    ax = axes[1, 1]
    for i, model in enumerate(models):
        vals = [max(get(model, p, o, "tail_sigma_z") * 1e3, 1e-4) for p, o in cells]
        ax.bar(x + (i - (len(models) - 1) / 2) * bar_w, vals, bar_w,
               label=model, color=_MODEL_COLOR.get(model, "0.5"))
    ax.set_title("tail_σ_z [mm] — residual HOLD oscillation")
    ax.set_xticks(x); ax.set_xticklabels(cell_labels, rotation=15)
    ax.set_ylabel("σ_z [mm]")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both", axis="y")

    # Panel 5: F_n during HOLD (the headline number)
    ax = axes[0, 2]
    for i, model in enumerate(models):
        vals = [get(model, p, o, "F_n_hold") for p, o in cells]
        ax.bar(x + (i - (len(models) - 1) / 2) * bar_w, vals, bar_w,
               label=model, color=_MODEL_COLOR.get(model, "0.5"))
        for xi, v in zip(x + (i - (len(models) - 1) / 2) * bar_w, vals):
            if v > 0.1:
                ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("F_n [N] during HOLD — actual operating normal force")
    ax.set_xticks(x); ax.set_xticklabels(cell_labels, rotation=15)
    ax.set_ylabel("F_n [N]")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 6: delta during HOLD (penetration depth)
    ax = axes[1, 2]
    for i, model in enumerate(models):
        vals = [get(model, p, o, "delta_hold_mm") for p, o in cells]
        ax.bar(x + (i - (len(models) - 1) / 2) * bar_w, vals, bar_w,
               label=model, color=_MODEL_COLOR.get(model, "0.5"))
        for xi, v in zip(x + (i - (len(models) - 1) / 2) * bar_w, vals):
            if v > 0.01:
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("δ [mm] during HOLD — commanded SQUEEZE depth")
    ax.set_xticks(x); ax.set_xticklabels(cell_labels, rotation=15)
    ax.set_ylabel("δ [mm]")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
