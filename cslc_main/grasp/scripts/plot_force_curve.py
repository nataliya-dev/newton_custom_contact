# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot F-vs-δ curves from :mod:`sweep_force_curve`'s CSV output.

One panel per (pad_kind, object_kind) combination, one curve per
contact_model, with F_n_std shown as a shaded error band.  Highlights
the chosen operating depth δ_op with a vertical line and prints the
F_n at δ_op for each model (the inputs to matched-F calibration).

Usage::

    uv run -m cslc_main.grasp.scripts.plot_force_curve \\
        --input outputs/sweep/force_curve_box_sphere.csv \\
        --output outputs/sweep/force_curve_box_sphere.png \\
        --delta-op-mm 1.0
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_MODEL_COLOR = {
    "cslc":  "#1f77b4",  # blue
    "hydro": "#2ca02c",  # green
    "point": "#d62728",  # red
}
# Numeric defaults shown in the legend so the reader can map curve →
# what primary stiffness produced it.  Keep in sync with GraspConfig
# defaults (CSLCParams.kc_per_volume, MaterialParams.kh_pad,
# MaterialParams.ke_pad_physical); also shows the OBJECT-side default
# (MaterialParams.ke_target_physical / kh_object) because that
# composes with the pad-side knob via the series-spring law and
# explains why some panels are softer than the pad knob alone would
# suggest.
_MODEL_LABEL = {
    "cslc":  ("CSLC  kc_per_volume = 1e10 [Pa·m⁻¹ᐟ²]\n"
              "      ke_target_physical = 5e4 [N/m] (object)"),
    "hydro": ("Hydro kh_pad = 5e9 [Pa/m]\n"
              "      kh_object = 5e9 [Pa/m]"),
    "point": ("Point ke_pad_physical = 5e4 [N/m]\n"
              "      ke_target_physical = 5e4 [N/m]"),
}


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _group_by_panel(rows: list[dict]) -> dict[tuple[str, str], dict[str, list[dict]]]:
    """``(pad_kind, object_kind) -> contact_model -> rows sorted by depth``."""
    panels: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["pad_kind"], r["object_kind"])
        panels[key][r["contact_model"]].append(r)
    for k in panels:
        for m in panels[k]:
            panels[k][m].sort(key=lambda r: float(r["face_pen_mm"]))
    return panels


def _plot_panel(ax, by_model: dict[str, list[dict]], delta_op_mm: float | None) -> None:
    """Plot one (pad, object) panel: F vs δ with one curve per model.

    No per-panel legend — the parent figure draws one shared legend at
    the bottom so each panel gets its full data area.
    """
    for model, rows in by_model.items():
        if not rows:
            continue
        depths = np.array([float(r["face_pen_mm"]) for r in rows])
        F = np.array([float(r["F_n_mean"]) for r in rows])
        F_std = np.array([float(r["F_n_std"]) for r in rows])
        color = _MODEL_COLOR.get(model, "#666666")
        ax.plot(depths, F, "-o", color=color, label=_MODEL_LABEL.get(model, model))
        ax.fill_between(depths, F - F_std, F + F_std, color=color, alpha=0.2)

        if delta_op_mm is not None:
            # Linear interpolation onto delta_op_mm (curves are sparse).
            F_at_op = float(np.interp(delta_op_mm, depths, F))
            ax.plot([delta_op_mm], [F_at_op], "x", color=color, markersize=10,
                    markeredgewidth=2)
            ax.annotate(
                f"{F_at_op:.1f} N", xy=(delta_op_mm, F_at_op),
                xytext=(6, 4), textcoords="offset points",
                fontsize=9, color=color,
            )

    if delta_op_mm is not None:
        ax.axvline(delta_op_mm, color="0.5", linestyle="--", alpha=0.6)
    ax.set_xlabel("Commanded face penetration δ [mm]")
    ax.set_ylabel("Steady-state normal force F_n [N]")
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Sweep CSV path (output of sweep_force_curve)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output PNG path")
    parser.add_argument("--delta-op-mm", type=float, default=None,
                        help="Operating depth to mark on the plot [mm].  "
                             "If omitted, no vertical line / F-at-op "
                             "annotation is drawn (use when there is no "
                             "single canonical operating depth).")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    panels = _group_by_panel(rows)

    n_panels = len(panels)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7.0 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    if args.delta_op_mm is not None:
        print(f"F_n at δ_op = {args.delta_op_mm:.2f} mm:")
        print(f"{'panel':<22} {'cslc':>10} {'hydro':>10} {'point':>10}")
    for ax, (key, by_model) in zip(axes.flat, sorted(panels.items())):
        pad, obj = key
        ax.set_title(f"pad = {pad}, object = {obj}")
        _plot_panel(ax, by_model, args.delta_op_mm)
        # Console summary for matched-F input (only when δ_op is provided).
        if args.delta_op_mm is not None:
            row = [f"{pad}/{obj}".ljust(22)]
            for m in ("cslc", "hydro", "point"):
                if m in by_model and by_model[m]:
                    depths = np.array([float(r["face_pen_mm"]) for r in by_model[m]])
                    F = np.array([float(r["F_n_mean"]) for r in by_model[m]])
                    row.append(f"{float(np.interp(args.delta_op_mm, depths, F)):>10.2f}")
                else:
                    row.append(f"{'-':>10}")
            print("  ".join(row))

    # Hide unused axes.
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    # Single shared legend at the figure level so each axis keeps its
    # full data area.  Pull handles+labels from the first axis (all
    # axes use the same model→label mapping).  Also add the δ_op
    # vertical-line entry.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if args.delta_op_mm is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="0.5", linestyle="--", alpha=0.6))
        labels.append(f"δ_op = {args.delta_op_mm:.1f} mm")
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, fontsize=9,
        bbox_to_anchor=(0.5, -0.02), frameon=True,
    )

    # Reserve bottom margin for the shared legend.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
