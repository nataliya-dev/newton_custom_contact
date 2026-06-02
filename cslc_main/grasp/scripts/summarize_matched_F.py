# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Print + plot the matched-F grasp comparison results.

Reads the CSV produced by :mod:`compare_grasp_matched_F` and emits:

  1. Per-(pad, object) console tables with one row per model variant.
  2. A small grid of bar charts (one metric per axis) for visual scan.

Usage::

    uv run -m cslc_main.grasp.scripts.summarize_matched_F \\
        --input outputs/sweep/matched_F_results.csv \\
        --output outputs/sweep/matched_F_summary.png
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


def _model_label(model: str) -> str:
    return model


def _is_nan_str(s: str) -> bool:
    try:
        return np.isnan(float(s))
    except (ValueError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    # Defensive: drop rows from any legacy CSV that still carries a
    # non-default ``smoothing_eps`` column (eps variant is no longer
    # part of the matched-F pipeline).
    rows = [
        r for r in rows
        if not (r.get("smoothing_eps", "")
                and r["smoothing_eps"].lower() != "nan"
                and not _is_nan_str(r["smoothing_eps"]))
    ]

    # Group by (pad, object, mode).  Within a panel each row is one model.
    panels: dict = defaultdict(list)
    for r in rows:
        key = (r["pad_kind"], r["object_kind"], r["mode"])
        panels[key].append(r)

    # ── Console tables ──
    metric_cols = [
        ("held",         "held",     "{:>4}",   1.0),
        ("lifted",       "lift",     "{:>4}",   1.0),
        ("max_z",        "max_z",    "{:>7.1f}", 1e3),   # m → mm
        ("final_z",      "fin_z",    "{:>7.1f}", 1e3),
        ("xy_slip_max",  "slip",     "{:>6.2f}", 1e3),
        ("tail_sigma_z", "tail_σ",   "{:>7.3f}", 1e3),
        ("n_active_hold","n_act",    "{:>6.1f}", 1.0),
        ("wall_s",       "wall_s",   "{:>6.1f}", 1.0),
    ]

    for (pad, obj, mode), prows in sorted(panels.items()):
        title = f"=== pad={pad}  object={obj}  mode={mode} ==="
        print(f"\n{title}")
        header = f"  {'model':<22s}  " + "  ".join(f"{h[1]:>7s}" for h in metric_cols)
        print(header)
        for r in sorted(prows, key=lambda x: (x["contact_model"], x.get("smoothing_eps", ""))):
            label = _model_label(r["contact_model"])
            cells = []
            for csv_key, _hdr, fmt, scale in metric_cols:
                val = r.get(csv_key, "")
                try:
                    v = float(val) * scale
                    cells.append(fmt.format(v))
                except (ValueError, TypeError):
                    cells.append(f"{'-':>7}")
            print(f"  {label:<22s}  " + "  ".join(cells))

    # ── Bar-chart grid ──
    # One metric per axis; bars grouped by (pad, object) panel,
    # colored by model.  Modes are kept on separate figures if there
    # are multiple modes (rare in practice: usually just rigid_object).
    plot_metrics = [
        ("max_z",        "max_z [mm]",        1e3, False),
        ("final_z",      "final_z [mm]",      1e3, False),
        ("xy_slip_max",  "xy slip [mm]",      1e3, True),  # smaller better
        ("tail_sigma_z", "tail σ_z [mm]",     1e3, True),
        ("n_active_hold","n_active",          1.0, False),
        ("wall_s",       "wall_s [s]",        1.0, True),
    ]

    # Collect unique (pad, object) keys and model labels.
    panel_keys = sorted({(r["pad_kind"], r["object_kind"]) for r in rows})
    model_labels = sorted({_model_label(r["contact_model"])
                           for r in rows})

    cmap = plt.get_cmap("tab10")
    model_color = {m: cmap(i) for i, m in enumerate(model_labels)}

    n_metrics = len(plot_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    x = np.arange(len(panel_keys))
    bar_w = 0.8 / max(len(model_labels), 1)

    for ax, (mkey, mlabel, mscale, lower_better) in zip(
            axes.flat, plot_metrics):
        for j, model in enumerate(model_labels):
            heights = []
            for pk in panel_keys:
                match = [r for r in rows
                         if (r["pad_kind"], r["object_kind"]) == pk
                         and _model_label(r["contact_model"]) == model]
                if not match:
                    heights.append(0.0)
                    continue
                try:
                    heights.append(float(match[0][mkey]) * mscale)
                except (ValueError, KeyError, TypeError):
                    heights.append(0.0)
            offsets = (j - (len(model_labels) - 1) / 2) * bar_w
            ax.bar(x + offsets, heights, bar_w,
                   label=model, color=model_color[model])

        ax.set_xticks(x)
        ax.set_xticklabels([f"{pad}/{obj}" for pad, obj in panel_keys],
                            rotation=15, fontsize=8)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel + ("  (lower = better)" if lower_better else ""))
        ax.grid(True, axis="y", alpha=0.3)
        if mkey == plot_metrics[0][0]:
            ax.legend(loc="upper left", fontsize=8)

    # Hide any unused axes.
    for ax in axes.flat[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
