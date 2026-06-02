# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot F-vs-k curves from :mod:`sweep_stiffness_curve` output.

Layout mirrors :mod:`plot_force_curve`: 2×2 grid of (pad, object)
panels, each panel overlays the three contact models (CSLC blue,
Hydro green, Point red) on a shared log-x axis of the stiffness
multiplier (k / k_default).  This lets the reader compare the three
models' force-vs-stiffness behaviour on the same x-axis even though
their absolute primary-stiffness units differ by orders of magnitude
(CSLC ~1e10 Pa·m⁻¹ᐟ², Hydro ~1e9 Pa/m, Point ~1e4 N/m).

Per-model default values appear in the shared footer legend so the
reader can convert the multiplier back to absolute units.

Usage::

    uv run -m cslc_main.grasp.scripts.plot_stiffness_curve \\
        --input outputs/sweep/stiffness_curve.csv \\
        --output outputs/sweep/stiffness_curve.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_MODEL_COLOR = {
    "cslc":  "#1f77b4",   # blue
    "hydro": "#2ca02c",   # green
    "point": "#d62728",   # red
}
_MODEL_LABEL = {
    "cslc":  "CSLC  (kc_per_volume,  default 1e10 [Pa·m⁻¹ᐟ²])",
    "hydro": "Hydro (kh_pad,         default 5e9  [Pa/m])",
    "point": "Point (ke_pad_physical, default 5e4  [N/m])",
}


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    # Group: (pad, object) -> model -> rows sorted by k_mult.
    panels: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        panels[(r["pad_kind"], r["object_kind"])][r["contact_model"]].append(r)
    for key in panels:
        for model in panels[key]:
            panels[key][model].sort(key=lambda r: float(r["k_mult"]))

    panel_keys = sorted(panels.keys())
    n_panels = len(panel_keys)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7.0 * n_cols, 4.5 * n_rows),
        squeeze=False, sharex=True,
    )

    print(f"\nF-vs-k (per (pad, object) panel, log multiplier x-axis):")
    for ax, key in zip(axes.flat, panel_keys):
        pad, obj = key
        ax.set_title(f"pad = {pad}, object = {obj}")
        for model in ("cslc", "hydro", "point"):
            if model not in panels[key]:
                continue
            rs = panels[key][model]
            mults = np.array([float(r["k_mult"]) for r in rs])
            F = np.array([float(r["F_n_mean"]) for r in rs])
            F_std = np.array([float(r["F_n_std"]) for r in rs])
            color = _MODEL_COLOR[model]
            ax.plot(mults, F, "-o", color=color, label=_MODEL_LABEL[model])
            ax.fill_between(mults, F - F_std, F + F_std,
                            color=color, alpha=0.15)
        ax.axvline(1.0, color="0.5", linestyle=":", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Stiffness × default")
        ax.set_ylabel("Steady-state normal force F_n [N]")
        ax.grid(True, which="both", alpha=0.3)

    # Hide unused axes.
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    # Shared footer legend.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=_MODEL_COLOR[m], marker="o", lw=1.5,
               label=_MODEL_LABEL[m])
        for m in ("cslc", "hydro", "point")
    ]
    handles.append(
        Line2D([0], [0], color="0.5", linestyle=":", alpha=0.5,
               label="default stiffness (mult = 1)")
    )
    fig.legend(
        handles=handles,
        loc="lower center", ncol=2, fontsize=9,
        bbox_to_anchor=(0.5, -0.02), frameon=True,
    )

    fig.suptitle("F-vs-stiffness at δ = 1 mm  (per-model defaults shown in legend)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
