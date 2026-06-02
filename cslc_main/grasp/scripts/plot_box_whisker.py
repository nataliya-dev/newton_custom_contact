# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Box-and-whisker of F_n during HOLD per (scenario, model).

For each (pad, object) scenario we have multiple (k, δ) samples per
model.  This plot shows the *distribution* of F_n values achieved when
the grasp held, broken out by model, so you can read at a glance:
  - the typical operating force regime per scenario
  - the spread (k × δ sensitivity)
  - which model achieves stable lifts at the lowest forces

Coulomb floor (W / (2·μ)) is overlaid per scenario as a horizontal
reference (the minimum two-pad normal force that can hold the object's
weight without slipping).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Bright, saturated per-model colours.
_MODEL_COLOR = {
    "cslc":  "#2a9df4",   # bright blue
    "hydro": "#3ddc55",   # bright green
    "point": "#ff4d4d",   # bright red
}
_MODEL_ORDER = ["point", "hydro", "cslc"]
_BOX_ALPHA = 0.85         # box fill opacity (brighter = higher)

# Object weights [N] (must match params.py: density × volume × g).
#   sphere — tennis ball, 58 g;  box — 67 mm cube, 111 g;
#   bunny  — 100 mm mesh, 100 g.
_OBJ_W = {"sphere": 0.569, "box": 1.086, "bunny": 0.982}
_MU = 0.5

# Maximum XY drift [m] for a grasp to count as physically held.  The
# runner's ``Metrics.held`` flag is purely KINEMATIC (object rose >5 mm
# and stayed <0.5 m); it does NOT check that the grasp was stable.  On
# the irregular bunny mesh that lets the object slide several mm through
# the pads while still "rising", producing held=1 rows whose F_n is far
# below the Coulomb floor (a physical impossibility: friction cannot
# hold the weight).  We require slip < this threshold so a "held" grasp
# actually stayed put.  5 mm keeps the genuine bunny grips (which settle
# with ~2.5-3 mm of lateral motion) while dropping the slid-through ones
# (5-22 mm).
_MAX_HOLD_SLIP_M = 5.0e-3


def _coulomb_floor(object_kind: str) -> float | None:
    """Two-pad slip floor F_n,min = W / (2μ) for an object, or None."""
    w = _OBJ_W.get(object_kind)
    return None if w is None else w / (2.0 * _MU)


def _is_physically_held(row: dict) -> bool:
    """Physical hold test: the kinematic ``held`` flag AND the grasp was
    stable, i.e. friction could support the weight (F_n ≥ Coulomb floor)
    and the object did not slide out (xy_slip < _MAX_HOLD_SLIP_M).

    This corrects the raw ``held`` column, which counts any object that
    merely rose >5 mm even if it was sliding through the pads the whole
    time at a force too low to hold it.

    If a precomputed ``held_physical`` column is present (cleaned CSV
    from clean_held.py), it is used directly.
    """
    if "held_physical" in row and row["held_physical"] != "":
        return int(row["held_physical"]) == 1
    if int(row["held"]) != 1:
        return False
    floor = _coulomb_floor(row["object_kind"])
    if floor is not None and float(row["F_n_hold"]) < floor:
        return False
    return float(row["xy_slip_max"]) < _MAX_HOLD_SLIP_M


def _load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, nargs="+",
                        help="One or more sweep CSVs (concatenated before plotting).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Operating-force distribution per scenario")
    parser.add_argument("--log-y", action="store_true",
                        help="Log scale on F_n axis (helpful when ranges span "
                             "two decades or more).")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    if not rows:
        raise SystemExit("No rows loaded")

    # Group by (scenario, model); split held vs failed.  Keyed by
    # (pad, object) so we can look up the object weight for the floor.
    by_cell: dict = defaultdict(lambda: {"held": [], "failed": 0})
    scen_object: dict[str, str] = {}
    for r in rows:
        pad, obj = r["pad_kind"], r["object_kind"]
        scenario = f"{pad}/{obj}"
        scen_object[scenario] = obj
        model = r["contact_model"]
        # Physical hold: kinematic rise AND friction-stable (F_n ≥ Coulomb
        # floor, slip < threshold).  Corrects the raw ``held`` flag, which
        # counts objects that merely rose while sliding through the pads.
        if _is_physically_held(r):
            by_cell[(scenario, model)]["held"].append(float(r["F_n_hold"]))
        else:
            by_cell[(scenario, model)]["failed"] += 1

    scenarios = sorted({s for (s, _m) in by_cell})
    models = [m for m in _MODEL_ORDER
              if any((s, m) in by_cell for s in scenarios)]

    # ── Styling ──
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 12, "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(max(11, 1.9 * len(scenarios) + 3), 7.0))

    # Narrow boxes clustered tightly within each scenario, so the gaps
    # *between* scenarios read clearly.
    width = 0.15
    step = width * 1.18          # inter-model spacing within a scenario
    n_models = len(models)
    x_pos = np.arange(len(scenarios))

    # Global data max for headroom + annotation placement (linear only).
    all_held = [v for cell in by_cell.values() for v in cell["held"]]
    data_max = max(all_held) if all_held else 1.0

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * step
        positions = x_pos + offset
        data = [by_cell.get((s, model), {}).get("held", []) or [np.nan]
                for s in scenarios]
        ax.boxplot(
            data, positions=positions, widths=width, patch_artist=True,
            boxprops=dict(facecolor=_MODEL_COLOR[model], alpha=_BOX_ALPHA,
                          edgecolor="black", linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            flierprops=dict(markerfacecolor=_MODEL_COLOR[model],
                            markeredgecolor="black", markersize=4, alpha=0.7),
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=6),
        )

    # ── Axis limits (set before annotating so offsets are stable) ──
    if args.log_y:
        ax.set_yscale("log")
        pos_vals = [v for v in all_held if v > 0]
        ax.set_ylim(max(0.05, 0.5 * min(pos_vals)) if pos_vals else 0.1,
                    data_max * 2.0)
    else:
        ax.set_ylim(0, data_max * 1.20)

    y0, y1 = ax.get_ylim()

    def _label_y(box_top: float) -> float:
        """A little above the box top, in the current y-scale."""
        if args.log_y:
            return box_top * 1.12
        return box_top + 0.018 * (y1 - y0)

    # ── Sample-count annotations: one compact line above each box ──
    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2) * step
        for j, s in enumerate(scenarios):
            cell = by_cell.get((s, model), {})
            n_held = len(cell.get("held", []))
            n_failed = cell.get("failed", 0)
            if n_held == 0 and n_failed == 0:
                continue
            held_vals = cell.get("held", [])
            top = max(held_vals) if held_vals else y0
            label = f"n={n_held}" + (f" ({n_failed}✗)" if n_failed else "")
            ax.annotate(label, xy=(x_pos[j] + offset, _label_y(top)),
                        ha="center", va="bottom", fontsize=8.5,
                        color=_MODEL_COLOR[model])

    # ── Coulomb floor per scenario: dotted line + one compact label ──
    for j, s in enumerate(scenarios):
        floor = _coulomb_floor(scen_object[s])
        if floor is None:
            continue
        ax.hlines(floor, j - 0.5, j + 0.5, colors="0.45",
                  linestyles=(0, (1, 1.5)), linewidth=1.6, zorder=1)
        ax.annotate(f"{floor:.2f} N", xy=(j - 0.48, floor),
                    xytext=(0, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color="0.45")

    # ── Legend ──
    handles = [Patch(facecolor=_MODEL_COLOR[m], edgecolor="black",
                     alpha=_BOX_ALPHA, label=m) for m in models]
    handles += [
        Line2D([0], [0], color="0.45", linestyle=(0, (1, 1.5)), linewidth=1.6,
               label=r"Coulomb floor  $W/2\mu$"),
        Line2D([0], [0], marker="D", markerfacecolor="white",
               markeredgecolor="black", linestyle="None", label="mean"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.95)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace("/", " / ") for s in scenarios])
    ax.set_xlim(-0.6, len(scenarios) - 0.4)
    ax.set_xlabel("scenario  (pad / object)")
    ax.set_ylabel(r"normal force $F_n$ during HOLD  [N]")
    ax.set_title(args.title, pad=12)
    ax.grid(True, axis="y", alpha=0.3,
            which="both" if args.log_y else "major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
