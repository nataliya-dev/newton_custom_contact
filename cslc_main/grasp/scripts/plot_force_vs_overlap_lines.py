# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""F_n during HOLD vs commanded overlap, one line per (model, k) variant.

Shows pure F_n scaling — no lift/no-lift markers, just the response
curves so the reader can see how each model's force law responds to
overlap at different stiffness levels.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Reference stiffness for computing the mult label per cell.
# CSLC: our project default. Hydro/Point: Newton library defaults
# (newton/_src/sim/builder.py).
_MODEL_DEFAULT_K = {
    "cslc":  1.0e10,
    "hydro": 1.0e10,
    "point": 2.5e3,
}

# Hue per model, lightness per stiffness mult.
_MODEL_BASE_COLOR = {
    "cslc":  "#1f77b4",
    "hydro": "#2ca02c",
    "point": "#d62728",
}
# Line style per stiffness mult, sorted ascending.
_LINESTYLE_BY_RANK = {
    0: ":",   # lowest stiffness — dotted
    1: "--",  # mid — dashed
    2: "-",   # default — solid
    3: "-.",  # higher — dash-dot
    4: (0, (3, 1, 1, 1)),  # highest — dash-dot-dot
}


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, nargs="+",
                        help="One or more sweep CSVs (merged before plotting).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="F_n during HOLD vs commanded overlap")
    parser.add_argument("--log-y", action="store_true",
                        help="Use log scale on the y-axis (helpful for "
                             "wide F_n spans across models).")
    args = parser.parse_args()

    rows: list[dict] = []
    for p in args.input:
        rows.extend(_load_rows(p))
    if not rows:
        raise SystemExit("No rows loaded")

    pad_kind = rows[0]["pad_kind"]
    obj_kind = rows[0]["object_kind"]

    # Group: (model, k_mult) → sorted by face_pen
    by_pair: dict = defaultdict(list)
    for r in rows:
        m = r["contact_model"]
        k = float(r["k_pad"])
        mult = round(k / _MODEL_DEFAULT_K[m], 4)
        by_pair[(m, mult)].append(r)
    for k in by_pair:
        by_pair[k].sort(key=lambda r: float(r["face_pen_mm"]))

    # Rank stiffness mults per model for linestyle assignment
    mults_per_model: dict = defaultdict(set)
    for (m, mult) in by_pair:
        mults_per_model[m].add(mult)
    rank_per_model: dict = {}
    for m, muset in mults_per_model.items():
        sorted_mults = sorted(muset)
        rank_per_model[m] = {mu: i for i, mu in enumerate(sorted_mults)}

    fig, ax = plt.subplots(figsize=(9, 6))
    _PLOT_ORDER = ["point", "hydro", "cslc"]
    for m in _PLOT_ORDER:
        for mult in sorted(mults_per_model.get(m, [])):
            rs = by_pair.get((m, mult), [])
            if not rs:
                continue
            depths = np.array([float(r["face_pen_mm"]) for r in rs])
            F = np.array([float(r["F_n_hold"]) for r in rs])
            rank = rank_per_model[m][mult]
            ls = _LINESTYLE_BY_RANK.get(rank, "-")
            color = _MODEL_BASE_COLOR.get(m, "0.5")
            label = f"{m} k={mult:.2f}×default"
            ax.plot(depths, F, color=color, linestyle=ls, marker="o",
                    markersize=5, linewidth=2, label=label)

    ax.set_xlabel("Commanded overlap δ [mm]")
    ax.set_ylabel("F_n during HOLD [N]")
    if args.log_y:
        ax.set_yscale("log")
    ax.set_title(f"{args.title} — pad={pad_kind}, object={obj_kind}")
    ax.grid(True, alpha=0.3, which="both" if args.log_y else "major")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
