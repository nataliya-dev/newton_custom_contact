# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep the lateral-coupling stiffness k_l and plot peak deflection.

Same scene as :mod:`cslc_main.theory.plot_flat_lattice_deflection`
(15×15 flat sphere lattice, single point-set target above the centre)
but with the lateral stiffness ``k_l`` swept across a range of values.
For each ``k_l`` we re-solve the contact equilibrium and record the
maximum normal compression ``max |δ_n|``, which always occurs at the
centre sphere by symmetry.

The plot shows that increasing ``k_l`` redistributes the contact load
to neighbouring spheres, dropping the peak deflection at the centre.

Run::

    uv run python -m cslc_main.theory.plot_kl_sweep
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    make_flat_grid,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import PointSetTarget


# ---------------------------------------------------------------------------
#  Scene parameters
# ---------------------------------------------------------------------------

N_U = 15
N_V = 15
SPACING_MM = 3.0
R_PAD_MM = SPACING_MM / 2.0
SPACING = SPACING_MM * 1.0e-3
R_PAD = R_PAD_MM * 1.0e-3

KA = 100.0                                   # N/m  — fixed
KC = 1.0e9                                   # N/m^(5/2)
EPS = 1.0e-9

H_OFFSET_MM = 1.0
H_OFFSET = H_OFFSET_MM * 1.0e-3
INDENT_AREA = np.pi * R_PAD * R_PAD

# k_l sweep range.  Logarithmically spaced with a leading 0 sample so
# we capture the "no lateral coupling" baseline.
KL_VALUES = np.concatenate([
    np.array([0.0]),
    np.geomspace(10.0, 2000.0, 24),
])                                           # N/m


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def make_target() -> PointSetTarget:
    positions = np.array([[0.0, 0.0, H_OFFSET]])
    normals = np.array([[0.0, 0.0, -1.0]])
    areas = np.array([INDENT_AREA])
    return PointSetTarget(positions=positions, normals=normals, areas=areas)


def sweep_kl(kl_values: np.ndarray) -> np.ndarray:
    """Return max |δ_n| in mm for each value of k_l."""
    target = make_target()
    max_dn_mm = np.zeros_like(kl_values)
    delta_prev = None
    for k, kl in enumerate(kl_values):
        lat = make_flat_grid(
            n_u=N_U, n_v=N_V, spacing=SPACING, ka=KA, kl=float(kl),
            normal=np.array([0.0, 0.0, 1.0]),
            diagonals=False,
        )
        deltas, info = solve_lattice_contact(
            lat, target, kc=KC, r_pad=R_PAD, eps=EPS,
            delta0=delta_prev, tol=1.0e-12, maxiter=2000,
        )
        delta_n = np.einsum("nj,nj->n", deltas, lat.n)
        max_dn_mm[k] = float(np.max(np.abs(delta_n))) * 1e3
        delta_prev = deltas
        print(f"  k_l = {kl:8.2f} N/m  →  max |δ_n| = {max_dn_mm[k]:.4f} mm  "
              f"(Jacobi iters = {info['jacobi_refine_iters']})")
    return max_dn_mm


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot(kl_values: np.ndarray, max_dn_mm: np.ndarray, out_path: Path) -> None:
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 13

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    # symlog x-axis: linear below threshold (to include k_l = 0), log above.
    ax.set_xscale("symlog", linthresh=10.0)

    ax.plot(kl_values, max_dn_mm, "o-", color="C0",
            ms=6, lw=1.6, label=fr"$k_a = {KA:g}$ N/m")

    ax.set_xlabel(r"lateral stiffness $k_\ell$ [N/m]",
                  fontsize=LABEL_SIZE)
    ax.set_ylabel(r"max $|\delta_n|$ [mm]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    print(f"Lattice: {N_U}x{N_V} flat grid, spacing={SPACING_MM:.2f} mm")
    print(f"Fixed: ka={KA} N/m, kc={KC:.1e} N/m^(5/2), "
          f"h={H_OFFSET_MM:.2f} mm")
    print(f"Sweeping k_l over {len(KL_VALUES)} values "
          f"from {KL_VALUES[0]:.0f} to {KL_VALUES[-1]:.0f} N/m")
    print("-" * 60)
    max_dn_mm = sweep_kl(KL_VALUES)
    print("-" * 60)
    out_path = (Path(__file__).parent / "figures"
                / "flat_lattice_kl_sweep.png")
    plot(KL_VALUES, max_dn_mm, out_path)


if __name__ == "__main__":
    main()
