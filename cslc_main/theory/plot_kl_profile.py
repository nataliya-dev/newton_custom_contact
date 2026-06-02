# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cross-section profiles of the lattice dimple at several k_l values.

Same scene as :mod:`cslc_main.theory.plot_flat_lattice_deflection`
(15×15 flat sphere lattice, single point-set target above the centre).
For each value of the lateral-coupling stiffness ``k_l`` we re-solve
the contact equilibrium and extract ``δ_n`` along the centre row of
the lattice; each row becomes a coloured curve.

At ``k_l = 0`` the centre sphere alone takes the load — a tall, narrow
peak.  As ``k_l`` grows the peak broadens and flattens because the
graph-Laplacian coupling shuttles load to neighbouring spheres.

Run::

    uv run python -m cslc_main.theory.plot_kl_profile
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

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

KA = 100.0
KC = 1.0e9
EPS = 1.0e-9

H_OFFSET_MM = 1.0
H_OFFSET = H_OFFSET_MM * 1.0e-3
INDENT_AREA = np.pi * R_PAD * R_PAD

# k_l values to compare.  Hand-picked so the curves are visually
# distinct: zero baseline, then geometrically spaced.
KL_VALUES = np.array([0.0, 25.0, 100.0, 400.0, 1600.0])


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def make_target() -> PointSetTarget:
    positions = np.array([[0.0, 0.0, H_OFFSET]])
    normals = np.array([[0.0, 0.0, -1.0]])
    areas = np.array([INDENT_AREA])
    return PointSetTarget(positions=positions, normals=normals, areas=areas)


def cross_section_for_kl(kl: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_mm, δ_n_mm) along the centre row of the lattice."""
    lat = make_flat_grid(
        n_u=N_U, n_v=N_V, spacing=SPACING, ka=KA, kl=float(kl),
        normal=np.array([0.0, 0.0, 1.0]),
        diagonals=False,
    )
    target = make_target()
    deltas, _ = solve_lattice_contact(
        lat, target, kc=KC, r_pad=R_PAD, eps=EPS,
        tol=1.0e-12, maxiter=2000,
    )
    delta_n = np.einsum("nj,nj->n", deltas, lat.n)
    # Centre row in (u, v): i fixed at N_U // 2, sweep j.
    centre_row_idx = (N_U // 2) * N_V + np.arange(N_V)
    x_mm = lat.p[centre_row_idx, 1] * 1e3
    return x_mm, delta_n[centre_row_idx] * 1e3


def sweep_all_profiles():
    """Solve every k_l and collect their centre-row profiles."""
    profiles = []
    for kl in KL_VALUES:
        x_mm, dn_mm = cross_section_for_kl(float(kl))
        profiles.append((kl, x_mm, dn_mm))
        peak = float(np.max(dn_mm))
        print(f"  k_l = {kl:8.2f} N/m  →  peak δ_n = {peak:.4f} mm")
    return profiles


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot(profiles, out_path: Path) -> None:
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 12

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Map k_l → colour via a log-ish normaliser so the 0 baseline
    # gets the darkest end and high k_l gets the brightest.
    kl_min = float(np.min([kl for kl, _, _ in profiles if kl > 0]))
    kl_max = float(np.max([kl for kl, _, _ in profiles]))
    norm = colors.LogNorm(vmin=kl_min, vmax=kl_max)
    cmap = cm.viridis

    # Plot the deformed centre position q_z = -δ_n so the profile reads
    # as a downward dimple (matches the deformation animation, where
    # the lattice is pressed into −z by the indenter from above).
    for kl, x_mm, dn_mm in profiles:
        if kl == 0.0:
            color = cmap(0.0)
            label = r"$k_\ell = 0$ N/m"
        else:
            color = cmap(norm(kl))
            label = rf"$k_\ell = {kl:g}$ N/m"
        ax.plot(x_mm, -dn_mm, "o-", color=color, ms=5, lw=1.6,
                label=label)

    ax.axhline(0.0, color="0.7", lw=0.5)
    ax.set_xlabel("position along centre row [mm]", fontsize=LABEL_SIZE)
    ax.set_ylabel("z [mm]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.grid(alpha=0.3)
    # Legend in the lower right, away from the dimple in the middle.
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    print(f"Lattice: {N_U}x{N_V} flat grid, spacing={SPACING_MM:.2f} mm")
    print(f"Fixed: ka={KA} N/m, kc={KC:.1e} N/m^(5/2), "
          f"h={H_OFFSET_MM:.2f} mm")
    print(f"Sweeping k_l over {len(KL_VALUES)} values: "
          f"{list(KL_VALUES)} N/m")
    print("-" * 60)
    profiles = sweep_all_profiles()
    print("-" * 60)
    out_path = (Path(__file__).parent / "figures"
                / "flat_lattice_kl_profile.png")
    plot(profiles, out_path)


if __name__ == "__main__":
    main()
