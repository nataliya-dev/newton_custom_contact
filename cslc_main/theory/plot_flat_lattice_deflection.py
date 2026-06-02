# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""3D + radial-decay visualisation of a flat CSLC lattice under a centred load.

Builds a 15x15 flat sphere lattice in the x-y plane and presses a
single point-set target sample into the centre of the lattice from
above (target outward face normal pointing down toward the pad).  The
lattice equilibrium is solved by
:func:`cslc_main.theory.cslc_lattice.solve_lattice_contact` under the
theory.md §3-§6 force law:

  * Hertz-like ``phi_eff = sigma_eps(raw) * sqrt(sigma_eps(raw) + eps)``
    (theory.md eq:phi-eff).
  * Tangential locality kernel ``w_t = Sigma_eps(r_pad - d_t)`` with
    half-width ``r_pad`` (theory.md §3.5).
  * Anchor + graph-Laplacian lateral coupling (theory.md §6.1-§6.2).

The contact force is delivered only to the centre sphere (the target
sample sits inside the centre sphere's tangential-locality disc; for
all other pad spheres ``d_t > r_pad`` so ``w_t ≈ 0`` and the contact
contribution vanishes).  The surrounding dimple is therefore built
entirely by lateral-spring spreading through the graph Laplacian.

Run::

    uv run python -m cslc_main.theory.plot_flat_lattice_deflection
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from cslc_main.theory.cslc_lattice import (
    lattice_contact_normal_forces,
    make_flat_grid,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import PointSetTarget


# ---------------------------------------------------------------------------
#  Scene parameters
# ---------------------------------------------------------------------------

N_U = 15
N_V = 15
SPACING_MM = 3.0                # mm  — lattice spacing
R_PAD_MM = SPACING_MM / 2.0     # mm  — half-spacing tiling (theory.md §3.5)

SPACING = SPACING_MM * 1.0e-3   # m
R_PAD = R_PAD_MM * 1.0e-3       # m

KA = 100.0                      # N/m  — anchor stiffness
KL = 200.0                      # N/m  — lateral stiffness (graph-Laplacian).
                                #        decay length ℓ = h·√(kl/ka) ≈ 1.4
                                #        spacings (√(kl/ka) is dimensionless)
KC = 1.0e9                      # N/m^(7/2)  — per-volume contact stiffness for
                                #              the Hertz-like force law
                                #              (theory.md §10)

EPS = 1.0e-9                    # smoothing width for the half-space surrogate

H_OFFSET_MM = 1.0               # mm  — target sample's z height above the pad
H_OFFSET = H_OFFSET_MM * 1.0e-3 # m
INDENT_AREA = np.pi * R_PAD * R_PAD    # m²  — one disc-of-radius-r tile


# ---------------------------------------------------------------------------
#  Build & solve
# ---------------------------------------------------------------------------


def build_scene():
    lat = make_flat_grid(
        n_u=N_U, n_v=N_V, spacing=SPACING, ka=KA, kl=KL,
        normal=np.array([0.0, 0.0, 1.0]),
        diagonals=False,
    )
    positions = np.array([[0.0, 0.0, H_OFFSET]])
    normals = np.array([[0.0, 0.0, -1.0]])
    areas = np.array([INDENT_AREA])
    target = PointSetTarget(positions=positions, normals=normals, areas=areas)
    return lat, target


def solve(lat, target):
    deltas, info = solve_lattice_contact(
        lat, target, kc=KC, r_pad=R_PAD, eps=EPS,
        tol=1.0e-12, maxiter=2000,
    )
    return deltas, info


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------


def summarise(lat, target, deltas: np.ndarray, info: dict) -> None:
    delta_n = np.einsum("nj,nj->n", deltas, lat.n)
    centre_idx = (N_U // 2) * N_V + (N_V // 2)
    # Per-sphere contact force at the equilibrium state.  F_contact is
    # the physical contact force on each pad sphere; f_n is its
    # magnitude projected on the pad's own outward normal.
    F_contact, f_n = lattice_contact_normal_forces(
        lat, target, deltas, kc=KC, r_pad=R_PAD, eps=EPS,
    )
    F_centre_vec = F_contact[centre_idx]                  # (3,) N
    F_centre_mag = float(np.linalg.norm(F_centre_vec))    # |F| N
    print("=" * 60)
    print(f"Lattice: {N_U}x{N_V} flat grid, spacing={SPACING_MM:.2f} mm")
    print(f"Target:  single sample at (0, 0, {H_OFFSET_MM:.2f} mm), "
          f"n_face=(0,0,-1)")
    print(f"Stiffnesses: ka={KA} N/m, kl={KL} N/m, "
          f"kc={KC:.2e} N/m^(7/2)")
    # √(kl/ka) is dimensionless (ratio of N/m stiffnesses); the physical
    # decay length is ℓ = h·√(kl/ka) with h = lattice spacing.
    print(f"Decay length ℓ = h·√(kl/ka) = {np.sqrt(KL/KA):.3f} spacings "
          f"= {np.sqrt(KL/KA)*SPACING_MM:.3f} mm")
    print(f"Rest overlap raw0 at centre: "
          f"{(R_PAD_MM - H_OFFSET_MM):.3f} mm")
    print("-" * 60)
    print(f"Solver: success={info['success']}, nit={info['nit']}, "
          f"|grad|={info['final_grad_norm']:.3e}")
    print(f"Active pairs: {info['n_active_pairs']}")
    print(f"Jacobi refine iters: {info['jacobi_refine_iters']}")
    print("-" * 60)
    print(f"Centre sphere δ_n: {delta_n[centre_idx]*1e3:.4f} mm")
    print(f"Centre sphere contact force:")
    print(f"  vector  F_contact = ({F_centre_vec[0]:+.4e}, "
          f"{F_centre_vec[1]:+.4e}, {F_centre_vec[2]:+.4e}) N")
    print(f"  magnitude         = {F_centre_mag*1e3:.3f} mN  "
          f"({F_centre_mag:.4e} N)")
    print(f"  projected on n_pad: {abs(F_centre_vec[2])*1e3:.3f} mN "
          f"(z-component magnitude)")
    print("δ_n along centre row (ring distance from centre):")
    for j in range(N_V):
        idx = (N_U // 2) * N_V + j
        ring = abs(j - N_V // 2)
        marker = " <-- centre" if ring == 0 else ""
        print(f"  ring={ring:2d}  j={j:2d}  "
              f"δ_n={delta_n[idx]*1e3:+.4f} mm{marker}")
    print(f"\nMax δ_n: {np.max(delta_n)*1e3:+.4f} mm")
    print(f"Min δ_n: {np.min(delta_n)*1e3:+.4f} mm")
    print("=" * 60)


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot(lat, deltas: np.ndarray, info: dict, out_path: Path) -> None:
    p_rest_mm = lat.p * 1e3                          # mm
    q_def_mm = (lat.p - deltas) * 1e3                # mm, deformed centres
    delta_n_mm = np.einsum("nj,nj->n", deltas, lat.n) * 1e3  # mm

    # Publication-ready typography for IEEE one-column figures.  Labels
    # and ticks are sized so they remain legible when the figure is
    # included via \includegraphics[width=\columnwidth].  Panel titles
    # are omitted — they live in the caption instead.
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 13

    fig = plt.figure(figsize=(9.0, 4.2))

    # --- Panel A: 3D view of deformed lattice (z in real mm).
    # matplotlib auto-fits the z axis to the data extent, so the dimple
    # is still visible inside the plot box even though δ_n is sub-mm
    # while the lattice spans ~42 mm.  All axis ticks are real mm.
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    for (i, j) in lat.edges:
        ax3d.plot(
            [q_def_mm[i, 0], q_def_mm[j, 0]],
            [q_def_mm[i, 1], q_def_mm[j, 1]],
            [q_def_mm[i, 2], q_def_mm[j, 2]],
            color="0.30", linewidth=0.6, alpha=0.7, zorder=1,
        )
    ax3d.scatter(
        q_def_mm[:, 0], q_def_mm[:, 1], q_def_mm[:, 2],
        c=delta_n_mm, cmap=cm.viridis, s=35, edgecolor="black",
        linewidth=0.25, zorder=3,
    )
    ax3d.set_xlabel("x [mm]", fontsize=LABEL_SIZE, labelpad=10)
    ax3d.set_ylabel("y [mm]", fontsize=LABEL_SIZE, labelpad=10)
    ax3d.set_zlabel("z [mm]", fontsize=LABEL_SIZE, labelpad=14)
    ax3d.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax3d.view_init(elev=22.0, azim=-55.0)

    # --- Panel B: log-y radial decay (lattice Green's function).
    # The continuum limit of the anchor + graph-Laplacian operator is
    # the 2D screened-Poisson (Helmholtz/Yukawa) operator, whose Green's
    # function is the modified Bessel function K_0(r/ℓ) with decay length
    # ℓ = h·√(kl/ka) (h = lattice spacing; √(kl/ka) is dimensionless).
    # Its large-r asymptote is the exponential envelope exp(−r/ℓ), so the
    # grey reference line plots that decay envelope (normalised at the
    # centre) rather than K_0 itself, which diverges as r→0.  It
    # establishes that the lattice's radial spread follows a known
    # analytic form set entirely by the lateral / anchor stiffness ratio
    # — without it the data alone would just be "some decay".  Reported
    # as ℓ in the legend; legend kept terse so caption can explain.
    axD = fig.add_subplot(1, 2, 2)
    r_xy_mm = np.linalg.norm(p_rest_mm[:, :2], axis=1)
    dn_abs_mm = np.abs(delta_n_mm)
    floor = max(1.0e-6, dn_abs_mm.max() * 1.0e-8)
    axD.semilogy(r_xy_mm, np.maximum(dn_abs_mm, floor),
                 "o", color="C3", ms=5, alpha=0.7, label="data")
    L_decay_mm = np.sqrt(KL / KA) * SPACING_MM   # ℓ = h·√(kl/ka), mm
    rr_mm = np.linspace(0.0, r_xy_mm.max(), 200)
    centre_value_mm = dn_abs_mm[(N_U // 2) * N_V + (N_V // 2)]
    axD.semilogy(rr_mm, centre_value_mm * np.exp(-rr_mm / L_decay_mm), "-",
                 color="0.35", lw=1.4,
                 label=r"envelope $\exp(-r/\ell)$")
    axD.set_xlabel("distance from centre [mm]",
                   fontsize=LABEL_SIZE)
    axD.set_ylabel(r"$|\delta_n|$ [mm]", fontsize=LABEL_SIZE, labelpad=8)
    axD.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    axD.grid(alpha=0.3, which="both")
    # Anchor the legend to the lower-left so it doesn't ride on top of
    # the data points / y-axis label in the upper-right corner.
    axD.legend(loc="lower left", fontsize=LEGEND_SIZE, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.5, w_pad=5.0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    lat, target = build_scene()
    deltas, info = solve(lat, target)
    summarise(lat, target, deltas, info)
    out_path = (Path(__file__).parent / "figures"
                / "flat_lattice_contact_deflection.png")
    plot(lat, deltas, info, out_path)


if __name__ == "__main__":
    main()
