# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dome (fingertip-like) lattice flattening against a flat target.

A spherical-cap CSLC lattice ("the dome") is built via
:func:`cslc_main.theory.cslc_lattice.make_dome` and pressed against a
flat rectangular target a small distance above its apex.  After
solving the contact equilibrium we extract a y ≈ 0 cross-section
through the apex and plot the rest dome vs the deformed dome in the
x-z plane — the apex flattens out against the target as the
lattice's compliant skin redistributes the contact load.

Run::

    uv run python -m cslc_main.theory.plot_dome_flattening
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    make_dome,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import make_flat_face_target


# ---------------------------------------------------------------------------
#  Scene parameters
# ---------------------------------------------------------------------------

# Dome geometry.  R_pad = 10 mm and half-angle = 45° is a reasonable
# stand-in for a fingertip cap.  Sphere count chosen so each pad
# sphere radius (r_pad = spacing/2) is large enough that the chosen
# apex-to-target gap leaves the apex inside the half-space form
# (raw_apex = r_pad − h > 0).
N_SPHERES = 80
R_PAD_DOME_MM = 10.0
HALF_ANGLE = np.pi / 4.0
R_PAD_DOME = R_PAD_DOME_MM * 1.0e-3

# Lattice stiffnesses.  Soft anchor + moderate lateral coupling so
# the apex compression is visible and the surrounding spheres follow
# through the graph-Laplacian.  ``kc`` is cranked up so the contact
# force saturates the apex sphere's available overlap (δ_apex
# approaches r_pad − h), which is what makes the flattening obvious.
KA = 20.0                                    # N/m
KL = 60.0                                    # N/m
KC = 5.0e10                                  # N/m^(5/2) per-volume
EPS = 1.0e-9

# Flat target sitting just above the dome's apex.  The face normal
# points down (toward the dome below) so the alignment gate is
# satisfied at the apex (α = -(n_face·n_pad) = +1 → fully active).
# Span generously to cover the whole dome footprint plus a margin so
# the contact patch can spread across the apex region.
H_OFFSET_MM = 0.5                            # apex-to-target gap
H_OFFSET = H_OFFSET_MM * 1.0e-3
TARGET_PITCH_FACTOR = 0.4                    # pitch / spacing


# ---------------------------------------------------------------------------
#  Build & solve
# ---------------------------------------------------------------------------


def build_scene():
    lat, spacing, cap_area = make_dome(
        N=N_SPHERES, R_pad=R_PAD_DOME, half_angle=HALF_ANGLE,
        ka=KA, kl=KL, k_neighbors=6,
    )
    # Per-sphere radius from the Lloyd-style spacing identity.
    r_pad_sphere = spacing / 2.0

    # Flat target above the apex.  The dome's xy footprint reaches
    # ±R·sin(half_angle); pad the span by ~20% so the target overhangs
    # the dome rim.
    target_z = R_PAD_DOME + H_OFFSET
    xy_extent = R_PAD_DOME * np.sin(HALF_ANGLE) * 1.25
    target = make_flat_face_target(
        centre=np.array([0.0, 0.0, target_z]),
        normal=np.array([0.0, 0.0, -1.0]),
        span_u=2.0 * xy_extent,
        span_v=2.0 * xy_extent,
        pitch=spacing * TARGET_PITCH_FACTOR,
    )
    return lat, target, spacing, r_pad_sphere, cap_area


def solve(lat, target, r_pad_sphere):
    deltas, info = solve_lattice_contact(
        lat, target, kc=KC, r_pad=r_pad_sphere, eps=EPS,
        tol=1.0e-12, maxiter=4000,
    )
    return deltas, info


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------


def summarise(lat, deltas: np.ndarray, info: dict,
              spacing: float, r_pad_sphere: float, cap_area: float) -> None:
    delta_n = np.einsum("nj,nj->n", deltas, lat.n)
    apex_idx = int(np.argmax(lat.p[:, 2]))
    print("=" * 60)
    print(f"Dome: N={N_SPHERES} spheres, R={R_PAD_DOME_MM:.2f} mm, "
          f"half-angle={np.degrees(HALF_ANGLE):.1f}°")
    print(f"  mean NN spacing: {spacing*1e3:.3f} mm  "
          f"(r_pad per sphere = {r_pad_sphere*1e3:.3f} mm)")
    print(f"  cap area: {cap_area*1e6:.2f} mm²")
    print(f"Target: flat plane at z = {(R_PAD_DOME + H_OFFSET)*1e3:.2f} mm "
          f"(h = {H_OFFSET_MM:.2f} mm above apex)")
    print(f"Stiffnesses: ka={KA} N/m, kl={KL} N/m, kc={KC:.1e}")
    print("-" * 60)
    print(f"Solver: success={info['success']}, nit={info['nit']}, "
          f"|grad|={info['final_grad_norm']:.3e}, "
          f"Jacobi iters={info['jacobi_refine_iters']}")
    print(f"Active pairs: {info['n_active_pairs']}")
    print("-" * 60)
    print(f"Apex sphere (idx={apex_idx}): "
          f"δ_n = {delta_n[apex_idx]*1e3:+.4f} mm")
    print(f"Max δ_n in lattice: {np.max(delta_n)*1e3:+.4f} mm")
    print(f"Mean δ_n (active contacts only, δ_n > 0): "
          f"{np.mean(delta_n[delta_n > 1e-7])*1e3:+.4f} mm")
    print("=" * 60)


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot(lat, deltas: np.ndarray, spacing: float, out_path: Path) -> None:
    p_rest_mm = lat.p * 1e3
    q_def_mm = (lat.p - deltas) * 1e3

    # Cross-section: filter spheres near y = 0, then sort by x.  Use a
    # band ≈ one spacing wide so we always pick up several spheres.
    y_band_mm = spacing * 1e3
    mask = np.abs(p_rest_mm[:, 1]) < y_band_mm
    order = np.argsort(p_rest_mm[mask, 0])
    p_cs = p_rest_mm[mask][order]
    q_cs = q_def_mm[mask][order]

    target_z_mm = (R_PAD_DOME + H_OFFSET) * 1e3
    xy_extent_mm = R_PAD_DOME * np.sin(HALF_ANGLE) * 1.25 * 1e3

    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 13

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    # Rest dome — dashed grey reference shape, drawn first/under.
    ax.plot(p_cs[:, 0], p_cs[:, 2], "--", color="0.45",
            lw=1.8, alpha=0.85, label="rest dome", zorder=1)
    ax.plot(p_cs[:, 0], p_cs[:, 2], "o", color="0.45",
            ms=4, alpha=0.6, zorder=1.5)

    # Deformed dome — solid blue on top of the rest reference.
    ax.plot(q_cs[:, 0], q_cs[:, 2], "o-", color="C0",
            lw=2.0, ms=6, label="deformed dome", zorder=3)

    # Flat target plane — bold red line.
    ax.plot([-xy_extent_mm, xy_extent_mm],
            [target_z_mm, target_z_mm],
            color="C3", lw=2.4, label="flat target", zorder=2)

    # Highlight the flattened cap region with a light shaded band.
    # Width: spheres whose deformed z lies within 5% of the apex.
    apex_q_z = float(np.max(q_cs[:, 2]))
    flat_mask = (apex_q_z - q_cs[:, 2]) < 0.05 * (apex_q_z - q_cs[:, 2].min())
    if flat_mask.sum() >= 2:
        flat_x = q_cs[flat_mask, 0]
        ax.axvspan(flat_x.min(), flat_x.max(),
                   color="C0", alpha=0.07, zorder=0)

    ax.set_xlabel("x [mm]", fontsize=LABEL_SIZE)
    ax.set_ylabel("z [mm]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.set_aspect("equal", adjustable="datalim")
    # Tighten y-limits to focus on the dome top + target.
    z_lo = float(np.min(q_cs[:, 2])) - 0.5
    z_hi = target_z_mm + 0.8
    ax.set_ylim(z_lo, z_hi)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", fontsize=LEGEND_SIZE, framealpha=0.95,
              ncol=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    lat, target, spacing, r_pad_sphere, cap_area = build_scene()
    deltas, info = solve(lat, target, r_pad_sphere)
    summarise(lat, deltas, info, spacing, r_pad_sphere, cap_area)
    out_path = (Path(__file__).parent / "figures"
                / "dome_flattening.png")
    plot(lat, deltas, spacing, out_path)


if __name__ == "__main__":
    main()
