# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dome flattening under a flat target, swept over several k_l values.

Same idea as :mod:`cslc_main.theory.plot_dome_flattening` but the
lateral-coupling stiffness ``k_l`` is varied across several values
and every deformed dome is overlaid on a single 2D cross-section.

The sweep makes the role of lateral coupling visible:

  * ``k_l = 0``: only the apex sphere deforms (it isolates from
    its neighbours), so the dome's tip plunges sharply but the rest
    of the cap is unchanged.
  * Small ``k_l``: lateral springs start dragging the immediate
    neighbours along, so the dimple broadens.
  * Large ``k_l``: the top of the cap moves almost uniformly, the
    apex flattens into a wide plateau.

We use a sharper / more "fingertip-like" dome ($R = 5$ mm, half-angle
$60^\\circ$) and a strong contact stiffness so the apex saturates
its available Hertz overlap.

Run::

    uv run python -m cslc_main.theory.plot_dome_kl_sweep
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from cslc_main.theory.cslc_lattice import (
    make_dome,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import make_flat_face_target


# ---------------------------------------------------------------------------
#  Scene parameters
# ---------------------------------------------------------------------------

# Sharper dome: half the radius and a wider cap angle than the
# basic ``plot_dome_flattening`` scene so the rest dome reads
# clearly as a curved fingertip.
N_SPHERES = 80
R_PAD_DOME_MM = 5.0
HALF_ANGLE = np.pi / 3.0                     # 60° cap
R_PAD_DOME = R_PAD_DOME_MM * 1.0e-3

# Fixed anchor + strong contact.  ``kc`` is cranked up so the apex
# sphere saturates its available overlap (δ_apex → r_pad − h),
# the regime where the flattening is most dramatic.
KA = 30.0                                    # N/m
KC = 1.0e11                                  # N/m^(5/2)
EPS = 1.0e-9

# Lateral coupling values to sweep [N/m].  Include 0 explicitly so
# you can see the "only-apex-deforms" baseline.
KL_VALUES = np.array([0.0, 20.0, 100.0, 500.0, 2000.0])

# Flat target above the apex.  H_OFFSET must be < r_pad ≈ spacing/2
# so the apex sphere starts inside the half-space (raw_apex > 0).
H_OFFSET_MM = 0.35
H_OFFSET = H_OFFSET_MM * 1.0e-3
TARGET_PITCH_FACTOR = 0.4                    # pitch / spacing


# ---------------------------------------------------------------------------
#  Build & solve
# ---------------------------------------------------------------------------


def build_target(spacing: float):
    target_z = R_PAD_DOME + H_OFFSET
    xy_extent = R_PAD_DOME * np.sin(HALF_ANGLE) * 1.25
    return make_flat_face_target(
        centre=np.array([0.0, 0.0, target_z]),
        normal=np.array([0.0, 0.0, -1.0]),
        span_u=2.0 * xy_extent,
        span_v=2.0 * xy_extent,
        pitch=spacing * TARGET_PITCH_FACTOR,
    )


def build_dome(kl: float):
    """Build the dome lattice at this ``k_l``.  ``k_a`` is fixed."""
    lat, spacing, cap_area = make_dome(
        N=N_SPHERES, R_pad=R_PAD_DOME, half_angle=HALF_ANGLE,
        ka=KA, kl=float(kl), k_neighbors=6,
    )
    return lat, spacing, cap_area


def solve_for_kl(kl: float):
    lat, spacing, cap_area = build_dome(kl)
    r_pad_sphere = spacing / 2.0
    target = build_target(spacing)
    deltas, info = solve_lattice_contact(
        lat, target, kc=KC, r_pad=r_pad_sphere, eps=EPS,
        tol=1.0e-12, maxiter=4000,
    )
    return lat, deltas, info, spacing, r_pad_sphere


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def cross_section_curve(lat, deltas, spacing):
    """Return (x_mm, z_def_mm, z_rest_mm) sorted along x for y ≈ 0."""
    p_rest_mm = lat.p * 1e3
    q_def_mm = (lat.p - deltas) * 1e3
    y_band_mm = spacing * 1e3
    mask = np.abs(p_rest_mm[:, 1]) < y_band_mm
    order = np.argsort(p_rest_mm[mask, 0])
    return (p_rest_mm[mask][order, 0],
            q_def_mm[mask][order, 2],
            p_rest_mm[mask][order, 2])


def main_plot(profiles, spacing, out_path: Path) -> None:
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 12

    target_z_mm = (R_PAD_DOME + H_OFFSET) * 1e3
    xy_extent_mm = R_PAD_DOME * np.sin(HALF_ANGLE) * 1.25 * 1e3

    # Map k_l → colour.  symlog so 0 sits near the dark end.
    kl_positive = [kl for kl, *_ in profiles if kl > 0]
    norm = colors.LogNorm(vmin=min(kl_positive), vmax=max(kl_positive))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Rest dome as the grey reference.
    x_rest, _, z_rest = profiles[0][1:4]
    ax.plot(x_rest, z_rest, "--", color="0.45",
            lw=1.8, alpha=0.9, label="rest dome", zorder=1)
    ax.plot(x_rest, z_rest, "o", color="0.45",
            ms=3.5, alpha=0.6, zorder=1.5)

    # Deformed curve per k_l.
    for kl, x_cs, z_def, _ in profiles:
        if kl == 0.0:
            color = cmap(0.0)
            label = r"$k_\ell = 0$ N/m"
        else:
            color = cmap(norm(kl))
            label = rf"$k_\ell = {kl:g}$ N/m"
        ax.plot(x_cs, z_def, "o-", color=color, lw=1.8, ms=5,
                label=label, zorder=3)

    # Flat target plane.
    ax.plot([-xy_extent_mm, xy_extent_mm],
            [target_z_mm, target_z_mm],
            color="C3", lw=2.4, label="flat target", zorder=2)

    ax.set_xlabel("x [mm]", fontsize=LABEL_SIZE)
    ax.set_ylabel("z [mm]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    # Tighten y to focus on the cap top.
    z_lo = min(profile[2].min() for profile in profiles) - 0.3
    ax.set_ylim(z_lo, target_z_mm + 0.5)
    ax.legend(loc="lower left", fontsize=LEGEND_SIZE, framealpha=0.95,
              ncol=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    print(f"Dome: N={N_SPHERES}, R={R_PAD_DOME_MM:.1f} mm, "
          f"half-angle={np.degrees(HALF_ANGLE):.0f}°")
    print(f"Fixed: ka={KA} N/m, kc={KC:.1e}, "
          f"h above apex = {H_OFFSET_MM:.2f} mm")
    print(f"Sweeping k_l over {list(KL_VALUES)} N/m")
    print("-" * 60)

    profiles = []
    spacing_ref = None
    for kl in KL_VALUES:
        lat, deltas, info, spacing, r_pad_sphere = solve_for_kl(float(kl))
        if spacing_ref is None:
            spacing_ref = spacing
        x_cs, z_def, z_rest = cross_section_curve(lat, deltas, spacing)
        profiles.append((float(kl), x_cs, z_def, z_rest))
        delta_n = np.einsum("nj,nj->n", deltas, lat.n)
        apex_idx = int(np.argmax(lat.p[:, 2]))
        print(f"  k_l = {kl:8.1f} N/m  →  "
              f"apex δ_n = {delta_n[apex_idx]*1e3:+.4f} mm  "
              f"max δ_n = {np.max(delta_n)*1e3:.4f} mm  "
              f"(Jacobi iters = {info['jacobi_refine_iters']})")
    print("-" * 60)

    out_path = (Path(__file__).parent / "figures"
                / "dome_kl_sweep.png")
    main_plot(profiles, spacing_ref, out_path)


if __name__ == "__main__":
    main()
