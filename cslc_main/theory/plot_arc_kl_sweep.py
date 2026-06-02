# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inverted curved arc lattice (finger) pressing down on a flat ground.

The arc is flipped so its apex points downward toward a rigid flat
ground plane that, in the undeformed state, just touches the apex.
A small extra "press" depth shifts the contact plane up into the
arc by ``PRESS_DEPTH_MM``, driving the lattice toward equilibrium
where the apex compresses upward into the finger body.

We sweep ``k_l`` from soft to firm and plot the deformed arcs to
show how the finger's compliance changes with lateral stiffness.

Run::

    uv run python -m cslc_main.theory.plot_arc_kl_sweep
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from cslc_main.theory.cslc_lattice import (
    Lattice,
    make_arc,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import make_flat_face_target


# ---------------------------------------------------------------------------
#  Scene parameters
# ---------------------------------------------------------------------------

# Arc geometry.  R = 5 mm with arc-length spacing 0.7 mm and N = 21
# spheres → an arc that spans ±80° around the apex, with each pad
# sphere of radius r_pad = 0.35 mm.  Sharper than the previous dome
# (curvature is high relative to sphere size) so the visual
# "fingertip" reads cleanly.
# Gentle fingertip-like arc.  R = 5 mm with 0.7 mm spacing gives a
# pad whose curvature reads as a fingertip cross-section, with each
# sphere small enough to render the deformed-shape evolution
# smoothly.
N_SPHERES = 31
R_ARC_MM = 5.0
ARC_LENGTH_SPACING_MM = 0.7
R_ARC = R_ARC_MM * 1.0e-3
ARC_LENGTH_SPACING = ARC_LENGTH_SPACING_MM * 1.0e-3

# r_pad is slightly oversized so the apex sphere has some Hertz
# overlap room — but small enough that the contact doesn't saturate
# at any of the swept ``k_l``.  This produces a smoothly curved
# deformed apex, like an actual finger pad being pressed, instead of
# a perfectly flat plateau.
R_PAD_SPHERE = ARC_LENGTH_SPACING * 0.9      # 1.8 × half-spacing

# Anchor + moderate contact.  kc is chosen so the apex δ_n is
# comfortably below saturation (r_pad − h) at every ``k_l``.
KA = 50.0                                    # N/m
KC = 3.0e9                                   # N/m^(5/2)
EPS = 1.0e-9

# k_l values spanning soft → firm fingertip behaviour: at low ``k_l``
# the apex carries most of the deformation locally; at higher ``k_l``
# the lateral springs shuttle load to neighbouring spheres so the
# deformed patch broadens.
KL_VALUES = np.array([20.0, 100.0, 500.0, 2500.0])

# "Press depth": how far the ground plane is pushed UP into the
# finger from the rest "just-touching" position.  This is the
# rest-overlap (= raw_apex_rest) the apex sphere sees in the
# half-space form.  Picked so the apex δ_n is visible but doesn't
# saturate at any swept k_l.
PRESS_DEPTH_MM = 0.35
PRESS_DEPTH = PRESS_DEPTH_MM * 1.0e-3
TARGET_PITCH_FACTOR = 0.3


# ---------------------------------------------------------------------------
#  Build & solve
# ---------------------------------------------------------------------------


def flip_lattice_upside_down(lat: Lattice) -> Lattice:
    """Mirror the lattice through y = 0 so its apex points to −y.

    Reuses the existing :func:`make_arc` (which builds the arc with
    apex at ``+y``), then negates the y-component of every sphere
    position and outward normal.  ``ka``, ``kl``, and the edge
    topology are preserved.
    """
    p = lat.p.copy()
    n = lat.n.copy()
    p[:, 1] *= -1.0
    n[:, 1] *= -1.0
    return Lattice(p=p, n=n, edges=lat.edges, ka=lat.ka, kl=lat.kl)


def build_target():
    """Flat ground plane BELOW the inverted finger, face normal +y.

    With the finger flipped so its apex sits at y ≈ −R, the ground
    plane is placed at ``y = −R − r_pad + PRESS_DEPTH``.  At
    PRESS_DEPTH = 0 the apex sphere bottom just kisses the plane;
    with PRESS_DEPTH > 0 the plane is moved up into the apex, which
    drives the lattice deformation.
    """
    target_centre_y = -R_ARC - R_PAD_SPHERE + PRESS_DEPTH
    xy_extent = R_ARC * np.sin(np.pi * 0.45)
    return make_flat_face_target(
        centre=np.array([0.0, target_centre_y, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),       # +y, pointing up at the finger
        span_u=2.0 * xy_extent,
        span_v=2.0 * xy_extent,
        pitch=ARC_LENGTH_SPACING * TARGET_PITCH_FACTOR,
    )


def solve_for_kl(kl: float):
    lat_up = make_arc(
        N=N_SPHERES, R_pad=R_ARC,
        arc_length_spacing=ARC_LENGTH_SPACING,
        ka=KA, kl=float(kl),
    )
    lat = flip_lattice_upside_down(lat_up)
    target = build_target()
    deltas, info = solve_lattice_contact(
        lat, target, kc=KC, r_pad=R_PAD_SPHERE, eps=EPS,
        tol=1.0e-12, maxiter=8000,
    )
    return lat, deltas, info


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot(profiles, out_path: Path) -> None:
    """Overlay rest finger + deformed fingers + ground plane.

    The finger is inverted (apex at minimum y).  The ground plane is
    drawn at the rest-apex level so visually the apex touches it in
    the undeformed case; deformed apexes sit ABOVE the plane (pushed
    up into the finger body).

    Only spheres deformed more than 5 µm from rest are drawn for
    each ``k_l`` curve so each curve's extent honestly tracks how
    widely lateral coupling has spread the load.
    """
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 12

    # Rest apex y (most negative) — the ground plane sits exactly here
    # so the figure reads "apex touches ground at rest".
    x_rest, _, y_rest = profiles[0][1:4]
    rest_apex_y_mm = float(np.min(y_rest))
    ground_y_mm = rest_apex_y_mm

    kl_positive = [kl for kl, *_ in profiles if kl > 0]
    norm = colors.LogNorm(vmin=min(kl_positive), vmax=max(kl_positive))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # x-window: focus on the apex/contact region.  Pick the widest
    # contact extent across curves but cap it so we don't show the
    # full half-circle.
    overall_contact_x_max = 0.0
    for _, x_arr, y_def, _ in profiles:
        order = np.argsort(x_arr)
        x_sorted = x_arr[order]
        y_rest_sorted = np.interp(x_sorted, x_rest, y_rest)
        devs = np.abs(y_def[order] - y_rest_sorted)
        in_contact = devs > 5.0e-3
        if in_contact.any():
            overall_contact_x_max = max(
                overall_contact_x_max,
                float(np.max(np.abs(x_sorted[in_contact])))
            )
    plot_x_max = min(max(overall_contact_x_max + ARC_LENGTH_SPACING_MM * 1.2,
                         1.5),
                     2.8)

    # Rest finger as the grey reference, restricted to the apex
    # window so the off-screen rim doesn't leak in.
    in_window_rest = np.abs(x_rest) <= plot_x_max + 1.0e-9
    ax.plot(x_rest[in_window_rest], y_rest[in_window_rest],
            "--", color="0.45", lw=1.8, alpha=0.9,
            label="rest finger", zorder=1)
    ax.plot(x_rest[in_window_rest], y_rest[in_window_rest],
            "o", color="0.45", ms=3.5, alpha=0.6, zorder=1.5)

    # Deformed arcs.  For each k_l curve plot only spheres within the
    # apex window so high-k_l curves don't drag the rim into view.
    deformed_apex_y_max_mm = rest_apex_y_mm
    for kl, x_arr, y_def, _ in profiles:
        order = np.argsort(x_arr)
        x_sorted = x_arr[order]
        y_sorted = y_def[order]
        in_window = np.abs(x_sorted) <= plot_x_max + 1.0e-9
        if not in_window.any():
            continue
        deformed_apex_y_max_mm = max(
            deformed_apex_y_max_mm,
            float(np.max(y_sorted[in_window]))
        )
        if kl == 0.0:
            color = cmap(0.0)
            label = r"$k_\ell = 0$ N/m"
        else:
            color = cmap(norm(kl))
            label = rf"$k_\ell = {kl:g}$ N/m"
        ax.plot(x_sorted[in_window], y_sorted[in_window],
                "o-", color=color, lw=2.2, ms=6,
                label=label, zorder=3)

    # Ground plane at the rest apex level — finger touches it at rest.
    ax.plot([-plot_x_max, plot_x_max],
            [ground_y_mm, ground_y_mm],
            color="C3", lw=2.4, label="rigid ground", zorder=2)
    ax.fill_between([-plot_x_max, plot_x_max],
                    ground_y_mm - 0.5, ground_y_mm,
                    facecolor="0.92", edgecolor="0.6",
                    hatch="///", lw=0.0, alpha=0.6, zorder=0)

    ax.set_xlabel("x [mm]", fontsize=LABEL_SIZE)
    ax.set_ylabel("y [mm]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

    ax.set_xlim(-plot_x_max, plot_x_max)
    # Finger above the ground plane → y from just below the ground
    # up to slightly past the deformed apex.
    ax.set_ylim(ground_y_mm - 0.12,
                deformed_apex_y_max_mm + 0.18)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    print(f"Arc: N={N_SPHERES}, R={R_ARC_MM:.1f} mm, "
          f"spacing={ARC_LENGTH_SPACING_MM:.2f} mm "
          f"(r_pad per sphere={R_PAD_SPHERE*1e3:.3f} mm)")
    print(f"Fixed: ka={KA} N/m, kc={KC:.1e}, "
          f"press depth = {PRESS_DEPTH_MM:.2f} mm "
          f"(ground plane pushed up into the finger by this much)")
    print(f"Sweeping k_l over {list(KL_VALUES)} N/m")
    print("-" * 60)

    profiles = []
    for kl in KL_VALUES:
        lat, deltas, info = solve_for_kl(float(kl))
        # With the finger flipped, apex = sphere with the SMALLEST y.
        apex_idx = int(np.argmin(lat.p[:, 1]))

        # The whole arc is the cross-section; sort by x.
        p_rest_mm = lat.p * 1e3
        q_def_mm = (lat.p - deltas) * 1e3
        order = np.argsort(p_rest_mm[:, 0])
        x_arr = p_rest_mm[order, 0]
        y_def = q_def_mm[order, 1]
        y_rest = p_rest_mm[order, 1]
        profiles.append((float(kl), x_arr, y_def, y_rest))

        delta_n = np.einsum("nj,nj->n", deltas, lat.n)
        print(f"  k_l = {kl:8.1f} N/m  →  "
              f"apex δ_n = {delta_n[apex_idx]*1e3:+.4f} mm  "
              f"max δ_n = {np.max(delta_n)*1e3:.4f} mm  "
              f"(Jacobi iters = {info['jacobi_refine_iters']})")
    print("-" * 60)

    out_path = (Path(__file__).parent / "figures"
                / "arc_kl_sweep.png")
    plot(profiles, out_path)


if __name__ == "__main__":
    main()
