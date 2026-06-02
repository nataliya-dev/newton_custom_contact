# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Aggregate contact-force scaling: curved pad is one power stiffer.

The per-pair contact law is Hertz-like, ``|f| ∝ φ^{3/2}`` (see
:mod:`cslc_main.theory.plot_contact_hertz`).  Summed over the engaged
contact patch the aggregate body force depends on whether the patch
*grows* with indentation (theory.md §1):

* **Flat pad on flat target** — every engaged sphere sees the same
  overlap ``φ = δ`` and the patch is fixed, so the body force is a
  constant number of spheres times ``δ^{3/2}``:  ``F ∝ δ^{3/2}``
  (pure Hertz).
* **Dome pad on flat target** — as ``δ`` grows the gap closes over a
  paraboloid, more rings engage, and the patch radius grows as
  ``√δ`` (area ∝ δ).  Integrating ``φ^{3/2}`` over the growing patch
  picks up one extra power:  ``F ∝ δ^{5/2}`` — one power stiffer than
  Hertz, the contact analogue of the hydroelastic patch integral.

We hold the lattice rigid (``δ_lattice = 0``) so the figure isolates
the contact force law's *patch-area scaling* (eq:contact summed over
the patch) rather than the lattice relaxation.  Net body force is
read off via :func:`cslc_lattice.lattice_contact_normal_forces`, the
same kernel-form sum the solver uses, and the log-log slope is fit in
the power-law region and compared against the reference exponents
3/2 and 5/2.

Run::

    uv run python -m cslc_main.theory.plot_contact_patch_scaling
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    lattice_contact_normal_forces,
    make_dome,
    make_flat_grid,
)
from cslc_main.theory.cslc_targets import make_flat_face_target


# ---------------------------------------------------------------------------
#  Scene parameters
# ---------------------------------------------------------------------------

KC = 1.0e10                                  # N·m^(−7/2)  (per-volume)
EPS = 1.0e-9                                 # tight smoothing → clean power law

# Dome (curved pad).  R and half-angle chosen so the small-angle
# paraboloid gap holds across the swept indentation range.
DOME_N = 600
DOME_R_MM = 12.0
DOME_HALF_ANGLE = np.deg2rad(55.0)
DOME_KA = 35_000.0
DOME_KL = 1_000.0

# Flat pad.  Wide enough that the target plane spans the whole pad, so
# every engaged sphere sees the same overlap (fixed patch).
FLAT_NU = 31
FLAT_NV = 31

# Indentation sweep (apex / surface overlap).  Log-spaced; kept below
# ~0.15·R on the dome so the paraboloid approximation holds.
DELTA_MIN_MM = 0.10
DELTA_MAX_MM = 1.80
N_DELTA = 24

# Fraction of the (log) sweep used to fit the asymptotic slope — drop
# the small-δ end where only the apex sphere engages (single-sphere
# δ^{3/2} before the patch has grown).
FIT_LO_FRAC = 0.45


# ---------------------------------------------------------------------------
#  Net body force at rigid indentation δ
# ---------------------------------------------------------------------------


def net_force_vs_delta(lat, z_apex: float, r_pad: float,
                       span: float, deltas_mm: np.ndarray) -> np.ndarray:
    """Net contact-force magnitude [N] sweeping rigid indentation δ.

    The flat target plane (outward normal ``-ẑ``) is placed so the apex
    sphere's half-space overlap equals ``δ``: ``raw_apex = r_pad −
    (t_z − z_apex) = δ`` ⇒ ``t_z = z_apex + r_pad − δ``.  The lattice is
    rigid (``δ_lattice = 0``); we sum the per-sphere contact force.
    """
    N = lat.N
    zeros = np.zeros((N, 3))
    # Target pitch fine relative to r_pad so the locality kernel sees
    # several samples per pad sphere (clean Riemann sum of the patch).
    pitch = 0.4 * r_pad
    forces = np.zeros_like(deltas_mm)
    for k, d_mm in enumerate(deltas_mm):
        t_z = z_apex + r_pad - d_mm * 1.0e-3
        target = make_flat_face_target(
            centre=np.array([0.0, 0.0, t_z]),
            normal=np.array([0.0, 0.0, -1.0]),
            span_u=span, span_v=span, pitch=pitch,
        )
        F_contact, _ = lattice_contact_normal_forces(
            lat, target, zeros, kc=KC, r_pad=r_pad, eps=EPS,
        )
        forces[k] = float(np.linalg.norm(F_contact.sum(axis=0)))
    return forces


def fit_slope(deltas_mm: np.ndarray, forces: np.ndarray) -> float:
    """Least-squares log-log slope over the asymptotic (upper) region."""
    mask = (forces > 0.0)
    lo = int(FIT_LO_FRAC * mask.sum())
    x = np.log(deltas_mm[mask][lo:])
    y = np.log(forces[mask][lo:])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot(deltas_mm: np.ndarray,
         f_flat: np.ndarray, f_dome: np.ndarray,
         s_flat: float, s_dome: float, out_path: Path) -> None:
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 12

    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    ax.loglog(deltas_mm, f_flat, "o-", color="C0", ms=5, lw=1.8,
              zorder=3,
              label=fr"flat pad  (fit slope {s_flat:.2f})")
    ax.loglog(deltas_mm, f_dome, "s-", color="C3", ms=5, lw=1.8,
              zorder=3,
              label=fr"dome pad  (fit slope {s_dome:.2f})")

    # Reference power laws δ^{3/2} and δ^{5/2}, anchored to each curve's
    # mid-point so they sit alongside the data as guides to the eye.
    mid = len(deltas_mm) // 2
    d = deltas_mm
    ref_flat = f_flat[mid] * (d / d[mid]) ** 1.5
    ref_dome = f_dome[mid] * (d / d[mid]) ** 2.5
    ax.loglog(d, ref_flat, "--", color="0.45", lw=1.3, zorder=1,
              label=r"$\delta^{3/2}$ (Hertz)")
    ax.loglog(d, ref_dome, ":", color="0.25", lw=1.6, zorder=1,
              label=r"$\delta^{5/2}$ (patch grows)")

    ax.set_xlabel(r"indentation $\delta$  [mm]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"net contact force $F^{\mathrm{body}}$  [N]",
                  fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_SIZE - 2)
    ax.grid(alpha=0.3, which="both")
    # Lower-right corner is the empty quadrant (both curves rise to the
    # upper right); placing the legend there avoids occluding the flat
    # pad line, which passes through the upper-left.
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    deltas_mm = np.geomspace(DELTA_MIN_MM, DELTA_MAX_MM, N_DELTA)

    # --- Dome pad ---------------------------------------------------------
    dome, dome_spacing, _ = make_dome(
        N=DOME_N, R_pad=DOME_R_MM * 1e-3, half_angle=DOME_HALF_ANGLE,
        ka=DOME_KA, kl=DOME_KL, k_neighbors=6,
    )
    r_pad_dome = dome_spacing / 2.0
    z_apex_dome = float(np.max(dome.p[:, 2]))
    span_dome = 2.0 * DOME_R_MM * 1e-3 * np.sin(DOME_HALF_ANGLE) * 1.1
    print(f"Dome: N={DOME_N}, R={DOME_R_MM} mm, "
          f"half-angle={np.degrees(DOME_HALF_ANGLE):.0f}°, "
          f"spacing={dome_spacing*1e3:.3f} mm, r_pad={r_pad_dome*1e3:.3f} mm")
    f_dome = net_force_vs_delta(dome, z_apex_dome, r_pad_dome,
                                span_dome, deltas_mm)

    # --- Flat pad ---------------------------------------------------------
    # Match the flat lattice spacing to the dome's so r_pad (hence the
    # per-sphere area weight) is comparable; only the exponent matters.
    flat = make_flat_grid(
        n_u=FLAT_NU, n_v=FLAT_NV, spacing=dome_spacing,
        ka=DOME_KA, kl=DOME_KL, normal=np.array([0.0, 0.0, 1.0]),
    )
    r_pad_flat = dome_spacing / 2.0
    span_flat = (FLAT_NU - 1) * dome_spacing * 1.05
    print(f"Flat: {FLAT_NU}x{FLAT_NV}, spacing={dome_spacing*1e3:.3f} mm")
    f_flat = net_force_vs_delta(flat, 0.0, r_pad_flat,
                                span_flat, deltas_mm)

    s_flat = fit_slope(deltas_mm, f_flat)
    s_dome = fit_slope(deltas_mm, f_dome)
    print("-" * 60)
    print(f"Fitted log-log slopes  (upper {1 - FIT_LO_FRAC:.0%} of sweep):")
    print(f"  flat pad : {s_flat:.3f}   (expected 1.50, pure Hertz)")
    print(f"  dome pad : {s_dome:.3f}   (expected 2.50, patch grows)")
    print("-" * 60)

    out_path = (Path(__file__).parent / "figures"
                / "contact_patch_scaling.png")
    plot(deltas_mm, f_flat, f_dome, s_flat, s_dome, out_path)


if __name__ == "__main__":
    main()
