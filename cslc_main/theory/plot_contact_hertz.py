# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-pair Hertz-like contact force law of one CSLC lattice sphere.

Characterises the third per-sphere force of the paper's "Forces on
Each Sphere" section: the Hertz-like contact force (overleaf
eq:contact + eq:phi_eff)::

    |f^contact|(φ) = k_c · A · φ_eff,    φ_eff = max(φ, 0)^{3/2}

where ``φ`` is the signed face-penetration overlap of one (lattice
sphere, target element) pair and ``A`` the target Voronoi area element.
The figure shows two properties the paper emphasises:

* **Soft, C¹ onset.**  The local stiffness ``d|f|/dφ = 3/2·k_c·A·√φ``
  vanishes at first contact (``φ → 0⁺``), so the force grows smoothly
  out of zero with no jump in stiffness.
* **Contrast with rigid point contact.**  A linear spring
  ``|f| = k_pt·φ`` (the point-contact idealisation) is matched to the
  same force at the deepest penetration, isolating the difference as
  the *onset shape*: the linear law has a finite stiffness at φ = 0,
  the Hertz law starts infinitely compliant.

Run::

    uv run python -m cslc_main.theory.plot_contact_hertz
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
#  Force-law parameters (theory.md §9 sample numbers + production kc)
# ---------------------------------------------------------------------------

# Per-volume contact stiffness (production grasp value) and a single
# target Voronoi area element at the sample-default sphere sampling
# (r_pad = 1.5 mm, n_samples = 1500 on a 33.5 mm tennis ball):
#     A_j = 4π R² / n_samples ≈ 9.4e-6 m².
KC = 1.0e10                                  # N·m^(−7/2)  (per-volume)
A_J = 9.4e-6                                 # m²  target area element

# Face-penetration sweep.  Operating depths in the grasp experiments sit
# below ~1 mm, so sweep φ ∈ [0, 1 mm].
PHI_MAX_MM = 1.0
N_SAMPLES = 600


# ---------------------------------------------------------------------------
#  Evaluate
# ---------------------------------------------------------------------------


def hertz_force(phi_m: np.ndarray) -> np.ndarray:
    """Per-pair Hertz contact force [N]: k_c · A · max(φ, 0)^{3/2}."""
    phi_eff = np.power(np.clip(phi_m, 0.0, None), 1.5)
    return KC * A_J * phi_eff


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot(out_path: Path) -> None:
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 12

    phi_mm = np.linspace(0.0, PHI_MAX_MM, N_SAMPLES)
    phi_m = phi_mm * 1.0e-3
    f_hertz = hertz_force(phi_m)

    # Linear point-contact reference, matched to the Hertz force at the
    # deepest penetration so the two laws end at the same point and the
    # only visible difference is the onset shape.
    phi_max_m = PHI_MAX_MM * 1.0e-3
    k_pt = hertz_force(np.array([phi_max_m]))[0] / phi_max_m   # N/m
    f_linear = k_pt * phi_m

    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    # Rigid point-contact spring — dashed grey reference, drawn under.
    ax.plot(phi_mm, f_linear, "--", color="0.45", lw=1.8, zorder=1,
            label="rigid point contact  " r"($\propto\,\phi$)")

    # Hertz-like CSLC contact force — solid blue.
    ax.plot(phi_mm, f_hertz, "-", color="C0", lw=2.4, zorder=3,
            label=r"CSLC contact  ($\propto\,\phi^{3/2}$)")

    # Mark the shared endpoint where the two laws coincide.
    ax.plot([PHI_MAX_MM], [f_hertz[-1]], "o", color="0.3", ms=6,
            mec="white", mew=0.8, zorder=4)

    # Annotate the soft onset: stiffness vanishes at first touch.
    ax.annotate(r"stiffness $\to 0$ at first touch"
                "\n" r"($d|f|/d\phi \propto \sqrt{\phi}$)",
                xy=(0.04 * PHI_MAX_MM, hertz_force(
                    np.array([0.04 * PHI_MAX_MM * 1e-3]))[0]),
                xytext=(0.30 * PHI_MAX_MM, 0.10 * f_hertz[-1]),
                fontsize=LEGEND_SIZE, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))

    ax.set_xlabel(r"face penetration $\phi$  [mm]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"contact force $|f^{\mathrm{contact}}|$  [N]",
                  fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.set_xlim(0.0, PHI_MAX_MM)
    ax.set_ylim(0.0, f_hertz[-1] * 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    print(f"Per-pair Hertz contact law: kc={KC:.1e}, A_j={A_J:.2e} m²")
    phi_max_m = PHI_MAX_MM * 1.0e-3
    print(f"  force at φ = {PHI_MAX_MM:.2f} mm:  "
          f"{hertz_force(np.array([phi_max_m]))[0]:.3f} N")
    out_path = (Path(__file__).parent / "figures"
                / "contact_hertz.png")
    plot(out_path)


if __name__ == "__main__":
    main()
