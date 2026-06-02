# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pre-sliding (elastoplastic) friction law of one CSLC lattice sphere.

Characterises the fourth per-sphere force of the paper's "Forces on
Each Sphere" section: the smooth stick-slip friction (overleaf
eq:friction, theory.md §6.4)::

    |f^friction|(s) = k_stick · μ · f_n · s / (k_stick · s + μ · f_n)

with ``s = ||δ_t||`` the tangential shear of the compliant skin.  The
curve interpolates smoothly between

    s → 0   (stick):  |f| ≈ k_stick · s     Hookean spring (slope k_stick)
    s → ∞   (slip):   |f| → μ · f_n         Coulomb plateau

We sweep ``s`` and plot the curve for three normal loads ``f_n`` so the
reader sees the Coulomb plateau ``μ·f_n`` scale with the normal force,
while the small-shear slope ``k_stick`` is shared by all three.  This
is the figure for the pre-sliding/LuGre-style friction the paper cites
(de Wit 1995, Cas23): static friction develops *before* macroscopic
slip, tied to a displacement rather than a sliding velocity.

Run::

    uv run python -m cslc_main.theory.plot_friction_presliding
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import friction_force_smooth


# ---------------------------------------------------------------------------
#  Force-law parameters (production grasp values, cslc_main.grasp.params)
# ---------------------------------------------------------------------------

K_STICK = 25_000.0                           # N/m  — tangential stick stiffness
MU = 0.5                                      # Coulomb coefficient

# Three normal loads spanning the paper's grasp-force regime (the lift
# experiment settles between ~0.57 N and ~2.9 N per pad).  The Coulomb
# plateau μ·f_n and the stick→slip transition s* = μ·f_n/k_stick both
# scale with f_n; the small-shear slope k_stick does not.
F_N_VALUES = np.array([0.5, 1.0, 2.0])       # N

# Sweep |δ_t| out to several transition widths so every plateau is
# clearly reached.  s* for the largest load is μ·f_n/k_stick = 40 µm,
# so 200 µm shows the full stick→slip crossover.
S_MAX_UM = 200.0                             # µm
N_SAMPLES = 600


# ---------------------------------------------------------------------------
#  Evaluate
# ---------------------------------------------------------------------------


def friction_curve(s_m: np.ndarray, f_n: float) -> np.ndarray:
    """Friction magnitude [N] over the |δ_t| grid for one normal load."""
    return np.array([friction_force_smooth(float(s), f_n, K_STICK, MU)
                     for s in s_m])


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot(out_path: Path) -> None:
    LABEL_SIZE = 16
    TICK_SIZE = 13
    LEGEND_SIZE = 12

    s_um = np.linspace(0.0, S_MAX_UM, N_SAMPLES)
    s_m = s_um * 1.0e-6

    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    colors = ["C0", "C1", "C2"]
    for f_n, color in zip(F_N_VALUES, colors):
        f = friction_curve(s_m, f_n)
        plateau = MU * f_n                        # Coulomb limit μ·f_n
        s_star_um = (MU * f_n / K_STICK) * 1.0e6  # transition |δ_t| [µm]

        # Smooth stick→slip curve.
        ax.plot(s_um, f, "-", color=color, lw=2.2, zorder=3,
                label=fr"$f_n = {f_n:g}$ N")
        # Coulomb plateau (dashed horizontal).
        ax.axhline(plateau, color=color, ls="--", lw=1.2, alpha=0.7,
                   zorder=2)
        # Transition point s* where |f| = μ·f_n / 2 (smooth crossover).
        ax.plot([s_star_um], [0.5 * plateau], "o", color=color,
                ms=6, mec="white", mew=0.8, zorder=4)

    # Shared elastic-stick tangent |f| = k_stick · s (slope at s = 0),
    # drawn only over the small-shear region where it is the envelope.
    s_tan_um = np.linspace(0.0, 0.45 * S_MAX_UM, 50)
    ax.plot(s_tan_um, K_STICK * (s_tan_um * 1.0e-6), ":", color="0.35",
            lw=1.6, zorder=1,
            label=fr"stick tangent $k_{{\mathrm{{stick}}}}\,\|\delta_t\|$")

    # Annotate one Coulomb plateau so the dashed lines read as μ·f_n.
    ax.annotate(r"Coulomb limit $\mu f_n$",
                xy=(S_MAX_UM, MU * F_N_VALUES[-1]),
                xytext=(0.42 * S_MAX_UM, MU * F_N_VALUES[-1] + 0.06),
                fontsize=LEGEND_SIZE, color="0.25")

    ax.set_xlabel(r"tangential shear $\|\delta_t\|$  [$\mu$m]",
                  fontsize=LABEL_SIZE)
    ax.set_ylabel(r"friction $|f^{\mathrm{friction}}|$  [N]",
                  fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.set_xlim(0.0, S_MAX_UM)
    ax.set_ylim(0.0, MU * F_N_VALUES[-1] * 1.18)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved figure: {out_path}")


def main() -> None:
    print(f"Friction pre-sliding law: k_stick={K_STICK:g} N/m, mu={MU:g}")
    for f_n in F_N_VALUES:
        s_star_um = (MU * f_n / K_STICK) * 1.0e6
        print(f"  f_n = {f_n:4.2f} N  →  Coulomb plateau μ·f_n = "
              f"{MU * f_n:.3f} N,  transition s* = {s_star_um:.1f} µm")
    out_path = (Path(__file__).parent / "figures"
                / "friction_presliding.png")
    plot(out_path)


if __name__ == "__main__":
    main()
