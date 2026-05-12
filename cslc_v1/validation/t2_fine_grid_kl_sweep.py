# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fix 1.2 + Fix 2 decisive experiment — does elastic-skin diffusion appear
when (a) the lattice is fine enough for many cells to engage inside the
indenter footprint and (b) the lateral coupling is strong enough that
ℓ_c ≳ 2·spacing in physical units?

Finding H concluded "lateral coupling does nothing" comparing
(ka=15000, kl=500) vs (ka=1000, kl=1000) at spacing=5 mm.  Two
confounds in that experiment:

1. At spacing=5 mm with R=10 mm indenter and δ=4 mm, only ~5 cells
   engage geometrically — too few for any pressure-profile shape
   difference to manifest (Winkler vs Hertz within 5 samples is
   indistinguishable).
2. The lateral correlation length in lattice-spacing units (Finding A)
   was 0.18 for regime A and 1.0 for regime B — neither is in the
   ≳2-spacing regime where elastic-skin diffusion would actually
   redistribute load across multiple cells.

This script controls for both: spacing=1 mm so ~125 cells engage at
δ=4 mm, and kl_kernel = 4·ka (ℓ_c = 2 spacings = 2 mm).  If the
pressure-profile shape STILL doesn't change between weak and strong
regimes, the elastic-skin framing is genuinely dead at the paper's
parameter scope.  If it does change, Fix 1.1 (resolution-independent
kl_physical scaling) is the lasting fix.

Run with:
    uv run python -m cslc_v1.validation.t2_fine_grid_kl_sweep

Outputs profile figures into ``cslc_v1/validation/figures/`` and prints
the r10/a_indent metric per regime for the headline comparison.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Reuse the existing Tier 2 infrastructure verbatim — different scenes
# are the only knob we vary.
from cslc_v1.validation.t2_indenter import (
    T2Scene,
    pressure_profile_comparison,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_THIS_DIR, "figures")
os.makedirs(_OUT_DIR, exist_ok=True)


def _scene(spacing_m: float, ka: float, kl_kernel: float, pad_half_m: float) -> T2Scene:
    """Build a fine-grid T2Scene at the supplied (ka, kl_kernel).

    ``kl_kernel`` is the value the kernel sees (legacy convention,
    N/m).  Per Fix 1.1 in cslc_data.py, the physically meaningful
    quantity is ``kl_physical = kl_kernel · spacing²`` [N·m], which is
    invariant to refinement.
    """
    return T2Scene(
        cslc_spacing=spacing_m,
        cslc_ka=ka,
        cslc_kl=kl_kernel,
        pad_half=pad_half_m,
    )


def main() -> None:
    # ── Experimental design ────────────────────────────────────────────
    # Spacing 1 mm gives ~10000 spheres on a 50 mm pad, ~125 cells
    # engaged at δ = 4 mm (Hertzian a = √(R·δ) = 6.32 mm).  Pad half
    # 25 mm keeps the lattice well outside the patch but caps memory:
    # A_inv at n=2601 is ~27 MB, well within the RTX 3070's headroom.
    spacing = 1.0e-3
    pad_half = 0.025
    R_indenter = 0.010
    delta_mm = 4.0

    ka = 25000.0   # current production default
    # Weak regime: ℓ_c_physical = √(kl/ka) · spacing = √(500/25000) · 1mm = 0.14 mm
    #              = 0.14 spacings → sub-grid (matches Finding A regime A)
    kl_weak = 500.0
    # Strong regime: ℓ_c_physical = 2 mm = 2 spacings
    #                kl_kernel = ka · (ℓ_c_phys / spacing)² = 25000·4 = 100000
    kl_strong = 4.0 * ka
    # Hydroelastic kh — fair calibration is irrelevant here (we compare
    # SHAPES not absolute force), but keep it fixed across regimes so
    # hydro contributes the same baseline.
    kh = 5.3e8

    # Headline derived quantities for the log.
    l_c_weak_phys = float(np.sqrt(kl_weak / ka)) * spacing
    l_c_strong_phys = float(np.sqrt(kl_strong / ka)) * spacing
    a_indent = float(np.sqrt(R_indenter * delta_mm * 1e-3))

    print("=" * 72)
    print("  Fine-grid kl sweep — does elastic-skin diffusion appear at")
    print("  spacing = 1 mm with kl strong enough for ℓ_c ≳ 2·spacing?")
    print("=" * 72)
    print(f"  Spacing       : {spacing * 1e3:.2f} mm")
    print(f"  Pad half      : {pad_half * 1e3:.1f} mm")
    print(f"  Indenter R    : {R_indenter * 1e3:.1f} mm  (a_indent at δ=4mm = {a_indent * 1e3:.2f} mm)")
    print(f"  ka            : {ka:.0f} N/m  (fixed across regimes)")
    print(f"  kl_weak       : {kl_weak:.0f} N/m  → ℓ_c = {l_c_weak_phys * 1e3:.3f} mm "
          f"({l_c_weak_phys / spacing:.3f} spacings)")
    print(f"  kl_strong     : {kl_strong:.0f} N/m  → ℓ_c = {l_c_strong_phys * 1e3:.3f} mm "
          f"({l_c_strong_phys / spacing:.3f} spacings)")
    print(f"  kh (hydro)    : {kh:.2e} Pa  (held constant; shape comparison only)")
    print(f"  δ_indenter    : {delta_mm:.2f} mm")
    print("=" * 72)

    # ── Regime A: weak (sub-grid) kl ───────────────────────────────────
    print("\n--- Regime A_fine: weak kl (sub-grid ℓ_c) -----------------------")
    scene_weak = _scene(spacing, ka, kl_weak, pad_half)
    res_weak = pressure_profile_comparison(
        scene_weak, delta_mm=delta_mm, kh_hydro=kh,
        out_dir=_OUT_DIR, regime_label="finegrid_A_weak",
    )

    # ── Regime B: strong (ℓ_c = 2 spacings) kl ─────────────────────────
    print("\n--- Regime B_fine: strong kl (ℓ_c = 2 spacings) -----------------")
    scene_strong = _scene(spacing, ka, kl_strong, pad_half)
    res_strong = pressure_profile_comparison(
        scene_strong, delta_mm=delta_mm, kh_hydro=kh,
        out_dir=_OUT_DIR, regime_label="finegrid_B_strong",
    )

    # ── Headline comparison ────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  HEADLINE — does strong kl change CSLC pressure-profile shape?")
    print("=" * 72)
    print(f"  Metric: r10 / a_indent  (radial distance at which p drops to 10% of peak,")
    print(f"          normalised by Hertzian footprint a = √(R·δ) = {a_indent * 1e3:.3f} mm)")
    print()
    print(f"  CSLC weak    r10/a = {res_weak['r10_c'] / res_weak['a_indent']:.3f}  "
          f"FWHM/a = {res_weak['fwhm_c'] / res_weak['a_indent']:.3f}  "
          f"F = {res_weak['F_c']:.3f} N  n_active = (see scene log above)")
    print(f"  CSLC strong  r10/a = {res_strong['r10_c'] / res_strong['a_indent']:.3f}  "
          f"FWHM/a = {res_strong['fwhm_c'] / res_strong['a_indent']:.3f}  "
          f"F = {res_strong['F_c']:.3f} N")

    # Profile SHAPE difference: compare normalised profiles point-wise.
    # If lateral coupling truly does nothing, the two normalised CSLC
    # profiles are pointwise identical.  Any meaningful difference
    # validates the elastic-skin diffusion claim.
    p_weak = res_weak["p_c"] / max(res_weak["p_c"].max(), 1e-30)
    p_strong = res_strong["p_c"] / max(res_strong["p_c"].max(), 1e-30)
    n = min(len(p_weak), len(p_strong))
    if n > 0:
        rms_diff = float(np.sqrt(np.mean((p_weak[:n] - p_strong[:n]) ** 2)))
        max_diff = float(np.max(np.abs(p_weak[:n] - p_strong[:n])))
        print()
        print(f"  Normalised-profile difference (CSLC weak vs CSLC strong):")
        print(f"    RMS  = {rms_diff:.4f}   (0 = identical, 1 = unrelated)")
        print(f"    max  = {max_diff:.4f}")
        if rms_diff < 5e-3:
            print("  ⇒ Shapes essentially IDENTICAL.  Elastic-skin diffusion DOES NOT")
            print("    appear even at fine spacing + strong coupling.  Finding H stands;")
            print("    the elastic-skin framing is dead for this paper.")
        elif rms_diff < 5e-2:
            print("  ⇒ Shapes DIFFER MILDLY.  Lateral coupling has a visible but small")
            print("    effect — defensible 'distributed-pressure' narrative; not a")
            print("    headline elastic-skin diffusion result.")
        else:
            print("  ⇒ Shapes DIFFER SIGNIFICANTLY.  Elastic-skin diffusion appears at")
            print("    fine spacing + strong coupling.  Fix 1.1 (resolution-independent")
            print("    kl_physical) is the lasting fix; cf the kl_physical hook in")
            print("    cslc_data.py:CSLCData.from_pads.")

    # Side-output: the active-cell ratio between regimes is also
    # diagnostic — strong kl should engage roughly the same set (gate
    # is geometry-bound, see prior session's lateral-coupling note) but
    # with redistributed forces.
    print()
    print("  Hydro / point reference (unchanged across regimes, by design):")
    print(f"    Hydro r10/a (weak run)   = {res_weak['r10_h'] / res_weak['a_indent']:.3f}")
    print(f"    Hydro r10/a (strong run) = {res_strong['r10_h'] / res_strong['a_indent']:.3f}")
    print(f"    Point r10/a              = {res_weak['r10_p'] / res_weak['a_indent']:.3f}  "
          "(single contact, by construction)")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
