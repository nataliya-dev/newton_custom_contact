# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic — does forcing the iterative jacobi_step path (rather than
the closed-form `lattice_solve_equilibrium` matvec) reveal kl-dependent
profile-shape differences that the closed-form washes out?

Finding D flagged that the closed-form
``δ = kc · A_inv · φ`` (with ``A_inv = (K + kc·I)^-1``) assumes saturated
active contact everywhere — the per-sphere gate ``Σ_ε(φ_rest − δ)`` is
absent from the formulation.  At strong lateral coupling, the
closed-form can therefore homogenise δ over the entire pad (active and
inactive spheres mixed), masking the within-patch redistribution that
real elastic-skin physics would produce.

This script repeats the fine-grid kl sweep from `t2_fine_grid_kl_sweep.py`
but disables ``A_inv`` post-construction (``handler.cslc_data.A_inv =
None``), which routes the kernel through the gated iterative
``jacobi_step`` path.  We crank ``n_iter`` because at strong kl the
Jacobi spectral radius approaches 1 and convergence slows.

Run with:
    uv run python -m cslc_v1.validation.t2_fine_grid_jacobi

If the iterative path produces meaningful weak-vs-strong kl differences
in the CSLC pressure profile shape, we have a smoking gun:
- The lattice physics IS doing elastic-skin redistribution
- The closed-form A_inv path was structurally hiding it
- Fix: route through iterative jacobi for contact onset (consistent with
  Finding D's recommendation in FINDINGS.md)
"""

from __future__ import annotations

import os
import sys

import numpy as np

from cslc_v1.validation.t2_indenter import (
    T2Scene,
    build_t2_model_cslc,
    measure_one,
    radial_profile,
    fwhm_from_profile,
    r_at_fraction,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_THIS_DIR, "figures")
os.makedirs(_OUT_DIR, exist_ok=True)


def _disable_a_inv(model, n_iter_override: int) -> None:
    """Force the handler onto the iterative jacobi_step path.

    Mutates the handler's cslc_data so the next launch() falls through
    the ``if data.A_inv is not None`` branch in
    `_launch_vs_sphere`.  Also bumps n_iter — at strong kl the
    Jacobi spectral radius approaches 1 and the 40-iteration default
    under-converges by ~50%.
    """
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        raise RuntimeError("No CSLC handler attached to model; check pad config.")
    handler.cslc_data.A_inv = None
    handler.n_iter = int(n_iter_override)


def _run_cslc_only(
    scene: T2Scene, delta_indenter_mm: float, n_iter_override: int
) -> dict:
    """Build model, force iterative path, measure per-sphere CSLC profile."""
    model = build_t2_model_cslc(scene)
    # contacts() triggers lazy collision-pipeline construction, which is
    # where the CSLC handler is built.  After this call,
    # model._collision_pipeline.cslc_handler exists and we can disable
    # A_inv to force the iterative jacobi path.
    contacts = model.contacts()
    _disable_a_inv(model, n_iter_override)
    res = measure_one(
        model, contacts, scene,
        delta_indenter=delta_indenter_mm * 1e-3, indenter_body_idx=1,
    )
    pos = res["pos_world"]
    f = res["f_n_per_sphere"]
    active = f > 1e-4
    pos = pos[active]
    f = f[active]
    R = scene.indenter_R
    a_indent = float(np.sqrt(R * delta_indenter_mm * 1e-3))
    r, p, cnt = radial_profile(
        pos, f, r_max=4 * a_indent + scene.cslc_spacing,
        n_bins=20, sphere_area=scene.cslc_spacing ** 2,
    )
    return {
        "n_active": int(active.sum()),
        "F": float(f.sum()),
        "r": r,
        "p": p,
        "fwhm": fwhm_from_profile(r, p),
        "r10": r_at_fraction(r, p, 0.1),
        "a_indent": a_indent,
    }


def main() -> None:
    spacing = 1.0e-3
    pad_half = 0.025
    R_indenter = 0.010
    delta_mm = 4.0
    ka = 25000.0
    kl_weak = 500.0
    kl_strong = 4.0 * ka      # 100 000  → ℓ_c = 2·spacing
    # Jacobi spectral radius at strong kl is ~0.94; raise n_iter so
    # under-convergence isn't a confound on top of the path comparison.
    n_iter_jacobi = 500

    a_indent = float(np.sqrt(R_indenter * delta_mm * 1e-3))
    print("=" * 72)
    print("  Iterative-jacobi vs closed-form: does the gated path reveal")
    print("  within-patch kl-dependence that A_inv hides?")
    print("=" * 72)
    print(f"  spacing = {spacing*1e3:.2f} mm   pad_half = {pad_half*1e3:.1f} mm")
    print(f"  R_indenter = {R_indenter*1e3:.1f} mm   δ = {delta_mm:.2f} mm")
    print(f"  ka = {ka:.0f}   kl_weak = {kl_weak:.0f}   kl_strong = {kl_strong:.0f}")
    print(f"  n_iter (forced jacobi) = {n_iter_jacobi}")
    print(f"  Hertzian a_indent = {a_indent*1e3:.3f} mm")
    print("=" * 72)

    print("\n--- weak kl, ITERATIVE path ----------------------------------")
    res_weak = _run_cslc_only(
        T2Scene(cslc_spacing=spacing, cslc_ka=ka, cslc_kl=kl_weak,
                pad_half=pad_half, cslc_n_iter=n_iter_jacobi),
        delta_indenter_mm=delta_mm, n_iter_override=n_iter_jacobi,
    )
    print(f"  F = {res_weak['F']:.3f} N    n_active = {res_weak['n_active']}")
    print(f"  r10/a = {res_weak['r10']/res_weak['a_indent']:.4f}    "
          f"FWHM/a = {res_weak['fwhm']/res_weak['a_indent']:.4f}")

    print("\n--- strong kl, ITERATIVE path --------------------------------")
    res_strong = _run_cslc_only(
        T2Scene(cslc_spacing=spacing, cslc_ka=ka, cslc_kl=kl_strong,
                pad_half=pad_half, cslc_n_iter=n_iter_jacobi),
        delta_indenter_mm=delta_mm, n_iter_override=n_iter_jacobi,
    )
    print(f"  F = {res_strong['F']:.3f} N    n_active = {res_strong['n_active']}")
    print(f"  r10/a = {res_strong['r10']/res_strong['a_indent']:.4f}    "
          f"FWHM/a = {res_strong['fwhm']/res_strong['a_indent']:.4f}")

    # Profile-shape comparison
    p_weak = res_weak["p"] / max(res_weak["p"].max(), 1e-30)
    p_strong = res_strong["p"] / max(res_strong["p"].max(), 1e-30)
    n = min(len(p_weak), len(p_strong))
    rms_diff = float(np.sqrt(np.mean((p_weak[:n] - p_strong[:n]) ** 2))) if n > 0 else 0.0
    max_diff = float(np.max(np.abs(p_weak[:n] - p_strong[:n]))) if n > 0 else 0.0

    print()
    print("=" * 72)
    print("  HEADLINE — iterative-path profile-shape difference (weak vs strong kl)")
    print("=" * 72)
    print(f"  ΔF (N)                        = {res_strong['F'] - res_weak['F']:+.4f}  "
          f"({100.0*(res_strong['F']-res_weak['F'])/max(res_weak['F'],1e-9):+.2f}%)")
    print(f"  Δn_active                     = {res_strong['n_active'] - res_weak['n_active']:+d}")
    print(f"  Δr10/a                        = {res_strong['r10']/res_strong['a_indent'] - res_weak['r10']/res_weak['a_indent']:+.4f}")
    print(f"  Δ(FWHM/a)                     = {res_strong['fwhm']/res_strong['a_indent'] - res_weak['fwhm']/res_weak['a_indent']:+.4f}")
    print(f"  RMS(normalised profile diff)  = {rms_diff:.4f}")
    print(f"  max(normalised profile diff)  = {max_diff:.4f}")
    print()
    if rms_diff < 5e-3:
        print("  ⇒ Even the iterative path shows IDENTICAL profiles.  Finding D's")
        print("    closed-form-degeneracy hypothesis is REFUTED for this scene.")
        print("    The kl-invariance is structural; routing through jacobi doesn't help.")
    elif rms_diff < 5e-2:
        print("  ⇒ Iterative path shows MILD profile-shape kl-dependence.  The")
        print("    closed-form was washing out a small but real elastic-skin signal.")
        print("    Worth defending in the paper if the Hertzian-vs-Winkler narrative")
        print("    survives the magnitude of the effect.")
    else:
        print("  ⇒ Iterative path shows SIGNIFICANT profile-shape kl-dependence!")
        print("    The closed-form A_inv solve was structurally hiding the elastic-")
        print("    skin signal.  Fix: route through iterative jacobi for contact-onset")
        print("    scenarios; reserve A_inv for saturated steady-state only.")
        print("    This recovers a defensible 'elastic-skin profile shape' claim.")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
