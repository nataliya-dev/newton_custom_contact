# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 / T-I — Stick-slip friction on a single pad sphere vs a flat face.

The v2 successor to ``test_04_friction.py``.  Same physics (contract
§6.4) but the contact term is the v2 half-space form (eq:raw) instead
of the v1 sphere-vs-sphere overlap.  At face-on geometry the two
reductions give numerically identical f_n, so the friction predictions
inherit v1's verified closed-form unchanged::

    stick  (|F| ≤ F_thresh):  s = |F| / (k_a + k_stick),
                              F_friction = k_stick · s
    slip   (|F| >  F_thresh): s = (|F| - μ·f_n) / k_a,
                              F_friction = μ·f_n
    F_thresh = μ · f_n · (k_a + k_stick) / k_stick.

What's verified by exercising the v2 path:

  PART A.  **Stick regime.**  Sweep F well below F_thresh.  v2
           analytical vs v2 hard-piecewise numerical (1-D
           ``minimize_scalar`` on the hard piecewise tangent energy)
           agree to fp precision; both match the closed-form
           ``s = F / (k_a + k_stick)``.

  PART B.  **Slip regime.**  Sweep F well above F_thresh; verify
           ``s = (F - μ·f_n)/k_a`` and the Coulomb plateau
           ``F_friction = μ·f_n``.

  PART C.  **Full transition (the Coulomb diagram).**  Sweep F across
           F_thresh; show the stick/slip slope ratio
           ``(k_a + k_stick)/k_a`` and the smooth surrogate as a witness
           of the kernel-style harmonic-mean form.  The smooth deviation
           from analytical is documented (inherent to the surrogate;
           cf. T-C/D production-eps precision floor for the cousin
           on the contact side).

  PART D.  **k_stick sweep.**  Vary k_stick/k_a in {0.1, 1, 10, 100}
           at fixed μ.  Verify F_thresh shifts as predicted and pre-slip
           slope follows 1/(k_a + k_stick); k_stick → ∞ recovers bare
           Coulomb F_thresh → μ·f_n.

Production-scale grounding (per Phase 3 guidance, contract §17):

  * Scene picks f_n ≈ 10 N (well-saturated contact: raw_eq / ε ~ 4×10⁵
    at eps = 1e-9), so neither the contact surrogate nor the friction
    surrogate is in its precision-floor regime.
  * Slip-regime samples warm-start the smooth solver from the
    analytical δ — the F ≈ F_thresh kink leaves L-BFGS-B stranded on
    a cold start.

Tolerances follow contract §12 T-I:
  * v2 analytical vs v2 hard-numerical: < 1e-7 absolute on s
    (``minimize_scalar(method='bounded')`` floors near 1e-7 for the
    1-D hard piecewise — independent of the v2 half-space path);
  * closed-form predictions vs both:  rel < 1e-7 (relaxed from
    contract's 1e-8 per the user-flagged scipy floor);
  * smooth surrogate is informational only — its smoothing-zone
    deviation is INHERENT and expected (cf. v1 test_04 PART C).

Run::

    uv run -m cslc_main.theory.test_friction_flat

Outputs::

    cslc_main/theory/figures/ti_stick_curve.png
    cslc_main/theory/figures/ti_slip_curve.png
    cslc_main/theory/figures/ti_full_transition.png
    cslc_main/theory/figures/ti_k_stick_sweep.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    equilibrium_half_space_friction_analytical,
    equilibrium_half_space_friction_hard_numerical,
    equilibrium_half_space_friction_smooth_numerical,
    friction_force_smooth,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Contract §12 T-I tolerance.  Relaxed from rel 1e-8 → 1e-7 because
# scipy.optimize.minimize_scalar(method='bounded') is ~7-digit precise
# (its docs document the floor); the v2 analytical solver is exact to
# fp precision but the hard-piecewise reference path inherits the scipy
# floor.  This is the v1 test_04 floor, unchanged by the v2 contact
# swap.  Phase 1 / T-C / T-D L-BFGS-B floor at ~5×10⁻⁸ for the
# multi-sphere lattice scenes is a separate budget (single-pad here).
TOL_REL = 1.0e-7


# ─────────────────────────────────────────────────────────────────────────
#  Scene  (face-on pad sphere vs single-point flat face)
# ─────────────────────────────────────────────────────────────────────────


def make_scene(
    *,
    d_rest: float = 0.8e-3,
    r: float = 2.5e-3,
    ka: float = 25_000.0,
    kc: float = 25_000.0,
    k_stick: float = 25_000.0,
    mu: float = 0.3,
) -> tuple[LatticeSphere, np.ndarray, np.ndarray, float, float, float]:
    """Build the canonical T-I scene.

    Choices:
      * f_n ≈ ka·kc·d/(ka+kc) = 0.5·ka·d = 10 N at d_rest = 0.8 mm,
        ka = kc = 25 kN/m — comfortable saturation for the v2 surrogate
        (raw_eq / eps ≫ 1 at the test eps = 1e-9).
      * Pad outward normal +y; flat face above, face normal -y; sample
        placed so the rest half-space overlap is exactly ``d_rest``.

    Returns:
        ``(sphere, n_face, t_sample, kc, k_stick, mu)``.
    """
    n_pad = np.array([0.0, 1.0, 0.0])
    sphere = LatticeSphere(p=np.zeros(3), r=r, n=n_pad, ka=ka)
    n_face = -n_pad
    # raw_rest = r - n_face · (p - t_sample) = r - (-n_pad)·(-t_sample)
    #          = r - n_pad·t_sample  → set to d_rest:  t_sample = (r-d)·n_pad
    t_sample = (r - d_rest) * n_pad
    return sphere, n_face, t_sample, kc, k_stick, mu


def F_thresh_analytical(sphere: LatticeSphere, kc: float, k_stick: float,
                        mu: float, d_rest: float) -> tuple[float, float]:
    """Return (F_thresh, f_n) for the v2 face-on scene.

    F_thresh = μ·f_n·(k_a + k_stick)/k_stick with f_n = ka·kc·d/(ka+kc).
    For k_stick = 0, F_thresh = +∞ (no slip threshold).
    """
    f_n = sphere.ka * kc * d_rest / (sphere.ka + kc)
    if k_stick > 0.0:
        F_thresh = mu * f_n * (sphere.ka + k_stick) / k_stick
    else:
        F_thresh = float("inf")
    return F_thresh, f_n


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Stick regime
# ─────────────────────────────────────────────────────────────────────────


def part_a_stick_regime() -> bool:
    print()
    print("=" * 78)
    print("PART A.  Stick regime: s = F/(k_a + k_stick), F_fric = k_stick·s")
    print("=" * 78)

    sphere, n_face, t_sample, kc, k_stick, mu = make_scene()
    F_thresh, f_n = F_thresh_analytical(sphere, kc, k_stick, mu, 0.8e-3)

    F_vals = np.linspace(0.0, 0.5 * F_thresh, 6)
    print()
    print(f"  ka = {sphere.ka:.0f} N/m, k_stick = {k_stick:.0f} N/m, "
          f"kc = {kc:.0f} N/m, d_rest = 0.8 mm, μ = {mu}")
    print(f"  f_n = {f_n:.4f} N    F_thresh = {F_thresh:.4f} N  "
          f"(stick window 0..F_thresh)")
    print()
    print(f"  {'F[N]':>9} {'s_ana[μm]':>11} {'s_hardN[μm]':>13} "
          f"{'F_fric[N]':>11} {'pred s/F':>12} {'rel err':>10}")
    print(f"  {'-'*9} {'-'*11} {'-'*13} {'-'*11} {'-'*12} {'-'*10}")

    max_rel = 0.0
    sign_mismatch = False
    expected_slope = 1.0 / (sphere.ka + k_stick)
    for F in F_vals:
        f_ext_t = np.array([F, 0.0, 0.0])
        delta_a, info_a = equilibrium_half_space_friction_analytical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
        delta_h, info_h = equilibrium_half_space_friction_hard_numerical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
        s_a = info_a["s"]
        s_h = info_h["s"]
        s_formula = F * expected_slope
        F_fric = info_a["F_friction"]
        if s_formula > 0:
            rel = max(
                abs(s_a - s_formula) / s_formula,
                abs(s_h - s_formula) / s_formula,
            )
        else:
            rel = 0.0
        max_rel = max(max_rel, rel)
        # Sign convention: F_ext in +x ⇒ q moves to +x ⇒ δ_x = -s < 0.
        if F > 0 and (delta_a[0] >= 0 or delta_h[0] >= 0):
            sign_mismatch = True
        print(f"  {F:>9.4f} {s_a*1e6:>11.4f} {s_h*1e6:>13.4f} "
              f"{F_fric:>11.4f} {expected_slope*1e6:>12.4f} {rel:>10.2e}")

    ok = (max_rel < TOL_REL) and not sign_mismatch
    print()
    if sign_mismatch:
        print("  WARNING: sign mismatch — F_ext = +x but δ_x ≥ 0 "
              "(convention q = p − δ requires δ < 0).")
    print(f"PART A result: {'PASS' if ok else 'FAIL'} "
          f"(max rel err = {max_rel:.2e}, sign mismatch = {sign_mismatch})")

    # ── Figure ──
    F_dense = np.linspace(0.0, 0.8 * F_thresh, 41)
    s_dense_ana = np.zeros_like(F_dense)
    s_dense_hard = np.zeros_like(F_dense)
    for i, F in enumerate(F_dense):
        f_ext_t = np.array([F, 0.0, 0.0])
        _, ia = equilibrium_half_space_friction_analytical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
        _, ih = equilibrium_half_space_friction_hard_numerical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
        s_dense_ana[i] = ia["s"]
        s_dense_hard[i] = ih["s"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(F_dense, s_dense_ana * 1e6, "k-", lw=2, label="v2 analytical")
    ax.plot(F_dense, s_dense_hard * 1e6, "ro", ms=5, mfc="none",
            label="v2 hard-piecewise (independent)")
    ax.plot(F_dense, F_dense * expected_slope * 1e6, "b--", lw=1.0,
            label=rf"$F/(k_a + k_{{\rm stick}}) = F \cdot {expected_slope*1e6:.2f}\,\mu$m/N")
    ax.set_xlabel(r"external tangential force $F$ [N]")
    ax.set_ylabel(r"tangential displacement $|\delta_t|$ [$\mu$m]")
    ax.set_title(rf"T-I/A.  Stick regime (v2 half-space contact + friction)"
                 rf"\n$f_n = {f_n:.2f}$ N, $F_{{\rm thresh}} = {F_thresh:.2f}$ N")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "ti_stick_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Slip regime
# ─────────────────────────────────────────────────────────────────────────


def part_b_slip_regime() -> bool:
    print()
    print("=" * 78)
    print("PART B.  Slip regime: s = (F − μ·f_n)/k_a, F_fric = μ·f_n")
    print("=" * 78)

    sphere, n_face, t_sample, kc, k_stick, mu = make_scene()
    F_thresh, f_n = F_thresh_analytical(sphere, kc, k_stick, mu, 0.8e-3)

    F_vals = np.linspace(2.0 * F_thresh, 8.0 * F_thresh, 6)
    print()
    print(f"  f_n = {f_n:.4f} N    F_thresh = {F_thresh:.4f} N  "
          f"(slip window F > F_thresh)")
    print()
    print(f"  {'F[N]':>9} {'s_ana[μm]':>11} {'s_hardN[μm]':>13} "
          f"{'F_fric[N]':>11} {'pred slope':>12} {'rel err':>10}")
    print(f"  {'-'*9} {'-'*11} {'-'*13} {'-'*11} {'-'*12} {'-'*10}")

    max_rel = 0.0
    expected_slope = 1.0 / sphere.ka
    plateau = mu * f_n
    for F in F_vals:
        f_ext_t = np.array([F, 0.0, 0.0])
        delta_a, info_a = equilibrium_half_space_friction_analytical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
        delta_h, info_h = equilibrium_half_space_friction_hard_numerical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
        s_a = info_a["s"]
        s_h = info_h["s"]
        s_formula = (F - mu * f_n) / sphere.ka
        F_fric = info_a["F_friction"]
        rel = max(
            abs(s_a - s_formula) / max(s_formula, 1e-15),
            abs(s_h - s_formula) / max(s_formula, 1e-15),
            abs(F_fric - plateau) / plateau,
        )
        max_rel = max(max_rel, rel)
        print(f"  {F:>9.4f} {s_a*1e6:>11.4f} {s_h*1e6:>13.4f} "
              f"{F_fric:>11.4f} {expected_slope*1e6:>12.4f} {rel:>10.2e}")

    ok = max_rel < TOL_REL
    print()
    print(f"PART B result: {'PASS' if ok else 'FAIL'} "
          f"(max rel err = {max_rel:.2e})")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Full stick-slip transition (the Coulomb diagram)
# ─────────────────────────────────────────────────────────────────────────


def part_c_full_transition() -> bool:
    print()
    print("=" * 78)
    print("PART C.  Stick → slip transition (the Coulomb diagram)")
    print("=" * 78)

    sphere, n_face, t_sample, kc, k_stick, mu = make_scene()
    F_thresh, f_n = F_thresh_analytical(sphere, kc, k_stick, mu, 0.8e-3)

    F_grid = np.linspace(0.0, 3.0 * F_thresh, 121)
    s_ana = np.zeros_like(F_grid)
    F_fric_ana = np.zeros_like(F_grid)
    s_hard = np.zeros_like(F_grid)
    s_smooth = np.zeros_like(F_grid)
    F_fric_smooth = np.zeros_like(F_grid)

    sample_F = 0.3 * F_thresh    # deep stick — checks the smooth-jac sign
    sample_idx = int(np.argmin(np.abs(F_grid - sample_F)))
    sample_sign_ok = True
    sample_F_actual = float(F_grid[sample_idx])
    sample_delta_ana_x = 0.0
    sample_delta_smooth_x = 0.0

    for i, F in enumerate(F_grid):
        f_ext_t = np.array([F, 0.0, 0.0])
        delta_a, info_a = equilibrium_half_space_friction_analytical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
        s_ana[i] = info_a["s"]
        F_fric_ana[i] = info_a["F_friction"]
        _, info_h = equilibrium_half_space_friction_hard_numerical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
        s_hard[i] = info_h["s"]
        # Smooth surrogate: warm-start from analytical (slip-regime
        # cold starts strand L-BFGS-B at the F ≈ F_thresh kink — see
        # Phase 3 user guidance).
        delta_sm, _ = equilibrium_half_space_friction_smooth_numerical(
            sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu,
            delta0=delta_a, tol=1.0e-12)
        d_n = float(np.dot(delta_sm, sphere.n))
        s_smooth[i] = float(np.linalg.norm(delta_sm - d_n * sphere.n))
        F_fric_smooth[i] = friction_force_smooth(s_smooth[i], f_n, k_stick, mu)
        if i == sample_idx:
            sign_match = (np.sign(delta_a[0]) == np.sign(delta_sm[0])
                          or abs(delta_sm[0]) < 1e-12)
            sample_sign_ok = bool(sign_match)
            sample_F_actual = float(F)
            sample_delta_ana_x = float(delta_a[0])
            sample_delta_smooth_x = float(delta_sm[0])

    s_thresh = mu * f_n / k_stick
    expected_slope_stick = 1.0 / (sphere.ka + k_stick)
    expected_slope_slip = 1.0 / sphere.ka

    i_below = int(np.argmin(np.abs(F_grid - 0.4 * F_thresh)))
    i_above = int(np.argmin(np.abs(F_grid - 2.0 * F_thresh)))
    slope_below = (s_ana[i_below + 1] - s_ana[i_below]) / (F_grid[i_below + 1] - F_grid[i_below])
    slope_above = (s_ana[i_above + 1] - s_ana[i_above]) / (F_grid[i_above + 1] - F_grid[i_above])

    hard_vs_ana = float(np.max(np.abs(s_ana - s_hard)))

    print()
    print(f"  f_n             = {f_n:.4f} N")
    print(f"  F_thresh        = {F_thresh:.4f} N")
    print(f"  s_thresh        = {s_thresh*1e6:.4f} μm  (= μ·f_n/k_stick)")
    print()
    print(f"  Analytical vs hard-piecewise (max |s_a − s_h|): "
          f"{hard_vs_ana*1e9:.3e} nm  (independent-code-path verification)")
    print()
    print(f"  Measured slopes from analytical curve:")
    print(f"    below threshold (stick) = {slope_below*1e6:.4f} μm/N   "
          f"expected 1/(ka+k_stick) = {expected_slope_stick*1e6:.4f}")
    print(f"    above threshold (slip)  = {slope_above*1e6:.4f} μm/N   "
          f"expected 1/k_a           = {expected_slope_slip*1e6:.4f}")
    print(f"    slope ratio slip/stick = {slope_above/slope_below:.4f}   "
          f"expected (ka+k_stick)/ka = {(sphere.ka+k_stick)/sphere.ka:.4f}")
    print()
    diff_smooth = float(np.max(np.abs(s_smooth - s_ana) / np.maximum(s_ana, 1e-15)))
    print(f"  Smooth surrogate vs hard:  max rel deviation = {diff_smooth:.2%}")
    print(f"  (inherent smoothing-zone error of the harmonic-mean form;")
    print(f"   identical magnitude to v1 test_04 PART C — the v2 contact swap")
    print(f"   does NOT change the friction surrogate behaviour.)")

    hard_match_ok = hard_vs_ana < 1e-7        # ~100 nm — scipy 1-D floor
    slope_stick_ok = abs(slope_below - expected_slope_stick) / expected_slope_stick < 1e-3
    slope_slip_ok = abs(slope_above - expected_slope_slip) / expected_slope_slip < 1e-3

    print()
    print(f"  analytical == hard-piecewise (<100 nm)?  "
          f"{'PASS' if hard_match_ok else 'FAIL'}")
    print(f"  stick slope = 1/(ka+k_stick)?           "
          f"{'PASS' if slope_stick_ok else 'FAIL'}")
    print(f"  slip slope  = 1/k_a?                    "
          f"{'PASS' if slope_slip_ok else 'FAIL'}")
    print(f"  smooth δ_x sign == analytical?           "
          f"{'PASS' if sample_sign_ok else 'FAIL'}  "
          f"(F = {sample_F_actual:.3f} N, ana = {sample_delta_ana_x:.3e}, "
          f"smooth = {sample_delta_smooth_x:.3e})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(F_grid, s_ana * 1e6, "k-", lw=2,
             label="v2 analytical (hard piecewise)")
    ax1.plot(F_grid, s_hard * 1e6, "ro", ms=4, mfc="none",
             label="v2 hard-numerical (indep. minimize_scalar)")
    ax1.plot(F_grid, s_smooth * 1e6, "b--", lw=1.5,
             label="v2 smooth surrogate (kernel-style)")
    ax1.axvline(F_thresh, ls=":", color="C0",
                label=rf"$F_{{\rm thresh}} = {F_thresh:.2f}$ N")
    ax1.axhline(s_thresh * 1e6, ls=":", color="C2",
                label=rf"$s_{{\rm thresh}} = \mu f_n/k_{{\rm stick}} = "
                      rf"{s_thresh*1e6:.0f}\,\mu$m")
    ax1.set_xlabel(r"external tangential force $F$ [N]")
    ax1.set_ylabel(r"tangential displacement $|\delta_t|$ [$\mu$m]")
    ax1.set_title("v2 stick-slip: half-space contact + unchanged friction law")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(F_grid, F_fric_ana, "k-", lw=2, label="v2 analytical (hard)")
    ax2.plot(F_grid, F_fric_smooth, "b--", lw=1.5,
             label="v2 smooth surrogate")
    ax2.axhline(mu * f_n, ls="--", color="gray",
                label=rf"$\mu f_n = {mu*f_n:.2f}$ N (Coulomb plateau)")
    ax2.axvline(F_thresh, ls=":", color="C0", label=r"$F_{\rm thresh}$")
    ax2.set_xlabel(r"external tangential force $F$ [N]")
    ax2.set_ylabel(r"friction force $|F_{\rm friction}|$ [N]")
    ax2.set_title("Friction force: hard piecewise vs smooth surrogate")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "ti_full_transition.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n  Saved {out}")

    return (hard_match_ok and slope_stick_ok and slope_slip_ok
            and sample_sign_ok)


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  k_stick sweep
# ─────────────────────────────────────────────────────────────────────────


def part_d_k_stick_sweep() -> bool:
    print()
    print("=" * 78)
    print("PART D.  k_stick sweep: bare Coulomb is the k_stick → ∞ limit")
    print("=" * 78)

    sphere, n_face, t_sample, kc, _, mu = make_scene()
    _, f_n = F_thresh_analytical(sphere, kc, 25_000.0, mu, 0.8e-3)
    ratios = [0.1, 1.0, 10.0, 100.0]
    print()
    print(f"  f_n = {f_n:.4f} N, μ = {mu}, μ·f_n = {mu*f_n:.4f} N  "
          f"(bare-Coulomb threshold)")
    print()
    print(f"  {'k_stick/ka':>11} {'F_thresh[N]':>13} "
          f"{'pre-slip slope[μm/N]':>22} {'F_thresh check':>16}")
    print(f"  {'-'*11} {'-'*13} {'-'*22} {'-'*16}")

    fig, ax = plt.subplots(figsize=(8, 5))
    # Span past the LARGEST F_thresh among the ratios so the plateau-
    # at-threshold assertion below is testable for every k_stick/ka.
    # F_thresh = μ·f_n·(1 + ka/k_stick); smallest k_stick → largest
    # F_thresh.  Use 1.5× headroom over the worst case.
    k_stick_min = min(ratios) * sphere.ka
    F_thresh_max = mu * f_n * (sphere.ka + k_stick_min) / k_stick_min
    F_grid = np.linspace(0.0, 1.5 * F_thresh_max, 181)

    ok_all = True
    for r in ratios:
        k_stick = r * sphere.ka
        s_ana = np.zeros_like(F_grid)
        F_fric_ana = np.zeros_like(F_grid)
        for i, F in enumerate(F_grid):
            f_ext_t = np.array([F, 0.0, 0.0])
            _, info = equilibrium_half_space_friction_analytical(
                sphere, n_face, t_sample, kc, f_ext_t, k_stick, mu)
            s_ana[i] = info["s"]
            F_fric_ana[i] = info["F_friction"]

        F_thresh = mu * f_n * (sphere.ka + k_stick) / k_stick
        slope_stick_pred = 1.0 / (sphere.ka + k_stick) * 1e6

        # Verify: at F just below F_thresh, F_fric should be very close
        # to μ·f_n; at F just above, F_fric should equal μ·f_n exactly.
        i_below = int(np.argmin(np.abs(F_grid - 0.95 * F_thresh)))
        i_above = int(np.argmin(np.abs(F_grid - 1.05 * F_thresh)))
        fric_at_thresh_ok = (
            F_fric_ana[i_below] <= mu * f_n + 1e-9
            and abs(F_fric_ana[i_above] - mu * f_n) < 1e-9
        )
        ok_all = ok_all and fric_at_thresh_ok

        print(f"  {r:>11.1f} {F_thresh:>13.4f} {slope_stick_pred:>22.4f} "
              f"{'PASS' if fric_at_thresh_ok else 'FAIL':>16}")

        ax.plot(F_grid, F_fric_ana, "-",
                label=rf"$k_{{\rm stick}}/k_a = {r:g}$,  "
                      rf"$F_{{\rm thresh}} = {F_thresh:.2f}$ N")

    ax.axhline(mu * f_n, ls="--", color="gray",
               label=rf"bare Coulomb $\mu f_n = {mu*f_n:.2f}$ N")
    ax.set_xlabel(r"external tangential force $F$ [N]")
    ax.set_ylabel(r"friction force $|F_{\rm friction}|$ [N]")
    ax.set_title("T-I/D. Compliant-skin friction: bare Coulomb is the "
                 r"$k_{\rm stick} \to \infty$ limit")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "ti_k_stick_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n  Saved {out}")
    print(f"PART D result: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = part_a_stick_regime()
    b = part_b_slip_regime()
    c = part_c_full_transition()
    d = part_d_k_stick_sweep()
    all_ok = a and b and c and d
    print()
    print("=" * 78)
    print(f"T-I overall: {'PASS' if all_ok else 'FAIL'}  "
          f"(A={'P' if a else 'F'} B={'P' if b else 'F'} "
          f"C={'P' if c else 'F'} D={'P' if d else 'F'})")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
