# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 4 / Stick-slip friction on the single lattice sphere.

Builds directly on step 1: same isolated sphere, face-on rigid target.
We add a tangential external force ``F`` perpendicular to the rest
normal, plus a friction model on the contact patch:

* **Stick** (when ``k_stick * |delta_t| <= mu * f_n``): friction is a
  Hookean spring  ``F_friction = k_stick * delta_t``.  The body's
  tangent equilibrium combines anchor (k_a) and friction (k_stick) in
  parallel, giving
      |delta_t| = |F| / (k_a + k_stick),
      |F_friction| = k_stick * |F| / (k_a + k_stick).

* **Slip** (when ``k_stick * |delta_t| > mu * f_n``): friction is
  clamped at the Coulomb cone, ``|F_friction| = mu * f_n``.  Anchor
  alone resists motion:
      |delta_t| = (|F| - mu * f_n) / k_a,
      |F_friction| = mu * f_n.

Transition at  ``F_thresh = mu * f_n * (k_a + k_stick) / k_stick``.

Math derivation: see notes.md step 4.

Test parts:

  PART A.  Stick regime sweep.  For F << F_thresh, verify
           |delta_t| = F / (k_a + k_stick) and friction force tracks
           Hooke's law on the parallel-spring composition.

  PART B.  Slip regime.  For F well above F_thresh, verify
           |delta_t| = (F - mu * f_n) / k_a and the friction force
           plateaus at mu * f_n.

  PART C.  Continuous transition.  Sweep F across F_thresh; show
           |delta_t| is continuous with a slope kink, and the friction
           force grows linearly then saturates.  This is the classic
           Coulomb diagram with finite stick stiffness.

  PART D.  k_stick sweep.  Vary k_stick / k_a from 0.1 to 100 at fixed
           mu.  Show how F_thresh shifts and the pre-slip stiffness
           changes.  k_stick -> inf recovers the bare-Coulomb law
           F_thresh -> mu * f_n.

Run::

    uv run -m cslc_main.theory.test_04_friction

Outputs::

    cslc_main/theory/figures/04a_stick_curve.png
    cslc_main/theory/figures/04b_slip_curve.png
    cslc_main/theory/figures/04c_full_stick_slip.png
    cslc_main/theory/figures/04d_k_stick_sweep.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    RigidTarget,
    equilibrium_face_on_analytical,
    equilibrium_with_friction_analytical,
    equilibrium_with_friction_hard_numerical,
    equilibrium_with_friction_smooth_numerical,
    friction_force_smooth,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ─────────────────────────────────────────────────────────────────────────
#  Scene
# ─────────────────────────────────────────────────────────────────────────


def make_scene(
    *,
    phi_rest: float = 1.0e-3,
    r: float = 2.5e-3,
    R: float = 33.5e-3,
    ka: float = 25_000.0,
    kc: float = 25_000.0,
    k_stick: float = 25_000.0,
    mu: float = 0.3,
) -> tuple[LatticeSphere, RigidTarget, float, float, float]:
    """Build the canonical scene for step 4.

    Returns (sphere, target, kc, k_stick, mu).
    """
    n = np.array([0.0, 1.0, 0.0])
    sphere = LatticeSphere(p=np.zeros(3), r=r, n=n, ka=ka)
    d = (r + R) - phi_rest
    target = RigidTarget(t=d * n, R=R)
    return sphere, target, kc, k_stick, mu


def F_thresh_analytical(sphere: LatticeSphere, kc: float, k_stick: float,
                        mu: float, phi_rest: float) -> tuple[float, float]:
    """Return (F_thresh, f_n) for the chosen scene.

    F_thresh is the external tangential force at which stick gives way
    to slip:  k_stick * s = mu * f_n  with  s = F / (k_a + k_stick).
    For k_stick = 0 the system never slips -- F_thresh = +inf.
    """
    # Normal equilibrium force (series spring, paper IV.A).
    f_n = sphere.ka * kc * phi_rest / (sphere.ka + kc)
    if k_stick > 0.0:
        F_thresh = mu * f_n * (sphere.ka + k_stick) / k_stick
    else:
        F_thresh = float("inf")
    return F_thresh, f_n


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Stick-regime verification
# ─────────────────────────────────────────────────────────────────────────


def part_a_stick_regime() -> bool:
    """Sweep F well below F_thresh; verify analytical and hard-numerical agree.

    Reference is the closed-form  s = F / (ka + k_stick); we verify
    against an independent code path (scipy minimize_scalar on the
    hard piecewise 1D energy).  Smoothed L-BFGS-B is shown only as a
    witness of the kernel-style differentiable surrogate, which has a
    known smoothing zone in the transition (see PART C plot).
    """
    print()
    print("=" * 72)
    print("PART A.  Stick regime: s = F / (k_a + k_stick), F_fric = k_stick * s")
    print("=" * 72)

    sphere, target, kc, k_stick, mu = make_scene()
    F_thresh, f_n = F_thresh_analytical(sphere, kc, k_stick, mu, 1e-3)

    F_vals = np.linspace(0.0, 0.5 * F_thresh, 6)
    print()
    print(f"  ka = {sphere.ka:.0f}, k_stick = {k_stick:.0f}, "
          f"kc = {kc:.0f}, phi = 1.0 mm, mu = {mu}")
    print(f"  f_n  = {f_n:.4f} N    F_thresh = {F_thresh:.4f} N "
          f"(stick window 0 .. F_thresh)")
    print()
    print(f"  {'F[N]':>9} {'s_ana[um]':>11} {'s_hardN[um]':>13} "
          f"{'F_fric[N]':>11} {'pred s/F':>12} {'rel err':>10}")
    print(f"  {'-'*9} {'-'*11} {'-'*13} {'-'*11} {'-'*12} {'-'*10}")

    max_rel = 0.0
    sign_mismatch = False
    expected_slope = 1.0 / (sphere.ka + k_stick)  # m/N
    for F in F_vals:
        f_ext_t = np.array([F, 0.0, 0.0])
        delta_ana, info_ana = equilibrium_with_friction_analytical(
            sphere, target, kc, f_ext_t, k_stick, mu)
        delta_hard, info_hard = equilibrium_with_friction_hard_numerical(
            sphere, target, kc, f_ext_t, k_stick, mu)
        s_ana = info_ana["s"]
        s_hard = info_hard["s"]
        # closed-form formula in stick: s = F * expected_slope.
        s_formula = F * expected_slope
        F_fric = info_ana["F_friction"]
        # Compare against the formula and the independent numerical.
        rel = max(
            abs(s_ana - s_formula) / max(s_formula, 1e-15),
            abs(s_hard - s_formula) / max(s_formula, 1e-15),
        ) if s_formula > 0 else 0.0
        max_rel = max(max_rel, rel)
        # Sign check on the vector: with q = p - delta, F_ext in +x gives
        # delta_x < 0.  Catch any future sign regression.
        if F > 0 and delta_ana[0] >= 0:
            sign_mismatch = True
        if F > 0 and delta_hard[0] >= 0:
            sign_mismatch = True
        print(f"  {F:>9.4f} {s_ana*1e6:>11.4f} {s_hard*1e6:>13.4f} "
              f"{F_fric:>11.4f} {expected_slope*1e6:>12.4f} {rel:>10.2e}")

    ok = (max_rel < 1e-8) and not sign_mismatch
    print()
    if sign_mismatch:
        print(f"  WARNING: sign mismatch -- F_ext = +x but delta_x >= 0; "
              f"convention is q = p - delta so delta should be -ve")
    print(f"PART A result: {'PASS' if ok else 'FAIL'} (max rel err = {max_rel:.2e}, "
          f"sign mismatch = {sign_mismatch})")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Slip-regime verification
# ─────────────────────────────────────────────────────────────────────────


def part_b_slip_regime() -> bool:
    """Sweep F well above F_thresh; verify s = (F - mu*f_n)/ka and plateau."""
    print()
    print("=" * 72)
    print("PART B.  Slip regime: s = (F - mu*f_n) / k_a, F_fric = mu * f_n")
    print("=" * 72)

    sphere, target, kc, k_stick, mu = make_scene()
    F_thresh, f_n = F_thresh_analytical(sphere, kc, k_stick, mu, 1e-3)

    F_vals = np.linspace(2.0 * F_thresh, 8.0 * F_thresh, 6)
    print()
    print(f"  f_n = {f_n:.4f} N    F_thresh = {F_thresh:.4f} N "
          f"(slip window F > F_thresh)")
    print()
    print(f"  {'F[N]':>9} {'s_ana[um]':>11} {'s_hardN[um]':>13} "
          f"{'F_fric[N]':>11} {'pred slope':>12} {'rel err':>10}")
    print(f"  {'-'*9} {'-'*11} {'-'*13} {'-'*11} {'-'*12} {'-'*10}")

    max_rel = 0.0
    expected_slope = 1.0 / sphere.ka
    plateau = mu * f_n
    for F in F_vals:
        f_ext_t = np.array([F, 0.0, 0.0])
        delta_ana, info_ana = equilibrium_with_friction_analytical(
            sphere, target, kc, f_ext_t, k_stick, mu)
        delta_hard, info_hard = equilibrium_with_friction_hard_numerical(
            sphere, target, kc, f_ext_t, k_stick, mu)
        s_ana = info_ana["s"]
        s_hard = info_hard["s"]
        s_formula = (F - mu * f_n) / sphere.ka
        F_fric = info_ana["F_friction"]
        rel = max(
            abs(s_ana - s_formula) / max(s_formula, 1e-15),
            abs(s_hard - s_formula) / max(s_formula, 1e-15),
            abs(F_fric - plateau) / plateau,
        )
        max_rel = max(max_rel, rel)
        print(f"  {F:>9.4f} {s_ana*1e6:>11.4f} {s_hard*1e6:>13.4f} "
              f"{F_fric:>11.4f} {expected_slope*1e6:>12.4f} {rel:>10.2e}")

    # scipy.optimize.minimize_scalar(method='bounded') is ~7-digit
    # precise; tighter would need a different method.  We use 1e-7 as
    # the floor, well below physically meaningful resolution.
    ok = max_rel < 1e-7
    print()
    print(f"PART B result: {'PASS' if ok else 'FAIL'} (max rel err = {max_rel:.2e})")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Full stick-slip transition
# ─────────────────────────────────────────────────────────────────────────


def part_c_full_transition() -> bool:
    """Plot the analytical hard curve + the smooth-surrogate witness."""
    print()
    print("=" * 72)
    print("PART C.  Stick -> slip transition (the Coulomb diagram)")
    print("=" * 72)

    sphere, target, kc, k_stick, mu = make_scene()
    F_thresh, f_n = F_thresh_analytical(sphere, kc, k_stick, mu, 1e-3)

    F_grid = np.linspace(0.0, 3.0 * F_thresh, 121)
    s_ana = np.zeros_like(F_grid)
    F_fric_ana = np.zeros_like(F_grid)
    s_hard = np.zeros_like(F_grid)
    # Smooth surrogate witness (the kernel's harmonic-mean surrogate)
    s_smooth = np.zeros_like(F_grid)
    F_fric_smooth = np.zeros_like(F_grid)

    # Track the SIGN of delta_x_smooth at a deep-stick sample to catch
    # the historic sign bug (smooth solver returning +x when it should
    # return -x for F in +x).  If the sign disagrees with analytical,
    # the smooth jac has regressed.
    sample_F = 0.3 * F_thresh   # deep stick
    sample_idx = int(np.argmin(np.abs(F_grid - sample_F)))

    for i, F in enumerate(F_grid):
        f_ext_t = np.array([F, 0.0, 0.0])
        delta_ana, info_ana = equilibrium_with_friction_analytical(
            sphere, target, kc, f_ext_t, k_stick, mu)
        s_ana[i] = info_ana["s"]
        F_fric_ana[i] = info_ana["F_friction"]
        _, info_hard = equilibrium_with_friction_hard_numerical(
            sphere, target, kc, f_ext_t, k_stick, mu)
        s_hard[i] = info_hard["s"]
        # Smooth surrogate: same scene through L-BFGS-B on the smoothed
        # friction energy.
        delta_sm, _ = equilibrium_with_friction_smooth_numerical(
            sphere, target, kc, f_ext_t, k_stick, mu, tol=1e-12)
        d_n = float(np.dot(delta_sm, sphere.n))
        s_smooth[i] = float(np.linalg.norm(delta_sm - d_n * sphere.n))
        F_fric_smooth[i] = friction_force_smooth(s_smooth[i], f_n, k_stick, mu)
        if i == sample_idx:
            sign_match = (np.sign(delta_ana[0]) == np.sign(delta_sm[0])) or (abs(delta_sm[0]) < 1e-12)
            sample_sign_ok = bool(sign_match)
            sample_F_actual = float(F)
            sample_delta_ana_x = float(delta_ana[0])
            sample_delta_smooth_x = float(delta_sm[0])

    # Continuity / property checks based on the ANALYTICAL hard curve.
    s_thresh = mu * f_n / k_stick
    expected_slope_stick = 1.0 / (sphere.ka + k_stick)
    expected_slope_slip = 1.0 / sphere.ka

    # Measured slopes from the analytical curve.
    i_below = np.argmin(np.abs(F_grid - 0.4 * F_thresh))
    i_above = np.argmin(np.abs(F_grid - 2.0 * F_thresh))
    slope_below = (s_ana[i_below + 1] - s_ana[i_below]) / (F_grid[i_below + 1] - F_grid[i_below])
    slope_above = (s_ana[i_above + 1] - s_ana[i_above]) / (F_grid[i_above + 1] - F_grid[i_above])

    # Cross-check analytical vs hard-numerical.
    hard_vs_ana = float(np.max(np.abs(s_ana - s_hard)))

    print()
    print(f"  f_n             = {f_n:.4f} N")
    print(f"  F_thresh        = {F_thresh:.4f} N")
    print(f"  s_thresh        = {s_thresh*1e6:.4f} um   (= mu*f_n/k_stick)")
    print()
    print(f"  Analytical vs hard-piecewise numerical (max |s_ana - s_hard|): "
          f"{hard_vs_ana*1e9:.3e} nm   (independent-code-path verification)")
    print()
    print(f"  Measured slopes from analytical curve:")
    print(f"    below threshold (stick) = {slope_below*1e6:.4f} um/N   "
          f"expected 1/(ka+k_stick) = {expected_slope_stick*1e6:.4f}")
    print(f"    above threshold (slip)  = {slope_above*1e6:.4f} um/N   "
          f"expected 1/k_a            = {expected_slope_slip*1e6:.4f}")
    print(f"    slope ratio slip/stick = {slope_above/slope_below:.4f}   "
          f"expected (ka+k_stick)/ka = {(sphere.ka+k_stick)/sphere.ka:.4f}")
    print()
    print(f"  Smooth surrogate (kernel-style harmonic-mean) compared to hard:")
    diff_in_transition = float(np.max(np.abs(s_smooth - s_ana)
                                      / np.maximum(s_ana, 1e-15)))
    print(f"    max rel deviation across full sweep = {diff_in_transition:.2%}")
    print(f"    (this is the smoothing-zone error, INHERENT to the harmonic-mean form;")
    print(f"     deep stick and deep slip recover the hard law exactly)")

    # Pass criteria: analytical agrees with hard piecewise numerical;
    # slopes match closed-form expectations.
    hard_match_ok = hard_vs_ana < 1e-9    # nm-level agreement
    slope_stick_ok = abs(slope_below - expected_slope_stick) / expected_slope_stick < 1e-3
    slope_slip_ok = abs(slope_above - expected_slope_slip) / expected_slope_slip < 1e-3

    print()
    print(f"  analytical == hard-piecewise (1 nm)?   "
          f"{'PASS' if hard_match_ok else 'FAIL'}")
    print(f"  stick slope = 1/(ka+k_stick)?          "
          f"{'PASS' if slope_stick_ok else 'FAIL'}")
    print(f"  slip slope = 1/k_a?                    "
          f"{'PASS' if slope_slip_ok else 'FAIL'}")
    print(f"  smooth delta_x sign == analytical?     "
          f"{'PASS' if sample_sign_ok else 'FAIL'}  "
          f"(F = {sample_F_actual:.3f} N, ana = {sample_delta_ana_x:.3e}, "
          f"smooth = {sample_delta_smooth_x:.3e})")

    # Plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(F_grid, s_ana * 1e6, "k-", lw=2, label="analytical (hard piecewise)")
    ax1.plot(F_grid, s_hard * 1e6, "ro", ms=4, mfc="none",
             label="numerical (hard, indep. minimize_scalar)")
    ax1.plot(F_grid, s_smooth * 1e6, "b--", lw=1.5,
             label="smooth surrogate (kernel-style)")
    ax1.axvline(F_thresh, ls=":", color="C0",
                label=rf"$F_{{\mathrm{{thresh}}}} = {F_thresh:.2f}$ N")
    ax1.axhline(s_thresh * 1e6, ls=":", color="C2",
                label=rf"$s_{{\mathrm{{thresh}}}} = \mu f_n/k_{{\mathrm{{stick}}}} = "
                      rf"{s_thresh*1e6:.0f}\,\mu$m")
    ax1.set_xlabel(r"external tangential force $F$ [N]")
    ax1.set_ylabel(r"tangential displacement $|\delta_t|$ [$\mu$m]")
    ax1.set_title("Tangential displacement: ideal hard law and smooth witness")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(F_grid, F_fric_ana, "k-", lw=2, label="analytical (hard)")
    ax2.plot(F_grid, F_fric_smooth, "b--", lw=1.5,
             label="smooth surrogate (kernel-style)")
    ax2.axhline(mu * f_n, ls="--", color="gray",
                label=rf"$\mu f_n = {mu*f_n:.2f}$ N (Coulomb plateau)")
    ax2.axvline(F_thresh, ls=":", color="C0", label=r"$F_{\mathrm{thresh}}$")
    ax2.set_xlabel(r"external tangential force $F$ [N]")
    ax2.set_ylabel(r"friction force $|F_{\mathrm{friction}}|$ [N]")
    ax2.set_title("Friction force: hard piecewise vs smooth surrogate")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "04c_full_stick_slip.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    return (hard_match_ok and slope_stick_ok and slope_slip_ok
            and sample_sign_ok)


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  k_stick sweep
# ─────────────────────────────────────────────────────────────────────────


def part_d_k_stick_sweep() -> None:
    """Sweep k_stick / k_a; show how the transition shape responds."""
    print()
    print("=" * 72)
    print("PART D.  k_stick sweep: bare Coulomb as k_stick -> inf limit")
    print("=" * 72)

    sphere, target, kc, _, mu = make_scene()
    F_thresh_base, f_n = F_thresh_analytical(sphere, kc, 25000.0, mu, 1e-3)
    ratios = [0.1, 1.0, 10.0, 100.0]
    print()
    print(f"  f_n = {f_n:.4f} N, mu = {mu}, mu*f_n = {mu*f_n:.4f} N (bare Coulomb threshold)")
    print()
    print(f"  {'k_stick/ka':>11} {'F_thresh[N]':>13} {'pre-slip slope[um/N]':>22}")
    print(f"  {'-'*11} {'-'*13} {'-'*22}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    F_grid = np.linspace(0.0, 6.0 * mu * f_n, 121)

    for r in ratios:
        k_stick = r * sphere.ka
        s_ana = np.zeros_like(F_grid)
        F_fric_ana = np.zeros_like(F_grid)
        for i, F in enumerate(F_grid):
            f_ext_t = np.array([F, 0.0, 0.0])
            _, info = equilibrium_with_friction_analytical(
                sphere, target, kc, f_ext_t, k_stick, mu)
            s_ana[i] = info["s"]
            F_fric_ana[i] = info["F_friction"]

        F_thresh = mu * f_n * (sphere.ka + k_stick) / k_stick
        slope_stick = 1.0 / (sphere.ka + k_stick) * 1e6  # um/N
        print(f"  {r:>11.1f} {F_thresh:>13.4f} {slope_stick:>22.4f}")

        ax.plot(F_grid, F_fric_ana, "-",
                label=rf"$k_{{\mathrm{{stick}}}}/k_a = {r:g}$,  "
                      rf"$F_{{\mathrm{{thresh}}}} = {F_thresh:.2f}$ N")

    ax.axhline(mu * f_n, ls="--", color="gray",
               label=rf"bare Coulomb $\mu f_n = {mu*f_n:.2f}$ N")
    ax.set_xlabel(r"external tangential force $F$ [N]")
    ax.set_ylabel(r"friction force $|F_{\mathrm{friction}}|$ [N]")
    ax.set_title("Compliant-skin friction: bare Coulomb is the "
                 "$k_{\\mathrm{stick}}\\to\\infty$ limit")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "04d_k_stick_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = part_a_stick_regime()
    b = part_b_slip_regime()
    c = part_c_full_transition()
    part_d_k_stick_sweep()
    all_ok = a and b and c
    print()
    print("=" * 72)
    print(f"Step 4 friction test: {'PASS' if all_ok else 'FAIL'}   "
          f"(A={'P' if a else 'F'} B={'P' if b else 'F'} C={'P' if c else 'F'})")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
