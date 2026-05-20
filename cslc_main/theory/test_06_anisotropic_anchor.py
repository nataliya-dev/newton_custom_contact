# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 6 / Anisotropic anchor on a single sphere with friction.

The CSLC anchor spring is the elastic recoil of the compliant skin
pulling its centre back toward the rest lattice site.  Until step 5 we
treated this spring as isotropic (k_a per Cartesian axis).  A real
compliant skin is closer to incompressible flesh: shear modulus
G = E / (2 (1 + nu)) with nu approaching 0.5, so G / K = ratio that
goes to zero as the material becomes incompressible.  For a uniform
isotropic solid, the relevant per-axis anchor ratio is

    ka_t / ka_n  =  G / E  =  1 / (2 (1 + nu))     (nu = 0.0  -> 0.5)
                                                    (nu = 0.25 -> 0.4)
                                                    (nu = 0.5  -> 1/3, incompressible)

so a tangent / normal anchor ratio of 1/3 corresponds to a fully
incompressible compliant skin.  The skin can still be compressed
normally (the tissue is held in by the rigid body underneath) but it
shears more easily.  See cslc_theory.LatticeSphere.ka_t_ratio.

What changes at the single-sphere level:

* Step 1 face-on contact:  unchanged.  Normal anchor sees ka, contact
  spring is along n_hat, no tangent motion -> ka_t_ratio is irrelevant.

* Step 1 off-axis contact:  off-axis delta has a tangent component
  that ka_t resists; softer ka_t means the delta tilts MORE toward the
  contact line.  Closed form below (PART A).

* Step 4 stick-slip threshold:  in stick mode, the tangent equation
  is now  (ka_t + k_stick) * s = F.  In slip mode it is  ka_t * s =
  F - mu * f_n.  The slip threshold becomes

      F_thresh = mu * f_n * (ka_t + k_stick) / k_stick

  At ka_t < ka the system slips at LOWER applied force (softer skin
  reaches the Coulomb cone earlier).  The Coulomb plateau itself
  (mu * f_n) is unchanged.

Test parts:

  PART A.  Off-axis contact at fixed (k_a, k_c, phi).  Sweep
           ka_t_ratio in {1.0, 0.5, 1/3, 0.1}.  Verify that the
           tangent component of the equilibrium delta scales as
           predicted by the closed form

               delta_t / delta_n = (ka_n / ka_t) * tan(theta)

           where theta is the off-axis angle and delta_n, delta_t are
           projections onto the apex's outward normal / tangent.  This
           formula falls out of decoupling the axes in the local
           {n, t} frame at small angles.

  PART B.  Stick-slip threshold shift.  Hold ka, kc, phi, k_stick,
           mu fixed.  Sweep ka_t_ratio in {2.0, 1.0, 1/3, 0.1}; for
           each ratio sweep F across the predicted F_thresh and
           verify the empirical kink lines up with the closed-form
           F_thresh(ka_t).

  PART C.  Stick-slope verification.  In deep stick, dF_friction / dF
           = k_stick / (ka_t + k_stick).  Soft tangent (ka_t -> 0)
           gives slope -> 1 (all of F goes into friction).  Stiff
           tangent (ka_t -> inf) gives slope -> 0 (anchor takes
           everything).  Plot dF_friction/dF vs ka_t.

  PART D.  Incompressible limit visualisation.  Run nu in {0, 0.25,
           0.4, 0.5} (via ka_t_ratio = 1/(2(1+nu))).  Show the
           pre-slip s vs F curve and the F_thresh shift.  Annotate
           nu = 0.5 as the "compliant flesh" reference.

Run::

    uv run -m cslc_main.theory.test_06_anisotropic_anchor

Outputs::

    cslc_main/theory/figures/06a_off_axis_tilt.png
    cslc_main/theory/figures/06b_stick_slip_threshold.png
    cslc_main/theory/figures/06c_stick_slope.png
    cslc_main/theory/figures/06d_poisson_sweep.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    RigidTarget,
    equilibrium_numerical,
    equilibrium_with_friction_analytical,
    equilibrium_with_friction_hard_numerical,
    equilibrium_with_friction_smooth_numerical,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ─────────────────────────────────────────────────────────────────────────
#  Convenience builders
# ─────────────────────────────────────────────────────────────────────────


def make_sphere(*, ka: float = 25_000.0, ka_t_ratio: float = 1.0,
                n: np.ndarray | None = None) -> LatticeSphere:
    if n is None:
        n = np.array([0.0, 1.0, 0.0])
    return LatticeSphere(p=np.zeros(3), r=2.5e-3, n=n, ka=ka,
                         ka_t_ratio=ka_t_ratio)


def make_face_on_target(sphere: LatticeSphere, phi_rest: float,
                        R: float = 33.5e-3) -> RigidTarget:
    d = (sphere.r + R) - phi_rest
    return RigidTarget(t=sphere.p + d * sphere.n, R=R)


def make_off_axis_target(sphere: LatticeSphere, phi_rest: float,
                         theta_deg: float, R: float = 33.5e-3) -> RigidTarget:
    """Place the target so the line of centres makes angle theta with sphere.n.

    Keeps closest-approach distance equal to (r + R) - phi_rest so the
    rest overlap is the same as the face-on case.  Off-axis direction
    is in the (n, x) plane.
    """
    theta = np.radians(theta_deg)
    # Unit vector at angle theta from sphere.n, in the (n, x_hat) plane.
    # sphere.n = +y by default, so the off-axis tangent direction is +x.
    n = sphere.n
    # Pick any tangent direction t_hat; here we use +x relative to n.
    # If n = +y then t_hat = +x.  We build t_hat = (sphere.n x +z) / |...| .
    z = np.array([0.0, 0.0, 1.0])
    t_hat = np.cross(n, z)
    t_hat /= np.linalg.norm(t_hat)
    u = np.cos(theta) * n + np.sin(theta) * t_hat
    d = (sphere.r + R) - phi_rest
    return RigidTarget(t=d * u + sphere.p, R=R)


def F_thresh(sphere: LatticeSphere, kc: float, phi_rest: float,
             k_stick: float, mu: float) -> tuple[float, float, float]:
    """Closed-form (F_thresh, f_n, ka_t) for the given anisotropic sphere.

    f_n stays at the isotropic-friction step-4 value (face-on normal
    equilibrium uses ka, not ka_t).  F_thresh shifts with ka_t.
    """
    ka_t = sphere.ka * sphere.ka_t_ratio
    f_n = sphere.ka * kc * phi_rest / (sphere.ka + kc)
    if k_stick > 0.0 and mu > 0.0:
        F_t = mu * f_n * (ka_t + k_stick) / k_stick
    else:
        F_t = float("inf")
    return F_t, f_n, ka_t


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Off-axis equilibrium: anisotropic tilt
# ─────────────────────────────────────────────────────────────────────────


def part_a_off_axis_tilt() -> bool:
    """Sweep ka_t_ratio; show off-axis delta tilts MORE for softer tangent.

    The off-axis equilibrium in the apex's local ``{n_hat, t_hat}`` frame
    (target along ``u = cos(theta) n_hat + sin(theta) t_hat``):

        q - t = -(delta_n + d cos(theta)) n_hat - (delta_t + d sin(theta)) t_hat
        L     = ||q - t||
        phi_eff = (r + R) - L

    Per-axis equilibrium (with anisotropic anchor):

        ka  * delta_n = kc * phi_eff * (delta_n + d cos(theta)) / L
        ka_t * delta_t = kc * phi_eff * (delta_t + d sin(theta)) / L

    Defining the effective contact coupling  A = kc * phi_eff / L  (a
    POSITIVE constant once the equilibrium is found), the EXACT ratio is

        delta_t / delta_n  =  (ka - A) / (ka_t - A) * tan(theta).

    The naive linear formula  delta_t/delta_n = (ka/ka_t) tan(theta)
    drops the  A  terms; it is accurate when  A << ka_t.  For our
    setup, A ≈ kc*phi/2 / (r+R) ≈ 347 N/m at kc = ka = 25 kN/m,
    phi = 1 mm, r+R ≈ 36 mm.  At ka_t/ka = 1/3 (ka_t = 8333 N/m),
    A/ka_t ≈ 4% -- the naive formula is wrong by that much.  We
    verify the EXACT formula instead, computing A self-consistently
    from the numerical equilibrium.
    """
    print()
    print("=" * 72)
    print("PART A.  Off-axis delta tilts as ka_t softens")
    print("=" * 72)

    ka = 25_000.0
    kc = 25_000.0
    phi = 1.0e-3
    # theta = 1 deg keeps delta_t small relative to delta_n at all
    # ratios down to 0.1 (predicted delta_t/delta_n ~ 0.18 at ratio
    # = 0.1).  Even at this small angle the naive  (ka/ka_t) tan(theta)
    # formula is wrong by a few percent at small ka_t -- so we verify
    # the EXACT formula  (ka-A)/(ka_t-A) tan(theta)  with A computed
    # self-consistently from the numerical equilibrium.
    theta_deg = 1.0
    ka_t_ratios = [1.0, 0.5, 1.0/3.0, 0.1]

    print()
    print(f"  ka = {ka:.0f} N/m, kc = {kc:.0f} N/m, phi = {phi*1e3:.2f} mm,  "
          f"theta = {theta_deg:.1f} deg")
    print()
    print(f"  {'ka_t/ka':>9} {'ka_t':>10} {'δ_n[µm]':>10} {'δ_t[µm]':>10} "
          f"{'δ_t/δ_n':>10} {'naive':>10} {'exact':>10} {'rel(exact)':>11}")
    print(f"  {'-'*9} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11}")

    n_hat = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    t_hat = np.cross(n_hat, z)
    t_hat /= np.linalg.norm(t_hat)

    rel_errs_exact = []
    series = []
    R_target = 33.5e-3
    r_sphere = 2.5e-3
    d_geom = (r_sphere + R_target) - phi   # rest line-of-centres distance
    for ratio in ka_t_ratios:
        sphere = make_sphere(ka=ka, ka_t_ratio=ratio, n=n_hat)
        target = make_off_axis_target(sphere, phi_rest=phi, theta_deg=theta_deg)
        # Warm-start with the isotropic series-spring direction.
        u = (np.cos(np.radians(theta_deg)) * n_hat
             + np.sin(np.radians(theta_deg)) * t_hat)
        delta0 = (kc / (ka + kc)) * phi * u
        delta, info = equilibrium_numerical(sphere, target, kc,
                                            eps=1e-9, delta0=delta0, tol=1e-12)
        delta_n_val = float(np.dot(delta, n_hat))
        delta_t_val = float(np.dot(delta, t_hat))
        measured_ratio = delta_t_val / max(delta_n_val, 1e-30)
        # Naive linear prediction (drops the A/ka_t correction).
        naive_ratio = (1.0 / ratio) * np.tan(np.radians(theta_deg))
        # Exact prediction with A computed self-consistently from the
        # numerical equilibrium.  A = kc * phi_eff / L  is the
        # "effective contact coupling" that appears in both axes.
        q_minus_t_n = -(delta_n_val + d_geom * np.cos(np.radians(theta_deg)))
        q_minus_t_t = -(delta_t_val + d_geom * np.sin(np.radians(theta_deg)))
        L = np.hypot(q_minus_t_n, q_minus_t_t)
        phi_eff = (r_sphere + R_target) - L
        A = kc * phi_eff / L
        ka_t = ka * ratio
        exact_ratio = (ka - A) / (ka_t - A) * np.tan(np.radians(theta_deg))
        rel = abs(measured_ratio - exact_ratio) / max(abs(exact_ratio), 1e-30)
        rel_errs_exact.append(rel)
        series.append((ratio, ka_t, delta_n_val, delta_t_val,
                       measured_ratio, naive_ratio, exact_ratio, rel))
        print(f"  {ratio:>9.4f} {ka_t:>10.0f} "
              f"{delta_n_val*1e6:>10.3f} {delta_t_val*1e6:>10.3f} "
              f"{measured_ratio:>10.4f} {naive_ratio:>10.4f} "
              f"{exact_ratio:>10.4f} {rel:>11.2e}")

    # Plot delta_t vs ka_t/ka.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ratios_arr = np.array([s[0] for s in series])
    delta_t_arr = np.array([s[3] for s in series]) * 1e6
    measured_ratio_arr = np.array([s[4] for s in series])
    naive_ratio_arr = np.array([s[5] for s in series])
    exact_ratio_arr = np.array([s[6] for s in series])

    ax1.plot(ratios_arr, delta_t_arr, "o-", color="C0", ms=8, lw=1.5,
             label=r"numerical $|\delta_t|$")
    ax1.set_xlabel(r"anchor anisotropy $k_{a,t} / k_a$")
    ax1.set_ylabel(r"tangent displacement $|\delta_t|$ [$\mu$m]")
    ax1.set_xscale("log")
    ax1.set_title("Soft tangent anchor -> larger $|\\delta_t|$")
    ax1.legend(loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(ratios_arr, measured_ratio_arr, "o", color="C0", ms=9,
             label=r"numerical $\delta_t/\delta_n$")
    ax2.plot(ratios_arr, exact_ratio_arr, "k-", lw=1.5,
             label=r"$(k_a - A)/(k_{a,t} - A)\,\tan\theta$ (exact)")
    ax2.plot(ratios_arr, naive_ratio_arr, "k:", lw=1.5,
             label=r"$(k_a/k_{a,t})\,\tan\theta$ (naive linear)")
    ax2.set_xlabel(r"anchor anisotropy $k_{a,t} / k_a$")
    ax2.set_ylabel(r"$\delta_t / \delta_n$ [-]")
    ax2.set_xscale("log")
    ax2.set_title(rf"Tilt ratio @ $\theta = {theta_deg:.0f}^\circ$")
    ax2.legend(loc="upper right")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "06a_off_axis_tilt.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    # Verify the EXACT formula across all ratios (the naive linear form
    # would fail at ratio = 0.1 by ~3-4% even at theta = 1 deg).
    delta_t_vals = [s[3] for s in series]
    monotone_ok = all(delta_t_vals[i] < delta_t_vals[i + 1]
                      for i in range(len(delta_t_vals) - 1))
    exact_ok = max(rel_errs_exact) < 1e-4
    ok = exact_ok and monotone_ok
    print(f"PART A result: {'PASS' if ok else 'FAIL'}  "
          f"(exact formula max rel err {max(rel_errs_exact):.2e}; "
          f"|delta_t| monotone-increasing as ka_t softens: "
          f"{'YES' if monotone_ok else 'NO'})")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Stick-slip threshold shift
# ─────────────────────────────────────────────────────────────────────────


def part_b_threshold_shift() -> bool:
    """Sweep ka_t_ratio; verify the empirical F_thresh tracks the closed form."""
    print()
    print("=" * 72)
    print("PART B.  Stick-slip threshold shifts with ka_t")
    print("=" * 72)

    ka = 25_000.0
    kc = 25_000.0
    phi = 1.0e-3
    k_stick = 25_000.0
    mu = 0.3
    ratios = [2.0, 1.0, 1.0/3.0, 0.1]

    print()
    print(f"  ka = {ka:.0f}, kc = {kc:.0f}, phi = {phi*1e3:.2f} mm, "
          f"k_stick = {k_stick:.0f}, mu = {mu}")
    print(f"  f_n = ka kc phi / (ka + kc) = "
          f"{ka*kc*phi/(ka+kc):.4f} N (unchanged across ratios)")
    print()
    print(f"  {'ka_t/ka':>9} {'F_thresh_th':>13} {'F_thresh_emp':>14} {'rel err':>10}")
    print(f"  {'-'*9} {'-'*13} {'-'*14} {'-'*10}")

    rel_errs = []
    rows = []
    for ratio in ratios:
        sphere = make_sphere(ka=ka, ka_t_ratio=ratio)
        target = make_face_on_target(sphere, phi_rest=phi)
        F_t_predicted, f_n, ka_t = F_thresh(sphere, kc, phi, k_stick, mu)

        # Empirical threshold: bisect on F to find where regime flips
        # from "stick" to "slip" using the analytical solver itself.
        # This is a tautology check against the analytical formula
        # (which we just changed), so we cross-validate against the
        # hard-piecewise scipy independent solver too.
        F_lo, F_hi = 0.0, 2.0 * F_t_predicted
        for _ in range(60):
            F_mid = 0.5 * (F_lo + F_hi)
            f_ext = np.array([F_mid, 0.0, 0.0])
            _, info_hard = equilibrium_with_friction_hard_numerical(
                sphere, target, kc, f_ext, k_stick, mu)
            if info_hard["regime"] == "stick":
                F_lo = F_mid
            else:
                F_hi = F_mid
        F_t_empirical = 0.5 * (F_lo + F_hi)
        rel = abs(F_t_empirical - F_t_predicted) / F_t_predicted
        rel_errs.append(rel)
        rows.append((ratio, F_t_predicted, F_t_empirical, rel))
        print(f"  {ratio:>9.4f} {F_t_predicted:>13.4f} {F_t_empirical:>14.4f} "
              f"{rel:>10.2e}")

    # Plot.  We illustrate by sweeping F and showing how the elbow
    # shifts across ratios.
    fig, ax = plt.subplots(figsize=(8, 5))
    F_max = 1.5 * max(F_thresh(make_sphere(ka=ka, ka_t_ratio=r), kc, phi,
                               k_stick, mu)[0]
                      for r in ratios)
    F_grid = np.linspace(0.0, F_max, 161)
    for ratio, color in zip(ratios, ["C2", "C0", "C1", "C3"]):
        sphere = make_sphere(ka=ka, ka_t_ratio=ratio)
        target = make_face_on_target(sphere, phi_rest=phi)
        F_fric = np.zeros_like(F_grid)
        for i, F in enumerate(F_grid):
            f_ext = np.array([F, 0.0, 0.0])
            _, info = equilibrium_with_friction_analytical(
                sphere, target, kc, f_ext, k_stick, mu)
            F_fric[i] = info["F_friction"]
        F_t_predicted = F_thresh(sphere, kc, phi, k_stick, mu)[0]
        ax.plot(F_grid, F_fric, "-", color=color, lw=1.5,
                label=rf"$k_{{a,t}}/k_a = {ratio:g}$,  "
                      rf"$F_{{\rm thresh}} = {F_t_predicted:.2f}$ N")
        ax.axvline(F_t_predicted, ls=":", color=color, lw=1, alpha=0.5)

    f_n = ka * kc * phi / (ka + kc)
    ax.axhline(mu * f_n, ls="--", color="gray",
               label=rf"$\mu f_n = {mu*f_n:.2f}$ N (Coulomb plateau)")
    ax.set_xlabel(r"external tangential force $F$ [N]")
    ax.set_ylabel(r"friction force $|F_{\rm friction}|$ [N]")
    ax.set_title("Anisotropic anchor: soft tangent shifts the slip threshold "
                 "(plateau unchanged)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "06b_stick_slip_threshold.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    ok = max(rel_errs) < 1e-6
    print(f"PART B result: {'PASS' if ok else 'FAIL'} "
          f"(max rel err empirical vs closed form = {max(rel_errs):.2e})")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Stick-slope verification
# ─────────────────────────────────────────────────────────────────────────


def part_c_stick_slope() -> bool:
    """Verify dF_friction/dF = k_stick / (ka_t + k_stick) in deep stick."""
    print()
    print("=" * 72)
    print("PART C.  Stick-mode slope vs ka_t_ratio")
    print("=" * 72)

    ka = 25_000.0
    kc = 25_000.0
    phi = 1.0e-3
    k_stick = 25_000.0
    mu = 0.3
    ratios = np.array([0.05, 0.1, 1.0/3.0, 1.0, 3.0, 10.0])

    print()
    print(f"  ka = {ka:.0f}, k_stick = {k_stick:.0f}")
    print()
    print(f"  {'ka_t/ka':>9} {'ka_t':>10} {'slope_th':>10} {'slope_meas':>12} "
          f"{'rel err':>10}")
    print(f"  {'-'*9} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    measured = np.zeros_like(ratios)
    predicted = np.zeros_like(ratios)
    rel_errs = np.zeros_like(ratios)

    for k, ratio in enumerate(ratios):
        sphere = make_sphere(ka=ka, ka_t_ratio=ratio)
        target = make_face_on_target(sphere, phi_rest=phi)
        F_t, _, ka_t = F_thresh(sphere, kc, phi, k_stick, mu)
        # Stay well below the threshold so we're in deep stick.
        F_samples = np.array([0.05, 0.10, 0.15]) * F_t
        F_fric_samples = []
        for F in F_samples:
            f_ext = np.array([F, 0.0, 0.0])
            _, info = equilibrium_with_friction_analytical(
                sphere, target, kc, f_ext, k_stick, mu)
            F_fric_samples.append(info["F_friction"])
        # Best-fit linear slope.
        slope_meas = np.polyfit(F_samples, F_fric_samples, 1)[0]
        slope_th = k_stick / (ka_t + k_stick)
        rel = abs(slope_meas - slope_th) / slope_th
        measured[k] = slope_meas
        predicted[k] = slope_th
        rel_errs[k] = rel
        print(f"  {ratio:>9.4f} {ka_t:>10.0f} {slope_th:>10.4f} "
              f"{slope_meas:>12.4f} {rel:>10.2e}")

    # Plot.
    fig, ax = plt.subplots(figsize=(7, 5))
    ratios_dense = np.logspace(np.log10(ratios.min() / 2),
                               np.log10(ratios.max() * 2), 200)
    ka_t_dense = ka * ratios_dense
    slope_dense = k_stick / (ka_t_dense + k_stick)
    ax.semilogx(ratios_dense, slope_dense, "k-", lw=1.5,
                label=r"$k_{\rm stick} / (k_{a,t} + k_{\rm stick})$ (closed form)")
    ax.semilogx(ratios, measured, "o", color="C0", ms=8,
                label="numerical slope (in stick regime)")
    ax.axhline(1.0, ls=":", color="gray", alpha=0.5, label="limit: all-F into friction")
    ax.axhline(0.0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel(r"$k_{a,t} / k_a$")
    ax.set_ylabel(r"stick-regime slope  $dF_{\rm friction} / dF$")
    ax.set_title("Soft tangent anchor -> more of the applied F flows into friction")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "06c_stick_slope.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    ok = float(np.max(rel_errs)) < 1e-6
    print(f"PART C result: {'PASS' if ok else 'FAIL'}  "
          f"(max rel err = {np.max(rel_errs):.2e})")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  Poisson sweep
# ─────────────────────────────────────────────────────────────────────────


def part_d_poisson_sweep() -> bool:
    """Sweep effective Poisson ratio nu -> ka_t_ratio = 1 / (2 (1 + nu)).

    Stops just shy of nu = 0.5 (where ka_t_ratio would be 1/3 exactly,
    matching the 'incompressible flesh' rule).  Shows the pre-slip s vs F
    curves and where each ratio's F_thresh sits.
    """
    print()
    print("=" * 72)
    print("PART D.  Poisson sweep: nu -> ka_t_ratio = 1/(2(1+nu))")
    print("=" * 72)

    ka = 25_000.0
    kc = 25_000.0
    phi = 1.0e-3
    k_stick = 25_000.0
    mu = 0.3
    nus = [0.0, 0.25, 0.4, 0.5]

    print()
    print(f"  {'nu':>5} {'ka_t/ka':>10} {'F_thresh':>11} {'note':<40}")
    for nu in nus:
        ratio = 1.0 / (2.0 * (1.0 + nu))
        sphere = make_sphere(ka=ka, ka_t_ratio=ratio)
        F_t, f_n, ka_t = F_thresh(sphere, kc, phi, k_stick, mu)
        note = ("isotropic"        if nu == 0.0
                else "intermediate"  if nu == 0.25
                else "near incompressible" if nu == 0.4
                else "incompressible flesh (ka_t = ka/3)")
        print(f"  {nu:>5.2f} {ratio:>10.4f} {F_t:>11.4f} {note:<40}")

    # Plot s vs F across all four nus.  We use F_max from the smallest
    # ratio so all curves complete their stick-to-slip transition on the
    # plot.
    sphere_min = make_sphere(ka=ka, ka_t_ratio=1.0/(2*(1+max(nus))))
    F_t_max = F_thresh(sphere_min, kc, phi, k_stick, mu)[0]
    F_grid = np.linspace(0.0, 1.6 * F_t_max, 161)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["C2", "C0", "C1", "C3"]
    for nu, color in zip(nus, colors):
        ratio = 1.0 / (2.0 * (1.0 + nu))
        sphere = make_sphere(ka=ka, ka_t_ratio=ratio)
        target = make_face_on_target(sphere, phi_rest=phi)
        s_vals = np.zeros_like(F_grid)
        F_fric = np.zeros_like(F_grid)
        for i, F in enumerate(F_grid):
            f_ext = np.array([F, 0.0, 0.0])
            _, info = equilibrium_with_friction_analytical(
                sphere, target, kc, f_ext, k_stick, mu)
            s_vals[i] = info["s"]
            F_fric[i] = info["F_friction"]
        F_t = F_thresh(sphere, kc, phi, k_stick, mu)[0]
        ax1.plot(F_grid, s_vals * 1e6, "-", color=color, lw=1.5,
                 label=rf"$\nu = {nu:.2f}$ ($k_{{a,t}}/k_a = {ratio:.3f}$)")
        ax1.axvline(F_t, ls=":", color=color, alpha=0.5)
        ax2.plot(F_grid, F_fric, "-", color=color, lw=1.5,
                 label=rf"$\nu = {nu:.2f}$")
        ax2.axvline(F_t, ls=":", color=color, alpha=0.5)

    ax1.set_xlabel(r"external tangential force $F$ [N]")
    ax1.set_ylabel(r"tangential displacement $|\delta_t|$ [$\mu$m]")
    ax1.set_title(r"Pre-slip slope steepens as $\nu \to 0.5$")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    f_n = ka * kc * phi / (ka + kc)
    ax2.axhline(mu * f_n, ls="--", color="gray",
                label=rf"$\mu f_n = {mu*f_n:.2f}$ N")
    ax2.set_xlabel(r"external tangential force $F$ [N]")
    ax2.set_ylabel(r"friction force $|F_{\rm friction}|$ [N]")
    ax2.set_title(r"$F_{\rm thresh}$ drops as $\nu \to 0.5$ (slips earlier)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "06d_poisson_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")
    print(f"PART D result: PASS (visual)")
    return True


# ─────────────────────────────────────────────────────────────────────────
#  Sign regression guard (same discipline as step 4)
# ─────────────────────────────────────────────────────────────────────────


def sign_regression_guard() -> bool:
    """Cross-check sign of delta_t across all three friction solvers at
    ka_t_ratio = 1/3 (incompressible flesh).  Magnitude-only checks
    would miss a sign flip; the regression guard is the discipline.
    """
    print()
    print("=" * 72)
    print("Sign-regression guard:  delta_t direction across solvers, ka_t/ka = 1/3")
    print("=" * 72)
    sphere = make_sphere(ka=25_000.0, ka_t_ratio=1.0/3.0)
    target = make_face_on_target(sphere, phi_rest=1e-3)
    kc, k_stick, mu = 25_000.0, 25_000.0, 0.3
    f_ext = np.array([1.0, 0.0, 0.0])
    d_a, _ = equilibrium_with_friction_analytical(
        sphere, target, kc, f_ext, k_stick, mu)
    d_h, _ = equilibrium_with_friction_hard_numerical(
        sphere, target, kc, f_ext, k_stick, mu)
    d_s, _ = equilibrium_with_friction_smooth_numerical(
        sphere, target, kc, f_ext, k_stick, mu)
    print(f"  analytical δ_x = {d_a[0]:+.3e}  (expect NEGATIVE)")
    print(f"  hard num   δ_x = {d_h[0]:+.3e}")
    print(f"  smooth num δ_x = {d_s[0]:+.3e}")
    ok = d_a[0] < 0 and d_h[0] < 0 and d_s[0] < 0
    print(f"  guard: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = part_a_off_axis_tilt()
    b = part_b_threshold_shift()
    c = part_c_stick_slope()
    part_d_poisson_sweep()
    g = sign_regression_guard()
    all_ok = a and b and c and g
    print()
    print("=" * 72)
    print(f"Step 6 anisotropic-anchor test: {'PASS' if all_ok else 'FAIL'}   "
          f"(A={'P' if a else 'F'} B={'P' if b else 'F'} C={'P' if c else 'F'} "
          f"sign-guard={'P' if g else 'F'})")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
