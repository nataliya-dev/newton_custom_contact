# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 / T-C — Face-on series-spring law for one pad sphere vs flat face.

The v2 successor to ``test_01_single_sphere.py``.  Single isolated
lattice sphere pressed against ONE flat target face element
(``PointSetTarget`` reduced to a single (position, normal, area)
triple — implemented here as the bare half-space ``n_face``,
``t_sample``).  No neighbours; no lateral spring.  All we verify is

    (anchor + half-space contact)  →  series spring equilibrium

with the deformed-centre formulation and half-space overlap

    raw  =  r - n_face · (q - t_sample),   q = p - delta.

Closed form (contract_v2.md §6, eq:1sphere-delta-equivalent):

    d         =  r - n_face · (p - t_sample)         (rest overlap, ≥ 0)
    delta_n*  =  k_c · d / (k_a + k_c)
    F*        =  k_a · k_c · d / (k_a + k_c)  =  k_eff · d

Four parts:

  PART A.  Closed form vs L-BFGS-B numerical equilibrium across a
           15-row sweep of (d, k_c) pairs.  Both should match to
           absolute ``2e-10 m`` on delta and ``1e-13 N`` on force
           (contract §12 T-C).  Uses ``eps = 1e-12`` (analytic-grade).
  PART B.  ``F*`` vs ``k_c`` at fixed ``d``.  Saturates at
           ``k_a · d`` as ``k_c → ∞`` (rigid contact limit); vanishes
           as ``k_c → 0``.
  PART C.  ``F*`` vs ``d`` at fixed ``k_c``.  Linear with slope
           ``k_eff = k_a k_c / (k_a + k_c)``.
  PART D.  **Production-eps precision floor.**  Re-runs the face-on
           equilibrium at ``eps = 5e-4 m`` (the production smoothing
           width used by the Phase 2+ lattice solver) and bounds the
           force precision loss as a function of the equilibrium
           overlap ``raw_eq / eps``.  Establishes a quantified Phase 2
           tolerance budget: contacts with ``raw_eq ≥ 10·eps`` agree
           with analytical to <1% rel err.

The v1 sphere-vs-sphere test_01 covered the same physics with the
(r + R) - ||q - t|| form.  This test exercises the same closed form
under the v2 half-space form (drop R, replace ``||q - t||`` with
``n_face · (q - t)``).  At face-on the two forms agree exactly because
the line-of-centres coincides with the face normal — but the v2 form
extends cleanly to tilted faces (T-D) and is well-defined at any
penetration depth (T-A, T-B).

Per ``cslc_main/theory/contract_v2.md`` §6, §12 T-C.

Run::

    uv run -m cslc_main.theory.test_face_on_series_spring

Outputs::

    cslc_main/theory/figures/tc_face_on_table.txt
    cslc_main/theory/figures/tc_series_spring_kc.png
    cslc_main/theory/figures/tc_force_vs_d.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    equilibrium_half_space_face_on_analytical,
    equilibrium_half_space_numerical,
    half_space_raw,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Contract §12 T-C tolerances.
TOL_DELTA_ABS = 2.0e-10   # [m]
TOL_FORCE_ABS = 1.0e-13   # [N]


def make_scene(*, d_rest: float = 1.0e-3, r: float = 2.5e-3,
               ka: float = 25_000.0,
               n: np.ndarray | None = None,
               ) -> tuple[LatticeSphere, np.ndarray, np.ndarray]:
    """Build a (sphere, n_face, t_sample) triple at face-on contact.

    Sphere at the origin with outward normal ``n`` (default +x).
    Face normal ``n_face = -n`` (face-on).  Sample position placed so
    ``half_space_raw(sphere, n_face, t_sample, 0) == d_rest`` exactly.

    Returns ``(LatticeSphere, n_face, t_sample)``.
    """
    if n is None:
        n = np.array([1.0, 0.0, 0.0])
    sphere = LatticeSphere(p=np.zeros(3), r=r, n=n, ka=ka)
    n_face = -n
    # Solve: r - n_face · (p - t_sample) = d_rest
    #     ⇒  -n_face · (-t_sample)        = d_rest - r
    #     ⇒   t_sample                    = (r - d_rest) · (-n_face) = (r - d_rest) · n
    t_sample = (r - d_rest) * n
    # Sanity.
    raw0 = half_space_raw(sphere, n_face, t_sample, np.zeros(3))
    assert np.isclose(raw0, d_rest, atol=1e-12), \
        f"setup error: rest raw = {raw0}, expected {d_rest}"
    return sphere, n_face, t_sample


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Closed form vs numerical, 15-row table
# ─────────────────────────────────────────────────────────────────────────


def part_a_table() -> bool:
    print()
    print("=" * 78)
    print("PART A.  Face-on equilibrium: analytical vs numerical")
    print("=" * 78)
    print()
    print("Setup: 1 lattice sphere, ka = 25,000 N/m, n_pad = +x,")
    print("       n_face = -x, rest overlap d as below.")
    print()

    ka = 25_000.0
    d_values = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]    # 0.1 mm .. 5 mm
    kc_values = [2_500.0, 25_000.0, 250_000.0]    # k_c/k_a = 0.1, 1, 10

    header = (
        f"{'d[mm]':>8} {'kc[N/m]':>11} "
        f"{'δ_ana[μm]':>11} {'δ_num[μm]':>11} "
        f"{'F_ana[N]':>10} {'F_num[N]':>10} "
        f"{'|Δδ|[m]':>11} {'|ΔF|[N]':>11}"
    )
    print(header)
    print("-" * len(header))

    lines = [header, "-" * len(header)]
    delta_max_err = 0.0
    F_max_err = 0.0

    for d in d_values:
        for kc in kc_values:
            sphere, n_face, t_sample = make_scene(d_rest=d, ka=ka)
            delta_ana, F_ana = equilibrium_half_space_face_on_analytical(
                sphere, n_face, t_sample, kc)
            delta_num, info = equilibrium_half_space_numerical(
                sphere, n_face, t_sample, kc,
                eps=1.0e-12, delta0=delta_ana, tol=1.0e-14)
            # Force from numerical: at equilibrium |f_anchor| = |f_contact|
            # = ka * |delta|.
            F_num = ka * float(np.linalg.norm(delta_num))
            delta_err = float(np.linalg.norm(delta_ana - delta_num))
            F_err = abs(F_ana - F_num)
            delta_max_err = max(delta_max_err, delta_err)
            F_max_err = max(F_max_err, F_err)
            row = (
                f"{d * 1e3:>8.3f} {kc:>11.0f} "
                f"{1e6 * delta_ana[0]:>11.6f} {1e6 * delta_num[0]:>11.6f} "
                f"{F_ana:>10.4f} {F_num:>10.4f} "
                f"{delta_err:>11.2e} {F_err:>11.2e}"
            )
            print(row)
            lines.append(row)

    delta_ok = delta_max_err < TOL_DELTA_ABS
    F_ok = F_max_err < TOL_FORCE_ABS
    all_ok = delta_ok and F_ok

    print()
    print(f"PART A result: {'PASS' if all_ok else 'FAIL'}")
    print(f"  worst |Δδ|  = {delta_max_err:.3e} m  (tol {TOL_DELTA_ABS:.0e})  "
          f"{'PASS' if delta_ok else 'FAIL'}")
    print(f"  worst |ΔF|  = {F_max_err:.3e} N  (tol {TOL_FORCE_ABS:.0e})  "
          f"{'PASS' if F_ok else 'FAIL'}")

    out = FIG_DIR / "tc_face_on_table.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved table to {out}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Series-spring saturation: sweep kc
# ─────────────────────────────────────────────────────────────────────────


def part_b_kc_sweep() -> bool:
    """Series-spring saturation in k_c.

    This part verifies the SHAPE of the F(kc) curve:
      * Saturates at  k_a · d  as kc → ∞ (rigid-contact limit).
      * Vanishes as kc → 0.
      * F(kc = k_a) = 0.5 · k_a · d  (equal-stiffness midpoint).

    The numerical-vs-analytical check is in Part A (where it can hit
    1e-13 N at canonical kc).  Part B sweeps kc 6 decades to visualise
    saturation; at the high-kc end (kc / k_a > 1000) δ_n is within
    O(d · k_a / kc) of its rigid limit ``d``, and any L-BFGS-B step
    of size <= float64 epsilon on δ amplifies to fp-noise force error
    via ``ka · ε_machine · |δ| ~ 1e-12 N`` — orders of magnitude
    smaller than the saturation magnitude k_a · d = 25 N but above the
    Part-A tolerance.  So Part B asserts asymptotics only.
    """
    print()
    print("=" * 78)
    print("PART B.  Series-spring saturation: sweep k_c")
    print("=" * 78)

    ka = 25_000.0
    d = 1.0e-3   # 1 mm rest overlap

    kc_dense = np.logspace(2, 9, 64)
    F_ana = np.zeros_like(kc_dense)
    delta_ana = np.zeros_like(kc_dense)
    for i, kc in enumerate(kc_dense):
        sphere, n_face, t_sample = make_scene(d_rest=d, ka=ka)
        delta, F = equilibrium_half_space_face_on_analytical(
            sphere, n_face, t_sample, kc)
        F_ana[i] = F
        delta_ana[i] = delta[0]

    # Spot-check numerical agreement at MODERATE kc (kc/ka in [0.01, 100]).
    # Avoids extreme kc where L-BFGS-B can perturb δ by fp epsilon and
    # ka·ε amplifies to a 1e-12 N force noise (acknowledged in Part A
    # gate; Part B is about shape, not numerical precision).
    kc_num = np.array([100.0, 1_000.0, 10_000.0, ka,
                       100_000.0, 1_000_000.0, 2_500_000.0])
    F_num = np.zeros_like(kc_num)
    delta_num_x = np.zeros_like(kc_num)
    for i, kc in enumerate(kc_num):
        sphere, n_face, t_sample = make_scene(d_rest=d, ka=ka)
        delta_a, _ = equilibrium_half_space_face_on_analytical(
            sphere, n_face, t_sample, kc)
        delta, info = equilibrium_half_space_numerical(
            sphere, n_face, t_sample, kc,
            eps=1.0e-12, delta0=delta_a, tol=1.0e-14)
        F_num[i] = ka * float(np.linalg.norm(delta))
        delta_num_x[i] = delta[0]

    # Asymptote checks.
    F_kc_to_inf = ka * d              # rigid-contact limit
    F_kc_eq_ka = 0.5 * ka * d         # equal-stiffness midpoint
    # F at the largest kc we swept (1e9) should match the rigid limit
    # to relative O(ka/kc) ≈ 2.5e-5.
    rigid_err = abs(F_ana[-1] - F_kc_to_inf)
    rigid_expected = F_kc_to_inf * ka / kc_dense[-1]
    asym_high_ok = rigid_err < 3.0 * rigid_expected
    print(f"  asymptote F(kc → ∞) = ka·d = {F_kc_to_inf:.4f} N  "
          f"vs F_ana(kc=1e9) = {F_ana[-1]:.4f} N  "
          f"(err {rigid_err:.3e}, expected ~{rigid_expected:.3e})  "
          f"{'PASS' if asym_high_ok else 'FAIL'}")
    # F at the smallest kc (1e2) → kc·d (linear at low kc):
    asym_low_ok = F_ana[0] < 0.05 * F_kc_to_inf
    print(f"  low-kc limit F(kc=100) = {F_ana[0]:.4f} N  "
          f"(must be << F_max = {F_kc_to_inf:.4f} N)  "
          f"{'PASS' if asym_low_ok else 'FAIL'}")
    # Midpoint check.
    j_eq = int(np.argmin(np.abs(kc_num - ka)))
    midpoint_err = abs(F_num[j_eq] - F_kc_eq_ka)
    midpoint_ok = midpoint_err < TOL_FORCE_ABS
    print(f"  midpoint F(kc=ka)   = ka·d/2 = {F_kc_eq_ka:.4f} N  "
          f"vs F_num = {F_num[j_eq]:.4f} N  "
          f"(err {midpoint_err:.3e}, tol {TOL_FORCE_ABS:.0e})  "
          f"{'PASS' if midpoint_ok else 'FAIL'}")
    # Moderate-kc numerical agreement (kc/ka ≤ 100; Part-A grade).
    moderate_err = float(np.max(np.abs(F_num - np.array([
        equilibrium_half_space_face_on_analytical(
            *make_scene(d_rest=d, ka=ka), kc=kc_v)[1]
        for kc_v in kc_num]))))
    moderate_ok = moderate_err < TOL_FORCE_ABS
    print(f"  worst |F_num - F_ana| on moderate-kc grid = {moderate_err:.3e} N  "
          f"(tol {TOL_FORCE_ABS:.0e})  "
          f"{'PASS' if moderate_ok else 'FAIL'}")
    ok = asym_high_ok and asym_low_ok and midpoint_ok and moderate_ok

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogx(kc_dense, F_ana * 1e3, "k-", lw=2, label="ideal (series spring)")
    ax1.semilogx(kc_num, F_num * 1e3, "ro", ms=6, mfc="none", label="numerical")
    ax1.axhline(F_kc_to_inf * 1e3, ls="--", color="grey",
                label=r"$k_a\,d$ (rigid limit)")
    ax1.axvline(ka, ls=":", color="C0", label=r"$k_c = k_a$")
    ax1.set_xlabel(r"$k_c$ [N/m]")
    ax1.set_ylabel(r"equilibrium force $|F^\ast|$ [mN]")
    ax1.set_title("T-C/B. Series-spring saturation\n"
                  rf"half-space face-on, $d = {d * 1e3:.1f}$ mm, "
                  rf"$k_a = {ka:.0f}$ N/m")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(kc_dense, delta_ana * 1e6, "k-", lw=2, label="ideal")
    ax2.semilogx(kc_num, delta_num_x * 1e6, "ro", ms=6, mfc="none",
                 label="numerical")
    ax2.axhline(d * 1e6, ls="--", color="grey", label=r"$d$ (rigid limit)")
    ax2.axvline(ka, ls=":", color="C0", label=r"$k_c = k_a$")
    ax2.set_xlabel(r"$k_c$ [N/m]")
    ax2.set_ylabel(r"equilibrium $\delta_n^\ast$ [$\mu$m]")
    ax2.set_title(r"Compression of the compliant skin")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "tc_series_spring_kc.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  F* linear in d at fixed kc
# ─────────────────────────────────────────────────────────────────────────


def part_c_d_sweep() -> bool:
    print()
    print("=" * 78)
    print("PART C.  Linearity: F* vs d at fixed k_a, k_c")
    print("=" * 78)

    ka = 25_000.0
    kcs = [2_500.0, 25_000.0, 250_000.0]
    d_grid = np.linspace(0.0, 3.0e-3, 24)

    fig, ax = plt.subplots(figsize=(7, 5))
    all_ok = True
    for kc, color in zip(kcs, ["C0", "C1", "C2"]):
        keff = ka * kc / (ka + kc)
        F = np.zeros_like(d_grid)
        for i, d in enumerate(d_grid):
            sphere, n_face, t_sample = make_scene(d_rest=d, ka=ka)
            _, F[i] = equilibrium_half_space_face_on_analytical(
                sphere, n_face, t_sample, kc)
        ax.plot(d_grid * 1e3, F * 1e3, "-", color=color, lw=2,
                label=rf"$k_c={kc:.0f}$, $k_{{eff}}={keff:.0f}$")
        ax.plot(d_grid * 1e3, keff * d_grid * 1e3, ":", color=color, lw=1)
        worst = float(np.max(np.abs(F - keff * d_grid)))
        ok = worst < TOL_FORCE_ABS
        all_ok = all_ok and ok
        print(f"  k_c = {kc:>7.0f} N/m   k_eff = {keff:>8.2f} N/m  "
              f"max |F - k_eff·d| = {worst:.3e} N  "
              f"{'PASS' if ok else 'FAIL'}")

    ax.set_xlabel(r"$d$ (rest half-space overlap) [mm]")
    ax.set_ylabel(r"equilibrium force $|F^\ast|$ [mN]")
    ax.set_title(
        rf"T-C/C. Linearity check: $F^\ast = k_{{eff}}\,d$  "
        rf"($k_a = {ka:.0f}$ N/m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tc_force_vs_d.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  Production-eps precision floor (Phase 2 budget reference)
# ─────────────────────────────────────────────────────────────────────────

# Production smoothing width [m] — matches contract §3 default
# and EPS_PRODUCTION used by T-B.
EPS_PRODUCTION = 5.0e-4


def part_d_production_eps_floor() -> bool:
    """Quantify how much precision is lost when switching from
    analytic-grade ``eps = 1e-12`` (used by Parts A-C) to production
    ``eps = 5e-4 m`` (used by the lattice solver in Phase 2+).

    Why this matters.  Parts A-C report ~1e-13 N agreement with the
    analytical series spring, but they're run at ``eps = 1e-12``
    specifically to get into the deep-saturated regime where
    ``phi_eff ≈ raw`` and ``gate ≈ 1``.  Phase 2 will run at
    production eps where these surrogates differ from raw by
    ``O(eps²/raw)``, which shifts the equilibrium.  This part
    establishes the precision FLOOR every Phase 2 test inherits,
    so downstream tolerances can be set knowingly instead of by
    trial and error.

    The shift has a predictable structure: equilibrium raw at depth
    ``d`` is ``raw_eq = d · k_a / (k_a + k_c)``, so the relative
    smoothing perturbation is governed by ``R = raw_eq / eps``.

    Empirically the rel err on F scales as ``~ 1/R⁴`` (not the naive
    ``1/R²``).  Reason: ``σ_ε`` overshoots raw by ``+ε²/(4·raw)``
    while ``Σ_ε`` undershoots 1 by ``-ε²/(4·raw²)``, and their
    PRODUCT cancels both first-order corrections, leaving only
    ``O(ε⁴/raw⁴)``.  Measured at production eps:

        R = raw_eq / eps |  rel err on F (measured)
        ──────────────── | ──────────────────────────
              1          |  ~1.6%
              5          |  ~5e-5
             10          |  ~3e-6
             25          |  ~8e-8
             50          |  ~9e-9
            100          |  ~5e-10

    So the actual precision floor is much better than the surface
    "phi_eff − raw ≈ eps/2" reading from T-B suggests, because the
    Σ_ε gate factor self-compensates.

    Assertion: at ``raw_eq ≥ 10·eps`` (the "well-saturated" regime
    the Jacobi solver should operate in for production contacts),
    force matches analytical to better than 1% rel err.  (In practice
    we beat this by ~5 orders of magnitude.)
    """
    print()
    print("=" * 78)
    print("PART D.  Production-eps precision floor (Phase 2 inherit)")
    print("=" * 78)
    print()
    print(f"  eps_production = {EPS_PRODUCTION:.0e} m,  ka = 25,000 N/m,")
    print(f"  kc = 25,000 N/m (k_c/k_a = 1, raw_eq = d/2)")
    print()

    ka = 25_000.0
    kc = 25_000.0   # raw_eq = d/2
    # Sweep d so raw_eq / eps spans {1, 5, 10, 25, 50, 100}.
    # raw_eq = d · k_a / (k_a + k_c) = d/2 at k_c = k_a, so:
    target_ratios = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    d_values = [r * EPS_PRODUCTION * 2.0 for r in target_ratios]

    header = (
        f"  {'d[mm]':>8} {'raw_eq/eps':>11} "
        f"{'F_ana[N]':>10} {'F_num[N]':>10} {'rel err':>12}  result"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    rel_errs = []
    ratios_actual = []
    forces_ana = []
    forces_num = []
    # Production-grade well-saturated bound (assertion).
    BOUND_WELL_SATURATED = 0.01   # 1% at raw_eq >= 10·eps
    bound_ok = True

    for d, ratio in zip(d_values, target_ratios):
        sphere, n_face, t_sample = make_scene(d_rest=d, ka=ka)
        delta_ana, F_ana = equilibrium_half_space_face_on_analytical(
            sphere, n_face, t_sample, kc)
        delta_num, info = equilibrium_half_space_numerical(
            sphere, n_face, t_sample, kc,
            eps=EPS_PRODUCTION, delta0=delta_ana, tol=1.0e-14)
        F_num = ka * float(np.linalg.norm(delta_num))
        rel = abs(F_num - F_ana) / max(F_ana, 1.0e-30)
        rel_errs.append(rel)
        ratios_actual.append(ratio)
        forces_ana.append(F_ana)
        forces_num.append(F_num)

        if ratio >= 10.0:
            ok = rel < BOUND_WELL_SATURATED
            bound_ok = bound_ok and ok
            verdict = "PASS" if ok else "FAIL"
        else:
            verdict = "(survey)"   # below the assertion gate
        print(f"  {d*1e3:>8.3f} {ratio:>11.1f} "
              f"{F_ana:>10.4f} {F_num:>10.4f} {rel:>12.3e}  {verdict}")

    print()
    print(f"  Assertion: rel err < {BOUND_WELL_SATURATED:.0%} for "
          f"raw_eq ≥ 10·eps  →  {'PASS' if bound_ok else 'FAIL'}")
    print()
    print("  Phase 2 budget: any contact at raw_eq < 10·eps "
          f"(~{10*EPS_PRODUCTION*1e3:.1f} mm of equilibrium half-space")
    print("  overlap) inherits surrogate-dominated precision; design "
          "the lattice and")
    print("  pad spacing so production contacts sit comfortably above "
          "this threshold.")

    # ── Figure: rel err vs raw_eq/eps on log-log ──
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(ratios_actual, np.maximum(rel_errs, 1.0e-12),
              "ko-", lw=1.5, ms=7, label="numerical (production eps)")
    # Theoretical leading-order: sigma_eps(x) ≈ x + eps²/(4x) and
    # Sigma_eps(x) ≈ 1 - eps²/(4x²); their PRODUCT cancels at order
    # eps²/x and leaves only O(eps⁴/x⁴) = O(1/R⁴) where R = raw_eq/eps.
    # So the rel err on F scales as 1/R⁴ (much better than naive 1/R²
    # — gate compensates phi_eff to leading order).
    rr = np.array(ratios_actual, dtype=float)
    ax.loglog(rr, 1.0 / (rr ** 4), "C3--", lw=1.2,
              label=r"$\sim 1/(\mathrm{raw}_{eq}/\varepsilon)^4$ "
              r"($\sigma_\varepsilon$, $\Sigma_\varepsilon$ corrections cancel)")
    ax.axvline(10.0, color="C2", ls=":", lw=1.0,
               label=r"assertion gate: $\mathrm{raw}_{eq} \geq 10\,\varepsilon$")
    ax.axhline(BOUND_WELL_SATURATED, color="C2", ls=":", lw=1.0,
               label=f"{BOUND_WELL_SATURATED:.0%} bound")
    ax.set_xlabel(r"$\mathrm{raw}_{eq} / \varepsilon$ "
                  r"(equilibrium half-space overlap, eps-multiples)")
    ax.set_ylabel(r"$|F_{num} - F_{ana}| / F_{ana}$")
    ax.set_title("T-C/D. Production-eps precision floor "
                 r"($\varepsilon = 5\!\times\!10^{-4}$ m)" "\n"
                 "Phase 2 inherits this floor — design contacts to land "
                 "right of the gate.")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG_DIR / "tc_production_eps_floor.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved figure to {out}")

    return bound_ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a_ok = part_a_table()
    b_ok = part_b_kc_sweep()
    c_ok = part_c_d_sweep()
    d_ok = part_d_production_eps_floor()
    print()
    print("=" * 78)
    print(f"T-C overall: {'PASS' if (a_ok and b_ok and c_ok and d_ok) else 'FAIL'}")
    print("=" * 78)
    return 0 if (a_ok and b_ok and c_ok and d_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
