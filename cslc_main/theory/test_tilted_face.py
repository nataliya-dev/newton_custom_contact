# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 / T-D — Anisotropic anchor under tilted flat face.

The v2 successor to ``test_06_anisotropic_anchor.py``.  Single pad
sphere pressed against a flat face whose outward normal makes angle
``θ`` with the pad's outward normal.  Anchor is per-axis anisotropic:
normal stiffness ``k_a``, tangent stiffness ``k_a · ρ``.

**Closed-form tilt ratio (v2, exact).**  At equilibrium in the pad's
local ``{n̂_pad, t̂_pad}`` frame, with face normal
``n̂_face = -cos θ · n̂_pad + sin θ · t̂_pad`` and the half-space
overlap ``raw = r − n̂_face · (q − t_sample)``::

    ka · δ_n  =  k_c · φ_eff · gate · cos θ            (n axis)
    ka_t · δ_t  =  −k_c · φ_eff · gate · sin θ          (t axis)

    ⇒  |δ_t| / |δ_n|  =  (1 / ρ) tan θ                   (eq:v2-tilt)

EXACT, with no ``A = k_c φ_eff / L`` correction.  The v1 sphere-vs-
sphere formula carried the ``A`` term because its contact direction
``(q − t) / ||q − t||`` rotated with δ; v2's face normal is fixed,
so its derivative w.r.t. δ vanishes and the correction goes away.

Empirically the v1 formula is wrong by ~3% at ρ = 1/3 and up to ~21%
at ρ = 0.1; the v2 formula matches numerical to ~1e-9 rel err.  This
test pins down the v2 formula across ρ ∈ {1, 1/2, 1/3, 0.1} and
θ ∈ {5°, 15°, 30°, 45°}.

Three parts:

  PART A.  Sweep ρ at fixed θ = 30°.  Verify v2 formula to 1e-7 rel err.
  PART B.  Sweep θ ∈ {5°, 15°, 30°, 45°} at fixed ρ = 1/3.  Verify
           the formula's tan(θ) dependence.
  PART C.  Side-by-side: v2 formula vs v1 formula (with A) across
           the same sweep.  v1 systematic deviation grows with
           anisotropy — the deviation IS the regression value of v2.

Anisotropic anchor + friction (test_06 PARTs B–D) moves to Phase 3 /
T-I + T-J under the unified flat-face friction solver.

Per ``cslc_main/theory/contract_v2.md`` §6, §12 T-D.

Run::

    uv run -m cslc_main.theory.test_tilted_face

Outputs::

    cslc_main/theory/figures/td_tilt_vs_rho.png
    cslc_main/theory/figures/td_tilt_vs_theta.png
    cslc_main/theory/figures/td_v1_vs_v2.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    equilibrium_half_space_numerical,
    half_space_raw,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"

TOL_REL = 1.0e-7   # contract §12 T-D


def build_tilted_scene(theta_deg: float, rho: float = 1.0,
                       d_rest: float = 1.0e-3, r: float = 2.5e-3,
                       ka: float = 25_000.0
                       ) -> tuple[LatticeSphere, np.ndarray, np.ndarray]:
    """Build (sphere, n_face, t_sample) for face tilted by theta_deg.

    Pad normal n_pad = +x; tangent t_pad = +y.  Face normal
    ``n_face = -cos θ · n_pad + sin θ · t_pad``.  Sample placed so the
    rest half-space overlap is exactly ``d_rest``.
    """
    theta = np.radians(theta_deg)
    n_pad = np.array([1.0, 0.0, 0.0])
    t_pad = np.array([0.0, 1.0, 0.0])
    n_face = -np.cos(theta) * n_pad + np.sin(theta) * t_pad
    # Place t_sample so ``r - n_face · (p - t_sample) = d_rest``.  Taking
    # p = 0 and choosing t_sample along -n_face gives
    # ``n_face · t_sample = -||t_sample||``, so ``t_sample = (r - d_rest) · (-n_face)``.
    sphere = LatticeSphere(p=np.zeros(3), r=r, n=n_pad, ka=ka, ka_t_ratio=rho)
    t_sample = (r - d_rest) * (-n_face)
    raw0 = half_space_raw(sphere, n_face, t_sample, np.zeros(3))
    assert np.isclose(raw0, d_rest, atol=1e-12), \
        f"setup error: raw0 = {raw0}, expected {d_rest}"
    return sphere, n_face, t_sample


def solve_and_decompose(sphere: LatticeSphere, n_face: np.ndarray,
                        t_sample: np.ndarray, kc: float
                        ) -> tuple[float, float, float, dict]:
    """Run the equilibrium and return (|δ_n|, |δ_t|, ratio, info).

    δ_n = δ · n_pad ; δ_t = || δ − δ_n n_pad ||.
    """
    delta, info = equilibrium_half_space_numerical(
        sphere, n_face, t_sample, kc, eps=1.0e-12, tol=1.0e-14)
    delta_n = float(np.dot(delta, sphere.n))
    delta_t_vec = delta - delta_n * sphere.n
    delta_t = float(np.linalg.norm(delta_t_vec))
    return abs(delta_n), delta_t, delta_t / max(abs(delta_n), 1e-30), info


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Sweep ρ at fixed θ = 30°
# ─────────────────────────────────────────────────────────────────────────


def part_a_rho_sweep() -> bool:
    print()
    print("=" * 78)
    print("PART A.  Sweep ρ at θ = 30°  —  v2 tilt formula |δ_t/δ_n| = tan(θ) / ρ")
    print("=" * 78)
    print()
    print(f"  ka = 25,000 N/m, kc = 25,000 N/m, d_rest = 1.0 mm, "
          f"θ = 30° (tan θ = {np.tan(np.radians(30.0)):.6f})")
    print()

    ka = 25_000.0
    kc = 25_000.0
    theta_deg = 30.0
    theta = np.radians(theta_deg)

    rhos = [1.0, 0.5, 1.0/3.0, 0.1]
    print(f"  {'ρ':>8} {'|δ_n|[μm]':>12} {'|δ_t|[μm]':>12} "
          f"{'|δ_t/δ_n|':>14} {'predicted':>12} {'rel err':>11}")
    print("  " + "-" * 80)

    all_ok = True
    for rho in rhos:
        sphere, n_face, t_sample = build_tilted_scene(theta_deg, rho=rho, ka=ka)
        dn, dt, ratio, info = solve_and_decompose(sphere, n_face, t_sample, kc)
        predicted = np.tan(theta) / rho
        rel = abs(ratio - predicted) / predicted
        ok = rel < TOL_REL
        all_ok = all_ok and ok
        print(f"  {rho:>8.4f} {dn*1e6:>12.4f} {dt*1e6:>12.4f} "
              f"{ratio:>14.6f} {predicted:>12.6f} {rel:>11.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

    print()
    print(f"PART A: {'PASS' if all_ok else 'FAIL'}  (tol rel < {TOL_REL:.0e})")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Sweep θ at fixed ρ = 1/3
# ─────────────────────────────────────────────────────────────────────────


def part_b_theta_sweep() -> bool:
    print()
    print("=" * 78)
    print("PART B.  Sweep θ at ρ = 1/3  —  verify tan(θ) dependence")
    print("=" * 78)
    print()

    ka = 25_000.0
    kc = 25_000.0
    rho = 1.0 / 3.0
    thetas_deg = [5.0, 15.0, 30.0, 45.0]
    print(f"  ka = {ka:.0f}, kc = {kc:.0f}, d_rest = 1.0 mm, ρ = 1/3")
    print()
    print(f"  {'θ [°]':>8} {'tan θ':>10} {'|δ_n|[μm]':>12} {'|δ_t|[μm]':>12} "
          f"{'|δ_t/δ_n|':>14} {'tan(θ)/ρ':>12} {'rel err':>11}")
    print("  " + "-" * 90)

    all_ok = True
    for theta_deg in thetas_deg:
        sphere, n_face, t_sample = build_tilted_scene(
            theta_deg, rho=rho, ka=ka)
        dn, dt, ratio, info = solve_and_decompose(sphere, n_face, t_sample, kc)
        predicted = np.tan(np.radians(theta_deg)) / rho
        rel = abs(ratio - predicted) / predicted
        ok = rel < TOL_REL
        all_ok = all_ok and ok
        print(f"  {theta_deg:>8.1f} {np.tan(np.radians(theta_deg)):>10.6f} "
              f"{dn*1e6:>12.4f} {dt*1e6:>12.4f} "
              f"{ratio:>14.6f} {predicted:>12.6f} {rel:>11.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

    print()
    print(f"PART B: {'PASS' if all_ok else 'FAIL'}  (tol rel < {TOL_REL:.0e})")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  v1 vs v2 formula across the ρ × θ grid + figures
# ─────────────────────────────────────────────────────────────────────────


def part_c_v1_vs_v2() -> bool:
    print()
    print("=" * 78)
    print("PART C.  v1 (with A) vs v2 (no A) tilt formulas")
    print("=" * 78)
    print()
    print("  The v1 formula  (k_a - A)/(k_a ρ - A) · tan(θ)  comes from")
    print("  line-of-centres direction rotating with δ.  v2 face normal is")
    print("  δ-independent, so A = 0.  v1 is wrong; the deviation IS the")
    print("  regression value of v2.")
    print()

    ka = 25_000.0
    kc = 25_000.0
    d_rest = 1.0e-3
    R_v1_proxy = 33.5e-3   # tennis-ball default used by v1 test_06 setup

    rhos = [1.0, 0.5, 1.0/3.0, 0.1]
    thetas_deg = np.array([5.0, 15.0, 30.0, 45.0])

    # Numerical sweep grid.
    ratios_num = np.zeros((len(rhos), len(thetas_deg)))
    v2_preds = np.zeros_like(ratios_num)
    v1_preds = np.zeros_like(ratios_num)

    for i, rho in enumerate(rhos):
        for j, theta_deg in enumerate(thetas_deg):
            sphere, n_face, t_sample = build_tilted_scene(
                theta_deg, rho=rho, ka=ka)
            _, _, ratio, _ = solve_and_decompose(sphere, n_face, t_sample, kc)
            ratios_num[i, j] = ratio
            v2_preds[i, j] = np.tan(np.radians(theta_deg)) / rho
            # v1 with A approximation: use face-on equilibrium for f_eff and L.
            delta_n_face_on = kc * d_rest / (ka + kc)
            phi_eff_face_on = d_rest - delta_n_face_on
            L_v1 = sphere.r + R_v1_proxy - phi_eff_face_on
            A = kc * phi_eff_face_on / L_v1
            v1_preds[i, j] = ((ka - A) / (ka * rho - A)
                              * np.tan(np.radians(theta_deg)))

    # Print summary.
    for i, rho in enumerate(rhos):
        rel_v2 = np.max(np.abs(ratios_num[i] - v2_preds[i]) / v2_preds[i])
        rel_v1 = np.max(np.abs(ratios_num[i] - v1_preds[i]) / v1_preds[i])
        print(f"  ρ = {rho:.4f}:  max rel err  v2 = {rel_v2:.2e}, "
              f"v1 = {rel_v1:.2e}  (v1 worse by {rel_v1/max(rel_v2, 1e-30):.0f}×)")

    # ── Figure: ratio vs θ for each ρ, v2 predicted vs numerical ──
    fig, ax = plt.subplots(figsize=(8, 5))
    theta_dense = np.linspace(0.0, 60.0, 60)
    for i, rho in enumerate(rhos):
        color = f"C{i}"
        ax.plot(theta_dense, np.tan(np.radians(theta_dense)) / rho,
                color=color, lw=2,
                label=rf"$\rho = {rho:.4g}$ (v2 prediction)")
        ax.plot(thetas_deg, ratios_num[i], "o", color=color, ms=8, mfc="white",
                label=rf"numerical")
    ax.set_xlabel(r"face tilt $\theta$ [°]")
    ax.set_ylabel(r"$|\delta_t / \delta_n|$")
    ax.set_title(r"T-D. v2 anisotropic tilt: $|\delta_t/\delta_n| = (1/\rho)\,\tan\theta$" "\n"
                 "(exact — no A correction; markers are numerical, lines are v2 closed form)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "td_tilt_vs_theta.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")

    # ── Figure: ratio vs ρ at θ = 30°, v1 vs v2 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    rho_dense = np.linspace(0.05, 1.5, 50)
    theta_eg = 30.0
    th = np.radians(theta_eg)
    ax.plot(rho_dense, np.tan(th) / rho_dense, "k-", lw=2,
            label=r"v2: $(1/\rho)\tan\theta$  (face-normal δ-indep)")
    # v1 prediction across rho_dense.
    delta_n_face_on = kc * d_rest / (ka + kc)
    phi_eff_face_on = d_rest - delta_n_face_on
    L_v1 = 2.5e-3 + R_v1_proxy - phi_eff_face_on
    A = kc * phi_eff_face_on / L_v1
    v1_curve = (ka - A) / (ka * rho_dense - A) * np.tan(th)
    ax.plot(rho_dense, v1_curve, "C3--", lw=1.5,
            label=r"v1: $(k_a - A)/(k_a\rho - A)\tan\theta$  (line-of-centres)")
    # Numerical points at θ = 30°
    j30 = int(np.argmin(np.abs(thetas_deg - theta_eg)))
    ax.plot(rhos, ratios_num[:, j30], "ko", ms=8, mfc="yellow",
            label="numerical (v2 solver)", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"tangent anchor ratio  $\rho = k_{a,t}/k_a$")
    ax.set_ylabel(r"$|\delta_t / \delta_n|$  at  $\theta = 30°$")
    ax.set_title(r"T-D. v1 vs v2 tilt formula at $\theta = 30°$" "\n"
                 r"(v1 diverges by up to ${\sim}21\%$ at $\rho = 0.1$; v2 is exact)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG_DIR / "td_v1_vs_v2.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")

    # PART C passes if v2 matches numerical at every grid point within
    # tolerance.  We separately ASSERT that v1 diverges (regression
    # target: v2 fixes the v1 bug).
    v2_max_err = float(np.max(np.abs(ratios_num - v2_preds) / v2_preds))
    v1_max_err = float(np.max(np.abs(ratios_num - v1_preds) / v1_preds))
    v2_pass = v2_max_err < TOL_REL
    v1_diverges = v1_max_err > 100.0 * v2_max_err  # v1 must be MUCH worse
    print(f"\n  v2 max rel err = {v2_max_err:.2e}  (tol {TOL_REL:.0e})  "
          f"{'PASS' if v2_pass else 'FAIL'}")
    print(f"  v1 max rel err = {v1_max_err:.2e}  "
          f"(must be ≫ v2 to count as regression: "
          f"{'PASS' if v1_diverges else 'FAIL'})")
    return v2_pass and v1_diverges


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a_ok = part_a_rho_sweep()
    b_ok = part_b_theta_sweep()
    c_ok = part_c_v1_vs_v2()
    print()
    print("=" * 78)
    print(f"T-D overall: {'PASS' if (a_ok and b_ok and c_ok) else 'FAIL'}")
    print("=" * 78)
    return 0 if (a_ok and b_ok and c_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
