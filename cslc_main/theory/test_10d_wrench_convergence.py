# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 10d -- convergence rate of the dome's lateral wrench residual vs N.

Diagnostic test that answers ONE binary question raised by Step 10c:

  Is the lateral wrench residual on the ball a finite-N discretization
  artifact that VANISHES at the central-limit rate ``1/sqrt(N)`` (in
  which case CSLC's distributed thesis is preserved and no global
  wrench correction is warranted) -- OR is it a SYSTEMATIC bias in the
  Fibonacci-spiral sampler that doesn't average out as N grows
  (in which case a wrench-level projection might be justified despite
  the architectural cost)?

We sweep ``N`` over 5 decades (50 -> 5000) on the SAME geometric
scene as Step 10c, solve the canonical baseline solver
(``solve_lattice_sphere_indenter``), measure the ratio
``|F_tangent| / |F_normal|`` on the ball at the converged equilibrium,
and fit a power law

    log10(ratio)  =  intercept  +  slope * log10(N).

Interpretation:

  * ``slope <= -0.4``  -> 1/sqrt(N) or faster.  Central-limit
                         convergence of independent per-sphere lateral
                         asymmetries.  The residual is a clean
                         discretization artifact; **the distributed
                         contact thesis is preserved exactly, no
                         wrench projection needed.**  Verdict PASS
                         (continue with the distributed model;
                         characterize the rate as the paper's
                         finite-N convergence story).

  * ``-0.1 < slope``   -> Essentially no convergence.  The Fibonacci
                         spiral has a systematic lateral bias.  Wrench
                         projection becomes a justified architectural
                         cost.  Verdict FAIL (consider projection
                         path).

  * ``-0.4 < slope <= -0.1``  -> Marginal.  Some convergence but
                                  slower than CLT.  Need more N
                                  values, or look for a different
                                  mechanism.

Run::

    uv run --extra importers -m cslc_main.theory.test_10d_wrench_convergence

Output::

    cslc_main/theory/figures/10d_wrench_convergence.png

Removability: same as test_10c.  This is an experimental diagnostic;
it does not modify the canonical theory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    SphereIndenter,
    make_dome,
    solve_lattice_sphere_indenter,
)
from cslc_main.theory.cslc_theory import INACTIVE_RAW_EPS_FACTOR


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _run_one_N(N: int, params: dict) -> dict:
    """Solve baseline equilibrium on the dome+ball scene at lattice size N.

    Returns measurements of the converged wrench on the ball, plus
    metadata.
    """
    R_pad = params["R_pad"]
    half_angle = params["half_angle"]
    ka = params["ka"]
    kl = params["kl"]
    r_lat = params["r_lat"]
    R_ball = params["R_ball"]
    kc = params["kc"]
    phi_apex = params["phi_apex"]
    eps = params["eps"]

    lat, spacing, cap_area = make_dome(
        N=N, R_pad=R_pad, half_angle=half_angle,
        ka=ka, kl=kl, k_neighbors=6)

    apex_idx = 0
    apex_p = lat.p[apex_idx]
    apex_n = lat.n[apex_idx]
    t_ball = apex_p + (r_lat + R_ball - phi_apex) * apex_n
    indenter = SphereIndenter(t=t_ball, R=R_ball, kc=kc)

    t0 = time.perf_counter()
    delta, info = solve_lattice_sphere_indenter(
        lat, indenter, r_lat, lateral="distance_preserving",
        eps=eps, tol=1.0e-12, maxiter=5000)
    solve_s = time.perf_counter() - t0

    # Compute the net contact wrench on the ball.  Force ON BALL from
    # sphere i is  +kc * phi_eff_i * step_i * (t - q_i) / ||t - q_i||
    # (Newton III: equal-and-opposite of the contact force on sphere i,
    # which is along (q_i - t)/dist).
    F = np.zeros(3)
    n_active = 0
    for i in range(lat.N):
        q_i = lat.p[i] - delta[i]
        diff = t_ball - q_i
        L = float(np.linalg.norm(diff))
        if L < 1.0e-15:
            continue
        raw = (r_lat + R_ball) - L
        if raw < INACTIVE_RAW_EPS_FACTOR * eps:
            continue
        denom = np.sqrt(raw * raw + eps * eps)
        phi_eff = 0.5 * (raw + denom)
        step = 0.5 * (1.0 + raw / denom)
        F += kc * phi_eff * step * (diff / L)
        n_active += 1

    F_n = float(np.dot(F, apex_n))
    F_t_vec = F - F_n * apex_n
    F_t_mag = float(np.linalg.norm(F_t_vec))
    # Direction of F_t in the apex tangent plane (unit vector or zero).
    F_t_hat = F_t_vec / F_t_mag if F_t_mag > 1.0e-30 else np.zeros(3)

    return {
        "N": N,
        "spacing": spacing,
        "cap_area": cap_area,
        "n_active": n_active,
        "F_n": F_n,
        "F_t_mag": F_t_mag,
        "F_t_hat": F_t_hat,
        "F_full": F,
        "ratio": F_t_mag / max(abs(F_n), 1.0e-30),
        "solve_s": solve_s,
        "nit": info["nit"],
        "final_grad_norm": info["final_grad_norm"],
    }


def main() -> int:
    print()
    print("=" * 78)
    print("Step 10d  Wrench-residual convergence rate vs N (dome + ball)")
    print("=" * 78)

    params = dict(
        R_pad=10.0e-3,
        half_angle=72.0 * np.pi / 180.0,
        ka=1.0e4,
        kl=0.2 * 1.0e4,
        r_lat=1.5e-3,
        R_ball=30.0e-3,
        kc=1.0e4,
        phi_apex=5.0e-4,
        eps=5.0e-4,
    )

    # Five points spanning two decades.  N = 5000 takes ~30 s; if you
    # need to drop it for a quick run, the slope from {50, 150, 500,
    # 1500} alone is usually within 0.05 of the 5-point slope.
    N_list = [50, 150, 500, 1500, 5000]

    results = []
    print(f"{'N':>6}  {'spacing':>10}  {'F_n [N]':>12}  {'|F_t| [N]':>12}  "
          f"{'ratio [%]':>10}  {'F_t direction (apex frame)':>32}  "
          f"{'t [s]':>6}")
    print("-" * 110)
    for N in N_list:
        r = _run_one_N(N, params)
        results.append(r)
        # Project F_t direction onto a fixed apex-tangent frame for
        # cross-N comparison (does the residual point in a consistent
        # direction across N, or rotate around?).
        F_t_hat_xyz = r["F_t_hat"]
        print(f"{N:>6d}  {r['spacing'] * 1e3:>8.2f} mm  "
              f"{r['F_n']:>12.4e}  {r['F_t_mag']:>12.4e}  "
              f"{r['ratio'] * 100:>9.3f}%  "
              f"({F_t_hat_xyz[0]:+.3f}, {F_t_hat_xyz[1]:+.3f}, {F_t_hat_xyz[2]:+.3f})  "
              f"{r['solve_s']:>5.1f}")

    # Direction consistency: dot products of F_t_hat between consecutive N's.
    # If the residual rotates (random direction at each N), dot products
    # are near zero; if it points consistently, dot products near +/-1.
    print()
    print("Direction consistency of F_t (dot products between consecutive N):")
    for i in range(len(results) - 1):
        dot = float(np.dot(results[i]["F_t_hat"], results[i + 1]["F_t_hat"]))
        print(f"  N = {results[i]['N']:5d} -> {results[i + 1]['N']:5d}:  "
              f"<F_t_hat_i, F_t_hat_j> = {dot:+.4f}")

    # ── Fit log-log power law to ratio vs N ──
    Ns = np.array([r["N"] for r in results], dtype=np.float64)
    ratios = np.array([r["ratio"] for r in results], dtype=np.float64)
    log_N = np.log10(Ns)
    log_r = np.log10(ratios)
    slope, intercept = np.polyfit(log_N, log_r, 1)

    # Residuals to the fit.
    fitted = intercept + slope * log_N
    residuals = log_r - fitted
    rms_residual_log10 = float(np.sqrt(np.mean(residuals ** 2)))

    print()
    print(f"Power-law fit: log10(|F_t|/|F_n|) = {intercept:.3f} + {slope:.3f} * log10(N)")
    print(f"Convergence rate exponent: {slope:.3f}")
    print(f"  (slope = -0.5  => exact 1/sqrt(N) CLT convergence)")
    print(f"  (slope =  0    => no convergence, systematic bias)")
    print(f"RMS residual of fit (log10 units): {rms_residual_log10:.3f}  "
          f"(< 0.1 = single power law dominates)")

    # ── Verdict ──
    if slope <= -0.4:
        verdict = ("PASS",
                   "1/sqrt(N) or faster convergence -- discretization "
                   "noise, distributed thesis preserved")
        rc = 0
    elif slope <= -0.1:
        verdict = ("MARGINAL",
                   "slower than CLT but still converging; investigate "
                   "further or sweep wider N")
        rc = 1
    else:
        verdict = ("FAIL",
                   "no significant convergence -- Fibonacci spiral has "
                   "a systematic lateral bias; wrench projection may be "
                   "justified")
        rc = 2

    print()
    print(f"VERDICT: {verdict[0]}  --  {verdict[1]}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: convergence plot with fit + 1/sqrt(N) reference.
    ax = axes[0]
    ax.loglog(Ns, ratios * 100, "o-", color="C0", lw=2, ms=8,
              label=f"measured  (slope = {slope:.3f})")
    # 1/sqrt(N) reference, normalised to pass through the first data point.
    ref = ratios[0] * np.sqrt(Ns[0]) / np.sqrt(Ns)
    ax.loglog(Ns, ref * 100, "--", color="gray", lw=1.5, alpha=0.7,
              label="1/sqrt(N) reference (slope = -0.5)")
    # Constant reference -- worst case.
    ax.loglog(Ns, np.full_like(Ns, ratios[0] * 100), ":", color="C3",
              lw=1.5, alpha=0.7,
              label="constant residual (slope = 0)")
    ax.set_xlabel("N (lattice spheres on dome)")
    ax.set_ylabel("|F_tangent| / |F_normal| on ball [%]")
    ax.set_title(f"Wrench residual convergence -- {verdict[0]}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # Right: F_t direction in the apex-tangent plane across N.  If the
    # residual is random, points scatter; if systematic, they cluster.
    ax = axes[1]
    # Pick a 2D basis in the apex tangent plane.
    apex_n = np.array([0., 0., 1.])  # results[*].F_t_hat is already in
                                      # the apex-tangent plane (we
                                      # subtracted the apex_n component
                                      # in _run_one_N), but built from
                                      # lat.n[0] which differs slightly
                                      # from (0,0,1).  For plotting we
                                      # take F_t_hat's x,y components
                                      # directly.
    for r in results:
        ax.scatter(r["F_t_hat"][0], r["F_t_hat"][1], s=80,
                   alpha=0.7, label=f"N = {r['N']}")
        ax.annotate(f"{r['N']}",
                    (r["F_t_hat"][0], r["F_t_hat"][1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    circ = plt.Circle((0, 0), 1.0, fill=False, color="gray", lw=0.5)
    ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.set_xlabel("F_t_hat . x")
    ax.set_ylabel("F_t_hat . y")
    ax.set_title("Direction of lateral residual\n(unit vectors; scatter = random, cluster = bias)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "10d_wrench_convergence.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"figure -> {out.relative_to(Path.cwd())}")

    print()
    print("=" * 78)
    return rc


if __name__ == "__main__":
    sys.exit(main())
