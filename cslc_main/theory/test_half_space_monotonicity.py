# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 / T-A — Half-space overlap monotonicity.

Sweeps the pad sphere's deformed centre ``q`` along the face normal
axis from ``+10 r`` outside the target body to ``-10 r`` deep inside,
parameterised by the signed *penetration depth*

    depth  =  -n_face · (q - t_sample)         (>0 when q sits past
                                                the face plane, inside
                                                the target body)

so that geometrically meaningful "depth increases" = "q moves into the
target body".  At every step asserts:

  A. ``raw(depth)`` strictly increasing (monotone, no flat regions,
     no sign-flip past face crossing).  Analytically
     ``raw = r + depth`` — a straight line — but the test verifies
     the implementation numerically at 401 points.
  B. ``raw`` covers both signs over the sweep (sanity: we span the
     face-crossing region).
  C. ``phi_eff = sigma_eps(raw)`` is monotonically non-decreasing
     for every smoothing width ``eps ∈ {1e-9, 5e-4}`` *to fp precision*.
     ``sigma_eps`` is analytically monotone, so any negative diff is
     pure float64 noise from the ``sqrt(x² + ε²)`` cancellation near
     ``raw ≈ 0``; the assertion uses a ``-1e-15`` fp tolerance.
  D. ``gate = Sigma_eps(raw)`` is monotone *to fp precision* and
     bounded in ``[0, 1]``.

This is the v2 regression that the v1 ``(r + R) - ||q - t||`` form
could not pass.  The figure overlays the v1 formula on the same
sweep — v1 is an inverted-V centred on ``q = t`` (its raw FLIPS sign
the moment ``q`` crosses through the target sample point), v2 is a
straight monotone line.

Per ``cslc_main/theory/contract_v2.md`` §3.2, §3.6, §12 T-A.

Run::

    uv run -m cslc_main.theory.test_half_space_monotonicity

Outputs::

    cslc_main/theory/figures/ta_half_space_monotonicity.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    half_space_gate,
    half_space_phi_eff,
    half_space_raw,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


# Sweep covers depth ∈ [-10 r, +10 r] (q far outside → q deep inside
# the target body).  N samples chosen so the spacing is fine enough
# that the smooth surrogate's O(eps) transition is well-resolved at
# the production-scale ``eps = 5e-4``.
N_SAMPLES = 401


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Canonical scene.  Sphere outward normal +x; face-on target with
    # n_face = -x and rest overlap d_rest = +1 mm at delta = 0.
    sphere = LatticeSphere(p=np.zeros(3), r=2.5e-3,
                           n=np.array([1.0, 0.0, 0.0]), ka=25_000.0)
    n_face = np.array([-1.0, 0.0, 0.0])
    d_rest = 1.0e-3
    # Place t_sample so half_space_raw(p, 0) == d_rest:
    #   r - n_face · (p - t_sample) = d_rest  =>  n_face · t_sample = d_rest - r
    #   with n_face = -x:  t_sample.x = r - d_rest.
    t_sample = np.array([sphere.r - d_rest, 0.0, 0.0])
    raw0 = half_space_raw(sphere, n_face, t_sample, np.zeros(3))
    assert np.isclose(raw0, d_rest, atol=1e-12), \
        f"setup error: rest raw = {raw0}, expected {d_rest}"

    # Sweep variable: signed half-space depth from face plane.
    # q lies along the face normal through t_sample:
    #   q(depth)  =  t_sample - depth * n_face
    # Then  -n_face · (q - t_sample)  =  -n_face · (-depth · n_face)
    #                                 =  depth · ||n_face||^2  =  depth.
    # And  raw  =  r - n_face · (q - t_sample)  =  r + depth  (linear).
    depth_grid = np.linspace(-10.0 * sphere.r, +10.0 * sphere.r, N_SAMPLES)
    raw_vals = np.zeros(N_SAMPLES)
    eps_values = [1.0e-9, 5.0e-4]   # tight + production
    phi_vals = {eps: np.zeros(N_SAMPLES) for eps in eps_values}
    gate_vals = {eps: np.zeros(N_SAMPLES) for eps in eps_values}

    # v1 (legacy) comparison: raw_v1 = (r + R) - ||q - t_sample||.
    # Use R = r/2 to mirror the production v1 form (matches T-B's
    # legacy R, where a per-sample radius was a discretisation hack
    # typically tied to pad spacing).  The inverted-V pathology is
    # present at any R; using R > 0 makes the cliff at depth = r + R
    # explicit instead of degenerate at R = 0.
    R_legacy = sphere.r / 2.0
    raw_v1_vals = np.zeros(N_SAMPLES)

    for i, depth in enumerate(depth_grid):
        q = t_sample - depth * n_face
        delta = sphere.p - q
        raw_vals[i] = half_space_raw(sphere, n_face, t_sample, delta)
        for eps in eps_values:
            phi_vals[eps][i] = half_space_phi_eff(
                sphere, n_face, t_sample, delta, eps=eps)
            gate_vals[eps][i] = half_space_gate(
                sphere, n_face, t_sample, delta, eps=eps)
        raw_v1_vals[i] = (sphere.r + R_legacy) - float(np.linalg.norm(q - t_sample))

    # ──────────────────────────────────────────────────────────────────
    # Assertions
    # ──────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("Phase 1 / T-A — Half-space overlap monotonicity")
    print("=" * 72)
    print()
    print(f"Sweep: depth ∈ [{-10*sphere.r*1e3:.1f}, {+10*sphere.r*1e3:.1f}] mm "
          f"(N = {N_SAMPLES})")
    print(f"       Sphere r = {sphere.r*1e3:.1f} mm, "
          f"rest overlap d = {d_rest*1e3:.1f} mm")
    print()

    # A. raw strictly increasing in depth.  Analytically raw = r + depth,
    # so the consecutive-sample step should be exactly the constant
    # ``(20 r) / (N - 1)``.  The metric below is the worst per-step
    # deviation from that constant — i.e., float64 noise on the linear
    # raw function, NOT a residual of a least-squares fit.
    raw_diffs = np.diff(raw_vals)
    raw_mono_strict = bool(np.all(raw_diffs > 0))
    expected_step = (20.0 * sphere.r) / (N_SAMPLES - 1)
    max_step_dev = float(np.max(np.abs(raw_diffs - expected_step)))
    a_pass = raw_mono_strict and max_step_dev < 1.0e-15
    print(f"A. raw strictly increasing:           "
          f"{'PASS' if a_pass else 'FAIL'}  "
          f"(min diff = {np.min(raw_diffs):.3e} m, "
          f"max |step − {expected_step:.3e}| = {max_step_dev:.3e} m)")

    # B. raw covers both signs (sweep spans the face crossing).
    raw_min, raw_max = float(np.min(raw_vals)), float(np.max(raw_vals))
    sign_crossings = int(np.sum(np.diff(np.sign(raw_vals)) != 0))
    b_pass = (raw_min < 0) and (raw_max > 0) and sign_crossings == 1
    print(f"B. raw spans face crossing:           "
          f"{'PASS' if b_pass else 'FAIL'}  "
          f"(raw ∈ [{raw_min*1e3:.3f}, {raw_max*1e3:.3f}] mm, "
          f"{sign_crossings} crossing)")

    # C. phi_eff non-decreasing (to fp precision: sigma_eps is
    # analytically monotone, but the sqrt(x²+ε²) cancellation near
    # raw ≈ 0 produces O(1e-18) negative diffs that are pure float64
    # noise, not violations of monotonicity).
    c_pass = True
    for eps in eps_values:
        diffs = np.diff(phi_vals[eps])
        ok = bool(np.all(diffs >= -1.0e-15))
        c_pass = c_pass and ok
        print(f"C. phi_eff(eps={eps:.0e}) non-decreasing (to fp): "
              f"{'PASS' if ok else 'FAIL'}  "
              f"(min signed diff = {np.min(diffs):.3e})")

    # D. gate non-decreasing AND bounded in [0, 1].
    d_pass = True
    for eps in eps_values:
        diffs = np.diff(gate_vals[eps])
        mono = bool(np.all(diffs >= -1.0e-15))
        bounded = bool(np.all(gate_vals[eps] >= -1.0e-15)
                       and np.all(gate_vals[eps] <= 1.0 + 1.0e-15))
        ok = mono and bounded
        d_pass = d_pass and ok
        print(f"D. gate(eps={eps:.0e}) monotone (to fp) + bounded: "
              f"{'PASS' if ok else 'FAIL'}  "
              f"(min signed diff = {np.min(diffs):.3e}, "
              f"range [{np.min(gate_vals[eps]):.6f}, "
              f"{np.max(gate_vals[eps]):.6f}])")

    # E. v1 form fails monotonicity (regression witness).
    v1_diffs = np.diff(raw_v1_vals)
    v1_sign_changes = int(np.sum(np.diff(np.sign(v1_diffs)) != 0))
    e_pass = v1_sign_changes > 0  # v1 SHOULD be non-monotone
    print(f"E. v1 raw non-monotone (regression):  "
          f"{'PASS' if e_pass else 'FAIL'}  "
          f"(v1 has {v1_sign_changes} slope flips; v2 has 0)")

    all_pass = a_pass and b_pass and c_pass and d_pass and e_pass

    # ──────────────────────────────────────────────────────────────────
    # Figure: stack raw / phi_eff / gate vs depth; overlay v1 raw
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    depth_mm = depth_grid * 1e3

    ax = axes[0]
    ax.plot(depth_mm, raw_vals * 1e3, "k-", lw=2.0,
            label=r"v2:  $r - \hat n_{\mathrm{face}}\cdot(q-t)$  (monotone)")
    ax.plot(depth_mm, raw_v1_vals * 1e3, "C3--", lw=1.4,
            label=r"v1:  $r - \|q-t\|$  (inverted-V, flips at face)")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":", label="face plane")
    ax.set_ylabel("raw [mm]")
    ax.set_title(
        "T-A. Half-space overlap monotonicity\n"
        "v2 (black) grows linearly with penetration depth; "
        "v1 (red) flips sign at face crossing")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = {1.0e-9: "C0", 5.0e-4: "C1"}
    for eps in eps_values:
        ax.plot(depth_mm, phi_vals[eps] * 1e3, color=colors[eps], lw=1.5,
                label=rf"$\varepsilon = {eps:.0e}$ m")
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel(r"$\varphi_{\mathrm{eff}}$ [mm]")
    ax.set_title(r"$\varphi_{\mathrm{eff}} = \sigma_\varepsilon(\mathrm{raw})$  "
                 r"(smooth ReLU; lifts $\varepsilon/2$ floor far below)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for eps in eps_values:
        ax.plot(depth_mm, gate_vals[eps], color=colors[eps], lw=1.5,
                label=rf"$\varepsilon = {eps:.0e}$ m")
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel(r"depth into target body  $-\hat n_{\mathrm{face}}\cdot(q-t)$ [mm]")
    ax.set_ylabel("gate")
    ax.set_title(r"$\mathrm{gate} = \Sigma_\varepsilon(\mathrm{raw})$  "
                 r"(smooth step; saturates at 1 for deep contact)")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "ta_half_space_monotonicity.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    print()
    print("=" * 72)
    print(f"T-A overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 72)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
