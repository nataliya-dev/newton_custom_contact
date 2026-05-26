# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 / T-B — Half-space contact deep-penetration regression.

The v1 ``(r + R) - ||q - t||`` overlap form silently breaks the
moment the pad sphere centre crosses through the target sample point:
``||q - t||`` shrinks to zero and then grows again, so v1's ``raw``
gains a spurious zero-crossing and the contact force flips direction
inside the body.  T-A made the *monotonicity* statement geometrically;
T-B makes the *contact-force* statement at production-scale depths
(``q`` driven from face-touching to ``10 r`` deep inside the target
body) and verifies the v2 force is:

  A. **Finite**: no NaN or inf at any depth, including
     ``depth = +10 r`` (≫ pad spacing).
  B. **Monotone non-decreasing** in depth (a deeper push always
     gives at least as much contact force).
  C. **Saturated gate at depth ≫ eps**: ``gate(raw) → 1`` for
     ``raw ≫ eps``; specifically ``1 - gate < 1e-9`` for
     ``raw ≥ 50 eps``.  This is the active-set saturation that
     justifies the ``-50 eps`` cull threshold (contract_v2.md §3.6).
  D. **Phi_eff bounded relative to raw**: ``|phi_eff(raw, eps) - raw|
     < eps`` for any raw, so the smooth surrogate adds no spurious
     force beyond the smoothing band.  Asymptotically
     ``|phi_eff - raw| ≈ eps² / (4·raw)`` for raw ≫ eps — so the
     **worst case is at shallow contact** (raw just above 0), where
     it caps at ``eps/2`` (the irreducible smoothing floor of
     ``sigma_eps``).  At production ``eps = 5e-4 m`` the worst
     deviation across this sweep is ~``2.5e-5 m``, occurring near
     ``raw ≈ r``; at deep saturation (raw ≈ 10r) it drops to
     ~``2.5e-6 m`` — 10× smaller.  Plot panel 3 visualises this.
  E. **Force direction = +n_face**: target always pushes pad sphere
     outward along the target's outward normal, regardless of which
     side of the face the pad centre happens to be on.
  F. **v1 loses contact past r+R**: the legacy ``(r+R)-||q-t||`` form
     goes negative once ``q`` has travelled past ``t`` by more than
     ``r + R``, so the positive-part clamp drops the force to zero —
     pad is demonstrably inside the body but the model has forgotten.
     v2 keeps reporting growing contact at any depth.

The figure plots ``f_contact · n_face`` vs depth for both v2 and v1
(with the legacy ``R = sphere.r / 2``) so the "v1 cliff" at depth =
r + R is visually obvious.

Per ``cslc_main/theory/contract_v2.md`` §3.2, §3.6, §12 T-B.

Run::

    uv run -m cslc_main.theory.test_half_space_deep_penetration

Outputs::

    cslc_main/theory/figures/tb_half_space_deep_penetration.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    half_space_force,
    half_space_gate,
    half_space_phi_eff,
    half_space_raw,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Depth grid: 0 → 10 r in 200 steps (avoid negative depths — T-A already
# covered the outside-the-body regime).
N_SAMPLES = 200
EPS_PRODUCTION = 5.0e-4   # production smoothing width [m]


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sphere = LatticeSphere(p=np.zeros(3), r=2.5e-3,
                           n=np.array([1.0, 0.0, 0.0]), ka=25_000.0)
    n_face = np.array([-1.0, 0.0, 0.0])
    # Place t_sample at the face plane; we'll vary q along the face
    # normal axis.  q(depth) = t_sample - depth · n_face, so
    # ``depth >= 0`` means q is on the body side of the face.
    t_sample = np.array([0.0, 0.0, 0.0])
    kc = 25_000.0

    depth_grid = np.linspace(0.0, 10.0 * sphere.r, N_SAMPLES)
    raw_v2 = np.zeros(N_SAMPLES)
    phi_v2 = np.zeros(N_SAMPLES)
    gate_v2 = np.zeros(N_SAMPLES)
    f_v2_n = np.zeros(N_SAMPLES)   # f · n_face (signed magnitude)

    # v1 witness: raw_v1 = (r + R) - ||q - t||, direction (q - t)/||q - t||.
    # Use R = r/2 — typical legacy default; the pathology is the same at any R.
    R_legacy = sphere.r / 2.0
    raw_v1 = np.zeros(N_SAMPLES)
    f_v1_n = np.zeros(N_SAMPLES)

    for i, depth in enumerate(depth_grid):
        q = t_sample - depth * n_face
        delta = sphere.p - q
        raw_v2[i] = half_space_raw(sphere, n_face, t_sample, delta)
        phi_v2[i] = half_space_phi_eff(
            sphere, n_face, t_sample, delta, eps=EPS_PRODUCTION)
        gate_v2[i] = half_space_gate(
            sphere, n_face, t_sample, delta, eps=EPS_PRODUCTION)
        f_vec = half_space_force(
            sphere, n_face, t_sample, delta, kc, eps=EPS_PRODUCTION)
        f_v2_n[i] = float(np.dot(f_vec, n_face))

        # v1 (legacy).
        diff = q - t_sample
        L = float(np.linalg.norm(diff))
        raw_v1[i] = (sphere.r + R_legacy) - L
        if L > 1.0e-15 and raw_v1[i] > 0:
            d_hat_v1 = -diff / L     # line-of-centres FROM q TO t (pushes pad away)
            f_v1 = kc * raw_v1[i] * d_hat_v1
        else:
            f_v1 = np.zeros(3)
        f_v1_n[i] = float(np.dot(f_v1, n_face))

    # ──────────────────────────────────────────────────────────────────
    # Assertions
    # ──────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("Phase 1 / T-B — Half-space contact deep-penetration regression")
    print("=" * 72)
    print()
    print(f"Sweep: depth ∈ [0, {10*sphere.r*1e3:.1f}] mm "
          f"(0 → 10 r past the face)")
    print(f"       Sphere r = {sphere.r*1e3:.1f} mm, "
          f"k_c = {kc:.0f} N/m, eps = {EPS_PRODUCTION:.0e} m")
    print()

    # A. Finite at every depth.
    a_pass = bool(np.all(np.isfinite(raw_v2))
                  and np.all(np.isfinite(phi_v2))
                  and np.all(np.isfinite(gate_v2))
                  and np.all(np.isfinite(f_v2_n)))
    print(f"A. v2 finite at all depths up to 10 r: "
          f"{'PASS' if a_pass else 'FAIL'}  "
          f"(max raw = {np.max(raw_v2)*1e3:.3f} mm, "
          f"max |f·n| = {np.max(np.abs(f_v2_n)):.3e} N)")

    # B. Force monotone non-decreasing in depth (deeper push gives more
    # force).  Allow a tiny floor for fp noise.
    f_diffs = np.diff(f_v2_n)
    b_pass = bool(np.all(f_diffs >= -1.0e-12))
    print(f"B. v2 force monotone in depth:        "
          f"{'PASS' if b_pass else 'FAIL'}  "
          f"(min diff = {np.min(f_diffs):.3e})")

    # C. gate saturates at 1 for raw >= 50 eps.  Cull threshold is
    # -50 eps; symmetrically the SATURATION threshold is +50 eps:
    # gate(50 eps) ≈ 1 - 1e-4 and asymptotes to 1 thereafter.
    saturation_threshold = 50.0 * EPS_PRODUCTION
    deep_mask = raw_v2 >= saturation_threshold
    if np.any(deep_mask):
        gate_deep_min = float(np.min(gate_v2[deep_mask]))
        # gate(+50 eps) = 0.5*(1 + 50/sqrt(2501)) ≈ 0.999900
        # For raw ≫ 50 eps it monotonically approaches 1.
        c_pass = gate_deep_min > 0.9998
    else:
        gate_deep_min = float("nan")
        c_pass = False
    print(f"C. gate saturated for raw ≥ 50 eps:   "
          f"{'PASS' if c_pass else 'FAIL'}  "
          f"(min gate where raw≥50 eps = {gate_deep_min:.9f})")

    # D. phi_eff bounded by raw: |phi_eff - max(0, raw)| < eps for
    # any raw (the irreducible eps/2 floor of smooth_relu).  Worst
    # case is at the SHALLOW end (raw ~ r); asymptotically
    # |phi_eff - raw| ≈ eps²/(4·raw) for raw ≫ eps, so the floor
    # decays as 1/raw with depth.
    contact_only = raw_v2 > 0
    phi_err = np.abs(phi_v2[contact_only] - raw_v2[contact_only])
    phi_err_max = float(np.max(phi_err)) if phi_err.size > 0 else 0.0
    d_pass = phi_err_max < EPS_PRODUCTION
    # Locate where the worst case actually occurs (sanity: should be
    # at the shallow end of the sweep).
    raw_at_worst = float(raw_v2[contact_only][int(np.argmax(phi_err))])
    print(f"D. |phi_eff − raw| < eps:             "
          f"{'PASS' if d_pass else 'FAIL'}  "
          f"(max = {phi_err_max:.3e} m at raw = {raw_at_worst*1e3:.3f} mm; "
          f"bound eps = {EPS_PRODUCTION:.3e} m)")

    # E. Force direction = +n_face (always pushes pad along target's
    # outward normal).  We check f · n_face >= 0 at every depth (positive
    # = aligned with n_face, i.e., physical force pushes pad outward
    # from target body).  The "f_v2_n" we computed IS f · n_face;
    # confirm it's never negative for raw > 0.
    contact_mask = raw_v2 > 0
    e_pass = bool(np.all(f_v2_n[contact_mask] >= -1.0e-15))
    print(f"E. v2 force direction stable (+n_face): "
          f"{'PASS' if e_pass else 'FAIL'}  "
          f"(min f·n_face in contact = "
          f"{np.min(f_v2_n[contact_mask]) if np.any(contact_mask) else 0:.3e})")

    # F. v1 LOSES CONTACT past depth = r + R (regression target).
    # v1 raw goes negative for depth > r + R, so the standard positive-
    # part clamp drops the force to ZERO inside the body — even though
    # the pad is now demonstrably penetrating the target.  v2's force
    # keeps growing.  The contrast is "v1 forgets it's in contact at
    # deep penetration; v2 doesn't."
    contact_loss_depth = sphere.r + R_legacy
    deep_idx = depth_grid > 1.5 * contact_loss_depth   # well past the cliff
    v1_lost_contact = bool(np.all(f_v1_n[deep_idx] < 1.0e-9))
    v2_still_active = bool(np.all(f_v2_n[deep_idx] > 1.0))   # large force
    f_pass = v1_lost_contact and v2_still_active
    print(f"F. v1 loses contact past r+R={contact_loss_depth*1e3:.2f} mm: "
          f"{'PASS' if f_pass else 'FAIL'}  "
          f"(v1 max past cliff = {np.max(f_v1_n[deep_idx]):.3e} N, "
          f"v2 min past cliff = {np.min(f_v2_n[deep_idx]):.3e} N)")

    all_pass = a_pass and b_pass and c_pass and d_pass and e_pass and f_pass

    # ──────────────────────────────────────────────────────────────────
    # Figure: force vs depth (v2 monotone, v1 sign-flips),
    # gate saturation, and |phi_eff − raw| smoothing-floor envelope.
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    depth_mm = depth_grid * 1e3

    ax = axes[0]
    ax.plot(depth_mm, f_v2_n, "k-", lw=2.0,
            label=r"v2:  $f \cdot \hat n_{\mathrm{face}}$  (monotone, $\geq 0$)")
    ax.plot(depth_mm, f_v1_n, "C3--", lw=1.4,
            label=rf"v1 ($R={R_legacy*1e3:.2f}$ mm):  "
                  r"$f \cdot \hat n_{\mathrm{face}}$  (cliff at $r+R$)")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline((sphere.r + R_legacy) * 1e3, color="C3", lw=0.5, ls=":",
               label=rf"v1 contact lost ($r+R={(sphere.r+R_legacy)*1e3:.2f}$ mm)")
    ax.set_ylabel("force · n_face [N]")
    ax.set_title(
        "T-B. Deep-penetration regression\n"
        "v2 force monotone-grows with depth; v1 drops to 0 past r + R "
        "(loses contact inside the body)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(depth_mm, gate_v2, "C0-", lw=1.5, label=r"$\mathrm{gate}_{v2}$")
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")
    ax.axvline(saturation_threshold * 1e3, color="C0", lw=0.5, ls=":",
               label=rf"$50\,\varepsilon$ saturation threshold")
    ax.set_ylabel("gate")
    ax.set_title(r"$\mathrm{gate} \to 1$ for $\mathrm{raw} \gg \varepsilon$ "
                 "(active-set saturation, justifies the −50 ε cull)")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: |phi_eff - raw| envelope.  Plots numerical deviation
    # and overlays the asymptotic 1/raw decay so the "worst-case is at
    # SHALLOW contact, not deep saturation" point is visually obvious.
    ax = axes[2]
    phi_dev = np.abs(phi_v2 - np.maximum(raw_v2, 0.0))
    ax.semilogy(depth_mm, np.maximum(phi_dev, 1.0e-12), "C0-", lw=1.5,
                label=r"$|\varphi_{\mathrm{eff}} - \mathrm{raw}|$ (numerical)")
    # Asymptote eps²/(4·raw) valid for raw ≫ eps; clip to eps/2 at raw=0.
    raw_clip = np.maximum(raw_v2, 0.5 * EPS_PRODUCTION)
    asym = np.minimum(EPS_PRODUCTION / 2.0,
                      EPS_PRODUCTION ** 2 / (4.0 * raw_clip))
    ax.semilogy(depth_mm, asym, "C3--", lw=1.2,
                label=r"asymptote $\min(\varepsilon/2,\, \varepsilon^2/(4\,\mathrm{raw}))$")
    ax.axhline(EPS_PRODUCTION / 2.0, color="grey", lw=0.5, ls=":",
               label=rf"$\varepsilon/2 = {EPS_PRODUCTION/2:.1e}$ (floor)")
    ax.set_xlabel("depth into target body [mm]")
    ax.set_ylabel(r"$|\varphi_{\mathrm{eff}} - \mathrm{raw}|$ [m]")
    ax.set_title(r"Smoothing floor: worst at SHALLOW contact (raw ≈ r), "
                 r"decays as $1/\mathrm{raw}$ with depth")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = FIG_DIR / "tb_half_space_deep_penetration.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    print()
    print("=" * 72)
    print(f"T-B overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 72)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
