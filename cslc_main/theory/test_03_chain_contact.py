# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 3 / Contact on the chain: how lateral coupling spreads the load.

Built on top of step 2 (chain + lateral) and step 1 (single-sphere
contact).  Same chain geometry as step 2; one rigid target placed
face-on above the centre sphere.

Math (derived in chat; recap below):

Graph-Laplacian equilibrium with one face-on contact at sphere k of an
N-sphere chain (scalar field along the outward normal n):

    (K + kc * e_k e_k^T) delta_n  =  kc * phi_rest * e_k

By Sherman-Morrison the closed form is

    delta_n_k = kc * phi_rest * g_kk / (1 + kc * g_kk),
    g_kk      = (K^{-1})_kk

i.e. the **single-sphere series-spring formula with ka replaced by
1/g_kk**.  Lateral coupling INCREASES the effective anchor stiffness
(neighbours share the load through their own anchor springs).

For the infinite chain the closed form is

    1/g_kk = sqrt(ka * (ka + 4*kl))

Per-sphere force on the body  F_i = ka * delta_n_i  forms the "patch
pressure profile".  By Newton's 3rd law

    sum_i F_i  =  kc * (phi_rest - delta_n_k)  =  F_contact

which we verify to machine precision.

Distance-preserving lateral on the SAME chain + perpendicular contact
is degenerate: edges along x_hat, source along y_hat, rank-1 projector
zeros the cross-axis coupling at linear order, so only sphere k
yields.  The 2D / 3D lattice case (future step) is where this degeneracy
goes away.

What the parts test:

  PART A.  Graph-Laplacian: solve the closed-form chain-contact system.
           Verify the centre-sphere delta against the Sherman-Morrison
           prediction.  Plot delta(i) and F(i) = ka * delta(i).  Verify
           force balance sum F_i = kc * (phi_rest - delta_k) to ~1e-12 N.

  PART B.  kl sweep.  For kl/ka in {0, 0.2, 1, 10, 100}, overlay the
           per-sphere delta profile.  Report peak delta, total force,
           and FWHM of the patch.  Demonstrate that
             kl = 0    -> delta_k = kc*phi/(ka+kc), neighbours = 0
             kl -> inf -> delta uniform across the chain, F_total -> kc*phi

  PART C.  Distance-preserving comparison.  The IDEAL nonlinear lateral
           law is *quasi-localised* in 1D chain + perpendicular contact:
           linear distance-pres is exactly localised (rank-1 projector
           zeros transverse coupling), but the nonlinearity contributes
           a third-order coupling that slightly REDUCES delta_k (the
           stretched spring resists compression) and pushes neighbours
           in the SAME direction as k by a small amount.

           Hand-prediction at delta_k ~ kc*phi/(ka+kc) = 500 um, h=2.5 mm,
           kl = 5000 N/m: stretched lateral spring force in the y
           direction on k is 2*kl*(delta/h)*(delta^2/(2h)) ~ 0.1 N, so
           delta_k drops to ~(kc*phi - 0.1)/(ka+kc) = 498 um (~0.4%
           below isolated).  Neighbour delta_y ~ 0.05 N / ka = 2 um
           (same sign as delta_k, not bulging).

           True outward bulging (negative delta_n at neighbours) is a
           2D / 3D phenomenon: it requires neighbours whose rest edges
           point even partly along the contact direction, which a 1D
           chain (all edges transverse to contact) does not have.

Run::

    uv run -m cslc_main.theory.test_03_chain_contact

Outputs::

    cslc_main/theory/figures/03a_chain_contact_profile.png
    cslc_main/theory/figures/03b_kl_sweep_patch.png
    cslc_main/theory/figures/03c_gl_vs_dp_chain_contact.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    ContactTarget,
    anchor_force_all,
    build_K_matrix,
    make_chain,
    solve_chain_contact_linear,
    solve_lattice_contact_numerical,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ─────────────────────────────────────────────────────────────────────────
#  Convenience: build the step-3 reference scene
# ─────────────────────────────────────────────────────────────────────────


def make_scene(
    *,
    N: int = 21,
    ka: float = 25_000.0,
    kl: float = 5_000.0,
    h: float = 2.5e-3,
    r_lat: float = 2.5e-3,   # not stored in Lattice; we use it for target placement
    R: float = 33.5e-3,
    kc: float = 25_000.0,
    phi_rest: float = 1.0e-3,
):
    """Build a chain + a single face-on target on the centre sphere.

    Returns (lat, target, phi_rest, i_centre).  ``phi_rest`` is also
    returned for convenience: the chain test driver passes it
    separately to the contact solvers (Lattice doesn't store a target).
    """
    lat = make_chain(N=N, h=h, ka=ka, kl=kl)
    i_centre = N // 2
    # Target sits at p_k + d * n_hat with d = (r_lat + R) - phi_rest so the
    # rest overlap (with sphere k) is exactly phi_rest.
    d = (r_lat + R) - phi_rest
    target = ContactTarget(
        sphere_idx=i_centre,
        t=lat.p[i_centre] + d * lat.n[i_centre],
        R=R,
        kc=kc,
    )
    return lat, target, phi_rest, i_centre


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Graph-Laplacian contact-on-chain at centre
# ─────────────────────────────────────────────────────────────────────────


def part_a_chain_contact_basic() -> bool:
    """Solve the closed-form chain-contact system; verify all consistency
    checks and produce the delta + force profile plot."""
    print()
    print("=" * 72)
    print("PART A.  Graph-Laplacian: contact at centre of N=21 chain")
    print("=" * 72)

    lat, target, phi_rest, i_centre = make_scene()
    deltas = solve_chain_contact_linear(lat, target, phi_rest)

    # Per-sphere outward delta (the only nonzero component).
    delta_n = np.array([float(np.dot(deltas[i], lat.n[i]))
                        for i in range(lat.N)])

    # Sherman-Morrison closed form for the centre sphere.
    K = build_K_matrix(lat)
    Kinv = np.linalg.inv(K)
    g_kk = float(Kinv[i_centre, i_centre])
    delta_k_predicted = target.kc * phi_rest * g_kk / (1.0 + target.kc * g_kk)
    ka_eff = 1.0 / g_kk

    # Anchor force per sphere = ka * delta_i.  Total = contact force by N3L.
    F_per_sphere = lat.ka * delta_n
    F_total = float(np.sum(F_per_sphere))
    F_contact_predicted = target.kc * (phi_rest - delta_n[i_centre])

    print()
    print(f"  N = {lat.N}, ka = {lat.ka:.0f} N/m, kl = {lat.kl:.0f} N/m, "
          f"kc = {target.kc:.0f} N/m, phi_rest = {phi_rest*1e3:.3f} mm")
    print()
    print(f"  Centre-sphere delta_n predicted (Sherman-Morrison):")
    print(f"    g_kk = (K^-1)_kk     = {g_kk:.6e}  [m/N]")
    print(f"    ka_eff = 1/g_kk      = {ka_eff:>10.2f} N/m  "
          f"({ka_eff/lat.ka:.3f}x ka -- lateral stiffens response)")
    print(f"    delta_k = kc phi g_kk / (1 + kc g_kk)")
    print(f"                          = {delta_k_predicted*1e6:.4f} um")
    print(f"  Centre-sphere delta_n numerical  = {delta_n[i_centre]*1e6:.4f} um  "
          f"rel err = {abs(delta_n[i_centre] - delta_k_predicted)/delta_k_predicted:.2e}")
    print()
    print(f"  Total body force (sum F_i = sum ka*delta_i)  = {F_total:.6f} N")
    print(f"  Contact force (kc * (phi - delta_k))         = {F_contact_predicted:.6f} N")
    print(f"  Force balance residual (N3L)                 "
          f"= {abs(F_total - F_contact_predicted):.3e} N")
    print()
    print(f"  Top of the per-sphere force profile (F_i = ka * delta_i):")
    print(f"    sphere  i - i_c  delta_n[um]  F_i[mN]   F_i/F_total")
    for di in range(-3, 4):
        i = i_centre + di
        if 0 <= i < lat.N:
            frac = F_per_sphere[i] / F_total if abs(F_total) > 1e-15 else 0
            print(f"    {i:>4}    {di:>+3}     {delta_n[i]*1e6:>8.4f}  "
                  f"{F_per_sphere[i]*1e3:>7.4f}   {frac:>5.2%}")

    # Pass criteria:
    pred_ok = abs(delta_n[i_centre] - delta_k_predicted) / delta_k_predicted < 1e-10
    n3l_ok = abs(F_total - F_contact_predicted) < 1e-9

    print()
    print(f"  Sherman-Morrison agreement: {'PASS' if pred_ok else 'FAIL'}")
    print(f"  Newton's 3rd law balance:    {'PASS' if n3l_ok else 'FAIL'}")
    print(f"PART A result: {'PASS' if (pred_ok and n3l_ok) else 'FAIL'}")

    # Plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    offsets = np.arange(lat.N) - i_centre

    ax1.plot(offsets, delta_n * 1e6, "o-", color="C0", ms=6)
    ax1.axhline(target.kc * phi_rest / (lat.ka + target.kc) * 1e6, ls="--",
                color="gray", label=rf"isolated $\delta = k_c\varphi/(k_a+k_c)$")
    ax1.set_xlabel("sphere offset $i - i_{\\mathrm{centre}}$")
    ax1.set_ylabel(r"$\delta_n(i)$ [$\mu$m]")
    ax1.set_title(rf"$\delta_n$ profile  ($k_l/k_a = {lat.kl/lat.ka:.2f}$)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.bar(offsets, F_per_sphere * 1e3, width=0.8, color="C2", alpha=0.7,
            edgecolor="C2")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("sphere offset $i - i_{\\mathrm{centre}}$")
    ax2.set_ylabel(r"$F_i = k_a\,\delta_n(i)$ [mN]")
    ax2.set_title(rf"Per-sphere body force  (total $F = {F_total*1e3:.2f}$ mN)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "03a_chain_contact_profile.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return pred_ok and n3l_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  kl sweep: how lateral shapes the patch
# ─────────────────────────────────────────────────────────────────────────


def part_b_kl_sweep() -> bool:
    """Sweep kl/ka; overlay delta profiles; track peak, FWHM, total force."""
    print()
    print("=" * 72)
    print("PART B.  kl sweep: patch shape vs lateral stiffness")
    print("=" * 72)

    ratios = [0.0, 0.2, 1.0, 10.0, 100.0]

    fig, ax = plt.subplots(figsize=(8, 5))
    print()
    print(f"  ka = 25000, kc = 25000, phi_rest = 1.0 mm")
    print()
    print(f"  {'kl/ka':>8} {'kl[N/m]':>10} {'ka_eff[N/m]':>13} "
          f"{'delta_k[um]':>13} {'F_total[mN]':>13} {'FWHM[spc]':>11}")
    print(f"  {'-'*8} {'-'*10} {'-'*13} {'-'*13} {'-'*13} {'-'*11}")

    all_ok = True

    for ratio in ratios:
        kl = ratio * 25_000.0
        lat, target, phi_rest, i_centre = make_scene(kl=kl)
        deltas = solve_chain_contact_linear(lat, target, phi_rest)
        delta_n = np.array([float(np.dot(deltas[i], lat.n[i]))
                            for i in range(lat.N)])
        F_per = lat.ka * delta_n
        F_total = float(np.sum(F_per))

        # FWHM: width at which delta_n drops to half the peak.
        peak = float(delta_n[i_centre])
        # find indices left and right where delta crosses peak/2
        left = i_centre
        while left > 0 and delta_n[left] > peak / 2:
            left -= 1
        right = i_centre
        while right < lat.N - 1 and delta_n[right] > peak / 2:
            right += 1
        fwhm = float(right - left)

        # ka_eff: from Sherman-Morrison.
        K = build_K_matrix(lat)
        g_kk = float(np.linalg.inv(K)[i_centre, i_centre])
        ka_eff = 1.0 / g_kk

        # Sanity check: force balance.
        F_contact = target.kc * (phi_rest - delta_n[i_centre])
        if abs(F_total - F_contact) > 1e-9:
            all_ok = False

        offsets = np.arange(lat.N) - i_centre
        ax.plot(offsets, delta_n * 1e6, "o-",
                label=f"$k_l/k_a = {ratio:g}$  (FWHM = {fwhm:.1f})", ms=5)

        print(f"  {ratio:>8.2f} {kl:>10.0f} {ka_eff:>13.2f} "
              f"{peak*1e6:>13.4f} {F_total*1e3:>13.4f} {fwhm:>11.1f}")

    # Limit check: isolated (kl=0) should match single-sphere series-spring.
    iso_pred = 25_000.0 * 1.0e-3 / (25_000.0 + 25_000.0)
    print()
    print(f"  isolated limit prediction  delta_k = kc phi/(ka+kc) = "
          f"{iso_pred*1e6:.4f} um")
    print(f"  rigid-chain limit prediction  delta_k -> kc phi/(N ka + kc) "
          f"= {25_000.0 * 1.0e-3 / (21*25_000.0 + 25_000.0)*1e6:.4f} um  "
          f"(N=21)")

    ax.set_xlabel("sphere offset $i - i_{\\mathrm{centre}}$")
    ax.set_ylabel(r"$\delta_n(i)$ [$\mu$m]")
    ax.set_title("Chain contact: lateral coupling widens and shortens the "
                 r"$\delta$ patch")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "03b_kl_sweep_patch.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")
    print(f"PART B result: {'PASS' if all_ok else 'FAIL'} (force balance held across sweep)")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Distance-preserving comparison
# ─────────────────────────────────────────────────────────────────────────


def part_c_distance_pres_comparison() -> bool:
    """Compare graph-Laplacian and distance-preserving on the chain-contact.

    Expected for distance-pres on 1D chain + perpendicular contact:

      * Centre delta SLIGHTLY less than isolated (kc phi/(ka+kc)):
        stretched lateral spring's y-component resists compression at
        sphere k.  Hand: drop ~kl * delta^3 / (h^2 * (ka+kc)) per edge.

      * Neighbours yield in the SAME direction as k, magnitude on
        order of kl * delta^3 / (ka * h^2).  Not bulging (no negative
        delta_n).

      * Both effects ~third-order in delta (since stretch = delta^2/(2h)
        and the y-projection picks up another delta/h factor).
    """
    print()
    print("=" * 72)
    print("PART C.  Graph-Laplacian vs distance-preserving on chain contact")
    print("=" * 72)

    lat, target, phi_rest, i_centre = make_scene()
    h = float(lat.p[1, 0] - lat.p[0, 0])

    delta_GL = solve_chain_contact_linear(lat, target, phi_rest)
    delta_GL_n = np.array([float(np.dot(delta_GL[i], lat.n[i]))
                           for i in range(lat.N)])

    # Distance-preserving via L-BFGS-B, warm-started from the GL answer.
    delta_DP, info = solve_lattice_contact_numerical(
        lat, target, phi_rest,
        lateral="distance_preserving",
        delta0=delta_GL,
        eps=1e-9, tol=1e-12,
    )
    delta_DP_n = np.array([float(np.dot(delta_DP[i], lat.n[i]))
                           for i in range(lat.N)])

    # Closed forms.
    iso_pred = target.kc * phi_rest / (lat.ka + target.kc)
    # Third-order distance-pres correction at sphere k: two stretched
    # springs, each contributes ~kl*(delta^2/(2h))*(delta/h) = kl*delta^3/(2h^2)
    # to the y-component of force on k.  Compensating at the anchor +
    # contact balance: delta_k_dp ~ (kc*phi - 2*kl*delta^3/(2h^2)) / (ka+kc).
    # Linearise around delta = iso_pred.
    third_order_force_k = 2 * lat.kl * iso_pred**3 / (2 * h**2)
    delta_k_dp_pred = (target.kc * phi_rest - third_order_force_k) / (lat.ka + target.kc)
    # Neighbour: anchor balances one stretched spring's y-component.
    third_order_force_neighbour = lat.kl * iso_pred**3 / (2 * h**2)
    delta_neighbour_pred = third_order_force_neighbour / lat.ka

    print()
    print(f"  Centre-sphere delta_n:")
    print(f"    isolated  (linear)          = kc phi / (ka + kc)   = "
          f"{iso_pred*1e6:>9.4f} um")
    print(f"    graph-Laplacian (chain)     = kc phi g_kk/(1+kc g_kk) = "
          f"{delta_GL_n[i_centre]*1e6:>9.4f} um  (lateral stiffens -> smaller)")
    print(f"    distance-pres prediction    (third-order correction)   = "
          f"{delta_k_dp_pred*1e6:>9.4f} um")
    print(f"    distance-pres numerical     = "
          f"{delta_DP_n[i_centre]*1e6:>9.4f} um  (~= isolated, ~slightly less)")
    print()
    print(f"  Neighbour delta_n (sphere k+/-1):")
    print(f"    distance-pres prediction    = kl delta_k^3 / (2 h^2 ka) = "
          f"{delta_neighbour_pred*1e6:>9.4f} um")
    print(f"    distance-pres numerical     = "
          f"{delta_DP_n[i_centre + 1]*1e6:>9.4f} um  (same sign as k)")
    print()
    print(f"  Sign of neighbour delta: distance-pres = "
          f"{'POSITIVE (co-compress)' if delta_DP_n[i_centre+1] > 0 else 'NEGATIVE (bulge)'}; "
          f"true Poisson bulging requires 2D/3D curvature.")

    # Pass criteria.
    # The third-order analytic prediction uses iso_pred = 500 um as the
    # reference delta; the numerical equilibrium runs at delta_k ~ 498
    # um, so the cube enters the next correction round.  We accept 5%
    # on the centre (where the leading correction dominates) and 25% on
    # the neighbour (which feels the second-order feedback through
    # delta_k).  Sign + order-of-magnitude is the real test.
    centre_pred_ok = abs(delta_DP_n[i_centre] - delta_k_dp_pred) / delta_k_dp_pred < 0.05
    neighbour_pred_ok = (abs(delta_DP_n[i_centre + 1] - delta_neighbour_pred)
                         / max(delta_neighbour_pred, 1e-15)) < 0.25
    gl_stiffens_ok = delta_GL_n[i_centre] < iso_pred
    dp_softens_ok = delta_DP_n[i_centre] < iso_pred
    dp_neighbour_same_sign = delta_DP_n[i_centre + 1] > 0

    print()
    print(f"  distance-pres centre ~ third-order prediction (5%)? "
          f"{'PASS' if centre_pred_ok else 'FAIL'}")
    print(f"  distance-pres neighbour ~ third-order prediction (25%)? "
          f"{'PASS' if neighbour_pred_ok else 'FAIL'}  "
          f"({(delta_DP_n[i_centre+1] - delta_neighbour_pred)/delta_neighbour_pred:+.1%})")
    print(f"  graph-Laplacian centre < isolated?         "
          f"{'PASS' if gl_stiffens_ok else 'FAIL'}")
    print(f"  distance-pres centre < isolated?           "
          f"{'PASS' if dp_softens_ok else 'FAIL'}")
    print(f"  distance-pres neighbours same sign as k?   "
          f"{'PASS' if dp_neighbour_same_sign else 'FAIL'}")

    # Plot overlay.
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = np.arange(lat.N) - i_centre
    ax.plot(offsets, delta_GL_n * 1e6, "o-", color="C3", ms=6,
            label="graph-Laplacian: load shared with neighbours")
    ax.plot(offsets, delta_DP_n * 1e6, "s-", color="C0", ms=6,
            label="distance-preserving: localised at centre")
    ax.axhline(iso_pred * 1e6, ls="--", color="gray",
               label=rf"isolated: $k_c \varphi/(k_a+k_c) = {iso_pred*1e6:.1f}\,\mu$m")
    ax.set_xlabel("sphere offset $i - i_{\\mathrm{centre}}$")
    ax.set_ylabel(r"$\delta_n(i)$ [$\mu$m]")
    ax.set_title("Chain contact: graph-Laplacian shares the load,\n"
                 "distance-preserving degenerates to single-sphere "
                 "(1D-chain projector zeros transverse coupling)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "03c_gl_vs_dp_chain_contact.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    ok = (centre_pred_ok and neighbour_pred_ok and gl_stiffens_ok
          and dp_softens_ok and dp_neighbour_same_sign)
    print(f"PART C result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = part_a_chain_contact_basic()
    b = part_b_kl_sweep()
    c = part_c_distance_pres_comparison()
    all_ok = a and b and c
    print()
    print("=" * 72)
    print(f"Step 3 chain-contact test: {'PASS' if all_ok else 'FAIL'}   "
          f"(A={'P' if a else 'F'} B={'P' if b else 'F'} C={'P' if c else 'F'})")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
