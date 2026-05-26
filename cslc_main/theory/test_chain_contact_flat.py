# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 / T-F — Chain pad lattice vs a flat target face.

The v2 successor to ``test_03_chain_contact.py``.  A 1-D chain of pad
lattice spheres pressed against ONE flat target sample.  The locality
kernel (half-width ``3·r_pad`` by default; we use a tighter value here
so only the centre pad sphere falls inside it) restricts contact to
the single pad sphere directly below the sample, reducing the system
to the rank-1 Sherman-Morrison update familiar from v1::

    K δ + k_c (e_k e_k^T) δ  =  k_c d e_k
    ⇒  δ_n_k  =  k_c · d · g_kk / (1 + k_c · g_kk)

where ``K = K_lattice + k_a·I`` is the scalar lattice stiffness from
:func:`cslc_main.theory.cslc_lattice.build_K_matrix`, ``g_kk =
(K^-1)_kk``, and ``d = r - n_face · (p - t_sample)`` is the rest
half-space overlap.

Three parts:

  PART A.  Sherman-Morrison agreement.  v2 ``solve_lattice_contact``
           equilibrium δ_n at the contact sphere vs S-M closed form
           across three k_l/k_a ratios.  Also overlay a DIRECT LINEAR
           SOLVE of the same rank-1 system as an fp-precision reference
           (the v1 test_03 used this; the v2 nonlinear solver bottoms
           out a few orders of magnitude higher due to L-BFGS-B's
           stopping criteria — see T-C/D production-eps budget for the
           related precision floor discussion).

  PART B.  Force balance.  Sum of anchor forces along the chain
           ``Σ_i k_a · |δ_i|`` equals the contact force at the target
           ``k_c · (d - δ_n_k)``.  This is Newton's third law expressed
           through the equilibrium state.

  PART C.  ``k_l`` sweep.  As ``k_l → 0`` the lattice decouples and
           the centre sphere acts in isolation (δ_n_k → k_c·d/(k_a+k_c),
           the single-sphere series spring).  As ``k_l → ∞`` the chain
           becomes rigid (uniform δ across all spheres; the contact force
           saturates at ``k_c·d`` as the centre sphere stops moving).

Per ``cslc_main/theory/contract_v2.md`` §6, §12 T-F.

Run::

    uv run -m cslc_main.theory.test_chain_contact_flat

Outputs::

    cslc_main/theory/figures/tf_sherman_morrison.png
    cslc_main/theory/figures/tf_kl_sweep.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    build_K_matrix,
    make_chain,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import PointSetTargetV2

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Contract §12 T-F tolerance.  L-BFGS-B floors at ~1e-8 rel err on
# multi-sphere problems even with eps = 1e-12 and gtol = 1e-16; the v1
# test hit 1e-12 only because it used a direct linear solve, which the
# v2 nonlinear path doesn't.  We carry a direct-linear reference in
# Part A so the fp-precision agreement is still visible alongside the
# v2 solver.  Worst rows are at low k_c (small forces, small δ — the
# absolute-noise floor dominates the relative error).
TOL_SM_REL = 1.0e-7


# Canonical setup.  Chain spacing > kernel half-width so only the
# centre pad sphere is "seen" by the single target sample.
N = 7                  # odd so there's a true centre
H = 8.0e-3             # chain spacing [m]  (> 3·r = 7.5 mm → neighbours
                       # are outside the locality kernel of half-width 7 mm)
R_PAD = 2.5e-3         # pad sphere radius [m]
KERNEL_HW = 7.0e-3     # locality kernel half-width [m]  (< H)
KA = 25_000.0          # anchor stiffness [N/m]
D_REST = 1.0e-3        # rest half-space overlap [m]
EPS = 1.0e-12          # analytic-grade smoothing (use tight to avoid
                       # the T-C/D production-eps surrogate floor here;
                       # Phase 4 bridge tests run at production eps).


def build_scene(kl: float, kc: float):
    """Return (lat, target, k_sphere_idx) for the canonical T-F scene."""
    lat = make_chain(N, H, ka=KA, kl=kl)
    k = N // 2
    # Pad outward normal is +y (set by make_chain); place target
    # face-on at distance (r - d_rest) along +y from p_k.
    t_centre = lat.p[k] + (R_PAD - D_REST) * lat.n[k]
    target = PointSetTargetV2(
        positions=t_centre[None, :],
        normals=(-lat.n[k])[None, :],         # face normal toward pad
        areas=np.array([1.0]),                # unit area for direct
                                              # series-spring comparison
    )
    return lat, target, k


def sherman_morrison_dn_k(lat, kc: float, k: int) -> float:
    """Closed-form δ_n_k = kc · d · g_kk / (1 + kc · g_kk)."""
    K = build_K_matrix(lat)
    g_kk = float(np.linalg.inv(K)[k, k])
    return kc * D_REST * g_kk / (1.0 + kc * g_kk)


def direct_linear_dn_k(lat, kc: float, k: int) -> float:
    """Direct linear solve of (K + kc · e_k e_k^T) δ = kc · d · e_k.

    fp-precision reference for the v2 nonlinear solver.
    """
    K = build_K_matrix(lat)
    A_sys = K.copy()
    A_sys[k, k] += kc
    rhs = np.zeros(lat.N)
    rhs[k] = kc * D_REST
    delta_n_axis = np.linalg.solve(A_sys, rhs)
    return float(delta_n_axis[k])


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Sherman-Morrison agreement across k_l/k_a regimes
# ─────────────────────────────────────────────────────────────────────────


def part_a_sherman_morrison() -> bool:
    print()
    print("=" * 78)
    print("PART A.  Sherman-Morrison: v2 nonlinear vs closed form vs direct")
    print("=" * 78)
    print()
    print(f"  Chain: N = {N}, h = {H*1e3:.1f} mm, k_a = {KA:.0f} N/m, "
          f"r_pad = {R_PAD*1e3:.1f} mm")
    print(f"  Target: 1 flat face sample face-on at sphere {N//2}, "
          f"d = {D_REST*1e3:.1f} mm")
    print(f"  Locality kernel half-width = {KERNEL_HW*1e3:.1f} mm "
          f"(< spacing = {H*1e3:.1f} mm  ⇒  only centre pad engages)")
    print()
    print(f"  {'k_l/k_a':>9} {'k_c/k_a':>9} {'δ_SM[μm]':>11} "
          f"{'δ_linear[μm]':>14} {'δ_v2[μm]':>11} "
          f"{'rel(v2 vs SM)':>14}  result")
    print("  " + "-" * 95)

    sm_pass = True
    for kl_ratio in [0.2, 1.0, 10.0]:
        kl = KA * kl_ratio
        for kc_ratio in [0.1, 1.0, 10.0]:
            kc = KA * kc_ratio
            lat, target, k = build_scene(kl, kc)
            dn_SM = sherman_morrison_dn_k(lat, kc, k)
            dn_lin = direct_linear_dn_k(lat, kc, k)
            # v2 nonlinear: warm-start from S-M projection for fast convergence.
            delta0 = np.zeros((lat.N, 3))
            delta0[k] = dn_SM * lat.n[k]
            delta, info = solve_lattice_contact(
                lat, target, kc=kc, r_pad=R_PAD,
                kernel_half_width=KERNEL_HW,
                eps=EPS, tol=1.0e-14, maxiter=20000,
                delta0=delta0)
            dn_v2 = float(np.dot(delta[k], lat.n[k]))
            rel = abs(dn_v2 - dn_SM) / max(abs(dn_SM), 1e-30)
            ok = rel < TOL_SM_REL
            sm_pass = sm_pass and ok
            print(f"  {kl_ratio:>9.2f} {kc_ratio:>9.2f} "
                  f"{dn_SM*1e6:>11.6f} {dn_lin*1e6:>14.9f} {dn_v2*1e6:>11.6f} "
                  f"{rel:>14.2e}  {'PASS' if ok else 'FAIL'}")

    print()
    print(f"PART A: {'PASS' if sm_pass else 'FAIL'}  "
          f"(tol rel < {TOL_SM_REL:.0e})")
    print(f"  Note: direct-linear column shows ~1e-16 agreement with S-M "
          f"(fp limit);")
    print(f"  v2 nonlinear is ~7 orders worse due to L-BFGS-B stopping "
          f"behaviour at this scale.")
    print(f"  This is the inherent precision of the iterative path the "
          f"production kernel uses.")

    # ── Figure ──
    # Sweep kc on a wider grid for plotting.
    kl_for_plot = KA
    kc_dense = np.logspace(np.log10(KA / 100.0), np.log10(KA * 100.0), 32)
    dn_SM_curve = np.zeros_like(kc_dense)
    dn_v2_curve = np.zeros_like(kc_dense)
    for i, kc in enumerate(kc_dense):
        lat, target, k = build_scene(kl_for_plot, kc)
        dn_SM_curve[i] = sherman_morrison_dn_k(lat, kc, k)
        delta0 = np.zeros((lat.N, 3))
        delta0[k] = dn_SM_curve[i] * lat.n[k]
        delta, _ = solve_lattice_contact(
            lat, target, kc=kc, r_pad=R_PAD,
            kernel_half_width=KERNEL_HW,
            eps=EPS, tol=1.0e-14, maxiter=20000, delta0=delta0)
        dn_v2_curve[i] = float(np.dot(delta[k], lat.n[k]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(kc_dense / KA, dn_SM_curve * 1e6, "k-", lw=2,
                label=r"Sherman-Morrison closed form")
    ax.semilogx(kc_dense / KA, dn_v2_curve * 1e6, "ro", ms=6, mfc="none",
                label=r"v2 ``solve_lattice_contact``")
    ax.axhline(D_REST * 1e6, ls="--", color="gray",
               label=rf"$d = {D_REST*1e3:.1f}$ mm (rigid contact limit)")
    ax.set_xlabel(r"$k_c / k_a$")
    ax.set_ylabel(r"$\delta_{n,k}$ [$\mu$m] at the contact sphere")
    ax.set_title("T-F/A. Chain + 1 flat contact: v2 vs Sherman-Morrison\n"
                 rf"($N = {N}$, $h = {H*1e3:.1f}$ mm, "
                 rf"$k_l = k_a = {KA:.0f}$ N/m, locality kernel cuts "
                 "off neighbour-pad contacts)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG_DIR / "tf_sherman_morrison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")
    return sm_pass


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Force balance: anchor sum == contact reaction
# ─────────────────────────────────────────────────────────────────────────


def part_b_force_balance() -> bool:
    """Verify Σ_i ka·|δ_i| (anchor reaction) = kc·(d - δ_n_k) (contact reaction).

    At equilibrium, the chain's anchor forces along the pad normal axis
    must equal the contact force pushing the contact sphere inward —
    Newton III on the lattice frame.
    """
    print()
    print("=" * 78)
    print("PART B.  Force balance: Σ anchor reaction == contact reaction")
    print("=" * 78)
    print()
    print(f"  {'k_l/k_a':>9} {'k_c/k_a':>9} "
          f"{'Σ_i F_anc[N]':>14} {'k_c·(d-δ_k)[N]':>15} "
          f"{'rel diff':>10}  result")
    print("  " + "-" * 76)

    fb_pass = True
    for kl_ratio in [0.2, 1.0, 10.0]:
        kl = KA * kl_ratio
        for kc_ratio in [0.1, 1.0, 10.0]:
            kc = KA * kc_ratio
            lat, target, k = build_scene(kl, kc)
            dn_SM = sherman_morrison_dn_k(lat, kc, k)
            delta0 = np.zeros((lat.N, 3))
            delta0[k] = dn_SM * lat.n[k]
            delta, _ = solve_lattice_contact(
                lat, target, kc=kc, r_pad=R_PAD,
                kernel_half_width=KERNEL_HW,
                eps=EPS, tol=1.0e-14, maxiter=20000, delta0=delta0)

            # Anchor reaction: sum of per-sphere anchor forces.  At
            # face-on (all spheres displace along pad normal n_pad =
            # +y), |F_anc_i · n_pad| = ka · δ_n_i.
            delta_n = np.einsum("nj,nj->n", delta, lat.n)
            anchor_force_total = float(np.sum(KA * np.abs(delta_n)))
            # Contact reaction at the target: kc · (d - δ_n_k).
            dn_k = float(delta_n[k])
            contact_force = kc * (D_REST - dn_k)
            rel = abs(anchor_force_total - contact_force) / max(contact_force, 1e-30)
            ok = rel < TOL_SM_REL
            fb_pass = fb_pass and ok
            print(f"  {kl_ratio:>9.2f} {kc_ratio:>9.2f} "
                  f"{anchor_force_total:>14.6f} {contact_force:>15.6f} "
                  f"{rel:>10.2e}  {'PASS' if ok else 'FAIL'}")

    print()
    print(f"PART B: {'PASS' if fb_pass else 'FAIL'}  "
          f"(tol rel < {TOL_SM_REL:.0e})")
    return fb_pass


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  k_l sweep: isolated → rigid limits
# ─────────────────────────────────────────────────────────────────────────


def part_c_kl_sweep() -> bool:
    """k_l sweep showing the two limiting regimes.

      * k_l → 0:  chain decouples; centre sphere alone resists.
                  δ_n_k → k_c · d / (k_a + k_c) (single-sphere series).
      * k_l → ∞:  chain rigid; all N spheres yield equally with
                  δ_uniform = k_c·d / (N·k_a + k_c) (N anchors in
                  parallel, NOT a truly rigid pad), and the contact
                  force saturates at
                      F_rigid = k_c · d · (N·k_a) / (N·k_a + k_c)
                  (= k_c · d only if N → ∞ as well).
    """
    print()
    print("=" * 78)
    print("PART C.  k_l sweep: isolated → N-fold anchor-bound limit")
    print("=" * 78)
    print()

    kc = KA   # fix at k_c = k_a
    kl_ratios = np.logspace(-3, 3, 13)
    dn_centre = np.zeros_like(kl_ratios)
    F_contact = np.zeros_like(kl_ratios)

    # Isolated limit: only centre sphere yields.
    iso_dn = kc * D_REST / (KA + kc)
    iso_F = kc * (D_REST - iso_dn)        # = ka·kc·d/(ka+kc) = k_eff·d
    # Rigid-pad limit: N spheres in parallel (each via its anchor).
    rigid_dn = kc * D_REST / (N * KA + kc)
    rigid_F = kc * (D_REST - rigid_dn)

    for i, kl_ratio in enumerate(kl_ratios):
        kl = KA * kl_ratio
        lat, target, k = build_scene(kl, kc)
        dn_SM = sherman_morrison_dn_k(lat, kc, k)
        delta0 = np.zeros((lat.N, 3))
        delta0[k] = dn_SM * lat.n[k]
        delta, _ = solve_lattice_contact(
            lat, target, kc=kc, r_pad=R_PAD,
            kernel_half_width=KERNEL_HW,
            eps=EPS, tol=1.0e-14, maxiter=20000, delta0=delta0)
        dn_centre[i] = float(np.dot(delta[k], lat.n[k]))
        F_contact[i] = kc * (D_REST - dn_centre[i])

    # Asymptote checks.
    iso_err = abs(dn_centre[0] - iso_dn) / iso_dn   # at k_l/k_a = 1e-3
    rigid_err = abs(F_contact[-1] - rigid_F) / rigid_F
    iso_ok = iso_err < 0.05            # within 5% (low-k_l asymptote)
    rigid_ok = rigid_err < 0.05        # within 5% (high-k_l asymptote)

    print(f"  k_l → 0 limit:  δ_n_k = k_c·d/(k_a+k_c) = {iso_dn*1e6:.3f} μm")
    print(f"     measured at k_l/k_a = {kl_ratios[0]:.0e}:  "
          f"{dn_centre[0]*1e6:.3f} μm  (rel err {iso_err:.2e})  "
          f"{'PASS' if iso_ok else 'FAIL'}")
    print(f"  k_l → ∞ limit (N={N} anchors in parallel):")
    print(f"     F_contact → k_c·d·N·k_a/(N·k_a + k_c) = {rigid_F:.4f} N")
    print(f"     measured at k_l/k_a = {kl_ratios[-1]:.0e}: "
          f"{F_contact[-1]:.4f} N  (rel err {rigid_err:.2e})  "
          f"{'PASS' if rigid_ok else 'FAIL'}")

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogx(kl_ratios, dn_centre * 1e6, "ko-", ms=6, lw=1.5,
                 label=r"v2 numerical")
    ax1.axhline(iso_dn * 1e6, ls="--", color="gray",
                label=rf"$k_c d / (k_a + k_c)$ (isolated)")
    ax1.axhline(0.0, ls=":", color="C2",
                label=r"$\delta \to 0$ (rigid limit)")
    ax1.set_xlabel(r"$k_l / k_a$")
    ax1.set_ylabel(r"$\delta_{n,k}$ at contact sphere [$\mu$m]")
    ax1.set_title("Centre sphere yields LESS as $k_l$ grows\n"
                  "(load shared with neighbours)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    ax2.semilogx(kl_ratios, F_contact, "ko-", ms=6, lw=1.5)
    ax2.axhline(rigid_F, ls="--", color="gray",
                label=rf"$k_c d \,(Nk_a)/(Nk_a+k_c) = {rigid_F:.3f}$ N "
                      "(rigid-pad limit)")
    ax2.axhline(iso_F, ls=":", color="C2",
                label=rf"$k_{{eff}} d = {iso_F:.3f}$ N (isolated)")
    ax2.set_xlabel(r"$k_l / k_a$")
    ax2.set_ylabel(r"contact force $F_{\mathrm{contact}}$ [N]")
    ax2.set_title(r"Contact force saturates at $k_c d\,(Nk_a)/(Nk_a+k_c)$"
                  "\n(N anchors in parallel resist the contact)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")
    fig.suptitle(rf"T-F/C. $k_l$ sweep at $k_c = k_a = {KA:.0f}$ N/m, "
                 rf"$d = {D_REST*1e3:.1f}$ mm")
    fig.tight_layout()
    out = FIG_DIR / "tf_kl_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n  Saved {out}")
    return iso_ok and rigid_ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a_ok = part_a_sherman_morrison()
    b_ok = part_b_force_balance()
    c_ok = part_c_kl_sweep()
    print()
    print("=" * 78)
    print(f"T-F overall: {'PASS' if (a_ok and b_ok and c_ok) else 'FAIL'}  "
          f"(A={'P' if a_ok else 'F'} B={'P' if b_ok else 'F'} "
          f"C={'P' if c_ok else 'F'})")
    print("=" * 78)
    return 0 if (a_ok and b_ok and c_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
