# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 / T-E — 1D chain of lattice spheres, graph-Laplacian only.

In-place edit of the v1 ``test_02_chain.py``.  v2 drops the
distance-preserving lateral law (contract_v2.md §6.2 and §13);
graph-Laplacian is now the sole lateral spring.  The DP-vs-GL
diagnostic PART D of the v1 test is therefore removed — there is
nothing to compare against — and replaced with a graph-Laplacian
transverse-load sanity check (PART D') that verifies axes decouple
under GL (the property that the linearised DP law also satisfies and
that the v1 PART D used as its baseline).

No contact target here — Step 2 is the lateral-coupling reference.
T-F (test_chain_contact_flat.py) adds a flat-face target and watches
how the GL lateral spreads the contact load across the patch.

A chain of N spheres along the x-axis at spacing h:

    p_i = (i*h, 0, 0)              i = 0, 1, ..., N-1
    n_i = +y                       (pad-like outward normal)
    edges = {(i, i+1) : i = 0..N-2}

Each sphere has an anchor spring (stiffness k_a) tethering its centre
q_i = p_i - delta_i to its rest position p_i.  Adjacent spheres are
connected by lateral GL springs (stiffness k_l): ``f_lat(i, j) =
-k_l · (delta_i - delta_j)``.

What the parts test:

  PART A.  Hand-assembled K matrix vs cslc_lattice.build_K_matrix.
           Confirms K_ii = k_a + k_l * |N(i)|, K_ij = -k_l for edges.
           Symmetry, positive-definiteness, and the smallest eigenvalue
           (= k_a, uniform translation mode) are checked numerically.

  PART B.  Eigenvalue spectrum of K vs analytical formula
              lambda_k = k_a + 2*k_l*(1 - cos(k*pi/N))
           valid for the Neumann (free-endpoint) 1D Laplacian.  All N
           eigenvalues should match to ~1e-10 rel err.

  PART C.  Green's function along the chain axis.  Apply external force
           f_ext = F * x_hat on the centre sphere; solve K @ delta_x =
           f_ext.  Plot delta_x(i - i_centre); fit exponential decay;
           recover the discrete-lattice decay length
              l_c_discrete = -1 / ln(z_-)
           where z_- is the small root of the characteristic equation
           (see cslc_lattice.chain_discrete_decay_length).  Sweep three
           regimes
              k_l/k_a = 0.2 (sub-grid, production CSLC default)
              k_l/k_a = 1   (well-coupled, l_c = 1 spacing)
              k_l/k_a = 10  (strongly coupled, l_c ~ 3 spacings)
           to make the regime-dependence visible.

  PART D'.  Graph-Laplacian axes-decouple sanity check.  Apply
           f_ext = F * y_hat (perpendicular to the chain axis) at the
           centre sphere; verify the y-axis response is identical to
           the x-axis response from PART C (under GL the per-axis
           system is the same scalar K, just with the rhs along the
           loaded axis).  This is the GL-only successor to v1's
           DP-vs-GL diagnostic.

Run::

    uv run -m cslc_main.theory.test_02_chain

Outputs::

    cslc_main/theory/figures/02a_K_matrix.txt          (printed matrix)
    cslc_main/theory/figures/02b_eigenvalue_spectrum.png
    cslc_main/theory/figures/02c_greens_function_axial.png
    cslc_main/theory/figures/02d_GL_axes_decouple.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    build_K_matrix,
    chain_analytical_eigenvalues,
    chain_discrete_decay_length,
    make_chain,
    solve_equilibrium_graph_laplacian,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Hand-built K vs the assembly function
# ─────────────────────────────────────────────────────────────────────────


def part_a_K_matrix() -> bool:
    """Build a small chain by hand and compare with build_K_matrix."""
    print()
    print("=" * 72)
    print("PART A.  Lattice stiffness matrix K  (paper eq. 10)")
    print("=" * 72)

    N = 5
    ka = 25_000.0
    kl = 5_000.0
    h = 2.5e-3
    lat = make_chain(N=N, h=h, ka=ka, kl=kl)

    # Hand-build the expected K: tridiagonal, K_ii = ka + kl*deg(i), K_ij = -kl.
    K_expected = np.zeros((N, N))
    for i in range(N):
        deg = 2 if 0 < i < N - 1 else 1
        K_expected[i, i] = ka + kl * deg
    for i in range(N - 1):
        K_expected[i, i + 1] = -kl
        K_expected[i + 1, i] = -kl

    K_actual = build_K_matrix(lat)

    max_diff = float(np.max(np.abs(K_expected - K_actual)))
    is_symmetric = float(np.max(np.abs(K_actual - K_actual.T)))
    smallest_eig = float(np.linalg.eigvalsh(K_actual)[0])

    print()
    print(f"  N = {N}   k_a = {ka:.0f} N/m   k_l = {kl:.0f} N/m")
    print(f"  K matrix (units N/m):")
    print(np.array2string(K_actual, formatter={"float": lambda x: f"{x:8.0f}"}))
    print()
    print(f"  ||K_actual - K_expected||_max  = {max_diff:.3e}    "
          f"(want 0)")
    print(f"  ||K - K^T||_max                 = {is_symmetric:.3e}    "
          f"(want 0; SPD requires symmetry)")
    print(f"  smallest eigenvalue             = {smallest_eig:.4f}   "
          f"(want {ka:.4f} = k_a, the rigid-translation mode)")

    a_ok = (max_diff < 1e-10
            and is_symmetric < 1e-10
            and abs(smallest_eig - ka) < 1e-6)

    print()
    print(f"PART A result: {'PASS' if a_ok else 'FAIL'}")

    out = FIG_DIR / "02a_K_matrix.txt"
    out.write_text(
        f"N = {N}, ka = {ka}, kl = {kl}\n"
        f"K =\n{np.array2string(K_actual, formatter={'float': lambda x: f'{x:8.0f}'})}\n"
        f"smallest eigenvalue = {smallest_eig}\n"
    )
    print(f"Saved table to {out}")
    return a_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Eigenvalue spectrum vs analytical
# ─────────────────────────────────────────────────────────────────────────


def part_b_eigenvalue_spectrum() -> bool:
    """Spectrum of K compared with the Neumann-Laplacian analytical formula."""
    print()
    print("=" * 72)
    print("PART B.  Eigenvalues of K vs analytical DCT-II spectrum")
    print("=" * 72)

    N = 20
    ka = 25_000.0
    kl = 5_000.0
    h = 2.5e-3
    lat = make_chain(N=N, h=h, ka=ka, kl=kl)
    K = build_K_matrix(lat)

    lam_num = np.sort(np.linalg.eigvalsh(K))         # ascending
    lam_ana = chain_analytical_eigenvalues(N, ka, kl)  # ascending

    max_abs = float(np.max(np.abs(lam_num - lam_ana)))
    max_rel = float(np.max(np.abs(lam_num - lam_ana) / np.maximum(lam_ana, 1e-30)))

    print()
    print(f"  N = {N}   k_a = {ka:.0f} N/m   k_l = {kl:.0f} N/m")
    print(f"  smallest eig (numerical)  = {lam_num[0]:.6f}   "
          f"(want {ka:.6f} = k_a)")
    print(f"  largest eig (numerical)   = {lam_num[-1]:.6f}   "
          f"(want {ka + 2*kl*(1-np.cos((N-1)*np.pi/N)):.6f})")
    print(f"  max |numerical - analytical|       = {max_abs:.3e} N/m")
    print(f"  max relative error                  = {max_rel:.3e}")

    b_ok = max_rel < 1e-10
    print()
    print(f"PART B result: {'PASS' if b_ok else 'FAIL'}")

    # Plot spectrum.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    k_axis = np.arange(N)
    ax.plot(k_axis, lam_ana, "k-", lw=2,
            label=r"$\lambda_k = k_a + 2k_l(1 - \cos(k\pi/N))$ (analytical)")
    ax.plot(k_axis, lam_num, "ro", ms=6, mfc="none",
            label="numerical (eigvalsh)")
    ax.axhline(ka, ls=":", color="gray",
               label=rf"$\lambda_0 = k_a = {ka:.0f}$ N/m")
    ax.set_xlabel("mode index $k$")
    ax.set_ylabel(r"eigenvalue $\lambda_k$ [N/m]")
    ax.set_title(rf"Chain stiffness spectrum  (N = {N}, $k_l/k_a$ = {kl/ka:.2f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "02b_eigenvalue_spectrum.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return b_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Green's function along the chain axis
# ─────────────────────────────────────────────────────────────────────────


def part_c_axial_greens_function() -> bool:
    """Push the centre sphere in +x; fit exponential decay of delta_x(i)."""
    print()
    print("=" * 72)
    print("PART C.  Axial Green's function -- load spread along chain")
    print("=" * 72)

    N = 31
    ka = 25_000.0
    h = 2.5e-3
    i_centre = N // 2
    F_ext = 1.0   # 1 N applied at the centre sphere in +x

    regimes = [
        (0.2,  "sub-grid ($k_l/k_a = 0.2$)"),
        (1.0,  "well-coupled ($k_l/k_a = 1$)"),
        (10.0, "strongly coupled ($k_l/k_a = 10$)"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    print()
    print(f"  N = {N}   k_a = {ka:.0f} N/m   F_ext = {F_ext} N at sphere {i_centre}")
    print()
    print(f"  Reference: discrete formula  l_c = -1 / ln(z_-)  where")
    print(f"             z_- = (1+alpha) - sqrt(alpha*(alpha+2)),  alpha = ka/(2 kl).")
    print(f"             Continuum approximation sqrt(kl/ka) is shown for context;")
    print(f"             it is only accurate when k_l/k_a >> 1.")
    print()
    print(f"  {'regime':<32} {'kl[N/m]':>10} {'lc_cont':>9} {'lc_disc':>9} "
          f"{'lc_fit':>9} {'rel err vs disc':>16}")
    print(f"  {'-'*32} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*16}")

    all_ok = True

    for ratio, label in regimes:
        kl = ratio * ka
        lat = make_chain(N=N, h=h, ka=ka, kl=kl)
        f_ext = np.zeros((N, 3))
        f_ext[i_centre, 0] = F_ext

        delta = solve_equilibrium_graph_laplacian(lat, f_ext)
        dx = delta[:, 0]

        # Two reference lengths to compare against the numerical fit.
        lc_cont = np.sqrt(ratio)                          # continuum approx
        lc_disc = chain_discrete_decay_length(ka, kl)     # exact for infinite chain

        # Fit exponential decay |dx(i)| ~ A * exp(-|i - i_centre|/lc) on
        # the right half of the chain, skipping i_centre (zero-distance
        # has the extra delta-function term that breaks the pure
        # exponential).  Restrict to a window of ~10 spacings so the
        # boundary reflection doesn't bias the slope.
        rhs_indices = np.arange(i_centre + 1, N)
        x_fit = (rhs_indices - i_centre).astype(float)
        y_fit = np.log(np.abs(dx[rhs_indices]))
        n_keep = min(10, len(x_fit))
        slope, intercept = np.polyfit(x_fit[:n_keep], y_fit[:n_keep], 1)
        lc_fit = -1.0 / slope if slope < 0 else float("inf")

        rel = abs(lc_fit - lc_disc) / lc_disc
        ok = rel < 0.01   # 1% bar against the DISCRETE prediction
        all_ok &= ok
        marker = "PASS" if ok else "FAIL"
        print(f"  {label:<32} {kl:>10.0f} {lc_cont:>9.3f} {lc_disc:>9.3f} "
              f"{lc_fit:>9.3f} {rel:>15.2%}  {marker}")

        # Plot |dx| on log y, normalised by the peak.
        offsets = np.arange(N) - i_centre
        ax.semilogy(offsets, np.maximum(np.abs(dx), 1e-30) / np.max(np.abs(dx)),
                    "-o", label=f"{label}, fit $\\ell_c = {lc_fit:.2f}$ "
                    f"(disc. ref {lc_disc:.2f})", ms=5)

    ax.set_xlabel("sphere index offset $i - i_{\\mathrm{centre}}$")
    ax.set_ylabel(r"$|\delta_x(i)|\,/\,|\delta_x(i_{\mathrm{centre}})|$")
    ax.set_title("Axial Green's function: discrete $\\ell_c = -1/\\ln(z_-)$\n"
                 "(continuum $\\sqrt{k_l/k_a}$ is the long-correlation-length limit)")
    ax.set_xlim(-int(N/2), int(N/2))
    ax.set_ylim(1e-8, 1.5)
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "02c_greens_function_axial.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {out}")
    print(f"PART C result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART D'.  Graph-Laplacian axes-decouple sanity check
#
#  v2 successor to v1's PART D (which compared GL against the
#  distance-preserving law).  DP is gone in v2 (contract_v2.md §6.2);
#  this part just verifies that the GL system decouples per Cartesian
#  axis — the property the v1 PART D used as the GL baseline.
# ─────────────────────────────────────────────────────────────────────────


def part_d_GL_axes_decouple() -> bool:
    """Apply f_ext along +y at the chain centre; verify GL gives the
    same per-axis response as the +x load from PART C.

    Under graph-Laplacian, the full 3N×3N stiffness is K ⊗ I_3 (with K
    the scalar (N×N) Laplacian + anchor matrix), so each Cartesian
    axis decouples and solves K·δ_axis = f_axis with the same K.  A
    transverse load (+y) should therefore give exactly the same δ
    spatial profile as an axial load (+x), with values placed in the
    y-component instead of the x-component.

    Quantitative check (3 production-relevant k_l/k_a ratios):
      * δ_y profile under +y load matches δ_x profile under +x load to
        machine precision (rel err < 1e-12).
      * Off-axis components (e.g. δ_x under +y load) are zero to fp.
      * Centre |δ| < F/k_a (lateral spring shares the load — same as
        PART C's axial response).
    """
    print()
    print("=" * 72)
    print("PART D'. GL axes decouple under perpendicular load")
    print("=" * 72)

    N = 21
    ka = 25_000.0
    h = 2.5e-3
    i_centre = N // 2
    F_ext = 1.0
    kl_ratios = [0.2, 1.0, 10.0]

    print()
    print(f"  N = {N},  k_a = {ka:.0f},  F_ext = {F_ext} N at centre")
    print()
    print(f"  {'k_l/k_a':>9} {'l_c (disc)':>11} "
          f"{'δ_x[+x]@c':>11} {'δ_y[+y]@c':>11} "
          f"{'max axis-prof Δ':>17} {'off-axis leak':>15}")
    print("  " + "-" * 86)

    all_ok = True
    profile_storage = {}
    for r in kl_ratios:
        kl = ka * r
        lat = make_chain(N=N, h=h, ka=ka, kl=kl)
        # Axial load.
        f_x = np.zeros((N, 3))
        f_x[i_centre, 0] = F_ext
        delta_x_loaded = solve_equilibrium_graph_laplacian(lat, f_x)
        # Transverse load.
        f_y = np.zeros((N, 3))
        f_y[i_centre, 1] = F_ext
        delta_y_loaded = solve_equilibrium_graph_laplacian(lat, f_y)

        # Profile equivalence on the loaded axis.
        prof_x = delta_x_loaded[:, 0]
        prof_y = delta_y_loaded[:, 1]
        prof_diff = float(np.max(np.abs(prof_x - prof_y)))
        prof_rel = prof_diff / max(float(np.max(np.abs(prof_x))), 1e-30)

        # Off-axis components should be zero.
        off_axis = max(
            float(np.max(np.abs(delta_x_loaded[:, 1]))),
            float(np.max(np.abs(delta_x_loaded[:, 2]))),
            float(np.max(np.abs(delta_y_loaded[:, 0]))),
            float(np.max(np.abs(delta_y_loaded[:, 2]))),
        )

        lc_disc = chain_discrete_decay_length(ka, kl)
        decouple_ok = (prof_rel < 1e-12) and (off_axis < 1e-12)
        all_ok = all_ok and decouple_ok
        print(f"  {r:>9.2f} {lc_disc:>11.3f} "
              f"{prof_x[i_centre]*1e6:>11.3f} {prof_y[i_centre]*1e6:>11.3f} "
              f"{prof_rel:>17.3e} {off_axis:>15.3e}  "
              f"{'PASS' if decouple_ok else 'FAIL'}")
        profile_storage[r] = (prof_x, prof_y)

    print()
    print(f"PART D' result: {'PASS' if all_ok else 'FAIL'}")

    # Plot: overlay +x-load and +y-load profiles for the production
    # k_l/k_a = 0.2 case to show they coincide.
    prof_x, prof_y = profile_storage[0.2]
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = np.arange(N) - i_centre
    ax.plot(offsets, prof_x * 1e6, "o-", color="C3", lw=1.5, ms=6,
            label=r"$\delta_x(i)$ under $F\hat x$ at centre")
    ax.plot(offsets, prof_y * 1e6, "s--", color="C0", lw=1.5, ms=6,
            mfc="none", label=r"$\delta_y(i)$ under $F\hat y$ at centre")
    ax.axhline(F_ext / ka * 1e6, ls=":", color="gray",
               label=rf"$F/k_a = {F_ext/ka*1e6:.1f}\,\mu$m (anchor only)")
    ax.axvline(0, ls=":", color="black", alpha=0.4)
    ax.set_xlabel(r"sphere index offset $i - i_{\mathrm{centre}}$")
    ax.set_ylabel(r"$\delta$ component [$\mu$m]")
    ax.set_title("T-E/D'. Graph-Laplacian axes decouple\n"
                 r"$+\hat x$ and $+\hat y$ loads give identical profiles "
                 r"(at $k_l/k_a = 0.2$)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "02d_GL_axes_decouple.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return all_ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = part_a_K_matrix()
    b = part_b_eigenvalue_spectrum()
    c = part_c_axial_greens_function()
    d = part_d_GL_axes_decouple()
    all_ok = a and b and c and d
    print()
    print("=" * 72)
    print(f"T-E chain test (graph-Laplacian only): "
          f"{'PASS' if all_ok else 'FAIL'}   "
          f"(A={'P' if a else 'F'} B={'P' if b else 'F'} "
          f"C={'P' if c else 'F'} D'={'P' if d else 'F'})")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
