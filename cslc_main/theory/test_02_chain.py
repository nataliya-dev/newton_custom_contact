# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 2 / 1D chain of lattice spheres with anchor + lateral springs.

No contact target here -- step 2 is the lateral-coupling reference.
Step 3 will add a target and watch how the lateral spreads the contact
load across the patch.

A chain of N spheres along the x-axis at spacing h:

    p_i = (i*h, 0, 0)              i = 0, 1, ..., N-1
    n_i = +y                       (pad-like outward normal)
    edges = {(i, i+1) : i = 0..N-2}

Each sphere has an anchor spring (stiffness k_a) tethering its centre
q_i = p_i - delta_i to its rest position p_i.  Adjacent spheres are
connected by lateral springs (stiffness k_l).

Two lateral laws are tested side by side (see cslc_lattice.py):
  * graph-Laplacian:   f_lat(i,j) = -k_l * (delta_i - delta_j)
  * distance-preserving: f_lat(i,j) = -k_l * (||q_j - q_i|| - L_ij) * e_hat_ij

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
           recover the lateral correlation length
              l_c = sqrt(k_l / k_a)
           in lattice-spacing units.  Sweep three regimes
              k_l/k_a = 0.2 (sub-grid, production CSLC default)
              k_l/k_a = 1   (well-coupled, l_c = 1 spacing)
              k_l/k_a = 10  (strongly coupled, l_c ~ 3 spacings)
           to make the regime-dependence visible.

  PART D.  The DIAGNOSTIC test that distinguishes graph-Laplacian from
           distance-preserving.  Apply f_ext = F * y_hat (perpendicular
           to the chain axis) at the centre sphere.  At small delta,
           the distance-preserving linearisation projects onto the
           rest edge direction (here, x_hat), so y-components of delta
           do NOT couple to neighbours.  Predicted result:
              graph-Laplacian: delta_y(i) spreads with l_c, same Green's
                  function as Part C (axes decouple).
              distance-pres:   delta_y(i_centre) = F / k_a (anchor only),
                                delta_y(i != centre) = 0.

Run::

    uv run -m cslc_main.theory.test_02_chain

Outputs::

    cslc_main/theory/figures/02a_K_matrix.txt          (printed matrix)
    cslc_main/theory/figures/02b_eigenvalue_spectrum.png
    cslc_main/theory/figures/02c_greens_function_axial.png
    cslc_main/theory/figures/02d_perpendicular_diagnostic.png
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
    solve_equilibrium_numerical,
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
#  PART D.  Perpendicular load: graph-Laplacian vs distance-preserving
# ─────────────────────────────────────────────────────────────────────────


def part_d_perpendicular_diagnostic() -> bool:
    """Apply f_ext in +y on the chain centre; compare lateral laws.

    Graph-Laplacian:  delta_y spreads with l_c = sqrt(kl/ka) (axes decouple).
    Distance-pres:    only the centre yields; delta_y(centre) = F / k_a.

    Quantitative test:
      * Distance-pres distant-sphere |delta_y| should be << anchor-only
        prediction (within a per-mille of zero in our default setup).
      * Distance-pres centre |delta_y| should equal F / k_a to within
        the second-order correction k_l * delta_y^2 / (k_a * h^2).
      * Graph-Laplacian centre |delta_y| should be SMALLER than F/k_a
        (load shares).
    """
    print()
    print("=" * 72)
    print("PART D.  Perpendicular-load diagnostic")
    print("         graph-Laplacian spreads | distance-preserving localises")
    print("=" * 72)

    N = 21
    ka = 25_000.0
    kl = 5_000.0          # production-ish ratio kl/ka = 0.2
    h = 2.5e-3
    i_centre = N // 2
    F_ext = 1.0           # 1 N in +y at centre

    lat = make_chain(N=N, h=h, ka=ka, kl=kl)
    f_ext = np.zeros((N, 3))
    f_ext[i_centre, 1] = F_ext

    # Graph-Laplacian: closed-form linear solve.
    delta_GL = solve_equilibrium_graph_laplacian(lat, f_ext)

    # Distance-preserving: nonlinear minimisation.
    delta_DP, info = solve_equilibrium_numerical(
        lat, f_ext, lateral="distance_preserving",
        delta0=delta_GL,  # warm-start from the linear answer
        tol=1e-12,
    )

    # Linearised prediction for distance-pres: anchor alone resists.
    delta_lin_DP = np.zeros_like(delta_GL)
    delta_lin_DP[i_centre, 1] = F_ext / ka

    print()
    print(f"  N = {N}   k_a = {ka:.0f}   k_l = {kl:.0f}   F_ext_y = {F_ext} N at centre")
    print(f"  l_c = sqrt(k_l/k_a) = {np.sqrt(kl/ka):.3f} spacings")
    print()
    print(f"  centre delta_y prediction:")
    print(f"    anchor only         = F / k_a            "
          f"= {F_ext / ka * 1e6:>8.3f} um")
    print(f"    graph-Laplacian num = (K^-1)_cc * F      "
          f"= {delta_GL[i_centre, 1] * 1e6:>8.3f} um   "
          f"(< anchor-only because load spreads)")
    print(f"    distance-pres num   = numerical optimum  "
          f"= {delta_DP[i_centre, 1] * 1e6:>8.3f} um   "
          f"(= anchor-only to leading order)")
    print()

    # Quantitative checks.
    eps_GL_lt = delta_GL[i_centre, 1] < F_ext / ka            # GL shares load
    eps_DP_eq = abs(delta_DP[i_centre, 1] - F_ext / ka) / (F_ext / ka)
    eps_DP_neighbours = float(np.max(
        np.abs(delta_DP[i_centre - 1: i_centre + 2, 1] -
               delta_lin_DP[i_centre - 1: i_centre + 2, 1])
    )) / (F_ext / ka)

    print(f"  graph-Laplacian centre < anchor-only?              "
          f"{'YES' if eps_GL_lt else 'NO'}")
    print(f"  distance-pres rel deviation from F/k_a at centre  "
          f"{eps_DP_eq:>8.2e}    (want < 1e-3)")
    print(f"  distance-pres max |delta_y(i) - lin_pred| / (F/ka)"
          f" {eps_DP_neighbours:>8.2e}    (want < 1e-3)")

    d_ok = eps_GL_lt and eps_DP_eq < 1e-3 and eps_DP_neighbours < 1e-3
    print()
    print(f"PART D result: {'PASS' if d_ok else 'FAIL'}")

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = np.arange(N) - i_centre
    ax.plot(offsets, delta_GL[:, 1] * 1e6, "o-", color="C3", lw=1.5, ms=6,
            label="graph-Laplacian: load spreads")
    ax.plot(offsets, delta_DP[:, 1] * 1e6, "s-", color="C0", lw=1.5, ms=6,
            label="distance-preserving: localised")
    ax.axhline(F_ext / ka * 1e6, ls="--", color="gray",
               label=rf"$F/k_a = {F_ext/ka*1e6:.1f}\,\mu$m (anchor-only)")
    ax.axvline(0, ls=":", color="black", alpha=0.5)
    ax.set_xlabel("sphere index offset $i - i_{\\mathrm{centre}}$")
    ax.set_ylabel(r"$\delta_y(i)$ [$\mu$m]")
    ax.set_title("Perpendicular load: distance-preserving lateral leaves "
                 "$\\delta_y$ uncoupled\n"
                 f"$k_l/k_a = {kl/ka:.2f}$, $F_y = {F_ext}$ N at centre")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "02d_perpendicular_diagnostic.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return d_ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = part_a_K_matrix()
    b = part_b_eigenvalue_spectrum()
    c = part_c_axial_greens_function()
    d = part_d_perpendicular_diagnostic()
    all_ok = a and b and c and d
    print()
    print("=" * 72)
    print(f"Step 2 chain test: {'PASS' if all_ok else 'FAIL'}   "
          f"(A={'P' if a else 'F'} B={'P' if b else 'F'} "
          f"C={'P' if c else 'F'} D={'P' if d else 'F'})")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
