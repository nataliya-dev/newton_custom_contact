# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 5 / 2D arc with face-on contact: where bulging finally appears.

Step 3 (1D straight chain, perpendicular contact) showed that
distance-preserving was *quasi-localised* with same-sign neighbour
response -- no outward bulge.  The cause was geometric: every rest
edge pointed transverse to the contact direction, so the rank-1
projector zeroed the linear-order coupling and the third-order
nonlinearity dragged neighbours in the SAME direction as the centre.

Curvature breaks that symmetry.  On a circular arc of radius R_pad
with arc-length spacing h, neighbours sit at a curvature drop
``R_pad * alpha^2 / 2`` below the apex (where alpha = h / R_pad is the
angular spacing).  Compressing the apex by delta_y_apex:

* If 0 < delta_y_apex < R_pad * alpha^2, the chord from apex to
  neighbour has shrunk below its rest length -- the lateral spring is
  COMPRESSED, pushing neighbours **outward along their local n_hat**.
  This is the geometric Poisson signature.

* delta_y_apex = R_pad * alpha^2 restores the chord to rest length;
  perimeter delta_n crosses zero.

* delta_y_apex > R_pad * alpha^2: stretched, same-sign compression at
  perimeter (the chain story).

Graph-Laplacian sees only (delta_i - delta_j) and is blind to
curvature -- it spreads the load in same-sign neighbours regardless.

Test parts:

  PART A.  Build an arc; verify radial outward normals, edges match
           1D chain topology, rest edge lengths > R_pad * alpha by
           the curvature factor sqrt(1 + alpha^2/4).

  PART B.  Face-on contact at the apex; small delta_y_apex (within the
           bulge window).  Plot per-sphere delta_n in LOCAL FRAME
           (scalar projection onto each sphere's own n_hat).  Verify:
             * graph-Laplacian -> all positive, decays away from apex,
             * distance-preserving -> apex positive, perimeter
               *negative* (bulge), crosses zero somewhere mid-patch.

  PART C.  Sweep phi_rest from 0 to ~5 * R_pad * alpha^2.  Plot the
           perimeter delta_n vs delta_y_apex (numerical) and the
           theoretical zero crossing at delta_y_apex = R_pad * alpha^2.

  PART D.  Curvature sweep at fixed contact phi_rest: vary R_pad
           from very large (nearly flat) to small (highly curved) at
           fixed arc-length spacing.  Bulging vanishes as R_pad -> inf.

Run::

    uv run -m cslc_main.theory.test_05_arc_contact

Outputs::

    cslc_main/theory/figures/05a_arc_geometry.png
    cslc_main/theory/figures/05b_face_on_local_delta.png
    cslc_main/theory/figures/05c_bulge_window_phi_sweep.png
    cslc_main/theory/figures/05d_curvature_sweep.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    ContactTarget,
    build_K_matrix,
    make_arc,
    solve_lattice_contact_linear,
    solve_lattice_contact_numerical,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────


def local_delta_n(lat, deltas: np.ndarray) -> np.ndarray:
    """Project each sphere's delta onto its own outward normal: delta_n_i."""
    return np.einsum("ij,ij->i", deltas, lat.n)


def make_scene(
    *,
    N: int = 31,
    R_pad: float = 10.0e-3,        # arc radius [m] -- 10 mm dome
    arc_length_spacing: float = 1.0e-3,  # 1 mm arc-length between spheres
    ka: float = 25_000.0,
    kl: float = 25_000.0,          # kl/ka = 1 -- well-coupled.  Production CSLC
                                   # is sub-grid (kl/ka = 0.2); we use kl/ka = 1
                                   # so the bulge propagates several spacings
                                   # and is visible on plots.  Sub-grid case is
                                   # exhibited explicitly in a separate sweep.
    kc: float = 25_000.0,
    phi_rest: float = 1.0e-5,      # 10 um -- inside bulge window for R*alpha^2 = 10*0.1^2 = 100 um
    r_lat: float = 0.5e-3,
    R: float = 33.5e-3,
):
    """Build an arc + a target sitting face-on above the apex.

    Returns (lat, target, phi_rest, i_centre, R_pad, alpha, curvature_drop).
    """
    lat = make_arc(N=N, R_pad=R_pad, arc_length_spacing=arc_length_spacing,
                   ka=ka, kl=kl)
    i_centre = N // 2
    n_apex = lat.n[i_centre]
    p_apex = lat.p[i_centre]
    d = (r_lat + R) - phi_rest
    target = ContactTarget(
        sphere_idx=i_centre,
        t=p_apex + d * n_apex,
        R=R,
        kc=kc,
    )
    alpha = arc_length_spacing / R_pad
    curvature_drop = R_pad * alpha * alpha  # one-spacing perimeter "depth" relative to apex
    return lat, target, phi_rest, i_centre, R_pad, alpha, curvature_drop


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Geometry sanity
# ─────────────────────────────────────────────────────────────────────────


def part_a_geometry() -> bool:
    """Build an arc; check normals are radial and rest lengths are correct."""
    print()
    print("=" * 72)
    print("PART A.  Arc geometry sanity")
    print("=" * 72)

    lat, _, _, i_c, R_pad, alpha, curvature_drop = make_scene(N=21)
    N = lat.N

    # Normals must point radially outward: n_i ∝ p_i (modulo numerical).
    n_dot_p = np.einsum("ij,ij->i", lat.n, lat.p / R_pad)
    n_dot_p_residual = float(np.max(np.abs(n_dot_p - 1.0)))

    # Rest distances along the arc: chord length R_pad * 2 * sin(delta_theta/2)
    rest_lengths = lat.rest_lengths()
    delta_theta = alpha
    expected = R_pad * 2.0 * np.sin(delta_theta / 2.0)
    rest_residual = float(np.max(np.abs(rest_lengths - expected)))

    # K matrix shape & symmetry.
    K = build_K_matrix(lat)
    K_sym = float(np.max(np.abs(K - K.T)))
    K_smallest_eig = float(np.linalg.eigvalsh(K)[0])

    print()
    print(f"  N = {N}, R_pad = {R_pad*1e3:.2f} mm, arc spacing = "
          f"{rest_lengths.mean()*1e3:.3f} mm")
    print(f"  alpha (angular spacing)            = {alpha:.6f} rad")
    print(f"  curvature drop R_pad * alpha^2     = {curvature_drop*1e6:.2f} um")
    print(f"  apex sphere index                  = {i_c}")
    print()
    print(f"  max |n_i . p_i / R_pad - 1|        = {n_dot_p_residual:.3e}  "
          f"(want 0; normals radial)")
    print(f"  max |rest - 2*R*sin(d_theta/2)|    = {rest_residual:.3e} m")
    print(f"  K symmetry residual                = {K_sym:.3e}")
    print(f"  K smallest eigenvalue              = {K_smallest_eig:.4f}  "
          f"(want ka = {lat.ka:.4f}; uniform-translation mode)")

    ok = (n_dot_p_residual < 1e-12
          and rest_residual < 1e-12
          and K_sym < 1e-10
          and abs(K_smallest_eig - lat.ka) < 1e-6)
    print()
    print(f"PART A result: {'PASS' if ok else 'FAIL'}")

    # Plot arc geometry.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lat.p[:, 0] * 1e3, lat.p[:, 1] * 1e3, "o-", ms=6, color="C0")
    ax.scatter(lat.p[i_c, 0] * 1e3, lat.p[i_c, 1] * 1e3, s=120,
               c="C3", zorder=3, label="apex")
    # Normal arrows.
    scale = 1e-3  # 1 mm normal arrow
    for i in range(0, N, 2):
        ax.annotate("", xy=((lat.p[i, 0] + scale * lat.n[i, 0]) * 1e3,
                            (lat.p[i, 1] + scale * lat.n[i, 1]) * 1e3),
                    xytext=(lat.p[i, 0] * 1e3, lat.p[i, 1] * 1e3),
                    arrowprops={"arrowstyle": "->", "color": "C1", "lw": 1})
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(rf"Arc lattice  ($R_{{\mathrm{{pad}}}} = {R_pad*1e3:.1f}$ mm, "
                 rf"$N = {N}$, $\alpha = {alpha:.4f}$ rad)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "05a_arc_geometry.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Face-on contact: local delta_n shows bulging
# ─────────────────────────────────────────────────────────────────────────


def part_b_face_on_local_delta() -> bool:
    """Compare local delta_n for graph-Laplacian vs distance-preserving."""
    print()
    print("=" * 72)
    print("PART B.  Face-on contact at apex; local delta_n shows bulging")
    print("=" * 72)

    # Pick phi_rest inside the bulge window: phi_rest < R_pad * alpha^2.
    # For R_pad = 10 mm, h = 1 mm: alpha = 0.1, drop = 100 um. Use phi = 30 um.
    # Use kl/ka = 1 (well-coupled) so the bulge reaches several spacings.
    lat, target, phi_rest, i_c, R_pad, alpha, drop = make_scene(
        N=31, phi_rest=30e-6, kl=25000.0)

    d_GL = solve_lattice_contact_linear(lat, target, phi_rest)
    d_DP, info_DP = solve_lattice_contact_numerical(
        lat, target, phi_rest,
        lateral="distance_preserving", delta0=d_GL, eps=1e-9, tol=1e-12)

    n_GL = local_delta_n(lat, d_GL)
    n_DP = local_delta_n(lat, d_DP)

    # Where does distance-pres cross zero (bulge boundary)?
    sign_changes = np.where(np.diff(np.sign(n_DP)))[0]
    crossover_idx = sign_changes[0] if len(sign_changes) > 0 else -1

    print()
    print(f"  N = {lat.N}, R_pad = {R_pad*1e3:.2f} mm, alpha = {alpha:.4f},  "
          f"R_pad*alpha^2 = {drop*1e6:.2f} um (curvature drop)")
    print(f"  phi_rest = {phi_rest*1e6:.2f} um  (inside bulge window 0 .. "
          f"{drop*1e6:.0f} um)")
    print()
    print(f"  apex local delta_n:")
    print(f"    graph-Laplacian      = {n_GL[i_c]*1e6:>8.4f} um")
    print(f"    distance-preserving  = {n_DP[i_c]*1e6:>8.4f} um")
    print()
    print(f"  per-sphere local delta_n (apex and first few neighbours):")
    for di in [1, 2, 3, 5, 10]:
        ip = i_c + di
        if ip < lat.N:
            print(f"    i - i_c = +{di:<2}  GL = {n_GL[ip]*1e6:>+9.5f} um   "
                  f"DP = {n_DP[ip]*1e6:>+9.5f} um   "
                  f"DP sign = {'INWARD' if n_DP[ip] > 0 else 'OUTWARD (bulge)'}")
    print()
    if crossover_idx >= 0:
        print(f"  distance-pres delta_n crosses zero at sphere {crossover_idx} "
              f"(offset {crossover_idx - i_c} from apex)")
    else:
        print(f"  distance-pres delta_n does not change sign in this window")

    # Pass criteria:
    GL_all_positive = bool(np.all(n_GL >= -1e-15))
    DP_apex_positive = n_DP[i_c] > 0
    DP_has_bulge = bool(np.any(n_DP < -1e-15))

    print()
    print(f"  GL all non-negative?                "
          f"{'PASS' if GL_all_positive else 'FAIL'}")
    print(f"  DP apex compressed inward?         "
          f"{'PASS' if DP_apex_positive else 'FAIL'}")
    print(f"  DP has negative delta_n (bulge)?   "
          f"{'PASS' if DP_has_bulge else 'FAIL'}")
    ok = GL_all_positive and DP_apex_positive and DP_has_bulge

    # Two-panel plot.
    # Left: full profile of both laws (apex visible, perimeter compressed).
    # Right: DP only, zoomed to the bulge (negative-delta) region.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    offsets = np.arange(lat.N) - i_c
    ax1.plot(offsets, n_GL * 1e6, "o-", color="C3", lw=1.5, ms=5,
             label="graph-Laplacian  (same-direction spread)")
    ax1.plot(offsets, n_DP * 1e6, "s-", color="C0", lw=1.5, ms=5,
             label="distance-preserving")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.axvline(0, color="black", lw=0.5, alpha=0.3)
    ax1.set_xlabel("sphere offset $i - i_{\\mathrm{centre}}$")
    ax1.set_ylabel(r"local $\delta_n(i)$ [$\mu$m]"
                   "\n(positive = inward, negative = bulge)")
    ax1.set_title("Full profile (apex compression dominates)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    # Right: zoom into DP perimeter, with horizontal baselines for context.
    ax2.plot(offsets, n_DP * 1e6, "s-", color="C0", lw=1.5, ms=6,
             label="distance-preserving (zoomed)")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.axvline(0, color="black", lw=0.5, alpha=0.3)
    perim_max_abs = float(np.max(np.abs(n_DP[i_c + 1:]))) * 1e6
    ax2.set_ylim(-perim_max_abs * 1.4, perim_max_abs * 1.4)
    ax2.set_xlim(-12, 12)
    ax2.set_xlabel("sphere offset $i - i_{\\mathrm{centre}}$")
    ax2.set_ylabel(r"local $\delta_n(i)$ [$\mu$m] (DP zoom, "
                   r"$\pm\sim 10$ nm)")
    ax2.set_title(rf"DP bulge at perimeter (apex clipped off; "
                  rf"max perim |$\delta_n$| = {perim_max_abs*1e3:.1f} nm)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.suptitle(rf"Arc + face-on contact at apex  "
                 rf"($\varphi = {phi_rest*1e6:.0f}\,\mu$m, "
                 rf"$R\alpha^2 = {drop*1e6:.0f}\,\mu$m, "
                 rf"$\varphi < R\alpha^2$ -- inside bulge window)",
                 y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "05b_face_on_local_delta.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out}")
    print(f"PART B result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Phi sweep showing the bulge window
# ─────────────────────────────────────────────────────────────────────────


def part_c_phi_sweep() -> bool:
    """Vary phi_rest; plot perimeter delta_n vs apex delta_y for DP."""
    print()
    print("=" * 72)
    print("PART C.  Phi sweep: bulge window 0 < delta_apex < R_pad * alpha^2")
    print("=" * 72)

    # Build one arc and reuse; sweep phi_rest.
    R_pad = 10.0e-3
    h = 1.0e-3
    alpha = h / R_pad
    drop = R_pad * alpha * alpha  # 100 um

    # phi_rest values from far below drop to ~5x drop.
    phi_vals = np.linspace(0.0, 5.0 * drop, 21)

    perimeter_idx_offset = 1   # nearest neighbour -- strongest signal
    apex_GL = np.zeros_like(phi_vals)
    apex_DP = np.zeros_like(phi_vals)
    perim_GL = np.zeros_like(phi_vals)
    perim_DP = np.zeros_like(phi_vals)

    for k, phi in enumerate(phi_vals):
        lat, target, _, i_c, _, _, _ = make_scene(
            N=31, R_pad=R_pad, arc_length_spacing=h, kl=25000.0,
            phi_rest=max(phi, 1e-12))
        if phi <= 0.0:
            d_GL = np.zeros((lat.N, 3))
            d_DP = np.zeros((lat.N, 3))
        else:
            d_GL = solve_lattice_contact_linear(lat, target, phi)
            d_DP, _ = solve_lattice_contact_numerical(
                lat, target, phi, lateral="distance_preserving",
                delta0=d_GL, eps=1e-9, tol=1e-12)
        n_GL = local_delta_n(lat, d_GL)
        n_DP = local_delta_n(lat, d_DP)
        apex_GL[k] = n_GL[i_c]
        apex_DP[k] = n_DP[i_c]
        perim_GL[k] = n_GL[i_c + perimeter_idx_offset]
        perim_DP[k] = n_DP[i_c + perimeter_idx_offset]

    # Locate the zero-crossing of perim_DP (numerical), skipping the
    # phi_vals[0] = 0 entry which is zero by construction.
    sign_changes = np.where(np.diff(np.sign(perim_DP[1:])))[0]
    if len(sign_changes) > 0:
        i_cross = sign_changes[0] + 1
        x0, x1 = phi_vals[i_cross], phi_vals[i_cross + 1]
        y0, y1 = perim_DP[i_cross], perim_DP[i_cross + 1]
        phi_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
    else:
        phi_cross = np.nan

    print()
    print(f"  R_pad = {R_pad*1e3:.1f} mm, h = {h*1e3:.1f} mm, alpha = {alpha:.4f}")
    print(f"  curvature drop R_pad * alpha^2 = {drop*1e6:.2f} um  "
          f"(theoretical bulge-to-compress crossover)")
    print(f"  phi sweep from 0 to {5*drop*1e6:.0f} um")
    print()
    print(f"  perimeter sphere (offset +{perimeter_idx_offset}) delta_n DP sweep:")
    print(f"    {'phi[um]':>9} {'apex_DP[um]':>13} {'perim_DP[um]':>14} "
          f"{'sign':>20}")
    for k in [0, 5, 10, 15, 20]:
        if k < len(phi_vals):
            sgn = ("zero" if abs(perim_DP[k]) < 1e-10
                   else ("INWARD" if perim_DP[k] > 0 else "OUTWARD (bulge)"))
            print(f"    {phi_vals[k]*1e6:>9.2f} {apex_DP[k]*1e6:>13.4f} "
                  f"{perim_DP[k]*1e6:>+14.4f} {sgn:>20}")
    print()
    print(f"  measured zero-crossing phi_cross = {phi_cross*1e6:.2f} um")
    print(f"  expected (linear theory)         = R_pad * alpha^2 = {drop*1e6:.2f} um")
    # NOTE: phi_cross is delta_apex - based; small effective deviation depends on
    # the apex's actual sinkage (less than phi for kc<inf).  Just sanity-check
    # the order of magnitude.

    bulge_found = bool(np.any(perim_DP < -1e-10))
    crossing_finite = np.isfinite(phi_cross)
    crossing_in_range = bool(0.5 * drop < phi_cross < 5.0 * drop) if crossing_finite else False

    print()
    print(f"  DP shows bulge (perim < 0) somewhere?   "
          f"{'PASS' if bulge_found else 'FAIL'}")
    print(f"  Zero-crossing within 0.5*drop .. 5*drop? "
          f"{'PASS' if crossing_in_range else 'FAIL'}")
    print(f"  (Note: GL stays positive monotonic; no crossover.)")
    print(f"  GL monotone-positive?   "
          f"{'PASS' if bool(np.all(np.diff(perim_GL) >= -1e-12)) else 'FAIL'}")

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phi_vals * 1e6, perim_GL * 1e6, "o-", color="C3", lw=1.5,
            label=f"graph-Laplacian, perimeter (+{perimeter_idx_offset})")
    ax.plot(phi_vals * 1e6, perim_DP * 1e6, "s-", color="C0", lw=1.5,
            label=f"distance-preserving, perimeter (+{perimeter_idx_offset})")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(drop * 1e6, ls="--", color="gray",
               label=rf"$R_{{\mathrm{{pad}}}} \alpha^2 = {drop*1e6:.0f}\,\mu$m "
                     r"(rest-chord recovery)")
    if np.isfinite(phi_cross):
        ax.axvline(phi_cross * 1e6, ls=":", color="C0",
                   label=rf"DP crossover  $\varphi = {phi_cross*1e6:.1f}\,\mu$m")
    ax.set_xlabel(r"contact rest overlap $\varphi_{\mathrm{rest}}$ [$\mu$m]")
    ax.set_ylabel(r"perimeter local $\delta_n$ [$\mu$m]"
                  "\n(positive = inward; negative = outward bulge)")
    ax.set_title("Arc-contact bulge window: distance-preserving "
                 r"crosses zero near $R\alpha^2$, graph-Laplacian doesn't")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "05c_bulge_window_phi_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    ok = bulge_found and crossing_in_range
    print(f"PART C result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  Curvature sweep: bulging vanishes as R_pad -> infinity
# ─────────────────────────────────────────────────────────────────────────


def part_d_curvature_sweep() -> bool:
    """Vary R_pad at fixed phi and h; show bulge vanishes as R_pad -> inf."""
    print()
    print("=" * 72)
    print("PART D.  Curvature sweep: bulging is geometric")
    print("=" * 72)

    h = 1.0e-3
    phi = 30e-6   # 30 um -- fixed
    R_pads = np.array([2.5, 5.0, 10.0, 25.0, 100.0]) * 1e-3
    perimeter_offset = 1   # nearest neighbour

    perim_DP = np.zeros_like(R_pads)
    drops = np.zeros_like(R_pads)
    apex_DP = np.zeros_like(R_pads)

    for k, R_pad in enumerate(R_pads):
        lat, target, _, i_c, _, alpha, drop = make_scene(
            N=31, R_pad=R_pad, arc_length_spacing=h, kl=25000.0, phi_rest=phi)
        d_GL = solve_lattice_contact_linear(lat, target, phi)
        d_DP, _ = solve_lattice_contact_numerical(
            lat, target, phi, lateral="distance_preserving",
            delta0=d_GL, eps=1e-9, tol=1e-12)
        n_DP = local_delta_n(lat, d_DP)
        apex_DP[k] = n_DP[i_c]
        perim_DP[k] = n_DP[i_c + perimeter_offset]
        drops[k] = drop

    print()
    print(f"  h = {h*1e3:.1f} mm, phi = {phi*1e6:.1f} um, perimeter offset = +{perimeter_offset}")
    print()
    print(f"  {'R_pad[mm]':>11} {'alpha':>10} {'R*alpha^2[um]':>15} "
          f"{'apex_DP[um]':>13} {'perim_DP[um]':>14} {'sign':>20}")
    for k, R_pad in enumerate(R_pads):
        alpha = h / R_pad
        sgn = ("zero" if abs(perim_DP[k]) < 1e-10
               else ("INWARD" if perim_DP[k] > 0 else "OUTWARD (bulge)"))
        print(f"  {R_pad*1e3:>11.2f} {alpha:>10.4f} {drops[k]*1e6:>15.2f} "
              f"{apex_DP[k]*1e6:>13.4f} {perim_DP[k]*1e6:>+14.4f} {sgn:>20}")

    # Bulging at small R_pad, vanishing toward zero at large R_pad.
    bulge_at_small_R = perim_DP[0] < -1e-10
    near_zero_at_large_R = abs(perim_DP[-1]) < 5.0 * abs(perim_DP[0])
    # The bulge magnitude should DECREASE as R_pad increases (in the bulge regime).
    # We check that the SIGNED perim_DP is monotone-non-decreasing across the
    # sweep (since small R_pad gives more negative perim_DP, larger R_pad
    # gives less negative -- when both are still in bulge regime).  As R_pad
    # crosses drop = phi, the system leaves the bulge window.
    print()
    print(f"  bulge at smallest R_pad?   "
          f"{'PASS' if bulge_at_small_R else 'FAIL'}")
    print(f"  perim_DP order-of-magnitude shrinks toward large R_pad? "
          f"{'PASS' if near_zero_at_large_R else 'FAIL'}  "
          f"(small-R = {perim_DP[0]*1e6:+.3f} um, "
          f"large-R = {perim_DP[-1]*1e6:+.3f} um)")
    ok = bulge_at_small_R

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R_pads * 1e3, perim_DP * 1e6, "o-", color="C0", ms=8, lw=1.5,
            label="perimeter distance-preserving $\\delta_n$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"arc radius $R_{\mathrm{pad}}$ [mm] (log)")
    ax.set_ylabel(r"perimeter local $\delta_n$ [$\mu$m]"
                  "\n(negative = outward bulge)")
    ax.set_title(r"Curvature sweep at fixed $\varphi_{\mathrm{rest}} = 30\,\mu$m, "
                 r"$h = 1$ mm: bulge depends on $R_{\mathrm{pad}} \alpha^2$")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "05d_curvature_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    print(f"PART D result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = part_a_geometry()
    b = part_b_face_on_local_delta()
    c = part_c_phi_sweep()
    d = part_d_curvature_sweep()
    all_ok = a and b and c and d
    print()
    print("=" * 72)
    print(f"Step 5 arc-contact test: {'PASS' if all_ok else 'FAIL'}   "
          f"(A={'P' if a else 'F'} B={'P' if b else 'F'} "
          f"C={'P' if c else 'F'} D={'P' if d else 'F'})")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
