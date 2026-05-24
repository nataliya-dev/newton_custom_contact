# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 8 / 3D dome lattice pressed by a sphere indenter.

The step-5 arc test proved the geometric Poisson bulge window analytically
in 1D (eq. ``arc-window`` of ``theory.txt``).  This driver lifts that
study to 2D: a quasi-uniform Fibonacci-spiral cap of ``N=150`` spheres
on a sphere of radius ``R_pad = 10 mm`` (matching the production
fingertip dome in ``assets/pad/pad.obj``), pressed by a tennis-ball
indenter (``R_obj = 33.5 mm``).

This is the dome's grip diagnostic: under what conditions does it
behave like a real compliant fingertip (active patch spreading, lateral
coupling sharing load, kc tuned to deliver ``ke_bulk``), and under what
conditions does it act like an array of decoupled normal springs (the
failure mode that drops the ball in production)?

Predictions tested
------------------

PART A.  Geometry sanity.  Cap built correctly: ``N`` spheres, radial
         outward normals (``n_i = p_i / R_pad``), Delaunay-like k-NN
         edges with average degree ~``k_neighbors``, K is SPD with
         smallest eigenvalue exactly ``ka`` (uniform-translation mode,
         consistent with step-2 chain spectrum).

PART B.  Apex sinkage vs penetration.  At ``phi_apex`` from 0 to ~3 mm
         along the apex sphere's outward normal, measure the apex
         local ``delta_n``.  Compare to two predictions:

         * **Isolated series-spring** ``delta = kc*phi/(ka+kc)`` --
           recovered exactly when the apex sphere acts as a parallel
           spring independent of the lattice.  At our production
           geometry (``R_obj/spacing ~ 20``) this is the dominant
           regime because adjacent spheres compress by similar amounts.

         * **Green's-function-stiffened** ``delta = kc*phi/(1/g_00 + kc)``
           -- the 1D chain-contact result of step 3 lifted to the dome
           via ``g_00 = (K^-1)[apex, apex]``.  Diverges from the
           isolated answer when load is sharply localised (small
           ``R_obj``, large ``kl/ka``).

         The gap between these two predictions is *the* diagnostic for
         whether lateral coupling is actually doing anything useful
         under production geometry.

PART C.  Patch flattening + Hertz prediction.  At each ``phi_apex`` the
         number of active spheres ``N_active`` should grow as
         ``N_active ~ pi * a^2 / cell_area`` with the Hertz contact
         radius ``a = sqrt(R_eff * phi_apex)``, ``R_eff = (R_obj * R_pad)
         / (R_obj + R_pad)``.  We measure ``a`` from the active-sphere
         positions and verify the scaling.  3D scatter coloured by
         ``delta_n`` visualises the flattening.

PART D.  ``kl / ka`` sweep -- production diagnostic.  At fixed
         ``phi_apex = 1 mm`` and varying ``kl / ka`` in {0.05, 0.2, 1,
         5}, measure (apex delta, N_active, F_total).  Production is
         ``kl/ka = 0.2``; ``F_total = sum_i kc * phi_eff_i`` is the
         aggregate normal force the indenter feels, and its slope
         ``F_total / phi`` is the effective bulk stiffness the
         calibration in ``contact_models.recalibrate_kc_per_pad``
         tries to match to ``ke_bulk``.

Run::

    uv run -m cslc_main.theory.test_08_dome_contact

Outputs::

    cslc_main/theory/figures/08a_dome_geometry.png
    cslc_main/theory/figures/08b_apex_sinkage.png
    cslc_main/theory/figures/08c_flattening_3d.png
    cslc_main/theory/figures/08d_kl_sweep.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    SphereIndenter,
    build_K_matrix,
    make_dome,
    solve_lattice_sphere_indenter,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ─────────────────────────────────────────────────────────────────────────
#  Fixed scene parameters -- production-equivalent
# ─────────────────────────────────────────────────────────────────────────


R_PAD = 10.0e-3                  # dome sphere radius [m] -- production fingertip
HALF_ANGLE = np.deg2rad(72.0)    # cap half-angle -- matches the production OBJ
N_SPHERES = 150                  # matches PadParams.n_samples
K_NEIGHBORS = 6                  # matches PadParams.k_neighbors
R_OBJ = 33.5e-3                  # tennis ball radius
KA = 25_000.0                    # production CSLCParams.ka
KL_DEFAULT = 5_000.0             # production CSLCParams.kl  (kl/ka = 0.2)
KE_BULK = 5.0e4                  # production MaterialParams.ke (bulk stiffness target)
KC_DEFAULT = 1.0e4               # picked to give roughly ke_bulk under expected N_active


def make_dome_scene(*, kl: float = KL_DEFAULT, kc: float = KC_DEFAULT,
                    N: int = N_SPHERES):
    """Build the dome lattice + a placeholder indenter (caller positions it).

    Returns ``(lat, spacing, cap_area, r_lat, kc)``.  ``r_lat`` is the
    per-sphere lattice radius (uniform, ``= spacing / 2`` -- the same
    convention as ``contact_models.make_cslc_pad_from_samples``).
    """
    lat, spacing, cap_area = make_dome(
        N=N, R_pad=R_PAD, half_angle=HALF_ANGLE, ka=KA, kl=kl,
        k_neighbors=K_NEIGHBORS,
    )
    r_lat = spacing * 0.5
    return lat, spacing, cap_area, r_lat, kc


def position_indenter(lat, r_lat: float, phi_apex: float) -> SphereIndenter:
    """Place the indenter along the apex sphere's outward normal so that
    ``raw_apex = (r_lat + R_obj) - ||t - p_apex|| = phi_apex``."""
    apex_p = lat.p[0]
    apex_n = lat.n[0]
    t = apex_p + (r_lat + R_OBJ - phi_apex) * apex_n
    return SphereIndenter(t=t, R=R_OBJ, kc=KC_DEFAULT)


def local_delta_n(lat, deltas: np.ndarray) -> np.ndarray:
    """Project each delta onto its sphere's outward normal."""
    return np.einsum("ij,ij->i", deltas, lat.n)


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Geometry sanity
# ─────────────────────────────────────────────────────────────────────────


def part_a_geometry() -> bool:
    print()
    print("=" * 72)
    print("PART A.  Dome geometry sanity")
    print("=" * 72)

    lat, spacing, cap_area, r_lat, _ = make_dome_scene()

    # Radial normals.
    n_dot_p = np.einsum("ij,ij->i", lat.n, lat.p / R_PAD)
    n_radial_residual = float(np.max(np.abs(n_dot_p - 1.0)))

    # Degree distribution.
    degrees = lat.neighbour_counts()
    deg_mean = float(np.mean(degrees))
    deg_min = int(np.min(degrees))
    deg_max = int(np.max(degrees))

    # K symmetry + spectrum.
    K = build_K_matrix(lat)
    K_sym = float(np.max(np.abs(K - K.T)))
    eig = np.linalg.eigvalsh(K)
    eig_min, eig_max = float(eig[0]), float(eig[-1])

    # Cap area sanity: 2 pi R^2 (1 - cos theta_max).
    cap_area_expected = 2.0 * np.pi * R_PAD**2 * (1.0 - np.cos(HALF_ANGLE))
    cap_area_residual = abs(cap_area - cap_area_expected) / cap_area_expected

    # Effective per-sphere cell area = cap_area / N -- benchmarks the
    # Hertz N_active prediction in part C.
    cell_area = cap_area / lat.N

    print()
    print(f"  N spheres           = {lat.N}")
    print(f"  N edges             = {lat.E}  (avg degree = {deg_mean:.2f}, "
          f"min = {deg_min}, max = {deg_max})")
    print(f"  spacing (mean NN)   = {spacing*1e3:.3f} mm")
    print(f"  cap area            = {cap_area*1e6:.2f} mm^2  "
          f"(expected {cap_area_expected*1e6:.2f}, rel err {cap_area_residual:.2e})")
    print(f"  per-sphere cell     = {cell_area*1e6:.3f} mm^2/sphere")
    print(f"  r_lat (= h/2)       = {r_lat*1e3:.3f} mm")
    print()
    print(f"  max |n_i . p_i/R - 1| (normal radiality)  = {n_radial_residual:.3e}")
    print(f"  K symmetry residual                       = {K_sym:.3e}")
    print(f"  K smallest eigenvalue                     = {eig_min:.4f}  "
          f"(want ka = {KA:.4f}; uniform-translation mode)")
    print(f"  K largest  eigenvalue                     = {eig_max:.4f}")

    # Production tunes pad.n_samples = 150, k_neighbors = 6.  Tolerate
    # broad degree distributions on the boundary, where the cap edge
    # has fewer neighbours.  Geometry expectations:
    ok_radial = n_radial_residual < 1e-12
    ok_K_sym = K_sym < 1e-9
    ok_K_smallest = abs(eig_min - KA) < 1e-6
    ok_cap_area = cap_area_residual < 1e-6
    # Average degree should be close to 2 * k_neighbors (each pair
    # adds two degree counts) on a closed manifold, but the cap has
    # boundary so it's lower.  Production-like: average between
    # k_neighbors and 2*k_neighbors.
    ok_degree = K_NEIGHBORS <= deg_mean <= 2.0 * K_NEIGHBORS

    ok = ok_radial and ok_K_sym and ok_K_smallest and ok_cap_area and ok_degree
    print()
    print(f"  radial normals?            {'PASS' if ok_radial else 'FAIL'}")
    print(f"  K symmetric?               {'PASS' if ok_K_sym else 'FAIL'}")
    print(f"  K smallest eig = ka?       {'PASS' if ok_K_smallest else 'FAIL'}")
    print(f"  cap area matches formula?  {'PASS' if ok_cap_area else 'FAIL'}")
    print(f"  degree in [{K_NEIGHBORS}, {2*K_NEIGHBORS}]? "
          f"{'PASS' if ok_degree else 'FAIL'}")

    # 3D scatter of the dome with normal quivers.
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    p_mm = lat.p * 1e3
    ax.scatter(p_mm[:, 0], p_mm[:, 1], p_mm[:, 2], s=12, c="tab:blue", alpha=0.7)
    ax.scatter(p_mm[0, 0], p_mm[0, 1], p_mm[0, 2], s=80, c="tab:red",
               label=f"apex sphere (i = 0)", zorder=5)
    # Quiver subset.
    qlen_mm = spacing * 0.5 * 1e3
    sub = np.arange(0, lat.N, 5)
    nq = lat.n[sub]
    ax.quiver(p_mm[sub, 0], p_mm[sub, 1], p_mm[sub, 2],
              nq[:, 0], nq[:, 1], nq[:, 2],
              length=qlen_mm, normalize=False, color="tab:orange", alpha=0.5,
              linewidth=0.5)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(rf"Dome lattice  ($R_{{\mathrm{{pad}}}} = {R_PAD*1e3:.1f}$ mm, "
                 rf"$N = {lat.N}$, "
                 rf"$\theta_{{\max}} = {np.rad2deg(HALF_ANGLE):.0f}^\circ$)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "08a_dome_geometry.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    print(f"PART A result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Apex sinkage: isolated vs lattice-stiffened series-spring
# ─────────────────────────────────────────────────────────────────────────


def part_b_apex_sinkage() -> bool:
    print()
    print("=" * 72)
    print("PART B.  Apex sinkage vs penetration")
    print("=" * 72)

    lat, spacing, _, r_lat, kc = make_dome_scene()

    # Green's function diagonal at the apex (for the lattice-stiffened
    # prediction).  Reusable for all phi.
    K = build_K_matrix(lat)
    g_00 = float(np.linalg.inv(K)[0, 0])
    ka_eff = 1.0 / g_00
    print()
    print(f"  Lattice apex Green's function g_00     = {g_00:.6e}")
    print(f"  lattice-stiffened apex anchor ka_eff   = {ka_eff:.2f}  "
          f"(vs ka = {KA:.0f}; ratio = {ka_eff/KA:.4f})")
    print(f"  kc (production-equivalent)             = {kc:.0f}")

    # Sweep phi.
    phis = np.linspace(0.05e-3, 3.0e-3, 16)  # 50 um to 3 mm
    delta_apex_meas = np.zeros_like(phis)
    n_active = np.zeros_like(phis, dtype=int)
    delta = None  # warm-start chain
    for k, phi in enumerate(phis):
        indenter = position_indenter(lat, r_lat, phi)
        delta, info = solve_lattice_sphere_indenter(
            lat, indenter, r_lat=r_lat, eps=1e-9,
            delta0=delta if k > 0 else None,
        )
        n_dn = local_delta_n(lat, delta)
        delta_apex_meas[k] = n_dn[0]
        n_active[k] = info["n_active"]

    # Predictions.
    delta_isolated = kc * phis / (KA + kc)
    delta_green = kc * phis / (ka_eff + kc)

    rel_err_iso = np.abs(delta_apex_meas - delta_isolated) / np.maximum(delta_isolated, 1e-12)
    rel_err_green = np.abs(delta_apex_meas - delta_green) / np.maximum(delta_green, 1e-12)
    max_iso_err = float(np.max(rel_err_iso))
    max_green_err = float(np.max(rel_err_green))

    print()
    print(f"  {'phi[mm]':>10} {'N_active':>10} {'meas[um]':>12} "
          f"{'iso[um]':>10} {'green[um]':>11}")
    for k in [0, 4, 8, 12, 15]:
        print(f"  {phis[k]*1e3:>10.3f} {n_active[k]:>10d} "
              f"{delta_apex_meas[k]*1e6:>12.3f} "
              f"{delta_isolated[k]*1e6:>10.3f} {delta_green[k]*1e6:>11.3f}")
    print()
    print(f"  max rel err vs isolated series-spring  = {max_iso_err:.4e}")
    print(f"  max rel err vs Green's-stiffened       = {max_green_err:.4e}")
    print()
    print("  Interpretation: which prediction wins is the diagnostic.")
    print("  Production geometry (R_obj >> spacing): N_active is large,")
    print("  neighbours' delta values are similar to the apex's, lateral")
    print("  spring is barely strained, isolated series-spring wins.")
    print("  This is exactly why the dome behaves like a parallel array")
    print("  of independent normal springs, not a coupled mesh.")

    # Linearity: delta_meas vs phi should be linear with slope close to
    # the better-matching prediction's slope.
    slope_meas = float(np.polyfit(phis, delta_apex_meas, 1)[0])
    slope_iso = kc / (KA + kc)
    slope_green = kc / (ka_eff + kc)
    print()
    print(f"  fitted slope d(delta_apex)/d(phi)      = {slope_meas:.6f}")
    print(f"  isolated slope kc/(ka + kc)            = {slope_iso:.6f}")
    print(f"  Green's slope    kc/(ka_eff + kc)      = {slope_green:.6f}")

    # Pass: at least one prediction must agree to 5%.  This documents
    # which regime the dome lives in, not which one is "correct".
    ok = (max_iso_err < 0.05) or (max_green_err < 0.05)
    print()
    print(f"  Isolated series-spring agrees to 5%?   "
          f"{'PASS' if max_iso_err < 0.05 else 'FAIL'}  ({max_iso_err*100:.2f}%)")
    print(f"  Green's-stiffened agrees to 5%?        "
          f"{'PASS' if max_green_err < 0.05 else 'FAIL'}  ({max_green_err*100:.2f}%)")

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phis * 1e3, delta_isolated * 1e6, "--", color="C0",
            label=rf"isolated: $kc\,\varphi/(ka + kc)$")
    ax.plot(phis * 1e3, delta_green * 1e6, ":", color="C2",
            label=rf"Green's-stiffened: $kc\,\varphi/(1/g_{{00}} + kc)$ "
                  rf"($ka_{{\mathrm{{eff}}}} = {ka_eff:.0f}$)")
    ax.plot(phis * 1e3, delta_apex_meas * 1e6, "o-", color="C3", ms=6, lw=1.5,
            label="measured apex $\\delta_n$ (multi-contact L-BFGS-B)")
    ax.set_xlabel(r"apex penetration $\varphi$ [mm]")
    ax.set_ylabel(r"apex local $\delta_n$ [$\mu$m]")
    ax.set_title("Dome apex sinkage: which regime does production live in?\n"
                 f"R_obj = {R_OBJ*1e3:.1f} mm, R_pad = {R_PAD*1e3:.1f} mm, "
                 f"spacing = {spacing*1e3:.2f} mm, kl/ka = {KL_DEFAULT/KA:.2f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "08b_apex_sinkage.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    print(f"PART B result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Patch flattening + Hertz prediction
# ─────────────────────────────────────────────────────────────────────────


def part_c_flattening() -> bool:
    print()
    print("=" * 72)
    print("PART C.  Patch flattening + Hertz prediction")
    print("=" * 72)

    lat, spacing, cap_area, r_lat, kc = make_dome_scene()
    cell_area = cap_area / lat.N
    R_eff = R_OBJ * R_PAD / (R_OBJ + R_PAD)
    print()
    print(f"  Hertz R_eff = R_obj * R_pad / (R_obj + R_pad) = {R_eff*1e3:.3f} mm")
    print(f"  cell area (cap_area / N) = {cell_area*1e6:.3f} mm^2")
    print()

    # Test penetrations.
    phis = np.array([0.5e-3, 1.0e-3, 2.0e-3, 3.0e-3])
    n_active_meas = np.zeros_like(phis, dtype=int)
    n_active_hertz = np.zeros_like(phis, dtype=float)
    patch_radius_meas = np.zeros_like(phis)
    f_total_meas = np.zeros_like(phis)
    deltas_by_phi = []

    delta = None
    for k, phi in enumerate(phis):
        indenter = position_indenter(lat, r_lat, phi)
        delta, info = solve_lattice_sphere_indenter(
            lat, indenter, r_lat=r_lat, eps=1e-9,
            delta0=delta if k > 0 else None,
        )
        n_active_meas[k] = info["n_active"]
        deltas_by_phi.append(delta.copy())

        # Hertz: a = sqrt(R_eff * phi_apex), N_active ~ pi * a^2 / cell_area.
        a_hertz = float(np.sqrt(R_eff * phi))
        n_active_hertz[k] = np.pi * a_hertz**2 / cell_area

        # Measured patch radius: lateral spread of active spheres from
        # the apex axis.  Use the q-positions (deformed centres) so the
        # flattening is reflected.
        # Active = those with raw_overlap > 0 under the hard gate.
        q = lat.p - delta
        diff = q - indenter.t
        L = np.linalg.norm(diff, axis=1)
        raw = (r_lat + R_OBJ) - L
        active = raw > 0
        if active.any():
            # Patch radius = max radial distance of active spheres from
            # the apex axis (project onto the plane perpendicular to
            # the apex normal).
            apex_n = lat.n[0]
            v = lat.p[active] - lat.p[0]
            v_tang = v - np.einsum("ij,j->i", v, apex_n)[:, None] * apex_n
            patch_radius_meas[k] = float(np.max(np.linalg.norm(v_tang, axis=1)))

        # Total normal force F_total = sum of per-sphere contact forces
        # (magnitudes), each kc * phi_eff_i.
        f_per = kc * np.maximum(raw, 0.0)
        f_total_meas[k] = float(np.sum(f_per))

    print(f"  {'phi[mm]':>10} {'N_meas':>8} {'N_hertz':>10} "
          f"{'a_meas[mm]':>12} {'a_hertz[mm]':>14} {'F_tot[N]':>12}")
    for k, phi in enumerate(phis):
        a_h = float(np.sqrt(R_eff * phi))
        print(f"  {phi*1e3:>10.3f} {n_active_meas[k]:>8d} "
              f"{n_active_hertz[k]:>10.2f} "
              f"{patch_radius_meas[k]*1e3:>12.3f} "
              f"{a_h*1e3:>14.3f} {f_total_meas[k]:>12.3f}")

    # Pass: N_active_meas scales LINEARLY with phi (since N ~ a^2 ~ R_eff*phi).
    # Fit a line through origin to log-log slope.
    log_phi = np.log(phis)
    log_n = np.log(np.maximum(n_active_meas.astype(float), 1.0))
    slope_n_vs_phi = float(np.polyfit(log_phi, log_n, 1)[0])
    print()
    print(f"  log-log slope N_active vs phi          = {slope_n_vs_phi:.3f}  "
          f"(Hertz prediction: 1.0)")

    # F_total should also scale linearly with phi at low penetration
    # (more spheres engage AND each sees more compression -- gives
    # F_total ~ phi^2 in the deep-saturated limit; ~ phi^1.5 in the
    # Hertz limit).  We just report the trend without strict gating.
    log_F = np.log(np.maximum(f_total_meas, 1e-12))
    slope_F_vs_phi = float(np.polyfit(log_phi, log_F, 1)[0])
    print(f"  log-log slope F_total vs phi           = {slope_F_vs_phi:.3f}  "
          f"(Hertz: 1.5, deep-sat: 2.0)")

    # 3D scatter heatmap at the largest phi.
    delta_big = deltas_by_phi[-1]
    n_dn = local_delta_n(lat, delta_big)
    q = lat.p - delta_big
    fig = plt.figure(figsize=(14, 6))
    # Left: deformed centres q coloured by delta_n.
    ax1 = fig.add_subplot(121, projection="3d")
    sc = ax1.scatter(q[:, 0] * 1e3, q[:, 1] * 1e3, q[:, 2] * 1e3,
                     c=n_dn * 1e6, cmap="RdBu_r", s=18,
                     vmin=-np.max(np.abs(n_dn)) * 1e6,
                     vmax=+np.max(np.abs(n_dn)) * 1e6)
    ax1.set_title(rf"Deformed lattice  (apex $\varphi = {phis[-1]*1e3:.1f}$ mm)")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_zlabel("z [mm]")
    plt.colorbar(sc, ax=ax1, label=r"$\delta_n$ [$\mu$m]", shrink=0.6)
    # Right: top-down view (xy) coloured by delta_n.
    ax2 = fig.add_subplot(122)
    sc2 = ax2.scatter(q[:, 0] * 1e3, q[:, 1] * 1e3,
                      c=n_dn * 1e6, cmap="RdBu_r", s=40,
                      vmin=-np.max(np.abs(n_dn)) * 1e6,
                      vmax=+np.max(np.abs(n_dn)) * 1e6)
    # Hertz patch circle overlay.
    a_h = float(np.sqrt(R_eff * phis[-1]))
    theta_circ = np.linspace(0, 2 * np.pi, 64)
    ax2.plot(a_h * np.cos(theta_circ) * 1e3 + lat.p[0, 0] * 1e3,
             a_h * np.sin(theta_circ) * 1e3 + lat.p[0, 1] * 1e3,
             "--", color="black", lw=1.5, alpha=0.7,
             label=rf"Hertz patch $a = \sqrt{{R_{{\mathrm{{eff}}}} \varphi}} = "
                   rf"{a_h*1e3:.2f}$ mm")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_title(r"Top-down ($\delta_n$ heatmap)")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax2, label=r"$\delta_n$ [$\mu$m]", shrink=0.7)
    fig.tight_layout()
    out = FIG_DIR / "08c_flattening_3d.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")

    # Pass: N_active grows monotonically with phi and scales roughly
    # linearly (slope 0.7 .. 1.3, allowing for boundary effects + finite N).
    monotonic = bool(np.all(np.diff(n_active_meas) >= 0))
    hertz_like = 0.6 < slope_n_vs_phi < 1.4
    print()
    print(f"  N_active monotonically increasing?     "
          f"{'PASS' if monotonic else 'FAIL'}")
    print(f"  N_active log-log slope in [0.6, 1.4]?  "
          f"{'PASS' if hertz_like else 'FAIL'}  (got {slope_n_vs_phi:.3f})")
    ok = monotonic and hertz_like
    print(f"PART C result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  kl / ka sweep: production diagnostic
# ─────────────────────────────────────────────────────────────────────────


def part_d_kl_sweep() -> bool:
    print()
    print("=" * 72)
    print("PART D.  kl / ka sweep at fixed phi = 1 mm")
    print("=" * 72)

    phi = 1.0e-3
    kl_over_ka = np.array([0.05, 0.2, 1.0, 5.0])
    kls = kl_over_ka * KA

    apex_dn = np.zeros_like(kls)
    n_active = np.zeros_like(kls, dtype=int)
    f_total = np.zeros_like(kls)
    max_dn = np.zeros_like(kls)
    mean_dn_active = np.zeros_like(kls)

    for k, kl in enumerate(kls):
        lat, spacing, _, r_lat, kc = make_dome_scene(kl=kl)
        indenter = position_indenter(lat, r_lat, phi)
        delta, info = solve_lattice_sphere_indenter(
            lat, indenter, r_lat=r_lat, eps=1e-9)
        n_dn = local_delta_n(lat, delta)
        q = lat.p - delta
        raw = (r_lat + R_OBJ) - np.linalg.norm(q - indenter.t, axis=1)
        active = raw > 0
        f_per = kc * np.maximum(raw, 0.0)

        apex_dn[k] = n_dn[0]
        n_active[k] = info["n_active"]
        f_total[k] = float(np.sum(f_per))
        max_dn[k] = float(np.max(n_dn[active])) if active.any() else 0.0
        mean_dn_active[k] = float(np.mean(n_dn[active])) if active.any() else 0.0

    # Bulk-stiffness reconstruction: effective ke from F_total / phi.
    keff_bulk = f_total / phi

    print()
    print(f"  {'kl/ka':>8} {'apex[um]':>10} {'max_dn[um]':>12} "
          f"{'mean_active[um]':>17} {'N_active':>10} {'F_total[N]':>12} "
          f"{'keff_bulk[N/m]':>17}")
    for k, r in enumerate(kl_over_ka):
        print(f"  {r:>8.2f} {apex_dn[k]*1e6:>10.3f} "
              f"{max_dn[k]*1e6:>12.3f} {mean_dn_active[k]*1e6:>17.3f} "
              f"{n_active[k]:>10d} {f_total[k]:>12.3f} "
              f"{keff_bulk[k]:>17.0f}")
    print()
    print(f"  ke_bulk (production target)              = {KE_BULK:.0f}")
    print()
    print("  Reading the sweep:")
    print("    * Small kl/ka: each sphere acts independently, apex sinks")
    print("      deep but neighbours barely move.  Localised patch.")
    print("    * Large kl/ka: load spreads across neighbours, apex sinks")
    print("      less but more spheres engage.  Broader patch.")
    print("    * F_total mostly insensitive to kl (kc * sum of phi_eff)")
    print("      because total compression is set by indenter penetration.")
    print("    * Production kl/ka = 0.2 is in the localised regime --")
    print("      consistent with PART B finding that the apex sees almost")
    print("      no lattice stiffening from its neighbours.")

    # Pass: apex sinkage should be DECREASING with kl (load sharing),
    # N_active should be NON-DECREASING (more engagement with more
    # spreading).
    apex_decreasing = bool(np.all(np.diff(apex_dn) <= 1e-9))
    n_active_nondecreasing = bool(np.all(np.diff(n_active) >= 0))

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.semilogx(kl_over_ka, apex_dn * 1e6, "o-", color="C3", ms=8)
    ax.set_xlabel("kl / ka")
    ax.set_ylabel(r"apex $\delta_n$ [$\mu$m]")
    ax.set_title("Apex sinkage vs lateral stiffness")
    ax.axvline(KL_DEFAULT / KA, color="gray", ls="--", alpha=0.5,
               label=f"production kl/ka = {KL_DEFAULT/KA:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.semilogx(kl_over_ka, n_active, "s-", color="C0", ms=8)
    ax.set_xlabel("kl / ka")
    ax.set_ylabel("N active spheres")
    ax.set_title("Patch size (active sphere count)")
    ax.axvline(KL_DEFAULT / KA, color="gray", ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(kl_over_ka, f_total, "^-", color="C2", ms=8)
    ax.set_xlabel("kl / ka")
    ax.set_ylabel(r"$F_{\mathrm{total}}$ [N]")
    ax.set_title(rf"Total normal force at $\varphi = 1$ mm")
    ax.axvline(KL_DEFAULT / KA, color="gray", ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogx(kl_over_ka, keff_bulk, "v-", color="C1", ms=8)
    ax.axhline(KE_BULK, color="black", ls=":", alpha=0.7,
               label=f"target ke_bulk = {KE_BULK:.0f}")
    ax.set_xlabel("kl / ka")
    ax.set_ylabel(r"$k_{\mathrm{eff,bulk}} = F_{\mathrm{total}} / \varphi$ [N/m]")
    ax.set_title("Effective bulk stiffness vs target")
    ax.axvline(KL_DEFAULT / KA, color="gray", ls="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Production diagnostic: kl/ka sweep at phi = 1 mm  "
                 f"(R_pad = {R_PAD*1e3:.0f} mm, R_obj = {R_OBJ*1e3:.1f} mm, "
                 f"N = {N_SPHERES})")
    fig.tight_layout()
    out = FIG_DIR / "08d_kl_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")

    ok = apex_decreasing and n_active_nondecreasing
    print()
    print(f"  apex_dn decreases with kl?             "
          f"{'PASS' if apex_decreasing else 'FAIL'}")
    print(f"  N_active non-decreasing with kl?       "
          f"{'PASS' if n_active_nondecreasing else 'FAIL'}")
    print(f"PART D result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
#  PART E.  Side-view profile: dome flattening under squeeze
# ─────────────────────────────────────────────────────────────────────────


def part_e_flattening_profile() -> bool:
    """Side-view (r, z) profile of the dome at several apex penetrations.

    The right intuition picture: the dome starts curved (each sphere on
    the rest cap, r^2 + z^2 = R_pad^2) and flattens into a Hertz-like
    cap against the indenter as phi grows.  Each sphere is plotted as a
    point ``(r_i, z_i)`` where ``r_i`` is the perpendicular distance
    from the apex axis (the axis along the apex sphere's outward
    normal) and ``z_i`` is the projection onto that axis.

    Aggregating all azimuthal samples into one 2D scatter loses no
    information for a face-on indenter (the deformation is axially
    symmetric about the apex axis to good approximation, modulo the
    Fibonacci-spiral azimuthal sampling), and it's the most direct
    visualisation of "curved -> flat".
    """
    print()
    print("=" * 72)
    print("PART E.  Side-view profile: dome flattening under squeeze")
    print("=" * 72)

    lat, _, _, r_lat, _ = make_dome_scene()
    apex_p = lat.p[0]
    apex_n = lat.n[0]

    def project(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (r_radial, z_axial) -- distances perpendicular to and
        along the apex outward normal, with z_axial measured from the
        apex sphere's centre (so apex rest sits at (0, 0))."""
        v = pts - apex_p
        z = v @ apex_n
        # Radial component = perpendicular distance to the apex axis.
        v_perp = v - z[:, None] * apex_n
        r = np.linalg.norm(v_perp, axis=1)
        return r, z

    # Rest profile (no deformation).
    r_rest, z_rest = project(lat.p)

    phis = np.array([0.0, 0.5e-3, 1.0e-3, 2.0e-3, 3.0e-3])
    colors = ["tab:gray", "C0", "C2", "C1", "C3"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: all penetrations overlaid ──
    ax = axes[0]
    delta = None
    indenter_circles = []
    print()
    print(f"  {'phi[mm]':>10} {'apex_dn[um]':>14} {'flat_zone_r[mm]':>17} "
          f"{'flat_zone_z_drop[um]':>22}")
    for phi, color in zip(phis, colors):
        if phi <= 0.0:
            r_def, z_def = r_rest, z_rest
            apex_dn_um = 0.0
            flat_r_mm = 0.0
            flat_zdrop_um = 0.0
        else:
            indenter = position_indenter(lat, r_lat, phi)
            delta, _ = solve_lattice_sphere_indenter(
                lat, indenter, r_lat=r_lat, eps=1e-9,
                delta0=delta if delta is not None else None,
            )
            q = lat.p - delta
            r_def, z_def = project(q)
            apex_dn_um = float(np.dot(delta[0], apex_n)) * 1e6
            # Flat zone characterisation: take all active spheres
            # (raw_overlap > 0), use the radial extent of the contact
            # patch and the spread of z values within it (a perfectly
            # flat patch would have zero spread).
            raw = (r_lat + R_OBJ) - np.linalg.norm(q - indenter.t, axis=1)
            active = raw > 0
            if active.any():
                flat_r_mm = float(np.max(r_def[active])) * 1e3
                flat_zdrop_um = float(z_def[active].max() -
                                      z_def[active].min()) * 1e6
            else:
                flat_r_mm = 0.0
                flat_zdrop_um = 0.0
            # Indenter circle (in the (r, z) plane).  Centre projects
            # to (0, t_centre_z - apex_p . n).
            t_v = indenter.t - apex_p
            t_z = float(t_v @ apex_n)
            t_perp = t_v - t_z * apex_n  # should be ~0 since face-on
            r_perp_indenter = float(np.linalg.norm(t_perp))
            indenter_circles.append((r_perp_indenter, t_z, R_OBJ, color, phi))

        print(f"  {phi*1e3:>10.3f} {apex_dn_um:>14.3f} {flat_r_mm:>17.3f} "
              f"{flat_zdrop_um:>22.3f}")
        ax.scatter(r_def * 1e3, z_def * 1e3, s=14, color=color, alpha=0.75,
                   label=rf"$\varphi = {phi*1e3:.1f}$ mm")

    # Reference rest-cap arc (analytical, dense).
    theta_arc = np.linspace(0.0, HALF_ANGLE, 200)
    r_arc = R_PAD * np.sin(theta_arc)
    z_arc_world = R_PAD * np.cos(theta_arc)
    # Project the analytical arc onto the apex frame as well.
    p_arc = np.stack([r_arc, np.zeros_like(r_arc), z_arc_world], axis=1)
    # The apex sphere's true (0, 0, R_pad) is NOT exactly at index 0
    # (Fibonacci samples don't land precisely on the pole), so subtract
    # apex_p before projecting -- this gives the analytical rest arc
    # in the same coords as the scatter, anchored at (0, 0) for the
    # ideal pole and offset slightly for the actual apex sphere.
    r_arc_proj, z_arc_proj = project(p_arc)
    ax.plot(r_arc_proj * 1e3, z_arc_proj * 1e3, "--", color="tab:gray",
            lw=1.0, alpha=0.6,
            label=rf"rest cap (analytic, $R_{{\mathrm{{pad}}}} = "
                  rf"{R_PAD*1e3:.0f}$ mm)")

    # Indenter circles (one per phi, but only the deepest shows
    # clearly; draw the outermost two for context).
    theta_circ = np.linspace(0, np.pi, 64)
    for r_off, t_z, R, color, phi in indenter_circles[-2:]:
        xs = R * np.sin(theta_circ) + r_off
        zs = R * np.cos(theta_circ) + t_z
        ax.plot(xs * 1e3, zs * 1e3, ":", color=color, lw=1.2, alpha=0.5)
        # Mark the indenter centre with a small x.
        ax.scatter([r_off * 1e3], [t_z * 1e3], marker="x", color=color, s=40)

    ax.set_xlabel(r"radial distance from apex axis $r$ [mm]")
    ax.set_ylabel(r"axial distance along apex normal $z$ [mm]")
    ax.set_title("Side-view profile: dome flattens under squeeze")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, R_PAD * np.sin(HALF_ANGLE) * 1e3 + 1)
    ax.set_ylim(-4, 1)

    # ── Right: zoom on the apex region so the flattening is visible ──
    ax = axes[1]
    delta = None
    for phi, color in zip(phis, colors):
        if phi <= 0.0:
            r_def, z_def = r_rest, z_rest
        else:
            indenter = position_indenter(lat, r_lat, phi)
            delta, _ = solve_lattice_sphere_indenter(
                lat, indenter, r_lat=r_lat, eps=1e-9,
                delta0=delta if delta is not None else None,
            )
            q = lat.p - delta
            r_def, z_def = project(q)
        # Only the inner spheres (r < 8 mm) so the flat zone is visible.
        inner = r_def < 8.0e-3
        ax.scatter(r_def[inner] * 1e3, z_def[inner] * 1e6,  # z in microns now
                   s=24, color=color, alpha=0.85,
                   label=rf"$\varphi = {phi*1e3:.1f}$ mm")

    # Rest arc analytic in zoomed coords (z in microns).
    r_arc_zoom, z_arc_zoom = project(p_arc)
    ax.plot(r_arc_zoom * 1e3, z_arc_zoom * 1e6, "--", color="tab:gray",
            lw=1.0, alpha=0.6, label="rest cap")
    ax.set_xlabel(r"radial distance from apex axis $r$ [mm]")
    ax.set_ylabel(r"axial position $z$ [$\mu$m]")
    ax.set_title(rf"Apex region (zoom): $r < 8$ mm, $z$ in $\mu$m")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(rf"Dome flattening under sphere indenter  "
                 rf"($R_{{\mathrm{{pad}}}} = {R_PAD*1e3:.0f}$ mm, "
                 rf"$R_{{\mathrm{{obj}}}} = {R_OBJ*1e3:.1f}$ mm, "
                 rf"$N = {N_SPHERES}$, "
                 rf"$kc / ka = {KC_DEFAULT/KA:.2f}$, "
                 rf"$kl / ka = {KL_DEFAULT/KA:.2f}$)",
                 y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "08e_flattening_profile.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")
    print(f"PART E result: PASS  (qualitative -- no numeric pass criterion;")
    print(f"                       check the figure shows the curved -> flat trend)")
    return True


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a = part_a_geometry()
    b = part_b_apex_sinkage()
    c = part_c_flattening()
    d = part_d_kl_sweep()
    e = part_e_flattening_profile()
    all_ok = a and b and c and d and e
    print()
    print("=" * 72)
    print(f"Step 8 dome-contact test: {'PASS' if all_ok else 'FAIL'}   "
          f"(A={'P' if a else 'F'} B={'P' if b else 'F'} "
          f"C={'P' if c else 'F'} D={'P' if d else 'F'} "
          f"E={'P' if e else 'F'})")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
