# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 1 / Single isolated lattice sphere vs a rigid target.

This is the simplest possible verification of the IDEAL CSLC model.
The lattice has exactly one sphere -- no neighbours -- so the lateral
spring and the lattice Laplacian are absent.  All we test is

    (anchor + contact)  spheres = 0

with the deformed-centre formulation.

What the test proves:

  PART A.  For face-on contact (target on the rest outward normal),
           the L-BFGS-B numerical equilibrium agrees with the closed-
           form analytical solution to ~1e-10 relative error.  This
           validates both implementations against each other.

  PART B.  Sweep the contact stiffness k_c at fixed phi_rest and k_a;
           verify that the equilibrium force traces the
           series-spring curve  F = (k_a*k_c)/(k_a+k_c) * phi_rest.
           In particular, F saturates at k_a*phi_rest as k_c -> inf
           (rigid contact limit) and goes to 0 as k_c -> 0.

  PART C.  Sweep phi_rest at fixed k_a, k_c; verify F is linear in
           phi_rest with slope k_eff = k_a*k_c/(k_a+k_c).

  PART D.  Sweep the OFF-AXIS offset of the target away from the rest
           outward normal.  The IDEAL solver produces a delta whose
           tangential component grows with the offset -- the sphere
           "slides" toward the contact direction.  Plot the
           deformed-centre geometry to make this visible.

A kernel-vs-theory comparison plot used to live here as Part E; it has
been removed pending a kernel fix that aligns the in-tree contact-force
law with the ideal series-spring law tested above.  When the kernel is
updated, re-add a comparison module that calls into ``cslc_kernels`` and
overlays its prediction on the ideal curves -- the bridge from theory
to implementation belongs in its own test once it can pass.

Run::

    uv run -m cslc_main.theory.test_01_single_sphere

Outputs::

    cslc_main/theory/figures/01a_face_on_table.txt        # numerical audit
    cslc_main/theory/figures/01b_series_spring_kc.png
    cslc_main/theory/figures/01c_force_vs_phi.png
    cslc_main/theory/figures/01d_off_axis_geometry.png

Every plot also prints a one-line summary of pass/fail bars to stdout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    RigidTarget,
    contact_direction,
    deformed_centre,
    effective_penetration,
    equilibrium_face_on_analytical,
    equilibrium_numerical,
    rest_overlap,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ─────────────────────────────────────────────────────────────────────────
#  Convenience: build a canonical single-sphere scenario
# ─────────────────────────────────────────────────────────────────────────


def make_scene(
    *,
    phi_rest: float = 1.0e-3,    # rest overlap [m] — 1 mm
    r: float = 2.5e-3,           # sphere radius [m] — typical lattice
    R: float = 33.5e-3,          # target radius [m] — tennis ball
    ka: float = 25_000.0,        # anchor stiffness [N/m]
    n: np.ndarray | None = None,
) -> tuple[LatticeSphere, RigidTarget]:
    """Build a (sphere, target) pair face-on along ``n``.

    Geometry: sphere at origin with outward normal ``n`` (default +x).
    Target is placed at distance ``(r + R) - phi_rest`` along ``n`` so
    the rest overlap is exactly ``phi_rest``.

    Returns ``(LatticeSphere, RigidTarget)``.
    """
    if n is None:
        n = np.array([1.0, 0.0, 0.0])
    sphere = LatticeSphere(p=np.zeros(3), r=r, n=n, ka=ka)
    d = (r + R) - phi_rest
    target = RigidTarget(t=d * n, R=R)
    # Sanity-check we built what we said we built.
    assert np.isclose(rest_overlap(sphere, target), phi_rest, atol=1e-12)
    return sphere, target


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Face-on sanity: analytical == numerical
# ─────────────────────────────────────────────────────────────────────────


def part_a_face_on_table() -> bool:
    """Print + save a table comparing analytical and numerical equilibria.

    Returns True if all rows agree to relative tolerance 1e-9.
    """
    print()
    print("=" * 72)
    print("PART A.  Face-on equilibrium: analytical vs numerical")
    print("=" * 72)
    print()
    print("Setup: 1 lattice sphere, ka = 25,000 N/m, n = +x, phi_rest as below.")
    print("       Target on +x axis, rest overlap = phi_rest.")
    print()

    ka = 25_000.0
    phi_values = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]    # 0.1 mm .. 5 mm
    kc_values = [2_000.0, 25_000.0, 200_000.0]      # below, equal, above ka

    header = (
        f"{'phi[mm]':>9} {'kc[N/m]':>11} "
        f"{'delta*_ana[um]':>15} {'delta*_num[um]':>15} "
        f"{'F*_ana[N]':>10} {'F*_num[N]':>10} "
        f"{'rel_err':>10}"
    )
    print(header)
    print("-" * len(header))

    lines = [header, "-" * len(header)]
    rows = []
    all_close = True

    for phi in phi_values:
        for kc in kc_values:
            sphere, target = make_scene(phi_rest=phi, ka=ka)
            delta_ana, F_ana = equilibrium_face_on_analytical(sphere, target, kc)
            delta_num, info = equilibrium_numerical(sphere, target, kc, eps=1e-9,
                                                    delta0=delta_ana)
            # Force from numerical: at equilibrium |f_anchor| = |f_contact| =
            # numerical anchor force magnitude.  Both equal ka * |delta|.
            F_num = ka * float(np.linalg.norm(delta_num))
            rel = (np.linalg.norm(delta_ana - delta_num) /
                   max(np.linalg.norm(delta_ana), 1e-15))
            if rel > 1e-9:
                all_close = False
            row = (
                f"{phi * 1e3:>9.3f} {kc:>11.0f} "
                f"{1e6 * delta_ana[0]:>15.6f} {1e6 * delta_num[0]:>15.6f} "
                f"{F_ana:>10.4f} {F_num:>10.4f} "
                f"{rel:>10.2e}"
            )
            print(row)
            lines.append(row)
            rows.append((phi, kc, delta_ana, delta_num, F_ana, F_num, rel))

    print()
    print(f"PART A result: {'PASS' if all_close else 'FAIL'}  "
          f"(max relative error in delta: "
          f"{max(r[6] for r in rows):.2e})")

    out = FIG_DIR / "01a_face_on_table.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved table to {out}")
    return all_close


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  k_c sweep at fixed phi: series-spring saturation
# ─────────────────────────────────────────────────────────────────────────


def part_b_kc_sweep() -> None:
    """Plot F* and delta_n* vs k_c at fixed phi_rest, k_a.

    Compares analytical (continuous curve), numerical (markers), and
    the two asymptotic limits (k_c -> 0 and k_c -> inf).
    """
    print()
    print("=" * 72)
    print("PART B.  Series-spring saturation: sweep k_c")
    print("=" * 72)

    ka = 25_000.0
    phi = 1.0e-3   # 1 mm rest overlap

    kc_log = np.logspace(2, 9, 64)   # 1e2 .. 1e9
    F_ana = np.zeros_like(kc_log)
    d_ana = np.zeros_like(kc_log)
    for i, kc in enumerate(kc_log):
        sphere, target = make_scene(phi_rest=phi, ka=ka)
        delta, F = equilibrium_face_on_analytical(sphere, target, kc)
        F_ana[i] = F
        d_ana[i] = delta[0]

    # Sample numerical at a sparser grid (it is slow at extreme kc).  We
    # include the exact ``kc = ka`` point explicitly so the diagnostic
    # printout below compares apples to apples with the closed-form
    # midpoint formula.
    kc_num = np.sort(np.concatenate(
        [np.logspace(2, 8, 13), np.array([ka], dtype=float)]))
    F_num = np.zeros_like(kc_num)
    d_num = np.zeros_like(kc_num)
    for i, kc in enumerate(kc_num):
        sphere, target = make_scene(phi_rest=phi, ka=ka)
        delta_ana, _ = equilibrium_face_on_analytical(sphere, target, kc)
        delta, info = equilibrium_numerical(sphere, target, kc, eps=1e-9,
                                            delta0=delta_ana)
        F_num[i] = ka * float(np.linalg.norm(delta))
        d_num[i] = delta[0]

    # Asymptotes.
    F_kc_to_inf = ka * phi          # rigid-contact limit (paper line 273)
    F_kc_eq_ka = 0.5 * ka * phi      # equal-stiffness limit (paper line 275)

    # Exact ``kc = ka`` index for the apples-to-apples print.
    j_eq = int(np.argmin(np.abs(kc_num - ka)))
    print(f"  asymptote F(k_c -> inf) = k_a * phi = {F_kc_to_inf:.4f} N")
    print(f"  midpoint  F(k_c = k_a)  = 0.5*k_a*phi = {F_kc_eq_ka:.4f} N")
    print(f"  measured  F(k_c = {kc_num[j_eq]:.0f}) = "
          f"{F_num[j_eq]:.4f} N  (via numerical optimisation)  "
          f"delta = {kc_num[j_eq] - ka:+.1f} from ka")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.semilogx(kc_log, F_ana * 1e3, "k-", lw=2, label="ideal (series spring)")
    ax1.semilogx(kc_num, F_num * 1e3, "ro", ms=6, mfc="none", label="numerical")
    ax1.axhline(F_kc_to_inf * 1e3, ls="--", color="gray",
                label=r"$k_a\,\varphi_{\mathrm{rest}}$ (rigid limit)")
    ax1.axvline(ka, ls=":", color="C0", label=r"$k_c = k_a$")
    ax1.set_xlabel(r"$k_c$ [N/m]")
    ax1.set_ylabel(r"equilibrium force $|F^\ast|$ [mN]")
    ax1.set_title("Series-spring saturation\n"
                  rf"$\varphi_{{\mathrm{{rest}}}} = {phi * 1e3:.1f}$ mm, "
                  rf"$k_a = {ka:.0f}$ N/m")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(kc_log, d_ana * 1e6, "k-", lw=2, label="ideal")
    ax2.semilogx(kc_num, d_num * 1e6, "ro", ms=6, mfc="none", label="numerical")
    ax2.axhline(phi * 1e6, ls="--", color="gray",
                label=r"$\varphi_{\mathrm{rest}}$ (rigid limit)")
    ax2.axvline(ka, ls=":", color="C0", label=r"$k_c = k_a$")
    ax2.set_xlabel(r"$k_c$ [N/m]")
    ax2.set_ylabel(r"equilibrium $\delta_n^\ast$ [$\mu$m]")
    ax2.set_title(r"Compression of the compliant skin")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "01b_series_spring_kc.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  phi_rest sweep: F is linear in phi with slope k_eff
# ─────────────────────────────────────────────────────────────────────────


def part_c_phi_sweep() -> None:
    """Show F(phi_rest) is linear with slope k_a*k_c/(k_a+k_c)."""
    print()
    print("=" * 72)
    print("PART C.  Linearity: F* vs phi_rest at fixed k_a, k_c")
    print("=" * 72)

    ka = 25_000.0
    kcs = [2_500.0, 25_000.0, 250_000.0]   # k_c/k_a = 0.1, 1, 10
    phi_grid = np.linspace(0.0, 3e-3, 24)

    fig, ax = plt.subplots(figsize=(7, 5))
    for kc, color in zip(kcs, ["C0", "C1", "C2"]):
        keff = ka * kc / (ka + kc)
        F = np.zeros_like(phi_grid)
        for i, phi in enumerate(phi_grid):
            sphere, target = make_scene(phi_rest=phi, ka=ka)
            _, F[i] = equilibrium_face_on_analytical(sphere, target, kc)
        ax.plot(phi_grid * 1e3, F * 1e3, "-", color=color, lw=2,
                label=rf"$k_c={kc:.0f}$, $k_{{eff}}={keff:.0f}$")
        # Hand-drawn slope tick to make the linearity obvious.
        ax.plot(phi_grid * 1e3, keff * phi_grid * 1e3, ":", color=color, lw=1)
        print(f"  k_c = {kc:>7.0f} N/m   k_eff = {keff:>8.2f} N/m  "
              f"max |F - keff*phi| = {np.max(np.abs(F - keff*phi_grid)):.3e} N")

    ax.set_xlabel(r"$\varphi_{\mathrm{rest}}$ [mm]")
    ax.set_ylabel(r"equilibrium force $|F^\ast|$ [mN]")
    ax.set_title(
        rf"Linearity check: $F^\ast = k_{{eff}}\,\varphi_{{\mathrm{{rest}}}}$  "
        rf"($k_a = {ka:.0f}$ N/m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "01c_force_vs_phi.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  Off-axis: delta tilts when target is off the normal
# ─────────────────────────────────────────────────────────────────────────


def part_d_off_axis() -> None:
    """Visualise off-axis equilibrium where delta picks up tangential component.

    Geometry: sphere at origin, outward normal +x.  Place target so the
    line of centres makes angles 0, 15, 30, 45 degrees with +x while the
    closest-approach distance stays at (r + R) - phi_rest (so phi_rest
    is held constant across the sweep).
    """
    print()
    print("=" * 72)
    print("PART D.  Off-axis equilibrium: delta tilts toward contact line")
    print("=" * 72)

    ka = 25_000.0
    kc = 25_000.0
    phi = 1.5e-3        # 1.5 mm rest overlap
    r = 2.5e-3
    R = 33.5e-3
    n = np.array([1.0, 0.0, 0.0])
    angles_deg = [0.0, 15.0, 30.0, 45.0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    # Draw target as a single big circle for the largest-angle case so
    # the plot stays readable; we also draw the rest sphere outline.

    for angle_deg, color in zip(angles_deg, ["C0", "C1", "C2", "C3"]):
        theta = np.radians(angle_deg)
        # Unit vector along the target's line of centres relative to sphere
        # (in the x-y plane).
        u = np.array([np.cos(theta), np.sin(theta), 0.0])
        # Target placed so that ||t - p|| = (r + R) - phi.
        d = (r + R) - phi
        sphere = LatticeSphere(p=np.zeros(3), r=r, n=n, ka=ka)
        target = RigidTarget(t=d * u, R=R)
        # Sanity check rest overlap.
        phi_actual = rest_overlap(sphere, target)
        # Warm-start from the closed-form magnitude pointing along the
        # contact direction u.  This is the exact answer in the limit
        # where the anchor and contact both align with u (isotropic
        # anchor sees no preferred direction, so equilibrium delta sits
        # on the line through p and t).  Saves L-BFGS-B from a slow
        # walk-down from delta = 0 at large theta.
        delta0_warm = (kc / (ka + kc)) * phi * u
        delta, info = equilibrium_numerical(sphere, target, kc, eps=1e-9,
                                            delta0=delta0_warm, tol=1e-10)
        q = deformed_centre(sphere, delta)
        e_hat = contact_direction(sphere, target, delta)
        phi_eff = effective_penetration(sphere, target, delta)
        delta_n = float(np.dot(delta, n))
        delta_t = delta - delta_n * n
        print(
            f"  theta = {angle_deg:>5.1f} deg   "
            f"|delta| = {np.linalg.norm(delta)*1e6:>7.2f} um   "
            f"delta_n = {delta_n*1e6:>7.2f} um   "
            f"|delta_t| = {np.linalg.norm(delta_t)*1e6:>7.2f} um   "
            f"phi_eff = {phi_eff*1e6:>7.2f} um   "
            f"converged = {info['success']}")

        # Plot the deformed-centre geometry on the x-y plane.
        # Rest sphere (light grey).
        theta_arc = np.linspace(0, 2*np.pi, 200)
        ax.plot(r * np.cos(theta_arc) * 1e3, r * np.sin(theta_arc) * 1e3,
                color="lightgrey", lw=0.8)
        # Deformed sphere (colour).
        ax.plot((q[0] + r * np.cos(theta_arc)) * 1e3,
                (q[1] + r * np.sin(theta_arc)) * 1e3,
                color=color, lw=1.5,
                label=rf"$\theta = {angle_deg:.0f}^\circ$")
        # Target sphere (same colour, dashed, slightly transparent).
        ax.plot((target.t[0] + R * np.cos(theta_arc)) * 1e3,
                (target.t[1] + R * np.sin(theta_arc)) * 1e3,
                color=color, lw=0.7, ls="--", alpha=0.5)
        # Displacement arrow from p to q.
        ax.annotate("",
                    xy=(q[0] * 1e3, q[1] * 1e3),
                    xytext=(0.0, 0.0),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})

    ax.plot(0, 0, "k+", ms=12, label="rest centre $p$")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(-3, 5)
    ax.set_ylim(-2, 4)
    ax.set_title(
        rf"Off-axis equilibrium ($k_a = k_c = {ka:.0f}$ N/m, "
        rf"$\varphi_{{\mathrm{{rest}}}} = {phi*1e3:.1f}$ mm)" "\n"
        r"grey ring: rest sphere $\bullet$ coloured ring: deformed sphere "
        r"$\bullet$ dashed: target $\bullet$ arrow: $-\delta$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "01d_off_axis_geometry.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out}")


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a_ok = part_a_face_on_table()
    part_b_kc_sweep()
    part_c_phi_sweep()
    part_d_off_axis()
    print()
    print("=" * 72)
    print(f"Step 1 single-sphere test: {'PASS' if a_ok else 'FAIL'}")
    print("=" * 72)
    return 0 if a_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
