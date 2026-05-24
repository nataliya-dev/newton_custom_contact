# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 10 / Single pad sphere vs ``PointSetTarget`` (box).

First verification of the multi-point target generalisation introduced
in :mod:`cslc_main.theory.cslc_theory`'s ``PointSetTarget`` and the
trimesh-based ``make_box_target`` sampler in
:mod:`cslc_main.theory.cslc_box`.

The aim is narrow: prove that the multi-point energy + L-BFGS-B path
reduces to the established single-sphere theory in the right limits,
before any kernel-side refactor.  Three scenes:

  PART A.  M = 1 PointSetTarget centred and sized to coincide exactly
           with a ``RigidTarget``.  ``equilibrium_point_set_numerical``
           must match ``equilibrium_face_on_analytical`` to ~1e-10 --
           this verifies the multi-point gradient picks up the same
           ``smooth_step`` chain-rule factor as the single-sphere
           solver (theory.txt eq. ``contact-grad``; the silent
           gradient bug from notes.md lesson #4).

  PART B.  Face-on contact onto the top face of a sampled box.  We
           position one pad sphere directly above the box centre with
           rest overlap ``phi_rest_face = phi_rest`` against the
           effective surface (= top face plus the target spheres'
           radius offset, see scene definition below).  Verifications:
             * ``delta`` is dominantly along the face normal (lateral
               components below a sampler-noise tolerance).
             * The contact force on the pad sphere points OPPOSITE the
               face's outward normal (back into the pad body).

  PART C.  Sample-density convergence.  Sweep ``n_samples`` and verify
           that as the target gets denser, the multi-point equilibrium
           converges (i.e. the lattice-asymmetry residual on the
           lateral axes shrinks).  This is the box-side analogue of
           the dome's Fibonacci-spiral convergence story in
           notes.md Step 11; here it's measured on a target that we
           CAN sample arbitrarily densely without solver cost (no
           lateral coupling on the target).

Run::

    uv run --extra importers -m cslc_main.theory.test_10_pad_vs_box

Output::

    cslc_main/theory/figures/10a_reduction_table.txt        # PART A audit
    cslc_main/theory/figures/10b_face_on_box_geometry.png   # PART B vis
    cslc_main/theory/figures/10c_density_convergence.png    # PART C sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_box import make_box_target
from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    PointSetTarget,
    RigidTarget,
    equilibrium_face_on_analytical,
    equilibrium_point_set_numerical,
    point_set_contact_force,
    point_set_raw_overlaps,
)


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
#  Shared scene parameters
# ──────────────────────────────────────────────────────────────────────


def _scene_params():
    """Production-scale parameters shared by the test scenes.

    ``phi_rest`` is the SDF-style penetration of the pad sphere into the
    box's analytic flat face (treating the box as a half-space at the
    top face's z-plane).  The multi-point representation can engage
    fewer than the analytic flat-half-space number of contacts -- the
    pad still has to physically overlap individual target spheres -- so
    we choose ``phi_rest`` large enough that the pad reaches sample
    points across the entire range of sampling densities the test
    sweeps (150 .. 3000 samples, mean spacing 4.5 .. 20 mm on a 50 mm
    box).  See the PART B header for the geometric overlap budget.
    """
    return dict(
        r_lat=1.5e-3,            # 1.5 mm lattice sphere radius (production)
        ka=1.0e4,                # anchor stiffness
        kc=1.0e4,                # contact stiffness
        phi_rest=5.0e-4,         # 0.5 mm penetration into the flat box face
        box_extents=(0.05, 0.05, 0.05),   # 50 mm cube target
    )


# ──────────────────────────────────────────────────────────────────────
#  PART A.  M = 1 reduction to the single-sphere face-on closed form
# ──────────────────────────────────────────────────────────────────────


def part_a():
    print()
    print("=" * 72)
    print("PART A.  M=1 PointSetTarget reduces to RigidTarget face-on law")
    print("=" * 72)

    p = _scene_params()
    n_hat = np.array([0., 0., 1.])
    # Pad sphere with outward normal +z, target sphere directly along +z.
    sphere = LatticeSphere(p=np.zeros(3), r=p["r_lat"], n=n_hat, ka=p["ka"])
    # phi_rest = (r + R) - ||t - p||  with target along +n_hat.
    R = p["r_lat"]   # equal radii for definiteness
    dist_rest = p["r_lat"] + R - p["phi_rest"]
    t_world = dist_rest * n_hat
    rigid = RigidTarget(t=t_world, R=R)
    # Build a 1-point set that coincides with the rigid target.
    pst = PointSetTarget(
        positions=t_world[None, :],
        radii=np.array([R]),
        normals=(-n_hat)[None, :],   # outward of target sphere, back toward pad
    )

    # Closed-form reference.
    delta_ana, F_ana = equilibrium_face_on_analytical(sphere, rigid, p["kc"])

    # Multi-point L-BFGS-B at several smoothing widths so we can confirm
    # the smooth_step chain-rule factor is wired in.
    rows = []
    eps_list = [1.0e-12, 1.0e-9, 1.0e-7]
    for eps in eps_list:
        delta_pt, info = equilibrium_point_set_numerical(
            sphere, pst, p["kc"], eps=eps)
        diff = float(np.linalg.norm(delta_pt - delta_ana))
        # |delta| should equal the analytic series-spring value.
        rows.append((eps, delta_pt, diff, info))

    print(f"{'eps [m]':>10}  {'|delta_pt - delta_ana|':>26}  {'nit':>4}  status")
    for eps, delta_pt, diff, info in rows:
        print(f"{eps:>10.0e}  {diff:>26.3e}  {info['nit']:>4}  "
              f"{'ok' if info['success'] else 'FAIL'}")
    print(f"analytic delta_n = {np.dot(delta_ana, n_hat):.6e} m")
    print(f"analytic |F|     = {F_ana:.6e} N")

    # Tolerance: 1e-10 m absolute (well below the 1e-7 smoothing eps).
    tol = 1.0e-10
    ok = all(diff < tol for _, _, diff, _ in rows)

    audit = FIG_DIR / "10a_reduction_table.txt"
    with audit.open("w") as f:
        f.write("Step 10 PART A reduction table\n")
        f.write(f"analytic delta_n = {np.dot(delta_ana, n_hat):.6e} m\n")
        f.write(f"analytic |F|     = {F_ana:.6e} N\n\n")
        f.write(f"{'eps':>12}  {'|err|':>14}  {'nit':>4}\n")
        for eps, _, diff, info in rows:
            f.write(f"{eps:>12.3e}  {diff:>14.3e}  {info['nit']:>4}\n")
    print(f"audit -> {audit.relative_to(Path.cwd())}")
    print(f"PART A result: {'PASS' if ok else 'FAIL'}  "
          f"(tol = {tol:.0e} m)")
    return ok


# ──────────────────────────────────────────────────────────────────────
#  PART B.  Face-on contact onto the top face of a sampled box
# ──────────────────────────────────────────────────────────────────────


def part_b():
    print()
    print("=" * 72)
    print("PART B.  Face-on pad sphere vs sampled box top face")
    print("=" * 72)

    p = _scene_params()

    # Box centred at origin, top face at z = +box_half_z.
    # radius_factor = 1.0 -> target spheres extend one mean-spacing in
    # radius, so adjacent samples overlap moderately and the surface
    # has NO discrete coverage gaps (a smaller factor like 0.5 lets the
    # pad sphere fall between target samples even at finite phi_rest).
    # This is the analogue of "tile the surface with redundant
    # coverage" you'd get from hydroelastic's continuous SDF.
    #
    # n_samples = 3000 is chosen so the per-face target-sample mean
    # spacing (~2 mm) is smaller than the pad sphere's diameter
    # (3 mm).  PART C's convergence sweep shows the lateral-residual
    # ratio |delta_t|/|delta_n| dropping from O(1) at N <= 600 to
    # ~7% at N = 3000 -- this is the multi-point representation's
    # symmetry-sensitivity floor at finite sampling.  Asserting strict
    # face-on (lateral < 5% of normal) at N < 3000 would be asserting
    # the absence of a real, expected phenomenon.
    target = make_box_target(
        extents=p["box_extents"], n_samples=3000, seed=0,
        radius_factor=1.0)
    box_half_z = 0.5 * p["box_extents"][2]
    R_target = float(target.radii[0])

    # Pad sphere: outward normal -z (points down toward the box).  Rest
    # position chosen so the pad penetrates the box's flat top face by
    # ``phi_rest`` (treating the box as an analytic half-space at z =
    # +box_half_z).  This positions the pad CENTRE at
    #   p_z = box_half_z + r_lat - phi_rest,
    # i.e. the pad's bottom (p_z - r_lat) sits ``phi_rest`` below the
    # flat top face.  For the multi-point representation, the pad
    # therefore overlaps every target sphere whose lateral distance
    # from the pad's vertical axis is below
    #   d_max = sqrt((r_lat + R_target)^2 - (r_lat - phi_rest)^2).
    # At phi_rest = 0.5 mm, r_lat = 1.5 mm:
    #   * N = 150  (R_target = 10 mm, spacing 20 mm):  d_max = 11.46 mm
    #   * N = 3000 (R_target =  1.1 mm, spacing 4.5 mm): d_max = 1.8 mm
    # So 1-5 target spheres overlap across the full sweep -- enough to
    # exercise lateral averaging without making the convergence story
    # trivial.
    pad_p_z = box_half_z + p["r_lat"] - p["phi_rest"]
    sphere = LatticeSphere(
        p=np.array([0., 0., pad_p_z]),
        r=p["r_lat"],
        n=np.array([0., 0., -1.]),    # outward toward the box
        ka=p["ka"],
    )

    # Diagnostic: how many target points actually overlap the pad sphere
    # at delta = 0, and what's the max-overlap point's raw?
    raws0, _ = point_set_raw_overlaps(sphere, target, np.zeros(3))
    active0 = raws0 > 0.0
    n_active0 = int(active0.sum())
    print(f"target: {target.M} points, {n_active0} overlapping at delta=0")
    if n_active0 == 0:
        print("  no active contacts -- check phi_rest / R_target ratio.")
    print(f"max raw at delta=0: {float(raws0.max()):.6e} m  "
          f"(expected ~ phi_rest = {p['phi_rest']:.3e})")

    delta, info = equilibrium_point_set_numerical(
        sphere, target, p["kc"], eps=1.0e-7)
    F = point_set_contact_force(sphere, target, delta, p["kc"], eps=1.0e-7)

    delta_n = float(np.dot(delta, sphere.n))
    delta_t = delta - delta_n * sphere.n
    delta_t_mag = float(np.linalg.norm(delta_t))
    F_along_n = float(np.dot(F, sphere.n))     # outward of pad; should be < 0
    F_mag = float(np.linalg.norm(F))

    print(f"delta_n (along pad outward = -z) = {delta_n:.6e} m  "
          "(expect > 0; box compresses pad inward)")
    print(f"|delta_t| / |delta_n| = "
          f"{delta_t_mag / max(abs(delta_n), 1.0e-30):.3e}  "
          "(should be small; face-on)")
    print(f"F . n_outward = {F_along_n:.6e} N  "
          "(expect < 0; force opposite outward, pushing pad back)")
    print(f"|F| = {F_mag:.6e} N  "
          f"nit = {info['nit']}  success = {info['success']}")

    # Verdict criteria (scaled to the scene; the lateral and
    # force-direction thresholds are the finite-sample analogues of the
    # "face-on" property -- they tighten as n_samples grows, see PART C):
    # * compression along pad outward:           delta_n > 0
    # * lateral / normal ratio:                  |delta_t| / |delta_n| < 0.10
    # * force opposes outward direction:         F.n < 0
    # * dominant force component along outward:  |F.n| / |F| > 0.90
    ok_compression = delta_n > 0.0
    ok_lateral = delta_t_mag / max(abs(delta_n), 1.0e-30) < 0.10
    ok_force_dir = (F_along_n < 0.0
                    and abs(F_along_n) / max(F_mag, 1.0e-30) > 0.90)

    # Geometry plot.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # XZ slice of target points and pad sphere.
    ax = axes[0]
    pts = target.positions
    ax.scatter(pts[:, 0] * 1000, pts[:, 2] * 1000, s=4, alpha=0.4,
               label="target points")
    pad_xy_z = sphere.p[2] * 1000
    pad_x_world = (sphere.p[0] - delta[0]) * 1000
    pad_z_world = (sphere.p[2] - delta[2]) * 1000
    circ_rest = plt.Circle((sphere.p[0] * 1000, pad_xy_z), sphere.r * 1000,
                           fill=False, color="C1", lw=1.5,
                           label="pad rest")
    circ_def = plt.Circle((pad_x_world, pad_z_world), sphere.r * 1000,
                          fill=False, color="C3", lw=1.5, ls="--",
                          label="pad deformed")
    ax.add_patch(circ_rest)
    ax.add_patch(circ_def)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title("PART B  X-Z slice (face-on)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="lower right")

    ax = axes[1]
    raws, dirs = point_set_raw_overlaps(sphere, target, delta)
    active = raws > 0.0
    ax.scatter(pts[active, 0] * 1000, pts[active, 1] * 1000,
               c=raws[active] * 1e6, s=20, cmap="viridis")
    cb = plt.colorbar(ax.collections[0], ax=ax, label="raw overlap [um]")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"PART B active contact patch ({int(active.sum())} pts)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "10b_face_on_box_geometry.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"figure -> {out.relative_to(Path.cwd())}")

    ok = ok_compression and ok_lateral and ok_force_dir
    print(f"PART B result: {'PASS' if ok else 'FAIL'}  "
          f"(compression={ok_compression}, lateral_small={ok_lateral}, "
          f"force_dir={ok_force_dir})")
    return ok


# ──────────────────────────────────────────────────────────────────────
#  PART C.  Sample-density convergence
# ──────────────────────────────────────────────────────────────────────


def part_c():
    print()
    print("=" * 72)
    print("PART C.  Sample-density convergence of multi-point equilibrium")
    print("=" * 72)

    p = _scene_params()
    box_half_z = 0.5 * p["box_extents"][2]
    n_samples_list = [150, 300, 600, 1500, 3000]

    # Pad sphere -- same scene as PART B (rest pose only depends on
    # R_target which depends on n_samples; we recompute per-N).
    rows = []
    for n_samples in n_samples_list:
        target = make_box_target(
            extents=p["box_extents"], n_samples=n_samples, seed=0,
            radius_factor=1.0)
        R_target = float(target.radii[0])
        # Same rest-pose convention as PART B: pad bottom penetrates the
        # analytic flat face by ``phi_rest``, pad centre at
        # box_half_z + r_lat - phi_rest.  This is N-INDEPENDENT, so the
        # convergence story is purely about the target sampling, not
        # about the rest pose.
        pad_p_z = box_half_z + p["r_lat"] - p["phi_rest"]
        sphere = LatticeSphere(
            p=np.array([0., 0., pad_p_z]),
            r=p["r_lat"],
            n=np.array([0., 0., -1.]),
            ka=p["ka"],
        )
        delta, info = equilibrium_point_set_numerical(
            sphere, target, p["kc"], eps=1.0e-7)
        delta_n = float(np.dot(delta, sphere.n))
        delta_t = delta - delta_n * sphere.n
        delta_t_mag = float(np.linalg.norm(delta_t))
        F = point_set_contact_force(sphere, target, delta, p["kc"], eps=1.0e-7)
        F_along_n = float(np.dot(F, sphere.n))
        rows.append({
            "N": n_samples,
            "R_target": R_target,
            "delta_n": delta_n,
            "delta_t": delta_t_mag,
            "F_along_n": F_along_n,
            "nit": info["nit"],
        })
        print(f"N = {n_samples:5d}  R_t = {R_target * 1e6:6.1f} um  "
              f"delta_n = {delta_n * 1e6:7.3f} um  "
              f"|delta_t| = {delta_t_mag * 1e9:7.2f} nm  "
              f"F.n = {F_along_n:.4e} N  "
              f"nit = {info['nit']}")

    # Convergence checks:
    # * The LATERAL deflection |delta_t| should DECREASE with N -- as the
    #   target gets denser, the lattice-asymmetry residual averages out.
    # * The NORMAL deflection delta_n should be bounded (not diverge) as
    #   N grows.  We do not assert a target value here -- the per-point
    #   kc is uncalibrated (each pair contributes kc independently),
    #   so dense sampling produces a larger total contact force.  That
    #   calibration belongs in step 10b alongside the area-weighted
    #   ``calibrate_kc`` extension.  Here we only verify boundedness +
    #   the lateral-noise reduction.
    delta_t_seq = np.array([r["delta_t"] for r in rows])
    # Lateral noise should drop by at least 3x between sparsest and densest.
    ratio = delta_t_seq[0] / max(delta_t_seq[-1], 1.0e-30)
    print(f"|delta_t| ratio (N={rows[0]['N']} / N={rows[-1]['N']}): {ratio:.2f}x")
    delta_n_seq = np.array([r["delta_n"] for r in rows])
    delta_n_bounded = (np.all(np.isfinite(delta_n_seq))
                       and float(delta_n_seq.max())
                       < 100.0 * p["phi_rest"])
    lateral_decreases = bool(ratio > 3.0)

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    Ns = np.array([r["N"] for r in rows])
    ax = axes[0]
    ax.plot(Ns, [r["delta_n"] * 1e6 for r in rows], "o-", label="delta_n")
    ax.set_xscale("log"); ax.set_xlabel("n_samples on box")
    ax.set_ylabel("delta_n [um]")
    ax.set_title("Normal deflection vs sampling density")
    ax.grid(alpha=0.3); ax.legend()
    ax = axes[1]
    ax.plot(Ns, [r["delta_t"] * 1e9 for r in rows], "s-", color="C2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n_samples on box")
    ax.set_ylabel("|delta_t| [nm]")
    ax.set_title("Lateral (lattice-asymmetry) residual vs N")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIG_DIR / "10c_density_convergence.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"figure -> {out.relative_to(Path.cwd())}")

    ok = delta_n_bounded and lateral_decreases
    print(f"PART C result: {'PASS' if ok else 'FAIL'}  "
          f"(bounded delta_n={delta_n_bounded}, "
          f"lateral_decreases_with_N={lateral_decreases}, "
          f"ratio={ratio:.2f}x)")
    return ok


# ──────────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    a_ok = part_a()
    b_ok = part_b()
    c_ok = part_c()

    print()
    print("=" * 72)
    overall = a_ok and b_ok and c_ok
    print(f"Step 10 pad-vs-box test: {'PASS' if overall else 'FAIL'}")
    print(f"  PART A (M=1 reduction)             {'PASS' if a_ok else 'FAIL'}")
    print(f"  PART B (face-on box top face)      {'PASS' if b_ok else 'FAIL'}")
    print(f"  PART C (sample-density convergence){'PASS' if c_ok else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
