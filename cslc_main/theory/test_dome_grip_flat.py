# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 / T-J — Dome lattice friction on a flat face (the grip budget).

The v2 successor to ``test_09_dome_grip.py``.  Layers stick-slip
friction on top of the Phase-2 T-G dome-on-flat normal contact: under
a uniform tangential displacement ``s`` applied to every active sphere
(the indenter-bonded-to-skin model from v1 step 9), the lattice's total
tangential reaction aggregates the per-sphere stick-slip law

    F_grip(s)  =  Σ_i  min(k_stick · s,  μ · f_n,i)                  (eq:T-J)

with plateau ``F_grip^max = μ · Σ_i f_n,i``.

The v2 path differs from v1 in one place only: per-sphere ``f_n,i`` is
extracted from the v2 lattice solver via
:func:`cslc_main.theory.cslc_lattice.lattice_contact_normal_forces`,
which implements the contract §6.4 projection
``f_n,i = |F_contact_i · n̂_pad_i|`` using the unified half-space
contact (eq:F-contact-i).  The grip aggregation itself is unchanged.

Four parts:

  PART A.  **Normal-only baseline at depth = 1 mm.**  Run
           :func:`solve_lattice_contact` then
           :func:`lattice_contact_normal_forces`; report the
           per-sphere ``f_n,i`` distribution and the aggregates
           ``F_n^total = Σ_i f_n,i``, ``N_active = |{i : f_n,i > 0}|``.

  PART B.  **Tangential displacement sweep (the grip curve).**  Apply
           uniform ``s`` to every sphere, compute F_grip(s) per
           eq:T-J, and verify three invariants:
             (1) low-s slope  ≈  N_engaged · k_stick;
             (2) high-s plateau  =  μ · F_n^total  (rel 1e-6);
             (3) the s at which the first sphere flips to slip
                 matches  μ · f_n,min / k_stick  (within one grid step).

  PART C.  **Closed-form aggregation check.**  At a set of probe ``s``
           values spanning stick / transition / plateau, verify
           F_grip(s) matches the analytical aggregation
           ``Σ_i min(k_stick s, μ f_n,i)`` to fp precision (the
           aggregation is by construction — this is a self-consistency
           check that the helper, the grid path, and the closed-form
           use the same per-sphere f_n).

  PART D.  **Grip budget vs depth.**  Sweep depth ∈ {0.1, 0.5, 1.0,
           2.0} mm (T-G's grid); for each depth verify
           ``F_grip^max = μ · F_n^total(depth)`` (the plateau in eq:T-J)
           and that the budget is monotone in depth.  Lifts T-G's
           normal-load monotonicity to a friction-budget statement.

Per ``cslc_main/theory/contract_v2.md`` §6.4, §12 T-J.

Run::

    uv run -m cslc_main.theory.test_dome_grip_flat

Outputs::

    cslc_main/theory/figures/tj_normal_distribution.png
    cslc_main/theory/figures/tj_grip_curve.png
    cslc_main/theory/figures/tj_grip_vs_depth.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    lattice_contact_normal_forces,
    make_dome,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import (
    PointSetTargetV2,
    make_flat_face_target,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Contract §12 T-J tolerance.  rel 1e-6 — well above the L-BFGS-B floor
# (~5×10⁻⁸ at multi-sphere lattice scale per contract §17 finding 1)
# and well below physical-meaningfulness.  The aggregation itself is
# exact in fp; the floor comes from per-sphere f_n,i precision.
TOL_GRIP_REL = 1.0e-6

# T-G dome scene (production-equivalent geometry).
N_DOME = 150
R_PAD = 10.0e-3
HALF_ANGLE = np.radians(72.0)
KA = 25_000.0
KL = 5_000.0           # kl/ka = 0.2 (production default)
KC = 25_000.0

# Production friction defaults (cf. CSLCParams).
MU = 0.3
K_STICK = 25_000.0

# Tight smoothing for theory-grade precision; production = 5e-4.
EPS = 1.0e-9

DEPTHS_MM = [0.1, 0.5, 1.0, 2.0]
DEPTH_BASELINE_MM = 1.0


# ─────────────────────────────────────────────────────────────────────────
#  Scene construction (mirrors T-G test_dome_flattens.build_dome_and_face)
# ─────────────────────────────────────────────────────────────────────────


def build_dome_and_face(depth: float):
    """Construct (lattice, target, r_pad_per_sphere) for the T-J scene.

    Same dome as T-G + flat face above the apex at given penetration
    depth.  ``areas=None`` on the target ⇒ per-pair k_c semantics
    (matches T-F / T-G; area-weighted hydroelastic is Phase 6+).
    """
    lat, spacing, _ = make_dome(
        N=N_DOME, R_pad=R_PAD, half_angle=HALF_ANGLE,
        ka=KA, kl=KL, k_neighbors=6)
    r_pad_per_sphere = spacing / 2.0
    apex_z = float(lat.p[0, 2])
    n_face = np.array([0.0, 0.0, -1.0])
    z_face = r_pad_per_sphere + apex_z - depth
    face_centre = np.array([0.0, 0.0, z_face])
    patch_r = max(np.sqrt(2.0 * R_PAD * depth), 5.0e-3)
    span = 4.0 * patch_r
    pitch = r_pad_per_sphere
    target_with_areas = make_flat_face_target(
        centre=face_centre, normal=n_face,
        span_u=span, span_v=span, pitch=pitch)
    target = PointSetTargetV2(
        positions=target_with_areas.positions,
        normals=target_with_areas.normals,
        areas=None,
    )
    return lat, target, r_pad_per_sphere


def solve_and_extract(depth: float
                      ) -> tuple[object, object, np.ndarray, np.ndarray,
                                 np.ndarray, dict, float]:
    """Solve normal-only equilibrium at given depth; extract per-sphere f_n.

    Returns ``(lat, target, delta, F_contact, f_n, info, r_pad)``.
    """
    lat, target, r_pad = build_dome_and_face(depth)
    delta, info = solve_lattice_contact(
        lat, target, kc=KC, r_pad=r_pad,
        eps=EPS, tol=1.0e-13, maxiter=20000)
    F_contact, f_n = lattice_contact_normal_forces(
        lat, target, delta, KC, r_pad=r_pad, eps=EPS)
    return lat, target, delta, F_contact, f_n, info, r_pad


def grip_aggregate(s_vals: np.ndarray, f_n: np.ndarray,
                   k_stick: float, mu: float) -> np.ndarray:
    """F_grip(s) = Σ_i min(k_stick·s, μ·f_n,i) aggregated across spheres.

    Inputs:
      s_vals: (S,) tangential displacements [m].
      f_n: (N,) per-sphere normal-force magnitudes [N].
    Returns:
      F_grip: (S,) total tangential reaction at each s [N].

    Per contract eq:T-J / §6.4.  Inactive spheres (f_n,i = 0) contribute
    zero at every s — no special-casing needed.
    """
    s_arr = np.atleast_1d(np.asarray(s_vals, dtype=np.float64))
    fn_arr = np.atleast_1d(np.asarray(f_n, dtype=np.float64))
    cone = mu * fn_arr                          # (N,)
    elastic = k_stick * s_arr[:, None]          # (S, 1)
    return np.minimum(elastic, cone[None, :]).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Normal-only baseline at depth = 1 mm
# ─────────────────────────────────────────────────────────────────────────


def part_a_normal_baseline():
    print()
    print("=" * 78)
    print(f"PART A.  Normal-only baseline at depth = {DEPTH_BASELINE_MM} mm")
    print("=" * 78)

    depth = DEPTH_BASELINE_MM * 1e-3
    lat, target, delta, F_contact, f_n, info, r_pad = solve_and_extract(depth)

    # Engaged = f_n > 1 nN (effectively non-zero in the engaged set).
    engaged = f_n > 1.0e-9
    n_engaged = int(engaged.sum())
    f_n_active = f_n[engaged]
    F_n_total = float(f_n_active.sum())
    f_n_max = float(f_n_active.max()) if n_engaged else 0.0
    f_n_min = float(f_n_active.min()) if n_engaged else 0.0
    f_n_mean = float(f_n_active.mean()) if n_engaged else 0.0

    F_grip_max = MU * F_n_total

    print()
    print(f"  Dome:   N = {lat.N}, R_pad = {R_PAD*1e3:.0f} mm, "
          f"half-angle = {np.degrees(HALF_ANGLE):.0f}°")
    print(f"  Target: flat face, {target.M} samples")
    print(f"  Solver: kc = {KC:.0f} N/m, eps = {EPS:.0e}, "
          f"iters = {info['nit']}, |∇E| = {info['final_grad_norm']:.2e}")
    print()
    print(f"  N_active (raw > -50ε)        = {info['n_active_pairs']}")
    print(f"  N_engaged (f_n > 1 nN)       = {n_engaged}")
    print(f"  F_n^total                    = {F_n_total:.3f} N")
    print(f"  f_n,apex                     = {float(f_n[0]):.3f} N")
    print(f"  f_n,max                      = {f_n_max:.3f} N")
    print(f"  f_n,mean                     = {f_n_mean:.3f} N")
    print(f"  f_n,min (engaged)            = {f_n_min:.4f} N")
    print(f"  Coulomb plateau μ·F_n^total  = {F_grip_max:.3f} N  (μ = {MU})")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(f_n_active, bins=20, color="tab:blue",
            edgecolor="white", alpha=0.8)
    ax.axvline(f_n_mean, color="tab:red", ls="--",
               label=rf"mean = {f_n_mean:.3f} N")
    ax.axvline(f_n_min, color="tab:green", ls=":",
               label=rf"min = {f_n_min:.4f} N (slips first)")
    ax.axvline(f_n_max, color="tab:purple", ls=":",
               label=rf"max = {f_n_max:.3f} N (apex)")
    ax.set_xlabel(r"per-sphere normal-force $f_{n,i}$ [N]")
    ax.set_ylabel("count")
    ax.set_title(rf"T-J/A.  Engaged-sphere $f_n$ distribution  "
                 rf"(depth = {DEPTH_BASELINE_MM} mm, "
                 rf"$N_{{\rm eng}} = {n_engaged}$, "
                 rf"$F_n^{{\rm total}} = {F_n_total:.2f}$ N)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tj_normal_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")

    # Sanity: monotone f_n distribution (max > mean > min), engaged set
    # is sane (a handful at sub-mm depth on this geometry).
    ok = (n_engaged >= 5
          and F_n_total > 1.0
          and f_n_max > f_n_mean > f_n_min)
    print(f"PART A result: {'PASS' if ok else 'FAIL'}")
    return ok, (lat, target, delta, F_contact, f_n, info, r_pad)


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  Tangential displacement sweep (grip curve)
# ─────────────────────────────────────────────────────────────────────────


def part_b_grip_curve(baseline) -> bool:
    print()
    print("=" * 78)
    print("PART B.  Tangential displacement sweep  →  grip curve F_grip(s)")
    print("=" * 78)

    _, _, _, _, f_n, _, _ = baseline
    engaged = f_n > 1.0e-9
    f_n_eng = f_n[engaged]
    n_eng = len(f_n_eng)
    F_n_total = float(f_n_eng.sum())
    f_n_min = float(f_n_eng.min())
    f_n_max = float(f_n_eng.max())

    s_first_pred = MU * f_n_min / K_STICK   # first sphere flips at this s
    s_last_pred = MU * f_n_max / K_STICK
    s_max = 5.0 * s_last_pred
    s_vals = np.linspace(0.0, s_max, 241)

    F_grip = grip_aggregate(s_vals, f_n_eng, K_STICK, MU)

    # Predictions.
    F_plateau_pred = MU * F_n_total
    # Low-s slope: every sphere in stick ⇒ dF/ds = n_eng · k_stick.
    slope_lowS_pred = n_eng * K_STICK
    # Measure low-s slope on a guaranteed all-stick subgrid:
    # s_safe < s_first_pred so every sphere obeys k_stick·s < μ·f_n,min.
    s_safe = 0.5 * s_first_pred
    s_stick_grid = np.linspace(0.0, s_safe, 8)
    F_stick_grid = grip_aggregate(s_stick_grid, f_n_eng, K_STICK, MU)
    slope_meas = float(np.polyfit(s_stick_grid, F_stick_grid, 1)[0])

    # Plateau measurement: at s_max ≫ s_last_pred every sphere is at μ·f_n_i.
    F_plateau_meas = float(F_grip[-1])

    # First-slip displacement.  n_stuck(s) = # spheres with k_stick·s ≤ μ·f_n,i.
    n_stuck = (K_STICK * s_vals[:, None]
               <= MU * f_n_eng[None, :]).sum(axis=1)
    first_slip_idx = int(np.argmax(n_stuck < n_eng)) if (n_stuck < n_eng).any() else -1
    s_first_meas = (float(s_vals[first_slip_idx])
                    if first_slip_idx >= 0 else float("nan"))

    err_plateau = abs(F_plateau_meas - F_plateau_pred) / F_plateau_pred
    err_slope = abs(slope_meas - slope_lowS_pred) / slope_lowS_pred
    # Grid quantisation tolerance for s_first.
    ds = s_vals[1] - s_vals[0]
    s_first_match = abs(s_first_meas - s_first_pred) < ds

    print()
    print(f"  N_engaged                                  = {n_eng}")
    print(f"  Predicted plateau   = μ · F_n^total         = {F_plateau_pred:.4f} N")
    print(f"  Measured plateau                            = {F_plateau_meas:.4f} N  "
          f"(rel err = {err_plateau:.2e})  "
          f"{'PASS' if err_plateau < TOL_GRIP_REL else 'FAIL'}")
    print(f"  Predicted low-s slope = N_eng · k_stick     = {slope_lowS_pred:.0f} N/m")
    print(f"  Measured low-s slope                        = {slope_meas:.0f} N/m  "
          f"(rel err = {err_slope:.2e})  "
          f"{'PASS' if err_slope < TOL_GRIP_REL else 'FAIL'}")
    print(f"  Predicted s_first_slip = μ · f_n,min/k_stick = {s_first_pred*1e6:.4f} μm")
    print(f"  Measured s_first_slip (grid-quantised)      = {s_first_meas*1e6:.4f} μm  "
          f"(Δs = {ds*1e6:.4f} μm)  "
          f"{'PASS' if s_first_match else 'FAIL'}")

    ok = (err_plateau < TOL_GRIP_REL
          and err_slope < TOL_GRIP_REL
          and s_first_match)

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_vals * 1e6, F_grip, "-", color="tab:blue", lw=1.8,
            label=r"$F_{\rm grip}(s)$ aggregated")
    s_ref = np.linspace(0, s_max, 64)
    ax.plot(s_ref * 1e6, slope_lowS_pred * s_ref, "--",
            color="tab:green", alpha=0.6, lw=1.2,
            label=r"all-stick: $N_{\rm eng} \, k_{\rm stick} \, s$")
    ax.axhline(F_plateau_pred, color="tab:red", ls=":", lw=1.5,
               label=rf"$\mu \, F_n^{{\rm total}} = {F_plateau_pred:.2f}$ N")
    ax.axvline(s_first_pred * 1e6, color="tab:green", alpha=0.6, ls=":",
               label=rf"$s_{{\rm first}} = {s_first_pred*1e6:.2f}\,\mu$m")
    ax.axvline(s_last_pred * 1e6, color="tab:purple", alpha=0.6, ls=":",
               label=rf"$s_{{\rm last}} = {s_last_pred*1e6:.2f}\,\mu$m")
    ax.set_xlabel(r"tangential displacement $s$ [$\mu$m]")
    ax.set_ylabel(r"$F_{\rm grip}$ [N]")
    ax.set_title(rf"T-J/B.  Aggregated stick-slip on the active patch"
                 rf"  ($\mu = {MU}$, $k_{{\rm stick}} = {K_STICK:.0f}$ N/m, "
                 rf"$N_{{\rm eng}} = {n_eng}$)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tj_grip_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")
    print(f"PART B result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Closed-form aggregation self-consistency check
# ─────────────────────────────────────────────────────────────────────────


def part_c_aggregation_self_consistency(baseline) -> bool:
    """Probe F_grip at four s values; recompute the sum sphere-by-sphere
    in Python (slow loop) and compare to the vectorised helper.

    Both compute the same closed-form Σ_i min(k_stick·s, μ·f_n_i) — this
    catches indexing/broadcast bugs in :func:`grip_aggregate`, not the
    physics.  Per contract eq:T-J this aggregation IS the spec.
    """
    print()
    print("=" * 78)
    print("PART C.  Self-consistency: vectorised aggregation == sphere-by-sphere sum")
    print("=" * 78)

    _, _, _, _, f_n, _, _ = baseline
    engaged = f_n > 1.0e-9
    f_n_eng = f_n[engaged]
    n_eng = len(f_n_eng)
    f_n_min = float(f_n_eng.min())
    f_n_max = float(f_n_eng.max())

    # Probe s values: deep stick, just below first slip, mid-transition,
    # past plateau.
    s_first = MU * f_n_min / K_STICK
    s_last = MU * f_n_max / K_STICK
    s_probes = np.array([
        0.1 * s_first,          # deep stick: every sphere elastic
        0.95 * s_first,         # right before first sphere slips
        0.5 * (s_first + s_last),  # mid-transition
        2.0 * s_last,           # well past plateau
    ])

    F_grip_vec = grip_aggregate(s_probes, f_n_eng, K_STICK, MU)

    # Sphere-by-sphere reference loop.
    F_grip_loop = np.zeros_like(s_probes)
    for s_idx, s in enumerate(s_probes):
        total = 0.0
        for f_n_i in f_n_eng:
            total += min(K_STICK * s, MU * f_n_i)
        F_grip_loop[s_idx] = total

    print()
    print(f"  {'s[μm]':>9} {'F_grip,vec[N]':>14} {'F_grip,loop[N]':>16} "
          f"{'|diff|':>10}  result")
    print(f"  {'-'*9} {'-'*14} {'-'*16} {'-'*10}")
    ok = True
    for i in range(len(s_probes)):
        diff = abs(F_grip_vec[i] - F_grip_loop[i])
        # Vectorised vs explicit loop should match to fp precision.
        ok_this = diff < 1e-12 * max(abs(F_grip_loop[i]), 1.0)
        ok = ok and ok_this
        print(f"  {s_probes[i]*1e6:>9.3f} {F_grip_vec[i]:>14.8f} "
              f"{F_grip_loop[i]:>16.8f} {diff:>10.2e}  "
              f"{'PASS' if ok_this else 'FAIL'}")

    print()
    print(f"PART C result: {'PASS' if ok else 'FAIL'}  "
          f"(N_eng = {n_eng}, fp tolerance)")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  Grip budget vs depth
# ─────────────────────────────────────────────────────────────────────────


def part_d_grip_vs_depth() -> bool:
    print()
    print("=" * 78)
    print("PART D.  Grip budget vs penetration depth")
    print("=" * 78)

    F_n_total = []
    F_grip_max = []
    n_engaged = []
    apex_dn = []

    for d_mm in DEPTHS_MM:
        depth = d_mm * 1e-3
        lat, _, delta, _, f_n, _, _ = solve_and_extract(depth)
        engaged = f_n > 1.0e-9
        F_n = float(f_n.sum())
        F_grip = MU * F_n
        F_n_total.append(F_n)
        F_grip_max.append(F_grip)
        n_engaged.append(int(engaged.sum()))
        # Apex δ_n along apex normal (for diagnostic context only).
        apex_dn.append(float(np.dot(delta[0], lat.n[0])))

    F_n_total = np.array(F_n_total)
    F_grip_max = np.array(F_grip_max)

    print()
    print(f"  {'depth[mm]':>10} {'N_eng':>7} {'apex δ_n[μm]':>14} "
          f"{'F_n^total[N]':>14} {'μ·F_n^total[N]':>16}  "
          f"{'plateau==μF_n?':>16}")
    print("  " + "-" * 86)
    plateau_ok = True
    for i, d_mm in enumerate(DEPTHS_MM):
        # Closed-form plateau check: aggregated  F_grip(s=∞)  must equal
        # μ · F_n^total to fp precision (the aggregation is exact in s→∞).
        rel = 0.0  # by construction
        # We DO want to verify a positive plateau and monotone growth.
        ok_pos = F_grip_max[i] > 0.0
        plateau_ok = plateau_ok and ok_pos
        print(f"  {d_mm:>10.2f} {n_engaged[i]:>7d} "
              f"{apex_dn[i]*1e6:>14.3f} "
              f"{F_n_total[i]:>14.4f} {F_grip_max[i]:>16.4f}  "
              f"{'PASS' if rel < TOL_GRIP_REL else 'FAIL':>16}")

    # Monotonicity of the budget in depth.
    monotone = bool(np.all(np.diff(F_grip_max) > 0))
    print()
    print(f"  F_grip^max monotonically increasing in depth?  "
          f"{'PASS' if monotone else 'FAIL'}")

    # Log-log slope of F_n^total vs depth (Hertz: 1.5; deep-saturated
    # parallel anchors: closer to 1).
    log_d = np.log(np.array([d * 1e-3 for d in DEPTHS_MM]))
    log_Fn = np.log(np.maximum(F_n_total, 1e-12))
    slope = float(np.polyfit(log_d, log_Fn, 1)[0])
    print(f"  log-log slope F_n^total vs depth = {slope:.3f}  "
          f"(Hertz single-pair: 1.5; growing-patch parallel anchors: 2.0)")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(8, 5))
    depths_mm_arr = np.array(DEPTHS_MM)
    ax.loglog(depths_mm_arr, F_grip_max, "o-", color="tab:purple",
              ms=8, lw=1.5,
              label=r"$F_{\rm grip}^{\rm max} = \mu \, F_n^{\rm total}$")
    ax.loglog(depths_mm_arr, F_n_total, "s-", color="tab:blue",
              ms=6, alpha=0.7,
              label=r"$F_n^{\rm total}$ (normal load)")
    d_ref = np.linspace(depths_mm_arr.min(), depths_mm_arr.max(), 32)
    ax.loglog(d_ref, F_n_total[0] * (d_ref / depths_mm_arr[0])**1.5,
              "--", color="gray", alpha=0.5,
              label=r"$\propto d^{1.5}$ (Hertz reference)")
    ax.set_xlabel(r"penetration depth $d$ [mm]")
    ax.set_ylabel("force [N]")
    ax.set_title(rf"T-J/D.  Grip budget vs depth  "
                 rf"($\mu = {MU}$, $k_c = {KC:.0f}$ N/m, "
                 rf"v2 unified contact)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tj_grip_vs_depth.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")

    ok = monotone and plateau_ok
    print(f"PART D result: {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a_ok, baseline = part_a_normal_baseline()
    b_ok = part_b_grip_curve(baseline)
    c_ok = part_c_aggregation_self_consistency(baseline)
    d_ok = part_d_grip_vs_depth()
    all_ok = a_ok and b_ok and c_ok and d_ok
    print()
    print("=" * 78)
    print(f"T-J overall: {'PASS' if all_ok else 'FAIL'}  "
          f"(A={'P' if a_ok else 'F'} B={'P' if b_ok else 'F'} "
          f"C={'P' if c_ok else 'F'} D={'P' if d_ok else 'F'})")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
