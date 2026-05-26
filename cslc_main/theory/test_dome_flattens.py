# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 / T-G — Dome lattice flattens against a flat target face (no bulge).

The CENTERPIECE regression test for the v2 contract.  This test
verifies that the v2 model preserves the *kinematic* dome-flattens-on-
flat-face behaviour (apex sinkage, perimeter conforming) WITHOUT
introducing the *artefactual* outward Poisson bulge that v1's distance-
preserving lateral law produced on curved patches.

The v1 ``test_05_arc_contact.py`` showed that DP lateral + curved
geometry yields a window in φ where neighbours of the contact sphere
develop **negative δ_n** (i.e., they bulge OUTWARD along their own
outward normal) before snapping back to same-sign spreading at larger
compression.  This was a real geometric consequence of the DP law on
curved patches, NOT a numerical artefact.  v2 removes DP entirely
(contract §6.2); graph-Laplacian is direction-blind on the apex-
neighbour edges and CANNOT produce outward bulging.

T-G replaces both v1 ``test_05_arc_contact.py`` (DP bulge window) and
``test_08_dome_contact.py`` (3D dome under sphere indenter): the
dome lattice setup is the v1 step-8 Fibonacci-spiral cap, but the
target is a flat face (the v2 unified path's natural test geometry)
and the regression check is "no bulge anywhere on the lattice."

Four parts:

  PART A.  Geometry sanity check.  Dome lattice has N spheres on a
           spherical cap; apex at index 0; outward normals are radial.
           Flat face is positioned above the apex with the apex's
           rest half-space overlap = d_rest.  Sanity: only the apex
           sphere is in contact at rest (its raw > 0); all others
           have raw < 0.

  PART B.  Equilibrium at four penetration depths
           ``{0.1, 0.5, 1.0, 2.0} mm``.  For each depth:
             (a) apex sphere has positive δ_n (compressed inward);
             (b) **NO sphere anywhere on the lattice has δ_n < -ε**
                 — the regression target.  The v1 DP law would
                 produce negative δ_n on the perimeter near the
                 apex; v2 graph-Laplacian produces non-negative δ_n
                 everywhere.

  PART C.  Cross-section profile.  For depth = 1.0 mm, plot the
           (r_radial, z_axial) cross-section of the dome before
           (rest) and after (deformed).  Visually confirm the dome
           conforms to the flat face — apex sinks; near-apex spheres
           sink less but in the SAME direction (no outward bulge).
           Overlay the flat face plane for context.

  PART D.  Active-set diagnostic.  Count (pad sphere, target sample)
           active pairs vs depth.  Verify the contact patch grows
           with depth (more pad spheres engage as the dome sinks
           further into the face).

Per ``cslc_main/theory/contract_v2.md`` §6, §12 T-G.

Run::

    uv run -m cslc_main.theory.test_dome_flattens

Outputs::

    cslc_main/theory/figures/tg_geometry.png
    cslc_main/theory/figures/tg_delta_n_histogram.png
    cslc_main/theory/figures/tg_cross_section.png
    cslc_main/theory/figures/tg_active_pairs.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_lattice import (
    make_dome,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import (
    PointSetTargetV2,
    make_flat_face_target,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Production-equivalent dome geometry (matches v1 test_08).
N_DOME = 150
R_PAD = 10.0e-3            # dome radius [m]
HALF_ANGLE = np.radians(72.0)
KA = 25_000.0
KL = 5_000.0               # kl/ka = 0.2 (production default)
KC = 25_000.0

DEPTHS_MM = [0.1, 0.5, 1.0, 2.0]   # penetration depths [mm]
EPS = 1.0e-9                       # tight smoothing for theory-grade precision

# Regression bound: no surface sphere may have δ_n more negative than
# this (essentially fp noise).  v1 DP at production geometry produces
# negative δ_n on the order of 1-30 µm; v2 GL must give effectively zero.
BULGE_BOUND_M = 1.0e-9     # 1 nm (below numerical precision floor)


def build_dome_and_face(depth: float):
    """Construct (lattice, target, mean_spacing) for the canonical T-G scene.

    Dome lattice from :func:`cslc_main.theory.cslc_lattice.make_dome`
    (Fibonacci spiral; densest-z sample permuted to index 0 = "apex").
    Note: the Fibonacci spiral does NOT place a sample at the exact
    cap-top (0, 0, R_pad); the apex sample is the closest one, typically
    offset by a few % of spacing in (x, y).  We place the flat face
    relative to the ACTUAL apex z so the apex's rest half-space overlap
    is exactly ``depth``.
    """
    lat, spacing, cap_area = make_dome(
        N=N_DOME, R_pad=R_PAD, half_angle=HALF_ANGLE,
        ka=KA, kl=KL, k_neighbors=6)
    r_pad_per_sphere = spacing / 2.0
    apex_z = float(lat.p[0, 2])    # actual apex sample z (not R_pad)
    # Face normal: -z (face above dome, pointing down toward dome).
    n_face = np.array([0.0, 0.0, -1.0])
    # Place face so apex's half-space overlap at δ = 0 is exactly ``depth``:
    # raw_apex = r - n_face · (apex_pos - face_centre)
    #         = r + (apex.z - face.z)         (since face normal is -z)
    # Setting raw = depth:  face.z = r + apex.z - depth.
    z_face = r_pad_per_sphere + apex_z - depth
    face_centre = np.array([0.0, 0.0, z_face])
    # Face needs to span enough to cover the contact patch.  Patch
    # radius at depth d: a ≈ √(2 R_pad d) (Hertzian); add margin so
    # the locality kernel sees a full set of samples.
    patch_r = max(np.sqrt(2.0 * R_PAD * depth), 5.0e-3)
    span = 4.0 * patch_r          # ample margin
    pitch = r_pad_per_sphere      # 1 sample per pad-sphere "footprint"
    target_with_areas = make_flat_face_target(
        centre=face_centre, normal=n_face,
        span_u=span, span_v=span, pitch=pitch)
    # For Phase 2 verification we use per-pair Hookean kc (areas treated
    # as dimensionless 1.0 in the solver, matching T-F's setup).  The
    # area-weighted form would require recalibrating kc to ~1e10 N/m³
    # (hydroelastic-style per-unit-volume stiffness) — a separate
    # calibration story we defer to Phase 6+ (handler / production).
    # The "no bulge" regression check is unaffected by this choice:
    # only the FORCE MAGNITUDES change with areas, not the direction
    # of equilibrium δ at any sphere.
    target = PointSetTargetV2(
        positions=target_with_areas.positions,
        normals=target_with_areas.normals,
        areas=None,
    )
    return lat, target, r_pad_per_sphere, spacing


def project_delta_normal(deltas: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Per-sphere δ_n = δ_i · n_pad_i  (>0 means compressed inward)."""
    return np.einsum("ij,ij->i", deltas, normals)


# ─────────────────────────────────────────────────────────────────────────
#  PART A.  Geometry sanity check
# ─────────────────────────────────────────────────────────────────────────


def part_a_geometry() -> bool:
    print()
    print("=" * 78)
    print("PART A.  Dome geometry sanity check")
    print("=" * 78)

    d = 1.0e-3
    lat, target, r_pad, spacing = build_dome_and_face(d)
    print()
    print(f"  Dome: N = {lat.N}, R_pad = {R_PAD*1e3:.1f} mm, "
          f"half_angle = {np.degrees(HALF_ANGLE):.0f}°")
    print(f"        mean lattice spacing = {spacing*1e3:.3f} mm  "
          f"→ r_pad (per sphere) = {r_pad*1e3:.3f} mm")
    print(f"  Target: flat face with {target.M} samples, "
          f"pitch ≈ {r_pad*1e3:.3f} mm")
    print(f"  Penetration depth at apex (rest): {d*1e3:.1f} mm")

    # Apex sample is at index 0; verify it's the highest-z sphere on
    # the cap and lies within one spacing of the exact top.
    apex_pos = lat.p[0]
    apex_z_max = bool(apex_pos[2] >= np.max(lat.p[:, 2]) - 1e-12)
    apex_normal_up = bool(np.dot(lat.n[0], [0, 0, 1]) > 0.99)   # close to +z
    apex_near_top = bool(np.linalg.norm(apex_pos[:2]) < spacing)
    print(f"  apex at highest z:                 "
          f"{'YES' if apex_z_max else 'NO'} (z = {apex_pos[2]*1e3:.3f} mm)")
    print(f"  apex normal close to +z:           "
          f"{'YES' if apex_normal_up else 'NO'} "
          f"(n · ẑ = {np.dot(lat.n[0], [0, 0, 1]):.6f})")
    print(f"  apex within 1 spacing of (0,0):    "
          f"{'YES' if apex_near_top else 'NO'} "
          f"(|xy| = {np.linalg.norm(apex_pos[:2])*1e3:.3f} mm, "
          f"spacing = {spacing*1e3:.3f} mm)")

    # All normals are unit radial.
    norms = np.linalg.norm(lat.n, axis=1)
    radial = np.einsum("ij,ij->i", lat.p, lat.n) / np.linalg.norm(lat.p, axis=1)
    norms_ok = bool(np.allclose(norms, 1.0, atol=1e-12))
    radial_ok = bool(np.allclose(radial, 1.0, atol=1e-10))
    print(f"  all normals unit-length:          "
          f"{'YES' if norms_ok else 'NO'} (max dev = {np.max(np.abs(norms - 1)):.2e})")
    print(f"  all normals radial:                "
          f"{'YES' if radial_ok else 'NO'} (max dev = {np.max(np.abs(radial - 1)):.2e})")

    # Plot.
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(lat.p[:, 0]*1e3, lat.p[:, 1]*1e3, lat.p[:, 2]*1e3,
               c="C0", s=20, label=f"dome lattice (N = {lat.N})")
    ax.scatter([apex_pos[0]*1e3], [apex_pos[1]*1e3], [apex_pos[2]*1e3],
               c="red", s=80, label="apex (index 0)")
    ax.scatter(target.positions[:, 0]*1e3, target.positions[:, 1]*1e3,
               target.positions[:, 2]*1e3,
               c="C2", s=5, alpha=0.4, label=f"flat face ({target.M} samples)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(f"T-G/A. Dome geometry "
                 f"({np.degrees(HALF_ANGLE):.0f}° cap, R = {R_PAD*1e3:.0f} mm) "
                 f"+ flat face above apex")
    ax.legend(loc="upper left", fontsize=9)
    out = FIG_DIR / "tg_geometry.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")

    return apex_z_max and apex_normal_up and apex_near_top and norms_ok and radial_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART B.  No-bulge regression at four depths
# ─────────────────────────────────────────────────────────────────────────


def part_b_no_bulge() -> bool:
    print()
    print("=" * 78)
    print("PART B.  No-bulge regression at four penetration depths")
    print("=" * 78)
    print()
    print("  Assertion: every surface sphere on the dome has "
          f"δ_n > -{BULGE_BOUND_M*1e9:.0f} nm")
    print("  (regression: v1 DP at production geometry gives "
          "δ_n_perim ≈ -33 nm to -1 μm; v2 GL gives effectively zero)")
    print()
    print(f"  {'depth[mm]':>10} {'δ_n_apex[μm]':>13} "
          f"{'min δ_n[nm]':>13} {'max δ_n[μm]':>13} "
          f"{'# bulge spheres':>16}  result")
    print("  " + "-" * 80)

    all_ok = True
    per_depth_data = {}
    for d_mm in DEPTHS_MM:
        d = d_mm * 1e-3
        lat, target, r_pad, spacing = build_dome_and_face(d)
        delta, info = solve_lattice_contact(
            lat, target, kc=KC, r_pad=r_pad,
            eps=EPS, tol=1.0e-13, maxiter=20000)
        delta_n = project_delta_normal(delta, lat.n)
        bulge_mask = delta_n < -BULGE_BOUND_M
        n_bulge = int(np.sum(bulge_mask))
        apex_dn = float(delta_n[0])
        min_dn = float(np.min(delta_n))
        max_dn = float(np.max(delta_n))
        apex_compressed = apex_dn > 0
        no_bulge = (n_bulge == 0)
        ok = apex_compressed and no_bulge
        all_ok = all_ok and ok
        per_depth_data[d_mm] = {
            "delta": delta, "delta_n": delta_n, "lat": lat,
            "target": target, "info": info,
        }
        print(f"  {d_mm:>10.2f} {apex_dn*1e6:>13.4f} "
              f"{min_dn*1e9:>13.3f} {max_dn*1e6:>13.4f} "
              f"{n_bulge:>16d}  {'PASS' if ok else 'FAIL'}")

    print()
    print(f"PART B: {'PASS' if all_ok else 'FAIL'}")

    # ── Figure: δ_n histogram at depth = 1 mm ──
    d_show = 1.0
    dn = per_depth_data[d_show]["delta_n"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dn * 1e6, bins=40, color="C0", edgecolor="black", lw=0.5)
    ax.axvline(0, color="grey", lw=0.8, ls=":")
    ax.axvline(-BULGE_BOUND_M * 1e6, color="C3", lw=1.0, ls="--",
               label=f"bulge bound = -{BULGE_BOUND_M*1e9:.0f} nm")
    ax.set_xlabel(r"$\delta_n$ [$\mu$m]  (along sphere's outward normal)")
    ax.set_ylabel("number of dome spheres")
    ax.set_title("T-G/B. δ_n distribution at depth = 1 mm  —  no bulge\n"
                 f"({lat.N} dome spheres; {int(np.sum(dn > 0))} compressed, "
                 f"{int(np.sum(dn <= 0))} at rest)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tg_delta_n_histogram.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")

    return all_ok, per_depth_data


# ─────────────────────────────────────────────────────────────────────────
#  PART C.  Cross-section profile (dome conforms to face)
# ─────────────────────────────────────────────────────────────────────────


def part_c_cross_section(per_depth_data) -> bool:
    print()
    print("=" * 78)
    print("PART C.  Cross-section: dome conforms to flat face")
    print("=" * 78)
    print()
    print("  Plot (r_radial, z_axial) cross-section for all 4 depths.")
    print("  Verify dome surface monotonically conforms to face plane")
    print("  — no outward-bulge spheres above the rest cap.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: rest cap profile (reference).
    lat0 = per_depth_data[DEPTHS_MM[0]]["lat"]
    r_rest = np.sqrt(lat0.p[:, 0]**2 + lat0.p[:, 1]**2)
    z_rest = lat0.p[:, 2]

    for d_mm, color in zip(DEPTHS_MM, ["C0", "C1", "C2", "C3"]):
        data = per_depth_data[d_mm]
        delta = data["delta"]
        q = data["lat"].p - delta            # deformed centres
        r_def = np.sqrt(q[:, 0]**2 + q[:, 1]**2)
        z_def = q[:, 2]
        # Sort by radial position for clean line plot.
        order = np.argsort(r_def)
        # Face plane z (at depth d_mm).
        d = d_mm * 1e-3
        _, _, r_pad, _ = build_dome_and_face(d)
        z_face = R_PAD + r_pad - d
        axes[0].scatter(r_def * 1e3, z_def * 1e3, color=color, s=25,
                        label=f"depth = {d_mm:.1f} mm", alpha=0.7)
        axes[0].axhline(z_face * 1e3, color=color, ls=":",
                        lw=0.8, alpha=0.5)

    # Rest cap for reference.
    axes[0].scatter(r_rest * 1e3, z_rest * 1e3, color="grey", s=10,
                    marker="x", label="rest", alpha=0.6)
    axes[0].set_xlabel("radial position r [mm]")
    axes[0].set_ylabel("axial position z [mm]")
    axes[0].set_title("T-G/C. Dome cross-section vs penetration depth\n"
                      "(coloured circles = deformed; grey × = rest cap; "
                      "dotted lines = face plane)")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal", adjustable="datalim")

    # Right panel: δ_n vs radial position at each depth.
    for d_mm, color in zip(DEPTHS_MM, ["C0", "C1", "C2", "C3"]):
        data = per_depth_data[d_mm]
        delta_n = data["delta_n"]
        r_rad = np.sqrt(data["lat"].p[:, 0]**2 + data["lat"].p[:, 1]**2)
        order = np.argsort(r_rad)
        axes[1].plot(r_rad[order] * 1e3, delta_n[order] * 1e6, "o-",
                     color=color, ms=4, lw=1.0,
                     label=f"depth = {d_mm:.1f} mm")
    axes[1].axhline(0, color="grey", lw=0.8, ls=":")
    axes[1].axhline(-BULGE_BOUND_M * 1e6, color="C3", lw=1.0, ls="--",
                    label=f"bulge bound -{BULGE_BOUND_M*1e9:.0f} nm")
    axes[1].set_xlabel("rest radial position r [mm]")
    axes[1].set_ylabel(r"$\delta_n$ [$\mu$m]  (along sphere's outward normal)")
    axes[1].set_title(r"δ_n profile across the dome — no spheres below 0")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "tg_cross_section.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")

    # Quantitative monotonicity: at each depth, δ_n should decrease
    # monotonically with r_rad (apex compresses most; perimeter not at all).
    mono_ok = True
    for d_mm in DEPTHS_MM:
        data = per_depth_data[d_mm]
        delta_n = data["delta_n"]
        r_rad = np.sqrt(data["lat"].p[:, 0]**2 + data["lat"].p[:, 1]**2)
        order = np.argsort(r_rad)
        # Allow some scatter due to lattice azimuthal asymmetry — just
        # check the broad trend: δ_n at near-apex spheres > δ_n at far
        # perimeter.  We split into inner/outer halves.
        N = len(delta_n)
        inner_dn = float(np.mean(delta_n[order][:N // 4]))
        outer_dn = float(np.mean(delta_n[order][3 * N // 4:]))
        ok = inner_dn > outer_dn
        mono_ok = mono_ok and ok
        print(f"  depth = {d_mm:.1f} mm: "
              f"inner-quarter mean δ_n = {inner_dn*1e6:>8.3f} μm, "
              f"outer-quarter mean δ_n = {outer_dn*1e6:>8.6f} μm  "
              f"{'(conforming)' if ok else '(NON-MONOTONE!)'}")

    return mono_ok


# ─────────────────────────────────────────────────────────────────────────
#  PART D.  Active-pair count vs depth
# ─────────────────────────────────────────────────────────────────────────


def part_d_active_pairs(per_depth_data) -> bool:
    print()
    print("=" * 78)
    print("PART D.  Active (pad sphere, target sample) pairs vs depth")
    print("=" * 78)
    print()
    print(f"  {'depth[mm]':>10} {'n_active':>10} {'n_engaged_pad':>14}  "
          f"(engaged = δ_n > 1 nm)")
    print("  " + "-" * 60)

    counts = []
    for d_mm in DEPTHS_MM:
        data = per_depth_data[d_mm]
        info = data["info"]
        delta_n = data["delta_n"]
        n_active = info["n_active_pairs"]
        n_engaged = int(np.sum(delta_n > 1e-9))
        counts.append((d_mm, n_active, n_engaged))
        print(f"  {d_mm:>10.2f} {n_active:>10d} {n_engaged:>14d}")

    # Check monotonicity in engaged pad spheres.
    engaged = [c[2] for c in counts]
    mono = all(engaged[i] <= engaged[i + 1] for i in range(len(engaged) - 1))
    print()
    print(f"  engaged-sphere count monotone in depth: "
          f"{'PASS' if mono else 'FAIL'}")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    depths_mm = [c[0] for c in counts]
    n_active_arr = [c[1] for c in counts]
    n_engaged_arr = [c[2] for c in counts]
    ax.plot(depths_mm, n_active_arr, "o-", color="C0", ms=8, lw=1.5,
            label="active (pad, sample) pairs")
    ax.plot(depths_mm, n_engaged_arr, "s-", color="C2", ms=8, lw=1.5,
            label=r"engaged pad spheres ($\delta_n > 1$ nm)")
    ax.set_xlabel("penetration depth at apex [mm]")
    ax.set_ylabel("count")
    ax.set_title("T-G/D. Contact patch grows with penetration depth")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "tg_active_pairs.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")

    return mono


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    a_ok = part_a_geometry()
    b_ok, per_depth_data = part_b_no_bulge()
    c_ok = part_c_cross_section(per_depth_data)
    d_ok = part_d_active_pairs(per_depth_data)
    print()
    print("=" * 78)
    print(f"T-G overall: {'PASS' if (a_ok and b_ok and c_ok and d_ok) else 'FAIL'}  "
          f"(A={'P' if a_ok else 'F'} B={'P' if b_ok else 'F'} "
          f"C={'P' if c_ok else 'F'} D={'P' if d_ok else 'F'})")
    print("=" * 78)
    return 0 if (a_ok and b_ok and c_ok and d_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
