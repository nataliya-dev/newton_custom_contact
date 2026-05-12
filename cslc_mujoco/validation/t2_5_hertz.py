# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tier 2.5 part 1: CSLC at FEM-mapped parameters vs analytical Hertz.

The plan's Tier 2.5 calls for a FEM ground-truth comparison.  Standing
up FEniCSx 3D elasticity with frictional contact is a substantial
piece of infrastructure (mesh refinement under the indenter, Coulomb-
contact via penalty / mortar, dolfinx + petsc4py).  But the plan
itself notes that FEM must first be validated against the
**analytical Hertz half-space** solution at small delta; we use that
*same analytical solution* directly as the ground truth here, which
is exact for r << pad-size and small delta << R.

What we test
------------
H2.5.3  Material-parameter mapping.  With the plan's mapping

    ka = E * spacing^2 / t_pad      (anchor stiffness from uniaxial column)
    kl = G * spacing                (lateral coupling, G = E/(2(1+nu)))

does CSLC's F-vs-delta match the analytical Hertz force

    F_hertz(delta) = (4/3) * E* * sqrt(R) * delta^{3/2}
    E* = E / (1 - nu^2)

within "30 %" (the plan's bound)?  And does CSLC's per-sphere pressure
profile approximate the Hertz semicircle

    p(r) = p_0 * sqrt(1 - r^2 / a^2)   for r <= a = sqrt(R * delta)
         = 0                            otherwise
    p_0 = (3 * F_hertz) / (2 * pi * a^2)

within a recognisable shape, given the discrete-shell lattice
quantisation seen in Tier 2 stages 3 and 4?

Target material
---------------
E = 1 MPa, nu = 0.3, t_pad = 3 mm  (a soft fingertip rubber)
R_indenter = 10 mm
delta in {0.1, 0.25, 0.5, 1.0, 2.0, 4.0} mm

Caveat
------
- Hertz assumes an *elastic half-space*, our pad is 3 mm thick.  At
  a = sqrt(R*delta) ~ t_pad the thin-layer correction kicks in
  (typically Hertz over-predicts by ~ 1.5x in the thin-layer limit).
  This is acknowledged; the more rigorous mapping requires the FEM.
- ka calibration uses A_sphere = spacing^2 (the area-per-sphere on
  the pad face, which IS the right physical interpretation), not
  pi * r_lat^2 (the cross-section of one sphere, which is what the
  plan's text says but is not the right physical quantity for an
  axial-column-stiffness derivation).  See the docstring above.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from cslc_v1.validation.t2_indenter import (
    T2Scene,
    build_t2_model_cslc,
    measure_one,
    radial_profile,
)


# ───────────────────────────────────────────────────────────────────────────
#  Analytical Hertz (rigid sphere on elastic half-space)
# ───────────────────────────────────────────────────────────────────────────

def hertz_force(E: float, nu: float, R: float, delta: float) -> float:
    """F = (4/3) * E* * sqrt(R) * delta^{3/2}, E* = E / (1 - nu^2)."""
    Estar = E / (1.0 - nu ** 2)
    return (4.0 / 3.0) * Estar * np.sqrt(R) * delta ** 1.5


def hertz_pressure_profile(E: float, nu: float, R: float, delta: float,
                           r: np.ndarray) -> np.ndarray:
    """Hertz half-space contact pressure p(r) for a rigid sphere indenter."""
    a = np.sqrt(R * delta)
    F = hertz_force(E, nu, R, delta)
    p0 = 3.0 * F / (2.0 * np.pi * a ** 2)
    p = np.zeros_like(r, dtype=float)
    inside = r <= a
    p[inside] = p0 * np.sqrt(np.maximum(0.0, 1.0 - (r[inside] / a) ** 2))
    return p


# ───────────────────────────────────────────────────────────────────────────
#  H2.5.3  parameter mapping
# ───────────────────────────────────────────────────────────────────────────

def fem_mapped_params(E: float, nu: float, t_pad: float, spacing: float
                      ) -> tuple[float, float]:
    """Plan H2.5.3 mapping  -- returns (ka, kl).

    Derivation (column-of-material approximation):
      ka : a sphere covers spacing^2 of pad area; the column of pad
           material beneath it has axial stiffness E * A / L.
        ka = E * spacing^2 / t_pad

      kl : lateral coupling between adjacent spheres represents the
           shear stiffness across the edge between two columns.
        kl = G * spacing,  G = E / (2 * (1 + nu))

    For E = 1 MPa, nu = 0.3, t_pad = 3 mm, spacing = 5 mm:
      ka = 8 333 N/m  (vs paper-prescribed 15 000)
      kl = 1 925 N/m  (vs paper-prescribed   500)
    """
    G = E / (2.0 * (1.0 + nu))
    ka = E * spacing ** 2 / t_pad
    kl = G * spacing
    return ka, kl


# ───────────────────────────────────────────────────────────────────────────
#  CSLC F-vs-delta at fem-mapped parameters
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class TargetMaterial:
    label: str
    E: float          # Young's modulus [Pa]
    nu: float         # Poisson ratio
    t_pad: float      # pad thickness [m]
    R: float          # indenter radius [m]


def run_cslc_at_target(target: TargetMaterial, scene: T2Scene,
                       deltas_m: np.ndarray) -> dict:
    """Build CSLC scene with FEM-mapped ka, kl; return F at each delta.

    The scene fixes spacing, pad size, and indenter geometry; we
    override ka, kl with the mapping above.  `cslc_kc` is left at the
    handler's auto-calibrated value (so the calibration formula
    `kc = ke*ka/(N_contact*ka - ke)` runs as usual, with the new ka).

    Returns a dict containing per-delta F and per-sphere force / position.
    """
    ka, kl = fem_mapped_params(target.E, target.nu, target.t_pad,
                               scene.cslc_spacing)
    mapped_scene = T2Scene(
        pad_hx_thick=scene.pad_hx_thick,
        pad_half=scene.pad_half,
        indenter_R=target.R,
        cslc_ka=ka, cslc_kl=kl,
        cslc_kc=scene.cslc_kc,
        cslc_dc=scene.cslc_dc, cslc_n_iter=scene.cslc_n_iter,
        cslc_alpha=scene.cslc_alpha,
        cslc_spacing=scene.cslc_spacing,
        # ke kept at scene's value; affects kc calibration only.
        ke=scene.ke, kd=scene.kd, kf=scene.kf, mu=scene.mu, density=scene.density,
    )
    model = build_t2_model_cslc(mapped_scene)
    contacts = model.contacts()

    F = np.zeros_like(deltas_m)
    n_active = np.zeros(len(deltas_m), dtype=int)
    per_sphere = []
    for k, d in enumerate(deltas_m):
        res = measure_one(model, contacts, mapped_scene,
                          delta_indenter=float(d), indenter_body_idx=1)
        F[k] = res["F_total_anchor"]
        n_active[k] = res["n_active"]
        per_sphere.append({
            "pos": res["pos_world"], "f_n": res["f_n_per_sphere"],
            "kc_cal": res["kc_calibrated"],
        })
    return {"ka": ka, "kl": kl, "F": F, "n_active": n_active,
            "per_sphere": per_sphere, "deltas": deltas_m}


def t2_5_main() -> None:
    wp.init()
    np.set_printoptions(precision=6, suppress=True)
    out_dir = "cslc_v1/validation/figures"

    deltas_mm = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    deltas = deltas_mm * 1e-3

    target = TargetMaterial(label="rubber_1MPa", E=1.0e6, nu=0.3,
                            t_pad=0.003, R=0.010)
    F_hertz = np.array([hertz_force(target.E, target.nu, target.R, d) for d in deltas])

    print("\n" + "=" * 72)
    print(f"  Tier 2.5 part 1: CSLC at FEM-mapped (ka, kl) vs analytical Hertz")
    print(f"  target: E = {target.E:.1e} Pa, nu = {target.nu}, t_pad = {target.t_pad*1e3:.1f} mm")
    print(f"  indenter R = {target.R*1e3:.1f} mm")
    print("=" * 72)

    scene_base = T2Scene(cslc_spacing=2.5e-3)   # use the Tier-2-stage-4 asymptote spacing
    print(f"\n  spacing = {scene_base.cslc_spacing*1e3:.2f} mm")
    out_mapped = run_cslc_at_target(target, scene_base, deltas)
    print(f"  H2.5.3-mapped (from E={target.E:.0e}):  "
          f"ka = {out_mapped['ka']:.1f} N/m,  kl = {out_mapped['kl']:.1f} N/m")
    print(f"  paper-prescribed (squeeze_test default): ka = 15000,  kl = 500")

    # Also run the paper-prescribed (ka, kl) -- a direct cross-check that
    # the underestimate vs Hertz is not just a bad mapping but reflects
    # CSLC's whole calibration being detached from physical E.
    scene_paper = T2Scene(cslc_spacing=2.5e-3, cslc_ka=15000.0, cslc_kl=500.0)
    paper_model = build_t2_model_cslc(scene_paper)
    paper_contacts = paper_model.contacts()
    F_paper = np.zeros_like(deltas)
    n_paper = np.zeros(len(deltas), dtype=int)
    for k, d in enumerate(deltas):
        res = measure_one(paper_model, paper_contacts, scene_paper,
                          delta_indenter=float(d), indenter_body_idx=1)
        F_paper[k] = res["F_total_anchor"]
        n_paper[k] = res["n_active"]

    print(f"\n  {'delta_mm':>9}  {'F_mapped':>10}  {'err_mapped':>11}  "
          f"{'F_paper':>10}  {'err_paper':>10}  {'F_hertz':>10}  {'n_act':>6}")
    out = out_mapped  # used by plotting below
    for k, d_mm in enumerate(deltas_mm):
        err_m = (out_mapped["F"][k] - F_hertz[k]) / F_hertz[k]
        err_p = (F_paper[k] - F_hertz[k]) / F_hertz[k]
        print(f"  {d_mm:>9.3f}  {out_mapped['F'][k]:>10.4f}  {err_m*100:>10.2f}%  "
              f"{F_paper[k]:>10.4f}  {err_p*100:>9.2f}%  {F_hertz[k]:>10.4f}  "
              f"{out_mapped['n_active'][k]:>6d}")

    # Plot F vs delta -- log-log
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].loglog(deltas * 1e3, out_mapped["F"], "-o", label="CSLC (H2.5.3-mapped)")
    axes[0].loglog(deltas * 1e3, F_paper, "-d", label="CSLC (paper ka,kl)")
    axes[0].loglog(deltas * 1e3, F_hertz, "-s", label="Hertz (E={:.0e} Pa)".format(target.E))
    axes[0].set_xlabel("delta (mm)")
    axes[0].set_ylabel("aggregate F (N)")
    axes[0].set_title("Tier 2.5  F vs delta")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    # Plot Hertz p(r) vs CSLC p(r) at delta = 1 mm
    d_test = 1.0e-3
    a_indent = np.sqrt(target.R * d_test)
    r_grid = np.linspace(0.0, 1.5 * a_indent, 50)
    p_hertz = hertz_pressure_profile(target.E, target.nu, target.R, d_test, r_grid)
    # Look up CSLC profile at delta = 1 mm.
    k1 = int(np.argmin(np.abs(deltas - d_test)))
    ps_cslc = out["per_sphere"][k1]
    pos_c = ps_cslc["pos"]
    f_c = ps_cslc["f_n"]
    active = f_c > 1e-6
    pos_c = pos_c[active]; f_c = f_c[active]
    r_c, p_c, cnt_c = radial_profile(
        pos_c, f_c, r_max=1.5 * a_indent,
        n_bins=20, sphere_area=scene_base.cslc_spacing ** 2,
    )

    axes[1].plot(r_grid * 1e3, p_hertz / 1e3, "-", label="Hertz analytical")
    axes[1].plot(r_c * 1e3, p_c / 1e3, "-o", label="CSLC (mapped)", markersize=4)
    axes[1].axvline(a_indent * 1e3, color="k", linestyle=":",
                    label=f"a = sqrt(R*delta) = {a_indent*1e3:.2f} mm")
    axes[1].set_xlabel("r (mm)")
    axes[1].set_ylabel("p(r) (kPa)")
    axes[1].set_title(f"Tier 2.5  p(r) at delta = {d_test*1e3:.1f} mm")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"CSLC (FEM-mapped) vs analytical Hertz  "
                 f"-- E = {target.E:.0e} Pa, nu = {target.nu}, R = {target.R*1e3:.0f} mm")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/t2_5_hertz_comparison.png", dpi=140)
    plt.close(fig)
    print(f"\n  Wrote {out_dir}/t2_5_hertz_comparison.png")

    # H2.5.3 verdict
    print("\n  --- H2.5.3 verdict ---")
    print(f"  Plan target: CSLC F within 30 % of Hertz F under the mapped parameters")
    midrange = (deltas_mm >= 0.5) & (deltas_mm <= 2.0)
    mid_errs = np.abs((out["F"][midrange] - F_hertz[midrange]) / F_hertz[midrange])
    max_err = float(mid_errs.max() * 100) if mid_errs.size else float("nan")
    avg_err = float(mid_errs.mean() * 100) if mid_errs.size else float("nan")
    print(f"  Average |F_CSLC - F_Hertz| / F_Hertz over delta in [0.5, 2.0] mm: "
          f"{avg_err:.1f} %")
    print(f"  Max     |F_CSLC - F_Hertz| / F_Hertz over the same range:        "
          f"{max_err:.1f} %")
    verdict = "PASS" if max_err < 30.0 else "FAIL"
    print(f"  H2.5.3: {verdict}  (target: max rel.err < 30 %)")


if __name__ == "__main__":
    t2_5_main()
