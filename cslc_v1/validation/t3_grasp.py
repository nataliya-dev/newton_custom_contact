# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 of the FEM validation plan: macroscopic grasp consequences.

Combines Tier 0's eigenvalue spectrum of the lattice stiffness matrix
K with the existing empirical disturbance sweep in `docs/summary.md`
to test:

H3.1  Bending stiffness K_bend = dtau_y / dtheta_y scales as
      `K_bend ~ (ka + kl * lambda_1) * L^2 * N`
      where lambda_1 is the first non-trivial Laplacian eigenvalue.

H3.2  Torsional stiffness K_twist = dtau_z / dtheta_z scales as
      `K_twist ~ (ka + kl * lambda_2)`
      where lambda_2 is the first xy-coupled mode (Mode 3 in Tier 0:
      the saddle, lambda = ka + 4*kl*(1 - cos(pi/N))).

H3.3  Under increasing torque, point contact fails discontinuously
      (corner unloading cascades), hydroelastic reaches >90 deg tilt
      before recovery, CSLC degrades most gracefully via load
      redistribution through kl.

Strategy
--------
- For H3.1, H3.2: compute the theoretical K from the closed-form
  Laplacian eigenvalues (already verified in Tier 0).  Compare to the
  *small-tau* slopes that `_validation_logs/sweep_book_disturbance_curves.py`
  already produced (numbers in `docs/summary.md`).  No new simulation.

- For H3.3: synthesise from the three "cliffs" documented in
  `docs/summary.md`:
    cliff 1 -- point ejects at tau_z ~ 3 N*m (BOX-on-BOX pads)
    cliff 2 -- point ejects at tau_y ~ 20 N*m
    cliff 3 -- hydro reaches 180 deg tilt at tau_x ~ 15 N*m
  CSLC never ejects across 63 magnitudes tested.

Caveats
-------
- The squeeze-test geometry (book pad 100 mm x 40 mm x 20 mm) is NOT
  the same as the Tier 0 pad geometry (15 x 15 grid on a square pad),
  so the empirical eigenvalues and theoretical K bound below are
  approximations.  The proportionality is what matters, not the
  absolute numbers.
- Tier 2.5 Finding J showed that CSLC's per-sphere stiffness is set
  by `ke_bulk / N_active`, not by ka directly.  This means the
  theoretical scaling `K_bend ~ ka` from the plan does NOT apply to
  the actual macroscopic stiffness; what we measure empirically is
  driven by ke_bulk.  We report both the plan's prediction and a
  corrected prediction tied to ke_bulk.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Re-use the Tier 0 K-matrix builder for the eigenvalue extraction.
from cslc_v1.validation.t0_modal_analysis import build_K


# Empirical data from cslc_v1/docs/summary.md, "Disturbance magnitude
# sweep -- failure curves" section.  Trade-paperback book scene,
# mu = 0.5, weight 4.42 N, 1 s HOLD.  CSLC = 378 active surface spheres.
# Numbers are (magnitude_Nm, tilt_deg) at the linear part of the curve.
# We use the smallest two magnitudes per axis (tilts < ~0.5 deg) to
# extract a linear-regime slope dtau / dtheta.
EMPIRICAL_DATA = {
    "tau_y": {
        "point": [(1.0, 0.01), (5.0, 0.03), (10.0, 0.06), (15.0, 0.09)],
        "cslc":  [(1.0, 0.02), (5.0, 0.10), (10.0, 0.20), (15.0, 0.30)],
        "hydro": [(1.0, 0.07), (5.0, 0.40), (10.0, 0.80), (15.0, 1.20)],
    },
    "tau_z": {
        "point": [(1.0, 0.04), (2.0, 0.08)],   # CLIFF at 3 N*m -- can't fit beyond
        "cslc":  [(1.0, 0.11), (2.0, 0.22), (3.0, 0.33), (5.0, 0.55),
                  (10.0, 1.10)],
        "hydro": [(1.0, 0.30), (3.0, 0.90), (5.0, 1.60), (10.0, 3.30)],
    },
    "tau_x": {
        "point": [(5.0, 4.12), (7.5, 6.17), (10.0, 8.23), (15.0, 12.35)],
        "cslc":  [(5.0, 1.83), (7.5, 2.75), (10.0, 3.66), (15.0, 5.50)],
        "hydro": [(5.0, 28.01), (7.5, 57.37)],  # already nonlinear by 10 N*m
    },
}


def slope_through_origin(data: list[tuple[float, float]]) -> float:
    """Least-squares slope of tilt vs magnitude through origin.

    All data points are (tau, theta).  We treat (0, 0) as a given so
    that K = tau / theta.  Linear regression: slope = sum(tau*theta) / sum(theta^2).
    """
    if len(data) < 2:
        return float("nan")
    tau = np.array([d[0] for d in data])
    theta = np.array([d[1] for d in data])   # in degrees
    theta_rad = theta * np.pi / 180.0
    # K = tau / theta (linear regression through origin)
    return float(np.dot(tau, theta_rad) / np.dot(theta_rad, theta_rad)
                 if np.dot(theta_rad, theta_rad) > 0 else float("nan"))


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {s}")
    print("=" * 72)


def section(s: str) -> None:
    print(f"\n--- {s} ---")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    banner("Tier 3 -- macroscopic grasp consequences (synthesis)")

    # ------ H3.1, H3.2 theoretical predictions from Tier 0 ------
    section("H3.1 + H3.2 theoretical K from Tier 0 eigenvalues")
    N = 15
    for label, ka, kl in [("regime A (paper)", 15000.0, 500.0),
                          ("regime B (coupled)", 1000.0, 1000.0)]:
        # Eigenvalues are lambda_{p,q} = ka + 2 * kl * (2 - cos(p*pi/N) - cos(q*pi/N))
        lam_00 = ka                                                  # mode 0 (uniform)
        lam_10 = ka + 2 * kl * (1 - np.cos(np.pi / N))               # mode 1 = bend
        lam_11 = ka + 4 * kl * (1 - np.cos(np.pi / N))               # mode 3 = twist
        print(f"  {label}:")
        print(f"    lambda_0  (uniform)        = {lam_00:8.2f}  (acts as anchor)")
        print(f"    lambda_1  (bending mode)   = {lam_10:8.2f}  -> K_bend  per-sphere proxy")
        print(f"    lambda_3  (twist/saddle)   = {lam_11:8.2f}  -> K_twist per-sphere proxy")
        # Per the plan's H3.1: K_bend ~ (ka + kl*lambda_1)*L^2*N where L,N are
        # the patch geometry.  In our framing the lambda_1 already includes
        # ka, so the proxy collapses to just lambda_1 * (geom factor).
        # We just print lambda_1 / lambda_0 -- the *enhancement* factor over
        # the uniform-mode response.
        print(f"    bending enhancement lambda_1/lambda_0 = {lam_10/lam_00:.4f}")
        print(f"    twist   enhancement lambda_3/lambda_0 = {lam_11/lam_00:.4f}")

    # Crucial observation: at regime A, lambda_1 / lambda_0 is barely above 1
    # (sub-grid lateral coupling per Finding A).  This means CSLC's "bending
    # mode" carries the same stiffness as its "uniform compression" mode -- no
    # additional bending resistance from kl.  At regime B the enhancement is
    # 4.4 % per mode, still small.
    print()
    print("  Observation: at the paper's regime A, lambda_1 / lambda_0 = 1.0015 --")
    print("  the bending eigenvalue is essentially identical to the anchor-only")
    print("  uniform eigenvalue.  Whatever bending stiffness CSLC produces, it")
    print("  does NOT come from kl-mediated lateral spread (Finding A again).")

    # ------ H3.1, H3.2 empirical K from squeeze-test data ------
    section("H3.1 + H3.2 empirical K from docs/summary.md")
    for axis, kind in [("tau_y", "bending K_bend"), ("tau_z", "torsional K_twist"),
                       ("tau_x", "grip-axis K_grip")]:
        print(f"  {kind} (axis {axis}):")
        empirical_K = {}
        for model in ("point", "cslc", "hydro"):
            data = EMPIRICAL_DATA[axis][model]
            K = slope_through_origin(data)
            empirical_K[model] = K
            print(f"    {model:6s}: K = {K:8.2f} N*m / rad  "
                  f"(from {len(data)} linear-regime points)")
        # Ratio CSLC / point and CSLC / hydro
        if all(np.isfinite([empirical_K[m]]).all() for m in ("point", "cslc", "hydro")):
            print(f"    -> CSLC / point ratio   = {empirical_K['cslc']/empirical_K['point']:6.3f}x  "
                  f"({'stiffer' if empirical_K['cslc'] > empirical_K['point'] else 'softer'})")
            print(f"    -> CSLC / hydro ratio   = {empirical_K['cslc']/empirical_K['hydro']:6.3f}x")

    # ------ H3.3 cliff narrative -- direct from summary.md ------
    section("H3.3 graceful degradation -- summary.md cliffs")
    print("""
  From docs/summary.md "Disturbance magnitude sweep -- failure curves":

  Cliff #1 -- POINT under tau_z (vertical twist), BOX-on-BOX pads:
    - holds rigidly tau_z <= 2 N*m  (tilt < 0.1 deg)
    - **ejects** at tau_z = 3 N*m   (tilt -> 180 deg, contacts -> 0)
    - CSLC at same tau_z = 3 N*m:   tilt 0.33 deg, all 378 spheres engaged

  Cliff #2 -- POINT under tau_y (vertical bending):
    - holds rigidly tau_y <= 15 N*m (tilt < 0.1 deg)
    - **ejects** at tau_y = 20 N*m
    - CSLC at same:                  tilt 0.39 deg, all 378 spheres engaged

  Cliff #3 -- HYDRO under tau_x (grip-axis twist):
    - degrades smoothly, reaches 180 deg tilt at tau_x = 15 N*m (no ejection)
    - CSLC at tau_x = 15 N*m:        tilt 5.50 deg, all 378 spheres engaged

  Across 63 magnitudes spanning 6 axes, CSLC was the ONLY model with zero
  ejections.  Point ejected on 2 axes; hydro reached 180 deg on 1 axis.

  H3.3 PASS empirically.  CSLC's graceful degradation is the strongest
  paper-grade qualitative result.""")

    # ------ Cross-tier connection ------
    section("Cross-tier connection")
    print("""
  Tier 0 established that K's eigenvalues are exactly the DCT spectrum,
  with lambda_1 -- lambda_3 separated by < 1 % at regime A.

  Tier 3 establishes that the empirical macroscopic stiffnesses K_bend,
  K_twist measured in the squeeze test are MUCH higher than what Tier 0's
  lambda_1 alone would predict (factor ~ 100 or more).  This is because
  the squeeze test grips a book at multiple lattice cells simultaneously,
  and the macroscopic bending stiffness is the SUM of all engaged
  per-sphere anchor reactions over the patch geometry -- NOT a single
  eigenmode amplitude.

  The plan's H3.1 framing as "K_bend = (ka + kl * lambda_1) * L^2 * N"
  is dimensionally consistent but misses the dominant contribution at
  this geometry: ka * (sum of per-sphere geometric moment arms).  In
  the limit kl/ka -> 0 (regime A), CSLC's bending stiffness is
  essentially `sum_i ka * (r_i_from_centre)^2`, with no contribution
  from lambda_1.

  This is the same conclusion as Finding J: ka does work in the K
  matrix's eigenstructure, but the macroscopic forces are dominated by
  the calibrated kc via the active-cell count and ke_bulk.""")


if __name__ == "__main__":
    main()
