# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 10b -- patch-resultant projection as a symmetry-residual ablation.

A controlled A/B against the Step 10 PART C convergence baseline:

  Baseline -- ``cslc_theory.equilibrium_point_set_numerical``.
              The canonical multi-point solver; each per-target-point
              contact force is summed in its own line-of-centres
              direction.  At finite sampling density the per-point
              direction variance produces a lateral residual
              ``|delta_t|`` that decays as N grows (Step 10 PART C
              showed 17x reduction from N = 150 to N = 3000).

  Variant  -- ``cslc_projection_experiment.equilibrium_point_set_projected_numerical``.
              The same energy, but the contact term is evaluated at
              ``delta_proj = (delta . n_outward) * n_outward``.  The
              tangential component of ``delta`` therefore does not
              enter the contact energy; the anchor is the only term
              that depends on ``delta_t``, so ``delta_t -> 0`` at
              equilibrium by construction.

Same scene as Step 10 PART C: 50 mm box, pad sphere face-on at
``phi_rest = 0.5 mm``, ``radius_factor = 1.0`` (target spheres tile
with redundancy), sweep ``n_samples in {150, 300, 600, 1500, 3000}``.

Verdict: the variant should
  (a) reduce ``|delta_t|`` BELOW the baseline at every N, with the
      effect strongest at sparse N (where the baseline residual is
      worst); and
  (b) NOT disturb ``delta_n`` substantially -- the projection only
      affects the lateral coupling, so the normal-axis equilibrium
      should agree with the baseline within the residual's order of
      magnitude.

This is an EXPERIMENT.  See the module docstring of
``cslc_projection_experiment.py`` for the removal checklist; deleting
both files (this driver + the variant module) restores the canonical
Step 10 path with no side effects.

Run::

    uv run --extra importers -m cslc_main.theory.test_10b_symmetry_projection

Output::

    cslc_main/theory/figures/10b_projection_ablation.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cslc_main.theory.cslc_box import make_box_target
from cslc_main.theory.cslc_projection_experiment import (
    equilibrium_point_set_projected_numerical,
)
from cslc_main.theory.cslc_theory import (
    LatticeSphere,
    equilibrium_point_set_numerical,
    point_set_contact_force,
)


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _scene_params():
    """Same scene as Step 10 PART C, kept verbatim so the A/B is honest."""
    return dict(
        r_lat=1.5e-3,
        ka=1.0e4,
        kc=1.0e4,
        phi_rest=5.0e-4,
        box_extents=(0.05, 0.05, 0.05),
    )


def _run_one_N(n_samples: int, p: dict) -> dict:
    """Solve baseline AND projected variant on the same scene, return both."""
    target = make_box_target(
        extents=p["box_extents"], n_samples=n_samples, seed=0,
        radius_factor=1.0)
    box_half_z = 0.5 * p["box_extents"][2]
    pad_p_z = box_half_z + p["r_lat"] - p["phi_rest"]
    sphere = LatticeSphere(
        p=np.array([0., 0., pad_p_z]),
        r=p["r_lat"],
        n=np.array([0., 0., -1.]),
        ka=p["ka"],
    )

    # Baseline.
    delta_b, info_b = equilibrium_point_set_numerical(
        sphere, target, p["kc"], eps=1.0e-7)
    F_b = point_set_contact_force(sphere, target, delta_b, p["kc"], eps=1.0e-7)

    # Projected variant.
    delta_p, info_p = equilibrium_point_set_projected_numerical(
        sphere, target, p["kc"], eps=1.0e-7)
    # The force on the BALL emitted by the variant is the projected
    # force -- only the n_outward component matters under the modified
    # potential.  But we still report the FULL line-of-centres force
    # for fair comparison; the variant's "effective contact force" is
    # the n_outward projection of that.
    F_p_full = point_set_contact_force(
        sphere, target, delta_p, p["kc"], eps=1.0e-7)
    F_p_along_n = float(np.dot(F_p_full, sphere.n)) * sphere.n

    def _decompose(delta, F, sphere):
        delta_n = float(np.dot(delta, sphere.n))
        delta_t = delta - delta_n * sphere.n
        F_n = float(np.dot(F, sphere.n))
        F_t = F - F_n * sphere.n
        return delta_n, float(np.linalg.norm(delta_t)), F_n, float(np.linalg.norm(F_t))

    db_n, db_t, Fb_n, Fb_t = _decompose(delta_b, F_b, sphere)
    dp_n, dp_t, Fp_n, Fp_t = _decompose(delta_p, F_p_full, sphere)

    return {
        "N": n_samples,
        "baseline": {"delta_n": db_n, "delta_t": db_t,
                     "F_n": Fb_n, "F_t": Fb_t, "nit": info_b["nit"]},
        "projected": {"delta_n": dp_n, "delta_t": dp_t,
                      "F_n": Fp_n, "F_t": Fp_t, "nit": info_p["nit"]},
    }


def main() -> int:
    print()
    print("=" * 78)
    print("Step 10b  Patch-resultant projection vs baseline (same scene as 10 PART C)")
    print("=" * 78)

    p = _scene_params()
    n_samples_list = [150, 300, 600, 1500, 3000]
    rows = [_run_one_N(N, p) for N in n_samples_list]

    print()
    print(f"{'N':>5}  {'delta_n base':>14}  {'delta_n proj':>14}  "
          f"{'|d_t| base':>13}  {'|d_t| proj':>13}  {'|d_t| ratio':>12}")
    for r in rows:
        b, q = r["baseline"], r["projected"]
        ratio = q["delta_t"] / max(b["delta_t"], 1.0e-30)
        print(f"{r['N']:>5d}  "
              f"{b['delta_n'] * 1e6:>12.2f} um  "
              f"{q['delta_n'] * 1e6:>12.2f} um  "
              f"{b['delta_t'] * 1e9:>11.1f} nm  "
              f"{q['delta_t'] * 1e9:>11.1f} nm  "
              f"{ratio:>10.2e}x")

    # Verdict criteria:
    # (a) projected |delta_t| < baseline |delta_t| at every N (the
    #     projection should always reduce, not amplify, the lateral
    #     residual).
    # (b) projected delta_n agrees with baseline delta_n to within the
    #     baseline lateral RESIDUAL (i.e. the projection didn't
    #     wholesale change the normal-axis equilibrium).
    crit_a_per_N = []
    crit_b_per_N = []
    for r in rows:
        b, q = r["baseline"], r["projected"]
        crit_a_per_N.append(q["delta_t"] < b["delta_t"])
        # Tolerance for delta_n agreement = O(baseline lateral residual).
        # If the baseline is contaminated by sampling noise at this
        # level, the projected delta_n shouldn't differ by more than
        # the contamination magnitude.
        tol_n = max(b["delta_t"], 1.0e-9)
        crit_b_per_N.append(abs(q["delta_n"] - b["delta_n"]) <= tol_n)
    crit_a = all(crit_a_per_N)
    crit_b = all(crit_b_per_N)

    print()
    print(f"Criterion (a): projected |delta_t| < baseline |delta_t| at every N    "
          f"{'PASS' if crit_a else 'FAIL'}")
    print(f"Criterion (b): |delta_n_proj - delta_n_base| <= baseline |delta_t|    "
          f"{'PASS' if crit_b else 'FAIL'}")

    # Plot.
    Ns = np.array([r["N"] for r in rows])
    base_t = np.array([r["baseline"]["delta_t"] for r in rows])
    proj_t = np.array([r["projected"]["delta_t"] for r in rows])
    base_n = np.array([r["baseline"]["delta_n"] for r in rows])
    proj_n = np.array([r["projected"]["delta_n"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.loglog(Ns, base_t * 1e9, "o-", label="baseline (Step 10)")
    ax.loglog(Ns, np.clip(proj_t * 1e9, 1.0e-3, None), "s-",
              label="projected variant (10b)")
    ax.set_xlabel("n_samples on box")
    ax.set_ylabel("|delta_t| [nm]")
    ax.set_title("Lateral residual vs sampling density")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    ax = axes[1]
    ax.plot(Ns, base_n * 1e6, "o-", label="baseline")
    ax.plot(Ns, proj_n * 1e6, "s-", label="projected variant")
    ax.set_xscale("log")
    ax.set_xlabel("n_samples on box")
    ax.set_ylabel("delta_n [um]")
    ax.set_title("Normal-axis equilibrium (should be ~unchanged)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "10b_projection_ablation.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"figure -> {out.relative_to(Path.cwd())}")

    ok = crit_a and crit_b
    print()
    print("=" * 78)
    print(f"Step 10b symmetry-projection ablation: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
