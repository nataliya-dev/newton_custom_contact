# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction sweep on the R=20mm/72deg dome (Step 11 closure follow-up).

Question (from ``cslc_main/theory/notes.md`` Step 11): when friction
was previously swept in {0.3, 1.0}, the HOLD ``dz/dt`` climb rate on
the R = 20 mm / 72 deg dome was UNCHANGED at ~3.0 mm/s.  Static math
says ``tan(theta) < mu`` should be stable (tilt ~ 11.5 deg -> 22 deg
gives tan in [0.20, 0.40], both < mu=1.0).  Where's the gap?

Post-unification (Tier 2.3): a single ``material.mu`` drives BOTH the
lattice stick-slip block and MuJoCo's Coulomb cone on emitted
contacts.  The historic split (``cslc.mu_friction`` vs the geom-pair
``material.mu``) is gone -- they were always the same knob in
practice once :func:`cslc_kernels.write_cslc_contacts` started writing
``out_friction = 1.0`` (= no extra scale).

This script sweeps ``material.mu`` from 0.5 to 5.0.  If the climb
collapses as mu grows, friction WAS the bottleneck and the wedge is
preventable with higher friction.  If the climb persists, the wedge
is geometric (pad-shape problem, not a parameter problem).

Falsification matrix:

  mu_material  expected static stability (tan_theta < mu?)    observed dz/dt drops?
  -----------  ----------------------------------------------  ---------------------
  0.5          marginal: tan(11.5°)=0.20 OK, tan(22°)=0.40 OK  TBD
  1.0          STABLE at all observed tilts                    TBD
  2.0          STABLE with comfortable margin                  TBD
  5.0          STABLE everywhere                                TBD

This runs the existing R=20mm/72deg dome grasp 4 times (one per mu)
at ~20 s wall each, ~90 s total.  Outputs land in
``outputs/grasp/_emit_fric_mu*/`` for cross-run analysis.
"""

from __future__ import annotations

import math

from cslc_main.grasp.params import GraspConfig
from cslc_main.grasp.runner import run_headless


def make_cfg(*, mu_material: float, label: str) -> GraspConfig:
    cfg = GraspConfig()
    cfg.pad.kind = "dome_param"
    cfg.pad.dome_param_R_pad = 0.020
    cfg.pad.dome_param_half_angle = 72.0 * math.pi / 180.0

    # Sweep the single unified friction knob.
    cfg.material.mu = mu_material
    cfg.cslc.k_stick = 2.5e4

    cfg.logging.run_label = label
    cfg.logging.use_timestamp = False
    cfg.logging.save_postsim_plots = False
    cfg.logging.save_lattice_preview = False
    return cfg


def main() -> None:
    runs = [
        ("baseline_mu0p5",  0.5,  "_emit_fric_mu0p5"),
        ("mu1p0",           1.0,  "_emit_fric_mu1p0"),
        ("mu2p0",           2.0,  "_emit_fric_mu2p0"),
        ("mu5p0",           5.0,  "_emit_fric_mu5p0"),
    ]
    for name, mu, label in runs:
        print()
        print("=" * 72)
        print(f"  material.mu = {mu}   ({name})   "
              f"(unified -- drives both lattice and emission)")
        print("=" * 72)
        run_headless(make_cfg(mu_material=mu, label=label))


if __name__ == "__main__":
    main()
