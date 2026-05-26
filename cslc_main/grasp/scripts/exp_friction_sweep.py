# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction sweep on the R=20mm/72deg dome.

Question: can stronger CSLC friction (raise ``material.mu``, raise
``k_stick``) overcome the geometric wedge instability on the curved
dome?  Static math says yes (tan(tilt) < mu should be stable); the
slow-LIFT experiment suggested no.  This is the empirical test.

Post-unification: a single ``material.mu`` drives both the lattice
stick-slip block AND MuJoCo's Coulomb cone on emitted contacts.

Runs:
  * baseline:          mu=0.3,  k_stick=2.5e4
  * high mu:           mu=1.0,  k_stick=2.5e4
  * high k_stick:      mu=0.3,  k_stick=2.5e5
  * both:              mu=1.0,  k_stick=2.5e5
"""

from __future__ import annotations

import math

from cslc_main.grasp.params import GraspConfig
from cslc_main.grasp.runner import run_headless


def make_cfg(*, mu: float, k_stick: float, label: str) -> GraspConfig:
    cfg = GraspConfig()
    cfg.pad.kind = "dome_param"
    cfg.pad.dome_param_R_pad = 0.020
    cfg.pad.dome_param_half_angle = 72.0 * math.pi / 180.0
    cfg.material.mu = mu
    cfg.cslc.k_stick = k_stick
    cfg.logging.run_label = label
    cfg.logging.use_timestamp = False
    cfg.logging.save_postsim_plots = False
    cfg.logging.save_lattice_preview = False
    return cfg


def main() -> None:
    runs = [
        ("baseline", 0.3, 2.5e4, "_fric_mu0p3_kstick25k"),
        ("hi-mu",    1.0, 2.5e4, "_fric_mu1p0_kstick25k"),
        ("hi-stick", 0.3, 2.5e5, "_fric_mu0p3_kstick250k"),
        ("both",     1.0, 2.5e5, "_fric_mu1p0_kstick250k"),
    ]
    for name, mu, ks, label in runs:
        print()
        print("=" * 72)
        print(f"  mu_friction={mu}  k_stick={ks:.0f}  ({name})")
        print("=" * 72)
        run_headless(make_cfg(mu=mu, k_stick=ks, label=label))


if __name__ == "__main__":
    main()
