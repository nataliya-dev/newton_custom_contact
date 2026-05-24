# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose dome climbing instability: dynamics-induced vs purely geometric.

Run the parametric dome (R=20mm, 72 deg) at three LIFT speeds, holding total
LIFT distance fixed at 22.5 mm.  If the climbing during HOLD is caused by
LIFT dynamics (overshoot leaves the ball above the pad's equator), slowing
the LIFT should reduce HOLD-phase z drift to near zero.  If z still climbs
at the slowest LIFT, the instability is purely geometric (wedge force on
any z offset, no matter how small) and trajectory tuning can't fix it.
"""

from __future__ import annotations

import math

from cslc_main.grasp.params import GraspConfig
from cslc_main.grasp.runner import run_headless


def make_cfg(*, lift_speed: float, lift_duration: float, label: str) -> GraspConfig:
    cfg = GraspConfig()
    cfg.pad.kind = "dome_param"
    cfg.pad.dome_param_R_pad = 0.020
    cfg.pad.dome_param_half_angle = 72.0 * math.pi / 180.0
    cfg.timing.lift_speed = lift_speed
    cfg.timing.lift_duration = lift_duration
    # Keep ramp short enough that it doesn't dominate the lift.
    cfg.timing.lift_ramp_duration = min(0.25, 0.2 * lift_duration)
    # Always 3 s of HOLD so creep is comparable across runs.
    cfg.timing.hold_duration = 3.0
    cfg.logging.run_label = label
    cfg.logging.use_timestamp = False
    cfg.logging.save_postsim_plots = False  # we'll make our own diagnostic
    cfg.logging.save_lattice_preview = False
    return cfg


def main() -> None:
    # Hold lift_speed * lift_duration = 0.0225 m  (= 22.5 mm total LIFT).
    runs = [
        ("fast (production)", 0.015, 1.5, "_slowlift_15mm_per_s"),
        ("medium",            0.005, 4.5, "_slowlift_5mm_per_s"),
        ("slow",              0.0015, 15.0, "_slowlift_1p5mm_per_s"),
    ]
    for name, speed, duration, label in runs:
        print()
        print("=" * 72)
        print(f"  LIFT speed = {speed*1e3:.2f} mm/s  duration = {duration:.1f} s "
              f"({name})")
        print("=" * 72)
        cfg = make_cfg(lift_speed=speed, lift_duration=duration, label=label)
        run_headless(cfg)


if __name__ == "__main__":
    main()
