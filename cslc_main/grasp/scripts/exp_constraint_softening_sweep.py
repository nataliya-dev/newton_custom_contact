# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constraint-softening sweep on the R=20mm/72deg dome (Step 11 mechanism).

Follow-up to ``exp_emitted_friction_sweep.py``.  That script showed
HOLD ``dz/dt`` is invariant to mu across 0.5 -> 5.0 (10x range).
This script tests the alternative hypothesis: **the wedge climb is
driven by MuJoCo's regularised contact constraint admitting per-step
position drift along the contact normal, with the drift scale set by
the constraint compliance (≈ solref / timeconst in MuJoCo
terminology).  If that's right, drift should scale with the EMITTED
contact stiffness MuJoCo applies, *independently* of mu.**

What we sweep: ``cfg.material.ke`` over four orders of magnitude.
This propagates to ``kc_series = (cslc_kc * target_ke) /
(cslc_kc + target_ke + eps^2)`` in
[cslc_kernels.py:933](../../newton/_src/geometry/cslc_kernels.py#L933),
which is exactly the per-contact stiffness MuJoCo turns into a
timeconst via ``timeconst = sqrt(imp/ke)``.  Higher ``material.ke``
-> tighter constraint -> less per-step drift -> less climb (if the
hypothesis holds).

Caveat: ``material.ke`` is also read by ``calibrate_kc`` as the
bulk-stiffness target, so the lattice's internal ``cslc_kc`` is
re-derived to make the per-pad aggregate match ``ke_bulk``.  Both the
internal physics and the emitted stiffness scale together.  This is
fine for a yes/no answer (does drift scale with the emitted
stiffness?) -- isolating "regularisation only" would require a
kernel knob that scales only the emission, which we defer.

What we hold fixed (compared with the emitted-friction sweep):

  * ``cfg.material.mu = 0.5`` -- production baseline; the
    friction-sweep result already shows mu is invisible to the wedge.
  * ``cfg.cslc.mu_friction = 0.3`` -- in-kernel knob, baseline.
  * ``cfg.cslc.k_stick = 2.5e4``  -- baseline.
  * Pad geometry: R=20mm/72deg dome (the wedge-canonical scene).

Outcomes interpreted in notes.md Step 11 closure:

  * Drift scales with ``material.ke`` -> constraint-regularisation
    artifact.  Closure rewrites to "wedge climb is a regularisation
    leak; tighter constraints reduce it; for practical grasps, use
    flat contacts."
  * Drift insensitive to ``material.ke`` -> not regularisation.
    Mechanism dig continues: per-step contact-set reformation,
    aggregation across mixed-normal per-sphere contacts, etc.
"""

from __future__ import annotations

import math

from cslc_main.grasp.params import GraspConfig
from cslc_main.grasp.runner import run_headless


def make_cfg(*, ke: float, label: str) -> GraspConfig:
    cfg = GraspConfig()
    cfg.pad.kind = "dome_param"
    cfg.pad.dome_param_R_pad = 0.020
    cfg.pad.dome_param_half_angle = 72.0 * math.pi / 180.0

    # Sweep target: MuJoCo-facing constraint stiffness via material.ke.
    cfg.material.ke = ke

    # Hold every other knob at production baseline so the variation is
    # attributable solely to the constraint-stiffness sweep.
    cfg.material.mu = 0.5
    cfg.cslc.mu_friction = 0.3
    cfg.cslc.k_stick = 2.5e4

    cfg.logging.run_label = label
    cfg.logging.use_timestamp = False
    cfg.logging.save_postsim_plots = False
    cfg.logging.save_lattice_preview = False
    return cfg


def main() -> None:
    # Four points spanning 4 orders of magnitude.  Production = 5e4.
    runs = [
        ("ke5e3_soft",         5.0e3,   "_softref_ke5e3"),
        ("ke5e4_baseline",     5.0e4,   "_softref_ke5e4"),
        ("ke5e5_stiff",        5.0e5,   "_softref_ke5e5"),
        ("ke5e6_very_stiff",   5.0e6,   "_softref_ke5e6"),
    ]
    for name, ke, label in runs:
        print()
        print("=" * 72)
        print(f"  material.ke = {ke:.1e}   ({name})   "
              f"holds mu_pad=0.5, mu_friction=0.3")
        print("=" * 72)
        run_headless(make_cfg(ke=ke, label=label))


if __name__ == "__main__":
    main()
