# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-run metrics for grasp tests.

Captures the bare minimum needed for pass/fail judgement and for the
regression-test assertions in :class:`runner.Example.test_final`.
Anything more granular (per-phase summaries, contact-force histories,
CSLC δ statistics) lives in the CSV logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    """A run's high-level outcome.

    Populated incrementally by the runner; queried by the summary
    printer and the regression-test assertion.
    """

    name: str = ""
    # Per-step held-object centre Z [m].
    object_z: list[float] = field(default_factory=list)
    # Per-step held-object XY displacement [m] from its t=0 position.
    object_xy_drift: list[float] = field(default_factory=list)
    # Per-step contact count (post-narrow-phase, ≥ 0).
    contacts: list[int] = field(default_factory=list)
    # Per-step normal force per pad side [N] from MuJoCo actuator torques on
    # the X-DOF (Newton's 3rd law: actuator must supply same magnitude as
    # contact reaction along X).  Populated for all contact models.
    F_n_left: list[float] = field(default_factory=list)
    F_n_right: list[float] = field(default_factory=list)
    # Per-step pad inward displacement [m] (commanded by SQUEEZE phase).
    # Used to recover steady-state penetration depth during HOLD.
    dx_left: list[float] = field(default_factory=list)

    # ── Derived (cheap) ─────────────────────────────────────────────

    @property
    def max_z(self) -> float:
        return max(self.object_z) if self.object_z else 0.0

    @property
    def final_z(self) -> float:
        return self.object_z[-1] if self.object_z else 0.0

    @property
    def min_z(self) -> float:
        return min(self.object_z) if self.object_z else 0.0

    @property
    def lifted(self) -> bool:
        """True if the object rose more than 5 mm above its settled height."""
        if len(self.object_z) < 2:
            return False
        return self.max_z > self.min_z + 0.005

    @property
    def held(self) -> bool:
        """True if the object is still clearly airborne AND at a plausible height.

        Two conditions, both required:

        1. ``final_z > min_z + 5mm`` — the object rose at least 5 mm
           above its low point, comfortably above numerical jitter and
           well below any sensible pad lift distance.

        2. ``final_z < 0.5 m`` — the object did not fly to orbit.  A
           well-behaved grasp lands the object within a few cm of the
           pads' final position (≈ 56 mm for the default tennis-ball
           scene).  Anything above 0.5 m indicates catastrophic solver
           divergence: contact force overshoots, the PD launches the
           pad, and the object follows.  Without this upper bound,
           ``held=True`` is reported for cases where the object reached
           18 m or higher (seen in the F*=6N matched-F sweep with
           under-soft rigid-body contact pairs).
        """
        floor_threshold = 0.005   # 5 mm
        ceiling_threshold = 0.5   # 50 cm — well above any reasonable grasp height
        return (
            self.final_z > self.min_z + floor_threshold
            and self.final_z < ceiling_threshold
        )

    @property
    def xy_slip_max(self) -> float:
        """Largest XY drift [m] seen during the run."""
        return max(self.object_xy_drift) if self.object_xy_drift else 0.0
