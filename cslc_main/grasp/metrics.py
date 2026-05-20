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
    def held(self, threshold: float = 0.005) -> bool:
        """True if the object is still clearly airborne at the end.

        ``threshold`` is the minimum clearance [m] between object centre
        and the highest point during the run that we count as "still
        held".  Default 5 mm — well above numerical jitter, well below
        a typical pad lift distance.
        """
        return self.final_z > self.min_z + threshold

    @property
    def xy_slip_max(self) -> float:
        """Largest XY drift [m] seen during the run."""
        return max(self.object_xy_drift) if self.object_xy_drift else 0.0
