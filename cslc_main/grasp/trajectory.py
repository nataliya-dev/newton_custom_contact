# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pad PD-position trajectories for APPROACH / SQUEEZE / LIFT / HOLD.

Both pads track the same ``(dx, dz)`` target each step; left pad sees
``+dx, +dz``, right pad sees ``-dx, +dz``.  ``dx`` is the inward travel
from each pad's spawn position; ``dz`` is the vertical lift.

The LIFT phase uses a C¹-smooth velocity ramp at both endpoints
(smoothstep up, constant, smoothstep down) so the pad's commanded
velocity eases from 0 → lift_speed → 0 without discontinuity.  Without
this, the high-gain PD drive turns the 0 → lift_speed step into an
impulsive pad velocity that saturates friction (μ·Fn) against the
stationary held object and launches it ballistically.
"""

from __future__ import annotations

import warp as wp

from .params import GraspConfig, TimingParams


# ── Velocity ramp helper ────────────────────────────────────────────────


def _lift_dz(t_lift: float, t: TimingParams) -> float:
    """Position [m] of the pad at local lift-time ``t_lift`` [s].

    Velocity profile: smoothstep over [0, ramp], constant over
    [ramp, T-ramp], smoothstep over [T-ramp, T] (where T = lift_duration,
    ramp = lift_ramp_duration).  The position is the integral of that
    velocity, renormalised so the total travel matches what the
    no-ramp trajectory would deliver in the same total duration —
    keeps lift_speed the user-facing average velocity.
    """
    ramp = t.lift_ramp_duration
    T = t.lift_duration
    if ramp <= 0.0:
        return t.lift_speed * t_lift
    # Two-ramp travel = lift_speed * (T - ramp).  Renormalise so it
    # equals the single-ramp design distance, lift_speed * (T - ramp/2).
    v_eff = t.lift_speed * (T - 0.5 * ramp) / max(T - ramp, 1e-9)

    # Phase 1: smoothstep up.
    if t_lift < ramp:
        sn = t_lift / ramp
        return v_eff * ramp * (sn**3 - 0.5 * sn**4)
    # Phase 2: constant velocity.
    if t_lift < T - ramp:
        return v_eff * ramp * 0.5 + v_eff * (t_lift - ramp)
    # Phase 3: smoothstep down — mirror of phase 1.
    s_back = max((T - t_lift) / ramp, 0.0)
    z_phase2_end = v_eff * (ramp * 0.5) + v_eff * (T - 2.0 * ramp)
    z_phase3 = v_eff * ramp * (0.5 - (s_back**3 - 0.5 * s_back**4))
    return z_phase2_end + z_phase3


# ── Public API ──────────────────────────────────────────────────────────


def pad_state(step: int, config: GraspConfig) -> tuple[float, float]:
    """Return ``(dx_inward, dz_up)`` for the current step.

    ``dx_inward`` is the inward travel of each pad (positive moves
    toward the object), ``dz_up`` is the vertical translation (positive
    rises).  Caller is expected to apply ``+dx`` to the left pad's
    X-target and ``-dx`` to the right pad's X-target.
    """
    phase, s_local = config.phase_of(step)
    t = config.timing
    t_local = s_local * t.dt

    if phase == "APPROACH":
        return t.approach_speed * t_local, 0.0

    dx_app = t.approach_speed * t.approach_duration
    if phase == "SQUEEZE":
        return dx_app + t.squeeze_speed * t_local, 0.0

    dx_total = dx_app + t.squeeze_speed * t.squeeze_duration

    if phase == "LIFT":
        return dx_total, _lift_dz(t_local, t)

    # HOLD: freeze at the end-of-LIFT position.
    return dx_total, _lift_dz(t.lift_duration, t)


def set_pad_targets(
    control,
    step: int,
    config: GraspConfig,
    dof_map: dict[str, int],
) -> tuple[float, float]:
    """Write joint position targets for both pads; return ``(dx, dz)``."""
    dx, dz = pad_state(step, config)

    target = control.joint_target_pos.numpy()
    target[dof_map["left_x"]] = +dx
    target[dof_map["left_z"]] = +dz
    target[dof_map["right_x"]] = -dx
    target[dof_map["right_z"]] = +dz
    control.joint_target_pos.assign(
        wp.array(target, dtype=wp.float32, device=control.joint_target_pos.device)
    )
    return dx, dz
