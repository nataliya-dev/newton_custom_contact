"""H3 Rung 2 — Inclined-plane direct friction minimal case.

Verifies that applying direct Coulomb friction forces (H3 mechanism) via
MuJoCo's qfrc_applied produces correct static/sliding behavior matching
Coulomb's law.

Setup:
  - Box (m=1 kg) on a horizontal ground plane.
  - Gravity tilted at θ to simulate an inclined plane: gx=g·sin(θ), gz=-g·cos(θ).
  - MuJoCo's own friction is disabled (friction="0 0 0") for the ground-box pair.
  - At each step, H3 direct friction is applied via data.qfrc_applied using
    a predictive (semi-implicit) Coulomb formulation:
        f_t = clamp(-m*(gx + vx/dt), -μ*fn, +μ*fn)
    Rationale: regularised explicit Coulomb causes limit cycling at dt=2 ms
    because the kinetic branch has no velocity damping; the predictive clamped
    form solves for the generalized force that would stop x-motion in one step,
    then clips it to the Coulomb bound.  This gives exact static equilibrium
    (vx → 0 to machine precision) with correct kinetic-sliding above θ_crit.
    Note: xfrc_applied is silently ignored in MuJoCo ≥3.x; qfrc_applied[dof]
    is the correct API for injecting generalized forces.

Analytical Coulomb threshold:
  θ_crit = arctan(μ) = arctan(0.5) ≈ 26.57°

Pass criteria (from experiment design):
  θ=20° (well below critical): HoldCreep ≤ 0.5 mm/s
  θ=25° (just below critical): HoldCreep ≤ 1.0 mm/s
  θ=30° (above critical):      sliding accel within ±5% of g·(sin θ − μ·cos θ)

Run with:
    uv run --extra dev -m unittest cslc_v1.test_h3_inclined -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

_G = 9.81   # m/s²
_MASS = 1.0  # kg
_MU = 0.5   # Coulomb friction coefficient
_DT = 0.002  # s — matches lift_test timestep
_HALF = 0.05  # box half-size [m]

_THETA_CRIT = math.degrees(math.atan(_MU))  # ≈ 26.57°


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xml(theta_deg: float, dt: float) -> str:
    """Minimal MJCF: unit-mass box constrained to move in x and z only.

    Uses two nested prismatic joints (z then x) so the box can:
      - Fall in z and contact the ground (z-joint active, contact normal develops).
      - Slide in x (x-joint, driven by gx = g·sin θ).
    Rotations and y-motion are fully constrained, avoiding tipping.

    Gravity is tilted to simulate inclined plane: gx = g·sin θ, gz = -g·cos θ.
    MuJoCo friction is zeroed on the contact pair; H3 friction applied via qfrc_applied.
    """
    theta = math.radians(theta_deg)
    gx = _G * math.sin(theta)
    gz = -_G * math.cos(theta)
    z0 = _HALF + 0.002  # 2 mm gap so box settles cleanly
    return f"""<mujoco model="inclined">
  <option timestep="{dt:.8f}" gravity="{gx:.8f} 0 {gz:.8f}"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <!-- z_carrier: can fall in z to establish ground contact.
         Tiny mass (0.1% of box) so MuJoCo accepts it as a moving body. -->
    <body name="z_carrier" pos="0 0 {z0:.8f}">
      <inertial pos="0 0 0" mass="0.001"
                diaginertia="1e-6 1e-6 1e-6"/>
      <joint name="z_joint" type="slide" axis="0 0 1"/>
      <!-- box: constrained inside z_carrier, free to slide in x -->
      <body name="box" pos="0 0 0">
        <joint name="x_joint" type="slide" axis="1 0 0"/>
        <geom name="box_g" type="box" size="{_HALF} {_HALF} {_HALF}"
              mass="{_MASS}" friction="0 0 0"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <pair geom1="ground" geom2="box_g" friction="0 0 0 0 0"/>
  </contact>
</mujoco>"""


def _run_inclined_h3(
    theta_deg: float,
    n_settle: int = 300,
    n_measure: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inclined-plane sim with H3 direct friction; return (pos_x, vel_x) traces.

    Steps:
      1. Settle for n_settle steps (friction active, let box reach steady state).
      2. Record pos_x and vel_x for n_measure steps.

    Returns arrays of length n_measure.
    """
    model = mujoco.MjModel.from_xml_string(_xml(theta_deg, _DT))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    theta = math.radians(theta_deg)
    f_n = _MASS * _G * math.cos(theta)    # analytical normal force [N]
    gx = _G * math.sin(theta)              # tangential gravity component [m/s²]
    f_coulomb = _MU * f_n                  # Coulomb limit [N]

    # x_joint DOF index = 1 (qvel[1] = vx of box in world frame).
    # MuJoCo ≥3.x: xfrc_applied is silently ignored; use qfrc_applied[dof] directly.
    x_dof = 1

    def apply_h3_friction():
        """Predictive Coulomb friction injected as a generalised force.

        Solves for the qfrc that holds vx = 0 next step (semi-implicit Euler),
        then clamps to the Coulomb bound.  Prevents limit-cycling that afflicts
        the explicit regularised form at dt = 2 ms.
        """
        vx = float(data.qvel[1])
        f_needed = -_MASS * (gx + vx / _DT)
        ft_x = max(-f_coulomb, min(f_coulomb, f_needed))
        data.qfrc_applied[x_dof] = ft_x

    total = n_settle + n_measure
    pos_x = np.zeros(n_measure)
    vel_x = np.zeros(n_measure)

    for i in range(total):
        apply_h3_friction()
        mujoco.mj_step(model, data)
        if i >= n_settle:
            j = i - n_settle
            pos_x[j] = float(data.qpos[1])   # x_joint displacement
            vel_x[j] = float(data.qvel[1])   # x_joint velocity

    return pos_x, vel_x


def _sliding_accel(theta_deg: float, mu: float = _MU) -> float:
    """Analytical sliding acceleration on inclined plane [m/s²]."""
    t = math.radians(theta_deg)
    return _G * (math.sin(t) - mu * math.cos(t))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_MUJOCO, "mujoco not installed")
class TestRung2InclinedAnalytical(unittest.TestCase):
    """Analytical reference tests (no MuJoCo)."""

    def test_critical_angle(self):
        # arctan(0.5) ≈ 26.57°
        self.assertAlmostEqual(_THETA_CRIT, 26.565, delta=0.01,
            msg=f"θ_crit={_THETA_CRIT:.3f}°, expected ≈26.57°")

    def test_20deg_static_regime(self):
        # tan(20°) = 0.364 < μ=0.5 → static
        self.assertLess(math.tan(math.radians(20.0)), _MU,
            msg="20° should be below critical angle")

    def test_25deg_static_regime(self):
        # tan(25°) = 0.466 < μ=0.5 → static
        self.assertLess(math.tan(math.radians(25.0)), _MU,
            msg="25° should be below critical angle")

    def test_30deg_sliding_regime(self):
        # tan(30°) = 0.577 > μ=0.5 → sliding
        self.assertGreater(math.tan(math.radians(30.0)), _MU,
            msg="30° should be above critical angle")
        a = _sliding_accel(30.0)
        self.assertGreater(a, 0.0,
            msg=f"Expected positive sliding acceleration at 30°, got {a:.4f}")
        # Numerical check: a = 9.81*(sin30° - 0.5*cos30°) = 9.81*(0.5 - 0.433) = 0.657
        self.assertAlmostEqual(a, 0.657, delta=0.005,
            msg=f"Sliding accel at 30°: {a:.4f} m/s², expected ≈0.657")


@unittest.skipUnless(HAS_MUJOCO, "mujoco not installed")
class TestRung2InclinedMuJoCo(unittest.TestCase):
    """MuJoCo simulation tests for H3 direct friction on inclined plane."""

    def test_no_nan_or_instability(self):
        # Simulation must not produce NaN or explosive values.
        for theta in [20.0, 25.0, 30.0]:
            pos, vel = _run_inclined_h3(theta)
            self.assertFalse(np.any(np.isnan(pos)) or np.any(np.isnan(vel)),
                msg=f"NaN at θ={theta}°")
            self.assertFalse(np.any(np.abs(vel) > 50.0),
                msg=f"Explosive velocity at θ={theta}°: max|v|={np.max(np.abs(vel)):.2f}")

    def test_20deg_static(self):
        # Well below critical: H3 friction holds the box static.
        # Pass criterion: |v_x| < 0.5 mm/s at steady state.
        _, vel = _run_inclined_h3(20.0)
        creep_mms = float(np.mean(np.abs(vel[-50:]))) * 1e3  # last 50 steps, mm/s
        print(f"\n  θ=20°  mean|v_x|={creep_mms:.4f} mm/s  (threshold 0.5 mm/s)")
        self.assertLess(creep_mms, 0.5,
            msg=f"θ=20° HoldCreep={creep_mms:.4f} mm/s > 0.5 mm/s threshold")

    def test_25deg_static(self):
        # Just below critical: H3 friction holds the box nearly static.
        # Pass criterion: |v_x| < 1.0 mm/s.
        _, vel = _run_inclined_h3(25.0)
        creep_mms = float(np.mean(np.abs(vel[-50:]))) * 1e3
        print(f"\n  θ=25°  mean|v_x|={creep_mms:.4f} mm/s  (threshold 1.0 mm/s)")
        self.assertLess(creep_mms, 1.0,
            msg=f"θ=25° HoldCreep={creep_mms:.4f} mm/s > 1.0 mm/s threshold")

    def test_30deg_sliding_acceleration(self):
        # Above critical: box slides at predicted acceleration ±5%.
        # Measure acceleration from position second-derivative.
        pos, vel = _run_inclined_h3(30.0, n_settle=100, n_measure=400)
        # Use last 300 samples (steady sliding regime).
        # a = Δv / Δt: fit linear trend to vel vs time.
        t = np.arange(len(vel)) * _DT
        coeffs = np.polyfit(t[-300:], vel[-300:], 1)
        a_measured = float(coeffs[0])   # slope = acceleration [m/s²]

        a_expected = _sliding_accel(30.0)
        err_frac = abs(a_measured - a_expected) / a_expected

        print(f"\n  θ=30°  a_measured={a_measured:.4f} m/s²  "
              f"a_expected={a_expected:.4f} m/s²  err={err_frac*100:.1f}%")

        self.assertLess(err_frac, 0.05,
            msg=f"θ=30° accel error {err_frac*100:.1f}% > 5% tolerance. "
                f"measured={a_measured:.4f}, expected={a_expected:.4f}")

    def test_30deg_sliding_positive_velocity(self):
        # Above critical: velocity must be positive (sliding downhill in +x direction).
        _, vel = _run_inclined_h3(30.0)
        mean_vel = float(np.mean(vel[-50:]))
        print(f"\n  θ=30°  mean v_x={mean_vel:.4f} m/s  (expected > 0)")
        self.assertGreater(mean_vel, 0.0,
            msg=f"θ=30°: box moving in wrong direction, v_x={mean_vel:.4f}")

    def test_creep_increases_with_angle_below_critical(self):
        # Static creep should increase with angle (both stay below thresholds).
        _, v20 = _run_inclined_h3(20.0)
        _, v25 = _run_inclined_h3(25.0)
        c20 = float(np.mean(np.abs(v20[-50:])))
        c25 = float(np.mean(np.abs(v25[-50:])))
        print(f"\n  creep: 20°={c20*1e3:.4f} mm/s  25°={c25*1e3:.4f} mm/s")
        self.assertLessEqual(c20, c25 + 1e-7,
            msg=f"Expected creep(20°) ≤ creep(25°): {c20:.2e} vs {c25:.2e}")


if __name__ == "__main__":
    unittest.main()
