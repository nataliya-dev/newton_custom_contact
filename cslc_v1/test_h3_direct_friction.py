"""H3 Rung 1 — Direct body-force friction math unit tests.

Pure Python / NumPy only: no GPU, no MuJoCo, no Warp.

H3 hypothesis: bypass MuJoCo's Anitescu friction constraint for CSLC contacts
and instead apply per-sphere tangential wrenches directly as body_f external
forces. This eliminates the Anitescu friction velocity gap entirely.

Run with:
    uv run --extra dev -m unittest cslc_v1.test_h3_direct_friction -v
"""

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def compute_friction_force(
    f_n: float,
    v_t: np.ndarray,
    mu: float,
    gate: float = 1.0,
) -> np.ndarray:
    """Per-sphere tangential friction force (max-dissipation, Coulomb bound).

    f_t = -gate * mu * f_n * v_t / |v_t|   when |v_t| > eps and f_n > 0
    f_t = 0                                  otherwise

    Args:
        f_n: Normal contact force magnitude [N]. Positive = compression.
        v_t: 3-D tangential relative velocity [m/s]. Perpendicular to contact normal.
        mu: Coulomb friction coefficient [dimensionless].
        gate: Smooth contact gate ∈ [0, 1].

    Returns:
        3-D friction force vector [N].
    """
    v_mag = float(np.linalg.norm(v_t))
    if v_mag < 1e-12 or f_n <= 0.0 or gate <= 0.0:
        return np.zeros(3)
    return -gate * mu * f_n * (v_t / v_mag)


def aggregate_wrench(
    positions: np.ndarray,
    forces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-sphere forces into a body wrench.

    Args:
        positions: (N, 3) sphere positions relative to body CoM [m].
        forces:    (N, 3) force vectors applied at each sphere [N].

    Returns:
        (F_total [N], tau_total [N·m])
    """
    F_total = forces.sum(axis=0)
    tau_total = np.sum(np.cross(positions, forces), axis=0)
    return F_total, tau_total


def tangential_component(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Project velocity onto the tangential plane of contact normal n."""
    n_unit = n / (np.linalg.norm(n) + 1e-15)
    return v - np.dot(v, n_unit) * n_unit


# ---------------------------------------------------------------------------
# Rung 1 tests — analytical / math only
# ---------------------------------------------------------------------------

class TestRung1WrenchSummation(unittest.TestCase):
    """Test 1: Σ_i (f_i, r_i × f_i) = (F_total, τ_total) within numerical precision."""

    def test_single_sphere_wrench(self):
        # Single sphere at (0.01, 0, 0), force (0, 0, 1) N.
        # Torque = r × f = (0.01, 0, 0) × (0, 0, 1) = (0·1-0·0, 0·0-0.01·1, 0.01·0-0·0)
        #                = (0, -0.01, 0)
        pos = np.array([[0.01, 0.0, 0.0]])
        frc = np.array([[0.0, 0.0, 1.0]])
        F, tau = aggregate_wrench(pos, frc)
        np.testing.assert_allclose(F, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(tau, [0.0, -0.01, 0.0], atol=1e-12)

    def test_four_sphere_symmetry(self):
        # 4 spheres at corners of a square in the XY plane, each with identical
        # downward force (0, 0, -F_n). By symmetry: F_total = (0, 0, -4·F_n),
        # τ_total = 0 (torques cancel by 4-fold symmetry).
        r = 0.02
        F_n = 5.0
        positions = np.array([[r, r, 0], [-r, r, 0], [-r, -r, 0], [r, -r, 0]])
        forces = np.tile([0.0, 0.0, -F_n], (4, 1))
        F, tau = aggregate_wrench(positions, forces)
        np.testing.assert_allclose(F, [0.0, 0.0, -4 * F_n], atol=1e-12)
        np.testing.assert_allclose(tau, [0.0, 0.0, 0.0], atol=1e-12)

    def test_wrench_linearity(self):
        # Scaling all forces by scalar s scales the wrench by s.
        rng = np.random.default_rng(42)
        pos = rng.standard_normal((8, 3))
        frc = rng.standard_normal((8, 3))
        s = 3.7
        F1, tau1 = aggregate_wrench(pos, frc)
        F2, tau2 = aggregate_wrench(pos, frc * s)
        np.testing.assert_allclose(F2, F1 * s, rtol=1e-12)
        np.testing.assert_allclose(tau2, tau1 * s, rtol=1e-12)

    def test_wrench_additivity(self):
        # Two disjoint sets of spheres: total wrench = sum of partial wrenches.
        rng = np.random.default_rng(43)
        pos = rng.standard_normal((12, 3))
        frc = rng.standard_normal((12, 3))
        F_all, tau_all = aggregate_wrench(pos, frc)
        F_a, tau_a = aggregate_wrench(pos[:6], frc[:6])
        F_b, tau_b = aggregate_wrench(pos[6:], frc[6:])
        np.testing.assert_allclose(F_all, F_a + F_b, atol=1e-12)
        np.testing.assert_allclose(tau_all, tau_a + tau_b, atol=1e-12)

    def test_analytical_torque_random(self):
        # Spot-check 100 random (pos, force) pairs against numpy cross product.
        rng = np.random.default_rng(44)
        for _ in range(100):
            N = rng.integers(2, 20)
            pos = rng.standard_normal((N, 3))
            frc = rng.standard_normal((N, 3))
            F, tau = aggregate_wrench(pos, frc)
            tau_ref = np.zeros(3)
            for p, f in zip(pos, frc):
                tau_ref += np.cross(p, f)
            np.testing.assert_allclose(tau, tau_ref, atol=1e-12)


class TestRung1MDPPreservation(unittest.TestCase):
    """Test 2: f_t opposes v_t (max dissipation principle holds per-sphere)."""

    def test_friction_opposes_velocity(self):
        # For any v_t ≠ 0, dot(f_t, v_t) < 0 (strictly dissipative).
        rng = np.random.default_rng(42)
        mu = 0.5
        f_n = 10.0
        for _ in range(200):
            v_t = rng.standard_normal(3)
            v_t_mag = np.linalg.norm(v_t)
            if v_t_mag < 1e-10:
                continue
            f_t = compute_friction_force(f_n, v_t, mu)
            dot = float(np.dot(f_t, v_t))
            self.assertLess(dot, 0.0,
                msg=f"Friction not dissipative: dot(f_t, v_t)={dot:.6f}")

    def test_tangential_velocity_decomposition(self):
        # v_rel = v_n + v_t; after tangential extraction, v_t ⊥ n.
        rng = np.random.default_rng(43)
        for _ in range(100):
            v_rel = rng.standard_normal(3)
            n = rng.standard_normal(3)
            n = n / np.linalg.norm(n)
            v_t = tangential_component(v_rel, n)
            # v_t is perpendicular to n
            self.assertAlmostEqual(float(np.dot(v_t, n)), 0.0, places=12,
                msg="v_t not perpendicular to contact normal")
            # v_rel = v_t + v_n
            v_n = np.dot(v_rel, n) * n
            np.testing.assert_allclose(v_t + v_n, v_rel, atol=1e-12)

    def test_zero_velocity_zero_friction(self):
        # No tangential velocity → zero friction (static case — MDP trivially satisfied).
        f_t = compute_friction_force(10.0, np.zeros(3), mu=0.5)
        np.testing.assert_array_equal(f_t, [0.0, 0.0, 0.0])

    def test_friction_direction_is_exactly_opposite(self):
        # f_t should point exactly opposite to v_t (not just have negative dot product).
        v_t = np.array([3.0, 4.0, 0.0])  # unit: (0.6, 0.8, 0)
        f_t = compute_friction_force(f_n=10.0, v_t=v_t, mu=0.5)
        # f_t = -0.5 * 10 * (3,4,0)/5 = (-3, -4, 0)
        np.testing.assert_allclose(f_t, [-3.0, -4.0, 0.0], atol=1e-12)


class TestRung1CoulombCone(unittest.TestCase):
    """Test 3: |f_t| ≤ μ · |f_n| (Coulomb bound, at equality for |v_t| > 0)."""

    def test_friction_at_coulomb_bound(self):
        # When |v_t| > 0 and f_n > 0: |f_t| = mu * f_n exactly (sliding contact).
        rng = np.random.default_rng(42)
        mu = 0.5
        for _ in range(100):
            f_n = rng.uniform(0.1, 100.0)
            v_t = rng.standard_normal(3)
            if np.linalg.norm(v_t) < 1e-10:
                continue
            f_t = compute_friction_force(f_n, v_t, mu)
            f_t_mag = float(np.linalg.norm(f_t))
            self.assertAlmostEqual(f_t_mag, mu * f_n, places=10,
                msg=f"|f_t|={f_t_mag:.6f} ≠ mu*f_n={mu*f_n:.6f}")

    def test_no_friction_from_tensile_contact(self):
        # f_n ≤ 0 (tensile or no contact): friction is zero.
        for f_n in [-1.0, 0.0, -100.0]:
            v_t = np.array([1.0, 0.0, 0.0])
            f_t = compute_friction_force(f_n, v_t, mu=0.5)
            np.testing.assert_array_equal(f_t, [0.0, 0.0, 0.0],
                err_msg=f"Expected zero friction for f_n={f_n}")

    def test_coulomb_bound_never_exceeded(self):
        # |f_t| ≤ mu * f_n for all valid inputs.
        rng = np.random.default_rng(43)
        mu = 0.7
        for _ in range(500):
            f_n = rng.uniform(0.0, 200.0)
            v_t = rng.standard_normal(3)
            gate = rng.uniform(0.0, 1.0)
            f_t = compute_friction_force(f_n, v_t, mu, gate)
            f_t_mag = float(np.linalg.norm(f_t))
            self.assertLessEqual(f_t_mag, mu * f_n + 1e-12,
                msg=f"|f_t|={f_t_mag:.6f} > mu*f_n={mu*f_n:.6f} with gate={gate:.3f}")


class TestRung1GateContinuity(unittest.TestCase):
    """Test 4: gate → 0 smoothly drives f_t → 0; no discontinuity."""

    def test_gate_zero_gives_zero_force(self):
        # gate=0 → f_t = 0 regardless of v_t and f_n.
        f_t = compute_friction_force(f_n=50.0, v_t=np.array([1.0, 2.0, 3.0]),
                                     mu=0.5, gate=0.0)
        np.testing.assert_array_equal(f_t, [0.0, 0.0, 0.0])

    def test_gate_one_gives_full_force(self):
        # gate=1 → f_t = mu * f_n * (-v_t/|v_t|).
        v_t = np.array([1.0, 0.0, 0.0])
        f_t = compute_friction_force(f_n=10.0, v_t=v_t, mu=0.5, gate=1.0)
        np.testing.assert_allclose(f_t, [-5.0, 0.0, 0.0], atol=1e-12)

    def test_gate_linear_scaling(self):
        # f_t scales linearly with gate: f_t(gate=s) = s · f_t(gate=1).
        v_t = np.array([0.6, 0.8, 0.0])
        f_n = 20.0
        mu = 0.4
        f_full = compute_friction_force(f_n, v_t, mu, gate=1.0)
        for s in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            f_s = compute_friction_force(f_n, v_t, mu, gate=s)
            np.testing.assert_allclose(f_s, s * f_full, atol=1e-12,
                err_msg=f"Gate scaling failed at gate={s}")

    def test_gate_continuity_near_zero(self):
        # |f_t| is monotonically non-increasing as gate decreases from 1 → 0.
        v_t = np.array([1.0, 0.0, 0.0])
        f_n = 10.0
        mu = 0.5
        gates = [1.0, 0.5, 0.1, 1e-3, 1e-6, 1e-9, 1e-12, 0.0]
        mags = [float(np.linalg.norm(compute_friction_force(f_n, v_t, mu, g)))
                for g in gates]
        for i in range(len(mags) - 1):
            self.assertGreaterEqual(mags[i], mags[i + 1] - 1e-14,
                msg=f"|f_t| not non-increasing: {mags[i]:.2e} < {mags[i+1]:.2e} "
                    f"(gate {gates[i]} → {gates[i+1]})")
        # At gate=0, exactly zero.
        np.testing.assert_array_equal(
            compute_friction_force(f_n, v_t, mu, gate=0.0),
            [0.0, 0.0, 0.0])

    def test_gate_continuity_near_one(self):
        # f_t(gate=1-ε) converges to f_t(gate=1) as ε → 0.
        v_t = np.array([0.0, 1.0, 0.0])
        f_n = 8.0
        mu = 0.6
        f_full = compute_friction_force(f_n, v_t, mu, gate=1.0)
        for eps in [1e-1, 1e-3, 1e-6]:
            f_near = compute_friction_force(f_n, v_t, mu, gate=1.0 - eps)
            delta = float(np.linalg.norm(f_near - f_full))
            self.assertLess(delta, mu * f_n * eps * 2,
                msg=f"Gate not continuous near 1: delta={delta:.2e} at eps={eps}")


if __name__ == "__main__":
    unittest.main()
