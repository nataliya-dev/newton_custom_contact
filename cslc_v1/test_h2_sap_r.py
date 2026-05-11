"""H2 Rung 1 — SAP-R formula math unit tests.

Pure Python / NumPy only: no GPU, no MuJoCo, no Warp.

Run with:
    uv run --extra dev -m unittest cslc_v1.test_h2_sap_r -v
"""

import math
import unittest


# ---------------------------------------------------------------------------
# Reference implementations (Castro 2022 eq 19 / eq 29)
# ---------------------------------------------------------------------------

def sap_r_inv(dt: float, k: float, tau_d: float) -> float:
    """R_n^{-1} = dt * k * (dt + tau_d)  — Castro22 eq 19."""
    return dt * k * (dt + tau_d)


def sap_r_to_solref(r_n_inv: float, ke: float, imp: float = 0.9) -> tuple[float, float]:
    """Convert SAP R_n^{-1} to MuJoCo (timeconst, dampratio) preserving F = ke * pen.

    Derivation: MuJoCo uses  F = imp * pen / (tc^2 * dr^2),  so setting
        tc  = sqrt(imp / R_n_inv)
        dr  = sqrt(R_n_inv / ke)
    gives F = ke * pen identically.
    """
    tc = math.sqrt(imp / r_n_inv)
    dr = math.sqrt(r_n_inv / ke)
    return tc, dr


def sap_r_inv_with_baumgarte(dt: float, k: float, tau_d: float,
                              r_n_inv_baumgarte: float) -> float:
    """Castro22 eq 29: max over physical and Baumgarte floors."""
    return max(sap_r_inv(dt, k, tau_d), r_n_inv_baumgarte)


# ---------------------------------------------------------------------------
# Rung 1 tests — analytical / math only
# ---------------------------------------------------------------------------

class TestRung1CastroFormula(unittest.TestCase):
    """Test 1: Castro22 eq 19 evaluated at canonical operating point."""

    def test_canonical_operating_point(self):
        # dt=2 ms, k=37500 N/m (kc/2 for a 2-contact pair), tau_d=0
        # Expected: 0.002 * 37500 * 0.002 = 0.15
        result = sap_r_inv(dt=2e-3, k=37500.0, tau_d=0.0)
        self.assertAlmostEqual(result, 0.15, delta=1e-3,
                               msg=f"R_n_inv={result:.6f}, expected 0.15")

    def test_tau_d_increases_r_inv(self):
        # Damping tau_d > 0 makes R_n_inv larger (R smaller → less creep).
        base = sap_r_inv(dt=2e-3, k=37500.0, tau_d=0.0)
        damped = sap_r_inv(dt=2e-3, k=37500.0, tau_d=1e-3)
        self.assertGreater(damped, base)

    def test_linear_in_k(self):
        # Doubling k doubles R_n_inv.
        r1 = sap_r_inv(dt=2e-3, k=37500.0, tau_d=0.0)
        r2 = sap_r_inv(dt=2e-3, k=75000.0, tau_d=0.0)
        self.assertAlmostEqual(r2 / r1, 2.0, places=10)

    def test_quadratic_in_dt(self):
        # Doubling dt (with tau_d=0) quadruples R_n_inv.
        r1 = sap_r_inv(dt=1e-3, k=37500.0, tau_d=0.0)
        r2 = sap_r_inv(dt=2e-3, k=37500.0, tau_d=0.0)
        self.assertAlmostEqual(r2 / r1, 4.0, places=10)


class TestRung1RigidLimit(unittest.TestCase):
    """Test 2: as k → ∞, R_n^{-1} → ∞ ⇒ R_n → 0 (hard constraint recovered)."""

    def test_r_inv_grows_with_k(self):
        ks = [1e3, 1e6, 1e9, 1e12]
        r_invs = [sap_r_inv(dt=2e-3, k=k, tau_d=0.0) for k in ks]
        for i in range(len(r_invs) - 1):
            self.assertGreater(r_invs[i + 1], r_invs[i])

    def test_rigid_limit_r_n_to_zero(self):
        r_n_inv_rigid = sap_r_inv(dt=2e-3, k=1e12, tau_d=0.0)
        r_n = 1.0 / r_n_inv_rigid
        self.assertLess(r_n, 1e-6,
                        msg=f"R_n={r_n:.2e} should be < 1e-6 for rigid k=1e12")

    def test_soft_spring_r_n_nontrivial(self):
        # For a soft spring (k=100), R_n should be substantial.
        r_n_inv_soft = sap_r_inv(dt=2e-3, k=100.0, tau_d=0.0)
        r_n = 1.0 / r_n_inv_soft
        self.assertGreater(r_n, 0.1,
                           msg=f"R_n={r_n:.4f} should be > 0.1 for soft k=100")


class TestRung1MuJoCoConversion(unittest.TestCase):
    """Test 3: (tc, dampratio) from SAP-R preserves F = ke * pen within ±0.5%."""

    def _ke_recovered(self, ke: float, dt: float, tau_d: float, imp: float = 0.9) -> float:
        r_n_inv = sap_r_inv(dt=dt, k=ke, tau_d=tau_d)
        tc, dr = sap_r_to_solref(r_n_inv, ke, imp=imp)
        # MuJoCo spring stiffness: F = imp * pen / (tc^2 * dr^2)
        return imp / (tc ** 2 * dr ** 2)

    def test_ke_preserved_canonical(self):
        ke = 37500.0
        ke_ref = self._ke_recovered(ke, dt=2e-3, tau_d=0.0)
        err_frac = abs(ke_ref - ke) / ke
        self.assertLess(err_frac, 0.005,
                        msg=f"ke_ref={ke_ref:.1f} vs ke={ke:.1f}, {err_frac*100:.3f}%")

    def test_ke_preserved_with_damping(self):
        ke = 75000.0
        ke_ref = self._ke_recovered(ke, dt=2e-3, tau_d=1e-3)
        err_frac = abs(ke_ref - ke) / ke
        self.assertLess(err_frac, 0.005,
                        msg=f"ke_ref={ke_ref:.1f} vs ke={ke:.1f}, {err_frac*100:.3f}%")

    def test_tc_positive_and_finite(self):
        r_n_inv = sap_r_inv(dt=2e-3, k=37500.0, tau_d=0.0)
        tc, dr = sap_r_to_solref(r_n_inv, ke=37500.0)
        self.assertGreater(tc, 0.0)
        self.assertFalse(math.isinf(tc))
        self.assertGreater(dr, 0.0)
        self.assertFalse(math.isinf(dr))

    def test_larger_r_inv_gives_smaller_tc(self):
        # Stiffer contact → larger R_n_inv → smaller timeconst (faster recovery).
        r1 = sap_r_inv(dt=2e-3, k=37500.0, tau_d=0.0)
        r2 = sap_r_inv(dt=2e-3, k=75000.0, tau_d=0.0)
        tc1, _ = sap_r_to_solref(r1, ke=37500.0)
        tc2, _ = sap_r_to_solref(r2, ke=75000.0)
        self.assertLess(tc2, tc1)


class TestRung1NearRigidSwitch(unittest.TestCase):
    """Test 4: Castro eq 29 Baumgarte floor — smooth transition, monotone in k."""

    # Baumgarte floor corresponds to MuJoCo default timeconst ≈ 0.02 s, imp=0.9
    # => R_n_inv_baumgarte = imp / tc^2 * 1/(dr^2) with dr~1 => ≈ 0.9/4e-4 = 2250
    _R_BAUMGARTE = 2250.0

    def _r_inv_final(self, k: float) -> float:
        return sap_r_inv_with_baumgarte(
            dt=2e-3, k=k, tau_d=0.0,
            r_n_inv_baumgarte=self._R_BAUMGARTE,
        )

    def test_soft_spring_hits_baumgarte_floor(self):
        # Very soft spring: physical R_n_inv << Baumgarte → floor dominates.
        r_inv = self._r_inv_final(k=10.0)
        self.assertAlmostEqual(r_inv, self._R_BAUMGARTE, delta=1.0,
                               msg="Soft spring should hit Baumgarte floor")

    def test_stiff_spring_above_floor(self):
        # Stiff spring: physical R_n_inv >> Baumgarte → physical dominates.
        # k_cross ≈ R_BAUMGARTE / dt^2 = 2250 / 4e-6 ≈ 5.6e8; use 100× above that.
        k_rigid = 5e10
        r_inv_physical = sap_r_inv(dt=2e-3, k=k_rigid, tau_d=0.0)
        r_inv_final = self._r_inv_final(k=k_rigid)
        self.assertAlmostEqual(r_inv_final, r_inv_physical, delta=1.0,
                               msg="Stiff spring should use physical R_n_inv")

    def test_monotone_in_k(self):
        ks = [10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
        r_invs = [self._r_inv_final(k) for k in ks]
        for i in range(len(r_invs) - 1):
            self.assertGreaterEqual(r_invs[i + 1], r_invs[i],
                                    msg=f"R_n_inv not monotone at k={ks[i+1]:.0e}")

    def test_no_discontinuity_at_crossover(self):
        # Find approximate crossover: physical ≈ baumgarte
        # physical = dt^2 * k = 4e-6 * k = 2250 => k_cross ≈ 562500
        k_cross = self._R_BAUMGARTE / (4e-6)
        delta_k = k_cross * 0.01  # 1% perturbation
        r_below = self._r_inv_final(k=k_cross - delta_k)
        r_above = self._r_inv_final(k=k_cross + delta_k)
        # The jump should be small relative to the value — no discontinuity.
        relative_jump = abs(r_above - r_below) / max(r_below, 1e-12)
        self.assertLess(relative_jump, 0.05,
                        msg=f"Discontinuity at crossover: Δ={relative_jump*100:.1f}%")


if __name__ == "__main__":
    unittest.main()
