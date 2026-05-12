# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 Rung 1 + Rung 2 — H1 (harmonic-mean compliance composition) tests.

Pure-Python math unit tests validating the harmonic-mean formula that
``newton/_src/geometry/cslc_kernels.py::write_cslc_contacts(_box)``
applies to compose per-constraint stiffness from lattice (``cslc_kc``)
and target (``shape_material_ke``) moduli. Equation per Masterjohn et
al. 2021 (PFC-V eq 23) and Castro et al. 2022 (SAP emission stage).

Run with:

    uv run --extra dev -m unittest cslc_v1.test_h1_compliance -v

Tests run without a GPU; they validate the analytical math, not the
kernel runtime. Kernel-level integration tests (against actual
sphere/box pair launches) are Phase 6 Rung 3 work and live separately.
"""

from __future__ import annotations

import math
import unittest


# Mirror of the kernel's harmonic-mean formula (cslc_kernels.py:531-543).
# Kept here as a pure-Python reference so tests can exercise the math
# without needing Warp / GPU initialization.
def harmonic_mean_stiffness(kc_lattice: float, ke_target: float, eps: float = 1.0e-5) -> float:
    """Series composition of two material stiffness moduli.

    Args:
        kc_lattice: lattice-side per-sphere effective stiffness [N/m].
        ke_target: target body's contact stiffness [N/m] (from
            ``shape_material_ke`` in Newton's Model).
        eps: smoothing width [m]; the ``eps²`` denominator floor guards
            against 0/0 when both stiffnesses are zero.

    Returns:
        Series-spring stiffness ``kc · ke / (kc + ke + eps²)`` [N/m].
    """
    return (kc_lattice * ke_target) / (kc_lattice + ke_target + eps * eps)


# Numerical derivative for the differentiability check.
def _numerical_derivative(f, x: float, h: float = 1.0e-4) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


class TestRung1HarmonicMeanMath(unittest.TestCase):
    """Rung 1 — math unit tests for the harmonic-mean composition."""

    def test_series_identity_equal_stiffness(self) -> None:
        """harmonic_mean(ke, ke) = 0.5·ke (exact, mod eps² floor)."""
        ke = 75000.0
        result = harmonic_mean_stiffness(ke, ke)
        # eps² floor ~ 1e-10 vs ke ~ 1e5 → relative effect ~ 1e-15
        self.assertAlmostEqual(result, 0.5 * ke, delta=1.0)
        self.assertAlmostEqual(result, 37500.0, delta=1.0)

    def test_rigid_limit(self) -> None:
        """As ke_target → ∞, harmonic_mean → cslc_kc.

        Leading-order error: H(kc, ke) ≈ kc - kc²/ke, so the absolute
        error scales as kc²/ke_target. Tolerance is set at 1.5× this
        leading-order term to allow for higher-order corrections.
        """
        kc = 75000.0
        for ke_target in (1.0e8, 1.0e10, 1.0e12):
            result = harmonic_mean_stiffness(kc, ke_target)
            expected_error = kc * kc / ke_target
            self.assertAlmostEqual(
                result, kc, delta=1.5 * expected_error,
                msg=f"At ke_target={ke_target:.0e}, harmonic mean is {result:.4f}, "
                f"expected ~{kc:.4f} (leading-order error budget {expected_error:.4f})",
            )

    def test_soft_limit(self) -> None:
        """harmonic_mean(kc, 0) = 0 (with eps² guard)."""
        kc = 75000.0
        result = harmonic_mean_stiffness(kc, 0.0)
        # numerator is 0; denominator is kc + eps² → result is exactly 0
        # within floating-point precision
        self.assertLess(abs(result), 1.0e-6,
                        f"Expected ~0, got {result:.3e}")

    def test_smooth_relu_compatibility(self) -> None:
        """smooth_relu(harmonic_mean) ≥ 0 for valid (kc, ke_target) pairs."""

        def smooth_relu(x: float, eps: float = 1.0e-9) -> float:
            return 0.5 * (x + math.sqrt(x * x + eps * eps))

        # Sweep both stiffnesses across the physically plausible range.
        kc_values = (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7)
        ke_target_values = (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7)
        for kc in kc_values:
            for ke_target in ke_target_values:
                result = harmonic_mean_stiffness(kc, ke_target)
                smoothed = smooth_relu(result)
                # smooth_relu of a non-negative input must be ≥ input/2,
                # i.e. trivially non-negative.
                self.assertGreaterEqual(
                    smoothed, 0.0,
                    f"smooth_relu(H({kc}, {ke_target})) = {smoothed:.3e}",
                )

    def test_differentiability_partial_kc(self) -> None:
        """∂/∂kc [H(kc, ke)] = ke² / (kc + ke + eps²)².

        Verified by finite-difference within 1e-4 relative tolerance.
        """
        ke_target = 50000.0
        eps = 1.0e-5
        # eps² is ~1e-10; for kc ~ 1e4 it's negligible vs (kc+ke)² ~ 1e10
        for kc in (1.0e3, 1.0e4, 5.0e4, 7.5e4, 1.0e5):
            analytic = (ke_target * ke_target) / (kc + ke_target + eps * eps) ** 2
            numeric = _numerical_derivative(
                lambda k: harmonic_mean_stiffness(k, ke_target, eps=eps), kc
            )
            rel_err = abs(analytic - numeric) / abs(analytic)
            self.assertLess(
                rel_err, 1.0e-4,
                msg=f"At kc={kc:.0f}, analytic={analytic:.6e}, numeric={numeric:.6e}, "
                f"rel_err={rel_err:.3e}",
            )


class TestRung2MinimalCaseSeriesSpring(unittest.TestCase):
    """Rung 2 — single-sphere series-spring equilibrium analytical check.

    Setup: one lattice sphere pressed against one target sphere with
    static force F along the normal.  Lattice stiffness kc_lattice,
    target stiffness ke_target (both compliant).  Series-spring
    equilibrium:

        F = k_series · δ_total
        δ_total = δ_lattice + δ_target = F · (1/kc + 1/ke)
        k_series = harmonic_mean(kc_lattice, ke_target)

    The MuJoCo constraint sees ``F = k_series · pen_3d`` where
    ``pen_3d = δ_total`` (the bodies' total approach distance).  H1
    correctly emits ``k_series`` so the constraint enforces the
    physically right force.
    """

    def test_equal_stiffness_split_evenly(self) -> None:
        """With equal stiffness, total compression splits evenly."""
        F = 5.0  # N
        kc = 75000.0  # N/m (lattice)
        ke = 75000.0  # N/m (target)

        k_series = harmonic_mean_stiffness(kc, ke)
        delta_total = F / k_series
        delta_lattice = F / kc
        delta_target = F / ke

        # Series-spring identity: total displacement = sum of per-spring
        # displacements (each spring sees the same force F).
        self.assertAlmostEqual(delta_total, delta_lattice + delta_target,
                               delta=1.0e-9)

        # Equal split: each spring takes half the total.
        self.assertAlmostEqual(delta_lattice, delta_total / 2, delta=1.0e-9)
        self.assertAlmostEqual(delta_target, delta_total / 2, delta=1.0e-9)

        # Numerical check against the experiment-design.md draft:
        # F=5 N, kc=ke=75000 → δ_total = 5 · 2/75000 = 133 µm
        # δ_each = 67 µm.
        self.assertAlmostEqual(delta_total, 133.33e-6, delta=1.0e-6)
        self.assertAlmostEqual(delta_lattice, 66.67e-6, delta=1.0e-6)

    def test_rigid_target_equivalent_to_lattice_alone(self) -> None:
        """With ke_target → ∞ (rigid), δ_total = F / kc_lattice.

        Leading-order relative error: δ_series / δ_lattice ≈ 1 + kc/ke_target,
        so tolerance scales as kc/ke_target with a 2× buffer for
        higher-order corrections.
        """
        F = 5.0
        kc = 75000.0
        for ke_target in (1.0e8, 1.0e10):
            k_series = harmonic_mean_stiffness(kc, ke_target)
            delta_total = F / k_series
            expected = F / kc
            rel_err = abs(delta_total - expected) / expected
            tol = 2.0 * kc / ke_target
            self.assertLess(
                rel_err, tol,
                msg=f"At ke_target={ke_target:.0e}, δ_total={delta_total*1e6:.4f} µm, "
                f"expected ~{expected*1e6:.4f} µm (rel_err {rel_err:.3e} vs tol {tol:.3e})",
            )

    def test_compliant_target_softer_than_lattice_alone(self) -> None:
        """With softer target, series compliance > lattice alone.

        Implication: in the existing CSLC tests where pad and target
        both have ke=5e4 and cslc_kc=75000, the H1-emitted constraint
        stiffness is k_series = 75000·50000/(125000) = 30000 N/m, vs
        the pre-H1 75000 N/m.  This is a 2.5× softer constraint —
        documenting the magnitude here as a reference point for the
        sphere lift Rung-3 expected re-calibration.
        """
        F = 5.0
        kc = 75000.0
        ke_target = 50000.0  # matches lift_test.py SceneParams.ke

        k_series = harmonic_mean_stiffness(kc, ke_target)
        # Verify the specific number we'd expect in the lift test scene.
        self.assertAlmostEqual(k_series, 30000.0, delta=1.0)

        # Series compliance is larger than lattice-alone compliance.
        delta_total_series = F / k_series
        delta_lattice_alone = F / kc
        self.assertGreater(delta_total_series, delta_lattice_alone)

        # Specifically: series compliance = sum of individual compliances.
        compliance_series = 1.0 / k_series
        compliance_lattice = 1.0 / kc
        compliance_target = 1.0 / ke_target
        self.assertAlmostEqual(
            compliance_series, compliance_lattice + compliance_target,
            delta=1.0e-9,
        )

    def test_force_at_equilibrium_matches_series_law(self) -> None:
        """Forward check: given δ_total, the emitted force matches series law.

        This is the property the MuJoCo constraint enforces under H1.
        F = k_series · pen_3d where pen_3d is the total approach
        distance the bodies have closed by.
        """
        kc = 75000.0
        ke_target = 50000.0
        delta_total = 100.0e-6  # 100 µm total approach

        k_series = harmonic_mean_stiffness(kc, ke_target)
        F_emitted = k_series * delta_total

        # Verify: each spring sees the same force, and each spring's
        # individual deflection sums to delta_total.
        delta_lattice_part = F_emitted / kc
        delta_target_part = F_emitted / ke_target
        self.assertAlmostEqual(
            delta_lattice_part + delta_target_part, delta_total,
            delta=1.0e-9,
        )


if __name__ == "__main__":
    unittest.main()
