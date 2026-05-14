# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UXPBD Phase 4: Position-Based Fluids."""

import unittest

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_uxpbd_empty_model_has_zero_fluid(test, device):
    """A Model finalized with no fluid has fluid_phase_count == 0."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    test.assertEqual(getattr(model, "fluid_phase_count", 0), 0)
    test.assertEqual(model.fluid_rest_density.shape[0], 0)
    test.assertEqual(model.fluid_smoothing_radius.shape[0], 0)
    test.assertEqual(model.fluid_viscosity.shape[0], 0)
    test.assertEqual(model.fluid_cohesion.shape[0], 0)


class TestSolverUXPBDPhase4(unittest.TestCase):
    pass


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_empty_model_has_zero_fluid",
    test_uxpbd_empty_model_has_zero_fluid,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
