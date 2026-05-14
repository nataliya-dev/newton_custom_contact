# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UXPBD Phase 4: Position-Based Fluids."""

import unittest

import numpy as np
import warp as wp

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


def test_uxpbd_add_fluid_grid_creates_phase_and_particles(test, device):
    """add_fluid_grid creates a fluid phase, particles tagged substrate=3."""
    builder = newton.ModelBuilder()
    phase_id = builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.01,
        cell_y=0.01,
        cell_z=0.01,
        particle_radius=0.005,
        rest_density=1000.0,
        viscosity=0.01,
        cohesion=0.01,
    )
    test.assertEqual(phase_id, 0)
    test.assertEqual(len(builder.particle_q), 8)
    test.assertEqual(len(builder.fluid_rest_density), 1)
    test.assertAlmostEqual(builder.fluid_rest_density[0], 1000.0)
    test.assertAlmostEqual(builder.fluid_viscosity[0], 0.01)
    test.assertAlmostEqual(builder.fluid_cohesion[0], 0.01)

    model = builder.finalize(device=device)
    substrate = model.particle_substrate.numpy()
    np.testing.assert_array_equal(substrate, [3] * 8)
    phase = model.particle_fluid_phase.numpy()
    np.testing.assert_array_equal(phase, [0] * 8)


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_add_fluid_grid_creates_phase_and_particles",
    test_uxpbd_add_fluid_grid_creates_phase_and_particles,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
