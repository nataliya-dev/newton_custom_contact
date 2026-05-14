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


def test_uxpbd_pbf_density_isolated_particle(test, device):
    """A fluid particle far from any neighbor has density = self-contribution.

    Self-contribution at r=0 is W(0, h) = 315/(64*pi*h^9) * h^6 = 315/(64*pi*h^3).
    For h=0.01, this should be ~1.79e7 * mass.
    """
    from newton._src.solvers.uxpbd.fluid import compute_fluid_density  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        particle_radius=0.005,
        rest_density=1000.0,
    )
    model = builder.finalize(device=device)
    state = model.state()

    # particle_grid is None when particle_count <= 1; build one manually.
    if model.particle_grid is None:
        model.particle_grid = wp.HashGrid(128, 128, 128)
    model.particle_grid.build(state.particle_q, model.particle_max_radius * 4.0)
    density_out = wp.zeros(model.particle_count, dtype=wp.float32, device=device)
    h = 2.0 * 0.005
    mass = 1000.0 * 1.0 * 1.0 * 1.0  # cell volume * rest density
    expected_density = mass * (315.0 / (64.0 * np.pi * h**3))

    wp.launch(
        kernel=compute_fluid_density,
        dim=model.particle_count,
        inputs=[
            model.particle_grid.id,
            state.particle_q,
            model.particle_mass,
            model.particle_substrate,
            model.particle_fluid_phase,
            model.fluid_smoothing_radius,
            model.fluid_solid_coupling_s,
        ],
        outputs=[density_out],
        device=device,
    )
    density = float(density_out.numpy()[0])
    test.assertAlmostEqual(
        density, expected_density, delta=0.01 * expected_density, msg=f"density {density}, expected {expected_density}"
    )


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_density_isolated_particle",
    test_uxpbd_pbf_density_isolated_particle,
    devices=get_test_devices(),
)


def test_uxpbd_pbf_lambda_at_rest_density(test, device):
    """When density == rest_density, lambda is 0 (unilateral)."""
    from newton._src.solvers.uxpbd.fluid import compute_fluid_lambda  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=1.0,
        cell_y=1.0,
        cell_z=1.0,
        particle_radius=0.005,
        rest_density=1000.0,
    )
    model = builder.finalize(device=device)
    state = model.state()
    if model.particle_grid is None:
        model.particle_grid = wp.HashGrid(128, 128, 128)

    n = model.particle_count
    density = wp.array([1000.0] * n, dtype=wp.float32, device=device)
    lambdas = wp.zeros(n, dtype=wp.float32, device=device)

    model.particle_grid.build(state.particle_q, model.particle_max_radius * 4.0)
    wp.launch(
        kernel=compute_fluid_lambda,
        dim=n,
        inputs=[
            model.particle_grid.id,
            state.particle_q,
            model.particle_mass,
            model.particle_substrate,
            model.particle_fluid_phase,
            model.fluid_rest_density,
            model.fluid_smoothing_radius,
            density,
            wp.float32(100.0),
        ],
        outputs=[lambdas],
        device=device,
    )
    test.assertAlmostEqual(float(lambdas.numpy()[0]), 0.0, delta=1e-6)


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_lambda_at_rest_density",
    test_uxpbd_pbf_lambda_at_rest_density,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
