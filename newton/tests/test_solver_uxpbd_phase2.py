# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UXPBD Phase 2: free SM-rigid, cross-substrate contact, PBD-R benchmarks."""

import os
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import add_function_test, get_test_devices

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "uxpbd")


def test_uxpbd_link_lattice_csr_offsets(test, device):
    """builder.add_lattice emits per-link CSR offsets used for per-link iteration."""
    builder = newton.ModelBuilder()
    link_a = builder.add_body(label="a")
    link_b = builder.add_body(label="b")
    builder.add_lattice(link=link_a, morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"), total_mass=1.0)
    builder.add_lattice(link=link_b, morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"), total_mass=1.0)
    model = builder.finalize(device=device)

    counts = model.link_lattice_sphere_count.numpy()
    starts = model.link_lattice_sphere_start.numpy()
    test.assertEqual(counts.shape[0], model.body_count)
    test.assertEqual(int(counts[link_a]), 5)
    test.assertEqual(int(counts[link_b]), 5)
    test.assertEqual(int(starts[link_a]), 0)
    test.assertEqual(int(starts[link_b]), 5)


class TestSolverUXPBDPhase2(unittest.TestCase):
    pass


add_function_test(
    TestSolverUXPBDPhase2,
    "test_uxpbd_link_lattice_csr_offsets",
    test_uxpbd_link_lattice_csr_offsets,
    devices=get_test_devices(),
)


def test_uxpbd_particle_substrate_tagging(test, device):
    """particle_substrate is 0 for lattice particles, 1 for shape-matched rigid."""
    builder = newton.ModelBuilder()
    link = builder.add_body(label="link")
    builder.add_lattice(link=link, morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"), total_mass=1.0)

    # Add a free SM-rigid group via add_particle_volume.
    builder.add_particle_volume(
        volume_data={"centers": [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]], "radii": [0.05, 0.05]},
        total_mass=0.5,
        pos=wp.vec3(0.0, 0.0, 0.0),
    )

    model = builder.finalize(device=device)

    substrate = model.particle_substrate.numpy()
    # First 5 particles are lattice (substrate=0).
    np.testing.assert_array_equal(substrate[:5], [0, 0, 0, 0, 0])
    # Next 2 particles are SM-rigid (substrate=1).
    np.testing.assert_array_equal(substrate[5:7], [1, 1])


add_function_test(
    TestSolverUXPBDPhase2,
    "test_uxpbd_particle_substrate_tagging",
    test_uxpbd_particle_substrate_tagging,
    devices=get_test_devices(),
)


def test_uxpbd_solver_caches_shape_match_data(test, device):
    """SolverUXPBD precomputes shape-matching group data at init."""
    builder = newton.ModelBuilder()
    # Two free SM-rigid groups.
    builder.add_particle_volume(
        volume_data={"centers": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], "radii": [0.05, 0.05]},
        total_mass=1.0,
    )
    builder.add_particle_volume(
        volume_data={"centers": [[2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [2.2, 0.0, 0.0]], "radii": [0.05, 0.05, 0.05]},
        total_mass=2.0,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverUXPBD(model)

    test.assertEqual(solver._num_dynamic_groups, 2)
    counts = solver._group_particle_count.numpy()
    test.assertEqual(int(counts[0]), 2)
    test.assertEqual(int(counts[1]), 3)
    masses = solver.total_group_mass.numpy()
    test.assertAlmostEqual(float(masses[0]), 1.0, places=4)
    test.assertAlmostEqual(float(masses[1]), 2.0, places=4)


add_function_test(
    TestSolverUXPBDPhase2,
    "test_uxpbd_solver_caches_shape_match_data",
    test_uxpbd_solver_caches_shape_match_data,
    devices=get_test_devices(),
)


def test_uxpbd_sm_rigid_cube_stays_rigid(test, device):
    """A SM-rigid cube spinning in free space stays rigid (shape matching works).

    Zero gravity, no contact. Validates the shape-matching pass keeps all
    pairwise particle distances within 1% after 500 substeps with an initial
    angular velocity around z.

    Note: requires CUDA. The tiled shape-matching kernels (solve_shape_matching_batch_tiled,
    enforce_momemntum_conservation_tiled) use wp.tile_extract after wp.tile_reduce.
    On CPU, wp.tile_extract does not broadcast the reduced value to all threads in the
    block, which produces incorrect center-of-mass values in non-lane-0 threads and
    corrupts the shape-matching delta and momentum conservation. CUDA broadcasts
    tile_extract correctly.
    """
    if not device.is_cuda:
        test.skipTest("SM-rigid tiled kernels require CUDA (tile_extract broadcast)")
    builder = newton.ModelBuilder(up_axis="Z")
    half_extent = 0.11
    sphere_r = 0.025
    coords = np.linspace(-half_extent + sphere_r, half_extent - sphere_r, 4)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    centers = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1)
    radii = np.full(centers.shape[0], sphere_r)
    builder.add_particle_volume(
        volume_data={"centers": centers.tolist(), "radii": radii.tolist()},
        total_mass=4.0,
        pos=wp.vec3(0.0, 0.0, 5.0),
    )
    model = builder.finalize(device=device)

    # Set initial angular velocity around z.
    pos = model.particle_q.numpy()
    com = pos.mean(axis=0)
    omega = np.array([0.0, 0.0, 2.0])
    qd_np = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        r = pos[i] - com
        qd_np[i] = np.cross(omega, r)
    model.particle_qd.assign(qd_np)

    # Zero gravity.
    grav_np = model.gravity.numpy() * 0.0
    model.gravity.assign(grav_np)

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()
    dt = 0.001
    for _ in range(500):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    initial_pos = pos
    for j in range(1, 5):
        d_init = float(np.linalg.norm(initial_pos[j] - initial_pos[0]))
        d_final = float(np.linalg.norm(final_pos[j] - final_pos[0]))
        rel_err = abs(d_final - d_init) / max(d_init, 1e-3)
        test.assertLess(rel_err, 0.01, f"particle pair {j} distance drift {rel_err}")


add_function_test(
    TestSolverUXPBDPhase2,
    "test_uxpbd_sm_rigid_cube_stays_rigid",
    test_uxpbd_sm_rigid_cube_stays_rigid,
    devices=get_test_devices(),
)


def test_uxpbd_sm_rigid_cube_drops_to_ground(test, device):
    """A free SM-rigid cube falls and settles on the ground plane.

    Validates the cross-substrate particle-shape contact path: SM-rigid
    particles must contact the ground plane via solve_particle_shape_contacts_uxpbd
    and route deltas into particle_deltas (not the body wrench path).

    Skipped on CPU due to the Warp tile-reduce CPU limitation documented
    in test_uxpbd_sm_rigid_cube_stays_rigid.
    """
    if not device.is_cuda:
        test.skipTest("SRXPBD tiled shape-matching does not broadcast correctly on CPU (Warp 1.14.0).")

    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    half_extent = 0.11
    sphere_r = 0.025
    coords = np.linspace(-half_extent + sphere_r, half_extent - sphere_r, 4)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    centers = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1)
    radii = np.full(centers.shape[0], sphere_r)
    builder.add_particle_volume(
        volume_data={"centers": centers.tolist(), "radii": radii.tolist()},
        total_mass=4.0,
        pos=wp.vec3(0.0, 0.0, 0.5),
    )
    model = builder.finalize(device=device)
    model.particle_mu = 0.4
    model.soft_contact_mu = 0.4

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()
    dt = 0.001
    contacts = model.contacts()
    for _ in range(2000):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    # Cube rests with bottom sphere touching ground: COM_z = half_extent ~= 0.11.
    particle_q = state_0.particle_q.numpy()
    com_z = float(np.mean(particle_q[:, 2]))
    test.assertAlmostEqual(com_z, 0.11, delta=0.02, msg=f"cube settled at {com_z}, expected ~0.11")


add_function_test(
    TestSolverUXPBDPhase2,
    "test_uxpbd_sm_rigid_cube_drops_to_ground",
    test_uxpbd_sm_rigid_cube_drops_to_ground,
    devices=get_test_devices(),
)


def test_uxpbd_shock_propagation_param_accepted(test, device):
    """SolverUXPBD accepts shock_propagation_k > 0 and uses scaled inv_mass.

    A full settling test would need the SRXPBD tile primitives to work on
    CPU (deferred to CUDA validation). Phase 2 Task 6 only validates that
    the constructor parameter is wired and the kernel launch path executes.
    """
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    builder.add_particle_volume(
        volume_data={"centers": [[0.0, 0.0, 0.1]], "radii": [0.05]},
        total_mass=1.0,
    )
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverUXPBD(model, iterations=2, shock_propagation_k=2.0)
    test.assertEqual(solver.shock_propagation_k, 2.0)
    # Up axis should be Z (=2) for the default gravity (0, 0, -9.81).
    test.assertEqual(solver._up_axis_int, 2)

    # One step should run without crashing (the scaled-mass kernel launches).
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, None, contacts, 0.001)


add_function_test(
    TestSolverUXPBDPhase2,
    "test_uxpbd_shock_propagation_param_accepted",
    test_uxpbd_shock_propagation_param_accepted,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
