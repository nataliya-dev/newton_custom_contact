# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UXPBD solver (Phase 1: articulated rigid + lattice + static contact)."""

import json
import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "uxpbd")


def test_uxpbd_solver_rejects_enable_cslc(test, device):
    """Phase 1: enable_cslc=True must raise NotImplementedError (reserved for v2)."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    with test.assertRaises(NotImplementedError):
        newton.solvers.SolverUXPBD(model, enable_cslc=True)


def test_uxpbd_solver_instantiates_with_empty_model(test, device):
    """SolverUXPBD can be constructed from an empty Model without errors."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverUXPBD(model)
    test.assertIs(solver.model, model)


class TestSolverUXPBD(unittest.TestCase):
    pass


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_solver_rejects_enable_cslc",
    test_uxpbd_solver_rejects_enable_cslc,
    devices=get_test_devices(),
)
add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_solver_instantiates_with_empty_model",
    test_uxpbd_solver_instantiates_with_empty_model,
    devices=get_test_devices(),
)


def test_uxpbd_empty_model_has_zero_lattice(test, device):
    """A Model finalized with no lattices has lattice_sphere_count == 0."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    test.assertEqual(getattr(model, "lattice_sphere_count", 0), 0)
    # Lattice arrays exist but are empty
    test.assertEqual(model.lattice_p_rest.shape[0], 0)
    test.assertEqual(model.lattice_r.shape[0], 0)
    test.assertEqual(model.lattice_link.shape[0], 0)
    test.assertEqual(model.lattice_delta.shape[0], 0)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_empty_model_has_zero_lattice",
    test_uxpbd_empty_model_has_zero_lattice,
    devices=get_test_devices(),
)


def test_uxpbd_add_lattice_populates_arrays(test, device):
    """builder.add_lattice loads MorphIt JSON, creates lattice + particle entries."""
    builder = newton.ModelBuilder()
    link = builder.add_body()  # one floating body (add_body auto-creates a free joint)

    json_path = os.path.join(_ASSET_DIR, "tiny_lattice.json")
    builder.add_lattice(link=link, morphit_json=json_path, total_mass=1.0)

    test.assertEqual(len(builder.lattice_p_rest), 5)
    test.assertEqual(len(builder.lattice_r), 5)
    test.assertEqual(len(builder.lattice_link), 5)
    test.assertEqual(builder.lattice_link[0], link)
    test.assertEqual(builder.lattice_link[4], link)
    test.assertAlmostEqual(builder.lattice_r[0], 0.05, places=6)
    test.assertAlmostEqual(builder.lattice_r[1], 0.04, places=6)

    # Particles were added to the builder
    test.assertEqual(len(builder.particle_q), 5)

    # Each lattice sphere's lattice_particle_index points to the matching slot
    test.assertEqual(list(builder.lattice_particle_index), [0, 1, 2, 3, 4])

    # Finalize produces non-empty model arrays
    model = builder.finalize(device=device)
    test.assertEqual(model.lattice_sphere_count, 5)
    test.assertEqual(model.lattice_link.shape[0], 5)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_add_lattice_populates_arrays",
    test_uxpbd_add_lattice_populates_arrays,
    devices=get_test_devices(),
)


def test_uxpbd_update_lattice_projects_body_q(test, device):
    """Lattice particle world positions match body_q x p_rest analytically."""
    builder = newton.ModelBuilder()
    link = builder.add_body()  # add_body auto-creates a free joint

    # One sphere at body-frame offset (0.1, 0.0, 0.0)
    builder.add_lattice(
        link=link,
        morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"),
        total_mass=1.0,
    )
    model = builder.finalize(device=device)

    state = model.state()
    # Set body_q to translation (1.0, 2.0, 3.0), rotation identity.
    body_q_np = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    state.body_q.assign(body_q_np)
    state.body_qd.zero_()

    solver = newton.solvers.SolverUXPBD(model)
    solver.update_lattice_world_positions(state)

    expected = np.array(
        [
            [1.0, 2.0, 3.0],  # (0,0,0)   shifted by (1,2,3)
            [1.1, 2.0, 3.0],  # (0.1,0,0) shifted
            [1.0, 2.1, 3.0],  # (0,0.1,0) shifted
            [1.0, 2.0, 3.1],  # (0,0,0.1) shifted
            [0.9, 2.0, 3.0],  # (-0.1,0,0) shifted
        ],
        dtype=np.float32,
    )
    got = state.particle_q.numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_update_lattice_projects_body_q",
    test_uxpbd_update_lattice_projects_body_q,
    devices=get_test_devices(),
)


def test_uxpbd_update_lattice_handles_rotation(test, device):
    """Lattice projection respects body rotation."""
    builder = newton.ModelBuilder()
    link = builder.add_body()  # add_body auto-creates a free joint
    builder.add_lattice(
        link=link,
        morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"),
        total_mass=1.0,
    )
    model = builder.finalize(device=device)

    state = model.state()
    # 90 degree rotation around Z. The sphere at (+0.1, 0, 0) body frame
    # should project to (0, +0.1, 0) in world frame.
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
    body_q_np = np.array([[0.0, 0.0, 0.0, rot[0], rot[1], rot[2], rot[3]]], dtype=np.float32)
    state.body_q.assign(body_q_np)
    state.body_qd.zero_()

    solver = newton.solvers.SolverUXPBD(model)
    solver.update_lattice_world_positions(state)

    got = state.particle_q.numpy()
    # Sphere at body-frame (+0.1, 0, 0) lands at world (0, +0.1, 0)
    np.testing.assert_allclose(got[1], [0.0, 0.1, 0.0], atol=1e-5)
    # Sphere at body-frame (0, +0.1, 0) lands at world (-0.1, 0, 0)
    np.testing.assert_allclose(got[2], [-0.1, 0.0, 0.0], atol=1e-5)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_update_lattice_handles_rotation",
    test_uxpbd_update_lattice_handles_rotation,
    devices=get_test_devices(),
)


def test_uxpbd_lattice_sphere_drops_to_ground(test, device):
    """A free body with one lattice sphere settles on the ground plane.

    Validates lattice-to-body wrench routing: contact on the lattice
    sphere must push the host body upward, not the particle.

    NOTE: This test will fail until Task 6 wires SolverUXPBD.step(). It
    is queued here as the integration target for the contact kernel.
    """
    builder = newton.ModelBuilder(up_axis="Z")
    link = builder.add_body(
        mass=1.0,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()),
    )
    sphere_json = {
        "centers": [[0.0, 0.0, 0.0]],
        "radii": [0.05],
        "normals": [[0.0, 0.0, -1.0]],
        "is_surface": [1],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sphere_json, f)
        temp_path = f.name
    try:
        builder.add_lattice(link=link, morphit_json=temp_path, total_mass=0.0)
    finally:
        os.unlink(temp_path)

    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverUXPBD(model, iterations=4)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 1.0 / 1000.0
    contacts = model.contacts()
    for _ in range(2000):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    body_z = float(body_q[0, 2])
    test.assertAlmostEqual(body_z, 0.05, delta=1e-3)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_lattice_sphere_drops_to_ground",
    test_uxpbd_lattice_sphere_drops_to_ground,
    devices=get_test_devices(),
)


def test_uxpbd_lattice_w_eff_helper_callable(test, device):
    """The lattice_sphere_w_eff device function is exposed and callable."""
    from newton._src.solvers.uxpbd.kernels import lattice_sphere_w_eff  # noqa: PLC0415

    test.assertTrue(callable(lattice_sphere_w_eff))


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_lattice_w_eff_helper_callable",
    test_uxpbd_lattice_w_eff_helper_callable,
    devices=get_test_devices(),
)


def test_uxpbd_free_fall_trajectory(test, device):
    """A free body with no lattice contact falls under gravity as g*t^2/2."""
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_body(
        mass=1.0,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 10.0), q=wp.quat_identity()),
    )
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverUXPBD(model, iterations=1)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 0.001
    t = 0.5
    n_steps = int(t / dt)
    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    body_z = float(body_q[0, 2])
    expected_z = 10.0 - 0.5 * 9.81 * t * t
    relative_err = abs(body_z - expected_z) / expected_z
    test.assertLess(relative_err, 0.005, f"free fall z={body_z}, expected={expected_z}")


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_free_fall_trajectory",
    test_uxpbd_free_fall_trajectory,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
