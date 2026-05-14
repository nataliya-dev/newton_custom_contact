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


def test_uxpbd_pendulum_period(test, device):
    """A simple pendulum of length L oscillates at period T = 2*pi*sqrt(L/g).

    Geometry: pivot at world origin, child body COM hangs L=1m below the pivot
    along -Z in the child frame (child_xform offset (0, 0, L)). A Y-axis
    revolute joint controls rotation. Initial joint angle 0.05 rad (small angle).

    For small angle theta around Y, the joint coordinate q[0] = theta.
    We read the angle directly from model.joint_q after FK, but during simulation
    we recover it from body_q quaternion: q_y = sin(theta/2) ~= theta/2, so
    theta ~= 2 * body_q[4].
    """
    builder = newton.ModelBuilder(up_axis="Z")
    L = 1.0
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.05)
    # child_xform offset (0, 0, L): pivot is L above the child's COM in child frame.
    j = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, L), q=wp.quat_identity()),
    )
    builder.add_articulation([j], label="pendulum")

    # Set initial joint angle before finalize. For this joint layout,
    # q[0] is the revolute DOF angle (radians about Y).
    initial_angle = 0.05
    builder.joint_q[0] = initial_angle

    model = builder.finalize(device=device)

    solver = newton.solvers.SolverUXPBD(model, iterations=8)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 0.001
    angles = []
    for _ in range(4000):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0
        bq = state_0.body_q.numpy()[0]
        # body_q is [tx, ty, tz, qx, qy, qz, qw].
        # For small angle theta around Y: qy ~= theta/2, so theta ~= 2 * qy.
        angles.append(2.0 * float(bq[4]))

    angles_np = np.array(angles)
    crossings = np.where(np.diff(np.signbit(angles_np)))[0]
    test.assertGreater(len(crossings), 2, f"Not enough zero crossings: {len(crossings)}")
    half_period_steps = float(crossings[2] - crossings[0]) / 2.0
    measured_T = half_period_steps * dt * 2.0
    expected_T = 2.0 * np.pi * np.sqrt(L / 9.81)
    relative_err = abs(measured_T - expected_T) / expected_T
    test.assertLess(relative_err, 0.01, f"T={measured_T:.4f}, expected={expected_T:.4f}")


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_pendulum_period",
    test_uxpbd_pendulum_period,
    devices=get_test_devices(),
)


def test_uxpbd_body_parent_f_revolute_to_world(test, device):
    """A revolute joint to the world reports a non-zero parent wrench when
    gravity acts on the child. Should match XPBD's output to within 5%.
    """

    def build_and_step(solver_class):
        builder = newton.ModelBuilder(up_axis="Z")
        L = 1.0
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.05)
        j = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, L), q=wp.quat_identity()),
        )
        builder.add_articulation([j], label="pendulum")
        builder.request_state_attributes("body_parent_f")
        model = builder.finalize(device=device)

        solver = solver_class(model)
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        for _ in range(20):
            state_0.clear_forces()
            solver.step(state_0, state_1, None, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        return state_0.body_parent_f.numpy()[0].copy()

    xpbd_wrench = build_and_step(newton.solvers.SolverXPBD)
    uxpbd_wrench = build_and_step(newton.solvers.SolverUXPBD)

    for i in range(6):
        denom = abs(xpbd_wrench[i]) + 1.0e-3
        rel_err = abs(uxpbd_wrench[i] - xpbd_wrench[i]) / denom
        test.assertLess(rel_err, 0.05, f"body_parent_f[{i}] xpbd={xpbd_wrench[i]} uxpbd={uxpbd_wrench[i]}")


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_body_parent_f_revolute_to_world",
    test_uxpbd_body_parent_f_revolute_to_world,
    devices=get_test_devices(),
)


def test_uxpbd_add_lattice_to_all_links_with_fallback(test, device):
    """add_lattice_to_all_links uses JSON when present, falls back to uniform packing otherwise."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    tmp = tempfile.mkdtemp(prefix="uxpbd_test_")
    try:
        shutil.copy(
            os.path.join(_ASSET_DIR, "tiny_lattice.json"),
            os.path.join(tmp, "link_a.json"),
        )

        builder = newton.ModelBuilder()
        builder.add_body(label="link_a")
        builder.add_body(label="link_b")

        builder.add_lattice_to_all_links(
            morphit_json_dir=tmp,
            fallback_uniform_n=2,  # 2x2x2 = 8 spheres
        )

        model = builder.finalize(device=device)
        # 5 spheres from JSON + 8 from fallback = 13.
        test.assertEqual(model.lattice_sphere_count, 13)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_add_lattice_to_all_links_with_fallback",
    test_uxpbd_add_lattice_to_all_links_with_fallback,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
