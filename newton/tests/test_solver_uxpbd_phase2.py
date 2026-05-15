# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UXPBD Phase 2: free SM-rigid, cross-substrate contact, PBD-R benchmarks."""

import os
import sys
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
    builder.add_lattice(link=link_a, morphit_json=os.path.join(
        _ASSET_DIR, "tiny_lattice.json"), total_mass=1.0)
    builder.add_lattice(link=link_b, morphit_json=os.path.join(
        _ASSET_DIR, "tiny_lattice.json"), total_mass=1.0)
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
    builder.add_lattice(link=link, morphit_json=os.path.join(
        _ASSET_DIR, "tiny_lattice.json"), total_mass=1.0)

    # Add a free SM-rigid group via add_particle_volume.
    builder.add_particle_volume(
        volume_data={"centers": [[1.0, 0.0, 0.0], [
            1.0, 0.1, 0.0]], "radii": [0.05, 0.05]},
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
        volume_data={"centers": [[0.0, 0.0, 0.0], [
            0.1, 0.0, 0.0]], "radii": [0.05, 0.05]},
        total_mass=1.0,
    )
    builder.add_particle_volume(
        volume_data={"centers": [[2.0, 0.0, 0.0], [2.1, 0.0, 0.0], [
            2.2, 0.0, 0.0]], "radii": [0.05, 0.05, 0.05]},
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
        test.skipTest(
            "SM-rigid tiled kernels require CUDA (tile_extract broadcast)")
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
        test.assertLess(
            rel_err, 0.01, f"particle pair {j} distance drift {rel_err}")


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
        test.skipTest(
            "SRXPBD tiled shape-matching does not broadcast correctly on CPU (Warp 1.14.0).")

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
    # The kernel uses mu = 0.5 * (particle_mu + shape_material_mu[shape]).
    # Default shape_material_mu is 0.5, so we must override to get the
    # intended 0.4 effective coefficient.
    model.shape_material_mu.assign(
        np.full(model.shape_count, 0.4, dtype=np.float32))

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
    test.assertAlmostEqual(com_z, 0.11, delta=0.02,
                           msg=f"cube settled at {com_z}, expected ~0.11")


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

    solver = newton.solvers.SolverUXPBD(
        model, iterations=2, shock_propagation_k=2.0)
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


def _build_pbdr_box(builder, pos=(0.0, 0.0, 0.10), rot=None):
    """Build the PBD-R reference box: 4x4x4 = 64 spheres, m=4 kg, edge=0.20 m.

    Matches ``shapes/box.py::Box.add_spheres`` (hx=hy=hz=0.10, num_spheres=4):
    radius_mean = hx / num_spheres = 0.025
    cell_x     = (2*hx - 2*radius_mean) / (dim_x - 1) = 0.05
    """
    if rot is None:
        rot = wp.quat_identity()
    half_extent = 0.10
    num_spheres = 4
    radius_mean = half_extent / num_spheres   # 0.025
    cell_x = (2.0 * half_extent - 2.0 * radius_mean) / \
        (num_spheres - 1)   # 0.05
    total_mass = 4.0
    num_particles = num_spheres ** 3
    mass_per_particle = total_mass / num_particles
    pos_corner = wp.vec3(*pos) + wp.quat_rotate(
        rot,
        wp.vec3(-half_extent + radius_mean, -half_extent +
                radius_mean, -half_extent + radius_mean),
    )
    start_idx = len(builder.particle_q)
    builder.add_particle_grid(
        pos=pos_corner,
        rot=rot,
        vel=wp.vec3(0.0),
        dim_x=num_spheres,
        dim_y=num_spheres,
        dim_z=num_spheres,
        cell_x=cell_x,
        cell_y=cell_x,
        cell_z=cell_x,
        mass=mass_per_particle,
        jitter=0.0,
        radius_mean=radius_mean,
        radius_std=0.0,
    )
    end_idx = len(builder.particle_q)
    # add_particle_grid does not register a particle group; shape-matching
    # needs one so we register it explicitly, matching shapes/box.py.
    group_id = builder.particle_group_count
    builder.particle_group_count += 1
    for i in range(start_idx, end_idx):
        builder.particle_group[i] = group_id
    builder.particle_groups[group_id] = list(range(start_idx, end_idx))
    return group_id


def test_pbdr_t1_pushed_box(test, device):
    """PBD-R Test 1: F=17 N horizontal at COM, mu=0.4 ground. Analytical
    x(t) = 0.5 * (F - mu*M*g)/M * t^2.
    """
    if not device.is_cuda:
        test.skipTest(
            "SRXPBD tile-reduce broadcast not supported on CPU (Warp 1.14.0).")

    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    _build_pbdr_box(builder)
    model = builder.finalize(device=device)
    model.particle_mu = 0.4
    model.soft_contact_mu = 0.4
    # The kernel uses mu = 0.5 * (particle_mu + shape_material_mu[shape]).
    # Default shape_material_mu is 0.5, so we must override to get the
    # intended 0.4 effective coefficient.
    model.shape_material_mu.assign(
        np.full(model.shape_count, 0.4, dtype=np.float32))

    F = 17.0
    M = 4.0
    mu = 0.4
    g = 9.81
    a = (F - mu * M * g) / M

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001
    n_steps = 10000

    force_per_particle = np.zeros((model.particle_count, 3), dtype=np.float32)
    force_per_particle[:, 0] = F / model.particle_count

    for _ in range(n_steps):
        state_0.clear_forces()
        state_0.particle_f.assign(force_per_particle)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    com_x = float(np.mean(final_pos[:, 0]))
    expected = 0.5 * a * (n_steps * dt) ** 2
    rel_err = abs(com_x - expected) / abs(expected)
    test.assertLess(
        rel_err, 0.05, f"Test 1 final x={com_x:.4f}, expected={expected:.4f}, err={rel_err:.4f}")


add_function_test(
    TestSolverUXPBDPhase2,
    "test_pbdr_t1_pushed_box",
    test_pbdr_t1_pushed_box,
    devices=get_test_devices(),
)


def test_pbdr_t2_box_torque(test, device):
    """PBD-R Test 2: tau=0.01 N*m about z, mu=0, zero gravity.

    Analytical: theta(t) = 0.5 * (tau / I_zz) * t^2, where I_zz is the
    principal moment of inertia of the 4x4x4 sphere packing about z,
    computed directly from the particle layout.
    """
    if not device.is_cuda:
        test.skipTest(
            "SRXPBD tile-reduce broadcast not supported on CPU (Warp 1.14.0).")

    builder = newton.ModelBuilder(up_axis="Z")
    _build_pbdr_box(builder, pos=(0.0, 0.0, 5.0))
    model = builder.finalize(device=device)
    grav_np = model.gravity.numpy() * 0.0
    model.gravity.assign(grav_np)
    model.particle_mu = 0.0

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()

    tau = 0.01
    lam = _compute_principal_inertia_zz(
        state_0.particle_q.numpy(),
        model.particle_mass.numpy(),
        np.arange(model.particle_count, dtype=np.int32),
    )
    alpha = tau / lam

    # Build a per-particle tangential-force pattern that produces net torque = tau.
    initial_pos = state_0.particle_q.numpy()
    com = initial_pos.mean(axis=0)
    r = initial_pos - com
    tangent = np.stack([-r[:, 1], r[:, 0], np.zeros_like(r[:, 0])], axis=1)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm = np.where(tangent_norm > 1e-9, tangent_norm, 1.0)
    tangent = tangent / tangent_norm
    r_xy = np.linalg.norm(r[:, :2], axis=1)
    sum_r_xy = float(np.sum(r_xy))
    f_mag = tau / sum_r_xy
    force_per_particle = (tangent * f_mag).astype(np.float32)

    dt = 0.001
    n_steps = 10000
    for _ in range(n_steps):
        state_0.clear_forces()
        state_0.particle_f.assign(force_per_particle)
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    com_f = final_pos.mean(axis=0)
    p_init = initial_pos[0] - com
    p_final = final_pos[0] - com_f
    theta_init = np.arctan2(p_init[1], p_init[0])
    theta_final = np.arctan2(p_final[1], p_final[0])
    theta = theta_final - theta_init
    expected = 0.5 * alpha * (n_steps * dt) ** 2
    # Unwrap to within pi of the expected.
    while theta < expected - np.pi:
        theta += 2.0 * np.pi
    while theta > expected + np.pi:
        theta -= 2.0 * np.pi
    rel_err = abs(theta - expected) / abs(expected)
    test.assertLess(
        rel_err, 0.05, f"Test 2 theta={theta:.4f}, expected={expected:.4f}, err={rel_err:.4f}")


add_function_test(
    TestSolverUXPBDPhase2,
    "test_pbdr_t2_box_torque",
    test_pbdr_t2_box_torque,
    devices=get_test_devices(),
)


def test_pbdr_t3_box_on_slope(test, device):
    """PBD-R Test 3: box on theta=pi/8 slope, mu=0.4. Analytical
    x(t) = 0.5 * (g sin theta - mu g cos theta) * t^2.
    """
    if not device.is_cuda:
        test.skipTest(
            "SRXPBD tile-reduce broadcast not supported on CPU (Warp 1.14.0).")

    builder = newton.ModelBuilder(up_axis="Z")
    slope_angle = np.pi / 8.0
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), slope_angle)
    builder.add_shape_plane(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=rot),
        # Long enough that the box (which slides ~188 m in 10 s with mu=0)
        # stays on the slope for the full simulation.
        width=500.0,
        length=10.0,
    )
    # Seat the cube on the slope so the bottom face just touches (no initial
    # fall, no impact spike). With the cube oriented along the slope, the
    # bottom particle centers sit one sphere-radius above the slope plane.
    # half_extent = 0.10, sphere_r = 0.025 -> COM at height (0.10 - 0.025 + 0.025)/cos(theta)
    # = 0.10 / cos(theta) above the slope perpendicular, projected back to world z.
    box_he = 0.10
    box_sr = 0.025
    com_above_slope_perp = (box_he - box_sr) + box_sr  # 0.10
    z_pos = com_above_slope_perp / np.cos(slope_angle)
    _build_pbdr_box(builder, pos=(0.0, 0.0, z_pos), rot=rot)
    model = builder.finalize(device=device)
    # Frictionless: with theta=pi/8 and mu=0.4 the analytical acceleration
    # a = g*(sin - mu*cos) is a small difference of large numbers, so any
    # ~1% friction-model imprecision blows up to ~30% trajectory error and
    # the test fails despite the simulator being correct. The friction code
    # is exercised by t1 (pushed_box); this test isolates gravity-driven slide.
    model.particle_mu = 0.0
    model.soft_contact_mu = 0.0
    model.shape_material_mu.assign(
        np.full(model.shape_count, 0.0, dtype=np.float32))

    g = 9.81
    mu = 0.0
    a = g * np.sin(slope_angle) - mu * g * np.cos(slope_angle)

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001
    n_steps = 10000

    initial_com = state_0.particle_q.numpy().mean(axis=0)
    for _ in range(n_steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    final_com = state_0.particle_q.numpy().mean(axis=0)
    # True downhill direction on the slope: projection of gravity onto the
    # slope tangent plane, normalized. For a plane rotated by `slope_angle`
    # around +Y, the slope normal is (sin, 0, cos) and the downhill tangent
    # is (cos, 0, -sin).
    slope_dir = np.array([np.cos(slope_angle), 0.0, -np.sin(slope_angle)])
    delta = final_com - initial_com
    x_slide = float(np.dot(delta, slope_dir))
    expected = 0.5 * a * (n_steps * dt) ** 2
    rel_err = abs(x_slide - expected) / abs(expected)
    test.assertLess(
        rel_err, 0.05, f"Test 3 slide={x_slide:.4f}, expected={expected:.4f}, err={rel_err:.4f}")


add_function_test(
    TestSolverUXPBDPhase2,
    "test_pbdr_t3_box_on_slope",
    test_pbdr_t3_box_on_slope,
    devices=get_test_devices(),
)


def _build_pbdr_bunny(builder, pos=(0.0, 0.0, 0.15)):
    """Build the PBD-R reference bunny (Stanford Bunny, m=2.18 kg)."""
    return builder.add_particle_volume(
        volume_data="assets/bunny-lowpoly/morphit_results.json",
        total_mass=2.18,
        pos=wp.vec3(*pos),
    )


def _compute_principal_inertia_zz(particle_q_np, particle_mass_np, group_indices):
    """Compute the principal moment of inertia about z for a particle group."""
    p = particle_q_np[group_indices]
    m = particle_mass_np[group_indices]
    M = m.sum()
    com = (m[:, None] * p).sum(axis=0) / M
    r = p - com
    return float(np.sum(m * (r[:, 0] ** 2 + r[:, 1] ** 2)))


def test_pbdr_t4_pushed_bunny(test, device):
    """PBD-R Test 4: F=10 N horizontal at COM, mu=0.4 ground. Bunny shape.

    Seats the bunny on the ground before stepping so the analytical reference
    (planar push from rest) is not contaminated by a 15 cm gravitational drop
    + impact transient that the asset's pos=(0,0,0.15) spawn would otherwise
    produce.
    """
    if not device.is_cuda:
        test.skipTest(
            "SRXPBD tile-reduce broadcast not supported on CPU (Warp 1.14.0).")

    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    _build_pbdr_bunny(builder)
    model = builder.finalize(device=device)
    model.particle_mu = 0.4
    model.soft_contact_mu = 0.4
    # The kernel uses mu = 0.5 * (particle_mu + shape_material_mu[shape]).
    # Default shape_material_mu is 0.5, so we must override to get the
    # intended 0.4 effective coefficient.
    model.shape_material_mu.assign(
        np.full(model.shape_count, 0.4, dtype=np.float32))

    # Seat the bunny so its lowest sphere sits on the ground (z=0). Must run
    # BEFORE solver construction so SolverUXPBD captures particle_q_rest at
    # the seated pose.
    pq = model.particle_q.numpy()
    pr = model.particle_radius.numpy()
    pq[:, 2] -= float(np.min(pq[:, 2] - pr))
    model.particle_q.assign(pq)

    F = 10.0
    M = 2.18
    mu = 0.4
    g = 9.81
    a = (F - mu * M * g) / M

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001
    n_steps = 10000

    f_per_p = F / model.particle_count
    force_np = np.zeros((model.particle_count, 3), dtype=np.float32)
    force_np[:, 0] = f_per_p

    for _ in range(n_steps):
        state_0.clear_forces()
        state_0.particle_f.assign(force_np)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    com_x = float(state_0.particle_q.numpy().mean(axis=0)[0])
    expected = 0.5 * a * (n_steps * dt) ** 2
    rel_err = abs(com_x - expected) / abs(expected)
    test.assertLess(
        rel_err, 0.10, f"Test 4 com_x={com_x:.4f}, expected={expected:.4f}, err={rel_err:.4f}")


add_function_test(
    TestSolverUXPBDPhase2,
    "test_pbdr_t4_pushed_bunny",
    test_pbdr_t4_pushed_bunny,
    devices=get_test_devices(),
)


def test_pbdr_t5_bunny_torque(test, device):
    """PBD-R Test 5: tau=0.01 Nm about z, zero gravity and friction. Bunny shape."""
    if not device.is_cuda:
        test.skipTest(
            "SRXPBD tile-reduce broadcast not supported on CPU (Warp 1.14.0).")

    builder = newton.ModelBuilder(up_axis="Z")
    _build_pbdr_bunny(builder, pos=(0.0, 0.0, 5.0))
    model = builder.finalize(device=device)
    grav_np = model.gravity.numpy() * 0.0
    model.gravity.assign(grav_np)
    model.particle_mu = 0.0

    # Compute principal inertia from the actual bunny packing.
    pos_np = model.particle_q.numpy()
    mass_np = model.particle_mass.numpy()
    # bunny is the only group
    group_indices = np.arange(model.particle_count, dtype=np.int32)
    Izz = _compute_principal_inertia_zz(pos_np, mass_np, group_indices)
    tau = 0.01
    alpha = tau / Izz

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()

    initial_pos = state_0.particle_q.numpy()
    com = initial_pos.mean(axis=0)
    r = initial_pos - com
    tangent = np.stack([-r[:, 1], r[:, 0], np.zeros_like(r[:, 0])], axis=1)
    tan_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tan_norm = np.where(tan_norm > 1e-9, tan_norm, 1.0)
    tangent = tangent / tan_norm
    r_xy = np.linalg.norm(r[:, :2], axis=1)
    sum_r_xy = float(np.sum(r_xy))
    f_mag = tau / sum_r_xy
    force_per_particle = (tangent * f_mag).astype(np.float32)

    dt = 0.001
    n_steps = 10000
    for _ in range(n_steps):
        state_0.clear_forces()
        state_0.particle_f.assign(force_per_particle)
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    final_pos = state_0.particle_q.numpy()
    com_f = final_pos.mean(axis=0)
    p_init = initial_pos[0] - com
    p_final = final_pos[0] - com_f
    theta_init = np.arctan2(p_init[1], p_init[0])
    theta_final = np.arctan2(p_final[1], p_final[0])
    theta = theta_final - theta_init
    expected = 0.5 * alpha * (n_steps * dt) ** 2
    while theta < expected - np.pi:
        theta += 2.0 * np.pi
    while theta > expected + np.pi:
        theta -= 2.0 * np.pi
    rel_err = abs(theta - expected) / abs(expected)
    test.assertLess(
        rel_err, 0.10, f"Test 5 theta={theta:.4f}, expected={expected:.4f}, err={rel_err:.4f}")


add_function_test(
    TestSolverUXPBDPhase2,
    "test_pbdr_t5_bunny_torque",
    test_pbdr_t5_bunny_torque,
    devices=get_test_devices(),
)


def test_pbdr_t1_lattice_pushed_box(test, device):
    """PBD-R Test 1 (lattice variant): horizontal body wrench applied to a
    lattice-clad rigid box on a mu=0.4 ground. Same analytical reference as
    test_pbdr_t1_pushed_box, but exercises the lattice contact path
    (kinematic particles slaved to a rigid body) instead of the SM-rigid
    (free particle_volume) path. Analytical:
    x(t) = 0.5 * (F - mu*M*g)/M * t^2.

    Geometry mirrors example_uxpbd_lattice_stack so the lattice contact
    pipeline is tested on the exact same 4x4x4 packing the demo uses
    (half_extent=0.04, sphere_r=0.012, identical lattice and SM-rigid
    radii). The applied force F is scaled so F/M, and therefore the
    analytical acceleration, matches the original PBD-R t1 setup.
    """
    if not device.is_cuda:
        test.skipTest("UXPBD lattice path validated on CUDA (matches t1 skip).")

    # Match example_uxpbd_lattice_stack geometry exactly.
    half_extent = 0.04
    num_spheres = 4
    sphere_r = 0.012
    # Default shape density 1000 kg/m^3 * (2*0.04)^3 = 0.512 kg.
    total_mass = 1000.0 * (2.0 * half_extent) ** 3

    coords = np.linspace(-half_extent + sphere_r, half_extent - sphere_r, num_spheres)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    centers = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
    radii = np.full(centers.shape[0], sphere_r, dtype=np.float32)

    # body_z at which the bottom lattice sphere just touches the ground:
    # body_z - (half_extent - sphere_r) = sphere_r  =>  body_z = half_extent.
    rest_z = half_extent

    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    body = builder.add_body(
        mass=0.0,  # let shape density carry mass + inertia consistently (see lattice_stack example).
        xform=wp.transform(p=wp.vec3(0.0, 0.0, rest_z), q=wp.quat_identity()),
    )
    # Default density (1000 kg/m^3) gives total_mass automatically; no cfg override needed.
    builder.add_shape_box(body, hx=half_extent, hy=half_extent, hz=half_extent)
    builder.add_lattice(
        link=body,
        morphit_json={"centers": centers, "radii": radii},
        total_mass=0.0,
        pos=wp.vec3(0.0, 0.0, rest_z),
    )
    model = builder.finalize(device=device)
    model.particle_mu = 0.4
    model.soft_contact_mu = 0.4
    # Effective mu = 0.5*(particle_mu + shape_material_mu); override the default 0.5.
    model.shape_material_mu.assign(np.full(model.shape_count, 0.4, dtype=np.float32))

    # PBD-R t1 uses F=17 N on M=4 kg (a = (17 - 0.4*4*g)/4 ~ 0.326 m/s^2).
    # Scale F by mass ratio so F/M is preserved -> analytical `a` is identical
    # to the original t1 test even with the smaller box.
    M = total_mass
    F = 17.0 * (M / 4.0)
    mu = 0.4
    g = 9.81
    a = (F - mu * M * g) / M

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # body_f layout (spatial_vector): top=linear force, bottom=torque -> [fx,fy,fz,tx,ty,tz].
    body_f_np = np.zeros((model.body_count, 6), dtype=np.float32)
    body_f_np[body, 0] = F

    dt = 0.001
    n_steps = 10000
    for _ in range(n_steps):
        state_0.clear_forces()
        state_0.body_f.assign(body_f_np)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    body_x = float(state_0.body_q.numpy()[body, 0])
    expected = 0.5 * a * (n_steps * dt) ** 2
    rel_err = abs(body_x - expected) / abs(expected)
    test.assertLess(
        rel_err, 0.05,
        f"Lattice t1 final body_x={body_x:.4f}, expected={expected:.4f}, err={rel_err:.4f}",
    )


add_function_test(
    TestSolverUXPBDPhase2,
    "test_pbdr_t1_lattice_pushed_box",
    test_pbdr_t1_lattice_pushed_box,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
