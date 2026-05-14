# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""UXPBD vs MuJoCo box-push comparison.

Reproduces PBD-R Test 1 (F=17 N horizontal on a 4 kg box, mu=0.4 ground)
under both SolverUXPBD and SolverMuJoCo, prints final-position error
relative to the analytical 0.5*(F-mu*M*g)/M*t^2.

Run:
    uv run python -m cslc_mujoco.uxpbd_comparison.box_push --solver uxpbd
    uv run python -m cslc_mujoco.uxpbd_comparison.box_push --solver mujoco
    uv run python -m cslc_mujoco.uxpbd_comparison.box_push --solver both
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton


def _build_box_sphere_packing(builder, pos=(0.0, 0.0, 0.11)):
    """The PBD-R reference box (4x4x4 spheres, M=4 kg, half-extent 0.11 m)."""
    half_extent = 0.11
    sphere_r = 0.025
    coords = np.linspace(-half_extent + sphere_r, half_extent - sphere_r, 4)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    centers = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1)
    radii = np.full(centers.shape[0], sphere_r)
    return builder.add_particle_volume(
        volume_data={"centers": centers.tolist(), "radii": radii.tolist()},
        total_mass=4.0,
        pos=wp.vec3(*pos),
    )


def run_uxpbd():
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    _build_box_sphere_packing(builder)
    model = builder.finalize()
    model.particle_mu = 0.4
    model.soft_contact_mu = 0.4

    F = 17.0
    M = 4.0
    mu = 0.4
    g = 9.81
    expected_x = 0.5 * (F - mu * M * g) / M * 100.0

    solver = newton.solvers.SolverUXPBD(model, iterations=10)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001
    n_steps = 10000

    force_np = np.zeros((model.particle_count, 3), dtype=np.float32)
    force_np[:, 0] = F / model.particle_count

    for _ in range(n_steps):
        state_0.clear_forces()
        state_0.particle_f.assign(force_np)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    com_x = float(state_0.particle_q.numpy().mean(axis=0)[0])
    rel_err = abs(com_x - expected_x) / abs(expected_x)
    print(f"UXPBD: final x = {com_x:.4f} m, expected = {expected_x:.4f} m, rel_err = {rel_err:.4f}")
    return com_x, expected_x


def run_mujoco():
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    box = builder.add_body(mass=4.0, xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.11), q=wp.quat_identity()))
    builder.add_shape_box(box, hx=0.11, hy=0.11, hz=0.11)

    try:
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    except AttributeError:
        pass  # not all Newton versions require this

    model = builder.finalize()
    for sh in range(model.shape_count):
        model.shape_material_mu[sh] = 0.4

    F = 17.0
    M = 4.0
    mu = 0.4
    g = 9.81
    expected_x = 0.5 * (F - mu * M * g) / M * 100.0

    solver = newton.solvers.SolverMuJoCo(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    dt = 0.001
    n_steps = 10000

    # body_f layout: spatial_vector = (angular, linear); indices 0-2 angular, 3-5 linear.
    body_f = np.zeros((model.body_count, 6), dtype=np.float32)
    body_f[box, 3] = F  # fx

    for _ in range(n_steps):
        state_0.clear_forces()
        state_0.body_f.assign(body_f)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    body_q_np = state_0.body_q.numpy()
    com_x = float(body_q_np[box, 0])
    rel_err = abs(com_x - expected_x) / abs(expected_x)
    print(f"MuJoCo: final x = {com_x:.4f} m, expected = {expected_x:.4f} m, rel_err = {rel_err:.4f}")
    return com_x, expected_x


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["uxpbd", "mujoco", "both"], default="both")
    args = p.parse_args()
    if args.solver in ("uxpbd", "both"):
        run_uxpbd()
    if args.solver in ("mujoco", "both"):
        run_mujoco()
