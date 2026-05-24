# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PBDR Sphere Drop
#
# Drops a single MorphIt sphere-packed rigid body onto the ground plane
# under SolverSRXPBD -- the PBD-R formulation from
#
#   Abderezaei et al., "Physically Accurate Rigid-Body Dynamics in
#   Particle-Based Simulation" (cslc_xpbd/papers/srxpbd.pdf)
#
# In that paper PBD-R extends standard PBD with (i) a corrected
# velocity update and (ii) a momentum-conservation constraint on top
# of the shape-matching step. In this codebase that solver is exposed
# as `newton.solvers.SolverSRXPBD` (Shape-matching Rigid XPBD).
#
# Purpose: read out per-frame telemetry (centroid position, centroid
# velocity, per-particle |v|_max) while the same MorphIt sphere used in
# example_uxpbd_lift_test free-falls and settles, so we can see whether
# the XY/Z oscillation observed under SolverUXPBD persists under PBD-R.
# No phases, no driven bodies, no asserts -- just the body and the floor.
# `--asset bunny` swaps the sphere packing for MorphIt's 100-sphere
# Stanford-bunny packing at native scale, to exercise an asymmetric
# rest pose / inertia tensor.
#
# Command: python -m newton.examples pbdr_sphere_drop
#          python -m newton.examples pbdr_sphere_drop --asset bunny
###########################################################################

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"

# Per-asset MorphIt packing config: JSON path, optional rescale target,
# and total mass.
#   - sphere: same 125-sphere unit-sphere packing as example_uxpbd_lift_test,
#     rescaled at load time to a 4 cm envelope radius (the proven-stable
#     SM-rigid regime per lift_test's obj_radius note; smaller radii hit
#     SVD-noise-dominated covariance in the shape-matching rotation
#     extraction).
#   - bunny: MorphIt's 100-sphere Stanford-bunny packing at native scale
#     (~17 cm bbox). Native is well above the SVD-noise floor, no rescale
#     needed; keeps MorphIt's joint mass/COM/inertia optimisation intact.
_ASSETS = {
    "sphere": {
        "path": _ASSETS_DIR / "sphere" / "sphere.json",
        "target_envelope": 0.04,   # rescale so max sphere-surface reach == 4 cm
        "mass": 1.0,
    },
    "bunny": {
        "path": _ASSETS_DIR / "bunny-lowpoly" / "bunny.json",
        "target_envelope": None,   # native MorphIt scale
        "mass": 1.0,
    },
    "box": {
        "path": _ASSETS_DIR / "box" / "box.json",
        "target_envelope": None,   # native MorphIt scale (~16 cm cube)
        "mass": 1.0,
    },
}

# Spawn the lowest sphere surface this high above z=0 for a clean
# free-fall onto contact (matches example_uxpbd_lift_test SETTLE).
SPAWN_CLEARANCE = 0.04

# Match example_uxpbd_lift_test's friction so the SM <-> contact tug-of-war
# (which drives the observed wobble) has the same tangential strength.
# mu_eff in the kernels = 0.5 * (particle_mu + shape_material_mu); setting
# both ends to the same value yields exactly this mu_eff.
MU = 1.0


class Example:
    def __init__(self, viewer, args):
        # Match example_uxpbd_lift_test integration cadence so per-substep
        # dt is identical: 16 substeps at 100 Hz = dt 6.25e-4 s. Smaller dt
        # changes the v_new = vp + d/dt amplification factor; matching it
        # keeps the wobble drivers apples-to-apples between the two runs.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # MorphIt packing for the selected asset.
        # native_envelope = max_i(|c_i| + r_i) is the furthest reach of
        # any sub-sphere surface from the body origin; dividing it into
        # `target_envelope` (if any) gives the uniform scale that maps
        # the native packing to a body of the requested envelope radius.
        cfg = _ASSETS[args.asset]
        with open(cfg["path"]) as f:
            data = json.load(f)
        native_centers = np.asarray(data["centers"], dtype=np.float32)
        native_radii = np.asarray(data["radii"], dtype=np.float32)
        native_masses = np.asarray(data["masses"], dtype=np.float32)

        if cfg["target_envelope"] is not None:
            native_envelope = float(
                (np.linalg.norm(native_centers, axis=1) + native_radii).max())
            scale = cfg["target_envelope"] / native_envelope
            obj_centers = (native_centers * scale).astype(np.float32)
            obj_radii = (native_radii * scale).astype(np.float32)
        else:
            obj_centers = native_centers
            obj_radii = native_radii

        # Spawn so the lowest sphere surface is `SPAWN_CLEARANCE` above z=0.
        # For the symmetric sphere this gives spawn_z = radius + clearance
        # (matches the original constant); for the bunny it accounts for
        # MorphIt's asymmetric body-frame z-offset of the packing.
        lowest_body_z = float((obj_centers[:, 2] - obj_radii).min())
        self._spawn_z = SPAWN_CLEARANCE - lowest_body_z

        self.obj_group = builder.add_particle_volume(
            volume_data={"centers": obj_centers.tolist(),
                         "radii": obj_radii.tolist()},
            total_mass=cfg["mass"],
            pos=wp.vec3(0.0, 0.0, self._spawn_z),
        )

        # Override add_particle_volume's volume-weighted mass distribution
        # with MorphIt's physics-optimised per-particle masses from the
        # asset JSON (jointly tuned with positions to match the true
        # body's mass / COM / inertia). Rescale so the total equals
        # cfg["mass"].
        mass_scale = cfg["mass"] / float(native_masses.sum())
        obj_masses = (native_masses * mass_scale).astype(np.float32)
        for idx, m in zip(builder.particle_groups[self.obj_group], obj_masses):
            builder.particle_mass[idx] = float(m)

        self.model = builder.finalize()

        # Friction override (same as example_uxpbd_lift_test). The contact
        # kernel computes mu_eff = 0.5 * (particle_mu + shape_material_mu).
        self.model.particle_mu = MU
        self.model.soft_contact_mu = MU
        self.model.shape_material_mu.assign(
            np.full(self.model.shape_count, MU, dtype=np.float32))

        # SolverSRXPBD = the paper's PBD-R: shape-matching + corrected
        # velocity update + momentum-conservation constraint. UXPBD imports
        # the same SM + momentum kernels and replicates the corrected
        # velocity update, so the kernels driving the wobble are shared.
        # UXPBD's extra `stabilization_iterations` and `shock_propagation_k`
        # have no SRXPBD analog, so those damping passes are absent here;
        # the wobble amplitude can differ even though the source is the same.
        self.solver = newton.solvers.SolverSRXPBD(
            self.model, iterations=args.iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Maximal-coordinate solvers need FK to initialise joint state
        # even when there are no articulated bodies.
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        obj_idx = self.model.particle_groups[self.obj_group]
        if hasattr(obj_idx, "numpy"):
            obj_idx = obj_idx.numpy()
        self._obj_idx = np.asarray(list(obj_idx), dtype=np.int32)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(0.3, -0.3, self._spawn_z + 0.10),
            pitch=-15.0, yaw=135.0,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

        # Telemetry: centroid + per-particle |v|_max each frame. Dense in
        # the first 20 frames (catches the t~0 SM-rigid spike) and every
        # 25 frames afterwards. Mean velocity cancels symmetric internal
        # dispersion; |v|_max exposes it.
        frame = int(round(self.sim_time * self.fps))
        if frame < 20 or frame % 25 == 0:
            q = self.state_0.particle_q.numpy()[self._obj_idx]
            v = self.state_0.particle_qd.numpy()[self._obj_idx]
            c = q.mean(axis=0)
            cv = v.mean(axis=0)
            v_max = float(np.linalg.norm(v, axis=1).max())
            # Spread = stddev of particle-z about the centroid. For a
            # rigid SM body this should stay constant; a growing spread
            # means the cluster is deforming (SM constraint losing grip).
            z_spread = float(q[:, 2].std())
            print(
                f"[f={frame:03d} t={self.sim_time:.3f}] "
                f"c=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}) "
                f"cv=({cv[0]:+.4f},{cv[1]:+.4f},{cv[2]:+.4f}) "
                f"|v|_max={v_max:.3f} z_spread={z_spread:.4f}"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--asset",
            type=str,
            default="sphere",
            choices=list(_ASSETS.keys()),
            help=(
                "MorphIt packing to drop. `sphere` is the proven-stable "
                "4 cm-radius rescaled unit-sphere packing from "
                "example_uxpbd_lift_test; `bunny` is the 100-sphere "
                "Stanford-bunny packing at native scale (~17 cm bbox); "
                "`box` is the 100-sphere box packing at native scale "
                "(~16 cm cube)."
            ),
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=8,
            help="PBD-R constraint solver iterations per substep. "
            "Default 8 matches example_uxpbd_lift_test.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
