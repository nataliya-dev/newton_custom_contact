# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Stack two cubes under SRXPBD.

Drops two ~0.22 m cubes onto a ground plane: a bottom cube placed just above
the ground, and a top cube hovering ~1 cm above the bottom cube. Both have
zero initial velocity so the only motion is gravity-driven settling. The
solver is :class:`~newton.solvers.SolverSRXPBD` (rigid-body shape matching
on sphere packings).

Two cube discretizations, selected by ``--packing``:

  * ``uniform`` — every cube is an n×n×n grid of equal-radius spheres
    (default n=4 → 64 particles per cube). Built via
    :class:`shapes.box.Box`.``add_spheres``.
  * ``morphit`` — every cube is a non-uniform sphere packing loaded from
    a JSON file produced by MorphIt. Built via
    :class:`shapes.box.Box`.``add_morphit_spheres``.

Both cubes use the same ``-m / --morphit_json`` file in morphit mode.

How to run
----------
    # Default: 4×4×4 uniform packing, GL viewer
    uv run python -m cslc_xpbd.stack_two_cubes

    # 6×6×6 uniform packing
    uv run python -m cslc_xpbd.stack_two_cubes --packing uniform -n 6

    # Morphit packing (uses the box JSON from assets/)
    uv run python -m cslc_xpbd.stack_two_cubes --packing morphit \
        -m assets/box/morphit_results.json

    # Headless (no GL window). Stops automatically after --num-frames.
    uv run python -m cslc_xpbd.stack_two_cubes --headless --num-frames 500
"""
from __future__ import annotations

import warp as wp

import newton
import newton.examples
from shapes.box import Box


# Cube geometry. The morphit JSON in assets/box/ has roughly 0.22 m extent,
# so the uniform packing matches the morphit physical size by construction.
HALF_EXTENT = 0.11  # m  → 0.22 m edge length
CUBE_MASS = 4.0  # kg, total mass per cube

# Stacking geometry.
GROUND_GAP = 0.001  # m, bottom cube floats just above the plane initially
DROP_GAP = 0.01  # m, top cube hovers 1 cm above the bottom cube ("gentle" drop)


class Example:
    """Two cubes stacked vertically, settling under gravity, solver = SRXPBD."""

    def __init__(self, viewer, packing: str = "uniform", n_spheres: int = 4, morphit_json: str | None = None, mu: float = 0.6):
        self.sim_time = 0.0
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.packing = packing

        builder = newton.ModelBuilder()
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=True,
            has_particle_collision=True,
        )
        builder.add_ground_plane(cfg=ground_cfg)

        # World positions for the two cube centers (z = half-extent puts the
        # bottom face flush with the plane).
        bottom_z = HALF_EXTENT + GROUND_GAP
        top_z = bottom_z + 2.0 * HALF_EXTENT + DROP_GAP
        rot = wp.quat_identity()

        # Bottom cube
        self.bottom = Box(
            pos0=wp.vec3(0.0, 0.0, bottom_z),
            rot0=rot,
            hx=HALF_EXTENT, hy=HALF_EXTENT, hz=HALF_EXTENT,
            mass=CUBE_MASS,
        )
        self._add_cube_particles(builder, self.bottom, packing, n_spheres, morphit_json)

        # Top cube
        self.top = Box(
            pos0=wp.vec3(0.0, 0.0, top_z),
            rot0=rot,
            hx=HALF_EXTENT, hy=HALF_EXTENT, hz=HALF_EXTENT,
            mass=CUBE_MASS,
        )
        self._add_cube_particles(builder, self.top, packing, n_spheres, morphit_json)

        # Shape-shape and shape-particle friction.
        for shape_idx in range(builder.shape_count):
            builder.shape_material_mu[shape_idx] = mu
            builder.shape_material_mu_torsional[shape_idx] = 0.0
            builder.shape_material_mu_rolling[shape_idx] = 0.0

        self.model = builder.finalize()

        # SRXPBD treats particle-particle friction as ``model.particle_mu`` and
        # particle-shape friction as ``model.soft_contact_mu``. These are only
        # available after ``finalize()``.
        self.model.particle_mu = mu
        self.model.soft_contact_mu = mu

        self.solver = newton.solvers.SolverSRXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)
        # Show the spheres in the viewer so it's obvious which cube is which.
        self.viewer.show_particles = True

        # SRXPBD needs joint state initialized like other maximal-coordinate solvers.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    @staticmethod
    def _add_cube_particles(builder, box, packing, n_spheres, morphit_json):
        if packing == "uniform":
            box.add_spheres(builder, num_spheres=n_spheres)
        elif packing == "morphit":
            if not morphit_json:
                raise ValueError("--packing morphit requires -m / --morphit_json <path>")
            box.add_morphit_spheres(builder, json_adrs=morphit_json)
        else:
            raise ValueError(f"unknown packing '{packing}' (expected 'uniform' or 'morphit')")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--packing",
        choices=["uniform", "morphit"],
        default="uniform",
        help="How to discretize each cube (uniform n^3 grid vs morphit JSON).",
    )
    parser.add_argument(
        "-n", "--n_spheres",
        type=int,
        default=4,
        help="Spheres per cube edge for --packing uniform (n^3 particles per cube).",
    )
    parser.add_argument(
        "-m", "--morphit_json",
        type=str,
        default="assets/box/morphit_results.json",
        help="Path to morphit sphere-packing JSON, used when --packing morphit.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.6,
        help="Coulomb friction (applied to shapes and particles).",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(
        viewer,
        packing=args.packing,
        n_spheres=args.n_spheres,
        morphit_json=args.morphit_json,
        mu=args.mu,
    )
    newton.examples.run(example, args)
