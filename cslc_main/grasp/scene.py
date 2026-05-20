# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene orchestrator.

Single entry point :func:`build_scene` consumes a :class:`GraspConfig`,
calls the pad / object / contact-model factories, builds the
``newton.Model``, and (for CSLC mode) attaches the custom mesh-pad
handler via the patch context manager.

Topology
~~~~~~~~

Each pad is a 2-DOF articulated arm::

    world ──[prismatic X]──> slider_body ──[prismatic Z]──> pad_body

so the X joint drives APPROACH/SQUEEZE and the Z joint drives LIFT.

The held object is a single rigid body with an explicit free joint
(required for the MuJoCo solver).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
import warp as wp

import newton
from newton import JointTargetMode

from . import contact_models, objects, pads
from .params import GraspConfig


@dataclass
class SceneArtifacts:
    """Everything the runner / visualiser need to query about the scene."""

    model: newton.Model
    # DOF index of each pad prismatic joint, e.g. {"left_x": 1, "left_z": 2, ...}.
    dof_map: dict[str, int]
    # Shape index of each pad's mesh shape.
    pad_shape_indices: dict[str, int]
    # Shape index of the held object.
    object_shape_index: int
    # Body indices of pad bodies (NOT slider bodies).
    pad_body_indices: dict[str, int]
    # Body index of the held object.
    object_body_index: int
    # Pad-local trimeshes (kept for the lattice-preview visualiser).
    pad_meshes: dict[str, trimesh.Trimesh] = field(default_factory=dict)
    # Per-side (points, normals) in pad-local frame.  None when contact
    # model isn't CSLC (no sampling happens).
    pad_lattices: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


# ── Internal helpers ────────────────────────────────────────────────────


def _add_pad(
    builder: newton.ModelBuilder,
    side: str,
    config: GraspConfig,
    pad_mesh: newton.Mesh,
    pad_shape_xform: wp.transform,
    pad_shape_cfg: newton.ModelBuilder.ShapeConfig,
    ghost_cfg: newton.ModelBuilder.ShapeConfig,
    ground_shape: int | None,
    spawn_x: float,
    spawn_z: float,
) -> tuple[int, int, int, int]:
    """Add one pad (slider + pad bodies, X + Z prismatic joints) and
    return ``(slider_body, pad_body, j_x, j_z)``."""
    slider = builder.add_link(
        xform=wp.transform((spawn_x, 0.0, spawn_z), wp.quat_identity()),
        mass=0.1,
        label=f"{side}_slider",
    )
    slider_shape = builder.add_shape_sphere(slider, radius=0.002, cfg=ghost_cfg)

    pad = builder.add_link(
        xform=wp.transform((spawn_x, 0.0, spawn_z), wp.quat_identity()),
        label=f"{side}_pad",
    )
    builder.add_shape_mesh(
        body=pad,
        mesh=pad_mesh,
        xform=pad_shape_xform,
        cfg=pad_shape_cfg,
        label=f"{side}_pad_mesh",
    )

    j_x = builder.add_joint_prismatic(
        parent=-1,
        child=slider,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform((spawn_x, 0.0, spawn_z), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        label=f"{side}_x",
    )
    j_z = builder.add_joint_prismatic(
        parent=slider,
        child=pad,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        label=f"{side}_z",
    )
    builder.add_articulation([j_x, j_z], label=f"{side}_arm")

    if ground_shape is not None:
        builder.add_shape_collision_filter_pair(slider_shape, ground_shape)

    return slider, pad, j_x, j_z


# ── Public entry ────────────────────────────────────────────────────────


def build_scene(config: GraspConfig) -> SceneArtifacts:
    """Build a complete ``newton.Model`` for the grasp test scene.

    Returns the model alongside enough metadata for the runner to
    drive joint targets, read body positions, and update the lattice
    visualiser.
    """
    # ── 1. Build pad trimesh and (if CSLC) sample its contact face. ──
    pad_mesh_tm, contact_mask = pads.build_pad_trimesh(config.pad)

    if config.contact_model == "cslc":
        sampled_pts, sampled_normals = pads.sample_pad_contact_face(
            pad_mesh_tm, contact_mask, config.pad
        )
    else:
        sampled_pts = np.empty((0, 3), np.float32)
        sampled_normals = np.empty((0, 3), np.float32)

    # ── 2. Promote the trimesh to a newton.Mesh shape. ──
    pad_mesh = newton.Mesh(
        pad_mesh_tm.vertices.astype(np.float32),
        pad_mesh_tm.faces.astype(np.int32).flatten(),
    )
    if config.contact_model == "hydro":
        # Hydroelastic needs an SDF on every mesh shape it sees.
        pad_mesh.build_sdf(
            max_resolution=config.hydro.sdf_resolution,
            margin=config.material.gap,
        )

    # ── 3. Compose ShapeConfigs for pad + object. ──
    pad_cfg = contact_models.make_pad_shape_cfg(
        config.pad, config.material, config.cslc, config.hydro, config.contact_model
    )
    obj_cfg = objects.make_object_shape_cfg(
        config.object, config.material, config.hydro, config.contact_model
    )

    # ── 4. ModelBuilder: ground, both pad arms, held object. ──
    b = newton.ModelBuilder()
    ground_shape = b.add_ground_plane()

    ghost_cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False, has_particle_collision=False, density=0.0
    )

    # Spawn each pad so its inner face sits ``approach_gap`` from the
    # object surface.  Pad body origin = object_radius + face_offset + gap.
    spawn_x = (
        config.object.radius
        + _pad_thickness(config)
        + config.pad.approach_gap
    )
    # Pad body z: explicit override, or auto-derive.  See
    # PadParams.pad_center_z for the auto logic.  Box pads need to
    # clear the ground (their box_hz extends below the body centre),
    # so the auto value lifts the body so the bottom of the pad is
    # 5 mm above z=0.  Dome pads are short — they sit on the
    # object's equator.
    if config.pad.pad_center_z is not None:
        spawn_z = config.pad.pad_center_z
    elif config.pad.kind == "box":
        spawn_z = max(config.object.radius, config.pad.box_hz + 0.005)
    else:  # dome
        spawn_z = config.object.radius

    pad_body_indices: dict[str, int] = {}
    pad_shape_indices: dict[str, int] = {}
    dof_map: dict[str, int] = {}
    pad_joints: list[int] = []

    for side, x0 in (("left", -spawn_x), ("right", +spawn_x)):
        xform = pads.pad_shape_xform(config.pad, side)
        # Track shape index — add_shape_mesh below returns it via builder state.
        before_shapes = len(b.shape_type)
        _slider, pad_body, j_x, j_z = _add_pad(
            b,
            side,
            config,
            pad_mesh,
            xform,
            pad_cfg,
            ghost_cfg,
            ground_shape,
            x0,
            spawn_z,
        )
        # The mesh shape is the second one added inside _add_pad (after
        # the ghost slider sphere); pull its index from the builder.
        # We know exactly two shapes were added per pad: slider ghost
        # then pad mesh — the mesh is the last in that batch.
        pad_shape_indices[side] = len(b.shape_type) - 1
        # Light sanity check that the index is what we expect.
        assert len(b.shape_type) - before_shapes == 2

        pad_body_indices[side] = pad_body
        pad_joints.extend([j_x, j_z])
        dof_map[f"{side}_x"] = b.joint_qd_start[j_x]
        dof_map[f"{side}_z"] = b.joint_qd_start[j_z]

    # Stiff PD on every pad joint (matches lift_test's drive).
    for ji in pad_joints:
        dof = b.joint_qd_start[ji]
        b.joint_target_ke[dof] = config.drive.ke
        b.joint_target_kd[dof] = config.drive.kd
        b.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        b.joint_armature[dof] = 0.01

    obj_body, obj_shape, _j_free = objects.add_object(
        b, config.object, obj_cfg, (0.0, 0.0, config.object.start_z)
    )

    # Request the per-contact "force" attribute so we can read solver-
    # applied normal/friction forces in the logger.
    b.request_contact_attributes("force")

    model = b.finalize()
    model.set_gravity(config.gravity)

    # ── 5. CSLC handler attach (mesh-pad path). ──
    if config.contact_model == "cslc":
        mesh_pads_by_shape = {
            pad_shape_indices["left"]: contact_models.make_cslc_pad_from_samples(
                sampled_pts,
                sampled_normals,
                pad_shape_indices["left"],
                k_neighbors=config.pad.k_neighbors,
            ),
            pad_shape_indices["right"]: contact_models.make_cslc_pad_from_samples(
                sampled_pts,
                sampled_normals,
                pad_shape_indices["right"],
                k_neighbors=config.pad.k_neighbors,
            ),
        }
        handler = contact_models.attach_cslc_handler_to_model(
            model, mesh_pads_by_shape, config
        )
        if handler is None:
            raise RuntimeError(
                "CSLC handler construction failed — check pad shape pairs."
            )
        # Realise the pipeline now (inside the patch context) so the
        # auto-discovered ``CSLCHandler._from_model`` call hands back our
        # pre-built handler.  After exit, ``model._collision_pipeline``
        # carries our handler and standard ``model.collide(...)`` works.
        with contact_models.patched_cslc_from_model(handler):
            _ = model.contacts()  # triggers _init_collision_pipeline

        # Per-scene kc recalibration so each pad's aggregate stiffness
        # matches ke_bulk under the expected contact fraction.
        contact_models.recalibrate_kc_per_pad(model, config.cslc.contact_fraction)

    return SceneArtifacts(
        model=model,
        dof_map=dof_map,
        pad_shape_indices=pad_shape_indices,
        object_shape_index=obj_shape,
        pad_body_indices=pad_body_indices,
        object_body_index=obj_body,
        pad_meshes={"left": pad_mesh_tm, "right": pad_mesh_tm},
        pad_lattices={"left": (sampled_pts, sampled_normals),
                      "right": (sampled_pts, sampled_normals)}
        if config.contact_model == "cslc"
        else {},
    )


def _pad_thickness(config: GraspConfig) -> float:
    """Distance from pad body origin to its contact face, along the
    contact normal in body-local coordinates [m].

    Used to convert the user-facing ``approach_gap`` (face-to-object-
    surface clearance at t=0) into the pad body spawn position.

    For box pads the trimesh spans ``[-box_hx, +box_hx]`` in body-local
    x with the body origin at the centre; the contact face is the +x
    side, so the offset is ``box_hx``.

    For dome pads the OBJ is rotated ±π/2 about y so its local +z
    apex maps to body-local ±x.  The shipped asset has z ∈ [0, t]
    (body origin sits on the OBJ's back face, not its centre), so the
    body-origin-to-apex distance is ``mesh.bounds[1, 2]`` (the apex
    z), NOT half the z-extent.  The previous half-extent formula was
    off by 2× and over-penetrated the held object during APPROACH.
    """
    p = config.pad
    if p.kind == "box":
        return p.box_hx
    if p.kind == "dome":
        try:
            mesh = trimesh.load(str(p.dome_obj), force="mesh")
            return float(mesh.bounds[1, 2])
        except Exception:
            return 0.01  # 10 mm — matches the shipped fingertip-scale OBJ.
    raise ValueError(f"Unknown pad kind: {p.kind!r}")
