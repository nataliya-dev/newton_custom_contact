# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Held-object factory.

For now only ``kind="sphere"`` is implemented (tennis-ball-sized
default).  The interface returns the new body's index and shape index
so the scene builder can wire collision filters and CSLC pairs.
"""

from __future__ import annotations

import warp as wp

import newton

from .params import HydroParams, MaterialParams, ObjectParams


def make_object_shape_cfg(
    obj: ObjectParams,
    material: MaterialParams,
    hydro: HydroParams,
    contact_model: str,
) -> newton.ModelBuilder.ShapeConfig:
    """Build the ``ShapeConfig`` for the held object's shape.

    For ``contact_model="hydro"`` the sphere primitive gets
    ``is_hydroelastic=True`` and ``kh``; Newton generates the SDF from
    the analytic sphere geometry.  Other contact models use the bare
    Hunt–Crossley + Coulomb material parameters.
    """
    kwargs = dict(
        ke=material.ke,
        kd=material.kd,
        kf=material.kf,
        mu=material.mu,
        gap=material.gap,
        density=obj.density,
    )
    if contact_model == "hydro":
        kwargs.update(
            kh=hydro.kh,
            is_hydroelastic=True,
            sdf_max_resolution=hydro.sdf_resolution,
        )
    return newton.ModelBuilder.ShapeConfig(**kwargs)


def add_object(
    builder: newton.ModelBuilder,
    obj: ObjectParams,
    shape_cfg: newton.ModelBuilder.ShapeConfig,
    spawn_xyz: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Add the held object as a free-joint body to the builder.

    Args:
        builder: live ``newton.ModelBuilder``.
        obj: object params (currently only ``kind="sphere"``).
        shape_cfg: pre-built shape config (see :func:`make_object_shape_cfg`).
        spawn_xyz: initial centre position of the body.

    Returns:
        ``(body_idx, shape_idx, free_joint_idx)``.
    """
    if obj.kind != "sphere":
        raise NotImplementedError(
            f"Held-object kind {obj.kind!r} not implemented yet "
            "(only 'sphere' for now)."
        )

    body_idx = builder.add_link(
        xform=wp.transform(spawn_xyz, wp.quat_identity()),
        label="object",
    )
    shape_idx = builder.add_shape_sphere(
        body_idx, radius=obj.radius, cfg=shape_cfg
    )
    j_free = builder.add_joint_free(body_idx, label="object_free")
    builder.add_articulation([j_free], label="object")
    return body_idx, shape_idx, j_free
