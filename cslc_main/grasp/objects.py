# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Held-object factory.

Two kinds supported:
  * ``kind="sphere"``: tennis-ball default (radius 33.5 mm, mass ~58 g).
  * ``kind="box"`` (C2e): axis-aligned cube/box with a uniformly-sampled
    point-set surface used by the CSLC point-set contact path.

The interface returns the new body's index and shape index so the scene
builder can wire collision filters and CSLC pairs.

Point-set sampling for box targets
----------------------------------
:func:`make_box_target` generates one target sphere per uniform face
sample (6 faces, ``ceil(2h / pitch)`` samples per axis per face).  Each
target sphere has radius ``pitch / 2`` so adjacent samples just tile
the face without overlap.  Positions are in the box's local frame; the
CSLC handler transforms them to world per-step via the box body's
``body_q``.

:func:`compute_k_max` derives the per-pad-sphere contact-buffer budget
``K_max`` from the active-set inclusion radius (``r_lat + R_target +
INCLUSION_FACTOR * eps``, where ``INCLUSION_FACTOR = 50`` comes from
:mod:`cslc_main.theory.cslc_theory`).  The handler's runtime
truncation counter (``cslc_handler._launch_vs_point_set``) is the
backstop if this formula under-counts on a new geometry.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton

from cslc_main.theory.cslc_theory import INCLUSION_FACTOR

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
    # C2 ke-split: the object's ke flows to MuJoCo as
    # ``pair.other_ke`` -> ``target_ke`` in the CSLC kc_series
    # composition (numerical / regularisation knob).  Under hydro,
    # ``kh`` is the physical compliance knob and ``ke`` here is the
    # constraint regularisation.
    kwargs = dict(
        ke=material.ke_target_constraint,
        kd=material.kd,
        kf=material.kf,
        mu=material.mu,
        gap=material.gap,
        density=obj.density,
    )
    if contact_model == "hydro":
        kwargs.update(
            kh=material.kh,
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
    if obj.kind not in ("sphere", "box"):
        raise NotImplementedError(
            f"Held-object kind {obj.kind!r} not implemented yet "
            "(supported: 'sphere', 'box')."
        )

    body_idx = builder.add_link(
        xform=wp.transform(spawn_xyz, wp.quat_identity()),
        label="object",
    )
    if obj.kind == "sphere":
        shape_idx = builder.add_shape_sphere(
            body_idx, radius=obj.radius, cfg=shape_cfg
        )
    else:  # box
        hx, hy, hz = obj.box_half_extents
        shape_idx = builder.add_shape_box(
            body_idx, hx=hx, hy=hy, hz=hz, cfg=shape_cfg
        )
    j_free = builder.add_joint_free(body_idx, label="object_free")
    builder.add_articulation([j_free], label="object")
    return body_idx, shape_idx, j_free


# ── Box point-set sampling (C2e) ─────────────────────────────────────────


_FACE_SPECS: dict[str, tuple[int, int, int, int, tuple[float, float, float]]] = {
    # face_name -> (fixed_axis, fixed_sign, a_axis, b_axis, normal_xyz)
    "+x": (0, +1, 1, 2, (+1.0, 0.0, 0.0)),
    "-x": (0, -1, 1, 2, (-1.0, 0.0, 0.0)),
    "+y": (1, +1, 0, 2, (0.0, +1.0, 0.0)),
    "-y": (1, -1, 0, 2, (0.0, -1.0, 0.0)),
    "+z": (2, +1, 0, 1, (0.0, 0.0, +1.0)),
    "-z": (2, -1, 0, 1, (0.0, 0.0, -1.0)),
}


def make_box_target(
    half_extents: tuple[float, float, float],
    pitch: float,
    faces: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Uniformly sample selected box faces into a point set for CSLC contact.

    Each face is tiled with cell-centre samples at the requested pitch.
    Cell count per axis is ``round(2*h / pitch)`` (minimum 1); actual
    cell step is ``2*h / cell_count`` -- usually equal to ``pitch``,
    differs slightly when ``2*h / pitch`` isn't an integer.

    Args:
        half_extents: box half-sides ``(hx, hy, hz)`` [m] in body-local
            frame.  A 25 mm cube uses ``(0.0125, 0.0125, 0.0125)``.
        pitch: target nominal spacing between adjacent samples [m].
            ``0.001`` (1 mm) at production gives 625 samples per 25 mm face.
        faces: which faces to sample, as a tuple of names from
            ``{"+x", "-x", "+y", "-y", "+z", "-z"}``.  Default ``None``
            samples all 6 faces (3750 points for default geometry).
            For a left-right grasp pipeline, pass ``("+x", "-x")`` to
            sample only the 1250 points the pads can physically reach
            -- the other faces would still be in the active-set
            inclusion radius (~27 mm at production eps) for pad spheres
            engaged at +/-x, padding the K_max budget and the Jacobi
            inner-loop cost without any wrench contribution.

    Returns:
        Dict with three ``np.ndarray`` fields, all body-local:
          * ``positions``: ``(N, 3) float32``
          * ``radii``:     ``(N,) float32``  -- uniform ``pitch / 2``
          * ``normals``:   ``(N, 3) float32``  -- per-face outward normal

    Notes:
        * ``radii = pitch / 2`` makes adjacent target spheres just tile
          their face without overlap.
        * ``normals`` is populated for API completeness (C3+ per-pair
          friction may consume it); the C2 production kernels use the
          line-of-centres direction ``(t - q)/||t-q||`` and ignore it.
    """
    if pitch <= 0.0:
        raise ValueError(f"pitch must be > 0, got {pitch}")
    hx, hy, hz = (float(h) for h in half_extents)
    if faces is None:
        faces = ("+x", "-x", "+y", "-y", "+z", "-z")
    unknown = [f for f in faces if f not in _FACE_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown box-face name(s) {unknown}; valid: "
            f"{sorted(_FACE_SPECS)}"
        )

    def axis_count(extent: float) -> int:
        return max(1, int(round(2.0 * extent / pitch)))

    halfs = (hx, hy, hz)

    pos_chunks, nrm_chunks = [], []
    for face_name in faces:
        fixed_axis, fixed_sign, a_axis, b_axis, normal_xyz = _FACE_SPECS[face_name]
        fixed_value = fixed_sign * halfs[fixed_axis]
        a_extent = halfs[a_axis]
        b_extent = halfs[b_axis]
        na = axis_count(a_extent)
        nb = axis_count(b_extent)
        step_a = 2.0 * a_extent / na
        step_b = 2.0 * b_extent / nb
        a_coords = -a_extent + 0.5 * step_a + step_a * np.arange(na)
        b_coords = -b_extent + 0.5 * step_b + step_b * np.arange(nb)
        A, B = np.meshgrid(a_coords, b_coords, indexing="ij")
        pts = np.zeros((na * nb, 3), dtype=np.float32)
        pts[:, fixed_axis] = fixed_value
        pts[:, a_axis] = A.ravel()
        pts[:, b_axis] = B.ravel()
        nrm = np.tile(np.array(normal_xyz, dtype=np.float32), (na * nb, 1))
        pos_chunks.append(pts)
        nrm_chunks.append(nrm)

    positions = np.concatenate(pos_chunks, axis=0).astype(np.float32)
    normals   = np.concatenate(nrm_chunks, axis=0).astype(np.float32)
    radii     = np.full(positions.shape[0], pitch * 0.5, dtype=np.float32)
    return {"positions": positions, "radii": radii, "normals": normals}


def box_face_area(half_extents: tuple[float, float, float], face: str) -> float:
    """Surface area [m^2] of one box face, looked up by name."""
    if face not in _FACE_SPECS:
        raise ValueError(f"Unknown face {face!r}; valid: {sorted(_FACE_SPECS)}")
    _, _, a_axis, b_axis, _ = _FACE_SPECS[face]
    return 4.0 * half_extents[a_axis] * half_extents[b_axis]


def box_surface_area(half_extents: tuple[float, float, float],
                     faces: tuple[str, ...] | None = None) -> float:
    """Sum of selected box face areas [m^2].  ``None`` -> all 6 faces."""
    if faces is None:
        faces = ("+x", "-x", "+y", "-y", "+z", "-z")
    return sum(box_face_area(half_extents, f) for f in faces)


def compute_k_max(
    *,
    lattice_radii_max: float,
    target_radii_max: float,
    smoothing_eps: float,
    target_count: int,
    target_surface_area: float,
    pad_face_clip_area: float | None = None,
    slack: int = 32,
) -> int:
    """Derive the per-pad-sphere contact-buffer budget for a point-set pair.

    A target point j contributes a non-trivial wrench to pad sphere i
    iff their centres are within the active-set inclusion radius::

        r_inclusion = r_lat + R_target + INCLUSION_FACTOR * smoothing_eps

    (see :data:`cslc_main.theory.cslc_theory.INCLUSION_FACTOR` for the
    derivation: the kernel's algebraic ``smooth_step`` surrogate hits
    its 1e-4 emission gate at ``raw = -INCLUSION_FACTOR * eps``, the
    same threshold as the active-set skip ``-50*eps``).  The maximum
    number of points within ``r_inclusion`` of any single pad sphere
    is bounded by the area of the inclusion disk times the target
    surface density.

    For a box target the inclusion disk is clipped by the face boundary
    -- a pad sphere centred on a face can never see points beyond the
    face's perimeter.  Pass ``pad_face_clip_area`` (= the face area) to
    apply this clip; otherwise the formula treats the target surface as
    an infinite plane and over-counts.

    Args:
        lattice_radii_max: max pad-sphere lattice radius [m].  Read
            from the CSLCLattice's ``radii`` array.
        target_radii_max: max target-sphere radius [m].  For
            :func:`make_box_target` outputs this is ``pitch / 2``.
        smoothing_eps: kernel smoothing width [m] (CSLCData.smoothing_eps).
        target_count: total number of target points (sum across faces
            for box targets).
        target_surface_area: total surface area of the target [m^2]
            (6 * (2hx)*(2hy) + ... for box).
        pad_face_clip_area: optional [m^2] clip cap for the inclusion
            disk.  Use the largest single face area for a box; omit for
            unclipped (infinite-plane) sizing.
        slack: extra slot count beyond the bounded estimate.  Default
            32 covers boundary effects (pad sphere near a corner of the
            face sees a partial adjacent face's worth of points).

    Returns:
        Integer ``K_max`` to use in :class:`CSLCShapePair`.  Memory
        cost per pair is ``n_surface_pad_spheres * K_max * ~96 bytes``;
        production scenes are O(10 MB), well within GPU budgets.
    """
    r_inclusion = (
        lattice_radii_max + target_radii_max
        + INCLUSION_FACTOR * smoothing_eps
    )
    density = target_count / target_surface_area
    inclusion_area = math.pi * r_inclusion * r_inclusion
    if pad_face_clip_area is not None:
        inclusion_area = min(inclusion_area, pad_face_clip_area)
    return int(math.ceil(inclusion_area * density)) + slack
