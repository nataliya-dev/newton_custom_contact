# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Box-target sampler for the theory-side ``PointSetTarget``.

Step 10 of the theory progression generalises the rigid target from a
single sphere (``RigidTarget`` in :mod:`cslc_main.theory.cslc_theory`)
to a discrete sampling of points on an arbitrary rigid surface
(``PointSetTarget``).  This module is the first concrete sampler -- it
builds a box via trimesh and emits a ``PointSetTarget`` whose points
tile the surface area-weighted across the six faces.

Design choices (with reasons), keeping in mind notes.md Step 11's
symmetry-robustness goal:

1.  ``trimesh.creation.box`` + ``trimesh.sample.sample_surface``.  The
    same path the grasp dome OBJ-loader uses, so calibration / sampling
    conventions stay consistent across pad and target geometry.
    Area-weighted random sampling (deterministic via ``seed``).

2.  Per-point radius ``R_j = radius_factor * mean_spacing`` where
    ``mean_spacing = sqrt(total_area / n_samples)``.  Default
    ``radius_factor = 0.5`` means adjacent target spheres just touch in
    the mean -- enough surface coverage to avoid pad spheres slipping
    between target spheres, without large neighbour-overlap regions
    where the discretisation noise is worst.  Calibration of ``kc`` for
    the integrated stiffness uses the per-point ``areas`` field.

3.  Per-point normal = the source face's outward normal.  trimesh
    provides ``face_normals`` and ``face_index`` from
    ``sample_surface``; we just index in.  These normals are NOT used
    by the basic sphere-vs-sphere overlap check (see ``PointSetTarget``
    docstring) but are the right metadata for the symmetry-robust
    direction work flagged in notes.md Step 11.

4.  ``trimesh`` is imported lazily inside the function so importing
    ``cslc_main.theory.*`` never triggers the dependency.  The
    grasp-side OBJ pipeline already ships trimesh via the
    ``importers`` extra.

The single-point reduction (M = 1, sampled to coincide with a
``RigidTarget``'s centre + radius) matches the closed-form face-on
series-spring law to machine precision -- see
``test_10_pad_vs_box.py`` scene A.
"""

from __future__ import annotations

import numpy as np

from .cslc_theory import PointSetTarget


__all__ = ["make_box_target"]


def make_box_target(
    extents: tuple[float, float, float],
    n_samples: int,
    *,
    center: np.ndarray | None = None,
    radius_factor: float = 0.5,
    seed: int = 0,
) -> PointSetTarget:
    """Sample a rigid box's surface as a ``PointSetTarget``.

    Args:
        extents: ``(W, H, D)`` box side lengths [m].  Box is centred at
            ``center`` (default origin), axis-aligned.
        n_samples: target number of surface points.  trimesh's
            ``sample_surface`` distributes area-weighted across the six
            faces; the actual count returned equals ``n_samples`` (the
            sampler never drops samples).
        center: box centre in world frame [m].  Default ``[0, 0, 0]``.
        radius_factor: per-point sphere radius as a multiple of the
            mean inter-sample spacing.  Default 0.5 (adjacent target
            spheres just touch in the mean).
        seed: numpy RNG seed for ``trimesh.sample.sample_surface``;
            controls sample positions deterministically.

    Returns:
        ``PointSetTarget`` with ``n_samples`` points, each carrying
        position, radius, outward face normal, and per-sample area
        ``total_area / n_samples`` for downstream calibration.
    """
    import trimesh  # lazy: not a theory-module dependency.

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if radius_factor <= 0.0:
        raise ValueError(f"radius_factor must be > 0, got {radius_factor}")
    if any(e <= 0.0 for e in extents):
        raise ValueError(f"all extents must be > 0, got {extents}")

    if center is None:
        center_arr = np.zeros(3, dtype=np.float64)
    else:
        center_arr = np.asarray(center, dtype=np.float64).reshape(3)

    mesh = trimesh.creation.box(extents=tuple(float(e) for e in extents))
    if not np.allclose(center_arr, 0.0):
        mesh.apply_translation(center_arr)

    positions, face_index = trimesh.sample.sample_surface(
        mesh, count=int(n_samples), seed=int(seed))
    positions = np.asarray(positions, dtype=np.float64)
    face_index = np.asarray(face_index, dtype=np.int64)

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    normals = face_normals[face_index]

    total_area = float(mesh.area)
    area_per_point = total_area / float(n_samples)
    mean_spacing = float(np.sqrt(area_per_point))
    radii = np.full(n_samples, radius_factor * mean_spacing, dtype=np.float64)
    areas = np.full(n_samples, area_per_point, dtype=np.float64)

    return PointSetTarget(
        positions=positions,
        radii=radii,
        normals=normals,
        areas=areas,
    )
