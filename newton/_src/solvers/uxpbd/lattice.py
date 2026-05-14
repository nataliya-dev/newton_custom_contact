# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MorphIt JSON loader and lattice metadata adapter for UXPBD.

The lattice file format is the same one used by
``ModelBuilder.add_particle_volume`` and ``Box.add_morphit_spheres``: a JSON
document (or in-memory dict) with at minimum the keys ``"centers"`` and
``"radii"``. UXPBD additionally consults optional keys ``"normals"`` and
``"is_surface"``; absent normals default to a unit vector pointing radially
outward from the packing centroid, and absent ``is_surface`` defaults to 1
for every sphere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp


def load_morphit_lattice(volume_data: str | Path | dict[str, Any]) -> dict[str, np.ndarray]:
    """Load a MorphIt sphere packing into NumPy arrays.

    Args:
        volume_data: Either a path to a JSON file, or an already-parsed dict.

    Returns:
        A dict with keys ``centers`` [N, 3] float32, ``radii`` [N] float32,
        ``normals`` [N, 3] float32 (unit), ``is_surface`` [N] uint8.

    Raises:
        ValueError: If ``centers`` and ``radii`` have mismatched lengths or
            ``centers`` is empty.
    """
    if isinstance(volume_data, (str, Path)):
        with open(volume_data) as f:
            data = json.load(f)
    else:
        data = volume_data

    centers = np.asarray(data["centers"], dtype=np.float32)
    radii = np.asarray(data["radii"], dtype=np.float32)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(f"MorphIt lattice: centers ({centers.shape[0]}) and radii ({radii.shape[0]}) length mismatch.")
    if centers.shape[0] == 0:
        raise ValueError("MorphIt lattice: cannot create lattice from zero spheres.")

    if "normals" in data:
        normals = np.asarray(data["normals"], dtype=np.float32)
        if normals.shape != centers.shape:
            raise ValueError(
                f"MorphIt lattice: normals shape {normals.shape} does not match centers shape {centers.shape}."
            )
        # Normalize for safety.
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        lens = np.where(lens > 1e-12, lens, 1.0)
        normals = normals / lens
    else:
        # Default: radial outward from centroid.
        centroid = centers.mean(axis=0, keepdims=True)
        offsets = centers - centroid
        lens = np.linalg.norm(offsets, axis=1, keepdims=True)
        lens = np.where(lens > 1e-12, lens, 1.0)
        normals = (offsets / lens).astype(np.float32)

    if "is_surface" in data:
        is_surface = np.asarray(data["is_surface"], dtype=np.uint8)
        if is_surface.shape[0] != centers.shape[0]:
            raise ValueError(
                f"MorphIt lattice: is_surface length {is_surface.shape[0]} "
                f"does not match centers length {centers.shape[0]}."
            )
    else:
        is_surface = np.ones(centers.shape[0], dtype=np.uint8)

    return {
        "centers": centers,
        "radii": radii,
        "normals": normals,
        "is_surface": is_surface,
    }


def add_lattice_to_builder(
    builder,
    link: int,
    morphit_json: str | Path | dict[str, Any],
    total_mass: float,
    pos: Any = None,
    rot: Any = None,
    k_anchor: float = 1.0e3,
    k_lateral: float = 5.0e2,
    k_bulk: float = 1.0e5,
    damping: float = 2.0,
) -> int:
    """Attach a MorphIt-generated lattice to an articulated link.

    The lattice spheres are added as particles in the builder's particle pool
    and registered in the lattice metadata accumulators. Per-particle mass is
    distributed by sphere volume fraction, matching ``add_particle_volume``.

    Args:
        builder: The :class:`~newton.ModelBuilder` to populate.
        link: The articulated link (``body_index``) that hosts this lattice.
        morphit_json: Path to a MorphIt JSON file, or an in-memory dict.
        total_mass: Total mass distributed across the lattice spheres [kg].
            This mass is metadata only; the host body's inertia is unchanged.
        pos: World-space pose offset for the lattice. Defaults to the origin.
            Pass the link's resting position so the lattice projects correctly
            at t=0.
        rot: Rotation applied to body-frame positions. Defaults to identity.
        k_anchor: Anchor spring stiffness for v2 CSLC [N/m]. Stored, unused
            in Phase 1.
        k_lateral: Lateral coupling stiffness for v2 CSLC [N/m]. Stored.
        k_bulk: Bulk material stiffness for v2 CSLC [N/m]. Stored.
        damping: Hunt-Crossley damping coefficient for v2 CSLC [s/m]. Stored.

    Returns:
        The starting index in the lattice arrays for this link's lattice.
    """
    data = load_morphit_lattice(morphit_json)
    centers = data["centers"]
    radii = data["radii"]
    normals = data["normals"]
    is_surface = data["is_surface"]
    n = centers.shape[0]

    if pos is None:
        pos_v = wp.vec3(0.0, 0.0, 0.0)
    else:
        pos_v = wp.vec3(*pos)
    if rot is None:
        rot_q = wp.quat_identity(float)
    else:
        rot_q = rot

    # Mass distribution by sphere volume fraction (matches add_particle_volume).
    volumes = (4.0 / 3.0) * np.pi * (radii**3)
    total_volume = float(np.sum(volumes))
    if total_volume <= 0.0:
        raise ValueError("Lattice total volume must be positive.")

    lattice_start = len(builder.lattice_link)

    for i in range(n):
        p_local = wp.vec3(float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2]))
        p_world = wp.quat_rotate(rot_q, p_local) + pos_v
        mass_i = total_mass * (float(volumes[i]) / total_volume)
        particle_idx = builder.add_particle(
            p_world,
            wp.vec3(0.0, 0.0, 0.0),
            mass_i,
            float(radii[i]),
        )

        builder.lattice_p_rest.append((float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2])))
        builder.lattice_r.append(float(radii[i]))
        builder.lattice_normal.append((float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2])))
        builder.lattice_is_surface.append(int(is_surface[i]))
        builder.lattice_link.append(int(link))
        builder.lattice_particle_index.append(int(particle_idx))
        builder.lattice_k_anchor.append(float(k_anchor))
        builder.lattice_k_lateral.append(float(k_lateral))
        builder.lattice_k_bulk.append(float(k_bulk))
        builder.lattice_damping.append(float(damping))

    return lattice_start


def _uniform_box_lattice(n: int, half_extent: float = 0.1) -> dict[str, np.ndarray]:
    """Build an ``n x n x n`` uniform sphere packing inscribed in a cube of
    half-extent ``half_extent``.

    Returns the same dict shape as :func:`load_morphit_lattice`.

    Args:
        n: Spheres per axis (n >= 1).
        half_extent: Half-extent of the inscribed cube [m]. Defaults to 0.1.

    Returns:
        A dict with keys ``centers`` [N, 3] float32, ``radii`` [N] float32,
        ``normals`` [N, 3] float32 (outward from centroid), ``is_surface``
        [N] uint8 (1 for boundary spheres, 0 for interior).
    """
    if n < 1:
        raise ValueError("fallback_uniform_n must be >= 1")
    spacing = (2.0 * half_extent) / n
    sphere_r = spacing * 0.5
    coords = np.linspace(-half_extent + sphere_r, half_extent - sphere_r, n)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    centers = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
    radii = np.full(centers.shape[0], sphere_r, dtype=np.float32)
    centroid = centers.mean(axis=0, keepdims=True)
    offsets = centers - centroid
    lens = np.linalg.norm(offsets, axis=1, keepdims=True)
    lens = np.where(lens > 1e-12, lens, 1.0)
    normals = (offsets / lens).astype(np.float32)
    boundary = (np.abs(np.abs(centers) - (half_extent - sphere_r)) < 1.0e-6).any(axis=1)
    is_surface = boundary.astype(np.uint8)
    return {"centers": centers, "radii": radii, "normals": normals, "is_surface": is_surface}


def add_lattice_to_all_links_in_builder(
    builder,
    morphit_json_dir: str | Path,
    *,
    fallback_uniform_n: int | None = 4,
    **lattice_kwargs,
) -> None:
    """Attach a lattice to every body in the builder.

    For each body, look up ``<morphit_json_dir>/<body_label>.json``. If the
    file exists, attach a MorphIt lattice from it. Otherwise, if
    ``fallback_uniform_n`` is not ``None``, attach a uniform ``n^3`` lattice
    inscribed in a unit cube; if ``fallback_uniform_n`` is ``None``, skip
    that body.

    Args:
        builder: The :class:`~newton.ModelBuilder` to populate.
        morphit_json_dir: Directory containing per-link MorphIt JSON files
            named ``<body_label>.json``.
        fallback_uniform_n: Sphere count per axis for the fallback packing,
            or ``None`` to skip bodies with no JSON.
        **lattice_kwargs: Forwarded to :func:`add_lattice_to_builder` per
            body (e.g., ``total_mass=...``, ``k_anchor=...``).
    """
    dir_path = Path(morphit_json_dir)
    body_count = len(builder.body_label)
    for link in range(body_count):
        label = builder.body_label[link]
        candidate = dir_path / f"{label}.json"
        if candidate.is_file():
            add_lattice_to_builder(builder, link=link, morphit_json=str(candidate), **lattice_kwargs)
        elif fallback_uniform_n is not None:
            data = _uniform_box_lattice(fallback_uniform_n)
            add_lattice_to_builder(builder, link=link, morphit_json=data, **lattice_kwargs)
