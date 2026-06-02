# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-target abstraction and concrete samplers.

A CSLC contact target is **any** surface sampled as a discrete set of
points carrying:

  * ``positions``  ∈ ℝ^(M×3) — sample positions [m]
  * ``normals``    ∈ ℝ^(M×3) — outward unit face normals at each sample
  * ``areas``      ∈ ℝ^M     — per-sample Voronoi area on the surface [m²]

Pad sphere ``i`` couples to target sample ``j`` via the signed half-space
overlap (theory.md §3.2)::

    raw_ij  =  r_i − n_face_j · (q_i − t_j)

with no per-sample radius — the radius is carried by the pad spheres.

This module provides:

  :class:`PointSetTarget`
      The target dataclass.

  :func:`make_flat_face_target`
      Regular grid on a flat face.

  :func:`make_sphere_target`
      Fibonacci-spiral surface sampling of a rigid sphere.  Each sample
      carries the radial outward normal; areas are uniform ``4πR²/M``.
      The half-space approximation degrades smoothly off-axis with
      error ``O(r_pad²/R)`` per active pair, fit-for-purpose at
      ``R ≫ r_pad``.

  :func:`make_box_target`
      Per-face area-weighted random sampling of an axis-aligned box.
      Each sample carries its face's outward normal; areas are uniform
      ``total_surface_area / M``.

  :func:`make_mesh_target`
      Area-weighted random surface sampling of a generic trimesh, with
      per-sample face normals.

All samplers are deterministic via ``seed``.  ``trimesh`` is imported
lazily so this module imports without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "PointSetTarget",
    "make_flat_face_target",
    "make_sphere_target",
    "make_box_target",
    "make_mesh_target",
]


# ─────────────────────────────────────────────────────────────────────────
#  Target dataclass
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PointSetTarget:
    """Contact target: discrete surface samples with face normals.

    A pad sphere ``i`` contacts every active sample ``j`` via the
    half-space overlap (theory.md §3.2)::

        raw_ij  =  r_i − n_face_j · (q_i − t_j)

    where ``r_i`` is the pad sphere radius and the target carries
    ``(t_j, n_face_j, A_j)`` per sample — no per-sample radius.

    Attributes:
        positions: (M, 3) sample positions [m].
        normals: (M, 3) outward unit face normals at each sample.
        areas: (M,) per-sample Voronoi area on the underlying surface
            [m²].  If None, treated as uniform ones by callers — pass
            real areas to make the discrete sum approximate the surface
            integral of contact pressure (theory.md eq:F-contact-i).

    Construction-time validation:
        * positions, normals must be 2-D arrays of shape (M, 3).
        * normals must be unit (within 1e-6 tolerance per sample).
        * areas, if supplied, must be 1-D of length M with all entries
          strictly positive.
    """

    positions: np.ndarray
    normals: np.ndarray
    areas: np.ndarray | None = None

    def __post_init__(self):
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (M, 3), got shape {self.positions.shape}")
        M = self.positions.shape[0]
        if M < 1:
            raise ValueError(f"M ≥ 1 required, got M = {M}")
        if self.normals.shape != (M, 3):
            raise ValueError(
                f"normals must be (M, 3) with M = {M}, got {self.normals.shape}")
        norms = np.linalg.norm(self.normals, axis=1)
        if not np.allclose(norms, 1.0, atol=1.0e-6):
            bad = int(np.argmax(np.abs(norms - 1.0)))
            raise ValueError(
                f"normals must be unit vectors; sample {bad} has |n|={norms[bad]:.6f}")
        if self.areas is not None:
            if self.areas.shape != (M,):
                raise ValueError(
                    f"areas must be (M,) with M = {M}, got {self.areas.shape}")
            if np.any(self.areas <= 0.0):
                raise ValueError("all sample areas must be > 0")

    @property
    def M(self) -> int:
        return int(self.positions.shape[0])

    def areas_or_ones(self) -> np.ndarray:
        """Return ``areas`` or an (M,) array of ones if not supplied."""
        if self.areas is None:
            return np.ones(self.M, dtype=np.float64)
        return np.asarray(self.areas, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────
#  Sampler 1: flat face on a regular grid
# ─────────────────────────────────────────────────────────────────────────


def make_flat_face_target(
    centre: np.ndarray,
    normal: np.ndarray,
    span_u: float,
    span_v: float,
    pitch: float,
) -> PointSetTarget:
    """Sample a flat rectangular face with a regular grid.

    The face passes through ``centre`` with outward unit normal
    ``normal``.  Two in-plane orthonormal axes ``u``, ``v`` are chosen
    automatically: ``u`` is whichever of (+x, +y) is more perpendicular
    to ``normal``, projected to the face plane and normalised;
    ``v = normal × u``.

    Samples populate a regular grid of ``ceil(span/pitch) + 1`` points
    along each axis, centred on ``centre``.  Each sample carries area
    ``(span_u · span_v) / M``.

    Args:
        centre: (3,) face centre [m].
        normal: (3,) outward unit face normal.
        span_u: face extent along u-axis [m].
        span_v: face extent along v-axis [m].
        pitch: target sample spacing [m].  Choose pitch ≲ r_pad so the
            tangential locality kernel (width r_pad) sees several
            samples per pad sphere.

    Returns:
        ``PointSetTarget`` with M = N_u × N_v samples, all carrying the
        same face normal.
    """
    centre = np.asarray(centre, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError(f"centre must be (3,), got {centre.shape}")
    if normal.shape != (3,):
        raise ValueError(f"normal must be (3,), got {normal.shape}")
    n_norm = float(np.linalg.norm(normal))
    if not np.isclose(n_norm, 1.0, atol=1e-9):
        normal = normal / n_norm
    if span_u <= 0 or span_v <= 0 or pitch <= 0:
        raise ValueError(
            f"span_u, span_v, pitch must be > 0; got {span_u}, {span_v}, {pitch}")

    # Pick an in-plane u-axis least parallel to ``normal`` to avoid a
    # degenerate cross product.
    x_hat = np.array([1.0, 0.0, 0.0])
    y_hat = np.array([0.0, 1.0, 0.0])
    seed = x_hat if abs(np.dot(normal, y_hat)) > abs(np.dot(normal, x_hat)) else y_hat
    u = seed - np.dot(seed, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    n_u = max(2, int(np.ceil(span_u / pitch)) + 1)
    n_v = max(2, int(np.ceil(span_v / pitch)) + 1)
    u_grid = np.linspace(-span_u / 2.0, span_u / 2.0, n_u)
    v_grid = np.linspace(-span_v / 2.0, span_v / 2.0, n_v)
    uu, vv = np.meshgrid(u_grid, v_grid, indexing="ij")
    positions = (centre[None, None, :]
                 + uu[:, :, None] * u[None, None, :]
                 + vv[:, :, None] * v[None, None, :]).reshape(-1, 3)
    M = positions.shape[0]
    normals = np.broadcast_to(normal, (M, 3)).copy()
    total_area = span_u * span_v
    areas = np.full(M, total_area / M, dtype=np.float64)
    return PointSetTarget(positions=positions, normals=normals, areas=areas)


# ─────────────────────────────────────────────────────────────────────────
#  Sampler 2: sphere surface via Fibonacci spiral
# ─────────────────────────────────────────────────────────────────────────


def make_sphere_target(
    t: np.ndarray,
    R: float,
    n_samples: int,
) -> PointSetTarget:
    """Sample a sphere surface deterministically via the Fibonacci spiral.

    Samples are quasi-uniformly distributed on the sphere of radius
    ``R`` centred at ``t``.  Each sample's normal is the radial outward
    direction; each sample carries the same area ``4πR² / n_samples``.

    Args:
        t: (3,) sphere centre [m].
        R: sphere radius [m] (> 0).
        n_samples: number of surface samples (≥ 4).
    """
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (3,):
        raise ValueError(f"t must be (3,), got {t.shape}")
    if R <= 0.0:
        raise ValueError(f"R must be > 0, got {R}")
    if n_samples < 4:
        raise ValueError(f"n_samples ≥ 4 required, got {n_samples}")

    indices = np.arange(n_samples, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / n_samples         # (-1, +1) cell centres
    r_xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    golden_angle = float(np.pi * (3.0 - np.sqrt(5.0)))
    phi = indices * golden_angle
    unit = np.stack([r_xy * np.cos(phi),
                     r_xy * np.sin(phi),
                     z], axis=1)
    positions = t[None, :] + R * unit
    normals = unit                                        # radial outward
    total_area = 4.0 * np.pi * R * R
    areas = np.full(n_samples, total_area / n_samples, dtype=np.float64)
    return PointSetTarget(positions=positions, normals=normals, areas=areas)


# ─────────────────────────────────────────────────────────────────────────
#  Sampler 3: axis-aligned box surface
# ─────────────────────────────────────────────────────────────────────────


def make_box_target(
    extents: tuple[float, float, float],
    n_samples: int,
    *,
    center: np.ndarray | None = None,
    seed: int = 0,
) -> PointSetTarget:
    """Sample an axis-aligned box surface area-weighted via trimesh.

    Args:
        extents: (W, H, D) box side lengths [m].  Box is axis-aligned.
        n_samples: number of surface points (area-weighted across
            the six faces).
        center: (3,) box centre [m].  Default origin.
        seed: numpy RNG seed for trimesh's area-weighted sampler.

    Returns:
        ``PointSetTarget`` with ``n_samples`` points; normal at each
        sample is the source face's outward normal; area per sample is
        ``total_surface_area / n_samples``.
    """
    import trimesh

    if n_samples < 1:
        raise ValueError(f"n_samples ≥ 1 required, got {n_samples}")
    if any(e <= 0.0 for e in extents):
        raise ValueError(f"all extents must be > 0, got {extents}")

    centre_arr = (np.zeros(3, dtype=np.float64) if center is None
                  else np.asarray(center, dtype=np.float64).reshape(3))

    mesh = trimesh.creation.box(extents=tuple(float(e) for e in extents))
    if not np.allclose(centre_arr, 0.0):
        mesh.apply_translation(centre_arr)

    positions, face_index = trimesh.sample.sample_surface(
        mesh, count=int(n_samples), seed=int(seed))
    positions = np.asarray(positions, dtype=np.float64)
    face_index = np.asarray(face_index, dtype=np.int64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    normals = face_normals[face_index]
    total_area = float(mesh.area)
    areas = np.full(int(n_samples), total_area / float(n_samples),
                    dtype=np.float64)
    return PointSetTarget(positions=positions, normals=normals, areas=areas)


# ─────────────────────────────────────────────────────────────────────────
#  Sampler 4: generic trimesh surface
# ─────────────────────────────────────────────────────────────────────────


def make_mesh_target(
    mesh,
    n_samples: int,
    *,
    seed: int = 0,
) -> PointSetTarget:
    """Sample a generic trimesh surface area-weighted.

    Per-sample normal is the source face's outward normal.  Area per
    sample is ``mesh.area / n_samples`` (uniform-density approximation;
    a true Voronoi area is a separate computation).

    Args:
        mesh: a ``trimesh.Trimesh`` instance.
        n_samples: number of surface samples.
        seed: numpy RNG seed for trimesh's sampler.
    """
    import trimesh  # noqa: F401

    if n_samples < 1:
        raise ValueError(f"n_samples ≥ 1 required, got {n_samples}")

    positions, face_index = mesh.sample(n_samples, return_index=True) \
        if hasattr(mesh, "sample") else \
        __import__("trimesh").sample.sample_surface(
            mesh, count=int(n_samples), seed=int(seed))
    positions = np.asarray(positions, dtype=np.float64)
    face_index = np.asarray(face_index, dtype=np.int64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    normals = face_normals[face_index]
    total_area = float(mesh.area)
    areas = np.full(int(n_samples), total_area / float(n_samples),
                    dtype=np.float64)
    return PointSetTarget(positions=positions, normals=normals, areas=areas)
