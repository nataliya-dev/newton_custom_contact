# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pad geometry — trimesh factories and Lloyd/CVT surface sampling.

Two pad shapes are supported:

* **box** (default): a flat brick built via ``trimesh.creation.box``,
  with its +x face designated as the contact surface.

* **dome**: loaded from an OBJ asset (``assets/pad/pad.obj`` by
  default), with the upward-facing curved face (``n_z > threshold``)
  sampled.  Originally introduced in ``cslc_mujoco/pad_lift_test.py``.

The contact face is sampled with Lloyd's algorithm (centroidal Voronoi
tessellation) via ``point_cloud_utils.sample_mesh_lloyd`` — same code
path for both kinds.  CVT converges to hexagonal close-packing on flat
patches and to as-uniform-as-possible spacing on curved patches, so
the box face produces a near-perfect hexagonal grid and the dome cap
produces a clean radial lattice — both with nearest-neighbour spacing
std/mean ≪ Poisson-disc.

Both paths return ``(mesh, points, normals)`` in pad-local coordinates;
the orientation of the mesh shape WITHIN each pad body is handled by
:func:`pad_shape_xform`, which rotates the mesh so the contact face
points inward toward the held object.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import trimesh
import warp as wp

from .params import PadParams


# ── Mesh factory ────────────────────────────────────────────────────────


def build_pad_trimesh(p: PadParams) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build the pad's trimesh and a face-id mask for its contact surface.

    The "contact surface" is the set of triangles on the pad whose face
    normal points along the pad's designated outward direction:
        box  → +x face   (all triangles with face_normal · x̂ > 0.9)
        dome → +z cap    (triangles with face_normal_z > nz_threshold)

    The orientation of the mesh within each pad body (i.e. rotating that
    contact surface to face the held object) is the caller's job — see
    :func:`pad_shape_xform`.

    Returns:
        mesh: the pad as a single trimesh in pad-local coordinates.
        contact_mask: (n_faces,) bool array, True for triangles on the
            contact surface.
    """
    if p.kind == "box":
        # trimesh.creation.box uses ``extents`` = full edge lengths.
        mesh = trimesh.creation.box(
            extents=(2.0 * p.box_hx, 2.0 * p.box_hy, 2.0 * p.box_hz)
        )
        face_normals = np.asarray(mesh.face_normals)
        contact_mask = face_normals[:, 0] > 0.9
        return mesh, contact_mask

    if p.kind == "dome":
        path = Path(p.dome_obj)
        if not path.exists():
            raise FileNotFoundError(
                f"Dome OBJ not found: {path}. "
                "Override via GraspConfig.pad.dome_obj or switch to "
                "pad.kind='box'."
            )
        mesh = trimesh.load(str(path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(
                f"Expected a triangle mesh from {path}, got {type(mesh)}")
        contact_mask = np.asarray(mesh.face_normals)[
            :, 2] > p.dome_nz_threshold
        if not contact_mask.any():
            raise RuntimeError(
                f"No dome faces with n_z > {p.dome_nz_threshold}; "
                "check OBJ orientation."
            )
        return mesh, contact_mask

    raise ValueError(
        f"Unknown pad kind: {p.kind!r} (expected 'box' or 'dome')")

#
# ── Contact-face sampling ────────────────────────────────────────────────


def sample_pad_contact_face(
    mesh: trimesh.Trimesh,
    contact_mask: np.ndarray,
    p: PadParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd/CVT-sample the contact face; return (points, normals).

    Uses ``point_cloud_utils.sample_mesh_lloyd`` to draw exactly
    ``p.n_samples`` points on the contact submesh.  Lloyd's algorithm
    iteratively moves each sample to the centroid of its surface
    Voronoi cell (centroidal Voronoi tessellation, CVT;
    https://en.wikipedia.org/wiki/Lloyd%27s_algorithm).  CVT minimises
    the quadratic energy ∫ ρ(x) ‖x − c_i(x)‖² dx, and on a flat patch
    the minimiser is hexagonal close-packing — i.e. the lattice you'd
    get from a perfect grid in 2-D — so the box face produces a near-
    hexagonal grid and the dome cap produces an as-uniform-as-possible
    radial pattern.  Compared to Poisson-disc, NN-spacing std/mean
    drops ≈ 3× on the box and ≈ 20 % on the dome.

    Points and normals are returned in pad-local coordinates as
    ``float32`` arrays of shape ``(N, 3)``.  Normals are the face
    normal of the submesh triangle nearest to each sample (via
    :func:`trimesh.proximity.closest_point`), so they encode the local
    outward direction of the contact surface.

    Args:
        mesh: the pad trimesh in pad-local frame.
        contact_mask: triangle subset to sample (from
            :func:`build_pad_trimesh`).
        p: pad params (uses ``n_samples``).  ``p.seed`` is unused —
            ``sample_mesh_lloyd`` is deterministic given (v, f, n).

    Returns:
        points: (N, 3) float32 samples on the contact face.
        normals: (N, 3) float32 outward normals.
    """
    face_ids = np.where(contact_mask)[0]
    sub = mesh.submesh([face_ids], append=True)
    if not isinstance(sub, trimesh.Trimesh):
        raise RuntimeError("submesh() did not return a single Trimesh")

    v = np.asarray(sub.vertices, dtype=np.float64)
    f = np.asarray(sub.faces, dtype=np.int32)
    pts = np.asarray(pcu.sample_mesh_lloyd(v, f, int(p.n_samples)))

    # Lloyd returns points only; recover per-sample face normals by
    # locating each point on the submesh via closest-point query.
    _, _, tri_id = trimesh.proximity.closest_point(sub, pts)
    normals = np.asarray(sub.face_normals)[tri_id]
    return pts.astype(np.float32), normals.astype(np.float32)


# ── Pad-shape orientation within the body frame ──────────────────────────


def pad_shape_xform(p: PadParams, side: str) -> wp.transform:
    """Return the mesh-shape transform that orients the contact face inward.

    Each pad body is world-aligned (its only motion is via prismatic
    joints).  Only the mesh shape inside the body is rotated, so the PD
    joints see a consistent frame regardless of pad kind.  Convention:

        * Left pad  is at world −x → contact face must point along world +x.
        * Right pad is at world +x → contact face must point along world −x.

    For ``pad.kind="box"``: the trimesh's local +x face already points
    along the pad body's local +x, so the left pad takes identity; the
    right pad gets a 180° rotation about z to flip its +x face to −x.

    For ``pad.kind="dome"``: the OBJ's contact face is along local +z;
    the left pad rotates +π/2 about y (mapping local +z → world +x),
    and the right pad rotates −π/2 about y (local +z → world −x).
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    if p.kind == "box":
        if side == "left":
            return wp.transform_identity()
        # Right: 180° around z → local +x face flips to world −x (inward).
        return wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
        )

    if p.kind == "dome":
        angle = +math.pi / 2 if side == "left" else -math.pi / 2
        return wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle),
        )

    raise ValueError(f"Unknown pad kind: {p.kind!r}")
