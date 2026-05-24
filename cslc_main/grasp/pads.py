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

    if p.kind == "dome_param":
        # The builder explicitly returns the cap-face mask so we
        # capture the entire spherical cap (including the wrap-around
        # region for half_angle > pi/2 where ``n_z < 0.3`` would have
        # cut it off) AND exclude the back cylinder's top/bottom caps
        # which the ``n_z > 0.3`` heuristic would otherwise leak.
        mesh, contact_mask = _build_dome_param_trimesh(
            R_pad=p.dome_param_R_pad,
            half_angle=p.dome_param_half_angle,
            back_height=p.dome_param_back_height,
            n_theta=p.dome_param_n_theta,
            n_phi=p.dome_param_n_phi,
        )
        if not contact_mask.any():
            raise RuntimeError(
                "Parametric dome produced no cap faces; check "
                f"half_angle ({math.degrees(p.dome_param_half_angle):.1f} deg)."
            )
        return mesh, contact_mask

    raise ValueError(
        f"Unknown pad kind: {p.kind!r} "
        "(expected 'box', 'dome', or 'dome_param')")


def _build_dome_param_trimesh(*, R_pad: float, half_angle: float,
                              back_height: float, n_theta: int,
                              n_phi: int
                              ) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build a parametric dome fingertip mesh + a cap-face mask.

    Geometry (apex points along local +z; ``pad_shape_xform`` rotates
    this to ±x in world for left/right pads):

    * **Spherical cap**, radius ``R_pad``, polar angle ``theta in
      [0, half_angle]``.  Apex sits at ``(0, 0, R_pad)``.  Sampled on
      an ``(n_theta + 1)`` ring × ``n_phi`` sector grid.
    * **Cylindrical sidewall**, axis along z, top-ring SHARED with the
      cap's base ring (single watertight mesh, not concatenation).
      Radius equals the cap's base-ring radius
      ``R_pad·sin(half_angle)`` for ``half_angle <= pi/2``.
    * **Bottom disk** at the bottom of the sidewall.  Triangle fan
      from a single bottom-centre vertex to the sidewall bottom ring.

    For ``half_angle <= pi/2`` the result is a single watertight mesh
    (``is_volume=True``, ``euler_number=2``) — required by Newton's
    hydroelastic SDF builder, which needs a proper interior to compute
    contact pressure (the v0.9b root cause finding,
    benchmark_spec.md §7 Block B retraction).

    For ``half_angle > pi/2`` (wrap-around / mushroom caps) the
    closure topology is non-trivial because the cap's widest part is
    the equator, not the base ring.  Falls back to the legacy open
    concatenation with a runtime warning; hydro and point contact
    modes will produce zero force on those pads.

    Tessellation: triangle fan at the apex pole, quad strips between
    successive ``theta`` rings (split into two triangles each), quad
    strips for the cylindrical sidewall (split into two triangles
    each), triangle fan at the bottom centre.

    Math matches :func:`cslc_main.theory.cslc_lattice.make_dome` so
    the grasp pad and the theory dome lattice share their cap.

    Returns:
        ``(mesh, cap_face_mask)`` -- the watertight cap+sidewall+bottom
        mesh and a boolean mask of length ``len(mesh.faces)``
        selecting only the spherical-cap faces (the first
        ``n_phi + 2 * n_phi * (n_theta - 1)`` faces by construction;
        used by CSLC's surface-sphere sampler).
    """
    if R_pad <= 0.0 or half_angle <= 0.0 or half_angle >= math.pi:
        raise ValueError(
            f"Require R_pad > 0 and 0 < half_angle < pi; got "
            f"R_pad={R_pad}, half_angle={half_angle}")

    thetas = np.linspace(0.0, half_angle, n_theta + 1)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)

    # ── 1.  Cap vertices: apex + n_theta rings ──
    # Apex at vertex index 0; ring r (r = 0..n_theta-1) starts at
    # index 1 + r * n_phi and contains n_phi vertices.
    vertices: list[list[float]] = [[0.0, 0.0, R_pad]]  # apex
    for theta in thetas[1:]:  # n_theta rings, from ring 0 (near apex) to last (base)
        cap_z = R_pad * math.cos(theta)
        cap_r_xy = R_pad * math.sin(theta)
        for phi in phis:
            vertices.append([cap_r_xy * math.cos(phi),
                             cap_r_xy * math.sin(phi),
                             cap_z])
    cap_base_ring_start = 1 + (n_theta - 1) * n_phi  # last cap ring index

    # ── 2.  Cap faces: apex fan + inter-ring quad strips ──
    cap_faces: list[list[int]] = []
    # Apex fan (n_phi triangles, normals out by +z·R_pad gradient).
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        cap_faces.append([0, 1 + j, 1 + j1])
    # Quad strips between successive rings.
    for r in range(n_theta - 1):
        ring0 = 1 + r * n_phi
        ring1 = ring0 + n_phi
        for j in range(n_phi):
            j1 = (j + 1) % n_phi
            cap_faces.append([ring0 + j, ring1 + j, ring1 + j1])
            cap_faces.append([ring0 + j, ring1 + j1, ring0 + j1])
    n_cap_faces = len(cap_faces)

    # ── 3.  Wrap-around fallback for half_angle > 90° ──
    if half_angle > 0.5 * math.pi:
        import warnings as _warnings
        _warnings.warn(
            f"dome_param with half_angle = {math.degrees(half_angle):.1f}° "
            f"(> 90°) uses the legacy OPEN-cap mesh; hydro and point "
            f"contact modes will produce zero contact force on this pad "
            f"(see benchmark_spec.md §7 Block B v0.9b retraction).  "
            f"Use --pad-kind box for non-CSLC modes at wrap-around half-"
            f"angles.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Legacy: separate cylinder concatenation, not closed.
        cap_vertices_arr = np.asarray(vertices, dtype=np.float64)
        cap = trimesh.Trimesh(
            vertices=cap_vertices_arr,
            faces=np.asarray(cap_faces, dtype=np.int64),
            process=False,
        )
        back_r = R_pad  # equator radius for wrap-around case
        base_ring_z = R_pad * math.cos(half_angle)
        back = trimesh.creation.cylinder(
            radius=back_r, height=back_height, sections=n_phi)
        back.apply_translation([0.0, 0.0, base_ring_z - 0.5 * back_height])
        combined = trimesh.util.concatenate([cap, back])
        cap_face_mask = np.zeros(len(combined.faces), dtype=bool)
        cap_face_mask[:n_cap_faces] = True
        return combined, cap_face_mask

    # ── 4.  Closed-mesh construction for half_angle <= 90° ──
    # Sidewall = cap base ring (already in `vertices`) + new bottom
    # ring at z = base_ring_z - back_height.  Sharing the top ring
    # with the cap is what makes the combined mesh watertight; the
    # legacy concatenation didn't share vertices and trimesh saw
    # the cap as having an open boundary edge.
    base_ring_z = R_pad * math.cos(half_angle)
    base_ring_r = R_pad * math.sin(half_angle)
    bottom_z = base_ring_z - back_height

    bottom_ring_start = len(vertices)
    for phi in phis:
        vertices.append([base_ring_r * math.cos(phi),
                         base_ring_r * math.sin(phi),
                         bottom_z])

    # Sidewall faces: 2 triangles per phi sector, normals pointing
    # radially outward.  Winding [top_j, bot_j, bot_j1] + [top_j,
    # bot_j1, top_j1] gives outward normals under right-hand rule
    # when ring vertices are listed counterclockwise viewed from +z
    # (which they are, since phi increases counterclockwise).
    sidewall_faces: list[list[int]] = []
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        top_j = cap_base_ring_start + j
        top_j1 = cap_base_ring_start + j1
        bot_j = bottom_ring_start + j
        bot_j1 = bottom_ring_start + j1
        sidewall_faces.append([top_j, bot_j, bot_j1])
        sidewall_faces.append([top_j, bot_j1, top_j1])

    # Bottom disk: single centre vertex + fan to bottom ring.
    # Winding [centre, bot_j1, bot_j] gives normals pointing in -z
    # (outward at the bottom face).
    bottom_centre_idx = len(vertices)
    vertices.append([0.0, 0.0, bottom_z])
    bottom_faces: list[list[int]] = []
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        bottom_faces.append([bottom_centre_idx,
                             bottom_ring_start + j1,
                             bottom_ring_start + j])

    all_vertices = np.asarray(vertices, dtype=np.float64)
    all_faces = np.asarray(
        cap_faces + sidewall_faces + bottom_faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces,
                           process=False)

    # Defensive: ensure face windings give consistently outward
    # normals.  fix_normals walks the mesh and flips inconsistent
    # faces; on a correctly-wound watertight mesh it's a no-op.
    mesh.fix_normals()

    # Cap-face mask for CSLC sampling (only the spherical-cap faces).
    cap_face_mask = np.zeros(len(all_faces), dtype=bool)
    cap_face_mask[:n_cap_faces] = True
    return mesh, cap_face_mask

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

    if p.kind in ("dome", "dome_param"):
        # Both dome variants put the contact apex at local +z; rotate
        # ±π/2 about y so the apex maps to world ±x (inward toward the
        # held object on each side).
        angle = +math.pi / 2 if side == "left" else -math.pi / 2
        return wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle),
        )

    raise ValueError(f"Unknown pad kind: {p.kind!r}")
