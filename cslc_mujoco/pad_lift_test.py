#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OBJ-pad lift test — mirrors cslc_mujoco/lift_test.py with mesh pads.

Same articulated scene and phase trajectory as lift_test.py, but the box
pads are replaced with the curved mesh from ``assets/pad/pad.obj``.  The
curved contact face is Poisson-disc sampled and the per-sample outward
normals (computed from the parent triangles, not predefined) follow the
pad bodies frame-by-frame.  These samples are visualization-only for
now; wiring them into the CSLC handler in place of the hard-coded
box-face grid is the next step.

Phase trajectory (identical to lift_test.py)::

      ┌─┐         ┌─┐
      │L│ ◄────── │R│      APPROACH    pads move inward
      └─┘         └─┘
      ┌─┐ ┌───┐ ┌─┐
      │L│ │obj│ │R│        SQUEEZE     pads press a few mm in
      └─┘ └───┘ └─┘
        ↑       ↑
      ┌─┐ ┌───┐ ┌─┐        LIFT        pads (and the gripped object)
      │L│ │obj│ │R│                    rise together
      └─┘ └───┘ └─┘
      ════════════         HOLD        pads stationary in the air

Geometry
--------
The OBJ pad is a 20 x 20 x 10 mm primitive with a flat base at local
z = 0 and a curved contact apex at local z = +0.01.  Each pad body sits
at world (+/-pad_center_x, 0, sphere_start_z) with identity orientation;
the MESH SHAPE is rotated by Ry(+/-pi/2) inside the body so the curved
face faces inward.  Curved apex sits at world x = +/-0.05 m — the same
inner-face position the box pads in lift_test.py occupy.

Contact model is selectable: "point" runs Hunt–Crossley mesh-vs-sphere
point contact (single closest-point pair); "hydro" runs the hydroelastic
pressure-field model on both bodies, with an SDF built on the pad mesh
at startup.  CSLC integration replaces the PDS samples + normals with a
working contact lattice in the next step.

Usage
-----
  uv run cslc_mujoco/pad_lift_test.py --viewer gl
  uv run cslc_mujoco/pad_lift_test.py --viewer gl --contact-model cslc
  uv run cslc_mujoco/pad_lift_test.py --viewer gl --contact-model hydro
  uv run cslc_mujoco/pad_lift_test.py --viewer gl --contact-model hydro --kh 1e8
  uv run cslc_mujoco/pad_lift_test.py --viewer gl --solver semi
  uv run cslc_mujoco/pad_lift_test.py --viewer gl --n-samples 250
"""

from __future__ import annotations
from contextlib import contextmanager

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.geometry import HydroelasticSDF

from cslc_mujoco.common import (
    HAS_MUJOCO,
    _log,
    _section,
    count_active_contacts,
    inspect_model,
    make_solver,
)


PAD_OBJ = Path(__file__).resolve().parent.parent / "assets" / "pad" / "pad.obj"


# ── OBJ + Poisson-disc sampling ──────────────────────────────────────────


def load_pad_mesh(path: Path) -> trimesh.Trimesh:
    """Load the OBJ as a single Trimesh (force-flatten if it loads as a scene)."""
    m = trimesh.load(str(path), force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise RuntimeError(
            f"Expected a triangle mesh from {path}, got {type(m)}")
    return m


def sample_curved_face(
    mesh: trimesh.Trimesh,
    n_samples: int,
    nz_threshold: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-disc sample the curved (n_z > threshold) face.

    Returns ``(points, normals)`` in pad-local coordinates as float32
    arrays of shape (N, 3).  Normals are the parent-triangle face
    normals (cross product of two triangle edges) — computed from the
    geometry, not predefined.
    """
    curved_face_ids = np.where(mesh.face_normals[:, 2] > nz_threshold)[0]
    if len(curved_face_ids) == 0:
        raise RuntimeError(
            f"No faces with n_z > {nz_threshold}; check pad orientation.")

    sub = mesh.submesh([curved_face_ids], append=True)
    if not isinstance(sub, trimesh.Trimesh):
        raise RuntimeError("submesh() did not return a single Trimesh")

    pts, face_idx = trimesh.sample.sample_surface_even(
        sub, count=n_samples, seed=seed)
    normals = sub.face_normals[face_idx]
    return pts.astype(np.float32), normals.astype(np.float32)


# ── Scene parameters (mirror lift_test.py SceneParams) ───────────────────


@dataclass
class SceneParams:
    """Knobs for the OBJ-pad lift scene.  Defaults mirror lift_test.py."""

    # OBJ pad + sampling
    pad_obj: Path = PAD_OBJ
    n_samples_per_pad: int = 150
    nz_threshold: float = 0.3

    # Sphere (gripped object)
    sphere_radius: float = 0.03
    sphere_density: float = 100.0
    sphere_start_z: float = 0.05     # sphere spawn height (falls to z~=r)

    # Pad geometry / mass
    # The OBJ pad spans local z = [0, 0.01]; after Ry(+/-pi/2) the curved
    # apex is offset 0.01 m along +/-x from the pad body origin.  Setting
    # pad_center_x = sphere_radius + pad_thickness + initial_gap places
    # the curved apex 20 mm from the sphere surface at t = 0, matching
    # lift_test's box-pad geometry (approach_gap/2 + pad_hx = 0.06).
    pad_center_x: float = 0.06
    # Pad mesh spans body_z in [-0.01, +0.01] (10 mm half-extent).  Set
    # pad_center_z to the sphere's settled centre (~ sphere_radius above
    # the ground) so the curved apex meets the sphere at its equator
    # rather than 20 mm above it.  lift_test's box pad was 100 mm tall in
    # z and could afford to sit at z = 0.05 and still span the sphere;
    # the OBJ pad is much shorter so it needs to be vertically centred.
    pad_center_z: float = 0.03
    pad_thickness: float = 0.01      # OBJ pad x-extent after rotation
    pad_density: float = 1000.0

    # Phase timing — identical to lift_test.py
    approach_speed: float = 20e-3 / 1.5   # 13.33 mm/s -> 20 mm in 1.5 s
    approach_duration: float = 1.5
    squeeze_speed: float = 1e-3 / 0.5     # 2 mm/s -> 1 mm in 0.5 s
    squeeze_duration: float = 0.5
    lift_speed: float = 0.015
    lift_duration: float = 1.5
    lift_ramp_duration: float = 0.25      # C1 ramp at both LIFT endpoints
    hold_duration: float = 1.0

    # Material — shared baseline (sphere + ground keep this value).
    ke: float = 5.0e4
    kd: float = 5.0e2
    kf: float = 100.0
    mu: float = 0.5

    # Point-contact PAD stiffness override.
    #
    # Newton's mesh-vs-sphere narrow phase emits one contact per triangle
    # in the proximity band, not one closest-point pair as box-vs-sphere
    # does.  At the 1 mm face_pen calibration pose this gives N = 32
    # contact polygons per pad — so with the shared `ke = 5e4`, the
    # per-pad aggregate stiffness becomes 32×5e4 = 1.6 MN/m, 32× too
    # stiff vs the lift_test box-pad reference (`ke_bulk = 50 000 N/m`).
    #
    # Calibrated by `calibrate_pad_stiffness(..., "point")`:
    #   k_pad with pad_ke = 5e4   →  1.6 MN/m  (3200 % of ke_bulk)
    #   pad_ke_fair = ke·ke_bulk/k_pad = 5e4·5e4 / 1.6e6 = 1562 N/m
    #
    # Only the pad shape uses this; sphere/ground keep the baseline ke
    # so sphere-ground contact stays stiff during APPROACH/SQUEEZE.
    # Re-run `pad_lift_test.py --calibrate-kh --contact-model point` if
    # pad geometry, sphere radius, or pad_thickness changes.
    pad_ke: float = 1562.0

    # Joint drive (stiff PD position tracking)
    drive_ke: float = 5.0e4
    drive_kd: float = 1.0e3

    # Integration
    dt: float = 1.0 / 500.0
    gravity: tuple = (0.0, 0.0, -9.81)

    # CSLC (Compliant Sphere Lattice Contact) parameters.  These mirror
    # `lift_test.py`'s fair-calibration defaults, except `cslc_spacing`
    # is recomputed at pad-build time as the mean nearest-neighbour
    # distance of the Poisson-disc samples (instead of being a free
    # parameter, since spacing is determined by N_samples on the curved
    # face).
    cslc_ka: float = 25_000.0
    cslc_kl: float = 500.0
    cslc_dc: float = 2.0
    cslc_n_iter: int = 20
    cslc_alpha: float = 0.6
    # Contact-fraction prior used by `recalibrate_cslc_kc_per_pad` to
    # set kc from ke_bulk.  Matched to lift_test.py.
    cslc_contact_fraction: float = 0.025
    # k for the k-NN neighbour graph used by the lattice Laplacian L.
    # k=6 is the natural choice for a 2-D Poisson-disc scatter (a typical
    # blue-noise neighbour has ~6 neighbours within its Delaunay cell).
    cslc_k_neighbors: int = 6

    # Hydroelastic contact (used when contact_model == "hydro").
    #
    # lift_test.py uses kh = 5.3e8 Pa for its flat box pad, calibrated so
    # kh_eff · A_patch(1 mm) = ke_bulk = 50 000 N/m on a 30 mm sphere
    # (A_patch ≈ π·(2·r·pen) ≈ 188 mm²).  The OBJ pad is a 10 mm-tall
    # dome — the dome-vs-sphere contact patch is much smaller than the
    # flat-pad patch at the same penetration, so kh must be scaled up
    # to preserve the per-pad fair invariant.
    #
    # Calibrated by `calibrate_hydro_kh()` at face_pen = 1 mm:
    #   k_left = k_right = 7287 N/m per pad  (with kh = 5.3e8)
    #   fairness factor 14.6 % of ke_bulk
    #   → kh_fair = kh · ke_bulk / k_pad = 5.3e8 · 50 000 / 7287
    #             ≈ 3.64e9 Pa   (6.86× the box-pad default)
    #
    # Re-run `pad_lift_test.py --calibrate-kh` if pad geometry, sphere
    # radius, or pad_thickness changes.
    kh: float = 3.64e9
    sdf_resolution: int = 64

    # Viz
    sample_radius: float = 0.0005   # 0.5 mm sphere markers
    arrow_length: float = 0.005     # 5 mm normal-arrow length
    seed: int = 0
    add_ground: bool = True

    @property
    def sphere_mass(self) -> float:
        return self.sphere_density * (4.0 / 3.0) * math.pi * self.sphere_radius ** 3

    @property
    def approach_steps(self) -> int:
        return int(self.approach_duration / self.dt)

    @property
    def squeeze_steps(self) -> int:
        return int(self.squeeze_duration / self.dt)

    @property
    def lift_steps(self) -> int:
        return int(self.lift_duration / self.dt)

    @property
    def hold_steps(self) -> int:
        return int(self.hold_duration / self.dt)

    @property
    def total_steps(self) -> int:
        return (self.approach_steps + self.squeeze_steps
                + self.lift_steps + self.hold_steps)

    def phase_of(self, step: int) -> tuple[str, int]:
        s = step
        for name, dur in (("APPROACH", self.approach_steps),
                          ("SQUEEZE",  self.squeeze_steps),
                          ("LIFT",     self.lift_steps),
                          ("HOLD",     self.hold_steps)):
            if s < dur:
                return name, s
            s -= dur
        return "HOLD", 0

    def dump(self) -> None:
        _section("SCENE PARAMETERS")
        m = self.sphere_mass
        _log(f"Sphere: r={self.sphere_radius*1e3:.1f}mm  "
             f"mass={m*1e3:.1f}g  weight={m*9.81:.3f}N")
        _log(f"Pad   : OBJ {self.pad_obj.name}  thickness={self.pad_thickness*1e3:.1f}mm "
             f"density={self.pad_density:.0f}  samples/pad={self.n_samples_per_pad}")
        _log(f"Phases: approach={self.approach_duration}s  "
             f"squeeze={self.squeeze_duration}s  lift={self.lift_duration}s  "
             f"hold={self.hold_duration}s")
        _log(f"Material: ke={self.ke:.0f}  kd={self.kd:.0f}  μ={self.mu:.2f}")
        _log(f"Drive   : ke={self.drive_ke:.0f}  kd={self.drive_kd:.0f}")
        _log(f"Steps: {self.total_steps} total  dt={self.dt*1e3:.2f}ms")


# ── Rotation helper ──────────────────────────────────────────────────────


def _Ry(angle: float) -> np.ndarray:
    """Right-handed rotation about +y.  R_y(+pi/2) maps local +z -> world +x."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], np.float32)


# ── Body-index layout (mirrors lift_test.py) ─────────────────────────────
#
#   0: left_slider    2: right_slider    4: sphere
#   1: left_pad       3: right_pad
# ─────────────────────────────────────────────────────────────────────────

LEFT_SLIDER = 0
LEFT_PAD = 1
RIGHT_SLIDER = 2
RIGHT_PAD = 3
SPHERE_BODY = 4


def _shape_rotations() -> tuple[np.ndarray, np.ndarray, wp.quat, wp.quat]:
    """Mesh-shape rotations within each pad body's frame.

    The pad body is world-aligned (prismatic joints only).  Only the
    mesh shape is rotated so the curved face points inward:
      Left  pad: R_y(+pi/2)  -> local +z -> world +x  (inward from -x)
      Right pad: R_y(-pi/2)  -> local +z -> world -x  (inward from +x)
    """
    R_L = _Ry(+math.pi / 2)
    R_R = _Ry(-math.pi / 2)
    q_L = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), +math.pi / 2)
    q_R = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.pi / 2)
    return R_L, R_R, q_L, q_R


# ── CSLC mesh-pad construction + monkey-patch ────────────────────────────
#
# Newton's `CSLCHandler._from_model` only knows how to auto-generate a
# lattice pad for BOX shapes (a 2-D face grid).  For our OBJ mesh pad
# we want to plug in a different lattice geometry: the Poisson-disc
# samples drawn from the curved contact face, with normals computed
# from the parent triangles and connectivity from a k-nearest-neighbour
# graph in 3-D.  We do this by building the `CSLCPad` objects ourselves
# *before* the collision pipeline is constructed, then temporarily
# replacing `_from_model` with a version that uses our pre-built pads
# instead of the box-face factory.
# ─────────────────────────────────────────────────────────────────────────


def _make_mesh_cslc_pad(
    local_pts: np.ndarray,
    local_norms: np.ndarray,
    shape_index: int,
    k_neighbors: int = 6,
):
    """Build a CSLCPad whose lattice is the Poisson-disc scatter on the
    pad's curved contact face.

    Each sample becomes one surface sphere.  The neighbour graph is
    k-NN in 3-D — for a Poisson-disc scatter on a 2-D manifold this is
    a close approximation of the surface Delaunay graph, and gives the
    same per-vertex valency profile (~k) the box-face grid relies on.

    Spacing is set to the mean nearest-neighbour distance (a Poisson-
    disc invariant: ~r_PDS for a tight disc).  Sphere radius = spacing/2
    matches Newton's box convention so the kc calibration formula in
    `recalibrate_cslc_kc_per_pad` reads the right per-sphere stiffness.

    Args:
        local_pts: (N, 3) PDS samples in pad-local coordinates.
        local_norms: (N, 3) outward normals from parent triangles.
        shape_index: which Newton shape this lattice belongs to.
        k_neighbors: k for the k-NN neighbour graph.

    Returns:
        A `CSLCPad` ready for `CSLCData.from_pads([...])`.
    """
    from scipy.spatial import cKDTree
    from newton._src.geometry.cslc_data import CSLCPad

    n = len(local_pts)
    pts32 = local_pts.astype(np.float32)
    norms32 = local_norms.astype(np.float32)

    # k-NN graph in 3-D: query k+1 to drop self.
    tree = cKDTree(pts32)
    dists, idxs = tree.query(pts32, k=k_neighbors + 1)
    spacing = float(np.mean(dists[:, 1]))   # mean NN distance (excluding self)
    sphere_radius = spacing * 0.5
    neighbor_indices = [idxs[i, 1:].astype(np.int32) for i in range(n)]

    return CSLCPad(
        positions=pts32,
        radii=np.full(n, sphere_radius, dtype=np.float32),
        is_surface=np.ones(n, dtype=bool),
        outward_normals=norms32,
        neighbor_indices=neighbor_indices,
        shape_index=shape_index,
        grid_shape=(n, 1, 1),    # placeholder; CSLCData doesn't use this
        spacing=spacing,
        sphere_radius=sphere_radius,
    )


def _build_cslc_handler_with_mesh_pads(model, mesh_pads_by_shape):
    """Build a CSLCHandler whose lattice pads come from `mesh_pads_by_shape`.

    Mirrors `CSLCHandler._from_model` (the parts that aren't shape-type-
    specific) and substitutes our pre-built `CSLCPad` objects for any
    MESH+CSLC shape.  Box+CSLC shapes still go through `create_pad_for_box_face`.
    Returns the handler, or None if there are no usable CSLC pairs.
    """
    from newton._src.geometry.cslc_handler import (
        CSLCHandler, CSLCShapePair, _CSLC_FLAG, _GEOTYPE_SPHERE, _GEOTYPE_BOX,
    )
    from newton._src.geometry.cslc_data import (
        CSLCData, calibrate_kc, create_pad_for_box_face,
    )
    _GEOTYPE_MESH = 1
    _GEOTYPE_CONVEX_MESH = 8
    _MESH_LIKE_TYPES = (_GEOTYPE_MESH, _GEOTYPE_CONVEX_MESH)

    shape_flags = model.shape_flags.numpy()
    shape_types = model.shape_type.numpy()
    cslc_shape_indices = [
        i for i in range(model.shape_count) if (shape_flags[i] & _CSLC_FLAG)
    ]
    if not cslc_shape_indices:
        return None

    cslc_set = set(cslc_shape_indices)
    shape_pairs: list = []
    if model.shape_contact_pairs is not None:
        for sa, sb in model.shape_contact_pairs.numpy():
            if sa in cslc_set and sb not in cslc_set:
                gt_other = int(shape_types[sb])
                # Skip pad-vs-ground (MESH ground) and similar unsupported
                # target types — the handler can only emit contacts for
                # sphere and box targets, and would just warn per step
                # otherwise.
                if gt_other not in (_GEOTYPE_SPHERE, _GEOTYPE_BOX):
                    continue
                shape_pairs.append(CSLCShapePair(
                    cslc_shape=int(sa), other_shape=int(sb),
                    other_geo_type=gt_other))
            elif sb in cslc_set and sa not in cslc_set:
                gt_other = int(shape_types[sa])
                if gt_other not in (_GEOTYPE_SPHERE, _GEOTYPE_BOX):
                    continue
                shape_pairs.append(CSLCShapePair(
                    cslc_shape=int(sb), other_shape=int(sa),
                    other_geo_type=gt_other))
    if not shape_pairs:
        return None

    cslc_spacing = model.shape_cslc_spacing.numpy()
    cslc_ka_arr = model.shape_cslc_ka.numpy()
    cslc_kl_arr = model.shape_cslc_kl.numpy()
    cslc_dc_arr = model.shape_cslc_dc.numpy()
    shape_ke = model.shape_material_ke.numpy()
    shape_scale_np = model.shape_scale.numpy()

    first_cslc = cslc_shape_indices[0]
    ka = float(cslc_ka_arr[first_cslc])
    kl = float(cslc_kl_arr[first_cslc])
    dc = float(cslc_dc_arr[first_cslc])

    pads = []
    for shape_idx in cslc_shape_indices:
        gt = int(shape_types[shape_idx])
        if gt == _GEOTYPE_BOX:
            hx, hy, hz = [float(shape_scale_np[shape_idx][j])
                          for j in range(3)]
            pads.append(create_pad_for_box_face(
                hx, hy, hz, face_axis=0, face_sign=+1,
                spacing=float(cslc_spacing[shape_idx]),
                shape_index=shape_idx))
        elif gt in _MESH_LIKE_TYPES:
            if shape_idx not in mesh_pads_by_shape:
                raise RuntimeError(
                    f"CSLC shape {shape_idx} is a MESH-like shape but no "
                    "pre-built CSLCPad was provided; build one with "
                    "`_make_mesh_cslc_pad(...)` and pass it in.")
            pads.append(mesh_pads_by_shape[shape_idx])
        else:
            raise RuntimeError(
                f"CSLC shape {shape_idx} has geo_type {gt}; only BOX and "
                "MESH/CONVEX_MESH are supported by this builder.")

    # kc calibration — start with the moderate prior (cf=0.3) that the
    # original `_from_model` uses; the user can call
    # `recalibrate_cslc_kc_per_pad(model, p.cslc_contact_fraction)` to
    # match the lift-test fair-calibration value.
    ke_bulk = float(shape_ke[first_cslc])
    kc = calibrate_kc(ke_bulk, pads, ka=ka, contact_fraction=0.3, per_pad=True)

    cslc_data = CSLCData.from_pads(
        pads, ka=ka, kl=kl, kc=kc, dc=dc,
        build_A_inv=True, device=model.device)

    # Filter CSLC pairs from the narrow phase (otherwise contacts double).
    if not hasattr(model, "shape_collision_filter_pairs"):
        model.shape_collision_filter_pairs = set()
    for pair in shape_pairs:
        a, b = sorted((pair.cslc_shape, pair.other_shape))
        model.shape_collision_filter_pairs.add((a, b))

    # Cache per-pair target info (same pattern as `_from_model`).
    shape_body_np = model.shape_body.numpy()
    shape_transform_np = model.shape_transform.numpy()
    for pair in shape_pairs:
        ke_raw = float(shape_ke[pair.other_shape])
        pair.other_ke = ke_raw if ke_raw > 0.0 else 1.0e9
        if pair.other_geo_type == _GEOTYPE_SPHERE:
            pair.other_body = int(shape_body_np[pair.other_shape])
            xf = shape_transform_np[pair.other_shape]
            pair.other_local_pos = (float(xf[0]), float(xf[1]), float(xf[2]))
            pair.other_radius = float(shape_scale_np[pair.other_shape][0])
        elif pair.other_geo_type == _GEOTYPE_BOX:
            pair.other_body = int(shape_body_np[pair.other_shape])
            xf = shape_transform_np[pair.other_shape]
            pair.other_local_xform = (float(xf[0]), float(xf[1]), float(xf[2]),
                                      float(xf[3]), float(xf[4]),
                                      float(xf[5]), float(xf[6]))
            pair.other_half_extents = (
                float(shape_scale_np[pair.other_shape][0]),
                float(shape_scale_np[pair.other_shape][1]),
                float(shape_scale_np[pair.other_shape][2]))

    # Surface slot map (one slot per surface sphere; the handler writes
    # one contact per slot per pair).
    is_surface_np = cslc_data.is_surface.numpy()
    surface_slot_map = np.full(cslc_data.n_spheres, -1, dtype=np.int32)
    slot = 0
    for i in range(cslc_data.n_spheres):
        if is_surface_np[i] == 1:
            surface_slot_map[i] = slot
            slot += 1
    slot_to_tid = np.full(slot, -1, dtype=np.int32)
    for tid in range(cslc_data.n_spheres):
        s = surface_slot_map[tid]
        if s >= 0:
            slot_to_tid[s] = tid

    supported = (_GEOTYPE_SPHERE, _GEOTYPE_BOX)
    supported_pairs = [p for p in shape_pairs if p.other_geo_type in supported]
    if not supported_pairs:
        return None

    n_iter = (int(model.shape_cslc_n_iter[first_cslc])
              if model.shape_cslc_n_iter is not None else 40)
    alpha = (float(model.shape_cslc_alpha[first_cslc])
             if model.shape_cslc_alpha is not None else 0.3)

    handler = CSLCHandler(
        cslc_data=cslc_data, shape_pairs=shape_pairs,
        n_iter=n_iter, alpha=alpha,
        surface_slot_map=wp.array(surface_slot_map, dtype=wp.int32,
                                  device=model.device),
        n_surface_contacts=slot, n_pair_blocks=len(supported_pairs),
        device=model.device)
    handler.slot_to_tid = slot_to_tid
    return handler


@contextmanager
def _patched_cslc_from_model(handler):
    """Temporarily make `CSLCHandler._from_model` return our pre-built handler.

    Newton's CollisionPipeline calls `CSLCHandler._from_model(model)` at
    construction time; the patch makes it return our externally-built
    handler so the pipeline wires up the contact-slot budget, excluded
    pairs, and per-step launch correctly.
    """
    from newton._src.geometry.cslc_handler import CSLCHandler
    original = CSLCHandler._from_model
    CSLCHandler._from_model = classmethod(lambda cls, model: handler)
    try:
        yield
    finally:
        CSLCHandler._from_model = original


# ── Scene builder ────────────────────────────────────────────────────────


def _pad_shape_cfg(p: SceneParams, contact_model: str):
    """ShapeConfig for the pad mesh.

    Point mode uses ``p.pad_ke`` (scaled to compensate for Newton's
    one-contact-per-triangle mesh narrow phase, see SceneParams.pad_ke);
    hydro uses ``p.ke`` for the Hunt–Crossley regularisation since the
    aggregate stiffness is dominated by ``kh`` anyway.

    For "hydro", the pad mesh must already carry an SDF (built via
    ``mesh.build_sdf()`` in ``_build_scene``) — Newton rejects
    ``sdf_max_resolution`` on mesh shapes because the SDF is owned by
    the mesh, not generated from the config.
    """
    ke = p.pad_ke if contact_model == "point" else p.ke
    kwargs = dict(ke=ke, kd=p.kd, kf=p.kf, mu=p.mu,
                  gap=0.002, density=p.pad_density)
    if contact_model == "hydro":
        kwargs.update(kh=p.kh, is_hydroelastic=True)
    elif contact_model == "cslc":
        # cslc_spacing is a placeholder — the actual lattice spacing is
        # derived from the Poisson-disc samples and stored on the
        # pre-built `CSLCPad`.  The handler reads
        # `model.shape_cslc_spacing[i]` for kc calibration only.
        kwargs.update(is_cslc=True,
                      cslc_spacing=0.005,  # placeholder, overridden by pad
                      cslc_ka=p.cslc_ka, cslc_kl=p.cslc_kl,
                      cslc_dc=p.cslc_dc,
                      cslc_n_iter=p.cslc_n_iter, cslc_alpha=p.cslc_alpha)
    return newton.ModelBuilder.ShapeConfig(**kwargs)


def _sphere_shape_cfg(p: SceneParams, contact_model: str):
    """ShapeConfig for the gripped sphere primitive.

    For "hydro", Newton generates the SDF internally from the analytic
    sphere geometry, controlled by ``sdf_max_resolution``.
    """
    kwargs = dict(ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu,
                  gap=0.002, density=p.sphere_density)
    if contact_model == "hydro":
        kwargs.update(kh=p.kh, is_hydroelastic=True,
                      sdf_max_resolution=p.sdf_resolution)
    return newton.ModelBuilder.ShapeConfig(**kwargs)


def _build_scene(p: SceneParams, contact_model: str = "point"):
    """Build ground + 2 articulated mesh pads + 1 free sphere.

    Each pad has the topology
        world ──[prismatic X]──> slider ──[prismatic Z]──> pad
    so the X joint drives APPROACH/SQUEEZE and the Z joint drives LIFT.

    ``contact_model`` selects the per-shape physics: "point" uses the
    standard Hunt–Crossley point-contact pipeline (single closest-point
    contact for the mesh-vs-sphere pair), "hydro" adds the hydroelastic
    pressure-field model on both bodies (requires an SDF on the pad mesh).

    Returns (model, dof_map, trimesh_obj).
    """
    if contact_model not in ("point", "hydro", "cslc"):
        raise ValueError(
            f"contact_model must be 'point', 'hydro', or 'cslc', got {contact_model!r}")

    tm = load_pad_mesh(p.pad_obj)
    pad_mesh = newton.Mesh(
        tm.vertices.astype(np.float32),
        tm.faces.astype(np.int32).flatten(),
    )

    # The hydroelastic pipeline expects a signed-distance field on every
    # mesh shape it sees; the sphere primitive is handled internally by
    # Newton from its analytic SDF, so only the pad mesh needs an explicit
    # build.  Margin matches the shape gap so the narrow band captures the
    # full contact-active region.
    if contact_model == "hydro":
        pad_mesh.build_sdf(max_resolution=p.sdf_resolution, margin=0.002)

    b = newton.ModelBuilder()

    ground_shape = b.add_ground_plane() if p.add_ground else None

    # Ghost config for slider bodies (no collision, no mass via density=0
    # but we still need a small mass for the prismatic joint to integrate).
    ghost_cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False, has_particle_collision=False, density=0.0)

    pad_cfg = _pad_shape_cfg(p, contact_model)

    _R_L, _R_R, q_L, q_R = _shape_rotations()

    dof_map: dict[str, int] = {}
    pad_joints: list[int] = []
    pad_shape_indices: dict[str, int] = {}  # "left"/"right" -> shape index

    lx0 = -p.pad_center_x
    rx0 = +p.pad_center_x
    pad_z0 = p.pad_center_z
    sphere_z0 = p.sphere_start_z

    for label, x0, q_shape in (("left", lx0, q_L), ("right", rx0, q_R)):
        # Slider body — intermediate link for X translation.
        slider = b.add_link(
            xform=wp.transform((x0, 0.0, pad_z0), wp.quat_identity()),
            mass=0.1, label=f"{label}_slider")
        slider_shape = b.add_shape_sphere(slider, radius=0.002, cfg=ghost_cfg)

        # Pad body — carries the mesh; world-aligned frame, mesh rotated
        # within the body so the curved face points inward.
        pad = b.add_link(
            xform=wp.transform((x0, 0.0, pad_z0), wp.quat_identity()),
            label=f"{label}_pad")
        pad_mesh_shape = b.add_shape_mesh(
            body=pad, mesh=pad_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), q_shape),
            cfg=pad_cfg, label=f"{label}_pad_mesh")
        pad_shape_indices[label] = pad_mesh_shape

        # Prismatic X: world -> slider (drives APPROACH/SQUEEZE).
        j_x = b.add_joint_prismatic(
            parent=-1, child=slider,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform((x0, 0.0, pad_z0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label=f"{label}_x")

        # Prismatic Z: slider -> pad (drives LIFT).
        j_z = b.add_joint_prismatic(
            parent=slider, child=pad,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label=f"{label}_z")

        b.add_articulation([j_x, j_z], label=f"{label}_arm")
        pad_joints.extend([j_x, j_z])

        dof_map[f"{label}_x"] = b.joint_qd_start[j_x]
        dof_map[f"{label}_z"] = b.joint_qd_start[j_z]

        if ground_shape is not None:
            b.add_shape_collision_filter_pair(slider_shape, ground_shape)

    # Joint drives — stiff PD position tracking on every pad DOF.
    for ji in pad_joints:
        dof = b.joint_qd_start[ji]
        b.joint_target_ke[dof] = p.drive_ke
        b.joint_target_kd[dof] = p.drive_kd
        b.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        b.joint_armature[dof] = 0.01

    # Dynamic sphere with explicit free joint (required for MuJoCo solver).
    sphere_cfg = _sphere_shape_cfg(p, contact_model)
    sphere = b.add_link(
        xform=wp.transform((0.0, 0.0, sphere_z0), wp.quat_identity()),
        label="sphere")
    b.add_shape_sphere(sphere, radius=p.sphere_radius, cfg=sphere_cfg)
    j_free = b.add_joint_free(sphere, label="sphere_free")
    b.add_articulation([j_free], label="sphere")

    b.request_contact_attributes("force")

    m = b.finalize()
    m.set_gravity(p.gravity)

    _log(f"DOF map: {dof_map}")
    _log(f"bodies={m.body_count} shapes={m.shape_count} "
         f"joints={m.joint_count} DOFs={m.joint_dof_count}")

    return m, dof_map, tm, pad_shape_indices


# ── Pad target trajectory (identical to lift_test.py) ────────────────────


def _pad_state(step: int, p: SceneParams) -> tuple[float, float]:
    """Return (dx_inward, dz_up) for a pad at the given step.

    The dz profile uses a C1-smooth velocity ramp at BOTH endpoints of
    LIFT so the pad's commanded velocity eases from 0 -> lift_speed -> 0
    rather than stepping discontinuously — avoiding the impulsive
    friction kick that would otherwise launch the sphere.
    """
    phase, s = p.phase_of(step)
    t = s * p.dt

    if phase == "APPROACH":
        return p.approach_speed * t, 0.0

    dx_app = p.approach_speed * p.approach_duration
    if phase == "SQUEEZE":
        return dx_app + p.squeeze_speed * t, 0.0

    dx_total = dx_app + p.squeeze_speed * p.squeeze_duration

    def _lift_dz(t_lift: float) -> float:
        ramp = p.lift_ramp_duration
        T = p.lift_duration
        if ramp <= 0.0:
            return p.lift_speed * t_lift
        # Renormalise so two-end-ramp travel matches single-end-ramp design.
        v_eff = p.lift_speed * (T - 0.5 * ramp) / max(T - ramp, 1e-9)
        if t_lift < ramp:
            sn = t_lift / ramp
            return v_eff * ramp * (sn ** 3 - 0.5 * sn ** 4)
        if t_lift < T - ramp:
            return v_eff * ramp * 0.5 + v_eff * (t_lift - ramp)
        s_back = max((T - t_lift) / ramp, 0.0)
        z_phase2_end = v_eff * (ramp * 0.5) + v_eff * (T - 2.0 * ramp)
        z_phase3 = v_eff * ramp * (0.5 - (s_back ** 3 - 0.5 * s_back ** 4))
        return z_phase2_end + z_phase3

    if phase == "LIFT":
        return dx_total, _lift_dz(t)

    # HOLD: freeze at the position reached at the end of LIFT.
    return dx_total, _lift_dz(p.lift_duration)


def set_pad_targets(control, step: int, p: SceneParams,
                    dof_map: dict[str, int], debug: bool = False) -> None:
    """Write joint position targets for both pads.

    Inward direction is +x for the LEFT pad (starting at world -pad_center_x)
    and -x for the RIGHT pad (starting at world +pad_center_x).  Both pads
    rise together by dz during LIFT.
    """
    dx, dz = _pad_state(step, p)

    target = control.joint_target_pos.numpy()
    target[dof_map["left_x"]] = +dx
    target[dof_map["left_z"]] = +dz
    target[dof_map["right_x"]] = -dx
    target[dof_map["right_z"]] = +dz
    control.joint_target_pos.assign(
        wp.array(target, dtype=wp.float32,
                 device=control.joint_target_pos.device))

    if debug:
        phase, _ = p.phase_of(step)
        # Curved apex is at body x +/- pad_thickness from the body origin.
        apex_gap = 2.0 * (p.pad_center_x - dx - p.pad_thickness)
        face_pen = 2.0 * p.sphere_radius - apex_gap
        _log(f"[{phase:8s}] step={step:5d}  dx={dx*1e3:+6.2f}mm  "
             f"dz={dz*1e3:+5.2f}mm  apex_gap={apex_gap*1e3:+6.1f}mm  "
             f"face_pen={face_pen*1e3:+5.2f}mm")


# ── Hydroelastic kh calibration ──────────────────────────────────────────


def calibrate_pad_stiffness(p_base: SceneParams,
                            contact_model: str = "hydro",
                            ke_bulk: float = 50_000.0,
                            verbose: bool = True) -> dict:
    """Measure per-pad aggregate stiffness at the squeeze-end pose.

    Hydro path: each contact polygon stores
    ``rigid_contact_stiffness[i] = kh_eff · polygon_area_i``; summing
    them over pad-vs-sphere pairs yields the aggregate per-pad normal
    stiffness ``k_pad``.  The fair invariant is ``k_pad = ke_bulk``;
    inverting linearly in kh gives ``kh_fair = kh_current · ke_bulk / k_pad``.

    Point path: ``rigid_contact_stiffness`` is unpopulated, so we fall
    back to the pad shape's material ke (per the t2_indenter convention).
    Mesh-vs-sphere produces one closest-point pair per shape pair, so
    ``k_pad ≡ ke`` and the model is fair by construction whenever
    ``ke = ke_bulk`` — no calibration knob to turn.

    A face_pen verification block is printed so the user can see (i) the
    theoretical apex penetration from joint state and (ii) the max
    contact-pen the narrow phase actually reports.  Disagreement
    between the two would indicate a geometry-orientation bug.

    Args:
        p_base: scene params (uses pad_obj, geometry, sphere_radius, kh).
        contact_model: ``"hydro"`` (recalibrates kh) or ``"point"``
            (sanity-checks ke against ke_bulk).
        ke_bulk: per-pad aggregate target [N/m] — default matches the
            paper's fair calibration (5e4 N/m).
        verbose: log derivation to stdout.

    Returns:
        ``{"k_pad": float, "kh_fair": float | None,
           "apex_pen": float, "max_contact_pen": float}``.
    """
    p = replace(p_base, add_ground=False, sphere_start_z=p_base.sphere_radius)
    model, dof_map, _, _ = _build_scene(p, contact_model)
    contacts = model.contacts()
    state = model.state()

    # Squeeze-end inward motion.  By geometry, apex_pen = dx_pen − initial_apex_gap.
    # initial_apex_gap = pad_center_x − pad_thickness − sphere_radius
    #                  = 0.06 − 0.01 − 0.03 = 0.02 m (20 mm).
    # dx_pen = approach_speed·approach_duration + squeeze_speed·squeeze_duration
    #        = 0.01333·1.5 + 0.002·0.5 = 0.021 m → apex_pen = 1 mm.
    dx_pen = (p.approach_speed * p.approach_duration
              + p.squeeze_speed * p.squeeze_duration)
    initial_apex_gap = p.pad_center_x - p.pad_thickness - p.sphere_radius
    apex_pen = dx_pen - initial_apex_gap

    joint_q = model.joint_q.numpy()
    joint_q[dof_map["left_x"]] = +dx_pen
    joint_q[dof_map["right_x"]] = -dx_pen
    model.joint_q.assign(wp.array(joint_q, dtype=wp.float32,
                                  device=model.joint_q.device))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    state.clear_forces()
    model.collide(state, contacts)

    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        if verbose:
            _log("CALIBRATE: no contacts at squeeze pose; check geometry.")
        return {"k_pad": 0.0, "kh_fair": None,
                "apex_pen": apex_pen, "max_contact_pen": 0.0}

    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    shape_body = model.shape_body.numpy()

    # Per-contact stiffness: hydro/CSLC populate it directly; for plain
    # point contact we read the pad shape's material ke (matches
    # t2_indenter.py's fallback).
    if contacts.rigid_contact_stiffness is not None:
        stiff = contacts.rigid_contact_stiffness.numpy()[:n]
    else:
        ke_per_shape = model.shape_material_ke.numpy()
        stiff = np.array(
            [ke_per_shape[int(s)] if s >= 0 else 0.0 for s in s0],
            dtype=np.float32)

    # Per-contact penetration: pen_i = (margin0+margin1) − dot(p1−p0, n).
    # We compute this both to (i) verify the claimed 1 mm apex penetration
    # and (ii) flag any direction-of-normal inversion.
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    normals = contacts.rigid_contact_normal.numpy()[:n]
    m0 = contacts.rigid_contact_margin0.numpy()[:n]
    m1 = contacts.rigid_contact_margin1.numpy()[:n]
    shape_transform = model.shape_transform.numpy()
    body_q = state.body_q.numpy()

    def _qrot(q, v):
        xyz = np.asarray(q[:3], dtype=np.float64)
        w = float(q[3])
        t = 2.0 * np.cross(xyz, v)
        return v + w * t + np.cross(xyz, t)

    k_left = k_right = 0.0
    n_left = n_right = 0
    pen_pad = []     # contact penetration depths for pad-vs-sphere only
    for i in range(n):
        if s0[i] < 0:
            continue
        s_idx0, s_idx1 = int(s0[i]), int(s1[i])
        b0 = int(shape_body[s_idx0])
        b1 = int(shape_body[s_idx1])
        if SPHERE_BODY not in (b0, b1):
            continue
        # World positions of p0 (shape0) and p1 (shape1):
        xs0 = shape_transform[s_idx0]
        p0_w = _qrot(body_q[b0, 3:7],
                     _qrot(xs0[3:7], np.asarray(p0[i], np.float64)) + xs0[:3]) \
            + body_q[b0, :3]
        xs1 = shape_transform[s_idx1]
        p1_w = _qrot(body_q[b1, 3:7],
                     _qrot(xs1[3:7], np.asarray(p1[i], np.float64)) + xs1[:3]) \
            + body_q[b1, :3]
        nrm = np.asarray(normals[i], np.float64)
        pen = (float(m0[i]) + float(m1[i])
               - float(np.dot(p1_w - p0_w, nrm)))
        pen_pad.append(max(pen, 0.0))
        s = float(max(stiff[i], 0.0))
        if LEFT_PAD in (b0, b1):
            k_left += s
            n_left += 1
        elif RIGHT_PAD in (b0, b1):
            k_right += s
            n_right += 1

    k_pad = 0.5 * (k_left + k_right)
    max_pen = float(max(pen_pad)) if pen_pad else 0.0
    kh_fair = None
    pad_ke_fair = None
    if k_pad > 0.0:
        if contact_model == "hydro":
            kh_fair = p_base.kh * ke_bulk / k_pad
        elif contact_model == "point":
            # k_pad scales linearly with the per-contact ke; invert it.
            # ``p_base.pad_ke`` is the actual ke the pad shape carried
            # during this calibration run (see _pad_shape_cfg).
            pad_ke_fair = p_base.pad_ke * ke_bulk / k_pad

    if verbose:
        _section(f"PAD STIFFNESS CALIBRATION  "
                 f"(model={contact_model}, ke_bulk={ke_bulk:.0f} N/m)")
        _log(f"squeeze inward dx     = {dx_pen*1e3:.3f} mm")
        _log(f"initial apex–surface gap = {initial_apex_gap*1e3:.3f} mm")
        _log(
            f"theoretical apex pen  = {apex_pen*1e3:.3f} mm  (← '1 mm face_pen')")
        _log(f"max contact pen (measured) = {max_pen*1e3:.3f} mm  "
             f"({len(pen_pad)} pad-vs-sphere contact polygons)")
        if contact_model == "hydro":
            _log(f"kh_current  = {p_base.kh:.3e} Pa")
        else:
            _log(f"pad_ke_current = {p_base.pad_ke:.1f} N/m  "
                 f"(sphere ke = {p_base.ke:.0f})")
        _log(f"k_left      = {k_left:7.1f} N/m  ({n_left:2d} polygons)")
        _log(f"k_right     = {k_right:7.1f} N/m  ({n_right:2d} polygons)")
        _log(f"k_pad (avg) = {k_pad:7.1f} N/m  "
             f"(current fairness: {100.0 * k_pad / ke_bulk:.1f} %)")
        if kh_fair is not None:
            _log(f"-> fair kh      = {kh_fair:.3e} Pa  "
                 f"(scale factor {kh_fair / p_base.kh:.2f}x)")
        if pad_ke_fair is not None:
            _log(f"-> fair pad_ke  = {pad_ke_fair:.1f} N/m  "
                 f"(scale factor {pad_ke_fair / p_base.pad_ke:.2f}x)")

    return {"k_pad": k_pad, "kh_fair": kh_fair, "pad_ke_fair": pad_ke_fair,
            "apex_pen": apex_pen, "max_contact_pen": max_pen}


# Back-compat shim — keeps the old name working but routes through the
# generalised helper.  Returns kh_fair as before so existing callers in
# `main` don't need to change shape.
def calibrate_hydro_kh(p_base: SceneParams, ke_bulk: float = 50_000.0,
                       verbose: bool = True) -> float:
    result = calibrate_pad_stiffness(p_base, "hydro", ke_bulk, verbose)
    return result["kh_fair"] if result["kh_fair"] is not None else p_base.kh


# ── Viewer example ───────────────────────────────────────────────────────


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(self.frame_dt / 0.002))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_step = 0

        self.solver_name = getattr(args, "solver", "mujoco")
        self.contact_model = getattr(args, "contact_model", "point")

        self.p = SceneParams(
            dt=self.sim_dt,
            n_samples_per_pad=getattr(args, "n_samples", 150),
            nz_threshold=getattr(args, "nz_threshold", 0.3),
            arrow_length=getattr(args, "arrow_length", 0.005),
            add_ground=not getattr(args, "no_ground", False),
        )
        if getattr(args, "kh", None) is not None:
            self.p.kh = float(args.kh)
        self.p.dump()
        _log(f"contact_model = {self.contact_model}"
             + (f"  kh = {self.p.kh:.2e} Pa" if self.contact_model == "hydro" else ""))

        self.model, self.dof_map, tm, self.pad_shape_indices = _build_scene(
            self.p, self.contact_model)
        inspect_model(self.model, f"obj_pad_lift_{self.contact_model}")

        # Sample PDS points BEFORE the collision pipeline is built, so
        # we can construct the CSLC pads first (the pipeline reads
        # `CSLCHandler._from_model` at construction time and needs the
        # handler ready).
        local_pts, local_norms = sample_curved_face(
            tm, self.p.n_samples_per_pad,
            nz_threshold=self.p.nz_threshold, seed=self.p.seed)

        # Two coordinate frames at play:
        #   - OBJ-local: the asset's native frame. `local_pts` are here.
        #     The mesh shape carries `xform = q_shape = Ry(±π/2)` so the
        #     CSLC kernel applies q_shape to every CSLCPad position to
        #     get body-frame coords.  Therefore the CSLC pad must store
        #     positions in OBJ-LOCAL — passing body-local would double-
        #     rotate (a bug we fixed: lattice ended up at wrong x,z).
        #   - body-local: OBJ-local rotated by R_L/R_R.  The viewer draws
        #     PDS markers in the pad body frame, so we keep this version
        #     for `_update_viz_arrays`.
        self._local_pts = local_pts.astype(np.float32)
        self._local_norms = local_norms.astype(np.float32)
        R_L, R_R, _, _ = _shape_rotations()
        self._body_local_pts_L = (local_pts @ R_L.T).astype(np.float32)
        self._body_local_pts_R = (local_pts @ R_R.T).astype(np.float32)
        self._body_local_norms_L = (local_norms @ R_L.T).astype(np.float32)
        self._body_local_norms_R = (local_norms @ R_R.T).astype(np.float32)

        # Hydro mode needs an explicit collision pipeline so we can flip
        # `output_contact_surface=True` and pull the pressure-field
        # triangles for the viewer.  CSLC mode needs the pre-built
        # CSLCHandler installed via the monkey-patch context manager so
        # the pipeline's auto-discovery hands back our mesh pads.
        hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=(self.contact_model == "hydro"))

        if self.contact_model == "cslc":
            # Build one mesh-pad lattice per side, keyed by shape index.
            # The mesh-shape rotation has already been applied to the
            # body-local PDS arrays, so the pad's positions are in the
            # pad body frame (= world frame at identity body rotation).
            # CSLC pads carry OBJ-LOCAL positions and normals — the
            # kernel applies the mesh shape's q_shape transform to get
            # body coords.  Both pads share the same OBJ-local lattice;
            # each pad's per-shape Ry rotation places it correctly in
            # the body frame (and the per-pad shape index disambiguates
            # which body to attach to).
            mesh_pads = {
                self.pad_shape_indices["left"]: _make_mesh_cslc_pad(
                    self._local_pts, self._local_norms,
                    self.pad_shape_indices["left"],
                    k_neighbors=self.p.cslc_k_neighbors),
                self.pad_shape_indices["right"]: _make_mesh_cslc_pad(
                    self._local_pts, self._local_norms,
                    self.pad_shape_indices["right"],
                    k_neighbors=self.p.cslc_k_neighbors),
            }
            handler = _build_cslc_handler_with_mesh_pads(self.model, mesh_pads)
            _log(f"CSLC: built handler with {handler.cslc_data.n_spheres} "
                 f"total spheres across {len(mesh_pads)} pad(s); "
                 f"spacing ≈ {1e3*np.mean([p.spacing for p in mesh_pads.values()]):.2f} mm")
            with _patched_cslc_from_model(handler):
                self.collision_pipeline = newton.CollisionPipeline(
                    self.model, sdf_hydroelastic_config=hydro_cfg)
        else:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model, sdf_hydroelastic_config=hydro_cfg)
        self.contacts = self.collision_pipeline.contacts()

        # kc recalibration for CSLC — two passes, both keyed on the
        # lift_test fair invariant `N_active · keff = ke_bulk`:
        #
        #   (a) First pass: apply the user-provided `cslc_contact_fraction`
        #       (default 0.025, matching lift_test's box-pad value) as a
        #       prior so the handler initialises with a sensible kc.
        #   (b) Second pass: snap the pads to the squeeze-end pose, run
        #       `pipeline.collide()` once to populate the lattice
        #       penetrations, count the actual active surface spheres
        #       (raw_penetration > 0), recompute
        #       `cf_actual = n_active_per_pad / n_surface_per_pad`, and
        #       call `recalibrate_cslc_kc_per_pad` again with that value.
        #       The box-pad cf prior was derived for a 9×21 grid; the
        #       PDS scatter on the curved face has a different
        #       contact-region-to-surface ratio at 1 mm pen, so without
        #       this empirical pass CSLC's per-pad aggregate stiffness
        #       drifts off `ke_bulk`.
        if self.contact_model == "cslc" and self.p.cslc_contact_fraction is not None:
            try:
                from cslc_mujoco.common import recalibrate_cslc_kc_per_pad
                _log(
                    f"CSLC: priming kc with cf = {self.p.cslc_contact_fraction:.3f}")
                recalibrate_cslc_kc_per_pad(
                    self.model, self.p.cslc_contact_fraction)

                # Empirical second pass.  We need a state for collide();
                # build a temporary one at the squeeze-end pose.
                cal_state = self.model.state()
                dx_pen = (self.p.approach_speed * self.p.approach_duration
                          + self.p.squeeze_speed * self.p.squeeze_duration)
                joint_q = self.model.joint_q.numpy().copy()
                joint_q[self.dof_map["left_x"]] = +dx_pen
                joint_q[self.dof_map["right_x"]] = -dx_pen
                model_q = wp.array(joint_q, dtype=wp.float32,
                                   device=self.model.joint_q.device)
                self.model.joint_q.assign(model_q)
                newton.eval_fk(self.model, self.model.joint_q,
                               self.model.joint_qd, cal_state)
                cal_state.clear_forces()
                self.collision_pipeline.collide(cal_state, self.contacts)

                h = self.collision_pipeline.cslc_handler
                is_surf = h.cslc_data.is_surface.numpy() == 1
                # Active per pad = surface spheres whose penetration is
                # ABOVE the smoothing baseline.  Naïve `pen > 0` catches
                # the eps/2 ≈ 5 µm tail of `σ_ε(x)` for non-contact
                # spheres (e.g. 100+ phantom-actives during APPROACH when
                # pads are 20 mm from the sphere).  Require the pen to
                # exceed 10·eps (≈ 0.1 mm) so only genuinely engaged
                # spheres count — that's well above the smoothing tail
                # but well below typical squeeze-end penetration (1 mm
                # at the apex).
                eps = float(h.cslc_data.smoothing_eps)
                pen_threshold = max(10.0 * eps, 1e-4)
                n_active_total = 0
                for pen_buf in h.raw_penetration_pairs:
                    pen_np = pen_buf.numpy()
                    n_active_total += int((pen_np[is_surf]
                                          > pen_threshold).sum())
                n_surf_per_pad = int(
                    is_surf.sum()) // max(len(h.raw_penetration_pairs), 1)
                n_active_per_pad = max(
                    1, n_active_total // max(len(h.raw_penetration_pairs), 1))
                cf_actual = float(n_active_per_pad) / float(n_surf_per_pad)
                _log(f"CSLC: measured at squeeze-end — "
                     f"{n_active_per_pad}/{n_surf_per_pad} surface spheres active "
                     f"(cf = {cf_actual:.3f})")

                # Only re-recalibrate if the empirical cf differs by
                # more than ~10 % from the prior — otherwise we'd be
                # chasing noise.
                if abs(cf_actual - self.p.cslc_contact_fraction) > 0.1 * self.p.cslc_contact_fraction:
                    _log(
                        f"CSLC: re-running kc recalibration with empirical cf = {cf_actual:.3f}")
                    recalibrate_cslc_kc_per_pad(self.model, cf_actual)
                else:
                    _log("CSLC: empirical cf matches prior; keeping initial kc.")

                # Restore joint_q to its build-time value.
                joint_q[self.dof_map["left_x"]] = 0.0
                joint_q[self.dof_map["right_x"]] = 0.0
                self.model.joint_q.assign(
                    wp.array(joint_q, dtype=wp.float32,
                             device=self.model.joint_q.device))
            except Exception as e:
                _log(f"WARN: CSLC kc recalibration skipped: {e}")

        # Build solver AFTER pipeline so the rigid_contact_max budget
        # (which the CSLC handler expands at pipeline init) is final.
        self.solver = make_solver(self.model, self.solver_name)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        n_total = 2 * len(local_pts)
        self._n_per_pad = len(local_pts)

        bbox = tm.bounding_box.extents
        _log(f"pad bbox extents (mm): "
             f"[{bbox[0]*1e3:.2f}, {bbox[1]*1e3:.2f}, {bbox[2]*1e3:.2f}]")
        _log(f"sampled {len(local_pts)} Poisson-disc points per pad "
             f"(n_z > {self.p.nz_threshold})")
        _log(f"pad apex at world x = +/-{self.p.pad_center_x - self.p.pad_thickness:.3f} m "
             f"at t=0, sphere surface at +/-{self.p.sphere_radius:.3f} m  -> "
             f"initial gap = {(self.p.pad_center_x - self.p.pad_thickness - self.p.sphere_radius)*1e3:.1f}mm")

        # Allocate viz buffers once; per-frame update writes new positions in.
        zeros3 = np.zeros((n_total, 3), np.float32)
        self._starts = wp.array(zeros3, dtype=wp.vec3)
        self._ends = wp.array(zeros3, dtype=wp.vec3)
        xforms = np.zeros((n_total, 7), np.float32)
        xforms[:, 6] = 1.0   # identity quat for marker spheres
        self._sphere_xforms = wp.array(xforms, dtype=wp.transform)
        self._sphere_colors = wp.array(
            np.tile([1.0, 0.2, 0.2], (n_total, 1)).astype(np.float32),
            dtype=wp.vec3)
        self._sphere_mats = wp.array(
            np.tile([0.5, 0.3, 0.0, 0.0], (n_total, 1)).astype(np.float32),
            dtype=wp.vec4)
        # Host-side scratch buffers so we don't reallocate every frame.
        self._starts_host = np.zeros((n_total, 3), np.float32)
        self._ends_host = np.zeros((n_total, 3), np.float32)
        self._xforms_host = xforms.copy()
        # CSLC engagement viz: colored per-sphere by current penetration —
        # gray for inactive (pen < threshold), warm red for compressed
        # contacts (pen growing).  Updated each frame in render().
        self._colors_host = np.tile(
            [0.35, 0.35, 0.40], (n_total, 1)).astype(np.float32)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.30, -0.30, self.p.sphere_start_z + 0.15),
            pitch=-15.0, yaw=135.0)
        # Show the hydroelastic isosurface (pressure-field triangles) by
        # default in hydro mode; off in point mode where there's nothing
        # to draw.  Toggle at runtime via the side-panel checkbox added
        # by `render_ui`.
        self.viewer.show_hydro_contact_surface = (
            self.contact_model == "hydro")
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        q = self.state_0.body_q.numpy()
        _log(f"INITIAL STATE ({self.model.body_count} bodies):")
        for bi in range(self.model.body_count):
            _log(
                f"  body {bi}: pos=({q[bi,0]:+.4f},{q[bi,1]:+.4f},{q[bi,2]:+.4f})", 1)
        _log(
            f"sim_substeps={self.sim_substeps}  sim_dt={self.sim_dt*1e3:.3f}ms")

        # ── Sphere xyz history for stability/lift metrics ──
        # One entry per frame (not per substep) to keep the buffer light.
        # `test_final()` reduces these into per-phase summary stats so we
        # can directly compare CSLC vs hydro vs point on (i) xy slip during
        # HOLD and (ii) how much z lift the grasp actually delivered.
        self._xyz_hist: list[tuple[int, str, float, float, float]] = []

    # ── Simulation step ──────────────────────────────────────────────────

    def simulate(self):
        for _ in range(self.sim_substeps):
            set_pad_targets(self.control, self.sim_step, self.p, self.dof_map,
                            debug=(self.sim_step % 500 == 0))
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_step += 1

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

        # Per-frame xyz record (cheap — one numpy read + one append).
        q = self.state_0.body_q.numpy()
        phase_now, _ = self.p.phase_of(self.sim_step)
        self._xyz_hist.append((
            self.sim_step, phase_now,
            float(q[SPHERE_BODY, 0]),
            float(q[SPHERE_BODY, 1]),
            float(q[SPHERE_BODY, 2])))

        # Coarse telemetry every ~100 ms.
        if self.sim_step % (self.sim_substeps * 6) < self.sim_substeps:
            phase, _ = self.p.phase_of(self.sim_step)
            sphere_z = float(q[SPHERE_BODY, 2])
            pad_z = float(q[LEFT_PAD, 2])
            nc = count_active_contacts(self.contacts)
            extra = ""
            if self.contact_model == "hydro":
                # Number of pressure-field triangles currently drawn.
                # Zero during APPROACH (no penetration); non-zero from
                # SQUEEZE onward.  If this stays 0 through SQUEEZE the
                # hydro pipeline isn't producing surface data and the
                # viewer has nothing to render.
                hsdf = self.collision_pipeline.hydroelastic_sdf
                if hsdf is not None:
                    s = hsdf.get_contact_surface()
                    if s is not None:
                        extra = f"  hydro_polys={int(s.face_contact_count.numpy()[0])}"
            elif self.contact_model == "cslc":
                # Active surface spheres — uses the same above-eps
                # threshold as the cf calibration so the count reflects
                # truly engaged spheres rather than smoothing-gate noise.
                h = self.collision_pipeline.cslc_handler
                if h is not None:
                    pen = h.raw_penetration.numpy()
                    is_surf = h.cslc_data.is_surface.numpy() == 1
                    eps = float(h.cslc_data.smoothing_eps)
                    thresh = max(10.0 * eps, 1e-4)
                    n_active = int((pen[is_surf] > thresh).sum())
                    n_surf = int(is_surf.sum())
                    extra = f"  cslc={n_active}/{n_surf}"
            _log(f"[{phase:8s}] step={self.sim_step:5d}  "
                 f"sphere_z={sphere_z:+.4f}  pad_z={pad_z:+.4f}  "
                 f"contacts={nc}{extra}")

    # ── Per-frame viz update ────────────────────────────────────────────

    def _update_viz_arrays(self):
        """Recompute world-frame PDS points + normal arrows from the
        current pad body poses.  Pad bodies have prismatic-only joints
        -> body rotation is identity -> world_pt = body_pos + body_local_pt.
        """
        q = self.state_0.body_q.numpy()
        pL = q[LEFT_PAD, :3]
        pR = q[RIGHT_PAD, :3]
        n = self._n_per_pad

        starts = self._starts_host
        ends = self._ends_host
        xforms = self._xforms_host
        L = self._body_local_pts_L
        R = self._body_local_pts_R
        nL = self._body_local_norms_L
        nR = self._body_local_norms_R
        a = self.p.arrow_length

        starts[:n] = L + pL
        starts[n:2*n] = R + pR
        ends[:n] = starts[:n] + nL * a
        ends[n:2*n] = starts[n:2*n] + nR * a
        xforms[:n, :3] = starts[:n]
        xforms[n:2*n, :3] = starts[n:2*n]

        self._starts.assign(wp.array(starts, dtype=wp.vec3,
                                     device=self._starts.device))
        self._ends.assign(wp.array(ends, dtype=wp.vec3,
                                   device=self._ends.device))
        self._sphere_xforms.assign(wp.array(xforms, dtype=wp.transform,
                                            device=self._sphere_xforms.device))

        # ── Per-sphere engagement colouring (CSLC only) ──
        # Each lattice sphere is shaded by its current penetration:
        #   gray  for inactive (pen below smoothing-baseline threshold)
        #   warm  for active   (pen growing toward saturated contact)
        # The colour scale is normalised to the current max penetration
        # so it tracks the relative engagement regardless of phase.
        # For non-CSLC modes the buffer never updates (initialised gray).
        if self.contact_model == "cslc":
            h = self.collision_pipeline.cslc_handler
            if h is not None:
                eps = float(h.cslc_data.smoothing_eps)
                thresh = max(10.0 * eps, 1e-4)
                # Concatenate per-pair raw_penetration buffers in
                # left-then-right order to match the (L, R) layout used
                # by the sample arrays.  The handler stores one buffer
                # per sphere-target pair; we walk shape_pairs to pick
                # them out in the correct pad order.
                pen_L = h.raw_penetration_pairs[0].numpy()
                pen_R = (h.raw_penetration_pairs[1].numpy()
                         if len(h.raw_penetration_pairs) > 1 else pen_L)
                is_surf = h.cslc_data.is_surface.numpy() == 1
                # The CSLCData arrays carry all spheres for both pads
                # concatenated; pick out surface entries per shape.
                shape_ids = h.cslc_data.sphere_shape.numpy()
                left_idx = self.pad_shape_indices["left"]
                right_idx = self.pad_shape_indices["right"]
                mask_L = is_surf & (shape_ids == left_idx)
                mask_R = is_surf & (shape_ids == right_idx)
                # Each `raw_penetration_pairs[i]` is zeroed for spheres
                # not belonging to that pair's pad, so we read the same
                # buffer indices but pick out per-pad masked entries.
                penL = pen_L[mask_L]
                penR = pen_R[mask_R]
                # Normalise to the largest currently-active penetration so
                # the colour scale auto-tracks the contact intensity.
                pmax = float(max(penL.max() if penL.size else 0.0,
                                 penR.max() if penR.size else 0.0,
                                 thresh * 2.0))
                colors = self._colors_host
                # Default gray.
                colors[:] = (0.35, 0.35, 0.40)
                for offset, pen_pad in ((0, penL), (n, penR)):
                    t = np.clip(pen_pad / pmax, 0.0, 1.0).astype(np.float32)
                    active = pen_pad > thresh
                    # Warm red for engaged; intensity grows with pen.
                    colors[offset:offset + len(pen_pad), 0] = np.where(
                        active, t,            0.35)
                    colors[offset:offset + len(pen_pad), 1] = np.where(
                        active, 0.2 * (1.0 - t), 0.35)
                    colors[offset:offset + len(pen_pad), 2] = np.where(
                        active, 1.0 - t,      0.40)
                self._sphere_colors.assign(wp.array(
                    colors, dtype=wp.vec3,
                    device=self._sphere_colors.device))

    def render(self):
        # The PDS sample markers + normal arrows ARE the CSLC lattice —
        # show them only in cslc mode.  Point contact has its own
        # narrow-phase contacts and hydro has its pressure-field
        # isosurface; the PDS preview would only clutter those.
        show_lattice_preview = (self.contact_model == "cslc")
        if show_lattice_preview:
            self._update_viz_arrays()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.contact_model == "hydro":
            # Pressure-field triangles from the hydroelastic isosurface.
            # The viewer respects `show_hydro_contact_surface`; when off
            # this is a no-op.
            surface = None
            if self.collision_pipeline.hydroelastic_sdf is not None:
                surface = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            self.viewer.log_hydro_contact_surface(
                surface, penetrating_only=True)
        if show_lattice_preview:
            self.viewer.log_shapes(
                "/pad_samples",
                newton.GeoType.SPHERE,
                self.p.sample_radius,
                self._sphere_xforms,
                self._sphere_colors,
                self._sphere_mats,
            )
            self.viewer.log_lines(
                "/pad_normals", self._starts, self._ends, (0.1, 0.9, 1.0))
        self.viewer.end_frame()

    def render_ui(self, imgui):
        """Add hydro isosurface toggle to the GL viewer's side panel.

        The GL viewer's built-in "Show Contacts" checkbox only controls
        the green normal arrows from ``log_contacts``; the hydroelastic
        pressure-field triangles need a separate ``show_hydro_contact_surface``
        toggle, exposed here.
        """
        if self.contact_model == "hydro":
            changed, val = imgui.checkbox(
                "Show Hydro Isosurface",
                bool(self.viewer.show_hydro_contact_surface))
            if changed:
                self.viewer.show_hydro_contact_surface = val

    def test_final(self):
        # Always print the metrics summary first — even if the assertion
        # below fails, the user wants to see *what happened* to xyz so the
        # failure has diagnostic context.
        self._print_xyz_metrics()
        # At end-of-sim the sphere should not have fallen through the floor.
        q = self.state_0.body_q.numpy()
        assert q[SPHERE_BODY, 2] > -0.05, (
            f"sphere fell through ground: z={q[SPHERE_BODY, 2]:.4f}")

    def _print_xyz_metrics(self) -> None:
        """Summarize sphere (x, y, z) over the four phases.

        Three things matter for "did the grasp succeed?":
          • LIFT achieved      = z_HOLD_end - z_SQUEEZE_end  (m)
          • XY slip during HOLD = max(|x - x_ref|, |y - y_ref|)  with
            x_ref, y_ref = sphere xy at start of LIFT (so any drift
            during LIFT/HOLD shows up).  A stable grasp keeps this
            sub-millimetre; visible slip is hundreds of microns +.
          • XY/Z stability     = std-dev of x, y, z during HOLD.
            High values mean oscillation; low values mean steady hold.
        """
        if not self._xyz_hist:
            return

        import numpy as _np
        arr = _np.array([(s, x, y, z) for (s, _ph, x, y, z) in self._xyz_hist],
                        dtype=_np.float64)
        phases = [ph for (_s, ph, _x, _y, _z) in self._xyz_hist]

        def _slice(name: str) -> _np.ndarray:
            mask = _np.array([p == name for p in phases])
            return arr[mask]

        approach = _slice("APPROACH")
        squeeze = _slice("SQUEEZE")
        lift = _slice("LIFT")
        hold = _slice("HOLD")

        _section(f"SPHERE XYZ METRICS  ({self.contact_model.upper()})")
        for name, sl in (("APPROACH", approach), ("SQUEEZE", squeeze),
                         ("LIFT", lift), ("HOLD", hold)):
            if sl.size == 0:
                continue
            x0, y0, z0 = sl[0, 1], sl[0, 2], sl[0, 3]
            xe, ye, ze = sl[-1, 1], sl[-1, 2], sl[-1, 3]
            _log(f"[{name:8s}] start xyz = "
                 f"({x0*1e3:+7.2f}, {y0*1e3:+7.2f}, {z0*1e3:+7.2f}) mm  "
                 f"end xyz = "
                 f"({xe*1e3:+7.2f}, {ye*1e3:+7.2f}, {ze*1e3:+7.2f}) mm")

        # Lift achievement.  Reference z is taken at the *end* of SQUEEZE
        # (= start of LIFT); the gripped sphere should rise from there.
        if squeeze.size and hold.size:
            z_ref = float(squeeze[-1, 3])
            z_end = float(hold[-1, 3])
            z_peak = float(hold[:, 3].max())
            lift_m = z_end - z_ref
            _log(
                f"LIFT achieved (HOLD end − SQUEEZE end) = {lift_m*1e3:+6.2f} mm")
            _log(
                f"PEAK z during HOLD                    = {z_peak*1e3:+6.2f} mm")

        # XY slip + xyz stability during HOLD (the most demanding phase —
        # pads are stationary in air; any drift is from contact dynamics).
        if hold.size:
            if lift.size:
                x_ref, y_ref = float(lift[0, 1]), float(lift[0, 2])
            else:
                x_ref, y_ref = float(hold[0, 1]), float(hold[0, 2])
            dx = hold[:, 1] - x_ref
            dy = hold[:, 2] - y_ref
            xy_slip = float(_np.sqrt(dx*dx + dy*dy).max())
            x_std = float(hold[:, 1].std())
            y_std = float(hold[:, 2].std())
            z_std = float(hold[:, 3].std())
            _log("HOLD-phase stability:")
            _log(
                f"  max XY slip from LIFT-start ref = {xy_slip*1e3:7.3f} mm", 1)
            _log(f"  std(x), std(y), std(z)          = "
                 f"{x_std*1e3:6.3f}, {y_std*1e3:6.3f}, {z_std*1e3:6.3f} mm",
                 1)
            # Verdict tags so the user can eyeball pass/fail across runs.
            ok_lift = (squeeze.size and (z_end - z_ref) > 5e-3)
            ok_slip = xy_slip < 5e-3
            ok_std = max(x_std, y_std) < 2e-3
            _log(f"VERDICT  lift={'PASS' if ok_lift else 'FAIL'}  "
                 f"xy_slip={'PASS' if ok_slip else 'FAIL'}  "
                 f"hold_stability={'PASS' if ok_std else 'FAIL'}")

    # ── CLI ──────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--n-samples", type=int, default=150,
                            help="Poisson-disc samples per pad.")
        parser.add_argument("--nz-threshold", type=float, default=0.3,
                            help="Keep faces with face_normal.z > threshold "
                                 "(filters out flat base + sidewalls).")
        parser.add_argument("--arrow-length", type=float, default=0.005,
                            help="Length of the normal-arrow line segments [m].")
        parser.add_argument("--no-ground", action="store_true",
                            help="Skip the ground plane.")
        parser.add_argument("--solver", type=str, default="mujoco",
                            choices=["mujoco", "semi"],
                            help="Physics solver backend.")
        parser.add_argument("--contact-model", type=str, default="point",
                            choices=["point", "hydro", "cslc"],
                            help="Contact model: point (Hunt–Crossley), "
                                 "hydro (hydroelastic pressure field), or "
                                 "cslc (compliant sphere lattice on PDS samples).")
        parser.add_argument("--kh", type=float, default=None,
                            help="Override hydroelastic modulus [Pa].")
        parser.add_argument("--calibrate-kh", action="store_true",
                            help="Measure A_patch at 1 mm face_pen, print the "
                                 "fair kh that matches per-pad aggregate "
                                 "stiffness to ke_bulk = 5e4 N/m, then exit.")
        return parser


def main():
    parser = Example.create_parser()
    args, _ = parser.parse_known_args()
    if args.solver == "mujoco" and not HAS_MUJOCO:
        print("MuJoCo not installed; falling back to semi-implicit solver.")
        args.solver = "semi"
    wp.init()

    # Standalone calibration mode: no viewer, no lift cycle — just
    # measure A_patch at 1 mm pen, print the fair kh, exit.  User
    # then either re-runs with --kh <value> or hard-codes the result
    # into SceneParams.kh as the new default.
    if getattr(args, "calibrate_kh", False):
        cm = getattr(args, "contact_model", "hydro")
        print(f"\n{'━' * 60}\n  OBJ-PAD STIFFNESS CALIBRATION ({cm})\n{'━' * 60}")
        p = SceneParams()
        if getattr(args, "kh", None) is not None:
            p = replace(p, kh=float(args.kh))
        calibrate_pad_stiffness(p, contact_model=cm, verbose=True)
        return

    print(f"\n{'━' * 60}\n  OBJ-PAD LIFT TEST\n{'━' * 60}")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
