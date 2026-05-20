# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-shape configs and CSLC mesh-lattice handler attach.

Two responsibilities:

1.  **Shape configs**: build ``newton.ModelBuilder.ShapeConfig`` objects
    for pad and object shapes, parameterised by the chosen contact
    model (``cslc`` | ``point`` | ``hydro``).

2.  **CSLC mesh-lattice pipeline**: a pre-built ``CSLCLattice`` per
    sampled pad (Poisson-disc scatter, k-NN neighbour graph), an
    out-of-band ``CSLCHandler`` built around those lattices via the
    generic ``CSLCHandler.from_model_with_lattices`` builder, a context
    manager that swaps the handler in during model finalisation, and a
    post-build kc recalibration routine.

Step 7 (notes.md) made the Newton CSLC stack generic: there are no
box-grid lattice helpers any more, and the handler refuses to
auto-build lattices from shape geometry.  Callers are responsible for
sampling and pass ``CSLCLattice`` objects in explicitly.  This module
is the canonical reference for that flow.

.. note::
   The ``point`` and ``hydro`` paths are wired through these factories
   but uncalibrated for mesh-as-box pads in this PR — Newton's
   mesh-vs-sphere narrow phase emits one contact per triangle in the
   proximity band rather than one closest-point pair, so the effective
   per-lattice aggregate stiffness will differ from a primitive-box
   reference.  CSLC is the only paper-grade path right now.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import warp as wp

import newton

# Newton CSLC internals.  Step 7 cleanup: only ``CSLCLattice`` (the
# generic per-body container) and the generic builder remain in this
# import surface.
from newton._src.geometry.cslc_data import (
    CSLCData,
    CSLCLattice,
    calibrate_kc,
)
from newton._src.geometry.cslc_handler import (
    _CSLC_FLAG,
    _GEOTYPE_SPHERE,
    CSLCHandler,
    CSLCShapePair,
)

from .params import CSLCParams, GraspConfig, HydroParams, MaterialParams, PadParams

# Newton geometry-type integers we need locally.  Newton doesn't export
# MESH / CONVEX_MESH separately from the cslc handler module, so we
# alias them by their integer values (matches newton.GeoType.MESH = 1).
_GEOTYPE_MESH = 1
_GEOTYPE_CONVEX_MESH = 8
_MESH_LIKE_TYPES = (_GEOTYPE_MESH, _GEOTYPE_CONVEX_MESH)


# ── Shape configs ────────────────────────────────────────────────────────


def make_pad_shape_cfg(
    pad: PadParams,
    material: MaterialParams,
    cslc: CSLCParams,
    hydro: HydroParams,
    contact_model: str,
) -> newton.ModelBuilder.ShapeConfig:
    """Build the ``ShapeConfig`` for a pad's mesh shape.

    For ``contact_model="cslc"`` the config carries the per-shape CSLC
    knobs (``is_cslc=True`` etc.).  ``cslc_spacing`` is a placeholder
    here — the real spacing is derived from the Poisson-disc samples and
    stored on the pre-built ``CSLCLattice`` (see
    :func:`make_cslc_pad_from_samples`).  Newton reads
    ``shape_cslc_spacing[i]`` only for its calibration heuristic, so a
    nominal value is sufficient.
    """
    kwargs: dict = dict(
        ke=material.ke,
        kd=material.kd,
        kf=material.kf,
        mu=material.mu,
        gap=material.gap,
        density=pad.density,
    )
    if contact_model == "cslc":
        kwargs.update(
            is_cslc=True,
            cslc_spacing=0.005,  # placeholder, real value lives on CSLCLattice
            cslc_ka=cslc.ka,
            cslc_kl=cslc.kl,
            cslc_dc=cslc.dc,
            cslc_n_iter=cslc.n_iter,
            cslc_alpha=cslc.alpha,
        )
    elif contact_model == "hydro":
        kwargs.update(kh=hydro.kh, is_hydroelastic=True)
    # "point": no extra kwargs — bare Hunt-Crossley + Coulomb.
    return newton.ModelBuilder.ShapeConfig(**kwargs)


# ── CSLCLattice construction from Poisson-disc samples ──────────────────────


def make_cslc_pad_from_samples(
    local_pts: np.ndarray,
    local_normals: np.ndarray,
    shape_index: int,
    k_neighbors: int = 6,
) -> CSLCLattice:
    """Build a ``CSLCLattice`` whose lattice IS the Poisson-disc scatter.

    Each sample becomes one surface lattice sphere.  The neighbour graph
    is k-NN in 3-D — for a Poisson-disc scatter on a 2-D manifold this
    is a close approximation of the surface Delaunay graph and produces
    the same per-vertex valency profile (~k) that the box-face grid
    relies on for its CSLC Laplacian operator to be well-conditioned.

    Spacing = mean nearest-neighbour distance (a Poisson-disc invariant,
    ≈ r_PDS for a tight disc).  Sphere radius = spacing / 2, matching
    Newton's box convention so the kc calibration formula in
    :func:`recalibrate_kc_per_pad` reads the right per-sphere stiffness.

    Args:
        local_pts: (N, 3) Poisson-disc samples in pad-local frame.
        local_normals: (N, 3) outward normals from parent triangles.
        shape_index: which Newton shape this lattice belongs to.
        k_neighbors: k for the k-NN neighbour graph.

    Returns:
        A ``CSLCLattice`` ready for ``CSLCData.from_lattices([...])``.
    """
    from scipy.spatial import cKDTree

    n = len(local_pts)
    pts32 = local_pts.astype(np.float32)
    norms32 = local_normals.astype(np.float32)

    tree = cKDTree(pts32)
    # Query k+1 to drop self (always the nearest).
    dists, idxs = tree.query(pts32, k=k_neighbors + 1)
    spacing = float(np.mean(dists[:, 1]))
    sphere_radius = spacing * 0.5
    neighbor_indices = [idxs[i, 1:].astype(np.int32) for i in range(n)]

    return CSLCLattice(
        positions=pts32,
        radii=np.full(n, sphere_radius, dtype=np.float32),
        is_surface=np.ones(n, dtype=bool),
        outward_normals=norms32,
        neighbor_indices=neighbor_indices,
        shape_index=shape_index,
        spacing=spacing,
        sphere_radius=sphere_radius,
    )


# ── CSLC handler attach ─────────────────────────────────────────────────


def build_cslc_handler_with_mesh_pads(
    model,
    mesh_pads_by_shape: dict[int, CSLCLattice],
    cslc: CSLCParams,
) -> CSLCHandler | None:
    """Build a ``CSLCHandler`` from caller-supplied ``CSLCLattice`` objects.

    Step 7 cleanup: only MESH-like CSLC shapes (with pre-built lattices)
    are supported; the box auto-gen path was removed alongside the
    box-target kernels.  Only sphere-vs-sphere pairs are supported in
    this pass; box / mesh / SDF target geometries are deferred (step
    7b in ``cslc_main/theory/notes.md``).

    Returns ``None`` if there are no usable CSLC pairs.
    """
    shape_flags = model.shape_flags.numpy()
    shape_types = model.shape_type.numpy()
    cslc_shape_indices = [
        i for i in range(model.shape_count) if (shape_flags[i] & _CSLC_FLAG)
    ]
    if not cslc_shape_indices:
        return None

    cslc_set = set(cslc_shape_indices)
    shape_pairs: list[CSLCShapePair] = []
    if model.shape_contact_pairs is not None:
        for sa, sb in model.shape_contact_pairs.numpy():
            if sa in cslc_set and sb not in cslc_set:
                gt_other = int(shape_types[sb])
                if gt_other != _GEOTYPE_SPHERE:
                    continue
                shape_pairs.append(
                    CSLCShapePair(
                        cslc_shape=int(sa),
                        other_shape=int(sb),
                        other_geo_type=gt_other,
                    )
                )
            elif sb in cslc_set and sa not in cslc_set:
                gt_other = int(shape_types[sa])
                if gt_other != _GEOTYPE_SPHERE:
                    continue
                shape_pairs.append(
                    CSLCShapePair(
                        cslc_shape=int(sb),
                        other_shape=int(sa),
                        other_geo_type=gt_other,
                    )
                )
    if not shape_pairs:
        return None

    cslc_ka_arr = model.shape_cslc_ka.numpy()
    cslc_kl_arr = model.shape_cslc_kl.numpy()
    cslc_dc_arr = model.shape_cslc_dc.numpy()
    shape_ke = model.shape_material_ke.numpy()
    shape_scale_np = model.shape_scale.numpy()

    first_cslc = cslc_shape_indices[0]
    ka = float(cslc_ka_arr[first_cslc])
    kl = float(cslc_kl_arr[first_cslc])
    dc = float(cslc_dc_arr[first_cslc])

    lattices: list[CSLCLattice] = []
    for shape_idx in cslc_shape_indices:
        gt = int(shape_types[shape_idx])
        if gt not in _MESH_LIKE_TYPES:
            raise RuntimeError(
                f"CSLC shape {shape_idx} has geo_type {gt}; only "
                "MESH / CONVEX_MESH are supported.  Sample your shape "
                "into a CSLCLattice and pass it in mesh_pads_by_shape."
            )
        if shape_idx not in mesh_pads_by_shape:
            raise RuntimeError(
                f"CSLC shape {shape_idx} has no pre-built CSLCLattice; "
                "build one with `make_cslc_pad_from_samples`."
            )
        lattices.append(mesh_pads_by_shape[shape_idx])

    # Initial kc calibration uses contact_fraction=0.3 (generic prior).
    # Post-build, callers should re-call :func:`recalibrate_kc_per_pad`
    # with the scene-specific fraction (e.g. cslc.contact_fraction).
    ke_bulk = float(shape_ke[first_cslc])
    kc = calibrate_kc(ke_bulk, lattices, ka=ka, contact_fraction=0.3, per_lattice=True)

    cslc_data = CSLCData.from_lattices(
        lattices,
        ka=ka,
        kl=kl,
        kc=kc,
        dc=dc,
        smoothing_eps=cslc.smoothing_eps,
        ka_tangent_ratio=cslc.ka_tangent_ratio,
        k_stick=cslc.k_stick,
        mu_friction=cslc.mu_friction,
        build_A_inv=cslc.build_A_inv,
        device=model.device,
    )

    # Suppress the regular narrow-phase output for CSLC pairs --
    # otherwise we'd double-count (once from CSLC, once from
    # mesh-vs-sphere).
    if not hasattr(model, "shape_collision_filter_pairs"):
        model.shape_collision_filter_pairs = set()
    for pair in shape_pairs:
        a, b = sorted((pair.cslc_shape, pair.other_shape))
        model.shape_collision_filter_pairs.add((a, b))

    # Cache per-pair sphere-target info on the CSLCShapePair (avoids a
    # GPU->CPU sync per kernel launch).
    shape_body_np = model.shape_body.numpy()
    shape_transform_np = model.shape_transform.numpy()
    for pair in shape_pairs:
        ke_raw = float(shape_ke[pair.other_shape])
        pair.other_ke = ke_raw if ke_raw > 0.0 else 1.0e9
        # Only sphere targets reach here (filter above).
        pair.other_body = int(shape_body_np[pair.other_shape])
        xf = shape_transform_np[pair.other_shape]
        pair.other_local_pos = (float(xf[0]), float(xf[1]), float(xf[2]))
        pair.other_radius = float(shape_scale_np[pair.other_shape][0])

    # One contact slot per surface sphere; the handler writes one
    # contact per slot per pair.
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

    n_iter = (
        int(model.shape_cslc_n_iter[first_cslc])
        if model.shape_cslc_n_iter is not None
        else cslc.n_iter
    )
    alpha = (
        float(model.shape_cslc_alpha[first_cslc])
        if model.shape_cslc_alpha is not None
        else cslc.alpha
    )

    handler = CSLCHandler(
        cslc_data=cslc_data,
        shape_pairs=shape_pairs,
        n_iter=n_iter,
        alpha=alpha,
        surface_slot_map=wp.array(
            surface_slot_map, dtype=wp.int32, device=model.device
        ),
        n_surface_contacts=slot,
        n_pair_blocks=len(shape_pairs),
        device=model.device,
    )
    handler.slot_to_tid = slot_to_tid
    return handler


@contextmanager
def patched_cslc_from_model(handler: CSLCHandler):
    """Monkey-patch ``CSLCHandler._from_model`` to return our pre-built handler.

    Newton's :class:`CollisionPipeline` calls
    ``CSLCHandler._from_model(model)`` at construction.  Inside this
    context, that call returns ``handler`` regardless of the model
    passed.  The original classmethod is restored on exit.
    """
    original = CSLCHandler._from_model
    CSLCHandler._from_model = classmethod(lambda cls, model: handler)
    try:
        yield
    finally:
        CSLCHandler._from_model = original


# ── Post-build kc recalibration ────────────────────────────────────────


def recalibrate_kc_per_pad(model, contact_fraction: float) -> float | None:
    """Override per-sphere ``kc`` so each pad's aggregate stiffness ≈ ke_bulk.

    H1-aware: includes the target body's contact stiffness
    ``ke_target`` (read from the first CSLC pair's cached ``other_ke``)
    in the series chain.  The fair-calibration identity composes three
    springs per sphere::

        1/keff_per_sphere = 1/ka + 1/kc + 1/ke_target
        N_contact_per_pad · keff_per_sphere = ke_bulk

    Solving for ``kc``::

        1/kc = N_contact/ke_bulk - 1/ka - 1/ke_target     (exact)
        ⇒ kc = ke_bulk / N_contact                        (fallback when 1/kc ≤ 0)

    Returns the new ``kc``, or ``None`` if no CSLC handler is attached.
    """
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None

    d = handler.cslc_data
    shape_flags = model.shape_flags.numpy()
    cslc_shape_idx = next(
        (i for i in range(model.shape_count) if (shape_flags[i] & _CSLC_FLAG)),
        0,
    )
    ke_bulk = float(model.shape_material_ke.numpy()[cslc_shape_idx])

    ke_target: float | None = None
    if hasattr(handler, "shape_pairs") and handler.shape_pairs:
        ke_raw = float(handler.shape_pairs[0].other_ke)
        if ke_raw > 0.0:
            ke_target = ke_raw

    shape_ids = d.sphere_shape.numpy()
    is_surface = d.is_surface.numpy()
    n_pads = int(len(np.unique(shape_ids)))
    n_surface_per_pad = int(is_surface.sum()) // max(n_pads, 1)
    n_contact_per_pad = max(int(n_surface_per_pad * contact_fraction), 1)

    ka = float(d.ka)
    inv_keff_target = float(n_contact_per_pad) / ke_bulk
    inv_kc = inv_keff_target - 1.0 / ka
    if ke_target is not None:
        inv_kc -= 1.0 / ke_target

    if inv_kc <= 0.0:
        new_kc = ke_bulk / max(n_contact_per_pad, 1)
    else:
        new_kc = 1.0 / inv_kc

    d.kc = new_kc
    return new_kc


# ── High-level convenience ──────────────────────────────────────────────


def attach_cslc_handler_to_model(
    model,
    mesh_pads_by_shape: dict[int, CSLCLattice],
    config: GraspConfig,
) -> CSLCHandler | None:
    """Build the CSLC handler from pre-built pads and attach via patch.

    This is the high-level call from the scene builder.  It (a) builds
    the handler, (b) records it for the patch context manager to surface
    during the collision pipeline's construction, and (c) returns the
    handler so the scene builder can keep it for diagnostics.

    The caller must then enter :func:`patched_cslc_from_model` BEFORE
    constructing the collision pipeline (i.e. before
    ``model.collide(...)`` is first called or before solver
    construction triggers pipeline creation).
    """
    return build_cslc_handler_with_mesh_pads(
        model, mesh_pads_by_shape, config.cslc
    )
