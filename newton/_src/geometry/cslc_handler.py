# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC collision handler for Newton's CollisionPipeline.

Constructed via CSLCHandler._from_model(model) during CollisionPipeline.__init__.
Called via CSLCHandler.launch() during CollisionPipeline.collide(), AFTER the
standard narrow phase has run.

File location: newton/_src/geometry/cslc_handler.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from .cslc_data import CSLCData, CSLCLattice, calibrate_kc
from .cslc_kernels import (
    compute_cslc_penetration,
    compute_cslc_penetration_point_set,
    compute_outward_normals_world,
    cslc_copy_active,
    jacobi_step,
    jacobi_step_point_set,
    lattice_solve_equilibrium,
    write_cslc_contacts,
    write_cslc_contacts_point_set,
)

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model
    from ..sim.state import State


# Must match the value in geometry/types.py
_CSLC_FLAG = 1 << 5
# GeoType.SPHERE — Newton stores this as int.  Step 7 cleanup: only
# sphere-vs-sphere pairs are supported; mesh / box / SDF target
# geometries are deferred (step 7b in cslc_main/theory/notes.md).
_GEOTYPE_SPHERE = 3   # GeoType.SPHERE


@dataclass
class CSLCShapePair:
    """A shape pair where one shape has the CSLC flag.

    Supports two target geometries (selected by ``is_point_set``):

    * Sphere target (``is_point_set = False``): single (local position,
      radius) pair.  Dispatched to ``_launch_vs_sphere``.
    * Point-set target (``is_point_set = True``): arrays of target
      positions / radii, plus the per-pair K_max contact-buffer budget.
      Dispatched to ``_launch_vs_point_set`` (C2d).

    The sphere fields stay valid (and unused) when ``is_point_set =
    True``; the point-set fields stay ``None`` / ``0`` when
    ``is_point_set = False``.
    """

    cslc_shape: int
    other_shape: int
    other_geo_type: int
    # Cached at construction — avoids GPU→CPU sync per step (Bug 3)
    other_body: int = 0
    # H1: target body's material stiffness [N/m], cached at construction
    # from model.shape_material_ke.  Used in the kernel for harmonic-
    # mean composition kc_series = kc·ke / (kc+ke+eps²).  Default 1e9
    # recovers the rigid-target limit (kc_series → kc) automatically.
    other_ke: float = 1.0e9
    # Sphere-target fields (consumed by _launch_vs_sphere):
    other_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    other_radius: float = 0.0
    # C2d point-set target fields (consumed by _launch_vs_point_set).
    # ``is_point_set`` is the dispatch flag in ``launch()``;
    # ``target_positions_local`` and ``target_radii`` are GPU-resident
    # wp.arrays in the target body's local frame, sized by
    # ``target_count``.  ``K_max`` is the per-pad-sphere contact-buffer
    # budget; size from geometry using INCLUSION_FACTOR from
    # cslc_main/theory/cslc_theory.py (see C2e helper
    # ``cslc_main.grasp.objects.compute_k_max``).  The handler's
    # runtime truncation counter (see ``_launch_vs_point_set``) is the
    # backstop if the sizing under-counts on a new geometry.
    is_point_set: bool = False
    target_positions_local: wp.array | None = None
    target_radii: wp.array | None = None
    target_count: int = 0
    K_max: int = 0

class CSLCHandler:
    """CSLC contact generation handler for Newton's collision pipeline.

    Constructed by :meth:`from_model_with_lattices`, which inspects
    the Model for CSLC-flagged shapes, consumes externally-supplied
    :class:`CSLCLattice` objects, calibrates ``kc``, and builds
    :class:`CSLCData`.  The legacy ``_from_model`` entry point now
    just warns and returns ``None`` (the box auto-gen path was
    removed in step 7).

    During ``collide()``, ``launch()`` runs the per-pair pipeline:
        1. Penetration computation (rest phi + line-of-centers normal,
           used by the linear warm-start and as a degenerate fallback).
        2. World-frame outward-normals precompute.
        3. Linear warm-start (``lattice_solve_equilibrium`` -- closed-form
           solve of ``(K + kc·I) δ = kc·phi_rest``).
        4. Damped-Jacobi refinement with the series-spring + deformed-
           centre contact law (Step 7-aligned ``jacobi_step``).
        5. Active-lattice selective copy of converged δ into
           ``CSLCData.sphere_delta`` (preserves other lattices'
           warm-start).
        6. Contact buffer writing at the deformed centre
           (``write_cslc_contacts``).

    Attributes:
        contact_count: Number of contact slots CSLC writes.  For a
            mixed-pair handler the slot budget is heterogeneous:

                sphere pair        -> n_surface_spheres   slots
                point-set pair     -> n_surface_spheres * pair.K_max

            ``contact_count`` returns the sum across all pairs.  This
            saves memory vs uniform K_max allocation -- typically a few
            MB at production densities -- and keeps the sphere kernel's
            single-slot-per-pad-sphere write pattern untouched.

    Diagnostic-array gap (C2d)
    --------------------------
    The per-contact diagnostic arrays
    (``dbg_pen_scale``, ``dbg_solver_pen``, ``dbg_effective_r``,
    ``dbg_d_proj``, ``dbg_radial``) are populated only by
    ``write_cslc_contacts`` (sphere targets).  ``write_cslc_contacts_point_set``
    does NOT take these as outputs -- per-pair diagnostics for point-set
    targets are deferred to C3+.  Readers of the diagnostic arrays must
    check ``shape_pairs[pair_idx].is_point_set`` and treat a True as
    "no per-contact diagnostics for this pair block; expect the
    sentinel pen_scale = -1.0".
    """

    def __init__(
        self,
        cslc_data: CSLCData,
        shape_pairs: list[CSLCShapePair],
        n_iter: int,
        alpha: float,
        surface_slot_map: wp.array,
        n_surface_contacts: int,
        n_pair_blocks: int,
        device: Any = None,
    ):
        self.cslc_data = cslc_data
        self.shape_pairs = shape_pairs
        self.n_iter = n_iter
        self.alpha = alpha
        self.surface_slot_map = surface_slot_map

        self.n_pair_blocks = n_pair_blocks


        self.n_surface_contacts = n_surface_contacts
        self.device = device or wp.get_device()

        self.slot_to_tid = np.full(0, -1, dtype=np.int32)
        self.debug_reason = wp.zeros(n_surface_contacts, dtype=wp.int32, device=self.device)

        # ── Per-contact diagnostic arrays (physics-neutral) ──
        # Laid out as n_pair_blocks × n_surface_contacts, indexed by the
        # global offset into the CSLC contact buffer (i.e. pair_idx *
        # n_surface_contacts + slot).  This MIRRORS the contacts buffer
        # layout so the CPU reader can index them the same way as
        # rigid_contact_stiffness[cslc_offset:cslc_offset+n_cslc].
        #
        # Previous single-n_surface_contacts sizing had a race: pair 1's
        # launch would overwrite pair 0's diagnostic writes at the same
        # slot_map indices (since slot_map is defined globally across
        # lattices).  Per-pair blocks eliminate that race.
        total_slots = n_surface_contacts * max(n_pair_blocks, 1)
        self.dbg_pen_scale   = wp.full(total_slots, -1.0, dtype=wp.float32, device=self.device)
        self.dbg_solver_pen  = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
        self.dbg_effective_r = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
        self.dbg_d_proj      = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
        self.dbg_radial      = wp.zeros(total_slots, dtype=wp.float32, device=self.device)


        n = cslc_data.n_spheres
        # One raw_penetration scratch buffer per sphere pair.  Kernel 1
        # zeros all non-active-lattice spheres in whichever buffer it
        # writes to, so a single shared buffer would leave every
        # non-most-recent lattice's phi reading as 0 in post-collide()
        # diagnostics.  Per-pair buffers preserve each lattice's
        # last-computed phi and let `get_phi_for_cslc_shape` look up the
        # right one by shape index.
        self.raw_penetration_pairs = [
            wp.zeros(n, dtype=wp.float32, device=self.device)
            for _ in range(max(n_pair_blocks, 1))
        ]
        # `self.raw_penetration` is repointed at each pair's buffer
        # inside `_launch_vs_sphere`; it's exposed for external readers
        # that want the most-recently-launched lattice's phi.
        self.raw_penetration = self.raw_penetration_pairs[0]
        self.contact_normal_scratch = wp.zeros(n, dtype=wp.vec3, device=self.device)
        # Per-sphere world-frame outward normal, refreshed once per pair
        # launch by `compute_outward_normals_world`.  Used by
        # `lattice_solve_equilibrium` to decompose the contact force
        # into normal/tangent components for the anisotropic-anchor
        # closed-form solve.  Bodies move every step so this cannot be
        # cached at construction.
        self.out_normal_world_scratch = wp.zeros(n, dtype=wp.vec3, device=self.device)
        # Vec3 ping-pong buffers for the damped Jacobi iteration; same
        # dtype + layout as CSLCData.sphere_delta.
        self._jacobi_a = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._jacobi_b = wp.zeros(n, dtype=wp.vec3, device=self.device)

        # C2d: per-pair truncation counter for point-set pairs.  One
        # int32 array per pair (sized 1, atomic-incremented inside
        # write_cslc_contacts_point_set, zeroed at the start of each
        # launch).  Sphere pairs allocate a buffer too -- unused, but
        # keeps the per-pair indexing trivial.  Read on CPU after
        # ``launch()`` returns; non-zero entries mean the K_max for
        # that pair undersized at runtime (see RuntimeWarning in
        # _launch_vs_point_set).
        self.truncation_count_pairs = [
            wp.zeros(1, dtype=wp.int32, device=self.device)
            for _ in range(max(n_pair_blocks, 1))
        ]


    @property
    def contact_count(self) -> int:
        """Total CSLC contact-buffer slots, summed across pairs.

        Heterogeneous per-pair sizing:
          * sphere pair    -> n_surface_contacts   slots
          * point-set pair -> n_surface_contacts * pair.K_max  slots
        """
        total = 0
        for pair in self.shape_pairs:
            if pair.is_point_set:
                total += self.n_surface_contacts * pair.K_max
            else:
                total += self.n_surface_contacts
        return total


    def get_phi_for_cslc_shape(self, cslc_shape_idx: int) -> wp.array | None:
        """Return the raw_penetration buffer last written for a CSLC body.

        Walks ``shape_pairs`` and returns the per-pair scratch buffer
        for the FIRST pair whose ``cslc_shape == cslc_shape_idx``.

        Phi semantics by pair type
        --------------------------
        * Sphere pair (``is_point_set == False``):
          ``phi_rest = smooth_relu((r_lat + R_target) - dist, eps) *
          smooth_step(dist, eps)``, single-target overlap; what every
          existing diagnostic reader (Step 11 figures, calibration
          scripts) was built against.

        * Point-set pair (``is_point_set == True``):
          argmax-overlap warm-start phi (the ``compute_cslc_penetration_point_set``
          output -- the per-pad-sphere most-overlapping target's phi).
          NOT a sum across all overlapping targets; if you need the
          aggregate normal-axis force you have to walk the contact
          buffer instead.

        Index correctness note (C2d)
        ----------------------------
        ``raw_penetration_pairs`` is allocated with one entry per
        ``shape_pairs`` entry, indexed by the FULL pair index (the same
        ``pair_idx`` that ``launch()`` passes to ``_launch_vs_sphere``
        / ``_launch_vs_point_set``).  An earlier version of this getter
        walked only sphere pairs with a separate counter -- which
        happened to coincide with the full pair index in sphere-only
        scenes, but returned the wrong buffer on the first mixed sphere
        + point-set scene.  The full-index walk below is correct for
        both.

        Returns:
            The raw_penetration buffer for the first matching pair, or
            ``None`` if no pair has ``cslc_shape == cslc_shape_idx``.
        """
        for pair_idx, pair in enumerate(self.shape_pairs):
            if pair.cslc_shape == cslc_shape_idx:
                return self.raw_penetration_pairs[pair_idx]
        return None



    @classmethod
    def _from_model(cls, model: "Model") -> "CSLCHandler | None":
        """Newton pipeline entry point.

        Step 7 cleanup: no shape-specific auto-gen.  This entry point now
        always returns ``None`` when any CSLC shape is flagged but no
        externally-supplied :class:`CSLCLattice` has been attached.
        Callers (e.g. ``cslc_main/grasp``) use
        :meth:`from_model_with_lattices` instead, passing the lattices
        they built from their own sampling pipeline.

        See ``cslc_main/theory/notes.md`` step 7 for the rationale; the
        box auto-gen path that used to live here was removed alongside
        the box-target kernels.
        """
        shape_flags = model.shape_flags.numpy()
        cslc_shape_indices = [
            i for i in range(model.shape_count) if (shape_flags[i] & _CSLC_FLAG)
        ]
        if not cslc_shape_indices:
            return None
        warnings.warn(
            f"CSLC shapes {cslc_shape_indices} are flagged but no lattice "
            "is attached.  Step 7 removed the box auto-gen path; callers "
            "must build CSLCLattice objects from a sampling pipeline (see "
            "cslc_main/grasp/contact_models.py for the reference flow) "
            "and call CSLCHandler.from_model_with_lattices.",
            RuntimeWarning, stacklevel=2,
        )
        return None

    @classmethod
    def from_model_with_lattices(
        cls,
        model: "Model",
        lattices_by_shape: "dict[int, CSLCLattice]",
        *,
        contact_fraction: float = 0.3,
    ) -> "CSLCHandler | None":
        """Build a :class:`CSLCHandler` from a Newton ``model`` plus
        externally-supplied :class:`CSLCLattice` objects.

        Args:
            model: Newton ``Model`` with CSLC-flagged shapes.
            lattices_by_shape: maps each CSLC-flagged shape index to a
                pre-built ``CSLCLattice``.  Every CSLC-flagged shape must
                have an entry (no auto-generation).
            contact_fraction: passed through to :func:`calibrate_kc`.

        Returns:
            The handler, or ``None`` if no CSLC shapes / no supported
            sphere-target pairs.
        """
        shape_flags = model.shape_flags.numpy()
        shape_types = model.shape_type.numpy()

        cslc_shape_indices = [
            i for i in range(model.shape_count) if (shape_flags[i] & _CSLC_FLAG)
        ]
        if not cslc_shape_indices:
            return None

        # Validate caller supplied a lattice for every CSLC-flagged shape.
        missing = [i for i in cslc_shape_indices if i not in lattices_by_shape]
        if missing:
            raise RuntimeError(
                f"CSLC-flagged shapes missing CSLCLattice: {missing}.  "
                "Provide one via lattices_by_shape."
            )

        # Find shape pairs involving CSLC shapes.  Only sphere targets
        # are supported in this pass.
        cslc_shape_set = set(cslc_shape_indices)
        shape_pairs: list[CSLCShapePair] = []
        if model.shape_contact_pairs is not None:
            for sa, sb in model.shape_contact_pairs.numpy():
                if sa in cslc_shape_set and sb not in cslc_shape_set:
                    gt = int(shape_types[sb])
                    if gt != _GEOTYPE_SPHERE:
                        continue
                    shape_pairs.append(CSLCShapePair(
                        cslc_shape=int(sa), other_shape=int(sb),
                        other_geo_type=gt,
                    ))
                elif sb in cslc_shape_set and sa not in cslc_shape_set:
                    gt = int(shape_types[sa])
                    if gt != _GEOTYPE_SPHERE:
                        continue
                    shape_pairs.append(CSLCShapePair(
                        cslc_shape=int(sb), other_shape=int(sa),
                        other_geo_type=gt,
                    ))
                # Both CSLC: not supported.

        if not shape_pairs:
            return None

        # Per-shape CSLC parameters.
        cslc_ka_arr = model.shape_cslc_ka.numpy()
        cslc_kl_arr = model.shape_cslc_kl.numpy()
        cslc_dc_arr = model.shape_cslc_dc.numpy()
        shape_ke = model.shape_material_ke.numpy()
        shape_scale_np = model.shape_scale.numpy()
        first_cslc = cslc_shape_indices[0]
        ka = float(cslc_ka_arr[first_cslc])
        kl = float(cslc_kl_arr[first_cslc])
        dc = float(cslc_dc_arr[first_cslc])

        # Calibrate kc on the externally-supplied lattices.
        lattices_ordered = [lattices_by_shape[i] for i in cslc_shape_indices]
        ke_bulk = float(shape_ke[first_cslc])
        kc = calibrate_kc(ke_bulk, lattices_ordered, ka=ka,
                          contact_fraction=contact_fraction, per_lattice=True)

        cslc_data = CSLCData.from_lattices(
            lattices_ordered, ka=ka, kl=kl, kc=kc, dc=dc,
            build_A_inv=True,
            device=model.device,
        )

        # Filter CSLC pairs from narrow phase (avoid double-counting).
        if not hasattr(model, 'shape_collision_filter_pairs'):
            model.shape_collision_filter_pairs = set()
        for pair in shape_pairs:
            a = min(pair.cslc_shape, pair.other_shape)
            b = max(pair.cslc_shape, pair.other_shape)
            model.shape_collision_filter_pairs.add((a, b))

        # Cache per-pair sphere-target info (one-time CPU read).
        shape_body_np = model.shape_body.numpy()
        shape_transform_np = model.shape_transform.numpy()
        for pair in shape_pairs:
            # Target stiffness for harmonic-mean composition; clamp to
            # rigid-ish if material reports zero.
            ke_raw = float(shape_ke[pair.other_shape])
            pair.other_ke = ke_raw if ke_raw > 0.0 else 1.0e9
            pair.other_body = int(shape_body_np[pair.other_shape])
            xform = shape_transform_np[pair.other_shape]
            pair.other_local_pos = (
                float(xform[0]), float(xform[1]), float(xform[2]),
            )
            pair.other_radius = float(shape_scale_np[pair.other_shape][0])

        # Surface slot map: surface sphere i -> sequential slot index.
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

        n_pair_blocks = len(shape_pairs)
        if n_pair_blocks == 0:
            return None

        # Solver params.
        if model.shape_cslc_n_iter is not None:
            n_iter = int(model.shape_cslc_n_iter[first_cslc])
        else:
            n_iter = 40
        if model.shape_cslc_alpha is not None:
            alpha = float(model.shape_cslc_alpha[first_cslc])
        else:
            alpha = 0.3

        handler = cls(
            cslc_data=cslc_data,
            shape_pairs=shape_pairs,
            n_iter=n_iter,
            alpha=alpha,
            surface_slot_map=wp.array(surface_slot_map, dtype=wp.int32, device=model.device),
            n_surface_contacts=slot,
            n_pair_blocks=n_pair_blocks,
            device=model.device,
        )
        handler.slot_to_tid = slot_to_tid
        return handler

    def launch(
        self,
        model: Model,
        state: State,
        contacts: Contacts,
        contact_offset: int,
    ) -> None:
        """Run CSLC narrow phase: penetration -> Jacobi -> contact writing.

        Called by CollisionPipeline.collide() AFTER the standard narrow phase.

        Dispatch:
          * ``pair.is_point_set``                 -> ``_launch_vs_point_set`` (C2d)
          * ``pair.other_geo_type == SPHERE``     -> ``_launch_vs_sphere``
          * otherwise                             -> RuntimeWarning, skipped

        Per-pair buffer offsets accumulate heterogeneously (sphere pair
        = ``n_surface_contacts`` slots; point-set pair = ``n_surface_contacts
        * pair.K_max`` slots) -- see the ``contact_count`` property.

        After all pairs launch, CPU-reads the per-pair truncation
        counters for point-set pairs and raises ``RuntimeWarning`` if
        any pad sphere overflowed its K_max budget -- the actionable
        signal to raise ``K_max`` in the caller's geometry-derived
        sizing.

        Args:
            model: The simulation Model.
            state: Current State (provides body_q).
            contacts: Contacts buffer to write to.
            contact_offset: Starting index in contacts buffer for CSLC slots.
        """

        # Accumulate the per-pair contact-buffer offset.  Heterogeneous
        # sizing means we cannot use ``pair_idx * n_surface_contacts``
        # uniformly any more.
        pair_offset = contact_offset
        truncation_pairs: list[tuple[int, int]] = []  # (pair_idx, K_max)
        for pair_idx, pair in enumerate(self.shape_pairs):
            if pair.is_point_set:
                self._launch_vs_point_set(
                    model, state, contacts, pair_offset, pair, pair_idx)
                pair_offset += self.n_surface_contacts * pair.K_max
                truncation_pairs.append((pair_idx, pair.K_max))
            elif pair.other_geo_type == _GEOTYPE_SPHERE:
                self._launch_vs_sphere(
                    model, state, contacts, pair_offset, pair, pair_idx)
                pair_offset += self.n_surface_contacts
            else:
                warnings.warn(
                    f"CSLC vs geometry type {pair.other_geo_type} not yet "
                    "implemented.  Supported: SPHERE (sphere-target path), "
                    "or any geo with ``pair.is_point_set = True`` (point-set "
                    "path).  Pair skipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # CPU read of truncation counters for point-set pairs.  Each
        # counter is the number of PAD SPHERES that overflowed K_max
        # (not the number of dropped pairs).  Non-zero means K_max was
        # under-sized for this scene's geometry -- the active-set
        # inclusion radius (r_lat + R + INCLUSION_FACTOR*eps from
        # cslc_main/theory/cslc_theory.py) is wider than the K_max
        # buffer accommodated, so a non-trivial slice of contact
        # wrench was silently dropped.
        for pair_idx, k_max in truncation_pairs:
            n_trunc = int(self.truncation_count_pairs[pair_idx].numpy()[0])
            if n_trunc > 0:
                warnings.warn(
                    f"CSLC point-set pair {pair_idx} truncated K_max={k_max} "
                    f"on {n_trunc} pad sphere(s).  Up to "
                    f"{n_trunc} * (active_pairs_per_pad - K_max) contact "
                    f"pairs were silently dropped, with per-pair force in "
                    f"the 0.025-0.4 N range each at production parameters. "
                    f"Raise K_max for this pair (either in the geometry-"
                    f"derived sizing in cslc_main/grasp/objects.compute_k_max, "
                    f"or via direct CSLCShapePair.K_max override).",
                    RuntimeWarning,
                    stacklevel=2,
                )


    def _launch_vs_sphere(
            self,
            model: Model,
            state: State,
            contacts: Contacts,
            contact_offset: int,
            pair: CSLCShapePair,
            pair_idx: int,
        ) -> None:
            """Per-pair kernel pipeline for CSLC lattice vs sphere target."""
            data = self.cslc_data

            # Each pair gets its own raw_penetration scratch so the
            # post-collide() diagnostic can inspect per-lattice phi.
            # Update the externally-visible alias to point at this pair.
            pen_buf = self.raw_penetration_pairs[pair_idx]
            self.raw_penetration = pen_buf

            # Use the per-pair sphere-target info cached at construction
            # in `from_model_with_lattices` -- avoids a GPU→CPU sync per
            # kernel launch.
            target_body = pair.other_body
            target_local_pos = wp.vec3(
                pair.other_local_pos[0],
                pair.other_local_pos[1],
                pair.other_local_pos[2],
            )
            target_radius = pair.other_radius

            # Smoothing width for the differentiable surrogates in
            # cslc_kernels.py.  Recovers the hard-cull behaviour in the
            # eps → 0 limit.
            eps = float(data.smoothing_eps)

            # ── Kernel 1: Raw penetration ──
            wp.launch(
                kernel=compute_cslc_penetration,
                dim=data.n_spheres,
                inputs=[
                    data.positions, data.radii, data.sphere_delta,
                    data.sphere_shape, data.is_surface, data.outward_normals,
                    state.body_q, model.shape_body, model.shape_transform,
                    pair.cslc_shape,
                    target_body, pair.other_shape, target_local_pos, target_radius,
                    eps,
                ],
                outputs=[pen_buf, self.contact_normal_scratch],
                device=self.device,
            )

            # ── Kernel 1b: World-frame outward normals ──
            wp.launch(
                kernel=compute_outward_normals_world,
                dim=data.n_spheres,
                inputs=[
                    data.outward_normals, data.sphere_shape,
                    state.body_q, model.shape_body, model.shape_transform,
                ],
                outputs=[self.out_normal_world_scratch],
                device=self.device,
            )

            # ── Kernel 2: Lattice equilibrium solve ──
            # Strategy: A_inv WARM-START + jacobi REFINE.
            #
            # `lattice_solve_equilibrium` solves the LINEARISED system
            # (K + kc·I) δ = kc · phi_rest -- a tangent-space approximation
            # of the series-spring law at δ = 0.  At finite δ the true
            # contact load is kc · phi_eff(δ) · smooth_step(raw) · n_eff_def
            # (Step 7 D1-D2), which the linear matvec doesn't see;
            # `jacobi_step` then iterates from this warm-start delta and
            # corrects the nonlinear residual.
            #
            # Empirically ~5-10 jacobi iterations on top of the warm-start
            # converge dome configurations; flat/face-on near-rest scenes
            # bottom out at 1-2 iterations (the linearisation is near-exact
            # there).
            if data.A_inv is not None:
                wp.launch(
                    kernel=lattice_solve_equilibrium,
                    dim=data.n_spheres,
                    inputs=[data.A_inv, data.A_inv_t, pen_buf,
                            self.contact_normal_scratch,
                            self.out_normal_world_scratch, data.kc],
                    outputs=[self._jacobi_a],
                    device=self.device,
                )
                src, dst = self._jacobi_a, self._jacobi_b
            else:
                # Fallback: cold-start from warm-state.
                wp.copy(self._jacobi_a, data.sphere_delta)
                src, dst = self._jacobi_a, self._jacobi_b

            # Nonlinear refinement: damped Jacobi with the distance-
            # preservation lateral force and the Step 7-aligned
            # series-spring contact at the deformed centre.
            # Iterations are governed by self.n_iter (defaults to 40
            # in the handler ctor; overridable per scene via
            # shape_cslc_n_iter).  The linear warm-start absorbs most
            # of the equilibrium displacement; the iterations only
            # need to converge the geometric-nonlinear correction
            # (distance-preserving spring on curved patches +
            # deformed-direction contact normal).
            for _ in range(self.n_iter):
                wp.launch(
                    kernel=jacobi_step,
                    dim=data.n_spheres,
                    inputs=[
                        src, dst,
                        # Step 7 D2: per-sphere radii so the contact
                        # block can compute the EXACT deformed overlap.
                        data.radii,
                        data.positions,
                        data.neighbor_rest_length,
                        data.is_surface,
                        data.neighbor_start, data.neighbor_count,
                        data.neighbor_list,
                        data.ka, data.kl, data.kc, self.alpha,
                        data.sphere_shape, pair.cslc_shape,
                        data.outward_normals,
                        state.body_q, model.shape_body, model.shape_transform,
                        data.ka_tangent_ratio,
                        data.k_stick, data.mu_friction,
                        # Step 7 D2: deformed contact direction +
                        # exact phi_def.  jacobi_step recomputes both
                        # n_eff and raw from q_i_world, target_local_pos,
                        # and target_radius each iter.
                        target_body, target_local_pos, target_radius,
                        # No external tangential load in production
                        # (apex_idx = -1 is the no-op sentinel).  Bridge
                        # test driver overrides for friction scenes.
                        int(-1), wp.vec3(0.0, 0.0, 0.0),
                        eps,
                    ],
                    device=self.device,
                )
                src, dst = dst, src

            # Warm-start: write converged delta back, but ONLY for the
            # active lattice's spheres.  In the dense-solve path, `src` carries
            # zeros for non-active spheres (because their phi was zeroed
            # in Kernel 1, and lattice_solve_equilibrium runs unfiltered);
            # an unconditional `wp.copy` would wipe the other lattice's
            # warm-start every step.  The iterative jacobi path passes
            # non-active deltas through unchanged via `jacobi_step`'s lattice
            # filter, so this selective copy is also a safe no-op for
            # non-active spheres in that path.  See cslc_kernels.py
            # `cslc_copy_active` for the rationale.
            wp.launch(
                kernel=cslc_copy_active,
                dim=data.n_spheres,
                inputs=[src, data.sphere_shape, pair.cslc_shape],
                outputs=[data.sphere_delta],
                device=self.device,
            )

            # ── Kernel 3: Write contacts ──
            wp.launch(
                kernel=write_cslc_contacts,
                dim=data.n_spheres,
                inputs=[
                    data.positions, data.radii, src,
                    data.sphere_shape, data.is_surface, data.outward_normals,
                    state.body_q, model.shape_body, model.shape_transform,
                    pair.cslc_shape,
                    target_body, pair.other_shape, target_local_pos, target_radius,
                    contact_offset, self.surface_slot_map,
                    pen_buf,
                    # Contacts buffer arrays
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    contacts.rigid_contact_tids,
                    # Per-contact material properties
                    model.shape_material_mu,
                    data.kc,
                    pair.other_ke,  # H1: target stiffness for harmonic-mean composition
                    data.dc,
                    eps,
                    contacts.rigid_contact_stiffness,
                    contacts.rigid_contact_damping,
                    contacts.rigid_contact_friction,
                    self.debug_reason,
                    # Per-pair diagnostic block: pair_idx * n_surface_contacts
                    # is the starting offset into each diagnostic array.
                    pair_idx * self.n_surface_contacts,  # diag_offset
                    self.dbg_pen_scale,
                    self.dbg_solver_pen,
                    self.dbg_effective_r,
                    self.dbg_d_proj,
                    self.dbg_radial,
                ],
                device=self.device,
            )


    def _launch_vs_point_set(
        self,
        model: "Model",
        state: "State",
        contacts: "Contacts",
        contact_offset: int,
        pair: CSLCShapePair,
        pair_idx: int,
    ) -> None:
        """Per-pair kernel pipeline for CSLC lattice vs PointSetTarget (C2d).

        Structural mirror of :meth:`_launch_vs_sphere` with the point-set
        kernel triple substituted in.  Launch order:

            1. ``compute_cslc_penetration_point_set``   (argmax-overlap warm-start)
            2. ``compute_outward_normals_world``        (shared with sphere path)
            3. ``lattice_solve_equilibrium``            (unchanged — consumes phi + n_eff)
            4. ``jacobi_step_point_set`` × ``n_iter``   (point-set Jacobi refinement)
            5. ``cslc_copy_active``                     (active-lattice selective copy)
            6. ``write_cslc_contacts_point_set``        (K_max contacts per pad sphere)

        The warm-start uses the argmax-overlap target per pad sphere
        (C2d option a').  This picks the single most-overlapping target
        per pad sphere, then feeds the resulting (phi_rest, n_eff)
        pair into the existing ``lattice_solve_equilibrium`` solver --
        no new linear-solve kernel.  The dominant single-contact
        equilibrium is absorbed by the warm-start; ``jacobi_step_point_set``
        sweeps refine the multi-point correction.  At production
        ``n_iter=40`` this matches the sphere-target convergence
        characterisation, so the same iteration budget applies to both
        paths.
        """
        data = self.cslc_data

        # Each pair gets its own raw_penetration scratch (reused across
        # sphere and point-set warm-starts).  The externally-visible
        # alias points at this pair so post-collide() readers see the
        # latest phi for this lattice.
        pen_buf = self.raw_penetration_pairs[pair_idx]
        self.raw_penetration = pen_buf

        eps = float(data.smoothing_eps)

        # Reset the per-pair truncation counter before this launch.
        # CPU read happens in ``launch()`` after collide() returns.
        self.truncation_count_pairs[pair_idx].zero_()

        # ── Kernel 1: Argmax-overlap warm-start penetration ──
        # NOTE: signature differs slightly from compute_cslc_penetration
        # -- no sphere_outward_normal (point-set kernel handles
        # degenerate-coincident targets by ``continue``, never falls
        # back to the rest outward normal).
        wp.launch(
            kernel=compute_cslc_penetration_point_set,
            dim=data.n_spheres,
            inputs=[
                data.positions, data.radii, data.sphere_delta,
                data.sphere_shape, data.is_surface,
                state.body_q, model.shape_body, model.shape_transform,
                pair.cslc_shape,
                pair.other_body,
                pair.target_positions_local, pair.target_radii,
                pair.target_count,
                eps,
            ],
            outputs=[pen_buf, self.contact_normal_scratch],
            device=self.device,
        )

        # ── Kernel 1b: World-frame outward normals (shared) ──
        wp.launch(
            kernel=compute_outward_normals_world,
            dim=data.n_spheres,
            inputs=[
                data.outward_normals, data.sphere_shape,
                state.body_q, model.shape_body, model.shape_transform,
            ],
            outputs=[self.out_normal_world_scratch],
            device=self.device,
        )

        # ── Kernel 2: Lattice equilibrium solve (unchanged from sphere path) ──
        # See _launch_vs_sphere for the linear-warm-start rationale; the
        # closed-form solve consumes the argmax-overlap (phi, n_eff) pair
        # exactly as it does for sphere targets.
        if data.A_inv is not None:
            wp.launch(
                kernel=lattice_solve_equilibrium,
                dim=data.n_spheres,
                inputs=[data.A_inv, data.A_inv_t, pen_buf,
                        self.contact_normal_scratch,
                        self.out_normal_world_scratch, data.kc],
                outputs=[self._jacobi_a],
                device=self.device,
            )
            src, dst = self._jacobi_a, self._jacobi_b
        else:
            wp.copy(self._jacobi_a, data.sphere_delta)
            src, dst = self._jacobi_a, self._jacobi_b

        # ── Damped Jacobi refinement with the point-set contact law ──
        # Same n_iter budget as the sphere path: the argmax-overlap
        # warm-start absorbs the dominant single-contact equilibrium,
        # leaving only the multi-point correction for the Jacobi
        # sweeps to converge.
        for _ in range(self.n_iter):
            wp.launch(
                kernel=jacobi_step_point_set,
                dim=data.n_spheres,
                inputs=[
                    src, dst,
                    data.radii,
                    data.positions,
                    data.neighbor_rest_length,
                    data.is_surface,
                    data.neighbor_start, data.neighbor_count,
                    data.neighbor_list,
                    data.ka, data.kl, data.kc, self.alpha,
                    data.sphere_shape, pair.cslc_shape,
                    data.outward_normals,
                    state.body_q, model.shape_body, model.shape_transform,
                    data.ka_tangent_ratio,
                    data.k_stick, data.mu_friction,
                    pair.target_positions_local, pair.target_radii,
                    pair.target_count, pair.other_body,
                    # No external tangential load in production
                    # (apex_idx = -1 is the no-op sentinel).
                    int(-1), wp.vec3(0.0, 0.0, 0.0),
                    eps,
                ],
                device=self.device,
            )
            src, dst = dst, src

        # ── Active-lattice selective copy of converged delta ──
        wp.launch(
            kernel=cslc_copy_active,
            dim=data.n_spheres,
            inputs=[src, data.sphere_shape, pair.cslc_shape],
            outputs=[data.sphere_delta],
            device=self.device,
        )

        # ── Kernel 3: Write per-pair contacts (up to K_max per pad sphere) ──
        # Per-pad-sphere diagnostic arrays (dbg_pen_scale, etc.) are NOT
        # populated by this kernel -- the point-set emit path does not
        # take them as outputs.  See the class docstring's "Diagnostic-
        # array gap" note; the diagnostic readers must check
        # ``shape_pairs[pair_idx].is_point_set`` and skip these blocks.
        wp.launch(
            kernel=write_cslc_contacts_point_set,
            dim=data.n_spheres,
            inputs=[
                data.positions, data.radii, src,
                data.sphere_shape, data.is_surface, data.outward_normals,
                state.body_q, model.shape_body, model.shape_transform,
                pair.cslc_shape,
                pair.other_body, pair.other_shape,
                pair.target_positions_local, pair.target_radii,
                pair.target_count,
                contact_offset, pair.K_max, self.surface_slot_map,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_tids,
                model.shape_material_mu,
                data.kc,
                pair.other_ke,
                data.dc,
                eps,
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
                contacts.rigid_contact_friction,
                self.truncation_count_pairs[pair_idx],
            ],
            device=self.device,
        )

