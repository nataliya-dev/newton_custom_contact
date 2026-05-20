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
    compute_outward_normals_world,
    cslc_copy_active,
    jacobi_step,
    lattice_solve_equilibrium,
    write_cslc_contacts,
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

    Only sphere targets are supported in this pass.  ``other_geo_type``
    is kept as a field so that re-adding mesh / box targets in step 7b
    (one of the deferred items in notes.md) is a localised change to
    the dispatch path, not a dataclass shape change.
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
    # Sphere-target fields:
    other_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    other_radius: float = 0.0

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
        contact_count: Number of contact slots CSLC writes
            (``= n_surface_spheres * n_pair_blocks``).
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


    @property
    def contact_count(self) -> int:
        return self.n_surface_contacts * self.n_pair_blocks


    def get_phi_for_cslc_shape(self, cslc_shape_idx: int) -> wp.array:
        """Return the raw_penetration buffer that was last written for a
        given CSLC body (by shape index).

        Each pair launch uses its own scratch buffer; kernel 1 zeros every
        sphere that doesn't belong to the active lattice.  So to read body P's
        phi after collide() we need the buffer from the pair that had
        cslc_shape == P.  Returns None if the CSLC body has no supported pair.
        """
        # Walk shape_pairs in the same order launch() does, but only count
        # sphere pairs (the ones that actually allocate a buffer).
        sphere_pair_idx = 0
        for pair in self.shape_pairs:
            if pair.other_geo_type != _GEOTYPE_SPHERE:
                continue
            if pair.cslc_shape == cslc_shape_idx:
                return self.raw_penetration_pairs[sphere_pair_idx]
            sphere_pair_idx += 1
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

        Args:
            model: The simulation Model.
            state: Current State (provides body_q).
            contacts: Contacts buffer to write to.
            contact_offset: Starting index in contacts buffer for CSLC slots.
        """

        # Walk sphere-target pairs.  Step 7 cleanup: box / mesh / SDF
        # targets are deferred to step 7b.
        pair_idx = 0
        for pair in self.shape_pairs:
            if pair.other_geo_type == _GEOTYPE_SPHERE:
                pair_contact_offset = contact_offset + pair_idx * self.n_surface_contacts
                self._launch_vs_sphere(
                    model, state, contacts, pair_contact_offset, pair, pair_idx)
                pair_idx += 1
            else:
                warnings.warn(
                    f"CSLC vs geometry type {pair.other_geo_type} not yet implemented. "
                    "Only CSLC vs SPHERE is supported (step 7).",
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

