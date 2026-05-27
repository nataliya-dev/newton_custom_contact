# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC collision handler for Newton's CollisionPipeline (CSLC v2 unified path).

Constructed via :meth:`CSLCHandler._from_model` during
``CollisionPipeline.__init__``.  Called via :meth:`CSLCHandler.launch`
during ``CollisionPipeline.collide()`` after the standard narrow phase
has run.

v2 unification (Phase 5).  The handler now has a single launch path:
each :class:`CSLCShapePair` carries point-set target arrays
``(target_positions_local, target_normals_local, target_areas_local)``
and dispatches to :meth:`_launch`, which runs the contract §6.5 Jacobi
solve against the unified ``cslc_kernels`` triple
(``compute_cslc_penetration`` → ``jacobi_step`` → ``write_cslc_contacts``).
The v1 sphere-target path (``_launch_vs_sphere`` and the
``is_point_set`` dispatch flag) was deleted in Phase 5 -- sphere
targets are now sampled as point-sets via, e.g.,
:func:`cslc_main.theory.cslc_targets.make_sphere_target`.

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


@dataclass
class CSLCShapePair:
    """A shape pair where one shape has the CSLC flag.

    v2 unified path: every pair is a point-set target.  Callers populate
    the ``target_*`` arrays from a sampling pipeline (sphere targets
    sampled via :func:`cslc_main.theory.cslc_targets.make_sphere_target`,
    box targets via :func:`make_box_target`, meshes via
    :func:`make_mesh_target`).  ``K_max`` is the per-pad-sphere
    contact-buffer budget; size from geometry using ``INCLUSION_FACTOR``
    from ``cslc_main.theory.cslc_theory`` (see helper
    :func:`cslc_main.grasp.objects.compute_k_max`).  The runtime
    truncation counter inside :meth:`CSLCHandler._launch` is the
    backstop if the sizing under-counts on a new geometry.

    Phase 5 (v2 unified path) changes:
      * Dropped ``is_point_set`` dispatch flag -- every pair is
        point-set.
      * Dropped ``other_local_pos`` and ``other_radius`` sphere-target
        fields -- sphere targets are now sampled as point-sets.
      * Dropped ``target_radii`` array -- the kernel half-space form
        ``raw = r_i − n_face · (q − t)`` does not consume per-sample
        radii.
    """

    cslc_shape: int
    other_shape: int
    other_geo_type: int
    # Cached at construction -- avoids GPU→CPU sync per step.
    other_body: int = 0
    # Target body's material stiffness [N/m], cached at construction
    # from ``model.shape_material_ke``.  Used in the kernel for
    # harmonic-mean composition ``kc_series = kc·ke / (kc + ke + ε²)``.
    # Default 1e9 recovers the rigid-target limit (kc_series → kc)
    # automatically.
    other_ke: float = 1.0e9
    # Point-set target arrays (GPU-resident wp.arrays, target body-local).
    target_positions_local: wp.array | None = None
    # Per-target outward face normal (target body-local).  The contact
    # direction is the target surface's outward normal at the sample
    # point.  Half-space raw = r_lat − n_face · (q − t) is monotone in
    # penetration depth and does not flip sign at face crossing.
    target_normals_local: wp.array | None = None
    # Per-target Voronoi area on the underlying surface [m²].  Used by
    # the area-weighted half-space contact form
    #     F_j = kc · A_j · w_tangent_j · phi_eff_j · n_face_j
    # so the discrete sum approximates the surface integral
    # ``∫ kc · phi · n_face dA`` over the contact patch.
    target_areas_local: wp.array | None = None
    target_count: int = 0
    K_max: int = 0


class CSLCHandler:
    """CSLC contact generation handler for Newton's collision pipeline.

    Constructed by :meth:`from_model_with_lattices`, which inspects
    the Model for CSLC-flagged shapes, consumes externally-supplied
    :class:`CSLCLattice` objects, calibrates ``kc``, and builds
    :class:`CSLCData`.  The legacy ``_from_model`` entry point now
    just warns and returns ``None`` (the box auto-gen path was removed
    in step 7).

    During ``collide()``, ``launch()`` runs the per-pair v2 pipeline:

        1. ``compute_cslc_penetration``  -- argmax-overlap warm-start
           penetration over the point-set target (half-space raw + the
           contract §3.6 alignment hard-cull).
        2. ``compute_outward_normals_world`` -- per-sphere world-frame
           outward normals for the linear warm-start solve.
        3. ``lattice_solve_equilibrium`` -- closed-form
           ``(K + kc·I) δ = kc · phi_rest`` warm-start.
        4. ``jacobi_step`` × ``n_iter`` -- damped-Jacobi refinement
           with the unified half-space contact, contract §6.5.
        5. ``cslc_copy_active`` -- active-lattice selective copy back
           into ``CSLCData.sphere_delta`` (preserves other lattices'
           warm-starts).
        6. ``write_cslc_contacts`` -- emit up to ``K_max`` MuJoCo
           contacts per pad sphere, contract §8 convention.

    Attributes:
        contact_count: Number of contact slots CSLC writes.  Every
            pair contributes ``n_surface_contacts * K_max`` slots.

    Diagnostic-array gap
    --------------------
    Per-contact diagnostic arrays
    (``dbg_pen_scale``, ``dbg_solver_pen``, ``dbg_effective_r``,
    ``dbg_d_proj``, ``dbg_radial``) were populated only by the v1
    sphere-target ``write_cslc_contacts`` and are NOT populated by the
    v2 unified emission kernel.  Per-pair diagnostics for the
    point-set path are deferred to C3+; readers should expect the
    sentinel ``pen_scale = -1.0`` everywhere.
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
        # Retained as sentinel buffers for diagnostic readers that still
        # query them; the v2 unified emission kernel does not populate
        # these (see class docstring's "Diagnostic-array gap").
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
        # last-computed phi and let :meth:`get_phi_for_cslc_shape` look
        # up the right one by shape index.
        self.raw_penetration_pairs = [
            wp.zeros(n, dtype=wp.float32, device=self.device)
            for _ in range(max(n_pair_blocks, 1))
        ]
        # ``self.raw_penetration`` is repointed at each pair's buffer
        # inside ``_launch``; exposed for external readers that want
        # the most-recently-launched lattice's phi.
        self.raw_penetration = self.raw_penetration_pairs[0]
        self.contact_normal_scratch = wp.zeros(n, dtype=wp.vec3, device=self.device)
        # Per-sphere world-frame outward normal, refreshed once per pair
        # launch by ``compute_outward_normals_world``.  Used by
        # ``lattice_solve_equilibrium`` to decompose the contact force
        # into normal/tangent components for the anisotropic-anchor
        # closed-form solve.  Bodies move every step so this cannot be
        # cached at construction.
        self.out_normal_world_scratch = wp.zeros(n, dtype=wp.vec3, device=self.device)
        # Vec3 ping-pong buffers for the damped Jacobi iteration; same
        # dtype + layout as ``CSLCData.sphere_delta``.
        self._jacobi_a = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._jacobi_b = wp.zeros(n, dtype=wp.vec3, device=self.device)

        # Per-pair truncation counter for the v2 emission kernel.  One
        # int32 array per pair (sized 1, atomic-incremented inside
        # ``write_cslc_contacts``, zeroed at the start of each launch).
        # Read on CPU after ``launch()`` returns; non-zero entries mean
        # the K_max for that pair undersized at runtime (see
        # RuntimeWarning in :meth:`_launch`).
        self.truncation_count_pairs = [
            wp.zeros(1, dtype=wp.int32, device=self.device)
            for _ in range(max(n_pair_blocks, 1))
        ]

    @property
    def contact_count(self) -> int:
        """Total CSLC contact-buffer slots, summed across pairs.

        Each pair contributes ``n_surface_contacts * K_max`` slots
        (one slot per (pad_sphere, target_sample) active pair, up to
        the per-pad ``K_max`` budget).
        """
        return sum(
            self.n_surface_contacts * pair.K_max for pair in self.shape_pairs
        )

    def get_phi_for_cslc_shape(self, cslc_shape_idx: int) -> wp.array | None:
        """Return the raw_penetration buffer last written for a CSLC body.

        Walks ``shape_pairs`` and returns the per-pair scratch buffer
        for the FIRST pair whose ``cslc_shape == cslc_shape_idx``.

        Phi semantics
        -------------
        Argmax-overlap warm-start phi (the
        ``compute_cslc_penetration`` output -- the per-pad-sphere
        most-overlapping target's phi).  NOT a sum across all
        overlapping targets; if you need the aggregate normal-axis
        force, walk the contact buffer instead.

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

        Step 7 cleanup: no shape-specific auto-gen.  Returns ``None``
        when any CSLC shape is flagged but no externally-supplied
        :class:`CSLCLattice` has been attached.  Callers
        (e.g. ``cslc_main/grasp``) use :meth:`from_model_with_lattices`
        instead, passing the lattices they built from their own
        sampling pipeline.
        """
        shape_flags = model.shape_flags.numpy()
        cslc_shape_indices = [
            i for i in range(model.shape_count) if (shape_flags[i] & _CSLC_FLAG)
        ]
        if not cslc_shape_indices:
            return None
        warnings.warn(
            f"CSLC shapes {cslc_shape_indices} are flagged but no lattice "
            "is attached.  Callers must build CSLCLattice objects from a "
            "sampling pipeline (see cslc_main/grasp/contact_models.py for "
            "the reference flow) and call CSLCHandler.from_model_with_lattices.",
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

        Phase 5 (v2 unified): this generic entry point builds CSLC pairs
        with *empty* target arrays -- callers (e.g.
        :func:`cslc_main.grasp.contact_models.build_cslc_handler_with_mesh_pads`)
        must populate ``target_positions_local`` /
        ``target_normals_local`` / ``target_areas_local`` and ``K_max``
        on each pair before the first :meth:`launch` call.

        Args:
            model: Newton ``Model`` with CSLC-flagged shapes.
            lattices_by_shape: maps each CSLC-flagged shape index to a
                pre-built ``CSLCLattice``.  Every CSLC-flagged shape
                must have an entry (no auto-generation).
            contact_fraction: passed through to :func:`calibrate_kc`.

        Returns:
            The handler, or ``None`` if no CSLC shapes / no contact
            pairs.
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

        # Build shape pairs involving CSLC shapes.  Phase 5: every pair
        # is a point-set target; callers populate target_* arrays
        # post-construction (see class docstring).
        cslc_shape_set = set(cslc_shape_indices)
        shape_pairs: list[CSLCShapePair] = []
        if model.shape_contact_pairs is not None:
            for sa, sb in model.shape_contact_pairs.numpy():
                if sa in cslc_shape_set and sb not in cslc_shape_set:
                    cslc_shape, other = int(sa), int(sb)
                elif sb in cslc_shape_set and sa not in cslc_shape_set:
                    cslc_shape, other = int(sb), int(sa)
                else:
                    continue  # both CSLC or neither: not supported
                shape_pairs.append(CSLCShapePair(
                    cslc_shape=cslc_shape, other_shape=other,
                    other_geo_type=int(shape_types[other]),
                ))

        if not shape_pairs:
            return None

        # Per-shape CSLC parameters.
        cslc_ka_arr = model.shape_cslc_ka.numpy()
        cslc_kl_arr = model.shape_cslc_kl.numpy()
        cslc_dc_arr = model.shape_cslc_dc.numpy()
        shape_ke = model.shape_material_ke.numpy()
        first_cslc = cslc_shape_indices[0]
        ka = float(cslc_ka_arr[first_cslc])
        kl = float(cslc_kl_arr[first_cslc])
        dc = float(cslc_dc_arr[first_cslc])

        # Read each pair's target body + target ke up front so the
        # subsequent kc composition has the right ke_target available.
        # Heterogeneous ke_target across pairs would require per-pair
        # kc storage; for now we use the first pair's ke (production
        # grasp has two pads on one object, single ke_target).
        shape_body_np = model.shape_body.numpy()
        for pair in shape_pairs:
            ke_raw = float(shape_ke[pair.other_shape])
            pair.other_ke = ke_raw if ke_raw > 0.0 else 1.0e9
            pair.other_body = int(shape_body_np[pair.other_shape])

        # Calibrate kc on the externally-supplied lattices.  ``calibrate_kc``
        # returns a per-sphere [N/m] stiffness satisfying the contract §10
        # series-spring identity ``1/k_c = N_contact/k_e_bulk − 1/k_a
        # − 1/k_e_target``.
        lattices_ordered = [lattices_by_shape[i] for i in cslc_shape_indices]
        ke_bulk = float(shape_ke[first_cslc])
        kc_per_sphere = calibrate_kc(
            ke_bulk, lattices_ordered, ka=ka,
            contact_fraction=contact_fraction, per_lattice=True,
        )

        # Compose kc with the target's contact stiffness ke_target
        # (Phase 6, contract §11 amendment).  The two contact springs
        # (pad + target body) are in series: every pair carries the
        # spring rate ``k_pair_eff = kc_per_sphere · ke_target /
        # (kc_per_sphere + ke_target)``.  v1 buried this composition
        # inside ``write_cslc_contacts`` (``kc_series = kc·ke/(kc+ke
        # +ε²)``) but mixed units — kc was per-volume [N/m³] there,
        # ke_target per-pair [N/m] — and ``jacobi_step`` skipped the
        # composition entirely, so the lattice solver and emission
        # disagreed on the per-pair force at any finite ke_target.
        # Pre-composing here (units consistent: both [N/m]) gives a
        # single global kc the kernels can use directly.  For multiple
        # pairs with the same target body, this is unambiguous; for
        # heterogeneous ke_targets a future extension would store kc
        # per-pair.
        ke_target = float(shape_pairs[0].other_ke)
        kc_per_sphere_eff = (
            kc_per_sphere * ke_target / (kc_per_sphere + ke_target)
        )

        # Per-volume rescale (Option-2 tiling): the v2 unified contact
        # kernels (``jacobi_step``, ``write_cslc_contacts``) multiply
        # ``kc`` by the per-sample Voronoi area ``A_j`` and the locality
        # kernel ``w_tangent`` (half-width = r_pad), so the kc the
        # kernel expects is per-volume [N/m³], not the per-sphere [N/m]
        # that ``calibrate_kc`` returns.  Divide by the kernel disc
        # area ``A_kernel = π·r_pad²``.  Pre-Option-2 this used 3·r_pad
        # and paired with a CSLC_SOFTENING ≈ 0.1 hack to cancel the
        # resulting 7× kernel-overlap error on a Lloyd lattice; with
        # kernel_h = r_pad the discs tile without overlap and the
        # identity ``kc_per_volume · A_kernel = kc_per_sphere`` is
        # exact.
        first_lat = lattices_ordered[0]
        r_pad_avg = float(np.mean(
            first_lat.radii[first_lat.is_surface.astype(bool)]
        ))
        A_kernel = float(np.pi * r_pad_avg * r_pad_avg)
        kc = kc_per_sphere_eff / A_kernel

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

        # (Per-pair target body + target ke were cached above before
        # the kc composition.)  Target arrays (positions/normals/areas)
        # and K_max are populated by the caller post-construction.

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

        Called by ``CollisionPipeline.collide()`` AFTER the standard
        narrow phase.  Every pair is dispatched to :meth:`_launch`
        (Phase 5 unified path); the per-pair contact-buffer offset
        accumulates ``n_surface_contacts * pair.K_max`` per pair.

        After all pairs launch, CPU-reads the per-pair truncation
        counters and raises ``RuntimeWarning`` if any pad sphere
        overflowed its K_max budget -- the actionable signal to raise
        ``K_max`` in the caller's geometry-derived sizing.

        Args:
            model: The simulation Model.
            state: Current State (provides body_q).
            contacts: Contacts buffer to write to.
            contact_offset: Starting index in contacts buffer for CSLC slots.
        """
        # B3 — Snapshot the previous step's converged sphere_delta into
        # sphere_delta_prev_step ONCE before any pair runs.  cslc_copy_active
        # at the end of each pair's _launch overwrites sphere_delta in
        # place, so we must capture the previous step's state here, before
        # the pair loop, to give the lattice velocity-damping term in
        # jacobi_step a consistent reference for δ̇ ≈ (δ - δ_prev)/dt.
        # When c_over_dt == 0 the kernel ignores delta_prev_step but the
        # copy still runs (one wp.copy of n_spheres·vec3, ~µs, negligible
        # vs the Jacobi launches).
        if self.cslc_data.sphere_delta_prev_step is not None:
            wp.copy(self.cslc_data.sphere_delta_prev_step, self.cslc_data.sphere_delta)

        pair_offset = contact_offset
        truncation_pairs: list[tuple[int, int]] = []  # (pair_idx, K_max)
        for pair_idx, pair in enumerate(self.shape_pairs):
            if pair.target_positions_local is None:
                warnings.warn(
                    f"CSLC pair {pair_idx} ({pair.cslc_shape} ↔ "
                    f"{pair.other_shape}) has no target arrays "
                    "populated.  Callers must fill "
                    "target_positions_local / target_normals_local / "
                    "target_areas_local + target_count + K_max before "
                    "the first launch.  Pair skipped.",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            self._launch(
                model, state, contacts, pair_offset, pair, pair_idx)
            pair_offset += self.n_surface_contacts * pair.K_max
            truncation_pairs.append((pair_idx, pair.K_max))

        # CPU read of truncation counters.  Each counter is the number
        # of PAD SPHERES that overflowed K_max (not the number of
        # dropped pairs).  Non-zero means K_max was under-sized for
        # this scene's geometry -- the active-set inclusion radius
        # (r_lat + INCLUSION_FACTOR·eps from
        # cslc_main/theory/cslc_theory.py) is wider than K_max
        # accommodated, so a non-trivial slice of contact wrench was
        # silently dropped.
        for pair_idx, k_max in truncation_pairs:
            n_trunc = int(self.truncation_count_pairs[pair_idx].numpy()[0])
            if n_trunc > 0:
                warnings.warn(
                    f"CSLC pair {pair_idx} truncated K_max={k_max} on "
                    f"{n_trunc} pad sphere(s).  Up to "
                    f"{n_trunc} * (active_pairs_per_pad - K_max) contact "
                    f"pairs were silently dropped, with per-pair force in "
                    f"the 0.025-0.4 N range each at production parameters. "
                    f"Raise K_max for this pair (either in the geometry-"
                    f"derived sizing in cslc_main/grasp/objects.compute_k_max, "
                    f"or via direct CSLCShapePair.K_max override).",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _launch(
        self,
        model: "Model",
        state: "State",
        contacts: "Contacts",
        contact_offset: int,
        pair: CSLCShapePair,
        pair_idx: int,
    ) -> None:
        """Per-pair kernel pipeline for the CSLC v2 unified path.

        Launch order (contract §6.5 + §8):

            1. ``compute_cslc_penetration``    (argmax-overlap warm-start)
            2. ``compute_outward_normals_world`` (per-sphere world-frame n̂)
            3. ``lattice_solve_equilibrium``   (closed-form linear warm-start)
            4. ``jacobi_step`` × ``n_iter``    (damped-Jacobi refinement)
            5. ``cslc_copy_active``            (active-lattice selective copy)
            6. ``write_cslc_contacts``         (K_max contacts per pad sphere)

        The warm-start uses the argmax-overlap target sample per pad
        sphere.  The dominant single-contact equilibrium is absorbed by
        the warm-start; ``jacobi_step`` sweeps refine the multi-point
        correction.  At production ``n_iter = 40`` this matches the
        Phase 4 bridge harness convergence characterisation, so the
        same iteration budget applies here.
        """
        data = self.cslc_data

        # Each pair gets its own raw_penetration scratch.  The
        # externally-visible alias points at this pair so post-collide()
        # readers see the latest phi for this lattice.
        pen_buf = self.raw_penetration_pairs[pair_idx]
        self.raw_penetration = pen_buf

        eps = float(data.smoothing_eps)

        # Reset the per-pair truncation counter before this launch.
        # CPU read happens in :meth:`launch` after collide() returns.
        self.truncation_count_pairs[pair_idx].zero_()

        # ── Kernel 1: Argmax-overlap warm-start penetration ──
        wp.launch(
            kernel=compute_cslc_penetration,
            dim=data.n_spheres,
            inputs=[
                data.positions, data.radii, data.sphere_delta,
                data.sphere_shape, data.is_surface, data.outward_normals,
                state.body_q, model.shape_body, model.shape_transform,
                pair.cslc_shape,
                pair.other_body,
                pair.target_positions_local,
                pair.target_normals_local,
                pair.target_count,
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

        # ── Kernel 2: Lattice equilibrium solve (linear warm-start) ──
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

        # ── Damped Jacobi refinement (contract §6.5) ──
        for _ in range(self.n_iter):
            wp.launch(
                kernel=jacobi_step,
                dim=data.n_spheres,
                inputs=[
                    src, dst,
                    data.radii,
                    data.positions,
                    data.is_surface,
                    data.neighbor_start, data.neighbor_count,
                    data.neighbor_list,
                    data.ka, data.kl, data.kc, self.alpha,
                    data.sphere_shape, pair.cslc_shape,
                    data.outward_normals,
                    state.body_q, model.shape_body, model.shape_transform,
                    data.ka_tangent_ratio,
                    data.k_stick, data.mu_friction,
                    pair.target_positions_local,
                    pair.target_normals_local, pair.target_areas_local,
                    pair.target_count, pair.other_body,
                    # No external tangential load in production
                    # (apex_idx = -1 is the no-op sentinel).
                    int(-1), wp.vec3(0.0, 0.0, 0.0),
                    eps,
                    # B3 — lattice velocity damping inputs.
                    data.sphere_delta_prev_step,
                    data.c_over_dt,
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
        # populated by the v2 emission kernel; see the class docstring's
        # "Diagnostic-array gap" note.
        wp.launch(
            kernel=write_cslc_contacts,
            dim=data.n_spheres,
            inputs=[
                data.positions, data.radii, src,
                data.sphere_shape, data.is_surface, data.outward_normals,
                state.body_q, model.shape_body, model.shape_transform,
                pair.cslc_shape,
                pair.other_body, pair.other_shape,
                pair.target_positions_local,
                pair.target_normals_local, pair.target_areas_local,
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
