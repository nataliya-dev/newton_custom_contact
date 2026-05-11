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

from .cslc_data import CSLCData, CSLCPad, calibrate_kc, create_pad_for_box, create_pad_for_box_face
from .cslc_kernels import (
    compute_cslc_penetration_box,
    compute_cslc_penetration_sphere,
    cslc_copy_active,
    h3_pcwc_friction,
    jacobi_step,
    lattice_solve_equilibrium,
    write_cslc_contacts,
    write_cslc_contacts_box,
)

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model
    from ..sim.state import State


# Must match the value in geometry/types.py
_CSLC_FLAG = 1 << 5
# GeoType.SPHERE — Newton stores this as int
_GEOTYPE_SPHERE = 3   # GeoType.SPHERE
_GEOTYPE_BOX = 7      # GeoType.BOX


@dataclass
class CSLCShapePair:
    """A shape pair where one shape has the CSLC flag.

    Both sphere and box targets share this dataclass; which fields are
    meaningful depends on `other_geo_type`.  Sphere targets populate
    `other_local_pos` and `other_radius`; box targets populate
    `other_local_xform` (the shape's body-local 7-vector transform) and
    `other_half_extents`.
    """

    cslc_shape: int
    other_shape: int
    other_geo_type: int
    # Cached at construction — avoids GPU→CPU sync per step (Bug 3)
    cslc_body: int = 0   # body index for the CSLC pad shape
    other_body: int = 0  # body index for the target shape
    # Sphere-target fields:
    other_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    other_radius: float = 0.0
    # Box-target fields:
    other_local_xform: tuple[float, float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
    other_half_extents: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # H1: target-body material stiffness, cached at construction.  Used by
    # write_cslc_contacts / write_cslc_contacts_box to compose
    # per-constraint compliance as the harmonic mean of lattice and target
    # moduli (Masterjohn21 eq 23).  Rigid target → very large value →
    # series stiffness ≈ cslc_kc (current behavior).
    other_ke: float = 0.0

class CSLCHandler:
    """CSLC contact generation handler for Newton's collision pipeline.

    Constructed by _from_model() which inspects the Model for CSLC-flagged
    shapes, auto-generates lattice pads, calibrates kc, and builds CSLCData.

    During collide(), launch() runs the three-kernel pipeline:
        1. Penetration computation (per surface sphere vs target)
        2. Jacobi equilibrium solve (warm-started from previous timestep)
        3. Contact buffer writing (pre-allocated slots, fully differentiable)

    Attributes:
        contact_count: Number of contact slots CSLC will write (= n_surface_spheres).
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

        # H2: SAP-R damping emission (Castro 2022 eq 19).
        # Set to the simulation timestep (s) to enable H2; leave at 0.0 to
        # use the kd=0 path (current default, R = 1/ke_series either way).
        self.sim_dt: float = 0.0

        # H3: Predictive Contact Wrench Compensation (PCWC).
        # Set h3_target_body >= 0 to enable.  Requires state.body_f to be
        # pre-allocated (wp.zeros(model.body_count, dtype=wp.spatial_vector)).
        # h3_v_desired: desired CoM velocity of the target body [m/s] — set to
        # (0,0,0) for static holding; update each step for dynamic tasks.
        # h3_gravity: full 3-D gravity vector [m/s²], e.g. (0, 0, -9.81).
        self.h3_target_body: int = -1
        self.h3_mass: float = 0.0
        self.h3_mu: float = 0.0
        self.h3_dt: float = 0.0
        self.h3_v_desired: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.h3_gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

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
        # pads).  Per-pair blocks eliminate that race.
        total_slots = n_surface_contacts * max(n_pair_blocks, 1)
        self.dbg_pen_scale   = wp.full(total_slots, -1.0, dtype=wp.float32, device=self.device)
        self.dbg_solver_pen  = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
        self.dbg_effective_r = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
        self.dbg_d_proj      = wp.zeros(total_slots, dtype=wp.float32, device=self.device)
        self.dbg_radial      = wp.zeros(total_slots, dtype=wp.float32, device=self.device)


        n = cslc_data.n_spheres
        # One raw_penetration buffer per sphere pair.  Kernel 1 zeros all
        # non-active-pad spheres in whichever buffer it writes to, so if a
        # single shared buffer were used, the LAST pair's launch would leave
        # every other pad's phi looking like 0 — that's what was making the
        # diagnostic print "pad 2  kernel1 phi>0: 0" after the (4,5) launch.
        # With per-pair buffers each pad's last-computed phi is preserved and
        # the diagnostic can inspect both pads after collide() returns.
        self.raw_penetration_pairs = [
            wp.zeros(n, dtype=wp.float32, device=self.device)
            for _ in range(max(n_pair_blocks, 1))
        ]
        # Back-compat alias.  _launch_vs_sphere repoints this to the current
        # pair's buffer before running kernels, so the SUMMARY print below
        # still reads "the pair we just launched".
        self.raw_penetration = self.raw_penetration_pairs[0]
        self.contact_normal_scratch = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._jacobi_a = wp.zeros(n, dtype=wp.float32, device=self.device)
        self._jacobi_b = wp.zeros(n, dtype=wp.float32, device=self.device)


    @property
    def contact_count(self) -> int:
        return self.n_surface_contacts * self.n_pair_blocks


    def get_phi_for_cslc_shape(self, cslc_shape_idx: int) -> wp.array:
        """Return the raw_penetration buffer that was last written for a
        given CSLC pad (by shape index).

        Each pair launch uses its own scratch buffer; kernel 1 zeros every
        sphere that doesn't belong to the active pad.  So to read pad P's
        phi after collide() we need the buffer from the pair that had
        cslc_shape == P.  Returns None if the pad has no supported pair.
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
    def _from_model(cls, model: Model) -> CSLCHandler | None:
        """Create a CSLCHandler from a Model, or None if no CSLC shapes exist.

        Follows the same pattern as HydroelasticSDF._from_model().
        """
        shape_flags = model.shape_flags.numpy()
        shape_types = model.shape_type.numpy()

        cslc_shape_indices = [
            i for i in range(model.shape_count) if (shape_flags[i] & _CSLC_FLAG)
        ]
        if not cslc_shape_indices:
            return None

        # Find shape pairs involving CSLC shapes
        cslc_shape_set = set(cslc_shape_indices)
        shape_pairs: list[CSLCShapePair] = []


        if model.shape_contact_pairs is not None:
            all_pairs = model.shape_contact_pairs.numpy()
            for sa, sb in all_pairs:
                if sa in cslc_shape_set and sb not in cslc_shape_set:
                    shape_pairs.append(CSLCShapePair(
                        cslc_shape=int(sa), other_shape=int(sb),
                        other_geo_type=int(shape_types[sb]),
                    ))
                elif sb in cslc_shape_set and sa not in cslc_shape_set:
                    shape_pairs.append(CSLCShapePair(
                        cslc_shape=int(sb), other_shape=int(sa),
                        other_geo_type=int(shape_types[sa]),
                    ))
                # Both CSLC: not yet supported

        if not shape_pairs:
            return None

        # Read per-shape CSLC parameters
        cslc_spacing = model.shape_cslc_spacing.numpy()
        cslc_ka_arr = model.shape_cslc_ka.numpy()
        cslc_kl_arr = model.shape_cslc_kl.numpy()
        cslc_dc_arr = model.shape_cslc_dc.numpy()
        shape_ke = model.shape_material_ke.numpy()

        first_cslc = cslc_shape_indices[0]
        ka = float(cslc_ka_arr[first_cslc])
        kl = float(cslc_kl_arr[first_cslc])
        dc = float(cslc_dc_arr[first_cslc])

        # Auto-generate lattice pads
        pads: list[CSLCPad] = []
        shape_scale_np = model.shape_scale.numpy()
        for shape_idx in cslc_shape_indices:
            geo_type = int(shape_types[shape_idx])
            spacing = float(cslc_spacing[shape_idx])

            if geo_type == _GEOTYPE_BOX:
                hx = float(shape_scale_np[shape_idx][0])
                hy = float(shape_scale_np[shape_idx][1])
                hz = float(shape_scale_np[shape_idx][2])
                # Use a 2-D face lattice instead of the full volumetric pad.
                #
                # Rationale:
                # The volumetric pad has surface spheres on all 6 faces with
                # averaged normals at edges/corners (e.g. (1,0,1)/√2 for the
                # top-inner-edge sphere).  When the grasped object moves
                # off-centre relative to the pad, these tilted-normal spheres
                # generate normal forces with components along the pad's
                # tangent plane.  That violates the paper's flat-patch
                # assumption (§3.1) and drives a self-reinforcing launch
                # instability during lift tests.
                #
                # The face lattice places all spheres on a single face with
                # parallel normals (face_axis, face_sign) — exactly the paper's
                # formulation.  Only the designated face contributes to
                # contact.  Interior, top, bottom, and side-face spheres do
                # not exist in this mode, so their geometry cannot leak into
                # the contact force.
                #
                # TEMP: this hardcodes the inner face to local +x.  The caller
                # must orient each CSLC shape so its local +x points toward
                # the grasped object.  For multi-DOF grippers this needs a
                # per-shape config (e.g. model.shape_cslc_face_axis/sign);
                # that's follow-up work.  See lift_test.py for the right-pad
                # 180°-around-z shape xform that handles the two-finger case.
                pad = create_pad_for_box_face(
                    hx, hy, hz,
                    face_axis=0, face_sign=+1,
                    spacing=spacing, shape_index=shape_idx,
                )
                pads.append(pad)
            else:
                warnings.warn(
                    f"CSLC shape {shape_idx} has geometry type {geo_type} "
                    f"which is not yet supported. Only BOX is implemented.",
                    RuntimeWarning, stacklevel=2,
                )

        if not pads:
            return None

        # Calibrate kc from bulk ke (per-pad basis — see calibrate_kc
        # docstring).  contact_fraction=0.3 is a generic moderate prior;
        # geometry-specific tuning (e.g. cf=0.46 for the squeeze pad/sphere
        # geometry) is exposed via `recalibrate_cslc_kc_per_pad` in
        # squeeze_test.py.  See TODOs in cslc_v1/convo_april_19.md for
        # the iterative bootstrap that would replace this prior.
        ke_bulk = float(shape_ke[first_cslc])
        kc = calibrate_kc(ke_bulk, pads, ka=ka, contact_fraction=0.3, per_pad=True)



        # build_A_inv=True: precompute the dense inverse of the lattice
        # system matrix A = K + kc·I so the per-step solve is one matvec
        # (kernel `lattice_solve_equilibrium`) instead of ~20 iterative
        # Jacobi launches.  See the SPEED note in `_launch_vs_sphere`.
        # For pads with n > ~2000 surface spheres this should be replaced
        # with a sparse Cholesky factorisation (see `cslc_data.py:430`).
        cslc_data = CSLCData.from_pads(
            pads, ka=ka, kl=kl, kc=kc, dc=dc,
            build_A_inv=True,
            device=model.device,
        )


        # ── Filter CSLC pairs from narrow phase (Bug 2) ──
        # Prevents double-counting: narrow phase would also generate
        # contacts for these pairs if we don't exclude them.
        if not hasattr(model, 'shape_collision_filter_pairs'):
            model.shape_collision_filter_pairs = set()
        for pair in shape_pairs:
            a = min(pair.cslc_shape, pair.other_shape)
            b = max(pair.cslc_shape, pair.other_shape)
            model.shape_collision_filter_pairs.add((a, b))

        # ── Cache target info per pair (Bug 3) ──
        # One-time CPU read at construction instead of per-step .numpy()
        shape_body_np = model.shape_body.numpy()
        shape_transform_np = model.shape_transform.numpy()
        for pair in shape_pairs:
            pair.cslc_body = int(shape_body_np[pair.cslc_shape])
            if pair.other_geo_type == _GEOTYPE_SPHERE:
                pair.other_body = int(shape_body_np[pair.other_shape])
                xform = shape_transform_np[pair.other_shape]
                pair.other_local_pos = (
                    float(xform[0]), float(xform[1]), float(xform[2]),
                )
                pair.other_radius = float(shape_scale_np[pair.other_shape][0])
            elif pair.other_geo_type == _GEOTYPE_BOX:
                pair.other_body = int(shape_body_np[pair.other_shape])
                xform = shape_transform_np[pair.other_shape]
                pair.other_local_xform = (
                    float(xform[0]), float(xform[1]), float(xform[2]),
                    float(xform[3]), float(xform[4]),
                    float(xform[5]), float(xform[6]),
                )
                # shape_scale entries are stored as half-extents for boxes.
                pair.other_half_extents = (
                    float(shape_scale_np[pair.other_shape][0]),
                    float(shape_scale_np[pair.other_shape][1]),
                    float(shape_scale_np[pair.other_shape][2]),
                )
            # H1: cache the target's material stiffness for harmonic-mean
            # composition in the kernel.  Done once at construction; the
            # array index never changes during the simulation.
            pair.other_ke = float(shape_ke[pair.other_shape])

        # Build surface slot map: surface sphere i -> sequential slot index
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



        supported_geo_types = (_GEOTYPE_SPHERE, _GEOTYPE_BOX)
        supported_pairs = [
            p for p in shape_pairs if p.other_geo_type in supported_geo_types
        ]
        n_pair_blocks = len(supported_pairs)

        if n_pair_blocks == 0:
            return None

        # Read solver params from model (Bug 6 fix)
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

        # Walk all supported pairs (sphere + box) with a single index so
        # they share the contiguous CSLC contact-slot range and the
        # per-pair raw_penetration scratch buffers.
        pair_idx = 0
        for pair in self.shape_pairs:
            if pair.other_geo_type == _GEOTYPE_SPHERE:
                pair_contact_offset = contact_offset + pair_idx * self.n_surface_contacts
                self._launch_vs_sphere(
                    model, state, contacts, pair_contact_offset, pair, pair_idx)
                pair_idx += 1
            elif pair.other_geo_type == _GEOTYPE_BOX:
                pair_contact_offset = contact_offset + pair_idx * self.n_surface_contacts
                self._launch_vs_box(
                    model, state, contacts, pair_contact_offset, pair, pair_idx)
                pair_idx += 1
            else:
                warnings.warn(
                    f"CSLC vs geometry type {pair.other_geo_type} not yet implemented. "
                    f"Only CSLC vs SPHERE and CSLC vs BOX are supported.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self.h3_target_body >= 0 and state.body_f is not None:
            self._apply_h3_friction(model, state)


    def _apply_h3_friction(self, model: Model, state: State) -> None:
        """Launch h3_pcwc_friction once per pair whose other_body matches h3_target_body.

        Runs after all pair launches so sphere_delta and contact_normal_scratch
        reflect the final equilibrium from the last matching pair.
        Accumulates force + torque directly into state.body_f[h3_target_body].

        Mass is split evenly across matching pairs so the total applied force
        is correct regardless of how many pads contact the same target body.
        """
        data = self.cslc_data
        gravity = wp.vec3(*self.h3_gravity)
        v_desired = wp.vec3(*self.h3_v_desired)

        n_pairs = sum(
            1 for pair in self.shape_pairs if pair.other_body == self.h3_target_body
        )
        effective_mass = self.h3_mass / max(1, n_pairs)

        for pair in self.shape_pairs:
            if pair.other_body != self.h3_target_body:
                continue
            wp.launch(
                kernel=h3_pcwc_friction,
                dim=1,
                inputs=[
                    data.positions,
                    data.sphere_delta,
                    data.is_surface,
                    self.contact_normal_scratch,
                    data.n_spheres,
                    float(data.kc),
                    state.body_q,
                    model.shape_transform,
                    pair.cslc_shape,
                    pair.cslc_body,
                    state.body_qd,
                    state.body_f,
                    self.h3_target_body,
                    effective_mass,
                    gravity,
                    v_desired,
                    self.h3_mu,
                    self.h3_dt,
                ],
                device=self.device,
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
            """Three-kernel pipeline for CSLC shape vs sphere target."""
            data = self.cslc_data

            # Each pair gets its own raw_penetration scratch so the diagnostic
            # can see per-pad phi after all launches complete.  Update the
            # back-compat alias so the SUMMARY print at the bottom of this
            # method reads the pair we're currently launching.
            pen_buf = self.raw_penetration_pairs[pair_idx]
            self.raw_penetration = pen_buf

            # ── Use cached target info (Bug 3 fix) ──
            # No .numpy() calls — these were cached at construction
            target_body = pair.other_body
            target_local_pos = wp.vec3(
                pair.other_local_pos[0],
                pair.other_local_pos[1],
                pair.other_local_pos[2],
            )
            target_radius = pair.other_radius

            # Smoothing width for the differentiable surrogates in
            # cslc_kernels.py.  Recovers the original non-smooth behaviour
            # in the eps → 0 limit.
            eps = float(data.smoothing_eps)

            # ── Kernel 1: Raw penetration ──
            wp.launch(
                kernel=compute_cslc_penetration_sphere,
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

            # ── Kernel 2: Lattice equilibrium solve ──
            # SPEED: when data.A_inv is precomputed (build_A_inv=True in
            # CSLCData.from_pads — the default from _from_model), replace
            # the iterative Jacobi (n_iter ≈ 20 launches) with a single
            # `lattice_solve_equilibrium` matvec.  That matvec is the
            # closed-form δ = kc · A⁻¹ · φ and is tape-compatible, so
            # wp.Tape backward also becomes a single graph linear
            # operation rather than a ping-pong buffer chain.
            # Iterative Jacobi is retained only as a fallback for pads
            # large enough that the dense A⁻¹ doesn't fit in memory
            # (the TODO in cslc_data.py for sparse Cholesky).
            if data.A_inv is not None:
                wp.launch(
                    kernel=lattice_solve_equilibrium,
                    dim=data.n_spheres,
                    inputs=[data.A_inv, pen_buf, data.kc],
                    outputs=[self._jacobi_a],
                    device=self.device,
                )
                src = self._jacobi_a
            else:
                wp.copy(self._jacobi_a, data.sphere_delta)
                src, dst = self._jacobi_a, self._jacobi_b
                for _ in range(self.n_iter):
                    wp.launch(
                        kernel=jacobi_step,
                        dim=data.n_spheres,
                        inputs=[
                            src, dst, pen_buf, data.is_surface,
                            data.neighbor_start, data.neighbor_count, data.neighbor_list,
                            data.ka, data.kl, data.kc, self.alpha,
                            data.sphere_shape, pair.cslc_shape, eps,
                        ],
                        device=self.device,
                    )
                    src, dst = dst, src

            # Warm-start: write converged delta back, but ONLY for the
            # active pad's spheres.  In the dense-solve path, `src` carries
            # zeros for non-active spheres (because their phi was zeroed
            # in Kernel 1, and lattice_solve_equilibrium runs unfiltered);
            # an unconditional `wp.copy` would wipe the other pad's
            # warm-start every step.  The iterative jacobi path passes
            # non-active deltas through unchanged via `jacobi_step`'s pad
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
                    pair.other_ke,
                    data.dc,
                    self.sim_dt,
                    eps,
                    contacts.rigid_contact_stiffness,
                    contacts.rigid_contact_damping,
                    contacts.rigid_contact_friction,
                    self.debug_reason,
                    # Diagnostic outputs (added this iteration)
                    pair_idx * self.n_surface_contacts,  # diag_offset
                    self.dbg_pen_scale,
                    self.dbg_solver_pen,
                    self.dbg_effective_r,
                    self.dbg_d_proj,
                    self.dbg_radial,
                ],
                device=self.device,
            )

    def _launch_vs_box(
        self,
        model: Model,
        state: State,
        contacts: Contacts,
        contact_offset: int,
        pair: CSLCShapePair,
        pair_idx: int,
    ) -> None:
        """Three-kernel pipeline for CSLC shape vs BOX target.

        Same structure as `_launch_vs_sphere` but uses the box-target
        kernels.  The dense lattice solve is shared (it's geometry-
        agnostic) and runs against the same per-pair raw_penetration
        scratch buffer.
        """
        data = self.cslc_data

        pen_buf = self.raw_penetration_pairs[pair_idx]
        self.raw_penetration = pen_buf

        target_body = pair.other_body
        target_local_xform = wp.transform(
            wp.vec3(pair.other_local_xform[0],
                    pair.other_local_xform[1],
                    pair.other_local_xform[2]),
            wp.quat(pair.other_local_xform[3],
                    pair.other_local_xform[4],
                    pair.other_local_xform[5],
                    pair.other_local_xform[6]),
        )
        target_half_extents = wp.vec3(
            pair.other_half_extents[0],
            pair.other_half_extents[1],
            pair.other_half_extents[2],
        )

        eps = float(data.smoothing_eps)

        # ── Kernel 1: per-sphere penetration vs box surface ──
        wp.launch(
            kernel=compute_cslc_penetration_box,
            dim=data.n_spheres,
            inputs=[
                data.positions, data.radii, data.sphere_delta,
                data.sphere_shape, data.is_surface, data.outward_normals,
                state.body_q, model.shape_body, model.shape_transform,
                pair.cslc_shape,
                target_body, pair.other_shape,
                target_local_xform, target_half_extents,
                eps,
            ],
            outputs=[pen_buf, self.contact_normal_scratch],
            device=self.device,
        )

        # ── Kernel 2: lattice equilibrium (same dense solve) ──
        if data.A_inv is not None:
            wp.launch(
                kernel=lattice_solve_equilibrium,
                dim=data.n_spheres,
                inputs=[data.A_inv, pen_buf, data.kc],
                outputs=[self._jacobi_a],
                device=self.device,
            )
            src = self._jacobi_a
        else:
            wp.copy(self._jacobi_a, data.sphere_delta)
            src, dst = self._jacobi_a, self._jacobi_b
            for _ in range(self.n_iter):
                wp.launch(
                    kernel=jacobi_step,
                    dim=data.n_spheres,
                    inputs=[
                        src, dst, pen_buf, data.is_surface,
                        data.neighbor_start, data.neighbor_count, data.neighbor_list,
                        data.ka, data.kl, data.kc, self.alpha,
                        data.sphere_shape, pair.cslc_shape, eps,
                    ],
                    device=self.device,
                )
                src, dst = dst, src

        # Selectively merge the active pad's delta back into
        # CSLCData.sphere_delta (preserves other pads' warm-starts).
        wp.launch(
            kernel=cslc_copy_active,
            dim=data.n_spheres,
            inputs=[src, data.sphere_shape, pair.cslc_shape],
            outputs=[data.sphere_delta],
            device=self.device,
        )

        # ── Kernel 3: write Newton contacts vs box surface ──
        wp.launch(
            kernel=write_cslc_contacts_box,
            dim=data.n_spheres,
            inputs=[
                data.positions, data.radii, src,
                data.sphere_shape, data.is_surface, data.outward_normals,
                state.body_q, model.shape_body, model.shape_transform,
                pair.cslc_shape,
                target_body, pair.other_shape,
                target_local_xform, target_half_extents,
                contact_offset, self.surface_slot_map,
                pen_buf,
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
                data.kc,
                pair.other_ke,
                data.dc,
                self.sim_dt,
                eps,
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
                contacts.rigid_contact_friction,
                self.debug_reason,
                # Diagnostic outputs — same per-pair-per-slot layout the
                # sphere variant uses, so external readers can index both
                # sphere and box pairs uniformly via
                # `pair_idx * n_surface_contacts + slot`.
                pair_idx * self.n_surface_contacts,  # diag_offset
                self.dbg_pen_scale,
                self.dbg_solver_pen,
                self.dbg_effective_r,
                self.dbg_d_proj,
                self.dbg_radial,
            ],
            device=self.device,
        )