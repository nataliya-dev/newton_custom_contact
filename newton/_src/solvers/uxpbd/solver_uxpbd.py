# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""UXPBD: Unified eXtended Position-Based Dynamics solver.

Phase 1 scope: articulated rigid bodies with a MorphIt-generated kinematic
lattice that contacts analytical static shapes through the particle pipeline.
See ``docs/superpowers/specs/2026-05-13-uxpbd-design.md`` for the full design.
"""

import warp as wp

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from ..srxpbd.kernels import (
    enforce_momemntum_conservation_tiled,
    solve_shape_matching_batch_tiled,
)
from ..xpbd.kernels import (
    apply_body_deltas,
    apply_joint_forces,
    convert_joint_impulse_to_parent_f,
    copy_kinematic_body_state_kernel,
    solve_body_joints,
)
from .fluid import (
    apply_cohesion_forces,
    apply_xsph_viscosity,
    compute_fluid_density_and_lambda,
    compute_fluid_position_delta,
)
from .kernels import (
    apply_particle_deltas_position_only,
    apply_particle_deltas_uxpbd,
    compute_mass_scale,
    solve_particle_particle_contacts_uxpbd,
    solve_particle_shape_contacts_uxpbd,
)
from .kernels import update_lattice_world_positions as update_lattice_world_positions_kernel
from .shape_match import build_shape_match_cache


class SolverUXPBD(SolverBase):
    """Unified position-based dynamics solver.

    Phase 1 implements articulated rigid bodies with a kinematic lattice
    shell. Subsequent phases add free shape-matched rigid (PBD-R), soft bodies
    (springs / bending / FEM tet), and liquids (PBF density constraint).
    The class preserves architectural seams for v2 CSLC (compliant sphere
    lattice contact); see :meth:`compute_compliant_contact_response`.

    Args:
        model: The :class:`~newton.Model` to simulate.
        iterations: Number of main constraint loop iterations per step.
        stabilization_iterations: UPPFRTA stabilization pre-pass iterations.
        soft_contact_relaxation: Relaxation factor for lattice-shape contact
            corrections; lower values damp contact response. Defaults to 0.8.
        joint_linear_compliance: Linear-axis compliance for joint constraints
            (XPBD ``alpha`` for translation). Defaults to 0.0 (hard).
        joint_angular_compliance: Angular-axis compliance for joint constraints.
            Defaults to 0.0 (hard).
        joint_angular_relaxation: Relaxation factor applied to angular joint
            corrections per iteration. Defaults to 0.4 (XPBD default).
        joint_linear_relaxation: Relaxation factor applied to linear joint
            corrections per iteration. Defaults to 0.7 (XPBD default).
        shock_propagation_k: UPPFRTA mass-scaling factor for stack stability
            (default 0 = off; positive values reduce upper-particle effective
            mass via m* = m * exp(-k * h)).
        fluid_iterations: Number of PBF sub-iterations per main iteration for
            incompressibility enforcement on fluid particles (Macklin and Muller
            2013). Default 4.
        enable_cslc: Must be False in Phase 1. Reserved for v2.

    Raises:
        NotImplementedError: If ``enable_cslc=True``.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 4,
        stabilization_iterations: int = 1,
        soft_contact_relaxation: float = 0.8,
        joint_linear_compliance: float = 0.0,
        joint_angular_compliance: float = 0.0,
        joint_angular_relaxation: float = 0.4,
        joint_linear_relaxation: float = 0.7,
        shock_propagation_k: float = 0.0,
        fluid_iterations: int = 4,
        enable_cslc: bool = False,
    ):
        super().__init__(model=model)
        if enable_cslc:
            raise NotImplementedError(
                "CSLC compliant contact is reserved for UXPBD v2. "
                "See docs/superpowers/specs/2026-05-13-uxpbd-design.md section 5.5."
            )
        self.iterations = iterations
        self.stabilization_iterations = stabilization_iterations
        self.soft_contact_relaxation = soft_contact_relaxation
        self.joint_linear_compliance = joint_linear_compliance
        self.joint_angular_compliance = joint_angular_compliance
        self.joint_angular_relaxation = joint_angular_relaxation
        self.joint_linear_relaxation = joint_linear_relaxation
        self.shock_propagation_k = shock_propagation_k
        self.fluid_iterations = fluid_iterations

        # Cache the up-axis integer (0=X, 1=Y, 2=Z) for shock propagation.
        # Derived from the dominant gravity axis: gravity is along -up.
        import numpy as _np  # noqa: PLC0415

        grav_np = model.gravity.numpy()
        if grav_np.shape[0] > 0:
            abs_grav = _np.abs(grav_np[0])
            self._up_axis_int: int = int(_np.argmax(abs_grav))
        else:
            self._up_axis_int = 2  # Z default

        self._init_kinematic_state()

        cache = build_shape_match_cache(model)
        self._num_dynamic_groups: int = cache["num_dynamic_groups"]
        self._dynamic_group_ids = cache["dynamic_group_ids"]
        self._group_particle_start = cache["group_particle_start"]
        self._group_particle_count = cache["group_particle_count"]
        self._group_particles_flat = cache["group_particles_flat"]
        self.total_group_mass = cache["total_group_mass"]
        self._shape_match_block_dim: int = cache["block_dim"]

        # Per-step rest pose snapshot. Phase 2 shape matching needs the initial
        # particle positions (at solver-create time) to compare against. Phase 3+
        # may need to refresh this on notify_model_changed.
        if model.particle_count > 0:
            self.particle_q_rest = wp.clone(model.particle_q)
        else:
            self.particle_q_rest = wp.empty(0, dtype=wp.vec3, device=model.device)

        # ---- Pre-allocated scratch buffers (perf #2) ------------------------
        # Avoid per-iteration wp.zeros / wp.empty_like / wp.clone churn in
        # step(). Buffers are sized to model and reused across substeps;
        # accumulators are .zero_()-d before each use, ping-pong scratches
        # are swapped via _alt_particle_q / _alt_body_q helpers.
        N_p = model.particle_count
        N_b = model.body_count
        N_j = model.joint_count
        N_g = self._num_dynamic_groups
        dev = model.device

        # Particle position/velocity ping-pong scratches. Two buffers each so
        # input/output of srxpbd_apply_particle_deltas etc. never alias.
        if N_p > 0:
            self._particle_q_scratch_a = wp.empty(N_p, dtype=wp.vec3, device=dev)
            self._particle_q_scratch_b = wp.empty(N_p, dtype=wp.vec3, device=dev)
            self._particle_qd_scratch_a = wp.empty(N_p, dtype=wp.vec3, device=dev)
            self._particle_qd_scratch_b = wp.empty(N_p, dtype=wp.vec3, device=dev)
            # Reusable particle-delta accumulator (.zero_() between phases).
            # All four contact phases (stab, shape-contact, PP, SM-rigid) use
            # this same buffer because they consume it before the next phase
            # zeros it again.
            self._particle_deltas = wp.zeros(N_p, dtype=wp.vec3, device=dev)
            # XSPH viscosity writes into this buffer in place of an
            # empty_like(particle_qd) per main iteration.
            self._xsph_v = wp.empty(N_p, dtype=wp.vec3, device=dev)
            # Scaled inverse mass for shock propagation (was lazily allocated).
            self._scaled_inv_mass = wp.zeros(N_p, dtype=wp.float32, device=dev)

        # PBF scratch (fluid density / lambda / position-delta).
        if model.fluid_phase_count > 0 and N_p > 0:
            self._fluid_density = wp.zeros(N_p, dtype=wp.float32, device=dev)
            self._fluid_lambdas = wp.zeros(N_p, dtype=wp.float32, device=dev)
            self._fluid_deltas = wp.zeros(N_p, dtype=wp.vec3, device=dev)

        # Body-side scratches.
        if N_b > 0:
            self._body_deltas = wp.zeros(N_b, dtype=wp.spatial_vector, device=dev)
            self._body_contact_count = wp.zeros(N_b, dtype=wp.float32, device=dev)
            # Two ping-pong body_q/body_qd buffers so apply_body_deltas never
            # aliases input/output.
            self._body_q_scratch_a = wp.empty(N_b, dtype=wp.transform, device=dev)
            self._body_q_scratch_b = wp.empty(N_b, dtype=wp.transform, device=dev)
            self._body_qd_scratch_a = wp.empty(N_b, dtype=wp.spatial_vector, device=dev)
            self._body_qd_scratch_b = wp.empty(N_b, dtype=wp.spatial_vector, device=dev)
            # body_f scratch used when joint feedforward is added (was wp.clone).
            if N_j > 0:
                self._body_f_scratch = wp.empty(N_b, dtype=wp.spatial_vector, device=dev)
        else:
            # Dummy 1-element buffers for fluid-only / particle-only scenes:
            # the particle-shape contact kernel signature still takes a body
            # delta buffer, but its writes are gated by is_lattice / shape_link>=0.
            self._body_deltas_dummy = wp.zeros(1, dtype=wp.spatial_vector, device=dev)
            self._body_contact_count_dummy = wp.zeros(1, dtype=wp.float32, device=dev)

        # Joint impulse accumulator (always preallocated when joints exist;
        # serves both the body_parent_f-reporting path and the per-iter
        # impulse_out temporary).
        if N_j > 0:
            self._joint_impulse_scratch = wp.zeros(N_j, dtype=wp.spatial_vector, device=dev)

        # SM-rigid group momentum scratches.
        if N_g > 0:
            self._P_b4 = wp.zeros(N_g, dtype=wp.vec3, device=dev)
            self._L_b4 = wp.zeros(N_g, dtype=wp.vec3, device=dev)

        # Empty placeholders used when state has no body_q/body_qd (for the
        # stabilization pass on fluid-only scenes). Allocated once on device.
        self._empty_body_q = wp.zeros(0, dtype=wp.transform, device=dev)
        self._empty_body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=dev)

    # ------- ping-pong helpers (perf #2) -------------------------------
    def _alt_particle_q(self, state_out):
        """Return the OTHER preallocated particle_q scratch buffer."""
        if state_out.particle_q is self._particle_q_scratch_a:
            return self._particle_q_scratch_b
        return self._particle_q_scratch_a

    def _alt_particle_qd(self, state_out):
        if state_out.particle_qd is self._particle_qd_scratch_a:
            return self._particle_qd_scratch_b
        return self._particle_qd_scratch_a

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation by ``dt`` seconds. Phase 1: articulated rigid + lattice.

        Args:
            state_in: Input state at time t.
            state_out: Output state at time t + dt.
            control: Joint actuation. If None, use the model's zero control.
            contacts: Contact list from a prior ``model.collide`` call. May be None.
            dt: Time step [s].
        """
        model = self.model

        if control is None:
            control = model.control(clone_variables=False)

        # Adopt state_out into our pre-allocated ping-pong scratches (perf #2).
        # This ensures every per-iteration srxpbd_apply_particle_deltas /
        # apply_particle_deltas_position_only / apply_body_deltas can pick the
        # opposite scratch as its output without aliasing the input.
        # We pick whichever scratch is NOT currently held by state_in, so the
        # subsequent integrate_particles / integrate_bodies read-write pair is
        # race-free. On the very first step state_in points to user buffers,
        # in which case we arbitrarily seed state_out with scratch_a.
        if model.particle_count > 0:
            if state_in.particle_q is self._particle_q_scratch_a:
                state_out.particle_q = self._particle_q_scratch_b
            else:
                state_out.particle_q = self._particle_q_scratch_a
            if state_in.particle_qd is self._particle_qd_scratch_a:
                state_out.particle_qd = self._particle_qd_scratch_b
            else:
                state_out.particle_qd = self._particle_qd_scratch_a
        if model.body_count > 0:
            if state_in.body_q is self._body_q_scratch_a:
                state_out.body_q = self._body_q_scratch_b
            else:
                state_out.body_q = self._body_q_scratch_a
            if state_in.body_qd is self._body_qd_scratch_a:
                state_out.body_qd = self._body_qd_scratch_b
            else:
                state_out.body_qd = self._body_qd_scratch_a

        # Zero the joint-impulse accumulator at step start; solve_body_joints
        # atomic-adds into it across iterations, and convert_joint_impulse_to_parent_f
        # reads the sum at end-of-step when body_parent_f reporting is on.
        if model.joint_count > 0:
            self._joint_impulse_scratch.zero_()

        # Akinci cohesion: accumulate cohesion forces into state_in.particle_f
        # before the predict step so the integrator sees them as external forces.
        if model.fluid_phase_count > 0 and model.particle_count > 0:
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(
                    state_in.particle_q,
                    model.particle_max_radius * 4.0,
                )
            wp.launch(
                kernel=apply_cohesion_forces,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state_in.particle_q,
                    model.particle_mass,
                    model.particle_substrate,
                    model.particle_fluid_phase,
                    model.fluid_smoothing_radius,
                    model.fluid_cohesion,
                ],
                outputs=[state_in.particle_f],
                device=model.device,
            )

        # 1. Predict body positions: integrate_bodies with joint feedforward.
        if model.body_count:
            body_f_local = state_in.body_f
            if model.joint_count:
                # Copy state_in.body_f into pre-allocated scratch (was wp.clone,
                # perf #5). apply_joint_forces writes into body_f_local on top
                # of the copy, so we cannot use state_in.body_f directly.
                body_f_local = self._body_f_scratch
                wp.copy(body_f_local, state_in.body_f)
                wp.launch(
                    kernel=apply_joint_forces,
                    dim=model.joint_count,
                    inputs=[
                        state_in.body_q,
                        model.body_com,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_axis,
                        control.joint_f,
                    ],
                    outputs=[body_f_local],
                    device=model.device,
                )
            if body_f_local is state_in.body_f:
                self.integrate_bodies(model, state_in, state_out, dt)
            else:
                body_f_prev = state_in.body_f
                state_in.body_f = body_f_local
                self.integrate_bodies(model, state_in, state_out, dt)
                state_in.body_f = body_f_prev

        # Predict SM-rigid particle positions under gravity (lattice particles get
        # overwritten by update_lattice_world_positions below).
        if model.particle_count:
            self.integrate_particles(model, state_in, state_out, dt)

        # 2. Project body_q onto lattice particles.
        self.update_lattice_world_positions(state_out)

        # 3. v2 CSLC hook (no-op in v1).
        self.compute_compliant_contact_response(state_in, state_out, contacts, dt)

        # 4. Main iteration loop.
        # apply_body_deltas requires distinct input and output arrays (no aliasing).
        # We keep a scratch buffer pair and ping-pong with state_out so that
        # cur_(q|qd) -> nxt_(q|qd) always refer to different allocations.
        # joint_impulse accumulates the spatial impulse atomic_added by
        # solve_body_joints across iterations; convert_joint_impulse_to_parent_f
        # reads it at end-of-step when body_parent_f reporting is enabled.
        # We always zero the preallocated scratch at step start (above) so this
        # alias is safe even when reporting is off (kernel still writes, we just
        # don't read the result).
        if state_out.body_parent_f is not None and model.joint_count > 0:
            joint_impulse = self._joint_impulse_scratch
        else:
            joint_impulse = None

        if model.body_count:
            body_deltas = self._body_deltas
            # Per-body contact count (UPPFRTA §4.2 constraint averaging at the
            # body level). Populated by the contact kernels alongside the
            # atomic_add into body_deltas; consumed by apply_body_deltas via
            # its constraint_inv_weights parameter, which divides the
            # accumulated wrench by max(count, 0). Zeroed between phases.
            # See the lattice-launch incident write-up in
            # docs/superpowers/specs/2026-05-13-uxpbd-design.md §9.4.
            body_contact_count = self._body_contact_count
            # state_out.body_q now holds one of our scratches (set above). Pick
            # the OTHER scratch as the ping-pong partner so apply_body_deltas's
            # input and output never alias (perf #2, was wp.clone).
            if state_out.body_q is self._body_q_scratch_a:
                _alt_body_q = self._body_q_scratch_b
            else:
                _alt_body_q = self._body_q_scratch_a
            if state_out.body_qd is self._body_qd_scratch_a:
                _alt_body_qd = self._body_qd_scratch_b
            else:
                _alt_body_qd = self._body_qd_scratch_a
            _body_q = [state_out.body_q, _alt_body_q]
            _body_qd = [state_out.body_qd, _alt_body_qd]
            _cur = 0  # index into _body_q/_body_qd that holds the current state
        else:
            # Dummy 1-element buffer so the particle-shape contact kernel signature
            # is satisfied even with zero rigid bodies. The kernel only writes to
            # body_deltas when is_lattice (no lattices without bodies) or
            # shape_link >= 0 (ground is -1), so no writes actually hit this buffer.
            body_deltas = self._body_deltas_dummy
            body_contact_count = self._body_contact_count_dummy
            _body_q = None
            _body_qd = None
            _cur = 0

        def _apply_deltas_flip(constraint_inv_weights=None):
            nonlocal _cur
            # No-op when there are no bodies: _body_q/_body_qd weren't allocated
            # and the kernel launch would have dim=0 anyway.
            if _body_q is None:
                return
            _nxt = 1 - _cur
            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    _body_q[_cur],
                    _body_qd[_cur],
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    body_deltas,
                    constraint_inv_weights,
                    dt,
                ],
                outputs=[_body_q[_nxt], _body_qd[_nxt]],
                device=model.device,
            )
            _cur = _nxt
            # Keep state_out.body_q/qd pointing at the authoritative data so that
            # every downstream kernel (lattice projection, contact solve, joints)
            # reads the most recent body state without extra indirection.
            state_out.body_q = _body_q[_cur]
            state_out.body_qd = _body_qd[_cur]

        # Compute scaled inverse mass for contact kernels (UPPFRTA §5.2).
        # _scaled_inv_mass is preallocated in __init__ (perf #2).
        if self.shock_propagation_k > 0.0 and model.particle_count > 0:
            wp.launch(
                kernel=compute_mass_scale,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    model.particle_mass,
                    self._up_axis_int,
                    self.shock_propagation_k,
                ],
                outputs=[self._scaled_inv_mass],
                device=model.device,
            )
            inv_mass_for_contact = self._scaled_inv_mass
        else:
            inv_mass_for_contact = model.particle_inv_mass

        # ──── 3.5 UPPFRTA §4.4 stabilization sub-loop ────────────────────
        # Resolve initial contact penetration BEFORE the main loop, applying
        # only POSITION corrections (no velocity update) so the substep's
        # final v = (q_final - q_init)/dt naturally cancels the stabilization
        # correction. Without this pass, a fluid block landing on the ground
        # injects a velocity impulse equal to penetration_depth/dt at the
        # first contact iteration; the PBF density solver then opposes the
        # contact correction and the contact-PBF pair pumps energy into the
        # stack until particles launch. Per UPPFRTA Algorithm 1 lines 10-15.
        #
        # Currently restricted to PARTICLE-only stabilization: body wrench
        # accumulated by the contact kernel is intentionally discarded here.
        # For pure-particle scenes (fluid block on ground, SM-rigid stacks)
        # this is sufficient. Lattice-bound articulated rigid bodies still
        # rely on the body-level contact-count averaging from design spec
        # §5.7 to avoid the "lattice launch" failure mode; their main-loop
        # corrections converge regardless of stabilization. A future
        # apply_body_deltas_position_only kernel would extend §4.4 to the
        # lattice path and is left as a follow-up.
        if (self.stabilization_iterations > 0
                and contacts is not None
                and model.particle_count > 0):
            _body_q_stab = (
                state_out.body_q
                if state_out.body_q is not None
                else self._empty_body_q
            )
            _body_qd_stab = (
                state_out.body_qd
                if state_out.body_qd is not None
                else self._empty_body_qd
            )
            for _stab_iter in range(self.stabilization_iterations):
                body_deltas.zero_()
                body_contact_count.zero_()
                # Reuse the shared particle-deltas accumulator (perf #2).
                self._particle_deltas.zero_()
                particle_deltas_stab = self._particle_deltas
                wp.launch(
                    kernel=solve_particle_shape_contacts_uxpbd,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        state_out.particle_q,
                        state_out.particle_qd,
                        inv_mass_for_contact,
                        model.particle_radius,
                        model.particle_flags,
                        model.particle_substrate,
                        model.particle_to_lattice,
                        model.lattice_link,
                        _body_q_stab,
                        _body_qd_stab,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                        model.shape_body,
                        model.shape_material_mu,
                        model.soft_contact_mu,
                        model.particle_adhesion,
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_max,
                        dt,
                        self.soft_contact_relaxation,
                    ],
                    outputs=[body_deltas, body_contact_count, particle_deltas_stab],
                    device=model.device,
                )
                # Apply position-only (no velocity update) per §4.4.
                new_q_stab = self._alt_particle_q(state_out)
                wp.launch(
                    kernel=apply_particle_deltas_position_only,
                    dim=model.particle_count,
                    inputs=[
                        state_out.particle_q,
                        model.particle_flags,
                        model.particle_mass,
                        particle_deltas_stab,
                    ],
                    outputs=[new_q_stab],
                    device=model.device,
                )
                state_out.particle_q = new_q_stab

                # §4.4 extension: SM-rigid rigidity restoration (position-only).
                # The contact pass above shifts only the penetrating particles,
                # which leaves any SM-rigid cluster non-rigidly deformed. The
                # main loop's apply_particle_deltas_uxpbd uses the PBD-R
                # v_new = vp + d/dt update, so leaving the deformation for the
                # main loop to undo injects velocity (sub-mm d over dt~6e-4 s
                # is m/s-scale v; see srxpbd.pdf §III-B for the velocity-update
                # rationale, uppfrta_preprint.pdf §4.4 for stabilization).
                # Resolving rigidity here -- still position-only, so particle_qd
                # is untouched and §4.4's no-velocity-injection contract holds
                # -- pre-empts that velocity injection without breaking the
                # fluid path (fluid particles aren't in any SM group, so the
                # SM kernel writes delta=0 for them and apply_particle_deltas_
                # position_only passes them through unchanged).
                # Required to make SM-rigid + fluid scenes (Macklin '14 Fig. 1
                # bunnies-in-water) stable in PBD-R-updated UXPBD.
                if (self._num_dynamic_groups > 0
                        and model.particle_count > 0):
                    self._particle_deltas.zero_()
                    self._P_b4.zero_()
                    self._L_b4.zero_()
                    bd_sm = self._shape_match_block_dim
                    wp.launch(
                        kernel=solve_shape_matching_batch_tiled,
                        dim=(self._num_dynamic_groups, bd_sm),
                        inputs=[
                            state_out.particle_q,
                            self.particle_q_rest,
                            state_out.particle_qd,
                            self.total_group_mass,
                            model.particle_mass,
                            self._group_particle_start,
                            self._group_particle_count,
                            self._group_particles_flat,
                        ],
                        outputs=[
                            self._particle_deltas,
                            self._P_b4,
                            self._L_b4,
                        ],
                        block_dim=bd_sm,
                        device=model.device,
                    )
                    new_q_stab_sm = self._alt_particle_q(state_out)
                    wp.launch(
                        kernel=apply_particle_deltas_position_only,
                        dim=model.particle_count,
                        inputs=[
                            state_out.particle_q,
                            model.particle_flags,
                            model.particle_mass,
                            self._particle_deltas,
                        ],
                        outputs=[new_q_stab_sm],
                        device=model.device,
                    )
                    state_out.particle_q = new_q_stab_sm

                # Re-sync lattice particle positions from the (unchanged)
                # body_q so the next stabilization iter sees consistent state.
                self.update_lattice_world_positions(state_out)

        for _ in range(self.iterations):
            if body_deltas is not None:
                body_deltas.zero_()
                body_contact_count.zero_()

            # Particle-shape contacts: dispatches on particle_substrate.
            # Lattice particles (substrate=0) route deltas into body_deltas.
            # SM-rigid particles (substrate=1) and fluid particles (substrate=3)
            # route into particle_deltas_contact via the ELSE branch.
            # Runs even when body_count==0 (e.g. fluid-only + static ground plane);
            # body_deltas / body_contact_count are pre-allocated as 1-element dummy
            # buffers in the body_count==0 branch above, and the kernel's atomic
            # writes to them are gated by is_lattice (needs bodies) or
            # shape_link >= 0 (needs bodies), so no real writes hit the dummy.
            if contacts is not None and model.particle_count > 0:
                _body_q_ps = (
                    state_out.body_q
                    if state_out.body_q is not None
                    else self._empty_body_q
                )
                _body_qd_ps = (
                    state_out.body_qd
                    if state_out.body_qd is not None
                    else self._empty_body_qd
                )
                # Reuse the shared particle-deltas accumulator (perf #2).
                self._particle_deltas.zero_()
                particle_deltas_contact = self._particle_deltas
                wp.launch(
                    kernel=solve_particle_shape_contacts_uxpbd,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        state_out.particle_q,
                        state_out.particle_qd,
                        inv_mass_for_contact,
                        model.particle_radius,
                        model.particle_flags,
                        model.particle_substrate,
                        model.particle_to_lattice,
                        model.lattice_link,
                        _body_q_ps,
                        _body_qd_ps,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                        model.shape_body,
                        model.shape_material_mu,
                        model.soft_contact_mu,
                        model.particle_adhesion,
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        contacts.soft_contact_max,
                        dt,
                        self.soft_contact_relaxation,
                    ],
                    outputs=[body_deltas, body_contact_count, particle_deltas_contact],
                    device=model.device,
                )

                # Per-body contact-count averaging (UPPFRTA §4.2 promoted to
                # the body level, design spec §5.7). Without dividing by the
                # per-body contact count, N synchronized lattice-particle
                # penetrations against one shape compound into N× the body
                # wrench and the body launches off the ground (lattice drop
                # regression). _apply_deltas_flip itself early-returns when
                # body_count==0, so this is safe for fluid-only scenes.
                _apply_deltas_flip(constraint_inv_weights=body_contact_count)

                # Re-sync lattice after body update so next iter sees consistent state.
                self.update_lattice_world_positions(state_out)

                # Apply particle-side deltas from shape contact (SM-rigid path).
                # Uses apply_particle_deltas_uxpbd which passes through v for
                # mass-0 lattice particles (instead of zeroing it like the
                # SRXPBD variant); the prior projection at line 629 already
                # set lattice x/qd to body-consistent values, and preserving
                # them through the apply eliminates the need for a redundant
                # projection here.
                if model.particle_count > 0:
                    new_q = self._alt_particle_q(state_out)
                    new_qd = self._alt_particle_qd(state_out)
                    wp.launch(
                        kernel=apply_particle_deltas_uxpbd,
                        dim=model.particle_count,
                        inputs=[
                            self.particle_q_rest,
                            state_out.particle_q,
                            state_out.particle_qd,
                            model.particle_flags,
                            model.particle_mass,
                            particle_deltas_contact,
                            dt,
                            model.particle_max_velocity,
                        ],
                        outputs=[new_q, new_qd],
                        device=model.device,
                    )
                    state_out.particle_q = new_q
                    state_out.particle_qd = new_qd

            # Cross-substrate particle-particle contact pass.
            if model.particle_count > 1 and model.particle_grid is not None and body_deltas is not None:
                # Reset body_deltas before the PP-contact pass writes into it. The shape-
                # contact pass above already applied its body deltas via _apply_deltas_flip,
                # so we must NOT include them again.
                body_deltas.zero_()
                body_contact_count.zero_()
                search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
                with wp.ScopedDevice(model.device):
                    model.particle_grid.build(state_out.particle_q, radius=search_radius)
                # Reuse the shared particle-deltas accumulator (perf #2).
                self._particle_deltas.zero_()
                pp_particle_deltas = self._particle_deltas
                wp.launch(
                    kernel=solve_particle_particle_contacts_uxpbd,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        state_out.particle_q,
                        state_out.particle_qd,
                        inv_mass_for_contact,
                        model.particle_radius,
                        model.particle_flags,
                        model.particle_group,
                        model.particle_substrate,
                        model.particle_to_lattice,
                        model.lattice_link,
                        state_out.body_q,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                        model.particle_mu,
                        model.particle_cohesion,
                        model.particle_max_radius,
                        dt,
                        self.soft_contact_relaxation,
                    ],
                    outputs=[pp_particle_deltas, body_deltas, body_contact_count],
                    device=model.device,
                )
                _apply_deltas_flip(constraint_inv_weights=body_contact_count)
                # PP-contact body apply moved bodies, so lattice particles
                # (whose x/qd are body-derived) are now stale. The post-apply
                # projection at the end of this block re-syncs them. Using
                # apply_particle_deltas_uxpbd avoids ALSO needing to restore
                # lattice qd that the SRXPBD variant would have zeroed.
                new_q = self._alt_particle_q(state_out)
                new_qd = self._alt_particle_qd(state_out)
                wp.launch(
                    kernel=apply_particle_deltas_uxpbd,
                    dim=model.particle_count,
                    inputs=[
                        self.particle_q_rest,
                        state_out.particle_q,
                        state_out.particle_qd,
                        model.particle_flags,
                        model.particle_mass,
                        pp_particle_deltas,
                        dt,
                        model.particle_max_velocity,
                    ],
                    outputs=[new_q, new_qd],
                    device=model.device,
                )
                state_out.particle_q = new_q
                state_out.particle_qd = new_qd
                self.update_lattice_world_positions(state_out)

            # Position-Based Fluids pipeline (Macklin and Muller 2013).
            # Runs fluid_iterations sub-iterations per main iteration.
            if model.fluid_phase_count > 0 and model.particle_count > 0:
                # PBF scratches are preallocated in __init__ (perf #2).
                fluid_density = self._fluid_density
                fluid_lambdas = self._fluid_lambdas
                fluid_deltas = self._fluid_deltas
                epsilon = wp.float32(100.0)
                k_corr = wp.float32(0.1)
                dq_factor = wp.float32(0.3)
                n_corr = wp.float32(4.0)

                for _pbf_iter in range(self.fluid_iterations):
                    # Halve the hash-grid rebuilds (perf #4): particles only
                    # drift by ~r·dt between consecutive sub-iterations, so
                    # rebuilding every other iteration with the same query
                    # radius is safe. Always rebuild on the first iter so we
                    # have a fresh grid on entry.
                    if _pbf_iter % 2 == 0:
                        with wp.ScopedDevice(model.device):
                            model.particle_grid.build(
                                state_out.particle_q,
                                model.particle_max_radius * 4.0,
                            )
                    fluid_density.zero_()
                    fluid_lambdas.zero_()
                    fluid_deltas.zero_()

                    # Fused density + lambda kernel (perf #B). Computes both
                    # in a single neighbor traversal. lambda only needs rho_i
                    # (not rho_j), so density can be computed inline; saves
                    # one grid query per fluid particle per sub-iteration.
                    wp.launch(
                        kernel=compute_fluid_density_and_lambda,
                        dim=model.particle_count,
                        inputs=[
                            model.particle_grid.id,
                            state_out.particle_q,
                            model.particle_mass,
                            model.particle_substrate,
                            model.particle_fluid_phase,
                            model.fluid_rest_density,
                            model.fluid_smoothing_radius,
                            model.fluid_solid_coupling_s,
                            epsilon,
                        ],
                        outputs=[fluid_density, fluid_lambdas],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=compute_fluid_position_delta,
                        dim=model.particle_count,
                        inputs=[
                            model.particle_grid.id,
                            state_out.particle_q,
                            model.particle_mass,
                            model.particle_substrate,
                            model.particle_fluid_phase,
                            model.fluid_rest_density,
                            model.fluid_smoothing_radius,
                            model.fluid_solid_coupling_s,
                            fluid_lambdas,
                            k_corr,
                            dq_factor,
                            n_corr,
                        ],
                        outputs=[fluid_deltas],
                        device=model.device,
                    )

                    new_q = self._alt_particle_q(state_out)
                    new_qd = self._alt_particle_qd(state_out)
                    wp.launch(
                        kernel=apply_particle_deltas_uxpbd,
                        dim=model.particle_count,
                        inputs=[
                            self.particle_q_rest,
                            state_out.particle_q,
                            state_out.particle_qd,
                            model.particle_flags,
                            model.particle_mass,
                            fluid_deltas,
                            dt,
                            model.particle_max_velocity,
                        ],
                        outputs=[new_q, new_qd],
                        device=model.device,
                    )
                    state_out.particle_q = new_q
                    state_out.particle_qd = new_qd
                    # NOTE (perf #1): the lattice projection that previously
                    # ran here every PBF sub-iteration is moved OUT of the
                    # loop. body_q is unchanged within the PBF loop, so the
                    # lattice particle positions derived from body_q are also
                    # unchanged; re-projecting them was pure waste (~7%/frame
                    # on combo).
                # End of PBF sub-iteration loop.
                # NOTE (perf 9-prime): trailing lattice projection removed.
                # The PBF loop does not move bodies (body_q is unchanged), and
                # apply_particle_deltas_uxpbd preserves lattice qd (instead of
                # zeroing it like the SRXPBD variant). Lattice particle_q /
                # particle_qd therefore remain valid through the entire PBF
                # loop without an explicit re-projection.

                # XSPH viscosity once per main iteration. Kept inside the loop
                # (not hoisted out of step()) because XSPH composes
                # non-linearly: applying it N times with coefficient c is NOT
                # equivalent to one pass with N*c when velocities are large
                # (e.g. fluid-solid impact). Moving it out caused volumes to
                # bounce on fluid impact and inflated x_extent on fluid_drop.
                # Kept here for behavior parity with the paper's PBF Algorithm
                # 1, which applies XSPH after each density-solve pass.
                xsph_v = self._xsph_v
                wp.launch(
                    kernel=apply_xsph_viscosity,
                    dim=model.particle_count,
                    inputs=[
                        model.particle_grid.id,
                        state_out.particle_q,
                        state_out.particle_qd,
                        model.particle_mass,
                        model.particle_substrate,
                        model.particle_fluid_phase,
                        model.fluid_smoothing_radius,
                        model.fluid_viscosity,
                        self._fluid_density,
                    ],
                    outputs=[xsph_v],
                    device=model.device,
                )
                state_out.particle_qd = xsph_v

            # Joints
            if model.joint_count and body_deltas is not None:
                body_deltas.zero_()
                # impulse_out always points to the preallocated joint scratch
                # (perf #2). When body_parent_f reporting is on it is the same
                # buffer as joint_impulse and accumulates across iterations;
                # otherwise it is a per-iter throwaway target (we never read
                # the result).
                impulse_out = self._joint_impulse_scratch
                wp.launch(
                    kernel=solve_body_joints,
                    dim=model.joint_count,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        model.body_com,
                        self.body_inv_mass_effective,
                        self.body_inv_inertia_effective,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_axis,
                        control.joint_target_pos,
                        control.joint_target_vel,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        self.joint_linear_compliance,
                        self.joint_angular_compliance,
                        self.joint_angular_relaxation,
                        self.joint_linear_relaxation,
                        dt,
                    ],
                    outputs=[body_deltas, impulse_out],
                    device=model.device,
                )
                _apply_deltas_flip()
                self.update_lattice_world_positions(state_out)

            # SM-rigid groups: shape matching + momentum-conservation post-pass.
            if self._num_dynamic_groups > 0 and model.particle_count > 0:
                # Reuse the shared particle-deltas accumulator (perf #2).
                self._particle_deltas.zero_()
                particle_deltas = self._particle_deltas
                self._P_b4.zero_()
                self._L_b4.zero_()
                P_b4 = self._P_b4
                L_b4 = self._L_b4
                bd = self._shape_match_block_dim

                wp.launch(
                    kernel=solve_shape_matching_batch_tiled,
                    dim=(self._num_dynamic_groups, bd),
                    inputs=[
                        state_out.particle_q,
                        self.particle_q_rest,
                        state_out.particle_qd,
                        self.total_group_mass,
                        model.particle_mass,
                        self._group_particle_start,
                        self._group_particle_count,
                        self._group_particles_flat,
                    ],
                    outputs=[particle_deltas, P_b4, L_b4],
                    block_dim=bd,
                    device=model.device,
                )

                new_q = self._alt_particle_q(state_out)
                new_qd = self._alt_particle_qd(state_out)
                wp.launch(
                    kernel=apply_particle_deltas_uxpbd,
                    dim=model.particle_count,
                    inputs=[
                        self.particle_q_rest,
                        state_out.particle_q,
                        state_out.particle_qd,
                        model.particle_flags,
                        model.particle_mass,
                        particle_deltas,
                        dt,
                        model.particle_max_velocity,
                    ],
                    outputs=[new_q, new_qd],
                    device=model.device,
                )
                state_out.particle_q = new_q
                state_out.particle_qd = new_qd

                # Momentum conservation post-pass. enforce_momemntum_conservation_tiled
                # writes x_out/v_out only for particles in dynamic SM-rigid groups, so
                # pre-seed final_q/final_qd with the current state to avoid leaving
                # non-group particles (lattice, static, ungrouped) with uninitialized
                # memory. Without this, the next iteration's contact pass reads
                # garbage lattice positions and routes junk wrenches into body_deltas.
                # Use the OTHER ping-pong scratch and wp.copy the seed (perf #5,
                # was wp.clone — saved 240 clones/frame on combo).
                final_q = self._alt_particle_q(state_out)
                final_qd = self._alt_particle_qd(state_out)
                wp.copy(final_q, state_out.particle_q)
                wp.copy(final_qd, state_out.particle_qd)
                wp.launch(
                    kernel=enforce_momemntum_conservation_tiled,
                    dim=(self._num_dynamic_groups, bd),
                    inputs=[
                        state_out.particle_q,
                        state_out.particle_qd,
                        self.total_group_mass,
                        model.particle_mass,
                        P_b4,
                        L_b4,
                        dt,
                        self._group_particle_start,
                        self._group_particle_count,
                        self._group_particles_flat,
                    ],
                    outputs=[final_q, final_qd],
                    block_dim=bd,
                    device=model.device,
                )
                state_out.particle_q = final_q
                state_out.particle_qd = final_qd

                # NOTE (perf 9-prime): trailing lattice projection removed.
                # SM-rigid shape matching does not move bodies (body_q is
                # unchanged), and apply_particle_deltas_uxpbd preserves
                # lattice qd through the apply. enforce_momemntum_conservation
                # only writes to group (SM-rigid) particles, and the wp.copy
                # seed carries correct lattice values from state_out into
                # final_q/final_qd untouched.

        # 5. Populate state_out.body_parent_f from joint_impulse (XPBD convention).
        if state_out.body_parent_f is not None:
            state_out.body_parent_f.zero_()
            if joint_impulse is not None:
                wp.launch(
                    kernel=convert_joint_impulse_to_parent_f,
                    dim=model.joint_count,
                    inputs=[
                        joint_impulse,
                        model.joint_enabled,
                        model.joint_type,
                        model.joint_child,
                        dt,
                    ],
                    outputs=[state_out.body_parent_f],
                    device=model.device,
                )

        # 6. Copy kinematic body state forward.
        if model.body_count:
            wp.launch(
                kernel=copy_kinematic_body_state_kernel,
                dim=model.body_count,
                inputs=[model.body_flags, state_in.body_q, state_in.body_qd],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )

    def update_lattice_world_positions(self, state: State) -> None:
        """Project ``body_q``/``body_qd`` onto every lattice particle.

        Updates ``state.particle_q``, ``state.particle_qd``, and
        ``model.particle_radius`` in place for all lattice particles. Non-lattice
        particles are left untouched.

        Args:
            state: The :class:`~newton.State` whose body_q drives the projection
                and whose particle_q is written.
        """
        model = self.model
        if model.lattice_sphere_count == 0:
            return
        wp.launch(
            kernel=update_lattice_world_positions_kernel,
            dim=model.lattice_sphere_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.lattice_link,
                model.lattice_p_rest,
                model.lattice_delta,
                model.lattice_r,
                model.lattice_particle_index,
            ],
            outputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
            ],
            device=model.device,
        )

    def compute_compliant_contact_response(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """v2 CSLC hook. No-op in Phase 1.

        v2 will solve the lattice compression vector :math:`\\delta` from the
        quasistatic equilibrium :math:`K\\delta = k_c (\\phi^{rest} - \\delta)_+`
        and write it into ``model.lattice_delta``, which the
        ``update_lattice_world_positions`` kernel then propagates into the
        per-particle effective radius.
        """
        return
