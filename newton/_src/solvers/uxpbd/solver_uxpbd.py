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
    apply_particle_deltas as srxpbd_apply_particle_deltas,
)
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
from .kernels import solve_lattice_shape_contacts
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

        # 1. Predict body positions: integrate_bodies with joint feedforward.
        if model.body_count:
            body_f_local = state_in.body_f
            if model.joint_count:
                body_f_local = wp.clone(state_in.body_f)
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
        # Allocate joint impulse accumulator only if the user requested body_parent_f reporting.
        if state_out.body_parent_f is not None and model.joint_count > 0:
            joint_impulse = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)
        else:
            joint_impulse = None

        if model.body_count:
            body_deltas = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
            _body_q = [state_out.body_q, wp.clone(state_out.body_q)]
            _body_qd = [state_out.body_qd, wp.clone(state_out.body_qd)]
            _cur = 0  # index into _body_q/_body_qd that holds the current state
        else:
            body_deltas = None
            _body_q = None
            _body_qd = None
            _cur = 0

        def _apply_deltas_flip():
            nonlocal _cur
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
                    None,
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

        for _ in range(self.iterations):
            if body_deltas is not None:
                body_deltas.zero_()

            # Lattice-shape contacts (Phase 1's only contact path).
            if contacts is not None and model.lattice_sphere_count and body_deltas is not None:
                wp.launch(
                    kernel=solve_lattice_shape_contacts,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        state_out.particle_q,
                        state_out.particle_qd,
                        model.particle_radius,
                        model.particle_flags,
                        model.lattice_particle_index,
                        model.lattice_link,
                        state_out.body_q,
                        state_out.body_qd,
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
                        model.particle_to_lattice,
                        dt,
                        self.soft_contact_relaxation,
                    ],
                    outputs=[body_deltas],
                    device=model.device,
                )

                _apply_deltas_flip()

                # Re-sync lattice after body update so next iter sees consistent state.
                self.update_lattice_world_positions(state_out)

            # Joints
            if model.joint_count and body_deltas is not None:
                body_deltas.zero_()
                if joint_impulse is not None:
                    impulse_out = joint_impulse
                else:
                    # Need a temp buffer because solve_body_joints requires an output array.
                    impulse_out = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)
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
                particle_deltas = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
                P_b4 = wp.zeros(self._num_dynamic_groups, dtype=wp.vec3, device=model.device)
                L_b4 = wp.zeros(self._num_dynamic_groups, dtype=wp.vec3, device=model.device)
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

                new_q = wp.empty_like(state_out.particle_q)
                new_qd = wp.empty_like(state_out.particle_qd)
                wp.launch(
                    kernel=srxpbd_apply_particle_deltas,
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

                # Momentum conservation post-pass.
                final_q = wp.empty_like(state_out.particle_q)
                final_qd = wp.empty_like(state_out.particle_qd)
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
