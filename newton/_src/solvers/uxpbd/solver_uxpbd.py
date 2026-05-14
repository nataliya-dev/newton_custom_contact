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
from .kernels import update_lattice_world_positions as update_lattice_world_positions_kernel


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
        enable_cslc: Must be False in Phase 1. Reserved for v2.

    Raises:
        NotImplementedError: If ``enable_cslc=True``.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 4,
        stabilization_iterations: int = 1,
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
        self._init_kinematic_state()

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation by ``dt`` seconds.

        Phase 1 stub: implementation is added in Task 6 of the UXPBD Phase 1 plan.

        Raises:
            NotImplementedError: Always, until Task 6 wires up the iteration loop.
        """
        raise NotImplementedError("SolverUXPBD.step() is implemented in Task 6.")

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
