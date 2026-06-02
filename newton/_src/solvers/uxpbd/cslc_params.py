# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the UXPBD Compliant Sphere Lattice Contact (CSLC) hook.

Passing a :class:`CSLCParams` instance into
:class:`~newton.solvers.SolverUXPBD` enables the compliant lattice path
inside :meth:`SolverUXPBD.compute_compliant_contact_response`. Passing
``None`` (the default) leaves the solver on the rigid-lattice Phase 1
path. New parameters get added here, not to the solver constructor.

v1 of the hook (active now) implements a per-sphere normal-axis anchor
spring in series with a per-sphere bulk-contact spring; the load-bearing
output is the per-sphere compression :math:`\\delta_{n,i}` written into
``model.lattice_delta``. v2 will replace the closed-form per-sphere
solve with the contract_v2 damped-Jacobi solve over the lattice
Laplacian and a per-pair half-space contact law; the v2 placeholders
below are carried on the dataclass so callers can configure them once
and the solver picks them up when the jacobi path lands.

Reference: ``cslc_main/theory/contract_v2.md`` (the per-pair force law,
sign conventions, and the damped-Jacobi solver this hook converges to).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CSLCParams:
    """Compliant Sphere Lattice Contact configuration for SolverUXPBD.

    Per-sphere overrides are read from the ``model.lattice_*`` arrays
    (``lattice_k_anchor``, ``lattice_k_lateral``, ``lattice_k_bulk``,
    ``lattice_damping``); the scalar fields here are the global solver
    knobs that gate the solve (iterations, damping, smoothing).

    The contract_v2 sign convention is :math:`q_i = p_i - \\delta_i`,
    with :math:`\\delta_{n,i} > 0` meaning the lattice sphere is
    compressed inward along its rest outward normal. The hook writes
    :math:`\\delta_{n,i}` into ``model.lattice_delta``; downstream
    ``update_lattice_world_positions`` physically displaces the contact
    particle inward by :math:`\\delta_{n,i} \\hat{n}_{world,i}` and adds
    :math:`-\\dot{\\delta}_{n,i} \\hat{n}_{world,i}` to its world
    velocity (Hunt-Crossley rate coupling).

    Attributes:
        iterations: Damped-Jacobi iterations per solver step. v1 anchor
            path is closed-form per-sphere and ignores this; v2 jacobi
            consumes it. Default 4 matches the cslc_handler default.
        alpha: Jacobi damping factor for v2 ([0, 1], 1 = no damping).
            Contract §6.5 default is 0.3.
        smoothing_eps: Width [m] of the smooth ReLU / smooth step
            surrogates (:math:`\\sigma_\\varepsilon`,
            :math:`\\Sigma_\\varepsilon`) used by the per-pair contact
            law (contract §3.4, §11). Production default 5e-4.
        ka_tangent_ratio: :math:`\\rho = k_{a,t}/k_a`, the anisotropic
            anchor's tangent-to-normal ratio (contract §6.1). 1.0 =
            isotropic; 1/3 = incompressible-skin limit. Anchor-only v1
            path uses the normal axis exclusively, so this is recorded
            for v2 and otherwise inert.
        clamp_delta_max: Optional per-step upper bound on
            :math:`\\delta_{n,i}` [m]. None disables. Used as a
            numerical guard against runaway compression on the first
            contact substep, when ``lattice_delta_prev = 0`` and a
            Hunt-Crossley velocity coupling would otherwise see a large
            :math:`\\dot{\\delta}`. Default None (no clamp); set to a
            few times the sphere radius for early debugging.
        clamp_delta_dot_max: Optional symmetric upper bound on
            :math:`|\\dot{\\delta}_{n,i}|` [m/s] passed into the
            velocity-coupling write in ``update_lattice_world_positions``.
            None disables. Defaults to None; the SETTLE phase typically
            avoids the warmup spike without needing a clamp.
    """

    iterations: int = 4
    alpha: float = 0.3
    smoothing_eps: float = 5.0e-4
    ka_tangent_ratio: float = 1.0
    clamp_delta_max: float | None = None
    clamp_delta_dot_max: float | None = None
    # Toggles the new Jacobi-sweep solver
    # (:func:`solve_lattice_jacobi_step`, modelled on
    # ``newton._src.geometry.cslc_kernels.jacobi_step``) in place of the
    # v1 anchor-only closed-form (:func:`solve_lattice_anchor_compression`).
    # When ``True``, :meth:`SolverUXPBD.compute_compliant_contact_response`
    # loops the Jacobi kernel ``jacobi_iterations`` times with a damped
    # update ``δ_new = (1-α)·δ_old + α·δ_jacobi`` per sweep. The Jacobi
    # path also enables (when their per-sphere stiffnesses are non-zero)
    # the graph-Laplacian lateral coupling and stick-slip friction terms
    # from the CSLC kernel.
    #
    # Default flipped to True (2026-05-28) -- v1 anchor-only path is
    # retained for reference but should not be used in production: its
    # ``lattice_k_bulk`` consumes a different unit convention than the
    # Jacobi path (N/m vs N·m^(-3.5)) and it lacks per-pair area
    # weighting + multi-contact summing.
    use_jacobi: bool = True
    jacobi_iterations: int = 4
    # Scalar stick-slip friction parameters used by the Jacobi kernel
    # (``f_t = -K·M·s / (K·s + M)`` with ``K = k_stick``,
    # ``M = μ·|F_n|``, ``s = |δ_t|``).  Defaults zero so single-pair
    # validation tests see no friction contribution; raise for grasp
    # scenarios where stick-slip dominates.
    k_stick: float = 0.0
    mu_friction: float = 0.0
    # Lattice velocity-damping rate [N·s/m].  When > 0, the Jacobi
    # kernel adds an implicit-Euler ``-c_lattice · δ̇`` term to each
    # pad sphere's per-substep update, where
    # ``δ̇ ≈ (δ_new − δ_prev_step) / dt`` and ``δ_prev_step`` is the
    # snapshot taken at the start of the contact substep (=
    # ``model.lattice_delta_prev`` after the snapshot in
    # :meth:`SolverUXPBD.compute_compliant_contact_response`).
    # Concretely the kernel adds ``c_lattice / dt`` to both
    # ``k_diag_n`` and ``k_diag_t`` (the per-axis Jacobi diagonal)
    # AND adds ``(c_lattice / dt) · δ_prev_step`` to ``rhs_explicit``.
    # The form is unconditionally stable -- raising ``c_lattice``
    # monotonically pulls ``δ_new`` toward ``δ_prev_step``, damping
    # lattice oscillatory modes.
    #
    # Default ``0.0`` makes the damping force identically zero and the
    # Jacobi kernel reduces to its pre-B3 form bit-for-bit (no
    # behaviour change unless this knob is opted into).
    c_lattice: float = 0.0
    # When True (the default and current behaviour), the pp contact
    # kernel SKIPS the lattice-side body wrench
    # (``cslc_owns_lattice_wrench=1`` in ``kernels.py``) and the CSLC
    # anchor reaction in ``accumulate_cslc_body_wrench`` is the SOLE
    # body-wrench source for lattice-vs-object contact.  This matches
    # the production CSLC contract: the pad body should feel only the
    # compliance term, not the rigid PP impulse.
    #
    # When False, the pp kernel ALSO writes the lattice-side body
    # wrench in addition to the anchor reaction.  This double-counts
    # the normal direction (anchor + pp both push the body), but adds
    # back the tangential (friction-drag) wrench on the body, which
    # the anchor-only path cannot capture (anchor encodes compression
    # along the rest normal, not friction along the contact tangent).
    # For grasp scenes where torque resistance from distributed
    # friction matters more than exact normal-stiffness calibration,
    # this is the right trade-off; for single-pair CSLC validation
    # tests where the contact normal force is the load-bearing
    # quantity, keep the default True.
    enable_lattice_pp_body_wrench: bool = False
