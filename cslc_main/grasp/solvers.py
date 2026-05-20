# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver factory.

MuJoCo is the production solver; semi-implicit is available as a
backup (slower but more differentiable).  The MuJoCo path auto-sizes
``njmax`` / ``nconmax`` to fit the CSLC contact-slot budget (one slot
per surface lattice sphere per pair) and bumps iterations to 100 in
CSLC mode so the ~hundreds of simultaneous constraints actually
converge.
"""

from __future__ import annotations

import warnings

import warp as wp

from newton.solvers import SolverSemiImplicit

try:
    from newton.solvers import SolverMuJoCo

    HAS_MUJOCO = True
except ImportError:  # pragma: no cover — depends on optional dep
    HAS_MUJOCO = False
    warnings.warn("SolverMuJoCo not available — semi-implicit only.")

# Must match newton/_src/geometry/types.py:ShapeFlags.CSLC.
_CSLC_FLAG = 1 << 5

from .params import SolverParams


def _count_cslc_contact_slots(model) -> int:
    """Sum the surface-sphere counts across all CSLC-flagged shapes.

    The CSLC handler reserves one MuJoCo contact slot per surface
    lattice sphere per pair.  We count the surface spheres by walking
    each CSLC shape and computing its grid-based surface count from
    ``shape_cslc_spacing`` and ``shape_scale`` — matches Newton's
    box-face factory.  Mesh-based CSLC pads override this estimate at
    runtime; over-estimating is safe (extra unused slots), under-
    estimating crashes MuJoCo on first kernel launch.
    """
    if model.shape_cslc_spacing is None:
        return 0
    spacing = model.shape_cslc_spacing.numpy()
    flags = model.shape_flags.numpy()
    scale = model.shape_scale.numpy()
    total = 0
    for i in range(model.shape_count):
        if not (flags[i] & _CSLC_FLAG):
            continue
        sp = float(spacing[i])
        if sp <= 0:
            continue
        hx, hy, hz = (float(scale[i][j]) for j in range(3))
        # n_per_axis = round(2h/spacing) + 1, ≥ 2 (always at least a 2×2×2 cube).
        nx, ny, nz = (max(int(round(2.0 * h / sp)) + 1, 2) for h in (hx, hy, hz))
        interior = max(nx - 2, 0) * max(ny - 2, 0) * max(nz - 2, 0)
        total += nx * ny * nz - interior
    return total


def make_solver(model, params: SolverParams):
    """Construct the requested solver, configured for the given model.

    Args:
        model: a finalised ``newton.Model``.
        params: solver selection + tuning (``SolverParams``).

    Returns:
        A Newton solver instance ready for ``solver.step(...)``.
    """
    if params.name == "semi":
        return SolverSemiImplicit(model)
    if params.name != "mujoco":
        raise ValueError(f"Unknown solver name: {params.name!r}")
    if not HAS_MUJOCO:
        raise RuntimeError(
            "MuJoCo solver requested but the import failed; install Newton "
            "with the mujoco extra or switch SolverParams.name to 'semi'."
        )

    has_cslc = model.shape_cslc_spacing is not None
    ncon = params.extra_ncon + _count_cslc_contact_slots(model)

    iterations = params.iterations
    if iterations is None:
        iterations = 100 if has_cslc else 20

    return SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver=params.solver,
        integrator=params.integrator,
        cone=params.cone,
        iterations=iterations,
        ls_iterations=params.ls_iterations,
        njmax=ncon,
        nconmax=ncon,
    )
