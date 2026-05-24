# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 10b -- symmetry-robust projection variant (EXPERIMENTAL).

This module is a SCOPED EXPERIMENT, not part of the canonical CSLC
theory.  It implements a single variant solver for one pad sphere
against a ``PointSetTarget``:

    equilibrium_point_set_projected_numerical

which differs from ``cslc_theory.equilibrium_point_set_numerical`` by a
single change: the per-target-point contact energy is evaluated at
``delta_proj = (delta . n_outward) * n_outward`` instead of the full
``delta``.  That is, the pad sphere's TANGENTIAL displacement is frozen
to zero from the contact's point of view; only the normal-axis
displacement participates in the overlap calculation.

This is the "patch-resultant projection" symmetry-robustness fix from
notes.md Step 11's discussion, in its cleanest single-pad-sphere form.

Why this is least disruptive to the general finger-contact theory:

  * The modification is LOCAL to each pad sphere -- it uses only that
    sphere's outward normal ``sphere.n``.  Multi-pad-sphere scenes (a
    full lattice) compose naturally: each sphere's contact energy is
    evaluated at its own ``delta_proj_i``; the anchor, the lateral
    distance-preserving coupling, the friction, and the per-target-
    point overlap machinery are all UNCHANGED.

  * It is a strict potential -- ``E_c(delta_proj(delta))`` is a valid
    smooth function of ``delta``, with gradient

        d/d delta  E_c(delta_proj)  =  (d E_c / d(delta_proj)) . n  *  n

    (from the chain rule ``d delta_proj / d delta = n n^T``).  L-BFGS-B
    converges to genuine stationary points rather than fixed points of
    an ad-hoc gradient field.

  * The lateral component of ``delta`` is then driven by the anchor
    alone (anchor is the only term that depends on ``delta_t``), so
    ``delta_t -> 0`` at equilibrium by construction.  This kills the
    finite-sample lateral residual measured in Step 10 PART C.

What this experiment does NOT cover, and is deferred:

  * Tangentially-active contact (sliding, stick-slip friction): freezing
    delta_t in the contact energy precludes the tangential spring that
    drives the stick-slip friction model in cslc_theory.py.  The
    projection variant here is intentionally limited to NORMAL contact,
    as a baseline.  Adding friction back on top is straightforward but
    not part of Step 10b.

  * Multi-sphere finger lattices: the variant function below handles
    one pad sphere.  A lattice version would loop the same projection
    per sphere; the design is local.

Removability checklist:

  1. Delete this file (``cslc_main/theory/cslc_projection_experiment.py``).
  2. Delete the driver (``cslc_main/theory/test_10b_symmetry_projection.py``).
  3. Step 10 (``test_10_pad_vs_box.py``) continues to PASS unchanged.

Nothing else imports from this module.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .cslc_lattice import (
    Lattice,
    LateralLaw,
    SphereIndenter,
    anchor_energy as lattice_anchor_energy,
    lateral_energy_distance_preserving,
    lateral_energy_graph_laplacian,
)
from .cslc_theory import (
    INACTIVE_RAW_EPS_FACTOR,
    LatticeSphere,
    PointSetTarget,
    anchor_energy,
    anchor_force,
    point_set_contact_energy,
    point_set_raw_overlaps,
)


__all__ = [
    "equilibrium_point_set_projected_numerical",
    "solve_lattice_sphere_indenter_projected",
]


def equilibrium_point_set_projected_numerical(
    sphere: LatticeSphere,
    target: PointSetTarget,
    kc: float,
    *,
    eps: float = 1.0e-7,
    delta0: np.ndarray | None = None,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """Frozen-tangent variant of ``equilibrium_point_set_numerical``.

    Minimises::

        E_total(delta) = (1/2) ka ||delta||^2
                       + sum_j  (1/2) kc  phi_eff_j(delta_proj)^2

    where ``delta_proj = (delta . sphere.n) * sphere.n``.  The contact
    energy therefore depends only on the normal-axis component
    ``delta_n = delta . sphere.n``; the lateral component ``delta_t``
    enters only through the anchor.  At equilibrium ``delta_t = 0``.

    See module docstring for the rationale.  Same call signature as
    ``cslc_theory.equilibrium_point_set_numerical`` so the two can be
    swapped freely in benchmark drivers.

    Args:
        sphere, target, kc, eps, delta0, tol: as in
            ``equilibrium_point_set_numerical``.

    Returns:
        ``(delta, info)``.  ``info["variant"] = "patch_resultant_projection"``.
    """
    if delta0 is None:
        delta0 = np.zeros(3)
    n_outward = sphere.n

    def _proj(d: np.ndarray) -> np.ndarray:
        return float(np.dot(d, n_outward)) * n_outward

    def fun(d: np.ndarray) -> float:
        # Anchor sees the FULL delta (full anisotropic potential).
        E_a = anchor_energy(sphere, d)
        # Contact sees only the projection onto n_outward.
        E_c = point_set_contact_energy(
            sphere, target, _proj(d), kc, eps=eps)
        return E_a + E_c

    def jac(d: np.ndarray) -> np.ndarray:
        d_proj = _proj(d)
        # Anchor gradient -- full vector, unchanged from the unprojected
        # solver.
        grad_a = anchor_force(sphere, d)
        # Contact gradient at d_proj (vec3), then project onto n_outward
        # via the chain rule d(delta_proj)/d(delta) = n n^T.
        raws, dirs = point_set_raw_overlaps(sphere, target, d_proj)
        if eps <= 0.0:
            phi_effs = np.maximum(0.0, raws)
            steps = np.where(raws > 0.0, 1.0,
                             np.where(raws == 0.0, 0.5, 0.0))
        else:
            denom = np.sqrt(raws * raws + eps * eps)
            phi_effs = 0.5 * (raws + denom)
            steps = 0.5 * (1.0 + raws / denom)
        g_c_at_proj = (kc * (phi_effs * steps)[:, None] * dirs).sum(axis=0)
        g_c_proj_along_n = float(np.dot(g_c_at_proj, n_outward)) * n_outward
        return grad_a + g_c_proj_along_n

    res = minimize(
        fun, delta0, jac=jac, method="L-BFGS-B",
        options={"gtol": tol, "ftol": tol, "maxiter": 500},
    )
    info = {
        "success": bool(res.success),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "final_grad_norm": float(np.linalg.norm(res.jac)),
        "energy": float(res.fun),
        "message": str(res.message),
        "M": target.M,
        "variant": "patch_resultant_projection",
    }
    return np.asarray(res.x, dtype=np.float64), info


def solve_lattice_sphere_indenter_projected(
    lat: Lattice,
    indenter: SphereIndenter,
    r_lat: np.ndarray | float,
    *,
    lateral: LateralLaw = "distance_preserving",
    delta0: np.ndarray | None = None,
    eps: float = 5.0e-4,
    tol: float = 1.0e-12,
    maxiter: int = 5000,
) -> tuple[np.ndarray, dict]:
    """Lattice-scale generalisation of the per-sphere projection variant.

    Mirrors ``cslc_lattice.solve_lattice_sphere_indenter`` exactly,
    except that each pad sphere's contact term sees the projected
    displacement ``delta_proj_i = (delta_i . lat.n[i]) * lat.n[i]``
    rather than the full ``delta_i``.  The anchor and the lateral
    distance-preserving (or graph-Laplacian) coupling continue to use
    the full ``delta`` -- those terms are PAD-internal physics that the
    projection deliberately does not touch.  The modification is local
    per sphere (each sphere projects onto its own ``lat.n[i]``), so
    multi-pad-sphere / curved-lattice / non-aligned-outward-normal
    scenes compose cleanly.

    What this means physically: every pad sphere can still translate
    in any direction under the anchor + lateral coupling, but the
    CONTACT FORCE it experiences from the indenter only acts along its
    own outward normal.  Lattice-asymmetry-induced lateral coupling
    between the contact term and ``delta_t_i`` -- the mechanism behind
    the dome y-drift in notes.md Step 11 -- is removed by construction.
    The contact contribution to ``delta_t_i`` is exactly zero; the
    only thing driving ``delta_t_i`` is the lateral coupling from
    neighbours, which is symmetric in expectation.

    Args:
        Identical signature to ``solve_lattice_sphere_indenter``.

    Returns:
        ``(deltas, info)``.  ``info["variant"] =
        "patch_resultant_projection_lattice"`` so consumers can confirm
        which solver produced the answer.
    """
    N = lat.N
    if np.isscalar(r_lat):
        r_arr = np.full(N, float(r_lat), dtype=np.float64)
    else:
        r_arr = np.asarray(r_lat, dtype=np.float64)
        if r_arr.shape != (N,):
            raise ValueError(
                f"r_lat must be scalar or shape ({N},), got {r_arr.shape}")

    t = np.asarray(indenter.t, dtype=np.float64)
    R = float(indenter.R)
    kc = float(indenter.kc)
    normals = np.asarray(lat.n, dtype=np.float64)   # (N, 3) cached for speed

    if delta0 is None:
        delta0 = np.zeros((N, 3))

    def _contact_terms_projected(d: np.ndarray):
        """Yield ``(i, phi_eff, step, e_hat_proj, n_i)`` for each pad sphere
        whose PROJECTED contact contribution is non-negligible.

        Same active-skip threshold (``raw < -50*eps``) as the baseline.
        ``e_hat_proj`` is the line-of-centres from the projected
        deformed centre ``q_i_proj = lat.p[i] - delta_proj_i`` to the
        ball, which can differ from the baseline ``e_hat`` once
        ``delta_t_i`` is nonzero -- but at convergence ``delta_t_i = 0``
        under the projected energy, so the two coincide there.
        """
        for i in range(N):
            n_i = normals[i]
            delta_n_i = float(np.dot(d[i], n_i))
            d_proj_i = delta_n_i * n_i
            q_i_proj = lat.p[i] - d_proj_i
            diff = q_i_proj - t
            L = float(np.linalg.norm(diff))
            if L < 1.0e-15:
                continue
            raw = (r_arr[i] + R) - L
            if eps <= 0.0:
                if raw <= 0.0:
                    continue
                phi_eff = raw
                step = 1.0
            else:
                if raw < INACTIVE_RAW_EPS_FACTOR * eps:
                    continue
                r2 = raw * raw + eps * eps
                sqr2 = np.sqrt(r2)
                phi_eff = 0.5 * (raw + sqr2)
                step = 0.5 * (1.0 + raw / sqr2)
            e_hat_proj = diff / L
            yield i, phi_eff, step, e_hat_proj, n_i

    def fun(x: np.ndarray) -> float:
        d = x.reshape(N, 3)
        # Anchor + lateral see the FULL delta (untouched pad-internal
        # physics).
        E = lattice_anchor_energy(lat, d)
        if lateral == "graph_laplacian":
            E += lateral_energy_graph_laplacian(lat, d)
        elif lateral == "distance_preserving":
            E += lateral_energy_distance_preserving(lat, d)
        else:
            raise ValueError(f"unknown lateral: {lateral!r}")
        # Contact sees the projected delta only.
        for _, phi_eff, _, _, _ in _contact_terms_projected(d):
            E += 0.5 * kc * phi_eff * phi_eff
        return E

    def jac(x: np.ndarray) -> np.ndarray:
        d = x.reshape(N, 3)
        g = np.zeros_like(d)
        # Anchor gradient (full delta).
        g += lat.ka * d
        # Lateral gradient (full delta).
        if lateral == "graph_laplacian":
            for (i, j) in lat.edges:
                diff = d[i] - d[j]
                g[i] += lat.kl * diff
                g[j] -= lat.kl * diff
        elif lateral == "distance_preserving":
            for (i, j) in lat.edges:
                q_i = lat.p[i] - d[i]
                q_j = lat.p[j] - d[j]
                v = q_j - q_i
                l = float(np.linalg.norm(v))
                if l < 1.0e-15:
                    continue
                L_rest = float(np.linalg.norm(lat.p[j] - lat.p[i]))
                e_hat = v / l
                contrib = lat.kl * (l - L_rest) * e_hat
                g[i] += contrib
                g[j] -= contrib
        else:
            raise ValueError(f"unknown lateral: {lateral!r}")
        # Contact gradient, projected per sphere onto its own outward
        # normal via the chain rule  d(delta_proj_i)/d(delta_i) =
        # n_i n_i^T.
        for i, phi_eff, step, e_hat_proj, n_i in _contact_terms_projected(d):
            g_c_full = kc * phi_eff * step * e_hat_proj
            g[i] += float(np.dot(g_c_full, n_i)) * n_i
        return g.reshape(-1)

    res = minimize(
        fun, delta0.reshape(-1), jac=jac, method="L-BFGS-B",
        options={"gtol": tol, "ftol": tol, "maxiter": maxiter},
    )
    info = {
        "success": bool(res.success),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "final_grad_norm": float(np.linalg.norm(res.jac)),
        "energy": float(res.fun),
        "message": str(res.message),
        "n_active": sum(1 for _ in
                        _contact_terms_projected(res.x.reshape(N, 3))),
        "variant": "patch_resultant_projection_lattice",
    }
    return np.asarray(res.x, dtype=np.float64).reshape(N, 3), info
