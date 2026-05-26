# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy reference implementation of the CSLC v2 model.

This module is the single source of truth for what the GPU kernels in
``newton/_src/geometry/cslc_kernels.py`` should converge to.  Every
function corresponds to an equation in
``cslc_main/theory/contract_v2.md``.

Phase 5 cleanup: v1 sphere-target primitives (``RigidTarget``,
``effective_penetration``, ``contact_force``, ``equilibrium_face_on_analytical``,
``equilibrium_numerical``, ``equilibrium_with_friction_*``, the v1
``PointSetTarget`` with ``radii``, and the ``point_set_*`` family) and
the ``kernel_contact_force_n_axis`` witness were removed.  Every target
in v2 is a :class:`cslc_main.theory.cslc_targets.PointSetTargetV2`
``(position, normal, area)`` triple-set; the unified path is the
``half_space_*`` primitives + ``equilibrium_half_space_*`` solvers.

Sign conventions (matching ``cslc_kernels.jacobi_step``, contract §2):

    q = p - delta            (deformed centre; delta along +n_hat
                              means the sphere is compressed INWARD,
                              i.e. toward the body interior)

    f_anchor = +k_a * delta  (restoring; pulls q back toward p)

    f_contact = +k_c * phi_eff * gate * n_face       (contract §4)
        phi_eff = sigma_eps(raw),  raw = r - n_face · (q - t_sample)
        gate    = Sigma_eps(raw)
    so contact pushes the pad sphere along +n_face (target's outward
    direction) at any depth, monotone in penetration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ────────────────────────────────────────────────────────────────────────
#  Active-set threshold constant
#
#  The active-set inactive-skip threshold for per-pair contact terms.
#  A pair with ``raw_ij < INACTIVE_RAW_EPS_FACTOR * eps`` is skipped:
#  ``smooth_relu(raw, eps) < 1e-9`` and ``smooth_step(raw, eps) < 1e-9``
#  at that distance, so the excluded contribution is below numerical
#  noise.  THIS IS THE LOAD-BEARING CONSTANT that keeps theory-side
#  gold-reference evaluators consistent with the Warp kernels' inner
#  loops -- e.g. ``compute_pad_force_vs_point_set``, ``jacobi_step``,
#  and the C2 ``jacobi_step_point_set`` all use the same threshold.
#
#  Both Python and Warp kernel sites must use this exact value:
#
#    Python: import INACTIVE_RAW_EPS_FACTOR from cslc_main.theory.cslc_theory
#    Warp  : kernels embed the literal ``-50.0`` with a comment pointing
#            back to this constant (Warp can't import Python module
#            constants -- kernel literals are compiled in).
#
#  test_07_kernel_bridge's scene H/I comparators assert kernel == theory
#  at <1e-5 relative error; any drift in this convention would surface
#  there immediately on asymmetric scenes (the B1 fix that landed the
#  threshold on the theory side was triggered exactly by this argument).
# ────────────────────────────────────────────────────────────────────────


INACTIVE_RAW_EPS_FACTOR: float = -50.0


# ────────────────────────────────────────────────────────────────────────
#  Emission inclusion factor (C2d)
#
#  ``INCLUSION_FACTOR = -INACTIVE_RAW_EPS_FACTOR = 50.0`` is the positive
#  magnitude used to size the K_max-per-pad-sphere contact-buffer
#  allocation for point-set targets.  A target point j contributes
#  measurably to a pad sphere i's wrench iff
#      raw_ij = (r_i + R_j) - ||t_j - q_i||  >=  -INCLUSION_FACTOR * eps,
#  equivalently iff ``||t_j - q_i|| <= r_i + R_j + INCLUSION_FACTOR*eps``.
#
#  Why 50, and not something smaller (e.g. ~3 for tanh/erf-style cutoffs)?
#  The smooth_step surrogate used in the kernels is the ALGEBRAIC form
#  ``0.5*(1 + x/sqrt(x*x + eps*eps))`` (cslc_kernels.smooth_step), not
#  tanh or erf.  Algebraic tails are heavy: smooth_step hits the 1e-4
#  emission-gate threshold at exactly ``x = -50*eps`` (k/sqrt(k^2+1) =
#  1 - 2e-4 -> k ~= 50).  So the active-set skip threshold and the
#  emission gate cull threshold COINCIDE at -50*eps; they are not
#  separated by an order of magnitude as a tanh/erf surrogate would
#  imply.  This is a load-bearing coincidence: it means
#  ``r_inclusion = r_lat + R + INCLUSION_FACTOR*eps`` is the EXACT
#  bound on pad-vs-target distance for any pair that will write a
#  non-trivial contact.  C2d's K_max sizing depends on this bound being
#  tight; under-sizing (e.g. using 3*eps from a tanh assumption) would
#  silently truncate ~0.025-0.4 N per-pair contributions in the
#  raw in [-50*eps, -3*eps] annular shell, losing 10s of N of total
#  wrench at production densities.  See cslc_handler._launch_vs_point_set
#  for the per-pair truncation counter that guards against this formula
#  being subtly wrong on new geometries.
#
#  If you change the smooth_step surrogate family (e.g. swap to erf or
#  tanh), both INACTIVE_RAW_EPS_FACTOR and INCLUSION_FACTOR need to be
#  recomputed against the new surrogate's 1e-9 / 1e-4 thresholds and
#  the kernel literals updated.
# ────────────────────────────────────────────────────────────────────────


INCLUSION_FACTOR: float = 50.0


# ────────────────────────────────────────────────────────────────────────
#  Alignment-gate smoothing half-width (contract §3.6)
#
#  ``EPS_ALIGN_DEFAULT`` is the half-width of the cubic-smoothstep
#  alignment gate used to suppress back-side contacts on closed convex
#  targets.  Both theory and the Warp kernel evaluate the gate with the
#  SAME constant; bridge parity (T-K / T-L) regression-guards it via the
#  literal-discipline check in test_07_kernel_bridge.
#
#  Same Python/Warp split as ``INACTIVE_RAW_EPS_FACTOR``:
#    Python: import EPS_ALIGN_DEFAULT from cslc_main.theory.cslc_theory
#            (used as the default of ``solve_lattice_contact(eps_align=…)``
#             and ``lattice_contact_normal_forces(eps_align=…)``).
#    Warp  : kernels embed the literal ``0.05`` with a comment pointing
#            back to this constant.
#
#  Picked to match typical pad-lattice and target-sampling angular
#  resolutions (~ 3° half-transition).  See contract §3.6 for the
#  derivation; changing this value re-tunes the perpendicular-face
#  smoothing band and requires kernel literal updates + bridge re-run.
# ────────────────────────────────────────────────────────────────────────


EPS_ALIGN_DEFAULT: float = 0.05


# ────────────────────────────────────────────────────────────────────────
#  Lattice sphere description
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LatticeSphere:
    """A single CSLC lattice sphere.

    Attributes:
        p:  Rest-position centre in world frame [m], shape (3,).
        r:  Rest radius [m].
        n:  Rest outward unit normal in world frame, shape (3,).
        ka: Anchor stiffness [N/m].  Isotropic in this single-sphere
            reference (the anisotropic version layers on a tangent-
            ratio multiplier; see ``ka_t`` below).
        ka_t_ratio: Tangent-anchor stiffness ratio.  Tangent anchor
            stiffness is ``ka * ka_t_ratio``.  Default 1.0 (isotropic);
            1/3 matches incompressible flesh (Poisson nu -> 0.5 gives
            shear modulus G = E/3, paper III-A footnote).
    """

    p: np.ndarray
    r: float
    n: np.ndarray
    ka: float
    ka_t_ratio: float = 1.0

    def __post_init__(self):
        # Defensive: a frozen dataclass still lets us validate.
        if self.p.shape != (3,):
            raise ValueError(f"p must be shape (3,), got {self.p.shape}")
        if self.n.shape != (3,):
            raise ValueError(f"n must be shape (3,), got {self.n.shape}")
        norm = float(np.linalg.norm(self.n))
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"n must be unit length, got |n|={norm}")


# ────────────────────────────────────────────────────────────────────────
#  Geometry primitives
# ────────────────────────────────────────────────────────────────────────


def deformed_centre(sphere: LatticeSphere, delta: np.ndarray) -> np.ndarray:
    """q = p - delta  (contract §2 / eq:def-centre)."""
    return sphere.p - delta


def smooth_step(x: float, eps: float) -> float:
    """C^infinity Heaviside surrogate:  S_eps(x) = 0.5 (1 + x / sqrt(x^2 + eps^2)).

    This is exactly  d/dx  of the smooth-relu  sigma_eps(x) = 0.5*(x + sqrt(x^2+eps^2)),
    so any gradient of the form  d/d delta (k_c/2 * phi_eff^2)  picks up a
    factor of  smooth_step(raw, eps)  via the chain rule, where
    phi_eff = sigma_eps(raw)  and  raw = (r + R) - ||t - q||.

    Limits:
        x >> eps   -> 1   (saturated active contact)
        x << -eps  -> 0   (separated)
        x ≈ 0      -> 0.5 (transition midpoint; jac is half the saturated value)

    Forgetting this factor is the silent gradient bug.  In deep
    saturated contact (|raw| >> eps), smooth_step ≈ 1 and the factor is
    invisible; near contact onset (|raw| ~ eps), the factor swings
    between 0 and 1 and the gradient is off by up to 2x.
    """
    if eps <= 0.0:
        return 1.0 if x > 0.0 else (0.5 if x == 0.0 else 0.0)
    return 0.5 * (1.0 + x / np.sqrt(x * x + eps * eps))


def smooth_relu(x: float, eps: float) -> float:
    """C^infinity ReLU surrogate: sigma_eps(x) = 0.5*(x + sqrt(x^2 + eps^2)).

    The smooth-positive-part that appears throughout the contact code.
    Its derivative is exactly :func:`smooth_step` (the chain-rule pair),
    so any gradient of  d/d delta (k_c/2 * phi_eff^2)  with
    phi_eff = sigma_eps(raw) picks up a :func:`smooth_step` factor.

    Limits:
        x >> eps   -> x          (active, identity)
        x << -eps  -> eps^2/(4*|x|) -> 0  (decays as 1/|x|)
        x ≈ 0      -> eps/2      (irreducible smoothing floor)

    Previously defined inline in :func:`effective_penetration` and other
    overlap helpers; extracted to a named function here so the v2
    half-space primitives can share the same C^infinity surrogate
    without duplicating the algebra.
    """
    if eps <= 0.0:
        return max(0.0, x)
    return 0.5 * (x + np.sqrt(x * x + eps * eps))


# ────────────────────────────────────────────────────────────────────────
#  Anchor force and energy (contract §6.1)
# ────────────────────────────────────────────────────────────────────────


def anchor_force(sphere: LatticeSphere, delta: np.ndarray) -> np.ndarray:
    """f_anchor = +k_a * delta   (restoring spring, paper eq. 8).

    The +sign: anchor force is ``-k * (q - p)`` and ``q - p = -delta``,
    so the force on the sphere is ``+k_a * delta``.  For isotropic
    anchor (the default) this is uniform across all 3 axes.  Anisotropic
    anchor with tangent ratio ``ka_t_ratio`` uses ``k_a`` on the n-axis
    and ``k_a * ka_t_ratio`` on the tangent axes; we resolve into the
    sphere's local frame and rescale per axis.
    """
    if sphere.ka_t_ratio == 1.0:
        return sphere.ka * delta
    # Anisotropic case: decompose into normal + tangent and re-scale.
    delta_n = float(np.dot(delta, sphere.n))
    delta_t = delta - delta_n * sphere.n
    return sphere.ka * delta_n * sphere.n + (sphere.ka * sphere.ka_t_ratio) * delta_t


def anchor_energy(sphere: LatticeSphere, delta: np.ndarray) -> float:
    """E_anchor = 0.5 * k_a * ||delta||^2   (isotropic)."""
    if sphere.ka_t_ratio == 1.0:
        return 0.5 * sphere.ka * float(np.dot(delta, delta))
    delta_n = float(np.dot(delta, sphere.n))
    delta_t_sq = float(np.dot(delta, delta)) - delta_n * delta_n
    return 0.5 * sphere.ka * delta_n * delta_n + 0.5 * (sphere.ka * sphere.ka_t_ratio) * delta_t_sq


# ────────────────────────────────────────────────────────────────────────
#  Solvers — sphere-target v1 deleted in Phase 5; see ``half_space_*``
#  primitives + ``equilibrium_half_space_*`` solvers below for the v2
#  unified half-space contact.
# ────────────────────────────────────────────────────────────────────────

def half_space_raw(sphere: LatticeSphere, n_face: np.ndarray,
                   t_sample: np.ndarray, delta: np.ndarray) -> float:
    """Signed half-space overlap for one (pad sphere, target sample) pair.

    ``raw = r - n_face · (q - t_sample),  q = p - delta``.
    See contract_v2.md eq:raw.

    Args:
        sphere: pad lattice sphere (uses ``sphere.p``, ``sphere.r``).
        n_face: target's outward unit face normal at the sample, shape (3,).
        t_sample: target sample position [m], shape (3,).
        delta: pad sphere displacement [m], shape (3,).

    Returns:
        Signed overlap [m]. Positive when the pad sphere center has
        penetrated the target half-space to within distance ``r``; negative
        when it sits outside the contact range.  Monotone-non-decreasing in
        the penetration depth at any depth, no sign flip at face crossing.
    """
    q = sphere.p - delta
    return float(sphere.r - np.dot(n_face, q - t_sample))


def half_space_phi_eff(sphere: LatticeSphere, n_face: np.ndarray,
                       t_sample: np.ndarray, delta: np.ndarray,
                       *, eps: float = 0.0) -> float:
    """Smoothed contact overlap: ``phi_eff = sigma_eps(raw)`` (eq:phi-eff)."""
    raw = half_space_raw(sphere, n_face, t_sample, delta)
    return smooth_relu(raw, eps)


def half_space_gate(sphere: LatticeSphere, n_face: np.ndarray,
                    t_sample: np.ndarray, delta: np.ndarray,
                    *, eps: float = 0.0) -> float:
    """Smoothed activation gate: ``gate = Sigma_eps(raw)`` (eq:gate).

    Required as the chain-rule factor on every contact gradient — see
    contract_v2.md §4 ("Why ``gate_ij`` appears in the load").
    """
    raw = half_space_raw(sphere, n_face, t_sample, delta)
    return smooth_step(raw, eps)


def half_space_force(sphere: LatticeSphere, n_face: np.ndarray,
                     t_sample: np.ndarray, delta: np.ndarray,
                     kc: float, *, eps: float = 0.0) -> np.ndarray:
    """Per-pair contact term, returned as ``+∂E/∂δ`` (= physical force on q).

    Under the contract_v2 convention ``q = p − δ`` (§2), the energy
    gradient w.r.t. ``δ`` equals the physical force on the deformed
    centre ``q`` — the two are the **same vector**::

        +∂E/∂δ  =  +k_c · phi_eff · gate · n_face  =  f_phys(q)

    The kernel-side "load" on ``δ`` (used by Jacobi as the negative
    of the gradient) is the opposite sign: ``f_load = −f_phys``.

    USAGE NOTE.  When summing with :func:`anchor_force` to form a
    per-sphere residual, both functions return ``+∂E/∂δ``, so the sum
    IS ``∂E_total/∂δ`` (gradient form, zero at equilibrium) — *not*
    "net force on q" in the Newtonian sense.  At equilibrium under
    the contract's convention the two are equal up to sign, but call
    sites should be explicit about which they want.  See contract §6.5.

    Both ``phi_eff`` and ``gate`` factors are present — together they
    are the chain-rule expansion of ``∂(½ k_c · phi_eff²)/∂δ``.
    Dropping ``gate`` is the silent-gradient bug (notes.md lesson #4).

    For sub-sums over a target ``PointSetTarget`` and area + tangential
    weight ``A_j · w_t``, see Phase 2+ in the lattice solver.
    """
    phi = half_space_phi_eff(sphere, n_face, t_sample, delta, eps=eps)
    gate = half_space_gate(sphere, n_face, t_sample, delta, eps=eps)
    return kc * phi * gate * np.asarray(n_face, dtype=np.float64)


def half_space_energy(sphere: LatticeSphere, n_face: np.ndarray,
                      t_sample: np.ndarray, delta: np.ndarray,
                      kc: float, *, eps: float = 0.0) -> float:
    """Per-pair contact energy: ``E = (1/2) k_c · phi_eff^2`` (eq:E-ij).

    Single-pair primitive (``A_j = 1``, ``w_t = 1`` — the area + locality
    weights enter at the lattice level, Phase 2+).
    """
    phi = half_space_phi_eff(sphere, n_face, t_sample, delta, eps=eps)
    return 0.5 * kc * phi * phi


def equilibrium_half_space_face_on_analytical(
    sphere: LatticeSphere,
    n_face: np.ndarray,
    t_sample: np.ndarray,
    kc: float,
) -> tuple[np.ndarray, float]:
    """Closed-form equilibrium for one pad sphere vs face-on flat face.

    Requires ``n_face`` anti-parallel to ``sphere.n`` (face-on geometry).
    Implements the contract_v2 §6 series-spring law::

        d         =  r - n_face · (p - t_sample)            (rest overlap)
        delta_n*  =  k_c · d / (k_a + k_c)
        F*        =  k_a · k_c · d / (k_a + k_c)  =  k_eff · d

    Args:
        sphere: pad sphere; the closed form requires isotropic anchor
                in the normal direction (anisotropic ``ka_t_ratio`` is
                irrelevant at face-on since the tangent component of
                delta is exactly zero by symmetry).
        n_face: target's outward face normal (must satisfy
                ``n_face · sphere.n ≈ -1``).
        t_sample: target sample position [m].
        kc: contact stiffness [N/m].

    Returns:
        ``(delta, F_magnitude)`` where ``delta`` is the full vec3
        displacement (zero tangential component by symmetry).

    Raises:
        ValueError: if not face-on (n_face not anti-parallel to sphere.n
                    within 1e-9 radian).
    """
    n_face = np.asarray(n_face, dtype=np.float64)
    t_sample = np.asarray(t_sample, dtype=np.float64)

    dot_nn = float(np.dot(n_face, sphere.n))
    if not np.isclose(dot_nn, -1.0, atol=1e-9):
        raise ValueError(
            "equilibrium_half_space_face_on_analytical requires face-on "
            f"(n_face = -sphere.n); got n_face·n = {dot_nn:.9f}. "
            "Use equilibrium_half_space_numerical for tilted faces.")

    d = half_space_raw(sphere, n_face, t_sample, np.zeros(3))
    if d <= 0.0:
        return np.zeros(3), 0.0

    delta_n = kc * d / (sphere.ka + kc)
    F = sphere.ka * kc * d / (sphere.ka + kc)
    # delta_n > 0 ⇒ pad compressed inward (q on body side of p) along
    # sphere.n.  delta = delta_n · sphere.n.
    return delta_n * sphere.n, F


def equilibrium_half_space_numerical(
    sphere: LatticeSphere,
    n_face: np.ndarray,
    t_sample: np.ndarray,
    kc: float,
    *,
    eps: float = 1.0e-9,
    delta0: np.ndarray | None = None,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """General-geometry equilibrium: one pad sphere vs one face element.

    Minimises ``E_total = E_anchor + E_contact`` over delta in R^3 via
    L-BFGS-B.  Handles anisotropic anchor (via ``sphere.ka_t_ratio``)
    and arbitrary face orientation ``n_face``.

    Gradient (gradient form per contract_v2.md §6.5)::

        dE/d delta  =  anchor_force(sphere, delta)
                       + k_c · phi_eff · gate · n_face

    where ``phi_eff = sigma_eps(raw)`` and ``gate = Sigma_eps(raw)`` —
    the ``gate`` factor is the chain-rule term required to keep the
    gradient consistent with the smooth energy (notes.md lesson #4).

    Args:
        sphere: pad sphere (optionally anisotropic).
        n_face: target's outward face normal, shape (3,).  Any
                orientation; need NOT be anti-parallel to sphere.n.
        t_sample: target sample position [m], shape (3,).
        kc: contact stiffness [N/m].
        eps: smoothing width [m].  Default 1e-9 (deep-saturated regime
             for high-precision verification; production uses 5e-4).
        delta0: warm start, shape (3,).  Defaults to zeros.
        tol: L-BFGS-B ``gtol`` and ``ftol``.

    Returns:
        ``(delta, info)`` where ``delta`` is the equilibrium displacement
        (3,) and ``info`` is the scipy diagnostics dict.
    """
    n_face = np.asarray(n_face, dtype=np.float64)
    t_sample = np.asarray(t_sample, dtype=np.float64)
    if delta0 is None:
        delta0 = np.zeros(3)
    delta0 = np.asarray(delta0, dtype=np.float64)

    def fun(d: np.ndarray) -> float:
        return (anchor_energy(sphere, d)
                + half_space_energy(sphere, n_face, t_sample, d, kc, eps=eps))

    def jac(d: np.ndarray) -> np.ndarray:
        # dE/d delta in GRADIENT form (per contract_v2.md §6.5).  Each
        # component is +∂E/∂δ; equilibrium sums them to zero.
        grad = anchor_force(sphere, d)
        raw = half_space_raw(sphere, n_face, t_sample, d)
        phi = smooth_relu(raw, eps)
        gate = smooth_step(raw, eps)
        # +∂E_contact/∂δ = +k_c · phi_eff · gate · n_face.  Both phi and
        # gate are required (chain rule); forgetting gate gives 2x error
        # at raw ~ eps (silent-gradient bug, notes.md lesson #4).
        grad = grad + kc * phi * gate * n_face
        return grad

    res = minimize(
        fun, delta0, jac=jac, method="L-BFGS-B",
        options={"gtol": tol, "ftol": tol, "maxiter": 2000},
    )
    info = {
        "success": bool(res.success),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "final_grad_norm": float(np.linalg.norm(res.jac)),
        "energy": float(res.fun),
        "message": str(res.message),
    }
    return np.asarray(res.x, dtype=np.float64), info


# ────────────────────────────────────────────────────────────────────────
#  Friction (step 4): stick-slip on the tangent axis
# ────────────────────────────────────────────────────────────────────────


def friction_force_smooth(delta_t_mag: float, f_n: float, k_stick: float,
                          mu: float, eps: float = 0.0) -> float:
    """Smooth stick-slip friction magnitude as a function of |delta_t|.

    Equivalent to the kernel's "harmonic-mean" surrogate (Micro-step 4
    in cslc_kernels.py jacobi_step lines 491-495).  Kernel writes::

        scale     = k_stick * cone_scale / (k_stick + cone_scale + eps_k)
        cone_scale = mu * f_n / |delta_t|
        F_friction = scale * |delta_t|

    which simplifies to  F = (K * M * s) / (K * s + M + eps_k * s)
    after multiplying numerator and denominator by s.  Here we drop
    eps_k entirely: with K > 0 and M, s >= 0 the denominator K*s + M
    is always strictly positive once we early-return at s = 0, so
    eps_k is unnecessary -- and removing it makes the closed-form
    integral below the exact antiderivative.

        F_friction(s) = (k_stick * mu * f_n * s) / (k_stick * s + mu * f_n)

    Limits:
        s -> 0  (deep stick): F ≈ k_stick * s   (Hookean stick spring)
        s -> inf (deep slip): F → mu * f_n      (Coulomb plateau)
        s = mu*f_n/k_stick   (transition):  F = (mu*f_n)/2  (smooth)
                                            F = mu*f_n      (hard)

    The smoothing zone has half-width O(mu*f_n/k_stick); see step 4
    notes.

    Args:
        delta_t_mag: |δ_t| [m] (>= 0).
        f_n: normal contact force [N] (>= 0).
        k_stick: tangential stick stiffness [N/m] (>= 0).
        mu: Coulomb friction coefficient (>= 0).
        eps: retained for API parity with the kernel signature; ignored.

    Returns:
        Friction force magnitude [N], opposing the tangential motion.
    """
    del eps  # accepted for API parity with the kernel; see docstring.
    if delta_t_mag <= 0.0:
        return 0.0
    M = mu * f_n
    K = k_stick
    if M <= 0.0 or K <= 0.0:
        return 0.0
    return (K * M * delta_t_mag) / (K * delta_t_mag + M)


def friction_energy_smooth(delta_t_mag: float, f_n: float, k_stick: float,
                           mu: float, eps: float = 0.0) -> float:
    """Exact antiderivative of friction_force_smooth, [J].

    Closed form::

        E(s) = mu * f_n * s
             - (mu * f_n)^2 / k_stick * ln(1 + k_stick * s / (mu * f_n))

    Derivation: with K = k_stick, M = mu * f_n,

        E(s) = integral_0^s  K*M*u / (K*u + M)  du
             = (sub v = K*u + M)
             = K*M/K^2 * [v - M*ln(v)] from M to K*s+M
             = M*s - M^2/K * ln(1 + K*s/M).

    Limits:
        s -> 0   (Taylor):  E ≈ (1/2) * k_stick * s^2     (quadratic stick)
        s -> inf (asym):    E ≈ mu*f_n*s - (mu*f_n)^2/k_stick * ln(...)
                                (linear with log correction)

    The log term makes the stick-to-slip energy C-infinity smooth.

    Args:
        eps: retained for API parity; ignored.  See friction_force_smooth.
    """
    del eps
    if delta_t_mag <= 0.0:
        return 0.0
    M = mu * f_n
    K = k_stick
    if M <= 0.0 or K <= 0.0:
        return 0.0
    return M * delta_t_mag - (M * M / K) * np.log1p(K * delta_t_mag / M)




# ────────────────────────────────────────────────────────────────────────
#  Friction equilibria  (v2 contract — Phase 3)
#
#  Smooth stick-slip on the tangent axis (contract §6.4):
#      f_t = K · M · s / (K · s + M),  K = k_stick, M = μ·f_n, s = |δ_t|.
#  ``f_n`` is the half-space series-spring magnitude from
#  ``equilibrium_half_space_face_on_analytical``.
# ────────────────────────────────────────────────────────────────────────


def equilibrium_half_space_friction_analytical(
    sphere: LatticeSphere,
    n_face: np.ndarray,
    t_sample: np.ndarray,
    kc: float,
    f_ext_tangent: np.ndarray,
    k_stick: float,
    mu: float,
) -> tuple[np.ndarray, dict]:
    """Closed-form face-on friction equilibrium for one pad sphere (v2).

    The v2 successor to :func:`equilibrium_with_friction_analytical`.
    Same physics (contract §6.4); only the ``f_n`` source changes from
    the v1 sphere-vs-sphere overlap to the half-space form (eq:raw).

    Decoupled normal / tangent axes (valid for isotropic-normal anchor
    and face-on geometry — anisotropy enters via ``ka_t = ka · ka_t_ratio``).

    Normal axis (eq:phi-eff at δ=0, eq:f-anchor):
        d         =  r - n_face · (p - t_sample)        (rest half-space overlap)
        δ_n*      =  k_c · d / (k_a + k_c)
        f_n       =  k_a · k_c · d / (k_a + k_c)

    Tangent axis (stick-slip as a function of |F_ext|):
        stick (|F| ≤ F_thresh):  s = |F| / (k_at + k_stick),
                                 F_friction = k_stick · s
        slip  (|F| >  F_thresh): s = (|F| - μ f_n) / k_at,
                                 F_friction = μ f_n
        F_thresh = μ f_n · (k_at + k_stick) / k_stick

    Sign convention (contract §2): ``q = p - δ``, so an external force
    F_ext along +f_hat moves q to +f_hat ⇒ δ_t = -s · f_hat.

    The friction tangent frame is the **pad's** outward normal
    ``sphere.n`` (contract §6.4), so ``f_ext_tangent`` must be
    perpendicular to ``sphere.n`` — NOT ``n_face``.  At face-on these
    are anti-parallel so the two perpendicular planes coincide.

    Args:
        sphere: pad sphere (anisotropic tangent via ``ka_t_ratio``).
        n_face: target outward face normal (must satisfy
                ``n_face · sphere.n ≈ -1``).
        t_sample: target sample position [m].
        kc: contact stiffness [N/m].
        f_ext_tangent: external tangential force [N], shape (3,).
            Perpendicular to ``sphere.n`` (raises ValueError otherwise).
        k_stick: tangent stick spring stiffness [N/m].
        mu: Coulomb friction coefficient.

    Returns:
        ``(delta, info)`` where ``delta = δ_normal + δ_tangent`` is the
        full 3-vector displacement and ``info`` carries ``regime``
        (``"stick"`` / ``"slip"`` / ``"no_friction"``), ``s``,
        ``F_friction``, ``F_thresh``, ``f_n``, ``delta_n``.

    Raises:
        ValueError: if ``n_face`` is not anti-parallel to ``sphere.n``
                    or if ``f_ext_tangent`` has a non-trivial component
                    along ``sphere.n``.
    """
    f_ext_tangent = np.asarray(f_ext_tangent, dtype=np.float64)
    if f_ext_tangent.shape != (3,):
        raise ValueError(
            f"f_ext_tangent must be (3,), got {f_ext_tangent.shape}")
    # Friction tangent frame uses sphere.n (contract §6.4).
    f_dot_n = float(np.dot(f_ext_tangent, sphere.n))
    if abs(f_dot_n) > 1e-9 * (np.linalg.norm(f_ext_tangent) + 1e-15):
        raise ValueError(
            "f_ext_tangent must be perpendicular to sphere.n; "
            f"got |f.n| = {abs(f_dot_n):.3e}")

    # Normal equilibrium via the v2 half-space series spring (face-on).
    delta_normal, F_normal = equilibrium_half_space_face_on_analytical(
        sphere, n_face, t_sample, kc)
    f_n = F_normal

    F_mag = float(np.linalg.norm(f_ext_tangent))
    if F_mag > 0:
        f_hat = f_ext_tangent / F_mag
    else:
        # Pick any unit tangent in the (sphere.n)-perpendicular plane;
        # s = 0 makes the choice immaterial.
        seed = np.array([1.0, 0.0, 0.0])
        f_hat = seed - sphere.n * float(np.dot(seed, sphere.n))
        f_hat /= max(np.linalg.norm(f_hat), 1e-12)

    ka_t = sphere.ka * sphere.ka_t_ratio

    # Threshold: k_stick·s = μ·f_n at  s = F_thresh / (ka_t + k_stick).
    # k_stick = 0 ⇒ no stick spring ⇒ no slip threshold (anchor alone resists).
    if k_stick > 0.0 and mu > 0.0:
        F_thresh = mu * f_n * (ka_t + k_stick) / k_stick
    else:
        F_thresh = float("inf")

    if k_stick <= 0.0 or mu <= 0.0:
        regime = "no_friction"
        s = F_mag / ka_t if ka_t > 0.0 else 0.0
        F_friction = 0.0
    elif F_mag <= F_thresh:
        regime = "stick"
        s = F_mag / (ka_t + k_stick)
        F_friction = k_stick * s
    else:
        regime = "slip"
        s = (F_mag - mu * f_n) / ka_t
        F_friction = mu * f_n

    delta_tangent = -s * f_hat
    delta = delta_normal + delta_tangent

    info = {
        "regime": regime,
        "s": s,
        "F_friction": F_friction,
        "F_thresh": F_thresh,
        "f_n": f_n,
        "delta_n": float(np.dot(delta_normal, sphere.n)),
    }
    return delta, info


def equilibrium_half_space_friction_hard_numerical(
    sphere: LatticeSphere,
    n_face: np.ndarray,
    t_sample: np.ndarray,
    kc: float,
    f_ext_tangent: np.ndarray,
    k_stick: float,
    mu: float,
) -> tuple[np.ndarray, dict]:
    """Numerical reference using the HARD piecewise friction law (v2).

    The v2 successor to :func:`equilibrium_with_friction_hard_numerical`.
    Independent code path from
    :func:`equilibrium_half_space_friction_analytical` — agreement
    between the two is the real verification.

    Decouples normal axis (half-space analytical) from tangent axis
    (1-D ``scipy.optimize.minimize_scalar`` on the hard piecewise
    tangent energy).  Tangent energy::

        E(s) = (1/2) k_at s²
             + E_friction(s)
             - |F_ext| · s

        E_friction(s) = (1/2) k_stick s²,                  s ≤ s_thresh
                      = (1/2) k_stick s_thresh²
                        + μ f_n (s - s_thresh),             s > s_thresh
    """
    from scipy.optimize import minimize_scalar

    delta_normal, F_normal = equilibrium_half_space_face_on_analytical(
        sphere, n_face, t_sample, kc)
    f_n = F_normal

    F_mag = float(np.linalg.norm(f_ext_tangent))
    if F_mag > 0:
        f_hat = np.asarray(f_ext_tangent, dtype=np.float64) / F_mag
    else:
        f_hat = np.array([1.0, 0.0, 0.0])

    ka_t = sphere.ka * sphere.ka_t_ratio

    if k_stick <= 0.0 or mu <= 0.0:
        s_opt = F_mag / ka_t if ka_t > 0.0 else 0.0
        regime = "no_friction"
        F_friction = 0.0
    else:
        s_thresh = mu * f_n / k_stick

        def E(s: float) -> float:
            E_anc = 0.5 * ka_t * s * s
            if s <= s_thresh:
                E_fric = 0.5 * k_stick * s * s
            else:
                E_fric = (0.5 * k_stick * s_thresh * s_thresh
                          + mu * f_n * (s - s_thresh))
            return E_anc + E_fric - F_mag * s

        upper = max(3.0 * (F_mag + mu * f_n) / max(ka_t, 1e-30), 1e-3)
        res = minimize_scalar(E, bounds=(0.0, upper), method="bounded",
                              options={"xatol": 1e-15})
        s_opt = float(res.x)
        regime = "stick" if s_opt <= s_thresh else "slip"
        F_friction = (k_stick * s_opt if regime == "stick" else mu * f_n)

    delta_tangent = -s_opt * f_hat
    delta = delta_normal + delta_tangent
    info = {
        "regime": regime,
        "s": s_opt,
        "F_friction": F_friction,
        "f_n": f_n,
    }
    return delta, info


def equilibrium_half_space_friction_smooth_numerical(
    sphere: LatticeSphere,
    n_face: np.ndarray,
    t_sample: np.ndarray,
    kc: float,
    f_ext_tangent: np.ndarray,
    k_stick: float,
    mu: float,
    *,
    eps_contact: float = 1.0e-9,
    eps_friction: float = 1.0e-12,
    f_n_override: float | None = None,
    delta0: np.ndarray | None = None,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """L-BFGS-B equilibrium with smooth contact + smooth friction (v2).

    The v2 successor to :func:`equilibrium_with_friction_smooth_numerical`.
    Replaces v1 sphere-vs-sphere contact energy/gradient with v2
    half-space contact (contract eq:raw); the smooth friction surrogate
    (eq:friction-energy / contract §6.4) is unchanged.

    Minimises::

        E_total(δ) = anchor_energy(sphere, δ)                    (anisotropic)
                   + half_space_energy(sphere, n_face, t_sample, δ, kc)
                   + friction_energy_smooth(|δ_t|; f_n, k_stick, μ)
                   + f_ext_tangent · δ                            (external pot.)

    The friction ``f_n`` is **frozen** at the value from the analytic
    face-on normal equilibrium and held constant during optimisation
    (i.e. ``f_n`` is not re-evaluated as δ changes).  This is the
    quasi-static normal/tangent decoupling approximation — exact for
    face-on geometry (where δ_n at convergence matches the analytic
    value to smoothing precision), but it **underrates the coupling
    on tilted faces** where the true f_n drifts with δ_n.  All
    Phase-3 tests use face-on; this caveat is load-bearing for Phase
    4+ scenes that mix friction with tilted contact.

    Args:
        eps_contact: smoothing width [m] for the half-space surrogate.
                     Default 1e-9 (deep-saturated regime for theory-
                     grade precision; production kernel uses 5e-4).
        eps_friction: API parity with v1 / kernel; ignored by the smooth
                      friction law (the antiderivative is exact).
        f_n_override: if not None, use this value as ``f_n`` in the
                friction surrogate instead of the analytical
                face-on value.  **Phase 4 bridge-side hook.**  The
                kernel computes f_n LIVE (= ``|F_contact · n̂_pad|``
                at the current iterate, contract §6.4) while this
                primitive defaults to the analytical face-on value
                (for the Phase 3 hard-law decoupling).  At theory-
                grade ``eps_contact = 1e-9`` the two agree to
                smoothing precision (matches Phase 3 T-I); at
                production ``eps_contact = 5e-4`` the smooth
                equilibrium δ_n drifts ~15-20% from analytical and
                f_n drifts proportionally.  For bridge parity pass
                ``f_n_override = ka·|δ_n_smooth|`` from the
                normal-only :func:`cslc_lattice.solve_lattice_contact`
                run at the same eps.  See contract §17 finding #12.
        delta0: warm start.  None ⇒ falls back to the analytical
                solution — strongly recommended for slip regimes where
                the F = F_thresh kink in the hard limit gives L-BFGS-B
                a hard line search at random starts.
        tol: L-BFGS-B ``gtol`` / ``ftol``.
    """
    n_face_arr = np.asarray(n_face, dtype=np.float64)
    t_sample_arr = np.asarray(t_sample, dtype=np.float64)
    f_ext_arr = np.asarray(f_ext_tangent, dtype=np.float64)

    if f_n_override is None:
        # f_n from the half-space normal equilibrium (decoupled).
        _, f_n = equilibrium_half_space_face_on_analytical(
            sphere, n_face_arr, t_sample_arr, kc)
    else:
        f_n = float(f_n_override)

    if delta0 is None:
        delta_ana, _ = equilibrium_half_space_friction_analytical(
            sphere, n_face_arr, t_sample_arr, kc, f_ext_arr, k_stick, mu)
        delta0 = delta_ana.copy()
    delta0 = np.asarray(delta0, dtype=np.float64)

    def fun(d: np.ndarray) -> float:
        E_a = anchor_energy(sphere, d)
        E_c = half_space_energy(sphere, n_face_arr, t_sample_arr, d, kc,
                                eps=eps_contact)
        d_n = float(np.dot(d, sphere.n))
        d_t = d - d_n * sphere.n
        d_t_mag = float(np.linalg.norm(d_t))
        E_f = friction_energy_smooth(d_t_mag, f_n, k_stick, mu,
                                     eps=eps_friction)
        # External potential.  q = p - δ ⇒ work by f_ext on q is
        #   W = f_ext · (q - p) = -f_ext · δ;  V_ext = -W = +f_ext · δ.
        # Same sign-convention fix as the v1 smooth solver (see v1
        # docstring comment).
        E_ext = +float(np.dot(f_ext_arr, d))
        return E_a + E_c + E_f + E_ext

    def jac(d: np.ndarray) -> np.ndarray:
        g = anchor_force(sphere, d)
        # Half-space contact gradient (contract §4):
        #   ∂E_c/∂δ = +k_c · phi_eff · gate · n_face
        # phi_eff = σ_ε(raw); gate = Σ_ε(raw) — both are required (chain
        # rule).  Forgetting ``gate`` is the silent-gradient bug from
        # notes.md lesson #4.
        raw = half_space_raw(sphere, n_face_arr, t_sample_arr, d)
        phi = smooth_relu(raw, eps_contact)
        gate = smooth_step(raw, eps_contact)
        g = g + kc * phi * gate * n_face_arr
        # Friction.
        d_n = float(np.dot(d, sphere.n))
        d_t = d - d_n * sphere.n
        d_t_mag = float(np.linalg.norm(d_t))
        if d_t_mag > 1e-15:
            F_fric = friction_force_smooth(d_t_mag, f_n, k_stick, mu,
                                           eps=eps_friction)
            g = g + F_fric * (d_t / d_t_mag)
        # External: V_ext = +f_ext·δ ⇒ grad = +f_ext.
        g = g + f_ext_arr
        return g

    res = minimize(
        fun, delta0, jac=jac, method="L-BFGS-B",
        options={"gtol": tol, "ftol": tol, "maxiter": 2000},
    )
    info = {
        "success": bool(res.success),
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "final_grad_norm": float(np.linalg.norm(res.jac)),
        "energy": float(res.fun),
        "message": str(res.message),
        "f_n": f_n,
    }
    return np.asarray(res.x, dtype=np.float64), info

__all__ = [
    # Active-set + alignment constants (literal-discipline with kernel).
    "INACTIVE_RAW_EPS_FACTOR",
    "INCLUSION_FACTOR",
    "EPS_ALIGN_DEFAULT",
    # Lattice primitive.
    "LatticeSphere",
    # Geometry helpers.
    "deformed_centre",
    "smooth_step",
    "smooth_relu",
    # Anchor (contract §6.1).
    "anchor_force",
    "anchor_energy",
    # Half-space contact primitives (contract §3-4).
    "half_space_raw",
    "half_space_phi_eff",
    "half_space_gate",
    "half_space_force",
    "half_space_energy",
    "equilibrium_half_space_face_on_analytical",
    "equilibrium_half_space_numerical",
    # Friction (contract §6.4).
    "friction_force_smooth",
    "friction_energy_smooth",
    "equilibrium_half_space_friction_analytical",
    "equilibrium_half_space_friction_hard_numerical",
    "equilibrium_half_space_friction_smooth_numerical",
]
