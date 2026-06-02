# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy reference implementation of the CSLC model.

Single source of truth for what the GPU kernels in
``newton/_src/geometry/cslc_kernels.py`` should converge to.  Every
function corresponds to an equation in
``cslc_main/theory/theory.md``.  Targets are
:class:`cslc_main.theory.cslc_targets.PointSetTarget` ``(position,
normal, area)`` triple-sets; the unified contact path is the
``half_space_*`` primitives plus ``equilibrium_half_space_*`` solvers.

Sign conventions (matching ``cslc_kernels.jacobi_step``, theory.md §2):

    q = p - delta            (deformed centre; delta along +n_hat
                              means the sphere is compressed inward,
                              i.e. toward the body interior)

    f_anchor = +k_a * delta  (restoring; pulls q back toward p)

    f_contact = +k_c * phi_eff * gate * n_face       (theory.md §4)
        phi_eff = sigma_eps(raw) * sqrt(sigma_eps(raw) + eps)
                                       (Hertz-like; raw^1.5 saturated)
        raw     = r - n_face · (q - t_sample)
        gate    = Sigma_eps(raw)

    The Hertz-like phi_eff makes per-pair dF/d(raw) ∝ sqrt(raw) vanish
    at first touch.  The kernel writes the force law directly
    (theory.md §4 "Note on energy form"): there is no globally-defined
    potential whose gradient equals this force, so lattice equilibrium
    is a fixed-point of forces, not the stationary point of an energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ────────────────────────────────────────────────────────────────────────
#  Active-set + alignment constants  (theory.md §3.6, §12)
#
#  ``INACTIVE_RAW_EPS_FACTOR``: a pair with
#  ``raw_ij < INACTIVE_RAW_EPS_FACTOR * eps`` is skipped because both
#  ``smooth_relu(raw, eps)`` and ``smooth_step(raw, eps)`` fall below
#  numerical noise at that depth.
#
#  ``INCLUSION_FACTOR = -INACTIVE_RAW_EPS_FACTOR``: positive magnitude
#  used by the handler to size the K_max-per-pad-sphere contact buffer.
#  A target sample j contributes measurably iff
#  ``raw_ij >= -INCLUSION_FACTOR * eps``.  The value 50 is set by the
#  algebraic smooth-step surrogate ``0.5·(1 + x/sqrt(x²+eps²))`` used
#  in the kernels: that surrogate hits its 1e-4 cull threshold exactly
#  at ``x = -50 eps``, so the active-set skip threshold and the
#  emission cull coincide.
#
#  ``EPS_ALIGN_DEFAULT``: half-width of the cubic-smoothstep alignment
#  gate that suppresses back-side contacts on closed convex targets.
#  0.05 ≈ a 3° angular half-transition, matched to typical pad lattice
#  and target-sampling angular resolutions.
#
#  All three constants are also embedded as literals in the Warp
#  kernels (kernel literals are compiled in and cannot import Python
#  module constants); changing any value requires updating the kernel
#  literal in lockstep.
# ────────────────────────────────────────────────────────────────────────


INACTIVE_RAW_EPS_FACTOR: float = -50.0
INCLUSION_FACTOR: float = 50.0
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
    """``q = p - delta``  (theory.md eq:def-centre)."""
    return sphere.p - delta


def smooth_step(x: float, eps: float) -> float:
    """C^∞ Heaviside surrogate: ``Σ_ε(x) = 0.5·(1 + x/sqrt(x²+eps²))``.

    The derivative of :func:`smooth_relu`: ``dσ_ε/dx = Σ_ε``.  Appears
    in any gradient of ``½ k_c · phi_eff²`` via the chain rule.

    Limits:
        x >> eps   → 1     (saturated, active contact)
        x << -eps  → 0     (separated)
        x = 0      → 0.5   (transition midpoint)
    """
    if eps <= 0.0:
        return 1.0 if x > 0.0 else (0.5 if x == 0.0 else 0.0)
    return 0.5 * (1.0 + x / np.sqrt(x * x + eps * eps))


def smooth_relu(x: float, eps: float) -> float:
    """C^∞ ReLU surrogate: ``σ_ε(x) = 0.5·(x + sqrt(x²+eps²))``.

    Its derivative is :func:`smooth_step` (chain-rule pair).

    Limits:
        x >> eps   → x                (identity, active)
        x << -eps  → eps²/(4|x|) → 0  (decays as 1/|x|)
        x = 0      → eps/2            (irreducible smoothing floor)
    """
    if eps <= 0.0:
        return max(0.0, x)
    return 0.5 * (x + np.sqrt(x * x + eps * eps))


# ────────────────────────────────────────────────────────────────────────
#  Anchor force and energy (theory.md §6.1)
# ────────────────────────────────────────────────────────────────────────


def anchor_force(sphere: LatticeSphere, delta: np.ndarray) -> np.ndarray:
    """Anchor restoring spring: ``f_anchor = +k_a · delta``.

    Sign: anchor force is ``-k·(q - p)`` and ``q - p = -delta``, so the
    force on the sphere is ``+k_a · delta``.  Isotropic when
    ``ka_t_ratio = 1``; otherwise normal/tangent axes are scaled
    separately in the sphere's local frame: ``k_a`` along ``n̂`` and
    ``k_a · ka_t_ratio`` along the tangent plane.
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
#  Half-space contact primitives (theory.md §3-§4)
# ────────────────────────────────────────────────────────────────────────


def half_space_raw(sphere: LatticeSphere, n_face: np.ndarray,
                   t_sample: np.ndarray, delta: np.ndarray) -> float:
    """Signed half-space overlap for one (pad sphere, target sample) pair.

    ``raw = r - n_face · (q - t_sample),  q = p - delta``
    (theory.md eq:raw).  Positive when the pad sphere center has
    penetrated the target half-space to within distance ``r``; negative
    when outside contact range.  Monotone non-decreasing in penetration
    depth; no sign flip at face crossing.

    Args:
        sphere: pad lattice sphere (uses ``sphere.p``, ``sphere.r``).
        n_face: target's outward unit face normal at the sample, shape (3,).
        t_sample: target sample position [m], shape (3,).
        delta: pad sphere displacement [m], shape (3,).
    """
    q = sphere.p - delta
    return float(sphere.r - np.dot(n_face, q - t_sample))


def half_space_phi_eff(sphere: LatticeSphere, n_face: np.ndarray,
                       t_sample: np.ndarray, delta: np.ndarray,
                       *, eps: float = 0.0) -> float:
    """Hertz-like smoothed overlap (theory.md eq:phi-eff).

    ``phi_eff = σ_ε(raw) · sqrt(σ_ε(raw) + eps)``.  In the saturated
    regime ``raw >> eps`` we have ``σ_ε ≈ raw`` so ``phi_eff ≈ raw^1.5``;
    the per-pair stiffness ``dF/d(raw) ∝ sqrt(raw)`` vanishes at first
    touch.  The ``+ eps`` inside the square root is a smoothing guard
    so the derivative stays bounded as ``raw → 0``.  When ``eps = 0``
    and ``raw ≤ 0`` the expression collapses to 0, matching the hard
    Hertz limit.
    """
    raw = half_space_raw(sphere, n_face, t_sample, delta)
    sigma = smooth_relu(raw, eps)
    return float(sigma * np.sqrt(sigma + max(eps, 0.0)))


def half_space_gate(sphere: LatticeSphere, n_face: np.ndarray,
                    t_sample: np.ndarray, delta: np.ndarray,
                    *, eps: float = 0.0) -> float:
    """Smoothed activation gate: ``gate = Σ_ε(raw)`` (theory.md eq:gate)."""
    raw = half_space_raw(sphere, n_face, t_sample, delta)
    return smooth_step(raw, eps)


def half_space_force(sphere: LatticeSphere, n_face: np.ndarray,
                     t_sample: np.ndarray, delta: np.ndarray,
                     kc: float, *, eps: float = 0.0) -> np.ndarray:
    """Per-pair physical force on the pad sphere (theory.md eq:f-phys).

    ``f_phys = + k_c · phi_eff · gate · n_face``

    with ``phi_eff = σ_ε(raw)·sqrt(σ_ε(raw)+eps)`` (Hertz) and
    ``gate = Σ_ε(raw)``.  Single-pair primitive: no area weight, no
    tangential locality kernel; those enter at the lattice level
    (see :func:`cslc_lattice.solve_lattice_contact`).

    **Force-form, not energy gradient.**  Under the Hertz lift the
    kernel's force law is not the gradient of any global potential
    (theory.md §4 "Note on energy form").  Equilibrium is a fixed-point
    of ``anchor_force + half_space_force = 0``, not the stationary
    point of a globally-defined potential.  :func:`half_space_energy`
    is retained as a heuristic Lyapunov scalar for warm-start use.
    """
    phi = half_space_phi_eff(sphere, n_face, t_sample, delta, eps=eps)
    gate = half_space_gate(sphere, n_face, t_sample, delta, eps=eps)
    return kc * phi * gate * np.asarray(n_face, dtype=np.float64)


def half_space_energy(sphere: LatticeSphere, n_face: np.ndarray,
                      t_sample: np.ndarray, delta: np.ndarray,
                      kc: float, *, eps: float = 0.0) -> float:
    """Heuristic per-pair contact scalar: ``E_heur = ½ k_c · phi_eff²``.

    Not the antiderivative of :func:`half_space_force` — under the
    Hertz lift the force law has no global potential (theory.md §4).
    Kept as a smooth, non-negative Lyapunov-like scalar useful for
    warm-starting L-BFGS-B (which finds zeros of the force-form
    Jacobian regardless of the heuristic energy mismatch).  Single-pair
    primitive (``A_j = 1``, ``w_t = 1``).
    """
    phi = half_space_phi_eff(sphere, n_face, t_sample, delta, eps=eps)
    return 0.5 * kc * phi * phi


def equilibrium_half_space_face_on_analytical(
    sphere: LatticeSphere,
    n_face: np.ndarray,
    t_sample: np.ndarray,
    kc: float,
    *,
    eps: float = 0.0,
) -> tuple[np.ndarray, float]:
    """1D face-on equilibrium for one pad sphere vs flat face (Hertz law).

    Under the Hertz form (theory.md eq:phi-eff) the per-pair force is
    nonlinear in delta_n.  Face-on symmetry (``n_face = -sphere.n``)
    keeps the displacement along ``sphere.n``, so the residual reduces
    to the scalar equation::

        ka · delta_n  =  kc · phi_eff(raw) · gate(raw)
        raw           =  d0 - delta_n
        d0            =  r - n_face · (p - t_sample)          (rest overlap)

    Solved by ``scipy.optimize.brentq`` on the bracket
    ``delta_n ∈ [0, d0]``: the residual is strictly monotone (anchor
    grows linearly with delta_n while the Hertz contact force decreases
    as raw shrinks), so a single bracket suffices.

    Args:
        sphere: pad sphere; anisotropic ``ka_t_ratio`` is irrelevant
                at face-on (tangent component is exactly zero by
                symmetry).
        n_face: target's outward face normal (must satisfy
                ``n_face · sphere.n ≈ -1``).
        t_sample: target sample position [m].
        kc: contact stiffness [N/m] (per-pair; lattice-level rescaling
            to per-volume k_c is handled by the CSLC handler — see
            theory.md §10).
        eps: smoothing width for the Hertz surrogate.  Default 0.0
             (saturated/sharp limit).  Production kernel uses 5e-4.

    Returns:
        ``(delta, F_magnitude)`` where ``delta = delta_n · sphere.n``.
        Returns zeros if not in contact (``d0 ≤ 0``).

    Raises:
        ValueError: if not face-on (``n_face`` not anti-parallel to
                    ``sphere.n`` within 1e-9 cosine tolerance).
    """
    from scipy.optimize import brentq

    n_face = np.asarray(n_face, dtype=np.float64)
    t_sample = np.asarray(t_sample, dtype=np.float64)

    dot_nn = float(np.dot(n_face, sphere.n))
    if not np.isclose(dot_nn, -1.0, atol=1e-9):
        raise ValueError(
            "equilibrium_half_space_face_on_analytical requires face-on "
            f"(n_face = -sphere.n); got n_face·n = {dot_nn:.9f}. "
            "Use equilibrium_half_space_numerical for tilted faces.")

    d0 = half_space_raw(sphere, n_face, t_sample, np.zeros(3))
    if d0 <= 0.0:
        return np.zeros(3), 0.0

    # Residual R(delta_n) = ka·delta_n − kc·phi_eff(raw)·gate(raw),
    # raw = d0 − delta_n.  Bracket [0, d0]: at delta_n=0, R = −kc·phi_eff(d0)·gate(d0) ≤ 0;
    # at delta_n=d0, raw≈0 ⇒ phi_eff→0, R = ka·d0 ≥ 0.  Strictly monotone increasing.
    def residual(delta_n: float) -> float:
        raw = d0 - delta_n
        sigma = smooth_relu(raw, eps)
        phi_eff = sigma * np.sqrt(sigma + max(eps, 0.0))
        gate = smooth_step(raw, eps)
        return sphere.ka * delta_n - kc * phi_eff * gate

    delta_n = float(brentq(residual, 0.0, d0, xtol=1e-15, rtol=1e-12))
    F = sphere.ka * delta_n
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
    """Force-fixed-point equilibrium: one pad sphere vs one face element.

    Solves the force-balance residual via ``scipy.optimize.root``
    (Powell's hybrid 'hybr')::

        R(delta)  =  anchor_force(sphere, delta)
                     + k_c · phi_eff · gate · n_face   =   0

    with ``phi_eff = σ_ε(raw)·sqrt(σ_ε(raw)+eps)`` (Hertz) and
    ``gate = Σ_ε(raw)``.  Handles anisotropic anchor (via
    ``sphere.ka_t_ratio``) and arbitrary face orientation.

    Under the Hertz lift the kernel's force law is not the gradient of
    any global potential (theory.md §4), so this root-finds on the
    force residual rather than minimising a heuristic energy with an
    inconsistent gradient.  Matches the production kernel, which also
    iterates to the force fixed-point.

    Args:
        sphere: pad sphere (optionally anisotropic).
        n_face: target's outward face normal, shape (3,).  Any
                orientation; need NOT be anti-parallel to sphere.n.
        t_sample: target sample position [m], shape (3,).
        kc: contact stiffness [N/m].
        eps: smoothing width [m].  Default 1e-9 (theory-side precision);
             production kernel uses 5e-4.
        delta0: warm start, shape (3,).  Defaults to zeros.
        tol: residual tolerance for the root finder.

    Returns:
        ``(delta, info)`` where ``delta`` is the equilibrium displacement
        (3,) and ``info`` carries ``success``, ``nfev``, ``final_grad_norm``
        (= ``||R||`` at convergence), and ``message``.
    """
    from scipy.optimize import root

    n_face = np.asarray(n_face, dtype=np.float64)
    t_sample = np.asarray(t_sample, dtype=np.float64)
    if delta0 is None:
        delta0 = np.zeros(3)
    delta0 = np.asarray(delta0, dtype=np.float64)

    def residual(d: np.ndarray) -> np.ndarray:
        # R(δ) = ∂E_anchor/∂δ + f_phys_contact.  Both terms are
        # +∂E_X/∂δ on the anchor side and the kernel's force-form
        # contact term (eq:f-phys); equilibrium ⇔ R = 0.
        r_vec = anchor_force(sphere, d)
        raw = half_space_raw(sphere, n_face, t_sample, d)
        sigma = smooth_relu(raw, eps)
        phi_eff = sigma * np.sqrt(sigma + max(eps, 0.0))
        gate = smooth_step(raw, eps)
        r_vec = r_vec + kc * phi_eff * gate * n_face
        return r_vec

    res = root(residual, delta0, method="hybr", tol=tol)
    info = {
        "success": bool(res.success),
        "nfev": int(res.nfev),
        "final_grad_norm": float(np.linalg.norm(res.fun)),
        "energy": float(
            anchor_energy(sphere, res.x)
            + half_space_energy(sphere, n_face, t_sample, res.x, kc, eps=eps)
        ),
        "message": str(res.message),
    }
    return np.asarray(res.x, dtype=np.float64), info


# ────────────────────────────────────────────────────────────────────────
#  Friction: smooth stick-slip on the tangent axis (theory.md §6.4)
# ────────────────────────────────────────────────────────────────────────


def friction_force_smooth(delta_t_mag: float, f_n: float, k_stick: float,
                          mu: float, eps: float = 0.0) -> float:
    """Smooth stick-slip friction magnitude as a function of |delta_t|.

    Matches the kernel's harmonic-mean surrogate (theory.md eq:friction)::

        F_friction(s) = (k_stick · μ · f_n · s) / (k_stick · s + μ · f_n)

    with ``s = |δ_t|``, ``M = μ·f_n``, ``K = k_stick``.

    Limits:
        s → 0    (deep stick):   F ≈ k_stick · s        (Hookean spring)
        s → ∞    (deep slip):    F → μ · f_n            (Coulomb plateau)
        s = M/K  (transition):   F = (μ · f_n) / 2      (smooth crossover)

    The smoothing zone has half-width ``O(μ · f_n / k_stick)``.

    Args:
        delta_t_mag: |δ_t| [m] (≥ 0).
        f_n: normal contact force [N] (≥ 0).
        k_stick: tangential stick stiffness [N/m] (≥ 0).
        mu: Coulomb friction coefficient (≥ 0).
        eps: accepted for API parity with the kernel signature; ignored
             (the closed-form below has no smoothing guard term).
    """
    del eps
    if delta_t_mag <= 0.0:
        return 0.0
    M = mu * f_n
    K = k_stick
    if M <= 0.0 or K <= 0.0:
        return 0.0
    return (K * M * delta_t_mag) / (K * delta_t_mag + M)


def friction_energy_smooth(delta_t_mag: float, f_n: float, k_stick: float,
                           mu: float, eps: float = 0.0) -> float:
    """Exact antiderivative of :func:`friction_force_smooth`, [J].

    Closed form (theory.md §6.4)::

        E(s) = μ·f_n·s − (μ·f_n)²/k_stick · ln(1 + k_stick·s/(μ·f_n))

    Derivation: with ``K = k_stick``, ``M = μ·f_n``,

        E(s) = ∫_0^s  K·M·u / (K·u + M)  du
             = M·s − (M²/K)·ln(1 + K·s/M).

    Limits:
        s → 0    (Taylor):  E ≈ ½·k_stick·s²   (quadratic stick)
        s → ∞    (asymp.):  E ≈ μ·f_n·s − (μ·f_n)²/k_stick · ln(...)
                                              (linear with log correction)

    The log term makes the stick-to-slip energy C^∞ smooth.

    Args:
        eps: accepted for API parity; ignored.
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
#  Friction equilibria  (theory.md §6.4)
#
#  Smooth stick-slip on the tangent axis:
#      f_t = K · M · s / (K · s + M),  K = k_stick, M = μ·f_n, s = |δ_t|.
#  ``f_n`` is the half-space series-spring magnitude from
#  :func:`equilibrium_half_space_face_on_analytical`.
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
    """Closed-form face-on friction equilibrium for one pad sphere.

    Decoupled normal / tangent axes (valid for face-on geometry;
    anisotropy enters via ``ka_t = ka · ka_t_ratio``).

    Normal axis: 1-D Hertz series spring (see
    :func:`equilibrium_half_space_face_on_analytical`) yields ``δ_n``
    and ``f_n = k_a · δ_n``.

    Tangent axis (stick-slip as a function of ``|F_ext|``)::

        stick (|F| ≤ F_thresh):  s = |F| / (k_at + k_stick),
                                 F_friction = k_stick · s
        slip  (|F| >  F_thresh): s = (|F| − μ·f_n) / k_at,
                                 F_friction = μ · f_n
        F_thresh = μ · f_n · (k_at + k_stick) / k_stick

    Sign (theory.md §2): ``q = p − δ``, so external force ``F_ext``
    along ``+f_hat`` moves q in that direction ⇒ ``δ_t = −s · f_hat``.

    The friction tangent frame is the **pad's** outward normal
    ``sphere.n`` (theory.md §6.4), so ``f_ext_tangent`` must be
    perpendicular to ``sphere.n`` — not ``n_face``.  At face-on these
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
        ``(delta, info)`` where ``delta = δ_normal + δ_tangent`` and
        ``info`` carries ``regime`` (``"stick"`` / ``"slip"`` /
        ``"no_friction"``), ``s``, ``F_friction``, ``F_thresh``,
        ``f_n``, ``delta_n``.

    Raises:
        ValueError: if ``n_face`` is not anti-parallel to ``sphere.n``,
                    or if ``f_ext_tangent`` has a non-trivial component
                    along ``sphere.n``.
    """
    f_ext_tangent = np.asarray(f_ext_tangent, dtype=np.float64)
    if f_ext_tangent.shape != (3,):
        raise ValueError(
            f"f_ext_tangent must be (3,), got {f_ext_tangent.shape}")
    f_dot_n = float(np.dot(f_ext_tangent, sphere.n))
    if abs(f_dot_n) > 1e-9 * (np.linalg.norm(f_ext_tangent) + 1e-15):
        raise ValueError(
            "f_ext_tangent must be perpendicular to sphere.n; "
            f"got |f.n| = {abs(f_dot_n):.3e}")

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
    """Numerical reference using the hard piecewise friction law.

    Independent code path from
    :func:`equilibrium_half_space_friction_analytical`; agreement
    between the two cross-validates the closed form.

    Decouples normal axis (1-D Hertz analytical) from tangent axis
    (1-D ``scipy.optimize.minimize_scalar`` on the hard piecewise
    tangent energy).  Tangent energy::

        E(s) = ½·k_at·s²  +  E_friction(s)  −  |F_ext|·s

        E_friction(s) = ½·k_stick·s²                          (s ≤ s_thresh)
                      = ½·k_stick·s_thresh² + μ·f_n·(s − s_thresh)
                                                              (s > s_thresh)
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
    """L-BFGS-B equilibrium with smooth contact + smooth friction.

    Minimises::

        E_total(δ) = anchor_energy(sphere, δ)              (anisotropic)
                   + half_space_energy(sphere, n_face, t_sample, δ, kc)
                   + friction_energy_smooth(|δ_t|; f_n, k_stick, μ)
                   + f_ext_tangent · δ                     (external potential)

    The Jacobian uses the force-form contact term (theory.md eq:f-phys);
    under the Hertz lift this differs from ``∂E_c/∂δ`` of the heuristic
    quadratic energy, so the objective is Lyapunov-like rather than a
    true potential.  L-BFGS-B still converges at zero Jacobian = force
    balance.

    ``f_n`` is **frozen** at the analytic face-on normal-equilibrium
    value during the optimisation (quasi-static normal/tangent
    decoupling).  Exact at face-on; underrates coupling on tilted
    faces where ``f_n`` drifts with ``δ_n``.

    Args:
        eps_contact: smoothing width [m] for the half-space surrogate.
                     Default 1e-9 (theory-side precision); production
                     kernel uses 5e-4.
        eps_friction: accepted for API parity; ignored (the smooth
                      friction antiderivative is exact).
        f_n_override: if not None, use this value as ``f_n`` in the
                friction surrogate instead of the analytical face-on
                value.  The kernel computes ``f_n`` live as
                ``|F_contact · n̂_pad|`` at each iterate
                (theory.md §6.4); pass an override when reproducing
                that behaviour against a non-face-on geometry.
        delta0: warm start.  None ⇒ analytical solution from
                :func:`equilibrium_half_space_friction_analytical` —
                strongly recommended for slip regimes where the
                ``F = F_thresh`` kink in the hard limit gives L-BFGS-B
                a difficult line search from a random start.
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
        # External potential: q = p − δ ⇒ work by f_ext on q is
        # W = -f_ext·δ; V_ext = +f_ext·δ.
        E_ext = +float(np.dot(f_ext_arr, d))
        return E_a + E_c + E_f + E_ext

    def jac(d: np.ndarray) -> np.ndarray:
        g = anchor_force(sphere, d)
        # Force-form contact term (theory.md eq:f-phys) with the Hertz
        # phi_eff = σ_ε·sqrt(σ_ε + eps):
        raw = half_space_raw(sphere, n_face_arr, t_sample_arr, d)
        sigma = smooth_relu(raw, eps_contact)
        phi_eff = sigma * np.sqrt(sigma + max(eps_contact, 0.0))
        gate = smooth_step(raw, eps_contact)
        g = g + kc * phi_eff * gate * n_face_arr
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
    # Active-set + alignment constants (kept in lockstep with the kernel).
    "INACTIVE_RAW_EPS_FACTOR",
    "INCLUSION_FACTOR",
    "EPS_ALIGN_DEFAULT",
    # Lattice primitive.
    "LatticeSphere",
    # Geometry helpers.
    "deformed_centre",
    "smooth_step",
    "smooth_relu",
    # Anchor (theory.md §6.1).
    "anchor_force",
    "anchor_energy",
    # Half-space contact primitives (theory.md §3-§4).
    "half_space_raw",
    "half_space_phi_eff",
    "half_space_gate",
    "half_space_force",
    "half_space_energy",
    "equilibrium_half_space_face_on_analytical",
    "equilibrium_half_space_numerical",
    # Friction (theory.md §6.4).
    "friction_force_smooth",
    "friction_energy_smooth",
    "equilibrium_half_space_friction_analytical",
    "equilibrium_half_space_friction_hard_numerical",
    "equilibrium_half_space_friction_smooth_numerical",
]
