# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy reference implementation of the IDEAL CSLC model.

This module is the single source of truth for what the GPU kernels in
``newton/_src/geometry/cslc_kernels.py`` should converge to.  Every
function corresponds to a printed equation; the docstrings cite both
the paper section and the in-tree Overleaf transcript at
``cslc_mujoco/docs/overleaf_theory_cslc_icra.txt``.

This file currently covers ONE lattice sphere against one rigid
target.  Subsequent steps will extend it to lateral coupling, friction,
and full lattice equilibrium.

Sign conventions (matching ``cslc_kernels.jacobi_step``):

    q = p - delta            (deformed centre; delta along +n_hat
                              means the sphere is compressed INWARD,
                              i.e. toward the body interior)

    f_anchor = +k_a * delta  (restoring; pulls q back toward p)

    f_contact = +k_c * phi_eff * e_hat   where
        phi_eff = max(0, (r + R) - ||t - q||)
        e_hat   = (q - t) / ||q - t||  (unit vector FROM target TO sphere)
    so contact pushes the sphere AWAY from the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


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


@dataclass(frozen=True)
class RigidTarget:
    """A rigid sphere we contact against.

    No anchor, no compliance: it sits where it sits.  This matches the
    paper's eq. 12 with the rigid-target limit ``k_e -> inf``, where the
    series composition collapses to ``k_c`` alone.
    """

    t: np.ndarray  # Centre in world frame [m].
    R: float       # Radius [m].

    def __post_init__(self):
        if self.t.shape != (3,):
            raise ValueError(f"t must be shape (3,), got {self.t.shape}")


# ────────────────────────────────────────────────────────────────────────
#  Geometry primitives
# ────────────────────────────────────────────────────────────────────────


def deformed_centre(sphere: LatticeSphere, delta: np.ndarray) -> np.ndarray:
    """q = p - delta.

    The IDEAL CSLC formulation moves the sphere centre with delta and
    keeps the radius at ``r``.  ``cslc_kernels.write_cslc_contacts`` does
    the opposite: it keeps the centre at ``p`` and shrinks the radius to
    ``r - dot(delta, n_eff)``.  See ``effective_radius_kernel`` below for
    the kernel-style reduction we keep around for comparison.
    """
    return sphere.p - delta


def rest_overlap(sphere: LatticeSphere, target: RigidTarget) -> float:
    """phi_rest = (r + R) - ||p - t||  [m].

    Paper eq. 11 evaluated at delta = 0.  Positive when the rest spheres
    geometrically overlap, zero when they just touch, negative when
    separated.  No clamp here -- callers decide how to gate.
    """
    return (sphere.r + target.R) - float(np.linalg.norm(sphere.p - target.t))


def effective_penetration(
    sphere: LatticeSphere,
    target: RigidTarget,
    delta: np.ndarray,
    *,
    eps: float = 0.0,
) -> float:
    """phi_eff = [r + R - ||t - q||]_+   (paper eq. 11 with q = p - delta).

    With ``eps = 0`` this is the hard ``max(0, ...)``.  With ``eps > 0`` it
    is the smooth surrogate sigma_eps(x) = 0.5*(x + sqrt(x^2 + eps^2))
    from the paper III.G smoothing scheme, so we can compare against the
    kernel's smooth gates.
    """
    q = deformed_centre(sphere, delta)
    dist = float(np.linalg.norm(target.t - q))
    raw = (sphere.r + target.R) - dist
    if eps <= 0.0:
        return max(0.0, raw)
    return 0.5 * (raw + np.sqrt(raw * raw + eps * eps))


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


def contact_raw_overlap(sphere: LatticeSphere, target: RigidTarget,
                        delta: np.ndarray) -> float:
    """Raw (signed) overlap before the positive-part clamp:
        raw = (r + R) - ||t - q||,    q = p - delta.

    The smooth phi_eff is sigma_eps(raw); the gradient picks up
    smooth_step(raw, eps).  Returned separately so callers don't have
    to re-derive raw to apply the gradient factor.
    """
    q = deformed_centre(sphere, delta)
    dist = float(np.linalg.norm(target.t - q))
    return (sphere.r + target.R) - dist


def contact_direction(sphere: LatticeSphere, target: RigidTarget,
                      delta: np.ndarray) -> np.ndarray:
    """e_hat = (q - t) / ||q - t||  (unit, pointing from target to sphere).

    This is the IDEAL contact direction: the line of centres of the
    DEFORMED sphere and the target.  It tilts with tangential delta when
    the target sits off-axis.  The kernel uses the REST line of centres
    (``(t - p) / ||t - p||``) instead, so tangential delta cannot tilt the
    kernel's contact direction.
    """
    q = deformed_centre(sphere, delta)
    diff = q - target.t
    n = float(np.linalg.norm(diff))
    if n < 1e-15:
        # Degenerate: centres coincide.  Fall back to the rest outward
        # normal so callers never see NaN.
        return sphere.n.copy()
    return diff / n


# ────────────────────────────────────────────────────────────────────────
#  Forces and energy (ideal model)
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


def contact_force(
    sphere: LatticeSphere,
    target: RigidTarget,
    delta: np.ndarray,
    kc: float,
    *,
    eps: float = 0.0,
) -> np.ndarray:
    """f_contact = +k_c * phi_eff(delta) * e_hat(delta)   (paper eq. 12).

    The IDEAL spring **force** (not the energy gradient): magnitude is
    k_c times the actual deformation of the contact layer, direction
    is from target toward sphere (pushes sphere away from target).

    GRADIENT WARNING.  This is NOT the gradient of the contact
    energy when ``eps > 0``.  The energy is E = 0.5 * k_c * phi_eff^2
    with phi_eff = sigma_eps(raw), so its gradient picks up an extra
    chain-rule factor:

        dE/d delta = k_c * phi_eff * smooth_step(raw, eps) * e_hat.

    For deep saturated contact (raw >> eps) the factor is ~1 and the
    two coincide.  Near contact onset (raw ~ eps) they disagree by up
    to 2x.  Always apply ``smooth_step(raw, eps)`` when using this for
    L-BFGS-B Jacobians.  See ``contact_raw_overlap`` for raw.
    """
    phi = effective_penetration(sphere, target, delta, eps=eps)
    if phi <= 0.0 and eps <= 0.0:
        return np.zeros(3)
    e = contact_direction(sphere, target, delta)
    return kc * phi * e


def anchor_energy(sphere: LatticeSphere, delta: np.ndarray) -> float:
    """E_anchor = 0.5 * k_a * ||delta||^2   (isotropic)."""
    if sphere.ka_t_ratio == 1.0:
        return 0.5 * sphere.ka * float(np.dot(delta, delta))
    delta_n = float(np.dot(delta, sphere.n))
    delta_t_sq = float(np.dot(delta, delta)) - delta_n * delta_n
    return 0.5 * sphere.ka * delta_n * delta_n + 0.5 * (sphere.ka * sphere.ka_t_ratio) * delta_t_sq


def contact_energy(
    sphere: LatticeSphere,
    target: RigidTarget,
    delta: np.ndarray,
    kc: float,
    *,
    eps: float = 0.0,
) -> float:
    """E_contact = 0.5 * k_c * phi_eff^2   (the spring's stored energy)."""
    phi = effective_penetration(sphere, target, delta, eps=eps)
    return 0.5 * kc * phi * phi


def total_energy(
    sphere: LatticeSphere,
    target: RigidTarget,
    delta: np.ndarray,
    kc: float,
    *,
    eps: float = 0.0,
) -> float:
    """E_total = E_anchor + E_contact.

    The IDEAL quasistatic equilibrium is the minimiser of this scalar.
    The minimum exists and is unique whenever ``k_a > 0`` (anchor
    coercivity dominates the bounded contact term).
    """
    return anchor_energy(sphere, delta) + contact_energy(sphere, target, delta, kc, eps=eps)


# ────────────────────────────────────────────────────────────────────────
#  Solvers
# ────────────────────────────────────────────────────────────────────────


def equilibrium_face_on_analytical(
    sphere: LatticeSphere,
    target: RigidTarget,
    kc: float,
) -> tuple[np.ndarray, float]:
    """Closed-form equilibrium when the target lies on the rest normal.

    Implements the paper's single-sphere closed form (paper IV.A and
    derivation in cslc_main/theory's README):

        delta_n* = k_c * phi_rest / (k_a + k_c),     if phi_rest > 0
        delta*   = delta_n* * n_hat
        |F|*     = k_a * k_c * phi_rest / (k_a + k_c)

    Returns:
        (delta, force_magnitude).  ``delta`` is the full vec3
        displacement (zero tangential component by symmetry).
    """
    # Sanity-check the "face-on" assumption: target must lie along the
    # outward normal.  We don't enforce strict colinearity (numerical
    # noise is fine) but we warn if the offset has any tangential
    # component beyond a tight tolerance.
    offset = target.t - sphere.p
    offset_n = float(np.dot(offset, sphere.n))
    offset_t = offset - offset_n * sphere.n
    if float(np.linalg.norm(offset_t)) > 1e-9:
        raise ValueError(
            "equilibrium_face_on_analytical called with non-face-on geometry "
            f"(tangential offset = {np.linalg.norm(offset_t):.3e} m). "
            "Use equilibrium_numerical for off-axis configurations.")

    phi = rest_overlap(sphere, target)
    if phi <= 0.0:
        return np.zeros(3), 0.0
    delta_n = kc * phi / (sphere.ka + kc)
    F = sphere.ka * kc * phi / (sphere.ka + kc)
    return delta_n * sphere.n, F


def equilibrium_numerical(
    sphere: LatticeSphere,
    target: RigidTarget,
    kc: float,
    *,
    eps: float = 1.0e-7,
    delta0: np.ndarray | None = None,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """General-geometry equilibrium by minimising E_total over delta in R^3.

    Uses scipy's L-BFGS-B (quasi-Newton) on the smoothed energy.  We
    smooth phi_eff with a tiny eps so the gradient is well-defined at
    phi = 0; eps -> 0 recovers the exact hard model and shifts the
    optimum by at most O(eps) (see paper III.G).

    Args:
        sphere, target, kc: as elsewhere.
        eps: smoothing width [m] for the contact potential.  Default
            1e-7 m: ~10x tighter than the kernel default, so the smooth
            answer is indistinguishable from the hard one at micrometre
            precision.
        delta0: initial guess.  Default zero.  For poorly-conditioned
            problems (very high k_c / k_a), passing the analytical
            face-on solution improves L-BFGS convergence.
        tol: optimiser gradient tolerance (default 1e-12).

    Returns:
        (delta, info) where info is the scipy ``OptimizeResult`` dict.
    """
    if delta0 is None:
        delta0 = np.zeros(3)

    def fun(d: np.ndarray) -> float:
        return total_energy(sphere, target, d, kc, eps=eps)

    def jac(d: np.ndarray) -> np.ndarray:
        # dE/d delta = dE_anchor/d delta + dE_contact/d delta.
        #
        # E_anchor = (1/2) ka ||delta||^2:
        #     dE_anchor/d delta = +ka * delta = anchor_force(...) by construction.
        #
        # E_contact = (1/2) kc * phi_eff^2  with  phi_eff = sigma_eps(raw),
        #     raw = (r + R) - ||t - q||,    q = p - delta.
        # Chain rule:
        #     dE_contact/d delta = kc * phi_eff * d(phi_eff)/d delta
        #                        = kc * phi_eff * sigma_eps'(raw) * d(raw)/d delta
        #                        = kc * phi_eff * smooth_step(raw, eps) * e_hat,
        # where e_hat = (q - t)/||q - t|| is the contact direction and
        # smooth_step is the C^infinity Heaviside surrogate (derivative of
        # sigma_eps).  Forgetting smooth_step is the silent gradient bug
        # (see smooth_step docstring); it's invisible at raw >> eps but
        # up to 2x wrong at raw ~ eps.
        grad = anchor_force(sphere, d)
        # Contact: emit even when phi <= 0 if eps > 0 (the smooth surrogate
        # has a soft tail that contributes a small gradient there).
        raw = contact_raw_overlap(sphere, target, d)
        emit_contact = (raw > 0.0) or (eps > 0.0 and raw > -10.0 * eps)
        if emit_contact:
            grad = grad + (smooth_step(raw, eps)
                           * contact_force(sphere, target, d, kc, eps=eps))
        return grad

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


def equilibrium_with_friction_analytical(
    sphere: LatticeSphere,
    target: RigidTarget,
    kc: float,
    f_ext_tangent: np.ndarray,
    k_stick: float,
    mu: float,
) -> tuple[np.ndarray, dict]:
    """Closed-form equilibrium for a face-on lattice sphere with friction.

    Decouples normal and tangent axes (valid for isotropic anchor, face-on
    contact, target along +n_hat).

    Normal: same as step 1.
        delta_n = k_c * phi_rest / (k_a + k_c)
        f_n     = k_a * k_c * phi_rest / (k_a + k_c)

    Tangent: stick-slip as a function of |f_ext_t|.
        stick (|F| <= F_thresh):  s = |F| / (k_a + k_stick)
                                  F_friction = k_stick * s
        slip  (|F| >  F_thresh):  s = (|F| - mu * f_n) / k_a
                                  F_friction = mu * f_n
        F_thresh = mu * f_n * (k_a + k_stick) / k_stick

    Returns:
        (delta, info) where delta = delta_n * n_hat + delta_t * (- F / |F|).
        Sign convention for delta_t: F is in +x_t direction, q moves in
        +x_t direction (s > 0), so delta_t = -s * x_t_hat (since
        delta = p - q).  info carries the regime, threshold, and force
        magnitudes for verification.
    """
    if f_ext_tangent.shape != (3,):
        raise ValueError(f"f_ext_tangent must be (3,), got {f_ext_tangent.shape}")
    # Make sure f_ext is genuinely tangential.
    f_dot_n = float(np.dot(f_ext_tangent, sphere.n))
    if abs(f_dot_n) > 1e-9 * (np.linalg.norm(f_ext_tangent) + 1e-15):
        raise ValueError(
            f"f_ext_tangent must be perpendicular to sphere.n; "
            f"got |f.n| = {abs(f_dot_n):.3e}")

    # Normal equilibrium (step 1).
    delta_normal, F_normal = equilibrium_face_on_analytical(sphere, target, kc)
    f_n = F_normal

    # Tangent direction (unit vector along +f_ext_t; if F is zero we
    # default to some arbitrary tangent direction).
    F_mag = float(np.linalg.norm(f_ext_tangent))
    if F_mag > 0:
        f_hat = f_ext_tangent / F_mag
    else:
        # Pick any unit tangent.  Doesn't matter, s will be 0.
        f_hat = np.array([1.0, 0.0, 0.0]) - sphere.n * float(np.dot(np.array([1.0, 0.0, 0.0]), sphere.n))
        f_hat /= max(np.linalg.norm(f_hat), 1e-12)

    # Tangent dynamics use ka_t = ka * ka_t_ratio (step 6).  Anisotropic
    # anchor has the SAME normal stiffness ka (so f_n is unchanged) but a
    # softer tangent stiffness ka_t.  ka_t_ratio = 1 recovers step-4.
    ka_t = sphere.ka * sphere.ka_t_ratio

    # F_thresh is the |F_ext| at which stick gives way to slip:
    #   k_stick * s = mu * f_n  with  s = F / (ka_t + k_stick).
    # For k_stick = 0 there IS no stick spring, so the system has no
    # slip threshold (anchor alone resists) -- F_thresh = +inf and the
    # regime is "no_friction".  Mirrors the test-helper convention.
    if k_stick > 0.0 and mu > 0.0:
        F_thresh = mu * f_n * (ka_t + k_stick) / k_stick
    else:
        F_thresh = float("inf")

    if k_stick <= 0.0 or mu <= 0.0:
        regime = "no_friction"
        s = F_mag / ka_t
        F_friction = 0.0
    elif F_mag <= F_thresh:
        regime = "stick"
        s = F_mag / (ka_t + k_stick)
        F_friction = k_stick * s
    else:
        regime = "slip"
        s = (F_mag - mu * f_n) / ka_t
        F_friction = mu * f_n

    # delta_t = -s * f_hat (since q moves in +f_hat direction, delta = p - q).
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


def equilibrium_with_friction_hard_numerical(
    sphere: LatticeSphere,
    target: RigidTarget,
    kc: float,
    f_ext_tangent: np.ndarray,
    k_stick: float,
    mu: float,
) -> tuple[np.ndarray, dict]:
    """Numerical reference using the HARD piecewise friction law.

    Decouples the normal axis (analytical, step 1) from the tangent axis
    (1D scipy.optimize.minimize_scalar on the hard piecewise energy).
    Independent code path from equilibrium_with_friction_analytical, so
    agreement between the two is a real verification.

    The 1D tangent energy:

        E(s) = (1/2) k_a s^2
             + E_friction(s)
             - F * s

        E_friction(s) = (1/2) k_stick s^2,                  if s <= s_thresh
                      = (1/2) k_stick s_thresh^2
                        + mu * f_n * (s - s_thresh),         if s > s_thresh

    The break point s_thresh = mu * f_n / k_stick.  Force law is
    continuous; energy is continuous and C^1.
    """
    from scipy.optimize import minimize_scalar

    # Normal equilibrium (decoupled, step 1).
    delta_normal, F_normal = equilibrium_face_on_analytical(sphere, target, kc)
    f_n = F_normal

    F_mag = float(np.linalg.norm(f_ext_tangent))
    if F_mag > 0:
        f_hat = f_ext_tangent / F_mag
    else:
        f_hat = np.array([1.0, 0.0, 0.0])

    # Step 6: anisotropic anchor.  Tangent stiffness is ka_t = ka * ka_t_ratio
    # (ka_t_ratio = 1 recovers step-4 isotropic behaviour).
    ka_t = sphere.ka * sphere.ka_t_ratio

    if k_stick <= 0.0 or mu <= 0.0:
        # No friction; pure tangent-anchor balance.
        s_opt = F_mag / ka_t
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

        # Upper bracket: pick a generous bound that comfortably contains
        # the slip-regime minimum.
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


def equilibrium_with_friction_smooth_numerical(
    sphere: LatticeSphere,
    target: RigidTarget,
    kc: float,
    f_ext_tangent: np.ndarray,
    k_stick: float,
    mu: float,
    *,
    eps_contact: float = 1.0e-9,
    eps_friction: float = 1.0e-12,
    delta0: np.ndarray | None = None,
    tol: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """Numerical L-BFGS-B equilibrium with smooth contact + smooth friction.

    Minimises::

        E_total(delta) =
              (1/2) k_a ||delta||^2                         (anchor, isotropic)
            + (1/2) k_c sigma_eps(phi_rest - dot(delta, n))^2  (contact)
            + E_friction(||delta_t||; f_n_quasi)              (friction)
            - dot(f_ext_tangent, delta)                       (external work)

    The friction f_n is taken from the quasi-static normal equilibrium of
    step 1 (decoupled from tangent for isotropic anchor + face-on
    contact).  This decoupling holds exactly in the analytic limit; the
    smooth contact makes the gradient C^infinity but f_n stays anchored
    to the analytic value within smoothing epsilon.

    Returns:
        (delta, info) with optimisation diagnostics.
    """
    # Pre-compute f_n from the (face-on) normal equilibrium.
    _, f_n = equilibrium_face_on_analytical(sphere, target, kc)

    if delta0 is None:
        delta_ana, _ = equilibrium_with_friction_analytical(
            sphere, target, kc, f_ext_tangent, k_stick, mu)
        delta0 = delta_ana.copy()

    def fun(d: np.ndarray) -> float:
        E_a = anchor_energy(sphere, d)
        E_c = contact_energy(sphere, target, d, kc, eps=eps_contact)
        # Tangent decomposition.
        d_n = float(np.dot(d, sphere.n))
        d_t = d - d_n * sphere.n
        d_t_mag = float(np.linalg.norm(d_t))
        E_f = friction_energy_smooth(d_t_mag, f_n, k_stick, mu, eps=eps_friction)
        # External potential.  With q = p - delta the displacement of q
        # from rest is (q - p) = -delta, so work done by an external
        # force f_ext on q is W = f_ext . (-delta) = -f_ext.delta and
        # the external potential is V_ext = -W = +f_ext.delta.  The
        # previous code used  E_ext = -f_ext.delta  (a sign error that
        # propagated to a flipped delta_t in stick mode; magnitude was
        # right but direction was wrong, only invisible because the
        # test pass criteria checked |delta_t|, not the vector).
        E_ext = +float(np.dot(f_ext_tangent, d))
        return E_a + E_c + E_f + E_ext

    def jac(d: np.ndarray) -> np.ndarray:
        # Anchor.  Use anchor_force() so the anisotropic ka_t_ratio path
        # in step 6 is consistent with anchor_energy() in fun(); writing
        # `sphere.ka * d` here would be isotropic and would silently
        # disagree with the energy for ka_t_ratio != 1.
        g = anchor_force(sphere, d)
        # Contact (face-on, smooth).
        #   dE_contact/d delta = kc * phi_eff * smooth_step(raw, eps) * e_hat,
        # where e_hat = (q - t)/||q - t||.  For face-on contact (target
        # along +n_hat, sphere compressed inward) e_hat = -n_hat exactly.
        # Without smooth_step, the gradient is wrong by up to 2x at
        # raw ~ eps -- the silent contact-gradient bug.
        raw = contact_raw_overlap(sphere, target, d)
        if (raw > 0.0) or (eps_contact > 0.0 and raw > -10.0 * eps_contact):
            phi = effective_penetration(sphere, target, d, eps=eps_contact)
            step = smooth_step(raw, eps_contact)
            g = g - kc * phi * step * sphere.n
        # Friction.
        d_n = float(np.dot(d, sphere.n))
        d_t = d - d_n * sphere.n
        d_t_mag = float(np.linalg.norm(d_t))
        if d_t_mag > 1e-15:
            F_fric = friction_force_smooth(d_t_mag, f_n, k_stick, mu,
                                           eps=eps_friction)
            # Friction force magnitude is dE_f / d(d_t_mag) at the smoothed
            # form (by construction of friction_energy_smooth).  Gradient
            # contribution: dE_f / d delta = (dE_f/d|d_t|) · (d_t / |d_t|).
            g = g + F_fric * (d_t / d_t_mag)
        # External.  V_ext = +f_ext . delta  =>  grad = +f_ext (see fun()).
        g = g + f_ext_tangent
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


# ────────────────────────────────────────────────────────────────────────
#  Kernel-style law (for comparison only)
# ────────────────────────────────────────────────────────────────────────


def kernel_contact_force_n_axis(
    sphere: LatticeSphere,
    target: RigidTarget,
    delta: np.ndarray,
    kc: float,
    *,
    eps: float = 1.0e-5,
) -> float:
    """The kernel's contact-force n-axis component (for comparison).

    Reimplements ``jacobi_step``'s ``f_contact = kc * phi_rest * gate``
    (line ~466 of newton/_src/geometry/cslc_kernels.py).  Returns the
    component along ``sphere.n`` so you can plot it against the ideal
    spring law.

    This is NOT used by the solvers in this file -- it's a witness
    function, kept here so the test scripts can plot kernel vs ideal on
    the same axes.
    """
    phi_rest = rest_overlap(sphere, target)
    if phi_rest <= 0.0 and eps <= 0.0:
        return 0.0
    # n_eff in the kernel is the LINE OF CENTRES FROM REST: (t - p)/||t - p||.
    diff_rest = target.t - sphere.p
    dist_rest = float(np.linalg.norm(diff_rest))
    if dist_rest < 1e-15:
        n_eff = sphere.n.copy()
    else:
        n_eff = diff_rest / dist_rest
    delta_proj = float(np.dot(delta, n_eff))
    eff_pen = phi_rest - delta_proj
    if eps > 0.0:
        gate = 0.5 * (1.0 + eff_pen / np.sqrt(eff_pen * eff_pen + eps * eps))
    else:
        gate = 1.0 if eff_pen > 0.0 else 0.0
    # Kernel writes f_contact = kc * phi_rest * gate * n_eff.  Force
    # FROM target TO sphere is -n_eff (since n_eff points TOWARD target
    # from rest).  But the kernel's force has the OPPOSITE sign in its
    # contribution to the equilibrium -- it's applied as a load on the
    # sphere.  See cslc_kernels.py:466.  Returning the magnitude along
    # the n-axis here keeps the sign aligned with the ideal model's
    # n-axis force for comparison plotting.
    return kc * phi_rest * gate


__all__ = [
    "LatticeSphere",
    "RigidTarget",
    "deformed_centre",
    "rest_overlap",
    "effective_penetration",
    "contact_direction",
    "anchor_force",
    "contact_force",
    "anchor_energy",
    "contact_energy",
    "total_energy",
    "equilibrium_face_on_analytical",
    "equilibrium_numerical",
    "friction_force_smooth",
    "friction_energy_smooth",
    "equilibrium_with_friction_analytical",
    "equilibrium_with_friction_hard_numerical",
    "equilibrium_with_friction_smooth_numerical",
    "kernel_contact_force_n_axis",
]
