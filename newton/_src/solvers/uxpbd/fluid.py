# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-Based Fluids kernels (Macklin and Muller 2013).

Implements the density constraint C_i = rho_i / rho_0 - 1 <= 0 with
fluid-solid coupling per UPPFRTA section 7.1 eq. 27. Provides separate
kernels for density computation, lambda (Lagrange multiplier), position
deltas with artificial pressure correction, XSPH viscosity, and Akinci
cohesion.

Notes:
    SI units throughout. Smoothing radius h is per fluid phase. Mass is
    per particle. The Poly6 kernel is used for density and cohesion; the
    Spiky kernel gradient is used for position deltas.
"""

from __future__ import annotations

import warp as wp

PI = wp.float32(3.14159265358979)


@wp.func
def poly6_kernel(r: wp.vec3, h: wp.float32) -> wp.float32:
    """Poly6 SPH smoothing kernel W(r, h).

    W(r, h) = 315 / (64 * pi * h^9) * (h^2 - |r|^2)^3,  |r| < h
              0,                                            otherwise

    Reference: Muller et al. 2003, "Particle-Based Fluid Simulation for
    Interactive Applications", eq. 20.

    Args:
        r: Displacement vector from neighbor j to particle i [m].
        h: Smoothing radius [m].

    Returns:
        Kernel value [1/m^3].
    """
    r2 = wp.dot(r, r)
    h2 = h * h
    if r2 >= h2:
        return wp.float32(0.0)
    coeff = wp.float32(315.0) / (wp.float32(64.0) * PI * wp.pow(h, wp.float32(9.0)))
    return coeff * wp.pow(h2 - r2, wp.float32(3.0))


@wp.func
def spiky_gradient(r: wp.vec3, h: wp.float32) -> wp.vec3:
    """Gradient of the Spiky SPH kernel grad W(r, h).

    grad W(r, h) = -45 / (pi * h^6) * (h - |r|)^2 * r_hat,  0 < |r| < h
                   0,                                          otherwise

    Reference: Muller et al. 2003, "Particle-Based Fluid Simulation for
    Interactive Applications", eq. 21. Used by Macklin and Muller 2013
    for the PBF position delta to avoid clustering artefacts of Poly6.

    Args:
        r: Displacement vector from neighbor j to particle i [m].
        h: Smoothing radius [m].

    Returns:
        Kernel gradient [1/m^4].
    """
    rl = wp.length(r)
    if rl >= h or rl < wp.float32(1.0e-9):
        return wp.vec3(0.0, 0.0, 0.0)
    coeff = wp.float32(-45.0) / (PI * wp.pow(h, wp.float32(6.0)))
    return (coeff * (h - rl) * (h - rl)) * (r / rl)


@wp.kernel
def compute_fluid_density(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_solid_coupling_s: wp.array[wp.float32],
    # output
    density: wp.array[wp.float32],
):
    """Compute per-particle SPH density rho_i via Poly6 kernel.

    Iterates over all particles within the smoothing radius h of fluid
    particle i using the hash grid. Fluid neighbors in the same phase
    contribute with their full mass; solid neighbors (substrate != 3)
    contribute with mass scaled by the per-phase coupling factor s per
    UPPFRTA section 7.1 eq. 27:

        rho_i = sum_{j in fluid} m_j * W(x_i - x_j, h)
              + s * sum_{j in solid} m_j * W(x_i - x_j, h)

    Args:
        grid: Warp hash grid id built from particle positions.
        particle_x: Particle positions [m], shape [particle_count].
        particle_mass: Particle masses [kg], shape [particle_count].
        particle_substrate: Substrate type per particle; value 3 denotes fluid.
        particle_fluid_phase: Fluid phase index per particle; -1 for non-fluid.
        fluid_smoothing_radius: Smoothing radius h per fluid phase [m].
        fluid_solid_coupling_s: Solid density contribution scale s per phase.
        density: Output density array [kg/m^3], shape [particle_count].
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        return
    h = fluid_smoothing_radius[phase]
    s_scale = fluid_solid_coupling_s[phase]

    x_i = particle_x[i]
    rho = wp.float32(0.0)

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        r = x_i - particle_x[j]
        w = poly6_kernel(r, h)
        if particle_substrate[j] == wp.uint8(3) and particle_fluid_phase[j] == phase:
            # Fluid-fluid contribution: full mass weight.
            rho += particle_mass[j] * w
        elif particle_substrate[j] != wp.uint8(3):
            # Fluid-solid contribution: scaled by coupling factor s.
            rho += s_scale * particle_mass[j] * w
    density[i] = rho


@wp.kernel
def compute_fluid_lambda(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_rest_density: wp.array[wp.float32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_solid_coupling_s: wp.array[wp.float32],
    density: wp.array[wp.float32],
    epsilon: wp.float32,
    # output
    lambdas: wp.array[wp.float32],
):
    """Compute per-fluid-particle Lagrange multiplier lambda.

    For the unilateral density constraint C_i = rho_i / rho_0 - 1 (active
    only when rho_i > rho_0):

        lambda_i = -C_i / (sum_k |grad_k C_i|^2 + epsilon)

    Self-gradient grad_i C_i = -sum_{k != i} grad_k C_i (Newton's 3rd law
    for the kernel). epsilon ~ 100 is Macklin and Muller's relaxation.

    Reference: Macklin and Muller 2013, "Position Based Fluids", eq. 11.
    Solid neighbors are scaled by the UPPFRTA eq.27 coupling factor s, matching
    the same s factor applied in :func:`compute_fluid_density`. Without this
    symmetry, lambda's numerator (C, from s-scaled density) and denominator
    (sum_grad_sq, from un-scaled gradient) used inconsistent definitions of
    rho_i, which inflated lambda at fluid-solid contact boundaries and drove
    fluid-particle launches when heavy SM-rigid particles impacted the pool.

    Args:
        grid: Warp hash grid id built from particle positions.
        particle_x: Particle positions [m], shape [particle_count].
        particle_mass: Particle masses [kg], shape [particle_count].
        particle_substrate: Substrate type per particle; value 3 denotes fluid.
        particle_fluid_phase: Fluid phase index per particle; -1 for non-fluid.
        fluid_rest_density: Rest density rho_0 per fluid phase [kg/m^3].
        fluid_smoothing_radius: Smoothing radius h per fluid phase [m].
        fluid_solid_coupling_s: Solid density contribution scale s per phase
            (UPPFRTA eq.27). Applied to solid neighbors' gradients so the
            constraint Jacobian matches the density formulation.
        density: Current density rho_i per particle [kg/m^3].
        epsilon: Relaxation regularizer to prevent division by zero [dimensionless].
        lambdas: Output Lagrange multiplier per particle [dimensionless].
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        lambdas[i] = wp.float32(0.0)
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        lambdas[i] = wp.float32(0.0)
        return

    rho0 = fluid_rest_density[phase]
    h = fluid_smoothing_radius[phase]
    s_scale = fluid_solid_coupling_s[phase]

    # Unilateral constraint: only active when over-dense (C_i > 0).
    c = density[i] / rho0 - wp.float32(1.0)
    if c <= wp.float32(0.0):
        lambdas[i] = wp.float32(0.0)
        return

    x_i = particle_x[i]
    grad_i_sum = wp.vec3(0.0, 0.0, 0.0)
    sum_grad_sq = wp.float32(0.0)

    # Accumulate squared gradient contributions from neighbor particles j != i.
    # For the mass-weighted density rho_i = sum_j m_j W(x_i - x_j, h), the
    # constraint gradient is grad_p_j C_i = (m_j / rho_0) * grad_W. Macklin &
    # Muller 2013 derive this with unit mass and drop the m_j factor; for
    # variable-mass particles the m_j must be carried, otherwise both lambda
    # and the position delta are wrong by a factor of m_j (which can be very
    # small in SI units, e.g. 4e-3 kg for a 1 cm^3 water particle).
    #
    # Solid neighbors are scaled by s_scale to keep the gradient consistent
    # with the s-scaled density contribution from the same neighbors (UPPFRTA
    # eq.27). Skipping this scaling inflated lambda at solid-fluid contact.
    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        r = x_i - particle_x[j]
        grad_w = spiky_gradient(r, h)
        m_eff = particle_mass[j]
        if particle_substrate[j] != wp.uint8(3) or particle_fluid_phase[j] != phase:
            # Non-fluid (lattice / SM-rigid / static-rigid) or different-phase
            # fluid: scale by the coupling factor s to match compute_fluid_density.
            m_eff = m_eff * s_scale
        g_j = m_eff * grad_w / rho0
        sum_grad_sq += wp.dot(g_j, g_j)
        grad_i_sum += g_j

    # Self-gradient via Newton's 3rd law: grad_i C_i = -sum_{k != i} grad_k C_i.
    grad_i = -grad_i_sum
    sum_grad_sq += wp.dot(grad_i, grad_i)

    lambdas[i] = -c / (sum_grad_sq + epsilon)


@wp.kernel
def compute_fluid_density_and_lambda(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_rest_density: wp.array[wp.float32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_solid_coupling_s: wp.array[wp.float32],
    epsilon: wp.float32,
    # outputs
    density: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
):
    """Fused density + lambda computation for PBF (perf opportunity #B).

    Mathematically equivalent to running :func:`compute_fluid_density`
    followed by :func:`compute_fluid_lambda`. lambda_i depends only on
    rho_i and neighbor positions (NOT on neighbor densities), so density
    can be computed inline within the lambda kernel and the global
    density read in the second pass is eliminated. Saves one neighbor
    traversal per fluid particle per PBF sub-iteration (3 traversals -> 2;
    position-delta still requires a separate kernel because it reads all
    lambda_j after they are written).

    Density follows :func:`compute_fluid_density` (UPPFRTA eq.27 with
    s-scaled solid contributions). Lambda follows
    :func:`compute_fluid_lambda` (Macklin and Muller 2013 eq.11 with the
    same s-scaling on solid neighbor gradients for Jacobian symmetry).
    Both outputs are still written globally: density for the downstream
    XSPH viscosity pass and lambda for compute_fluid_position_delta.

    Args:
        grid: Warp hash grid id built from particle positions.
        particle_x: Particle positions [m].
        particle_mass: Particle masses [kg].
        particle_substrate: Substrate type; value 3 denotes fluid.
        particle_fluid_phase: Fluid phase index per particle; -1 for non-fluid.
        fluid_rest_density: Rest density rho_0 per fluid phase [kg/m^3].
        fluid_smoothing_radius: Smoothing radius h per fluid phase [m].
        fluid_solid_coupling_s: Solid density coupling factor per phase
            (UPPFRTA eq.27).
        epsilon: Macklin & Muller relaxation regularizer.
        density: Output density rho_i per particle [kg/m^3].
        lambdas: Output Lagrange multiplier per particle [dimensionless].
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        # Non-fluid: density not meaningful, lambda zero.
        lambdas[i] = wp.float32(0.0)
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        lambdas[i] = wp.float32(0.0)
        return

    rho0 = fluid_rest_density[phase]
    h = fluid_smoothing_radius[phase]
    s_scale = fluid_solid_coupling_s[phase]

    x_i = particle_x[i]
    rho = wp.float32(0.0)
    grad_i_sum = wp.vec3(0.0, 0.0, 0.0)
    sum_grad_sq = wp.float32(0.0)

    # Single neighbor traversal: accumulate density (Poly6 W) and gradient
    # contributions (Spiky gradient) simultaneously. Self contributes to
    # density (W at r=0 is finite) but not to gradient (Spiky gradient is
    # zero at r=0; skip the self term explicitly to match the unfused
    # kernels' behavior).
    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        r = x_i - particle_x[j]
        w = poly6_kernel(r, h)
        if particle_substrate[j] == wp.uint8(3) and particle_fluid_phase[j] == phase:
            # Same-phase fluid: full mass for both density and gradient.
            rho += particle_mass[j] * w
            if j != i:
                grad_w = spiky_gradient(r, h)
                g_j = particle_mass[j] * grad_w / rho0
                sum_grad_sq += wp.dot(g_j, g_j)
                grad_i_sum += g_j
        elif particle_substrate[j] != wp.uint8(3):
            # Non-fluid (solid) neighbor: s-scaled contributions to both
            # density and gradient, per UPPFRTA eq.27.
            rho += s_scale * particle_mass[j] * w
            grad_w = spiky_gradient(r, h)
            g_j = (s_scale * particle_mass[j]) * grad_w / rho0
            sum_grad_sq += wp.dot(g_j, g_j)
            grad_i_sum += g_j

    density[i] = rho

    # Unilateral constraint: C = rho/rho_0 - 1, active only when overdense.
    c = rho / rho0 - wp.float32(1.0)
    if c <= wp.float32(0.0):
        lambdas[i] = wp.float32(0.0)
        return

    # Self-gradient closes the gradient sum via Newton's 3rd law.
    grad_i = -grad_i_sum
    sum_grad_sq += wp.dot(grad_i, grad_i)

    lambdas[i] = -c / (sum_grad_sq + epsilon)


@wp.kernel
def compute_fluid_position_delta(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_rest_density: wp.array[wp.float32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_solid_coupling_s: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    k_corr: wp.float32,
    dq_factor: wp.float32,
    n_corr: wp.float32,
    # output
    deltas: wp.array[wp.vec3],
):
    """Per-fluid-particle position delta.

        delta_i = (1/rho_0) * sum_j (lambda_i + lambda_j + s_corr) * grad W(x_i - x_j, h)

    s_corr is Macklin-Muller's artificial pressure for tensile-instability
    correction: s_corr = -k_corr * (W(r, h) / W(dq, h))^n_corr, with
    dq = dq_factor * h.

    Reference: Macklin and Muller 2013, "Position Based Fluids", eq. 13.

    Args:
        grid: Warp hash grid id built from particle positions.
        particle_x: Particle positions [m], shape [particle_count].
        particle_mass: Particle masses [kg], shape [particle_count].
        particle_substrate: Substrate type per particle; value 3 denotes fluid.
        particle_fluid_phase: Fluid phase index per particle; -1 for non-fluid.
        fluid_rest_density: Rest density rho_0 per fluid phase [kg/m^3].
        fluid_smoothing_radius: Smoothing radius h per fluid phase [m].
        fluid_solid_coupling_s: Solid density contribution scale s per phase
            (UPPFRTA eq.27). Applied to solid neighbors' position-delta terms
            so the fluid particle is pushed back when an SM-rigid or lattice
            particle penetrates its smoothing-kernel support.
        lambdas: Lagrange multipliers from compute_fluid_lambda [dimensionless].
        k_corr: Artificial pressure magnitude coefficient [dimensionless].
        dq_factor: Reference distance as fraction of h for s_corr [dimensionless].
        n_corr: Exponent for artificial pressure kernel ratio [dimensionless].
        deltas: Output position corrections [m], shape [particle_count].
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        return

    rho0 = fluid_rest_density[phase]
    h = fluid_smoothing_radius[phase]
    s_scale = fluid_solid_coupling_s[phase]
    lam_i = lambdas[i]

    x_i = particle_x[i]
    delta = wp.vec3(0.0, 0.0, 0.0)

    # Reference kernel value at dq = dq_factor * h for s_corr denominator.
    dq_vec = wp.vec3(dq_factor * h, 0.0, 0.0)
    w_dq = poly6_kernel(dq_vec, h)

    # Macklin's artificial pressure s_corr is meant to fix *tensile*
    # instability (the spiky gradient pulls particles into clumps when
    # they are already over-dense). With Macklin & Muller's BILATERAL
    # density constraint (rho_i / rho_0 - 1 = 0), s_corr always fires
    # because lambda is always non-zero. UPPFRTA section 7 (and our
    # compute_fluid_lambda) clamps to a UNILATERAL constraint that is
    # inactive when rho_i < rho_0 (lam_i == 0). In that regime there
    # is no clumping force to counter, so firing s_corr just pumps
    # spurious outward velocity each iteration.
    #
    # SECOND failure mode: even when lam_i is barely-active (e.g.
    # 1e-8 from a slightly over-dense moment), the s_corr term scales
    # as (W(r,h)/W(dq,h))^n_corr * |grad_W|, both of which diverge
    # at near-overlap (r -> 0). With m_j weighting in SI water (m
    # ~= 4e-3 kg) and Spiky |grad_W| ~ 4e+7 at r ~= 1 mm, a single
    # neighbour pair generates a position correction on the order of
    # tens of metres per pass, even though lambda is 1e-8.
    #
    # Gate on `lam_i < threshold` rather than just `lam_i < 0` so
    # numerically-tiny lambdas (from particles right at the unilateral
    # threshold) do not trigger s_corr. Threshold chosen so lam_i must
    # represent at least ~1% over-density (C >= 0.01).
    s_corr_active = lam_i < wp.float32(-1.0e-5)

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        same_phase_fluid = (particle_substrate[j] == wp.uint8(3)
                            and particle_fluid_phase[j] == phase)
        # Skip neighbors that contribute nothing to fluid density (different
        # fluid phase, or substrate not handled by Eq.27). Solid neighbors
        # (lattice / SM-rigid / static) DO contribute via the s-scaled term.
        # Different-phase fluid is also folded into the solid path so a
        # heavy phase pushes a lighter one through the same s coupling.
        r = x_i - particle_x[j]
        grad_w = spiky_gradient(r, h)

        # Artificial pressure s_corr prevents tensile instability (clumping).
        # Only fire when this particle is over-dense (lam_i < 0); see comment
        # at s_corr_active definition above. Solid neighbors carry no
        # tensile-instability contribution: lam_j = 0 for them (compute_fluid_lambda
        # returns 0 for non-fluid particles), and s_corr is a fluid-self-clumping
        # fix that does not apply to a fluid-solid pair. Restricting s_corr to
        # same-phase fluid neighbors avoids amplifying the impact pressure.
        s_corr = wp.float32(0.0)
        if same_phase_fluid and s_corr_active and w_dq > wp.float32(1.0e-12):
            w_r = poly6_kernel(r, h)
            ratio = w_r / w_dq
            s_corr = -k_corr * wp.pow(ratio, n_corr)

        # m_j factor: textbook PBD derivation says m cancels here for uniform
        # mass (since w_i = 1/m and grad_p_i C_j = (m/rho_0) grad_W gives
        # w_i * grad_p_i C_j = grad_W / rho_0). BUT: Macklin & Muller 2013
        # use a fixed regularizer epsilon = 100 calibrated for normalized
        # units (m = rho_0 = 1). In SI units with water-density particles
        # (m_j ~= 4e-3 kg), sum_grad_sq >> epsilon, so the regularizer is
        # ineffective and lambda stays undamped. The Macklin position-delta
        # formula then produces meter-scale corrections per neighbor.
        #
        # Multiplying by m_j here mass-scales the position correction to
        # millimeter scale in SI, matching the magnitudes Macklin gets in
        # his normalized examples. For SOLID neighbors, the mass is further
        # scaled by s (UPPFRTA eq.27) so the position-delta Jacobian matches
        # the density formulation and compute_fluid_lambda. Without the s
        # symmetry, a heavy SM-rigid particle would push a fluid particle
        # disproportionately and a single impact pass would saturate the
        # per-pass clamp, launching the fluid particle at clamp/dt m/s.
        if same_phase_fluid:
            m_eff = particle_mass[j]
            lam_j = lambdas[j]
        else:
            m_eff = particle_mass[j] * s_scale
            # lam_j for solid neighbors is 0 (compute_fluid_lambda gates on
            # substrate == 3). Read it anyway for code symmetry; this is
            # semantically just adding 0.
            lam_j = lambdas[j]
        delta += (lam_i + lam_j + s_corr) * m_eff * grad_w

    delta_out = delta / rho0

    # Clamp the per-iteration position correction. PBF + Spiky gradient
    # diverges at small r (|grad_W| ~ (h-r)^2 / h^6 prefactor blows up as
    # r -> 0), so when two particles approach overlap, a single iteration
    # can ask for METRES of correction. We clamp the per-pass move to a
    # small fraction of the smoothing radius.
    #
    # Why 0.02 * h (not Macklin & Muller's looser 0.1 * h): with the velocity
    # update v_new = v_pred + dx/dt (srxpbd apply_particle_deltas), each
    # saturated position-delta pass injects dx/dt into the particle velocity.
    # At dt ~ 1e-3 s, h = 0.024 m: 0.1*h/dt = 2.4 m/s vs 0.02*h/dt = 0.48 m/s
    # per pass. With ~24 PBF passes per substep and a heavy SM-rigid neighbor
    # driving the delta direction consistently outward, the looser clamp let
    # fluid particles reach 10+ m/s at cross-substrate impact. The 0.02
    # factor caps per-substep injection at a few m/s even in the worst case
    # and matches Akinci 2012 DFSPH's stricter clamp recommendation.
    h_clamp = wp.float32(0.02) * h
    d_mag = wp.length(delta_out)
    if d_mag > h_clamp:
        delta_out = delta_out * (h_clamp / d_mag)

    deltas[i] = delta_out


@wp.kernel
def apply_xsph_viscosity(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_viscosity: wp.array[wp.float32],
    density: wp.array[wp.float32],
    # output
    new_v: wp.array[wp.vec3],
):
    """XSPH viscosity smoothing on fluid particles.

    Smooths velocities between neighbors once per main iteration after the
    PBF sub-iteration loop:

        v_i_new = v_i + c * sum_j m_j * (v_j - v_i) * W(x_i - x_j, h) / rho_j

    The m_j factor is the SPH "particle volume" V_j = m_j / rho_j. Macklin
    and Muller 2013 eq. 17 omits it because their derivation uses unit
    particle mass; for SI water-density particles (m_j ~= 4e-3 kg) the
    omission turns the kernel from a damping operator into an amplifier
    (per-neighbor gain ~ c * W / rho_j ~= 1 instead of c * V_j ~= 1e-3),
    which causes velocity differences to OVER-shoot and flip sign each
    pass. Multiplying by m_j is the same calibration applied to the PBF
    constraint gradients in compute_fluid_lambda /
    compute_fluid_position_delta and recovers the textbook XSPH operator.

    Reference: Macklin and Muller 2013, "Position Based Fluids", eq. 17;
    Monaghan 1989 / 2005 SPH review for the variable-mass form.

    Args:
        grid: Warp hash grid id built from particle positions.
        particle_x: Particle positions [m], shape [particle_count].
        particle_v: Particle velocities [m/s], shape [particle_count].
        particle_mass: Particle masses [kg], shape [particle_count].
        particle_substrate: Substrate type per particle; value 3 denotes fluid.
        particle_fluid_phase: Fluid phase index per particle; -1 for non-fluid.
        fluid_smoothing_radius: Smoothing radius h per fluid phase [m].
        fluid_viscosity: XSPH viscosity coefficient c per fluid phase [dimensionless].
        density: Current density rho per particle [kg/m^3].
        new_v: Output smoothed velocity per particle [m/s], shape [particle_count].
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        new_v[i] = particle_v[i]
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        new_v[i] = particle_v[i]
        return

    h = fluid_smoothing_radius[phase]
    c = fluid_viscosity[phase]
    if c <= wp.float32(0.0):
        new_v[i] = particle_v[i]
        return

    x_i = particle_x[i]
    v_i = particle_v[i]
    delta_v = wp.vec3(0.0, 0.0, 0.0)

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if particle_substrate[j] != wp.uint8(3) or particle_fluid_phase[j] != phase:
            continue
        r = x_i - particle_x[j]
        w = poly6_kernel(r, h)
        rho_j = density[j]
        if rho_j > wp.float32(1.0e-12):
            # m_j * W / rho_j == V_j (particle volume); see docstring.
            delta_v += (particle_v[j] - v_i) * (particle_mass[j] * w / rho_j)

    new_v[i] = v_i + c * delta_v


@wp.func
def akinci_cohesion_kernel(r_len: wp.float32, h: wp.float32) -> wp.float32:
    """Akinci 2013 cohesion support function C(r, h).

    C(r, h) = 32 / (pi * h^9) * (h - r)^3 * r^3,              h/2 < r < h
              32 / (pi * h^9) * (2*(h-r)^3*r^3 - h^6/64),     0  < r <= h/2
              0,                                                 otherwise

    Reference: Akinci et al. 2013, "Versatile Surface Tension and Adhesion for
    SPH Fluids", eq. 2.

    Args:
        r_len: Distance between particles |r| [m].
        h: Smoothing radius [m].

    Returns:
        Kernel value [1/m^6] (unnormalized cohesion weight).
    """
    if r_len <= wp.float32(0.0) or r_len >= h:
        return wp.float32(0.0)
    coeff = wp.float32(32.0) / (PI * wp.pow(h, wp.float32(9.0)))
    base = wp.pow(h - r_len, wp.float32(3.0)) * wp.pow(r_len, wp.float32(3.0))
    if wp.float32(2.0) * r_len > h:
        return coeff * base
    h6 = wp.pow(h, wp.float32(6.0))
    return coeff * (wp.float32(2.0) * base - h6 / wp.float32(64.0))


@wp.kernel
def apply_cohesion_forces(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_cohesion: wp.array[wp.float32],
    # output
    particle_f: wp.array[wp.vec3],
):
    """Akinci 2013 cohesion forces between same-phase fluid neighbors.

    f_ij = -kc * m_i * m_j * C(|r|, h) * (r / |r|)

    Accumulates into particle_f via atomic_add (composes with other
    external-force sources).

    Reference: Akinci et al. 2013, "Versatile Surface Tension and Adhesion for
    SPH Fluids", eq. 1.

    Args:
        grid: Warp hash grid id built from particle positions.
        particle_x: Particle positions [m], shape [particle_count].
        particle_mass: Particle masses [kg], shape [particle_count].
        particle_substrate: Substrate type per particle; value 3 denotes fluid.
        particle_fluid_phase: Fluid phase index per particle; -1 for non-fluid.
        fluid_smoothing_radius: Smoothing radius h per fluid phase [m].
        fluid_cohesion: Cohesion coefficient kc per fluid phase [N/kg^2].
        particle_f: Output force accumulator per particle [N], shape [particle_count].
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        return

    h = fluid_smoothing_radius[phase]
    kc = fluid_cohesion[phase]
    if kc <= wp.float32(0.0):
        return

    x_i = particle_x[i]
    m_i = particle_mass[i]

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if particle_substrate[j] != wp.uint8(3) or particle_fluid_phase[j] != phase:
            continue
        r = x_i - particle_x[j]
        r_len = wp.length(r)
        if r_len < wp.float32(1.0e-9):
            continue
        c_val = akinci_cohesion_kernel(r_len, h)
        if c_val <= wp.float32(0.0):
            continue
        m_j = particle_mass[j]
        f_dir = -r / r_len
        f_mag = kc * m_i * m_j * c_val
        wp.atomic_add(particle_f, i, f_mag * f_dir)
