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

    Args:
        grid: Warp hash grid id built from particle positions.
        particle_x: Particle positions [m], shape [particle_count].
        particle_mass: Particle masses [kg], shape [particle_count].
        particle_substrate: Substrate type per particle; value 3 denotes fluid.
        particle_fluid_phase: Fluid phase index per particle; -1 for non-fluid.
        fluid_rest_density: Rest density rho_0 per fluid phase [kg/m^3].
        fluid_smoothing_radius: Smoothing radius h per fluid phase [m].
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

    # Unilateral constraint: only active when over-dense (C_i > 0).
    c = density[i] / rho0 - wp.float32(1.0)
    if c <= wp.float32(0.0):
        lambdas[i] = wp.float32(0.0)
        return

    x_i = particle_x[i]
    grad_i_sum = wp.vec3(0.0, 0.0, 0.0)
    sum_grad_sq = wp.float32(0.0)

    # Accumulate squared gradient contributions from neighbor particles j != i.
    # grad_j C_i = (1/rho_0) * grad_W(x_i - x_j, h)
    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        r = x_i - particle_x[j]
        grad_w = spiky_gradient(r, h)
        g_j = grad_w / rho0
        sum_grad_sq += wp.dot(g_j, g_j)
        grad_i_sum += g_j

    # Self-gradient via Newton's 3rd law: grad_i C_i = -sum_{k != i} grad_k C_i.
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
    lam_i = lambdas[i]

    x_i = particle_x[i]
    delta = wp.vec3(0.0, 0.0, 0.0)

    # Reference kernel value at dq = dq_factor * h for s_corr denominator.
    dq_vec = wp.vec3(dq_factor * h, 0.0, 0.0)
    w_dq = poly6_kernel(dq_vec, h)

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if particle_substrate[j] != wp.uint8(3) or particle_fluid_phase[j] != phase:
            continue
        r = x_i - particle_x[j]
        grad_w = spiky_gradient(r, h)

        # Artificial pressure s_corr prevents tensile instability (clumping).
        w_r = poly6_kernel(r, h)
        s_corr = wp.float32(0.0)
        if w_dq > wp.float32(1.0e-12):
            ratio = w_r / w_dq
            s_corr = -k_corr * wp.pow(ratio, n_corr)

        lam_j = lambdas[j]
        delta += (lam_i + lam_j + s_corr) * grad_w

    deltas[i] = delta / rho0


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

        v_i_new = v_i + c * sum_j (v_j - v_i) * W(x_i - x_j, h) / rho_j

    Reference: Macklin and Muller 2013, "Position Based Fluids", eq. 17.

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
            delta_v += (particle_v[j] - v_i) * (w / rho_j)

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
