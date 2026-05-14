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
