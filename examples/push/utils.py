import warp as wp
import numpy as np


@wp.kernel
def apply_constant_particle_force(
        particle_force_array: wp.array(dtype=wp.vec3),
        const_f_total: wp.vec3,
        particle_mass_array: wp.array(dtype=float),
        total_mass: float,
):
    i = wp.tid()
    mass_ratio = particle_mass_array[i] / total_mass
    particle_force_array[i] = mass_ratio * const_f_total


@wp.kernel
def apply_constant_body_force(
        body_force_array: wp.array(dtype=wp.spatial_vector),
        const_f_total: wp.vec3,
):
    i = wp.tid()
    body_force_array[i] = wp.spatial_vector(const_f_total, wp.vec3(0.0))


@wp.kernel
def apply_constant_force_and_torque_body(
        body_force_array: wp.array(dtype=wp.spatial_vector),
        F_world: wp.vec3,
        tau_world: wp.vec3,
):
    body_force_array[0] = wp.spatial_vector(F_world, tau_world)


def compute_lambda_for_torque_particles(particle_positions, particle_masses, com_world, tau_world, eps=1e-8):
    '''
        Helper function to calculate lambda for applying a constant torque to a set of particles.
    '''
    r = particle_positions - com_world.reshape((1, 3))   # (N,3)
    m = particle_masses.reshape((-1, 1))                 # (N,1)
    # mass-weighted rr and ||r||^2
    rr = (m * r).T @ r                                   # Σ m r r^T   (3x3)
    M = np.sum(particle_masses * np.sum(r*r, axis=1))    # Σ m ||r||^2 scalar
    # Σ m (||r||^2 I - r r^T)
    A = M * np.eye(3) - rr
    A += eps * np.eye(3)
    return np.linalg.solve(A, tau_world)

@wp.kernel
def apply_constant_force_and_torque_particle(
        particle_force_array: wp.array(dtype=wp.vec3),
        particle_q: wp.array(dtype=wp.vec3),
        particle_mass_array: wp.array(dtype=float),
        total_mass: float,
        com_world: wp.vec3,
        F_world: wp.vec3,
        lambda_vec: wp.vec3,
):
    i = wp.tid()
    mi = particle_mass_array[i]
    mass_ratio = mi / total_mass
    r = particle_q[i] - com_world
    f_tau = mi * wp.cross(lambda_vec, r)
    particle_force_array[i] = mass_ratio * F_world + f_tau



@wp.kernel
def apply_constant_particle_force_range(
        particle_force_array: wp.array(dtype=wp.vec3),
        const_f_total: wp.vec3,
        particle_mass_array: wp.array(dtype=float),
        total_mass: float,
        start_idx: int,
):
    i = wp.tid() + start_idx
    mass_ratio = particle_mass_array[i] / total_mass
    particle_force_array[i] = mass_ratio * const_f_total



@wp.kernel
def apply_constant_force_to_shape(body_force_array: wp.array(dtype=wp.spatial_vector), const_f: wp.vec3, shape_body_idx: int):
    # Only apply force to the desired body index
    i = wp.tid()
    if i == shape_body_idx:
        body_force_array[i] = wp.spatial_vector(const_f, wp.vec3(0.0))