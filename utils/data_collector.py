import warp as wp
import numpy as np
from utils.rot_helper import distance_between_quaternions, estimate_omega_from_particles, integrate_omega
from scipy.spatial.transform import Rotation


class COLLECTOR_TYPES:
    POS_A = 'pos_analytic'
    VEL_A = 'vel_analytic'
    ROT_A = 'rot_analytic'
    OMEGA_A = 'w_analytic'
    POS_S = 'pos_solver'
    VEL_S = 'vel_solver'
    ROT_S = 'rot_solver'
    OMEGA_S = 'w_solver'
    POS_E = 'pos_err'
    VEL_E = 'vel_err'
    ROT_E = 'rot_err'
    OMEGA_E = 'w_err'
    POS_L2_E = 'pos_l2_err'
    DT = 'frame_dt'
    ANG_MOM_A = 'L_analytic'  # angular momentum
    ANG_MOM_S = 'L_solver'
    ANG_MOM_E = 'L_err'
    LIN_MOM_A = 'p_analytic'  # linear momentum
    LIN_MOM_S = 'p_solver'
    LIN_MOM_E = 'p_err'
    FORCE_S = 'force_solver'  # net force on body
    PARTICLE_FORCE_SUM = 'force_array'
    PARTICLE_TORQUE_SUM = 'tau_array'


def setup_data_collection(frame_dt):
    return {
        COLLECTOR_TYPES.POS_A: [],
        COLLECTOR_TYPES.VEL_A: [],
        COLLECTOR_TYPES.ROT_A: [],
        COLLECTOR_TYPES.OMEGA_A: [],
        COLLECTOR_TYPES.POS_S: [],
        COLLECTOR_TYPES.VEL_S: [],
        COLLECTOR_TYPES.ROT_S: [],
        COLLECTOR_TYPES.OMEGA_S: [],
        COLLECTOR_TYPES.POS_E: [],
        COLLECTOR_TYPES.POS_L2_E: [],
        COLLECTOR_TYPES.VEL_E: [],
        COLLECTOR_TYPES.ROT_E: [],
        COLLECTOR_TYPES.OMEGA_E: [],
        COLLECTOR_TYPES.ANG_MOM_A: [],
        COLLECTOR_TYPES.ANG_MOM_S: [],
        COLLECTOR_TYPES.ANG_MOM_E: [],
        COLLECTOR_TYPES.LIN_MOM_A: [],
        COLLECTOR_TYPES.LIN_MOM_S: [],
        COLLECTOR_TYPES.LIN_MOM_E: [],
        COLLECTOR_TYPES.FORCE_S: [], # TODO REMOVE PARTICLE FORCE/TORQUE SUM AND ONLY USE FORCE + TORQUE KEYS
        COLLECTOR_TYPES.PARTICLE_FORCE_SUM: [],
        COLLECTOR_TYPES.PARTICLE_TORQUE_SUM: [],
        COLLECTOR_TYPES.DT: frame_dt,
    }


def collect_data_from_spheres(pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, p_analytic,
                              particles_q, particles_qd, particle_m, prev_rot_solver, dt, particle_f=None, particle_indices=None):
    '''
    This function  assumes that all particles belong to a single rigid body
    particle_f: optional force array for particles (shape: [n_particles, 3])
    particle_indices: optional array of particle indices to sum forces over (if particle_f is provided)
    '''
    # TODO: Remove particle indices + particle_f must not have a default and must be always provided. 
    M = particle_m.sum()
    com_pos = np.sum(particles_q * particle_m[:, None], axis=0) / M
    com_vel = np.sum(particles_qd * particle_m[:, None], axis=0) / M
    pos_solver = com_pos
    vel_solver = com_vel
    p_solver = M * com_vel
    omega_solver,  L_solver = estimate_omega_from_particles(
        particles_q, particles_qd, particle_m)
    rot_solver = integrate_omega(prev_rot_solver, omega_solver, dt)
    
    # Calculate net force if force data is provided
    force_solver = np.array([0.0, 0.0, 0.0])
    if particle_f is not None and particle_indices is not None:
        force_solver = np.sum(particle_f[particle_indices], axis=0)

    result = {
        COLLECTOR_TYPES.POS_A: pos_analytic,
        COLLECTOR_TYPES.VEL_A: vel_analytic,
        COLLECTOR_TYPES.ROT_A: rot_analytic,
        COLLECTOR_TYPES.OMEGA_A: omega_analytic,
        COLLECTOR_TYPES.ANG_MOM_A: L_analytic,
        COLLECTOR_TYPES.LIN_MOM_A: p_analytic,

        COLLECTOR_TYPES.POS_S: pos_solver,
        COLLECTOR_TYPES.VEL_S: vel_solver,
        COLLECTOR_TYPES.ROT_S: rot_solver,
        COLLECTOR_TYPES.OMEGA_S: omega_solver,
        COLLECTOR_TYPES.ANG_MOM_S: L_solver,
        COLLECTOR_TYPES.LIN_MOM_S: p_solver,
        COLLECTOR_TYPES.FORCE_S: force_solver,

        COLLECTOR_TYPES.POS_E: pos_analytic - pos_solver,
        COLLECTOR_TYPES.POS_L2_E: np.linalg.norm(pos_analytic - pos_solver),
        COLLECTOR_TYPES.VEL_E: vel_analytic - vel_solver,
        COLLECTOR_TYPES.LIN_MOM_E: p_analytic - p_solver,
        COLLECTOR_TYPES.OMEGA_E: omega_analytic - omega_solver,
        COLLECTOR_TYPES.ANG_MOM_E: L_analytic - L_solver,
        # note this is overwritten by the unwrapped version in recompute_rot_error_unwrapped, but we keep it here for the initial error before unwrapping
        COLLECTOR_TYPES.ROT_E: distance_between_quaternions(rot_analytic, rot_solver, in_degrees=True),
    }
    return result


def collect_data_from_bodies(pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, p_analytic, body_q, body_qd, mass, I, body_idx, body_f=None):
    # TODO: body_f must not have a default and must be always provided. 
    pos_solver = body_q[body_idx][:3]
    rot_solver = body_q[body_idx][3:]
    vel_solver = body_qd[body_idx][:3]
    omega_solver = body_qd[body_idx][3:].reshape(3, )
    L_solver = I @ omega_solver
    p_solver = mass * vel_solver
    
    # Extract net force from body_f (spatial_vector has linear force component first)
    force_solver = np.array([0.0, 0.0, 0.0])
    if body_f is not None:
        # body_f is a spatial_vector array where the linear force is the first 3 components
        force_solver = np.asarray(body_f[body_idx][:3])

    return {
        COLLECTOR_TYPES.POS_A: pos_analytic,
        COLLECTOR_TYPES.VEL_A: vel_analytic,
        COLLECTOR_TYPES.ROT_A: rot_analytic,
        COLLECTOR_TYPES.OMEGA_A: omega_analytic,
        COLLECTOR_TYPES.ANG_MOM_A: L_analytic,
        COLLECTOR_TYPES.LIN_MOM_A: p_analytic,

        COLLECTOR_TYPES.POS_S: pos_solver,
        COLLECTOR_TYPES.VEL_S: vel_solver,
        COLLECTOR_TYPES.ROT_S: rot_solver,
        COLLECTOR_TYPES.OMEGA_S: omega_solver,
        COLLECTOR_TYPES.ANG_MOM_S: L_solver,
        COLLECTOR_TYPES.LIN_MOM_S: p_solver,

        COLLECTOR_TYPES.FORCE_S: force_solver,
        COLLECTOR_TYPES.POS_E: pos_analytic - pos_solver,
        COLLECTOR_TYPES.POS_L2_E: np.linalg.norm(pos_analytic - pos_solver),
        COLLECTOR_TYPES.VEL_E: vel_analytic - vel_solver,
        COLLECTOR_TYPES.LIN_MOM_E: p_analytic - p_solver,
        COLLECTOR_TYPES.OMEGA_E: omega_analytic - omega_solver,
        COLLECTOR_TYPES.ANG_MOM_E: L_analytic - L_solver,
        # note this is overwritten by the unwrapped version in recompute_rot_error_unwrapped, but we keep it here for the initial error before unwrapping
        COLLECTOR_TYPES.ROT_E: distance_between_quaternions(
            rot_analytic, rot_solver, in_degrees=True)
    }


def recompute_rot_error_unwrapped(collected_data, custom_rot_keys:list=None):
    """Call this after all frames are collected, before saving.
    
    Args:
        collected_data: Dictionary of collected data
        custom_rot_keys: List of tuples (rot_a_key, rot_s_key, rot_e_key) for custom rotation keys.
            If None, uses standard COLLECTOR_TYPES keys.
    """
    if custom_rot_keys is not None: # TODO Need a cleaner way to handle custom keys
        # Process custom rotation keys for experiments with multiple tracked objects
        for rot_a_key, rot_s_key, rot_e_key in custom_rot_keys:
            if rot_a_key not in collected_data or rot_s_key not in collected_data:
                continue
            
            rot_a = np.array(collected_data[rot_a_key])  # (N, 4)
            rot_s = np.array(collected_data[rot_s_key])  # (N, 4)
            
            # scipy expects (x, y, z, w), same as warp
            yaw_a = np.unwrap(Rotation.from_quat(rot_a).as_euler('ZYX')[:, 0])
            yaw_s = np.unwrap(Rotation.from_quat(rot_s).as_euler('ZYX')[:, 0])
            
            collected_data[rot_e_key] = list(
                np.degrees(yaw_a - yaw_s).reshape(-1, 1))
    else:
        # Standard processing for default keys
        rot_a = np.array(collected_data[COLLECTOR_TYPES.ROT_A])  # (N, 4)
        rot_s = np.array(collected_data[COLLECTOR_TYPES.ROT_S])  # (N, 4)

        # scipy expects (x, y, z, w), same as warp
        yaw_a = np.unwrap(Rotation.from_quat(rot_a).as_euler('ZYX')[:, 0])
        yaw_s = np.unwrap(Rotation.from_quat(rot_s).as_euler('ZYX')[:, 0])

        collected_data[COLLECTOR_TYPES.ROT_E] = list(
            np.degrees(yaw_a - yaw_s).reshape(-1, 1))
