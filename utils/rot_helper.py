import numpy as np
from scipy.spatial.transform import Rotation as R
import warp as wp


def distance_between_quaternions(q1, q2, in_degrees=False):
    '''
    Formula from Jim @ https://math.stackexchange.com/questions/90081/quaternion-distance
    '''
    q1 = wp.quat(*q1)
    q2 = wp.quat(*q2)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)
    dot = np.clip(abs(dot), -1.0, 1.0)

    angle = 2.0 * np.arccos(dot)

    if in_degrees:
        return np.degrees(angle)
    return angle


def estimate_omega_from_particles(particles_q, particles_qd, masses, eps=1e-10):
    """
    Returns omega in WORLD frame, best-fit rigid body angular velocity.
    """
    N = particles_q.shape[0]
    masses = masses.astype(np.float32)

    M = masses.sum()
    com = (particles_q * masses[:, None]).sum(axis=0) / M
    vcom = (particles_qd * masses[:, None]).sum(axis=0) / M

    r = particles_q - com[None, :]
    vrel = particles_qd - vcom[None, :]

    # L = Σ m (r × vrel)
    L = np.sum(masses[:, None] * np.cross(r, vrel), axis=0)

    # I = Σ m ( (r·r)I - r r^T )
    r2 = np.einsum("ni,ni->n", r, r)
    rrT = np.einsum("ni,nj->nij", r, r)
    I = np.sum(masses[:, None, None] *
               (r2[:, None, None] * np.eye(3) - rrT), axis=0)

    omega = np.linalg.solve(I, L)
    return np.float32(omega), np.float32(L)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def integrate_omega(q, omega, dt):
    omega = np.array(omega, dtype=np.float32)
    q = np.array(q, dtype=np.float32)
    w = np.linalg.norm(omega)
    if w < 1e-12:
        return q  # no rotation

    axis = omega / w
    theta = w * dt
    dq = np.array([
        axis[0] * np.sin(theta * 0.5),
        axis[1] * np.sin(theta * 0.5),
        axis[2] * np.sin(theta * 0.5),
        np.cos(theta * 0.5)
    ])
    q_new = quat_mul(dq, q)
    return q_new / np.linalg.norm(q_new)
