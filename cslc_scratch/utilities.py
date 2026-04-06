"""
utilities.py — Quaternion and rotation helpers.

Convention:  q = [w, x, y, z]  (scalar-first, Hamilton convention).
"""

import numpy as np


def quat_to_rotmat(q):
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_multiply(q1, q2):
    """Hamilton product  q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_from_axis_angle(axis, angle):
    """Create unit quaternion from axis (unit vector) and angle (radians)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])


def quat_normalize(q):
    """Re-normalise quaternion to unit length."""
    return q / (np.linalg.norm(q) + 1e-30)


def rotate_vector(q, v):
    """Rotate vector v by quaternion q.  Returns rotated vector."""
    R = quat_to_rotmat(q)
    return R @ v


def rotate_points(q, pts):
    """Rotate an (n, 3) array of points by quaternion q."""
    R = quat_to_rotmat(q)
    return (R @ pts.T).T
