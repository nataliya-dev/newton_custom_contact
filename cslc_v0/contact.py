"""
contact.py — Vectorised contact force computation.
"""

import numpy as np
from config import K_CONTACT, D_CONTACT, FRICTION_EPS, MU
from utilities import quat_to_rotmat


def sphere_to_sphere_forces(finger_pos, finger_r,
                            object_pos, object_r,
                            object_vel, object_omega, object_com,
                            finger_vel=None, mu=None):
    """
    Vectorised sphere-to-sphere contact forces.

    Returns (f_on_finger, f_on_object, tau_on_object, info_dict).
    """
    mu_val = MU if mu is None else mu
    nf = len(finger_pos)

    diff = finger_pos[:, None, :] - object_pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, 1e-12)
    pen = (finger_r + object_r) - dist

    if not np.any(pen > 0):
        return (np.zeros((nf, 3)), np.zeros(3), np.zeros(3),
                {'n_contacts': 0, 'total_normal': 0.0})

    n_hat = diff / dist[:, :, None]

    r_from_com = object_pos - object_com
    v_obj = object_vel + np.cross(object_omega, r_from_com)
    if finger_vel is None:
        finger_vel = np.zeros((nf, 3))

    v_rel = finger_vel[:, None, :] - v_obj[None, :, :]
    v_n = np.sum(v_rel * n_hat, axis=2)

    hc = np.maximum(1.0 + D_CONTACT * np.maximum(0.0, -v_n), 0.0)
    pen_pos = np.maximum(pen, 0.0)
    f_n_mag = K_CONTACT * pen_pos * hc

    v_t = v_rel - v_n[:, :, None] * n_hat
    v_t_mag = np.linalg.norm(v_t, axis=2)
    v_t_safe = np.maximum(v_t_mag, 1e-12)
    t_hat = v_t / v_t_safe[:, :, None]
    fric_scale = v_t_mag / (v_t_mag + FRICTION_EPS)
    f_t_vec = -mu_val * f_n_mag[:, :, None] * fric_scale[:, :, None] * t_hat

    f_per_pair = f_n_mag[:, :, None] * n_hat + f_t_vec
    f_on_finger = np.sum(f_per_pair, axis=1)

    f_on_obj_pairs = -f_per_pair
    f_on_object = np.sum(f_on_obj_pairs, axis=(0, 1))

    contact_pts = object_pos[None, :, :] + object_r * n_hat
    r_cp = contact_pts - object_com
    tau_on_object = np.sum(np.cross(r_cp, f_on_obj_pairs), axis=(0, 1))

    return (f_on_finger, f_on_object, tau_on_object,
            {'n_contacts': int(np.sum(pen > 0)),
             'total_normal': float(np.sum(f_n_mag))})


# ═══════════════════════════════════════════════════════════════════════
#  CLASSICAL POINT CONTACT — force against the flat box face
# ═══════════════════════════════════════════════════════════════════════

def classical_point_forces(finger_pos, finger_r,
                           object_com, object_quat,
                           object_vel, object_omega,
                           box_halfs,
                           face_axis=0, face_sign=-1,
                           finger_vel=None, mu=None, k_c=None, d_c=None):
    """
    Classical point contact on a flat face of a rigid box.

    Penetration is measured against the face plane.  No sphere
    discretisation on the object — the face is a true flat surface.
    This is the standard point-contact-with-friction (PCWF) model.

    Parameters
    ----------
    finger_pos : (n, 3) — finger contact point positions
    finger_r   : float — finger sphere radius (penetration offset)
    object_com : (3,)  — box centre of mass, world frame
    object_quat: (4,)  — box orientation [w, x, y, z]
    object_vel : (3,)  — box linear velocity
    object_omega: (3,) — box angular velocity
    box_halfs  : (3,)  — (half_w, half_d, half_h) along x, y, z
    face_axis  : 0 = x-faces (default), 1 = y, 2 = z
    face_sign  : -1 = left/bottom face, +1 = right/top face
    finger_vel : (n, 3) or None
    mu         : Coulomb friction coefficient override
    k_c        : contact stiffness override

    Returns
    -------
    f_on_object : (3,) — net force on object
    tau_on_object: (3,) — net torque on object about COM
    info : dict with 'n_contacts', 'total_normal', 'contact_pts'
    """

    mu_val = MU if mu is None else mu
    kc = K_CONTACT if k_c is None else k_c
    nf = len(finger_pos)
    if finger_vel is None:
        finger_vel = np.zeros((nf, 3))

    R = quat_to_rotmat(object_quat)
    hw, hd, hh = box_halfs

    # Face outward normal in world frame
    local_normal = np.zeros(3)
    local_normal[face_axis] = float(face_sign)
    outward_n = R @ local_normal

    # Face centre in world frame
    local_face_c = np.zeros(3)
    local_face_c[face_axis] = face_sign * [hw, hd, hh][face_axis]
    face_centre = R @ local_face_c + object_com

    # Signed distance from each finger point to face plane
    # positive ↔ finger centre is on the outward (external) side
    disp = finger_pos - face_centre
    signed_dist = np.sum(disp * outward_n, axis=1)

    # Penetration: finger sphere overlaps face when signed_dist < finger_r
    pen = finger_r - signed_dist

    # Contact point: projection of finger centre onto face plane
    cp = finger_pos - signed_dist[:, None] * outward_n

    # Clip to face extent (reject contacts outside the face rectangle)
    local_cp = (R.T @ (cp - object_com).T).T   # (nf, 3) in body frame
    extents = np.array([hw, hd, hh])
    face_axes = [i for i in range(3) if i != face_axis]
    in_face = np.ones(nf, dtype=bool)
    for ax in face_axes:
        in_face &= np.abs(local_cp[:, ax]) <= extents[ax] * 1.05

    active = (pen > 0) & in_face

    if not np.any(active):
        return (np.zeros(3), np.zeros(3),
                {'n_contacts': 0, 'total_normal': 0.0,
                 'contact_pts': np.zeros((0, 3))})

    # ── Velocities at contact points ──
    r_cp_from_com = cp - object_com
    v_obj_at_cp = object_vel + np.cross(object_omega, r_cp_from_com)
    v_rel = finger_vel - v_obj_at_cp   # finger relative to object surface

    v_n_scalar = np.sum(v_rel * outward_n, axis=1)  # approach along outward n

    # ── Normal force (Hunt & Crossley) ──
    dc = D_CONTACT if d_c is None else d_c
    hc = np.maximum(1.0 + dc * np.maximum(0.0, -v_n_scalar), 0.0)
    pen_pos = np.maximum(pen, 0.0)
    f_n_mag = kc * pen_pos * hc
    f_n_mag[~active] = 0.0

    # Force on finger: pushes finger outward  → f_n_mag * outward_n
    # Force on object: reaction, pushes object inward → -f_n_mag * outward_n
    f_normal_on_obj = -f_n_mag[:, None] * outward_n   # (nf, 3)

    # ── Friction (regularised Coulomb) ──
    v_t = v_rel - v_n_scalar[:, None] * outward_n
    v_t_mag = np.linalg.norm(v_t, axis=1)
    v_t_safe = np.maximum(v_t_mag, 1e-12)
    t_hat = v_t / v_t_safe[:, None]
    fric_scale = v_t_mag / (v_t_mag + FRICTION_EPS)

    # Friction on finger opposes relative tangential slip: -mu*fn * t_hat
    # Friction on object is the reaction: +mu*fn * t_hat
    f_fric_on_obj = mu_val * f_n_mag[:, None] * fric_scale[:, None] * t_hat

    # ── Totals ──
    f_per_contact = f_normal_on_obj + f_fric_on_obj
    f_per_contact[~active] = 0.0

    f_on_object = np.sum(f_per_contact, axis=0)

    tau_on_object = np.zeros(3)
    for i in range(nf):
        if active[i]:
            tau_on_object += np.cross(r_cp_from_com[i], f_per_contact[i])

    return (f_on_object, tau_on_object,
            {'n_contacts': int(np.sum(active)),
             'total_normal': float(np.sum(f_n_mag[active])),
             'contact_pts': cp[active]})
