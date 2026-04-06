"""
contact_pipeline.py — Glue between cslc.py, contact.py, and 3D dynamics.

Imports:
  cslc.solve_contact_local   → lattice quasi-static solve
  contact.classical_point_forces → point-contact baseline

New code here (not in cslc.py or contact.py):
  - Flat-face geometric penetration query
  - Contact-point projection onto face plane
  - Wrench accumulation from lattice-resolved forces + H&C damping + friction

    ┌────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
    │ penetration    │ →  │ cslc.solve_contact_   │ →  │ wrench from      │
    │ _against_flat_ │    │ local()               │    │ lattice forces   │
    │ face()         │    │  (YOUR lattice solver) │    │ + H&C + friction │
    └────────────────┘    └──────────────────────┘    └──────────────────┘
"""

import numpy as np

# ── Import YOUR existing modules ──
from cslc import solve_contact_local
from contact import classical_point_forces


# ═══════════════════════════════════════════════════════════════════════
#  GEOMETRY — flat-face penetration and contact-point projection
#
#  These are thin helpers that don't exist in cslc.py (which is 1D)
#  or contact.py (which bundles geometry + forces together).
# ═══════════════════════════════════════════════════════════════════════

def penetration_against_flat_face(sphere_pos, sphere_radii,
                                  face_point, face_outward_normal):
    """
    Penetration of each sphere against a flat face (plane).

    Returns
    -------
    pen         : (n,) — penetration per sphere (positive = overlap)
    signed_dist : (n,) — signed distance from centre to face plane
    """
    disp = sphere_pos - face_point
    signed_dist = disp @ face_outward_normal
    pen = sphere_radii - signed_dist
    return pen, signed_dist


def project_onto_face(sphere_pos, signed_dist, face_outward_normal):
    """
    Project sphere centres onto the face plane.

    CRITICAL for torque:  Using sphere centres instead of face-projected
    contact points introduces a spurious moment arm along the face normal.
    """
    return sphere_pos - signed_dist[:, None] * face_outward_normal


# ═══════════════════════════════════════════════════════════════════════
#  WRENCH — convert lattice-resolved forces to 3D wrench on object
#
#  This is genuinely new:  cslc.py gives us per-sphere SCALAR forces
#  along the pad normal.  We need to:
#    1. Apply Hunt & Crossley velocity damping
#    2. Compute regularised Coulomb friction
#    3. Sum force and torque contributions about object COM
#
#  The H&C and friction formulas are the SAME as in your contact.py
#  (Hunt & Crossley §3.3, regularised Coulomb §3.4 of overview.md).
# ═══════════════════════════════════════════════════════════════════════

def _wrench_from_lattice_forces(
        contact_pts, lattice_forces, contact_normals,
        object_com, object_vel, object_omega,
        finger_vel_at_cp,
        dc, mu, friction_eps):
    """
    Convert per-sphere lattice forces into a 3D wrench on the object.

    Parameters
    ----------
    contact_pts      : (n, 3) — points ON THE FACE (from project_onto_face)
    lattice_forces   : (n,)   — scalar contact forces from cslc.solve_contact_local
    contact_normals  : (n, 3) — unit normal from object toward finger
    dc, mu, friction_eps : Hunt & Crossley damping, friction coeff, regularisation

    Returns  (force_on_object, torque_on_object, info_dict)
    """
    # Moment arms
    r_from_com = contact_pts - object_com

    # Relative velocity at each contact point
    v_obj_at_cp = object_vel + np.cross(object_omega, r_from_com)
    v_rel = finger_vel_at_cp - v_obj_at_cp

    # ── Normal velocity (same formula as contact.py) ──
    v_n_scalar = np.einsum('ij,ij->i', v_rel, contact_normals)

    # ── Hunt & Crossley damping (same as contact.py line: hc = max(1+d*v, 0)) ──
    hc = np.maximum(1.0 + dc * np.maximum(0.0, -v_n_scalar), 0.0)
    fn_damped = lattice_forces * hc

    # Normal force on object (pushes away from finger = -n_hat)
    f_normal = -fn_damped[:, None] * contact_normals

    # ── Regularised Coulomb friction (same as contact.py) ──
    v_t = v_rel - v_n_scalar[:, None] * contact_normals
    v_t_mag = np.linalg.norm(v_t, axis=1)
    v_t_safe = np.maximum(v_t_mag, 1e-12)
    t_hat = v_t / v_t_safe[:, None]
    fric_scale = v_t_mag / (v_t_mag + friction_eps)
    f_friction = mu * fn_damped[:, None] * fric_scale[:, None] * t_hat

    # ── Sum to wrench ──
    f_per = f_normal + f_friction
    active = lattice_forces > 1e-10
    f_per[~active] = 0.0

    f_total = np.sum(f_per, axis=0)
    tau_total = np.sum(np.cross(r_from_com, f_per), axis=0)

    n_active = int(np.sum(active))
    return f_total, tau_total, dict(
        n_contacts=n_active,
        total_normal=float(np.sum(fn_damped[active])) if n_active else 0.0,
        forces_per_sphere=fn_damped,
        active=active,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def cslc_contact_wrench(pad, ka, kl, kc, dc, mu, eps,
                        pad_sphere_pos_world, pad_radii,
                        face_point, face_outward_normal,
                        finger_vel, object_vel, object_omega, object_com,
                        prev_delta=None, n_iter=60, alpha=0.30):
    """
    Full CSLC pipeline: geometry → cslc.solve_contact_local → wrench.

    This is the GLUE function.  It calls:
      1. penetration_against_flat_face (this file)
      2. cslc.solve_contact_local (YOUR lattice solver)
      3. _wrench_from_lattice_forces (this file)

    Returns  (f_on_object, tau_on_object, info_dict)
    """
    # ── Stage 1: geometric penetration ──
    pen_nominal, signed_dist = penetration_against_flat_face(
        pad_sphere_pos_world, pad_radii,
        face_point, face_outward_normal)

    # ── Stage 2: lattice solve (YOUR solver from cslc.py) ──
    sol = solve_contact_local(
        pad, ka, kl, kc, pen_nominal,
        n_iter=n_iter, alpha=alpha, warm_start=prev_delta)

    # ── Stage 3: wrench from lattice output ──
    contact_pts = project_onto_face(
        pad_sphere_pos_world, signed_dist, face_outward_normal)
    normals = np.tile(face_outward_normal, (pad['n_spheres'], 1))

    f_on_obj, tau_on_obj, wrench_info = _wrench_from_lattice_forces(
        contact_pts, sol['forces'], normals,
        object_com, object_vel, object_omega,
        finger_vel, dc, mu, eps)

    info = {**wrench_info,
            'delta': sol['delta_x'],
            'pen_nominal': pen_nominal,
            'pen_equilibrium': sol['pen']}
    return f_on_obj, tau_on_obj, info


def point_contact_wrench(finger_pos, finger_radius, kc, dc, mu, eps,
                         face_point, face_outward_normal,
                         finger_vel_3d,
                         object_com, object_quat,
                         object_vel, object_omega,
                         box_halfs, face_axis=0, face_sign=-1):
    """
    Point-contact baseline using YOUR contact.classical_point_forces.

    Thin wrapper that reshapes inputs and calls your existing function.
    Returns the same (f_on_object, tau_on_object, info) format.
    """
    pos = finger_pos.reshape(1, 3)
    vel = finger_vel_3d.reshape(1, 3)

    f_on_obj, tau_on_obj, info = classical_point_forces(
        pos, finger_radius,
        object_com, object_quat,
        object_vel, object_omega,
        box_halfs,
        face_axis=face_axis, face_sign=face_sign,
        finger_vel=vel, mu=mu, k_c=kc, d_c=dc)

    return f_on_obj, tau_on_obj, info
