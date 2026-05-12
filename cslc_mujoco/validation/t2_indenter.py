# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 of the FEM validation plan: single hemispherical indenter on a CSLC pad.

This is the **keystone** tier of the validation plan -- where CSLC's
distributed-contact physics differentiates from point and hydroelastic
empirically.  The scene is the simplest indenter test: a rigid
hemispherical indenter pressed into a flat CSLC pad, with the indenter
descended kinematically through a sweep of penetration depths.

Hypotheses
----------
H2.1  At small penetration delta (single-sphere contact), the force
      response is linear:  F ~ k_eff * delta, with
      k_eff = ka * kc / (ka + kc) (the series effective stiffness of
      one anchor + one contact spring).

H2.2  As delta grows and the contact patch expands, the curve transitions
      toward Hertzian scaling F ~ delta^{3/2}.  The transition
      penetration scales as delta_transition ~ spacing^2 / R for a
      hemispherical indenter of radius R on a lattice of spacing s.

H2.3  CSLC's radial pressure profile p(r) has a Gaussian-like tail
      extending well outside the indenter footprint (FWHM ~ 1.5x or more
      of the footprint radius) -- because the lateral coupling kl
      spreads the load.  Hydroelastic's profile has FWHM ~ 1.1x the
      footprint because pressure is bounded by the volumetric SDF
      intersection, with no lateral spread.

H2.4  Doubling the sphere count N halves the residual between CSLC and
      its asymptotic (large-N) limit -- first-order convergence as
      predicted for graph-Laplacian Galerkin discretisations.

Stage 1 (THIS SCRIPT, current state):
  - CSLC-only.
  - Single (N, delta) sweep at one regime to validate the scene works.
  - Print aggregate force, peak per-sphere force, active count.
  - Plot F vs delta, p(r), and the per-sphere force heatmap.

Stages 2-4 add the point and hydroelastic baselines, the N sweep, and
the convergence-in-N fit.

Reading the per-sphere normal force at equilibrium
--------------------------------------------------
At quasistatic equilibrium, the CSLC solver satisfies the per-sphere
balance
    [K * delta]_i = kc * (phi_i^rest - delta_i) * gate_i        (eq.1)
which gives the per-sphere normal force `f_n,i = kc * (phi - delta) * gate`
on the indenter.  By Newton's 3rd, the indenter feels the same.

Summing eq.1 over all surface spheres:
    sum_i [K * delta]_i  =  ka * sum_i delta_i
because the lateral terms cancel pairwise across each lattice edge.
Therefore the aggregate normal force on the pad body is
    F_total = ka * sum_i delta_i                                (eq.2)
which is a closed-form readout from the CSLC handler state without
inspecting the contacts buffer.  We use eq.2 for the aggregate F and
eq.1 for the per-sphere radial profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton

from cslc_v1.common import (
    CSLC_FLAG,
    count_active_contacts,
    inspect_model,
    make_mujoco_solver,
    read_cslc_state,
    shape_cfg_cslc_pad,
    shape_cfg_hydro,
    shape_cfg_point,
)


# ───────────────────────────────────────────────────────────────────────────
#  Scene
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class T2Scene:
    # Pad geometry: a thin box, 100 x 100 mm wide, 3 mm thick.
    # The CSLC handler hardcodes face_axis=0, face_sign=+1, so the lattice
    # lives on the local +x face.  We rotate the pad body so its local +x
    # axis points world +z (up).  The pad SHAPE is hx (thickness) x hy
    # (half width) x hz (half depth) along its LOCAL axes; after the
    # rotation, hx becomes vertical thickness in the world.
    pad_hx_thick: float = 0.0015   # 3 mm pad thickness (half-extent 1.5 mm)
    pad_half: float = 0.05         # 100 mm side of the square face
    indenter_R: float = 0.010      # 10 mm hemispherical indenter radius
    # Indenter sits along world +z directly above the pad's centre.
    # Initial gap = ~0.5 mm so we can descend smoothly into a penetration.

    # CSLC material -- regime A (paper-prescribed) by default.
    cslc_ka: float = 15000.0
    cslc_kl: float = 500.0
    cslc_kc: float = 75000.0       # matches the per-pad-calibrated value in squeeze
    cslc_dc: float = 0.0
    cslc_n_iter: int = 40
    cslc_alpha: float = 0.6

    # Lattice spacing -> N x N grid.  For 100mm face and 5mm spacing: 21x21 = 441.
    cslc_spacing: float = 0.005

    # Material (the standard ke/kd/kf/mu hooks for Newton).
    ke: float = 5.0e4
    kd: float = 500.0
    kf: float = 0.0
    mu: float = 0.5
    density: float = 1000.0


def _build_t2_scene(scene: T2Scene, pad_cfg, indenter_cfg) -> newton.Model:
    """Internal scene builder shared by all three contact models.

    Pad is a flat box, rotated so its local +x face points world +z.
    Indenter is a sphere body above the pad's centre, kinematic.
    Shape configs determine the contact model (CSLC, hydro, or plain).
    """
    b = newton.ModelBuilder()

    # Pad body: rotate so local +x maps to world +z.
    # A rotation of -90 degrees around the y axis takes local (1,0,0) to
    # world (0,0,1).  (Right-hand rule: a +90 rotation around +y takes
    # local +x to world -z, which would aim the CSLC face DOWN — that
    # bug bit me on the first run.)
    pad_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.pi / 2.0)
    pad_xform = wp.transform((0.0, 0.0, 0.0), pad_q)
    pad_body = b.add_body(xform=pad_xform, is_kinematic=True, label="pad")
    # Box half-extents are LOCAL: hx=thickness, hy/hz=plan dims.
    b.add_shape_box(pad_body, hx=scene.pad_hx_thick,
                    hy=scene.pad_half, hz=scene.pad_half, cfg=pad_cfg)

    pad_top_z = scene.pad_hx_thick                # world z of the upward face
    init_z = pad_top_z + scene.indenter_R + 0.001
    indenter_xform = wp.transform((0.0, 0.0, init_z), wp.quat_identity())
    indenter_body = b.add_body(
        xform=indenter_xform, is_kinematic=True, label="indenter",
    )
    b.add_shape_sphere(indenter_body, radius=scene.indenter_R, cfg=indenter_cfg)

    model = b.finalize()
    model.set_gravity((0.0, 0.0, 0.0))   # quasistatic: gravity off
    return model


def build_t2_model_cslc(scene: T2Scene) -> newton.Model:
    pad_cfg = shape_cfg_cslc_pad(
        ke=scene.ke, kd=scene.kd, kf=scene.kf, mu=scene.mu, density=scene.density,
        cslc_spacing=scene.cslc_spacing,
        cslc_ka=scene.cslc_ka, cslc_kl=scene.cslc_kl,
        cslc_dc=scene.cslc_dc, cslc_n_iter=scene.cslc_n_iter, cslc_alpha=scene.cslc_alpha,
    )
    indenter_cfg = shape_cfg_point(
        ke=scene.ke, kd=scene.kd, kf=scene.kf, mu=scene.mu, density=scene.density,
    )
    return _build_t2_scene(scene, pad_cfg, indenter_cfg)


# Back-compat: keep `build_t2_model` pointing at the CSLC scene.
def build_t2_model(scene: T2Scene) -> newton.Model:
    return build_t2_model_cslc(scene)


def build_t2_model_point(scene: T2Scene) -> newton.Model:
    """Baseline: plain box pad + plain sphere indenter (single contact point)."""
    pad_cfg = shape_cfg_point(
        ke=scene.ke, kd=scene.kd, kf=scene.kf, mu=scene.mu, density=scene.density,
    )
    indenter_cfg = shape_cfg_point(
        ke=scene.ke, kd=scene.kd, kf=scene.kf, mu=scene.mu, density=scene.density,
    )
    return _build_t2_scene(scene, pad_cfg, indenter_cfg)


def build_t2_model_hydro(scene: T2Scene, kh: float) -> newton.Model:
    """Hydroelastic pad + indenter.

    `kh` (hydroelastic modulus, Pa) controls patch-area stiffness.  For
    indenter-pad geometry we calibrate so kh * (indenter-footprint area
    at 1 mm) equals the same per-area force a CSLC pad would produce.
    """
    pad_cfg = shape_cfg_hydro(
        ke=scene.ke, kd=scene.kd, kf=scene.kf, mu=scene.mu, density=scene.density,
        kh=kh, sdf_resolution=64,
    )
    indenter_cfg = shape_cfg_hydro(
        ke=scene.ke, kd=scene.kd, kf=scene.kf, mu=scene.mu, density=scene.density,
        kh=kh, sdf_resolution=64,
    )
    return _build_t2_scene(scene, pad_cfg, indenter_cfg)


def set_indenter_z(state, indenter_body_idx: int, z: float) -> None:
    """Pin the indenter body's z coordinate; preserve x=y=0 and identity orient."""
    q = state.body_q.numpy()
    q[indenter_body_idx, 0] = 0.0
    q[indenter_body_idx, 1] = 0.0
    q[indenter_body_idx, 2] = float(z)
    q[indenter_body_idx, 3] = 0.0
    q[indenter_body_idx, 4] = 0.0
    q[indenter_body_idx, 5] = 0.0
    q[indenter_body_idx, 6] = 1.0
    state.body_q.assign(wp.array(q, dtype=wp.transform,
                                 device=state.body_q.device))


# ───────────────────────────────────────────────────────────────────────────
#  Per-step CSLC readout
# ───────────────────────────────────────────────────────────────────────────


def cslc_per_sphere(model, scene: T2Scene
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read per-sphere positions (world), deltas, raw penetrations, and forces.

    Returns
    -------
    pos_world : (n_surface, 3)  -- lattice sphere world-frame centres
    delta     : (n_surface,)    -- compression of each surface sphere
    phi_rest  : (n_surface,)    -- raw penetration phi_i^rest (post smooth-relu)
    f_n       : (n_surface,)    -- per-sphere normal force kc * (phi - delta) * gate_active
    agg       : dict with aggregate fields
        F_total_anchor   : ka * sum(delta)        -- eq.2 above
        F_total_contact  : sum( kc * (phi - delta) ) over active spheres -- eq.1
        n_active         : number of surface spheres with phi > 0
        max_delta        : max compression
        peak_force       : max per-sphere force
    """
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        raise RuntimeError("No CSLC handler on this model")
    d = handler.cslc_data
    is_surface = d.is_surface.numpy() == 1
    n_sphere = int(np.sum(is_surface))

    # Local positions (in pad body frame after shape transform applied).
    pos_local = d.positions.numpy()[is_surface]   # (n_surface, 3) in pad body frame
    delta = d.sphere_delta.numpy()[is_surface]
    phi_rest = handler.raw_penetration.numpy()[is_surface]

    # World positions of each surface sphere.
    # Pad body transform: the rotation maps local +x -> world +z.  But the
    # CSLC pad lives on a SHAPE attached to the body; we have to compose
    # body_q with the shape_transform.  For the simple scene built above,
    # shape_transform is identity so body_q * pos_local is the world point.
    state = model.state()
    body_q = state.body_q.numpy()
    shape_body_arr = model.shape_body.numpy()
    shape_flags = model.shape_flags.numpy()
    # First CSLC-flagged shape index:
    cslc_shape = next(i for i in range(model.shape_count)
                      if shape_flags[i] & CSLC_FLAG)
    pad_body_idx = int(shape_body_arr[cslc_shape])
    pad_q = body_q[pad_body_idx]
    px, py, pz, qx, qy, qz, qw = pad_q
    # Rotate each local position by (qx, qy, qz, qw), then translate.
    def qrot(q, v):
        x, y, z, w = q
        # Hamilton product: vec3 rotated by quat = q * v * q*
        xyz = np.array([x, y, z]); wq = w
        t = 2.0 * np.cross(xyz, v)
        return v + wq * t + np.cross(xyz, t)
    pos_world = np.array([qrot((qx, qy, qz, qw), pl) + np.array([px, py, pz])
                          for pl in pos_local])

    # Per-sphere force (eq.1 above).  At equilibrium the gate is ~1 wherever
    # phi > 0; we evaluate the smooth gate explicitly to avoid double-
    # counting eps-band cells.
    eps = d.smoothing_eps
    eff_pen = phi_rest - delta
    gate = 0.5 * (1.0 + eff_pen / np.sqrt(eff_pen * eff_pen + eps * eps))
    f_n = d.kc * eff_pen * gate

    # n_active threshold = 1 micron.  The kernel's smooth_relu returns a
    # tiny positive sliver (~ eps^2 / |pen_3d|) for negative pen_3d -- fp32
    # noise masquerading as active contact.  For eps = 1e-5 m and typical
    # negative pen_3d ~ 1 mm, that floor is ~ 1e-7 m; a 1 micron cutoff is
    # well above the noise and well below typical physical sub-mm phi.
    n_phys_active = int(np.sum(phi_rest > 1.0e-6))

    agg = {
        "F_total_anchor": float(d.ka * delta.sum()),
        "F_total_contact": float(f_n.sum()),
        "n_active": n_phys_active,
        "n_active_raw": int(np.sum(phi_rest > 0.0)),
        "max_delta": float(delta.max() if len(delta) else 0.0),
        "peak_force": float(f_n.max() if len(f_n) else 0.0),
        "k_eff_series": d.ka * d.kc / (d.ka + d.kc),
        "kc_calibrated": float(d.kc),
        "ka": float(d.ka),
        "kl": float(d.kl),
        "r_lat": float(d.radii.numpy()[0]),
    }
    return pos_world, delta, phi_rest, f_n, agg


# ───────────────────────────────────────────────────────────────────────────
#  Quasistatic single-shot measurement
# ───────────────────────────────────────────────────────────────────────────


def measure_one_baseline(model, contacts, scene: T2Scene, delta_indenter: float,
                         indenter_body_idx: int, n_settle: int = 5
                         ) -> dict:
    """Measure aggregate F and per-contact (position, force) for point/hydro.

    Reads from the standard contacts buffer:
      - rigid_contact_count, rigid_contact_shape0  (active iff shape0 >= 0)
      - rigid_contact_point0  (contact point in shape0 body frame)
      - rigid_contact_normal  (contact normal, world frame)
      - rigid_contact_stiffness  (per-contact ke; for point, = shape ke;
        for hydro, kh * polygon_area projected onto contact)
      - rigid_contact_margin0, margin1, and body_q to compute penetration

    Force per contact: f = stiffness * pen, applied along the normal.
    pen = (margin0 + margin1) - dot(p1_w - p0_w, normal).

    Returns dict with F_total, per_contact_pos, per_contact_force,
    plus diagnostic counts.  Notes for `delta_indenter` zero point: same
    convention as for CSLC -- distance from the lattice apex.  For point
    and hydro there is no lattice, so we use pad_top + r_lat where
    r_lat = cslc_spacing/2 to match the CSLC scene's coordinate.
    """
    state = model.state()
    pad_top_z = scene.pad_hx_thick
    # For point/hydro the "physical pad surface" IS the box top face --
    # there is no CSLC sphere-apex offset.  delta_indenter is measured
    # from pad_top_z directly, so the indenter centre sits at
    # `pad_top_z + R - delta_indenter`.  This is the FAIR comparison
    # against CSLC's "measured from lattice apex" convention (Finding G):
    # both define delta_indenter as the descent from the model's own
    # "effective load-bearing surface".
    z_indenter = pad_top_z + scene.indenter_R - delta_indenter
    set_indenter_z(state, indenter_body_idx, z_indenter)

    for _ in range(n_settle):
        model.collide(state, contacts)

    n_total = int(contacts.rigid_contact_count.numpy()[0])
    if n_total == 0:
        return {
            "F_total": 0.0, "n_active": 0, "delta_indenter": delta_indenter,
            "contact_pos_world": np.zeros((0, 3)), "contact_force": np.zeros(0),
            "contact_pen": np.zeros(0),
        }
    s0 = contacts.rigid_contact_shape0.numpy()[:n_total]
    s1 = contacts.rigid_contact_shape1.numpy()[:n_total]
    p0_local = contacts.rigid_contact_point0.numpy()[:n_total]
    p1_local = contacts.rigid_contact_point1.numpy()[:n_total]
    normals = contacts.rigid_contact_normal.numpy()[:n_total]
    margin0 = contacts.rigid_contact_margin0.numpy()[:n_total]
    margin1 = contacts.rigid_contact_margin1.numpy()[:n_total]
    # Per-contact stiffness only populated for hydro/CSLC pipelines.  For
    # plain point contact, fall back to the pad shape's material ke.
    if contacts.rigid_contact_stiffness is not None:
        stiff = contacts.rigid_contact_stiffness.numpy()[:n_total]
    else:
        ke_per_shape = model.shape_material_ke.numpy()
        stiff = np.array([ke_per_shape[int(s)] if s >= 0 else 0.0 for s in s0],
                         dtype=np.float32)
    active_mask = s0 >= 0
    if not np.any(active_mask):
        return {
            "F_total": 0.0, "n_active": 0, "delta_indenter": delta_indenter,
            "contact_pos_world": np.zeros((0, 3)), "contact_force": np.zeros(0),
            "contact_pen": np.zeros(0),
        }

    body_q = state.body_q.numpy()
    shape_body_arr = model.shape_body.numpy()
    shape_transform = model.shape_transform.numpy()

    def qrot(q, v):
        x, y, z, w = q
        xyz = np.array([x, y, z]); wq = w
        t = 2.0 * np.cross(xyz, v)
        return v + wq * t + np.cross(xyz, t)

    pos_world = np.zeros((n_total, 3), dtype=np.float64)
    pen_arr = np.zeros(n_total, dtype=np.float64)
    forces = np.zeros(n_total, dtype=np.float64)

    for i in range(n_total):
        if not active_mask[i]:
            continue
        # Compute world position of p0 (on the pad shape, shape index s0[i])
        s_idx = int(s0[i]); b_idx = int(shape_body_arr[s_idx])
        # Apply shape_transform then body_q to p0_local
        xs = shape_transform[s_idx]
        p_body = qrot(xs[3:7], np.array(p0_local[i], dtype=np.float64)) + xs[:3]
        p_world0 = qrot(body_q[b_idx, 3:7], p_body) + body_q[b_idx, :3]
        # And p1 on shape s1[i]
        s_idx1 = int(s1[i]); b_idx1 = int(shape_body_arr[s_idx1])
        xs1 = shape_transform[s_idx1]
        p_body1 = qrot(xs1[3:7], np.array(p1_local[i], dtype=np.float64)) + xs1[:3]
        p_world1 = qrot(body_q[b_idx1, 3:7], p_body1) + body_q[b_idx1, :3]
        # Penetration: margin0 + margin1 - dot(p1 - p0, normal)
        n = np.array(normals[i], dtype=np.float64)
        pen = (float(margin0[i]) + float(margin1[i])
               - float(np.dot(p_world1 - p_world0, n)))
        pen_arr[i] = max(pen, 0.0)
        forces[i] = max(0.0, float(stiff[i])) * pen_arr[i]
        pos_world[i] = p_world0

    pos_active = pos_world[active_mask]
    force_active = forces[active_mask]
    pen_active = pen_arr[active_mask]
    F_total = float(force_active.sum())
    return {
        "F_total": F_total,
        "n_active": int(active_mask.sum()),
        "delta_indenter": delta_indenter,
        "contact_pos_world": pos_active,
        "contact_force": force_active,
        "contact_pen": pen_active,
    }


def measure_one(model, contacts, scene: T2Scene, delta_indenter: float,
                indenter_body_idx: int, n_settle: int = 10, debug: bool = False
                ) -> dict:
    """Set indenter depth, collide n_settle times, read CSLC state.

    `delta_indenter` is the descent from the *lattice-sphere apex*,
    NOT from the pad face: the topmost lattice sphere has its top at
    world z = pad_top_z + r_lat (since the sphere of radius r_lat sits
    with its centre on the pad face).  delta_indenter = 0 means the
    indenter is just kissing that apex; positive values are compression.

    This shift avoids the trivial r_lat-baseline force that arose when
    we used `pad_top_z + R - delta` as the indenter z and naively called
    `delta = 0` a zero-force operating point.  See README/t2_results.md.

    n_settle warm-started collide() calls are usually enough; the Jacobi
    sweeps per call (cslc_n_iter) take a fresh delta vector to near
    convergence on the first one and the rest are warm-start polish.
    """
    state = model.state()
    pad_top_z = scene.pad_hx_thick
    # Topmost lattice apex sits r_lat above the pad face.  r_lat is set
    # by the CSLC factory to spacing/2 for uniform tiling.
    r_lat = scene.cslc_spacing * 0.5
    apex_z = pad_top_z + r_lat
    # Indenter centre at apex_z + R - delta_indenter:
    #   delta_indenter = 0  -> indenter just kissing the apex sphere.
    #   delta_indenter > 0  -> compressive penetration.
    z_indenter = apex_z + scene.indenter_R - delta_indenter
    set_indenter_z(state, indenter_body_idx, z_indenter)

    handler = model._collision_pipeline.cslc_handler
    d = handler.cslc_data

    for k in range(n_settle):
        model.collide(state, contacts)
        if debug:
            phi = handler.raw_penetration.numpy()
            delta = d.sphere_delta.numpy()
            is_surf = d.is_surface.numpy() == 1
            print(f"    [collide {k}] phi(surface): max = {phi[is_surf].max():.4e}, "
                  f"n_active = {int((phi[is_surf] > 0).sum())} ; "
                  f"delta: max = {delta[is_surf].max():.4e}, "
                  f"mean = {delta[is_surf].mean():.4e}")

    pos, delta, phi, f_n, agg = cslc_per_sphere(model, scene)
    agg["delta_indenter"] = delta_indenter
    agg["pos_world"] = pos
    agg["delta_per_sphere"] = delta
    agg["phi_per_sphere"] = phi
    agg["f_n_per_sphere"] = f_n
    return agg


# ───────────────────────────────────────────────────────────────────────────
#  Stage 1 driver
# ───────────────────────────────────────────────────────────────────────────


def stage1_demo(scene: T2Scene, *, delta_indenter_mm: float = 1.0):
    """Build the scene, measure at one (N, delta), print + plot a snapshot.

    This is the smoke test before the full sweep.  Verifies that:
      - the pad+indenter scene builds cleanly
      - CSLC handler is attached and reports the right N
      - measure_one() returns sensible numbers
      - a 2D heatmap of per-sphere force looks like a localised indenter
    """
    print("\n" + "=" * 72)
    print(f"  Tier 2 stage 1 demo: indenter at delta = {delta_indenter_mm} mm")
    print(f"  pad {2*scene.pad_half*1e3:.0f} x {2*scene.pad_half*1e3:.0f} mm, "
          f"spacing {scene.cslc_spacing*1e3:.1f} mm, "
          f"indenter R = {scene.indenter_R*1e3:.1f} mm")
    print("=" * 72)
    model = build_t2_model(scene)
    inspect_model(model, "Tier 2 stage 1")
    contacts = model.contacts()

    # Indenter body index = 1 (pad = 0).  Verify via label.
    body_labels = list(model.body_key) if hasattr(model, "body_key") else None
    print(f"  body labels: {body_labels}")
    indenter_idx = 1

    # One collide before adjusting indenter, so the CSLC handler initialises
    # and we can read N.
    state = model.state()
    model.collide(state, contacts)
    state_info = read_cslc_state(model)
    if state_info is not None:
        print(f"  CSLC: n_surface = {state_info['n_surface']}  "
              f"(expected ~{int(round(2*scene.pad_half/scene.cslc_spacing))+1}^2)")

    res = measure_one(model, contacts, scene,
                      delta_indenter=delta_indenter_mm * 1e-3,
                      indenter_body_idx=indenter_idx, debug=False)
    print()
    print(f"  RESULT @ delta = {delta_indenter_mm} mm")
    print(f"    F_total (sum of anchor reactions, ka * sum delta) = {res['F_total_anchor']:.4f} N")
    print(f"    F_total (sum of contact forces, kc * phi_eff)     = {res['F_total_contact']:.4f} N")
    print(f"    n_active surface contacts                          = {res['n_active']}")
    print(f"    max delta per sphere                              = {res['max_delta']*1e3:.4f} mm")
    print(f"    peak per-sphere normal force                      = {res['peak_force']:.4f} N")
    print(f"    k_eff = ka * kc / (ka + kc)                       = {res['k_eff_series']:.2f} N/m")
    print(f"    linear F estimate = k_eff * delta_indenter        = "
          f"{res['k_eff_series'] * delta_indenter_mm * 1e-3:.4f} N")

    # Visual: per-sphere force heatmap on the pad face.
    pos = res["pos_world"]
    f = res["f_n_per_sphere"]
    # The pad face lives at world z = pad_hx_thick (pos_world[:, 2] all
    # approximately equal).  Plot in (x, y) world.
    xs = pos[:, 0]
    ys = pos[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    sc = ax.scatter(xs * 1e3, ys * 1e3, c=f, cmap="hot", s=60, edgecolors="k",
                    linewidths=0.2)
    fig.colorbar(sc, ax=ax, label="per-sphere f_n (N)")
    R_mm = scene.indenter_R * 1e3
    theta = np.linspace(0, 2 * np.pi, 64)
    ax.plot(R_mm * np.cos(theta), R_mm * np.sin(theta), "b--",
            label=f"indenter footprint (R = {R_mm:.0f} mm)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"Per-sphere normal force at delta = {delta_indenter_mm} mm  "
                 f"(F_total = {res['F_total_anchor']:.3f} N)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("cslc_v1/validation/figures/t2_stage1_heatmap.png", dpi=140)
    plt.close(fig)
    print()
    print(f"  Wrote cslc_v1/validation/figures/t2_stage1_heatmap.png")
    return res


def sweep_delta(scene: T2Scene, deltas_mm: list[float]) -> dict:
    """Run the indenter scene at a sweep of penetration depths.

    Returns a dict with one list per (delta-keyed) metric.
    Reuses the same model across all deltas -- the CSLC handler's
    `sphere_delta` warm-starts each new measurement.
    """
    print("\n" + "=" * 72)
    print(f"  Tier 2 stage 1 F-vs-delta sweep at  ka={scene.cslc_ka:g}, "
          f"kl={scene.cslc_kl:g}, spacing={scene.cslc_spacing*1e3:.1f} mm")
    print("=" * 72)
    model = build_t2_model(scene)
    contacts = model.contacts()
    indenter_idx = 1

    # Warm-up: one collide at small penetration so the handler is fully
    # initialised with sensible delta before the first measurement.
    _ = model.state()

    rows = []
    print(f"  {'delta_mm':>10}  {'F_anchor':>10}  {'F_contact':>10}  "
          f"{'max_delta':>10}  {'n_active':>9}  {'peak_f':>9}")
    for d_mm in deltas_mm:
        res = measure_one(model, contacts, scene,
                          delta_indenter=d_mm * 1e-3,
                          indenter_body_idx=indenter_idx)
        rows.append(res)
        print(f"  {d_mm:>10.3f}  {res['F_total_anchor']:>10.4f}  "
              f"{res['F_total_contact']:>10.4f}  "
              f"{res['max_delta']*1e3:>10.4f}  "
              f"{res['n_active']:>9d}  {res['peak_force']:>9.4f}")

    delta_arr = np.array(deltas_mm) * 1e-3
    F_arr = np.array([r["F_total_anchor"] for r in rows])
    n_act = np.array([r["n_active"] for r in rows])

    # Plot F vs delta on both linear and log-log axes.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].plot(delta_arr * 1e3, F_arr, "-o", label="CSLC")
    keff = rows[0]["k_eff_series"]
    r_lat = rows[0]["r_lat"]
    # Naive linear with sphere-tiling offset: F_predict = keff * N_active * (r_lat + delta)
    axes[0].plot(delta_arr * 1e3,
                 keff * n_act * (r_lat + delta_arr),
                 "--", label="N_active * keff * (r_lat + delta)  (baseline only)")
    axes[0].set_xlabel("indenter penetration delta (mm)")
    axes[0].set_ylabel("aggregate F (N)")
    axes[0].set_title("Tier 2  F vs delta  (linear)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(delta_arr * 1e3, F_arr, "-o", label="CSLC")
    # Hertzian asymptote: F ~ delta^{3/2}
    ref_F = F_arr[len(F_arr) // 2]
    ref_d = delta_arr[len(F_arr) // 2]
    axes[1].loglog(delta_arr * 1e3, ref_F * (delta_arr / ref_d) ** 1.5,
                   "--", label="Hertzian F ~ delta^{3/2}", alpha=0.6)
    axes[1].loglog(delta_arr * 1e3, ref_F * (delta_arr / ref_d) ** 1.0,
                   ":", label="linear F ~ delta", alpha=0.6)
    axes[1].set_xlabel("indenter penetration delta (mm)")
    axes[1].set_ylabel("aggregate F (N)")
    axes[1].set_title("Tier 2  F vs delta  (log-log)")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.suptitle(f"ka={scene.cslc_ka:g}, kl={scene.cslc_kl:g}, "
                 f"spacing={scene.cslc_spacing*1e3:.1f} mm, "
                 f"kc_cal={rows[0]['kc_calibrated']:.1f} N/m")
    fig.tight_layout()
    fig.savefig("cslc_v1/validation/figures/t2_stage1_F_vs_delta.png", dpi=140)
    plt.close(fig)
    print(f"\n  Wrote cslc_v1/validation/figures/t2_stage1_F_vs_delta.png")

    # Empirical power-law fit on the log-log.  Restrict to deltas above
    # the spacing scale so we are out of the "single-cell" regime.
    mask = delta_arr > scene.cslc_spacing * 0.5
    if mask.sum() >= 3:
        slope, intercept = np.polyfit(np.log(delta_arr[mask]),
                                      np.log(F_arr[mask]), 1)
        print(f"  Empirical exponent in F ~ delta^p  (delta > spacing/2):  p = {slope:.3f}")
    else:
        slope = float("nan")
    return {"deltas": delta_arr, "F": F_arr, "n_active": n_act,
            "rows": rows, "slope": slope}


def radial_profile(positions_world: np.ndarray, forces: np.ndarray, *,
                   r_max: float, n_bins: int = 25,
                   sphere_area: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin per-contact forces into annular bins and return p(r).

    positions_world : (n, 3)
    forces          : (n,) per-contact normal force [N]
    sphere_area     : if given, per-contact "area" used to convert force
                      to pressure (for CSLC, this is spacing**2).  If
                      None, returns sum-of-forces per annulus, normalised
                      to the annulus area to give a force per unit area.

    Returns (r_centres, p, count_per_bin) -- r in metres, p in Pa or
    N/m^2 (force per annulus area), count = number of contacts per bin.
    """
    if len(forces) == 0:
        r_centres = np.linspace(0, r_max, n_bins)
        return r_centres, np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    r = np.linalg.norm(positions_world[:, :2], axis=1)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centres = 0.5 * (edges[:-1] + edges[1:])
    annulus_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    p = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=int)
    bin_idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_bins - 1)
    for i, f in enumerate(forces):
        p[bin_idx[i]] += float(f)
        count[bin_idx[i]] += 1
    # Convert force per bin into force per area.
    # If a per-contact sphere area is given, we use a weighted-mean style:
    #   p_bin = sum(f_i) / (count_bin * sphere_area)   when count_bin > 0
    # Otherwise we use the annulus area as the denominator.
    p_out = np.zeros(n_bins)
    if sphere_area is not None:
        for k in range(n_bins):
            if count[k] > 0:
                p_out[k] = p[k] / (count[k] * sphere_area)
    else:
        for k in range(n_bins):
            if annulus_area[k] > 0:
                p_out[k] = p[k] / annulus_area[k]
    return r_centres, p_out, count


def fwhm_from_profile(r: np.ndarray, p: np.ndarray) -> float:
    """Compute the full-width-half-maximum of a peaked radial profile.

    The profile is assumed to peak near r=0 and decay monotonically.
    Returns the smallest r at which p falls below 0.5 * p.max().
    If the profile never crosses half-max within r_max, returns r_max.
    """
    if len(p) == 0 or p.max() <= 0:
        return float("nan")
    half = 0.5 * p.max()
    below = np.where(p < half)[0]
    if len(below) == 0:
        return float(r[-1])
    # Linear interp between the last point above half and the first below.
    k = below[0]
    if k == 0:
        return float(r[0])
    r_above = r[k - 1]; p_above = p[k - 1]
    r_below = r[k];     p_below = p[k]
    if p_above == p_below:
        return float(r_above)
    return float(r_above + (half - p_above) * (r_below - r_above) / (p_below - p_above))


def r_at_fraction(r: np.ndarray, p: np.ndarray, frac: float) -> float:
    """Smallest r at which p falls below frac * p.max() (linearly interpolated)."""
    if len(p) == 0 or p.max() <= 0:
        return float("nan")
    target = frac * p.max()
    below = np.where(p < target)[0]
    if len(below) == 0:
        return float(r[-1])
    k = below[0]
    if k == 0:
        return float(r[0])
    r_above = r[k - 1]; p_above = p[k - 1]
    r_below = r[k];     p_below = p[k]
    if p_above == p_below:
        return float(r_above)
    return float(r_above + (target - p_above) * (r_below - r_above) / (p_below - p_above))


def pressure_profile_comparison(scene: T2Scene, delta_mm: float,
                                kh_hydro: float, out_dir: str,
                                regime_label: str = ""
                                ) -> dict:
    """Run all three models at delta_mm and return radial profiles.

    Calibration note.  We do NOT match aggregate F across models -- each
    uses its native stiffness calibration.  The interesting comparison
    is the SHAPE of p(r), specifically FWHM relative to indenter
    footprint radius `a_indent = sqrt(R*delta)` (Hertzian footprint).
    """
    print("\n" + "=" * 72)
    print(f"  Tier 2 stage 3 pressure profile comparison @ delta = {delta_mm} mm")
    print(f"  hydro kh = {kh_hydro:.3g} Pa")
    print("=" * 72)
    R = scene.indenter_R
    a_indent = float(np.sqrt(R * delta_mm * 1e-3))   # Hertzian footprint radius

    # CSLC
    model_c = build_t2_model_cslc(scene)
    contacts_c = model_c.contacts()
    res_c = measure_one(model_c, contacts_c, scene,
                        delta_indenter=delta_mm * 1e-3, indenter_body_idx=1)
    pos_c = res_c["pos_world"]
    f_c   = res_c["f_n_per_sphere"]
    # Filter to only physically-active spheres (f > 1e-4 N, well above noise).
    active = f_c > 1e-4
    pos_c = pos_c[active]; f_c = f_c[active]
    r_c, p_c, cnt_c = radial_profile(
        pos_c, f_c, r_max=4 * a_indent + scene.cslc_spacing,
        n_bins=20, sphere_area=scene.cslc_spacing ** 2,
    )

    # Point
    model_p = build_t2_model_point(scene)
    contacts_p = model_p.contacts()
    res_p = measure_one_baseline(model_p, contacts_p, scene,
                                 delta_indenter=delta_mm * 1e-3,
                                 indenter_body_idx=1)
    pos_p = res_p["contact_pos_world"]; f_p = res_p["contact_force"]
    # Point contact: a single delta at r=0 -- represent as a delta-spike
    # at the smallest r bin so it still appears on the plot.
    r_p, p_p, cnt_p = radial_profile(
        pos_p, f_p, r_max=4 * a_indent + scene.cslc_spacing, n_bins=20,
    )

    # Hydro
    model_h = build_t2_model_hydro(scene, kh=kh_hydro)
    contacts_h = model_h.contacts()
    res_h = measure_one_baseline(model_h, contacts_h, scene,
                                 delta_indenter=delta_mm * 1e-3,
                                 indenter_body_idx=1)
    pos_h = res_h["contact_pos_world"]; f_h = res_h["contact_force"]
    r_h, p_h, cnt_h = radial_profile(
        pos_h, f_h, r_max=4 * a_indent + scene.cslc_spacing, n_bins=20,
    )

    fwhm_c = fwhm_from_profile(r_c, p_c)
    fwhm_p = fwhm_from_profile(r_p, p_p)
    fwhm_h = fwhm_from_profile(r_h, p_h)
    # 10 %-fall-off radius: more discriminating than FWHM for the
    # "tail outside the indenter footprint" claim.  H2.3 predicts
    # CSLC's r_10 / a_indent >> hydro's.
    r10_c = r_at_fraction(r_c, p_c, 0.1)
    r10_p = r_at_fraction(r_p, p_p, 0.1)
    r10_h = r_at_fraction(r_h, p_h, 0.1)
    print(f"  Hertzian indenter footprint radius  a_indent = sqrt(R * delta) = {a_indent*1e3:.3f} mm")
    print(f"  CSLC   F = {f_c.sum():7.3f} N  n_act = {len(f_c):3d}  "
          f"FWHM/a_indent = {fwhm_c/a_indent:.3f}  r10/a_indent = {r10_c/a_indent:.3f}")
    print(f"  Point  F = {f_p.sum():7.3f} N  n_act = {len(f_p):3d}  "
          f"FWHM/a_indent = {fwhm_p/a_indent:.3f}  r10/a_indent = {r10_p/a_indent:.3f}")
    print(f"  Hydro  F = {f_h.sum():7.3f} N  n_act = {len(f_h):3d}  "
          f"FWHM/a_indent = {fwhm_h/a_indent:.3f}  r10/a_indent = {r10_h/a_indent:.3f}")

    # Plot.  Normalise each profile by its own peak to compare shapes
    # (heights are not comparable across models given different kc/kh
    # calibrations).
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, label_unit in zip(axes, ("raw", "normalised")):
        if label_unit == "raw":
            ax.plot(r_c * 1e3, p_c, "-o", label="CSLC", markersize=4)
            ax.plot(r_p * 1e3, p_p, "-s", label="point", markersize=4)
            ax.plot(r_h * 1e3, p_h, "-^", label="hydroelastic", markersize=4)
            ax.set_ylabel("local pressure (Pa or N/m^2)")
            ax.set_yscale("log")
        else:
            def norm(p):
                return p / max(p.max(), 1e-30)
            ax.plot(r_c * 1e3, norm(p_c), "-o", label="CSLC", markersize=4)
            ax.plot(r_p * 1e3, norm(p_p), "-s", label="point", markersize=4)
            ax.plot(r_h * 1e3, norm(p_h), "-^", label="hydroelastic", markersize=4)
            ax.set_ylabel("p(r) / peak(p)")
            ax.set_ylim(-0.05, 1.1)
        ax.axvline(a_indent * 1e3, color="k", linestyle=":",
                   label=f"a_indent = {a_indent*1e3:.2f} mm")
        ax.set_xlabel("radial distance from indenter axis (mm)")
        ax.set_title(f"Tier 2  pressure profile ({label_unit})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"delta = {delta_mm} mm,  ka={scene.cslc_ka:g}, kl={scene.cslc_kl:g}, "
                 f"spacing={scene.cslc_spacing*1e3:.1f} mm, kh={kh_hydro:.1e} Pa")
    fig.tight_layout()
    suffix = f"_{regime_label}" if regime_label else ""
    path = f"{out_dir}/t2_stage3_profile_d{int(delta_mm*10):02d}{suffix}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  Wrote {path}")
    return {"r_c": r_c, "p_c": p_c, "r_p": r_p, "p_p": p_p,
            "r_h": r_h, "p_h": p_h,
            "fwhm_c": fwhm_c, "fwhm_p": fwhm_p, "fwhm_h": fwhm_h,
            "r10_c": r10_c, "r10_p": r10_p, "r10_h": r10_h,
            "a_indent": a_indent,
            "F_c": f_c.sum(), "F_p": f_p.sum(), "F_h": f_h.sum()}


def n_convergence_sweep(spacings_mm: list[float], deltas_mm: list[float],
                        out_dir: str, *, ka: float = 15000.0, kl: float = 500.0
                        ) -> dict:
    """H2.4 sweep: vary lattice spacing while keeping pad and indenter fixed.

    For each spacing, build a fresh CSLC scene and run the F-vs-delta
    sweep.  The finest spacing is treated as the asymptotic limit; for
    every coarser one we compute the residual error at each delta and
    fit the convergence order in h (= spacing).
    """
    print("\n" + "#" * 72)
    print(f"# STAGE 4: N convergence sweep  (ka={ka:g}, kl={kl:g})")
    print("#" * 72)
    runs = {}
    for sp_mm in spacings_mm:
        scene = T2Scene(cslc_spacing=sp_mm * 1e-3, cslc_ka=ka, cslc_kl=kl)
        # Build model once per spacing, then sweep deltas.
        model = build_t2_model_cslc(scene)
        contacts = model.contacts()
        n_side = int(round(2 * scene.pad_half / scene.cslc_spacing)) + 1
        print(f"\n  spacing = {sp_mm:.2f} mm  ->  grid ~ {n_side}x{n_side} = {n_side*n_side} spheres")
        Fs = []
        for d_mm in deltas_mm:
            res = measure_one(model, contacts, scene,
                              delta_indenter=d_mm * 1e-3, indenter_body_idx=1)
            Fs.append(res["F_total_anchor"])
            print(f"    delta = {d_mm:5.2f} mm   F = {Fs[-1]:8.4f} N   "
                  f"n_active = {res['n_active']}")
        runs[sp_mm] = {"F": np.array(Fs), "n_side": n_side}
    deltas = np.array(deltas_mm) * 1e-3

    # Asymptotic: finest spacing.
    sp_fine = min(spacings_mm)
    F_asymp = runs[sp_fine]["F"]
    print(f"\n  Asymptote: spacing = {sp_fine:.2f} mm  (treated as N-> infinity)")
    print(f"  {'spacing':>9}  " + "  ".join(f"d={d:.2f}".rjust(8) for d in deltas_mm))
    print(f"  {'(mm)':>9}  " + "  ".join(["F (N)".rjust(8)] * len(deltas_mm)))
    for sp_mm in spacings_mm:
        line = f"  {sp_mm:>9.2f}  " + "  ".join(f"{f:8.4f}" for f in runs[sp_mm]["F"])
        print(line)
    print()

    # Convergence order: for each delta, fit log|F - F_asymp| ~ p * log(h).
    print(f"  {'delta_mm':>9}  {'order p':>9}  {'err at finest non-asymp':>22}")
    orders = {}
    for di, d in enumerate(deltas_mm):
        hs = np.array([sp for sp in spacings_mm if sp > sp_fine])
        errs = np.array([abs(runs[sp]["F"][di] - F_asymp[di]) for sp in hs])
        if (errs > 0).sum() >= 2:
            slope, _ = np.polyfit(np.log(hs[errs > 0]), np.log(errs[errs > 0]), 1)
        else:
            slope = float("nan")
        orders[d] = slope
        print(f"  {d:>9.2f}  {slope:>9.3f}  {errs.min() if errs.size else float('nan'):>22.4f}")

    # Plot: F vs delta for each spacing; convergence error vs h on log-log
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(spacings_mm)))
    for sp_mm, c in zip(spacings_mm, cmap):
        axes[0].loglog(deltas, runs[sp_mm]["F"], "-o", color=c,
                       label=f"spacing = {sp_mm:.2f} mm  "
                             f"({runs[sp_mm]['n_side']}x{runs[sp_mm]['n_side']})")
    axes[0].set_xlabel("delta (m)")
    axes[0].set_ylabel("aggregate F (N)")
    axes[0].set_title("Tier 2 stage 4  F vs delta vs spacing")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    for di, d in enumerate(deltas_mm):
        hs = np.array([sp for sp in spacings_mm if sp > sp_fine]) * 1e-3
        errs = np.array([abs(runs[sp]["F"][di] - F_asymp[di]) for sp in spacings_mm if sp > sp_fine])
        if (errs > 0).any():
            axes[1].loglog(hs * 1e3, errs, "-o", label=f"delta = {d:.2f} mm")
    # First-order reference line.
    h_ref = np.array(sorted([sp for sp in spacings_mm if sp > sp_fine])) * 1e-3
    if h_ref.size > 0:
        axes[1].loglog(h_ref * 1e3, 0.1 * (h_ref / h_ref[0]),
                       "k--", label="O(h) reference", alpha=0.5)
    axes[1].set_xlabel("spacing h (mm)")
    axes[1].set_ylabel("|F(h) - F(h_min)|  (N)")
    axes[1].set_title("Tier 2 stage 4  convergence in h")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/t2_stage4_convergence.png", dpi=140)
    plt.close(fig)
    print(f"\n  Wrote {out_dir}/t2_stage4_convergence.png")
    return {"runs": runs, "orders": orders, "F_asymp": F_asymp,
            "deltas": deltas, "sp_fine": sp_fine}


def baseline_smoke(scene: T2Scene, delta_mm: float = 1.0,
                   kh: float = 2.65e8) -> None:
    """Build point and hydro scenes at one delta, dump active contacts.

    Verifies the baseline builders work, that contacts are emitted with
    sensible counts, and that the aggregate F is in a plausible range
    (compared to the CSLC F at the same delta).
    """
    print("\n" + "=" * 72)
    print(f"  Tier 2 stage 3 smoke test: point + hydro baselines at delta = {delta_mm} mm")
    print(f"  hydro kh = {kh:.3g} Pa")
    print("=" * 72)

    for model_name, build_fn in [
        ("point", lambda s: build_t2_model_point(s)),
        ("hydro", lambda s: build_t2_model_hydro(s, kh=kh)),
    ]:
        print(f"\n  ---- {model_name} ----")
        model = build_fn(scene)
        inspect_model(model, model_name)
        contacts = model.contacts()
        res = measure_one_baseline(model, contacts, scene,
                                   delta_indenter=delta_mm * 1e-3,
                                   indenter_body_idx=1)
        print(f"    F_total = {res['F_total']:.4f} N   "
              f"n_active = {res['n_active']}   "
              f"max contact force = "
              f"{(res['contact_force'].max() if len(res['contact_force']) else 0):.4f} N")
        # Distance distribution of active contacts
        if res["n_active"] > 0:
            r = np.linalg.norm(res["contact_pos_world"][:, :2], axis=1)
            print(f"    contact radial r (mm): min = {r.min()*1e3:.3f}, "
                  f"max = {r.max()*1e3:.3f}, mean = {r.mean()*1e3:.3f}")


def main() -> None:
    wp.init()
    np.set_printoptions(precision=6, suppress=True)
    deltas = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
              1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    out_dir = "cslc_v1/validation/figures"

    # Smoke-test the point and hydro baselines before any sweep.
    baseline_smoke(T2Scene(), delta_mm=1.0)

    # Stage 1: CSLC F-vs-delta sweep at both regimes -- re-run to keep
    # the figures in sync with any code changes in the stage-3 work.
    print("\n" + "#" * 72)
    print("# STAGE 1: CSLC F-vs-delta sweep (regimes A and B)")
    print("#" * 72)
    res_a = sweep_delta(T2Scene(cslc_ka=15000.0, cslc_kl=500.0), deltas)
    res_b = sweep_delta(T2Scene(cslc_ka=1000.0, cslc_kl=1000.0), deltas)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for res, label, marker in [(res_a, "regime A (paper)", "o"),
                                (res_b, "regime B (coupled)", "s")]:
        axes[0].plot(res["deltas"] * 1e3, res["F"], "-" + marker, label=label)
        axes[1].loglog(res["deltas"] * 1e3, res["F"], "-" + marker, label=label)
    ref = res_a
    ref_F = ref["F"][len(ref["F"]) // 2]
    ref_d = ref["deltas"][len(ref["F"]) // 2]
    axes[1].loglog(ref["deltas"] * 1e3, ref_F * (ref["deltas"] / ref_d) ** 1.5,
                   "--", label="Hertzian F ~ delta^{3/2}", alpha=0.5)
    axes[1].loglog(ref["deltas"] * 1e3, ref_F * (ref["deltas"] / ref_d) ** 1.0,
                   ":", label="linear F ~ delta", alpha=0.5)
    for ax, kind in zip(axes, ("linear", "log-log")):
        ax.set_xlabel("indenter penetration delta (mm)")
        ax.set_ylabel("aggregate F (N)")
        ax.set_title(f"Tier 2  F vs delta  ({kind})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("CSLC F-vs-delta at two stiffness regimes")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/t2_stage1_F_vs_delta_both.png", dpi=140)
    plt.close(fig)

    # Stage 3: pressure profile comparison at three representative deltas,
    # at BOTH regimes.
    print("\n" + "#" * 72)
    print("# STAGE 3: PRESSURE PROFILE COMPARISON (CSLC vs point vs hydroelastic)")
    print("#" * 72)
    for regime_label, ka, kl in [("A", 15000.0, 500.0), ("B", 1000.0, 1000.0)]:
        scene = T2Scene(cslc_ka=ka, cslc_kl=kl)
        for d_mm in [0.5, 1.0, 2.0, 4.0]:
            print(f"\n  >>> regime {regime_label}, delta = {d_mm} mm <<<")
            res = pressure_profile_comparison(
                scene, delta_mm=d_mm, kh_hydro=2.65e8, out_dir=out_dir,
                regime_label=regime_label,
            )

    # Stage 4: H2.4 convergence in N (sphere count).  At fixed pad and
    # indenter, sweep spacing -> finer lattice -> measure F convergence
    # to the asymptote (treated as the smallest-spacing run).
    n_convergence_sweep(
        spacings_mm=[12.5, 8.0, 6.25, 5.0, 4.0, 3.125, 2.5],
        deltas_mm=[0.5, 1.0, 2.0, 4.0],
        out_dir=out_dir,
        ka=15000.0, kl=500.0,
    )

if __name__ == "__main__":
    main()
