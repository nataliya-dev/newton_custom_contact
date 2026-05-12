"""Disturbance sweep, with PROPER cf=1.0 for book CSLC calibration."""
import sys, math
sys.path.insert(0, "/home/nataliya/newton_custom_contact")
import warp as wp; wp.config.quiet = True
import numpy as np, newton
from cslc_v1 import squeeze_test as st
from cslc_v1.common import make_solver, apply_external_wrench, recalibrate_cslc_kc_per_pad

def build_sphere_pad_box_book(p):
    b = newton.ModelBuilder()
    pad_cfg = st._pad_cfg(p); target_cfg = st._target_cfg(p, "point")
    target_z = st._target_z(p)
    pad_radius = p.pad_hx
    lx = -(p.pad_gap_initial/2 + pad_radius); rx = +(p.pad_gap_initial/2 + pad_radius)
    left  = b.add_body(xform=wp.transform((lx, 0, target_z), wp.quat_identity()), is_kinematic=True, label="left_pad")
    right = b.add_body(xform=wp.transform((rx, 0, target_z), wp.quat_identity()), is_kinematic=True, label="right_pad")
    b.add_shape_sphere(left,  radius=pad_radius, cfg=pad_cfg)
    b.add_shape_sphere(right, radius=pad_radius, cfg=pad_cfg)
    target = b.add_body(xform=wp.transform((0,0,target_z), wp.quat_identity()), label="book")
    b.add_shape_box(target, hx=p.book_hx, hy=p.book_hy, hz=p.book_hz, cfg=target_cfg)
    m = b.finalize(); m.set_gravity(p.gravity); return m

def metrics_for(model, p, disturbance, recalibrate=False):
    solver = make_solver(model, "mujoco")
    if recalibrate:
        recalibrate_cslc_kc_per_pad(model, contact_fraction=p.cslc_contact_fraction)
    s0, s1 = model.state(), model.state()
    ctrl = model.control(); con = model.contacts()
    z_hist, q_hist = [], []
    sphere_body_idx = 2
    for step in range(p.n_total_steps):
        st.set_kinematic_pads(s0, step, p); s0.clear_forces()
        if step >= p.n_squeeze_steps and disturbance is not None:
            apply_external_wrench(s0, sphere_body_idx,
                                  force=disturbance.get("force", (0,0,0)),
                                  torque=disturbance.get("torque", (0,0,0)))
        model.collide(s0, con)
        solver.step(s0, s1, ctrl, con, p.dt); wp.synchronize()
        s0, s1 = s1, s0
        q = s0.body_q.numpy()
        z_hist.append(float(q[2, 2]))
        q_hist.append(q[2, 3:7].copy())
    n = int(con.rigid_contact_count.numpy()[0])
    sh0 = con.rigid_contact_shape0.numpy()[:n]
    z_hold_start = z_hist[p.n_squeeze_steps]; z_final = z_hist[-1]
    drop_full_mm = (z_hist[0] - min(z_hist)) * 1000
    drop_hold_mm = (z_hold_start - z_final) * 1000
    n_hold = len(z_hist) - p.n_squeeze_steps
    half = p.n_squeeze_steps + n_hold//2
    creep = (z_hist[-1] - z_hist[half]) / ((len(z_hist)-1-half) * p.dt) * (-1) * 1000
    tilts = []
    for q in q_hist[p.n_squeeze_steps:]:
        sin_h = min(1.0, (q[0]**2 + q[1]**2 + q[2]**2)**0.5)
        tilts.append(2.0 * math.degrees(math.asin(sin_h)))
    return {"contacts": int((sh0 != -1).sum()),
            "drop_full": drop_full_mm, "drop_hold": drop_hold_mm,
            "creep": creep, "max_tilt": max(tilts) if tilts else 0.0}

def make_p(book_as_mesh=False):
    p = st.SceneParams()
    p.object_kind = "book"
    p.book_as_mesh = book_as_mesh
    p.pad_gap_initial = 2.0 * (p.book_hx - 0.0015)
    p.cslc_contact_fraction = 1.0          # full pad-on-cover overlap
    pad_face_area = (2.0 * p.pad_hy) * (2.0 * p.pad_hz)
    # H1-style fair calibration: kh_eff = kh/2 (both bodies hydroelastic
    # at the same modulus → harmonic mean), so kh = 2·ke/A_patch.
    p.kh = 2.0 * p.ke / pad_face_area
    return p

disturbances = [
    ("none",         None),
    ("τ_y = 1 N·m",  {"torque": (0, 1.0, 0)}),
    ("τ_y = 5 N·m",  {"torque": (0, 5.0, 0)}),
    ("τ_x = 5 N·m",  {"torque": (5.0, 0, 0)}),
    ("τ_z = 5 N·m",  {"torque": (0, 0, 5.0)}),
    ("F_z = -2 N",   {"force":  (0, 0, -2.0)}),    # pull down 45% of weight
]

print(f"\n{'='*108}")
print("  DISTURBANCE SWEEP — book scene, μ=0.5, weight=4.42 N, cf=1.0 (book), kh patch-matched")
print(f"{'='*108}\n")

scenes = [
    ("BOX book + BOX pads (default — paper baseline)",  False, ("point", "cslc", "hydro")),
    ("MESH book + BOX pads (--book-as-mesh)",            True, ("point", "cslc", "hydro")),
    ("BOX book + SPHERE pads (true 1-pt/pad baseline)",  None, ("point",)),
]

for scene_label, mesh_or_sphere_pads, models in scenes:
    print(f"  {scene_label}")
    print(f"  {'-'*104}")
    print(f"    {'disturbance':<14} {'model':<10} {'contacts':>9} {'drop_full[mm]':>14} {'drop_hold[mm]':>14} {'creep[mm/s]':>13} {'maxTilt[°]':>11}")
    for dist_label, dist in disturbances:
        for model_name in models:
            p = make_p(book_as_mesh=(mesh_or_sphere_pads is True))
            try:
                if mesh_or_sphere_pads is None:
                    model = build_sphere_pad_box_book(p)
                else:
                    model = st.BUILDERS[model_name](p)
                r = metrics_for(model, p, dist, recalibrate=(model_name == "cslc"))
                print(f"    {dist_label:<14} {model_name:<10} {r['contacts']:>9d} {r['drop_full']:>+14.3f} {r['drop_hold']:>+14.3f} {r['creep']:>+13.3f} {r['max_tilt']:>+11.2f}")
            except Exception as e:
                print(f"    {dist_label:<14} {model_name:<10} ERROR: {type(e).__name__}: {str(e)[:60]}")
    print()
