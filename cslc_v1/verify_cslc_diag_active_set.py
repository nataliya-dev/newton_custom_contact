#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CSLC active-set diagnostic against lift_test.

Runs the lift_test gripper scene up to a chosen phase and prints, for each
CSLC pad, how many spheres:
  - actually have 3-D sphere-sphere overlap with the target  (pen_3d > 0)
  - would pass the old 2 mm radial gate                       (radial <= cutoff)
  - were marked active by kernel 1                            (raw_penetration > 0)

Usage
─────
    # snapshot at end of SQUEEZE (no lift yet — steady grip)
    python cslc_diag_active_set.py --until squeeze_end

    # snapshot mid-LIFT
    python cslc_diag_active_set.py --until lift_mid

    # same thing but with the Newton viewer running alongside
    python cslc_diag_active_set.py --until lift_mid --viewer

What to look for AFTER the 2026-04-17 scene retune
──────────────────────────────────────────────────
Scene now targets ~1 mm FACE compression (from 12.5 mm before), so lattice
pen_3d is ~r_lat + face_pen ≈ 6 mm at the patch centre, not ~17 mm.

At SQUEEZE_END you should see, for BOTH pads:
    pen_3d>0: on the order of 10–25
    kernel1 phi>0: equal-ish to (pen_3d>0 minus the edge-corner spheres
                   that fail d_proj>0 — roughly 50–80 % of pen_3d>0)
    max pen_3d: around 5–8 mm (NOT 16 mm!)

Both pads should now show similar kernel1 phi>0 counts, because each pad
has its own raw_penetration buffer.  If pad 2 shows 0 while pad 4 shows a
positive number, the per-pair buffer routing is broken.
"""
from __future__ import annotations

import argparse
import numpy as np
import warp as wp

import newton
import newton.examples
from lift_test import (
    SceneParams,
    build_cslc_scene,
    set_pad_targets,
    make_solver,
    SPHERE_BODY,
)

GEO_SPHERE = 3             # newton.GeoType.SPHERE
CUTOFF_2MM = 2.0e-3        # the old CSLC_RADIAL_CUTOFF, for comparison

# ── Small quaternion helpers (wp.transform layout: [px,py,pz, qx,qy,qz,qw]) ──
def _qrot(q, v):
    xyz = np.array([q[0], q[1], q[2]])
    t = 2.0 * np.cross(xyz, v)
    return v + q[3] * t + np.cross(xyz, t)

def _xf_pt (xf, p): return _qrot(xf[3:7], p) + xf[0:3]
def _xf_vec(xf, v): return _qrot(xf[3:7], v)


def analyze_state(model, state, tag: str = "") -> None:
    """Walk every surface lattice sphere in numpy; report gate statistics.

    For each CSLC pad, prints:
      pen_3d>0          — spheres with any real 3-D overlap with the target
      radial<=2mm       — spheres the OLD 2 mm radial gate would have kept
      OLD-gate AND      — spheres both gates agreed on (the baseline's
                          "1 contact per pad" degenerate active set)
      kernel1 phi>0     — spheres kernel 1 actually accepted for this pad
                          (read from the pair's own raw_penetration buffer,
                          so this is NOT confused by whichever pair happened
                          to launch last)
      max pen_3d        — worst-case lattice overlap across the face
    """
    handler    = model._collision_pipeline.cslc_handler
    d          = handler.cslc_data

    pos_local  = d.positions.numpy()
    normals    = d.outward_normals.numpy()
    is_surf    = d.is_surface.numpy()
    shape_id   = d.sphere_shape.numpy()
    r_lat_all  = d.radii.numpy()

    bodyq      = state.body_q.numpy()
    shape_xf   = model.shape_transform.numpy()
    shape_body = model.shape_body.numpy()
    shape_type = model.shape_type.numpy()
    shape_sc   = model.shape_scale.numpy()

    # Locate the target sphere shape.  In lift_test the target body index is
    # SPHERE_BODY (imported from lift_test).  It has a SPHERE geometry type.
    sph_idx = None
    for i in range(model.shape_count):
        if int(shape_type[i]) == GEO_SPHERE and int(shape_body[i]) == SPHERE_BODY:
            sph_idx = i
            break
    if sph_idx is None:
        raise RuntimeError("Could not find target sphere shape — scene layout changed?")

    R_tgt   = float(shape_sc[sph_idx][0])
    Xtb     = bodyq[int(shape_body[sph_idx])]
    t_world = _xf_pt(Xtb, shape_xf[sph_idx][0:3])

    print(f"\n══ {tag}  target sphere z={t_world[2]:.4f} m,  R={R_tgt*1e3:.1f} mm ══")
    for pad in np.unique(shape_id):
        tids = np.where((shape_id == pad) & (is_surf == 1))[0]
        Xws  = shape_xf[pad]
        Xwb  = bodyq[int(shape_body[pad])]

        # Each pad has its own raw_penetration scratch.  Fetch this pad's
        # buffer so we compare against what kernel 1 ACTUALLY wrote for it,
        # not whatever pair happened to launch last.
        phi_buf = handler.get_phi_for_cslc_shape(int(pad))
        phi_pad = phi_buf.numpy() if phi_buf is not None else np.zeros(len(is_surf))

        n_pen3d = n_rad = n_phi = n_both = 0
        max_pen3d = -1e9
        best_tid  = -1
        for t in tids:
            q_world = _xf_pt(Xwb, _xf_pt(Xws, pos_local[t]))
            n_world = _xf_vec(Xwb, _xf_vec(Xws, normals[t]))
            diff    = t_world - q_world
            dist    = float(np.linalg.norm(diff))
            dproj   = float(np.dot(diff, n_world))
            radial  = (max(dist * dist - dproj * dproj, 0.0)) ** 0.5
            pen3d   = (r_lat_all[t] + R_tgt) - dist

            n_pen3d += (pen3d > 0)
            n_rad   += (radial <= CUTOFF_2MM)
            n_phi   += (phi_pad[t] > 0)
            n_both  += (pen3d > 0 and radial <= CUTOFF_2MM)
            if pen3d > max_pen3d:
                max_pen3d = pen3d
                best_tid  = t

        print(f"  pad shape {pad:2d}  N_surf={len(tids):4d}  "
              f"pen_3d>0: {n_pen3d:4d}   "
              f"radial<=2mm: {n_rad:4d}   "
              f"OLD-gate AND: {n_both:4d}   "
              f"kernel1 phi>0: {n_phi:4d}   "
              f"max pen_3d: {max_pen3d*1e3:6.2f} mm   best_tid: {best_tid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true",
                    help="open the Newton 3-D viewer while stepping")
    ap.add_argument("--solver", default="mujoco", choices=["mujoco", "semi"])
    ap.add_argument("--until",  default="squeeze_end",
                    choices=["approach_end", "squeeze_end", "lift_mid", "lift_end"])
    args, _ = ap.parse_known_args()

    wp.init()
    p = SceneParams()
    p.dump()

    # Build scene.  Viewer init must happen before collision pipeline setup
    # so model.contacts() exists before we start stepping.
    viewer = None
    if args.viewer:
        parser = newton.examples.create_parser()
        viewer, _ = newton.examples.init(parser)

    model, dof_map = build_cslc_scene(p)
    contacts  = model.contacts()      # forces collision pipeline init
    solver    = make_solver(model, args.solver, p)
    s0, s1    = model.state(), model.state()
    ctrl      = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    if viewer is not None:
        viewer.set_model(model)

    targets = {
        "approach_end": p.approach_steps,
        "squeeze_end":  p.approach_steps + p.squeeze_steps,
        "lift_mid":     p.approach_steps + p.squeeze_steps + p.lift_steps // 2,
        "lift_end":     p.approach_steps + p.squeeze_steps + p.lift_steps,
    }
    n_run = targets[args.until]

    print(f"\nStepping to phase '{args.until}' = {n_run} steps...")
    for step in range(n_run):
        set_pad_targets(ctrl, step, p, dof_map)
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, p.dt)
        wp.synchronize()
        if viewer is not None and (step % 30 == 0):
            viewer.begin_frame(step * p.dt)
            viewer.log_state(s0)
            viewer.log_contacts(contacts, s0)
            viewer.end_frame()
        s0, s1 = s1, s0

    # One final collide so handler buffers match the current s0 body_q.
    s0.clear_forces()
    model.collide(s0, contacts)
    wp.synchronize()
    analyze_state(model, s0, tag=args.until.upper())

    # Keep the viewer responsive for 2 s so you can look around.
    if viewer is not None:
        import time
        t0 = time.time()
        while time.time() - t0 < 2.0:
            viewer.begin_frame(n_run * p.dt)
            viewer.log_state(s0)
            viewer.log_contacts(contacts, s0)
            viewer.end_frame()


if __name__ == "__main__":
    main()