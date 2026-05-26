"""Quick probe for the CSLC production failure handoff.

Runs the grasp scene for a small number of steps, capturing the
lattice state at key points so we can localise the bug (warm-start vs.
Jacobi vs. emission vs. solver feedback).

Run: ``uv run python -m cslc_main.grasp.scripts.probe_failure --object-kind box``
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

import newton

from cslc_main.grasp.params import GraspConfig
from cslc_main.grasp.scene import build_scene
from cslc_main.grasp.solvers import make_solver
from cslc_main.grasp.runner import _attach_qfrc_actuator, _simulate_one_step
from cslc_main.grasp.logger import read_cslc_state, count_active_contacts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object-kind", choices=["sphere", "box"], default="box")
    p.add_argument("--n-steps", type=int, default=250)
    p.add_argument("--every", type=int, default=25)
    args = p.parse_args()

    wp.init()

    cfg = GraspConfig()
    cfg.object.kind = args.object_kind
    # Avoid heavy postsim plots.
    cfg.logging.save_postsim_plots = False
    cfg.logging.save_lattice_preview = False
    cfg.logging.use_timestamp = False
    cfg.logging.run_label = "probe_failure"

    artifacts = build_scene(cfg)
    model = artifacts.model
    solver = make_solver(model, cfg.solver)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    _attach_qfrc_actuator((state_0, state_1), model)

    # Warm-up.
    _simulate_one_step(model, solver, control, contacts,
                       state_0, state_1, 0, cfg, artifacts.dof_map)
    wp.synchronize()

    handler = model._collision_pipeline.cslc_handler
    print(f"n_pair_blocks={handler.n_pair_blocks}, n_surface_contacts={handler.n_surface_contacts}")
    print("pair[0]: K_max=", handler.shape_pairs[0].K_max,
          "target_count=", handler.shape_pairs[0].target_count,
          "other_ke=", handler.shape_pairs[0].other_ke)
    print("ka=", float(handler.cslc_data.ka),
          "kl=", float(handler.cslc_data.kl),
          "kc=", float(handler.cslc_data.kc),
          "eps=", float(handler.cslc_data.smoothing_eps))
    print("n_iter=", handler.n_iter, "alpha=", handler.alpha)

    # Pre-step lattice state.
    cs = read_cslc_state(model)
    print(f"PRE-STEP cslc state: {cs}")

    t0 = time.perf_counter()
    for step in range(1, args.n_steps + 1):
        state_0, state_1, dx, dz = _simulate_one_step(
            model, solver, control, contacts,
            state_0, state_1, step, cfg, artifacts.dof_map,
        )
        if step % args.every == 0 or step in (1, 5, 10):
            cs = read_cslc_state(model)
            n_c = count_active_contacts(contacts)
            ncc = int(contacts.rigid_contact_count.numpy()[0])
            q = state_0.body_q.numpy()
            obj_z = float(q[artifacts.object_body_index, 2])
            obj_x = float(q[artifacts.object_body_index, 0])
            pad_x_left = float(q[artifacts.pad_body_indices["left"], 0])
            phase, _ = cfg.phase_of(step)
            mdelta = cs["max_delta_mm"] if cs else 0.0
            mpen = cs["max_pen_mm"] if cs else 0.0
            print(f" step={step:4d} [{phase:8s}] obj=({obj_x:+.4f},{obj_z:+.4f}) "
                  f"pad_x_L={pad_x_left:+.4f} dx={dx*1e3:+.2f}mm "
                  f"n_active={n_c} n_count={ncc} "
                  f"max_delta={mdelta:.2f}mm max_pen={mpen:.2f}mm")
    wall = time.perf_counter() - t0
    print(f"  WALLCLOCK {wall:.2f}s for {args.n_steps} steps "
          f"= {1000*wall/args.n_steps:.1f} ms/step")


if __name__ == "__main__":
    main()
