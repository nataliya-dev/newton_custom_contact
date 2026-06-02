# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Steady-state per-step wall-clock benchmark: CSLC vs point contact.

Why this exists
---------------
The headless runner prints a ``TIMING per-step`` line, but that number
is **not** a clean per-step cost:

  * its warm-up is a single step at step 0 (APPROACH, no contacts), so
    the contact/lattice kernels JIT-compile *inside* the timed loop;
  * it averages over all four phases (cheap no-contact APPROACH ...
    expensive full-contact HOLD).

This script measures the cost that actually matters for a model
comparison: the **steady-state, contact-active** per-step wall time,
with every kernel pre-compiled and the GPU explicitly synchronised.

Method (per model)
------------------
1. Build the identical grasp scene + solver.
2. Warm up by running the real trajectory through APPROACH + SQUEEZE up
   to the first LIFT step ("grip"): this compiles every kernel and
   presses the pads firmly onto the object so contacts are active.
3. ``wp.synchronize()`` once to drain all outstanding GPU work.
4. Time ``reps`` blocks of ``measure_steps`` steps each, holding the
   step index fixed at the grip configuration (constant PD targets, so
   the contact set stays steady).  Each block ends with a single
   ``wp.synchronize()`` before reading the clock -- this is the only
   valid way to time async GPU work.
5. Report mean +/- std and the minimum (least-noisy) ms/step, the
   throughput in steps/s, and the active-contact count so the numbers
   are interpretable (CSLC's lattice engages more contacts than point).

The minimum across reps is the cleanest estimate of the underlying cost
(OS jitter and background load only ever *add* time).

Usage::

    uv run -m cslc_main.grasp.scripts.benchmark_per_step
    uv run -m cslc_main.grasp.scripts.benchmark_per_step \\
        --pad-kind dome --object-kind sphere --reps 9 --measure-steps 300
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import statistics
import time
from pathlib import Path

import warp as wp

import newton

from ..logger import count_active_contacts
from ..params import GraspConfig
from ..runner import _attach_qfrc_actuator, _simulate_one_step
from ..scene import build_scene
from ..solvers import make_solver


@contextlib.contextmanager
def _silence(quiet: bool):
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


def benchmark_model(
    model_name: str,
    pad_kind: str,
    object_kind: str,
    *,
    measure_steps: int,
    reps: int,
    quiet: bool = True,
) -> dict:
    """Measure steady-state per-step wall time for one contact model.

    Returns a result dict with per-step timing statistics [ms], the
    active-contact count at the measured configuration, and the held
    object's Z [m] (a sanity check that it is still gripped).
    """
    config = GraspConfig()
    config.contact_model = model_name
    config.pad.kind = pad_kind
    config.object.kind = object_kind

    dt = config.timing.dt
    approach_steps = int(config.timing.approach_duration / dt)
    squeeze_steps = int(config.timing.squeeze_duration / dt)
    # First LIFT step: pads fully closed at the commanded squeeze depth,
    # object gripped, all kernels exercised.  PD targets are constant
    # from here on the X (closing) axis.
    grip_step = approach_steps + squeeze_steps

    with _silence(quiet):
        artifacts = build_scene(config)
        model = artifacts.model
        solver = make_solver(model, config.solver)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        _attach_qfrc_actuator((state_0, state_1), model)

        # ── Warm-up: run the real trajectory to the grip configuration.
        # Compiles every kernel and presses the pads onto the object.
        for step in range(grip_step + 1):
            state_0, state_1, _, _ = _simulate_one_step(
                model, solver, control, contacts,
                state_0, state_1, step, config, artifacts.dof_map,
            )
        wp.synchronize()  # drain all warm-up / JIT work before timing

        # Sanity readback at the measured configuration.
        n_contacts = count_active_contacts(contacts)
        object_z = float(state_0.body_q.numpy()[artifacts.object_body_index, 2])

        # ── Timed reps, step index pinned at grip (constant targets). ──
        ms_per_step: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(measure_steps):
                state_0, state_1, _, _ = _simulate_one_step(
                    model, solver, control, contacts,
                    state_0, state_1, grip_step, config, artifacts.dof_map,
                )
            wp.synchronize()  # the only valid point to stop the clock
            elapsed = time.perf_counter() - t0
            ms_per_step.append(1000.0 * elapsed / measure_steps)

    mean_ms = statistics.mean(ms_per_step)
    std_ms = statistics.stdev(ms_per_step) if len(ms_per_step) > 1 else 0.0
    min_ms = min(ms_per_step)
    return {
        "contact_model": model_name,
        "pad_kind": pad_kind,
        "object_kind": object_kind,
        "active_contacts": n_contacts,
        "object_z_m": object_z,
        "measure_steps": measure_steps,
        "reps": reps,
        "mean_ms_per_step": mean_ms,
        "std_ms_per_step": std_ms,
        "min_ms_per_step": min_ms,
        "steps_per_s_mean": 1000.0 / mean_ms,
    }


def main() -> None:
    default_out = (Path(__file__).resolve().parents[1]
                   / "outputs" / "benchmark" / "per_step.csv")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", default="cslc,point",
                        help="Comma-separated contact models to benchmark.")
    parser.add_argument("--pad-kind", default="box", choices=["box", "dome"])
    parser.add_argument("--object-kind", default="sphere",
                        choices=["sphere", "box"])
    parser.add_argument("--measure-steps", type=int, default=300,
                        help="Steps timed per rep (steady-state window).")
    parser.add_argument("--reps", type=int, default=7,
                        help="Timed blocks; the minimum is the cleanest estimate.")
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--verbose", action="store_true",
                        help="Leave the scene-build prints on.")
    args = parser.parse_args()

    wp.init()
    models = [m.strip() for m in args.models.split(",")]

    print(f"Steady-state per-step benchmark  (pad={args.pad_kind}, "
          f"object={args.object_kind}, {args.reps} reps x "
          f"{args.measure_steps} steps)\n")

    results = []
    for mdl in models:
        print(f"  benchmarking {mdl} ...", flush=True)
        r = benchmark_model(
            mdl, args.pad_kind, args.object_kind,
            measure_steps=args.measure_steps, reps=args.reps,
            quiet=not args.verbose,
        )
        results.append(r)
        print(f"    {r['mean_ms_per_step']:.3f} +/- {r['std_ms_per_step']:.3f} "
              f"ms/step  (min {r['min_ms_per_step']:.3f})  "
              f"{r['steps_per_s_mean']:.0f} steps/s  "
              f"active_contacts={r['active_contacts']}  "
              f"obj_z={r['object_z_m'] * 1e3:.1f}mm")

    # ── Summary table ──
    print("\n  model   ms/step (mean+/-std)   min     steps/s   contacts")
    print("  " + "-" * 58)
    for r in results:
        print(f"  {r['contact_model']:6s}  "
              f"{r['mean_ms_per_step']:6.3f} +/- {r['std_ms_per_step']:5.3f}     "
              f"{r['min_ms_per_step']:6.3f}  {r['steps_per_s_mean']:7.0f}   "
              f"{r['active_contacts']:6d}")
    # Pairwise ratio when exactly cslc vs point are present.
    by = {r["contact_model"]: r for r in results}
    if "cslc" in by and "point" in by:
        ratio = by["cslc"]["min_ms_per_step"] / by["point"]["min_ms_per_step"]
        print(f"\n  CSLC is {ratio:.2f}x the per-step cost of point "
              f"(min ms/step, same scene).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  Wrote {args.output}")


if __name__ == "__main__":
    main()
