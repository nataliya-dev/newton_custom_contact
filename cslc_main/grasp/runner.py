# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless runner and viewer-mode ``Example`` class.

Both run paths share the same per-step body of work (apply pad
targets, collide, step, log) — packaged in :func:`_simulate_one_step`.
The headless runner consumes that and the metrics in a tight loop; the
viewer's ``Example`` exposes ``step``/``render``/``test_final`` so the
``newton.examples.run`` driver can drive it at 60 fps with 8× substeps.
"""

from __future__ import annotations

import time
import types
from typing import Any

import numpy as np
import warp as wp

import newton

from . import trajectory
from .logger import CSVLogger, count_active_contacts, read_cslc_state
from .metrics import Metrics
from .params import GraspConfig
from .scene import SceneArtifacts, build_scene
from .solvers import make_solver
from .visualization import (
    LatticeRenderer,
    StatsPanel,
    TargetPointsRenderer,
    save_lattice_preview,
    save_postsim_plots,
)


# ── Wrench instrumentation ─────────────────────────────────────────────


def _attach_qfrc_actuator(states: tuple, model) -> None:
    """Pre-allocate ``state.mujoco.qfrc_actuator`` on each state.

    Benchmark §7.1 / Appendix A wrench-readout pattern.  MuJoCo's
    solver writes the per-DOF joint-actuator force into
    ``state.mujoco.qfrc_actuator`` only if the field exists at
    ``solver.step`` call time.  The conventional
    ``Contacts.rigid_contact_force`` buffer is allocated by Newton
    but NEVER populated by MuJoCo's solver path — qfrc_actuator is
    the only working route for per-pad real contact wrench.

    Safe to call unconditionally: on non-MuJoCo solvers the buffer
    is allocated but ignored.  Adds ~``model.joint_dof_count *
    4 bytes`` of GPU memory per state (negligible).
    """
    for state in states:
        state.mujoco = types.SimpleNamespace()
        state.mujoco.qfrc_actuator = wp.zeros(
            model.joint_dof_count, dtype=wp.float32, device=model.device,
        )


# ── Shared sim step ─────────────────────────────────────────────────────


def _simulate_one_step(
    model,
    solver,
    control,
    contacts,
    state_0,
    state_1,
    step: int,
    config: GraspConfig,
    dof_map: dict[str, int],
) -> tuple[Any, Any, float, float]:
    """Advance the simulation by one ``dt`` and return ``(state_a, state_b, dx, dz)``."""
    dx, dz = trajectory.set_pad_targets(control, step, config, dof_map)
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, config.timing.dt)
    return state_1, state_0, dx, dz


def _gather_step_logs(
    artifacts: SceneArtifacts,
    state,
    contacts,
    cslc_state: dict | None,
) -> tuple[tuple[float, float, float], float, float, int]:
    """Read everything the logger needs from the current state."""
    q = state.body_q.numpy()
    obj = artifacts.object_body_index
    obj_xyz = (float(q[obj, 0]), float(q[obj, 1]), float(q[obj, 2]))
    left_pad_z = float(q[artifacts.pad_body_indices["left"], 2])
    right_pad_z = float(q[artifacts.pad_body_indices["right"], 2])
    n_contacts = count_active_contacts(contacts)
    return obj_xyz, left_pad_z, right_pad_z, n_contacts


# ── Headless ────────────────────────────────────────────────────────────


def run_headless(config: GraspConfig) -> Metrics:
    """Run a complete grasp test without a viewer; returns Metrics.

    Side-effects: writes CSVs and PNGs under ``config.run_dir()``.
    """
    print(config.dump_summary())
    artifacts = build_scene(config)
    model = artifacts.model
    solver = make_solver(model, config.solver)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    _attach_qfrc_actuator((state_0, state_1), model)

    logger = CSVLogger(config)
    if config.logging.save_lattice_preview:
        save_lattice_preview(artifacts, logger.run_dir / "pad_lattice.png")

    metrics = Metrics(name=config.run_dir_name())
    t_step = config.timing.dt

    # Warm-up pass — primes kernel JIT + solver, so the timed loop sees
    # steady-state cost.  Output state intentionally discarded.
    _simulate_one_step(
        model, solver, control, contacts,
        state_0, state_1, 0, config, artifacts.dof_map,
    )
    wp.synchronize()

    t0 = time.perf_counter()
    # Per-window timing: ms/step over the last `print_every` steps.
    # Wall-clock between print boundaries divided by the step count;
    # picks up real per-step cost variation across phases (APPROACH
    # has no contacts, HOLD has full contact load).
    print_every = 200
    t_window_start = time.perf_counter()
    for step in range(config.total_steps):
        state_0, state_1, dx, dz = _simulate_one_step(
            model, solver, control, contacts,
            state_0, state_1, step, config, artifacts.dof_map,
        )

        obj_xyz, left_pad_z, right_pad_z, n_contacts = _gather_step_logs(
            artifacts, state_0, contacts, None
        )
        cslc_state = read_cslc_state(model) if config.contact_model == "cslc" else None
        phase, _ = config.phase_of(step)
        logger.log_step(
            step, step * t_step, phase, dx, dz, obj_xyz,
            left_pad_z, right_pad_z, n_contacts, cslc_state, t_step,
        )

        metrics.object_z.append(obj_xyz[2])
        if len(metrics.object_z) == 1:
            metrics._x0 = obj_xyz[0]  # type: ignore[attr-defined]
            metrics._y0 = obj_xyz[1]  # type: ignore[attr-defined]
        metrics.object_xy_drift.append(
            float(np.hypot(obj_xyz[0] - metrics._x0, obj_xyz[1] - metrics._y0))  # type: ignore[attr-defined]
        )
        metrics.contacts.append(n_contacts)

        if (step + 1) % print_every == 0 or step == config.total_steps - 1:
            extra = ""
            if cslc_state is not None:
                extra = (
                    f"  cslc={cslc_state['n_active']}/{cslc_state['n_surface']}  "
                    f"max_δ={cslc_state['max_delta_mm']:.2f}mm"
                )
            wp.synchronize()  # one sync per print, so per-window ms reflects GPU work
            now = time.perf_counter()
            window_steps = (((step + 1) % print_every) or print_every)
            window_ms = 1000.0 * (now - t_window_start) / window_steps
            t_window_start = now
            print(
                f"  step={step + 1:5d}/{config.total_steps}  "
                f"[{phase:8s}]  obj_z={obj_xyz[2]:+.5f}  "
                f"pad_z={left_pad_z:+.4f}  n={n_contacts}{extra}  "
                f"({window_ms:.2f} ms/step)"
            )

    wall = time.perf_counter() - t0
    rtx = config.total_steps * t_step / max(wall, 1e-9)
    print(
        f"  TIMING wall={wall:.3f}s  per-step={1000 * wall / config.total_steps:.3f}ms  "
        f"realtime×={rtx:.2f}"
    )

    logger.close()
    if config.logging.save_postsim_plots:
        save_postsim_plots(logger.run_dir)

    _print_summary(metrics)
    return metrics


def _print_summary(m: Metrics) -> None:
    print(
        f"  RESULT  max_z={m.max_z:.4f}  final_z={m.final_z:.4f}  "
        f"lifted={'YES' if m.lifted else 'NO'}  held={'YES' if m.held else 'NO'}  "
        f"xy_slip_max={m.xy_slip_max * 1e3:.2f}mm"
    )


# ── Viewer-mode Example ─────────────────────────────────────────────────


class Example:
    """Viewer-mode driver; compatible with ``newton.examples.run``.

    Substeps the physics dt down from the viewer's frame interval so
    rendering stays at 60 fps while the simulation runs at 500 Hz (or
    whatever ``GraspConfig.timing.dt`` is set to).
    """

    def __init__(self, viewer, args, config: GraspConfig):
        self.viewer = viewer
        self.config = config
        self.test_mode = bool(getattr(args, "test", False))

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(round(self.frame_dt / config.timing.dt)))
        # Actual substep dt may drift slightly from config.timing.dt to
        # ensure integer substeps per frame.  Keep the underlying physics
        # dt — drop frames if necessary, but don't change phase timing.
        self.sim_dt = config.timing.dt

        print(config.dump_summary())
        self.artifacts = build_scene(config)
        self.model = self.artifacts.model
        self.solver = make_solver(self.model, config.solver)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        _attach_qfrc_actuator((self.state_0, self.state_1), self.model)

        self.logger = CSVLogger(config)
        if config.logging.save_lattice_preview:
            save_lattice_preview(self.artifacts, self.logger.run_dir / "pad_lattice.png")

        self.metrics = Metrics(name=config.run_dir_name())
        self.sim_step = 0
        self.sim_time = 0.0
        self.last_object_z = config.object.start_z

        self.lattice = LatticeRenderer(self.model, self.viewer)
        self.target_points = TargetPointsRenderer(self.model, self.viewer)
        self.stats = StatsPanel(config)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.3, -0.3, config.object.start_z + 0.15),
            pitch=-15.0,
            yaw=135.0,
        )
        # Enable hydroelastic contact-surface rendering when the viewer
        # supports it AND the scene built the pipeline with
        # ``output_contact_surface=True`` (see scene.py hydro branch).
        # Safe to set unconditionally — viewer ignores it when the
        # kernels weren't compiled with the surface-output path.
        if config.contact_model == "hydro" and hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = True

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            if self.sim_step >= self.config.total_steps:
                return
            self.state_0, self.state_1, dx, dz = _simulate_one_step(
                self.model, self.solver, self.control, self.contacts,
                self.state_0, self.state_1, self.sim_step,
                self.config, self.artifacts.dof_map,
            )

            obj_xyz, lz, rz, nc = _gather_step_logs(
                self.artifacts, self.state_0, self.contacts, None
            )
            self.last_object_z = obj_xyz[2]
            cslc_state = (
                read_cslc_state(self.model)
                if self.config.contact_model == "cslc"
                else None
            )
            phase, _ = self.config.phase_of(self.sim_step)
            self.logger.log_step(
                self.sim_step,
                self.sim_step * self.sim_dt,
                phase, dx, dz, obj_xyz, lz, rz, nc,
                cslc_state, self.sim_dt,
            )
            self.metrics.object_z.append(obj_xyz[2])
            self.metrics.contacts.append(nc)
            self.stats.update(obj_xyz[2])

            self.sim_step += 1

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.lattice.update(self.state_0)
        self.target_points.update(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui) -> None:
        """Render the side-panel stats overlay.  Invoked by the viewer."""
        self.stats.render(ui)

    def test_final(self) -> None:
        """Regression-test assertion at the end of the run."""
        if self.config.contact_model == "cslc":
            assert self.last_object_z > 0.01, (
                f"Object fell during sim: z={self.last_object_z:.4f}"
            )
        self.logger.close()
        if self.config.logging.save_postsim_plots:
            save_postsim_plots(self.logger.run_dir)
