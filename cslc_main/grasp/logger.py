# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-run CSV logger.

The logger owns one output directory and writes several CSV streams in
parallel as the sim runs.  Each stream is opened once at
``__init__`` and closed by ``close()``; rows are flushed every
``log_every`` steps so a crash mid-run still leaves usable partial
output on disk.

Files produced::

    config.json        Frozen ``GraspConfig`` snapshot at run start
                       (reproducibility — the only artifact you need to
                       re-create the run).
    timeseries.csv     Per-step kinematics: step, t, phase, dx, dz,
                       object_xyz, object_vxvyvz, pad_z (both sides),
                       n_contacts.
    cslc_state.csv     Per-step CSLC stats (CSLC mode only):
                       step, t, n_active, n_surface, max_delta_mm,
                       max_pen_mm, mean_delta_mm, mean_pen_active_mm.

Later expansions (per-phase summary, bulging diagnostic, contact-force
totals) can be added as new streams without touching the call sites.
"""

from __future__ import annotations

import csv
from io import TextIOBase
from pathlib import Path

import numpy as np

from .params import GraspConfig


class CSVLogger:
    """One run = one output directory + a few CSV streams."""

    def __init__(self, config: GraspConfig):
        self.config = config
        self.run_dir: Path = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = max(1, config.logging.log_every)

        # Freeze config to JSON so the run is reproducible.
        config.freeze_to_json(self.run_dir / "config.json")

        # Timeseries — always logged.
        self._ts_file: TextIOBase = open(
            self.run_dir / "timeseries.csv", "w", newline=""
        )
        self._ts_writer = csv.DictWriter(
            self._ts_file,
            fieldnames=[
                "step", "t", "phase", "dx_mm", "dz_mm",
                "obj_x", "obj_y", "obj_z",
                "obj_vx", "obj_vy", "obj_vz",
                "left_pad_z", "right_pad_z",
                "n_contacts",
            ],
        )
        self._ts_writer.writeheader()

        # CSLC state — only when CSLC is active.
        self._cslc_file: TextIOBase | None = None
        self._cslc_writer: csv.DictWriter | None = None
        if config.contact_model == "cslc":
            self._cslc_file = open(self.run_dir / "cslc_state.csv", "w", newline="")
            self._cslc_writer = csv.DictWriter(
                self._cslc_file,
                fieldnames=[
                    "step", "t",
                    "n_active", "n_surface",
                    "max_delta_mm", "max_pen_mm",
                    "mean_delta_mm", "mean_pen_active_mm",
                ],
            )
            self._cslc_writer.writeheader()

        # Track previous object position so we can finite-difference velocity.
        # (MuJoCo GPU body_qd is sometimes stale relative to body_q.)
        self._prev_obj_xyz: tuple[float, float, float] | None = None

    # ── Step logging ───────────────────────────────────────────────

    def log_step(
        self,
        step: int,
        t: float,
        phase: str,
        dx: float,
        dz: float,
        obj_xyz: tuple[float, float, float],
        left_pad_z: float,
        right_pad_z: float,
        n_contacts: int,
        cslc_state: dict | None,
        dt: float,
    ) -> None:
        """Append one row to each open CSV stream.

        ``cslc_state`` is the dict returned by :func:`read_cslc_state` —
        passed in by the runner so the logger doesn't depend on the
        handler directly.  Pass ``None`` outside CSLC mode.
        """
        if step % self.log_every != 0:
            return

        if self._prev_obj_xyz is None:
            vx = vy = vz = 0.0
        else:
            vx = (obj_xyz[0] - self._prev_obj_xyz[0]) / dt
            vy = (obj_xyz[1] - self._prev_obj_xyz[1]) / dt
            vz = (obj_xyz[2] - self._prev_obj_xyz[2]) / dt
        self._prev_obj_xyz = tuple(obj_xyz)

        self._ts_writer.writerow(
            {
                "step": step,
                "t": f"{t:.6f}",
                "phase": phase,
                "dx_mm": f"{dx * 1e3:.4f}",
                "dz_mm": f"{dz * 1e3:.4f}",
                "obj_x": f"{obj_xyz[0]:.6f}",
                "obj_y": f"{obj_xyz[1]:.6f}",
                "obj_z": f"{obj_xyz[2]:.6f}",
                "obj_vx": f"{vx:.6f}",
                "obj_vy": f"{vy:.6f}",
                "obj_vz": f"{vz:.6f}",
                "left_pad_z": f"{left_pad_z:.6f}",
                "right_pad_z": f"{right_pad_z:.6f}",
                "n_contacts": n_contacts,
            }
        )

        if self._cslc_writer is not None and cslc_state is not None:
            self._cslc_writer.writerow(
                {
                    "step": step,
                    "t": f"{t:.6f}",
                    "n_active": cslc_state.get("n_active", 0),
                    "n_surface": cslc_state.get("n_surface", 0),
                    "max_delta_mm": f"{cslc_state.get('max_delta_mm', 0.0):.4f}",
                    "max_pen_mm": f"{cslc_state.get('max_pen_mm', 0.0):.4f}",
                    "mean_delta_mm": f"{cslc_state.get('mean_delta', 0.0) * 1e3:.4f}",
                    "mean_pen_active_mm": f"{cslc_state.get('mean_pen_active', 0.0) * 1e3:.4f}",
                }
            )

    def close(self) -> None:
        for f in (self._ts_file, self._cslc_file):
            if f is not None:
                f.close()
        self._ts_file = None  # type: ignore[assignment]
        self._cslc_file = None


# ── Helper: read CSLC state from a model ────────────────────────────


def read_cslc_state(model) -> dict | None:
    """Snapshot the CSLC handler's per-step lattice state.

    Returns a dict the logger can consume directly, or ``None`` if no
    CSLC handler is attached.  Same shape as
    ``cslc_mujoco/common.read_cslc_state``.
    """
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None
    d = handler.cslc_data
    is_surf = d.is_surface.numpy() == 1
    deltas_vec = d.sphere_delta.numpy()[is_surf]
    delta_mags = (
        np.linalg.norm(deltas_vec, axis=-1)
        if deltas_vec.ndim == 2
        else deltas_vec
    )
    pen = handler.raw_penetration.numpy()[is_surf]
    active = pen > 0
    n_active = int(active.sum())
    n_surface = int(is_surf.sum())
    max_delta = float(delta_mags.max()) if len(delta_mags) else 0.0
    max_pen = float(pen.max()) if len(pen) else 0.0
    mean_delta = float(delta_mags.mean()) if len(delta_mags) else 0.0
    mean_pen_active = float(pen[active].mean()) if n_active else 0.0
    return {
        "n_active": n_active,
        "n_surface": n_surface,
        "max_delta_mm": max_delta * 1e3,
        "max_pen_mm": max_pen * 1e3,
        "max_delta": max_delta,
        "max_pen": max_pen,
        "mean_delta": mean_delta,
        "mean_pen_active": mean_pen_active,
    }


def count_active_contacts(contacts) -> int:
    """Count contacts that survived narrow-phase culls.

    A "valid" contact has ``shape0 ≥ 0``; CSLC's hybrid emission policy
    writes ``shape0 = -1`` for slots whose smooth gate fell below 1e-4
    (see ``cslc_kernels.write_cslc_contacts``).
    """
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0
    return int(np.sum(contacts.rigid_contact_shape0.numpy()[:n] >= 0))
