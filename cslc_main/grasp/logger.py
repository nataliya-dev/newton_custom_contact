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
                    # Repro-A diagnostics (H2: per-pad active count;
                    # H3: apex sphere normal/tangential delta split).
                    "n_active_left", "n_active_right",
                    "apex_left_delta_n_mm", "apex_left_delta_t_mm",
                    "apex_right_delta_n_mm", "apex_right_delta_t_mm",
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
                    "n_active_left": cslc_state.get("n_active_left", 0),
                    "n_active_right": cslc_state.get("n_active_right", 0),
                    "apex_left_delta_n_mm":
                        f"{cslc_state.get('apex_left_delta_n_mm', 0.0):.4f}",
                    "apex_left_delta_t_mm":
                        f"{cslc_state.get('apex_left_delta_t_mm', 0.0):.4f}",
                    "apex_right_delta_n_mm":
                        f"{cslc_state.get('apex_right_delta_n_mm', 0.0):.4f}",
                    "apex_right_delta_t_mm":
                        f"{cslc_state.get('apex_right_delta_t_mm', 0.0):.4f}",
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

    Extended (2026-05-24) with per-pad active counts and per-pad
    apex-sphere delta decomposition (``apex_*_delta_n_mm`` /
    ``apex_*_delta_t_mm`` for the per-pad most-compressed surface
    sphere).  These exist to discriminate H2 (contact-fraction
    mismatch — too few spheres engaged) and H3 (lateral springs
    popping the loaded sphere off-axis) in Repro A; see
    plans/i-need-you-to-jolly-beacon.md.
    """
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None
    d = handler.cslc_data
    is_surf_full = d.is_surface.numpy() == 1
    deltas_full = d.sphere_delta.numpy()
    pen_full = handler.raw_penetration.numpy()
    shape_full = d.sphere_shape.numpy()
    normals_full = d.outward_normals.numpy()

    is_surf = is_surf_full
    deltas_vec = deltas_full[is_surf]
    delta_mags = (
        np.linalg.norm(deltas_vec, axis=-1)
        if deltas_vec.ndim == 2
        else deltas_vec
    )
    pen = pen_full[is_surf]
    active = pen > 0
    n_active = int(active.sum())
    n_surface = int(is_surf.sum())

    # Stricter "real contact" count for kc auto-tune: thresh at the
    # smoothing width eps so the smooth_relu tail doesn't inflate the
    # count.  smooth_relu(0, eps) = eps/2, so any sphere with pen > eps
    # has raw > 0 confidently (force contribution > smoothing leak).
    # Used by :func:`cslc_main.grasp.contact_models.auto_tune_kc_per_step`
    # as the active fraction signal -- ``n_active`` (pen > 0) over-counts
    # by ~10x during APPROACH because every near-contact pad sphere has
    # a smoothing-tail pen > 0.
    eps = float(getattr(handler.cslc_data, "smoothing_eps", 5.0e-4))
    n_active_strict = int((pen > eps).sum())
    max_delta = float(delta_mags.max()) if len(delta_mags) else 0.0
    max_pen = float(pen.max()) if len(pen) else 0.0
    mean_delta = float(delta_mags.mean()) if len(delta_mags) else 0.0
    mean_pen_active = float(pen[active].mean()) if n_active else 0.0

    # Per-pad split.  Assumes 2 pads (the grasp pipeline always does);
    # smaller shape_id = "left", larger = "right".  This matches the
    # left-before-right insertion order in scene.py.
    #
    # Active-count signal: ``handler.raw_penetration`` is overwritten by
    # each pair's compute_cslc_penetration launch, so by the time we read
    # it only the LAST pair's surface spheres have meaningful values
    # (others get zeroed).  Use the per-sphere displacement projected
    # onto the OUTWARD normal instead: negative values = sphere pushed
    # inward = contact compression.  ``sphere_delta`` is per-sphere
    # persistent and reflects both pads correctly.
    unique_shapes = sorted(int(s) for s in np.unique(shape_full[is_surf_full]))
    # |δ·n̂_outward| > 5 µm threshold ≈ pen > 10 µm (since δ_n ≈ phi·kc/(ka+kc)),
    # well above noise floor and below any meaningful contact compression.
    DELTA_INWARD_THRESH_M = 5.0e-6
    pad_state = {}
    for label, shape_id in zip(
        ("left", "right"), (unique_shapes + [-1, -1])[:2]
    ):
        if shape_id < 0:
            pad_state[f"n_active_{label}"] = 0
            pad_state[f"apex_{label}_delta_n_mm"] = 0.0
            pad_state[f"apex_{label}_delta_t_mm"] = 0.0
            continue
        mask = is_surf_full & (shape_full == shape_id)
        pad_deltas = deltas_full[mask]
        pad_normals = normals_full[mask]
        # Signed delta along outward normal: negative = compressed inward.
        n_mags = np.linalg.norm(pad_normals, axis=-1)
        # Guard against zero-magnitude normals (degenerate; shouldn't happen
        # for surface spheres, but be defensive).
        safe = n_mags > 1e-12
        d_dot_n = np.zeros(len(pad_deltas), dtype=np.float32)
        if safe.any():
            n_hat = pad_normals[safe] / n_mags[safe, None]
            d_dot_n[safe] = np.einsum("ij,ij->i", pad_deltas[safe], n_hat)
        # Active = pushed inward beyond noise floor.
        active_mask = d_dot_n < -DELTA_INWARD_THRESH_M
        pad_state[f"n_active_{label}"] = int(active_mask.sum())
        if active_mask.any():
            # Apex = most-compressed sphere (most negative δ·n̂).
            local_idx = int(np.argmin(d_dot_n))
            d_vec = pad_deltas[local_idx]
            n_vec = pad_normals[local_idx]
            n_mag = float(np.linalg.norm(n_vec))
            if n_mag > 1e-12:
                n_hat = n_vec / n_mag
                d_n = float(np.dot(d_vec, n_hat))
                d_t_vec = d_vec - d_n * n_hat
                d_t = float(np.linalg.norm(d_t_vec))
            else:
                d_n = float(np.linalg.norm(d_vec))
                d_t = 0.0
            pad_state[f"apex_{label}_delta_n_mm"] = d_n * 1e3
            pad_state[f"apex_{label}_delta_t_mm"] = d_t * 1e3
        else:
            pad_state[f"apex_{label}_delta_n_mm"] = 0.0
            pad_state[f"apex_{label}_delta_t_mm"] = 0.0

    return {
        "n_active": n_active,
        "n_active_strict": n_active_strict,
        "n_surface": n_surface,
        "max_delta_mm": max_delta * 1e3,
        "max_pen_mm": max_pen * 1e3,
        "max_delta": max_delta,
        "max_pen": max_pen,
        "mean_delta": mean_delta,
        "mean_pen_active": mean_pen_active,
        **pad_state,
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
