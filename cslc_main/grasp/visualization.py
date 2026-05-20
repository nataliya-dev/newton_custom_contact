# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualisation helpers — pure functions, no global state.

Three responsibilities, each one function from the runner's POV:

* :func:`save_lattice_preview` — pre-sim 3-D scatter of the sampled
  pad lattices (points + outward-normal quivers).  Run once before the
  sim starts; writes ``pad_lattice.png`` to the run directory.
* :func:`render_in_sim_lattice` — per-frame lattice rendering in the
  GL viewer (compression-coloured spheres).  Runner calls this from
  ``Example.render`` only when CSLC is active and the viewer is
  interactive.
* :func:`save_postsim_plots` — post-sim matplotlib renders of the CSV
  data.  Runs after the sim ends; writes ``sphere_trajectory.png``,
  ``contact_count.png``, and (if CSLC) ``cslc_delta.png``.

Matplotlib is the only visualisation dependency; it's already a
transitive dep via trimesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import warp as wp

from .scene import SceneArtifacts


# ── Pre-sim lattice preview ─────────────────────────────────────────────


def save_lattice_preview(artifacts: SceneArtifacts, path: Path) -> None:
    """Save a 3-D scatter of both pads' sampled lattice points + normals.

    The two pads are plotted as separate colour groups so left/right are
    distinguishable.  Each point gets a short arrow quiver showing its
    outward normal.  No interactive window is opened — the PNG is
    written and we move on.

    Off-screen rendering uses Matplotlib's ``Agg`` backend explicitly to
    keep the call safe inside headless/batch runs.
    """
    if not artifacts.pad_lattices:
        return  # nothing to draw (non-CSLC run)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    colours = {"left": "tab:blue", "right": "tab:orange"}
    for side, (pts, normals) in artifacts.pad_lattices.items():
        if pts.size == 0:
            continue
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=12, c=colours[side], label=f"{side} ({len(pts)})",
            depthshade=True,
        )
        # Short outward-normal quivers, length ≈ 25 % of mean NN spacing.
        scale = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1))) * 0.5
        if scale <= 0.0:
            scale = 1e-3
        ax.quiver(
            pts[:, 0], pts[:, 1], pts[:, 2],
            normals[:, 0], normals[:, 1], normals[:, 2],
            length=scale, normalize=False, color=colours[side], alpha=0.4,
            linewidth=0.5,
        )

    ax.set_xlabel("x [m] (pad-local)")
    ax.set_ylabel("y [m] (pad-local)")
    ax.set_zlabel("z [m] (pad-local)")
    ax.set_title("Pad lattice samples + outward normals")
    ax.legend(loc="upper right", fontsize=9)
    # Equal-ish axes for the scatter so the geometry isn't distorted.
    _set_equal_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _set_equal_axes(ax) -> None:
    """Equalise the data ranges across x/y/z for a 3-D Axes."""
    extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    centers = extents.mean(axis=1)
    half = (extents[:, 1] - extents[:, 0]).max() * 0.5
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


# ── In-sim lattice rendering (GL viewer) ────────────────────────────────


class LatticeRenderer:
    """Holds the per-frame buffers used by :func:`render_in_sim_lattice`.

    Stays out of the runner's main step loop — the runner just creates
    one of these once and calls :meth:`update`.  No effect when the
    viewer is headless or no CSLC handler is attached.
    """

    def __init__(self, model, viewer):
        self.model = model
        self.viewer = viewer
        self.enabled = False
        pipeline = getattr(model, "_collision_pipeline", None)
        handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
        if handler is None:
            return

        d = handler.cslc_data
        n = d.n_surface
        if n <= 0:
            return
        self.n = n
        # Render lattice spheres at 40 % of their lattice radius so the
        # underlying pad geometry stays visible behind them.
        self.radius = float(d.radii.numpy()[0]) * 0.4
        self.xforms = np.zeros((n, 7), np.float32)
        self.xforms[:, 6] = 1.0  # identity quaternion
        self.colors = np.zeros((n, 3), np.float32)
        self.mats = np.tile([0.5, 0.3, 0.0, 0.0], (n, 1)).astype(np.float32)
        self.enabled = True

    def update(self, state) -> None:
        if not self.enabled:
            return
        viz = _compute_lattice_viz(self.model, state)
        if viz is None:
            return
        pw, dl_scalar, _radii, surface_mask = viz

        idx = 0
        max_dl = max(float(np.max(np.abs(dl_scalar))), 1e-6)
        for i in range(len(dl_scalar)):
            if surface_mask[i] == 0 or idx >= self.n:
                continue
            self.xforms[idx, :3] = pw[i]
            t = min(abs(dl_scalar[i]) / max_dl, 1.0)
            # Red = compressed, gray = uncompressed.  Bulging (dl < 0)
            # treated as compression magnitude for now.
            if dl_scalar[i] > 1e-8:
                self.colors[idx] = [t, 0.2 * (1.0 - t), 1.0 - t]
            else:
                self.colors[idx] = [0.3, 0.3, 0.35]
            idx += 1
        if idx == 0:
            return

        import newton

        self.viewer.log_shapes(
            "/cslc_lattice", newton.GeoType.SPHERE, self.radius,
            wp.array(self.xforms[:idx], dtype=wp.transform),
            wp.array(self.colors[:idx], dtype=wp.vec3),
            wp.array(self.mats[:idx], dtype=wp.vec4),
        )


def _quat_rotate(q, v):
    xyz = np.array([q[0], q[1], q[2]])
    t = 2.0 * np.cross(xyz, v)
    return v + q[3] * t + np.cross(xyz, t)


def _compute_lattice_viz(model, state):
    """World-space (positions, signed-δ-along-normal, radii, surface-mask)."""
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None
    d = handler.cslc_data
    n = d.n_spheres
    pl = d.positions.numpy()
    nm = d.outward_normals.numpy()
    dl_vec = d.sphere_delta.numpy()
    rd = d.radii.numpy()
    sf = d.is_surface.numpy()
    si = d.sphere_shape.numpy()
    bq = state.body_q.numpy()
    sb = model.shape_body.numpy()
    sx = model.shape_transform.numpy()

    pw = np.zeros((n, 3), np.float32)
    dl_scalar = np.zeros(n, np.float32)
    for i in range(n):
        if sf[i] == 0:
            continue
        s = si[i]
        b = sb[s]
        qb = _quat_rotate(sx[s, 3:7], pl[i]) + sx[s, :3]
        p_rest_world = _quat_rotate(bq[b, 3:7], qb) + bq[b, :3]
        pw[i] = p_rest_world - dl_vec[i]
        n_body = _quat_rotate(sx[s, 3:7], nm[i])
        n_world = _quat_rotate(bq[b, 3:7], n_body)
        dl_scalar[i] = float(np.dot(dl_vec[i], n_world))
    return pw, dl_scalar, rd, sf


# ── In-sim GUI stats panel (GL viewer) ──────────────────────────────────


class StatsPanel:
    """Per-frame sphere-z readout for the viewer's GUI side panel.

    ``update`` records the latest object z (called each substep from the
    runner); ``render`` is invoked from ``Example.gui`` and writes the
    z value to the side panel as ``ui.text``.  Rendered in millimetres
    to match the rest of the project's reporting convention.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.current_z = config.object.start_z

    def update(self, obj_z: float) -> None:
        self.current_z = obj_z

    def render(self, ui) -> None:
        ui.text(f"Sphere z: {self.current_z * 1e3:8.3f} mm")


# ── Post-sim plots from logged CSVs ─────────────────────────────────────


def save_postsim_plots(run_dir: Path) -> None:
    """Render summary plots from the CSV files in ``run_dir``.

    Always renders ``sphere_trajectory.png`` and ``contact_count.png``.
    If ``cslc_state.csv`` is present, also renders ``cslc_delta.png``.
    """
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts_path = run_dir / "timeseries.csv"
    if not ts_path.exists():
        return

    rows = list(csv.DictReader(open(ts_path)))
    if not rows:
        return

    t = np.array([float(r["t"]) for r in rows])
    obj_z = np.array([float(r["obj_z"]) for r in rows])
    obj_x = np.array([float(r["obj_x"]) for r in rows])
    obj_y = np.array([float(r["obj_y"]) for r in rows])
    pad_z = np.array([float(r["left_pad_z"]) for r in rows])
    n_contacts = np.array([int(r["n_contacts"]) for r in rows])
    phases = [r["phase"] for r in rows]

    # Phase background bands.
    phase_changes = [0] + [
        i for i in range(1, len(phases)) if phases[i] != phases[i - 1]
    ] + [len(phases)]

    def _phase_bands(ax, ymin, ymax):
        colours = {"APPROACH": "#eef", "SQUEEZE": "#fef", "LIFT": "#efe", "HOLD": "#ffd"}
        for a, b in zip(phase_changes[:-1], phase_changes[1:]):
            ph = phases[a]
            ax.axvspan(t[a], t[b - 1], color=colours.get(ph, "#fff"), alpha=0.4, zorder=0)

    # ── Sphere trajectory + pad height ──
    fig, ax = plt.subplots(figsize=(8, 4))
    _phase_bands(ax, obj_z.min(), obj_z.max())
    ax.plot(t, obj_z * 1e3, label="object_z [mm]", color="tab:blue", lw=1.5)
    ax.plot(t, pad_z * 1e3, label="left_pad_z [mm]", color="tab:gray", lw=1.0, ls="--")
    ax.plot(t, obj_x * 1e3, label="object_x [mm]", color="tab:red", lw=1.0, alpha=0.6)
    ax.plot(t, obj_y * 1e3, label="object_y [mm]", color="tab:green", lw=1.0, alpha=0.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [mm]")
    ax.set_title("Object & pad trajectories")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "sphere_trajectory.png", dpi=120)
    plt.close(fig)

    # ── Contact count ──
    fig, ax = plt.subplots(figsize=(8, 3))
    _phase_bands(ax, 0, max(int(n_contacts.max()), 1))
    ax.plot(t, n_contacts, color="tab:purple", lw=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("# active contacts")
    ax.set_title("Active contact count over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "contact_count.png", dpi=120)
    plt.close(fig)

    # ── CSLC δ ──
    cslc_path = run_dir / "cslc_state.csv"
    if cslc_path.exists():
        cslc_rows = list(csv.DictReader(open(cslc_path)))
        if cslc_rows:
            t_c = np.array([float(r["t"]) for r in cslc_rows])
            n_active = np.array([int(r["n_active"]) for r in cslc_rows])
            max_delta = np.array([float(r["max_delta_mm"]) for r in cslc_rows])
            mean_delta = np.array([float(r["mean_delta_mm"]) for r in cslc_rows])
            max_pen = np.array([float(r["max_pen_mm"]) for r in cslc_rows])

            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax = axes[0]
            ax.plot(t_c, max_delta, label="max δ [mm]", color="tab:red")
            ax.plot(t_c, mean_delta, label="mean δ [mm]", color="tab:blue")
            ax.plot(t_c, max_pen, label="max pen [mm]", color="tab:purple", ls="--")
            ax.set_ylabel("[mm]")
            ax.set_title("CSLC lattice deformation")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.plot(t_c, n_active, color="tab:green")
            ax.set_ylabel("active spheres")
            ax.set_xlabel("time [s]")
            ax.set_title("Active surface spheres")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / "cslc_delta.png", dpi=120)
            plt.close(fig)
