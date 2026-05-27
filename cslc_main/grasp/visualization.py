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

    # Engagement cutoff: any |delta_n| below this is treated as rest
    # (gray).  10 µm is well below the meaningful CSLC operating depth
    # (~0.1-1 mm at silicone calibration) but well above the bounded
    # GPU-non-determinism noise floor (<<1 µm).  Old code used 10 nm
    # which is below the noise floor and caused spurious red coloring
    # before contact engagement.
    ENGAGED_THRESHOLD_M = 1.0e-5  # 10 µm
    # Color-intensity reference scale.  Old code normalised against the
    # per-frame MAX |delta_n|, which made tiny deltas saturate to full
    # red when no real contact was happening AND made the color of any
    # given sphere flicker frame-to-frame depending on which OTHER
    # sphere happened to have the largest delta.  Fixed 1 mm scale ≈
    # the typical operating depth so red intensity is physically
    # interpretable.
    INTENSITY_REFERENCE_M = 1.0e-3  # 1 mm

    def update(self, state) -> None:
        if not self.enabled:
            return
        viz = _compute_lattice_viz(self.model, state)
        if viz is None:
            return
        pw, dl_scalar, _radii, surface_mask = viz

        idx = 0
        for i in range(len(dl_scalar)):
            if surface_mask[i] == 0 or idx >= self.n:
                continue
            self.xforms[idx, :3] = pw[i]
            d = float(dl_scalar[i])
            if d > self.ENGAGED_THRESHOLD_M:
                # Compressed inward (engaged).  Red intensity scales
                # against a FIXED reference depth (1 mm), so values are
                # physically comparable frame-to-frame and pad-to-pad.
                t = min(d / self.INTENSITY_REFERENCE_M, 1.0)
                self.colors[idx] = [t, 0.2 * (1.0 - t), 1.0 - t]
            elif d < -self.ENGAGED_THRESHOLD_M:
                # Bulging outward — geometric Poisson signature of the
                # distance-preserving lateral spring at the patch
                # perimeter (theory step 5, notes.md §5).  Cyan so it's
                # visually distinct from both compressed (red) and at-
                # rest (gray).  Real CSLC characteristic worth seeing.
                t = min(-d / self.INTENSITY_REFERENCE_M, 1.0)
                self.colors[idx] = [0.0, 0.5 + 0.5 * t, 0.5 + 0.5 * t]
            else:
                # At rest (within ±10 µm of the rest position).
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


class ContactNormalRenderer:
    """Visualises the live per-frame contact normals emitted by the solver.

    Each active emitted slot carries the contact location on the pad
    (``rigid_contact_point0``, in pad body-local frame) and on the
    target (``rigid_contact_point1``, in target body-local frame), plus
    the world-frame target outward normal (``rigid_contact_normal``).
    We transform the two locals to world via the relevant body's
    ``body_q`` transform, draw the anchor at the world-frame target
    sample, and the arrow runs ``ARROW_LENGTH_M`` along the target
    normal (i.e. the direction the target's surface pushes the pad).

    Anchoring at the TARGET sample (point1) is the right choice for
    "where is the force applied to the object" semantics: the visual
    sits on the object's surface, not the deformed pad-sphere centre.
    The pad-side contact point (point0) ends up ~r_lat below the
    target sample along -normal -- not where the user wants the arrow
    base to sit.

    Useful for spotting:

    * Tilted contact normals during SQUEEZE (pads not squeezing
      head-on -> upward component pumps the object up during LIFT).
    * Normals flipping direction frame-to-frame (lattice tangent
      oscillation feeding the rigid solver).
    * Contact emission gaps -- many lattice spheres engaged but few
      contacts rendered (e.g. ``w_tangent < 1e-2`` cull biting hard on
      a curved pad).

    Falls back to ``viewer.log_lines`` when the viewer lacks
    ``log_arrows`` (e.g. the rerun viewer).
    """

    ARROW_LENGTH_M = 0.008  # 8 mm — visible without obscuring the lattice

    def __init__(self, model, viewer):
        self.model = model
        self.viewer = viewer
        self.enabled = hasattr(viewer, "log_lines") or hasattr(viewer, "log_arrows")
        # Cache shape→body map once; shapes are static for the run.
        self._shape_body = model.shape_body.numpy().astype(np.int32)

    def update(self, contacts, state) -> None:
        if not self.enabled or contacts is None or state is None:
            return
        n = int(contacts.rigid_contact_count.numpy()[0])
        if n == 0:
            self._clear()
            return
        shape0 = contacts.rigid_contact_shape0.numpy()[:n]
        shape1 = contacts.rigid_contact_shape1.numpy()[:n]
        active = np.where(shape0 >= 0)[0]
        if active.size == 0:
            self._clear()
            return

        # Body-local contact points (see write_cslc_contacts kernel
        # cslc_kernels.py: p0_body = X_wb_inv · q_world_def,
        # p1_body = X_tb_inv · t_world).
        p1_local = contacts.rigid_contact_point1.numpy()[active]
        # ``rigid_contact_normal`` IS world-frame (set to -n_face_world).
        nrm = contacts.rigid_contact_normal.numpy()[active].astype(np.float32)

        body_q = state.body_q.numpy()
        # Transform each contact's target-side anchor to world.
        target_body_idx = self._shape_body[shape1[active]]
        starts = np.empty((len(active), 3), dtype=np.float32)
        for k, (b_idx, p_local) in enumerate(zip(target_body_idx, p1_local)):
            X_wb = body_q[b_idx]
            starts[k] = _quat_rotate(X_wb[3:7], p_local) + X_wb[:3]

        # Defensive normalize -- emitted normals should be unit length.
        nrm_mag = np.linalg.norm(nrm, axis=1, keepdims=True)
        safe = nrm_mag > 1e-8
        nrm_unit = np.where(safe, nrm / np.where(safe, nrm_mag, 1), nrm)
        # Draw the arrow in the direction the TARGET pushes back on the
        # pad: that is the target's outward normal at the contact, which
        # ``write_cslc_contacts`` writes as ``normal_ab = -n_face_world``.
        # Flip back to +n_face for visualization so the arrow points
        # outward from the object surface (the physically intuitive
        # "this is the contact normal" direction).
        ends = starts + self.ARROW_LENGTH_M * (-nrm_unit)
        colors = np.tile([1.0, 0.85, 0.0], (len(active), 1)).astype(np.float32)

        starts_wp = wp.array(starts, dtype=wp.vec3)
        ends_wp = wp.array(ends, dtype=wp.vec3)
        colors_wp = wp.array(colors, dtype=wp.vec3)
        if hasattr(self.viewer, "log_arrows"):
            self.viewer.log_arrows(
                "/contact_normals", starts_wp, ends_wp, colors_wp,
            )
        else:
            self.viewer.log_lines(
                "/contact_normals", starts_wp, ends_wp, colors_wp,
            )

    def _clear(self) -> None:
        if hasattr(self.viewer, "log_arrows"):
            self.viewer.log_arrows("/contact_normals", None, None, None)
        else:
            self.viewer.log_lines("/contact_normals", None, None, None)


class TargetPointsRenderer:
    """Visualises the CSLC box-target (point-set) sample points.

    Under the Phase 5 v2 unified path, the CSLC kernels iterate per
    pad sphere over every target point sampled on the held object's
    approach faces.  This viewer overlay logs those target points as
    small grey spheres so the user can see the sampling density the
    kernel is grinding against.  Cost: one ``viewer.log_shapes`` call
    per frame with N_target transforms; cheap relative to the
    simulation step.

    No-op when no CSLC handler is attached or no pair has populated
    target arrays.
    """

    def __init__(self, model, viewer):
        self.model = model
        self.viewer = viewer
        self.enabled = False
        pipeline = getattr(model, "_collision_pipeline", None)
        handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
        if handler is None:
            return
        self.handler = handler
        # Collect all (pair, positions_local, normals_local, body_idx,
        # cslc_shape) for each CSLC pair.  Concatenate across pairs so
        # we render with a single log_shapes call.  The v2 unified
        # path has no per-target radii -- use the pad lattice's mean
        # sphere radius (halved) as the rendered point size.
        pair_data = []
        for pair in getattr(handler, "shape_pairs", []):
            if pair.target_positions_local is None:
                continue
            pos = pair.target_positions_local.numpy().astype(np.float32)
            normals = pair.target_normals_local.numpy().astype(np.float32)
            pair_data.append({
                "pos_local": pos,
                "normals_local": normals,
                "body_idx": int(pair.other_body),
                "cslc_shape": int(pair.cslc_shape),
            })
        if not pair_data:
            return
        self.pair_data = pair_data
        # Rendered ball radius derived from the pad lattice's mean
        # surface-sphere radius; halved so points appear as dots that
        # don't overlap at typical Poisson-disc sampling density.
        pad_radii = handler.cslc_data.radii.numpy()
        pad_is_surface = handler.cslc_data.is_surface.numpy().astype(bool)
        self.radius = float(np.mean(pad_radii[pad_is_surface])) * 0.5
        self.n_total = int(sum(p["pos_local"].shape[0] for p in pair_data))
        # Pre-allocated CPU buffers reused each frame.
        self.xforms = np.zeros((self.n_total, 7), np.float32)
        self.xforms[:, 6] = 1.0  # identity quaternion
        self.colors = np.tile([0.6, 0.6, 0.7],
                              (self.n_total, 1)).astype(np.float32)
        self.mats = np.tile([0.4, 0.2, 0.0, 0.0],
                            (self.n_total, 1)).astype(np.float32)
        self.enabled = True

    # Engagement reference depth for color intensity (1 mm = saturated red).
    ENGAGEMENT_REFERENCE_M = 1.0e-3

    def update(self, state) -> None:
        if not self.enabled:
            return
        body_q = state.body_q.numpy()
        # CSLC state needed to compute per-target engagement (which pad
        # spheres are currently penetrating each target sample).
        d = self.handler.cslc_data
        pad_pos_local = d.positions.numpy()            # (N_pad_total, 3)
        pad_radii    = d.radii.numpy()
        pad_delta    = d.sphere_delta.numpy()
        pad_shape    = d.sphere_shape.numpy()
        pad_surface  = d.is_surface.numpy()
        shape_body   = self.model.shape_body.numpy()
        shape_xform  = self.model.shape_transform.numpy()
        eps = float(d.smoothing_eps)

        cursor = 0
        for pair in self.pair_data:
            pos_local    = pair["pos_local"]
            normals_local = pair["normals_local"]
            t_body_xform = body_q[pair["body_idx"]]
            t_world_pos  = t_body_xform[:3]
            t_world_quat = t_body_xform[3:7]
            # Transform target samples + normals to world frame.
            t_world = np.array([
                _quat_rotate(t_world_quat, pos_local[k]) + t_world_pos
                for k in range(pos_local.shape[0])
            ], dtype=np.float32)
            n_world = np.array([
                _quat_rotate(t_world_quat, normals_local[k])
                for k in range(normals_local.shape[0])
            ], dtype=np.float32)

            # For this pair's CSLC pad: compute world-frame deformed
            # pad-sphere positions (q_i = X_wb * X_ws * p_local - delta).
            s_idx = pair["cslc_shape"]
            mask_pad = (pad_shape == s_idx) & (pad_surface == 1)
            pad_idx = np.where(mask_pad)[0]
            if pad_idx.size == 0:
                # Still draw the sample positions; just no engagement.
                for k in range(pos_local.shape[0]):
                    self.xforms[cursor, :3] = t_world[k]
                    self.colors[cursor] = [0.6, 0.6, 0.7]
                    cursor += 1
                continue
            b_idx = int(shape_body[s_idx])
            X_wb = body_q[b_idx]
            X_ws = shape_xform[s_idx]
            pad_p_local = pad_pos_local[pad_idx]                # (Np, 3)
            pad_r       = pad_radii[pad_idx]                    # (Np,)
            pad_d       = pad_delta[pad_idx]                    # (Np, 3)
            # World positions of pad-sphere REST centres:
            #   p_w = X_wb · (X_ws · p_local)
            pad_p_shape = np.array([
                _quat_rotate(X_ws[3:7], p) + X_ws[:3]
                for p in pad_p_local
            ], dtype=np.float32)
            pad_p_world = np.array([
                _quat_rotate(X_wb[3:7], p) + X_wb[:3]
                for p in pad_p_shape
            ], dtype=np.float32)
            q_world = pad_p_world - pad_d                       # (Np, 3)

            # For each target sample j: max over pad spheres of
            #   engagement_ij = kernel_w_ij · phi_eff_ij (half-space)
            # phi_eff_ij = smooth_relu(r_i - n_face_j · (q_i - t_j))   (v2)
            # kernel_w_ij = smooth_step(3·r_i - ||(q_i - t_j)_tangent||)
            # Visual saturated at ENGAGEMENT_REFERENCE_M.
            #
            # Vectorised: diffs[i, j] = q_i - t_j, shape (Np, M, 3).
            diffs = q_world[:, None, :] - t_world[None, :, :]   # (Np, M, 3)
            projs = np.einsum("ijk,jk->ij", diffs, n_world)     # (Np, M)
            raw_half = pad_r[:, None] - projs                   # (Np, M)
            phi_eff = 0.5 * (raw_half + np.sqrt(raw_half ** 2 + eps ** 2))
            d_t_vec = diffs - projs[:, :, None] * n_world[None, :, :]
            d_t_mag = np.linalg.norm(d_t_vec, axis=2)            # (Np, M)
            arg = 3.0 * pad_r[:, None] - d_t_mag                 # (Np, M)
            kernel_w = 0.5 * (1.0 + arg / np.sqrt(arg ** 2 + eps ** 2))
            engagement_ij = kernel_w * phi_eff                   # (Np, M)
            engagement_j = engagement_ij.max(axis=0)             # (M,)

            # Color: red when engaged, gray otherwise.
            for k in range(pos_local.shape[0]):
                self.xforms[cursor, :3] = t_world[k]
                eng = float(engagement_j[k])
                if eng > 1.0e-6:    # above noise floor (1 µm)
                    intensity = min(eng / self.ENGAGEMENT_REFERENCE_M, 1.0)
                    self.colors[cursor] = [
                        intensity, 0.2 * (1.0 - intensity), 1.0 - intensity
                    ]
                else:
                    self.colors[cursor] = [0.6, 0.6, 0.7]
                cursor += 1
        if cursor == 0:
            return
        import newton
        self.viewer.log_shapes(
            "/cslc_target_points", newton.GeoType.SPHERE, self.radius,
            wp.array(self.xforms[:cursor], dtype=wp.transform),
            wp.array(self.colors[:cursor], dtype=wp.vec3),
            wp.array(self.mats[:cursor], dtype=wp.vec4),
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

            # ── Repro-A diagnostics (H2 + H3) ───────────────────────
            # Per-pad active counts (H2) and apex normal/tangential
            # delta split (H3).  Older CSVs without these columns
            # simply skip this plot.
            try:
                n_act_l = np.array(
                    [int(r["n_active_left"]) for r in cslc_rows])
                n_act_r = np.array(
                    [int(r["n_active_right"]) for r in cslc_rows])
                ap_l_n = np.array(
                    [float(r["apex_left_delta_n_mm"]) for r in cslc_rows])
                ap_l_t = np.array(
                    [float(r["apex_left_delta_t_mm"]) for r in cslc_rows])
                ap_r_n = np.array(
                    [float(r["apex_right_delta_n_mm"]) for r in cslc_rows])
                ap_r_t = np.array(
                    [float(r["apex_right_delta_t_mm"]) for r in cslc_rows])
            except KeyError:
                pass
            else:
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                ax = axes[0]
                _phase_bands(ax, 0, max(n_act_l.max(), n_act_r.max(), 1))
                ax.plot(t_c, n_act_l, label="left pad", color="tab:blue")
                ax.plot(t_c, n_act_r, label="right pad", color="tab:orange")
                ax.set_ylabel("active surface spheres")
                ax.set_title("Per-pad active sphere count (H2)")
                ax.legend(loc="upper left", fontsize=9)
                ax.grid(True, alpha=0.3)

                ax = axes[1]
                ax.plot(t_c, ap_l_n, label="left δ·n̂", color="tab:blue")
                ax.plot(t_c, ap_l_t, label="left |δ_t|",
                        color="tab:blue", ls="--")
                ax.plot(t_c, ap_r_n, label="right δ·n̂", color="tab:orange")
                ax.plot(t_c, ap_r_t, label="right |δ_t|",
                        color="tab:orange", ls="--")
                ax.set_ylabel("apex δ [mm]")
                ax.set_xlabel("time [s]")
                ax.set_title(
                    "Apex sphere normal vs tangential δ (H3 — tangential >> "
                    "normal means lateral springs dominate)"
                )
                ax.legend(loc="upper left", fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(run_dir / "cslc_apex_delta.png", dpi=120)
                plt.close(fig)
