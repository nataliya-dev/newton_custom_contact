# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Animation: indenter pressing progressively into a flat CSLC lattice.

Same scene as :mod:`cslc_main.theory.plot_flat_lattice_deflection`
(15×15 flat sphere lattice, single point-set target above the centre,
n_face pointing down toward the pad).  Animates the loading process
by sweeping the indenter's vertical offset ``h`` from "just out of
contact" down to "deep penetration", re-solving
:func:`solve_lattice_contact` at each frame and re-rendering the 3D
deformed lattice.

The per-frame solve is warm-started from the previous frame's δ, which
brings each subsequent solve down to a handful of Jacobi iterations.

Run::

    uv run python -m cslc_main.theory.animate_flat_lattice_deflection

Output: GIF (always) + MP4 (when ffmpeg is on PATH).
"""

from __future__ import annotations

from pathlib import Path
from shutil import which

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from cslc_main.theory.cslc_lattice import (
    lattice_contact_normal_forces,
    make_flat_grid,
    solve_lattice_contact,
)
from cslc_main.theory.cslc_targets import PointSetTarget


# ---------------------------------------------------------------------------
#  Scene parameters (mirror the static plot)
# ---------------------------------------------------------------------------

N_U = 15
N_V = 15
SPACING_MM = 3.0
R_PAD_MM = SPACING_MM / 2.0
SPACING = SPACING_MM * 1.0e-3
R_PAD = R_PAD_MM * 1.0e-3

KA = 100.0
KL = 200.0
KC = 1.0e9
EPS = 1.0e-9
INDENT_AREA = np.pi * R_PAD * R_PAD          # one disc-of-radius-r tile

# Indenter offset sweep, in mm.  H_START > R_pad ⇒ no contact (rest
# overlap raw0 < 0); H_END < R_pad ⇒ deep contact.  We sweep linearly.
H_START_MM = 1.55                            # just out of contact
H_END_MM = 0.30                              # deep contact
N_FRAMES = 90                                # ~6 s at 15 fps
FPS = 15

# Visual-only downward shift of the indenter marker, in mm.  The
# contact model treats the target as the single sample at z = h; this
# offset just moves the rendered marker lower so it visually meets
# the lattice during contact instead of floating above it.  Picked so
# the marker descends from above, touches the lattice plane near
# h ≈ MARKER_VISUAL_OFFSET_MM, and presses just into the dimple at
# the deepest frame.
MARKER_VISUAL_OFFSET_MM = 0.5


# ---------------------------------------------------------------------------
#  Lattice (rebuilt once)
# ---------------------------------------------------------------------------


def build_lattice():
    return make_flat_grid(
        n_u=N_U, n_v=N_V, spacing=SPACING, ka=KA, kl=KL,
        normal=np.array([0.0, 0.0, 1.0]),
        diagonals=False,
    )


def make_target(h_mm: float) -> PointSetTarget:
    h = h_mm * 1.0e-3
    positions = np.array([[0.0, 0.0, h]])
    normals = np.array([[0.0, 0.0, -1.0]])
    areas = np.array([INDENT_AREA])
    return PointSetTarget(positions=positions, normals=normals, areas=areas)


# ---------------------------------------------------------------------------
#  Pre-solve all frames (warm-started chain so each solve is cheap).
# ---------------------------------------------------------------------------


def solve_all_frames(lat):
    """Solve the equilibrium for every frame, warm-starting from the previous.

    Returns a list of dicts: {h_mm, deltas, f_centre_mN, max_dn_mm}.
    """
    h_values_mm = np.linspace(H_START_MM, H_END_MM, N_FRAMES)
    frames: list[dict] = []
    delta_prev = np.zeros((lat.N, 3))
    centre_idx = (N_U // 2) * N_V + (N_V // 2)
    for k, h_mm in enumerate(h_values_mm):
        target = make_target(h_mm)
        deltas, info = solve_lattice_contact(
            lat, target, kc=KC, r_pad=R_PAD, eps=EPS,
            delta0=delta_prev, tol=1.0e-12, maxiter=2000,
        )
        F_contact, _ = lattice_contact_normal_forces(
            lat, target, deltas, kc=KC, r_pad=R_PAD, eps=EPS,
        )
        f_centre = float(np.linalg.norm(F_contact[centre_idx]))
        delta_n = np.einsum("nj,nj->n", deltas, lat.n)
        max_dn_mm = float(np.max(np.abs(delta_n))) * 1e3
        frames.append({
            "h_mm": float(h_mm),
            "deltas": deltas.copy(),
            "f_centre_mN": f_centre * 1e3,
            "max_dn_mm": max_dn_mm,
            "jacobi_iters": int(info["jacobi_refine_iters"]),
        })
        delta_prev = deltas
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  frame {k+1:3d}/{N_FRAMES}: h = {h_mm:.3f} mm, "
                  f"max δ_n = {max_dn_mm:.3f} mm, "
                  f"F_centre = {f_centre*1e3:.2f} mN, "
                  f"Jacobi iters = {info['jacobi_refine_iters']}")
    return frames


# ---------------------------------------------------------------------------
#  Animation
# ---------------------------------------------------------------------------


def build_animation(lat, frames: list[dict], out_dir: Path):
    p_rest_mm = lat.p * 1e3
    edges = lat.edges

    # Pre-compute per-frame deformed positions and δ_n colours (mm).
    q_all = []
    dn_all = []
    for fr in frames:
        q = (lat.p - fr["deltas"]) * 1e3
        dn = np.einsum("nj,nj->n", fr["deltas"], lat.n) * 1e3
        q_all.append(q)
        dn_all.append(dn)
    q_all = np.stack(q_all, axis=0)              # (F, N, 3) mm
    dn_all = np.stack(dn_all, axis=0)            # (F, N) mm

    # Fixed axis ranges so frames are comparable.  z is intentionally
    # cropped close to the lattice plane: the indenter (which moves
    # over a much larger range than the lattice deflects) enters from
    # above as it descends, focusing the viewer on the dimple.  The
    # marker is drawn slightly below the target sample position so it
    # visually meets the lattice during contact rather than floating
    # above it (see ``MARKER_VISUAL_OFFSET_MM`` and ``update()``).
    xy_lim = float(np.max(np.abs(p_rest_mm[:, :2]))) * 1.05
    z_min = float(np.min(q_all[:, :, 2]))
    z_lim_lo = min(z_min - 0.05, -0.4)
    z_lim_hi = 0.6
    dn_max_global = float(np.max(np.abs(dn_all)))
    vmin, vmax = -dn_max_global, dn_max_global

    LABEL_SIZE = 16
    TICK_SIZE = 13

    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # Persistent artists.
    edge_lines = []
    for (i, j) in edges:
        ln, = ax.plot([0, 0], [0, 0], [0, 0],
                      color="0.30", lw=0.7, alpha=0.7, zorder=1)
        edge_lines.append(ln)
    scatter = ax.scatter(
        q_all[0, :, 0], q_all[0, :, 1], q_all[0, :, 2],
        c=dn_all[0], cmap=cm.viridis, vmin=vmin, vmax=vmax,
        s=35, edgecolor="black", linewidth=0.25, zorder=3,
    )
    # Visual indenter marker, drawn slightly below the target sample's
    # actual z (see MARKER_VISUAL_OFFSET_MM).  Cosmetic only — the
    # contact model still uses the sample at z = h.
    indenter = ax.scatter(
        [0.0], [0.0], [frames[0]["h_mm"] - MARKER_VISUAL_OFFSET_MM],
        c="crimson", s=140, marker="v", edgecolor="black",
        linewidth=0.6, zorder=4,
    )
    annotation = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes,
        fontsize=TICK_SIZE, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9),
    )

    ax.set_xlim(-xy_lim, xy_lim)
    ax.set_ylim(-xy_lim, xy_lim)
    ax.set_zlim(z_lim_lo, z_lim_hi)
    ax.set_xlabel("x [mm]", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel("y [mm]", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_zlabel("z [mm]", fontsize=LABEL_SIZE, labelpad=14)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.view_init(elev=22.0, azim=-55.0)
    fig.tight_layout()

    def update(frame_idx: int):
        q = q_all[frame_idx]
        dn = dn_all[frame_idx]
        for k, (i, j) in enumerate(edges):
            edge_lines[k].set_data_3d(
                [q[i, 0], q[j, 0]],
                [q[i, 1], q[j, 1]],
                [q[i, 2], q[j, 2]],
            )
        # scatter._offsets3d is the documented mutation point in 3D.
        scatter._offsets3d = (q[:, 0], q[:, 1], q[:, 2])
        scatter.set_array(dn)
        h = frames[frame_idx]["h_mm"]
        indenter._offsets3d = (np.array([0.0]),
                               np.array([0.0]),
                               np.array([h - MARKER_VISUAL_OFFSET_MM]))
        annotation.set_text(
            f"h = {h:.2f} mm\n"
            f"raw₀ = {(R_PAD_MM - h):+.2f} mm\n"
            f"max |δ_n| = {frames[frame_idx]['max_dn_mm']:.3f} mm\n"
            f"F_centre = {frames[frame_idx]['f_centre_mN']:.1f} mN"
        )
        return edge_lines + [scatter, indenter, annotation]

    anim = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000.0 / FPS, blit=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # GIF (always-available writer via Pillow).
    gif_path = out_dir / "flat_lattice_contact_deflection.gif"
    anim.save(gif_path, writer="pillow", fps=FPS, dpi=120)
    print(f"saved animation: {gif_path}")

    # MP4 (if ffmpeg is present).  Better compression + smoother playback.
    if which("ffmpeg") is not None:
        mp4_path = out_dir / "flat_lattice_contact_deflection.mp4"
        anim.save(mp4_path, writer="ffmpeg", fps=FPS, dpi=160,
                  extra_args=["-pix_fmt", "yuv420p"])
        print(f"saved animation: {mp4_path}")
    else:
        print("ffmpeg not on PATH — skipping MP4 export.")

    plt.close(fig)


def main() -> None:
    lat = build_lattice()
    print(f"Lattice: {N_U}x{N_V} flat grid, spacing={SPACING_MM:.2f} mm")
    print(f"Sweeping indenter h from {H_START_MM:.2f} mm to {H_END_MM:.2f} mm "
          f"over {N_FRAMES} frames at {FPS} fps.")
    print("Solving frames (warm-started)…")
    frames = solve_all_frames(lat)
    print("Rendering animation…")
    out_dir = Path(__file__).parent / "figures"
    build_animation(lat, frames, out_dir)


if __name__ == "__main__":
    main()
