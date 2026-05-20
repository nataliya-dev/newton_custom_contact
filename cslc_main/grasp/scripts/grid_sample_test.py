# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare current Poisson-disc sampling vs pcu's Lloyd/CVT sampling.

Run:
    uv run python cslc_main/grasp/scripts/grid_sample_test.py

For both pad kinds (box, dome — if asset present):
  * Method A: trimesh.sample.sample_surface_even          (current)
  * Method B: pcu.sample_mesh_lloyd                       (proposed)

Lloyd's algorithm iteratively moves each sample to the centroid of its
surface Voronoi cell, converging to a centroidal Voronoi tessellation
(CVT).  On a flat patch this converges to a hexagonal close-packed
arrangement (the "grid in an ideal scenario" pattern); on a curved
patch it produces the most uniform spacing achievable on that surface.

Outputs ``grid_sample_test.png`` (2 rows: box, dome) with the contact-
face scatter for each method and nearest-neighbour spacing statistics
(mean / std/mean) in each subtitle.  Lower std/mean means more uniform.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu
import trimesh
from scipy.spatial import cKDTree

from cslc_main.grasp.pads import build_pad_trimesh, sample_pad_contact_face
from cslc_main.grasp.params import PadParams


def lloyd_sample(sub: trimesh.Trimesh, n: int) -> np.ndarray:
    """Lloyd / CVT samples on the submesh.  Returns (M, 3) points."""
    v = np.asarray(sub.vertices, dtype=np.float64)
    f = np.asarray(sub.faces, dtype=np.int32)
    pts = pcu.sample_mesh_lloyd(v, f, n)
    return np.asarray(pts)


def nn_stats(points_2d: np.ndarray) -> dict:
    """Mean / std / min / max of nearest-neighbour distance in 2-D."""
    tree = cKDTree(points_2d)
    d, _ = tree.query(points_2d, k=2)
    nn = d[:, 1]
    return dict(mean=nn.mean(), std=nn.std(), min=nn.min(), max=nn.max())


def plot_panel(ax, pts_2d, title, stats, x_label, y_label, bbox):
    ax.scatter(pts_2d[:, 0] * 1e3, pts_2d[:, 1] * 1e3, s=10)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{x_label} [mm]")
    ax.set_ylabel(f"{y_label} [mm]")
    ax.set_title(
        f"{title}\nN={len(pts_2d)}  NN mean={stats['mean']*1e3:.2f} mm  "
        f"std/mean={stats['std']/max(stats['mean'],1e-12):.2f}"
    )
    ax.grid(True, alpha=0.3)
    # bounding outline
    (x0, y0), (x1, y1) = bbox
    ax.plot(
        [x0*1e3, x1*1e3, x1*1e3, x0*1e3, x0*1e3],
        [y0*1e3, y0*1e3, y1*1e3, y1*1e3, y0*1e3],
        "k-", lw=1,
    )


def run_kind(kind: str, n_samples: int = 150):
    """Return list of (label, pts_3d, axes_pair, bbox) tuples for one pad kind."""
    p = PadParams(kind=kind, n_samples=n_samples)
    mesh, mask = build_pad_trimesh(p)

    # Build the submesh once (same surface both methods sample).
    face_ids = np.where(mask)[0]
    sub = mesh.submesh([face_ids], append=True)

    # Poisson via the production sampler (already routes through submesh).
    pts_poisson, _ = sample_pad_contact_face(mesh, mask, p)

    # Lloyd / CVT on the same submesh.
    pts_lloyd = lloyd_sample(sub, n_samples)

    # For 2-D plotting: box face is +x (project to y, z); dome cap is +z (project to x, y).
    if kind == "box":
        proj = (1, 2)            # show (y, z)
        x_label, y_label = "y", "z"
        bbox = ((-p.box_hy, -p.box_hz), (p.box_hy, p.box_hz))
    else:
        proj = (0, 1)            # show (x, y) for the dome cap
        x_label, y_label = "x", "y"
        # take a square bbox from the actual mesh extents on the cap
        vmin = sub.vertices.min(axis=0)
        vmax = sub.vertices.max(axis=0)
        bbox = ((vmin[0], vmin[1]), (vmax[0], vmax[1]))

    return [
        (f"{kind} — Poisson-disc (current)",
         pts_poisson[:, list(proj)], bbox, x_label, y_label),
        (f"{kind} — Lloyd/CVT (pcu)",
         pts_lloyd[:, list(proj)],   bbox, x_label, y_label),
    ]


def main() -> None:
    rows = [run_kind("box")]
    dome_panels = None
    try:
        dome_panels = run_kind("dome")
        rows.append(dome_panels)
    except FileNotFoundError as e:
        print(f"[skip dome] {e}")

    fig, axes = plt.subplots(
        len(rows), 2, figsize=(10, 5 * len(rows)),
    )
    if len(rows) == 1:
        axes = axes[None, :]

    for i, row in enumerate(rows):
        for j, (label, pts_2d, bbox, xl, yl) in enumerate(row):
            stats = nn_stats(pts_2d)
            plot_panel(axes[i, j], pts_2d, label, stats, xl, yl, bbox)
            print(
                f"{label:<40s} N={len(pts_2d)}  "
                f"NN mean={stats['mean']*1e3:.2f} mm  "
                f"std/mean={stats['std']/max(stats['mean'],1e-12):.2f}  "
                f"(min={stats['min']*1e3:.2f}, max={stats['max']*1e3:.2f})"
            )

    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
