"""
sphere_body.py — 3D sphere representations wrapping cslc.py.

Wraps cslc.create_pad_2d to add 3D world-frame positioning.
This is the module that MORPHIT will eventually replace:
swap create_finger_pad() with a MORPHIT-based function that
returns the same dict structure.

The pad dict returned here is a SUPERSET of what cslc.py returns,
so all cslc.py functions (solve_contact_local, etc.) work on it.
"""

import numpy as np
from cslc import create_pad_2d


def create_finger_pad(ny, nz, spacing, radius_factor=0.55):
    """
    3D finger pad for the contact pipeline.

    Wraps cslc.create_pad_2d and adds:
      - positions_3d : (n, 3) — centred at origin, pad in YZ plane
      - radii        : (n,)   — per-sphere radii (with overlap factor)

    The cslc.py neighbor graph, degrees, etc. are preserved so
    solve_contact_local() works directly on this dict.

    Parameters
    ----------
    radius_factor : radius / spacing.  >0.5 means spheres OVERLAP.
                    This is the eventual MORPHIT plug-in point.
    """
    # ── Use YOUR core lattice creation ──
    pad = create_pad_2d(ny, nz, spacing)

    # Override radius with the overlap factor
    radius = spacing * radius_factor
    pad['radius'] = radius

    # ── Add 3D positions centred at origin ──
    ns = pad['n_spheres']
    y_offset = (ny - 1) * spacing / 2.0
    z_offset = (nz - 1) * spacing / 2.0

    pos3d = np.zeros((ns, 3))
    pos3d[:, 1] = pad['positions'][:, 0] - y_offset   # y
    pos3d[:, 2] = pad['positions'][:, 1] - z_offset   # z
    # x = 0 (pad surface — contact happens along x)

    pad['positions_3d'] = pos3d
    pad['radii'] = np.full(ns, radius)

    return pad


def create_single_sphere(radius=0.001):
    """
    Degenerate 1-sphere pad for point-contact baseline.

    Returns a dict with the SAME keys that cslc.py solvers expect
    (neighbors, degrees, n_spheres) so it can go through the
    same pipeline.
    """
    return dict(
        n_spheres=1,
        positions=np.zeros((1, 2)),
        positions_3d=np.zeros((1, 3)),
        radii=np.array([radius]),
        radius=radius,
        neighbors=[[]],
        degrees=np.array([0.0]),
        spacing=0.0,
        kind='point',
        shape=(1, 1),
        ny=1, nz=1,
        # cslc.py also expects 'indices' for row_profile_2d
        indices=np.zeros((1, 2), dtype=int),
    )
