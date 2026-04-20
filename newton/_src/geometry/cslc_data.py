# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC data structures: lattice pad geometry, topology, and GPU upload.

CSLCPad: CPU-side geometry and neighbor topology for a single shape.
CSLCData: merged GPU arrays for all pads in a simulation.

File location: newton/_src/geometry/cslc_data.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from ..core.types import Devicelike


# ═══════════════════════════════════════════════════════════════════════════
#  CSLCPad — CPU-side lattice geometry for one shape
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CSLCPad:
    """Lattice pad for one rigid body shape.

    Attributes:
        positions: (N, 3) float32 — sphere centers in shape-local frame.
        radii: (N,) float32 — sphere radii.
        is_surface: (N,) bool — True for spheres that participate in contact.
        outward_normals: (N, 3) float32 — outward-pointing normal for surface
            spheres. Interior spheres have (0,0,0). Used as the displacement
            direction in the Jacobi solve and contact writing.
        neighbor_indices: list of arrays — CSR neighbor lists.
        shape_index: int — which shape in the Model this pad belongs to.
        grid_shape: tuple — (nx, ny, nz) grid dimensions.
        spacing: float — distance between sphere centers [m].
        sphere_radius: float — radius of each lattice sphere [m].
    """

    positions: np.ndarray
    radii: np.ndarray
    is_surface: np.ndarray
    outward_normals: np.ndarray
    neighbor_indices: list[np.ndarray]
    shape_index: int
    grid_shape: tuple[int, int, int]
    spacing: float
    sphere_radius: float

    @property
    def n_spheres(self) -> int:
        return len(self.positions)

    @property
    def n_surface(self) -> int:
        return int(self.is_surface.sum())


def _compute_box_outward_normals(
    grid_indices: np.ndarray,
    nx: int, ny: int, nz: int,
) -> np.ndarray:
    """Compute outward-pointing normals for surface spheres of a box grid.

    For face spheres: normal points along the face axis.
    For edge spheres: average of the two adjacent face normals, normalized.
    For corner spheres: average of the three adjacent face normals, normalized.
    Interior spheres get (0,0,0).

    Args:
        grid_indices: (N, 3) int array of (i, j, k) grid indices.
        nx, ny, nz: grid dimensions.

    Returns:
        (N, 3) float32 outward normals.
    """
    normals = np.zeros((len(grid_indices), 3), dtype=np.float32)

    for idx in range(len(grid_indices)):
        i, j, k = grid_indices[idx]
        n = np.zeros(3, dtype=np.float32)

        # X faces
        if i == 0:
            n[0] -= 1.0
        if i == nx - 1:
            n[0] += 1.0

        # Y faces
        if j == 0:
            n[1] -= 1.0
        if j == ny - 1:
            n[1] += 1.0

        # Z faces
        if k == 0:
            n[2] -= 1.0
        if k == nz - 1:
            n[2] += 1.0

        length = np.linalg.norm(n)
        if length > 1e-8:
            normals[idx] = n / length

    return normals


def create_pad_for_box(
    hx: float,
    hy: float,
    hz: float,
    *,
    spacing: float | None = None,
    grid_shape: tuple[int, int, int] | None = None,
    shape_index: int = 0,
) -> CSLCPad:
    """Create a volumetric lattice pad for a box shape.

    Fills the box interior with a regular 3D grid of spheres.
    Surface spheres (outer layer) participate in contact. All spheres
    participate in the Jacobi solve via neighbor coupling.

    Provide either `spacing` or `grid_shape`, not both.

    Args:
        hx, hy, hz: Box half-extents [m].
        spacing: Distance between sphere centers [m].
        grid_shape: (nx, ny, nz) number of spheres per axis.
        shape_index: Shape index in the Model.

    Returns:
        CSLCPad with volumetric packing and outward normals.
    """
    if spacing is not None and grid_shape is not None:
        raise ValueError("Provide either spacing or grid_shape, not both.")
    if spacing is None and grid_shape is None:
        raise ValueError("Provide either spacing or grid_shape.")

    if grid_shape is not None:
        nx, ny, nz = grid_shape
        spacing_val = min(
            (2.0 * hx) / max(nx - 1, 1) if nx > 1 else 2.0 * hx,
            (2.0 * hy) / max(ny - 1, 1) if ny > 1 else 2.0 * hy,
            (2.0 * hz) / max(nz - 1, 1) if nz > 1 else 2.0 * hz,
        )
    else:
        spacing_val = spacing
        nx = max(int(round(2.0 * hx / spacing_val)) + 1, 2)
        ny = max(int(round(2.0 * hy / spacing_val)) + 1, 2)
        nz = max(int(round(2.0 * hz / spacing_val)) + 1, 2)

    sphere_radius = spacing_val * 0.5

    xs = np.linspace(-hx, hx, nx, dtype=np.float32)
    ys = np.linspace(-hy, hy, ny, dtype=np.float32)
    zs = np.linspace(-hz, hz, nz, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    n_total = len(positions)

    radii = np.full(n_total, sphere_radius, dtype=np.float32)

    # Grid indices for surface/normal computation
    grid_indices = np.mgrid[0:nx, 0:ny, 0:nz].reshape(3, -1).T

    is_surface = (
        (grid_indices[:, 0] == 0)
        | (grid_indices[:, 0] == nx - 1)
        | (grid_indices[:, 1] == 0)
        | (grid_indices[:, 1] == ny - 1)
        | (grid_indices[:, 2] == 0)
        | (grid_indices[:, 2] == nz - 1)
    )

    # Outward normals for surface spheres
    outward_normals = _compute_box_outward_normals(grid_indices, nx, ny, nz)

    # 6-connected neighbor topology
    def flat_idx(i: int, j: int, k: int) -> int:
        return i * (ny * nz) + j * nz + k

    neighbor_indices = []
    for idx in range(n_total):
        i = idx // (ny * nz)
        j = (idx % (ny * nz)) // nz
        k = idx % nz
        neighbors = []
        for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                neighbors.append(flat_idx(ni, nj, nk))
        neighbor_indices.append(np.array(neighbors, dtype=np.int32))

    return CSLCPad(
        positions=positions,
        radii=radii,
        is_surface=is_surface,
        outward_normals=outward_normals,
        neighbor_indices=neighbor_indices,
        shape_index=shape_index,
        grid_shape=(nx, ny, nz),
        spacing=spacing_val,
        sphere_radius=sphere_radius,
    )


# Backwards compatibility
def create_pad_for_box_face(
    hx: float, hy: float, hz: float, *,
    face_axis: int, face_sign: int, spacing: float, shape_index: int = 0,
) -> CSLCPad:
    """Create a 2D lattice pad on one face of a box (legacy API)."""
    half_extents = [hx, hy, hz]
    axes = [i for i in range(3) if i != face_axis]
    a0, a1 = axes
    h0, h1 = half_extents[a0], half_extents[a1]
    n0 = max(int(round(2.0 * h0 / spacing)) + 1, 2)
    n1 = max(int(round(2.0 * h1 / spacing)) + 1, 2)
    sphere_radius = spacing * 0.5
    face_coord = face_sign * half_extents[face_axis]

    coords_0 = np.linspace(-h0, h0, n0, dtype=np.float32)
    coords_1 = np.linspace(-h1, h1, n1, dtype=np.float32)
    g0, g1 = np.meshgrid(coords_0, coords_1, indexing="ij")
    g0, g1 = g0.ravel(), g1.ravel()
    n_total = len(g0)

    positions = np.zeros((n_total, 3), dtype=np.float32)
    positions[:, face_axis] = face_coord
    positions[:, a0] = g0
    positions[:, a1] = g1

    radii = np.full(n_total, sphere_radius, dtype=np.float32)
    is_surface = np.ones(n_total, dtype=bool)

    # Outward normal: all spheres face the same direction
    outward_normals = np.zeros((n_total, 3), dtype=np.float32)
    outward_normals[:, face_axis] = float(face_sign)

    def flat_idx(i: int, j: int) -> int:
        return i * n1 + j

    neighbor_indices = []
    for idx in range(n_total):
        i, j = idx // n1, idx % n1
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n0 and 0 <= nj < n1:
                neighbors.append(flat_idx(ni, nj))
        neighbor_indices.append(np.array(neighbors, dtype=np.int32))

    return CSLCPad(
        positions=positions, radii=radii, is_surface=is_surface,
        outward_normals=outward_normals, neighbor_indices=neighbor_indices,
        shape_index=shape_index, grid_shape=(n0, n1, 1),
        spacing=spacing, sphere_radius=sphere_radius,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Stiffness calibration
# ═══════════════════════════════════════════════════════════════════════════


def calibrate_kc(
    ke_bulk: float,
    pads: list[CSLCPad],
    *,
    ka: float,
    contact_fraction: float = 0.3,
    per_pad: bool = True,
) -> float:
    """Derive per-sphere contact stiffness kc from bulk ke.

    At quasistatic equilibrium for uniform flat contact, the Laplacian
    vanishes (all delta equal), so per-sphere: kc*(pen-delta) = ka*delta,
    giving effective stiffness per sphere = kc*ka/(ka+kc).

    With per_pad=True (default), each pad's aggregate stiffness at uniform
    contact equals ke_bulk:
        N_contact_per_pad * kc*ka/(ka+kc) = ke_bulk
    This is the fair multi-pad analogue of a single point contact at
    ke_bulk per pad. With per_pad=False, the calibration sums N across
    all pads and matches one ke_bulk to that total — appropriate only
    when ke_bulk is a global "system" stiffness rather than a per-pad
    material property.

    Solving for kc:
        kc = ke_bulk * ka / (N_contact * ka - ke_bulk)
    """
    if per_pad:
        # Average n_surface across pads — assumes pads are roughly uniform
        # in size. For mixed pad sizes, promote to per-shape kc storage
        # in CSLCData (see TODOs in cslc_v1/convo_april_19.md).
        n_surface = int(np.mean([p.n_surface for p in pads]))
    else:
        n_surface = sum(p.n_surface for p in pads)
    n_contact = max(int(n_surface * contact_fraction), 1)
    denom = n_contact * ka - ke_bulk
    if denom <= 0.0:
        return ke_bulk / max(n_contact, 1)
    return ke_bulk * ka / denom


# ═══════════════════════════════════════════════════════════════════════════
#  CSLCData — merged GPU arrays
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CSLCData:
    """GPU-resident CSLC lattice data, merged from one or more CSLCPads.

    Neighbor lookups use CSR format:
        sphere i's neighbors are
        neighbor_list[neighbor_start[i] : neighbor_start[i] + neighbor_count[i]]
    """

    n_spheres: int
    n_surface: int
    positions: wp.array       # (n_spheres,) vec3 — shape-local rest positions
    radii: wp.array           # (n_spheres,) float32
    is_surface: wp.array      # (n_spheres,) int32 — 1 = surface, 0 = interior
    outward_normals: wp.array # (n_spheres,) vec3 — outward normal for surface spheres
    sphere_shape: wp.array    # (n_spheres,) int32 — shape index per sphere
    sphere_delta: wp.array    # (n_spheres,) float32 — warm-start displacement
    ka: float
    kl: float
    kc: float
    dc: float
    neighbor_start: wp.array  # (n_spheres,) int32 — CSR row pointer
    neighbor_count: wp.array  # (n_spheres,) int32
    neighbor_list: wp.array   # (n_edges,) int32
    device: str | None = None

    @classmethod
    def from_pads(
        cls, pads: list[CSLCPad], *, ka: float, kl: float, kc: float,
        dc: float, device: Devicelike | None = None,
    ) -> CSLCData:
        """Merge CSLCPads into GPU-resident CSLCData with global indexing."""
        if device is None:
            device = wp.get_device()

        offsets = []
        offset = 0
        for pad in pads:
            offsets.append(offset)
            offset += pad.n_spheres
        n_total = offset

        all_pos = np.zeros((n_total, 3), dtype=np.float32)
        all_radii = np.zeros(n_total, dtype=np.float32)
        all_surface = np.zeros(n_total, dtype=np.int32)
        all_normals = np.zeros((n_total, 3), dtype=np.float32)
        all_shape = np.zeros(n_total, dtype=np.int32)

        for pad, off in zip(pads, offsets):
            sl = slice(off, off + pad.n_spheres)
            all_pos[sl] = pad.positions
            all_radii[sl] = pad.radii
            all_surface[sl] = pad.is_surface.astype(np.int32)
            all_normals[sl] = pad.outward_normals
            all_shape[sl] = pad.shape_index

        # CSR neighbor structure
        all_start = np.zeros(n_total, dtype=np.int32)
        all_count = np.zeros(n_total, dtype=np.int32)
        neighbor_lists = []
        edge_offset = 0

        for pad, glob_off in zip(pads, offsets):
            for local_i, neighbors in enumerate(pad.neighbor_indices):
                global_i = glob_off + local_i
                all_start[global_i] = edge_offset
                all_count[global_i] = len(neighbors)
                neighbor_lists.append(neighbors + glob_off)
                edge_offset += len(neighbors)

        all_neighbor_list = (
            np.concatenate(neighbor_lists).astype(np.int32)
            if neighbor_lists else np.zeros(0, dtype=np.int32)
        )

        return cls(
            n_spheres=n_total,
            n_surface=int(all_surface.sum()),
            positions=wp.array(all_pos, dtype=wp.vec3, device=device),
            radii=wp.array(all_radii, dtype=wp.float32, device=device),
            is_surface=wp.array(all_surface, dtype=wp.int32, device=device),
            outward_normals=wp.array(all_normals, dtype=wp.vec3, device=device),
            sphere_shape=wp.array(all_shape, dtype=wp.int32, device=device),
            sphere_delta=wp.zeros(n_total, dtype=wp.float32, device=device),
            ka=ka, kl=kl, kc=kc, dc=dc,
            neighbor_start=wp.array(all_start, dtype=wp.int32, device=device),
            neighbor_count=wp.array(all_count, dtype=wp.int32, device=device),
            neighbor_list=wp.array(all_neighbor_list, dtype=wp.int32, device=device),
            device=device,
        )