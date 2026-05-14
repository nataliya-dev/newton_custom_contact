# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-group caching helpers for UXPBD shape-matching.

The SRXPBD ``solve_shape_matching_batch_tiled`` and
``enforce_momentum_conservation_tiled`` kernels require per-group CSR-indexed
data: a flat ``group_particles_flat`` array of particle indices, parallel
``group_particle_start`` / ``group_particle_count``, plus per-group total mass.
UXPBD caches these once at solver init (mirrors SRXPBD).
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _calculate_group_particle_mass(
    particle_mass: wp.array[wp.float32],
    group_particle_start: wp.array[wp.int32],
    group_particle_count: wp.array[wp.int32],
    group_particles_flat: wp.array[wp.int32],
    total_group_mass: wp.array[wp.float32],
):
    group_id = wp.tid()
    start_idx = group_particle_start[group_id]
    num_particles = group_particle_count[group_id]
    total = wp.float32(0.0)
    for p in range(num_particles):
        idx = group_particles_flat[start_idx + p]
        total += particle_mass[idx]
    total_group_mass[group_id] = total


def build_shape_match_cache(model):
    """Identify dynamic SM-rigid groups and pack their particle indices flat.

    A group is "dynamic" if at least one particle has nonzero mass AND no
    particle is a lattice particle (substrate=0). Lattice groups are excluded
    because they are kinematically driven by their host body, not shape-matched.

    Args:
        model: The :class:`~newton.Model` to inspect.

    Returns:
        A dict with keys ``num_dynamic_groups`` (int), ``dynamic_group_ids``,
        ``group_particle_start``, ``group_particle_count``,
        ``group_particles_flat``, ``total_group_mass``, ``block_dim`` (int).
    """
    device = model.device
    if model.particle_count == 0 or model.particle_group_count == 0:
        empty_i32 = wp.empty(0, dtype=wp.int32, device=device)
        empty_f32 = wp.empty(0, dtype=wp.float32, device=device)
        return {
            "num_dynamic_groups": 0,
            "dynamic_group_ids": empty_i32,
            "group_particle_start": empty_i32,
            "group_particle_count": empty_i32,
            "group_particles_flat": empty_i32,
            "total_group_mass": empty_f32,
            "block_dim": 32,
        }

    particle_mass_np = model.particle_mass.numpy()
    particle_substrate_np = model.particle_substrate.numpy()

    dynamic_ids: list[int] = []
    starts: list[int] = []
    counts: list[int] = []
    flat: list[int] = []

    offset = 0
    for group_id in range(model.particle_group_count):
        group_particle_indices = model.particle_groups[group_id]
        if hasattr(group_particle_indices, "numpy"):
            idxs = group_particle_indices.numpy().astype(np.int32)
        else:
            idxs = np.asarray(list(group_particle_indices), dtype=np.int32)
        if idxs.size == 0:
            continue
        # Exclude lattice particles (kinematically driven, not shape-matched).
        substrate_vals = particle_substrate_np[idxs]
        if np.any(substrate_vals == 0):
            continue
        masses = particle_mass_np[idxs]
        if not np.any(masses > 0.0):
            continue
        dynamic_ids.append(group_id)
        starts.append(offset)
        counts.append(int(idxs.size))
        flat.extend(int(i) for i in idxs)
        offset += int(idxs.size)

    num_dynamic = len(dynamic_ids)
    dynamic_ids_arr = wp.array(dynamic_ids, dtype=wp.int32, device=device)
    starts_arr = wp.array(starts, dtype=wp.int32, device=device)
    counts_arr = wp.array(counts, dtype=wp.int32, device=device)
    flat_arr = wp.array(flat, dtype=wp.int32, device=device)

    max_n = max(counts) if counts else 0
    block_dim = min(256, max_n)
    block_dim = max(32, ((block_dim + 31) // 32) * 32)

    total_mass = wp.zeros(num_dynamic, dtype=wp.float32, device=device)
    if num_dynamic > 0:
        wp.launch(
            kernel=_calculate_group_particle_mass,
            dim=num_dynamic,
            inputs=[model.particle_mass, starts_arr, counts_arr, flat_arr],
            outputs=[total_mass],
            device=device,
        )

    return {
        "num_dynamic_groups": num_dynamic,
        "dynamic_group_ids": dynamic_ids_arr,
        "group_particle_start": starts_arr,
        "group_particle_count": counts_arr,
        "group_particles_flat": flat_arr,
        "total_group_mass": total_mass,
        "block_dim": block_dim,
    }
