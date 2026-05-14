# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels specific to UXPBD: lattice projection and lattice-aware contact.

The remaining kernels used by the solver (joint resolution, body integration,
restitution, body_parent_f reporting) live in
:mod:`newton._src.solvers.xpbd.kernels` and are imported there.
"""

import warp as wp


@wp.kernel
def update_lattice_world_positions(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    lattice_link: wp.array[wp.int32],
    lattice_p_rest: wp.array[wp.vec3],
    lattice_delta: wp.array[float],
    lattice_r: wp.array[float],
    lattice_particle_index: wp.array[wp.int32],
    # outputs
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
):
    """Project body_q onto lattice particles.

    For lattice sphere ``i`` with body-frame offset ``p_rest`` hosted by
    ``link``, set:

    ``particle_q[pidx] = body_q[link] x p_rest``,
    ``particle_qd[pidx] = v_lin + omega x (R . (p_rest - com))``,
    ``particle_radius[pidx] = lattice_r[i] - lattice_delta[i]``  (delta == 0 in v1).

    The ``particle_radius`` write is the load-bearing CSLC v2 seam: v2 writes
    nonzero ``lattice_delta`` and downstream contact kernels automatically
    see the compressed effective radius.
    """
    sid = wp.tid()
    link = lattice_link[sid]
    p_local = lattice_p_rest[sid]
    tf = body_q[link]
    pidx = lattice_particle_index[sid]

    # World position
    particle_q[pidx] = wp.transform_point(tf, p_local)

    # World velocity at offset
    rot = wp.transform_get_rotation(tf)
    r_world = wp.quat_rotate(rot, p_local - body_com[link])
    twist = body_qd[link]
    v_lin = wp.spatial_top(twist)
    omega = wp.spatial_bottom(twist)
    particle_qd[pidx] = v_lin + wp.cross(omega, r_world)

    # Effective contact radius. v1: delta == 0 so radius == rest radius.
    particle_radius[pidx] = lattice_r[sid] - lattice_delta[sid]
