# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compliant Sphere Lattice Contact (CSLC) kernels for SolverUXPBD.

Two solvers share the per-sphere displacement buffer
``model.lattice_delta`` (vec3, contract sign convention ``q_i = p_i -
δ_i``):

* :func:`solve_lattice_anchor_compression` — v1 anchor-only, per-sphere
  closed form for the normal-axis compression.  Writes ``δ_n ·
  n_outward`` (vec3 along the rest outward normal).  Used when
  ``CSLCParams.use_jacobi = False`` (default).

* :func:`solve_lattice_jacobi_step` — damped Jacobi sweep that mirrors
  ``newton._src.geometry.cslc_kernels.jacobi_step`` (the production CSLC
  solver).  Same per-pair Hertz-like force law, anisotropic-anchor
  decomposition, graph-Laplacian lateral coupling, and stick-slip
  friction.  Differences vs. the original kernel:

  - Pairs come from ``model.particle_grid`` (UXPBD's hash grid over the
    unified particle pool), not a PointSetTarget on the object surface.
  - Per-pair geometry is sphere-vs-sphere (``raw = r_pad + r_obj −
    |q_pad − q_obj|``) rather than half-space
    (``raw = r_pad − n_face · (q_pad − t_sample)``).  For SM-rigid
    sphere-packed objects this is the natural form; the CSLC kernel
    assumed a continuum object whose surface was sampled into
    ``(pos, n_face, area)`` triples.
  - Per-pair area weight ``A_j`` is derived from each object particle's
    own radius (``A_j ≈ (2·r_obj)²``, the spacing-squared of a
    close-packed packing), not from a Voronoi area baked at load.
  - Hard culls on the active-set gates (``raw < 0`` and
    ``alignment ≤ 0``) instead of the C∞ smooth gates the CSLC kernel
    uses for ``wp.Tape``-friendly backprop.  ``phi_eff`` keeps the
    smooth_relu lift so the contact force has no first-touch
    discontinuity.

Both kernels write the same buffer with the same sign convention, so
:func:`accumulate_cslc_body_wrench` consumes either output uniformly:
the body wrench is ``F = -k_a · δ`` (anisotropic when
``ka_tangent_ratio ≠ 1``), torque from the lever arm at the deformed
sphere position.

Reference: ``newton._src.geometry.cslc_kernels.jacobi_step`` lines
~503-844 for the per-pair force law and per-axis decomposition we
mirror here.
"""

from __future__ import annotations

import warp as wp

from ...geometry import ParticleFlags


# ═══════════════════════════════════════════════════════════════════════════
#  Smooth surrogates — mirror cslc_kernels.smooth_relu / smooth_step
#  (only smooth_relu is used; hard culls replace the smooth gates).
# ═══════════════════════════════════════════════════════════════════════════


@wp.func
def smooth_relu(x: float, eps: float) -> float:
    """``0.5·(x + sqrt(x² + ε²))`` — C∞ surrogate for max(x, 0).

    Matches ``cslc_kernels.smooth_relu`` exactly.  Used here for the
    Hertz-like ``phi_eff = σ_ε(raw) · sqrt(σ_ε(raw) + ε)`` lift so the
    per-contact stiffness vanishes smoothly at first touch (``raw → 0``)
    rather than producing a step-function impulse.
    """
    return 0.5 * (x + wp.sqrt(x * x + eps * eps))


# ═══════════════════════════════════════════════════════════════════════════
#  v1 closed-form (anchor-only, scalar-projection of δ)
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def solve_lattice_anchor_compression(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_radius: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_substrate: wp.array[wp.uint8],
    particle_to_lattice: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_articulation: wp.array[wp.int32],
    lattice_link: wp.array[wp.int32],
    lattice_normal: wp.array[wp.vec3],
    lattice_is_surface: wp.array[wp.uint8],
    lattice_k_anchor: wp.array[wp.float32],
    lattice_k_bulk: wp.array[wp.float32],
    lattice_particle_index: wp.array[wp.int32],
    max_radius: float,
    clamp_delta_max: float,
    # outputs
    lattice_delta: wp.array[wp.vec3],
):
    """v1 per-sphere anchor + bulk-contact series-spring compression.

    For each surface lattice sphere ``i``, sum the inward-normal-
    projected overlap from every object-particle neighbour ``j`` (via
    ``model.particle_grid``), apply the series-spring closed form, and
    write :math:`\\delta = \\delta_n \\cdot \\hat n_{outward}` to
    ``lattice_delta[sid]`` (vec3, along the rest outward normal).  The
    Jacobi solver writes the full vec3; this closed form populates only
    the normal-axis component.

    The series-spring identity here is the v1 single-pair form
    ``δ_n = k_c / (k_a + k_c) · L`` where
    ``L = Σ_j overlap_j · alignment_j`` aggregates overlaps from all
    aligned neighbours.  The aggregation is dimensionally consistent
    (overlap is in meters, alignment is dimensionless) but does not
    incorporate per-contact area weighting or the proper
    multi-contact ``ka + N·kc`` denominator — those are the job of the
    Jacobi solver.

    Pair filtering matches ``solve_particle_particle_contacts_uxpbd``
    (kernels.py): skip self, skip inactive particles, skip
    same-articulation lattice spheres (URDF joint overlaps), skip
    same-body lattice spheres, skip fluid (substrate 3).
    """
    sid = wp.tid()

    if lattice_is_surface[sid] == wp.uint8(0):
        lattice_delta[sid] = wp.vec3(0.0, 0.0, 0.0)
        return

    pidx = lattice_particle_index[sid]
    if (particle_flags[pidx] & ParticleFlags.ACTIVE) == 0:
        lattice_delta[sid] = wp.vec3(0.0, 0.0, 0.0)
        return

    link = lattice_link[sid]
    rot = wp.transform_get_rotation(body_q[link])
    n_world = wp.quat_rotate(rot, lattice_normal[sid])

    x_i = particle_x[pidx]
    r_i = particle_radius[pidx]

    art_i = wp.int32(-1)
    if link >= 0:
        art_i = body_articulation[link]

    query = wp.hash_grid_query(grid, x_i, r_i + max_radius)
    load = float(0.0)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == pidx:
            continue
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            continue
        sub_j = particle_substrate[j]
        if sub_j == wp.uint8(0):
            other_lat = particle_to_lattice[j]
            if other_lat < 0:
                continue
            host_j = lattice_link[other_lat]
            if host_j == link:
                continue
            if host_j >= 0 and art_i >= 0 and art_i == body_articulation[host_j]:
                continue
        if sub_j == wp.uint8(3):
            continue

        delta_vec = x_i - particle_x[j]
        d = wp.length(delta_vec)
        if d < 1.0e-12:
            continue
        r_j = particle_radius[j]
        overlap = (r_i + r_j) - d
        if overlap <= 0.0:
            continue
        n_pair = delta_vec / d
        alignment = -wp.dot(n_pair, n_world)
        if alignment <= 0.0:
            continue
        load += overlap * alignment

    k_a = lattice_k_anchor[sid]
    k_c = lattice_k_bulk[sid]
    denom = k_a + k_c
    if denom <= 0.0:
        lattice_delta[sid] = wp.vec3(0.0, 0.0, 0.0)
        return
    delta_n = (k_c / denom) * load

    if clamp_delta_max >= 0.0 and delta_n > clamp_delta_max:
        delta_n = clamp_delta_max

    # vec3 along the rest outward normal — Jacobi solver writes the
    # full vec3 directly; both consumers (update_lattice_world_positions,
    # accumulate_cslc_body_wrench) read this buffer uniformly.
    lattice_delta[sid] = delta_n * n_world


# ═══════════════════════════════════════════════════════════════════════════
#  Jacobi sweep — mirrors cslc_kernels.jacobi_step
#  ───────────────────────────────────────────────────────────────────────
#  Per pad sphere i, in i's local rest-normal frame (CSLC contract §6.5
#  load form ``load = −∂E/∂δ``):
#
#    0 = -K_anchor · δ_i                                  (anchor; anisotropic)
#        - k_l · Σ_{j ∈ N(i)} (δ_i − δ_j)                 (graph-Laplacian lateral)
#        + Σ_{m ∈ neighbours} k_c · A_m · phi_eff_im · n_eff_im
#                                                         (contact, sphere-sphere)
#        + f_friction(δ_t, |F_n|, k_stick, μ)             (stick-slip)
#
#  where for each object-particle neighbour m:
#    raw_im      = r_pad_i + r_obj_m − |q_i − q_m|        (sphere-sphere overlap;
#                                                          UXPBD's substitute for
#                                                          the CSLC half-space form
#                                                          ``r − n_face·(q−t)``)
#    phi_eff_im  = σ_ε(raw_im) · √(σ_ε(raw_im) + ε)        (Hertz-like δ^1.5;
#                                                          matches cslc_kernels)
#    α_im        = -(n_pair_im · n_pad_i)                 (pair-normal alignment;
#                                                          UXPBD's substitute for
#                                                          ``-(n_face · n_pad)``)
#    n_eff_im    = -n_pair_im                             (load form: pushes pad
#                                                          along −n_pair = away from
#                                                          obj particle)
#    A_im        ≈ (2·r_obj_m)²                           (sphere packing's
#                                                          spacing²; UXPBD's
#                                                          substitute for the
#                                                          CSLC target Voronoi area)
#
#  Active-set gates: HARD cull on raw ≤ 0 and α ≤ 0 (UXPBD is
#  forward-only — the smooth gates in cslc_kernels exist for wp.Tape).
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def solve_lattice_jacobi_step(
    delta_src: wp.array[wp.vec3],
    delta_dst: wp.array[wp.vec3],
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_radius: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    particle_substrate: wp.array[wp.uint8],
    particle_to_lattice: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_articulation: wp.array[wp.int32],
    lattice_link: wp.array[wp.int32],
    lattice_p_rest: wp.array[wp.vec3],
    lattice_normal: wp.array[wp.vec3],
    lattice_is_surface: wp.array[wp.uint8],
    lattice_k_anchor: wp.array[wp.float32],
    lattice_k_lateral: wp.array[wp.float32],
    lattice_k_bulk: wp.array[wp.float32],
    lattice_neighbor_start: wp.array[wp.int32],
    lattice_neighbor_count: wp.array[wp.int32],
    lattice_neighbor_list: wp.array[wp.int32],
    lattice_particle_index: wp.array[wp.int32],
    max_radius: float,
    eps: float,
    alpha_damping: float,
    ka_tangent_ratio: float,
    k_stick: float,
    mu_friction: float,
    clamp_delta_max: float,
):
    """One damped Jacobi sweep over the UXPBD lattice (mirror of
    ``cslc_kernels.jacobi_step``).

    Read ``delta_src``, accumulate per-pair contact load + lateral
    Laplacian + stick-slip friction over each surface pad sphere's
    object-particle neighbours, perform the anisotropic-anchor Jacobi
    update in the pad's local rest-normal frame, write the
    α-damped result to ``delta_dst``.

    Caller (``SolverUXPBD.compute_compliant_contact_response``) loops
    this kernel ``CSLCParams.jacobi_iterations`` times with src/dst
    ping-pong scratches and ends by copying the converged buffer into
    ``model.lattice_delta``.

    Args:
        delta_src: Per-sphere δ at the start of this sweep, shape
            [lattice_sphere_count]; read-only here.
        delta_dst: Output buffer; ``(1-α)·δ_src + α·δ_jacobi`` per
            sphere.  Different storage than ``delta_src`` (ping-pong).
        grid: Hash-grid id (``model.particle_grid.id``); the caller
            must rebuild the grid against ``particle_x`` before
            launching this kernel.
        particle_x: Current particle positions (post-projection lattice
            + SM-rigid + fluid).
        particle_radius: Per-particle radius (lattice spheres use
            ``lattice_r``; object spheres their own radii).
        particle_flags, particle_substrate, particle_to_lattice: Pair
            filter inputs (mirror ``solve_particle_particle_contacts_uxpbd``).
        body_q, body_articulation, lattice_link: Pose + topology lookups
            for the pair filter and the world-frame outward normal.
        lattice_p_rest, lattice_normal: Pad sphere body-local rest
            position and outward normal.
        lattice_is_surface: Skip interior spheres (they have no contact
            term; the lateral Laplacian still couples to them, but no
            δ to propagate without contact).
        lattice_k_anchor, lattice_k_lateral, lattice_k_bulk: Per-sphere
            stiffnesses (anchor along rest normal, lateral edge weight,
            bulk contact spring).
        lattice_neighbor_start/_count/_list: CSR neighbour topology for
            the lateral Laplacian.  Empty arrays disable lateral
            coupling (each sphere sees count=0).
        lattice_particle_index: Particle pool index per sphere; used to
            (a) skip the sphere's own particle in the hash sweep,
            (b) read the pad sphere's current world position via
            ``particle_x`` (already projected with the previous sweep's
            δ in update_lattice_world_positions).
        max_radius: ``model.particle_max_radius``; the hash query
            radius is ``r_i + max_radius``, matching the contact pass.
        eps: Smoothing width for ``smooth_relu`` (CSLCParams.smoothing_eps).
        alpha_damping: Jacobi damping factor ∈ (0, 1].  1.0 = undamped.
        ka_tangent_ratio: ``k_a_t / k_a``; anisotropic anchor stiffness
            on the two in-plane (tangential) axes.
        k_stick, mu_friction: stick-slip parameters (zero disables
            friction; see ``cslc_kernels.jacobi_step`` for the form).
        clamp_delta_max: Optional upper bound on ``|δ|`` [m]; pass a
            negative sentinel to disable.
    """
    sid = wp.tid()

    # Mirror cslc_kernels.jacobi_step's "lattice filter" — preserve
    # non-active or interior spheres' warm-start by copying src to dst.
    if lattice_is_surface[sid] == wp.uint8(0):
        delta_dst[sid] = delta_src[sid]
        return

    pidx = lattice_particle_index[sid]
    if (particle_flags[pidx] & ParticleFlags.ACTIVE) == 0:
        delta_dst[sid] = delta_src[sid]
        return

    link = lattice_link[sid]
    tf = body_q[link]
    rot = wp.transform_get_rotation(tf)
    n_world = wp.quat_rotate(rot, lattice_normal[sid])

    # δ at start of this sweep (in CSLC sign convention ``q = p − δ``).
    delta_old = delta_src[sid]

    # World-frame REST sphere position (the "anchor" — where the
    # spring's body end is attached).  Read from body_q + p_rest, NOT
    # from particle_x[pidx] (which carries the deformed position from
    # the previous sweep's update_lattice_world_positions).  The
    # deformed sphere centre is ``q_i = p_rest_world − δ_old``.
    p_pin_world = wp.transform_point(tf, lattice_p_rest[sid])
    q_i_world = p_pin_world - delta_old
    r_i = particle_radius[pidx]

    art_i = wp.int32(-1)
    if link >= 0:
        art_i = body_articulation[link]

    # ────────────────────────────────────────────────────────────────
    # Lateral graph-Laplacian force (mirror cslc_kernels.jacobi_step
    # lines ~611-616).  CSR neighbour list; empty list ⇒ no contribution.
    # ────────────────────────────────────────────────────────────────
    f_lateral = wp.vec3(0.0, 0.0, 0.0)
    kl = lattice_k_lateral[sid]
    n_neighbors = lattice_neighbor_count[sid]
    if n_neighbors > 0 and kl > 0.0:
        start = lattice_neighbor_start[sid]
        for k_idx in range(n_neighbors):
            j_sid = lattice_neighbor_list[start + k_idx]
            f_lateral = f_lateral - kl * (delta_old - delta_src[j_sid])

    # ────────────────────────────────────────────────────────────────
    # Contact sum over object-particle neighbours via the hash grid.
    # Mirrors cslc_kernels.jacobi_step's inner loop with two
    # substitutions:
    #   - target_count loop → hash-grid query
    #   - half-space raw   → sphere-sphere overlap
    # ────────────────────────────────────────────────────────────────
    kc = lattice_k_bulk[sid]
    f_contact_vec = wp.vec3(0.0, 0.0, 0.0)
    # Per-axis Jacobi diagonal stabilisers, split by alignment angle:
    # cos²α → S_n, sin²α → S_t (mirror cslc_kernels lines ~767-775).
    # On flat patches cos²α = 1 ⇒ S_t = 0; on curved pads off-axis
    # spheres contribute kc·sin²α to the tangent-axis diagonal so the
    # iteration's contraction property holds along the tangent axis.
    sum_gate_n = float(0.0)
    sum_gate_t = float(0.0)

    query = wp.hash_grid_query(grid, q_i_world, r_i + max_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == pidx:
            continue
        if (particle_flags[j] & ParticleFlags.ACTIVE) == 0:
            continue
        sub_j = particle_substrate[j]
        # Skip lattice-vs-lattice on same body or articulation, skip fluid
        # (parity with solve_particle_particle_contacts_uxpbd).
        if sub_j == wp.uint8(0):
            other_lat = particle_to_lattice[j]
            if other_lat < 0:
                continue
            host_j = lattice_link[other_lat]
            if host_j == link:
                continue
            if host_j >= 0 and art_i >= 0 and art_i == body_articulation[host_j]:
                continue
        if sub_j == wp.uint8(3):
            continue

        diff_qt = q_i_world - particle_x[j]
        d = wp.length(diff_qt)
        if d < 1.0e-12:
            continue
        r_j = particle_radius[j]
        # Sphere-vs-sphere overlap form (UXPBD substitute for the CSLC
        # half-space form ``raw = r − n_face · (q − t)``; for SM-rigid
        # sphere-packed objects the line-of-centres overlap is exact).
        raw = (r_i + r_j) - d
        # Hard cull on raw ≤ 0 — no smooth gate here (UXPBD is
        # forward-only; the smooth_step in cslc_kernels exists so
        # wp.Tape backprops cleanly through the contact threshold).
        if raw <= 0.0:
            continue
        # Pair-normal alignment: positive when the neighbour sits on
        # the inward side of the pad's outward normal.  UXPBD's
        # substitute for ``α = -(n_face · n_pad)`` in cslc_kernels.
        n_pair = diff_qt / d  # from neighbour TO pad sphere
        align_arg = -wp.dot(n_pair, n_world)
        if align_arg <= 0.0:
            continue
        # Hertz-like phi_eff with smoothed ReLU (mirrors
        # cslc_kernels.jacobi_step lines ~715-716).  C∞ at raw = 0 so
        # per-contact stiffness vanishes at first touch.
        raw_pos = smooth_relu(raw, eps)
        phi_eff = raw_pos * wp.sqrt(raw_pos + eps)
        # Per-particle area weight A_j (UXPBD substitute for the CSLC
        # target Voronoi area).  For a close-packed sphere object,
        # spacing ≈ 2·r_obj_j, so A_j = (2·r_obj_j)² = 4·r_obj_j².
        A_j = 4.0 * r_j * r_j
        # Contact load on the pad sphere: ``+kc·A_j·phi_eff·n_face``
        # in cslc_kernels, where ``n_face`` is the OBJECT's outward
        # face normal.  Here we use ``n_eff = -n_pair`` (pair normal
        # from neighbour to pad sphere = pad outward direction at the
        # contact = analogue of ``-n_face`` since the object is "on
        # the other side" of the contact).  Compression pushes the
        # pad sphere along ``+n_eff`` (away from the obj particle);
        # the load form on δ is the negative of that — see CSLC
        # contract §6.5 ``load = -∂E/∂δ``.  With ``δ`` measured
        # outward, the contact force on δ is along ``+n_eff`` (pad
        # outward from contact = pad's own outward = compressed
        # inward in δ-space).
        n_eff = -n_pair
        f_contact_vec = f_contact_vec + kc * A_j * phi_eff * n_eff
        # Diagonal stabilisation, split per-axis by alignment angle.
        # align_arg = -(n_pair · n_pad) = (n_eff · n_pad) = cos α.
        cos_a = align_arg
        cos2 = cos_a * cos_a
        contrib = kc * A_j
        sum_gate_n = sum_gate_n + contrib * cos2
        sum_gate_t = sum_gate_t + contrib * (1.0 - cos2)

    # ────────────────────────────────────────────────────────────────
    # Stick-slip friction (mirror cslc_kernels.jacobi_step
    # lines ~777-802).  ``M = μ·|F_n|`` is the Coulomb cone magnitude;
    # ``K = k_stick`` the pre-slip shear stiffness.  Defaults
    # ``k_stick = mu_friction = 0`` disable.
    # ────────────────────────────────────────────────────────────────
    f_friction_vec = wp.vec3(0.0, 0.0, 0.0)
    if k_stick > 0.0 and mu_friction > 0.0:
        f_n_signed = wp.dot(f_contact_vec, n_world)
        f_n_mag = wp.abs(f_n_signed)
        delta_proj_n = wp.dot(delta_old, n_world)
        delta_t = delta_old - delta_proj_n * n_world
        delta_t_mag = wp.length(delta_t)
        M = mu_friction * f_n_mag
        scale_used = (k_stick * M) / (k_stick * delta_t_mag + M + 1.0e-30)
        f_friction_vec = -scale_used * delta_t

    # ────────────────────────────────────────────────────────────────
    # Anisotropic block-Jacobi in pad sphere's local rest-normal frame
    # (mirror cslc_kernels.jacobi_step lines ~813-844).
    # ────────────────────────────────────────────────────────────────
    rhs_explicit = f_contact_vec + f_lateral + f_friction_vec
    rhs_n_scalar = wp.dot(rhs_explicit, n_world)
    rhs_t_vec = rhs_explicit - rhs_n_scalar * n_world

    delta_old_n = wp.dot(delta_old, n_world)
    delta_old_t = delta_old - delta_old_n * n_world

    ka = lattice_k_anchor[sid]
    ka_t = ka * ka_tangent_ratio
    S_n = kl * float(n_neighbors) + sum_gate_n
    S_t = kl * float(n_neighbors) + sum_gate_t
    k_diag_n = ka + S_n
    k_diag_t = ka_t + S_t

    rhs_n_total = rhs_n_scalar + S_n * delta_old_n
    rhs_t_total = rhs_t_vec + S_t * delta_old_t

    if k_diag_n <= 0.0:
        delta_dst[sid] = delta_src[sid]
        return
    delta_jacobi_n = rhs_n_total / k_diag_n
    delta_jacobi_t = wp.vec3(0.0, 0.0, 0.0)
    if k_diag_t > 0.0:
        delta_jacobi_t = rhs_t_total / k_diag_t
    delta_jacobi = delta_jacobi_n * n_world + delta_jacobi_t

    new_delta = (1.0 - alpha_damping) * delta_old + alpha_damping * delta_jacobi

    if clamp_delta_max >= 0.0:
        mag = wp.length(new_delta)
        if mag > clamp_delta_max and mag > 0.0:
            new_delta = new_delta * (clamp_delta_max / mag)

    delta_dst[sid] = new_delta


# ═══════════════════════════════════════════════════════════════════════════
#  Body wrench from CSLC anchor reactions
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def accumulate_cslc_body_wrench(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    lattice_link: wp.array[wp.int32],
    lattice_p_rest: wp.array[wp.vec3],
    lattice_normal: wp.array[wp.vec3],
    lattice_is_surface: wp.array[wp.uint8],
    lattice_k_anchor: wp.array[wp.float32],
    lattice_delta: wp.array[wp.vec3],
    ka_tangent_ratio: float,
    dt: float,
    # outputs
    body_delta: wp.array[wp.spatial_vector],
):
    """Per-sphere anchor-reaction body wrench from CSLC compression.

    The anchor connects pad sphere ``i`` to its host body via an
    anisotropic spring with normal stiffness ``k_a`` and tangent
    stiffness ``k_a · ratio`` in the pad's local rest-normal frame.
    A vec3 displacement ``δ_i`` (CSLC sign: outward positive)
    decomposes as ``δ_n_scalar = δ · n_outward``,
    ``δ_t = δ − δ_n_scalar · n_outward``; the spring force on the
    sphere is

        F_sphere = -k_a · δ_n_scalar · n_outward  -  k_a · ratio · δ_t

    By Newton's 3rd, ``F_body = -F_sphere = +k_a · δ_n · n_outward +
    k_a · ratio · δ_t``... wait that's pushing the body INTO the
    object, which is wrong.  The right derivation:

    The spring REST length is set so the SPHERE rests at ``p_rest``.
    When the sphere is compressed inward by ``δ`` (sphere moves to
    ``p_rest - δ``), the spring is COMPRESSED along ``+n_outward`` by
    ``δ_n_scalar`` (positive δ_n means closer body↔sphere distance).
    A compressed spring pushes its ends APART: sphere outward (along
    ``+n_outward``), body inward (along ``-n_outward``).  So
    ``F_body = -k_a · δ_n_scalar · n_outward`` (= ``-k_a · (δ ·
    n_outward) · n_outward``, the normal-axis component of ``-k_a · δ``).

    Tangentially, ``δ_t`` represents the sphere shearing relative to
    the body; the spring exerts ``+k_a·ratio · δ_t`` on the body to
    restore relative position.

    Combining: ``F_body = -k_a · δ_n_scalar · n_outward + k_a·ratio · δ_t``.
    When ratio = 1 (isotropic), this collapses to
    ``F_body = -k_a · (δ_n · n_outward) + k_a · (δ − δ_n · n_outward)
              = k_a · (δ − 2·δ_n·n_outward) = k_a · (δ_t − δ_n · n_outward)``.
    That's NOT equal to ``-k_a · δ`` in general — the normal axis
    flips sign relative to the tangent under our convention.

    Cross-check: when ``δ = δ_n · n_outward`` (pure compression, no
    shear), ``δ_t = 0``, ``F_body = -k_a · δ_n · n_outward``.  For the
    left pad with ``n_outward = +X`` and ``δ_n > 0``, ``F_body``
    points along ``-X`` — pushes the pad body AWAY from the object
    (the ball at +X side pressing on the pad pushes the pad to more
    negative X).  ✓

    Run inside the iteration loop after the pp contact pass has
    zeroed ``body_delta`` and written its non-lattice contributions.
    ``body_contact_count`` is NOT incremented here: anchor reactions
    are physically additive parallel springs and should not be
    averaged down by UPPFRTA constraint-count division.
    """
    sid = wp.tid()

    if lattice_is_surface[sid] == wp.uint8(0):
        return
    delta = lattice_delta[sid]
    # Cheap early-out for spheres with no compression at all (norm-zero
    # vec3 contributes nothing).
    delta_mag_sq = wp.dot(delta, delta)
    if delta_mag_sq <= 0.0:
        return
    link = lattice_link[sid]
    if link < 0:
        return

    tf = body_q[link]
    rot = wp.transform_get_rotation(tf)
    n_world = wp.quat_rotate(rot, lattice_normal[sid])

    # Normal / tangent split of δ in the pad's local rest-normal frame.
    delta_n_scalar = wp.dot(delta, n_world)
    delta_t = delta - delta_n_scalar * n_world

    # Anchor spring force on the BODY.  Normal axis: compressed spring
    # pushes body inward (-n_world).  Tangent axes: shear spring drags
    # body along the same direction the sphere was sheared (+δ_t).
    k_a = lattice_k_anchor[sid]
    k_a_t = k_a * ka_tangent_ratio
    F_body = (-k_a * delta_n_scalar) * n_world + k_a_t * delta_t  # [N]

    # Cheap early-out if numerically zero after the split.
    if wp.length(F_body) <= 0.0:
        return

    # World-frame point where the force is applied (the deformed sphere
    # centre).  ``r = sphere_world - com_world`` is the lever arm.
    p_rest_world = wp.transform_point(tf, lattice_p_rest[sid])
    sphere_world = p_rest_world - delta
    com_world = wp.transform_point(tf, body_com[link])
    r_sphere = sphere_world - com_world

    # Convert force → impulse for body_delta accumulator (matches the
    # unit convention of pp/shape contact kernels in this solver).
    impulse = F_body * dt
    torque_impulse = wp.cross(r_sphere, impulse)

    wp.atomic_add(body_delta, link, wp.spatial_vector(impulse, torque_impulse))
