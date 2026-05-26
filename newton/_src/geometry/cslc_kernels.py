#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC Warp GPU kernels for contact generation (CSLC v2, unified path).

v2 contract: every target geometry (sphere, box, mesh, dome, ...) is a
:class:`PointSetTarget` of ``(position, normal, area)`` triples — no
target sample radius.  The contact direction is the target's outward
face normal, and the per-pair raw overlap is the half-space form

    raw = r_i − n̂_face · (q − t)                                       (v2)

monotone in penetration depth at any depth (no sign flip past face
crossing).  See ``cslc_main/theory/contract_v2.md`` §3 for the full
spec; the v1 sphere-vs-sphere kernels were removed in Phase 5.

Active-set culling, per contract §3.6:

* **Raw gate** — pairs with ``raw < −50·ε`` are skipped (the threshold
  literal MUST match ``INACTIVE_RAW_EPS_FACTOR`` in
  ``cslc_main/theory/cslc_theory.py``).
* **Alignment gate** — one-sided smoothstep on ``[0, +EPS_ALIGN]`` with
  ``EPS_ALIGN = 0.05``.  Back-side and perpendicular samples are
  hard-culled; the smooth taper is one-sided on the face-on edge of
  the band.

File location: newton/_src/geometry/cslc_kernels.py
"""

import warp as wp


# ═══════════════════════════════════════════════════════════════════════════
#  Smooth differentiable surrogates for ReLU and step
#
#  Replace hard `if x > 0` ops with smooth analogues so wp.Tape can backprop
#  through CSLC contact dynamics for MPC and RL workflows.  All branches
#  that gate on continuous physical quantities (raw, d_t, ...) are smoothed
#  with eps as the transition width [m].  eps → 0 recovers the non-smooth
#  behavior; production default eps = 5e-4 m gives essentially-binary
#  forces above ~5 mm and a smooth C^∞ transition through the threshold.
#
#  smooth_relu(x, eps) = 0.5 * (x + sqrt(x² + eps²))
#      → max(x, 0) as eps → 0
#      derivative is smooth_step(x, eps), well-defined at x = 0
#
#  smooth_step(x, eps) = 0.5 * (1 + x / sqrt(x² + eps²))
#      → 1 if x >> eps, → 0 if x << -eps, smooth sigmoid-like through 0
#
#  Both are C^∞ for eps > 0 and have bounded gradients.
# ═══════════════════════════════════════════════════════════════════════════


@wp.func
def smooth_relu(x: float, eps: float) -> float:
    return 0.5 * (x + wp.sqrt(x * x + eps * eps))


@wp.func
def smooth_step(x: float, eps: float) -> float:
    return 0.5 * (1.0 + x / wp.sqrt(x * x + eps * eps))


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 1: Argmax-overlap warm-start penetration (point-set target)
#
#  For each active surface pad sphere, walks the target point set and
#  picks the SINGLE most-overlapping target sample (argmax over j of
#  raw_ij at the rest position), writing a scalar ``phi_rest`` and the
#  rest-frame load-direction normal.  Consumed by
#  ``lattice_solve_equilibrium`` as a tape-compatible linear warm-start
#  for the Jacobi solve.
#
#  Non-smoothness caveat
#  ---------------------
#  argmax(j) is non-smooth in pose at the boundary where two target
#  points have near-equal overlap with the same pad sphere.  Fine for
#  warm-start convergence (the nonlinear Jacobi sweeps absorb the
#  resulting transient) but breaks differentiability of the warm-start
#  with respect to body pose.  Downstream gradient-based optimisation
#  (MPC, RL, sim-to-real policy gradients) that wants to backprop
#  through the lattice solve will eventually need a softmax-blended
#  weighted-average warm-start; see C3+ deferred items in notes.md.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_cslc_penetration(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    # Kept in the signature for symmetry with the production data flow
    # (the warm-start is evaluated at REST position; the Jacobi
    # iteration recomputes raw at the deformed centre each iteration).
    sphere_delta: wp.array(dtype=wp.vec3),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    # Pad outward normal (rest, shape-local).  Used for the contract
    # §3.6 alignment gate; transformed to world inline.
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    # Point-set target.  v2: no per-sample radius (raw = r_i − n̂·(q−t)).
    target_positions_local: wp.array(dtype=wp.vec3),
    # Per-target outward face normal (target body-local).  The contact
    # direction is the target surface's outward normal at the sample;
    # half-space raw = r_lat − n_face · (q − t) is monotone in
    # penetration depth at any depth (no sign flip past face crossing).
    target_normals_local: wp.array(dtype=wp.vec3),
    target_count: int,
    eps: float,
    raw_penetration: wp.array(dtype=wp.float32),
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """REST half-space argmax-overlap penetration per lattice sphere (warm-start).

    For each active surface pad sphere i, picks the target sample j*
    with the largest rest half-space overlap

        raw_ij = r_i − n̂_face_j · (p_i − t_j)                             (v2)

    subject to ALL THREE active-set gates from contract §3.6 / §3.5:

      (1) ``raw_ij ≥ −50·eps`` (raw cull; literal MUST match
          ``INACTIVE_RAW_EPS_FACTOR`` in cslc_main/theory/cslc_theory.py),
      (2) one-sided alignment ``align_arg = −(n̂_face · n̂_pad) > 0``
          (back-side AND perpendicular samples hard-culled; literal
          0.05 MUST match ``EPS_ALIGN_DEFAULT``),
      (3) tangential locality ``w_tangent ≥ 1e-2`` where ``w_tangent =
          smooth_step(3·r_i − d_t, eps)`` and ``d_t`` is the tangential
          distance from the pad centre to the sample in the sample's
          face plane.  Without this cull the argmax can pick samples
          where ``n_face`` is steeply tilted relative to (q − t),
          inflating raw to ``R + r_i + offset`` — well-defined for an
          infinite half-space but nonsense for closed curved targets
          (a sphere sample on the far side of the surface from the pad
          still satisfies (1)+(2) and yields raw ≈ R + |c − p_i|).
          The iteration kernel ``jacobi_step`` already gates contact
          force by ``w_tangent``; mirroring it here keeps the warm-start
          phi_rest consistent with the equilibrium phi the iteration
          converges to (contract §3.5 eq:w_t).  Literal 1e-2 matches
          the hard-cull threshold in :func:`write_cslc_contacts`.

    All three gates match :func:`jacobi_step`'s effective active set
    so the warm-start picks the same sample the iteration kernel would
    saturate on.

    Outputs:
      * ``raw_penetration[i] = smooth_relu(raw_ij*, eps) * smooth_step(L*, eps)``
      * ``contact_normal_out[i] = -n_face_world_j*``  (load direction)

    Pad spheres with no target sample passing the gates get phi = 0
    and n = 0; the warm-start linear solve gives near-zero local
    displacement for those spheres (lateral coupling still propagates
    other spheres' warm-start delta through the Laplacian).
    """
    tid = wp.tid()

    # Active-lattice filter.
    if sphere_shape[tid] != active_cslc_shape_idx:
        raw_penetration[tid] = 0.0
        contact_normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    if is_surface[tid] == 0:
        raw_penetration[tid] = 0.0
        contact_normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws = shape_transform[s_idx]
    X_wb = body_q[b_idx]

    p_local = sphere_pos_local[tid]
    r_lat = sphere_radii[tid]

    q_body = wp.transform_point(X_ws, p_local)
    q_world = wp.transform_point(X_wb, q_body)

    out_n_local = sphere_outward_normal[tid]
    out_n_world = wp.transform_vector(
        X_wb, wp.transform_vector(X_ws, out_n_local))

    X_tb = body_q[target_body_idx]

    # Argmax search over target points.  ``found`` distinguishes "no
    # target passed both gates" (phi := 0) from "at least one active
    # pair".  Track dist_best separately for the smooth_step output
    # factor.
    found = int(0)
    raw_best = float(0.0)
    dist_best = float(0.0)
    n_best = wp.vec3(0.0, 0.0, 0.0)

    for j in range(target_count):
        t_j_world = wp.transform_point(X_tb, target_positions_local[j])
        diff_qt = q_world - t_j_world
        dist = wp.length(diff_qt)
        # Degenerate centres-coincide check.  1e-15 is numerical zero
        # (NOT ``eps``, which is the smooth-gate width); skipping pairs
        # at the smooth-gate radius would silently drop deeply-
        # overlapping contacts.
        if dist < 1.0e-15:
            continue
        n_face_world = wp.transform_vector(X_tb, target_normals_local[j])
        # Half-space raw (v2): monotone in penetration depth at any
        # depth, doesn't flip sign at face crossing.  Contract eq:raw.
        raw_j = r_lat - wp.dot(diff_qt, n_face_world)
        # Raw active-set gate (contract §3.6 eq:inactive).  MUST match
        # INACTIVE_RAW_EPS_FACTOR = -50.0 in cslc_main/theory/cslc_theory.py.
        if raw_j < -50.0 * eps:
            continue
        # One-sided alignment gate (contract §3.6, amended Phase 4b):
        #     align_arg = -(n_face · n_pad)        (+1 face-on, -1 back, 0 perp)
        # Hard-cull at align_arg <= 0 (back-to-back AND perpendicular).
        # The smooth taper on (0, EPS_ALIGN_DEFAULT) lives in
        # ``jacobi_step``; here we only need to KNOW the pair is
        # active for the argmax search, so the binary hard-cull edge
        # of the gate is sufficient.  The smooth taper does not change
        # which sample is argmax (it only scales the magnitude, which
        # we threshold against once at the loop's end via raw).
        align_arg = -wp.dot(n_face_world, out_n_world)
        if align_arg <= 0.0:
            continue
        # Tangential-locality gate (contract §3.5 eq:w_t).  Without
        # this cull the argmax over the half-space form can pick a
        # sample where ``n_face`` is steeply tilted relative to
        # ``(q − t)`` — e.g. a sphere sample on the far side of the
        # surface from the pad still satisfies the raw + align gates
        # and yields raw ≈ R + |c − p_i|.  The iteration kernel
        # (jacobi_step) multiplies contact force by ``w_tangent``, so
        # such a sample contributes ZERO force at equilibrium; emitting
        # phi_rest with it as the argmax drives the lattice solver to a
        # spurious δ ≈ kc·raw/(ka+kc) at step 0.  Hard cull at
        # ``w_tangent < 1e-2`` matches the emission cull in
        # write_cslc_contacts and keeps the warm-start consistent with
        # the equilibrium force law.
        d_t_vec = diff_qt - wp.dot(diff_qt, n_face_world) * n_face_world
        d_t_mag = wp.length(d_t_vec)
        kernel_h = 3.0 * r_lat
        w_tangent = smooth_step(kernel_h - d_t_mag, eps)
        if w_tangent < 1.0e-2:
            continue
        if (found == 0) or (raw_j > raw_best):
            found = 1
            raw_best = raw_j
            dist_best = dist
            # Warm-start contact direction is the load-form -n_face
            # (matches ``jacobi_step``'s n_eff).  The downstream
            # lattice_solve_equilibrium reads this as the warm-start
            # contact direction and projects it onto the pad sphere's
            # local frame.
            n_best = -n_face_world

    if found == 1:
        # DEFENSIVE: the dist < 1e-15 continue above ensures dist_best > 0
        # here, so smooth_step ~= 1.  Do NOT remove the degenerate-check
        # continue without re-deriving this gate.
        phi = smooth_relu(raw_best, eps) * smooth_step(dist_best, eps)
        raw_penetration[tid] = phi
        contact_normal_out[tid] = n_best
    else:
        raw_penetration[tid] = 0.0
        contact_normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2a: Linear warm-start solve  (NOT theory-aligned for finite delta)
#
#  Closed-form solve of the LINEARISED system
#      (K + kc·I) δ = kc · φ_rest
#  where K is the CSLC Laplacian (anchor ka + lateral kl) and φ_rest is
#  kernel 1's argmax-overlap rest overlap.  Uses a precomputed dense
#  A_inv built once in CSLCData.from_lattices (build_A_inv=True) for an
#  O(n²) matvec.
#
#  Role: a tape-compatible warm-start for jacobi_step.  At δ = 0 the
#  series-spring contact load is exactly kc · φ_rest · n_eff, so this
#  matvec recovers the small-δ equilibrium in closed form.  At finite
#  δ the true load is kc · A_j · w_t · a · φ_eff · gate · n̂_face
#  (jacobi_step), which this kernel does NOT see -- jacobi_step
#  refines from the warm-start delta and corrects it.
#
#  Why keep it: production grasp scenes save 10-20 jacobi iterations
#  per step by warm-starting with a δ that's already in the right
#  ballpark.  Also preserves wp.Tape backward through the lattice
#  solve when downstream MPC / RL needs gradients (the iterative
#  jacobi src/dst ping-pong breaks wp.Tape).
#
#  Notes:
#    – Ungated: treats all surface spheres as contributing.  For
#      non-overlap spheres, φ ≈ 0 (kernel 1 smooth_relu clip), so their
#      warm-start δ stays near zero.
#    – Lateral kl coupling is baked into K and therefore A_inv.
#    – O(n²) time and memory; for n ≳ 1000 prefer sparse Cholesky.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def lattice_solve_equilibrium(
    # Per-axis dense inverses.  A_inv_n absorbs the contact spring
    # (+kc·I) on its diagonal and solves the NORMAL-axis scalar field.
    # A_inv_t uses the anisotropic-scaled anchor (ka·ratio) with no
    # contact spring and solves the TANGENT-axis vec3 field.
    A_inv_n: wp.array2d(dtype=wp.float32),      # (n_spheres, n_spheres)
    A_inv_t: wp.array2d(dtype=wp.float32),      # (n_spheres, n_spheres)
    phi: wp.array(dtype=wp.float32),            # (n_spheres,) -- phi_rest
    # Rest-frame load direction per sphere (written by
    # `compute_cslc_penetration` as -n_face_world at the argmax j*).
    # Per-sphere contact force in the linearised model is
    # kc·phi_rest·(-n_face); we decompose it in each sphere's own local
    # rest-normal frame so the anisotropic anchor (normal vs tangent
    # stiffness) applies on the correct components.
    contact_normal_world: wp.array(dtype=wp.vec3),
    # World-frame rest outward normal per sphere, precomputed by
    # `compute_outward_normals_world` once per launch.  Defines each
    # sphere's local axis frame for the normal/tangent decomposition.
    out_normal_world: wp.array(dtype=wp.vec3),
    kc: float,
    delta_out: wp.array(dtype=wp.vec3),         # (n_spheres,) vec3
):
    """Linear warm-start for the CSLC lattice solve.

    This kernel solves the LINEARISED equilibrium ``(K + kc·I) δ =
    kc · φ_rest`` -- a tangent-space approximation of the series-
    spring law at δ = 0.  At finite δ the true contact load is the
    half-space form ``kc · A_j · w_t · a · φ_eff · gate · n̂_face``
    (see ``jacobi_step``); this kernel does NOT capture that
    nonlinearity.  It is used only as a tape-compatible warm-start;
    every callsite follows it with damped-Jacobi refinement that
    corrects the nonlinear residual.

    For each lattice sphere j, splits the contact force kc·φ_j·n_eff_j
    into its own local rest-normal frame:

        f_n_j  =  φ_j · dot(n_eff_j, n_outward_j)         (scalar, j-frame)
        f_t_j  =  φ_j · n_eff_j  −  f_n_j · n_outward_j   (vec3, j-frame)

    Applies the appropriate per-axis closed-form inverse:

        δ_n_i  =  kc · Σ_j  A_inv_n[i, j] · f_n_j         (scalar)
        δ_t_i  =  kc · Σ_j  A_inv_t[i, j] · f_t_j         (vec3 in world)

    Recomposes at sphere i using i's own outward normal:

        δ_i  =  δ_n_i · n_outward_i  +  (δ_t_i − dot(δ_t_i, n_outward_i)·n_outward_i)

    The final tangent projection at sphere i removes any normal-axis
    component that crept into the world-frame accumulator from neighbours
    whose tangent plane differs from i's (a graph-Laplacian cross-axis
    coupling).  For flat patches this projection is exact; for curved
    domes it is the same near-flat approximation used inside
    `jacobi_step` for its per-iter normal/tangent split.

    Reduces algebraically to the legacy isotropic A⁻¹·(kc·φ·n_eff) solve
    when ka_tangent_ratio = 1.0 and all n_outward are parallel (face-on
    contact), because A_inv_n − A_inv_t differs only by the +kc·I
    term on A_inv_n, which is multiplied only by the normal-axis force
    scalar in that limit.
    """
    i = wp.tid()
    n = A_inv_n.shape[1]
    n_i = out_normal_world[i]

    # δ_n_i (scalar, i-frame normal axis)
    acc_n = float(0.0)
    # δ_t_i (world-frame, perp i-normal in expectation)
    acc_t = wp.vec3(0.0, 0.0, 0.0)

    for j in range(n):
        phi_j = phi[j]
        n_eff_j = contact_normal_world[j]
        n_j = out_normal_world[j]

        # Decompose per-sphere force in j's local frame.
        proj_j = wp.dot(n_eff_j, n_j)
        f_n_j_scalar = phi_j * proj_j
        f_t_j_vec = phi_j * (n_eff_j - proj_j * n_j)

        a_n = A_inv_n[i, j]
        a_t = A_inv_t[i, j]

        acc_n = acc_n + a_n * f_n_j_scalar
        acc_t = acc_t + a_t * f_t_j_vec

    # Recompose in i's local frame.  Project the world-frame tangent
    # accumulator onto i's tangent plane to remove any normal-axis
    # bleed from curvature-induced cross-axis coupling.
    acc_t_proj = acc_t - wp.dot(acc_t, n_i) * n_i
    delta_out[i] = kc * (acc_n * n_i + acc_t_proj)


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2a': World-frame outward normal precompute
#
#  Per-launch transform of each lattice sphere's rest outward normal
#  (stored in CSLCData.outward_normals, expressed in the shape's local
#  body frame) into world frame.  Used by `lattice_solve_equilibrium` to
#  decompose the contact force into the sphere's local normal/tangent
#  components for the anisotropic-anchor closed-form solve.
#
#  Could be inlined inside `lattice_solve_equilibrium` (the kernel reads
#  out_normal_world[j] for every j in its inner loop), but factoring it
#  out avoids recomputing the same transform n_spheres times per matvec
#  and keeps the solve kernel free of body_q / shape_transform reads.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_outward_normals_world(
    # rest frame, shape-local
    sphere_outward_normal_local: wp.array(dtype=wp.vec3),
    sphere_shape: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    out_normal_world: wp.array(dtype=wp.vec3),
):
    """Transform each lattice sphere's rest outward normal into world frame."""
    tid = wp.tid()
    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws = shape_transform[s_idx]
    X_wb = body_q[b_idx]
    n_local = sphere_outward_normal_local[tid]
    n_body = wp.transform_vector(X_ws, n_local)
    out_normal_world[tid] = wp.transform_vector(X_wb, n_body)


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2b: Active-lattice-selective copy
#
#  When `lattice_solve_equilibrium` runs for one pair (active lattice P), it
#  writes a per-sphere δ for *every* sphere — including spheres on other
#  pads, where φ was zeroed in Kernel 1 and so δ ends up at zero.
#  Unconditionally copying that full δ buffer back into
#  `CSLCData.sphere_delta` would wipe the other lattice body's warm-start every
#  step, leaving only the *last* pair's lattice with non-zero compression in
#  `sphere_delta`.  That breaks visualisation (lattice viz reads
#  `sphere_delta` and would render only one lattice as compressed) and warm
#  starts (next step's φ for the wiped lattice starts from δ=0 again).
#
#  The iterative jacobi path doesn't have this problem because
#  `jacobi_step` has an active-lattice branch that pass-throughs δ for
#  non-active lattices.  This selective-copy kernel adds the same guard to
#  the dense-solve path's writeback: only update sphere_delta[i] when
#  sphere i belongs to the active lattice.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def cslc_copy_active(
    src: wp.array(dtype=wp.vec3),
    sphere_shape: wp.array(dtype=wp.int32),
    active_cslc_shape_idx: int,
    dst: wp.array(dtype=wp.vec3),
):
    """dst[i] ← src[i] only for spheres on the active CSLC lattice.

    Used by ``cslc_handler._launch`` to merge the dense-solve output
    back into ``CSLCData.sphere_delta`` without clobbering other
    lattice bodies' warm-starts.
    """
    i = wp.tid()
    if sphere_shape[i] == active_cslc_shape_idx:
        dst[i] = src[i]


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2: Damped Jacobi iteration (CSLC v2 unified half-space contact)
#
#  Each pad sphere accumulates contact contributions from every active
#  target sample in an inner loop; anchor, lateral, friction, and the
#  damped Jacobi update follow contract §6.5.  See
#  ``cslc_main/theory/contract_v2.md`` §4 for the per-pair energy /
#  force, §6 for the lattice equilibrium, and §3.6 for the active-set
#  gates.
#
#  Friction uses the pad sphere's own outward normal as the local frame
#  (same as the anisotropic-anchor decomposition).  ``f_n_mag = |F_contact
#  · n_outward|`` -- the aggregate normal-axis component of the multi-
#  point contact wrench.  Contract §6.4 ``f_t = K·M·s/(K·s + M)`` form
#  with K = k_stick, M = μ·f_n, s = |δ_t|; rewritten to avoid 1/s (see
#  Phase 4a finding #10 in contract §17).
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def jacobi_step(
    delta_src: wp.array(dtype=wp.vec3),
    delta_dst: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_pos_local: wp.array(dtype=wp.vec3),
    is_surface: wp.array(dtype=wp.int32),
    neighbor_start: wp.array(dtype=wp.int32),
    neighbor_count: wp.array(dtype=wp.int32),
    neighbor_list: wp.array(dtype=wp.int32),
    ka: float,
    kl: float,
    kc: float,
    alpha: float,
    sphere_shape: wp.array(dtype=wp.int32),
    active_cslc_shape_idx: int,
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    ka_tangent_ratio: float,
    k_stick: float,
    mu_friction: float,
    # Point-set target.  v2: no per-sample radius (raw = r_i − n̂·(q−t)).
    target_positions_local: wp.array(dtype=wp.vec3),
    # Per-target outward face normal (target body-local).  The contact
    # direction is the target surface's outward normal at the sample;
    # half-space raw = r_lat − n_face · (q − t) is monotone in
    # penetration depth at any depth (no sign flip past face crossing).
    target_normals_local: wp.array(dtype=wp.vec3),
    # Per-target Voronoi area on the underlying surface [m^2].  Folded
    # into the contact force with a tangential locality kernel
    # w_tangent to reconstruct the surface integral
    #     F = ∫_{contact_patch} kc · phi · n_face dA
    # from the discrete sample set (contract §5).
    target_areas_local: wp.array(dtype=wp.float32),
    target_count: int,
    target_body_idx: int,
    # External tangential load (bridge / experiment driver only).
    # ``f_ext_apex_idx = -1`` is the no-op sentinel (production).
    f_ext_apex_idx: int,
    f_ext_apex: wp.vec3,
    eps: float,
):
    """One damped Jacobi sweep for the ACTIVE lattice against a PointSetTarget.

    Equilibrium (per active surface sphere i, in pad sphere i's local
    rest-normal frame; contract §6.5 load form, ``load = −∂E/∂δ``):

      0 =  -K_anchor · δ_i
           - k_l · Σ_{j∈N(i)} (δ_i − δ_j)                          (graph-Laplacian lateral)
           + Σ_{m=0..M-1} k_c · A_m · w_t_im · a_im · phi_eff_im · gate_im · n_eff_im
                                                                   (contact, point-set)
           + f_friction(δ_t, |F_contact · n_outward|, k_stick, μ)  (stick-slip)
           + f_ext_apex                                            (only at apex_idx)

    where for each target sample m (contract §3.2, §3.4, §3.5, §3.6):
       raw_im      = r_lat_i − n_face_m · (q_i − t_m)               (half-space)
       phi_eff_im  = smooth_relu(raw_im, eps)
       gate_im     = smooth_step(raw_im, eps)
       d_t_im      = ‖(q_i − t_m) − n_face_m · (q_i − t_m) · n_face_m‖
       w_t_im      = smooth_step(3 r_lat_i − d_t_im, eps)
       a_im        = smoothstep(α_im / EPS_ALIGN, 0, 1) on [0, +EPS_ALIGN]
                   = 0 if α_im ≤ 0 (back-side / perpendicular hard-cull)
                   = 1 if α_im ≥ +EPS_ALIGN (face-on)
       α_im        = -(n_face_m · n_pad_i)
       n_eff_im    = -n_face_m       (load form: pushes pad along -n_face)

    Active-set gates (contract §3.6):
      (a) ``raw_im < -50·eps``  -- raw cull, literal MUST match
          INACTIVE_RAW_EPS_FACTOR in cslc_main/theory/cslc_theory.py.
      (b) ``a_im = 0`` (α_im ≤ 0)  -- alignment hard-cull; literal 0.05
          MUST match EPS_ALIGN_DEFAULT in cslc_main/theory/cslc_theory.py.
    """
    tid = wp.tid()

    # Lattice filter — preserve warm-start on non-active lattices.
    if sphere_shape[tid] != active_cslc_shape_idx:
        delta_dst[tid] = delta_src[tid]
        return

    delta_old = delta_src[tid]
    n_neighbors = neighbor_count[tid]

    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws = shape_transform[s_idx]
    X_wb = body_q[b_idx]

    p_i_local = sphere_pos_local[tid]
    p_i_world = wp.transform_point(X_wb, wp.transform_point(X_ws, p_i_local))
    q_i_world = p_i_world - delta_old

    out_n_local = sphere_outward_normal[tid]
    out_n_world = wp.transform_vector(
        X_wb, wp.transform_vector(X_ws, out_n_local))

    # Graph-Laplacian lateral force (contract §6.2):
    #     f_lat_i = -k_l · Σ_{j∈N(i)} (δ_i − δ_j)
    # No rest-length term — this is the linearisation of the distance-
    # preserving spring around δ = 0; the nonlinear DP law is deleted
    # in v2 (contract §15).
    f_lateral = wp.vec3(0.0, 0.0, 0.0)
    start = neighbor_start[tid]
    for n in range(n_neighbors):
        edge = start + n
        j = neighbor_list[edge]
        f_lateral = f_lateral - kl * (delta_old - delta_src[j])

    # Point-set CONTACT: sum over target samples.
    f_contact_vec = wp.vec3(0.0, 0.0, 0.0)
    sum_gate = float(0.0)
    f_friction_vec = wp.vec3(0.0, 0.0, 0.0)

    if is_surface[tid] == 1:
        r_i = sphere_radii[tid]
        X_tb = body_q[target_body_idx]
        for j in range(target_count):
            t_j_world = wp.transform_point(X_tb, target_positions_local[j])
            # Degenerate centres-coincide check.  1e-15 is numerical
            # zero (NOT ``eps``, the smooth-gate width); skipping pairs
            # at the smooth-gate radius would silently drop deeply-
            # overlapping contacts.
            diff_qt = q_i_world - t_j_world
            if wp.length(diff_qt) < 1.0e-15:
                continue
            n_face_world = wp.transform_vector(X_tb, target_normals_local[j])
            # Half-space raw (v2, contract eq:raw):
            #     raw = r_i − n_face · (q − t)
            # Monotone in penetration depth at any depth, doesn't flip
            # sign at face crossing (unlike the v1 sphere-overlap
            # form which assumed a single sphere target with radius
            # R and used (r_i + R) − ‖q − t‖).
            raw = r_i - wp.dot(diff_qt, n_face_world)
            # Half-space gate (literal MUST match INACTIVE_RAW_EPS_FACTOR
            # = -50.0 in cslc_main/theory/cslc_theory.py).
            if raw < -50.0 * eps:
                continue
            # Smooth alignment gate (contract §3.6, amended Phase 4b
            # after finding #11).  Closed convex targets (sphere, box,
            # mesh) have back-side AND side-face samples whose outward
            # normal points AWAY from or perpendicular to the pad — the
            # half-space form alone would treat them as deep contacts
            # because the pad centre IS inside their half-plane.  The
            # one-sided alignment cull
            #
            #     align_arg = -(n_face · n_pad)         (+1 face-on, -1 back, 0 perp)
            #     align_w   = 1                                if align_arg >= +EPS_ALIGN
            #                 cubic smoothstep on [0, +EPS_ALIGN]
            #                 0                                if align_arg <= 0
            #
            # gives a C¹-smooth weight (no force discontinuity at the
            # face-on edge of the band) with COMPACT support — back-side
            # AND perpendicular contributions are EXACTLY zero, not a
            # 1/x² tail, not the 0.5-midpoint of the earlier symmetric
            # band that over-coupled side faces of closed convex targets
            # (Phase 4a finding #11; full box scene previously
            # non-convergent at production eps).  Theory parity in
            # cslc_main/theory/cslc_lattice.solve_lattice_contact uses
            # the same one-sided form.  The literal 0.05 below MUST
            # match EPS_ALIGN_DEFAULT in cslc_main/theory/cslc_theory.py;
            # bridge parity (Phase 4 / T-K) regression-guards it via
            # _check_constant_discipline in test_07_kernel_bridge.
            align_arg = -wp.dot(n_face_world, out_n_world)
            if align_arg <= 0.0:
                continue
            align_w = float(1.0)
            if align_arg < 0.05:
                t_lerp = wp.clamp(align_arg / 0.05, 0.0, 1.0)
                align_w = t_lerp * t_lerp * (3.0 - 2.0 * t_lerp)
            phi_eff = smooth_relu(raw, eps)
            gate = smooth_step(raw, eps)
            # Tangential locality kernel (contract §3.5 eq:w_t).
            # Kernel half-width = 3 · r_pad (covers the typical Hertz
            # patch + several sample spacings; under-sampling at
            # ``r_pad ≈ pitch`` would otherwise give an empty active
            # set).  The (A_j · w_tangent) factor reconstructs the
            # surface integral F = ∫ kc · phi · n_face dA from the
            # discrete sample set; without it the coherent face-normal
            # sum overcounts by a factor of (face_area_in_reach /
            # contact_patch_area).
            d_t_vec = diff_qt - wp.dot(diff_qt, n_face_world) * n_face_world
            d_t_mag = wp.length(d_t_vec)
            kernel_h = r_i
            w_tangent = smooth_step(kernel_h - d_t_mag, eps)
            A_j = target_areas_local[j]
            area_kernel = A_j * w_tangent
            # Load form: load = -∂E_contact/∂δ.  Physical force on the
            # pad sphere is +kc · A_j · w_tangent · align · phi_eff
            # · gate · n_face (face's outward normal); load form is
            # the negative of that.  ``phi_eff = smooth_relu(raw, eps)``
            # is always ≥ 0, so the contact force never reverses sign
            # near the smooth-cull boundary — critical for the
            # no-bulge regression on multi-sphere lattices (contract
            # §6.2 / T-G).
            n_eff = -n_face_world
            f_contact_vec = f_contact_vec + kc * \
                area_kernel * align_w * phi_eff * gate * n_eff
            # Diagonal stabilisation needs the same area+kernel+align
            # weights so the contraction bound matches the actual operator.
            sum_gate = sum_gate + area_kernel * align_w * gate

        # Stick-slip friction.  Aggregate normal-axis magnitude used as
        # the cone reference; tangent decomposition done in pad outward
        # frame (same as the anisotropic-anchor decomposition below).
        f_n_signed = wp.dot(f_contact_vec, out_n_world)
        # Compression should give f_contact_vec · n_outward < 0 (force
        # pushes pad sphere along -n_outward, i.e. into the body).
        # Take absolute value for the cone magnitude either way; the
        # cone is symmetric.
        f_n_mag = wp.abs(f_n_signed)

        delta_proj_n_outward = wp.dot(delta_old, out_n_world)
        delta_t = delta_old - delta_proj_n_outward * out_n_world
        delta_t_mag = wp.length(delta_t)
        # Contract §6.4 friction law: f_t = K·M·s/(K·s + M).  Rewritten
        # without the legacy ``inv_dt_mag = s/(s² + eps²)`` regulariser
        # (Phase 4a finding #10) — at production eps = 5×10⁻⁴ m,
        # stick-mode |δ_t| ~ μm gives ``inv_dt_mag ≈ s/eps²``, three
        # orders of magnitude too small, and friction effectively
        # vanishes.  The contract form avoids 1/s entirely.  ``1e-30``
        # is a denormalise-floor that protects the (k_stick = 0 AND
        # μ·f_n = 0) no-friction corner; in any active friction regime
        # K·s + M ≫ 1e-30 so the floor is numerically invisible.
        # Matches ``cslc_theory.friction_force_smooth`` exactly.
        M = mu_friction * f_n_mag
        scale_used = (k_stick * M) / (k_stick * delta_t_mag + M + 1.0e-30)
        f_friction_vec = -scale_used * delta_t

    # External tangential load (bridge / experiment driver only; -1 in
    # production).  Sign convention: f_ext_apex is the physical
    # external force on the sphere body (i.e. on q, where q = p − δ).
    # Its potential is V_ext = +f_ext · δ, so the LOAD on δ is
    # −dV/dδ = −f_ext.
    f_ext_vec = wp.vec3(0.0, 0.0, 0.0)
    if tid == f_ext_apex_idx:
        f_ext_vec = -f_ext_apex

    # Anisotropic block-Jacobi in pad sphere's local rest-normal frame
    # (contract §6.5).  S_n picks up the SUM of contact gates × area
    # × align over the target samples.  This sum upper-bounds
    # |d(f_contact·n)/d(δ_n)| over the active set, so the iteration's
    # contraction property holds.
    rhs_explicit = f_contact_vec + f_lateral + f_friction_vec + f_ext_vec
    rhs_n_scalar = wp.dot(rhs_explicit, out_n_world)
    rhs_t_vec = rhs_explicit - rhs_n_scalar * out_n_world

    delta_old_n = wp.dot(delta_old, out_n_world)
    delta_old_t = delta_old - delta_old_n * out_n_world

    ka_t = ka * ka_tangent_ratio
    S_n = kl * float(n_neighbors) + kc * sum_gate
    S_t = kl * float(n_neighbors)
    k_diag_n = ka + S_n
    k_diag_t = ka_t + S_t

    rhs_n_total = rhs_n_scalar + S_n * delta_old_n
    rhs_t_total = rhs_t_vec + S_t * delta_old_t

    delta_jacobi_n = rhs_n_total / k_diag_n
    delta_jacobi_t = rhs_t_total / k_diag_t
    delta_jacobi = delta_jacobi_n * out_n_world + delta_jacobi_t

    delta_dst[tid] = (1.0 - alpha) * delta_old + alpha * delta_jacobi


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 3: Per-pair contact emission for PointSetTarget
#
#  Per (pad_sphere, target_sample) pair that passes the active-set
#  gates at the converged delta, emit one MuJoCo contact -- contract §8
#  emission convention:
#
#    point0    = q_def = p_world - sphere_delta[i]   (deformed pad centre)
#    point1    = t_j_world                           (target sample)
#    normal    = -n_face_world                       (target outward -> pad)
#    margin0   = r_i
#    margin1   = 0                                   (v2: was R_j in v1)
#    stiffness = cslc_kc · A_j · w_tangent · align · gate
#                (cslc_kc already carries the (pad ⊕ target) series-spring
#                composition, done up front in
#                CSLCHandler.from_model_with_lattices)
#    friction  = μ
#
#  MuJoCo reconstructs
#    solver_pen = margin0 + margin1 - (point1 - point0) · normal
#               = r_i + 0 - (t_j - q_def) · (-n_face)
#               = r_i - n_face · (q_def - t_j)
#               = raw                                              ✓
#  so the per-contact force MuJoCo applies, ``stiffness · solver_pen``,
#  equals ``cslc_kc · A_j · w_tangent · align · gate · raw`` --
#  matches ``jacobi_step``'s per-pair force in the deep-saturated
#  limit (``phi_eff ≈ raw`` when ``raw ≫ ε``).  Contract §8.
#
#  Buffer layout
#  -------------
#  Pad sphere ``i`` (with ``surface_slot_map[i] = s_i >= 0``) writes its
#  contacts to absolute slots
#
#       contact_offset + s_i · K_max + 0 .. K_max-1.
#
#  Excess slots (beyond the actual K_i ≤ K_max active pairs) are filled
#  with the ``out_shape0 = -1`` sentinel, so the downstream MuJoCo
#  conversion kernel culls them via its ``shape_a < 0`` early-out.
#
#  K_max sizing
#  ------------
#  Total active pairs per pad sphere is bounded by the number of target
#  samples inside the inclusion radius r_inclusion = r_lat +
#  INCLUSION_FACTOR · eps of the pad sphere's deformed centre that pass
#  both gates (raw and alignment).  See
#  ``cslc_main.grasp.objects.compute_k_max`` for the production sizing
#  helper.  K_max = 32 is the production default; comfortable margin,
#  doesn't blow the buffer at production lattice sizes.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def write_cslc_contacts(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.vec3),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    target_shape_idx: int,
    # Point-set target.  v2: no per-sample radius.
    target_positions_local: wp.array(dtype=wp.vec3),
    # Per-target outward face normal (target body-local).  Used to (a)
    # compute the half-space penetration that MuJoCo reconstructs as
    # solver_pen, and (b) emit the contact normal as the FACE NORMAL
    # so the resolved contact force on the pad sphere points along
    # the face's outward direction regardless of which side of the
    # face the pad sphere center is on.
    target_normals_local: wp.array(dtype=wp.vec3),
    # Per-target Voronoi area [m^2] -- folded into the emitted contact
    # stiffness so MuJoCo applies force ``kc_series · A_j · w_tangent
    # · align · gate · solver_pen`` per contact, reconstructing the
    # surface integral of the half-space pressure field.
    target_areas_local: wp.array(dtype=wp.float32),
    target_count: int,
    contact_offset: int,
    K_max: int,
    surface_slot_map: wp.array(dtype=wp.int32),
    out_shape0: wp.array(dtype=wp.int32),
    out_shape1: wp.array(dtype=wp.int32),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_margin0: wp.array(dtype=wp.float32),
    out_margin1: wp.array(dtype=wp.float32),
    out_tids: wp.array(dtype=wp.int32),
    # shape_material_mu: no longer read inside this kernel.  Previously
    # used as ``out_friction = mu`` (the lattice body's friction
    # coefficient), which caused a double-count: the MuJoCo conversion
    # kernel multiplies rigid_contact_friction onto the geom pair base
    # friction (already = mu), giving effective_mu = mu² instead of mu.
    # Fix writes out_friction = 1.0 (no scale), so geom friction is
    # used as-is.  Kept in the signature to avoid breaking the handler
    # call; remove in a future cleanup.
    shape_material_mu: wp.array(dtype=wp.float32),
    cslc_kc: float,
    target_ke: float,
    cslc_dc: float,
    eps: float,
    out_stiffness: wp.array(dtype=wp.float32),
    out_damping: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
    # Per-pair truncation counter.  Atomic-incremented once per pad
    # sphere whose active-pair count would have exceeded K_max.  Read
    # on CPU by the handler after collide() to RuntimeWarning if any
    # pair block under-sized K_max; see cslc_handler._launch.
    truncation_count: wp.array(dtype=wp.int32),
):
    """Emit one MuJoCo contact per active (pad_sphere, target_sample) pair.

    Active pairs pass both contract §3.6 gates:
        (a) ``raw_ij >= -50 * eps``  -- raw cull, MUST match
            INACTIVE_RAW_EPS_FACTOR in cslc_main/theory/cslc_theory.py;
        (b) ``align_arg = -(n_face · n_pad) > 0``  -- one-sided
            alignment cull (back-side + perpendicular hard-culled);
            literal 0.05 below MUST match EPS_ALIGN_DEFAULT.
    Plus the production emission threshold:
        (c) ``contact_gate >= 1e-4``  -- deep tail is sub-nN force,
            machine-zero gradient; emitting these slots measurably
            degrades MuJoCo's soft-constraint solver via the per-slot
            compliance leak (verified against the lift test).

    Each emitted contact carries margin0 = r_lat[i], margin1 = 0;
    MuJoCo reconstructs solver_pen = r_lat - n_face · (q_def - t_j) =
    raw_ij per pair, so per-contact force = stiffness · solver_pen
    = kc_series · A_j · w_tangent · align · gate · raw -- the same
    series-spring law that ``jacobi_step`` converges on.
    """
    tid = wp.tid()
    base_slot = surface_slot_map[tid]
    if base_slot < 0:
        return

    # Initialise this pad sphere's K_max slot block with the "no
    # contact" sentinel.  Excess slots beyond the active pair count
    # stay culled.
    for k in range(K_max):
        buf_idx_init = contact_offset + base_slot * K_max + k
        out_shape0[buf_idx_init] = -1
        out_stiffness[buf_idx_init] = 0.0

    # Pair filter -- only the active CSLC lattice writes.
    if sphere_shape[tid] != active_cslc_shape_idx:
        return

    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws = shape_transform[s_idx]
    X_wb = body_q[b_idx]
    X_wb_inv = wp.transform_inverse(X_wb)

    p_i_local = sphere_pos_local[tid]
    r_i = sphere_radii[tid]
    q_world = wp.transform_point(X_wb, wp.transform_point(X_ws, p_i_local))
    q_world_def = q_world - sphere_delta[tid]

    out_n_local = sphere_outward_normal[tid]
    out_n_world = wp.transform_vector(
        X_wb, wp.transform_vector(X_ws, out_n_local))

    X_tb = body_q[target_body_idx]
    X_tb_inv = wp.transform_inverse(X_tb)

    # Series stiffness composition is now done UPFRONT in
    # ``CSLCHandler.from_model_with_lattices`` (Phase 6 amendment to
    # contract §11): the per-sphere kc returned by ``calibrate_kc`` is
    # composed with the target body's ke before the per-volume rescale,
    # so ``cslc_kc`` here already carries the (pad ⊕ target) series
    # spring rate.  Using ``cslc_kc`` directly matches
    # ``jacobi_step``'s contact load formula (no kc_series re-composition
    # in either kernel).  ``target_ke`` is retained in the signature
    # for backward compat; it is no longer read.
    kc_emit = cslc_kc

    pair_count = int(0)
    truncated = int(0)
    for j in range(target_count):
        t_world = wp.transform_point(X_tb, target_positions_local[j])
        diff_qt = q_world_def - t_world
        if wp.length(diff_qt) < 1.0e-15:
            continue

        n_face_world = wp.transform_vector(X_tb, target_normals_local[j])
        # Half-space raw (v2, contract eq:raw):
        #     raw = r_i − n_face · (q_def − t)
        raw = r_i - wp.dot(diff_qt, n_face_world)
        # Half-space gate (literal MUST match INACTIVE_RAW_EPS_FACTOR
        # = -50.0 in cslc_main/theory/cslc_theory.py).
        if raw < -50.0 * eps:
            continue
        # One-sided alignment gate (contract §3.6).  Literal 0.05 MUST
        # match EPS_ALIGN_DEFAULT in cslc_main/theory/cslc_theory.py.
        # See ``jacobi_step`` for the full rationale; bridge parity
        # (Phase 4) regression-guards both sites via
        # ``_check_constant_discipline`` in test_07_kernel_bridge.
        align_arg = -wp.dot(n_face_world, out_n_world)
        if align_arg <= 0.0:
            continue
        align_w = float(1.0)
        if align_arg < 0.05:
            t_lerp = wp.clamp(align_arg / 0.05, 0.0, 1.0)
            align_w = t_lerp * t_lerp * (3.0 - 2.0 * t_lerp)

        normal_ab = -n_face_world
        contact_gate = smooth_step(raw, eps)
        # Tangential locality kernel -- folded into emitted stiffness
        # below so MuJoCo applies per-contact force
        #     stiffness · solver_pen
        #     = kc_series · A_j · w_tangent · align · gate · raw
        # matching the lattice solver's per-pair force exactly.
        d_t_vec = diff_qt - wp.dot(diff_qt, n_face_world) * n_face_world
        d_t_mag = wp.length(d_t_vec)
        # Kernel half-width = r_pad (Option-2 tiling, no overlap).
        # Must match jacobi_step's kernel reach; the emitted contact
        # stiffness uses the same area_kernel so MuJoCo's per-contact
        # force = stiffness · solver_pen equals the lattice solver's
        # per-pair force exactly.
        kernel_h = r_i
        w_tangent = smooth_step(kernel_h - d_t_mag, eps)
        A_j = target_areas_local[j]
        area_kernel = A_j * w_tangent

        # Hard cull below the production gate threshold.  Two
        # cumulative checks:
        #   (a) ``contact_gate < 0.5`` -- equivalent to ``raw < 0``: only
        #       slots with positive half-space penetration (physical
        #       contact) get emitted to MuJoCo.  The lattice solver
        #       still uses the full smooth-tail force law internally
        #       (jacobi_step keeps contact_gate / phi_eff at all
        #       ``raw >= -50·eps`` samples), so this cull does NOT
        #       affect the lattice equilibrium.  It only controls which
        #       slots become MuJoCo solver constraints — and emitting
        #       the deep tail ``raw ∈ (-25mm, 0)`` flooded MuJoCo's CG
        #       solver with ~5-15k near-zero-force constraints per
        #       step (per HOLD-phase diagnostic), turning a 2ms step
        #       into a 45ms step.  Tightening to ``raw >= 0`` drops
        #       MuJoCo's active constraint count by ~20× at HOLD with
        #       no change to grasp stability.  The negative-raw tail
        #       was also slightly unphysical: emission applied force
        #       ``kc·A·w·α·gate·raw`` which is NEGATIVE when raw < 0
        #       (an attractive "adhesion" force), while the lattice
        #       solver uses ``kc·A·w·α·phi_eff·gate`` where
        #       phi_eff = smooth_relu(raw) ≈ 0 for raw < 0 (no
        #       attraction).  Hard-culling raw < 0 in emission removes
        #       this asymmetry.
        #   (b) ``w_tangent < 1e-2`` -- sample is tangentially far
        #       outside the contact kernel.  1e-2 corresponds to d_t ≈
        #       3·r_pad + 3·eps (the physical contact patch boundary);
        #       samples past this contribute < 1% of a central sample's
        #       force, well below MuJoCo's solver resolution.
        if contact_gate < 0.5 or w_tangent < 1.0e-2:
            continue

        # Pair j is active AND emittable.  Now check buffer space:
        # placing the overflow check HERE (rather than at the top of
        # the loop) avoids false-positive truncation warnings when the
        # remaining target indices j..target_count-1 are all inactive
        # (would be skipped by the raw / contact_gate culls above).
        if pair_count >= K_max:
            truncated = 1
            break

        # Body-frame contact geometry.  Contract §8 emission convention:
        # point0 = deformed pad centre, point1 = target sample,
        # margin0 = r_i, margin1 = 0 (v2 change vs v1, which carried
        # margin1 = R_j).
        p0_body = wp.transform_point(X_wb_inv, q_world_def)
        p1_body = wp.transform_point(X_tb_inv, t_world)
        offset0_body = wp.transform_vector(X_wb_inv,  r_i * normal_ab)
        offset1_body = wp.transform_vector(X_tb_inv, wp.vec3(0.0, 0.0, 0.0))

        buf_idx = contact_offset + base_slot * K_max + pair_count
        out_shape0[buf_idx] = s_idx
        out_shape1[buf_idx] = target_shape_idx
        out_point0[buf_idx] = p0_body
        out_point1[buf_idx] = p1_body
        out_offset0[buf_idx] = offset0_body
        out_offset1[buf_idx] = offset1_body
        out_normal[buf_idx] = normal_ab
        out_margin0[buf_idx] = r_i
        # Contract §8: v2 sets margin1 = 0 (v1 used R_j).  MuJoCo's
        # reconstructed solver_pen then reduces to the half-space raw
        # algebraically: see kernel docstring.
        out_margin1[buf_idx] = 0.0
        out_tids[buf_idx] = 0
        # Fold A_j · w_tangent · align into the emitted stiffness so
        # MuJoCo's per-contact force ``stiffness · solver_pen`` equals
        # the lattice solver's per-pair force ``kc · A_j · w_tangent
        # · align · gate · raw``.  ``cslc_kc`` already carries the
        # (pad ⊕ target) series-spring composition (done upfront in
        # ``CSLCHandler.from_model_with_lattices``); here we add the
        # discrete-area + locality + alignment factors that convert
        # the per-volume stiffness into a per-contact spring constant.
        out_stiffness[buf_idx] = smooth_relu(
            kc_emit * area_kernel * align_w * contact_gate, 1.0e-9)
        # cslc_dc retained in signature.  Writing 0.0 uses MuJoCo's
        # ``kd = 0`` branch ⇒ timeconst = sqrt(imp/ke) ≈ 0.030 s.
        # Setting kd > 0 would trigger timeconst = 2/kd, making
        # friction constraints 250× softer than the standard contact
        # and causing excessive Coulomb creep in the HOLD phase.
        out_damping[buf_idx] = 0.0
        # FRICTION SCALE: the MuJoCo conversion kernel treats
        # rigid_contact_friction as a SCALE FACTOR multiplied onto the
        # geom pair's base friction:
        #     effective_mu = geom_friction_max × rigid_contact_friction
        # The geom pair base friction is max(mu_pad, mu_sphere) = mu.
        # Writing 1.0 here gives effective_mu = mu × 1.0 = mu.  Writing
        # ``shape_material_mu`` instead would give mu² (a bug fixed
        # 2026-04-19).
        out_friction[buf_idx] = 1.0

        pair_count = pair_count + 1

    if truncated == 1:
        # One atomic per pad sphere that overflowed -- not per dropped
        # pair -- so the CPU read after collide() reports the number of
        # affected pad spheres, which is the more actionable count.
        wp.atomic_add(truncation_count, 0, 1)
