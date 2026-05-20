#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC Warp GPU kernels for contact generation.


File location: newton/_src/geometry/cslc_kernels.py
"""

import warp as wp


# ═══════════════════════════════════════════════════════════════════════════
#  Smooth differentiable surrogates for ReLU and step
#
#  Replace hard `if x > 0` ops with smooth analogues so wp.Tape can backprop
#  through CSLC contact dynamics for MPC and RL workflows.  All four kernel
#  branches that gate on continuous physical quantities (pen_3d, d_proj,
#  effective_pen, delta clamps) are smoothed with eps as the transition
#  width [m].  eps → 0 recovers the original non-smooth behavior; default
#  eps = 1e-5 m gives essentially-binary forces above 0.1 mm and a smooth
#  C^1 transition through the threshold.
#
#  smooth_relu(x, eps) = 0.5 * (x + sqrt(x² + eps²))
#      → max(x, 0) as eps → 0
#      derivative is smooth_step(x, eps), well-defined at x=0
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
#  Kernel 1: Penetration (lattice sphere vs target sphere)
#
#  Only sphere-vs-sphere is supported.  Mesh / box / SDF targets are
#  follow-on work (step 7b): re-add a target-specific penetration kernel
#  plus the matching contact-emit, both reusing the theory-aligned
#  contact-force code in `jacobi_step` and `write_cslc_contacts`.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_cslc_penetration(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    # Not read here.  This kernel returns the REST overlap phi_rest
    # and the rest-frame line-of-centers normal; jacobi_step
    # recomputes the EXACT deformed overlap phi_def and direction
    # n_eff_def inline each iteration (Step 7 D2).  sphere_delta is
    # kept in the signature so the kernel could be re-extended later
    # without an API change (e.g. for a deformed-overlap kernel-1
    # variant); current callers do not depend on it.
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
    target_local_pos: wp.vec3,
    target_radius: float,
    eps: float,
    raw_penetration: wp.array(dtype=wp.float32),
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """REST 3-D sphere-sphere overlap per lattice sphere (differentiable).

    Returns ``phi_rest = (r_lat + R) - ||p_world - t_world||`` at the
    rest position, plus the rest-frame line-of-centers normal.
    Consumers:

    * ``lattice_solve_equilibrium`` (warm-start) reads phi_rest as its
      RHS load and contact_normal as the direction.
    * ``jacobi_step`` and ``write_cslc_contacts`` use the rest normal
      as the degenerate fallback for the deformed direction; they
      otherwise recompute phi_def and n_eff_def inline from
      sphere_radii + target_local_pos each iteration (Step 7 D2/D4).
    """
    tid = wp.tid()
    phi     = 0.0
    n_world = wp.vec3(0.0, 0.0, 0.0)

    # Active-lattice filter.  Discrete index branch — not differentiable, but
    # the parameter being indexed (sphere_shape) is integer-valued and
    # never a learning target.
    if sphere_shape[tid] != active_cslc_shape_idx:
        raw_penetration[tid] = 0.0
        return

    if is_surface[tid] == 1:
        s_idx = sphere_shape[tid]
        b_idx = shape_body[s_idx]
        X_ws  = shape_transform[s_idx]
        X_wb  = body_q[b_idx]

        p_local = sphere_pos_local[tid]
        r_lat   = sphere_radii[tid]

        q_body  = wp.transform_point(X_ws, p_local)
        q_world = wp.transform_point(X_wb, q_body)

        X_tb    = body_q[target_body_idx]
        t_world = wp.transform_point(X_tb, target_local_pos)

        diff = t_world - q_world
        dist = wp.length(diff)

        # Rest-frame geometric (line-of-centers) contact normal from
        # p_world to t_world.  jacobi_step / write_cslc_contacts use
        # this only as a degenerate fallback when their own deformed
        # direction (t - q_def)/||t - q_def|| would divide by zero.
        if dist > eps:
            n_world = diff / dist
        else:
            # Degenerate: lattice and target centers coincide.  Cannot
            # happen during normal operation (radii prevent it), so fall
            # back to the pre-baked outward normal just to avoid NaN.
            out_n  = sphere_outward_normal[tid]
            n_body = wp.transform_vector(X_ws, out_n)
            n_world = wp.transform_vector(X_wb, n_body)
        # With normal == diff/dist, d_proj = dist and is positive whenever
        # the target is in front (always, by construction of pen_3d > 0).
        # The smooth_step(d_proj, eps) factor therefore evaluates to ~1.0,
        # leaving the contact-active gate driven purely by pen_3d > 0.
        d_proj = dist

        pen_3d = (r_lat + target_radius) - dist

        # Smooth contact-active gate:
        #   phi ≈ pen_3d when pen_3d > 0 AND d_proj > 0
        #   phi ≈ 0 otherwise
        # Continuous and C^∞ for eps > 0.
        phi = smooth_relu(pen_3d, eps) * smooth_step(d_proj, eps)

    raw_penetration[tid] = phi
    contact_normal_out[tid] = n_world



# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2a: Linear warm-start solve  (NOT theory-aligned for finite delta)
#
#  Closed-form solve of the LINEARISED system
#      (K + kc·I) δ = kc · φ_rest
#  where K is the CSLC Laplacian (anchor ka + lateral kl) and φ_rest is
#  kernel 1's rest overlap.  Uses a precomputed dense A_inv built once
#  in CSLCData.from_lattices (build_A_inv=True) for an O(n²) matvec.
#
#  Role: a tape-compatible warm-start for jacobi_step.  At δ = 0 the
#  series-spring contact load is exactly kc · φ_rest · n_eff, so this
#  matvec recovers the small-δ equilibrium in closed form.  At finite
#  δ the true load is kc · (φ_rest − dot(δ, n_eff_def)) · gate · n_eff_def
#  (Step 7 jacobi_step), which this kernel does NOT see -- jacobi_step
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
    # Rest-frame line-of-centers contact normal per sphere (written by
    # `compute_cslc_penetration`).  Per-sphere contact force in the
    # linearised model is kc·phi_rest·n_eff; we decompose it in each
    # sphere's own local rest-normal frame so the anisotropic anchor
    # (normal vs tangent stiffness) applies on the correct components.
    contact_normal_world: wp.array(dtype=wp.vec3),
    # World-frame rest outward normal per sphere, precomputed by
    # `compute_outward_normals_world` once per launch.  Defines each
    # sphere's local axis frame for the normal/tangent decomposition.
    out_normal_world: wp.array(dtype=wp.vec3),
    kc: float,
    delta_out: wp.array(dtype=wp.vec3),         # (n_spheres,) vec3
):
    """Linear warm-start for the CSLC lattice solve (Step 7b status note).

    This kernel solves the LINEARISED equilibrium ``(K + kc·I) δ =
    kc · φ_rest`` -- a tangent-space approximation of the
    series-spring law at δ = 0.  At finite δ the true contact load
    becomes ``kc · phi_eff(δ) · smooth_step(raw) · n_eff_def`` (see
    ``jacobi_step``); this kernel does NOT capture that
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

    acc_n = float(0.0)                         # δ_n_i (scalar, i-frame normal axis)
    acc_t = wp.vec3(0.0, 0.0, 0.0)             # δ_t_i (world-frame, perp i-normal in expectation)

    for j in range(n):
        phi_j = phi[j]
        n_eff_j = contact_normal_world[j]
        n_j = out_normal_world[j]

        # Decompose per-sphere force in j's local frame.
        proj_j = wp.dot(n_eff_j, n_j)
        f_n_j_scalar = phi_j * proj_j
        f_t_j_vec    = phi_j * (n_eff_j - proj_j * n_j)

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
    sphere_outward_normal_local: wp.array(dtype=wp.vec3),  # rest frame, shape-local
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
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]
    n_local = sphere_outward_normal_local[tid]
    n_body  = wp.transform_vector(X_ws, n_local)
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

    Used by `cslc_handler._launch_vs_sphere` to merge the dense-solve
    output back into `CSLCData.sphere_delta` without clobbering other
    lattice bodies' warm-starts.
    """
    i = wp.tid()
    if sphere_shape[i] == active_cslc_shape_idx:
        dst[i] = src[i]


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2: Damped Jacobi iteration
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def jacobi_step(
    # delta_src / delta_dst: per-sphere vec3 displacements in world
    # frame; deformed centre is q_i = p_i_world − δ_i.  Free-sign δ
    # everywhere (no clamp): side spheres may expand outward (negative
    # δ along their own outward normal) under the geometric Poisson
    # coupling produced by the distance-preservation lateral on
    # curved patches (verified in theory step 5).
    delta_src: wp.array(dtype=wp.vec3),
    delta_dst: wp.array(dtype=wp.vec3),
    # Per-sphere radii.  Used to compute the EXACT deformed overlap
    # phi_def = (r_lat + target_radius) - ||t_world - q_i_world||
    # (Step 7 D2).  Replaces the older linear approximation
    # raw = phi_rest - dot(δ, n_eff) which collapsed to ~0 at non-
    # contact spheres (kernel 1's smooth_relu clips phi_rest there),
    # making smooth_relu(0, eps) = eps/2 leak a spurious kc·eps/2
    # contact force at every surface sphere -- observable in
    # test_07 scene C as 4-15% chain delta error vs theory.
    sphere_radii: wp.array(dtype=wp.float32),
    # Per-sphere rest positions (body-local) and per-edge Euclidean
    # rest length L_ij.  Used by the distance-preservation lateral
    # spring f = k_l·(‖q_j − q_i‖ − L_ij)·ê_ij, which linearises to
    # the graph-Laplacian at δ→0 (so the paper's tangent-space
    # analysis still applies near rest) and produces geometric
    # Poisson bulging at finite δ on curved patches.
    sphere_pos_local: wp.array(dtype=wp.vec3),
    neighbor_rest_length: wp.array(dtype=wp.float32),
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
    # Stick-slip friction coefficients.  k_stick is the friction-patch
    # spring constant; mu_friction is the Coulomb cone slope.  At
    # k_stick = 0 or mu_friction = 0, no friction term contributes
    # (matches theory step 4's "no-friction" branch).
    k_stick: float,
    mu_friction: float,
    # Step 7 D2: target world position computed inline each iteration
    # (one transform_point per call) so the contact direction n_eff and
    # the deformed overlap raw both track the current q_def = p - δ.
    target_body_idx: int,
    target_local_pos: wp.vec3,
    target_radius: float,
    # Optional external tangential load for friction bridge tests.
    # If f_ext_apex_idx >= 0, the kernel adds `f_ext_apex` to the
    # rhs of that sphere (rhs_explicit += f_ext at tid == apex_idx).
    # Used by test_07's friction scenes (F stick / G slip) to drive
    # tangential displacement against the stick-slip cone.  Production
    # passes apex_idx = -1 (no-op).
    f_ext_apex_idx: int,
    f_ext_apex: wp.vec3,
    eps: float,
):

    """One damped Jacobi sweep for the ACTIVE lattice.

    Equilibrium (per active surface sphere i, with K_anchor diagonal
    in i's local rest-normal frame):

      0  =  −K_anchor · δ_i
            +  Σ_{j∈N(i)} k_l · ( ‖q_j − q_i‖ − L_ij ) · ê_ij(δ)   (lateral)
            +  k_c · phi_eff · gate · n_eff_def                    (contact, Step 7 D1+D2)
            +  f_friction(δ_t, f_n, k_stick, μ)                    (stick-slip)
            +  f_ext_apex            (only at sphere f_ext_apex_idx; bridge test driver)

    with  phi_eff = smooth_relu(raw, eps),  gate = smooth_step(raw, eps),
    and   raw = (r_lat + target_radius) − ||t_world − q_i_world||
    (the EXACT deformed overlap, not the linear phi_rest − dot(δ, n_eff)
    approximation; see the sphere_radii signature note for the leak
    that motivated this change).

    Damped Picard iteration with stabilised diagonal:

       (ka + k_l·|N| + k_c·gate) δ_n^(k+1)
            =  rhs_n  +  (k_l·|N| + k_c·gate) δ_n^(k)
       (ka_t + k_l·|N|)            δ_t^(k+1)
            =  rhs_t  +  k_l·|N| δ_t^(k)

    where rhs is the EXPLICIT per-iteration residual (contact + lateral
    + friction + external).  The gate's δ-dependence is handled
    implicitly via k_c·gate on the normal-axis LHS, with the matching
    k_c·gate·δ_old_n term added to rhs.  No clamp on δ_n -- anchor +
    lateral govern sign at equilibrium.
    """
    tid = wp.tid()

    # Lattice filter — preserve warm-start on non-active lattices.
    if sphere_shape[tid] != active_cslc_shape_idx:
        delta_dst[tid] = delta_src[tid]
        return

    delta_old = delta_src[tid]
    n_neighbors = neighbor_count[tid]

    # World-frame transforms (same chain as penetration kernel).
    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]

    p_i_local = sphere_pos_local[tid]
    p_i_world = wp.transform_point(X_wb, wp.transform_point(X_ws, p_i_local))
    q_i_world = p_i_world - delta_old

    out_n_local = sphere_outward_normal[tid]
    out_n_world = wp.transform_vector(X_wb, wp.transform_vector(X_ws, out_n_local))

    # Distance-preservation lateral force.  For each neighbour j on
    # this lattice body, evaluate the nonlinear spring
    # f = k_l·(dist − L_ij)·d̂  using the lagged δ_j.  Neighbours share
    # the same body+shape transforms (intra-lattice topology), so the
    # inverse transform applied to p_j_local is identical to i's.
    f_lateral = wp.vec3(0.0, 0.0, 0.0)
    start = neighbor_start[tid]
    for n in range(n_neighbors):
        edge = start + n
        j = neighbor_list[edge]
        p_j_local = sphere_pos_local[j]
        p_j_world = wp.transform_point(X_wb, wp.transform_point(X_ws, p_j_local))
        q_j_world = p_j_world - delta_src[j]
        d = q_j_world - q_i_world
        dist = wp.length(d)
        # Smooth reciprocal handles dist → 0 without branching: for
        # dist ≫ eps recovers 1/dist; at dist = 0 the d·0 product
        # vanishes safely.
        inv_dist = dist / (dist * dist + eps * eps)
        L_ij = neighbor_rest_length[edge]
        # Step 7 sign fix: f_lateral is the LOAD on delta, not the
        # physical force on q.  q = p - delta, so the load direction
        # is opposite to physical force.  Stretched edge (l > L) pulls
        # q_i toward q_j (physical force +kl*(l-L)*e_hat_ij), which
        # makes q_i move toward q_j and delta_i move AWAY from j (i.e.
        # in -e_hat_ij direction).  Load (= direction delta moves)
        # therefore carries a minus sign relative to the physical
        # force.  Hidden pre-step-7 because step 3 / production tests
        # only checked normal-axis macroscopic quantities that this
        # tangent-direction sign error doesn't move.
        f_lateral = f_lateral - kl * (dist - L_ij) * d * inv_dist

    f_contact_vec = wp.vec3(0.0, 0.0, 0.0)
    f_friction_vec = wp.vec3(0.0, 0.0, 0.0)
    gate          = float(0.0)

    if is_surface[tid] == 1:
        # Series-spring contact at the deformed centre (Step 7 D1+D2).
        #
        # Theory eq. 12 (cslc_main/theory/notes.md step 1):
        #
        #   raw       = (r_lat + target_radius) - ||t_world - q_def||
        #   phi_eff   = smooth_relu(raw, eps)
        #   gate      = smooth_step(raw, eps)
        #   n_eff_def = (t_world - q_def) / ||t_world - q_def||
        #   f_contact = kc * phi_eff * gate * n_eff_def
        #
        # The EXACT deformed overlap form (raw above) replaces the
        # older linear approximation `raw = phi_rest - dot(δ, n_eff)`.
        # Kernel 1's raw_penetration is smooth_relu-clipped to ~0 at
        # every non-contact sphere, so the linear form produced raw≈0
        # and smooth_relu(0, eps) = eps/2 leaked a spurious kc·eps/2
        # contact force at every surface sphere -- visible in
        # test_07 scene C as 4-15% chain error vs theory.  The exact
        # form gives strongly negative raw far from contact, so
        # smooth_relu(raw, eps) ≈ 0 there.  At the contact sphere
        # itself raw == phi_def in both formulations.
        #
        # n_eff_def is recomputed each iteration from the current
        # deformed centre; for face-on contact n_eff_def == n_eff_rest
        # exactly, but for off-axis / curved-lattice scenes the
        # difference is observable as the arc bulge sign in test_07
        # scene D and the kc≥ka magnitude error in scene B.
        r_lat = sphere_radii[tid]
        X_tb = body_q[target_body_idx]
        t_world = wp.transform_point(X_tb, target_local_pos)
        diff_def = t_world - q_i_world
        dist_def = wp.length(diff_def)
        if dist_def > eps:
            n_eff = diff_def / dist_def
        else:
            # Degenerate fallback (radii prevent it physically): use
            # the sphere's own world-frame outward normal so the
            # iteration stays well-defined.
            n_eff = out_n_world
        raw = (r_lat + target_radius) - dist_def
        phi_eff = smooth_relu(raw, eps)
        gate = smooth_step(raw, eps)
        # smooth_step gradient factor (theory bug #4).  Energy
        # E_contact = 0.5·kc·phi_eff² with phi_eff = smooth_relu(raw)
        # gives  dE/dδ = kc·phi_eff·smooth_step(raw)·d(raw)/dδ, so the
        # LOAD on δ (= −dE/dδ) carries the smooth_step factor.
        # Forgetting it is correct in deep saturated contact
        # (step → 1) but gives a spurious kc·(eps/2) leak at
        # non-contact spheres where step → 0 while smooth_relu still
        # has its irreducible eps/2 floor; the leak's projection onto
        # each sphere's out_n_world sign-flips the arc bulge
        # (test_07 scene D, formerly sign=FAIL across all kc/ka).
        # The implicit diagonal S_n = kl·|N| + kc·gate is unchanged:
        # an upper bound on |d(f_load)/d(δ_n)| at every raw, so the
        # damped Jacobi iteration stays contracting.
        f_contact_vec = kc * phi_eff * gate * n_eff
        # delta_proj_neff retained for friction's tangent split below.
        delta_proj_neff = wp.dot(delta_old, n_eff)
        # Friction normal-force magnitude inherits the same gate
        # factor, so non-contact spheres carry no friction either.
        f_n_mag = kc * phi_eff * gate

        # Stick-slip friction via tangential δ (theory step 4 + 6).
        # In the contact plane (perpendicular to n_eff), the compliant
        # skin's tangential deformation δ_t = δ − dot(δ, n_eff)·n_eff
        # generates a friction force f_t = −k_stick·δ_t in STICK mode.
        # The Coulomb cone clamps the magnitude at μ·f_n in SLIP mode.
        # Smoothed form:
        #     f_t = − δ_t · min(k_stick, μ·f_n / ‖δ_t‖)
        # implemented as a harmonic-mean smooth-min surrogate.  At
        # ‖δ_t‖ ≈ 0 the friction vanishes (no tangential deformation
        # → no force); as ‖δ_t‖ grows the linear stick law
        # k_stick·‖δ_t‖ takes over, then saturates at μ·f_n.
        delta_t = delta_old - delta_proj_neff * n_eff
        delta_t_mag = wp.length(delta_t)
        inv_dt_mag = delta_t_mag / (delta_t_mag * delta_t_mag + eps * eps)
        cone_scale = mu_friction * f_n_mag * inv_dt_mag
        # scale_used = harmonic mean of k_stick and cone_scale.  The
        # `+ eps` in the denominator is a kernel-parity regulariser
        # against an in-flight Jacobi iterate where k_stick +
        # cone_scale would otherwise hit exact zero (e.g. first
        # iteration with δ_t = 0 and f_n = 0).  Theory dropped this
        # term in its smooth surrogate (bug #5 fix) because
        # scipy.minimize_scalar / L-BFGS-B never see those transient
        # residuals; kernel keeps it consciously.  Below test
        # resolution at production eps; intentional, documented
        # divergence from theory.
        scale_used = (k_stick * cone_scale) / (k_stick + cone_scale + eps)
        f_friction_vec = -scale_used * delta_t

    # Optional external tangential load (bridge test only).  Production
    # passes f_ext_apex_idx = -1 so this branch is a no-op everywhere.
    # Used by test_07 scenes F / G to drive a single sphere against
    # the stick-slip cone and verify the kernel converges to the same
    # δ as the theory's hard-piecewise + smooth solvers.
    #
    # Sign convention (theory bug #6): `f_ext_apex` is the physical
    # external force on the sphere body (i.e. on q, where q = p − δ).
    # Its potential is V_ext = +f_ext · δ, so the LOAD on δ is
    # −dV/dδ = −f_ext.  We therefore subtract `f_ext_apex` from
    # rhs_explicit (it acts as a negative load on δ that drives δ
    # opposite to the physical force direction, since q moving in
    # +f_ext means δ moves in −f_ext).
    f_ext_vec = wp.vec3(0.0, 0.0, 0.0)
    if tid == f_ext_apex_idx:
        f_ext_vec = -f_ext_apex

    # Anisotropic block-Jacobi in the local rest-normal frame.  The
    # k_l·|N| stabilisation upper-bounds the linearised distance-
    # preservation operator at δ→0.  Friction enters the RHS like
    # lateral -- it acts only in the contact-plane tangent direction,
    # so its normal projection vanishes and the tangent-axis
    # equilibrium gains a restoring term anchor + lateral cannot
    # provide on their own.
    rhs_explicit = f_contact_vec + f_lateral + f_friction_vec + f_ext_vec
    rhs_n_scalar = wp.dot(rhs_explicit, out_n_world)
    rhs_t_vec    = rhs_explicit - rhs_n_scalar * out_n_world

    delta_old_n = wp.dot(delta_old, out_n_world)
    delta_old_t = delta_old - delta_old_n * out_n_world

    ka_t = ka * ka_tangent_ratio
    S_n  = kl * float(n_neighbors) + kc * gate
    S_t  = kl * float(n_neighbors)
    k_diag_n = ka + S_n
    k_diag_t = ka_t + S_t

    # Add S·δ_old to both sides — gives a contraction iteration whose
    # Lipschitz factor on the explicit residual is bounded by S/(ka+S).
    rhs_n_total = rhs_n_scalar + S_n * delta_old_n
    rhs_t_total = rhs_t_vec    + S_t * delta_old_t

    delta_jacobi_n = rhs_n_total / k_diag_n
    delta_jacobi_t = rhs_t_total / k_diag_t
    delta_jacobi   = delta_jacobi_n * out_n_world + delta_jacobi_t

    # No clamp on δ_n.  Free-sign δ everywhere: anchor resists
    # displacement of any sign; lateral distance-preservation drives
    # signs from the geometry of the neighbour graph.
    delta_dst[tid] = (1.0 - alpha) * delta_old + alpha * delta_jacobi


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 3: Write contacts to Newton's Contacts buffer
#
#  Emission convention (Step 7 D4):
#    point0    = q_def = q_world - sphere_delta[tid]   (deformed centre)
#    point1    = t_world
#    normal    = (t_world - q_def) / ||t_world - q_def||  (sphere -> target)
#    margin0   = r_lat                                 (untouched, no shim)
#    margin1   = target_radius
#    offset0   = +r_lat         * normal               (sphere surface point)
#    offset1   = -target_radius * normal               (target surface point)
#
#  MuJoCo reconstructs solver_pen = margin0 + margin1 - dot(point1_w -
#  point0_w, normal) = (r_lat + target_radius) - ||t_world - q_def||
#  = phi_def exactly -- the true deformed overlap that jacobi_step uses
#  for the contact spring.  pen_scale ≡ 1 by construction; the
#  per-contact force MuJoCo applies is `kc_series * contact_gate *
#  phi_def`, the theory's series-spring law (Step 7 D6).
#
#  History:
#    v3: point0 = rest sphere centre; offset0 = effective_r * outward_normal.
#        Worked for flat bodies, broke on curved (dome) bodies.
#    v4: line-of-centers normal from rest q_world; effective_r =
#        r_lat - dot(delta, normal_ab) shim approximated the deformed
#        overlap.  Linearly correct for face-on; diverged off-axis.
#    Step 7 D4 (current): deformed-centre point0 + r_lat untouched; pen_scale
#        collapses to 1; emission matches jacobi_step's deformed overlap
#        and the theory's series-spring law exactly.
# ═══════════════════════════════════════════════════════════════════════════

@wp.kernel
def write_cslc_contacts(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    # Converged vec3 δ from the lattice solve.  point0 is shifted by
    # -δ to emit at the deformed sphere centre q_def = p_world - δ
    # (Step 7 D4); no separate radius-reduction shim.
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
    target_local_pos: wp.vec3,
    target_radius: float,
    contact_offset: int,
    surface_slot_map: wp.array(dtype=wp.int32),
    raw_penetration: wp.array(dtype=wp.float32),
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
    # shape_material_mu: no longer read inside this kernel (2026-04-19).
    # Previously used as out_friction = mu (the lattice body's friction coefficient),
    # which caused a double-count: the MuJoCo conversion kernel multiplies
    # rigid_contact_friction onto the geom pair base friction (already = mu),
    # giving effective_mu = mu * mu = mu^2 instead of mu.
    # The fix writes out_friction = 1.0 (no scale), so geom friction is used as-is.
    # This parameter is kept in the signature to avoid breaking the handler call;
    # remove it from both here and cslc_handler.py in a future cleanup.
    shape_material_mu: wp.array(dtype=wp.float32),
    cslc_kc: float,
    target_ke: float,
    cslc_dc: float,
    eps: float,
    out_stiffness: wp.array(dtype=wp.float32),
    out_damping: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
    debug_reason: wp.array(dtype=wp.int32),
    # ── Diagnostic outputs (physics-neutral; read back by the handler) ──
    # Indexed by (diag_offset + slot) where diag_offset =
    # pair_idx * n_surface_contacts.  This lays each pair's diagnostics
    # out in its own block (same layout as the contacts buffer), so
    # pair_1's launch can't overwrite pair_0's diagnostic writes.
    diag_offset: int,
    dbg_pen_scale: wp.array(dtype=wp.float32),
    dbg_solver_pen: wp.array(dtype=wp.float32),
    dbg_effective_r: wp.array(dtype=wp.float32),
    dbg_d_proj: wp.array(dtype=wp.float32),
    dbg_radial: wp.array(dtype=wp.float32),
):
    """Write one rigid contact per lattice sphere at the deformed centre.

    Convention (Step 7 D4):
      point0  = q_def = q_world − sphere_delta[tid]  (deformed centre, body frame)
      point1  = t_world                              (target centre, body frame)
      normal  = (t_world − q_def) / ||t_world − q_def||  (sphere → target)
      margin0 = r_lat                                (untouched, no shim)
      margin1 = target_radius
      offset0 = +r_lat         · normal              (body frame)
      offset1 = −target_radius · normal              (body frame)

    The rigid-body solver reconstructs
        solver_pen = margin0 + margin1 − dot(point1_w − point0_w, normal)
                   = (r_lat + R) − ||t_world − q_def|| = phi_def
    which is the true deformed overlap.  pen_scale collapses to 1 by
    construction, and per-contact force = stiffness * solver_pen =
    kc_series * gate * phi_def matches jacobi_step's series-spring load.

    debug_reason codes:
        0 = wrote successfully
        3 = culled on contact_gate <= 1e-4  (smooth tail below threshold)
        4 = wrong lattice body for this pair
    """
    tid = wp.tid()

    slot = surface_slot_map[tid]
    if slot < 0:
        return
    buf_idx = contact_offset + slot
    # dslot: where THIS pair's diagnostic for THIS slot lives in the
    # handler's per-pair-per-slot diagnostic arrays.  Mirrors the layout
    # of the contacts buffer: pair_idx * n_surface_contacts + slot.
    dslot = diag_offset + slot

    # Pair filter.
    if sphere_shape[tid] != active_cslc_shape_idx:
        out_shape0[buf_idx] = -1
        debug_reason[slot]  = 4
        # Sentinel: negative pen_scale signals "no contact this slot".
        dbg_pen_scale[dslot]   = -1.0
        dbg_solver_pen[dslot]  = 0.0
        dbg_effective_r[dslot] = 0.0
        dbg_d_proj[dslot]      = 0.0
        dbg_radial[dslot]      = 0.0
        return

    # Shape A: lattice sphere.
    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]

    p_local  = sphere_pos_local[tid]
    r_lat    = sphere_radii[tid]

    q_body  = wp.transform_point(X_ws, p_local)
    q_world = wp.transform_point(X_wb, q_body)

    # Step 7 D4: deformed-centre emission.  Shift point0 by -delta so
    # the contact lives at q_def = p_world - delta instead of the rest
    # position.  Together with the unchanged r_lat (no radius-reduction
    # shim), MuJoCo's reconstructed
    #     solver_pen = margin0 + margin1 - dot(point1_w - point0_w, normal)
    # becomes (r_lat + target_radius) - dist_def = phi_def exactly --
    # the same deformed overlap jacobi_step uses for the contact
    # spring.  This makes the kernel emit `kc * phi_def` per contact
    # (series-spring law) instead of the constant `kc * phi_rest`
    # the radius-reduction shim used to approximate.
    q_world_def = q_world - sphere_delta[tid]

    # Shape B: target sphere centre in world.
    X_tb    = body_q[target_body_idx]
    t_world = wp.transform_point(X_tb, target_local_pos)

    diff = t_world - q_world_def
    dist = wp.length(diff)

    # Geometric (line-of-centers) contact normal from the DEFORMED
    # sphere centre toward the target.  Matches jacobi_step's D2 n_eff.
    if dist > eps:
        normal_ab = diff / dist
    else:
        # Degenerate: deformed centre coincides with target centre.
        # Physically unreachable for non-overlapping radii; fall back
        # to outward_normal to avoid NaN.
        out_n     = sphere_outward_normal[tid]
        n_body    = wp.transform_vector(X_ws, out_n)
        normal_ab = wp.transform_vector(X_wb, n_body)
    # With normal_ab = diff / dist, d_proj = dist by construction.  Kept
    # explicit so the smooth-step gate and the diagnostic field below
    # read the same value the rigid-body solver will reconstruct.
    d_proj = wp.dot(diff, normal_ab)

    # Step 7 D4: true 3-D deformed overlap phi_def = (r_lat + R) - dist_def.
    # No shim on r_lat, no separate solver_pen computation -- the
    # deformed-centre point0 makes MuJoCo's projected solver_pen
    # numerically identical to pen_3d, so pen_scale collapses to 1.0
    # by construction.  Kept as a constant so the diagnostic and the
    # rest of the kernel's structure stay readable.
    pen_3d = (r_lat + target_radius) - dist
    pen_scale = float(1.0)

    # Smooth contact-active gate.  d_proj == dist is always positive
    # for non-degenerate geometry; we keep smooth_step(d_proj, eps) as
    # a no-op for safety / parity with jacobi_step's gate structure.
    contact_gate = smooth_step(d_proj, eps) * smooth_step(pen_3d, eps)

    # Body-frame contact geometry.  point0 is the DEFORMED sphere
    # centre, offset0 is r_lat (untouched) along normal_ab.
    X_wb_inv = wp.transform_inverse(X_wb)
    X_tb_inv = wp.transform_inverse(X_tb)

    p0_body      = wp.transform_point(X_wb_inv, q_world_def)
    p1_body      = wp.transform_point(X_tb_inv, t_world)
    offset0_body = wp.transform_vector(X_wb_inv,  r_lat         * normal_ab)
    offset1_body = wp.transform_vector(X_tb_inv, -target_radius * normal_ab)

    # Diagnostic radial for optional prints (cheap — already have the pieces).
    radial_sq = dist * dist - d_proj * d_proj
    if radial_sq < 0.0:
        radial_sq = 0.0
    radial = wp.sqrt(radial_sq)

    # ── Record per-contact diagnostics (physics-neutral) ──
    # Post-D4: pen_scale ≡ 1 by construction, solver_pen ≡ pen_3d, and
    # there is no effective_r shim -- r_lat is used directly.  The
    # diagnostic field names are kept for backwards compatibility with
    # existing readers; the semantics now are:
    #   dbg_pen_scale   = contact_gate    (= pen_scale * gate, since pen_scale = 1)
    #   dbg_solver_pen  = pen_3d          (= phi_def, the true deformed overlap)
    #   dbg_effective_r = r_lat           (radius is no longer reduced)
    dbg_pen_scale[dslot]   = pen_scale * contact_gate
    dbg_solver_pen[dslot]  = pen_3d
    dbg_effective_r[dslot] = r_lat
    dbg_d_proj[dslot]      = d_proj
    dbg_radial[dslot]      = radial

    # Hybrid emission policy: emit the contact to the downstream solver
    # ONLY when the smooth gate is non-negligible (> 1e-4).  This keeps
    # the kernel-to-solver interface C^∞ across the physically meaningful
    # transition region (|d_proj| or |pen_3d| ≲ 30·eps, where smooth_step
    # varies between ~2.5e-4 and ~0.99975) while hard-culling the deep
    # tail where gate ≲ 1e-4 — a regime in which the smooth force is
    # already sub-nanoNewton *and* its gradient is machine-zero, so the
    # discrete cull costs nothing for gradient-based optimisation.
    #
    # Why the cull exists: MuJoCo's soft-constraint solver carries a
    # per-contact compliance term (c · f_n with c = 1/k).  Every live slot
    # contributes some constraint leak per step, so writing ALL 378
    # surface-sphere slots — even with near-zero stiffness — measurably
    # degrades static friction during HOLD (verified against the lift
    # test: 4 mm → 20 mm creep regression without this cull).
    #
    # gate_threshold is set so the cull activates at |d_proj| ≈ 30 · eps
    # ≈ 300 µm for eps = 1e-5 m — several times the transition width of
    # the smooth step, so gradient flow through contact onset is
    # unaffected.  This is a smooth-in-practice, hard-in-the-tail hybrid.
    gate_threshold = float(1.0e-4)
    if contact_gate < gate_threshold:
        out_shape0[buf_idx]    = -1
        out_stiffness[buf_idx] = 0.0
        debug_reason[slot]     = 3
        return

    out_shape0[buf_idx]    = s_idx
    out_shape1[buf_idx]    = target_shape_idx
    out_point0[buf_idx]    = p0_body
    out_point1[buf_idx]    = p1_body
    out_offset0[buf_idx]   = offset0_body
    out_offset1[buf_idx]   = offset1_body
    out_normal[buf_idx]    = normal_ab
    # Step 7 D4: margin0 = r_lat (untouched), no radius-reduction shim.
    out_margin0[buf_idx]   = r_lat
    out_margin1[buf_idx]   = target_radius
    out_tids[buf_idx]      = 0

    # Series composition of the lattice's contact stiffness cslc_kc and
    # the target body's material stiffness target_ke.  Combined with
    # the anchor `ka` enforced inside jacobi_step, this gives a
    # THREE-SPRING series chain (anchor — contact — target).  At
    # equilibrium each sphere obeys
    #     1/keff = 1/ka + 1/cslc_kc + 1/target_ke
    # which is exactly what `calibrate_kc` (newton/_src/geometry/
    # cslc_data.py) inverts to derive cslc_kc from a user-supplied
    # bulk stiffness `ke_bulk`.  Recovers cslc_kc in the rigid-target
    # limit (target_ke ≫ cslc_kc); the eps² floor guards against 0/0
    # when both stiffnesses are zero.
    kc_series = (cslc_kc * target_ke) / (cslc_kc + target_ke + eps * eps)

    # Step 7 D6: stiffness handed to MuJoCo is `kc_series * contact_gate`
    # (no pen_scale factor; pen_scale ≡ 1 under D4 because the
    # deformed-centre point0 makes solver_pen = pen_3d exactly).  MuJoCo
    # then applies a per-contact force `stiffness * solver_pen =
    # kc_series * gate * phi_def`, the theory's series-spring law.  At
    # saturated contact (gate ≈ 1) and rigid target (kc_series → cslc_kc),
    # the force evolves as `kc * (phi_rest - delta_n)` -- the spring law
    # that test_07_kernel_bridge verifies jacobi_step against.
    # The smooth_relu floor keeps MuJoCo's `timeconst = sqrt(imp/ke)`
    # finite when gate drives ke → 0.
    out_stiffness[buf_idx] = smooth_relu(
        kc_series * contact_gate, 1.0e-9)
    # DAMPING BUG (2026-04-19):
    # cslc_dc=2.0 N·s/m is calibrated for Newton's semi-implicit solver.
    # In the MuJoCo conversion kernel, kd>0 triggers timeconst = 2/kd = 1.0s,
    # making both normal AND friction constraints 250× softer than standard
    # contacts (timeconst=0.004s for ke=50000, kd=500). This soft friction
    # timeconst causes excessive Coulomb creep in the HOLD phase.
    # FIX: write 0.0 → uses kd=0 branch → timeconst = sqrt(imp/ke) ≈ 0.030s.
    # Normal force is unchanged by design of the stiffness fix; only friction
    # stiffness improves (33× stiffer timeconst). cslc_dc retained in signature.
    out_damping[buf_idx]   = 0.0
    # FRICTION BUG FIX (2026-04-19):
    # The MuJoCo conversion kernel (kernels.py) treats rigid_contact_friction as a
    # SCALE FACTOR multiplied onto the geom pair's base friction:
    #   effective_mu = geom_friction_max × rigid_contact_friction
    # The geom pair base friction is max(mu_pad, mu_sphere) = mu (from shape materials).
    # ORIGINAL: out_friction = shape_material_mu → effective_mu = mu × mu = mu² (WRONG!)
    # FIX:      out_friction = 1.0              → effective_mu = mu × 1.0 = mu (CORRECT)
    out_friction[buf_idx]  = 1.0

    debug_reason[slot] = 0

