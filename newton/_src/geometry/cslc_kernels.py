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
        # DEFENSIVE: smooth_step(d_proj, eps) is ~1 under current
        # control flow (d_proj = dist > eps via the `if dist > eps`
        # branch above; degenerate path falls back to out_n and never
        # reaches this line).  Kept as belt-and-suspenders against a
        # future refactor that removes the dist > eps guard or routes
        # the degenerate case through here — do NOT remove without
        # adding an explicit `if dist > 0` precondition.
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
        # scale_used = harmonic mean of k_stick and cone_scale.
        # Two-part regulariser, both dimensionally consistent with
        # k_stick + cone_scale [N/m]:
        #   (1) `1e-6 * (k_stick + cone_scale)` — relative floor that
        #       scales with the dominant stiffness.  Biases scale_used
        #       by 1/(1+1e-6) ≈ 1 - 1 ppm, far below any test
        #       resolution.  Stays fixed FRACTION of the operating
        #       stiffness across any k_stick > 0, so a low-friction
        #       sweep where k_stick is reduced doesn't see the
        #       regulariser become non-trivial.  Prior form `+ eps`
        #       (m only) was dimensionally inconsistent AND would
        #       have shifted scale_used by ~50% at k_stick ≈ 1 N/m —
        #       a latent footgun in any low-friction regime.
        #   (2) `+ 1.0e-30` — absolute denormalise-floor for the
        #       k_stick = 0 AND cone_scale = 0 corner (non-friction
        #       scenes where mu_friction = 0; numerator is also 0 so
        #       result is 0/1e-30 = 0).  Without (2), (1) alone
        #       NaN-poisons every non-friction scene (verified by
        #       regression).  1e-30 is well inside IEEE 754 normal
        #       range; conventionally treated as N/m for unit
        #       accounting, magnitude is too tiny for the convention
        #       to matter.
        # Theory dropped both regularisers entirely (cslc_theory
        # friction_force_smooth, bug #5 fix) because L-BFGS-B never
        # sees the transient; kernel keeps a floor for the in-flight
        # Jacobi residual.  Intentional, documented divergence.
        scale_used = (k_stick * cone_scale) / (
            (1.0 + 1.0e-6) * (k_stick + cone_scale) + 1.0e-30)
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
    # DEFENSIVE: d_proj is non-negative by construction of normal_ab
    # (line above) and the dist > eps branch.  smooth_step(d_proj, eps)
    # below is therefore ~1, but is kept for symmetry with the
    # jacobi_step gate and as a future-refactor safety net.
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


# ═══════════════════════════════════════════════════════════════════════════
#  Step 10 / C1b: per-pad-sphere aggregate force vs a PointSetTarget
#
#  This is a STANDALONE smoke-test kernel that does NOT participate in the
#  production sphere-target pipeline.  It exists so the kernel-vs-theory
#  bridge can verify that the GPU implementation of
#
#      F_i = sum_j  kc * smooth_relu(raw_ij, eps) * (q_i - t_j) / ||q_i - t_j||
#      raw_ij = (r_lat_i + R_j) - ||q_i - t_j||,  q_i = p_i_world - delta_i
#
#  matches `cslc_main.theory.kernel_bridge.compute_contact_force_point_set`
#  element-wise at production deltas.  Once verified, C1c extends test_07
#  with box-target scenes; C2 then wires per-pair contact emission to
#  MuJoCo (no aggregation -- this kernel is for verification of the
#  per-pad SUM, not for emission).
#
#  Kept independent so production sphere-target code paths cannot
#  accidentally regress through edits here.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_pad_force_vs_point_set(
    # ── Pad lattice state (same arrays as compute_cslc_penetration) ──
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
    # ── Point-set target ──
    # Positions / radii / normals are TARGET BODY LOCAL; transformed
    # to world per-thread using the target body's body_q transform.
    target_positions_local: wp.array(dtype=wp.vec3),
    target_radii: wp.array(dtype=wp.float32),
    target_normals_local: wp.array(dtype=wp.vec3),   # currently unused;
                                                     # stored for C2+ emission
    target_count: int,
    target_body_idx: int,
    # ── Constants ──
    kc: float,
    eps: float,
    # ── Output ──
    pad_force_world: wp.array(dtype=wp.vec3),
):
    """Per-pad-sphere aggregate contact force from a point-set target.

    Each kernel thread = one pad lattice sphere.  Inactive lattice
    spheres (sphere_shape != active_cslc_shape_idx) and non-surface
    spheres write a zero force.  No active-gate culling on the
    pad-sphere level -- the per-pair raw threshold (raw > -50*eps)
    decides which target points contribute.

    This kernel does NOT write Newton contacts; it only computes the
    aggregate force vector for verification against the theory's
    ``compute_contact_force_point_set``.  The contact-emission kernel
    (one MuJoCo contact per overlapping (pad_sphere, target_point)
    pair) lives in C2.
    """
    tid = wp.tid()

    # Lattice filter + surface filter -- both produce a zero force.
    if sphere_shape[tid] != active_cslc_shape_idx:
        pad_force_world[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    if is_surface[tid] == 0:
        pad_force_world[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    # Pad sphere world-frame deformed centre.  Same transform chain
    # the production kernels use: q_world = X_wb * X_ws * p_local - delta.
    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]

    p_i_local = sphere_pos_local[tid]
    p_i_world = wp.transform_point(X_wb, wp.transform_point(X_ws, p_i_local))
    q_i_world = p_i_world - sphere_delta[tid]
    r_i = sphere_radii[tid]

    # Target body transform (constant across all target points; lift
    # out of the inner loop).
    X_tb = body_q[target_body_idx]

    # Accumulate per-target-point contributions.  No early-exit on the
    # pad sphere -- a fully-disengaged pad sphere will accumulate the
    # zero vector through every smooth-relu floor.  (Cost is bounded
    # by target_count, which is fixed at scene init.)
    F = wp.vec3(0.0, 0.0, 0.0)
    for j in range(target_count):
        t_j_world = wp.transform_point(X_tb, target_positions_local[j])
        diff = q_i_world - t_j_world
        L = wp.length(diff)
        if L < 1.0e-15:
            continue
        R_j = target_radii[j]
        raw = (r_i + R_j) - L
        # Inactive-pair skip threshold.  MUST match the canonical
        # Python constant ``INACTIVE_RAW_EPS_FACTOR = -50.0`` defined
        # in ``cslc_main/theory/cslc_theory.py``.  The literal is
        # baked in here because Warp kernel constants are compiled
        # in -- can't import a Python module-level value at kernel
        # build time.  If you change the constant on the Python side,
        # update this literal too and rerun
        # test_07_kernel_bridge to confirm scenes H/I still pass.
        # At raw < -50*eps: smooth_relu(raw, eps) and smooth_step(raw, eps)
        # are both < 1e-9, so the excluded contribution is below
        # numerical noise.
        if raw < -50.0 * eps:
            continue
        phi_eff = smooth_relu(raw, eps)
        # Force on the PAD sphere is in the direction (q_i - t_j)/L,
        # i.e. AWAY from the target point.  Matches the theory's
        # ``point_set_contact_force`` and ``cslc_theory.contact_force``
        # sign convention.
        F = F + kc * phi_eff * (diff / L)

    pad_force_world[tid] = F


# ═══════════════════════════════════════════════════════════════════════════
#  C2d: point-set warm-start penetration (argmax-overlap)
#
#  Structural twin of ``compute_cslc_penetration`` for PointSetTarget
#  targets.  For each pad sphere, walks the target point set and picks
#  the SINGLE most-overlapping target point (argmax over j of raw_ij at
#  the rest position), then writes the same two outputs as the sphere
#  kernel: scalar ``phi_rest`` and rest-frame line-of-centres normal.
#
#  This is the option-(a') warm-start: ``lattice_solve_equilibrium`` is
#  unchanged and consumes the same scalar+vec3 fields per pad sphere as
#  in the sphere-target path.  No new linear-solve kernel needed.  The
#  remaining multi-point correction is exactly what
#  ``jacobi_step_point_set`` is built to handle.
#
#  Non-smoothness caveat (deferred to C3+)
#  ---------------------------------------
#  argmax(j) is non-smooth in pose at the boundary where two target
#  points have near-equal overlap with the same pad sphere.  This is
#  fine for warm-start convergence -- the nonlinear Jacobi sweeps
#  absorb the resulting transient -- but it breaks differentiability
#  of the warm-start with respect to body pose.  Downstream gradient-
#  based optimisation (MPC, RL, sim-to-real policy gradients) that
#  wants to backprop through the lattice solve will eventually need a
#  softmax-blended weighted-average warm-start; see C3+ deferred items
#  in cslc_main/theory/notes.md.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_cslc_penetration_point_set(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    # Kept in the signature for symmetry with ``compute_cslc_penetration``.
    # Not read here (the warm-start works at the REST position;
    # jacobi_step_point_set recomputes overlap at the deformed centre
    # each iteration).
    sphere_delta: wp.array(dtype=wp.vec3),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    # NOTE: ``compute_cslc_penetration`` (sphere variant) takes
    # ``sphere_outward_normal`` for the degenerate-centres-coincide
    # fallback.  This point-set variant ``continue``s past the
    # degenerate target instead and relies on a NEAR-coincident
    # target to supply a non-degenerate normal, so the outward normal
    # is genuinely unused -- omitted from this kernel's signature.
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    # Point-set target (replaces target_local_pos + target_radius).
    target_positions_local: wp.array(dtype=wp.vec3),
    target_radii: wp.array(dtype=wp.float32),
    target_count: int,
    eps: float,
    raw_penetration: wp.array(dtype=wp.float32),
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """REST 3-D argmax-overlap penetration per lattice sphere (warm-start).

    For each active surface pad sphere i, picks the target point j* with
    the largest rest overlap ``raw_ij = (r_i + R_j) - ||t_j - p_i||``
    (subject to the same active-set skip ``raw_ij >= -50*eps`` used by
    every other point-set kernel; MUST match ``INACTIVE_RAW_EPS_FACTOR``
    in cslc_main/theory/cslc_theory.py).

    Outputs match ``compute_cslc_penetration`` so that
    ``lattice_solve_equilibrium`` can consume the result with no
    signature change:

      * ``raw_penetration[i] = smooth_relu(raw_ij*, eps) * smooth_step(dist*, eps)``
      * ``contact_normal_out[i] = (t_j* - p_i_world) / ||t_j* - p_i_world||``

    Pad spheres with no target points passing the active-set skip get
    ``phi = 0`` and ``n = 0``; the warm-start linear solve produces a
    near-zero local displacement for those spheres (lateral coupling
    still propagates other spheres' warm-start delta through the
    Laplacian).
    """
    tid = wp.tid()

    # Active-lattice filter (matches compute_cslc_penetration).
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
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]

    p_local = sphere_pos_local[tid]
    r_lat   = sphere_radii[tid]

    q_body  = wp.transform_point(X_ws, p_local)
    q_world = wp.transform_point(X_wb, q_body)

    X_tb = body_q[target_body_idx]

    # Argmax search over target points.  ``found`` distinguishes "no
    # target passed the active-set skip" (phi := 0) from "found at
    # least one active pair".  Tracking dist_best separately avoids a
    # back-derivation from raw_best (which would need R_j*).
    found     = int(0)
    raw_best  = float(0.0)
    dist_best = float(0.0)
    n_best    = wp.vec3(0.0, 0.0, 0.0)

    for j in range(target_count):
        t_j_world = wp.transform_point(X_tb, target_positions_local[j])
        diff = t_j_world - q_world
        dist = wp.length(diff)
        # Degenerate centres-coincide: same convention as
        # compute_pad_force_vs_point_set / jacobi_step_point_set --
        # 1e-15 is numerical zero, NOT eps (which is the smooth-gate
        # width, 5e-4 m in production; using eps here would silently
        # drop deeply-overlapping contacts).
        if dist < 1.0e-15:
            continue
        R_j = target_radii[j]
        raw_j = (r_lat + R_j) - dist
        # Active-set skip.  MUST match INACTIVE_RAW_EPS_FACTOR = -50.0 in
        # cslc_main/theory/cslc_theory.py.  Below this threshold both
        # smooth_relu and smooth_step are <1e-9, so the per-pair
        # contribution is below numerical noise.  Skip is also a real
        # performance win: at target_count = 3750 (full 25mm box, 1mm
        # pitch) most points are >25mm from any given pad sphere and
        # get culled in the first 'continue'.
        if raw_j < -50.0 * eps:
            continue
        if (found == 0) or (raw_j > raw_best):
            found     = 1
            raw_best  = raw_j
            dist_best = dist
            n_best    = diff / dist

    if found == 1:
        # Same smooth_relu * smooth_step composition as
        # compute_cslc_penetration.  smooth_step(dist_best, eps) ~= 1 for
        # any reasonable dist > eps; it's kept for symmetry and to
        # preserve C^inf behaviour at the degenerate dist -> 0 limit.
        # DEFENSIVE: the `dist < 1.0e-15` continue above ensures
        # dist_best > 0 here, so smooth_step ~= 1.  Do NOT remove the
        # degenerate-check `continue` without re-deriving this gate.
        phi = smooth_relu(raw_best, eps) * smooth_step(dist_best, eps)
        raw_penetration[tid] = phi
        contact_normal_out[tid] = n_best
    else:
        raw_penetration[tid] = 0.0
        contact_normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Step 10 / C2b: point-set Jacobi iteration
#
#  Structural twin of ``jacobi_step`` for PointSetTarget targets.  Each pad
#  sphere accumulates contact contributions from every overlapping target
#  point in an inner loop; anchor, lateral, friction, and the damped Jacobi
#  update are otherwise identical in shape to the sphere-target kernel.
#
#  Differences from ``jacobi_step``, contained to the contact block:
#    * No single (target_local_pos, target_radius) -- instead arrays
#      ``target_positions_local[0:M]``, ``target_radii[0:M]``.
#    * Contact force = SUM over target points of per-pair series-spring
#      contribution.  Same active-set threshold (-50 * eps) as the kernel-1
#      smoke-test ``compute_pad_force_vs_point_set`` (MUST match
#      INACTIVE_RAW_EPS_FACTOR in cslc_main/theory/cslc_theory.py).
#    * Implicit-diagonal stabilisation S_n picks up the SUM of contact
#      gates over j instead of a single gate -- contractive Jacobi update
#      still holds because Σ kc·gate_j upper-bounds the linearised contact
#      operator on the normal axis.
#
#  Friction uses the pad sphere's own outward normal as the local frame
#  (same as the existing anisotropic-anchor decomposition in
#  jacobi_step).  ``f_n_mag = |F_contact · n_outward|`` -- the aggregate
#  normal-axis component of the multi-point contact wrench.  This is the
#  least-surprising extension from the single-target friction physics; a
#  per-pair friction model is C3+ work.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def jacobi_step_point_set(
    delta_src: wp.array(dtype=wp.vec3),
    delta_dst: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
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
    k_stick: float,
    mu_friction: float,
    # ── Point-set target (replaces target_local_pos + target_radius) ──
    target_positions_local: wp.array(dtype=wp.vec3),
    target_radii: wp.array(dtype=wp.float32),
    target_count: int,
    target_body_idx: int,
    # External tangential load (same as jacobi_step; -1 = no-op).
    f_ext_apex_idx: int,
    f_ext_apex: wp.vec3,
    eps: float,
):
    """One damped Jacobi sweep for the ACTIVE lattice against a PointSetTarget.

    Equilibrium (per active surface sphere i, in pad sphere i's local
    rest-normal frame):

      0 =  -K_anchor · δ_i
           + Σ_{j ∈ N(i)} k_l (‖q_j − q_i‖ − L_ij) ê_ij(δ)     (lateral)
           + Σ_{m=0..M-1} k_c · phi_eff_im · gate_im · n_eff_im (contact, point-set)
           + f_friction(δ_t, |F_contact · n_outward|, k_stick, μ) (stick-slip)
           + f_ext_apex                                          (only at apex_idx)

    where for each target point m:
       raw_im   = (r_lat_i + R_m) - ||t_m_world - q_i_world||
       phi_eff_im = smooth_relu(raw_im, eps)
       gate_im    = smooth_step(raw_im, eps)
       n_eff_im   = (t_m_world - q_i_world) / ||·||
    -- the same series-spring law as ``jacobi_step``, summed.

    Active-set skip: pairs with ``raw_im < -50 * eps`` are excluded
    (MUST match INACTIVE_RAW_EPS_FACTOR in cslc_theory.py).
    """
    tid = wp.tid()

    # Lattice filter (same as jacobi_step).
    if sphere_shape[tid] != active_cslc_shape_idx:
        delta_dst[tid] = delta_src[tid]
        return

    delta_old = delta_src[tid]
    n_neighbors = neighbor_count[tid]

    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]

    p_i_local = sphere_pos_local[tid]
    p_i_world = wp.transform_point(X_wb, wp.transform_point(X_ws, p_i_local))
    q_i_world = p_i_world - delta_old

    out_n_local = sphere_outward_normal[tid]
    out_n_world = wp.transform_vector(X_wb, wp.transform_vector(X_ws, out_n_local))

    # Lateral (distance-preserving), identical to jacobi_step.
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
        inv_dist = dist / (dist * dist + eps * eps)
        L_ij = neighbor_rest_length[edge]
        # Step 7 sign fix: load form (= -gradient) carries minus sign
        # relative to physical force.  Matches jacobi_step exactly.
        f_lateral = f_lateral - kl * (dist - L_ij) * d * inv_dist

    # Point-set CONTACT: sum over target points.
    f_contact_vec = wp.vec3(0.0, 0.0, 0.0)
    sum_gate = float(0.0)
    f_friction_vec = wp.vec3(0.0, 0.0, 0.0)

    if is_surface[tid] == 1:
        r_i = sphere_radii[tid]
        X_tb = body_q[target_body_idx]
        for j in range(target_count):
            t_j_world = wp.transform_point(X_tb, target_positions_local[j])
            diff = t_j_world - q_i_world
            L = wp.length(diff)
            # Degenerate centres-coincide check.  MUST be ~1e-15
            # (numerical zero), NOT ``eps`` -- ``eps`` is the smooth-
            # gate width (production: 5e-4 m), and skipping pairs at
            # that radius would silently drop deeply-overlapping
            # contacts.  See compute_pad_force_vs_point_set above for
            # the same convention.
            if L < 1.0e-15:
                continue
            R_j = target_radii[j]
            raw = (r_i + R_j) - L
            # MUST match INACTIVE_RAW_EPS_FACTOR = -50.0 in
            # cslc_main/theory/cslc_theory.py.  Excluded contribution
            # is below 1e-9 in both phi_eff and step there, so this is
            # numerically-safe pruning that keeps the inner loop tight
            # at production densities (e.g. 600-point box at production
            # eps = 5e-4 m means a pair is skipped if it's more than
            # ~25 mm away from the pad sphere, comfortably outside the
            # engaged patch).
            if raw < -50.0 * eps:
                continue
            phi_eff = smooth_relu(raw, eps)
            gate = smooth_step(raw, eps)
            n_eff = diff / L
            # Load form: kernel uses (t-q)/||·|| direction; load
            # gradient sign convention matches theory's (q-t)/||·||
            # force direction (both negations cancel; see theory.txt
            # tab:signs).
            f_contact_vec = f_contact_vec + kc * phi_eff * gate * n_eff
            sum_gate = sum_gate + gate

        # Stick-slip friction.  Aggregate normal-axis magnitude used
        # as the cone reference; tangent decomposition done in pad
        # outward frame (same as the existing anisotropic-anchor
        # decomposition below).
        f_n_signed = wp.dot(f_contact_vec, out_n_world)
        # Compression should give f_contact_vec · n_outward < 0 (force
        # pushes pad sphere along -n_outward, i.e. into the body).
        # Take absolute value for the cone magnitude either way; the
        # cone is symmetric.
        f_n_mag = wp.abs(f_n_signed)

        delta_proj_n_outward = wp.dot(delta_old, out_n_world)
        delta_t = delta_old - delta_proj_n_outward * out_n_world
        delta_t_mag = wp.length(delta_t)
        inv_dt_mag = delta_t_mag / (delta_t_mag * delta_t_mag + eps * eps)
        cone_scale = mu_friction * f_n_mag * inv_dt_mag
        # Harmonic-mean smooth-min surrogate.  v0.8 dimensionally-clean
        # regulariser: relative floor `1e-6 * (k_stick + cone_scale)`
        # (dimensionless × N/m = N/m, ~1 ppm bias on scale_used) +
        # absolute denormalise-floor `1.0e-30` for the k_stick = 0
        # AND cone_scale = 0 corner (non-friction scenes that would
        # otherwise NaN).  See jacobi_step's matching block for full
        # rationale; the two sites MUST stay in sync.
        scale_used = (k_stick * cone_scale) / (
            (1.0 + 1.0e-6) * (k_stick + cone_scale) + 1.0e-30)
        f_friction_vec = -scale_used * delta_t

    # External tangential load (bridge / experiment driver only; -1 in
    # production).
    f_ext_vec = wp.vec3(0.0, 0.0, 0.0)
    if tid == f_ext_apex_idx:
        f_ext_vec = -f_ext_apex

    # Anisotropic block-Jacobi in pad sphere's local rest-normal frame.
    # S_n picks up the SUM of contact gates (vs the single gate in
    # jacobi_step).  This sum upper-bounds |d(f_contact·n)/d(δ_n)|
    # over the active set, so the iteration's contraction property
    # carries over from the single-target case.
    rhs_explicit = f_contact_vec + f_lateral + f_friction_vec + f_ext_vec
    rhs_n_scalar = wp.dot(rhs_explicit, out_n_world)
    rhs_t_vec    = rhs_explicit - rhs_n_scalar * out_n_world

    delta_old_n = wp.dot(delta_old, out_n_world)
    delta_old_t = delta_old - delta_old_n * out_n_world

    ka_t = ka * ka_tangent_ratio
    S_n  = kl * float(n_neighbors) + kc * sum_gate
    S_t  = kl * float(n_neighbors)
    k_diag_n = ka + S_n
    k_diag_t = ka_t + S_t

    rhs_n_total = rhs_n_scalar + S_n * delta_old_n
    rhs_t_total = rhs_t_vec    + S_t * delta_old_t

    delta_jacobi_n = rhs_n_total / k_diag_n
    delta_jacobi_t = rhs_t_total / k_diag_t
    delta_jacobi   = delta_jacobi_n * out_n_world + delta_jacobi_t

    delta_dst[tid] = (1.0 - alpha) * delta_old + alpha * delta_jacobi


# ═══════════════════════════════════════════════════════════════════════════
#  Step 10 / C2c: per-pair contact emission for PointSetTarget
#
#  Per (pad_sphere, target_point) pair that's geometrically overlapping at
#  the converged delta, emit one MuJoCo contact -- same emission convention
#  as ``write_cslc_contacts`` (deformed-centre point0, r_lat margin, target
#  radius R_j, line-of-centres normal), just K of them per pad sphere
#  instead of one.  No aggregation, no resultant-projection hack -- each
#  contact is a genuine sphere-vs-sphere contact pair.
#
#  Buffer layout
#  -------------
#  Pad sphere ``i`` (with ``surface_slot_map[i] = s_i >= 0``) writes its
#  contacts to absolute slots
#
#       contact_offset + s_i · K_max + 0 .. K_max-1.
#
#  Excess slots (beyond the actual K_i ≤ K_max overlapping pairs for this
#  pad sphere) are filled with the ``out_shape0 = -1`` sentinel, so the
#  downstream MuJoCo conversion kernel
#  (``convert_newton_contacts_to_mjwarp_kernel``) culls them via its
#  ``shape_a < 0`` early-out.
#
#  K_max sizing
#  ------------
#  Total contacts per pad sphere is bounded by the number of target points
#  inside radius ``r_lat + R_target`` of the pad sphere's deformed centre.
#  For a 25 mm box face sampled at ~1 mm pitch with r_lat=1.5 mm and
#  R_target=1 mm, that bound is ~π(2.5mm)²/(1mm)² ≈ 20 points.  K_max = 32
#  is the production default in the handler -- comfortable margin,
#  doesn't blow the buffer at production lattice sizes.  At N_pad = 150,
#  K_max = 32, n_pair_blocks = 2: total = 9600 slots, well under MuJoCo's
#  default naconmax (100k+).
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def write_cslc_contacts_point_set(
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
    target_positions_local: wp.array(dtype=wp.vec3),
    target_radii: wp.array(dtype=wp.float32),
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
    shape_material_mu: wp.array(dtype=wp.float32),
    cslc_kc: float,
    target_ke: float,
    cslc_dc: float,
    eps: float,
    out_stiffness: wp.array(dtype=wp.float32),
    out_damping: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
    # C2d: per-pair truncation counter.  Atomic-incremented once per pad
    # sphere whose active-pair count would have exceeded K_max.  Read on
    # CPU by the handler after collide() to RuntimeWarning if any pair
    # block under-sized K_max; see cslc_handler._launch_vs_point_set.
    truncation_count: wp.array(dtype=wp.int32),
):
    """Emit one MuJoCo contact per overlapping (pad_sphere, target_point) pair.

    Mirror of ``write_cslc_contacts`` for PointSetTarget targets.  Active
    pairs are those passing both
        (a) ``raw_ij >= -50 * eps``  -- MUST match
            ``INACTIVE_RAW_EPS_FACTOR`` in cslc_main/theory/cslc_theory.py;
        (b) ``smooth_step(d_proj)·smooth_step(raw_ij) >= 1e-4`` -- the same
            ``gate_threshold`` cull the sphere-target emission applies
            (deep tail is sub-nN force, machine-zero gradient).

    Note: with the algebraic smooth_step (``0.5*(1 + x/sqrt(x^2+eps^2))``)
    these two thresholds COINCIDE at raw = -50*eps (see INCLUSION_FACTOR
    in cslc_main/theory/cslc_theory.py); any pair passing (a) also
    passes (b).  ``K_max`` must therefore be sized against the active-
    set inclusion radius r_inclusion = r_lat + R + INCLUSION_FACTOR*eps,
    NOT against geometric overlap (raw > 0) alone.

    Each emitted contact carries margin0 = r_lat[i], margin1 = R_j; MuJoCo
    reconstructs solver_pen = (r_lat + R_j) - ||t_j - q_def||  = phi_def
    per pair, so per-contact force = stiffness · solver_pen
    = kc_series · gate · phi_def -- the same series-spring law that
    ``jacobi_step_point_set`` converges on.
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
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]
    X_wb_inv = wp.transform_inverse(X_wb)

    p_i_local = sphere_pos_local[tid]
    r_i = sphere_radii[tid]
    q_world = wp.transform_point(X_wb, wp.transform_point(X_ws, p_i_local))
    q_world_def = q_world - sphere_delta[tid]

    X_tb = body_q[target_body_idx]
    X_tb_inv = wp.transform_inverse(X_tb)

    # Series stiffness composition: same as write_cslc_contacts.
    kc_series = (cslc_kc * target_ke) / (cslc_kc + target_ke + eps * eps)

    pair_count = int(0)
    truncated  = int(0)
    for j in range(target_count):
        t_world = wp.transform_point(X_tb, target_positions_local[j])
        diff = t_world - q_world_def
        dist = wp.length(diff)
        if dist < 1.0e-15:
            continue
        R_j = target_radii[j]
        pen_3d = (r_i + R_j) - dist
        # Active-set skip.  MUST match INACTIVE_RAW_EPS_FACTOR = -50.0.
        if pen_3d < -50.0 * eps:
            continue

        normal_ab = diff / dist
        d_proj = dist
        contact_gate = smooth_step(d_proj, eps) * smooth_step(pen_3d, eps)

        # Hard cull below the production gate threshold -- matches
        # write_cslc_contacts's hybrid emission policy (line ~903).
        if contact_gate < 1.0e-4:
            continue

        # Pair j is active AND emittable.  Now check buffer space:
        # placing the overflow check HERE (rather than at the top of
        # the loop) avoids false-positive truncation warnings when the
        # remaining target indices j..target_count-1 are all inactive
        # (would be skipped by the pen_3d / contact_gate culls above).
        # At geometry-derived K_max sizing this matters: K_max is
        # tuned to the active-pair count, so an off-by-one in
        # over-warning erodes signal quality.  We may under-report
        # by leaving emittable pairs at indices > j uncounted, but
        # the metric is "did any pad sphere overflow", and ANY
        # overflow is sufficient to fire the warning -- the per-
        # dropped-pair count was never the actionable number.
        if pair_count >= K_max:
            truncated = 1
            break

        # Body-frame contact geometry (same convention as
        # write_cslc_contacts).
        p0_body      = wp.transform_point(X_wb_inv, q_world_def)
        p1_body      = wp.transform_point(X_tb_inv, t_world)
        offset0_body = wp.transform_vector(X_wb_inv,  r_i  * normal_ab)
        offset1_body = wp.transform_vector(X_tb_inv, -R_j  * normal_ab)

        buf_idx = contact_offset + base_slot * K_max + pair_count
        out_shape0[buf_idx]   = s_idx
        out_shape1[buf_idx]   = target_shape_idx
        out_point0[buf_idx]   = p0_body
        out_point1[buf_idx]   = p1_body
        out_offset0[buf_idx]  = offset0_body
        out_offset1[buf_idx]  = offset1_body
        out_normal[buf_idx]   = normal_ab
        out_margin0[buf_idx]  = r_i
        out_margin1[buf_idx]  = R_j
        out_tids[buf_idx]     = 0
        out_stiffness[buf_idx] = smooth_relu(
            kc_series * contact_gate, 1.0e-9)
        # Match sphere-target conventions (see comments at the existing
        # write_cslc_contacts for the friction/damping rationale).
        out_damping[buf_idx]   = 0.0
        out_friction[buf_idx]  = 1.0

        pair_count = pair_count + 1

    if truncated == 1:
        # One atomic per pad sphere that overflowed -- not per dropped
        # pair -- so the CPU read after collide() reports the number of
        # affected pad spheres, which is the more actionable count.
        wp.atomic_add(truncation_count, 0, 1)

