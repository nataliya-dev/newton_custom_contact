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


# ─────────────────────────────────────────────────────────────────────────────
#  smooth_blend: a one-sided polynomial blend of max(x, 0)
#
#  Theory issue (cslc theory §smoothing): the analytic surrogate
#  smooth_relu(x, ε) = 0.5·(x + √(x² + ε²)) does not pass through (0, 0):
#  smooth_relu(0, ε) = ε/2.  Plugged into φ_eff = σ(raw)·√(σ(raw) + ε)
#  this yields φ_eff(raw = 0) ≈ 0.61·ε^1.5, a non-zero force baseline
#  whose magnitude scales as ε^1.5 and cannot be tuned away without
#  losing C¹ smoothness (no analytic C^∞ approximation of max(x, 0)
#  passes through 0 while staying non-negative — by IVT, f'(0) ∈ (0, 1)
#  forces f < 0 for some x < 0).
#
#  smooth_blend trades C^∞ regularity for an EXACT zero floor and an
#  EXACT raw^1.5 regime at depth, accepting C² regularity instead.  The
#  Hermite quintic H₅(t) = t³·(10 − 15t + 6t²) is C² at both endpoints
#  (H₅(0) = H₅'(0) = H₅''(0) = 0, H₅(1) = 1, H₅'(1) = H₅''(1) = 0), so
#  the blend
#
#      smooth_blend(x, ε) = 0           if x ≤ 0
#                          x · H₅(x/ε)  if 0 < x < ε
#                          x            if x ≥ ε
#
#  is C² everywhere.  Composed with the existing Hertz lift
#  φ_eff = blend · √(blend + ε), the floor vanishes (blend = 0 ⇒ φ_eff =
#  0) and at depth the +ε inside the sqrt is dwarfed by blend → raw, so
#  φ_eff → raw^1.5 cleanly (TRUE Hertz, not the raw·√ε linear regime
#  that smooth_relu's ε/2 floor produced near raw ≈ ε).
#
#  C² is sufficient for reverse-mode autodiff through the lattice solve
#  (∇F = ∂F/∂δ is C¹, which is what wp.Tape backward needs).  Tradeoff:
#  loses C^∞ relative to smooth_relu, but in exchange:
#    1. φ_eff(raw=0) = 0 exactly — no force baseline (theory issue #1)
#    2. φ_eff = raw^1.5 above raw=ε exactly — true Hertz (issue #2)
#    3. Active set is sharper (exactly 0 below 0 instead of an ε-scaled
#       smooth tail), which speeds Jacobi convergence as a side effect.
# ─────────────────────────────────────────────────────────────────────────────


@wp.func
def smooth_blend(x: float, eps: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= eps:
        return x
    t = x / eps
    t2 = t * t
    t3 = t2 * t
    # H₅(t) = t³·(10 − 15t + 6t²); C² at t=0 and t=1.
    h = t3 * (10.0 - 15.0 * t + 6.0 * t2)
    return x * h


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
          smooth_step(r_i − d_t, eps)`` (Option-2 tiling, kernel
          half-width = r_pad — DIVERGES from contract §3.5 which
          specifies 3·r_pad) and ``d_t`` is the tangential
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
      * ``raw_penetration[i] = σ_ε(raw_ij*) · √(σ_ε(raw_ij*) + ε)
        · smooth_step(L*, eps)``  (Hertz-like δ^1.5 lift; matches
        :func:`jacobi_step`'s force law — DIVERGES from contract §4
        eq:phi-eff which specifies linear ``σ_ε(raw)``)
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
    # factor.  Theory fix D1: argmax is over raw_face = raw - r_lat
    # (signed distance from sphere centre to target plane along
    # outward normal), so the warm-start phi reads off the face-
    # penetration scalar — zero at face contact, positive at depth.
    found = int(0)
    raw_face_best = float(0.0)
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
        # Theory fix D1 — face-penetration form.  ``raw_j`` carries an
        # r_lat shelf (raw_j = r_lat at first face contact for a flat
        # lattice surface) which contributes φ ≈ r_lat^1.5 per sphere at
        # δ=0, summing to ~84 N on box/box pre-fix-D1 (the "shelf"
        # diagnosed in the t2_box_box_shelf probe).  ``raw_face`` is
        # the signed distance from the lattice sphere centre to the
        # target plane along the OUTWARD target normal, zero at face
        # contact, positive at face penetration.  All downstream uses
        # of raw for the force law and active-set gates now consume
        # raw_face so phi(face contact) = 0 exactly.
        raw_face_j = raw_j - r_lat
        # Active-set gate now on face penetration (contract §3.6 eq:
        # inactive, post-fix-D1).  MUST match INACTIVE_RAW_EPS_FACTOR
        # = -50.0 in cslc_main/theory/cslc_theory.py.
        if raw_face_j < -50.0 * eps:
            continue
        # Theory fix D2 — distance-magnitude cull.  The half-space
        # form (raw_face, gates) is LOCAL: it assumes the target's
        # tangent plane approximates the target surface near the
        # sample.  For a curved target (sphere, dome, etc.) this
        # breaks down for pad spheres far from the target sample on
        # the OPPOSITE side of the target body.  Such pairs can pass
        # alignment (both outward normals end up antiparallel in
        # world frame), tangential locality (d_t = 0 when on the same
        # axis through the body), and the raw_face cull (the half-
        # space form sees them as 50-100 mm "penetrating"), producing
        # phantom contacts with huge phi^1.5 force.  Discovered on the
        # dome scene where a dome-rim sphere paired with a ball back-
        # side target across the scene.  Bound the pair distance to
        # ``3·r_lat + 5·ε`` (well above legitimate ||q-t|| ≤ √5·r_lat
        # at the deepest typical penetration; well below scene-scale
        # phantom distances of 50-100×r_lat).  MUST match jacobi_step,
        # compute_target_W, write_cslc_contacts so the active set is
        # parity-locked across the pipeline.
        if dist > 3.0 * r_lat + 5.0 * eps:
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
        # Kernel half-width = r_pad (Option-2 tiling, no overlap).
        # Must match jacobi_step and write_cslc_contacts so the
        # warm-start argmax picks the same sample the iteration kernel
        # will saturate on.
        kernel_h = r_lat
        w_tangent = smooth_step(kernel_h - d_t_mag, eps)
        if w_tangent < 1.0e-2:
            continue
        # argmax over face-penetration form (theory fix D1).  Since
        # raw_face_j = raw_j - r_lat is monotone in raw_j (r_lat is the
        # SAME pad sphere's radius across the search), argmax(raw_face)
        # = argmax(raw); we just track the face-form value so the warm-
        # start phi reads off the right shelf-free quantity below.
        if (found == 0) or (raw_face_j > raw_face_best):
            found = 1
            raw_face_best = raw_face_j
            dist_best = dist
            # Warm-start contact direction is the load-form -n_face
            # (matches ``jacobi_step``'s n_eff).  The downstream
            # lattice_solve_equilibrium reads this as the warm-start
            # contact direction and projects it onto the pad sphere's
            # local frame.
            n_best = -n_face_world

    if found == 1:
        # Hertz-like phi = raw_face^1.5 (matches jacobi_step's force
        # law post-fix-D1).  The lattice solver converges on this same
        # scaling; the warm start has to agree or it bootstraps from
        # the wrong load shape.  DEFENSIVE: dist < 1e-15 was caught
        # above, so smooth_step ~= 1.
        #
        # Theory fix #1+#2 (Hertz floor + low-raw shape): smooth_blend
        # replaces smooth_relu so phi(raw_face≤0) = 0 EXACTLY (no
        # ε^1.5 baseline) and phi(raw_face≥ε) = raw_face^1.5 EXACTLY
        # (true Hertz at depth).  The √(blend + ε) inside is retained
        # as a gradient-safety regularizer; since blend = 0 for raw_face
        # ≤ 0, the floor it would otherwise create is multiplicatively
        # killed.  Theory fix D1 (face-penetration form): input is
        # raw_face, not raw, so phi vanishes at FACE contact, not at
        # lattice-sphere-centre contact (which had an r_lat shelf).
        raw_pos = smooth_blend(raw_face_best, eps)
        phi = raw_pos * wp.sqrt(raw_pos + eps) * smooth_step(dist_best, eps)
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
#  Kernel 2a'': Per-sphere world-frame rest-position precompute
#
#  Body poses ``X_wb`` and ``X_ws`` are constant during the quasi-static
#  Jacobi sweep (n_iter = 40 sweeps at fixed body_q).  Hoisting the rest-
#  position transform
#
#      p_world_i = X_wb · X_ws · p_local_i
#
#  out of ``jacobi_step``'s per-iter inner loop saves
#  ``n_spheres × (n_iter − 1)`` transform_point pairs per pair launch.
#  Mirror of ``compute_outward_normals_world`` (which already precomputes
#  the per-sphere world-frame normal); both are launched once per pair
#  before the Jacobi loop.  See cslc_handler._launch.
#
#  Fidelity: this is an algebraic refactor — the precomputed world-frame
#  position is the same value the inner loop computed every iteration, so
#  the converged ``δ`` is bit-identical (modulo fp non-associativity).
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_pad_pos_world(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_shape: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    pad_pos_world: wp.array(dtype=wp.vec3),
):
    """Transform each lattice sphere's rest centre into world frame."""
    tid = wp.tid()
    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws = shape_transform[s_idx]
    X_wb = body_q[b_idx]
    p_local = sphere_pos_local[tid]
    pad_pos_world[tid] = wp.transform_point(
        X_wb, wp.transform_point(X_ws, p_local))


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2a''': Per-target world-frame pose precompute
#
#  The target body's pose ``X_tb = body_q[target_body_idx]`` is constant
#  during the Jacobi sweep, so the per-sample world-frame position and
#  face-normal can be hoisted out of jacobi_step's inner loop:
#
#      t_world_j  = X_tb · t_local_j         (transform_point)
#      n_face_world_j = X_tb · n_local_j     (transform_vector)
#
#  This saves ``target_count × n_iter`` transform ops per pair launch
#  (the per-iter inner-loop hot path).  For a 50-target pair at n_iter =
#  40 that's 4000 transform_point/_vector pairs per sphere, all wasted
#  recomputation of the same values.
#
#  Fidelity: algebraic refactor — same values, fewer ops.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_target_world_state(
    target_positions_local: wp.array(dtype=wp.vec3),
    target_normals_local: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    target_body_idx: int,
    target_pos_world: wp.array(dtype=wp.vec3),
    target_normal_world: wp.array(dtype=wp.vec3),
):
    """Transform target samples into world frame."""
    tid = wp.tid()
    X_tb = body_q[target_body_idx]
    target_pos_world[tid] = wp.transform_point(X_tb, target_positions_local[tid])
    target_normal_world[tid] = wp.transform_vector(X_tb, target_normals_local[tid])


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel 2a'''': Target-side partition-of-unity normalizer
#
#  Theory issue #3 + #4 — the pre-Fix-B force law summed per pair
#
#      F_i = Σ_j  kc · A_j · w_t_ij · α_ij · phi_eff_ij · gate · n_face
#
#  with ``w_t_ij = smooth_step(r_pad − d_t_ij, ε)`` a smooth indicator
#  ∈ [0, 1] — NOT a partition-of-unity weight.  Each target sample j is
#  reached by ~π·r_pad²/h² pad spheres at Option-2 tiling (kernel half-
#  width = r_pad = lattice spacing h), so its Voronoi area A_j is
#  counted ~π× across the lattice.  Total force scales with lattice
#  density rather than physical contact-patch area — refining the pad
#  lattice at fixed r_pad multiplies F by ~(h_old/h_new)², the
#  resolution-knob #7 finding in the handoff.
#
#  The fix is to renormalize:
#
#      W_j = Σ_i  w_t_ij · α_ij        (sum over reaching pad spheres)
#      share_ij = w_t_ij · α_ij / W_j
#
#  Then ``Σ_i share_ij = 1`` for every target j (proper partition of
#  unity), and the total force becomes
#
#      Σ_i F_i = Σ_j  kc · A_j · ⟨phi_eff⟩_j · n_face_j
#
#  — a Riemann sum over target Voronoi cells, lattice-density-invariant,
#  matching hydroelastic's ∫_Ω k_h·φ·n dA structure (Elandt 2019 §3.2).
#
#  W is computed ONCE per pair launch on the warm-start delta (between
#  ``lattice_solve_equilibrium`` and the Jacobi loop).  Rationale: the
#  δ corrections during Jacobi sweeps are sub-mm; target spacing is mm–
#  cm; so d_t = ‖tangential separation‖ barely moves across iterations,
#  and W ≈ const.  If lift fidelity drifts vs the per-iter reduction,
#  promote W to per-iter (~2× kernel launches inside the Jacobi loop).
#
#  Active-set gates here MUST mirror ``jacobi_step`` exactly so the
#  numerator (w_t_ij · α_ij used in jacobi_step) and the denominator
#  (W_j summed here) include / exclude the same pairs.  Any divergence
#  breaks the partition-of-unity property.
# ═══════════════════════════════════════════════════════════════════════════


@wp.kernel
def compute_target_W(
    delta_src: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_shape: wp.array(dtype=wp.int32),
    active_cslc_shape_idx: int,
    n_spheres: int,
    pad_pos_world: wp.array(dtype=wp.vec3),
    out_normal_world_in: wp.array(dtype=wp.vec3),
    target_pos_world: wp.array(dtype=wp.vec3),
    target_normal_world: wp.array(dtype=wp.vec3),
    eps: float,
    W_out: wp.array(dtype=wp.float32),
):
    """For each target j, sum w_tangent_ij · align_w_ij over all active
    surface pad spheres i in the active lattice.

    Gates are bit-identical to ``jacobi_step``'s inner loop so the
    numerator there and the denominator here cover the same pairs.
    """
    j = wp.tid()
    t_j_world = target_pos_world[j]
    n_face_world = target_normal_world[j]

    W = float(0.0)
    for i in range(n_spheres):
        # Lattice filter
        if sphere_shape[i] != active_cslc_shape_idx:
            continue
        if is_surface[i] == 0:
            continue

        r_i = sphere_radii[i]
        p_i_world = pad_pos_world[i]
        q_i_world = p_i_world - delta_src[i]
        out_n_world = out_normal_world_in[i]

        diff_qt = q_i_world - t_j_world
        dist = wp.length(diff_qt)
        if dist < 1.0e-15:
            continue

        # Raw cull (contract §3.6; literal must match
        # INACTIVE_RAW_EPS_FACTOR = -50.0).  Theory fix D1: cull on
        # raw_face = raw - r_i (face-penetration form), parity-locked
        # with jacobi_step.  If this cull and jacobi_step's cull
        # disagree, partition-of-unity (Σ_i share_ij = 1) breaks and
        # the per-pair force divides by a stale W.
        raw = r_i - wp.dot(diff_qt, n_face_world)
        raw_face = raw - r_i
        if raw_face < -50.0 * eps:
            continue
        # Theory fix D2 — distance-magnitude cull (mirror of
        # jacobi_step).  W_j sums over active pad spheres for target
        # j; this cull MUST match jacobi_step's so the numerator there
        # and denominator here cover the same pairs (otherwise
        # share_ij = w_t·α/W_j doesn't sum to 1 and per-pair force
        # divides by a wrong W).
        if dist > 3.0 * r_i + 5.0 * eps:
            continue

        # One-sided alignment cull (contract §3.6, EPS_ALIGN_DEFAULT
        # = 0.05).  Mirror of jacobi_step.
        align_arg = -wp.dot(n_face_world, out_n_world)
        if align_arg <= 0.0:
            continue
        align_w = float(1.0)
        if align_arg < 0.05:
            t_lerp = wp.clamp(align_arg / 0.05, 0.0, 1.0)
            align_w = t_lerp * t_lerp * (3.0 - 2.0 * t_lerp)

        # Tangential locality.  kernel_h = r_i (Option-2 tiling), hard
        # cull at w_tangent < 1e-2 — mirrors jacobi_step.
        d_t_vec = diff_qt - wp.dot(diff_qt, n_face_world) * n_face_world
        d_t_mag = wp.length(d_t_vec)
        kernel_h = r_i
        w_tangent = smooth_step(kernel_h - d_t_mag, eps)
        if w_tangent < 1.0e-2:
            continue

        W += w_tangent * align_w

    W_out[j] = W


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
    # Per-sphere world-frame rest centre, precomputed once per pair by
    # ``compute_pad_pos_world``.  Hoisted out of the n_iter Jacobi loop
    # because the underlying transforms (X_ws, X_wb) are constant during
    # the quasi-static solve.
    pad_pos_world: wp.array(dtype=wp.vec3),
    # Per-sphere world-frame outward normal, precomputed once per pair
    # by ``compute_outward_normals_world``.  Hoisted out of the n_iter
    # Jacobi loop for the same reason.
    out_normal_world_in: wp.array(dtype=wp.vec3),
    ka_tangent_ratio: float,
    k_stick: float,
    mu_friction: float,
    # Per-target world-frame position and outward face normal,
    # precomputed once per pair by ``compute_target_world_state``.
    # Replace the in-loop ``transform_point`` / ``transform_vector``
    # of ``target_positions_local`` / ``target_normals_local`` -- the
    # target body's pose ``X_tb`` is constant during the Jacobi sweep.
    target_pos_world: wp.array(dtype=wp.vec3),
    target_normal_world: wp.array(dtype=wp.vec3),
    # Per-target Voronoi area on the underlying surface [m^2].  Folded
    # into the contact force with a tangential locality kernel
    # w_tangent to reconstruct the surface integral
    #     F = ∫_{contact_patch} kc · phi · n_face dA
    # from the discrete sample set (contract §5).
    target_areas_local: wp.array(dtype=wp.float32),
    # Per-target partition-of-unity normalizer
    # ``W_j = Σ_i w_tangent_ij · align_w_ij`` (theory fix #3+#4).  Each
    # per-pair contribution is divided by W_j so the share weights sum
    # to 1 across pad spheres reaching target j; total force becomes a
    # Riemann sum over target Voronoi cells, lattice-density-invariant
    # and matching hydroelastic's ∫_Ω k_h·φ·n dA structure.  Computed
    # once per pair launch by ``compute_target_W`` on the warm-start δ
    # (see CSLCHandler._launch).  Targets with W_j ≈ 0 (no pad sphere
    # reaches them) are skipped — their share is undefined and their
    # phi_eff would be 0 anyway.
    W_targets: wp.array(dtype=wp.float32),
    target_count: int,
    # External tangential load (bridge / experiment driver only).
    # ``f_ext_apex_idx = -1`` is the no-op sentinel (production).
    f_ext_apex_idx: int,
    f_ext_apex: wp.vec3,
    eps: float,
    # B3 — lattice velocity damping.  ``delta_prev_step`` is a snapshot
    # of ``sphere_delta`` taken at the start of this simulation step;
    # ``c_over_dt = c_lattice / dt`` is the damping rate.  Setting
    # ``c_over_dt = 0`` makes the damping force identically zero and
    # the kernel reduces to the pre-B3 form.
    delta_prev_step: wp.array(dtype=wp.vec3),
    c_over_dt: float,
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
       phi_eff_im  = σ_ε(raw_im) · √(σ_ε(raw_im) + ε)               (Hertz-like δ^1.5;
                                                                     DIVERGES from contract §4
                                                                     eq:phi-eff which specifies
                                                                     linear σ_ε(raw))
       gate_im     = smooth_step(raw_im, eps)
       d_t_im      = ‖(q_i − t_m) − n_face_m · (q_i − t_m) · n_face_m‖
       w_t_im      = smooth_step(r_lat_i − d_t_im, eps)             (Option-2 tiling,
                                                                     kernel half-width = r_pad;
                                                                     DIVERGES from contract §3.5
                                                                     eq:w_t which specifies 3·r_pad)
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

    # Per-sphere world-frame rest centre and outward normal are
    # precomputed once per pair launch (compute_pad_pos_world,
    # compute_outward_normals_world) -- they don't change across the
    # n_iter Jacobi sweeps because X_wb / X_ws are constant during the
    # quasi-static solve.  Hoisting them out of the per-iter inner loop
    # saves n_iter-1 transform_point/_vector pairs per sphere.
    p_i_world = pad_pos_world[tid]
    q_i_world = p_i_world - delta_old
    out_n_world = out_normal_world_in[tid]

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
    # Per-axis contact-spring diagonal stabilisers (Phase 5a fix).  The
    # contact spring acts along n_eff = -n_face with stiffness kc; its
    # per-sphere stiffness tensor is kc · n_eff·n_eff^T.  Projecting
    # onto pad sphere i's local outward-normal frame (axis = n_pad,
    # angle α = ∠(n_eff, n_pad)) gives:
    #     normal-axis diag contribution: kc · cos²α
    #     tangent-axis diag contribution: kc · sin²α
    # where cos α = -(n_face · n_pad) = align_arg (already computed for
    # the alignment gate).
    #
    # The pre-Phase-5a kernel accumulated a single sum_gate and used it
    # for the normal axis only (S_t had no kc term).  For flat pads
    # cos²α ≈ 1 everywhere so the bug was silent; for curved pads
    # (dome) off-apex spheres have α up to ~70°, so sin²α ≈ 0.9 of the
    # contact stiffness was MISSING from S_t -- letting the per-iter
    # tangent update over-relax against the under-braced denominator
    # k_diag_t = ka·ratio + kl·deg ≈ 4·10⁴.  With the true contact
    # stiffness kc ≈ 3·10¹⁰ on the tangent axis, the iteration
    # amplifies tangent residuals by ~10⁶ per step and the lattice
    # cannot settle.
    #
    # Splitting the accumulator into sum_gate_n / sum_gate_t and using
    # both on the Jacobi diagonal restores the contraction property on
    # curved pads while leaving flat-pad behaviour bit-identical
    # (cos²α = 1 ⇒ sum_gate_n = sum_gate_old, sum_gate_t = 0, exactly
    # matching the pre-Phase-5a formulation).
    sum_gate_n = float(0.0)
    sum_gate_t = float(0.0)
    f_friction_vec = wp.vec3(0.0, 0.0, 0.0)

    if is_surface[tid] == 1:
        r_i = sphere_radii[tid]
        # Target body pose ``X_tb`` is constant during the quasi-static
        # Jacobi sweep, so ``t_j_world`` and ``n_face_world`` are
        # precomputed once per pair launch (compute_target_world_state)
        # instead of recomputed every inner-loop iter.
        for j in range(target_count):
            t_j_world = target_pos_world[j]
            # Degenerate centres-coincide check.  1e-15 is numerical
            # zero (NOT ``eps``, the smooth-gate width); skipping pairs
            # at the smooth-gate radius would silently drop deeply-
            # overlapping contacts.  ``dist`` is reused below for the
            # theory-fix-D2 distance-magnitude cull.
            diff_qt = q_i_world - t_j_world
            dist = wp.length(diff_qt)
            if dist < 1.0e-15:
                continue
            n_face_world = target_normal_world[j]
            # Half-space raw (v2, contract eq:raw):
            #     raw = r_i − n_face · (q − t)
            # Monotone in penetration depth at any depth, doesn't flip
            # sign at face crossing (unlike the v1 sphere-overlap
            # form which assumed a single sphere target with radius
            # R and used (r_i + R) − ‖q − t‖).
            raw = r_i - wp.dot(diff_qt, n_face_world)
            # Theory fix D1 — face-penetration form.  ``raw`` carries
            # an r_i shelf (= r_i at face contact for flat lattice
            # surfaces), summing to a 84 N spurious force at face
            # contact on a 490-sphere box pad.  raw_face = raw - r_i =
            # -n_face · (q - t) is zero at face contact, positive at
            # face penetration; the active-set gate and phi below
            # consume raw_face so phi(face contact) = 0 exactly.
            raw_face = raw - r_i
            # Half-space gate (literal MUST match INACTIVE_RAW_EPS_FACTOR
            # = -50.0 in cslc_main/theory/cslc_theory.py).  Gate is on
            # raw_face post-fix-D1: cull pad spheres whose surface has
            # separated by more than 50·ε from face contact.
            if raw_face < -50.0 * eps:
                continue
            # Theory fix D2 — distance-magnitude cull (mirror of
            # compute_cslc_penetration).  Drops phantom pairs where
            # the pad sphere is on the OPPOSITE side of the target
            # body from the target sample but where alignment +
            # tangential locality + raw_face all happen to pass.
            # Discovered on the dome scene; see compute_cslc_penetration
            # for the full diagnosis.  Threshold parity-locked.
            if dist > 3.0 * r_i + 5.0 * eps:
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
            # phi_eff = raw_face^1.5 (smooth-blend lifted to power 1.5).
            # The 1.5 exponent makes per-contact stiffness vanish at
            # first touch (raw_face→0): dF/d(raw_face) ∝ √raw_face → 0.
            # Sphere-on-flat gives F ∝ δ^2.5 (one power stiffer than
            # Hertz's δ^1.5) because the area-weighted sum adds one
            # power of δ via the contact-patch area scaling with δ.
            # Trade-off: the impulse-on-engagement problem that linear
            # contact had is eliminated, at the cost that kc no longer
            # equals the material's Young modulus directly -- see
            # CSLCParams.
            #
            # Theory fix #1+#2 (Hertz floor + low-raw shape): swapped
            # smooth_relu → smooth_blend.  smooth_blend is exactly 0
            # for raw_face ≤ 0 (kills the ε^1.5 force baseline),
            # exactly raw_face for raw_face ≥ ε (so phi_eff →
            # raw_face·√raw_face = raw_face^1.5, TRUE Hertz at depth),
            # C² blend in between.  smooth_step(raw_face, eps) gate
            # below is now structurally redundant for the FORCE (phi_eff
            # already vanishes at raw_face=0) but retained for the
            # diagonal stabiliser via sum_gate_{n,t} weighting so the
            # iteration's contraction proof carries over unchanged.
            #
            # Theory fix D1 (face-penetration form): both phi_eff and
            # gate consume raw_face = raw - r_i so they vanish at FACE
            # contact rather than at lattice-sphere-centre contact.
            # The pre-D1 form left an r_i shelf at face onset that
            # summed to ~84 N spurious force on box/box (every engaged
            # sphere saw raw = r_i ≈ 0.3 mm at δ=0).
            raw_pos = smooth_blend(raw_face, eps)
            phi_eff = raw_pos * wp.sqrt(raw_pos + eps)
            gate = smooth_step(raw_face, eps)
            # Tangential locality kernel (Option-2 tiling, no overlap).
            # Kernel half-width = r_pad — DIVERGES from contract §3.5
            # eq:w_t which specifies 3·r_pad.  The (A_j · w_tangent)
            # factor reconstructs the surface integral
            # F = ∫ kc · phi · n_face dA from the discrete sample set;
            # without it the coherent face-normal sum overcounts by a
            # factor of (face_area_in_reach / contact_patch_area).
            # Must match compute_cslc_penetration and write_cslc_contacts.
            d_t_vec = diff_qt - wp.dot(diff_qt, n_face_world) * n_face_world
            d_t_mag = wp.length(d_t_vec)
            kernel_h = r_i
            w_tangent = smooth_step(kernel_h - d_t_mag, eps)
            # Phase 6 fix: HARD CULL on w_tangent (parity with
            # compute_cslc_penetration line ~250 and write_cslc_contacts
            # line ~1054).  Without this cull, the smooth_step tail at
            # d_t > kernel_h evaluates to ~eps/(2·d_t) -- e.g.
            # ≈ 6e-4 at d_t = 10·r_pad.  For a curved pad sphere whose
            # outward normal is tilted from the apex direction, target
            # samples on the FAR side of the convex object (sphere /
            # box) pass the alignment cull because their face normals
            # align with the pad sphere's tilted normal.  The half-
            # space raw for those far samples is unbounded
            # (raw = r_lat - n_face·(q - t) grows with object scale),
            # so phi_eff = raw^1.5 is HUGE.  Multiplied by kc and the
            # nonzero w_tangent tail, each far-side sample contributes
            # tens of N of phantom contact force to a pad sphere that
            # isn't actually touching the object.  Empirically: a single
            # rim sphere on the production dome had 25 out of 50 sphere
            # samples passing align+gate but with d_t > kernel_h, summing
            # to 157 N of spurious force in the load buffer.  The other
            # two kernels in the CSLC pipeline already cull this tail
            # at w_tangent < 1e-2; jacobi_step diverged from them.
            # See cslc_main/grasp/scripts/probe_failure.py (dome rim
            # diagnostic) for the discovery trace.
            if w_tangent < 1.0e-2:
                continue
            # Theory fix #3+#4 — target-side partition of unity:
            # divide A_j by W_j = Σ_i w_t_ij · α_ij so the per-pair area
            # share sums to 1 across pad spheres reaching target j.  The
            # 1e-30 floor protects targets that survive the gates here
            # but have W_j ≈ 0 from upstream-launch parity issues; in
            # practice such targets contribute zero force because their
            # numerator (w_tangent · align_w) is also ≈ 0.
            W_j = W_targets[j]
            inv_W = 1.0 / (W_j + 1.0e-30)
            A_j = target_areas_local[j] * inv_W
            area_kernel = A_j * w_tangent
            # Load form: load = -∂E_contact/∂δ.  Physical force on the
            # pad sphere is +kc · A_j · w_tangent · align · phi_eff
            # · gate · n_face (face's outward normal); load form is
            # the negative of that.  ``phi_eff = σ_ε(raw) · √(σ_ε(raw) + ε)``
            # is always ≥ 0, so the contact force never reverses sign
            # near the smooth-cull boundary — critical for the
            # no-bulge regression on multi-sphere lattices (contract
            # §6.2 / T-G).
            n_eff = -n_face_world
            f_contact_vec = f_contact_vec + kc * \
                area_kernel * align_w * phi_eff * gate * n_eff
            # Diagonal stabilisation -- split per-axis by alignment angle.
            # align_arg = -(n_face · n_pad) = (n_eff · n_pad) = cos α.
            # cos²α goes to S_n, sin²α = 1 - cos²α goes to S_t.  For flat
            # pads cos²α = 1 ⇒ S_t contribution = 0 ⇒ bit-identical to
            # the pre-Phase-5a sum_gate path on the normal axis.
            cos2 = align_arg * align_arg
            contrib = area_kernel * align_w * gate
            sum_gate_n = sum_gate_n + contrib * cos2
            sum_gate_t = sum_gate_t + contrib * (1.0 - cos2)

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

    # B3 — Lattice velocity-damping (IMPLICIT formulation).
    # Continuum equation: c · δ̇ + ka · δ = forces  (quasi-static).
    # Backward-Euler: δ̇ ≈ (δ_new - δ_prev_step)/dt, so:
    #     (c/dt + ka) · δ_new = forces + (c/dt) · δ_prev_step
    # The (c/dt) coefficient appears on BOTH sides.  In our Jacobi
    # update that means:
    #   * EXPLICIT part: ``+ c_over_dt · δ_prev_step`` added to rhs
    #     (constant within the iteration, evaluated below as f_damping)
    #   * IMPLICIT part: ``+ c_over_dt`` added to k_diag_n and k_diag_t
    #     (handled where those are assembled below)
    # The pre-fix version put the entire damping force into the rhs as
    # ``-c_over_dt·(δ_old - δ_prev_step)`` which made the iteration
    # explicit in δ_old and unstable when c_over_dt > k_diag (ball
    # ejected on production sweeps).  The implicit form below is
    # unconditionally stable: increasing c_over_dt monotonically pulls
    # δ_new toward δ_prev_step.
    f_damping = c_over_dt * delta_prev_step[tid]

    # Anisotropic block-Jacobi in pad sphere's local rest-normal frame
    # (contract §6.5).  S_n picks up the SUM of contact gates × area
    # × align over the target samples.  This sum upper-bounds
    # |d(f_contact·n)/d(δ_n)| over the active set, so the iteration's
    # contraction property holds.
    rhs_explicit = f_contact_vec + f_lateral + \
        f_friction_vec + f_ext_vec + f_damping  # B3
    rhs_n_scalar = wp.dot(rhs_explicit, out_n_world)
    rhs_t_vec = rhs_explicit - rhs_n_scalar * out_n_world

    delta_old_n = wp.dot(delta_old, out_n_world)
    delta_old_t = delta_old - delta_old_n * out_n_world

    ka_t = ka * ka_tangent_ratio
    # Phase 5a: split kc·sum_gate into per-axis contributions weighted
    # by cos²α / sin²α (see initialisation comment above).  On flat
    # pads sum_gate_t = 0 so S_t reduces to the pre-Phase-5a value
    # exactly; on curved pads S_t now carries the missing
    # kc·sin²α·sum(area·align·gate) so the iteration's contraction
    # property holds along the tangent axis too.
    S_n = kl * float(n_neighbors) + kc * sum_gate_n
    S_t = kl * float(n_neighbors) + kc * sum_gate_t
    # B3 — implicit-Euler damping adds c/dt to the diagonal on BOTH
    # axes.  Pairs with the (c/dt)·δ_prev_step term added to rhs_explicit
    # above.  When c_over_dt = 0 this collapses to the pre-B3 form.
    k_diag_n = ka + S_n + c_over_dt
    k_diag_t = ka_t + S_t + c_over_dt

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
#    stiffness = 1.5 · cslc_kc · A_j · w_tangent · align · gate · √raw_pos
#                (DIVERGES from contract §8 which specifies
#                cslc_kc · A_j · w_tangent · gate; the extra
#                1.5·√raw factor encodes the Hertz-like phi_eff = raw^1.5
#                force law used in jacobi_step.  cslc_kc already carries
#                the (pad ⊕ target) series-spring composition, done up
#                front in CSLCHandler.from_model_with_lattices)
#    friction  = 1.0                                 (DIVERGES from contract §8
#                which specifies μ; MuJoCo treats rigid_contact_friction
#                as a SCALE on the geom pair base friction (= μ).
#                Writing μ would give effective μ² — bug fix 2026-04-19)
#
#  MuJoCo reconstructs
#    solver_pen = margin0 + margin1 - (point1 - point0) · normal
#               = r_i + 0 - (t_j - q_def) · (-n_face)
#               = r_i - n_face · (q_def - t_j)
#               = raw                                              ✓
#  so the per-contact force MuJoCo applies, ``stiffness · solver_pen``,
#  equals ``1.5 · cslc_kc · A_j · w_tangent · align · gate · raw^1.5`` --
#  matches ``jacobi_step``'s per-pair Hertz-like force law exactly
#  (jacobi_step uses phi_eff = raw^1.5 internally).
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
    # UNUSED.  Previously written to ``out_friction = mu``, which
    # double-counted: MuJoCo's conversion kernel treats
    # rigid_contact_friction as a SCALE on the geom pair base friction
    # (= mu), so writing mu gave effective_mu = mu².  Fix (2026-04-19)
    # writes out_friction = 1.0 below; this arg is retained in the
    # signature only to match the handler call site.  TODO: drop from
    # both the kernel signature and the handler launch in a follow-up.
    shape_material_mu: wp.array(dtype=wp.float32),
    cslc_kc: float,
    target_ke: float,
    cslc_dc: float,
    eps: float,
    # Theory fix #3+#4 — partition-of-unity normalizer (see jacobi_step
    # docstring).  Same W computed once per pair launch and consumed by
    # both jacobi_step (during the Jacobi sweep) and here (when
    # emitting contacts).  Sharing W keeps the lattice solver's
    # per-pair force and MuJoCo's emitted per-contact force bit-
    # identical: both divide A_j by the same W_j.
    W_targets: wp.array(dtype=wp.float32),
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
    Plus two production emission thresholds NOT in the contract:
        (c) ``contact_gate >= 0.5`` (equivalent to ``raw >= 0``) -- only
            slots with positive half-space penetration are emitted to
            MuJoCo.  Lattice solver still uses the full smooth-tail
            internally; this cull only controls MuJoCo solver slots.
            Emitting the negative-raw tail flooded MuJoCo CG with
            ~5–15k near-zero constraints, turning a 2ms step into a
            45ms step.
        (d) ``w_tangent >= 1e-2`` -- sample is tangentially within the
            contact kernel.  See in-body comment for the threshold
            derivation.

    Each emitted contact carries margin0 = r_lat[i], margin1 = 0;
    MuJoCo reconstructs solver_pen = r_lat - n_face · (q_def - t_j) =
    raw_ij per pair, so per-contact force = stiffness · solver_pen
    = 1.5 · kc_series · A_j · w_tangent · align · gate · raw^1.5 --
    the Hertz-like force law that ``jacobi_step`` uses (DIVERGES from
    contract §8 which specifies linear ``kc · A_j · w_t · gate · raw``).
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
        dist = wp.length(diff_qt)
        if dist < 1.0e-15:
            continue

        n_face_world = wp.transform_vector(X_tb, target_normals_local[j])
        # Half-space raw (v2, contract eq:raw):
        #     raw = r_i − n_face · (q_def − t)
        raw = r_i - wp.dot(diff_qt, n_face_world)
        # Theory fix D1 — face-penetration form (mirror of jacobi_step).
        # raw carries an r_i shelf at face contact; raw_face zeroes
        # the shelf so phi(face contact) = 0 and MuJoCo's emitted
        # constraint goes inactive at the geometric face onset rather
        # than r_i below it.
        raw_face = raw - r_i
        # Half-space gate (literal MUST match INACTIVE_RAW_EPS_FACTOR
        # = -50.0 in cslc_main/theory/cslc_theory.py).  Cull on
        # raw_face post-fix-D1, parity-locked with jacobi_step.
        if raw_face < -50.0 * eps:
            continue
        # Theory fix D2 — distance-magnitude cull (mirror of
        # jacobi_step).  Prevents phantom contacts from being emitted
        # to MuJoCo where the pad sphere is on the opposite side of
        # the target body.  Parity-locked threshold.
        if dist > 3.0 * r_i + 5.0 * eps:
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
        # Theory fix D1: contact_gate on raw_face so the emission cull
        # ``contact_gate < 0.5`` is equivalent to ``raw_face < 0``
        # (face has not penetrated) rather than ``raw < 0`` (sphere
        # centre has not crossed the target plane, which was r_i
        # earlier than face contact).
        contact_gate = smooth_step(raw_face, eps)
        # Tangential locality kernel -- folded into emitted stiffness
        # below so MuJoCo applies per-contact force
        #     stiffness · solver_pen
        #     = 1.5 · kc_series · A_j · w_tangent · align · gate · raw^1.5
        # matching the lattice solver's Hertz-like per-pair force exactly
        # (DIVERGES from contract §8 linear form; see kernel 3 header).
        d_t_vec = diff_qt - wp.dot(diff_qt, n_face_world) * n_face_world
        d_t_mag = wp.length(d_t_vec)
        # Kernel half-width = r_pad (Option-2 tiling, no overlap).
        # DIVERGES from contract §3.5 eq:w_t which specifies 3·r_pad.
        # Must match jacobi_step's kernel reach; the emitted contact
        # stiffness uses the same area_kernel so MuJoCo's per-contact
        # force = stiffness · solver_pen equals the lattice solver's
        # per-pair Hertz-like force exactly.
        kernel_h = r_i
        w_tangent = smooth_step(kernel_h - d_t_mag, eps)
        # Theory fix #3+#4 — partition-of-unity normalizer (parity with
        # jacobi_step).  Divide A_j by W_j so the per-pair emitted
        # stiffness encodes the same share = w_t · α / W_j the lattice
        # solver applied during the Jacobi sweep.  1e-30 floor protects
        # the (W_j → 0) degenerate; in that regime w_tangent → 0 anyway
        # so the cull below drops the contact.
        W_j = W_targets[j]
        inv_W = 1.0 / (W_j + 1.0e-30)
        A_j = target_areas_local[j] * inv_W
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
        #       no change to grasp stability.  The lattice solver uses
        #       ``kc·A·w·α·phi_eff·gate`` with
        #       ``phi_eff = σ_ε(raw) · √(σ_ε(raw) + ε)`` (Hertz-like
        #       δ^1.5; see jacobi_step), which is ≈ 0 for raw ≪ −ε —
        #       so hard-culling raw < 0 in emission removes only the
        #       negligible smooth-tail contribution and keeps lattice
        #       equilibrium δ unchanged.
        #   (b) ``w_tangent < 1e-2`` -- sample is tangentially far
        #       outside the contact kernel.  With kernel_h = r_pad
        #       (Option-2 tiling), 1e-2 corresponds to d_t ≈ r_pad + 5·eps;
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
        # Theory fix D1: margin0 = 0 (was r_i) so MuJoCo's
        #     solver_pen = margin0 + margin1 − (point1 − point0)·normal
        #                = 0 + 0 − (t_j − q_def)·(−n_face)
        #                = −n_face · (q_def − t_j)
        #                = raw − r_i
        #                = raw_face
        # reconstructs the FACE-PENETRATION scalar, not the half-space
        # raw with its r_i shelf.  The emitted stiffness below is
        # likewise tied to raw_face (depth_factor = √(raw_face_pos +
        # ε)), so MuJoCo's per-contact force ``stiffness · solver_pen``
        # = 1.5 · kc · A · w · α · gate · raw_face^1.5 matches
        # jacobi_step's per-pair Hertz-like force at the converged δ.
        out_margin0[buf_idx] = 0.0
        # Contract §8: v2 sets margin1 = 0 (v1 used R_j).  MuJoCo's
        # reconstructed solver_pen now reduces to raw_face (= raw - r_i),
        # not raw — see margin0 fix above.
        out_margin1[buf_idx] = 0.0
        out_tids[buf_idx] = 0
        # Hertz-like force law: F = kc · A_j · w · α · gate · raw^1.5.
        # MuJoCo applies F = stiffness · solver_pen = stiffness · raw,
        # so the stiffness must be the LOCAL derivative of F w.r.t. raw
        # evaluated at the current depth:
        #     dF/d(raw) = 1.5 · kc · A_j · w · α · gate · √raw
        # This goes to ZERO at first touch (raw → 0), eliminating the
        # constant-stiffness impulse that the previous linear law had,
        # and growing as √raw at depth — same scaling as Hertz's local
        # stiffness 2 E* √(R·δ) ∝ √δ.  Couples with the jacobi_step
        # change above (phi_eff = raw^1.5) so the lattice solver and
        # MuJoCo agree on the per-contact force.
        #
        # Theory fix #1+#2: smooth_blend in place of smooth_relu so the
        # emitted stiffness vanishes EXACTLY for raw_face ≤ 0 (matching
        # jacobi_step's phi_eff) and equals √raw_face EXACTLY at depth.
        # This site is only reached when raw_face passes the
        # contact_gate ≥ 0.5 cull above (raw_face ≥ 0 in practice).
        #
        # Theory fix D1: smooth_blend consumes raw_face = raw - r_i so
        # the depth_factor — and therefore the emitted stiffness —
        # zeroes at FACE contact, not r_i below it.  Paired with the
        # margin0 = 0 change above so MuJoCo's solver_pen = raw_face,
        # the per-contact force ``stiffness · solver_pen`` is
        # 1.5 · kc · A · w · α · gate · raw_face^1.5, matching the
        # lattice solver's per-pair force exactly post-fix-D1.
        raw_pos = smooth_blend(raw_face, eps)
        depth_factor = wp.sqrt(raw_pos + eps)
        out_stiffness[buf_idx] = smooth_relu(
            kc_emit * area_kernel * align_w * contact_gate * depth_factor,
            1.0e-9)
        # Per-contact damping (A1: was hardcoded 0.0 in v2).
        # MuJoCo's solref/solimp branch logic:
        #   kd = 0   ⇒ timeconst = sqrt(imp/ke) ≈ 0.030 s (stiffness-derived)
        #   kd > 0   ⇒ timeconst = 2 / kd     (explicit, tighter)
        # Tradeoff: tightening contact timeconst also tightens the
        # friction-constraint timeconst (same MuJoCo solref slot), so
        # excessive ``cslc_dc`` makes Coulomb friction softer and the
        # held object can creep down during HOLD.  Caller sets via
        # ``CSLCParams.dc`` (params.py); default 0 preserves legacy
        # behavior.
        out_damping[buf_idx] = cslc_dc
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
