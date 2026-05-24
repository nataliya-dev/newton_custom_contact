# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step 7 / Kernel-vs-theory bridge.

Drive the Warp kernels in ``newton/_src/geometry/cslc_kernels.py``
against the theory shim in :mod:`cslc_main.theory.kernel_bridge`.
Both consume the same :class:`KernelScene`; the test compares
per-sphere equilibrium delta vectors.

Scenes (one per validated theory step) -- each runs at
``kc/ka ∈ {0.1, 1, 10}``:

  A.  single sphere face-on                            (step 1A;   tol=1e-3)
  B.  single sphere off-axis (30 deg)                  (step 1D;   tol=1e-3)
  C.  chain face-on contact at centre, small target    (step 3A;   tol=1e-3)
  D.  arc face-on contact at apex (bulge window)       (step 5B;   tol=1e-3)
  E.  single sphere anisotropic ka_t = ka/3, off-axis  (step 6A;   tol=1e-3)
  F.  single sphere + tangential force, STICK mode     (step 4A;   tol=1e-3)
  G.  single sphere + tangential force, SLIP mode      (step 4B;   tol=2e-2)

Tolerance for G is relaxed to 2% to absorb the known geometric
divergence between the kernel's deformed-direction contact normal
and the theory's analytical slip solver, which uses the rest
direction.  Slip mode requires ``δ_t > F_thresh / ka ≈ 0.3 mm`` at
production-scale parameters; at that scale the rest-direction
assumption breaks at the 1-2% level (the kernel is more accurate).

Run::

    uv run -m cslc_main.theory.test_07_kernel_bridge
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from cslc_main.theory.cslc_theory import INACTIVE_RAW_EPS_FACTOR
from cslc_main.theory.kernel_bridge import (
    KernelScene,
    KernelSolution,
    compute_contact_force,
    compute_contact_force_point_set,
    solve_theory,
)
from newton._src.geometry.cslc_kernels import (
    compute_cslc_penetration,
    compute_pad_force_vs_point_set,
    jacobi_step,
)


# ─────────────────────────────────────────────────────────────────────────
#  Constant-discipline guard (C2d Obs 3)
#
#  Warp kernels can't import Python module constants -- the active-set
#  threshold ``raw < INACTIVE_RAW_EPS_FACTOR * eps`` is hardcoded as the
#  literal ``-50.0 * eps`` at every kernel call site.  ``MUST match`` is
#  the only enforcement.  This check parses cslc_kernels.py for the
#  ``if <var> < <num> * eps:`` pattern and asserts every magnitude
#  matches ``abs(INACTIVE_RAW_EPS_FACTOR)``.  If the Python value ever
#  changes (e.g. to -60) and a kernel site is forgotten, this test
#  catches the split before it ships.
# ─────────────────────────────────────────────────────────────────────────


def _check_constant_discipline() -> bool:
    """Two-part guard on the kernel's `-50.0 * eps` threshold literals.

    Part 1 (v0.7 original): magnitude check.  Every kernel site
    matching ``if <var> [<>=]= <num> * eps`` must have ``abs(<num>)``
    equal to ``abs(INACTIVE_RAW_EPS_FACTOR)``.  Catches a drift where
    the Python value changes but a kernel literal is forgotten.

    Part 2 (v0.8): adjacent-comment check.  Each matching site must
    have a comment within the preceding ``PROXIMITY_LINES`` lines (or
    on the site line itself) that names the symbol
    ``INACTIVE_RAW_EPS_FACTOR``.  Catches a refactor that splits the
    expression into pieces the magnitude regex can no longer
    recognise -- if the symbol comment is also gone, the kernel
    silently loses its tie back to the canonical constant.
    """
    kernel_path = (
        Path(__file__).resolve().parents[2]
        / "newton" / "_src" / "geometry" / "cslc_kernels.py"
    )
    src = kernel_path.read_text()
    lines = src.split("\n")
    # Match a conditional skip on the active-set threshold:
    # ``if <var> < -<num>[.<dec>] * eps:`` or `` >= -<num> * eps``.
    # Comments and docstrings won't match because they don't start
    # with ``if ... <`` / ``... >=`` at the right structural position.
    pattern = re.compile(
        r"(?:if|elif|while|return)\s+[A-Za-z_][\w]*\s*[<>]=?\s*"
        r"(-?\d+(?:\.\d+)?)\s*\*\s*eps"
    )
    expected_mag = abs(INACTIVE_RAW_EPS_FACTOR)
    SYMBOL = "INACTIVE_RAW_EPS_FACTOR"
    # Scan up to this many lines back from each match for the symbol
    # mention.  12 is comfortably more than the deepest comment block
    # any site currently uses (~4 lines); generous against re-flowing.
    PROXIMITY_LINES = 12

    n_sites = 0
    mag_mismatches: list[tuple[int, str]] = []
    symbol_mismatches: list[int] = []
    for line_idx, line in enumerate(lines, start=1):
        m = pattern.search(line)
        if m is None:
            continue
        n_sites += 1
        # Part 1: magnitude check.
        mag = abs(float(m.group(1)))
        if abs(mag - expected_mag) > 1.0e-9:
            mag_mismatches.append((line_idx, m.group(1)))
            continue  # don't bother with symbol check if magnitude is wrong
        # Part 2: adjacent-comment check.  Look back PROXIMITY_LINES
        # lines (inclusive of the site line) for the symbol mention.
        scan_start = max(0, line_idx - 1 - PROXIMITY_LINES)
        adjacent = "\n".join(lines[scan_start:line_idx])
        if SYMBOL not in adjacent:
            symbol_mismatches.append(line_idx)

    if n_sites == 0:
        print(
            f"  WARNING: no active-set threshold sites found in "
            f"{kernel_path.name}; regex may have rotted."
        )
        return False
    if mag_mismatches:
        print(
            f"  FAIL: {len(mag_mismatches)} of {n_sites} kernel "
            f"hardcoded thresholds don't match magnitude "
            f"{expected_mag}: {mag_mismatches}"
        )
        return False
    if symbol_mismatches:
        print(
            f"  FAIL: {len(symbol_mismatches)} of {n_sites} kernel "
            f"threshold sites have no '{SYMBOL}' mention within "
            f"{PROXIMITY_LINES} preceding lines: line(s) {symbol_mismatches}.  "
            f"Add a comment naming the symbol at each site so future "
            f"refactors don't silently drift away from the canonical "
            f"constant in cslc_theory.py."
        )
        return False
    print(
        f"  Kernel hardcoded active-set thresholds ({n_sites} sites) "
        f"all match {SYMBOL} = {INACTIVE_RAW_EPS_FACTOR} AND have "
        f"an adjacent comment naming the symbol: PASS"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────
#  Kernel driver: builds minimal wp.arrays, launches the kernels, reads
#  back sphere_delta.  Skips the full Newton Model / CollisionPipeline
#  path -- step 7 is about the kernel physics, not the wiring.
# ─────────────────────────────────────────────────────────────────────────


def _csr_from_neighbor_indices(neighbor_indices: list[np.ndarray]):
    N = len(neighbor_indices)
    counts = np.array([len(nl) for nl in neighbor_indices], dtype=np.int32)
    starts = np.zeros(N, dtype=np.int32)
    starts[1:] = np.cumsum(counts)[:-1]
    if N > 0 and counts.sum() > 0:
        flat = np.concatenate([nl.astype(np.int32) for nl in neighbor_indices])
    else:
        flat = np.zeros(0, dtype=np.int32)
    return starts, counts, flat


def _rest_lengths_csr(positions: np.ndarray, neighbor_indices: list[np.ndarray]):
    pieces: list[np.ndarray] = []
    for i, nl in enumerate(neighbor_indices):
        if len(nl) == 0:
            continue
        deltas = positions[nl] - positions[i]
        pieces.append(np.linalg.norm(deltas, axis=-1).astype(np.float32))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


@dataclass
class KernelOutput:
    delta: np.ndarray
    raw_penetration: np.ndarray
    contact_normal: np.ndarray
    iters_run: int
    final_residual: float
    convergence_history: np.ndarray


def run_kernel(scene: KernelScene, *,
               n_iter: int = 2000,
               alpha: float = 0.3,
               tol: float = 1.0e-10) -> KernelOutput:
    """Drive the kernels on a ``KernelScene``.

    Builds minimal wp.array inputs (one body, one shape; identity
    transforms; lattice-host shape index = 0; target shape index = 1).
    Launches ``compute_cslc_penetration`` once, then iterates
    ``jacobi_step`` until ``||delta^(k+1) - delta^(k)||_inf < tol``
    or ``n_iter`` is reached.

    Returns per-sphere delta + diagnostics.
    """
    device = wp.get_device()
    N = scene.n_spheres
    eps = float(scene.eps)

    # Lattice host body + target body: 2 distinct bodies, identity transforms.
    body_q_np = np.zeros((2, 7), dtype=np.float32)
    body_q_np[:, 6] = 1.0   # identity quaternion
    body_q = wp.array(body_q_np, dtype=wp.transform, device=device)

    # Two shapes: idx 0 = lattice host (body 0), idx 1 = target (body 1).
    shape_body = wp.array(np.array([0, 1], dtype=np.int32),
                          dtype=wp.int32, device=device)
    shape_transform_np = np.zeros((2, 7), dtype=np.float32)
    shape_transform_np[:, 6] = 1.0
    shape_transform = wp.array(shape_transform_np, dtype=wp.transform, device=device)

    sphere_pos_local = wp.array(scene.positions.astype(np.float32),
                                dtype=wp.vec3, device=device)
    sphere_radii = wp.array(scene.radii.astype(np.float32),
                            dtype=wp.float32, device=device)
    sphere_delta_a = wp.zeros(N, dtype=wp.vec3, device=device)
    sphere_delta_b = wp.zeros(N, dtype=wp.vec3, device=device)
    is_surface = wp.array(scene.is_surface.astype(np.int32),
                          dtype=wp.int32, device=device)
    outward_normals = wp.array(scene.outward_normals.astype(np.float32),
                               dtype=wp.vec3, device=device)
    sphere_shape = wp.zeros(N, dtype=wp.int32, device=device)   # all on shape 0

    neighbor_start_np, neighbor_count_np, neighbor_list_np = \
        _csr_from_neighbor_indices(scene.neighbor_indices)
    neighbor_start = wp.array(neighbor_start_np, dtype=wp.int32, device=device)
    neighbor_count = wp.array(neighbor_count_np, dtype=wp.int32, device=device)
    neighbor_list = wp.array(neighbor_list_np, dtype=wp.int32, device=device)
    neighbor_rest_length = wp.array(
        _rest_lengths_csr(scene.positions, scene.neighbor_indices),
        dtype=wp.float32, device=device,
    )

    raw_penetration = wp.zeros(N, dtype=wp.float32, device=device)
    contact_normal_world = wp.zeros(N, dtype=wp.vec3, device=device)

    # ── Kernel 1: penetration (one-shot, reads rest positions only) ──
    target_local_pos = wp.vec3(*scene.target_position.tolist())
    wp.launch(
        kernel=compute_cslc_penetration,
        dim=N,
        inputs=[
            sphere_pos_local, sphere_radii, sphere_delta_a,
            sphere_shape, is_surface, outward_normals,
            body_q, shape_body, shape_transform,
            0,           # active CSLC shape idx
            1,           # target body idx
            1,           # target shape idx
            target_local_pos,
            float(scene.target_radius),
            eps,
        ],
        outputs=[raw_penetration, contact_normal_world],
        device=device,
    )

    # External tangential force: jacobi_step now accepts one via
    # (f_ext_apex_idx, f_ext_apex).  Single-sphere scenes pass apex_idx=0;
    # multi-sphere scenes with no external load pass -1.  Scene
    # validation already restricts f_ext_tangent to N==1, so we only
    # need to handle that case here.
    if scene.f_ext_tangent is not None:
        f_ext_apex_idx = 0
        f_ext_apex = wp.vec3(*(float(c) for c in scene.f_ext_tangent.tolist()))
    else:
        f_ext_apex_idx = -1
        f_ext_apex = wp.vec3(0.0, 0.0, 0.0)

    # ── Kernel 2: damped Jacobi to convergence ──────────────────────
    src = sphere_delta_a
    dst = sphere_delta_b
    prev = np.zeros((N, 3), dtype=np.float32)
    history: list[float] = []
    iters_run = 0
    final_res = float("inf")
    for k in range(n_iter):
        wp.launch(
            kernel=jacobi_step,
            dim=N,
            inputs=[
                src, dst,
                # Step 7 D2: per-sphere radii (exact deformed overlap).
                sphere_radii,
                sphere_pos_local, neighbor_rest_length, is_surface,
                neighbor_start, neighbor_count, neighbor_list,
                float(scene.ka), float(scene.kl), float(scene.kc),
                alpha,
                sphere_shape, 0,        # active CSLC shape
                outward_normals,
                body_q, shape_body, shape_transform,
                float(scene.ka_t_ratio),
                float(scene.k_stick), float(scene.mu_friction),
                # Step 7 D2: deformed contact direction + exact phi_def.
                1,                # target body idx (matches scene setup)
                target_local_pos, # cached above for kernel 1
                float(scene.target_radius),
                # Optional external tangential load (friction scenes).
                f_ext_apex_idx, f_ext_apex,
                eps,
            ],
            device=device,
        )
        src, dst = dst, src
        cur = src.numpy()
        res = float(np.max(np.abs(cur - prev)))
        history.append(res)
        iters_run = k + 1
        if res < tol:
            final_res = res
            break
        prev = cur
        final_res = res

    delta = src.numpy().astype(np.float64)
    return KernelOutput(
        delta=delta,
        raw_penetration=raw_penetration.numpy().copy(),
        contact_normal=contact_normal_world.numpy().copy(),
        iters_run=iters_run,
        final_residual=final_res,
        convergence_history=np.array(history, dtype=np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────
#  Scene builders
# ─────────────────────────────────────────────────────────────────────────


def _neighbor_lists_from_edges(N: int, edges: np.ndarray) -> list[np.ndarray]:
    nl: list[list[int]] = [[] for _ in range(N)]
    for (i, j) in edges:
        nl[int(i)].append(int(j))
        nl[int(j)].append(int(i))
    return [np.array(x, dtype=np.int32) for x in nl]


def scene_single_face_on(*, ka: float, kc: float, kl: float = 0.0,
                         phi_rest: float = 1.0e-3,
                         r_lat: float = 2.5e-3,
                         R: float = 33.5e-3,
                         eps: float = 1.0e-5) -> KernelScene:
    n_hat = np.array([0.0, 1.0, 0.0])
    p = np.zeros(3)
    t = p + (r_lat + R - phi_rest) * n_hat
    return KernelScene(
        positions=p[None, :].astype(np.float32),
        radii=np.array([r_lat], dtype=np.float32),
        outward_normals=n_hat[None, :].astype(np.float32),
        neighbor_indices=[np.zeros(0, dtype=np.int32)],
        is_surface=np.array([True]),
        target_position=t.astype(np.float32),
        target_radius=float(R),
        ka=float(ka), kl=float(kl), kc=float(kc),
        eps=eps,
    )


def scene_single_off_axis(*, ka: float, kc: float,
                          theta_deg: float = 30.0,
                          phi_rest: float = 1.0e-3,
                          r_lat: float = 2.5e-3,
                          R: float = 33.5e-3,
                          eps: float = 1.0e-5) -> KernelScene:
    n_hat = np.array([0.0, 1.0, 0.0])
    t_hat = np.array([1.0, 0.0, 0.0])
    theta = np.radians(theta_deg)
    u = np.cos(theta) * n_hat + np.sin(theta) * t_hat
    p = np.zeros(3)
    d = (r_lat + R) - phi_rest
    target = p + d * u
    return KernelScene(
        positions=p[None, :].astype(np.float32),
        radii=np.array([r_lat], dtype=np.float32),
        outward_normals=n_hat[None, :].astype(np.float32),
        neighbor_indices=[np.zeros(0, dtype=np.int32)],
        is_surface=np.array([True]),
        target_position=target.astype(np.float32),
        target_radius=float(R),
        ka=float(ka), kl=0.0, kc=float(kc),
        eps=eps,
    )


def scene_chain_single_contact(*, ka: float, kc: float,
                               kl_over_ka: float = 0.2,
                               N: int = 21,
                               h: float = 2.5e-3,
                               r_lat: float = 2.5e-3,
                               eps: float = 1.0e-5) -> KernelScene:
    from cslc_main.theory.cslc_lattice import make_chain
    lat = make_chain(N=N, h=h, ka=ka, kl=kl_over_ka * ka)
    i_c = N // 2
    # Small target so only the centre sphere overlaps at rest.
    R = 1.0e-3
    phi = 0.5e-3
    target = lat.p[i_c] + (r_lat + R - phi) * lat.n[i_c]
    return KernelScene(
        positions=lat.p.astype(np.float32),
        radii=np.full(N, r_lat, dtype=np.float32),
        outward_normals=lat.n.astype(np.float32),
        neighbor_indices=_neighbor_lists_from_edges(N, lat.edges),
        is_surface=np.ones(N, dtype=bool),
        target_position=target.astype(np.float32),
        target_radius=float(R),
        ka=float(ka), kl=float(kl_over_ka * ka), kc=float(kc),
        eps=eps,
    )


def scene_arc_single_contact(*, ka: float, kc: float,
                             R_pad: float = 10.0e-3,
                             N: int = 31,
                             eps: float = 1.0e-5) -> KernelScene:
    from cslc_main.theory.cslc_lattice import make_arc
    arc = make_arc(N=N, R_pad=R_pad, arc_length_spacing=1.0e-3,
                   ka=ka, kl=ka)
    i_a = N // 2
    r_lat = 0.5e-3
    # Tiny target so only the apex sphere is in contact (bulge
    # geometry is visible in the multi-sphere lattice via the
    # distance-preserving lateral, but the contact itself is at the
    # apex only).
    R = 0.6e-3
    phi = 30.0e-6
    target = arc.p[i_a] + (r_lat + R - phi) * arc.n[i_a]
    return KernelScene(
        positions=arc.p.astype(np.float32),
        radii=np.full(N, r_lat, dtype=np.float32),
        outward_normals=arc.n.astype(np.float32),
        neighbor_indices=_neighbor_lists_from_edges(N, arc.edges),
        is_surface=np.ones(N, dtype=bool),
        target_position=target.astype(np.float32),
        target_radius=float(R),
        ka=float(ka), kl=float(ka), kc=float(kc),
        eps=eps,
    )


def scene_single_friction_stick(*, ka: float, kc: float,
                                phi_rest: float = 1.0e-3,
                                r_lat: float = 2.5e-3,
                                R: float = 33.5e-3,
                                k_stick: float | None = None,
                                mu: float = 0.3,
                                F_ext_x: float = 0.3,
                                eps: float = 1.0e-7) -> KernelScene:
    """Single sphere face-on + tangential load **in stick mode**.

    F_thresh = mu * f_n * (ka + k_stick) / k_stick is the slip
    threshold; with defaults (ka = kc = k_stick = 25000, mu = 0.3,
    phi_rest = 1 mm) we get f_n at most ~22.7 N (kc/ka = 10) so
    F_thresh ≥ 1.36 N for every kc factor tested.  F_ext = 0.3 N
    stays well inside the stick cone at every kc.
    """
    scene = scene_single_face_on(
        ka=ka, kc=kc, phi_rest=phi_rest, r_lat=r_lat, R=R, eps=eps,
    )
    scene.k_stick = float(ka if k_stick is None else k_stick)
    scene.mu_friction = float(mu)
    scene.f_ext_tangent = np.array([F_ext_x, 0.0, 0.0], dtype=np.float32)
    return scene


def scene_single_friction_slip(*, ka: float, kc: float,
                               phi_rest: float = 1.0e-3,
                               r_lat: float = 2.5e-3,
                               R: float = 33.5e-3,
                               k_stick: float | None = None,
                               mu: float = 0.3,
                               F_ext_x: float = 20.0,
                               eps: float = 1.0e-7) -> KernelScene:
    """Single sphere face-on + tangential load **in slip mode**.

    With kc=ka, F_thresh ≈ 7.5 N; with kc/ka = 10, F_thresh ≈ 13.6 N.
    F_ext = 20 N exceeds both, putting all three kc factors past the
    cone boundary into Coulomb plateau (friction force = mu * f_n,
    anchor takes the rest of the load).
    """
    scene = scene_single_face_on(
        ka=ka, kc=kc, phi_rest=phi_rest, r_lat=r_lat, R=R, eps=eps,
    )
    scene.k_stick = float(ka if k_stick is None else k_stick)
    scene.mu_friction = float(mu)
    scene.f_ext_tangent = np.array([F_ext_x, 0.0, 0.0], dtype=np.float32)
    return scene


def scene_anisotropic_off_axis(*, ka: float, kc: float,
                               ka_t_ratio: float = 1.0 / 3.0,
                               theta_deg: float = 5.0,
                               phi_rest: float = 1.0e-3,
                               r_lat: float = 2.5e-3,
                               R: float = 33.5e-3,
                               eps: float = 1.0e-5) -> KernelScene:
    scene = scene_single_off_axis(
        ka=ka, kc=kc, theta_deg=theta_deg, phi_rest=phi_rest,
        r_lat=r_lat, R=R, eps=eps,
    )
    scene.ka_t_ratio = float(ka_t_ratio)
    return scene


# ─────────────────────────────────────────────────────────────────────────
#  Step 10 / C1: point-set target scene builders + kernel driver
#
#  Scenes built here exercise the NEW kernel
#  ``compute_pad_force_vs_point_set`` against the theory-side
#  ``compute_contact_force_point_set`` from kernel_bridge.  Because C1
#  only ships the FORCE aggregation kernel (no Jacobi yet for point-set
#  targets), the comparator below evaluates the kernel at the THEORY's
#  converged delta and asserts the per-pad force vector matches.
#  Equilibrium-level comparison for the lattice + point-set path is
#  deferred to C2 once the Jacobi kernel for point-set targets lands.
# ─────────────────────────────────────────────────────────────────────────


def scene_single_vs_box_face_on(*, ka: float, kc: float,
                                r_lat: float = 1.5e-3,
                                box_half: float = 0.025,
                                phi_rest: float = 0.5e-3,
                                n_samples: int = 600,
                                eps: float = 1.0e-7) -> KernelScene:
    """One pad sphere face-on against a sampled box top face.

    The pad sits ``phi_rest`` into the box's flat top face -- same
    geometry as test_10 PART B, sized for the box at 600 samples with
    radius_factor = 1.0 (touching, slightly overlapping target spheres).
    """
    from cslc_main.theory.cslc_box import make_box_target
    box = make_box_target(
        extents=(2.0 * box_half, 2.0 * box_half, 2.0 * box_half),
        n_samples=n_samples, seed=0, radius_factor=1.0)
    n_hat = np.array([0.0, 0.0, -1.0])   # pad outward toward the box
    p = np.array([0.0, 0.0, box_half + r_lat - phi_rest])
    return KernelScene(
        positions=p[None, :].astype(np.float32),
        radii=np.array([r_lat], dtype=np.float32),
        outward_normals=n_hat[None, :].astype(np.float32),
        neighbor_indices=[np.zeros(0, dtype=np.int32)],
        is_surface=np.array([True]),
        target_position=np.zeros(3, dtype=np.float32),   # ignored
        target_radius=0.0,                                # ignored
        target_point_set=box,
        ka=float(ka), kl=0.0, kc=float(kc),
        eps=eps,
    )


def scene_chain_vs_box_face_on(*, ka: float, kc: float,
                               kl_over_ka: float = 0.2,
                               N: int = 5,
                               h: float = 2.5e-3,
                               r_lat: float = 1.5e-3,
                               box_half: float = 0.010,
                               phi_rest: float = 0.5e-3,
                               n_samples: int = 200,
                               eps: float = 5.0e-4) -> KernelScene:
    """Chain (multi-pad-sphere) face-on against a sampled box top face.

    Box centred over the chain's middle sphere; chain spheres press
    upward into the bottom face.  This exercises the lattice +
    point-set dispatch path in solve_theory.
    """
    from cslc_main.theory.cslc_box import make_box_target
    from cslc_main.theory.cslc_lattice import make_chain
    lat = make_chain(N=N, h=h, ka=ka, kl=kl_over_ka * ka)
    i_centre = N // 2
    centre_p = lat.p[i_centre]
    centre_n = lat.n[i_centre]   # chain default outward = +y

    # Box centre = ``phi_rest`` past the chain centre along its outward
    # normal so the chain's middle sphere has ``phi_rest`` worth of
    # penetration into the box face nearest it.
    box_center = (centre_p
                  + (centre_n
                     * (r_lat + box_half - phi_rest)))
    box = make_box_target(
        extents=(2.0 * box_half, 2.0 * box_half, 2.0 * box_half),
        n_samples=n_samples, seed=0,
        center=box_center, radius_factor=1.0)
    return KernelScene(
        positions=lat.p.astype(np.float32),
        radii=np.full(N, r_lat, dtype=np.float32),
        outward_normals=lat.n.astype(np.float32),
        neighbor_indices=_neighbor_lists_from_edges(N, lat.edges),
        is_surface=np.ones(N, dtype=bool),
        target_position=np.zeros(3, dtype=np.float32),
        target_radius=0.0,
        target_point_set=box,
        ka=float(ka), kl=float(kl_over_ka * ka), kc=float(kc),
        eps=eps,
    )


@dataclass
class KernelPointSetOutput:
    """Per-pad-sphere aggregate contact force from the new C1b kernel."""
    pad_force: np.ndarray   # (N, 3) world-frame per-pad force


def run_kernel_point_set(scene: KernelScene,
                         deltas: np.ndarray) -> KernelPointSetOutput:
    """Drive ``compute_pad_force_vs_point_set`` at the supplied deltas.

    Identical wp.array plumbing to ``run_kernel`` (one body for the
    lattice host, one for the target).  The kernel returns the
    per-pad-sphere aggregate force; no Jacobi iteration, so this is a
    one-shot evaluation at the given delta field rather than a full
    equilibrium solve.

    Args:
        scene: must have ``scene.is_point_set``.
        deltas: (N, 3) per-pad-sphere displacements at which to
            evaluate the kernel.  Typically the theory's converged
            delta (so we're verifying kernel force == theory force at
            the same configuration).

    Returns:
        ``KernelPointSetOutput`` with ``pad_force[i]`` = world-frame
        aggregate contact force on pad sphere i.
    """
    if not scene.is_point_set:
        raise ValueError(
            "run_kernel_point_set requires a point-set scene; use "
            "run_kernel for sphere-target scenes.")
    device = wp.get_device()
    N = scene.n_spheres
    pst = scene.target_point_set

    # Lattice host (body 0) + target (body 1); identity transforms.
    body_q_np = np.zeros((2, 7), dtype=np.float32); body_q_np[:, 6] = 1.0
    body_q_wp = wp.array(body_q_np, dtype=wp.transform, device=device)
    shape_body_wp = wp.array(np.array([0, 1], dtype=np.int32),
                             dtype=wp.int32, device=device)
    shape_transform_np = np.zeros((2, 7), dtype=np.float32)
    shape_transform_np[:, 6] = 1.0
    shape_transform_wp = wp.array(shape_transform_np, dtype=wp.transform,
                                  device=device)

    sphere_pos_local_wp = wp.array(scene.positions.astype(np.float32),
                                   dtype=wp.vec3, device=device)
    sphere_radii_wp = wp.array(scene.radii.astype(np.float32),
                               dtype=wp.float32, device=device)
    sphere_delta_wp = wp.array(deltas.astype(np.float32),
                               dtype=wp.vec3, device=device)
    sphere_shape_wp = wp.zeros(N, dtype=wp.int32, device=device)   # shape 0
    is_surface_wp = wp.array(scene.is_surface.astype(np.int32),
                             dtype=wp.int32, device=device)
    outward_normals_wp = wp.array(
        scene.outward_normals.astype(np.float32),
        dtype=wp.vec3, device=device)

    target_pos_wp = wp.array(pst.positions.astype(np.float32),
                             dtype=wp.vec3, device=device)
    target_rad_wp = wp.array(pst.radii.astype(np.float32),
                             dtype=wp.float32, device=device)
    target_norm_wp = wp.array(pst.normals.astype(np.float32),
                              dtype=wp.vec3, device=device)

    pad_force_wp = wp.zeros(N, dtype=wp.vec3, device=device)

    wp.launch(
        kernel=compute_pad_force_vs_point_set,
        dim=N,
        inputs=[
            sphere_pos_local_wp, sphere_radii_wp, sphere_delta_wp,
            sphere_shape_wp, is_surface_wp, outward_normals_wp,
            body_q_wp, shape_body_wp, shape_transform_wp,
            0,   # active_cslc_shape_idx
            target_pos_wp, target_rad_wp, target_norm_wp,
            int(pst.M),
            1,   # target_body_idx
            float(scene.kc),
            float(scene.eps),
        ],
        outputs=[pad_force_wp],
        device=device,
    )
    return KernelPointSetOutput(
        pad_force=pad_force_wp.numpy().astype(np.float64))


# ─────────────────────────────────────────────────────────────────────────
#  Comparator
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ScenePassReport:
    name: str
    kc_over_ka: float
    delta_rel_err: float
    sign_match: bool
    pass_tol: float
    pass_: bool
    kernel_iters: int
    extra: dict


def compare(name: str, scene: KernelScene, *,
            tol: float = 1.0e-3,
            verbose: bool = True) -> ScenePassReport:
    theory_sol = solve_theory(scene)
    kernel_out = run_kernel(scene)
    theory_delta = theory_sol.delta
    kernel_delta = kernel_out.delta

    # Relative L2 error on the full delta field.
    norm_theory = float(np.linalg.norm(theory_delta))
    if norm_theory < 1.0e-15:
        rel = float(np.linalg.norm(kernel_delta))
    else:
        rel = float(np.linalg.norm(theory_delta - kernel_delta) / norm_theory)

    # Sign-regression guard: per-sphere along the SCENE's outward normal,
    # the sign of delta_n must agree between kernel and theory on every
    # sphere where the theory has |delta_n| > 1e-9.
    theory_delta_n = np.einsum("ij,ij->i", theory_delta, scene.outward_normals)
    kernel_delta_n = np.einsum("ij,ij->i", kernel_delta, scene.outward_normals)
    nontrivial = np.abs(theory_delta_n) > 1.0e-9
    if nontrivial.any():
        sign_match = bool(np.all(
            np.sign(theory_delta_n[nontrivial]) ==
            np.sign(kernel_delta_n[nontrivial])
        ))
    else:
        sign_match = True

    passed = (rel < tol) and sign_match
    if verbose:
        print(f"  {name:<40} ka={scene.ka:>7.0f} kc/ka={scene.kc/scene.ka:>5.2f}  "
              f"||δ_k - δ_t||/||δ_t|| = {rel:.3e}  "
              f"sign={'OK' if sign_match else 'FAIL'}  "
              f"kernel iters={kernel_out.iters_run}  "
              f"{'PASS' if passed else 'FAIL'}")
    return ScenePassReport(
        name=name,
        kc_over_ka=scene.kc / scene.ka,
        delta_rel_err=rel,
        sign_match=sign_match,
        pass_tol=tol,
        pass_=passed,
        kernel_iters=kernel_out.iters_run,
        extra={
            "kernel_final_residual": kernel_out.final_residual,
            "theory_regime": theory_sol.solver_info.get("regime"),
            "theory_iters": theory_sol.solver_info.get("nit"),
        },
    )


def compare_point_set(name: str, scene: KernelScene, *,
                      tol: float = 1.0e-3,
                      verbose: bool = True) -> ScenePassReport:
    """C1 comparator: per-pad-sphere force from kernel vs theory.

    Because C1 only ships the FORCE aggregation kernel (no Jacobi yet
    for point-set targets), we evaluate the kernel at the THEORY's
    converged delta -- this checks that the kernel's per-pair sum
    matches numpy element-wise, which is the C1 deliverable.  The
    full equilibrium comparison comes when C2 lands the point-set
    Jacobi kernel.

    Metric: per-pad-sphere relative L2 error of the force vector.
    """
    theory_sol = solve_theory(scene)
    kernel_out = run_kernel_point_set(scene, theory_sol.delta)
    theory_F = theory_sol.contact_force      # from compute_contact_force_point_set
    kernel_F = kernel_out.pad_force

    norm_theory = float(np.linalg.norm(theory_F))
    if norm_theory < 1.0e-15:
        rel = float(np.linalg.norm(kernel_F))
    else:
        rel = float(np.linalg.norm(theory_F - kernel_F) / norm_theory)

    # Sign-regression guard: per-pad-sphere the force projection onto
    # the pad's own outward normal must agree in sign.  Only checks
    # pads where the theory has |F . n| > 1e-9 (i.e. genuinely in
    # contact).  Same idea as the sphere-target compare().
    theory_F_n = np.einsum("ij,ij->i", theory_F, scene.outward_normals)
    kernel_F_n = np.einsum("ij,ij->i", kernel_F, scene.outward_normals)
    nontrivial = np.abs(theory_F_n) > 1.0e-9
    if nontrivial.any():
        sign_match = bool(np.all(
            np.sign(theory_F_n[nontrivial]) == np.sign(kernel_F_n[nontrivial])
        ))
    else:
        sign_match = True

    passed = (rel < tol) and sign_match
    if verbose:
        # Per-pad force norm distribution to give a sense of patch
        # engagement.  Re-uses ScenePassReport for output uniformity.
        n_active = int((np.linalg.norm(theory_F, axis=1) > 1.0e-9).sum())
        print(f"  {name:<40} ka={scene.ka:>7.0f} kc/ka={scene.kc/scene.ka:>5.2f}  "
              f"||F_k - F_t||/||F_t|| = {rel:.3e}  "
              f"sign={'OK' if sign_match else 'FAIL'}  "
              f"active={n_active}/{scene.n_spheres}  "
              f"{'PASS' if passed else 'FAIL'}")
    return ScenePassReport(
        name=name,
        kc_over_ka=scene.kc / scene.ka,
        delta_rel_err=rel,    # actually FORCE rel err under C1
        sign_match=sign_match,
        pass_tol=tol,
        pass_=passed,
        kernel_iters=0,       # no iteration in C1; one-shot kernel
        extra={
            "regime": theory_sol.solver_info.get("regime"),
            "theory_iters": theory_sol.solver_info.get("nit"),
            "n_active_pad": int((np.linalg.norm(theory_F, axis=1)
                                 > 1.0e-9).sum()),
        },
    )


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def _regression_n_sweep() -> bool:
    """Guard #3 (C2e): N_pad sweep on chain-vs-box at production eps.

    Asserts that the kernel-vs-theory force gap on the chain-vs-box
    scene stays bounded as the lattice grows.  N varies; the active
    contact patch over the box face stays roughly constant (~31
    spheres for the default geometry), so growth in error indicates
    spurious contributions from non-active spheres -- a real bug, not
    fp32 noise.  Bounded noise (rel_err scaling slower than ~sqrt(N))
    is the expected float32 floor and passes.

    Assertions per N:
      * rel_err <= 2e-2  (the handoff's upper-bound noise budget)
      * active count is within +/-5 of the cohort median (validates
        that the lattice's non-active spheres don't spuriously
        enter the active set at high N).

    Cohort assertion:
      * rel_err(N=800) <= 20 * rel_err(N=50) -- guards against
        superlinear N-scaling.  sqrt(800/50) = 4, so 20x is ~5x the
        sqrt-N floor; a real bug would blow far past this.

    Takes ~17s wall time at the three N values.
    """
    ka = 25_000.0
    kc = ka  # kc/ka = 1.0 baseline; not sweeping kc here, that's main()'s job
    N_sweep = [50, 200, 800]
    results: list[tuple[int, float, int]] = []  # (N, rel_err, n_active)
    for N in N_sweep:
        scene = scene_chain_vs_box_face_on(ka=ka, kc=kc, N=N, eps=5.0e-4)
        report = compare_point_set(
            f"N-sweep-N={N}", scene, tol=2.0e-2, verbose=False)
        # Re-derive active count via the same `>1e-9` rule
        # compare_point_set uses (kept here so we don't have to change
        # ScenePassReport just to surface it).
        theory_sol = solve_theory(scene)
        n_active = int(
            (np.linalg.norm(theory_sol.contact_force, axis=1) > 1.0e-9).sum()
        )
        results.append((N, report.delta_rel_err, n_active))
        print(
            f"  N={N:>4}  active={n_active:>3}/{N:<4}  "
            f"||F_k-F_t||/||F_t|| = {report.delta_rel_err:.3e}  "
            f"({'PASS' if report.pass_ else 'FAIL'})"
        )

    all_pass = True
    # Per-N upper bound.
    for N, rel, _ in results:
        if rel > 2.0e-2:
            print(f"  FAIL: N={N} rel_err {rel:.3e} > 2e-2 budget")
            all_pass = False

    # Active-count consistency.  Median +/- 5 catches a "non-active
    # sphere leaks into the active set" bug without false positives
    # from boundary jitter.
    actives = [n_active for _, _, n_active in results]
    median_active = sorted(actives)[len(actives) // 2]
    for N, _, n_active in results:
        if abs(n_active - median_active) > 5:
            print(
                f"  FAIL: N={N} active count {n_active} drifts >5 "
                f"from cohort median {median_active}"
            )
            all_pass = False

    # No superlinear N-scaling.  sqrt(N) growth is the float32 floor;
    # 20x ceiling between N=50 and N=800 leaves ~5x headroom above
    # sqrt(16) = 4x.
    rel_50  = next(rel for N, rel, _ in results if N == 50)
    rel_800 = next(rel for N, rel, _ in results if N == 800)
    if rel_50 > 0.0 and (rel_800 / rel_50) > 20.0:
        print(
            f"  FAIL: rel_err(N=800)/rel_err(N=50) = "
            f"{rel_800 / rel_50:.1f}x exceeds 20x sqrt-N ceiling -- "
            f"likely a real N-scaling bug, not fp32 noise"
        )
        all_pass = False

    if all_pass:
        print("  N-sweep regression: PASS")
    return all_pass


def main() -> int:
    print()
    print("=" * 88)
    print("Step 7 / Kernel-vs-theory bridge")
    print("=" * 88)

    # Constant-discipline guard (C2d Obs 3) -- run before any physics
    # scenes so a forgotten kernel literal update fails fast, with a
    # clear error pointing back at the Python/kernel split.
    print()
    print("── Constant-discipline guard ─────────────────────────────────")
    if not _check_constant_discipline():
        return 1

    # N_pad sweep regression (Guard #3, C2e).  Validates that the
    # kernel handles non-active lattice spheres correctly as N grows.
    print()
    print("── N-pad sweep regression (chain vs box, eps=5e-4) ───────────")
    if not _regression_n_sweep():
        return 1

    # Sweep kc/ka in {0.1, 1, 10} per the plan, plus a baseline kc=ka.
    ka = 25_000.0
    kc_factors = [0.1, 1.0, 10.0]
    reports: list[ScenePassReport] = []

    # ── Scene A: single sphere face-on ────────────────────────────────
    print()
    print("── A. single sphere, face-on ─────────────────────────────────")
    for f in kc_factors:
        reports.append(compare("A-single-face-on", scene_single_face_on(
            ka=ka, kc=f * ka)))

    # ── Scene B: single sphere off-axis (deformed contact direction) ──
    print()
    print("── B. single sphere, off-axis 30 deg ─────────────────────────")
    for f in kc_factors:
        reports.append(compare("B-single-off-axis-30", scene_single_off_axis(
            ka=ka, kc=f * ka)))

    # ── Scene C: chain single-contact ─────────────────────────────────
    print()
    print("── C. chain single-contact (small target, only centre) ──────")
    for f in kc_factors:
        reports.append(compare("C-chain-single-contact",
                               scene_chain_single_contact(ka=ka, kc=f * ka)))

    # ── Scene D: arc single-contact (curvature -> bulge geometry) ─────
    print()
    print("── D. arc single-contact (apex; perimeter bulge expected) ───")
    for f in kc_factors:
        reports.append(compare("D-arc-single-contact",
                               scene_arc_single_contact(ka=ka, kc=f * ka)))

    # ── Scene E: anisotropic off-axis ─────────────────────────────────
    print()
    print("── E. anisotropic anchor (ka_t = ka/3) off-axis ─────────────")
    for f in kc_factors:
        reports.append(compare("E-aniso-off-axis-5deg",
                               scene_anisotropic_off_axis(ka=ka, kc=f * ka)))

    # ── Scene F: single sphere + tangential load, stick mode ──────────
    print()
    print("── F. single sphere + tangential load, STICK mode ───────────")
    for f in kc_factors:
        reports.append(compare("F-single-friction-stick",
                               scene_single_friction_stick(ka=ka, kc=f * ka)))

    # ── Scene G: single sphere + tangential load, slip mode ───────────
    # Relaxed tolerance (2%): slip mode requires δ_t > F_thresh/ka
    # ≈ 0.5 mm at production scales, which produces a 1-2% geometric
    # tilt between the kernel's deformed-direction contact normal and
    # the theory's rest-direction analytical solver.  The kernel is
    # the more physically accurate of the two (the rest-direction
    # assumption breaks at finite δ); the looser tolerance captures
    # the known geometric divergence + the smooth surrogate's
    # transition zone.  See cslc_main/theory/notes.md step 4
    # "smoothing-zone observation" for the friction surrogate width.
    print()
    print("── G. single sphere + tangential load, SLIP mode (tol=2e-2) ─")
    for f in kc_factors:
        reports.append(compare("G-single-friction-slip",
                               scene_single_friction_slip(ka=ka, kc=f * ka),
                               tol=2.0e-2))

    # ── Scene H: single pad sphere vs sampled box (Step 10 / C1) ──────
    # Verifies the new compute_pad_force_vs_point_set kernel matches
    # the theory's compute_contact_force_point_set element-wise at the
    # theory's converged delta.  Float32 precision floor is ~1e-6 rel,
    # so 5e-6 is a comfortable tolerance.
    print()
    print("── H. single pad sphere vs box (point-set kernel, tol=5e-6) ──")
    for f in kc_factors:
        reports.append(compare_point_set(
            "H-single-vs-box-face-on",
            scene_single_vs_box_face_on(ka=ka, kc=f * ka),
            tol=5.0e-6))

    # ── Scene I: chain (multi-pad) vs sampled box (Step 10 / C1) ──────
    print()
    print("── I. chain vs box (point-set kernel, tol=5e-6) ───────────────")
    for f in kc_factors:
        reports.append(compare_point_set(
            "I-chain-vs-box-face-on",
            scene_chain_vs_box_face_on(ka=ka, kc=f * ka),
            tol=5.0e-6))

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 88)
    n_pass = sum(1 for r in reports if r.pass_)
    n_total = len(reports)
    unique_tols = sorted({r.pass_tol for r in reports})
    tol_str = ", ".join(f"{t:g}" for t in unique_tols)
    print(f"SUMMARY: {n_pass}/{n_total} scenes pass at per-scene tol in {{{tol_str}}}")
    print()
    print(f"  {'scene':<26} {'kc/ka':>6} {'rel err':>12} {'sign':>6} {'iters':>7} verdict")
    print(f"  {'-'*26} {'-'*6} {'-'*12} {'-'*6} {'-'*7} -------")
    for r in reports:
        print(f"  {r.name:<26} {r.kc_over_ka:>6.2f} {r.delta_rel_err:>12.3e} "
              f"{'OK' if r.sign_match else 'FAIL':>6} {r.kernel_iters:>7d} "
              f"{'PASS' if r.pass_ else 'FAIL'}")
    print("=" * 88)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
