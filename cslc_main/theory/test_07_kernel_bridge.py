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

from dataclasses import dataclass

import numpy as np
import warp as wp

from cslc_main.theory.kernel_bridge import (
    KernelScene,
    KernelSolution,
    compute_contact_force,
    solve_theory,
)
from newton._src.geometry.cslc_kernels import (
    compute_cslc_penetration,
    jacobi_step,
)


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


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    print()
    print("=" * 88)
    print("Step 7 / Kernel-vs-theory bridge")
    print("=" * 88)

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
