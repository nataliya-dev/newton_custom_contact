#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
CSLC Squeeze Test — Comparison of contact models and solvers.

Scene: Two kinematic box pads squeeze a dynamic sphere under gravity.
       The pads start already gripping the sphere (initial penetration),
       then gently squeeze further. Gravity pulls the sphere down;
       friction from the contacts must hold it.

         ┌─────┐         ┌─────┐
         │     │  ←───   │     │
         │ left│  sphere  │right│       ↓ gravity (-z)
         │ pad │   (●)   │ pad │
         │     │  ───→   │     │
         └─────┘         └─────┘
         body 0          body 1   body 2 (dynamic)

Key physics:
  - Point contact: 2 contact points → low aggregate friction stiffness
  - CSLC:          ~100+ contact points → high aggregate friction → holds object

Friction stability (semi-implicit integrator):
  The solver uses regularized Coulomb friction: ft = min(kf·|vt|, μ·|fn|).
  The viscous term (kf·|vt|) acts like a tangential spring, which requires:
      n_contacts × kf × dt / mass < 2    (stability criterion)
  Default kf=1000 is UNSTABLE for this scene. We set kf=3.0 so that:
      point (2 contacts):   2 × 3 × 0.002 / 0.5 = 0.024  ✓
      CSLC  (128 contacts): 128 × 3 × 0.002 / 0.5 = 1.54  ✓

Usage:
  python squeeze_test.py                       # run all configs
  python squeeze_test.py --configs cslc_semi   # run one config
  python squeeze_test.py --list                # list available configs
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import warp as wp
import newton
from newton.solvers import SolverSemiImplicit

# Optional solvers — graceful fallback if not installed
try:
    from newton.solvers import SolverMuJoCo
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    warnings.warn("SolverMuJoCo not available — skipping MuJoCo configs.")

try:
    from newton.solvers import SolverXPBD
    HAS_XPBD = True
except ImportError:
    HAS_XPBD = False
    warnings.warn("SolverXPBD not available — skipping XPBD configs.")


# ═══════════════════════════════════════════════════════════════════════════
#  Scene parameters
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SceneParams:
    """All physical parameters in one place for fair comparison.

    Friction stability note:
        kf (friction stiffness) must satisfy  n_contacts × kf × dt / mass < 2
        for the semi-implicit integrator.  With dt=0.002, mass=0.5, and up to
        ~128 CSLC contacts, kf must be < ~3.9.  We use kf=3.0.
    """
    # Sphere (dynamic object)
    sphere_radius: float = 0.03          # 30 mm
    sphere_density: float = 4421.0       # kg/m³  → ~0.5 kg for r=30mm
    sphere_start_z: float = 0.15         # initial height [m]

    # Boxes (kinematic pads)
    pad_hx: float = 0.04                 # half-extent in grip (x) direction [m]
    pad_hy: float = 0.04                 # half-extent in y [m]
    pad_hz: float = 0.15                 # half-extent in z [m] — tall so sphere can't fall out

    # Squeeze kinematics
    #   Initial gap < sphere diameter → sphere starts already gripped.
    #   gap=0.04 with sphere_diameter=0.06 → 10mm penetration per side.
    pad_gap_initial: float = 0.04        # distance between pad inner faces [m]
    pad_squeeze_speed: float = 0.005     # m/s inward per pad (gentle)
    pad_squeeze_duration: float = 0.5    # seconds of squeezing
    pad_hold_duration: float = 1.5       # seconds holding after squeeze

    # Contact material (shared across ALL contact models for fair comparison)
    ke: float = 5.0e4                    # elastic stiffness [N/m]
    kd: float = 5.0e2                    # normal damping [N·s/m]
    kf: float = 3.0                      # friction stiffness [N·s/m] — LOW for stability
    mu: float = 0.5                      # friction coefficient

    # CSLC-specific
    cslc_spacing: float = 0.005          # lattice spacing [m] (5 mm)
    cslc_ka: float = 5000.0              # anchor stiffness [N/m]
    cslc_kl: float = 500.0               # lateral stiffness [N/m]
    cslc_dc: float = 2.0                 # Hunt-Crossley damping [s/m]
    cslc_n_iter: int = 40                # Jacobi iterations
    cslc_alpha: float = 0.3              # under-relaxation

    # Hydroelastic-specific
    kh: float = 1.0e10                   # hydroelastic pressure modulus
    sdf_resolution: int = 64             # SDF voxel resolution

    # Simulation
    dt: float = 1.0 / 500.0             # 500 Hz
    gravity: tuple = (0.0, 0.0, -9.81)

    @property
    def sphere_mass(self) -> float:
        vol = (4.0 / 3.0) * math.pi * self.sphere_radius ** 3
        return self.sphere_density * vol

    @property
    def penetration_per_side(self) -> float:
        return max(self.sphere_radius - self.pad_gap_initial / 2.0, 0.0)

    @property
    def n_squeeze_steps(self) -> int:
        return int(self.pad_squeeze_duration / self.dt)

    @property
    def n_hold_steps(self) -> int:
        return int(self.pad_hold_duration / self.dt)

    @property
    def n_total_steps(self) -> int:
        return self.n_squeeze_steps + self.n_hold_steps


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics collector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    """Time-series metrics collected during simulation."""
    name: str = ""
    sphere_z: list[float] = field(default_factory=list)
    active_contacts: list[int] = field(default_factory=list)
    normal_force_mean: list[float] = field(default_factory=list)
    normal_force_std: list[float] = field(default_factory=list)
    step_times_ms: list[float] = field(default_factory=list)
    gradient_norm: float | None = None

    @property
    def z_drop_mm(self) -> float:
        if len(self.sphere_z) < 2:
            return 0.0
        z0 = self.sphere_z[0]
        z_min = min(self.sphere_z)
        return (z0 - z_min) * 1000.0

    @property
    def avg_step_ms(self) -> float:
        return sum(self.step_times_ms) / len(self.step_times_ms) if self.step_times_ms else 0.0

    @property
    def peak_contacts(self) -> int:
        return max(self.active_contacts) if self.active_contacts else 0

    def summary(self) -> str:
        lines = [
            f"{'─' * 50}",
            f"  Config: {self.name}",
            f"  Z-drop:          {self.z_drop_mm:8.3f} mm",
            f"  Peak contacts:   {self.peak_contacts:8d}",
            f"  Avg step time:   {self.avg_step_ms:8.3f} ms",
        ]
        if self.gradient_norm is not None:
            lines.append(f"  Gradient norm:   {self.gradient_norm:8.4e}")
        lines.append(f"{'─' * 50}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Scene builders — BODY LAYOUT: 0=left pad, 1=right pad, 2=sphere
# ═══════════════════════════════════════════════════════════════════════════


def _pad_cfg(p: SceneParams, **overrides) -> newton.ModelBuilder.ShapeConfig:
    """Shape config for kinematic pads (density=0: no mass on kinematic body)."""
    kwargs = dict(ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=0.0)
    kwargs.update(overrides)
    return newton.ModelBuilder.ShapeConfig(**kwargs)


def _sphere_cfg(p: SceneParams) -> newton.ModelBuilder.ShapeConfig:
    """Shape config for the dynamic sphere."""
    return newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.sphere_density,
    )


def _add_pads_and_sphere(
    b: newton.ModelBuilder,
    p: SceneParams,
    pad_cfg: newton.ModelBuilder.ShapeConfig,
    sphere_cfg: newton.ModelBuilder.ShapeConfig,
) -> tuple[int, int, int]:
    """Add the standard two-pad + sphere layout.

    Returns (left_body, right_body, sphere_body) indices.
    """
    left_x = -(p.pad_gap_initial / 2.0 + p.pad_hx)
    right_x =  (p.pad_gap_initial / 2.0 + p.pad_hx)

    left = b.add_body(
        xform=wp.transform((left_x, 0.0, p.sphere_start_z), wp.quat_identity()),
        is_kinematic=True,
        label="left_pad",
    )
    b.add_shape_box(left, hx=p.pad_hx, hy=p.pad_hy, hz=p.pad_hz, cfg=pad_cfg)

    right = b.add_body(
        xform=wp.transform((right_x, 0.0, p.sphere_start_z), wp.quat_identity()),
        is_kinematic=True,
        label="right_pad",
    )
    b.add_shape_box(right, hx=p.pad_hx, hy=p.pad_hy, hz=p.pad_hz, cfg=pad_cfg)

    sphere = b.add_body(
        xform=wp.transform((0.0, 0.0, p.sphere_start_z), wp.quat_identity()),
        label="sphere",
    )
    b.add_shape_sphere(sphere, radius=p.sphere_radius, cfg=sphere_cfg)

    return left, right, sphere


def build_point_contact_scene(p: SceneParams) -> newton.Model:
    """Standard point-contact: box primitives + sphere primitive."""
    b = newton.ModelBuilder()
    _add_pads_and_sphere(b, p, _pad_cfg(p), _sphere_cfg(p))
    model = b.finalize()
    model.set_gravity(p.gravity)
    return model


def build_cslc_scene(p: SceneParams) -> newton.Model:
    """CSLC contact: box pads with CSLC lattice vs sphere."""
    b = newton.ModelBuilder()
    cslc_pad = _pad_cfg(
        p,
        is_cslc=True,
        cslc_spacing=p.cslc_spacing,
        cslc_ka=p.cslc_ka,
        cslc_kl=p.cslc_kl,
        cslc_dc=p.cslc_dc,
        cslc_n_iter=p.cslc_n_iter,
        cslc_alpha=p.cslc_alpha,
    )
    _add_pads_and_sphere(b, p, cslc_pad, _sphere_cfg(p))
    model = b.finalize()
    model.set_gravity(p.gravity)
    return model


def build_hydroelastic_scene(p: SceneParams) -> newton.Model:
    """Hydroelastic: both shapes need HYDROELASTIC flag + SDF."""
    b = newton.ModelBuilder()
    hydro_pad = _pad_cfg(
        p,
        kh=p.kh,
        is_hydroelastic=True,
        sdf_max_resolution=p.sdf_resolution,
    )
    hydro_sphere = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002,
        density=p.sphere_density,
        kh=p.kh,
        is_hydroelastic=True,
        sdf_max_resolution=p.sdf_resolution,
    )
    _add_pads_and_sphere(b, p, hydro_pad, hydro_sphere)
    model = b.finalize()
    model.set_gravity(p.gravity)
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Kinematic pad controller
# ═══════════════════════════════════════════════════════════════════════════


def compute_pad_x(step: int, p: SceneParams, sign: float) -> float:
    """Compute x-position of a pad at a given timestep.

    sign = -1 for left pad, +1 for right pad.
    Phase 1 (squeeze): pads move inward at constant speed.
    Phase 2 (hold):    pads stay at final squeezed position.
    """
    initial_x = sign * (p.pad_gap_initial / 2.0 + p.pad_hx)
    t = step * p.dt
    t_squeeze = min(t, p.pad_squeeze_duration)
    displacement = p.pad_squeeze_speed * t_squeeze
    return initial_x - sign * displacement


def set_kinematic_pads(state: Any, step: int, p: SceneParams) -> None:
    """Overwrite body_q for the two kinematic pads (bodies 0 and 1).

    body_q is wp.array(dtype=wp.transform) → numpy shape (N, 7)
    layout per row: [px, py, pz, qx, qy, qz, qw]
    """
    q = state.body_q.numpy()

    left_x  = compute_pad_x(step, p, sign=-1.0)
    right_x = compute_pad_x(step, p, sign=+1.0)

    q[0, 0] = left_x    # body 0 (left pad), px
    q[1, 0] = right_x   # body 1 (right pad), px

    state.body_q.assign(wp.array(q, dtype=wp.transform, device=state.body_q.device))

    # Zero out pad velocities — body_qd is (N, 6) spatial vectors
    qd = state.body_qd.numpy()
    qd[0] = 0.0
    qd[1] = 0.0
    state.body_qd.assign(wp.array(qd, dtype=wp.spatial_vector, device=state.body_qd.device))


# ═══════════════════════════════════════════════════════════════════════════
#  Contact metrics extraction
# ═══════════════════════════════════════════════════════════════════════════


def extract_contact_metrics(contacts: Any) -> tuple[int, float, float]:
    """Extract active contact count and normal force statistics.

    Returns (n_active, force_mean, force_std).
    """
    count_arr = contacts.rigid_contact_count.numpy()
    n_contacts = int(count_arr[0]) if len(count_arr) > 0 else 0
    if n_contacts == 0:
        return 0, 0.0, 0.0

    shape0 = contacts.rigid_contact_shape0.numpy()[:n_contacts]
    margin0 = contacts.rigid_contact_margin0.numpy()[:n_contacts]
    margin1 = contacts.rigid_contact_margin1.numpy()[:n_contacts]

    valid_mask = shape0 >= 0
    n_active = int(np.sum(valid_mask))
    if n_active == 0:
        return 0, 0.0, 0.0

    margins = margin0[valid_mask] + margin1[valid_mask]
    return n_active, float(np.mean(margins)), float(np.std(margins))


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation runner
# ═══════════════════════════════════════════════════════════════════════════


def run_config(name: str, model: newton.Model, solver: Any, p: SceneParams) -> Metrics:
    """Run a single configuration and collect metrics."""
    print(f"\n{'═' * 60}")
    print(f"  Running: {name}")
    print(f"  Sphere mass: {p.sphere_mass:.3f} kg, gravity force: {p.sphere_mass * 9.81:.2f} N")
    print(f"  Initial penetration/side: {p.penetration_per_side * 1000:.1f} mm")
    print(f"{'═' * 60}")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    metrics = Metrics(name=name)

    # Warm-up: JIT compile + allocate (use a throwaway state so warm-start is clean)
    warmup_s0 = model.state()
    warmup_s1 = model.state()
    for _ in range(3):
        set_kinematic_pads(warmup_s0, 0, p)
        warmup_s0.clear_forces()
        model.collide(warmup_s0, contacts)
        solver.step(warmup_s0, warmup_s1, control, contacts, p.dt)
        warmup_s0, warmup_s1 = warmup_s1, warmup_s0
    del warmup_s0, warmup_s1

    # Fresh start for actual test
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()

    for step in range(p.n_total_steps):
        set_kinematic_pads(state_0, step, p)
        state_0.clear_forces()

        t0 = time.perf_counter()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, p.dt)
        wp.synchronize()
        t1 = time.perf_counter()

        # ── Diagnostics on first step ──
        if step == 0:
            count = int(contacts.rigid_contact_count.numpy()[0])
            print(f"  [DIAG] step 0: contacts={count}")
            if contacts.rigid_contact_stiffness is not None:
                stiff = contacts.rigid_contact_stiffness.numpy()[:count]
                nonzero = stiff[stiff > 0]
                if len(nonzero) > 0:
                    print(f"  [DIAG] per-contact stiffness: min={nonzero.min():.1f} max={nonzero.max():.1f}")
            # Check sphere velocity after first step
            qd = state_1.body_qd.numpy()
            vz = float(qd[2, 2])  # body 2 z-velocity
            print(f"  [DIAG] sphere vz after step 0: {vz:.4f} m/s")

        q = state_1.body_q.numpy()
        sphere_z = float(q[2, 2])  # body 2 = sphere, pz
        metrics.sphere_z.append(sphere_z)

        n_active, f_mean, f_std = extract_contact_metrics(contacts)
        metrics.active_contacts.append(n_active)
        metrics.normal_force_mean.append(f_mean)
        metrics.normal_force_std.append(f_std)
        metrics.step_times_ms.append((t1 - t0) * 1000.0)

        state_0, state_1 = state_1, state_0

        if (step + 1) % 200 == 0 or step == p.n_total_steps - 1:
            phase = "squeeze" if step < p.n_squeeze_steps else "hold"
            print(
                f"  step {step + 1:5d}/{p.n_total_steps}  "
                f"[{phase:7s}]  "
                f"z={sphere_z:+.4f}  "
                f"contacts={n_active:4d}  "
                f"dt={metrics.step_times_ms[-1]:.2f}ms"
            )

    return metrics


def run_gradient_test(name: str, model: newton.Model, p: SceneParams) -> float | None:
    """Forward+backward pass to measure gradient magnitude."""
    try:
        solver = SolverSemiImplicit(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        mid_step = p.n_squeeze_steps // 2
        set_kinematic_pads(state_0, mid_step, p)

        # Gradient test needs a scalar loss
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=model.device)

        tape = wp.Tape()
        with tape:
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, p.dt)

            # Scalar loss: z-position of sphere (body 2)
            wp.launch(
                kernel=_scalar_loss_kernel,
                dim=1,
                inputs=[state_1.body_q, 2],
                outputs=[loss],
                device=model.device,
            )

        tape.backward(loss=loss)
        grad = tape.gradients.get(state_0.body_q)
        if grad is not None:
            norm = float(np.linalg.norm(grad.numpy()))
            print(f"  Gradient norm ({name}): {norm:.4e}")
            return norm
        print(f"  Gradient not available for {name}")
        return None
    except Exception as e:
        print(f"  Gradient test failed for {name}: {e}")
        return None


@wp.kernel
def _scalar_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_index: int,
    loss: wp.array(dtype=wp.float32),
):
    """Extract z-position of a body as a scalar loss for gradient testing."""
    q = body_q[body_index]
    pos = wp.transform_get_translation(q)
    loss[0] = pos[2]


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration registry
# ═══════════════════════════════════════════════════════════════════════════


def get_configs(p: SceneParams) -> dict[str, dict]:
    configs = {}

    configs["point_semi"] = dict(
        build=lambda: build_point_contact_scene(p),
        solver_cls=SolverSemiImplicit, solver_kwargs={},
        label="Point Contact + SemiImplicit",
    )
    if HAS_MUJOCO:
        configs["point_mujoco"] = dict(
            build=lambda: build_point_contact_scene(p),
            solver_cls=SolverMuJoCo,
            solver_kwargs=dict(iterations=20, ls_iterations=10, solver="cg", integrator="implicitfast"),
            label="Point Contact + MuJoCo",
        )
    if HAS_XPBD:
        configs["point_xpbd"] = dict(
            build=lambda: build_point_contact_scene(p),
            solver_cls=SolverXPBD,
            solver_kwargs=dict(rigid_contact_relaxation=0.8, rigid_contact_con_weighting=True),
            label="Point Contact + XPBD",
        )
    configs["hydro_semi"] = dict(
        build=lambda: build_hydroelastic_scene(p),
        solver_cls=SolverSemiImplicit, solver_kwargs={},
        label="Hydroelastic + SemiImplicit",
    )
    if HAS_MUJOCO:
        configs["hydro_mujoco"] = dict(
            build=lambda: build_hydroelastic_scene(p),
            solver_cls=SolverMuJoCo,
            solver_kwargs=dict(iterations=20, ls_iterations=10, solver="cg", integrator="implicitfast"),
            label="Hydroelastic + MuJoCo",
        )
    configs["cslc_semi"] = dict(
        build=lambda: build_cslc_scene(p),
        solver_cls=SolverSemiImplicit, solver_kwargs={},
        label="CSLC + SemiImplicit",
    )
    if HAS_MUJOCO:
        configs["cslc_mujoco"] = dict(
            build=lambda: build_cslc_scene(p),
            solver_cls=SolverMuJoCo,
            solver_kwargs=dict(iterations=20, ls_iterations=10, solver="cg", integrator="implicitfast"),
            label="CSLC + MuJoCo",
        )
    return configs


# ═══════════════════════════════════════════════════════════════════════════
#  Results reporting
# ═══════════════════════════════════════════════════════════════════════════


def save_results_csv(all_metrics: list[Metrics], path: str = "squeeze_results.csv"):
    with open(path, "w") as f:
        f.write("config,z_drop_mm,peak_contacts,avg_step_ms,gradient_norm\n")
        for m in all_metrics:
            g = f"{m.gradient_norm:.4e}" if m.gradient_norm is not None else "N/A"
            f.write(f"{m.name},{m.z_drop_mm:.4f},{m.peak_contacts},{m.avg_step_ms:.4f},{g}\n")
    print(f"\nResults saved to {path}")


def save_timeseries_csv(all_metrics: list[Metrics], p: SceneParams, path: str = "squeeze_timeseries.csv"):
    with open(path, "w") as f:
        header = "step,time_s," + ",".join(f"z_{m.name}" for m in all_metrics)
        f.write(header + "\n")
        n_steps = max(len(m.sphere_z) for m in all_metrics) if all_metrics else 0
        for i in range(n_steps):
            vals = [f"{m.sphere_z[i]:.6f}" if i < len(m.sphere_z) else "" for m in all_metrics]
            f.write(f"{i},{i * p.dt:.6f},{','.join(vals)}\n")
    print(f"Timeseries saved to {path}")


def print_comparison_table(all_metrics: list[Metrics]):
    print(f"\n{'═' * 72}")
    print(f"  SQUEEZE TEST COMPARISON")
    print(f"{'═' * 72}")
    print(f"  {'Config':<30s} {'Z-drop':>10s} {'Contacts':>10s} {'Step':>10s} {'Grad':>12s}")
    print(f"  {'':30s} {'(mm)':>10s} {'(peak)':>10s} {'(ms)':>10s} {'norm':>12s}")
    for m in all_metrics:
        g = f"{m.gradient_norm:.2e}" if m.gradient_norm is not None else "—"
        print(f"  {m.name:<30s} {m.z_drop_mm:10.3f} {m.peak_contacts:10d} {m.avg_step_ms:10.3f} {g:>12s}")
    print(f"{'═' * 72}")
    if len(all_metrics) >= 2:
        print(f"\n  Best grip:       {min(all_metrics, key=lambda m: m.z_drop_mm).name}")
        print(f"  Fastest:         {min(all_metrics, key=lambda m: m.avg_step_ms).name}")
        print(f"  Largest patch:   {max(all_metrics, key=lambda m: m.peak_contacts).name}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="CSLC Squeeze Test")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-gradient", action="store_true")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    p = SceneParams()
    if args.dt is not None:
        p.dt = args.dt

    configs = get_configs(p)

    if args.list:
        print("Available configurations:")
        for key, cfg in configs.items():
            print(f"  {key:<20s}  {cfg['label']}")
        return

    selected = args.configs if args.configs is not None else list(configs.keys())
    for key in selected:
        if key not in configs:
            print(f"Unknown config: {key}.  Available: {', '.join(configs.keys())}")
            return

    all_metrics: list[Metrics] = []
    for key in selected:
        cfg = configs[key]
        try:
            model = cfg["build"]()
            solver = cfg["solver_cls"](model, **cfg["solver_kwargs"])
            metrics = run_config(name=key, model=model, solver=solver, p=p)
            if not args.no_gradient and cfg["solver_cls"] == SolverSemiImplicit:
                metrics.gradient_norm = run_gradient_test(key, cfg["build"](), p)
            all_metrics.append(metrics)
            print(metrics.summary())
        except Exception as e:
            print(f"\n  ERROR in {key}: {e}")
            import traceback; traceback.print_exc()

    if all_metrics:
        print_comparison_table(all_metrics)
        save_results_csv(all_metrics)
        save_timeseries_csv(all_metrics, p)


if __name__ == "__main__":
    main()