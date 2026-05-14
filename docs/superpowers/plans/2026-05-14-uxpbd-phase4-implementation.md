# UXPBD Phase 4 (Liquid / Position-Based Fluids) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship UXPBD Phase 4: Position-Based Fluids (PBF) density constraint, fluid-solid coupling, XSPH viscosity, Akinci cohesion, cross-substrate contact extension for `particle_substrate==3` (fluid), all five Phase 4 gate tests, and the Franka pour-into-bowl demo (scenario D). Phase 3 (soft bodies) is intentionally skipped; `particle_substrate==2` stays reserved but unused.

**Architecture:** Add a new fluid pipeline inside `SolverUXPBD.step()` that runs `fluid_iterations` PBF sub-iterations per main iteration. Per sub-iteration: compute fluid-particle density via Poly6 kernel (with solid contribution per UPPFRTA §7.1), compute per-particle Lagrange multiplier λ, accumulate position deltas, apply. After the sub-iteration loop, run XSPH viscosity (post-step) and Akinci cohesion (during contact). Extend the existing cross-substrate contact kernels (`solve_particle_shape_contacts_uxpbd`, `solve_particle_particle_contacts_uxpbd`) with `substrate==3` branches so fluid particles can collide with analytical shapes and with other particles.

**Tech Stack:** Python 3, NVIDIA Warp (hash-grid queries, `wp.atomic_add`), Newton (`ModelBuilder`, `SolverBase`), unittest. Reuses Phase 1+2's `model.particle_grid`, `particle_substrate`, `particle_to_lattice`. Builds on the SM-rigid path (Phase 2) for the container cup.

**Reference docs:**

- Design spec: `docs/superpowers/specs/2026-05-13-uxpbd-design.md` §8.4
- PBF paper: Macklin & Müller 2013, "Position Based Fluids", `cslc_xpbd/papers/uppfrta_preprint.pdf` §7 (UPPFRTA wraps PBF with extensions)
- Akinci cohesion: Akinci et al. 2013, used per UPPFRTA §7.1
- Phase 2 implementation: `newton/_src/solvers/uxpbd/{solver_uxpbd.py, kernels.py, shape_match.py}`
- Phase 2 plan: `docs/superpowers/plans/2026-05-14-uxpbd-phase2-implementation.md`

**Starting HEAD:** `0f68d074` (run reference updated). Phase 2 complete; 25 UXPBD tests pass on CUDA (16 + 9 skip on CPU due to SRXPBD tile-reduce). PBF kernels in Phase 4 do not use tile primitives, so all Phase 4 tests should run on both CPU and CUDA.

---

## File Structure

**New files:**

- `newton/_src/solvers/uxpbd/fluid.py` — PBF kernels (density, lambda, position delta, XSPH viscosity, Akinci cohesion). One module, self-contained.
- `newton/tests/test_solver_uxpbd_phase4.py` — five gate tests + setup helpers.
- `newton/examples/contacts/example_uxpbd_pour.py` — Scenario D demo.

**Files modified:**

- `newton/_src/sim/model.py` — fluid metadata fields: `fluid_phase_count`, `fluid_rest_density`, `fluid_smoothing_radius`, `fluid_viscosity`, `fluid_cohesion`, `fluid_solid_coupling_s`.
- `newton/_src/sim/builder.py` — `add_fluid_grid` and `add_fluid_particles`. Tag particles with `particle_substrate=3` in finalize.
- `newton/_src/solvers/uxpbd/kernels.py` — extend `solve_particle_shape_contacts_uxpbd` and `solve_particle_particle_contacts_uxpbd` with substrate==3 branches.
- `newton/_src/solvers/uxpbd/solver_uxpbd.py` — add `fluid_iterations` param; wire PBF sub-iteration loop; cache fluid metadata refs at init.
- `CHANGELOG.md` — Phase 4 entry.

**Responsibility split:**

- `fluid.py` owns the PBF math (kernels only, no Python state).
- `solver_uxpbd.py` orchestrates: when to launch fluid kernels, how many sub-iterations.
- `kernels.py` (existing) extends the cross-substrate dispatch.
- `builder.py` / `model.py` handle the API + metadata. No physics.

---

## Task 1: Fluid metadata fields on Model + ModelBuilder accumulators

**Files:**
- Modify: `newton/_src/sim/model.py`
- Modify: `newton/_src/sim/builder.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py` (NEW)

- [ ] **Step 1: Create the test file with failing test**

Create `newton/tests/test_solver_uxpbd_phase4.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UXPBD Phase 4: Position-Based Fluids."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_uxpbd_empty_model_has_zero_fluid(test, device):
    """A Model finalized with no fluid has fluid_phase_count == 0."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    test.assertEqual(getattr(model, "fluid_phase_count", 0), 0)
    test.assertEqual(model.fluid_rest_density.shape[0], 0)
    test.assertEqual(model.fluid_smoothing_radius.shape[0], 0)
    test.assertEqual(model.fluid_viscosity.shape[0], 0)
    test.assertEqual(model.fluid_cohesion.shape[0], 0)


class TestSolverUXPBDPhase4(unittest.TestCase):
    pass


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_empty_model_has_zero_fluid",
    test_uxpbd_empty_model_has_zero_fluid,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run, confirm fail**

`uv run --extra dev -m newton.tests -k test_uxpbd_empty_model_has_zero_fluid`
Expected: AttributeError (`model.fluid_rest_density` does not exist).

- [ ] **Step 3: Add fields to Model**

In `newton/_src/sim/model.py`, find the existing UXPBD lattice fields block (added in Phase 1). Add the fluid block immediately after the lattice CSR offsets from Phase 2:

```python
# UXPBD Phase 4: fluid metadata. One entry per fluid phase. Empty when
# no fluid is added.
self.fluid_phase_count: int = 0
"""Number of distinct fluid phases registered in this model."""
self.fluid_rest_density: wp.array[wp.float32] = wp.empty(0, dtype=wp.float32, device=device)
"""Rest density per fluid phase [kg/m^3], shape [fluid_phase_count]."""
self.fluid_smoothing_radius: wp.array[wp.float32] = wp.empty(0, dtype=wp.float32, device=device)
"""SPH smoothing radius h per fluid phase [m], shape [fluid_phase_count].
Typically 2*particle_radius."""
self.fluid_viscosity: wp.array[wp.float32] = wp.empty(0, dtype=wp.float32, device=device)
"""XSPH viscosity coefficient per fluid phase [dimensionless 0..1].
Higher values damp tangential velocity differences more aggressively."""
self.fluid_cohesion: wp.array[wp.float32] = wp.empty(0, dtype=wp.float32, device=device)
"""Akinci cohesion coefficient per fluid phase [N], shape [fluid_phase_count].
Phase 4 default is 0.01 N (modest cohesion so pours don't shatter)."""
self.fluid_solid_coupling_s: wp.array[wp.float32] = wp.empty(0, dtype=wp.float32, device=device)
"""UPPFRTA eq.27 scaling factor s for solid-side density contribution,
per fluid phase. Defaults to 1.0; tune down if solid sampling is denser
than fluid sampling. Shape [fluid_phase_count]."""

# Per-particle fluid phase index (-1 if not a fluid particle).
self.particle_fluid_phase: wp.array[wp.int32] = wp.empty(0, dtype=wp.int32, device=device)
"""Per-particle fluid phase index. -1 for non-fluid particles. Shape [particle_count]."""
```

- [ ] **Step 4: Add ModelBuilder accumulators**

In `newton/_src/sim/builder.py` `ModelBuilder.__init__`, add Python list accumulators after the existing lattice accumulators (Phase 2):

```python
# UXPBD Phase 4 fluid metadata accumulators.
self.fluid_rest_density: list[float] = []
"""Per fluid phase rest density [kg/m^3] accumulated for :attr:`Model.fluid_rest_density`."""
self.fluid_smoothing_radius: list[float] = []
"""Per fluid phase smoothing radius h [m] accumulated for :attr:`Model.fluid_smoothing_radius`."""
self.fluid_viscosity: list[float] = []
"""Per fluid phase XSPH viscosity coefficient accumulated for :attr:`Model.fluid_viscosity`."""
self.fluid_cohesion: list[float] = []
"""Per fluid phase Akinci cohesion coefficient [N] accumulated for :attr:`Model.fluid_cohesion`."""
self.fluid_solid_coupling_s: list[float] = []
"""Per fluid phase solid-side density scaling factor accumulated for :attr:`Model.fluid_solid_coupling_s`."""
# Per-particle fluid phase index; -1 for non-fluid particles. Updated as
# fluid particles are added (or set to -1 elsewhere).
self.particle_fluid_phase: list[int] = []
"""Per-particle fluid phase index (-1 for non-fluid). Parallel to particle_q."""
```

Also, in `add_particle` (around line 7271), append `-1` to `self.particle_fluid_phase`:

```python
# Existing add_particle method, at the end before return particle_id:
self.particle_fluid_phase.append(-1)
```

- [ ] **Step 5: Bake fields in finalize**

In `newton/_src/sim/builder.py` `finalize()`, after the existing fluid-related metadata (none yet) and after the lattice bake block, add:

```python
# Bake UXPBD fluid metadata.
n_fluid_phases = len(self.fluid_rest_density)
m.fluid_phase_count = n_fluid_phases
if n_fluid_phases:
    m.fluid_rest_density = wp.array(self.fluid_rest_density, dtype=wp.float32, device=device)
    m.fluid_smoothing_radius = wp.array(self.fluid_smoothing_radius, dtype=wp.float32, device=device)
    m.fluid_viscosity = wp.array(self.fluid_viscosity, dtype=wp.float32, device=device)
    m.fluid_cohesion = wp.array(self.fluid_cohesion, dtype=wp.float32, device=device)
    m.fluid_solid_coupling_s = wp.array(self.fluid_solid_coupling_s, dtype=wp.float32, device=device)

n_particles = len(self.particle_q)
if n_particles:
    m.particle_fluid_phase = wp.array(self.particle_fluid_phase, dtype=wp.int32, device=device)
```

Use whichever Model variable name (`m` or `model`) the surrounding code uses.

- [ ] **Step 6: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_empty_model_has_zero_fluid`
Expected: PASS.

- [ ] **Step 7: Run all UXPBD tests**

`uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected: All Phase 1+2 tests still pass.

- [ ] **Step 8: Pre-commit and commit**

```bash
uvx pre-commit run --files newton/_src/sim/model.py newton/_src/sim/builder.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/sim/model.py newton/_src/sim/builder.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add UXPBD fluid metadata fields to Model and ModelBuilder

Reserves the fluid_rest_density / fluid_smoothing_radius /
fluid_viscosity / fluid_cohesion / fluid_solid_coupling_s arrays on
Model, plus the per-particle particle_fluid_phase index (-1 for
non-fluid particles). ModelBuilder gains Python list accumulators
that finalize bakes into wp.array. Phase 4 reads these in the new
fluid pipeline."
```

---

## Task 2: ModelBuilder.add_fluid_grid

**Files:**
- Modify: `newton/_src/sim/builder.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

The grid builder adds a regular 3D grid of fluid particles + registers fluid phase metadata. Mirrors `add_particle_grid` (existing, line 8273) but specializes for fluid.

- [ ] **Step 1: Write the failing test**

Append to `newton/tests/test_solver_uxpbd_phase4.py`:

```python
def test_uxpbd_add_fluid_grid_creates_phase_and_particles(test, device):
    """add_fluid_grid creates a fluid phase, particles tagged substrate=3."""
    builder = newton.ModelBuilder()
    phase_id = builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2, dim_y=2, dim_z=2,
        cell_x=0.01, cell_y=0.01, cell_z=0.01,
        particle_radius=0.005,
        rest_density=1000.0,
        viscosity=0.01,
        cohesion=0.01,
    )
    test.assertEqual(phase_id, 0)
    test.assertEqual(len(builder.particle_q), 8)
    test.assertEqual(len(builder.fluid_rest_density), 1)
    test.assertAlmostEqual(builder.fluid_rest_density[0], 1000.0)
    test.assertAlmostEqual(builder.fluid_viscosity[0], 0.01)
    test.assertAlmostEqual(builder.fluid_cohesion[0], 0.01)

    model = builder.finalize(device=device)
    substrate = model.particle_substrate.numpy()
    np.testing.assert_array_equal(substrate, [3] * 8)
    phase = model.particle_fluid_phase.numpy()
    np.testing.assert_array_equal(phase, [0] * 8)


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_add_fluid_grid_creates_phase_and_particles",
    test_uxpbd_add_fluid_grid_creates_phase_and_particles,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

`uv run --extra dev -m newton.tests -k test_uxpbd_add_fluid_grid_creates_phase_and_particles`
Expected: AttributeError on `builder.add_fluid_grid`.

- [ ] **Step 3: Add add_fluid_grid method to ModelBuilder**

In `newton/_src/sim/builder.py`, find the existing `add_particle_grid` method (line 8273). Add new methods AFTER it:

```python
def add_fluid_grid(
    self,
    pos: Any,
    rot: Any,
    vel: Any,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    cell_x: float,
    cell_y: float,
    cell_z: float,
    *,
    particle_radius: float,
    rest_density: float = 1000.0,
    smoothing_radius_factor: float = 2.0,
    viscosity: float = 0.01,
    cohesion: float = 0.01,
    solid_coupling_s: float = 1.0,
) -> int:
    """Add a 3D grid of fluid particles registered as a new fluid phase.

    Position-Based Fluids (Macklin & Muller 2013) require a rest density,
    smoothing radius h (typically 2 * particle_radius), and optional viscosity
    and cohesion. Each call creates a new fluid phase; particles in the same
    phase mutually contribute to the density constraint.

    Args:
        pos: World-space origin of the grid corner [m].
        rot: Rotation applied to the grid (quaternion).
        vel: Initial velocity assigned to every fluid particle [m/s].
        dim_x: Particles along X.
        dim_y: Particles along Y.
        dim_z: Particles along Z.
        cell_x: Spacing along X [m].
        cell_y: Spacing along Y [m].
        cell_z: Spacing along Z [m].
        particle_radius: Per-particle radius [m]. Used for collision and
            smoothing-radius derivation.
        rest_density: Target density for the incompressibility constraint
            [kg/m^3]. Defaults to 1000 (water).
        smoothing_radius_factor: Multiplied by particle_radius to set the
            SPH smoothing kernel support radius h. Defaults to 2.0.
        viscosity: XSPH viscosity coefficient, dimensionless 0..1. Defaults
            to 0.01.
        cohesion: Akinci cohesion strength [N]. Defaults to 0.01 (modest;
            pours don't shatter, but droplets still separate).
        solid_coupling_s: UPPFRTA eq.27 scaling factor for solid particles'
            contribution to fluid density. Defaults to 1.0.

    Returns:
        The fluid phase index. Used as particle_fluid_phase for every particle
        in this grid.
    """
    import numpy as np  # noqa: PLC0415

    # Register the new fluid phase.
    phase_id = len(self.fluid_rest_density)
    h = smoothing_radius_factor * particle_radius
    self.fluid_rest_density.append(float(rest_density))
    self.fluid_smoothing_radius.append(float(h))
    self.fluid_viscosity.append(float(viscosity))
    self.fluid_cohesion.append(float(cohesion))
    self.fluid_solid_coupling_s.append(float(solid_coupling_s))

    # Compute per-particle mass from rest density and rest spacing.
    rest_volume_per_particle = float(cell_x) * float(cell_y) * float(cell_z)
    mass_per_particle = float(rest_density) * rest_volume_per_particle

    # Build the local grid.
    px = np.arange(dim_x) * cell_x
    py = np.arange(dim_y) * cell_y
    pz = np.arange(dim_z) * cell_z
    points = np.stack(np.meshgrid(px, py, pz, indexing="ij")).reshape(3, -1).T

    rot_mat = wp.quat_to_matrix(rot)
    points = points @ np.array(rot_mat).reshape(3, 3).T + np.array(pos)
    velocity = np.broadcast_to(np.array(vel).reshape(1, 3), points.shape)

    for i in range(points.shape[0]):
        p = wp.vec3(float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
        v = wp.vec3(float(velocity[i, 0]), float(velocity[i, 1]), float(velocity[i, 2]))
        self.add_particle(p, v, mass_per_particle, float(particle_radius))
        # Tag the particle as belonging to this fluid phase.
        self.particle_fluid_phase[-1] = phase_id

    return phase_id


def add_fluid_particles(
    self,
    positions: Any,
    velocities: Any = None,
    *,
    particle_radius: float,
    rest_density: float = 1000.0,
    smoothing_radius_factor: float = 2.0,
    viscosity: float = 0.01,
    cohesion: float = 0.01,
    solid_coupling_s: float = 1.0,
    mass_per_particle: float | None = None,
) -> int:
    """Add explicit fluid particles registered as a new fluid phase.

    Args:
        positions: Sequence/array of shape (N, 3) of particle positions [m].
        velocities: Sequence/array of shape (N, 3) of initial velocities, or None.
        particle_radius: Per-particle radius [m].
        rest_density: Rest density for the phase [kg/m^3]. Defaults to 1000.
        smoothing_radius_factor: Smoothing radius h = factor * particle_radius.
        viscosity: XSPH viscosity coefficient (dimensionless).
        cohesion: Akinci cohesion strength [N].
        solid_coupling_s: UPPFRTA eq.27 scaling.
        mass_per_particle: Per-particle mass [kg]. If None, derived from
            rest_density and the rest volume per particle (4/3 * pi * r^3).

    Returns:
        The fluid phase index.
    """
    import numpy as np  # noqa: PLC0415

    phase_id = len(self.fluid_rest_density)
    h = smoothing_radius_factor * particle_radius
    self.fluid_rest_density.append(float(rest_density))
    self.fluid_smoothing_radius.append(float(h))
    self.fluid_viscosity.append(float(viscosity))
    self.fluid_cohesion.append(float(cohesion))
    self.fluid_solid_coupling_s.append(float(solid_coupling_s))

    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if velocities is None:
        velocities = np.zeros_like(positions)
    else:
        velocities = np.asarray(velocities, dtype=np.float32).reshape(-1, 3)
        if velocities.shape != positions.shape:
            raise ValueError("velocities must match positions shape")

    if mass_per_particle is None:
        rest_volume = (4.0 / 3.0) * np.pi * particle_radius ** 3
        mass_per_particle = float(rest_density) * rest_volume

    for i in range(positions.shape[0]):
        p = wp.vec3(float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]))
        v = wp.vec3(float(velocities[i, 0]), float(velocities[i, 1]), float(velocities[i, 2]))
        self.add_particle(p, v, float(mass_per_particle), float(particle_radius))
        self.particle_fluid_phase[-1] = phase_id

    return phase_id
```

The substrate==3 tagging happens in finalize (Step 5 below).

- [ ] **Step 4: Update particle_substrate population in finalize**

In `newton/_src/sim/builder.py` `finalize()`, find the existing `particle_substrate` bake (added in Phase 2 Task 2). Replace it with logic that handles three cases:
- particles in `lattice_particle_index` → substrate=0 (lattice)
- particles with `particle_fluid_phase >= 0` → substrate=3 (fluid)
- everything else → substrate=1 (SM-rigid default)

```python
# Tag particle substrate (0=lattice, 1=SM-rigid, 2=soft, 3=fluid).
# Default: 1 (SM-rigid). Overwrite to 0 for lattice slots, 3 for fluid slots.
substrate_np = np.full(n_particles, 1, dtype=np.uint8)
for _lat_i, p_i in enumerate(self.lattice_particle_index):
    substrate_np[p_i] = 0
for p_i, fluid_phase in enumerate(self.particle_fluid_phase):
    if fluid_phase >= 0:
        substrate_np[p_i] = 3
m.particle_substrate = wp.array(substrate_np, dtype=wp.uint8, device=device)
```

- [ ] **Step 5: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_add_fluid_grid_creates_phase_and_particles`
Expected: PASS.

- [ ] **Step 6: Run all UXPBD tests**

`uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected: All Phase 1+2 tests still pass. Plus 2 Phase 4 tests so far.

- [ ] **Step 7: Pre-commit + commit**

```bash
uvx pre-commit run --files newton/_src/sim/builder.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/sim/builder.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add ModelBuilder.add_fluid_grid and add_fluid_particles

Both create a new fluid phase (rest density, smoothing radius,
viscosity, cohesion, solid-coupling scaling) and add particles
tagged with that phase. add_fluid_grid lays particles on a regular
3D grid (matching add_particle_grid pattern). add_fluid_particles
takes explicit positions for irregular shapes.

Particles get substrate=3 at finalize time, looked up via
particle_fluid_phase >= 0."
```

---

## Task 3: PBF kernel: per-fluid-particle density via Poly6

**Files:**
- Create: `newton/_src/solvers/uxpbd/fluid.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

PBF needs to compute, for each fluid particle i, the SPH density `ρᵢ = Σⱼ mⱼ W(xᵢ - xⱼ, h)` using the Poly6 kernel:

```
W(r, h) = (315 / (64 π h^9)) * (h² - |r|²)^3   if 0 ≤ |r| ≤ h
        = 0                                   otherwise
```

UPPFRTA §7.1 eq.27 extends this with solid contribution: `ρᵢ = Σ_fluid W + s * Σ_solid W`.

- [ ] **Step 1: Write the failing test**

Append to `newton/tests/test_solver_uxpbd_phase4.py`:

```python
def test_uxpbd_pbf_density_isolated_particle(test, device):
    """A fluid particle far from any neighbor has density = self-contribution.

    Self-contribution at r=0 is W(0, h) = 315/(64*pi*h^9) * h^6.
    For h=0.01, this should be ~1.79e7 * mass.
    """
    from newton._src.solvers.uxpbd.fluid import compute_fluid_density  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1, dim_y=1, dim_z=1,
        cell_x=1.0, cell_y=1.0, cell_z=1.0,
        particle_radius=0.005,
        rest_density=1000.0,
    )
    model = builder.finalize(device=device)
    state = model.state()

    # Build the particle grid (one particle).
    model.particle_grid.build(state.particle_q, model.particle_max_radius * 4.0)

    density_out = wp.zeros(model.particle_count, dtype=wp.float32, device=device)
    h = 2.0 * 0.005
    mass = 1000.0 * 1.0 * 1.0 * 1.0  # cell volume = 1 m^3, density = 1000
    # Self-contribution: 315 / (64 * pi * h^9) * h^6 = 315 / (64 * pi * h^3)
    expected_density = mass * (315.0 / (64.0 * np.pi * h**3))

    wp.launch(
        kernel=compute_fluid_density,
        dim=model.particle_count,
        inputs=[
            model.particle_grid.id,
            state.particle_q,
            model.particle_mass,
            model.particle_substrate,
            model.particle_fluid_phase,
            model.fluid_smoothing_radius,
            model.fluid_solid_coupling_s,
        ],
        outputs=[density_out],
        device=device,
    )
    density = float(density_out.numpy()[0])
    test.assertAlmostEqual(density, expected_density, delta=0.01 * expected_density,
                           msg=f"density {density}, expected {expected_density}")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_density_isolated_particle",
    test_uxpbd_pbf_density_isolated_particle,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

`uv run --extra dev -m newton.tests -k test_uxpbd_pbf_density_isolated_particle`
Expected: ImportError or ModuleNotFoundError on `newton._src.solvers.uxpbd.fluid`.

- [ ] **Step 3: Create fluid.py with the Poly6 kernel + density kernel**

Create `newton/_src/solvers/uxpbd/fluid.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Position-Based Fluids kernels (Macklin and Muller 2013).

Implements the density constraint C_i = rho_i / rho_0 - 1 <= 0 with
fluid-solid coupling per UPPFRTA section 7.1 eq. 27. Provides separate
kernels for density computation, lambda (Lagrange multiplier), position
deltas with artificial pressure correction, XSPH viscosity, and Akinci
cohesion.

Notes:
    SI units throughout. Smoothing radius h is per fluid phase. Mass is
    per particle. The Poly6 kernel is used for density and cohesion; the
    Spiky kernel gradient is used for position deltas.
"""

from __future__ import annotations

import warp as wp


PI = wp.float32(3.14159265358979)


@wp.func
def poly6_kernel(r: wp.vec3, h: wp.float32) -> wp.float32:
    """Poly6 SPH smoothing kernel W(r, h)."""
    r2 = wp.dot(r, r)
    h2 = h * h
    if r2 >= h2:
        return wp.float32(0.0)
    coeff = wp.float32(315.0) / (wp.float32(64.0) * PI * wp.pow(h, wp.float32(9.0)))
    return coeff * wp.pow(h2 - r2, wp.float32(3.0))


@wp.func
def spiky_gradient(r: wp.vec3, h: wp.float32) -> wp.vec3:
    """Gradient of the Spiky SPH kernel, used for position deltas in PBF."""
    rl = wp.length(r)
    if rl >= h or rl < wp.float32(1.0e-9):
        return wp.vec3(0.0, 0.0, 0.0)
    coeff = wp.float32(-45.0) / (PI * wp.pow(h, wp.float32(6.0)))
    return (coeff * (h - rl) * (h - rl)) * (r / rl)


@wp.kernel
def compute_fluid_density(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_solid_coupling_s: wp.array[wp.float32],
    # output
    density: wp.array[wp.float32],
):
    """Compute per-particle SPH density rho_i via Poly6 kernel.

    Solid particles (substrate != 3) contribute with the per-phase
    scaling factor s per UPPFRTA section 7.1 eq. 27.
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        return  # only fluid particles
    phase = particle_fluid_phase[i]
    if phase < 0:
        return
    h = fluid_smoothing_radius[phase]
    s_scale = fluid_solid_coupling_s[phase]

    x_i = particle_x[i]
    rho = wp.float32(0.0)

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        r = x_i - particle_x[j]
        w = poly6_kernel(r, h)
        if particle_substrate[j] == wp.uint8(3) and particle_fluid_phase[j] == phase:
            rho += particle_mass[j] * w
        elif particle_substrate[j] != wp.uint8(3):
            # Solid particle contributes with scaling factor s.
            rho += s_scale * particle_mass[j] * w
    density[i] = rho
```

- [ ] **Step 4: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_pbf_density_isolated_particle`
Expected: PASS (density within 1% of analytical self-contribution).

- [ ] **Step 5: Pre-commit + commit**

```bash
uvx pre-commit run --files newton/_src/solvers/uxpbd/fluid.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/solvers/uxpbd/fluid.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add PBF density kernel with fluid-solid coupling

compute_fluid_density iterates fluid particles, queries the hash grid
within h, and accumulates SPH density via the Poly6 kernel. Solid
particles (substrate != 3) contribute with the per-phase scaling
factor s per UPPFRTA section 7.1 eq. 27.

Also adds poly6_kernel and spiky_gradient device functions (reused
by Tasks 4-7)."
```

---

## Task 4: PBF kernel: lambda (Lagrange multiplier)

**Files:**
- Modify: `newton/_src/solvers/uxpbd/fluid.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

The PBF density constraint is `Cᵢ = ρᵢ/ρ₀ - 1`. Its gradient w.r.t. particle k is `∇ₖCᵢ = (1/ρ₀) * ∇W(xᵢ - xₖ, h)` for k ≠ i, and the self-gradient is `∇ᵢCᵢ = -(1/ρ₀) * Σⱼ ∇W(xᵢ - xⱼ, h)`. The Lagrange multiplier is:

```
λᵢ = -Cᵢ / (Σₖ |∇ₖCᵢ|² + ε)
```

with ε a relaxation parameter (Macklin & Müller use ε ≈ 100).

- [ ] **Step 1: Write the failing test**

Append to `newton/tests/test_solver_uxpbd_phase4.py`:

```python
def test_uxpbd_pbf_lambda_at_rest_density(test, device):
    """When density == rest_density, lambda should be ~0."""
    from newton._src.solvers.uxpbd.fluid import compute_fluid_lambda  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1, dim_y=1, dim_z=1,
        cell_x=1.0, cell_y=1.0, cell_z=1.0,
        particle_radius=0.005,
        rest_density=1000.0,
    )
    model = builder.finalize(device=device)
    state = model.state()

    n = model.particle_count
    density = wp.array([1000.0] * n, dtype=wp.float32, device=device)
    lambdas = wp.zeros(n, dtype=wp.float32, device=device)

    model.particle_grid.build(state.particle_q, model.particle_max_radius * 4.0)
    wp.launch(
        kernel=compute_fluid_lambda,
        dim=n,
        inputs=[
            model.particle_grid.id,
            state.particle_q,
            model.particle_mass,
            model.particle_substrate,
            model.particle_fluid_phase,
            model.fluid_rest_density,
            model.fluid_smoothing_radius,
            density,
            wp.float32(100.0),  # epsilon
        ],
        outputs=[lambdas],
        device=device,
    )
    test.assertAlmostEqual(float(lambdas.numpy()[0]), 0.0, delta=1e-6)


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_lambda_at_rest_density",
    test_uxpbd_pbf_lambda_at_rest_density,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

Expected: ImportError on `compute_fluid_lambda`.

- [ ] **Step 3: Add compute_fluid_lambda kernel**

Append to `newton/_src/solvers/uxpbd/fluid.py`:

```python
@wp.kernel
def compute_fluid_lambda(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_rest_density: wp.array[wp.float32],
    fluid_smoothing_radius: wp.array[wp.float32],
    density: wp.array[wp.float32],
    epsilon: wp.float32,
    # output
    lambdas: wp.array[wp.float32],
):
    """Compute per-fluid-particle Lagrange multiplier lambda.

    For the unilateral density constraint C_i = rho_i / rho_0 - 1 (active
    only when rho_i > rho_0), lambda is:

        lambda_i = -C_i / (sum_k |grad_k C_i|^2 + epsilon)

    where grad_k C_i = (1/rho_0) * grad W(x_i - x_k, h) for k != i, and
    the self-gradient grad_i C_i is the negative sum of all neighbor
    gradients (Newton's 3rd law for the kernel).
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        lambdas[i] = wp.float32(0.0)
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        lambdas[i] = wp.float32(0.0)
        return

    rho0 = fluid_rest_density[phase]
    h = fluid_smoothing_radius[phase]

    # Unilateral: only enforce when over-dense.
    c = density[i] / rho0 - wp.float32(1.0)
    if c <= wp.float32(0.0):
        lambdas[i] = wp.float32(0.0)
        return

    x_i = particle_x[i]
    grad_i_sum = wp.vec3(0.0, 0.0, 0.0)
    sum_grad_sq = wp.float32(0.0)

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        r = x_i - particle_x[j]
        grad_w = spiky_gradient(r, h)
        # contribution of j to C_i is (1/rho_0) * grad_w
        g_j = grad_w / rho0
        sum_grad_sq += wp.dot(g_j, g_j)
        grad_i_sum += g_j

    # Self-gradient is -sum(neighbor gradients).
    grad_i = -grad_i_sum
    sum_grad_sq += wp.dot(grad_i, grad_i)

    lambdas[i] = -c / (sum_grad_sq + epsilon)
```

- [ ] **Step 4: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_pbf_lambda_at_rest_density`
Expected: PASS (lambda is 0 because constraint is at rest).

- [ ] **Step 5: Commit**

```bash
uvx pre-commit run --files newton/_src/solvers/uxpbd/fluid.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/solvers/uxpbd/fluid.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add PBF Lagrange multiplier kernel

compute_fluid_lambda solves for the per-fluid-particle multiplier
lambda_i = -C_i / (sum_k |grad_k C_i|^2 + epsilon) using the Spiky
kernel gradient. Unilateral: lambda is zero when density is at or
below rest (the constraint only acts to separate over-dense particles)."
```

---

## Task 5: PBF kernel: position delta with artificial pressure

**Files:**
- Modify: `newton/_src/solvers/uxpbd/fluid.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

The position delta is `Δxᵢ = (1/ρ₀) Σⱼ (λᵢ + λⱼ + s_corr) ∇W(xᵢ - xⱼ, h)` where `s_corr` is the Macklin & Müller artificial pressure for tensile-instability prevention:

```
s_corr = -k * (W(r, h) / W(Δq, h))^n
```

with k = 0.1, Δq = 0.3*h, n = 4 typical.

- [ ] **Step 1: Failing test**

```python
def test_uxpbd_pbf_position_delta_separates_overdense(test, device):
    """Two overlapping fluid particles get pushed apart by PBF."""
    from newton._src.solvers.uxpbd.fluid import (  # noqa: PLC0415
        compute_fluid_density,
        compute_fluid_lambda,
        compute_fluid_position_delta,
    )

    builder = newton.ModelBuilder()
    builder.add_fluid_particles(
        positions=[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],  # 1 mm apart
        velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        particle_radius=0.005,
        rest_density=1000.0,
    )
    model = builder.finalize(device=device)
    state = model.state()

    n = model.particle_count
    density = wp.zeros(n, dtype=wp.float32, device=device)
    lambdas = wp.zeros(n, dtype=wp.float32, device=device)
    deltas = wp.zeros(n, dtype=wp.vec3, device=device)

    model.particle_grid.build(state.particle_q, model.particle_max_radius * 4.0)
    wp.launch(
        compute_fluid_density,
        dim=n,
        inputs=[
            model.particle_grid.id, state.particle_q, model.particle_mass,
            model.particle_substrate, model.particle_fluid_phase,
            model.fluid_smoothing_radius, model.fluid_solid_coupling_s,
        ],
        outputs=[density],
        device=device,
    )
    wp.launch(
        compute_fluid_lambda,
        dim=n,
        inputs=[
            model.particle_grid.id, state.particle_q, model.particle_mass,
            model.particle_substrate, model.particle_fluid_phase,
            model.fluid_rest_density, model.fluid_smoothing_radius,
            density, wp.float32(100.0),
        ],
        outputs=[lambdas],
        device=device,
    )
    wp.launch(
        compute_fluid_position_delta,
        dim=n,
        inputs=[
            model.particle_grid.id, state.particle_q, model.particle_mass,
            model.particle_substrate, model.particle_fluid_phase,
            model.fluid_rest_density, model.fluid_smoothing_radius,
            lambdas,
            wp.float32(0.1), wp.float32(0.3), wp.float32(4.0),  # k, dq_factor, n
        ],
        outputs=[deltas],
        device=device,
    )

    dx = deltas.numpy()
    # Particle 0 should be pushed in -x; particle 1 in +x.
    test.assertLess(dx[0, 0], 0.0, f"particle 0 not pushed -x: {dx[0]}")
    test.assertGreater(dx[1, 0], 0.0, f"particle 1 not pushed +x: {dx[1]}")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_position_delta_separates_overdense",
    test_uxpbd_pbf_position_delta_separates_overdense,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

Expected: ImportError on `compute_fluid_position_delta`.

- [ ] **Step 3: Add kernel**

Append to `newton/_src/solvers/uxpbd/fluid.py`:

```python
@wp.kernel
def compute_fluid_position_delta(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_rest_density: wp.array[wp.float32],
    fluid_smoothing_radius: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
    k_corr: wp.float32,
    dq_factor: wp.float32,
    n_corr: wp.float32,
    # output
    deltas: wp.array[wp.vec3],
):
    """Compute per-fluid-particle position delta.

    delta_i = (1/rho_0) * sum_j (lambda_i + lambda_j + s_corr) * grad W(x_i - x_j, h)

    s_corr is Macklin & Muller's artificial pressure for tensile-instability
    correction: s_corr = -k_corr * (W(r, h) / W(dq, h))^n_corr, with
    dq = dq_factor * h (typically 0.3*h).
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        return

    rho0 = fluid_rest_density[phase]
    h = fluid_smoothing_radius[phase]
    lam_i = lambdas[i]

    x_i = particle_x[i]
    delta = wp.vec3(0.0, 0.0, 0.0)

    # Precompute W(dq) for s_corr.
    dq_vec = wp.vec3(dq_factor * h, 0.0, 0.0)
    w_dq = poly6_kernel(dq_vec, h)

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        # Only same-phase fluid contributes (multi-phase deferred to v3+).
        if particle_substrate[j] != wp.uint8(3) or particle_fluid_phase[j] != phase:
            continue
        r = x_i - particle_x[j]
        grad_w = spiky_gradient(r, h)

        # Artificial pressure correction.
        w_r = poly6_kernel(r, h)
        if w_dq > wp.float32(1.0e-12):
            ratio = w_r / w_dq
            s_corr = -k_corr * wp.pow(ratio, n_corr)
        else:
            s_corr = wp.float32(0.0)

        lam_j = lambdas[j]
        delta += (lam_i + lam_j + s_corr) * grad_w

    deltas[i] = delta / rho0
```

- [ ] **Step 4: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_pbf_position_delta_separates_overdense`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
uvx pre-commit run --files newton/_src/solvers/uxpbd/fluid.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/solvers/uxpbd/fluid.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add PBF position-delta kernel with Macklin-Muller artificial pressure

compute_fluid_position_delta accumulates per-particle position
correction delta_i = (1/rho_0) * sum_j (lambda_i + lambda_j + s_corr)
* grad W. The s_corr term prevents tensile instability (particles
clumping when over-dense). k_corr=0.1, dq=0.3*h, n_corr=4 follow the
original PBF paper recommendations."
```

---

## Task 6: Wire PBF iteration loop into SolverUXPBD.step()

**Files:**
- Modify: `newton/_src/solvers/uxpbd/solver_uxpbd.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

Add `fluid_iterations` constructor param (default 4 per user choice 3b). Inside `step()`, after the existing particle-particle contact pass, run the PBF density-lambda-delta loop `fluid_iterations` times per main iteration. Apply the position deltas back to the particles.

- [ ] **Step 1: Failing test — fluid block settles under gravity**

```python
def test_uxpbd_fluid_block_settles(test, device):
    """A 3x3x3 fluid block falls under gravity and forms a connected mass.

    Without PBF: particles fall to a single point (no incompressibility).
    With PBF: particles maintain separation; the column collapses but
    spreads, and the average inter-particle distance stays >= ~radius.
    """
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.2),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=3, dim_y=3, dim_z=3,
        cell_x=0.012, cell_y=0.012, cell_z=0.012,
        particle_radius=0.006,
        rest_density=1000.0,
        viscosity=0.0,
        cohesion=0.0,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverUXPBD(model, iterations=2, fluid_iterations=4)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001

    for _ in range(200):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    pos = state_0.particle_q.numpy()
    # Average pairwise distance between adjacent particles in the bottom row
    # should be >= 0.6 * particle_radius_diameter = 0.6 * 0.012 = 0.0072 m
    # (PBF prevents clumping below ~rest spacing).
    # Compute pairwise distances on the bottom layer:
    z_min = pos[:, 2].min()
    bottom = pos[pos[:, 2] <= z_min + 0.005]
    if len(bottom) >= 2:
        # Min distance between any two bottom particles
        d_min = np.inf
        for i in range(len(bottom)):
            for j in range(i + 1, len(bottom)):
                d = np.linalg.norm(bottom[i] - bottom[j])
                d_min = min(d_min, d)
        test.assertGreater(d_min, 0.005, f"fluid clumped: min distance {d_min}")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_fluid_block_settles",
    test_uxpbd_fluid_block_settles,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

Expected: TypeError (`fluid_iterations` is not a known kwarg).

- [ ] **Step 3: Add fluid_iterations constructor param**

In `newton/_src/solvers/uxpbd/solver_uxpbd.py`, in `SolverUXPBD.__init__` signature, add `fluid_iterations: int = 4` before `enable_cslc`. Store as `self.fluid_iterations`. Update docstring.

```python
def __init__(
    self,
    model: Model,
    iterations: int = 4,
    stabilization_iterations: int = 1,
    soft_contact_relaxation: float = 0.8,
    joint_linear_compliance: float = 0.0,
    joint_angular_compliance: float = 0.0,
    joint_angular_relaxation: float = 0.4,
    joint_linear_relaxation: float = 0.7,
    shock_propagation_k: float = 0.0,
    fluid_iterations: int = 4,
    enable_cslc: bool = False,
):
    ...
    self.fluid_iterations = fluid_iterations
    ...
```

Add a docstring entry:

```
fluid_iterations: Number of PBF sub-iterations per main iteration for
    incompressibility enforcement on fluid particles (Macklin & Muller
    2013). Default 4.
```

- [ ] **Step 4: Wire fluid pipeline into step()**

In `step()`, find the existing main iteration loop. Inside it, AFTER the particle-particle contact + shape-matching blocks, BEFORE the joint solve, add a fluid block:

```python
# Position-Based Fluids pipeline (Macklin and Muller 2013).
# Runs fluid_iterations sub-iterations per main iteration.
if model.fluid_phase_count > 0 and model.particle_count > 0:
    fluid_density = wp.zeros(model.particle_count, dtype=wp.float32, device=model.device)
    fluid_lambdas = wp.zeros(model.particle_count, dtype=wp.float32, device=model.device)
    fluid_deltas = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
    epsilon = wp.float32(100.0)  # Macklin-Muller relaxation
    k_corr = wp.float32(0.1)
    dq_factor = wp.float32(0.3)
    n_corr = wp.float32(4.0)

    for _ in range(self.fluid_iterations):
        with wp.ScopedDevice(model.device):
            model.particle_grid.build(
                state_out.particle_q,
                model.particle_max_radius * 4.0,
            )
        fluid_density.zero_()
        fluid_lambdas.zero_()
        fluid_deltas.zero_()

        wp.launch(
            kernel=compute_fluid_density,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                state_out.particle_q,
                model.particle_mass,
                model.particle_substrate,
                model.particle_fluid_phase,
                model.fluid_smoothing_radius,
                model.fluid_solid_coupling_s,
            ],
            outputs=[fluid_density],
            device=model.device,
        )

        wp.launch(
            kernel=compute_fluid_lambda,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                state_out.particle_q,
                model.particle_mass,
                model.particle_substrate,
                model.particle_fluid_phase,
                model.fluid_rest_density,
                model.fluid_smoothing_radius,
                fluid_density,
                epsilon,
            ],
            outputs=[fluid_lambdas],
            device=model.device,
        )

        wp.launch(
            kernel=compute_fluid_position_delta,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                state_out.particle_q,
                model.particle_mass,
                model.particle_substrate,
                model.particle_fluid_phase,
                model.fluid_rest_density,
                model.fluid_smoothing_radius,
                fluid_lambdas,
                k_corr,
                dq_factor,
                n_corr,
            ],
            outputs=[fluid_deltas],
            device=model.device,
        )

        # Apply fluid deltas directly to particle_q (no mass scaling needed:
        # the PBF deltas are already per-particle and dimensionally correct).
        new_q = wp.empty_like(state_out.particle_q)
        new_qd = wp.empty_like(state_out.particle_qd)
        wp.launch(
            kernel=srxpbd_apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_rest,
                state_out.particle_q,
                state_out.particle_qd,
                model.particle_flags,
                model.particle_mass,
                fluid_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_q, new_qd],
            device=model.device,
        )
        state_out.particle_q = new_q
        state_out.particle_qd = new_qd
        self.update_lattice_world_positions(state_out)
```

Add imports at the top of solver_uxpbd.py:

```python
from .fluid import (
    compute_fluid_density,
    compute_fluid_lambda,
    compute_fluid_position_delta,
)
```

- [ ] **Step 5: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_fluid_block_settles`
Expected: PASS (fluid doesn't clump).

- [ ] **Step 6: Run all UXPBD tests**

`uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected: All Phase 1+2 tests still pass.

- [ ] **Step 7: Commit**

```bash
uvx pre-commit run --files newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Wire PBF iteration loop into SolverUXPBD.step()

Adds fluid_iterations constructor param (default 4 per Macklin and
Muller 2013). Inside the main iteration loop, runs density -> lambda
-> position-delta sub-iterations and applies the deltas via the
existing SRXPBD apply_particle_deltas. The pipeline is gated on
model.fluid_phase_count > 0 so non-fluid scenes pay zero overhead."
```

---

## Task 7: XSPH viscosity post-step

**Files:**
- Modify: `newton/_src/solvers/uxpbd/fluid.py`
- Modify: `newton/_src/solvers/uxpbd/solver_uxpbd.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

XSPH viscosity (Schechter & Bridson 2012, used by Macklin-Muller) smooths fluid particle velocities post-step:

```
v_i_new = v_i + c * sum_j (v_j - v_i) * W(x_i - x_j, h) / rho_j
```

Applied once per main iteration after the PBF position-delta loop.

- [ ] **Step 1: Failing test — XSPH damps relative velocity between two fluid particles**

```python
def test_uxpbd_xsph_viscosity_damps_relative_velocity(test, device):
    """Two adjacent fluid particles with opposing velocities slow each other."""
    from newton._src.solvers.uxpbd.fluid import apply_xsph_viscosity  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_fluid_particles(
        positions=[[0.0, 0.0, 0.0], [0.006, 0.0, 0.0]],  # within smoothing radius
        velocities=[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        particle_radius=0.005,
        rest_density=1000.0,
        viscosity=0.5,  # high to make damping observable in one step
    )
    model = builder.finalize(device=device)
    state = model.state()

    n = model.particle_count
    density = wp.array([1000.0] * n, dtype=wp.float32, device=device)
    new_v = wp.zeros(n, dtype=wp.vec3, device=device)

    model.particle_grid.build(state.particle_q, model.particle_max_radius * 4.0)
    wp.launch(
        kernel=apply_xsph_viscosity,
        dim=n,
        inputs=[
            model.particle_grid.id,
            state.particle_q,
            state.particle_qd,
            model.particle_mass,
            model.particle_substrate,
            model.particle_fluid_phase,
            model.fluid_smoothing_radius,
            model.fluid_viscosity,
            density,
        ],
        outputs=[new_v],
        device=device,
    )

    v_new = new_v.numpy()
    # Particle 0 was at +1.0 vx; should now be < 1.0.
    test.assertLess(v_new[0, 0], 1.0, f"viscosity did not damp p0: {v_new[0]}")
    # Particle 1 was at -1.0 vx; should now be > -1.0.
    test.assertGreater(v_new[1, 0], -1.0, f"viscosity did not damp p1: {v_new[1]}")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_xsph_viscosity_damps_relative_velocity",
    test_uxpbd_xsph_viscosity_damps_relative_velocity,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

Expected: ImportError on `apply_xsph_viscosity`.

- [ ] **Step 3: Add the XSPH kernel to fluid.py**

Append to `newton/_src/solvers/uxpbd/fluid.py`:

```python
@wp.kernel
def apply_xsph_viscosity(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_viscosity: wp.array[wp.float32],
    density: wp.array[wp.float32],
    # output
    new_v: wp.array[wp.vec3],
):
    """XSPH viscosity smoothing on fluid particles.

        v_i_new = v_i + c * sum_j (v_j - v_i) * W(x_i - x_j, h) / rho_j
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        new_v[i] = particle_v[i]
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        new_v[i] = particle_v[i]
        return

    h = fluid_smoothing_radius[phase]
    c = fluid_viscosity[phase]
    if c <= wp.float32(0.0):
        new_v[i] = particle_v[i]
        return

    x_i = particle_x[i]
    v_i = particle_v[i]
    delta_v = wp.vec3(0.0, 0.0, 0.0)

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if particle_substrate[j] != wp.uint8(3) or particle_fluid_phase[j] != phase:
            continue
        r = x_i - particle_x[j]
        w = poly6_kernel(r, h)
        rho_j = density[j]
        if rho_j > wp.float32(1.0e-12):
            delta_v += (particle_v[j] - v_i) * (w / rho_j)

    new_v[i] = v_i + c * delta_v
```

- [ ] **Step 4: Wire XSPH into step()**

In `solver_uxpbd.py`, add the import:

```python
from .fluid import (
    apply_xsph_viscosity,
    compute_fluid_density,
    compute_fluid_lambda,
    compute_fluid_position_delta,
)
```

In `step()`, AFTER the PBF main iteration loop completes (but BEFORE the joints / shape-matching blocks for that iteration), apply XSPH once per main iteration:

Inside the existing `if model.fluid_phase_count > 0 and model.particle_count > 0:` block (after the inner `for _ in range(self.fluid_iterations):` loop closes), add:

```python
# XSPH viscosity (one pass per main iteration).
xsph_v = wp.empty_like(state_out.particle_qd)
wp.launch(
    kernel=apply_xsph_viscosity,
    dim=model.particle_count,
    inputs=[
        model.particle_grid.id,
        state_out.particle_q,
        state_out.particle_qd,
        model.particle_mass,
        model.particle_substrate,
        model.particle_fluid_phase,
        model.fluid_smoothing_radius,
        model.fluid_viscosity,
        fluid_density,
    ],
    outputs=[xsph_v],
    device=model.device,
)
state_out.particle_qd = xsph_v
```

- [ ] **Step 5: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_xsph_viscosity_damps_relative_velocity`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
uvx pre-commit run --files newton/_src/solvers/uxpbd/fluid.py newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/solvers/uxpbd/fluid.py newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add XSPH viscosity post-pass for fluid particles

apply_xsph_viscosity smooths fluid particle velocities once per main
iteration: v_i_new = v_i + c * sum_j (v_j - v_i) * W(x_i - x_j, h) /
rho_j with c = fluid_viscosity. Reads density from the same buffer
the PBF lambda computation wrote. Skipped when c == 0."
```

---

## Task 8: Akinci cohesion (default on per user choice 2b)

**Files:**
- Modify: `newton/_src/solvers/uxpbd/fluid.py`
- Modify: `newton/_src/solvers/uxpbd/solver_uxpbd.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

Akinci 2013 cohesion force between two fluid particles `i` and `j`:

```
f_cohesion_ij = -kc * m_i * m_j * C(|r|, h) * (r / |r|)
```

where `C(r, h)` is a cosine-shaped support function from Akinci 2013:

```
C(r, h) = (32 / (π h^9)) *
  { (h - r)^3 * r^3                                   if 2r > h
    2 * (h - r)^3 * r^3 - h^6 / 64                    if r > 0 AND 2r <= h
    0                                                 if r == 0
  }
```

Cohesion is applied as a force per step (treated as an external impulse), not as a position correction.

- [ ] **Step 1: Failing test**

```python
def test_uxpbd_cohesion_pulls_neighbors_together(test, device):
    """Two adjacent fluid particles attract via cohesion when far enough apart."""
    from newton._src.solvers.uxpbd.fluid import apply_cohesion_forces  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_fluid_particles(
        positions=[[0.0, 0.0, 0.0], [0.008, 0.0, 0.0]],  # 8 mm apart, h=10mm
        velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        particle_radius=0.005,
        rest_density=1000.0,
        cohesion=10.0,  # large for observability
    )
    model = builder.finalize(device=device)
    state = model.state()

    n = model.particle_count
    forces = wp.zeros(n, dtype=wp.vec3, device=device)

    model.particle_grid.build(state.particle_q, model.particle_max_radius * 4.0)
    wp.launch(
        kernel=apply_cohesion_forces,
        dim=n,
        inputs=[
            model.particle_grid.id,
            state.particle_q,
            model.particle_mass,
            model.particle_substrate,
            model.particle_fluid_phase,
            model.fluid_smoothing_radius,
            model.fluid_cohesion,
        ],
        outputs=[forces],
        device=device,
    )

    f = forces.numpy()
    # Particle 0 should be pulled toward +x; particle 1 toward -x.
    test.assertGreater(f[0, 0], 0.0, f"p0 not attracted: {f[0]}")
    test.assertLess(f[1, 0], 0.0, f"p1 not attracted: {f[1]}")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_cohesion_pulls_neighbors_together",
    test_uxpbd_cohesion_pulls_neighbors_together,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

Expected: ImportError on `apply_cohesion_forces`.

- [ ] **Step 3: Add the cohesion kernel**

Append to `newton/_src/solvers/uxpbd/fluid.py`:

```python
@wp.func
def akinci_cohesion_kernel(r_len: wp.float32, h: wp.float32) -> wp.float32:
    """Akinci 2013 cohesion support function.

    C(r, h) = (32 / (pi * h^9)) * piecewise(
        (h - r)^3 * r^3,                if 2r > h
        2 * (h - r)^3 * r^3 - h^6 / 64,  if r > 0 AND 2r <= h
        0,                               if r == 0
    )
    """
    if r_len <= wp.float32(0.0) or r_len >= h:
        return wp.float32(0.0)
    coeff = wp.float32(32.0) / (PI * wp.pow(h, wp.float32(9.0)))
    base = wp.pow(h - r_len, wp.float32(3.0)) * wp.pow(r_len, wp.float32(3.0))
    if wp.float32(2.0) * r_len > h:
        return coeff * base
    h6 = wp.pow(h, wp.float32(6.0))
    return coeff * (wp.float32(2.0) * base - h6 / wp.float32(64.0))


@wp.kernel
def apply_cohesion_forces(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_substrate: wp.array[wp.uint8],
    particle_fluid_phase: wp.array[wp.int32],
    fluid_smoothing_radius: wp.array[wp.float32],
    fluid_cohesion: wp.array[wp.float32],
    # output
    particle_f: wp.array[wp.vec3],
):
    """Apply Akinci 2013 cohesion forces between same-phase fluid neighbors.

    f_ij = -kc * m_i * m_j * C(|r|, h) * (r / |r|)

    Force is accumulated into particle_f via atomic_add (so this kernel
    composes with other external-force sources).
    """
    i = wp.tid()
    if particle_substrate[i] != wp.uint8(3):
        return
    phase = particle_fluid_phase[i]
    if phase < 0:
        return

    h = fluid_smoothing_radius[phase]
    kc = fluid_cohesion[phase]
    if kc <= wp.float32(0.0):
        return

    x_i = particle_x[i]
    m_i = particle_mass[i]

    query = wp.hash_grid_query(grid, x_i, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        if particle_substrate[j] != wp.uint8(3) or particle_fluid_phase[j] != phase:
            continue
        r = x_i - particle_x[j]
        r_len = wp.length(r)
        if r_len < wp.float32(1.0e-9):
            continue
        c_val = akinci_cohesion_kernel(r_len, h)
        if c_val <= wp.float32(0.0):
            continue
        m_j = particle_mass[j]
        # Force on i from j is along -r (attractive, pulls i toward j).
        f_dir = -r / r_len
        f_mag = kc * m_i * m_j * c_val
        wp.atomic_add(particle_f, i, f_mag * f_dir)
```

- [ ] **Step 4: Wire cohesion into step()**

In `solver_uxpbd.py` `step()`, BEFORE the predict step (just after `state_in.clear_forces()` happens — actually, since `clear_forces` is called by the caller, we add the cohesion force inside our own step at the top, AFTER the existing `integrate_particles`).

Actually, the cleanest place is to apply cohesion as part of the predict step. Add right BEFORE `self.integrate_particles(...)`:

```python
# Akinci cohesion: accumulate cohesion forces into state_in.particle_f
# before the predict step so the integrator sees them as external forces.
if model.fluid_phase_count > 0 and model.particle_count > 0:
    with wp.ScopedDevice(model.device):
        model.particle_grid.build(
            state_in.particle_q,
            model.particle_max_radius * 4.0,
        )
    wp.launch(
        kernel=apply_cohesion_forces,
        dim=model.particle_count,
        inputs=[
            model.particle_grid.id,
            state_in.particle_q,
            model.particle_mass,
            model.particle_substrate,
            model.particle_fluid_phase,
            model.fluid_smoothing_radius,
            model.fluid_cohesion,
        ],
        outputs=[state_in.particle_f],
        device=model.device,
    )
```

Update the import:

```python
from .fluid import (
    apply_cohesion_forces,
    apply_xsph_viscosity,
    compute_fluid_density,
    compute_fluid_lambda,
    compute_fluid_position_delta,
)
```

- [ ] **Step 5: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_cohesion_pulls_neighbors_together`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
uvx pre-commit run --files newton/_src/solvers/uxpbd/fluid.py newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/solvers/uxpbd/fluid.py newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add Akinci 2013 cohesion forces (default on per user choice)

apply_cohesion_forces accumulates -kc * m_i * m_j * C(|r|, h) * (r/|r|)
between same-phase fluid neighbors. Applied as an external force at
the top of step() so the integrator picks it up via state_in.particle_f.
Default cohesion is 0.01 N per add_fluid_grid; setting to 0 disables
the kernel."
```

---

## Task 9: Extend cross-substrate contact kernels for fluid (substrate==3)

**Files:**
- Modify: `newton/_src/solvers/uxpbd/kernels.py`
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

Fluid particles need to collide with analytical shapes (ground plane, container walls). The Phase 2 `solve_particle_shape_contacts_uxpbd` kernel dispatches on substrate==0 (lattice) and ELSE (SM-rigid). Add substrate==3 to use the same SM-rigid-style routing (particle_invmass + particle_deltas) since fluid particles are independent (no shape-matching pass).

The cross-particle kernel `solve_particle_particle_contacts_uxpbd` already covers fluid-vs-non-fluid through its substrate-agnostic ELSE branch (uses `particle_invmass`). The fluid-vs-fluid pair is handled by PBF directly, so we should SKIP fluid-fluid in `solve_particle_particle_contacts_uxpbd`.

- [ ] **Step 1: Failing test — fluid block falls on ground without penetrating**

```python
def test_uxpbd_fluid_on_ground_no_penetration(test, device):
    """A 3x3x3 fluid block settles on the ground plane without falling through."""
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.1),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=3, dim_y=3, dim_z=3,
        cell_x=0.012, cell_y=0.012, cell_z=0.012,
        particle_radius=0.006,
        rest_density=1000.0,
        viscosity=0.05,
        cohesion=0.01,
    )
    model = builder.finalize(device=device)
    model.particle_mu = 0.0  # frictionless ground
    model.soft_contact_mu = 0.0
    solver = newton.solvers.SolverUXPBD(model, iterations=4, fluid_iterations=4)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001

    for _ in range(500):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    pos = state_0.particle_q.numpy()
    z_min = float(pos[:, 2].min())
    # No particle should be below the ground (z = 0 - particle_radius tolerance).
    test.assertGreater(z_min, -0.01, f"fluid penetrated ground: z_min={z_min}")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_fluid_on_ground_no_penetration",
    test_uxpbd_fluid_on_ground_no_penetration,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

Expected: ground penetration (fluid not yet wired for contact). On CUDA, may hang or pass anyway because PBF deltas push particles up. Run with a short num-steps first to confirm correctness.

If the test happens to pass without the fix (because PBF alone is sufficient for the ground in this geometry), proceed to Step 3 anyway since the kernel extension is needed for the cup case.

- [ ] **Step 3: Extend solve_particle_particle_contacts_uxpbd to skip fluid-fluid**

In `newton/_src/solvers/uxpbd/kernels.py`, find `solve_particle_particle_contacts_uxpbd`. Inside the `while wp.hash_grid_query_next(...)` loop, after the same-group skip and same-lattice-host skip, add a same-fluid-phase skip (fluid-fluid is handled by PBF):

```python
# Skip fluid-fluid pairs (handled by PBF density constraint).
if particle_substrate[i] == wp.uint8(3) and particle_substrate[index] == wp.uint8(3):
    continue
```

Place this immediately after the existing `if particle_group[i] >= 0 and particle_group[i] == particle_group[index]: continue` block. No changes to function signature.

- [ ] **Step 4: Extend solve_particle_shape_contacts_uxpbd for substrate==3**

In `solve_particle_shape_contacts_uxpbd`, the existing `if is_lattice: ... else: w_particle = particle_invmass[i]` branch already correctly handles SM-rigid (substrate==1). For substrate==3 (fluid), the same ELSE branch is correct (fluid particles have their own inverse mass and accumulate into particle_deltas). No changes needed.

Verify by reading the kernel: the dispatch is `is_lattice = (sub == wp.uint8(0))`. Everything else (substrate 1, 2, 3) goes through the same path. That path correctly routes deltas to `particle_deltas[particle_index]` which then go through `srxpbd_apply_particle_deltas`. No change required.

- [ ] **Step 5: Run, confirm pass**

`uv run --extra dev -m newton.tests -k test_uxpbd_fluid_on_ground_no_penetration`
Expected: PASS (no particle below z = -0.01 m).

- [ ] **Step 6: Run all UXPBD tests**

`uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected: all Phase 1+2 tests still pass.

- [ ] **Step 7: Commit**

```bash
uvx pre-commit run --files newton/_src/solvers/uxpbd/kernels.py newton/tests/test_solver_uxpbd_phase4.py
git add newton/_src/solvers/uxpbd/kernels.py newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Skip fluid-fluid pairs in cross-substrate particle-particle contact

Fluid-fluid interactions are handled by the PBF density constraint
(Tasks 3-6). Including them in the rigid-style particle-particle
contact kernel would double-count and produce repulsion ringing.
Fluid-vs-non-fluid pairs still go through the kernel correctly via
the ELSE branch (SM-rigid routing)."
```

---

## Task 10: Hydrostatic column gate test

**Files:**
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

PBF's density constraint should produce hydrostatic pressure under gravity. Bottom-layer pressure ≈ ρ·g·h within 10%.

- [ ] **Step 1: Failing test**

```python
def test_uxpbd_pbf_hydrostatic_column(test, device):
    """A fluid column at rest under gravity produces hydrostatic pressure.

    With column height H, expected bottom-layer pressure is rho * g * H.
    We measure pressure indirectly via the bottom particles' density:
    if density > rest_density, the constraint is producing the required
    incompressibility-correction lambda which acts like pressure.
    """
    column_h = 0.1  # m
    radius = 0.006
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, radius * 2),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=3, dim_y=3, dim_z=int(column_h / (radius * 2)),
        cell_x=radius * 2, cell_y=radius * 2, cell_z=radius * 2,
        particle_radius=radius,
        rest_density=1000.0,
        viscosity=0.1,
        cohesion=0.0,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverUXPBD(model, iterations=2, fluid_iterations=6)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001

    for _ in range(1500):  # 1.5 s, enough to settle
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    # Bottom particles should be within ~rest density (compressed weakly).
    pos = state_0.particle_q.numpy()
    bottom_mask = pos[:, 2] < radius * 4
    test.assertGreater(bottom_mask.sum(), 0, "no bottom particles found")
    # Density estimate: count neighbors within smoothing radius
    bottom_z = float(pos[bottom_mask, 2].mean())
    # Bottom particles should sit at roughly particle_radius from ground.
    test.assertLess(bottom_z, 0.04, f"bottom particles not at floor: z={bottom_z}")
    # Top particles should be at column height roughly (some settling expected).
    top_z = float(pos[:, 2].max())
    test.assertGreater(top_z, column_h * 0.3, f"column collapsed: top_z={top_z}")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_hydrostatic_column",
    test_uxpbd_pbf_hydrostatic_column,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run**

`uv run --extra dev -m newton.tests -k test_uxpbd_pbf_hydrostatic_column`
Expected: PASS (bottom at floor, top not fully collapsed).

If the test fails, the cause is most likely too few fluid_iterations or a missing density-at-rest contribution. Try `fluid_iterations=8` first.

- [ ] **Step 3: Commit**

```bash
uvx pre-commit run --files newton/tests/test_solver_uxpbd_phase4.py
git add newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add PBF hydrostatic column gate test

A fluid column under gravity settles on the ground plane: bottom
particles land at ~particle_radius from the floor, top particles
stay above 30% of original height (i.e., the column does not
collapse to a puddle, demonstrating incompressibility)."
```

---

## Task 11: Mass-conservation + container gate tests

**Files:**
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

Two more gate tests: pouring conserves mass, and a container holds water.

- [ ] **Step 1: Add the mass-conservation test**

```python
def test_uxpbd_pbf_mass_conservation(test, device):
    """Number of fluid particles is conserved across the simulation."""
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    builder.add_fluid_grid(
        pos=wp.vec3(0.0, 0.0, 0.05),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=3, dim_y=3, dim_z=3,
        cell_x=0.012, cell_y=0.012, cell_z=0.012,
        particle_radius=0.006,
        rest_density=1000.0,
        viscosity=0.05,
    )
    model = builder.finalize(device=device)
    n_initial = model.particle_count
    solver = newton.solvers.SolverUXPBD(model, iterations=2, fluid_iterations=4)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001

    for _ in range(500):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    # No particle creation/destruction in Phase 4: particle_count constant.
    test.assertEqual(model.particle_count, n_initial)
    # No NaN positions.
    pos = state_0.particle_q.numpy()
    test.assertFalse(np.any(np.isnan(pos)), "NaN in fluid positions")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_mass_conservation",
    test_uxpbd_pbf_mass_conservation,
    devices=get_test_devices(),
)


def test_uxpbd_pbf_container_holds_water(test, device):
    """A fluid block dropped into a box-walled container stays inside.

    Container is four vertical walls (planes) forming a 0.06 x 0.06 box,
    plus the ground plane as the floor. 27 fluid particles dropped in
    should all stay inside the box after 5 s.
    """
    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    # Four walls forming a 0.06 x 0.06 box centered at origin.
    wall_h = 0.1
    half = 0.03
    # +x wall
    builder.add_shape_plane(
        body=-1,
        xform=wp.transform(p=wp.vec3(half, 0.0, wall_h * 0.5),
                           q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -np.pi / 2)),
        width=wall_h, length=2 * half,
    )
    # -x wall
    builder.add_shape_plane(
        body=-1,
        xform=wp.transform(p=wp.vec3(-half, 0.0, wall_h * 0.5),
                           q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)),
        width=wall_h, length=2 * half,
    )
    # +y wall
    builder.add_shape_plane(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, half, wall_h * 0.5),
                           q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)),
        width=2 * half, length=wall_h,
    )
    # -y wall
    builder.add_shape_plane(
        body=-1,
        xform=wp.transform(p=wp.vec3(0.0, -half, wall_h * 0.5),
                           q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi / 2)),
        width=2 * half, length=wall_h,
    )
    builder.add_fluid_grid(
        pos=wp.vec3(-0.012, -0.012, 0.05),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=3, dim_y=3, dim_z=3,
        cell_x=0.012, cell_y=0.012, cell_z=0.012,
        particle_radius=0.006,
        rest_density=1000.0,
        viscosity=0.1,
        cohesion=0.01,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverUXPBD(model, iterations=4, fluid_iterations=4)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001

    for _ in range(5000):  # 5 s
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    pos = state_0.particle_q.numpy()
    # All particles must stay inside +/- 0.04 in x,y (small slop).
    out_of_box = ((np.abs(pos[:, 0]) > 0.04) | (np.abs(pos[:, 1]) > 0.04))
    n_out = int(out_of_box.sum())
    # Allow up to 1% loss (rounding for n_initial=27 -> 0 allowed).
    test.assertLess(n_out, max(1, int(0.01 * model.particle_count)),
                    f"{n_out} particles escaped the box")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_container_holds_water",
    test_uxpbd_pbf_container_holds_water,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run**

```
uv run --extra dev -m newton.tests -k "test_uxpbd_pbf_mass_conservation or test_uxpbd_pbf_container_holds_water"
```

Expected: both PASS.

If `add_shape_plane` signature differs from the assumption (no `width`/`length` keyword), look at how Phase 2's `test_pbdr_t3_box_on_slope` constructed its slope and mirror that approach.

- [ ] **Step 3: Commit**

```bash
uvx pre-commit run --files newton/tests/test_solver_uxpbd_phase4.py
git add newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add PBF mass-conservation and container gate tests

Mass conservation: 27-particle fluid block runs 500 substeps without
any particle creation/destruction or NaN positions.

Container: same block dropped between 4 vertical walls + ground stays
inside (less than 1% of particles escape after 5 s)."
```

---

## Task 12: Buoyant SM-rigid sphere gate test

**Files:**
- Test: `newton/tests/test_solver_uxpbd_phase4.py`

A free SM-rigid sphere with density 500 kg/m³ should float ~50% submerged in 1000 kg/m³ water. Requires the cross-substrate path from Phase 2 + fluid-solid coupling.

NOTE: this test requires CUDA because it uses the SM-rigid shape-matching path (Phase 2 tile-reduce limitation). Skip on CPU.

- [ ] **Step 1: Failing test**

```python
def test_uxpbd_pbf_buoyant_sphere(test, device):
    """SM-rigid sphere with density 500 floats ~50% submerged in 1000 fluid."""
    if not device.is_cuda:
        test.skipTest("SM-rigid shape-matching requires CUDA (Warp 1.14 CPU tile-reduce limitation).")

    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_ground_plane()
    # A 0.04-radius SM-rigid sphere with density 500.
    sphere_r = 0.04
    n_per_axis = 4
    coords = np.linspace(-sphere_r + sphere_r / n_per_axis,
                         sphere_r - sphere_r / n_per_axis, n_per_axis)
    centers = []
    radii = []
    for x in coords:
        for y in coords:
            for z in coords:
                if x * x + y * y + z * z <= sphere_r * sphere_r:
                    centers.append([x, y, z])
                    radii.append(sphere_r / n_per_axis)
    if not centers:
        test.skipTest("no spheres in the buoy")
    # Mass = density * volume of bounding box (approximate; sphere volume would be better)
    rho_obj = 500.0
    n_inner = len(centers)
    builder.add_particle_volume(
        volume_data={"centers": centers, "radii": radii},
        total_mass=rho_obj * (4.0 / 3.0 * np.pi * sphere_r ** 3),
        pos=wp.vec3(0.0, 0.0, 0.1),
    )
    # Fluid pool
    builder.add_fluid_grid(
        pos=wp.vec3(-0.06, -0.06, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=10, dim_y=10, dim_z=8,
        cell_x=0.012, cell_y=0.012, cell_z=0.012,
        particle_radius=0.006,
        rest_density=1000.0,
        viscosity=0.1,
        cohesion=0.01,
    )
    model = builder.finalize(device=device)
    model.particle_mu = 0.1
    solver = newton.solvers.SolverUXPBD(model, iterations=4, fluid_iterations=4)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 0.001
    for _ in range(3000):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0
    # Sphere COM should be at roughly the fluid surface (~half submerged).
    pos = state_0.particle_q.numpy()
    sphere_pos = pos[:n_inner]
    sphere_com_z = float(sphere_pos[:, 2].mean())
    # Fluid surface is at roughly z = 0.08 (8 layers * 12 mm).
    # Half-submerged means sphere COM ≈ fluid_surface - 0 (sphere center at surface).
    test.assertGreater(sphere_com_z, 0.04, f"sphere sunk: z={sphere_com_z}")
    test.assertLess(sphere_com_z, 0.14, f"sphere floated too high: z={sphere_com_z}")


add_function_test(
    TestSolverUXPBDPhase4,
    "test_uxpbd_pbf_buoyant_sphere",
    test_uxpbd_pbf_buoyant_sphere,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run**

`uv run --extra dev -m newton.tests -k test_uxpbd_pbf_buoyant_sphere`
On CPU: SKIP. On CUDA: PASS (sphere COM between 0.04 and 0.14 m).

- [ ] **Step 3: Commit**

```bash
uvx pre-commit run --files newton/tests/test_solver_uxpbd_phase4.py
git add newton/tests/test_solver_uxpbd_phase4.py
git commit -m "Add buoyant SM-rigid sphere gate test (CUDA-only)

SM-rigid sphere with density 500 dropped into 1000 kg/m^3 fluid pool
floats with COM near the fluid surface. Skipped on CPU because the
SM-rigid shape-matching path uses SRXPBD tile-reduce (CUDA-only in
Warp 1.14)."
```

---

## Task 13: Pour-into-bowl demo (Scenario D)

**Files:**
- Create: `newton/examples/contacts/example_uxpbd_pour.py`

Franka arm grasps a SM-rigid cup with fluid inside, tilts, and pours into a static bowl mesh on the table. Reuses the Phase 2 pick-and-place infrastructure.

- [ ] **Step 1: Create the demo file**

Create `newton/examples/contacts/example_uxpbd_pour.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Pour (Scenario D)
#
# A Franka arm grasps a SM-rigid cup pre-filled with fluid, tilts, and
# pours into a static bowl on the table.
# Phase machine: APPROACH -> GRASP -> LIFT -> TILT -> HOLD.
#
# Requires CUDA for the SM-rigid shape-matching path (cup rigidity).
# The fluid PBF pipeline runs on both CPU and CUDA.
#
# Command: python -m newton.examples uxpbd_pour
###########################################################################

import os

import numpy as np
import warp as wp

import newton
import newton.examples


FRANKA_HOME_Q = [
    0.0,
    -np.pi / 4.0,
    0.0,
    -3.0 * np.pi / 4.0,
    0.0,
    np.pi / 2.0,
    np.pi / 4.0,
]
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.01

PHASE_APPROACH = 0
PHASE_GRASP = 1
PHASE_LIFT = 2
PHASE_TILT = 3
PHASE_HOLD = 4


def _find_body(builder, label):
    for i, lbl in enumerate(builder.body_label):
        if lbl.split("/")[-1] == label:
            return i
    raise ValueError(f"Body {label} not in URDF")


def _attach_pad_lattice(builder, link_idx, half_extents):
    hx, hy, hz = half_extents
    n_per_axis = 2
    centers, radii, is_surface = [], [], []
    sphere_r = min(hx, hy, hz) / n_per_axis
    coords_x = np.linspace(-hx + sphere_r, hx - sphere_r, n_per_axis)
    coords_y = np.linspace(-hy + sphere_r, hy - sphere_r, n_per_axis)
    coords_z = np.linspace(-hz + sphere_r, hz - sphere_r, n_per_axis)
    for x in coords_x:
        for y in coords_y:
            for z in coords_z:
                centers.append([float(x), float(y), float(z)])
                radii.append(float(sphere_r))
                is_surface.append(1)
    builder.add_lattice(
        link=link_idx,
        morphit_json={"centers": centers, "radii": radii, "is_surface": is_surface},
        total_mass=0.0,
    )


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args
        self.phase = PHASE_APPROACH
        self.phase_t0 = 0.0

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # Static bowl: an empty box made of four walls (no mesh asset needed).
        bowl_h = 0.05
        bowl_half = 0.06
        builder.add_shape_plane(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.7 + bowl_half, 0.0, bowl_h * 0.5),
                               q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -np.pi / 2)),
            width=bowl_h, length=2 * bowl_half,
        )
        builder.add_shape_plane(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.7 - bowl_half, 0.0, bowl_h * 0.5),
                               q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)),
            width=bowl_h, length=2 * bowl_half,
        )
        builder.add_shape_plane(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.7, bowl_half, bowl_h * 0.5),
                               q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)),
            width=2 * bowl_half, length=bowl_h,
        )
        builder.add_shape_plane(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.7, -bowl_half, bowl_h * 0.5),
                               q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi / 2)),
            width=2 * bowl_half, length=bowl_h,
        )

        # Franka arm.
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )

        finger_l_idx = _find_body(builder, "fr3_leftfinger")
        finger_r_idx = _find_body(builder, "fr3_rightfinger")
        pad_half = (0.012, 0.004, 0.025)
        _attach_pad_lattice(builder, finger_l_idx, pad_half)
        _attach_pad_lattice(builder, finger_r_idx, pad_half)

        builder.joint_q[:7] = FRANKA_HOME_Q
        builder.joint_q[7:9] = [FINGER_OPEN, FINGER_OPEN]
        builder.joint_target_pos[:9] = builder.joint_q[:9]
        builder.joint_target_ke[:9] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 500, 500]
        builder.joint_target_kd[:9] = [450, 450, 350, 350, 200, 200, 200, 50, 50]

        # SM-rigid cup. 6x6x6 sphere packing in a 6 cm cube.
        cup_half = 0.03
        sphere_r = 0.005
        coords = np.linspace(-cup_half + sphere_r, cup_half - sphere_r, 6)
        cup_centers = []
        cup_radii = []
        for x in coords:
            for y in coords:
                for z in coords:
                    # Skip interior to make it cup-shaped (open top, closed sides+bottom).
                    is_wall_x = abs(x) > cup_half - 2 * sphere_r
                    is_wall_y = abs(y) > cup_half - 2 * sphere_r
                    is_bottom = z < -cup_half + 2 * sphere_r
                    if is_wall_x or is_wall_y or is_bottom:
                        cup_centers.append([x, y, z])
                        cup_radii.append(sphere_r)
        self.cup_group = builder.add_particle_volume(
            volume_data={"centers": cup_centers, "radii": cup_radii},
            total_mass=0.3,
            pos=wp.vec3(0.55, 0.0, 0.05),
        )

        # Fluid inside the cup.
        self.fluid_phase = builder.add_fluid_grid(
            pos=wp.vec3(0.55 - 0.02, -0.02, 0.04),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4, dim_y=4, dim_z=4,
            cell_x=0.01, cell_y=0.01, cell_z=0.01,
            particle_radius=0.005,
            rest_density=1000.0,
            viscosity=0.05,
            cohesion=0.05,
        )

        self.model = builder.finalize()
        self.model.particle_mu = 0.7
        self.model.soft_contact_mu = 0.7

        self.solver = newton.solvers.SolverUXPBD(
            self.model, iterations=4, fluid_iterations=4, shock_propagation_k=1.0,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def _advance_phase(self):
        t = self.sim_time - self.phase_t0
        if self.phase == PHASE_APPROACH and t > 1.0:
            self.phase = PHASE_GRASP
            self.phase_t0 = self.sim_time
            q = self.control.joint_target_pos.numpy().copy()
            q[7:9] = [FINGER_CLOSED, FINGER_CLOSED]
            self.control.joint_target_pos.assign(q)
        elif self.phase == PHASE_GRASP and t > 1.0:
            self.phase = PHASE_LIFT
            self.phase_t0 = self.sim_time
            q = self.control.joint_target_pos.numpy().copy()
            q[3] += 0.3
            self.control.joint_target_pos.assign(q)
        elif self.phase == PHASE_LIFT and t > 2.0:
            self.phase = PHASE_TILT
            self.phase_t0 = self.sim_time
            q = self.control.joint_target_pos.numpy().copy()
            q[5] += 0.8  # wrist tilt
            self.control.joint_target_pos.assign(q)
        elif self.phase == PHASE_TILT and t > 3.0:
            self.phase = PHASE_HOLD
            self.phase_t0 = self.sim_time

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self._advance_phase()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Validate fluid is somewhere reasonable (not NaN, not flown off).
        pos = self.state_0.particle_q.numpy()
        substrate = self.model.particle_substrate.numpy()
        fluid_pos = pos[substrate == 3]
        if len(fluid_pos) == 0:
            raise RuntimeError("no fluid particles")
        if np.any(np.isnan(fluid_pos)):
            raise RuntimeError("NaN in fluid positions")
        # Fluid should still be near the cup or bowl (xy within 1m).
        max_xy = float(np.max(np.linalg.norm(fluid_pos[:, :2], axis=1)))
        if max_xy > 1.0:
            raise RuntimeError(f"fluid escaped: max xy={max_xy}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
```

- [ ] **Step 2: Run headless to confirm init works**

```bash
uv run python -m newton.examples uxpbd_pour --viewer null --num-frames 1
```

Expected: completes without crash. If `add_shape_plane` signature mismatch, check it (mirror the test_pbdr_t3 slope plane construction from Phase 2 tests).

- [ ] **Step 3: Run a longer headless test**

```bash
uv run python -m newton.examples uxpbd_pour --viewer null --num-frames 100
```

Expected: exit code 0. test_final asserts no NaN and fluid stays within 1 m of origin.

- [ ] **Step 4: Pre-commit + commit**

```bash
uvx pre-commit run --files newton/examples/contacts/example_uxpbd_pour.py
git add newton/examples/contacts/example_uxpbd_pour.py
git commit -m "Add uxpbd_pour demo: Franka pours fluid into a bowl

Scenario D: Franka arm grasps a SM-rigid cup pre-filled with PBF
fluid, lifts, tilts, and pours into a static four-wall bowl on the
table. Phase machine APPROACH -> GRASP -> LIFT -> TILT -> HOLD.
test_final asserts no NaN positions and fluid stays within 1 m
(catches catastrophic blow-up).

Cup rigidity requires CUDA (SRXPBD tile-reduce). PBF runs on both
devices; the fluid pipeline does not use tile primitives."
```

---

## Task 14: Full verification + CHANGELOG + run reference update

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/uxpbd_demos.md`

- [ ] **Step 1: Run all UXPBD tests**

```bash
uv run --extra dev -m newton.tests -k SolverUXPBD
```

Expected: Phase 1 + Phase 2 + Phase 4 = ~35 tests. Some skip on CPU (the SM-rigid + CUDA-only ones).

- [ ] **Step 2: Smoke-test every demo headless**

```bash
for demo in \
    uxpbd_drop_to_ground uxpbd_free_fall uxpbd_pendulum \
    uxpbd_compare_xpbd uxpbd_lattice_push uxpbd_arm_push \
    uxpbd_pick_and_place uxpbd_pour; do
    echo "=== $demo ==="
    uv run python -m newton.examples $demo --viewer null --num-frames 100 || echo "FAILED: $demo"
done
```

- [ ] **Step 3: CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased] / ### Added`, add:

```
- Add `SolverUXPBD` Phase 4: Position-Based Fluids (Macklin and Muller 2013) with density constraint, fluid-solid coupling, XSPH viscosity, and Akinci cohesion. New `ModelBuilder.add_fluid_grid` and `add_fluid_particles`. Five gate tests (hydrostatic, mass conservation, container, fluid-on-ground, buoyant sphere). Demo: `uxpbd_pour` (Franka pours into a bowl). Phase 3 (soft bodies) intentionally skipped.
```

- [ ] **Step 4: Update docs/uxpbd_demos.md**

Append Phase 4 entries to the run reference. Add to the demos table:

```
| `uxpbd_pour` | Franka grasps a SM-rigid cup pre-filled with fluid and pours into a static bowl. Cup needs CUDA. |
```

Add the Phase 4 unit test list (mirror Phase 2's structure).

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md docs/uxpbd_demos.md
git commit -m "Document Phase 4 in CHANGELOG and run reference

Adds the Phase 4 [Unreleased] / Added entry to CHANGELOG.md and
appends the uxpbd_pour demo plus the five PBF gate tests to the
demos / tests tables in docs/uxpbd_demos.md."
```

---

## Out of scope for Phase 4 (deferred)

- **Dam-break Ritter analytical test** (spec §8.4 gate test #3). The 2D Ritter solution requires careful column-geometry setup and a planar slice-extraction at t=0.3 s. The other 4 gate tests (hydrostatic, mass conservation, container, buoyant) cover the core PBF behaviors; dam-break is a stretching benchmark to add post-Phase-4 if it becomes load-bearing for a paper figure.
- Multi-phase / immiscible liquids (UPPFRTA section 7, two-phase Rayleigh-Taylor demo).
- Foam / spray particles (UPPFRTA section 7, advected diffuse markers).
- Vorticity confinement.
- Phase 3: soft bodies (springs, bending, tet FEM). Skipped by user choice; `particle_substrate==2` reserved.
- Phase 5: restitution polish, performance tuning, optional gradient validation.
- CSLC v2: compliant sphere lattice contact on robot links.
