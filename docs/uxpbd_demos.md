# UXPBD Run Reference

How to run every UXPBD viewer demo, every unit test, and the MuJoCo comparison harness. All commands use `uv run` and assume the repo root as the working directory.

**CUDA caveat:** the SRXPBD tile-reduce primitives (`solve_shape_matching_batch_tiled`, `enforce_momemntum_conservation_tiled`) require a CUDA-enabled Warp build. On CPU, every test or demo that exercises free shape-matched rigid bodies will skip (unit tests) or run-but-not-validate (demos). The lattice-only Phase 1 path runs identically on CPU and CUDA.

---

## TL;DR

```bash
# Watch a demo
uv run python -m newton.examples uxpbd_pick_and_place

# Run all UXPBD unit tests
uv run --extra dev -m newton.tests -k SolverUXPBD

# Run the PBD-R analytical benchmarks (CUDA only)
uv run --extra dev -m newton.tests -k test_pbdr

# UXPBD vs MuJoCo box-push comparison
uv run python -m cslc_mujoco.uxpbd_comparison.box_push --solver both
```

---

## 1. Viewer demos

Each demo opens an interactive OpenGL window by default. Add `--headless` to run without a window, or `--viewer null` for a no-graphics CI run, or `--viewer usd --output-path file.usd` to export.

### Phase 1 demos (articulated rigid + lattice, no SM-rigid required)

These run on both CPU and CUDA.

| Command | What you'll see |
|---|---|
| `uv run python -m newton.examples uxpbd_pendulum` | A single-link revolute pendulum of length L swings under gravity at the analytical period 2π·√(L/g). `test_final` checks the period within 1%. |
| `uv run python -m newton.examples uxpbd_lattice_drop` | A free rigid body wrapped in a kinematic lattice (4×4×4 packing inscribed in a 0.04 m half-extent cube) falls under gravity and settles on the ground at `body_z = 0.048` ± 5 mm. Exercises the per-body wrench accumulation pipeline with N synchronized bottom-face contacts (regression for the "lattice launches off the ground" bug — see design doc §5.7). |

### Phase 2 demos (cross-substrate: lattice ↔ SM-rigid, CUDA only)

The SM-rigid path uses SRXPBD tile-reduce primitives that require a CUDA-enabled Warp build.

| Command | What you'll see |
|---|---|
| `uv run python -m newton.examples uxpbd_lattice_stack` | A free SM-rigid cube drops on top of a lattice-clad rigid body sharing the **exact same** 4×4×4 sphere packing (same centers, same radii). Both substrates settle: body at `z = 0.04`, cube COM at `z = 0.12 ± 4 cm`. Validates lattice ↔ SM-rigid cross-substrate contact through `solve_particle_particle_contacts_uxpbd` plus the UPPFRTA per-body averaging in `apply_body_deltas`. |
| `uv run python -m newton.examples uxpbd_lift_test` | Two articulated lattice-clad gripper pads (driven by prismatic-X + prismatic-Z joints) close on a free SM-rigid cube, squeeze, then lift it together against gravity. APPROACH → SQUEEZE → LIFT → HOLD phases. `test_final` asserts the object did not fall and slip is < 1 cm over a ~20 mm lift. Exercises the friction-driven grasp path. |
| `uv run python -m newton.examples uxpbd_box_push` | Visualizes PBD-R Test 1: a 4×4×4 sphere-packed SM-rigid cube is pushed by a constant horizontal body force on a μ=0.4 ground. Analytical trajectory x(t) = ½·(F−μMg)/M·t². |
| `uv run python -m newton.examples uxpbd_box_torque` | Visualizes PBD-R Test 2: torque-only rotation, μ=0. |
| `uv run python -m newton.examples uxpbd_box_on_slope` | Visualizes PBD-R Test 3: SM-rigid cube on a π/8 inclined slope, μ=0.4. |
| `uv run python -m newton.examples uxpbd_bunny_push` | Visualizes PBD-R Test 4: the Stanford Bunny sphere-packed and pushed. Loose tolerance because of the asymmetric mesh approximation. |
| `uv run python -m newton.examples uxpbd_particle_drop` | A free SM-rigid sphere-packed cube falls onto the ground and settles. Sanity check for the SM-rigid + ground-contact path on its own. |
| `uv run python -m newton.examples uxpbd_pick_and_place` | A Franka arm with lattice-shelled finger pads friction-grasps a 64-sphere SM-rigid cube, lifts, translates, and releases on a stack. APPROACH → SQUEEZE → LIFT → HOLD phases. |

### Phase 4 demos (PBF fluid)

PBF density solve + Akinci cohesion + XSPH viscosity. Run on CPU and CUDA (no SRXPBD tile primitives required for fluid-only scenes).

| Command | What you'll see |
|---|---|
| `uv run python -m newton.examples uxpbd_fluid_drop` | A 5×5×5 = 125-particle fluid block falls from `z = 0.20 m` onto the ground and spreads into a puddle. `test_final` asserts `z_min > -0.02 m` (no ground penetration) and `x_extent > 1.2× initial` (the block spread). Smallest scene for PBF + ground-contact regression. |
| `uv run python -m newton.examples uxpbd_fluid_column_collapse` | A 3×3×12 = 108-particle tall narrow column collapses onto the ground (dam-break lite). `test_final` asserts the column got shorter (`z_max < 0.5 × initial`), the footprint widened (`x_extent > 1.5× initial`), and no ground penetration. |
| `uv run python -m newton.examples uxpbd_fluid_cohesive_blob` | A 5×5×5 fluid block with Akinci cohesion `kc = 5 N` (the empirical stability sweet spot for SI water at this scale — bigger values compound through the contact-PBF chain). `test_final` asserts the blob stayed compact (bounding-box diagonal < 3× initial). |

### Cross-substrate combo demos

Multi-substrate scenes that exercise more than one of {lattice, SM-rigid, fluid, loose-particle} simultaneously. All require CUDA when SM-rigid is involved.

| Command | What you'll see |
|---|---|
| `uv run python -m newton.examples uxpbd_combo` | Side-by-side stress test of every substrate: a lattice-clad rigid body, a free SM-rigid cube, a sphere-masked SM-rigid ball (via `add_particle_volume`), and a 4×4×4 PBF fluid block, all on one ground plane. `test_final` asserts each substrate reached its expected rest height with `|v| < 0.5 m/s` and the fluid did not launch above its spawn height. **CUDA only** (SM-rigid path). |
| `uv run python -m newton.examples uxpbd_lattice_into_fluid` | A lattice-clad cube drops from `z = 0.30 m` into an 8×8×5 = 320-particle PBF pool. Tests lattice ↔ fluid coupling: lattice contact deltas route into the host body wrench, fluid density estimation includes lattice spheres as "solid" per UPPFRTA eq. 27. `test_final` asserts the cube sank/displaced fluid, did not pierce the ground, and the fluid did not splash above the spawn height. Runs on CPU + CUDA. |
| `uv run python -m newton.examples uxpbd_volume_into_fluid` | A 2000 kg/m³ sphere-masked SM-rigid ball (`add_particle_volume`) sinks through a 1000 kg/m³ PBF pool. Tests SM-rigid ↔ fluid coupling with the shape-matching post-pass keeping the ball rigid through the splash. **CUDA only**. |
| `uv run python -m newton.examples uxpbd_multi_fluid` | Two fluid phases with different rest densities (heavy ρ₀ = 1500 kg/m³ stacked above light ρ₀ = 1000 kg/m³). Tests UPPFRTA §7.1.1 mass-weighted PBF buoyancy / Rayleigh-Taylor inversion. `test_final` asserts the heavy phase migrated downward (mean z dropped or inter-phase gap shrank). Runs on CPU + CUDA. |
| `uv run python -m newton.examples uxpbd_raining_on_stack` | A 5×5×3 = 75-particle grid of loose particles (`add_particle_grid`) rains down from `z = 0.40 m` onto a lattice-clad cube resting on the ground. Tests lattice ↔ loose-particle PP contact: rain hits route into the host body's wrench without launching it off the ground. `test_final` asserts the lattice body stayed in `0.02 < z < 0.10 m` and no rain particle launched above its spawn height. Runs on CPU + CUDA. |

### Useful demo invocations

```bash
# Run for 5 seconds simulated (500 frames at 100 fps)
uv run python -m newton.examples uxpbd_pick_and_place --num-frames 500

# CI / smoke test: no viewer, asserts the demo's test_final()
uv run python -m newton.examples uxpbd_lattice_drop --viewer null --test --num-frames 500
uv run python -m newton.examples uxpbd_lattice_stack --viewer null --test --num-frames 500
uv run python -m newton.examples uxpbd_lift_test  --viewer null --test --num-frames 400

# CPU explicitly (skip CUDA initialization) -- Phase 1 demos only
uv run python -m newton.examples uxpbd_lattice_drop --device cpu

# Live-streamed to a Rerun viewer
uv run python -m newton.examples uxpbd_pick_and_place --viewer rerun
```

### CLI flags (common to all examples)

```bash
--viewer {gl,usd,rerun,null,viser}   # default: gl
--num-frames N                       # default: 100
--headless                           # GL viewer without a window
--test                               # run test_final(), exit non-zero on failure
--device cpu | cuda:0                # override Warp device
--quiet                              # suppress Warp compilation messages
--output-path FILE.usd               # required for --viewer usd
--benchmark [SECONDS]                # measure FPS after a warmup
```

Full list per demo: `uv run python -m newton.examples <name> --help`.

---

## 2. Unit tests

Tests are plain `unittest` assertions. No viewer. Test discovery picks up everything under `newton/tests/` matching the filter.

### Run all UXPBD tests

```bash
uv run --extra dev -m newton.tests -k SolverUXPBD
```

### Run a single specific test

```bash
# Phase 1 pendulum period (1% of analytical 2π·√(L/g))
uv run --extra dev -m newton.tests -k test_uxpbd_pendulum_period

```

### Full UXPBD test list

**Phase 1** (`newton/tests/test_solver_uxpbd_phase1.py`):

| Test | Validates |
|---|---|
| `test_uxpbd_solver_rejects_enable_cslc` | `enable_cslc=True` raises `NotImplementedError` |
| `test_uxpbd_solver_instantiates_with_empty_model` | Empty model + ground plane constructs cleanly |
| `test_uxpbd_empty_model_has_zero_lattice` | `model.lattice_*` arrays are size 0 when no lattice attached |
| `test_uxpbd_add_lattice_populates_arrays` | `add_lattice` loads MorphIt JSON, sets lattice metadata + particle slots |
| `test_uxpbd_update_lattice_projects_body_q` | Lattice particles match `body_q ⊗ p_rest` analytically (translation) |
| `test_uxpbd_update_lattice_handles_rotation` | Same, with 90° Z rotation |
| `test_uxpbd_lattice_w_eff_helper_callable` | `lattice_sphere_w_eff` device function exists |
| `test_uxpbd_lattice_sphere_drops_to_ground` | Single sphere settles at z = sphere radius |
| `test_uxpbd_free_fall_trajectory` | Free-fall body trajectory matches `g·t²/2` within 0.5% |
| `test_uxpbd_pendulum_period` | Pendulum period within 1% of 2π·√(L/g) |
| `test_uxpbd_body_parent_f_revolute_to_world` | `body_parent_f` matches SolverXPBD within 5% on revolute joint |
| `test_uxpbd_add_lattice_to_all_links_with_fallback` | Bulk lattice attach uses JSON when present, falls back to uniform |

**Phase 2** (`newton/tests/test_solver_uxpbd_phase2.py`):

| Test | Validates | Device |
|---|---|---|
| `test_uxpbd_link_lattice_csr_offsets` | Per-link CSR offsets populated | CPU + CUDA |
| `test_uxpbd_particle_substrate_tagging` | `particle_substrate` is 0 for lattice, 1 for SM-rigid | CPU + CUDA |
| `test_uxpbd_solver_caches_shape_match_data` | Dynamic group CSR arrays cached at solver init | CPU + CUDA |
| `test_uxpbd_shock_propagation_param_accepted` | `shock_propagation_k` accepted, kernel launches | CPU + CUDA |
| `test_uxpbd_sm_rigid_cube_stays_rigid` | SM-rigid cube preserves pairwise distances under free spin | **CUDA only** |
| `test_uxpbd_sm_rigid_cube_drops_to_ground` | SM-rigid cube settles at z ≈ 0.11 m on ground plane | **CUDA only** |
| `test_pbdr_t1_pushed_box` | F=17 N, μ=0.4 → x(t) = 0.5·(F−μMg)/M·t² within 5% | **CUDA only** |
| `test_pbdr_t2_box_torque` | τ=0.01 N·m, μ=0 → θ(t) = 0.5·(τ/λ)·t² within 5% | **CUDA only** |
| `test_pbdr_t3_box_on_slope` | π/8 slope, μ=0.4 → x(t) = 0.5·(g sinθ − μg cosθ)·t² within 5% | **CUDA only** |
| `test_pbdr_t4_pushed_bunny` | Same as t1 on Stanford Bunny, 10% tolerance | **CUDA only** |
| `test_pbdr_t5_bunny_torque` | Same as t2 on bunny, 10% tolerance | **CUDA only** |
| `test_pbdr_t1_lattice_pushed_box` | PBD-R t1 with the box replaced by a **lattice-clad articulated body** that shares its geometry with `example_uxpbd_lattice_stack` (half-extent 0.04, sphere_r 0.012, 4×4×4 packing). Force F scaled to preserve F/M so the analytical reference matches t1's SM-rigid variant. Within 5%. | **CUDA only** |

**Phase 4** (`newton/tests/test_solver_uxpbd_phase4.py`):

PBF fluid path. Each test runs on both CPU and CUDA (10 tests × 2 devices = 20 cases).

| Test | Validates |
|---|---|
| `test_uxpbd_empty_model_has_zero_fluid` | A model with no fluids has `fluid_phase_count == 0` and empty fluid arrays |
| `test_uxpbd_add_fluid_grid_creates_phase_and_particles` | `add_fluid_grid` registers a new phase, allocates particles, sets `particle_substrate == 3` and `particle_fluid_phase` |
| `test_uxpbd_pbf_density_isolated_particle` | An isolated fluid particle has density = `m · W(0, h)` (self-contribution only) |
| `test_uxpbd_pbf_lambda_at_rest_density` | When density equals `ρ_0` (C = 0), the unilateral constraint cuts off and `lambda = 0` |
| `test_uxpbd_pbf_position_delta_separates_overdense` | Over-dense particles get a position delta pointing away from neighbours (correct repulsion direction) |
| `test_uxpbd_pbf_position_delta_bounded_under_realistic_mass` | Position deltas stay bounded for SI water-mass particles; regression for the mass-weighting bug that produced metre-scale corrections |
| `test_uxpbd_fluid_block_settles` | A 3×3×3 fluid block falls without collapsing to a point (PBF incompressibility keeps inter-particle distance ≳ radius) |
| `test_uxpbd_xsph_viscosity_damps_relative_velocity` | XSPH reduces neighbour velocity differences; regression for the missing-`m_j` over-amplification |
| `test_uxpbd_cohesion_pulls_neighbors_together` | Akinci cohesion pair force attracts same-phase fluid neighbours |
| `test_uxpbd_fluid_on_ground_no_penetration` | A fluid block on the ground does not penetrate below `z = -0.02`; regression for the contact-PBF launch (s_corr gate + position-delta clamp) |

### Run the entire Newton test suite (regression check)

```bash
uv run --extra dev -m newton.tests
```

Takes several minutes. UXPBD-related tests are a subset; the rest are pre-existing Newton tests that should continue to pass.

---

## 3. MuJoCo vs UXPBD comparison

```bash
# Run both solvers on PBD-R Test 1 (F=17 N horizontal, μ=0.4 ground), print errors
uv run python -m cslc_mujoco.uxpbd_comparison.box_push --solver both

# Just UXPBD
uv run python -m cslc_mujoco.uxpbd_comparison.box_push --solver uxpbd

# Just MuJoCo
uv run python -m cslc_mujoco.uxpbd_comparison.box_push --solver mujoco
```

Each run prints `final x [m]`, `expected x [m]`, and `rel_err`. Both solvers should land within 5% of the analytical on CUDA hardware. UXPBD requires CUDA because the SM-rigid path uses SRXPBD tile-reduce primitives.

---

## 4. End-to-end smoke check

When you want to confirm nothing is broken locally:

```bash
# All UXPBD unit tests
uv run --extra dev -m newton.tests -k SolverUXPBD

# Every viewer demo to 400 frames, headless, with test_final asserted
for demo in \
    uxpbd_pendulum \
    uxpbd_lattice_drop \
    uxpbd_lattice_stack \
    uxpbd_lift_test \
    uxpbd_box_push \
    uxpbd_box_torque \
    uxpbd_box_on_slope \
    uxpbd_bunny_push \
    uxpbd_particle_drop \
    uxpbd_pick_and_place \
    uxpbd_fluid_drop \
    uxpbd_fluid_column_collapse \
    uxpbd_fluid_cohesive_blob \
    uxpbd_combo \
    uxpbd_lattice_into_fluid \
    uxpbd_volume_into_fluid \
    uxpbd_multi_fluid \
    uxpbd_raining_on_stack; do
    echo "=== $demo ==="
    uv run python -m newton.examples $demo --viewer null --test --num-frames 400 || break
done
```

Expected: every demo exit code 0; UXPBD test suite passes (Phase 1 + Phase 4 on CPU+CUDA, Phase 2 SM-rigid tests skip on CPU and pass on CUDA, combo demos involving SM-rigid skip on CPU).

---

