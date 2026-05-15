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

# Every viewer demo to 100 frames, headless, with test_final asserted
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
    uxpbd_pick_and_place; do
    echo "=== $demo ==="
    uv run python -m newton.examples $demo --viewer null --test --num-frames 400 || break
done
```

Expected: every demo exit code 0; UXPBD test suite passes (Phase 1 on CPU+CUDA, Phase 2 SM-rigid tests skip on CPU and pass on CUDA).

---

