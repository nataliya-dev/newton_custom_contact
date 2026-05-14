# UXPBD Run Reference

How to run every UXPBD viewer demo, every unit test, and the MuJoCo comparison harness. All commands use `uv run` and assume the repo root as the working directory.

**Implementation status (2026-05-14):**
- **Phase 1**: articulated rigid + lattice + static contact. Complete.
- **Phase 2**: free shape-matched rigid + cross-substrate contact + 7 PBD-R benchmarks + Franka pick-and-place + MuJoCo comparison. Complete.
- **Phase 3**: soft bodies. Not started.
- **Phase 4**: liquids. Not started.

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

### Phase 1 demos

| Command | What you'll see |
|---|---|
| `uv run python -m newton.examples uxpbd_drop_to_ground` | A box with a 16-sphere lattice falls and settles on the ground |
| `uv run python -m newton.examples uxpbd_free_fall` | A free rigid body falls under gravity, no contact |
| `uv run python -m newton.examples uxpbd_pendulum` | Small-angle pendulum hangs from a revolute joint |
| `uv run python -m newton.examples uxpbd_compare_xpbd` | Same pendulum simulated under SolverXPBD and SolverUXPBD in lockstep |
| `uv run python -m newton.examples uxpbd_lattice_push` | Single revolute link with a lattice rotates and pushes a static box |
| `uv run python -m newton.examples uxpbd_arm_push` | 3-link arm with per-link lattices pushes a static target box |

### Phase 2 demos

| Command | What you'll see |
|---|---|
| `uv run python -m newton.examples uxpbd_pick_and_place` | Franka arm friction-grasps a 64-sphere SM-rigid cube and lifts it. APPROACH → SQUEEZE → LIFT → HOLD phases. Needs CUDA for the SM-rigid shape-matching to converge correctly. |

### Useful demo invocations

```bash
# Run for 5 seconds simulated (500 frames at 100 fps)
uv run python -m newton.examples uxpbd_arm_push --num-frames 500

# CI / smoke test: no viewer, asserts the demo's test_final()
uv run python -m newton.examples uxpbd_drop_to_ground --viewer null --test --num-frames 500

# CPU explicitly (skip CUDA initialization)
uv run python -m newton.examples uxpbd_lattice_push --device cpu

# Export to USD for paper figures
uv run python -m newton.examples uxpbd_pendulum \
    --viewer usd --output-path docs/figures/uxpbd_pendulum.usd --num-frames 600

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

On CPU you should see roughly:
```
Ran 25 tests in ~30s
OK (skipped=9)
```
The 9 skips are all CUDA-only tests that exercise SRXPBD tile-reduce primitives. On CUDA, all 25 pass.

### Run only Phase 1 tests

```bash
uv run --extra dev -m newton.tests -k "SolverUXPBD and not Phase2"
```

12 tests; all pass on CPU.

### Run only Phase 2 tests

```bash
uv run --extra dev -m newton.tests -k SolverUXPBDPhase2
```

13 tests; 4 pass on CPU + 9 skip. On CUDA all 13 pass.

### Run a single specific test

```bash
# Phase 1 lattice projection
uv run --extra dev -m newton.tests -k test_uxpbd_update_lattice_projects_body_q

# Phase 1 pendulum period (1% of analytical 2π·√(L/g))
uv run --extra dev -m newton.tests -k test_uxpbd_pendulum_period

# Phase 1 body_parent_f cross-validation against XPBD
uv run --extra dev -m newton.tests -k test_uxpbd_body_parent_f_revolute_to_world

# Phase 2 PBD-R box benchmarks (CUDA only)
uv run --extra dev -m newton.tests -k "test_pbdr_t1 or test_pbdr_t2 or test_pbdr_t3"

# Phase 2 PBD-R bunny benchmarks (CUDA only)
uv run --extra dev -m newton.tests -k "test_pbdr_t4 or test_pbdr_t5 or test_pbdr_t6"

# Phase 2 PBD-R rod-pushing-box (CUDA only)
uv run --extra dev -m newton.tests -k test_pbdr_t7
```

### Full UXPBD test list

**Phase 1** (`newton/tests/test_solver_uxpbd.py`):

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
| `test_uxpbd_free_fall_trajectory` | Body falls under gravity at g·t²/2 (0.5% tolerance) |
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
| `test_pbdr_t6_bunny_on_slope` | Same as t3 on bunny, 10% tolerance | **CUDA only** |
| `test_pbdr_t7_rod_pushing_box` | Rod stays in contact with pushed box (gap < 5 cm) | **CUDA only** |

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

# Every viewer demo to 100 frames headless
for demo in \
    uxpbd_drop_to_ground uxpbd_free_fall uxpbd_pendulum \
    uxpbd_compare_xpbd uxpbd_lattice_push uxpbd_arm_push \
    uxpbd_pick_and_place; do
    echo "=== $demo ==="
    uv run python -m newton.examples $demo --viewer null --num-frames 100 || break
done
```

Expected:
- 25 unit tests pass / skip per device (16 + 9 on CPU; 25 + 0 on CUDA).
- All 7 demos exit code 0.

---

## 5. Not yet implemented (deferred phases)

Scenarios from the design spec (`docs/superpowers/specs/2026-05-13-uxpbd-design.md`) that current code cannot demonstrate:

- **Compliant grasp of a deformable object** (Spec scenario B). Requires **Phase 3**: soft body FEM tetrahedra, springs, bending.
- **Pouring liquid from a cup into a bowl** (Spec scenario D). Requires **Phase 4**: Position-Based Fluids density constraint and fluid-solid coupling.
- **CSLC compliant contact on robot fingers** (Spec section 5). Reserved as v2; `enable_cslc=True` raises `NotImplementedError` today. Architectural seams are present in Phase 1's lattice arrays.

See `docs/superpowers/specs/2026-05-13-uxpbd-design.md` §8.3 and §8.4 for the planned scope of Phases 3 and 4.

---

## 6. Where the code lives

| Component | Path |
|---|---|
| Solver class | `newton/_src/solvers/uxpbd/solver_uxpbd.py` |
| Warp kernels | `newton/_src/solvers/uxpbd/kernels.py` |
| Shape-matching cache helper | `newton/_src/solvers/uxpbd/shape_match.py` |
| MorphIt lattice loader | `newton/_src/solvers/uxpbd/lattice.py` |
| Public re-export | `newton/solvers.py` (`newton.solvers.SolverUXPBD`) |
| Phase 1 tests | `newton/tests/test_solver_uxpbd.py` |
| Phase 2 tests | `newton/tests/test_solver_uxpbd_phase2.py` |
| Demos | `newton/examples/contacts/example_uxpbd_*.py` |
| MuJoCo comparison | `cslc_mujoco/uxpbd_comparison/box_push.py` |
| Design spec | `docs/superpowers/specs/2026-05-13-uxpbd-design.md` |
| Phase 1 plan | `docs/superpowers/plans/2026-05-14-uxpbd-phase1-implementation.md` |
| Phase 2 plan | `docs/superpowers/plans/2026-05-14-uxpbd-phase2-implementation.md` |
