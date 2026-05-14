# UXPBD Phase 1 Demos

Phase 1 ships viewer-enabled demos for every validation scenario plus a 3-link arm push.

## Quick start

From the repo root:

```bash
# Default interactive GL viewer
uv run python -m newton.examples uxpbd_drop_to_ground
uv run python -m newton.examples uxpbd_free_fall
uv run python -m newton.examples uxpbd_pendulum
uv run python -m newton.examples uxpbd_compare_xpbd
uv run python -m newton.examples uxpbd_lattice_push
uv run python -m newton.examples uxpbd_arm_push
```

## Demo summary

| Name | Scenario | Validates |
| --- | --- | --- |
| `uxpbd_drop_to_ground` | Single lattice sphere on a free body, falls to ground | Lattice to body wrench routing |
| `uxpbd_free_fall` | Body without contact under gravity | Body integrator accuracy |
| `uxpbd_pendulum` | Small-angle pendulum on a Y-axis revolute joint | Joint dynamics, period within 1% of analytical |
| `uxpbd_compare_xpbd` | Same pendulum under both SolverXPBD and SolverUXPBD | Joint path matches XPBD reference |
| `uxpbd_lattice_push` | Single revolute link with lattice rotates and pushes static cube | End-to-end lattice contact + joint loop |
| `uxpbd_arm_push` | 3-link arm with per-link lattices, joint torques push static box | Multi-link articulation + multi-link lattice contact |

## CLI flags (common to all)

```bash
--viewer {gl,usd,rerun,null,viser}   # default: gl (interactive window)
--num-frames N                       # default: 100
--headless                           # OpenGL viewer without a window
--test                               # run test_final at the end, exit non-zero on failure
--device cpu | cuda:0                # override Warp device
--quiet                              # suppress Warp compilation messages
--output-path FILE.usd               # required for --viewer usd
--benchmark [SECONDS]                # measure FPS after warmup
```

Run `uv run python -m newton.examples <name> --help` for the full list.

## Useful invocations

```bash
# Watch the arm push for 5 seconds (500 frames at 100 fps)
uv run python -m newton.examples uxpbd_arm_push --num-frames 500

# CPU-only run (good if Warp CUDA init is slow on your machine)
uv run python -m newton.examples uxpbd_drop_to_ground --device cpu

# Export to USD for paper figures
uv run python -m newton.examples uxpbd_pendulum \
    --viewer usd --output-path docs/figures/uxpbd_pendulum.usd --num-frames 600

# CI-style test mode (no viewer, asserts test_final)
uv run python -m newton.examples uxpbd_drop_to_ground --viewer null --test --num-frames 500
```

## Unit tests (no viewer)

The full UXPBD test suite runs via `unittest`:

```bash
uv run --extra dev -m newton.tests -k SolverUXPBD
```

12 tests cover lattice population, FK projection, contact resolution, free fall trajectory,
pendulum period, body_parent_f comparison to XPBD, and `add_lattice_to_all_links` fallback.
These are assertion-based and have no visualization.

## Not yet implemented (deferred to Phase 2-4)

Scenarios from the spec that Phase 1 cannot demonstrate because the supporting
substrates are not yet wired up:

- **Robot grasping a free cube** (Spec scenario A, e.g. Franka pick and place).
  Requires Phase 2: free shape-matched rigid bodies via `SolverSRXPBD` kernels.
  Phase 1 can only push static targets; the cube needs to be a separate dynamic
  body for grasping to mean anything.
- **Compliant grasp of a deformable object** (Spec scenario B).
  Requires Phase 3: soft body FEM tetrahedra, springs, bending.
- **Pouring liquid from a cup into a bowl** (Spec scenario D).
  Requires Phase 4: Position-Based Fluids density constraint and fluid-solid
  coupling.

See `docs/superpowers/specs/2026-05-13-uxpbd-design.md` section 8 for the
phasing plan; sections 8.2-8.4 describe what each later phase will add.

## CSLC v2

Phase 1 reserves architectural seams for Compliant Sphere Lattice Contact (v2)
but does not implement it. Setting `enable_cslc=True` on `SolverUXPBD` raises
`NotImplementedError`. See spec section 5 for the v2 integration plan.
