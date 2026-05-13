# Solver experiments (SRXPBD, MRXPBD, MuJoCo, ...)

Headless scenarios for benchmarking the `SolverSRXPBD` (sphere-rigid XPBD) implementation
against `SolverMuJoCo`, `SolverSemiImplicit`, and `SolverXPBD` on canonical rigid-body tasks
(push, slope-slide, tip). Each scenario has a known analytical reference; data is collected
every frame and dumped to `outputs/<scenario>/<solver>_<config>.npz` + matching `.png`.

> These examples were ported from `phys-experiments`. Bunny scenarios need the morphit sphere
> packing JSONs (see `assets/`). Box scenarios run without extra assets.

## Quick start

```bash
# Box pushed by a constant force, srxpbd 4×4×4 sphere packing, 10 s sim
uv run python -m examples.push.pushed_box -e srxpbd \
    -t 10.0 --num-frames 1100 --headless
# → outputs/pushed_box/srxpbd_n64_i10.npz  +  .png

# or run the bunny
uv run python -m examples.push.pushed_box -e mrxpbd \
  -m assets/box/morphit_results.json \
  -t 10.0 --num-frames 1100 --headless

```

Open the PNG, that's the per-solver "solver vs analytical" plot.

## Solver knob (`-e / --experiment`)

| Value      | What it builds                                                | Solver                     |
|------------|---------------------------------------------------------------|----------------------------|
| `srxpbd`   | Uniform `n × n × n` sphere packing of the rigid body           | `SolverSRXPBD`             |
| `mrxpbd`   | Morphit sphere packing from `-m / --morphit_json` file          | `SolverSRXPBD`             |
| `bxpbd`    | Same packing as srxpbd, uses a different XPBD variant          | `SolverBXPBD` *(not ported here)* |
| `tetxpbd`  | Single rigid body, tetrahedral mesh                            | `SolverXPBD`               |
| `mujoco`   | Single rigid body                                              | `SolverMuJoCo`             |
| `semieuler`| Single rigid body                                              | `SolverSemiImplicit`       |

`srxpbd`/`mrxpbd`/`bxpbd` operate on a particle representation of the body (rigid-body shape
matching). `tetxpbd`/`mujoco`/`semieuler` operate on a single rigid body — the analytical
reference comparison uses the same closed-form solution either way.

## Common CLI flags (`get_base_parser` + per-example extras)

| Flag                          | Default            | Meaning                                                                            |
|-------------------------------|--------------------|------------------------------------------------------------------------------------|
| `-e / --experiment`           | `tetxpbd`          | Solver choice (table above).                                                       |
| `-i / --num_iterations`       | `10`               | Iterations per solver step.                                                        |
| `-n / --num_spheres`          | `4`                | (push examples) Sphere-packing dimension; `n³` particles total. Used only for `srxpbd`/`bxpbd`. |
| `-m / --morphit_json`         | —                  | Path to morphit sphere-packing JSON. Required for `mrxpbd`.                        |
| `-t / --plot_sim_time`        | `10.0`             | Simulated seconds before saving NPZ + plotting and exiting.                        |
| `--num-frames`                | `inf`              | Frame budget. Must exceed `t * 100` (fps is hard-coded to 100) for plotting to fire. |
| `--constant-force`            | `2.0 0.0 0.0`      | (push examples) Constant world-space force applied each step.                      |
| `--mu`                        | `0.0`              | (push examples) Coulomb friction between body and ground.                          |
| `--headless`                  | off                | Skip GL window. Recommended for batch runs.                                        |
| `--no-vis-plot`               | off                | Skip the per-solver PNG (still saves NPZ).                                         |
| `--exp-address`               | auto               | Override NPZ output path.                                                          |
| `--seed`                      | `0`                | RNG seed (jitter, etc.).                                                           |

The full grammar comes from `examples/base_example.py:get_base_parser()` and each example's
own `argparse` block at the bottom of the file.

## Output layout

```
outputs/
├── pushed_box/
│   ├── srxpbd_n64_i10.npz   ← per-frame data
│   ├── srxpbd_n64_i10.png   ← per-solver vs analytical
│   ├── mujoco_i10.npz
│   ├── mujoco_i10.png
│   ├── mrxpbd_i10.npz
│   └── mrxpbd_i10.png
├── pushed_box_experiment_comparison.png   ← cross-solver overlay
└── pushed_bunny/
    └── ...
```

NPZ keys are defined by `utils.data_collector.COLLECTOR_TYPES`. Notable ones:

| Key            | Shape          | Meaning                                            |
|----------------|----------------|----------------------------------------------------|
| `pos_solver`   | `(T, 3)`       | Body position over time, from the solver.          |
| `pos_analytic` | `(T, 3)`       | Closed-form analytical position.                   |
| `pos_l2_err`   | `(T,)`         | L2 norm of `pos_solver - pos_analytic`.            |
| `rot_err`      | `(T,)`         | Quaternion distance to analytical orientation, degrees. |
| `w_solver` / `w_analytic` / `w_err` | `(T, 3)` | Angular velocity (rad/s).                  |
| `L_solver` / `L_analytic` / `L_err` | `(T, 3)` | Angular momentum.                          |
| `p_solver` / `p_analytic` / `p_err` | `(T, 3)` | Linear momentum.                           |
| `force_array` / `tau_array`         | `(T, 3)` | Per-frame summed particle force/torque (sphere-packing solvers). |
| `frame_dt`     | scalar         | Time between frames.                               |

## Scenarios

### `examples/push/`

| Module                          | What it does                                                     |
|---------------------------------|-------------------------------------------------------------------|
| `pushed_box`                    | Cube on ground, constant force at COM. Analytical reference exists. |
| `pushed_box_not_com`            | Same, force off-COM (induces rotation).                          |
| `pushed_bunny`                  | Stanford bunny, force at COM. Mesh from `assets/bunny-lowpoly/`. |
| `pushed_bunny_not_com`          | Bunny, off-COM force.                                            |
| `rod_pushed_box`                | Box pushed by a sphere-packed rod. No analytical reference for full state. |
| `rod_pushed_t`                  | T-shaped block pushed by a rod. Analytical solution is stub. *Sim runs; post-processing raises.* |

### `examples/slide_on_slope/`

| Module                          | What it does                                                     |
|---------------------------------|-------------------------------------------------------------------|
| `box_on_slope`                  | Cube sliding down a slope under gravity, Coulomb friction.       |
| `bunny_on_slope`                | Bunny sliding down a slope.                                      |

### `examples/tip/`

| Module                          | What it does                                                     |
|---------------------------------|-------------------------------------------------------------------|
| `box_tip`                       | Cube tipping on one edge. *Sim runs; tip example only collects `omega_err`, post-processing of unused rotation buffers raises.* |

## Cross-solver comparison plots

After you've generated NPZs for several solvers on the same scenario:

```bash
uv run python -m scripts.compare_experiments \
    -e pushed_box/srxpbd_n64_i10 \
       pushed_box/mujoco_i10 \
       pushed_box/mrxpbd_i10 \
    -t pushed_box
# → outputs/pushed_box_experiment_comparison.png
```

The script ([scripts/compare_experiments.py](../scripts/compare_experiments.py)) overlays all
solvers for keys `L_solver`, `L_analytic`, `w_solver`, `w_analytic`, `w_err`, `tau_array`,
`rot_err`, `pos_err`, `pos_l2_err`. Each row is one key; columns are X/Y/Z components.

`-e` takes paths relative to `outputs/` *without* the `.npz` extension. `-t` is the comparison
filename tag (becomes `outputs/<tag>_experiment_comparison.png`).

## Full reproduction recipe — box and bunny at 10 s, three solvers

```bash
# Generate all six NPZs
for FORCE in srxpbd mujoco mrxpbd; do
  case $FORCE in
    mrxpbd) BOX_MORPHIT="-m assets/box/morphit_results.json"
            BUNNY_MORPHIT="-m assets/bunny-lowpoly/morphit_results.json" ;;
    srxpbd) BOX_MORPHIT="";  BUNNY_MORPHIT="-m assets/bunny-lowpoly/morphit_results.json" ;;
    *)      BOX_MORPHIT="";  BUNNY_MORPHIT="" ;;
  esac
  uv run python -m examples.push.pushed_box   -e $FORCE $BOX_MORPHIT   -t 10.0 --num-frames 1100 --headless
  uv run python -m examples.push.pushed_bunny -e $FORCE $BUNNY_MORPHIT -t 10.0 --num-frames 1100 --headless
done

# Overlay (replace n64/n2191 with whatever filenames actually got written)
uv run python -m scripts.compare_experiments \
    -e pushed_box/srxpbd_n64_i10 pushed_box/mujoco_i10 pushed_box/mrxpbd_i10 \
    -t pushed_box

uv run python -m scripts.compare_experiments \
    -e pushed_bunny/srxpbd_n2191_i10 pushed_bunny/mujoco_i10 pushed_bunny/mrxpbd_i10 \
    -t pushed_bunny
```

Note `pushed_bunny -e srxpbd` calls `manual_sphere_packing` on the bunny `.obj` mesh; the
resulting particle count depends on the mesh and is reflected in the output filename suffix
(e.g. `srxpbd_n2191_i10`). Check `outputs/pushed_bunny/` for the actual name before running
the comparison.

## Plotting backend

Plots use matplotlib's non-interactive `Agg` backend (PNG-only, no popups). This is set in
[utils/plot.py](../utils/plot.py) and [scripts/compare_experiments.py](../scripts/compare_experiments.py).
To view results, open the saved PNG (`xdg-open`, IDE preview, browser, etc.).

## Adding a new scenario

1. Subclass `BaseExample` (or one of `BasePush`/`BaseSlide`/`BaseTip`).
2. Implement `build_scene(builder)`, `launch_scenario_force_kernels()`,
   `calculate_analytical_state(t)`, and the data collectors.
3. Choose which `-e` solver branches to support (the standard set is
   `srxpbd / mrxpbd / mujoco / semieuler / tetxpbd / bxpbd`).
4. Wire `argparse` in `if __name__ == "__main__"` and call
   `newton.examples.init(parser) → init(...) → run(example, args)`.

See `examples/push/pushed_box.py` for the minimal pattern.
