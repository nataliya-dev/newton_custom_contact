# cslc_main.grasp

Two-finger grasp test harness for the CSLC contact model. Decoupled,
modular, one concern per file.

## Setup

The grasp tests pull in mesh-processing deps (`trimesh`,
`point-cloud-utils` for Lloyd/CVT pad sampling, etc.) that live in the
`importers` optional extra.  On a fresh machine — or if you hit
`ModuleNotFoundError: No module named 'point_cloud_utils'` /
`'trimesh'` — sync that extra once:

```bash
uv sync --extra importers
```

(`--extra examples` or `--extra dev` also work; both pull in
`importers` transitively.)

## Run

```bash
# Default: headless lift cycle on a tennis-ball sphere + box pads + CSLC + MuJoCo.
uv run -m cslc_main.grasp.main

# Phase sequence is fixed: APPROACH -> SQUEEZE -> LIFT -> HOLD.
# Bump TimingParams.hold_duration in params.py to extend the HOLD
# observation window for slip/creep measurement.

# Pad geometry
uv run -m cslc_main.grasp.main --pad-kind box    # default; flat brick (16x40x80 mm)
uv run -m cslc_main.grasp.main --pad-kind dome   # curved fingertip (assets/pad/pad.obj)

# Contact model (only cslc is calibrated for the mesh-pad path right now)
uv run -m cslc_main.grasp.main --contact-model cslc      # default
uv run -m cslc_main.grasp.main --contact-model point     # uncalibrated for mesh pads
uv run -m cslc_main.grasp.main --contact-model hydro     # uncalibrated for mesh pads

# Solver
uv run -m cslc_main.grasp.main --solver mujoco           # default
uv run -m cslc_main.grasp.main --solver semi             # semi-implicit

# Hot CSLC tuning knobs
uv run -m cslc_main.grasp.main --cslc-kl 25000           # lateral stiffness
uv run -m cslc_main.grasp.main --cslc-ka 25000           # anchor stiffness
uv run -m cslc_main.grasp.main --cslc-contact-fraction 0.1

# Output management
uv run -m cslc_main.grasp.main --no-timestamp            # overwrite same dir each run
uv run -m cslc_main.grasp.main --run-label my_ablation   # custom run dir name

# Viewer mode (opt-in)
uv run -m cslc_main.grasp.main --viewer gl
```

All other knobs live in `params.py` defaults — edit there for permanent
changes; use the CLI only for one-off overrides.

## Outputs

Every run writes to `outputs/grasp/<run_dir>/`:

| File | Contents |
|------|----------|
| `config.json` | Frozen `GraspConfig` snapshot (reproducibility) |
| `timeseries.csv` | Per-step: step, t, phase, dx, dz, object xyz/v, pad z (both), n_contacts |
| `cslc_state.csv` | Per-step CSLC stats (CSLC mode only): n_active, max_delta_mm, max_pen_mm, ... |
| `pad_lattice.png` | Pre-sim 3-D scatter of both pads' sampled points + outward-normal quivers |
| `sphere_trajectory.png` | Post-sim plot of object + pad Z, phase-coloured |
| `contact_count.png` | Post-sim contact-count timeline |
| `cslc_delta.png` | Post-sim max/mean δ + active-sphere count (CSLC mode only) |

Run dir name = `{timestamp}_{pad_kind}_{object_kind}_{contact_model}`
by default; `--no-timestamp` drops the prefix so iteration overwrites the
same dir.

## File layout

| File | Responsibility |
|------|----------------|
| `params.py` | Nested `@dataclass` config: `GraspConfig` holds `ObjectParams`, `PadParams`, `MaterialParams`, `CSLCParams`, `HydroParams`, `TimingParams`, `SolverParams`, `DriveParams`, `LoggingParams`. Every tuning knob lives here. |
| `pads.py` | Pad trimesh factory (`box` / `dome`) + Lloyd/CVT contact-face sampler (via `point_cloud_utils.sample_mesh_lloyd`) + per-side mesh-shape rotation. |
| `objects.py` | Held-object factory. Only `kind="sphere"` for now (tennis ball default). |
| `contact_models.py` | Per-shape `ShapeConfig` builders + CSLC mesh-pad pipeline: `make_cslc_pad_from_samples`, `build_cslc_handler_with_mesh_pads`, `patched_cslc_from_model`, `recalibrate_kc_per_pad`. |
| `solvers.py` | MuJoCo / semi-implicit factory, auto-sizing the contact-slot budget for CSLC. |
| `scene.py` | `build_scene(config)` → orchestrates pads + object + contact-model attach; returns `SceneArtifacts`. |
| `trajectory.py` | Per-step pad PD targets; smooth-ramp LIFT velocity. |
| `metrics.py` | `Metrics` dataclass + lifted/held checks. |
| `logger.py` | `CSVLogger` writing several CSVs in parallel + `read_cslc_state` / `count_active_contacts` helpers. |
| `visualization.py` | `save_lattice_preview` (pre-sim PNG), `LatticeRenderer` (in-sim GL), `save_postsim_plots` (post-sim PNGs). |
| `runner.py` | `run_headless(config)` + viewer-mode `Example` class. Both share `_simulate_one_step`. |
| `main.py` | CLI entry. Minimal argparse surface; defaults to headless. |

