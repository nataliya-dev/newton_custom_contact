# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for the modular grasp test.

Run with ``uv run -m cslc_main.grasp.main`` — argparse surface is kept
intentionally tiny.  Most knobs live in :class:`GraspConfig` defaults;
this CLI only exposes the choices that change frequently between runs
(mode, pad kind, contact model, solver) plus a few "hot" CSLC overrides
for grip tuning.

Examples::

    # Default: tennis-ball + box pads + CSLC + MuJoCo, full lift cycle.
    uv run -m cslc_main.grasp.main

    # Interactive GL viewer.
    uv run -m cslc_main.grasp.main --viewer gl

    # Tune CSLC lateral stiffness on the fly.
    uv run -m cslc_main.grasp.main --cslc-kl 25000

    # Overwrite the same output dir each run for fast iteration.
    uv run -m cslc_main.grasp.main --no-timestamp
"""

from __future__ import annotations

import argparse

import warp as wp

import newton.examples

from .params import GraspConfig
from .runner import Example, run_headless


# ── CLI ─────────────────────────────────────────────────────────────────


def _add_grasp_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Grasp test")
    g.add_argument(
        "--contact-model", choices=["cslc", "point", "hydro"], default=None,
        help="Per-shape contact model (default: cslc).",
    )
    g.add_argument(
        "--pad-kind", choices=["box", "dome", "dome_param"], default=None,
        help="Pad geometry (default: box).  'dome' loads the shipped "
             "fingertip OBJ; 'dome_param' generates a parametric "
             "spherical-cap pad from --pad-r-pad and --pad-half-angle.",
    )
    g.add_argument(
        "--pad-r-pad", type=float, default=None,
        help="Parametric-dome cap radius [m] (only used with "
             "--pad-kind dome_param).  Production-equivalent default: 0.010.",
    )
    g.add_argument(
        "--pad-half-angle", type=float, default=None,
        help="Parametric-dome cap half-angle [deg] (only used with "
             "--pad-kind dome_param).  Production-equivalent default: 72.",
    )

    # C2e: held-object kind + box-target sampling.
    g.add_argument(
        "--object-kind", choices=["sphere", "box"], default=None,
        help="Held-object kind (default: sphere).  'box' uses the v2 "
             "unified CSLC contact path -- each box face is uniformly "
             "sampled at --box-face-pitch and routed through the single "
             "CSLCHandler._launch.",
    )
    g.add_argument(
        "--box-side", type=float, default=None,
        help="Cube full side length [m] when --object-kind box (default "
             "0.025 = 25mm).  Below ~25mm the contact patch overflows "
             "the box edges and mixes normals; the scene builder warns "
             "when 2*hx < 2*patch_radius + 6mm.",
    )
    g.add_argument(
        "--box-face-pitch", type=float, default=None,
        help="Target-point pitch [m] on each box face when --object-kind "
             "box (default 0.001 = 1mm).  Pitch sets target-sphere "
             "radius (= pitch/2) and the per-pair K_max budget.",
    )
    g.add_argument(
        "--object-spawn-y-offset", type=float, default=None,
        help="Lateral spawn jitter [m] along Y.  Used to drive multi-"
             "seed statistics for the C2 ke-sweep falsification "
             "(3 seeds at {-1mm, 0, +1mm}).  Default 0 = centred.",
    )
    g.add_argument(
        "--object-density", type=float, default=None,
        help="Held-object material density [kg/m³].  Default 368 = "
             "tennis-ball mass on a sphere; pass 7800 for steel cube "
             "(matches benchmark §7.1 silicone-target scene).",
    )
    g.add_argument(
        "--pad-n-samples", type=int, default=None,
        help="Number of CSLC lattice spheres per pad (PadParams.n_samples).  "
             "Default 100 (see PadParams.n_samples).  Per-step cost is "
             "O(N_pad × N_target) for box-target scenes; reducing N_pad "
             "gives proportional speedup at the cost of coarser lattice "
             "resolution.",
    )

    g.add_argument(
        "--solver", choices=["mujoco", "semi"], default=None,
        help="Physics solver (default: mujoco).",
    )
    g.add_argument(
        "--run-label", default=None,
        help="Override the auto-generated run-directory label.",
    )
    g.add_argument(
        "--no-timestamp", action="store_true",
        help="Drop the timestamp prefix from the output dir; overwrites "
             "the previous run each time (good for fast iteration).",
    )

    # Hot CSLC overrides — not strictly minimal, but they save a lot of
    # round-trips while tuning grip behaviour.
    g.add_argument("--cslc-kl", type=float, default=None,
                   help="Override CSLC lateral stiffness kl [N/m].")
    g.add_argument("--cslc-ka", type=float, default=None,
                   help="Override CSLC anchor stiffness ka [N/m].")
    g.add_argument("--cslc-alpha", type=float, default=None,
                   help="Override damped-Jacobi damping factor alpha [-]. "
                        "Default 0.6 (see CSLCParams.alpha).  Lower = more "
                        "damped (more stable, slower convergence); higher = "
                        "more aggressive.")
    g.add_argument("--cslc-n-iter", type=int, default=None,
                   help="Override damped-Jacobi iteration count.  "
                        "Default 20 (see CSLCParams.n_iter).  n_iter "
                        "below 20 risks divergence on stiff scenes; "
                        "40 gives paper-grade convergence margin.")
    g.add_argument("--material-ke", type=float, default=None,
                   help="LEGACY alias: sets BOTH --ke-physical and "
                        "--ke-constraint to the same value.  Preserves "
                        "pre-C2-split recipes (dome_curved_flat, day-1 "
                        "ke-sweep) bit-identically.  New code should use "
                        "the explicit per-role flags.")
    g.add_argument("--ke-physical", type=float, default=None,
                   help="CSLC pad's bulk Young's-modulus-equivalent [N/m]. "
                        "Drives calibrate_kc on the pad lattice.  Unused "
                        "under --contact-model hydro (hydro uses --kh "
                        "instead).  See MaterialParams.ke_pad_physical.")
    g.add_argument("--ke-constraint", "--ke-target", type=float, default=None,
                   dest="ke_constraint",
                   help="Object's physical contact stiffness [N/m].  Enters "
                        "the CSLC series-spring composition "
                        "1/kc_eff = 1/kc + 1/ke_target; also flows to MuJoCo "
                        "as the rigid-contact stiffness (and thus the "
                        "regularisation timeconst -- intrinsic to MuJoCo's "
                        "API, see MaterialParams docstring).  See "
                        "MaterialParams.ke_target_physical.")
    g.add_argument("--kh", type=float, default=None,
                   help="Hydroelastic physical-compliance modulus [Pa/m]. "
                        "Used only under --contact-model hydro; ignored "
                        "under CSLC / point.  See MaterialParams.kh.")
    g.add_argument(
        "--cslc-contact-fraction", type=float, default=None,
        help="Override CSLC contact-fraction prior used by kc recalibration.",
    )
    g.add_argument(
        "--cslc-smoothing-eps", type=float, default=None,
        help="Override differentiability width [m] for the kernel smooth-step "
             "gates (CSLCParams.smoothing_eps).  Production default 5e-4.  "
             "Tighter values give stiffer contact (less smoothing tail) at the "
             "cost of differentiability and lattice-solve stability; see "
             "params.py:397-403.  Sweep for the H4 dome-grip hypothesis.",
    )


def _apply_args_to_config(args, config: GraspConfig) -> GraspConfig:
    """Mutate ``config`` in place from CLI args; return for convenience."""
    if args.contact_model is not None:
        config.contact_model = args.contact_model
    if args.pad_kind is not None:
        config.pad.kind = args.pad_kind
    if args.solver is not None:
        config.solver.name = args.solver
    if args.run_label is not None:
        config.logging.run_label = args.run_label
    if args.no_timestamp:
        config.logging.use_timestamp = False
    if args.cslc_kl is not None:
        config.cslc.kl = args.cslc_kl
    if args.cslc_ka is not None:
        config.cslc.ka = args.cslc_ka
    if args.cslc_contact_fraction is not None:
        config.cslc.contact_fraction = args.cslc_contact_fraction
    if args.cslc_smoothing_eps is not None:
        config.cslc.smoothing_eps = args.cslc_smoothing_eps
    if args.cslc_alpha is not None:
        config.cslc.alpha = args.cslc_alpha
    if args.cslc_n_iter is not None:
        config.cslc.n_iter = args.cslc_n_iter
    # Legacy --material-ke alias goes first; --ke-physical /
    # --ke-constraint / --kh overrides apply on top so explicit
    # per-role flags win over the alias.
    if args.material_ke is not None:
        config.material.ke = args.material_ke   # setter writes both ke fields
    if args.ke_physical is not None:
        config.material.ke_pad_physical = args.ke_physical
    if args.ke_constraint is not None:
        config.material.ke_target_physical = args.ke_constraint
    if args.kh is not None:
        config.material.kh = args.kh
    if args.pad_r_pad is not None:
        config.pad.dome_param_R_pad = args.pad_r_pad
    if args.pad_half_angle is not None:
        import math as _math
        config.pad.dome_param_half_angle = args.pad_half_angle * _math.pi / 180.0

    # C2e: held-object overrides.
    if args.object_kind is not None:
        config.object.kind = args.object_kind
    if args.box_side is not None:
        half = args.box_side * 0.5
        config.object.box_half_extents = (half, half, half)
    if args.box_face_pitch is not None:
        config.object.box_face_pitch = args.box_face_pitch
    if args.object_spawn_y_offset is not None:
        config.object.spawn_y_offset = args.object_spawn_y_offset
    if args.object_density is not None:
        config.object.density = args.object_density
    if args.pad_n_samples is not None:
        config.pad.n_samples = args.pad_n_samples

    # C2e geometric-constraint warning: dome contact patch must fit
    # inside one box face (with a 3mm margin per side) -- otherwise the
    # patch overflows the edge, mixes face normals, and creates a local
    # wedge that confounds the ke-sweep falsification.
    if config.object.kind == "box" and config.pad.kind in ("dome", "dome_param"):
        import math as _math
        import warnings as _warnings
        R_pad = config.pad.dome_param_R_pad
        half_angle = config.pad.dome_param_half_angle  # radians
        patch_radius = R_pad * _math.sin(half_angle)
        margin = 0.003  # 3 mm per handoff
        min_side = 2.0 * patch_radius + 2.0 * margin
        actual_side = 2.0 * min(config.object.box_half_extents)
        if actual_side < min_side:
            _warnings.warn(
                f"box side {actual_side * 1000:.1f}mm < geometric "
                f"minimum {min_side * 1000:.1f}mm "
                f"(2 * patch_radius {patch_radius * 1000:.1f}mm + "
                f"2 * 3mm margin).  Contact patch will overflow the box "
                f"edges, mixing normals and creating a local wedge.  "
                f"This invalidates the C2 ke-sweep falsification.  "
                f"Raise --box-side to >= {min_side * 1000:.0f}mm.",
                RuntimeWarning,
                stacklevel=2,
            )
    return config


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = newton.examples.create_parser()
    # Default to headless — newton.examples.create_parser() sets
    # ``--viewer=gl`` which would open a GL window on every run.  Flip
    # the default so ``uv run -m cslc_main.grasp.main`` is headless;
    # users opt into the viewer with ``--viewer gl``.
    parser.set_defaults(viewer="null")
    _add_grasp_args(parser)
    args, _ = parser.parse_known_args()

    wp.init()
    print(f"\n{'━' * 60}\n  cslc_main.grasp\n{'━' * 60}")

    config = _apply_args_to_config(args, GraspConfig())

    viewer_choice = getattr(args, "viewer", None)
    if viewer_choice in (None, "none", "null"):
        run_headless(config)
        return

    # Viewer mode — let newton.examples.init build the right viewer.
    viewer, args = newton.examples.init(parser)
    config = _apply_args_to_config(args, config)
    newton.examples.run(Example(viewer, args, config), args)


if __name__ == "__main__":
    main()
