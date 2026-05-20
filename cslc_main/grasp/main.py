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
        "--pad-kind", choices=["box", "dome"], default=None,
        help="Pad geometry (default: box).",
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
    g.add_argument(
        "--cslc-contact-fraction", type=float, default=None,
        help="Override CSLC contact-fraction prior used by kc recalibration.",
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
