# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# UXPBD Book Disturbance Test -- CSLC vs rigid rotational stiffness
#
# Runs the lift-test scene twice (CSLC on / off) at a fixed configuration,
# both ending in a DISTURB phase: after HOLD a short transverse force
# pulse is applied to the bottom of a tall "book" object, and the
# resulting tilt(t) ring-down is recorded. Emits one PNG with tilt(t)
# and obj_y(t) overlaid for the two methods -- the rotational analogue
# of the lift-test slip plot.
#
# The point this test makes:
#
#   Rigid (point/box) contact has near-zero patch area, so it can't
#   store a distributed pressure gradient. Under a transverse force,
#   the only thing resisting rotation is the friction at the
#   contact line -- low rotational stiffness, large peak tilt,
#   slow damping.
#
#   CSLC has a real contact patch with per-sphere compliance. When
#   the book tries to tilt, the spheres on the loading edge compress
#   while the opposite edge unloads, generating a distributed pressure
#   gradient that is a real restoring couple. Higher rotational
#   stiffness, smaller peak tilt, faster damping.
#
# Command:
#
#   uv run python -m newton.examples.contacts.example_uxpbd_book_disturb \
#       --force 5.0 --mass 0.5 --mu 0.5 \
#       --output-dir results/book_disturb
###########################################################################

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import newton.viewer

from .example_uxpbd_lift_test import SceneParams, _disturb_phase_start
from .example_uxpbd_lift_test_sweep import run_one_config, _write_csv


def _run_once(params: SceneParams, *, use_cslc: bool,
              pad: str = "box", num_pads: int = 2) -> list[dict]:
    """Headless run of one (method, config) pair. Returns history."""
    return run_one_config(
        params, object="book", pad=pad,
        num_pads=num_pads, use_cslc=use_cslc,
    )


def _disturb_window(p: SceneParams) -> tuple[float, float]:
    """``(t_start, t_end)`` of the DISTURB phase in world time [s]."""
    t0 = _disturb_phase_start(p)
    return t0, t0 + p.disturb_duration


def _pulse_window(p: SceneParams) -> tuple[float, float]:
    """``(t_pulse_start, t_pulse_end)`` of the force pulse [s]."""
    t0 = _disturb_phase_start(p)
    return t0, t0 + p.disturb_force_duration


def plot_disturb_comparison(
    history_cslc: list[dict],
    history_rigid: list[dict],
    p: SceneParams,
    output_path: Path,
) -> None:
    """Three-panel time-series PNG showing the whole run (SETTLE
    through DISTURB), not just the disturbance window:

      * obj_z(t) for both methods, with pad_z(t) overlaid as the
        target the grip is trying to reach. The vertical gap between
        ``obj_z`` and ``pad_z`` is the slip; a perfect grasp tracks
        the dashed pad line. Phase boundaries are annotated.
      * Applied disturbance force(t) -- a square pulse of
        ``disturb_force_amplitude`` over ``disturb_force_duration``,
        zero elsewhere. Makes the disturbance event obvious even when
        the CSLC response is too small to see in the tilt panel.
      * Tilt about the disturbance axis (deg) -- the rotational
        response. CSLC expected near zero; rigid expected to swing.

    The DISTURB pulse window is shaded in all panels for context.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[plot] matplotlib not available ({exc}); skipping plot.")
        return

    # Read the actual phase boundaries from the history rather than
    # computing them from the un-overridden ``SceneParams`` -- the
    # Example may shorten LIFT / SETTLE inside __init__ for certain
    # object kinds, so the analytical phase ladder doesn't agree with
    # what the simulation actually ran.
    def _phase_starts(h: list[dict]) -> dict[str, float]:
        starts: dict[str, float] = {}
        prev = None
        for r in h:
            if r["phase"] != prev:
                starts.setdefault(r["phase"], r["t"])
                prev = r["phase"]
        return starts

    starts_cslc = _phase_starts(history_cslc)
    starts_rigid = _phase_starts(history_rigid)
    # Use whichever run contains the phase (they should agree closely
    # because both runs share the same SceneParams + override).
    def _phase_start(phase: str) -> float:
        return starts_cslc.get(phase, starts_rigid.get(phase, float("nan")))

    t_disturb_start = _phase_start("disturb")
    t_pulse_start = t_disturb_start
    t_pulse_end = t_disturb_start + p.disturb_force_duration

    # Pick the tilt axis aligned with the disturbance direction. Force
    # along +X -> book pitches about Y -> use ``tilt_about_y_deg``.
    # Force along +Y -> book rolls about X -> use ``tilt_about_x_deg``.
    use_y_axis = abs(p.disturb_force_dir_x) > abs(p.disturb_force_dir_y)
    tilt_key = "tilt_about_y_deg" if use_y_axis else "tilt_about_x_deg"
    tilt_label = ("Tilt about Y [deg]  (pitch from +X force)"
                  if use_y_axis else
                  "Tilt about X [deg]  (roll from +Y force)")

    def _t(h): return [r["t"] for r in h]
    def _s(h, k): return [r[k] for r in h]

    phase_boundaries = [
        ("APPROACH", _phase_start("approach")),
        ("SQUEEZE", _phase_start("squeeze")),
        ("LIFT", _phase_start("lift")),
        ("HOLD", _phase_start("hold")),
        ("DISTURB", _phase_start("disturb")),
    ]
    # Drop boundaries that came back NaN (phase didn't occur).
    phase_boundaries = [(lbl, t) for lbl, t in phase_boundaries
                        if not (t != t)]  # NaN-safe

    fig, (ax_z, ax_f, ax_tilt) = plt.subplots(
        3, 1, figsize=(9, 10), sharex=True)

    for ax in (ax_z, ax_f, ax_tilt):
        # Shade the disturbance force pulse window.
        ax.axvspan(t_pulse_start, t_pulse_end,
                   color="orange", alpha=0.18, zorder=0)
        # Phase boundary vertical lines.
        for _label, t in phase_boundaries:
            ax.axvline(t, color="gray", linewidth=0.5,
                       linestyle=":", alpha=0.5, zorder=0)

    # ---------------------------------------------------------------
    # Panel 1: obj_z(t) with pad_z(t) reference
    # ---------------------------------------------------------------
    ax_z.plot(_t(history_cslc),
              [v * 1e3 for v in _s(history_cslc, "obj_z")],
              "-", color="tab:blue", linewidth=2.0, label="CSLC obj_z")
    ax_z.plot(_t(history_rigid),
              [v * 1e3 for v in _s(history_rigid, "obj_z")],
              "--", color="tab:red", linewidth=2.0, label="Rigid obj_z")
    # Pad target: use CSLC's pad trace as the canonical target -- both
    # methods drive the same joint trajectory so the pad_z paths
    # coincide to within solver noise.
    ax_z.plot(_t(history_cslc),
              [v * 1e3 for v in _s(history_cslc, "left_pad_z")],
              ":", color="black", linewidth=1.3,
              label="pad target z (= where the book should be)")
    ax_z.set_ylabel("Book z position [mm]")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc="best", fontsize=9)
    ax_z.set_title("Book z(t) vs pad target -- grip quality")

    # Annotate phase boundaries on the top panel only.
    y_max = ax_z.get_ylim()[1]
    for label, t in phase_boundaries:
        ax_z.text(t, y_max, label,
                  rotation=90, fontsize=7, color="gray",
                  va="top", ha="right", alpha=0.7)

    # ---------------------------------------------------------------
    # Panel 2: applied force(t) -- a square pulse
    # ---------------------------------------------------------------
    # The disturbance force is constant ``disturb_force_amplitude``
    # during the pulse window, zero everywhere else, and the same for
    # both methods. We synthesise the trace analytically from
    # SceneParams instead of reading per-frame telemetry -- the per-
    # frame ``pulse_active`` flag is 0/1 and would step quantise.
    t_all = sorted(set(_t(history_cslc) + _t(history_rigid)))
    f_trace = []
    for t in t_all:
        if t_pulse_start <= t < t_pulse_end:
            f_trace.append(p.disturb_force_amplitude)
        else:
            f_trace.append(0.0)
    ax_f.plot(t_all, f_trace,
              "-", color="tab:purple", linewidth=2.0,
              label=f"applied force = {p.disturb_force_amplitude:.1f} N along "
              + ("+X (perpendicular to book cover)" if use_y_axis else "+Y"))
    ax_f.fill_between(t_all, 0.0, f_trace,
                      color="tab:purple", alpha=0.2)
    ax_f.set_ylabel("Disturbance force [N]")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc="best", fontsize=9)
    ax_f.set_title(
        "Applied force on bottom band of the book "
        f"(pulse {p.disturb_force_duration * 1e3:.0f} ms)")

    # ---------------------------------------------------------------
    # Panel 3: tilt(t) -- rotational response
    # ---------------------------------------------------------------
    ax_tilt.plot(_t(history_cslc), _s(history_cslc, tilt_key),
                 "-", color="tab:blue", linewidth=2.0, label="CSLC")
    ax_tilt.plot(_t(history_rigid), _s(history_rigid, tilt_key),
                 "--", color="tab:red", linewidth=2.0, label="Rigid")
    ax_tilt.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax_tilt.set_xlabel("Time [s]")
    ax_tilt.set_ylabel(tilt_label)
    ax_tilt.grid(True, alpha=0.3)
    ax_tilt.legend(loc="best", fontsize=9)
    ax_tilt.set_title("Book tilt response -- rotational stiffness comparison")

    fig.suptitle(
        f"Book DISTURB test: m={p.obj_mass:.2f} kg, mu={p.mu:.2f}, "
        f"sq={p.squeeze_depth * 1e3:.1f} mm, "
        f"F={p.disturb_force_amplitude:.1f} N",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {output_path}")


def run_comparison(
    *,
    force: float,
    mass: float,
    mu: float,
    squeeze_depth: float,
    output_dir: Path,
    num_pads: int = 2,
    pad: str = "box",
    pulse_duration: float = 0.05,
    hold_duration: float | None = None,
    disturb_duration: float | None = None,
) -> Path:
    """Run CSLC + rigid back to back at the same setup and emit the
    PNG + per-method CSVs. Returns the PNG path.
    """
    overrides: dict = {
        "obj_mass": mass,
        "mu": mu,
        "squeeze_depth": squeeze_depth,
        "disturb_force_amplitude": force,
        "disturb_force_duration": pulse_duration,
    }
    if hold_duration is not None:
        overrides["hold_duration"] = hold_duration
    if disturb_duration is not None:
        overrides["disturb_duration"] = disturb_duration
    p = dataclasses.replace(SceneParams(), **overrides)

    output_dir.mkdir(parents=True, exist_ok=True)
    histories: dict[bool, list[dict]] = {}
    for use_cslc in (True, False):
        method = "cslc" if use_cslc else "rigid"
        print(f"\n=== book DISTURB {method}: m={mass:.3f}kg mu={mu:.2f} "
              f"sq={squeeze_depth * 1e3:.1f}mm F={force:.1f}N ===")
        hist = _run_once(p, use_cslc=use_cslc, pad=pad, num_pads=num_pads)
        csv_path = output_dir / (
            f"book_disturb_{method}_m_{mass:.2f}kg_mu_{mu:.2f}"
            f"_sq_{squeeze_depth * 1e3:.1f}mm_F_{force:.1f}N.csv"
        )
        _write_csv(hist, csv_path)
        print(f"  -> {csv_path.name} ({len(hist)} frames)")
        histories[use_cslc] = hist

    plot_path = output_dir / (
        f"book_disturb_m_{mass:.2f}kg_mu_{mu:.2f}"
        f"_sq_{squeeze_depth * 1e3:.1f}mm_F_{force:.1f}N.png"
    )
    plot_disturb_comparison(
        history_cslc=histories[True],
        history_rigid=histories[False],
        p=p,
        output_path=plot_path,
    )
    return plot_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="example_uxpbd_book_disturb",
        description=(
            "Run the book-tilt disturbance test for CSLC vs rigid "
            "contact and emit a tilt(t) / obj_y(t) / obj_z(t) PNG."
        ),
    )
    parser.add_argument("--force", type=float, default=5.0,
                        help="Peak transverse force amplitude [N]. The "
                             "sweet spot at default stiffnesses is "
                             "F in [2, 8] -- both methods hold but the "
                             "tilt difference is visible.")
    parser.add_argument("--mass", type=float, default=0.5,
                        help="Book mass [kg].")
    parser.add_argument("--mu", type=float, default=0.5,
                        help="Coulomb friction coefficient.")
    parser.add_argument("--squeeze-depth", type=float, default=0.015,
                        help="Grip compression depth [m].")
    parser.add_argument("--pulse-duration", type=float, default=0.05,
                        help="Force pulse width [s].")
    parser.add_argument("--hold-duration", type=float, default=None,
                        help="Stabilisation buffer between LIFT and "
                             "DISTURB [s]. Default: SceneParams value.")
    parser.add_argument("--disturb-duration", type=float, default=None,
                        help="DISTURB-phase length [s] (covers the "
                             "pulse window + ring-down). Default: "
                             "SceneParams value (1.5 s).")
    parser.add_argument("--num-pads", type=int, default=2,
                        choices=(0, 1, 2))
    parser.add_argument("--pad", choices=("box", "curved"), default="box")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "book_disturb",
                        help="Directory for the comparison CSVs + PNG.")
    args = parser.parse_args(argv)

    run_comparison(
        force=args.force,
        mass=args.mass,
        mu=args.mu,
        squeeze_depth=args.squeeze_depth,
        output_dir=args.output_dir,
        num_pads=args.num_pads,
        pad=args.pad,
        pulse_duration=args.pulse_duration,
        hold_duration=args.hold_duration,
        disturb_duration=args.disturb_duration,
    )


if __name__ == "__main__":
    main()
