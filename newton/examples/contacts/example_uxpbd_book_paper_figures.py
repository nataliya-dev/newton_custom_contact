# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Paper-quality CSLC vs rigid comparison figures, mimicking the
``cslc_main/theory/plot_dome_kl_sweep.py`` stylistic conventions
(LABEL_SIZE / TICK_SIZE / LEGEND_SIZE, ``lw=1.8+``, dpi=200,
``bbox_inches='tight'``).

Two separate PNGs are emitted:

  * **Figure 1 — Lift quality**
    `book_z(t)` for CSLC vs rigid, cropped to the LIFT phase
    onwards, with the pad's terminal z as a horizontal "target"
    reference. At ``m = 0.5 kg`` the CSLC grip carries the book to
    the target while the rigid pads slide past and drop the book.

  * **Figure 2 — Disturbance response**
    `tilt(t)` for CSLC vs rigid during the DISTURB phase, run at a
    lighter mass so the rigid grip also succeeds and both books are
    actually in the air when the transverse force pulse fires. The
    rotational-stiffness differential between CSLC's distributed
    compliant patch and the rigid sphere-lattice baseline is what
    the figure isolates.

Run::

    uv run python -m newton.examples.contacts.example_uxpbd_book_paper_figures \\
        --output-dir results/book_paper_figs
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .example_uxpbd_lift_test import SceneParams
from .example_uxpbd_lift_test_sweep import run_one_config


# ----- Style (matches plot_dome_kl_sweep.py) ---------------------------

LABEL_SIZE = 16
TICK_SIZE = 13
LEGEND_SIZE = 13
LINE_WIDTH = 2.4
DPI = 200
FIG_SIZE = (6.5, 5.0)

COLOUR_CSLC = "tab:blue"
COLOUR_RIGID = "tab:red"
COLOUR_TARGET = "0.30"
COLOUR_PULSE = "orange"


# ----- Phase helpers ---------------------------------------------------


def _phase_start(history: list[dict], phase: str) -> float | None:
    """World time [s] at which ``phase`` first appears, or ``None`` if
    the run never entered it."""
    for r in history:
        if r["phase"] == phase:
            return float(r["t"])
    return None


def _crop_from(history: list[dict], t0: float) -> list[dict]:
    return [r for r in history if r["t"] >= t0]


# ----- Figure 1: lift quality ------------------------------------------


def _crop_between(history: list[dict], t0: float,
                  t_end: float | None) -> list[dict]:
    """Return rows with t0 ≤ t < t_end (or t ≥ t0 if t_end is None)."""
    if t_end is None:
        return [r for r in history if r["t"] >= t0]
    return [r for r in history if t0 <= r["t"] < t_end]


def plot_lift_figure(
    history_cslc: list[dict],
    history_rigid: list[dict],
    output_path: Path,
) -> None:
    """One-panel ``Δz(t)`` vs target. Δz is anchored at the LIFT-start
    book z, so the curve begins at 0 mm. Target is the pad's terminal
    rise relative to the same anchor -- a horizontal line at the
    distance the book *should* travel if the grip carried it perfectly.

    Cropped to ``[LIFT_start, DISTURB_start)`` so the lift-quality
    figure is not contaminated by the disturbance-response transient
    (a separate figure handles that). With CSLC actually carrying
    load (post-Fix A), the disturbance kicks the pad bodies around
    after HOLD ends; including those frames blows out the y-axis
    scale and hides the lift kinematics the figure is meant to
    isolate.
    """
    import matplotlib.pyplot as plt

    t_lift_cslc = _phase_start(history_cslc, "lift")
    t_lift_rigid = _phase_start(history_rigid, "lift")
    if t_lift_cslc is None or t_lift_rigid is None:
        raise RuntimeError("LIFT phase missing from one of the histories")
    # Crop at DISTURB start so the lift-quality figure shows only the
    # LIFT + HOLD window. DISTURB is the subject of fig 2.
    t_d_cslc = _phase_start(history_cslc, "disturb")
    t_d_rigid = _phase_start(history_rigid, "disturb")

    cslc = _crop_between(history_cslc, t_lift_cslc, t_d_cslc)
    rigid = _crop_between(history_rigid, t_lift_rigid, t_d_rigid)

    # Anchor each Δz curve to its own object z at LIFT start. Pads
    # are driven by the same joint trajectory so the pad-rise target
    # is identical for both methods; take it from the CSLC run.
    z0_cslc = float(cslc[0]["obj_z"])
    z0_rigid = float(rigid[0]["obj_z"])
    pad_z0_cslc = float(cslc[0]["left_pad_z"])
    pad_target_delta_mm = (float(cslc[-1]["left_pad_z"]) - pad_z0_cslc) * 1e3

    t_cslc = [r["t"] - t_lift_cslc for r in cslc]
    t_rigid = [r["t"] - t_lift_rigid for r in rigid]
    dz_cslc = [(r["obj_z"] - z0_cslc) * 1e3 for r in cslc]
    dz_rigid = [(r["obj_z"] - z0_rigid) * 1e3 for r in rigid]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Target line first so the method curves overlay it.
    ax.axhline(pad_target_delta_mm, color=COLOUR_TARGET,
               linestyle=":", lw=2.0, label="target Δz", zorder=1)

    ax.plot(t_cslc, dz_cslc, "-", color=COLOUR_CSLC,
            lw=LINE_WIDTH, label="CSLC", zorder=3)
    ax.plot(t_rigid, dz_rigid, "--", color=COLOUR_RIGID,
            lw=LINE_WIDTH, label="rigid", zorder=2)

    ax.set_xlabel("time [s]", fontsize=LABEL_SIZE)
    ax.set_ylabel("book Δz [mm]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {output_path}")


# ----- Figure 2: disturbance response ----------------------------------


def plot_disturb_figure(
    history_cslc: list[dict],
    history_rigid: list[dict],
    p: SceneParams,
    output_path: Path,
    *,
    log_time: bool = True,
    log_floor_ms: float = 1.0,
) -> None:
    """One-panel ``tilt(t)`` during the DISTURB phase, with the pulse
    window shaded. Tilt axis auto-selected from the disturbance force
    direction.

    ``log_time=True`` puts the x axis on a log scale so the early
    transient response and the long settling tail are both legible on
    the same plot -- on a linear time axis the 50 ms pulse + 100 ms
    ring-down is crammed against the y axis while 1+ s of empty
    settled-state tail dominates the width. ``log_floor_ms`` is the
    smallest ``t > 0`` shown; anything earlier is clipped (log scale
    can't include 0 directly).
    """
    import matplotlib.pyplot as plt

    t_d_cslc = _phase_start(history_cslc, "disturb")
    t_d_rigid = _phase_start(history_rigid, "disturb")
    if t_d_cslc is None or t_d_rigid is None:
        raise RuntimeError("DISTURB phase missing from one of the histories")

    cslc = _crop_from(history_cslc, t_d_cslc)
    rigid = _crop_from(history_rigid, t_d_rigid)

    # Pick the tilt axis aligned with the disturbance direction. +X
    # force -> pitch about Y -> tilt_about_y_deg; +Y force -> roll
    # about X -> tilt_about_x_deg.
    use_y_axis = abs(p.disturb_force_dir_x) > abs(p.disturb_force_dir_y)
    tilt_key = "tilt_about_y_deg" if use_y_axis else "tilt_about_x_deg"

    # Time in ms relative to DISTURB start (matches the figure caption
    # convention in the existing paper draft).
    t_cslc = [(r["t"] - t_d_cslc) * 1e3 for r in cslc]
    t_rigid = [(r["t"] - t_d_rigid) * 1e3 for r in rigid]
    # Δtilt anchored at the disturb-start tilt for each run. Both
    # baselines may have a small non-zero pre-pulse tilt left over
    # from the LIFT->HOLD transition (especially the rigid baseline,
    # whose flat box face can't suppress book wobble during HOLD).
    # Normalising to Δ from the disturb start makes the actual force-
    # induced response readable instead of being biased by an
    # incidental pre-pulse offset.
    tilt0_cslc = float(cslc[0][tilt_key])
    tilt0_rigid = float(rigid[0][tilt_key])
    tilt_cslc = [r[tilt_key] - tilt0_cslc for r in cslc]
    tilt_rigid = [r[tilt_key] - tilt0_rigid for r in rigid]

    pulse_end_ms = p.disturb_force_duration * 1e3

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    if log_time:
        ax.set_xscale("log")
        # Shade the pulse window from the visible floor to its end.
        # The pulse actually starts at t=0 but a log axis can't show
        # that point; the shaded band makes the window's *extent*
        # readable even though its left edge is compressed.
        ax.axvspan(log_floor_ms, pulse_end_ms, color=COLOUR_PULSE,
                   alpha=0.18, zorder=0,
                   label=f"force pulse ({p.disturb_force_amplitude:.1f} N)")
    else:
        ax.axvspan(0.0, pulse_end_ms, color=COLOUR_PULSE,
                   alpha=0.18, zorder=0,
                   label=f"force pulse ({p.disturb_force_amplitude:.1f} N)")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5, zorder=1)

    ax.plot(t_cslc, tilt_cslc, "-", color=COLOUR_CSLC,
            lw=LINE_WIDTH, label="CSLC", zorder=3)
    ax.plot(t_rigid, tilt_rigid, "--", color=COLOUR_RIGID,
            lw=LINE_WIDTH, label="rigid", zorder=2)

    if log_time:
        # Constrain the lower bound so the y axis isn't dominated by
        # the t<floor region; the upper bound expands to fit the
        # longest run.
        t_max = max(t_cslc[-1] if t_cslc else log_floor_ms,
                    t_rigid[-1] if t_rigid else log_floor_ms)
        ax.set_xlim(log_floor_ms, t_max)

    ax.set_xlabel("time [ms]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Δtilt [deg]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=LEGEND_SIZE, framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {output_path}")


def plot_disturb_dz_figure(
    history_cslc: list[dict],
    history_rigid: list[dict],
    p: SceneParams,
    output_path: Path,
    *,
    log_time: bool = True,
    log_floor_ms: float = 1.0,
) -> None:
    """Companion to :func:`plot_disturb_figure`: book ``Δz(t)`` during
    the DISTURB phase. The rigid point contact lets the book slide out
    of the grasp under the transverse pulse (large negative Δz), while
    CSLC's distributed patch holds the book at its grip height (Δz ≈ 0).

    Same axes/style as the Δtilt figure (log time [ms] from DISTURB
    start, pulse window shaded, Δ anchored at the disturb-start height)
    so the two read as a matched pair.
    """
    import matplotlib.pyplot as plt

    t_d_cslc = _phase_start(history_cslc, "disturb")
    t_d_rigid = _phase_start(history_rigid, "disturb")
    if t_d_cslc is None or t_d_rigid is None:
        raise RuntimeError("DISTURB phase missing from one of the histories")

    cslc = _crop_from(history_cslc, t_d_cslc)
    rigid = _crop_from(history_rigid, t_d_rigid)

    t_cslc = [(r["t"] - t_d_cslc) * 1e3 for r in cslc]
    t_rigid = [(r["t"] - t_d_rigid) * 1e3 for r in rigid]
    # Δz anchored at each run's disturb-start height, in mm. A drop reads
    # as negative Δz (book sliding down out of the grip).
    z0_cslc = float(cslc[0]["obj_z"])
    z0_rigid = float(rigid[0]["obj_z"])
    dz_cslc = [(r["obj_z"] - z0_cslc) * 1e3 for r in cslc]
    dz_rigid = [(r["obj_z"] - z0_rigid) * 1e3 for r in rigid]

    pulse_end_ms = p.disturb_force_duration * 1e3

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    if log_time:
        ax.set_xscale("log")
        ax.axvspan(log_floor_ms, pulse_end_ms, color=COLOUR_PULSE,
                   alpha=0.18, zorder=0,
                   label=f"force pulse ({p.disturb_force_amplitude:.1f} N)")
    else:
        ax.axvspan(0.0, pulse_end_ms, color=COLOUR_PULSE,
                   alpha=0.18, zorder=0,
                   label=f"force pulse ({p.disturb_force_amplitude:.1f} N)")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5, zorder=1)

    ax.plot(t_cslc, dz_cslc, "-", color=COLOUR_CSLC,
            lw=LINE_WIDTH, label="CSLC", zorder=3)
    ax.plot(t_rigid, dz_rigid, "--", color=COLOUR_RIGID,
            lw=LINE_WIDTH, label="rigid", zorder=2)

    if log_time:
        t_max = max(t_cslc[-1] if t_cslc else log_floor_ms,
                    t_rigid[-1] if t_rigid else log_floor_ms)
        ax.set_xlim(log_floor_ms, t_max)

    ax.set_xlabel("time [ms]", fontsize=LABEL_SIZE)
    ax.set_ylabel("book Δz [mm]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=LEGEND_SIZE, framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {output_path}")


# ----- Runner ----------------------------------------------------------


def generate_figures(
    *,
    output_dir: Path,
    lift_mass: float,
    disturb_mass: float,
    mu: float,
    force: float,
    pad: str,
    num_pads: int,
    do_lift: bool = True,
) -> tuple[Path | None, Path, Path]:
    """Run the experiments and emit the PNGs. Returns
    ``(lift_path_or_None, disturb_tilt_path, disturb_dz_path)``.

    ``do_lift=False`` skips the lift figure (and its two sims) so the
    disturbance pair can be iterated on without recomputing the lift.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1_path: Path | None = None
    if do_lift:
        # --- Figure 1: lift quality ---
        print(f"\n== Figure 1: lift comparison (m={lift_mass:.2f} kg) ==")
        p_lift = dataclasses.replace(SceneParams(), obj_mass=lift_mass, mu=mu)
        hist_lift_cslc = run_one_config(p_lift, object="book", pad=pad,
                                        num_pads=num_pads, use_cslc=True)
        hist_lift_rigid = run_one_config(p_lift, object="book", pad=pad,
                                         num_pads=num_pads, use_cslc=False)
        fig1_path = output_dir / "fig_book_lift.png"
        plot_lift_figure(hist_lift_cslc, hist_lift_rigid, fig1_path)

    # --- Figure 2: disturbance response (tilt + Δz companion) ---
    print(f"\n== Figure 2: disturbance response (m={disturb_mass:.2f} kg, "
          f"F={force:.1f} N) ==")
    p_dist = dataclasses.replace(
        SceneParams(),
        obj_mass=disturb_mass, mu=mu,
        disturb_force_amplitude=force,
    )
    hist_dist_cslc = run_one_config(p_dist, object="book", pad=pad,
                                    num_pads=num_pads, use_cslc=True)
    hist_dist_rigid = run_one_config(p_dist, object="book", pad=pad,
                                     num_pads=num_pads, use_cslc=False)
    fig2_path = output_dir / "fig_book_disturb.png"
    plot_disturb_figure(hist_dist_cslc, hist_dist_rigid, p_dist, fig2_path)
    fig2dz_path = output_dir / "fig_book_disturb_dz.png"
    plot_disturb_dz_figure(hist_dist_cslc, hist_dist_rigid, p_dist, fig2dz_path)

    return fig1_path, fig2_path, fig2dz_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="example_uxpbd_book_paper_figures",
        description=(
            "Generate two paper-quality figures: book z(t) lift "
            "comparison and book tilt(t) disturbance response."
        ),
    )
    parser.add_argument("--lift-mass", type=float, default=0.5,
                        help="Book mass [kg] for the lift figure. "
                             "0.5 kg keeps both grips in regime; CSLC "
                             "tracks smoothly to the converged height "
                             "while rigid overshoots and oscillates.")
    parser.add_argument("--disturb-mass", type=float, default=0.3,
                        help="Book mass [kg] for the disturbance "
                             "figure. 0.3 kg keeps both grips clean so "
                             "the response is recorded in mid-air.")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Coulomb friction coefficient. 1.0 "
                             "(the SceneParams default) keeps the "
                             "grip in the static-friction regime for "
                             "both methods on a 0.5 kg book.")
    parser.add_argument("--force", type=float, default=3.0,
                        help=(
                            "Peak transverse force amplitude [N]. The "
                            "default (3 N) sits in CSLC's elastic "
                            "regime: the lattice anchor absorbs the "
                            "impulse and the book Δtilt stays within "
                            "±0.1°, while the rigid box transmits the "
                            "full impulse and oscillates at ±0.4° with "
                            "persistent ring-down. Raise above ~10 N "
                            "to push CSLC past elastic and into the "
                            "lattice-saturation regime where both "
                            "methods tilt comparably."
                        ))
    parser.add_argument("--pad", choices=("box", "curved"), default="box")
    parser.add_argument("--num-pads", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--disturb-only", action="store_true",
                        help="Skip the lift figure (and its 2 sims); emit "
                             "only the disturbance pair (tilt + Δz). Faster "
                             "when iterating on the disturbance story.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "book_paper_figs")
    args = parser.parse_args(argv)

    fig1, fig2, fig2dz = generate_figures(
        output_dir=args.output_dir,
        lift_mass=args.lift_mass,
        disturb_mass=args.disturb_mass,
        mu=args.mu,
        force=args.force,
        pad=args.pad,
        num_pads=args.num_pads,
        do_lift=not args.disturb_only,
    )
    print()
    print("Figures saved:")
    if fig1 is not None:
        print(f"  Figure 1 (lift quality):       {fig1}")
    print(f"  Figure 2 (disturbance tilt):   {fig2}")
    print(f"  Figure 2b (disturbance Δz):    {fig2dz}")


if __name__ == "__main__":
    main()
