# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# UXPBD Lift Test -- parameter sweep runner
#
# Two complementary sweeps over the same scene used in
# ``example_uxpbd_lift_test``:
#
#   --sweep mass         vary the object mass at a fixed compression
#                        depth. Expectation: rigid-lattice grasp slips /
#                        drops at a lower mass than the CSLC-equipped
#                        grasp because CSLC deforms its skin and
#                        recruits more contact pairs (more available
#                        friction).
#
#   --sweep compression  vary the squeeze depth at a fixed mass.
#                        Expectation: at light compression both
#                        variants slip; at heavy compression both hold;
#                        the *transition* depth at which each variant
#                        first holds is the result of interest.
#
# Each (sweep_var, value, cslc_on/off) combination runs the same full
# settle -> approach -> squeeze -> lift -> hold sequence headlessly via
# ``newton.viewer.ViewerNull`` and writes a per-frame CSV row containing
# obj/pad pose, slip, lift, and lattice compression diagnostics. A
# stdout summary table at the end prints which configs held vs which
# dropped the object using the same thresholds as
# ``Example.test_final``.
#
# Object / pad geometry is pulled through ``OBJECT_BUILDERS`` /
# ``PAD_BUILDERS`` in example_uxpbd_lift_test.py so swapping a new
# object kind (register it via ``register_object_builder``) makes it
# immediately sweep-compatible -- no edits needed in this file.
#
# Command (mass sweep, defaults):
#   uv run python -m newton.examples.contacts.example_uxpbd_lift_test_sweep \
#       --sweep mass --values 0.5,1,2,4,8,16
#
# Command (compression sweep, mm input):
#   uv run python -m newton.examples.contacts.example_uxpbd_lift_test_sweep \
#       --sweep compression --values 5,10,15,20,25,30 --obj-mass 4
###########################################################################

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer

from .example_uxpbd_lift_test import (
    OBJECT_BUILDERS,
    PAD_BUILDERS,
    Example,
    SceneParams,
)


# Slip / drop thresholds: kept aligned with the assertions in
# Example.test_final so "held" in the sweep means the same thing as a
# passing test_final on that config. If you tighten the example test,
# tighten these in lock-step.
DROP_TOL: float = 5.0e-3   # obj_z below obj_z0 by this much -> dropped
SLIP_TOL: float = 1.0e-2   # |pad_lift - obj_lift| above -> slipped


# ----- Internal helpers ------------------------------------------------


def _make_args(*, num_pads: int, object: str, pad: str,
               no_compliant: bool) -> argparse.Namespace:
    """Construct the minimal argparse.Namespace ``Example.__init__``
    inspects. ``test`` is added so any future test-mode branch in
    Example sees a defined attribute (no AttributeError)."""
    return argparse.Namespace(
        num_pads=num_pads,
        object=object,
        pad=pad,
        no_compliant=no_compliant,
        test=False,
    )


def _verdict(history: list[dict]) -> dict:
    """Classify the run from the telemetry history.

    The metrics in ``Example.history`` are anchored to spawn positions
    (the values ``test_final`` uses for its single-config asserts). For
    sweep verdicts we instead anchor to the **start-of-LIFT** frame:
    spawn-anchored slip double-counts the SETTLE drop (the ball falls
    a few cm onto the ground before the pad ever touches it) and makes
    every CSLC run look like it slipped by ~35 mm when in fact the
    object tracked the pad to within ~1 mm. The lift-anchored metrics
    are what "did the grasp survive the lift?" actually means.

    Returned dict carries both spawn-anchored (final_obj_lift,
    final_pad_lift, final_slip) and lift-anchored
    (lift_obj_lift, lift_pad_lift, lift_slip) measurements so plots and
    later analysis can pick the right one. Verdict is decided from the
    lift-anchored slip.
    """
    nan_row = {
        "verdict": "no_data",
        "final_obj_z": float("nan"),
        "final_slip": float("nan"),
        "final_obj_lift": float("nan"),
        "final_pad_lift": float("nan"),
        "final_n_active": 0,
        "final_delta_max": 0.0,
        "lift_slip": float("nan"),
        "lift_obj_lift": float("nan"),
        "lift_pad_lift": float("nan"),
    }
    if not history:
        return nan_row
    final = history[-1]
    # Find the first row tagged "lift". If the scene never reached LIFT
    # (shouldn't happen for a full-length run), fall back to spawn.
    lift_start = next(
        (r for r in history if r["phase"] == "lift"),
        None,
    )
    if lift_start is None:
        lift_obj_z0 = history[0]["obj_z"]
        lift_pad_z0 = history[0]["left_pad_z"]
    else:
        lift_obj_z0 = lift_start["obj_z"]
        lift_pad_z0 = lift_start["left_pad_z"]
    lift_obj_lift = final["obj_z"] - lift_obj_z0
    lift_pad_lift = final["left_pad_z"] - lift_pad_z0
    lift_slip = lift_pad_lift - lift_obj_lift

    # Continuous "grip ratio": what fraction of the pad's motion did
    # the object follow? 1.0 = perfect grip, 0.0 = full slip, negative
    # = object went the wrong way. Clipped to [-0.2, 1.5] to bound
    # plot ranges when the pad barely moves (denominator ≈ 0).
    if abs(lift_pad_lift) > 1.0e-4:
        grip_ratio = lift_obj_lift / lift_pad_lift
    else:
        grip_ratio = float("nan")  # pad never lifted; ratio undefined

    # Verdict: held if slip is inside tolerance; otherwise classify by
    # whether the object actually rose at all. "Dropped" reserves the
    # strong claim that the pad lifted significantly while the object
    # did not (lift ratio < 30%). "Slipped" covers the in-between case
    # where the object followed the pad but lagged measurably.
    if abs(lift_slip) <= SLIP_TOL:
        verdict = "held"
    elif lift_pad_lift > 1.0e-3 and lift_obj_lift < 0.3 * lift_pad_lift:
        verdict = "dropped"
    else:
        verdict = "slipped"

    return {
        "verdict": verdict,
        # Spawn-anchored (matches Example.test_final semantics):
        "final_obj_z": final["obj_z"],
        "final_slip": final["slip"],
        "final_obj_lift": final["obj_lift"],
        "final_pad_lift": final["pad_lift"],
        "final_n_active": final["n_active"],
        "final_delta_max": final["delta_max"],
        # Lift-anchored (what the verdict is actually computed from):
        "lift_slip": lift_slip,
        "lift_obj_lift": lift_obj_lift,
        "lift_pad_lift": lift_pad_lift,
        # Continuous grip metric (preferred for sweep plots):
        "grip_ratio": grip_ratio,
    }


def _write_csv(history: list[dict], path: Path) -> None:
    if not history:
        path.write_text("")
        return
    keys = list(history[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow(row)


def _apply_overrides(params: SceneParams, *, mu: float | None) -> SceneParams:
    """Apply optional per-sweep overrides on top of the supplied
    ``SceneParams``. Currently only friction; add more here when new
    sweep axes show up so the call sites stay one-liners."""
    if mu is not None:
        params = dataclasses.replace(params, mu=float(mu))
    return params


def run_one_config(
    params: SceneParams,
    *,
    object: str = "sphere",
    pad: str = "box",
    num_pads: int = 2,
    use_cslc: bool = True,
) -> list[dict]:
    """Run one headless instance of the lift-test scene and return
    ``Example.history`` (one dict per frame).

    Parameters
    ----------
    params:
        Fully-specified ``SceneParams``. The sweep callers build this
        per-config via ``dataclasses.replace(SceneParams(), ...)`` so
        every knob the lift test exposes is reachable.
    object, pad:
        Keys into ``OBJECT_BUILDERS`` / ``PAD_BUILDERS``. Custom
        builders registered via ``register_object_builder`` are
        accepted without further plumbing.
    use_cslc:
        When False, ``args.no_compliant`` is set so the solver runs the
        rigid-lattice baseline (Phase 1 path). When True, the CSLC
        path runs with the per-sphere stiffnesses in ``params``.
    """
    if object not in OBJECT_BUILDERS:
        raise ValueError(
            f"unknown object {object!r}; registered: "
            f"{sorted(OBJECT_BUILDERS)}")
    if pad not in PAD_BUILDERS:
        raise ValueError(
            f"unknown pad {pad!r}; registered: {sorted(PAD_BUILDERS)}")

    # +10 frames of headroom on ViewerNull so the loop never trips its
    # own frame-count guard before we drive total_frames externally.
    viewer = newton.viewer.ViewerNull(num_frames=params.total_frames + 10)
    args = _make_args(
        num_pads=num_pads,
        object=object,
        pad=pad,
        no_compliant=not use_cslc,
    )
    ex = Example(viewer, args, params=params)
    # Use the *post-override* SceneParams the Example actually runs
    # with -- ``Example.__init__`` may rewrite phase durations for
    # certain object kinds (e.g. ``--object book`` shortens LIFT), so
    # ``ex.p.total_frames`` is the canonical frame budget. Using the
    # caller-passed ``params.total_frames`` here would either undershoot
    # (skip frames) or overshoot (run extra frames in the last phase).
    for _ in range(ex.p.total_frames):
        ex.step()
    return ex.history


# ----- Sweep drivers ---------------------------------------------------


def mass_sweep(
    masses: list[float],
    *,
    squeeze_depth: float,
    object: str,
    pad: str,
    output_dir: Path,
    num_pads: int = 2,
    mu: float | None = None,
) -> list[dict]:
    """For each ``m`` in ``masses`` and each (cslc on/off), run the
    full scene at ``squeeze_depth`` and write a per-frame CSV.

    ``mu`` overrides ``SceneParams.mu`` if given; ``None`` keeps the
    scene default. Returns the per-config records used to build the
    stdout summary and (optionally) the plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for m in masses:
        for use_cslc in (True, False):
            params = dataclasses.replace(
                SceneParams(),
                obj_mass=float(m),
                squeeze_depth=float(squeeze_depth),
            )
            params = _apply_overrides(params, mu=mu)
            tag = (f"mass_{m:08.4f}kg"
                   f"_mu_{params.mu:0.2f}"
                   f"_cslc_{int(use_cslc)}")
            print(f"\n=== sweep=mass mass={m:.4f} kg "
                  f"compression={squeeze_depth * 1e3:.1f} mm "
                  f"mu={params.mu:.2f} cslc={use_cslc} ===")
            history = run_one_config(
                params, object=object, pad=pad,
                num_pads=num_pads, use_cslc=use_cslc)
            csv_path = output_dir / f"{tag}.csv"
            _write_csv(history, csv_path)
            v = _verdict(history)
            print(f"  -> {csv_path.name} | verdict={v['verdict']:>7s} "
                  f"obj_z={v['final_obj_z']:+.4f} "
                  f"slip={v['final_slip'] * 1e3:+.2f}mm "
                  f"n_active={v['final_n_active']} "
                  f"delta_max={v['final_delta_max'] * 1e3:.3f}mm")
            results.append({
                "sweep": "mass",
                "mass": float(m),
                "squeeze_depth": float(squeeze_depth),
                "mu": float(params.mu),
                "cslc": use_cslc,
                "csv": str(csv_path),
                **v,
            })
    _print_summary(results, sweep_var="mass", unit_scale=1.0, unit_label="kg")
    return results


def compression_sweep(
    depths: list[float],
    *,
    obj_mass: float,
    object: str,
    pad: str,
    output_dir: Path,
    num_pads: int = 2,
    mu: float | None = None,
) -> list[dict]:
    """For each ``d`` in ``depths`` [m] and each (cslc on/off), run the
    full scene at ``obj_mass`` kg and write a per-frame CSV.

    ``mu`` overrides ``SceneParams.mu`` if given.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for d in depths:
        for use_cslc in (True, False):
            params = dataclasses.replace(
                SceneParams(),
                obj_mass=float(obj_mass),
                squeeze_depth=float(d),
            )
            params = _apply_overrides(params, mu=mu)
            tag = (f"squeeze_{d * 1e3:06.2f}mm"
                   f"_mu_{params.mu:0.2f}"
                   f"_cslc_{int(use_cslc)}")
            print(f"\n=== sweep=compression squeeze={d * 1e3:.2f} mm "
                  f"mass={obj_mass:.3f} kg "
                  f"mu={params.mu:.2f} cslc={use_cslc} ===")
            history = run_one_config(
                params, object=object, pad=pad,
                num_pads=num_pads, use_cslc=use_cslc)
            csv_path = output_dir / f"{tag}.csv"
            _write_csv(history, csv_path)
            v = _verdict(history)
            print(f"  -> {csv_path.name} | verdict={v['verdict']:>7s} "
                  f"obj_z={v['final_obj_z']:+.4f} "
                  f"slip={v['final_slip'] * 1e3:+.2f}mm "
                  f"n_active={v['final_n_active']} "
                  f"delta_max={v['final_delta_max'] * 1e3:.3f}mm")
            results.append({
                "sweep": "compression",
                "squeeze_depth": float(d),
                "mass": float(obj_mass),
                "mu": float(params.mu),
                "cslc": use_cslc,
                "csv": str(csv_path),
                **v,
            })
    _print_summary(results, sweep_var="squeeze_depth",
                   unit_scale=1.0e3, unit_label="mm")
    return results


def _plot_sweep(results: list[dict], *, sweep_var: str,
                unit_scale: float, unit_label: str,
                output_path: Path) -> None:
    """Three-panel comparison plot saved as PNG.

    Panel 1: ``obj_rise`` (lift-anchored) and ``pad_rise`` overlaid.
    Panel 2: slip = ``pad_rise - obj_rise``. Zero = no slip.
    Panel 3: continuous ``grip_ratio = obj_rise / pad_rise``. 1.0 =
        perfect grip, 0.0 = full slip. The most informative panel for
        spotting the transition regime.

    All three panels use lift-anchored metrics (post-SETTLE / start of
    LIFT baseline) so the SETTLE drop doesn't contaminate the curves.
    Lines are plotted continuously (no per-point verdict markers on
    the curves themselves); a tiny verdict-color stripe below each
    point records the categorical class without breaking the line.

    Imports matplotlib lazily so the sweep itself doesn't require it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[plot] matplotlib not available ({exc}); skipping plot. "
              f"Install with: uv pip install matplotlib")
        return

    cslc_on = sorted(
        (r for r in results if r["cslc"]),
        key=lambda r: r[sweep_var],
    )
    cslc_off = sorted(
        (r for r in results if not r["cslc"]),
        key=lambda r: r[sweep_var],
    )

    def _xs(rows: list[dict]) -> list[float]:
        return [r[sweep_var] * unit_scale for r in rows]

    fig, (ax_lift, ax_slip, ax_grip) = plt.subplots(
        3, 1, figsize=(8, 10), sharex=True)

    # --- Panel 1: obj_rise vs sweep variable, with pad_rise overlay ---
    for label, rows, color, ls in (
        ("CSLC", cslc_on, "tab:blue", "-"),
        ("Rigid", cslc_off, "tab:red", "--"),
    ):
        xs = _xs(rows)
        ys = [r["lift_obj_lift"] * 1e3 for r in rows]
        ax_lift.plot(xs, ys, ls + "o", color=color, label=f"{label} obj rise",
                     linewidth=1.8, markersize=5, zorder=2)
    if cslc_on:
        ax_lift.plot(
            _xs(cslc_on),
            [r["lift_pad_lift"] * 1e3 for r in cslc_on],
            ":", color="gray", label="Pad rise (ideal target)",
            linewidth=1.5, zorder=1,
        )
    ax_lift.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax_lift.set_ylabel("obj rise during lift+hold [mm]")
    ax_lift.grid(True, alpha=0.3)
    ax_lift.legend(loc="best", fontsize=9)

    # --- Panel 2: slip vs sweep variable ---
    for label, rows, color, ls in (
        ("CSLC", cslc_on, "tab:blue", "-"),
        ("Rigid", cslc_off, "tab:red", "--"),
    ):
        xs = _xs(rows)
        ys = [r["lift_slip"] * 1e3 for r in rows]
        ax_slip.plot(xs, ys, ls + "o", color=color, label=f"{label} slip",
                     linewidth=1.8, markersize=5, zorder=2)
    ax_slip.axhspan(-SLIP_TOL * 1e3, SLIP_TOL * 1e3,
                    color="green", alpha=0.08, zorder=0,
                    label=f"|slip| < {SLIP_TOL * 1e3:.1f}mm (held band)")
    ax_slip.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax_slip.set_ylabel("slip = pad_rise - obj_rise [mm]")
    ax_slip.grid(True, alpha=0.3)
    ax_slip.legend(loc="best", fontsize=9)

    # --- Panel 3: grip_ratio (continuous), the headline metric ---
    for label, rows, color, ls in (
        ("CSLC", cslc_on, "tab:blue", "-"),
        ("Rigid", cslc_off, "tab:red", "--"),
    ):
        xs = _xs(rows)
        ys = [r["grip_ratio"] for r in rows]
        ax_grip.plot(xs, ys, ls + "o", color=color, label=f"{label} grip ratio",
                     linewidth=1.8, markersize=5, zorder=2)
    # Reference bands so the regime is readable at a glance.
    ax_grip.axhline(1.0, color="gray", linewidth=0.8, linestyle=":",
                    alpha=0.6, label="perfect grip")
    ax_grip.axhline(0.0, color="gray", linewidth=0.8, linestyle=":",
                    alpha=0.6, label="full slip")
    ax_grip.set_xlabel(f"{sweep_var} [{unit_label}]")
    ax_grip.set_ylabel("grip ratio = obj_rise / pad_rise")
    ax_grip.set_ylim(-0.2, 1.3)
    ax_grip.grid(True, alpha=0.3)
    ax_grip.legend(loc="best", fontsize=9)

    # Title summarises the held-fixed config so the same panel can be
    # filed under multiple mu / compression / pad combinations. Use
    # .get() because the replot path doesn't always re-derive every
    # held-fixed value (it only carries what's encoded in the filename).
    if results:
        r0 = results[0]
        mu_str = f"mu={r0.get('mu', float('nan')):.2f}"
        if sweep_var == "mass":
            sq = r0.get("squeeze_depth")
            held = (f"compression={sq * 1e3:.1f}mm {mu_str}"
                    if sq is not None else mu_str)
        else:
            m = r0.get("mass")
            held = (f"mass={m:.2f}kg {mu_str}"
                    if m is not None else mu_str)
        fig.suptitle(
            f"Lift test sweep over {sweep_var}  ({held})",
            fontsize=12)

    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] saved -> {output_path}")


def _print_summary(results: list[dict], *, sweep_var: str,
                   unit_scale: float, unit_label: str) -> None:
    """Stdout table comparing CSLC on/off at each swept value.

    Shows the lift-anchored metric (``lift_*``) because that's what
    drives the verdict. The trailing columns expose ``n_active`` and
    ``delta_max`` so you can spot CSLC participation independent of
    whether the grasp held.
    """
    print("\n" + "=" * 84)
    print(f"SUMMARY ({sweep_var}, {unit_label})  -- metrics anchored to start of LIFT")
    print("=" * 84)
    header = (
        f"{'value':>10s} | {'cslc':>4s} | {'verdict':>7s} | "
        f"{'obj_rise':>9s} | {'pad_rise':>9s} | {'slip':>9s} | "
        f"{'grip':>6s} | {'n_act':>5s} | {'d_max':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        val = r[sweep_var] * unit_scale
        cslc = "ON" if r["cslc"] else "off"
        gr = r.get("grip_ratio", float("nan"))
        gr_str = "nan" if gr != gr else f"{gr:+.3f}"  # NaN-safe formatter
        print(
            f"{val:>10.4f} | {cslc:>4s} | {r['verdict']:>7s} | "
            f"{r['lift_obj_lift'] * 1e3:>+8.2f}mm | "
            f"{r['lift_pad_lift'] * 1e3:>+8.2f}mm | "
            f"{r['lift_slip'] * 1e3:>+8.2f}mm | "
            f"{gr_str:>6s} | "
            f"{r['final_n_active']:>5d} | "
            f"{r['final_delta_max'] * 1e3:>6.3f}mm"
        )
    print("=" * 84)


def _history_from_csv(path: Path) -> list[dict]:
    """Reload a per-frame CSV into the same list-of-dicts shape that
    ``Example.history`` produces. Used by :func:`replot_from_csv_dir`
    so an existing sweep output directory can be re-analysed without
    re-running the simulation (useful when the verdict logic or the
    plotting changes).
    """
    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict = {}
            for k, v in raw.items():
                if k in ("phase",):
                    row[k] = v
                elif k in ("frame", "n_active", "lattice_sphere_count"):
                    row[k] = int(v)
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


def replot_from_csv_dir(
    output_dir: Path,
    *,
    sweep_var: str = "mass",
    unit_scale: float | None = None,
    unit_label: str | None = None,
    plot_name: str | None = None,
) -> Path:
    """Re-derive verdicts + plot from already-written CSVs in
    ``output_dir``. Filenames are expected to follow the
    ``mass_<kg>kg_mu_<mu>_cslc_<0|1>.csv`` /
    ``squeeze_<mm>mm_mu_<mu>_cslc_<0|1>.csv`` patterns this script
    emits. Returns the saved PNG path.
    """
    if sweep_var == "mass":
        unit_scale = unit_scale if unit_scale is not None else 1.0
        unit_label = unit_label or "kg"
        prefix = "mass_"
    elif sweep_var == "squeeze_depth":
        unit_scale = unit_scale if unit_scale is not None else 1.0e3
        unit_label = unit_label or "mm"
        prefix = "squeeze_"
    else:
        raise ValueError(f"unknown sweep_var: {sweep_var!r}")

    results: list[dict] = []
    for csv_path in sorted(output_dir.glob(f"{prefix}*.csv")):
        # Filename schema: <prefix><value><suffix>_mu_<mu>_cslc_<0|1>.csv
        name = csv_path.stem
        try:
            parts = name.split("_")
            cslc = bool(int(parts[-1]))
            mu = float(parts[-3])
            # Find the value token: 2nd index for mass, varies for squeeze.
            if sweep_var == "mass":
                # mass_<X.YYYY>kg_mu_<>_cslc_<>
                val = float(parts[1].rstrip("kg"))
            else:
                val = float(parts[1].rstrip("mm")) * 1.0e-3
        except (IndexError, ValueError) as exc:
            print(f"[replot] skipping {csv_path.name}: {exc}")
            continue
        hist = _history_from_csv(csv_path)
        v = _verdict(hist)
        results.append({
            "sweep": sweep_var,
            sweep_var: val,
            "mu": mu,
            "cslc": cslc,
            "csv": str(csv_path),
            **v,
        })

    if not results:
        raise FileNotFoundError(f"no CSVs matching {prefix}* in {output_dir}")

    # Print the refreshed verdict table so the caller can compare.
    _print_summary(
        results,
        sweep_var=sweep_var,
        unit_scale=unit_scale,
        unit_label=unit_label,
    )

    if plot_name is None:
        mu = results[0]["mu"]
        if sweep_var == "mass":
            plot_name = f"mass_sweep_mu_{mu:.2f}.png"
        else:
            plot_name = f"compression_sweep_mu_{mu:.2f}.png"
    plot_path = output_dir / plot_name
    _plot_sweep(
        results,
        sweep_var=sweep_var,
        unit_scale=unit_scale,
        unit_label=unit_label,
        output_path=plot_path,
    )
    return plot_path


# ----- CLI -------------------------------------------------------------


def _parse_values(spec: str) -> list[float]:
    """Parse a comma-separated list of floats. Trims whitespace; ignores
    empty entries so trailing commas don't error."""
    out: list[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise argparse.ArgumentTypeError(f"--values is empty: {spec!r}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="example_uxpbd_lift_test_sweep",
        description=(
            "Sweep object mass or compression depth over the UXPBD "
            "lift-test scene with CSLC on/off; emit per-frame CSV "
            "telemetry per config."
        ),
    )
    parser.add_argument(
        "--sweep", required=True, choices=("mass", "compression"),
        help=(
            "Which sweep to run. 'mass' holds compression fixed at "
            "--squeeze-depth and varies obj_mass over --values [kg]. "
            "'compression' holds mass fixed at --obj-mass and varies "
            "squeeze_depth over --values [mm]."
        ),
    )
    parser.add_argument(
        "--values", required=True, type=_parse_values,
        help=(
            "Comma-separated values. Units depend on --sweep: kg for "
            "mass, mm for compression. Example: '0.5,1,2,4,8,16' for "
            "a mass sweep; '5,10,15,20,25,30' for compression."
        ),
    )
    parser.add_argument(
        "--object", default="sphere", choices=tuple(OBJECT_BUILDERS),
        help=(
            "Object kind from OBJECT_BUILDERS. Defaults to 'sphere' "
            "(MorphIt packing). Register custom kinds via "
            "register_object_builder() and they appear here."
        ),
    )
    parser.add_argument(
        "--pad", default="box", choices=tuple(PAD_BUILDERS),
        help=(
            "Pad kind from PAD_BUILDERS. Defaults to 'box' for "
            "speed (~32 spheres per pad vs ~125 for 'curved')."
        ),
    )
    parser.add_argument(
        "--num-pads", type=int, default=2, choices=(0, 1, 2),
        help=(
            "Number of pads. Default 2 (full grasp). Lower values "
            "reuse the example's debug-staging modes."
        ),
    )
    parser.add_argument(
        "--obj-mass", type=float, default=1.0,
        help="Fixed object mass [kg] for the compression sweep.",
    )
    parser.add_argument(
        "--squeeze-depth", type=float, default=0.015,
        help="Fixed compression depth [m] for the mass sweep.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results") / "lift_sweep",
        help="Directory for per-config CSV files. Created if missing.",
    )
    parser.add_argument(
        "--mu", type=float, default=None,
        help=(
            "Override SceneParams.mu (Coulomb friction). When unset, "
            "the scene default (1.0) is used. Apply per sweep, e.g. "
            "--mu 0.5 to reduce available friction by half."
        ),
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help=(
            "Skip matplotlib comparison plot at end. Default is to "
            "save a PNG alongside the CSVs."
        ),
    )
    args = parser.parse_args(argv)

    if args.sweep == "mass":
        results = mass_sweep(
            args.values,
            squeeze_depth=args.squeeze_depth,
            object=args.object,
            pad=args.pad,
            output_dir=args.output_dir,
            num_pads=args.num_pads,
            mu=args.mu,
        )
        plot_path = args.output_dir / (
            f"mass_sweep_mu_{results[0]['mu']:.2f}_"
            f"sq_{args.squeeze_depth * 1e3:.1f}mm.png"
        )
        if not args.no_plot:
            _plot_sweep(
                results, sweep_var="mass",
                unit_scale=1.0, unit_label="kg",
                output_path=plot_path,
            )
    else:
        # --values came in as mm; the scene stores depths in m.
        depths_m = [v * 1.0e-3 for v in args.values]
        results = compression_sweep(
            depths_m,
            obj_mass=args.obj_mass,
            object=args.object,
            pad=args.pad,
            output_dir=args.output_dir,
            num_pads=args.num_pads,
            mu=args.mu,
        )
        plot_path = args.output_dir / (
            f"compression_sweep_mu_{results[0]['mu']:.2f}_"
            f"m_{args.obj_mass:.2f}kg.png"
        )
        if not args.no_plot:
            _plot_sweep(
                results, sweep_var="squeeze_depth",
                unit_scale=1.0e3, unit_label="mm",
                output_path=plot_path,
            )


if __name__ == "__main__":
    main()
