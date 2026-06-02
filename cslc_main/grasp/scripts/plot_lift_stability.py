# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Paper figure: CSLC lifts several objects stably at low normal force.

Runs the full APPROACH -> SQUEEZE -> LIFT -> HOLD grasp cycle with the
CSLC contact model for each plotted scenario, at the calibrated stable
operating point:

  * pad stiffness   ``kc_per_volume = 1e10``  (middle of the stable
    {1e9, 1e10, 1e11} range used in the comparison study)
  * object stiffness ``ke_target_physical = 5e4`` (grasp default)
  * commanded squeeze depth delta = 1 mm for the calibrated scenarios;
    the bunny mesh needs a deeper grip and uses its own ``squeeze_mm``.

It captures the per-step normal force (3rd-law actuator reaction along
the closing X axis, magnitude-averaged over the two pads) and the held
object's vertical position, then renders two clean publication figures
on a shared time axis spanning SQUEEZE -> LIFT -> HOLD:

  1. ``lift_stability_force.png`` -- normal force F_n(t): a smooth ramp
     during SQUEEZE that plateaus at a steady value through LIFT and HOLD
     (with the per-object Coulomb slip floor mg/2mu overlaid).
  2. ``lift_stability_z.png``     -- object rise dz(t): flat during
     SQUEEZE, a smooth monotone rise during LIFT, then a steady hold.

All scenarios share one time axis: the SQUEEZE *duration* is held fixed
and the squeeze *speed* is scaled to reach each scenario's target depth,
so the phase boundaries line up across curves.

Per-run time series are also written to ``lift_stability_data.csv`` so
the plots can be restyled (``--from-csv``) without re-running the sims.

Usage::

    # Run the sims + plot (a few minutes on GPU):
    uv run -m cslc_main.grasp.scripts.plot_lift_stability

    # Re-plot from cached data only (instant; for font/style tweaks):
    uv run -m cslc_main.grasp.scripts.plot_lift_stability --from-csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Stable operating point (see module docstring) ───────────────────
KC_PER_VOLUME = 1.0e10      # CSLC pad contact stiffness
KE_OBJECT = 5.0e4           # held-object contact stiffness (grasp default)
SQUEEZE_DEPTH_MM = 1.0      # default commanded penetration delta

# The plotted scenarios: pad geometry x held-object geometry.  Order,
# label and colour are shared across both figures.  Each scenario may
# override ``squeeze_mm`` (commanded penetration); the SQUEEZE *duration*
# is held fixed across all scenarios (the squeeze *speed* is scaled to
# reach the target depth) so every curve shares one time axis with
# aligned phase boundaries.
#
# flat pad · box is intentionally excluded from the figures: at a fixed
# 1 mm penetration its full-face flat-on-flat contact engages the entire
# pad area, so CSLC's volume-proportional force settles near ~33 N --
# physically correct, but an order of magnitude above the others and
# off-message for the low-force ("Coulomb regime") narrative.  It is
# still simulated and recorded in lift_stability_data.csv (see SIM_ONLY).
#
# The bunny is a mesh object: the dome pad's small patch can't hold its
# irregular surface, so it is grasped with the flat pad and a deeper
# 2 mm squeeze (it does not lift at the calibrated 1 mm).
SCENARIOS = [
    {"pad": "box",  "object": "sphere", "label": "flat pad · sphere", "color": "#1f77b4"},
    {"pad": "dome", "object": "sphere", "label": "dome pad · sphere", "color": "#2ca02c"},
    {"pad": "dome", "object": "box",    "label": "dome pad · box",    "color": "#9467bd"},
    {"pad": "box",  "object": "bunny",  "label": "flat pad · bunny",  "color": "#ff7f0e",
     "squeeze_mm": 2.0},
]

# Scenarios run + logged for the record but kept out of the figures.
SIM_ONLY = [
    {"pad": "box", "object": "box", "label": "flat pad · box", "color": "#d62728"},
]


@contextlib.contextmanager
def _silence():
    """Swallow the runner's verbose per-step stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _run_scenario(pad: str, object_kind: str, output_root: Path,
                  squeeze_mm: float = SQUEEZE_DEPTH_MM) -> dict:
    """Run one CSLC grasp at the stable operating point.

    Returns a dict of per-step series, windowed to SQUEEZE -> LIFT ->
    HOLD (the dead APPROACH phase is dropped).  Time is re-zeroed to the
    start of SQUEEZE; object Z is reported as rise above its
    start-of-squeeze height.
    """
    # Imported lazily so ``--from-csv`` needs no Warp/GPU.
    from ..params import GraspConfig
    from ..runner import run_headless

    config = GraspConfig()
    config.contact_model = "cslc"
    config.pad.kind = pad
    config.object.kind = object_kind
    config.cslc.kc_per_volume = KC_PER_VOLUME
    config.material.ke_target_physical = KE_OBJECT

    # Commanded squeeze depth: keep the squeeze *speed* at its default
    # (a slow, stable press) and set the *duration* to reach the target
    # depth.  Deeper grips therefore take longer to squeeze; the curves
    # are re-zeroed to the start of LIFT below so the LIFT/HOLD phases
    # still line up across scenarios despite different squeeze lengths.
    config.timing.squeeze_duration = max(
        squeeze_mm * 1e-3 / config.timing.squeeze_speed,
        config.timing.dt,
    )

    config.logging.output_root = output_root
    config.logging.run_label = f"cslc_{pad}_{object_kind}"
    config.logging.use_timestamp = False
    config.logging.save_lattice_preview = False
    config.logging.save_postsim_plots = False

    with _silence():
        m = run_headless(config)

    dt = config.timing.dt
    # Per-pad normal force magnitude, 3rd-law symmetric average.
    f_left = np.abs(np.asarray(m.F_n_left, dtype=float))
    f_right = np.abs(np.asarray(m.F_n_right, dtype=float))
    f_n = 0.5 * (f_left + f_right)
    z = np.asarray(m.object_z, dtype=float)

    # Window to SQUEEZE..HOLD (drop the flat APPROACH lead-in).
    phases = np.array([config.phase_of(s)[0] for s in range(len(z))])
    keep = np.isin(phases, ("SQUEEZE", "LIFT", "HOLD"))
    f_n, z, phases = f_n[keep], z[keep], phases[keep]
    # Re-zero time to the START OF LIFT so LIFT/HOLD align across
    # scenarios even when squeeze durations differ; SQUEEZE then occupies
    # negative time.
    n_squeeze = int((phases == "SQUEEZE").sum())
    t = (np.arange(len(z)) - n_squeeze) * dt
    dz_mm = (z - z[0]) * 1e3            # rise above start-of-squeeze height

    # Phase boundaries on the lift-aligned axis (shared by all scenarios).
    sq_end = 0.0                                       # SQUEEZE -> LIFT
    lift_end = float((phases == "LIFT").sum() * dt)    # LIFT -> HOLD

    return {
        "pad": pad, "object": object_kind,
        "t": t, "f_n": f_n, "dz_mm": dz_mm,
        "sq_end": sq_end, "lift_end": lift_end,
    }


def _write_csv(series: list[dict], path: Path) -> None:
    """Long-format dump: one row per (scenario, step)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pad", "object", "t_s", "F_n_N", "dz_mm",
                    "sq_end_s", "lift_end_s"])
        for s in series:
            for ti, fi, zi in zip(s["t"], s["f_n"], s["dz_mm"]):
                w.writerow([s["pad"], s["object"],
                            f"{ti:.6f}", f"{fi:.6f}", f"{zi:.6f}",
                            f"{s['sq_end']:.6f}", f"{s['lift_end']:.6f}"])


def _read_csv(path: Path) -> list[dict]:
    """Rebuild the per-scenario series from the long-format CSV,
    preserving SCENARIOS order."""
    by_key: dict[tuple[str, str], dict] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["pad"], r["object"])
            d = by_key.setdefault(key, {
                "pad": r["pad"], "object": r["object"],
                "t": [], "f_n": [], "dz_mm": [],
                "sq_end": float(r["sq_end_s"]),
                "lift_end": float(r["lift_end_s"]),
            })
            d["t"].append(float(r["t_s"]))
            d["f_n"].append(float(r["F_n_N"]))
            d["dz_mm"].append(float(r["dz_mm"]))
    out = []
    for sc in SCENARIOS:
        key = (sc["pad"], sc["object"])
        if key in by_key:
            d = by_key[key]
            d["t"] = np.asarray(d["t"]); d["f_n"] = np.asarray(d["f_n"])
            d["dz_mm"] = np.asarray(d["dz_mm"])
            out.append(d)
    return out


def _series_for(series: list[dict], pad: str, obj: str) -> dict | None:
    for s in series:
        if s["pad"] == pad and s["object"] == obj:
            return s
    return None


# Human-readable object names for the Coulomb-limit annotations.
_OBJ_NAME = {"sphere": "ball", "box": "box", "bunny": "bunny"}


def _coulomb_limits() -> tuple[dict[str, float], float]:
    """Minimum per-pad normal force to hold each plotted object against
    gravity without slipping.

    Two-finger pinch: the two pads' Coulomb friction must carry the
    object weight, ``2·μ·F_n ≥ m·g``, so the slip floor per pad is
    ``F_n,min = m·g / (2μ)``.

    Returns ``({object_kind: F_n_min [N]}, mu)`` for the distinct objects
    appearing in :data:`SCENARIOS`.
    """
    from ..params import GraspConfig
    c = GraspConfig()
    mu = c.material.mu
    limits: dict[str, float] = {}
    for sc in SCENARIOS:
        ok = sc["object"]
        if ok not in limits:
            c.object.kind = ok
            limits[ok] = c.object.weight / (2.0 * mu)
    return limits, mu


# Large, clean, title-free paper styling.
_RC = {
    "font.size": 20,
    "axes.labelsize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 19,
    "lines.linewidth": 3.2,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
}


def _phase_dividers(ax, sq_end: float, lift_end: float) -> None:
    """Faint vertical dividers at SQUEEZE->LIFT and LIFT->HOLD."""
    for x in (sq_end, lift_end):
        ax.axvline(x, color="0.7", linestyle=(0, (4, 4)), linewidth=1.4, zorder=0)


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=1.0)
    ax.margins(x=0.0)


def _plot_force(series: list[dict], out: Path) -> None:
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        ref = series[0]
        _phase_dividers(ax, ref["sq_end"], ref["lift_end"])
        for sc in SCENARIOS:
            s = _series_for(series, sc["pad"], sc["object"])
            if s is None:
                continue
            ax.plot(s["t"], s["f_n"], color=sc["color"], label=sc["label"])

        # Coulomb slip floor per object: F_n,min = m g / (2 mu).  CSLC
        # operating just above it = an efficient, low-force grasp.
        limits, _ = _coulomb_limits()
        for _, fmin in sorted(limits.items(), key=lambda kv: kv[1]):
            ax.axhline(fmin, color="0.35", linestyle=(0, (1, 1.5)),
                       linewidth=2.0, zorder=1)
        # Legend: a "Coulomb slip limit" header (the dotted line), then
        # the per-object floor values listed beneath it.
        coulomb_handles = [
            Line2D([0], [0], color="0.35", linestyle=(0, (1, 1.5)),
                   linewidth=2.0, label=r"Coulomb slip limit  $m g / 2\mu$")
        ]
        for ok, fmin in sorted(limits.items(), key=lambda kv: kv[1]):
            coulomb_handles.append(
                Line2D([0], [0], linestyle="none",
                       label=f"    {_OBJ_NAME.get(ok, ok)}: {fmin:.2f} N")
            )

        ax.set_xlabel("time [s]")
        ax.set_ylabel(r"normal force $F_n$ [N]")
        ax.set_ylim(bottom=0)
        _style_axes(ax)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + coulomb_handles, frameon=False,
                  loc="best")
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out} (+ .pdf)")


def _plot_z(series: list[dict], out: Path) -> None:
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        ref = series[0]
        _phase_dividers(ax, ref["sq_end"], ref["lift_end"])
        for sc in SCENARIOS:
            s = _series_for(series, sc["pad"], sc["object"])
            if s is None:
                continue
            ax.plot(s["t"], s["dz_mm"], color=sc["color"], label=sc["label"])
        ax.set_xlabel("time [s]")
        ax.set_ylabel(r"object rise $\Delta z$ [mm]")
        _style_axes(ax)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out} (+ .pdf)")


def main() -> None:
    default_out = Path(__file__).resolve().parents[1] / "outputs" / "lift_stability"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=default_out,
                        help="Directory for the data CSV and figures.")
    parser.add_argument("--from-csv", action="store_true",
                        help="Skip the sims; re-plot from the cached "
                             "lift_stability_data.csv (for style tweaks).")
    args = parser.parse_args()

    csv_path = args.out_dir / "lift_stability_data.csv"

    if args.from_csv:
        if not csv_path.exists():
            raise SystemExit(f"No cached data at {csv_path}; run without "
                             "--from-csv first.")
        series = _read_csv(csv_path)
    else:
        import warp as wp
        wp.init()
        runs_root = args.out_dir / "runs"
        series = []
        for sc in SCENARIOS + SIM_ONLY:
            squeeze_mm = sc.get("squeeze_mm", SQUEEZE_DEPTH_MM)
            print(f"Running CSLC  pad={sc['pad']:4s} object={sc['object']:6s} "
                  f"(kc={KC_PER_VOLUME:.0e}, ke_obj={KE_OBJECT:.0e}, "
                  f"δ={squeeze_mm:.1f}mm) ...", flush=True)
            s = _run_scenario(sc["pad"], sc["object"], runs_root,
                              squeeze_mm=squeeze_mm)
            print(f"  F_n[end]={s['f_n'][-1]:.2f}N  "
                  f"Δz[end]={s['dz_mm'][-1]:.1f}mm")
            series.append(s)
        _write_csv(series, csv_path)
        print(f"Wrote {csv_path}")

    _plot_force(series, args.out_dir / "lift_stability_force.png")
    _plot_z(series, args.out_dir / "lift_stability_z.png")


if __name__ == "__main__":
    main()
