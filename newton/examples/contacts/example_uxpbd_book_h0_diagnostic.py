# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""H0 diagnostic: is the CSLC-vs-rigid lift tie a *friction-capacity*
limit, or something else?

Context
-------
In the book-pinch lift figure (``example_uxpbd_book_paper_figures``)
CSLC and the rigid baseline tie: both carry the 0.5 kg book to ~17 mm
of the 30 mm pad target, lagging by ~13 mm. The question this script
answers is *why*, by measuring the two quantities that decide whether
the grip can hold the book:

  1. The **squeeze normal force** N transmitted through each pad.
  2. The **vertical friction** the book actually receives.

Physics
-------
Vertical hold is a Coulomb-friction problem. With two pads each
pressing with normal force N at coefficient mu, the maximum friction
available to oppose gravity is

    F_cap = mu * (N_left + N_right)          [Amontons' law]

and this is **independent of contact area / number of contact points**
-- distributing the same total normal load over many lattice spheres
does not raise F_cap. The book holds iff F_cap >= m*g (plus the
inertial term m*a during the lift accel ramp).

Two regimes are possible, and they imply completely different fixes:

  * **Regime A -- Coulomb-bound tie.**  F_cap ~ m*g at the slip onset.
    The tie is then a genuine theory bound (Amontons): CSLC cannot beat
    rigid on vertical capacity because friction = mu*N regardless of
    how the patch is distributed. Fix = change the *quantity* being
    shown (rotational stability / pose-error robustness), not tuning.

  * **Regime B -- ample capacity, lag is an artifact.**  The squeeze is
    *displacement*-controlled: the pad is driven 5 mm past contact into
    a stiff drive (drive_ke = 5e4 N/m), so N can be hundreds of N and
    F_cap >> m*g. If the book still lags, the 13 mm is NOT a friction
    limit -- it is a contact-resolution / SM-rigid / lift-transient
    artifact, and neither "theory" nor "conformance" is the lever.

How N is measured (no scene/solver changes)
-------------------------------------------
The squeeze axis (prismatic X) is held at its full-squeeze target
through LIFT and HOLD (see ``_pad_target_xz``). The pad is a near-
massless body on a PD position drive, so in quasi-static equilibrium
the contact normal balances the drive force:

    N ~= drive_ke * (target_x_disp - actual_x_disp)

The kd (velocity) term vanishes during HOLD (target and pad both
static), so this is exact where it matters. ``target_x_disp`` comes
from the pure ``_pad_target_xz`` trajectory; ``actual_x_disp`` is the
pad body's travel from its spawn x (recovered from
``_pad_thickness`` / ``approach_gap``). Everything is reconstructed
offline from the existing per-frame history.

The vertical friction the book receives is read from its own motion
via Newton's 2nd law (the squeeze normal is horizontal, so it drops
out of the z balance):

    F_fric_z = m * (a_z + g),   a_z = d(obj_vz)/dt

Run::

    uv run python -m newton.examples.contacts.example_uxpbd_book_h0_diagnostic \\
        --output-dir results/book_h0
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer

from .example_uxpbd_lift_test import (
    Example,
    SceneParams,
    _pad_target_xz,
    _pad_thickness,
)


# ----- Style (matches plot_dome_kl_sweep.py / book_paper_figures) ------

LABEL_SIZE = 16
TICK_SIZE = 13
LEGEND_SIZE = 13
LINE_WIDTH = 2.4
DPI = 200
FIG_SIZE = (6.5, 5.0)

COLOUR_CSLC = "tab:blue"
COLOUR_RIGID = "tab:red"
COLOUR_REF = "0.30"


# ----- Run plumbing (mirrors run_one_config, but keeps the Example) ----


def _run(*, lift_mass: float, mu: float, pad: str,
         use_cslc: bool) -> Example:
    """Run one headless book-lift instance and return the *Example* (so
    the caller can read ``ex.p`` post-override params, ``ex.history``,
    ``ex.model`` gravity, and the final pad poses). Matches the lift
    figure's config: object='book', num_pads=2."""
    params = dataclasses.replace(SceneParams(), obj_mass=lift_mass, mu=mu)
    viewer = newton.viewer.ViewerNull(num_frames=params.total_frames + 10)
    args = argparse.Namespace(
        num_pads=2, object="book", pad=pad,
        no_compliant=not use_cslc, test=False,
    )
    ex = Example(viewer, args, params=params)
    for _ in range(ex.p.total_frames):
        ex.step()
    return ex


def _gravity_mag(ex: Example) -> float:
    """Gravity magnitude [m/s^2] from the model, falling back to 9.81."""
    try:
        g = ex.model.gravity
        if hasattr(g, "numpy"):
            g = g.numpy()
        return abs(float(np.asarray(g).reshape(-1)[2]))
    except Exception:
        return 9.81


# ----- Squeeze normal force N(t) ---------------------------------------


def _pad_spawn_x(ex: Example, *, side: str, use_cslc: bool) -> float:
    """World x of a pad's spawn (the prismatic-X joint origin). Left pad
    spawns at -(gap/2 + thickness); right at the mirror. Thickness uses
    the protruded lattice envelope in CSLC mode (see ``_pad_thickness``)."""
    thick = _pad_thickness(ex.args.pad, ex.p, use_cslc=use_cslc)
    x0 = ex.p.approach_gap / 2.0 + thick
    return -x0 if side == "left" else +x0


def normal_force_series(ex: Example, *, use_cslc: bool) -> dict:
    """Reconstruct the left-pad squeeze normal force N(t) [N] from the
    history, plus time/phase axes. N>0 means contact is pushing the pad
    back off its target (the physical squeeze).

    Returned arrays are frame-aligned: ``t``, ``phase`` (list),
    ``N`` [N], ``commanded_disp`` [m], ``actual_disp`` [m].
    """
    hist = ex.history
    lx0 = _pad_spawn_x(ex, side="left", use_cslc=use_cslc)
    ke = ex.p.drive_ke
    substeps = ex.p.sim_substeps

    t = np.array([r["t"] for r in hist])
    phase = [r["phase"] for r in hist]
    commanded = np.empty(len(hist))
    actual = np.empty(len(hist))
    for i, r in enumerate(hist):
        # step index at record time = frame * substeps (see step()).
        step = int(round(r["frame"])) * substeps
        dx, _dz = _pad_target_xz(step, ex.p)
        commanded[i] = dx
        # left pad target is +dx; actual inward travel = pad_x - lx0.
        actual[i] = float(r["left_pad_x"]) - lx0
    N = ke * (commanded - actual)
    return {"t": t, "phase": phase, "N": N,
            "commanded_disp": commanded, "actual_disp": actual}


def final_pad_symmetry(ex: Example, *, use_cslc: bool) -> tuple[float, float]:
    """Read both pads' squeeze force at the final (HOLD) frame directly
    from the live state, to verify the left/right symmetry assumption
    used when doubling N for the capacity. Returns (N_left, N_right)."""
    body_q = ex.state_0.body_q.numpy()
    step = ex.sim_step
    dx, _dz = _pad_target_xz(step, ex.p)
    ke = ex.p.drive_ke
    lx0 = _pad_spawn_x(ex, side="left", use_cslc=use_cslc)
    rx0 = _pad_spawn_x(ex, side="right", use_cslc=use_cslc)
    left_x = float(body_q[ex.pad_bodies[0]][0])
    right_x = float(body_q[ex.pad_bodies[1]][0])
    n_left = ke * (dx - (left_x - lx0))
    n_right = ke * (dx - (rx0 - right_x))   # right travels inward (-x)
    return n_left, n_right


# ----- Vertical friction the book receives -----------------------------


def vertical_friction_series(ex: Example, g: float) -> dict:
    """Vertical friction F_fric_z(t) [N] the book receives, from its own
    motion: F_fric_z = m*(a_z + g). a_z is the smoothed time-derivative
    of the recorded mean book vz."""
    hist = ex.history
    t = np.array([r["t"] for r in hist])
    vz = np.array([r["obj_vz"] for r in hist])
    # Light smoothing before differentiating: SM-rigid mean velocity has
    # frame-scale jitter that the derivative would amplify.
    if len(vz) >= 5:
        kernel = np.ones(5) / 5.0
        vz_s = np.convolve(vz, kernel, mode="same")
    else:
        vz_s = vz
    a_z = np.gradient(vz_s, t)
    f_fric = ex.p.obj_mass * (a_z + g)
    return {"t": t, "f_fric": f_fric, "a_z": a_z}


# ----- Phase helpers ----------------------------------------------------


def _phase_mask(phase: list, name: str) -> np.ndarray:
    return np.array([p == name for p in phase])


def _slip_series(ex: Example) -> dict:
    hist = ex.history
    t = np.array([r["t"] for r in hist])
    slip = np.array([r["slip"] for r in hist])
    obj_lift = np.array([r["obj_lift"] for r in hist])
    pad_lift = np.array([r["pad_lift"] for r in hist])
    return {"t": t, "slip": slip, "obj_lift": obj_lift, "pad_lift": pad_lift}


def lift_tracking_series(ex: Example) -> dict:
    """Book vs pad kinematics over LIFT+HOLD, anchored at LIFT start.

    Returns lift-relative time and the book/pad rise + velocities, so we
    can see *how* the 13 mm slip accrues: a one-time re-seat at LIFT
    onset (step in slip, then flat) vs continuous tangential creep
    (slip grows with pad travel). pad velocity is differentiated from
    the recorded pad z (telemetry has no pad vz)."""
    hist = ex.history
    t = np.array([r["t"] for r in hist])
    phase = [r["phase"] for r in hist]
    lift_idx = np.where(np.array([p == "lift" for p in phase]))[0]
    if lift_idx.size == 0:
        return {"t_rel": np.array([]), "obj_dz": np.array([]),
                "pad_dz": np.array([]), "obj_vz": np.array([]),
                "pad_vz": np.array([])}
    i0 = lift_idx[0]
    sel = slice(i0, len(hist))             # LIFT start to end of run
    t_rel = t[sel] - t[i0]
    obj_z = np.array([r["obj_z"] for r in hist])[sel]
    pad_z = np.array([r["left_pad_z"] for r in hist])[sel]
    obj_vz = np.array([r["obj_vz"] for r in hist])[sel]
    pad_vz = np.gradient(pad_z, t[sel])
    return {"t_rel": t_rel, "obj_dz": obj_z - obj_z[0],
            "pad_dz": pad_z - pad_z[0], "obj_vz": obj_vz, "pad_vz": pad_vz}


# ----- Summary table ----------------------------------------------------


def summarize(label: str, ex: Example, *, use_cslc: bool, g: float) -> dict:
    nf = normal_force_series(ex, use_cslc=use_cslc)
    sl = _slip_series(ex)
    n_left_final, n_right_final = final_pad_symmetry(ex, use_cslc=use_cslc)

    hold = _phase_mask(nf["phase"], "hold")
    lift = _phase_mask(nf["phase"], "lift")
    # Mean squeeze force over HOLD (steady, kd term ~ 0 -> N exact here).
    n_hold = float(np.mean(nf["N"][hold])) if hold.any() else float("nan")
    # Use the verified two-pad sum at the final frame for capacity.
    cap = ex.p.mu * (n_left_final + n_right_final)
    weight = ex.p.obj_mass * g
    ratio = cap / weight if weight > 0 else float("inf")

    # Where does slip accrue? slip at end of lift vs end of hold,
    # anchored to the start-of-lift slip.
    lift_idx = np.where(lift)[0]
    hold_idx = np.where(hold)[0]
    if lift_idx.size:
        slip_lift_start = sl["slip"][lift_idx[0]]
        slip_lift_end = sl["slip"][lift_idx[-1]]
    else:
        slip_lift_start = slip_lift_end = float("nan")
    slip_hold_end = sl["slip"][hold_idx[-1]] if hold_idx.size else slip_lift_end
    slip_during_lift = slip_lift_end - slip_lift_start
    slip_during_hold = slip_hold_end - slip_lift_end

    print(f"\n--- {label} ---")
    print(f"  squeeze force N (per pad, HOLD mean) : {n_hold:8.2f} N")
    print(f"  N_left / N_right (final frame)        : "
          f"{n_left_final:8.2f} / {n_right_final:8.2f} N  "
          f"(symmetry check)")
    print(f"  friction capacity 2*mu*N (final)      : {cap:8.2f} N")
    print(f"  book weight m*g                        : {weight:8.2f} N "
          f"(m={ex.p.obj_mass:.3f} kg, g={g:.2f})")
    print(f"  capacity / weight (safety factor)     : {ratio:8.2f} x")
    print(f"  slip accrued during LIFT               : "
          f"{slip_during_lift * 1e3:+8.2f} mm")
    # Sub-window breakdown: is the slip a one-time re-seat at LIFT onset
    # or continuous creep? Sample slip = pad_dz - obj_dz at lift-relative
    # 0.5 s (end accel ramp), 1.0 s (end cruise), 1.5 s (end lift).
    lt = lift_tracking_series(ex)
    if lt["t_rel"].size:
        slip_track = lt["pad_dz"] - lt["obj_dz"]
        for t_mark in (ex.p.lift_ramp_duration,
                       ex.p.lift_duration - ex.p.lift_ramp_duration,
                       ex.p.lift_duration):
            j = int(np.argmin(np.abs(lt["t_rel"] - t_mark)))
            print(f"     slip at lift t={t_mark:.2f}s "
                  f"(pad +{lt['pad_dz'][j] * 1e3:5.1f}mm)         : "
                  f"{slip_track[j] * 1e3:+8.2f} mm")
    print(f"  slip accrued during HOLD               : "
          f"{slip_during_hold * 1e3:+8.2f} mm")
    print(f"  total slip (end of run)                : "
          f"{slip_hold_end * 1e3:+8.2f} mm")
    return {
        "label": label, "n_hold": n_hold, "cap": cap, "weight": weight,
        "ratio": ratio, "slip_lift": slip_during_lift,
        "slip_hold": slip_during_hold, "slip_total": slip_hold_end,
        "n_left": n_left_final, "n_right": n_right_final,
    }


def verdict(s_cslc: dict, s_rigid: dict) -> None:
    print("\n" + "=" * 70)
    print("H0 VERDICT")
    print("=" * 70)
    rA = s_cslc["ratio"]
    rB = s_rigid["ratio"]
    # Regime classification on the rigid baseline (the simplest contact).
    if max(rA, rB) < 1.5:
        print("Regime A (Coulomb-bound): capacity ~ weight. The tie is a")
        print("genuine friction limit (Amontons): F_cap = mu*N, area-")
        print("independent, so CSLC cannot beat rigid on vertical capacity.")
    elif min(rA, rB) > 3.0:
        print("Regime B (artifact): capacity >> weight for BOTH methods, yet")
        print("the book still lags. The 13 mm is NOT a friction-capacity")
        print("limit -- it is a contact-resolution / SM-rigid / lift-")
        print("transient artifact. Neither theory nor conformance is the")
        print("lever here; the lift dynamics are.")
    else:
        print(f"Mixed/intermediate: safety factors CSLC={rA:.2f}x "
              f"rigid={rB:.2f}x. Inspect the slip-accrual timing and the")
        print("friction-vs-capacity figure to localize the loss.")

    dn = s_cslc["n_hold"] - s_rigid["n_hold"]
    print(f"\nN(CSLC) - N(rigid) at HOLD = {dn:+.2f} N "
          f"(CSLC {s_cslc['n_hold']:.1f} vs rigid {s_rigid['n_hold']:.1f}).")
    if dn < -0.05 * max(abs(s_rigid["n_hold"]), 1.0):
        print("  -> CSLC transmits LESS squeeze force at the same commanded")
        print("     5 mm penetration: the displacement-controlled squeeze")
        print("     penalizes the softer compliant pad (series stiffness).")
    elif abs(dn) <= 0.05 * max(abs(s_rigid["n_hold"]), 1.0):
        print("  -> CSLC and rigid transmit ~equal N: same capacity, so the")
        print("     tie is expected (both bound by the same mu*N).")
    else:
        print("  -> CSLC transmits MORE N than rigid (unexpected under")
        print("     displacement control -- worth a closer look).")
    print("=" * 70)


# ----- Figures ----------------------------------------------------------


def plot_normal_force(nf_cslc: dict, nf_rigid: dict,
                      out_path: Path) -> None:
    """N(t) for CSLC vs rigid across the whole run. The decisive panel:
    if the two curves coincide (or CSLC sits below), capacity is equal
    (or worse) for CSLC and the lift tie is not a CSLC deficiency."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(nf_cslc["t"], nf_cslc["N"], "-", color=COLOUR_CSLC,
            lw=LINE_WIDTH, label="CSLC", zorder=3)
    ax.plot(nf_rigid["t"], nf_rigid["N"], "--", color=COLOUR_RIGID,
            lw=LINE_WIDTH, label="rigid", zorder=2)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("time [s]", fontsize=LABEL_SIZE)
    ax.set_ylabel("squeeze normal force N [N]", fontsize=LABEL_SIZE,
                  labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=LEGEND_SIZE, framealpha=0.95)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def plot_capacity_vs_demand(ex_cslc: Example, ex_rigid: Example,
                            nf_cslc: dict, nf_rigid: dict,
                            g: float, out_path: Path) -> None:
    """Friction capacity 2*mu*N vs the demand (book weight m*g and the
    actual vertical friction the book receives). If capacity sits far
    above weight for both methods, the lift lag is not friction-bound."""
    import matplotlib.pyplot as plt

    ff_cslc = vertical_friction_series(ex_cslc, g)
    ff_rigid = vertical_friction_series(ex_rigid, g)
    weight = ex_cslc.p.obj_mass * g
    mu = ex_cslc.p.mu

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # Capacity (two pads) for each method.
    ax.plot(nf_cslc["t"], 2.0 * mu * nf_cslc["N"], "-", color=COLOUR_CSLC,
            lw=LINE_WIDTH, label=r"CSLC capacity $2\mu N$", zorder=3)
    ax.plot(nf_rigid["t"], 2.0 * mu * nf_rigid["N"], "--", color=COLOUR_RIGID,
            lw=LINE_WIDTH, label=r"rigid capacity $2\mu N$", zorder=2)
    # Demand: the book's weight (what must be held).
    ax.axhline(weight, color=COLOUR_REF, linestyle=":", lw=2.0,
               label=r"book weight $mg$", zorder=4)
    ax.set_xlabel("time [s]", fontsize=LABEL_SIZE)
    ax.set_ylabel("vertical force [N]", fontsize=LABEL_SIZE, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=LEGEND_SIZE, framealpha=0.95)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def plot_lift_tracking(ex_cslc: Example, ex_rigid: Example,
                       out_path: Path) -> None:
    """Diagnostic (2-panel): how the book trails the pad during LIFT.

    Top: book Δz and pad Δz vs lift-relative time. The gap between a
    method's book curve and the shared pad curve IS the slip. A step at
    t=0 then parallel tracking => one-time re-seat; a widening gap =>
    continuous tangential creep. Bottom: book vz vs pad vz -- if book vz
    sits below pad vz throughout cruise, the contact is creeping."""
    import matplotlib.pyplot as plt

    lt_c = lift_tracking_series(ex_cslc)
    lt_r = lift_tracking_series(ex_rigid)

    fig, (ax_z, ax_v) = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True)

    ax_z.plot(lt_c["t_rel"], lt_c["pad_dz"] * 1e3, ":", color=COLOUR_REF,
              lw=2.0, label="pad Δz (target)", zorder=1)
    ax_z.plot(lt_c["t_rel"], lt_c["obj_dz"] * 1e3, "-", color=COLOUR_CSLC,
              lw=LINE_WIDTH, label="CSLC book Δz", zorder=3)
    ax_z.plot(lt_r["t_rel"], lt_r["obj_dz"] * 1e3, "--", color=COLOUR_RIGID,
              lw=LINE_WIDTH, label="rigid book Δz", zorder=2)
    ax_z.set_ylabel("Δz [mm]", fontsize=LABEL_SIZE, labelpad=6)
    ax_z.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax_z.grid(alpha=0.3)
    ax_z.legend(loc="upper left", fontsize=LEGEND_SIZE, framealpha=0.95)

    ax_v.plot(lt_c["t_rel"], lt_c["pad_vz"] * 1e3, ":", color=COLOUR_REF,
              lw=2.0, label="pad vz", zorder=1)
    ax_v.plot(lt_c["t_rel"], lt_c["obj_vz"] * 1e3, "-", color=COLOUR_CSLC,
              lw=LINE_WIDTH, label="CSLC book vz", zorder=3)
    ax_v.plot(lt_r["t_rel"], lt_r["obj_vz"] * 1e3, "--", color=COLOUR_RIGID,
              lw=LINE_WIDTH, label="rigid book vz", zorder=2)
    ax_v.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax_v.set_xlabel("time since LIFT start [s]", fontsize=LABEL_SIZE)
    ax_v.set_ylabel("vz [mm/s]", fontsize=LABEL_SIZE, labelpad=6)
    ax_v.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="best", fontsize=LEGEND_SIZE, framealpha=0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


# ----- Main -------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="example_uxpbd_book_h0_diagnostic",
        description="Measure squeeze force & friction capacity to "
                    "diagnose the CSLC-vs-rigid lift tie.",
    )
    parser.add_argument("--lift-mass", type=float, default=0.5,
                        help="Book mass [kg] (matches the lift figure).")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Coulomb friction coefficient.")
    parser.add_argument("--pad", choices=("box", "curved"), default="box")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "book_h0")
    args = parser.parse_args(argv)

    print(f"== H0 diagnostic: book lift, m={args.lift_mass} kg, "
          f"mu={args.mu}, pad={args.pad} ==")

    print("\n[run] CSLC ...")
    ex_cslc = _run(lift_mass=args.lift_mass, mu=args.mu,
                   pad=args.pad, use_cslc=True)
    print("\n[run] rigid ...")
    ex_rigid = _run(lift_mass=args.lift_mass, mu=args.mu,
                    pad=args.pad, use_cslc=False)

    g = _gravity_mag(ex_cslc)
    nf_cslc = normal_force_series(ex_cslc, use_cslc=True)
    nf_rigid = normal_force_series(ex_rigid, use_cslc=False)

    s_cslc = summarize("CSLC", ex_cslc, use_cslc=True, g=g)
    s_rigid = summarize("rigid", ex_rigid, use_cslc=False, g=g)
    verdict(s_cslc, s_rigid)

    plot_normal_force(nf_cslc, nf_rigid,
                      args.output_dir / "h0_normal_force.png")
    plot_capacity_vs_demand(ex_cslc, ex_rigid, nf_cslc, nf_rigid, g,
                            args.output_dir / "h0_capacity_vs_demand.png")
    plot_lift_tracking(ex_cslc, ex_rigid,
                       args.output_dir / "h0_lift_tracking.png")


if __name__ == "__main__":
    main()
