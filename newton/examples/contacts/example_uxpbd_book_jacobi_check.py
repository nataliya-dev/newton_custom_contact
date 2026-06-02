# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Jacobi convergence check: is ``jacobi_iterations=4`` enough at depth?

The CSLC lattice δ is solved by a damped-Jacobi sweep run
``CSLCParams.jacobi_iterations`` times per substep
(solver_uxpbd.compute_compliant_contact_response). Over-bracing
(stiff anchors / many contacts) slows Jacobi convergence, so the risk
is that 4 sweeps don't fully converge δ where the contact is deepest,
which would bias the emitted contact force.

This script measures, with **no solver changes**, the per-sweep
residual at depth. The sweep ping-pongs two buffers; after the loop
``solver._cslc_delta_a`` holds the LAST iterate and
``solver._cslc_delta_b`` the penultimate one, so

    residual = max_i || δ_a[i] - δ_b[i] ||   (surface spheres)

is exactly "max change in δ on the last sweep." We report it against
the depth (max ||δ||) over the engaged phases, and bracket
``jacobi_iterations`` to confirm the *physical* δ (and the resulting
squeeze force N) has converged: if depth(4) ≈ depth(16) and the
residual is a small fraction of the depth, 4 sweeps suffice.

Uses the box pad (the config where the lattice actually carries the
contact, so "depth" is real) at the lift-figure mass.

Run::

    uv run python -m newton.examples.contacts.example_uxpbd_book_jacobi_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer

from .example_uxpbd_lift_test import Example, SceneParams
from .example_uxpbd_book_h0_diagnostic import normal_force_series

# Phases where the lattice is meaningfully loaded (contact engaged).
_ENGAGED = ("squeeze", "lift", "hold")


def _build(jacobi_iters: int, *, mass: float, mu: float) -> Example:
    import dataclasses
    params = dataclasses.replace(SceneParams(), obj_mass=mass, mu=mu)
    viewer = newton.viewer.ViewerNull(num_frames=params.total_frames + 10)
    args = argparse.Namespace(
        num_pads=2, object="book", pad="box",
        no_compliant=False, test=False,
    )
    ex = Example(viewer, args, params=params)
    # CSLCParams is a plain (non-frozen) dataclass; the Jacobi loop reads
    # ``self.cslc_params.jacobi_iterations`` every substep, so mutating it
    # post-build takes effect immediately.
    ex.solver.cslc_params.jacobi_iterations = int(jacobi_iters)
    return ex


def run_one(jacobi_iters: int, *, mass: float, mu: float) -> dict:
    """Step the full scene; sample the last-sweep residual and depth on
    every engaged-phase frame. Returns worst-case and steady (last-HOLD)
    values, plus the converged squeeze force N for context."""
    ex = _build(jacobi_iters, mass=mass, mu=mu)
    surf = ex.model.lattice_is_surface.numpy().astype(bool)

    worst_resid = 0.0
    worst_resid_depth = 0.0
    hold_resid = hold_depth = float("nan")
    for _ in range(ex.p.total_frames):
        ex.step()
        row = ex.history[-1]
        if row["phase"] not in _ENGAGED:
            continue
        a = ex.solver._cslc_delta_a.numpy()
        b = ex.solver._cslc_delta_b.numpy()
        d = ex.model.lattice_delta.numpy()
        resid = np.linalg.norm((a - b)[surf], axis=1)
        depth = np.linalg.norm(d[surf], axis=1)
        r_max = float(resid.max()) if resid.size else 0.0
        d_max = float(depth.max()) if depth.size else 0.0
        if r_max > worst_resid:
            worst_resid = r_max
            worst_resid_depth = d_max
        if row["phase"] == "hold":
            hold_resid = r_max
            hold_depth = d_max

    # Converged squeeze force at HOLD (per pad), for the N-bias question.
    nf = normal_force_series(ex, use_cslc=True)
    hold_mask = np.array([p == "hold" for p in nf["phase"]])
    n_hold = float(np.mean(nf["N"][hold_mask])) if hold_mask.any() else float("nan")

    return {
        "iters": jacobi_iters,
        "worst_resid": worst_resid,
        "worst_resid_depth": worst_resid_depth,
        "hold_resid": hold_resid,
        "hold_depth": hold_depth,
        "n_hold": n_hold,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="example_uxpbd_book_jacobi_check",
        description="Measure the Jacobi last-sweep residual at depth and "
                    "bracket jacobi_iterations to confirm convergence.",
    )
    parser.add_argument("--mass", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--iters", type=int, nargs="+", default=[4, 16],
                        help="jacobi_iterations values to compare.")
    args = parser.parse_args(argv)

    print(f"== Jacobi convergence check: book/box, m={args.mass} kg, "
          f"mu={args.mu} ==")
    results = [run_one(it, mass=args.mass, mu=args.mu) for it in args.iters]

    print("\n" + "=" * 78)
    print("JACOBI RESIDUAL AT DEPTH  (units: micrometres of δ; surface spheres)")
    print("=" * 78)
    hdr = (f"{'iters':>5s} | {'depth(hold)':>11s} | {'resid(hold)':>11s} | "
           f"{'resid/depth':>11s} | {'worst resid':>11s} | {'N/pad(hold)':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ratio = (r["hold_resid"] / r["hold_depth"] * 100.0
                 if r["hold_depth"] and r["hold_depth"] == r["hold_depth"]
                 else float("nan"))
        print(f"{r['iters']:>5d} | "
              f"{r['hold_depth'] * 1e6:>9.2f}um | "
              f"{r['hold_resid'] * 1e6:>9.3f}um | "
              f"{ratio:>10.3f}% | "
              f"{r['worst_resid'] * 1e6:>9.3f}um | "
              f"{r['n_hold']:>9.2f}N")
    print("=" * 78)

    # Verdict: converged if the residual is a small fraction of the depth
    # at the production setting (first --iters entry, default 4) AND the
    # depth/N are stable as iterations increase.
    base = results[0]
    ratio0 = (base["hold_resid"] / base["hold_depth"]
              if base["hold_depth"] else float("nan"))
    print(f"\nAt jacobi_iterations={base['iters']}: last-sweep residual is "
          f"{ratio0 * 100:.3f}% of the contact depth at HOLD.")
    if len(results) > 1:
        d0, d1 = base["hold_depth"], results[-1]["hold_depth"]
        n0, n1 = base["n_hold"], results[-1]["n_hold"]
        dd = abs(d1 - d0) / d0 * 100 if d0 else float("nan")
        dn = abs(n1 - n0) / n0 * 100 if n0 else float("nan")
        print(f"depth shift {base['iters']}->{results[-1]['iters']} sweeps: "
              f"{dd:.2f}%   |   N shift: {dn:.2f}%")
        if ratio0 < 0.01 and dd < 1.0 and dn < 1.0:
            print("=> CONVERGED: 4 sweeps suffice; δ and N are iteration-"
                  "independent at depth. No bias from under-iteration.")
        else:
            print("=> NOT fully converged: bump jacobi_iterations "
                  "(cheap) until depth/N stabilise.")


if __name__ == "__main__":
    main()
