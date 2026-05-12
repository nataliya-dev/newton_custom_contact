"""Magnitude sweep across all 6 disturbance axes for the BOX-on-BOX book scene.

For each (axis, model) cell, sweep magnitude from 0 → cliff and record:
  - active contacts (drops to 0 = ejection)
  - drop / creep / max-tilt during HOLD

Output:
  - CSV at sweep_book_disturbance_curves.csv (one row per (axis, mag, model))
  - text summary printed to stdout grouping by axis

Trade-paperback defaults: 152×229×25 mm, 0.45 kg, μ=0.5, weight 4.42 N.

Dependencies: assumes apply_external_wrench has the 2026-05-03 layout
fix applied (force in slots 0:3, torque in slots 3:6).  Re-run if you
revert that fix.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import warp as wp  # noqa: E402

wp.config.quiet = True
import numpy as np  # noqa: E402

from cslc_v1 import squeeze_test as st  # noqa: E402
from cslc_v1.common import (  # noqa: E402
    apply_external_wrench,
    make_solver,
    recalibrate_cslc_kc_per_pad,
)


def make_p() -> st.SceneParams:
    p = st.SceneParams()
    p.object_kind = "book"
    p.book_as_mesh = False
    p.pad_gap_initial = 2.0 * (p.book_hx - 0.0015)
    p.cslc_contact_fraction = 1.0
    pad_face_area = (2.0 * p.pad_hy) * (2.0 * p.pad_hz)
    # H1-style fair calibration: kh_eff = kh/2 (both bodies hydroelastic
    # at the same modulus → harmonic mean), so kh = 2·ke/A_patch.
    p.kh = 2.0 * p.ke / pad_face_area
    return p


def metrics_for(model, p, force, torque, recalibrate: bool):
    solver = make_solver(model, "mujoco")
    if recalibrate:
        recalibrate_cslc_kc_per_pad(model, contact_fraction=p.cslc_contact_fraction)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    con = model.contacts()
    sphere_body_idx = 2

    z_hist = []
    q_hist = []
    contact_hist = []

    for step in range(p.n_total_steps):
        st.set_kinematic_pads(s0, step, p)
        s0.clear_forces()
        if step >= p.n_squeeze_steps:
            apply_external_wrench(s0, sphere_body_idx, force=force, torque=torque)
        model.collide(s0, con)
        solver.step(s0, s1, ctrl, con, p.dt)
        wp.synchronize()
        s0, s1 = s1, s0
        q = s0.body_q.numpy()
        z_hist.append(float(q[2, 2]))
        q_hist.append(q[2, 3:7].copy())
        n = int(con.rigid_contact_count.numpy()[0])
        sh0 = con.rigid_contact_shape0.numpy()[:n]
        contact_hist.append(int((sh0 != -1).sum()))

    z_hold_start = z_hist[p.n_squeeze_steps]
    z_final = z_hist[-1]

    # Sentinel: numerical-blowup (z runs to >1m or NaN) → mark as ejection.
    ejected = (
        not math.isfinite(z_final)
        or abs(z_final - z_hist[0]) > 1.0
        or contact_hist[-1] == 0
    )

    n_hold = len(z_hist) - p.n_squeeze_steps
    half = p.n_squeeze_steps + n_hold // 2
    if half + 1 < len(z_hist):
        creep = (z_hist[-1] - z_hist[half]) / ((len(z_hist) - 1 - half) * p.dt) * (-1) * 1000
    else:
        creep = 0.0

    tilts = []
    for q in q_hist[p.n_squeeze_steps :]:
        sin_h = min(1.0, (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5)
        tilts.append(2.0 * math.degrees(math.asin(sin_h)))
    max_tilt = max(tilts) if tilts else 0.0

    return {
        "contacts_final": contact_hist[-1],
        "contacts_min": min(contact_hist[p.n_squeeze_steps :]) if n_hold else 0,
        "drop_full_mm": (z_hist[0] - min(z_hist)) * 1000.0 if not ejected else float("inf"),
        "drop_hold_mm": (z_hold_start - z_final) * 1000.0 if not ejected else float("inf"),
        "creep_mm_s": float(creep) if math.isfinite(creep) else float("inf"),
        "max_tilt_deg": max_tilt,
        "ejected": ejected,
    }


# ────────────────────────────────────────────────────────────────────────
# Sweep design — magnitudes chosen to cover both smooth regime and cliff
# ────────────────────────────────────────────────────────────────────────

SWEEP_AXES = [
    # (label, axis_dispatch, magnitudes [N for force, N·m for torque])
    ("F_x", "fx", [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]),
    ("F_y", "fy", [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]),
    ("F_z", "fz", [-15.0, -10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0]),
    ("τ_x", "tx", [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0, 15.0]),
    ("τ_y", "ty", [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]),
    # τ_z is the flagship cliff axis; use finer resolution near 5 N·m
    ("τ_z", "tz", [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 4.75, 5.0, 5.25, 5.5, 6.0, 7.5, 10.0]),
]


def axis_to_wrench(axis: str, mag: float):
    f = [0.0, 0.0, 0.0]
    t = [0.0, 0.0, 0.0]
    idx = "xyz".index(axis[1])
    if axis.startswith("f"):
        f[idx] = mag
    else:
        t[idx] = mag
    return tuple(f), tuple(t)


# ────────────────────────────────────────────────────────────────────────
# Run sweep
# ────────────────────────────────────────────────────────────────────────

OUT_CSV = Path(__file__).with_name("sweep_book_disturbance_curves.csv")
fields = [
    "axis", "magnitude", "model",
    "contacts_final", "contacts_min",
    "drop_full_mm", "drop_hold_mm", "creep_mm_s", "max_tilt_deg",
    "ejected",
]
rows = []

print("\n" + "=" * 96)
print(
    "  BOOK SCENE DISTURBANCE-MAGNITUDE SWEEP — μ=0.5, weight=4.42 N "
    "(F in N, τ in N·m)"
)
print("=" * 96)

for axis_label, axis_key, mags in SWEEP_AXES:
    print(f"\n  AXIS: {axis_label}")
    print(f"  {'-'*92}")
    print(
        f"    {'magnitude':>10}  {'model':<6}  {'contacts':>9}  "
        f"{'drop_full':>10}  {'drop_hold':>10}  {'creep':>10}  {'tilt':>9}  status"
    )
    for mag in mags:
        force, torque = axis_to_wrench(axis_key, mag)
        for model_name in ("point", "cslc", "hydro"):
            p = make_p()
            try:
                model = st.BUILDERS[model_name](p)
                r = metrics_for(
                    model, p, force=force, torque=torque,
                    recalibrate=(model_name == "cslc"),
                )
                status = "EJECT" if r["ejected"] else "ok"
                drop_full = (
                    f"{r['drop_full_mm']:>+10.3f}" if math.isfinite(r["drop_full_mm"])
                    else f"{'+inf':>10}"
                )
                drop_hold = (
                    f"{r['drop_hold_mm']:>+10.3f}" if math.isfinite(r["drop_hold_mm"])
                    else f"{'+inf':>10}"
                )
                creep = (
                    f"{r['creep_mm_s']:>+10.3f}" if math.isfinite(r["creep_mm_s"])
                    else f"{'+inf':>10}"
                )
                print(
                    f"    {mag:>+10.2f}  {model_name:<6}  {r['contacts_final']:>9d}  "
                    f"{drop_full}  {drop_hold}  {creep}  "
                    f"{r['max_tilt_deg']:>+9.2f}  {status}"
                )
                rows.append({
                    "axis": axis_label, "magnitude": mag, "model": model_name,
                    "contacts_final": r["contacts_final"],
                    "contacts_min": r["contacts_min"],
                    "drop_full_mm": r["drop_full_mm"],
                    "drop_hold_mm": r["drop_hold_mm"],
                    "creep_mm_s": r["creep_mm_s"],
                    "max_tilt_deg": r["max_tilt_deg"],
                    "ejected": int(r["ejected"]),
                })
            except Exception as e:
                print(f"    {mag:>+10.2f}  {model_name:<6}  ERROR: {type(e).__name__}: {str(e)[:50]}")

# Save CSV
with OUT_CSV.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"\n  CSV → {OUT_CSV}")

# ────────────────────────────────────────────────────────────────────────
# Failure-threshold summary per (axis, model)
# ────────────────────────────────────────────────────────────────────────

def first_threshold(rows_subset, key, threshold):
    """First magnitude at which key crosses threshold."""
    for r in rows_subset:
        v = r[key]
        if isinstance(v, float) and not math.isfinite(v):
            return r["magnitude"]
        if v >= threshold:
            return r["magnitude"]
    return None

print("\n" + "=" * 96)
print("  FAILURE THRESHOLDS — first |magnitude| where each metric crosses")
print("=" * 96)
print(f"  {'axis':<6}  {'model':<6}  {'tilt > 5°':>10}  {'tilt > 30°':>11}  "
      f"{'|creep| > 5 mm/s':>17}  {'ejected':>10}")
print("  " + "-" * 92)

for axis_label, _, _ in SWEEP_AXES:
    for model_name in ("point", "cslc", "hydro"):
        # Sort by |magnitude| ascending to find first crossing
        subset = [r for r in rows if r["axis"] == axis_label and r["model"] == model_name]
        subset.sort(key=lambda r: abs(r["magnitude"]))

        t5 = first_threshold(subset, "max_tilt_deg", 5.0)
        t30 = first_threshold(subset, "max_tilt_deg", 30.0)
        # Use absolute creep for threshold detection
        creep_subset = [
            {**r, "_abs_creep": abs(r["creep_mm_s"]) if math.isfinite(r["creep_mm_s"]) else float("inf")}
            for r in subset
        ]
        c5 = first_threshold(creep_subset, "_abs_creep", 5.0)
        ej_row = next((r for r in subset if r["ejected"]), None)
        ej_mag = ej_row["magnitude"] if ej_row else None

        def fmt(v):
            return f"{v:>+10.2f}" if v is not None else f"{'—':>10}"

        print(
            f"  {axis_label:<6}  {model_name:<6}  {fmt(t5)}  {fmt(t30):>11}  "
            f"{fmt(c5):>17}  {fmt(ej_mag):>10}"
        )
