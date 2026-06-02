# CSLC v2 vs Hydro vs Point — Reproducible sweeps

This directory holds the paper-grade artifacts from the contact-model
comparison study. All commands below regenerate the corresponding CSVs
and PNGs from scratch. All times are wall-clock on RTX 3070.

Prerequisites: CSLC theory fixes A + B + D1 + D2 must be present in
`newton/_src/geometry/cslc_kernels.py`. The grasp scenario is
pad ∈ {box, dome} × object ∈ {sphere, box}, gripper closing on a
33.5 mm-radius tennis-ball-density object.

---

## 1. Headline figure: box-whisker F_n distribution per scenario

**The publication plot.** For each (pad × object) cell, samples F_n over
9 (k × δ) combinations within each model's effective operating range:
- CSLC: kc ∈ {1e9, 1e10, 1e11}, k_obj at our default 5e4
- Hydro: kh ∈ {1e9, 1e10, 1e11}, k_obj at Newton default 1e10
- Point: ke ∈ {2.5e3, 2.5e4, 2.5e5}, k_obj at Newton default 2.5e3
- δ ∈ {0.5, 1.0, 1.5} mm

**Run in parallel by scenario** (~20 min wall on 4 GPU streams):

```bash
# 1a. Build the per-scenario calibration CSVs (already present; regenerate if lost)
python3 -c "
import csv
k_per_model = {'cslc':[1e9,1e10,1e11], 'hydro':[1e9,1e10,1e11], 'point':[2.5e3,2.5e4,2.5e5]}
obj_defaults = {'cslc':5e4, 'hydro':1e10, 'point':2.5e3}
for pad,obj in [('box','sphere'),('box','box'),('dome','sphere'),('dome','box')]:
    rows=[{'contact_model':m,'pad_kind':pad,'object_kind':obj,'delta_op_mm':1.0,
           'F_target_N':float('nan'),'F_at_default':float('nan'),
           'k_default':k,'k_symmetric':k,'k_rigid_object_pad':k,'k_rigid_object_obj':obj_defaults[m]}
          for m,ks in k_per_model.items() for k in ks]
    with open(f'outputs/sweep_v2/calib_bw_{pad}_{obj}.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
"

# 1b. Run all four sweeps in parallel (background; ~16 min each)
for cell in box_sphere box_box dome_sphere dome_box; do
    uv run -m cslc_main.grasp.scripts.compare_grasp_matched_F \
        --calibration outputs/sweep_v2/calib_bw_${cell}.csv --modes rigid_object \
        --face-pens-mm 0.5,1.0,1.5 \
        --output outputs/sweep_v2/sweep_bw_${cell}.csv &
done
wait

# 1c. Merge and plot
python3 -c "
import csv; rows=[]
for c in ['box_sphere','box_box','dome_sphere','dome_box']:
    rows += list(csv.DictReader(open(f'outputs/sweep_v2/sweep_bw_{c}.csv')))
with open('outputs/sweep_v2/sweep_bw_merged.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
"
uv run -m cslc_main.grasp.scripts.plot_box_whisker \
    --input outputs/sweep_v2/sweep_bw_merged.csv \
    --output outputs/sweep_v2/boxwhisker_log.png --log-y \
    --title "Lift F_n distribution per scenario"
uv run -m cslc_main.grasp.scripts.plot_box_whisker \
    --input outputs/sweep_v2/sweep_bw_merged.csv \
    --output outputs/sweep_v2/boxwhisker.png \
    --title "Lift F_n distribution per scenario"
```

Artifacts: `boxwhisker.png`, `boxwhisker_log.png`, `sweep_bw_merged.csv`,
plus the 4 per-scenario CSVs.

---

## 2. Operating-range sweep around Newton defaults (box/sphere)

Finds where each model is responsive vs saturated. 30 runs, ~18 min.
Sweeps k ∈ {0.01×, 0.1×, 1×, 10×, 100×} of each model's library default,
at δ ∈ {0.5, 1.0} mm.

```bash
uv run -m cslc_main.grasp.scripts.compare_grasp_matched_F \
    --calibration outputs/sweep_v2/calib_newton_defaults.csv \
    --modes rigid_object --face-pens-mm 0.5,1.0 \
    --output outputs/sweep_v2/sweep_newton_defaults.csv

# Extended: push Point further (ke up to 10000× Newton) to find ceiling
uv run -m cslc_main.grasp.scripts.compare_grasp_matched_F \
    --calibration outputs/sweep_v2/calib_point_extend.csv \
    --modes rigid_object --face-pens-mm 0.5,1.0 \
    --output outputs/sweep_v2/sweep_point_extend.csv

# Merge, plot heatmap + log/linear lines
python3 -c "
import csv; rows=[]
for p in ('sweep_newton_defaults.csv','sweep_point_extend.csv'):
    rows += list(csv.DictReader(open(f'outputs/sweep_v2/{p}')))
seen={}
for r in rows:
    seen[(r['contact_model'],round(float(r['k_pad']),4),round(float(r['face_pen_mm']),3))]=r
m=list(seen.values())
with open('outputs/sweep_v2/sweep_newton_defaults_extended.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(m[0].keys())); w.writeheader(); w.writerows(m)
"
uv run -m cslc_main.grasp.scripts.plot_2d_envelope \
    --input outputs/sweep_v2/sweep_newton_defaults_extended.csv \
    --output outputs/sweep_v2/sweep_newton_defaults_extended.png \
    --title "Operating range around Newton defaults"
uv run -m cslc_main.grasp.scripts.plot_force_vs_overlap_lines \
    --input outputs/sweep_v2/sweep_newton_defaults_extended.csv \
    --output outputs/sweep_v2/lines_newton_defaults_extended_log.png --log-y
uv run -m cslc_main.grasp.scripts.plot_force_vs_overlap_lines \
    --input outputs/sweep_v2/sweep_newton_defaults_extended.csv \
    --output outputs/sweep_v2/lines_newton_defaults_extended.png
```

Side experiment: Point symmetric (both pad and object scaled) — proves
Point CAN reach 30+ N if k_obj also scales:

```bash
uv run -m cslc_main.grasp.scripts.compare_grasp_matched_F \
    --calibration outputs/sweep_v2/calib_point_symmetric.csv \
    --modes rigid_object --face-pens-mm 1.0 \
    --output outputs/sweep_v2/sweep_point_symmetric.csv
```

Artifacts: `sweep_newton_defaults_extended.csv` + `.png`,
`lines_newton_defaults_extended*.png`,
`sweep_point_symmetric.csv`.

---

## 3. 2D lift envelope (box/sphere, 5×5 grid)

Finds the failure boundary in (k, δ) space. 75 runs, ~45 min sequential.

```bash
# Run one full 5×5 grid: stiffness mult × overlap on box/sphere only
uv run -m cslc_main.grasp.scripts.compare_grasp_matched_F \
    --calibration outputs/sweep_v2/calib_3models.csv --modes rigid_object \
    --face-pens-mm 0.05,0.10,0.30,0.70,1.50 \
    --output outputs/sweep_v2/sweep_2d_merged.csv
# (Re-derive calib with --primary-stiffness sweep if needed; the calib_3models.csv
# in this dir uses default stiffness only. Use compare_grasp_matched_F with multiple
# k_pad rows in the calibration CSV to cover the full 5 stiffness values.)

uv run -m cslc_main.grasp.scripts.plot_2d_envelope \
    --input outputs/sweep_v2/sweep_2d_merged.csv \
    --output outputs/sweep_v2/sweep_2d_merged.png \
    --title "2D lift envelope (5 stiffness × 5 overlap)"
```

Artifacts: `sweep_2d_merged.csv` + `.png`.

---

## 4. Lift envelope vs overlap at default stiffness (all 4 scenarios)

Shows the minimum overlap each model needs to lift, at default stiffness.
72 runs, ~45 min sequential.

```bash
uv run -m cslc_main.grasp.scripts.compare_grasp_matched_F \
    --calibration outputs/sweep_v2/calib_3models.csv --modes rigid_object \
    --face-pens-mm 0.2,0.5,0.8,1.0,1.5,2.0 \
    --output outputs/sweep_v2/lift_envelope_overlap.csv
uv run -m cslc_main.grasp.scripts.plot_lift_envelope \
    --input outputs/sweep_v2/lift_envelope_overlap.csv \
    --output outputs/sweep_v2/lift_envelope_overlap.png
```

Artifacts: `lift_envelope_overlap.csv` + `.png`.

---

## 5. Default-stiffness 3-model comparison (baseline)

12 runs, ~5 min. Each model at its own GraspConfig default stiffness.

```bash
uv run -m cslc_main.grasp.scripts.compare_grasp_matched_F \
    --calibration outputs/sweep_v2/calib_3models.csv --modes rigid_object \
    --output outputs/sweep_v2/stability_3models.csv
uv run -m cslc_main.grasp.scripts.plot_stability \
    --input outputs/sweep_v2/stability_3models.csv \
    --output outputs/sweep_v2/stability_3models.png \
    --title "Grasp dynamics at default stiffness — CSLC vs Hydro vs Point"
```

Artifacts: `stability_3models.csv` + `.png`.

---

## Per-run grasp logs

Each `compare_grasp_matched_F` run also creates a subdir under
`outputs/matched_F/<contact_model>_<pad>_<obj>_<mode>[_pen<X>mm]/`
with the full per-step `timeseries.csv` and (for CSLC) `cslc_state.csv`.
These are regenerated on every sweep and can be deleted safely; the
main artifacts above (the sweep CSVs and PNGs) are self-contained.

## What was archived

- `outputs/sweep_v2/_archive/` — intermediate calibration CSVs, early
  sweep variants, t1/t2/t3/t4/t6 verification probes, superseded plots.
- `outputs/_archive_pre_fix/sweep/` — entire pre-CSLC-fixes sweep
  directory (force_curve_matrix, calibrated_stiffness, eps/ka probes
  from the broken-kernel era).
