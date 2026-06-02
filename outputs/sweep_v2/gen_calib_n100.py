"""Generate per-scenario calibration CSVs for the n=100 box-whisker sweep.

10 log-spaced stiffnesses per model (same ranges as the n=9 sweep), to be
crossed with 10 δ values at run time → 100 (k, δ) samples per cell.
"""
import csv
import numpy as np

# 10 log-spaced stiffnesses per model, over each model's operating range.
k_per_model = {
    "cslc":  np.logspace(9, 11, 10),                 # 1e9 … 1e11
    "hydro": np.logspace(9, 11, 10),                 # 1e9 … 1e11
    "point": np.logspace(np.log10(2.5e3), np.log10(2.5e5), 10),  # 2.5e3 … 2.5e5
}
# Object (target) stiffness held at each model's default (rigid_object mode
# reads k_rigid_object_obj).  Matches the n=9 calib_bw_*.csv files.
obj_defaults = {"cslc": 5e4, "hydro": 1e10, "point": 2.5e3}

scenarios = [("box", "sphere"), ("box", "box"),
             ("dome", "sphere"), ("dome", "box"), ("box", "bunny")]

for pad, obj in scenarios:
    rows = [{"contact_model": m, "pad_kind": pad, "object_kind": obj,
             "delta_op_mm": 1.0, "F_target_N": float("nan"),
             "F_at_default": float("nan"),
             "k_default": k, "k_symmetric": k,
             "k_rigid_object_pad": k, "k_rigid_object_obj": obj_defaults[m]}
            for m, ks in k_per_model.items() for k in ks]
    path = f"outputs/sweep_v2/calib_bw100_{pad}_{obj}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows = 3 models × 10 k)")
