"""Emit a cleaned box-whisker CSV with a physically-corrected hold column.

The raw ``held`` flag from the runner is purely kinematic (object rose
>5 mm and stayed <0.5 m).  It wrongly counts grasps where the object
slid through the pads at a force too low to support its weight -- e.g.
bunny rows with F_n ~ 0.1 N (below the 0.98 N Coulomb floor) and 5-22 mm
of slip.  Here we add ``held_physical``: held AND F_n >= W/(2 mu) AND
xy_slip < 5 mm.  Original columns are preserved; nothing is deleted.
"""
import csv

OBJ_W = {"sphere": 0.569, "box": 1.086, "bunny": 0.982}
MU = 0.5
MAX_SLIP_M = 5.0e-3

src = "outputs/sweep_v2/sweep_bw25_merged.csv"
dst = "outputs/sweep_v2/sweep_bw25_merged_clean.csv"

rows = list(csv.DictReader(open(src)))
n_raw = sum(int(r["held"]) for r in rows)
for r in rows:
    floor = OBJ_W[r["object_kind"]] / (2 * MU)
    phys = (int(r["held"]) == 1
            and float(r["F_n_hold"]) >= floor
            and float(r["xy_slip_max"]) < MAX_SLIP_M)
    r["held_physical"] = 1 if phys else 0

fields = list(rows[0].keys())
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

n_phys = sum(r["held_physical"] for r in rows)
print(f"wrote {dst}")
print(f"  raw held=1: {n_raw}/{len(rows)}   physically held: {n_phys}/{len(rows)}")
print(f"  removed {n_raw - n_phys} spurious holds")
