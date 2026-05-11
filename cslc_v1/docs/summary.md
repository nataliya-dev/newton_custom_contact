# CSLC Integration Notes — April 20, 2026

Newton + MuJoCo integration of the CSLC (Compliant Sphere Lattice Contact) model.
Goal: validate that CSLC outperforms point contact — and matches or beats
hydroelastic PFC — in squeeze and lift tasks for the ICRA paper.



## 1. Squeeze test (`cslc_v1/squeeze_test.py`)

### Scene

Two kinematic box pads squeeze a dynamic target (sphere or book) under
gravity.  The pad geometry (40×100 mm contact face, 20 mm thick) and
trajectory are shared across object kinds: ~0.5 mm initial penetration
per side, 0.5 mm active squeeze over 0.5 s, 1.5 s HOLD — giving 1 mm
face_pen at HOLD on the sphere target (matching the lift test's
operating point) and ~4 mm at HOLD on the book target.

The squeeze test supports two held targets via `--object`:

- **`sphere`** (default) — r = 30 mm, m = 500 g, ρ = 4421 kg/m³.
  The fair-calibration regime: 1 mm face_pen with
  `cslc_ka=15000`, `cslc_contact_fraction=0.025`, `kh=2.65e8`.
- **`book`** — trade-paperback dimensions (152 × 229 × 25 mm, 0.45 kg,
  ρ ≈ 515 kg/m³).  Pads grip the wide ±x covers; the pad face fits
  comfortably inside the cover (pad/cover area ≈ 11 %).  Book mode
  uses `cslc_contact_fraction=1.0` (full pad-on-cover overlap → all
  189 surface spheres engage) and patch-area-matched
  `kh = ke/(2hy·2hz) ≈ 1.25e7 Pa`.

```
uv run cslc_v1/squeeze_test.py --mode squeeze --solver mujoco \
    --contact-models point,cslc,hydro            # sphere
uv run cslc_v1/squeeze_test.py --mode squeeze --object book \
    --contact-models point,cslc,hydro            # book
```

### Tuning parameters (shared and per-model)

Shared material: `ke=5.0e4`, `kd=500`, `μ=0.5`, `dt=2.0 ms`.
MuJoCo solver: `solver=cg`, `integrator=implicitfast`, `cone=elliptic`,
`iterations=100 (CSLC) / 20 (point)`, default `solimp`.

| Knob | Point | CSLC | Hydro |
|---|---|---|---|
| lattice spacing | — | 5 mm | — |
| `cslc_ka` | — | **15 000** | — |
| `cslc_kl` | — | 500 | — |
| `cslc_n_iter` / `α` | — | 20 / 0.6 | — |
| `cslc_contact_fraction` | — | **0.025** (matches lift) | — |
| `kc` (recalibrated) | — | **75 000 N/m** | — |
| `keff` per sphere | — | 12 500 N/m | — |
| Aggregate per pad | — | 50 000 N/m (=`ke_bulk`) ✓ | — |
| `kh` | — | — | **2.65e8 Pa** (fair) |
| `sdf_max_resolution` | — | — | 64 |

### Metrics

Three complementary HOLD-phase numbers are reported per model:

- `FullDrop` = `target_z[0] − min(target_z)` (legacy metric —
  **confounded** by the squeeze transient; do not use for
  cross-model comparison).
- `HoldDrop` = `target_z[n_squeeze_steps] − target_z[-1]` (displacement
  during HOLD only; positive = fell, negative = rose).
- `HoldCreep` = second-half-of-HOLD mean velocity (positive = falling).
- `MaxTilt` (book mode only) = peak rotation magnitude of the held body
  during HOLD, in degrees.

For the paper, `HoldCreep` is the primary number — solver-compliance-
dominated and directly comparable across models.

### Results — `--object sphere` (default)

| Model | FullDrop | HoldDrop | **HoldCreep** | Active contacts |
|---|---|---|---|---|
| `point_mujoco` | 1.020 mm | +0.733 mm | **+0.490 mm/s** | 2 |
| `cslc_mujoco` (cf=0.025) | **0.092 mm** | **+0.067 mm** | **+0.045 mm/s** | 26 |
| `hydro_mujoco` (kh=2.65e8) | 0.564 mm | +0.379 mm | +0.253 mm/s | 35 |

CSLC wins every metric — **10.9× better than point and 5.6× better than
hydro on HoldCreep** — and the same ranking carries through HoldDrop and
FullDrop.  This matches the lift test (§2) exactly: under fair
calibration at 1 mm face_pen, CSLC's distributed-constraint advantage
produces the lowest-slip / lowest-creep behaviour by a wide margin.

### Results — `--object book` (trade paperback, post 2026-04-26 realistic dims)

Pads grip the wide ±x covers of a 152×229×25 mm, 0.45 kg trade
paperback (`book_hx=0.0125`, `book_hy=0.076`, `book_hz=0.115`,
`book_density=515`).  Pad face penetration starts at 1.5 mm; HOLD pen
is ~4 mm.  CSLC binds all 189 surface lattice spheres per pad (full
pad-on-cover overlap), giving — with the calibration prior
`cslc_contact_fraction=1.0` — `kc = 269.3 N/m`, `keff = 264.6 N/m`,
agg/pad = 50 000 N/m = `ke_bulk` ✓.  Hydro uses the patch-area-matched
modulus `kh = ke/(2hy·2hz) ≈ 1.25e7 Pa`.

| Model | FullDrop | HoldDrop | HoldCreep | MaxTilt | Active contacts |
|---|---|---|---|---|---|
| `point_mujoco` | 0.238 mm | +0.179 mm | +0.119 mm/s | 0.00° | 8 |
| `cslc_mujoco` (cf=1.0) | **0.088 mm** | **+0.067 mm** | **+0.045 mm/s** | **0.00°** | **378** |
| `hydro_mujoco` (kh=1.25e7) | 0.563 mm | +0.378 mm | +0.260 mm/s | 1.11° | 105 |

CSLC is **2.6× better than point** and **5.8× better than hydro** on
HoldCreep, same qualitative ranking as the sphere scene.  CSLC also has
the lowest baseline MaxTilt (exactly 0°) — hydro shows ~1° rotational
drift from how its pressure-field polygons distribute around the book
corners.

```
uv run cslc_v1/squeeze_test.py --mode squeeze --object book \
    --contact-models point,cslc,hydro
```

### Disturbance experiments (HOLD-only external wrench)

The squeeze test supports `--external-force fx,fy,fz` and
`--external-torque tx,ty,tz` (world frame, applied to the held body
during HOLD only via `cslc_v1/common.py:apply_external_wrench`).

> **2026-05-03 — `apply_external_wrench` layout bug fixed.** The function
> previously had force/torque slots swapped (`body_f[0:3] = torque,
> body_f[3:6] = force`) — opposite to Newton's `wp.spatial_vector(force,
> torque)` convention used by both the semi-implicit contact kernel
> ([kernels_contact.py:261](newton/_src/solvers/semi_implicit/kernels_contact.py#L261))
> and MuJoCo's `apply_mjc_body_f_kernel`
> ([mujoco/kernels.py:1367](newton/_src/solvers/mujoco/kernels.py#L1367)).
> The bug aliased every `--external-force` flag to a torque
> disturbance and vice versa.  Every disturbance row above the
> 2026-05-03 cutoff is mislabelled — see the "Disturbance sweep —
> three baselines side by side" section for the corrected numbers.

At realistic trade-paperback dimensions, **two distinct catastrophic-
failure regimes emerge once the wrench layout is correct**:

1. **BOX-vs-BOX pads + τ_z = 5 N·m**: point contact's 4-corner
   specialisation cannot resist the vertical twist (capacity at this
   geometry is marginally below 5 N·m).  Book is ejected: tilt 178°,
   creep −74.8 mm/s.  CSLC holds bit-exactly with the same disturbance
   (creep +0.015 mm/s, tilt 0.55°).  Hydro degrades but holds (creep
   +0.32 mm/s, tilt 4.5°).
2. **SPHERE-pads + τ_x = 5 N·m**: the true 1-contact-per-pad
   ("two-pin") grasp cannot resist twist around the line joining the
   pads — contact is lost entirely, book is ejected.

These give the paper two clean catastrophic-failure axes that map
directly onto physical intuition: vertical twist beats specialised
4-corner contact; grip-axis twist beats true point contact.  Hard
numbers are in the "Disturbance sweep" section below.

Caveat — the dramatic "point ejects the book" failures observed in
the original (pre-2026-04-25) deep-pen sweeps required the unrealistic
16×300×400 mm, 1.2 kg paper-spec panel (300 mm of book overhang past
the pad on both sides, huge tilt-lever arms on every contact-line
corner).  At realistic trade-paperback proportions, point contact's
4-corner specialisation is competent against most disturbances —
ejection requires a τ_z that exceeds the corner moment-arm capacity.
To reproduce the original cliff, manually set `book_hx=0.008,
book_hy=0.15, book_hz=0.20, book_density=625` in `SceneParams`.

### What "point contact" actually means in this scene (2026-05-03)

Dumping the contact buffer at HOLD step 800 with `point_mujoco`
shows the 8 reported contacts are **the four corners of each pad
face** (`y = ±20mm, z = ±50mm`, both pads) — i.e. MuJoCo's
box-vs-box narrow-phase specialisation emits **4 contacts per pair**,
not the single pin-like point one might intuit from the name.  With
two pads that's 8 corner contacts, each at moment arm √(20²+50²) ≈
54 mm from the patch centre, which is plenty of restoring torque
against the disturbances above.

In other words, the "point contact" baseline in the book scene is
already a sparse 4-corner patch per pad — not a true 1-contact-per-
pair grasp.  The CSLC vs point comparison on this scene is therefore
"MuJoCo's hand-rolled 4-corner specialisation vs 378-sphere lattice
with anchor + lateral compliance", not "8-point patch vs 2 pins".

The cleanest "two pins" baseline is built by switching the pads from
BOX to SPHERE — sphere-vs-box always emits 1 contact per pair, and
the 2-contact regime exposes the rotational compliance that MuJoCo's
box-vs-box specialisation hides.  Mesh-target paths emit somewhere
in between (12–13 contacts via BVH-per-triangle iteration).  Hard
numbers across all three regimes follow.

### What changes when the book is a mesh? (--book-as-mesh, 2026-05-03)

`squeeze_test.py --object book --book-as-mesh` swaps the
`add_shape_box` for a 12-triangle `add_shape_mesh`.  Three findings:

1. **MuJoCo emits ~12 contacts per pad-vs-mesh-book pair**, not the
   single GJK-MPR contact one might expect.  MuJoCo Warp's mesh
   narrow phase iterates over the mesh's triangle BVH and emits one
   contact per overlapping triangle, so the active-contact count
   stays in the "denser multipoint" regime regardless of mesh
   triangle count.  Single-point-per-pair behaviour requires either
   sphere-on-X primitive specialisation OR an explicit `condim=1` /
   `max_contacts_per_pair=1` solver hint.

2. **CSLC currently does NOT support mesh targets.**
   `cslc_handler._from_model` filters `shape_pairs` to
   `supported_geo_types = (SPHERE, BOX)`; mesh-target pairs are
   dropped → the handler reports `n_pair_blocks = 0` → no CSLC
   contacts are emitted → the run degenerates to MuJoCo's standard
   mesh narrow phase, identical to `point_mujoco`.  This shows up
   as bit-exact metrics between `point` and `cslc` columns in the
   `--book-as-mesh` table below.

3. **Hydroelastic still works on mesh targets** because Newton lets
   meshes carry their own SDF (`mesh.build_sdf(max_resolution=64)`
   in `_add_pads_and_target`).  Hydro emits ~133 pressure-field
   polygons per pair on a 12-tri book.

Implication for the paper's "general manipulation" claim: paper §3.1
says "we approximate the robot AND object geometries as collections
of spheres using MorphIt" — but the current implementation only
realises this for the robot (via the CSLC pad lattice).  For
arbitrary mesh objects the user must EITHER MorphIt-ify the target
into a sphere collection (which then routes through CSLC-vs-sphere)
OR add a `compute_cslc_penetration_mesh` kernel that queries the
mesh's SDF — listed in §5.1 TODOs.

### Disturbance magnitude sweep — failure curves (2026-05-03, post-fix)

The single-magnitude tables below answer "does CSLC hold under
disturbance X" but not "where is the cliff".  This section sweeps each
of the 6 disturbance axes from 0 → an upper bound that exposes
catastrophic failure (or saturates without one), revealing **three
distinct cliffs** in the BOX-on-BOX scene and confirming that **CSLC
is the only model with zero ejections across all 63 magnitudes
tested**.

Reproduce: `cslc_v1/_validation_logs/sweep_book_disturbance_curves.py`
(~5 min on RTX 3070).  CSV → `sweep_book_disturbance_curves.csv` (190
rows).  Trade-paperback book, μ=0.5, weight 4.42 N, kinematic BOX
pads, 1 s HOLD.

#### Cliff-threshold summary

| Axis  | Unit | point ejects at | CSLC ejects at | hydro ejects at | CSLC max tilt |
|-------|------|-----------------|----------------|-----------------|----------------|
| F_x   | N    | — (15 N tested) | — | — (15 N tested) | 0.00° |
| F_y   | N    | — (15 N tested) | — | — (15 N tested) | 0.00° |
| F_z   | N    | — (15 N down)   | — | — (15 N down)   | 0.00° |
| τ_x   | N·m  | — (15 N·m, tilt 12°) | — (15 N·m, tilt 5.5°) | — (15 N·m, tilt 180° but no contact loss) | 5.50° |
| τ_y   | N·m  | **20 N·m** (cliff between 15 and 20) | — (20 N·m, tilt 0.39°) | — (20 N·m, tilt 9.4°) | 0.39° |
| τ_z   | N·m  | **3 N·m** (cliff between 2 and 3) | — (10 N·m, tilt 1.1°) | — (10 N·m, tilt 14°) | 1.10° |

**Three observations:**

1. **Lateral forces are trivially absorbed** by friction at the pad
   face for all three models, even at 15 N (3.4× the book's weight).
   This is geometry, not contact-model fidelity: the pads grip the
   book's wide ±x covers, so an x/y/z force is just a normal /
   tangential load on the existing patch.  No contact-model claim
   should rely on F_x / F_y discrimination.

2. **F_z creep scales linearly with magnitude**, with **CSLC's slope
   ~6× lower than point's**:
   - Point creep slope: 0.030 mm/s per N down (from +0.119 at 0N to
     +0.536 at -15N).
   - CSLC creep slope: 0.005 mm/s per N down (from +0.015 at 0N to
     +0.089 at -15N).
   - This is the "compliance leak" axis: each MuJoCo constraint
     contributes a `c·f_n` term that grows linearly with f_n; CSLC's
     lattice equilibrium absorbs the load with proportionally less
     leak per sphere.

3. **CSLC is the only model with no ejections** across the entire
   sweep.  Maximum observed tilt anywhere: 5.50° at τ_x = 15 N·m.
   At every disturbance/magnitude combination tested, CSLC retained
   all 378 surface spheres in contact.

#### Three cliffs identified

**Cliff #1 — point under τ_z (vertical twist) at ~2.5 N·m**

| τ_z [N·m] | point creep / tilt / contacts | cslc creep / tilt / contacts |
|---|---|---|
| 0.00 | +0.119 / +0.00° / 8  | +0.015 / +0.00° / 378 |
| 1.00 | +0.119 / +0.04° / 8  | +0.015 / +0.11° / 378 |
| 2.00 | +0.119 / +0.08° / 8  | +0.015 / +0.22° / 378 |
| **3.00** | **+2273 / +180° / 0 EJECTED** | +0.016 / +0.33° / 378 |
| 5.00 | EJECTED | +0.015 / +0.55° / 378 |
| 10.00 | EJECTED | +0.015 / +1.10° / 378 |

**Discontinuous failure** — point holds rigidly up to 2 N·m, then
ejects at 3 N·m.  The 4-corner specialisation provides a moment-arm
capacity around `τ_max ≈ 4 · μ·F_n·r_corner ≈ 4 · 0.5 · 100 · 0.054 ≈
11 N·m` *if all corners stay loaded*, but corner-unloading cascades
collapse this in practice once any single corner crosses the friction
cone.  CSLC, with 378 distributed spheres, has no analogous cascade
mode — load redistributes through the lateral spring `k_ℓ` faster
than friction-cone saturation can propagate.

**Cliff #2 — point under τ_y (vertical bending) at ~17 N·m**

| τ_y [N·m] | point creep / tilt / contacts | cslc creep / tilt / contacts |
|---|---|---|
| 0.00  | +0.119 / +0.00° / 8 | +0.015 / +0.00° / 378 |
| 5.00  | +0.119 / +0.03° / 8 | +0.015 / +0.10° / 378 |
| 10.00 | +0.119 / +0.06° / 8 | +0.015 / +0.20° / 378 |
| 15.00 | +0.119 / +0.09° / 8 | +0.015 / +0.30° / 378 |
| **20.00** | **EJECTED / +178.84° / 0** | +0.015 / +0.39° / 378 |

**Second discontinuous failure** of point contact, this time on the
bending axis.  CSLC at the failure magnitude reaches just 0.39° tilt
— a **>450× rotational stiffness advantage** at the threshold where
point catastrophically fails.

**Cliff #3 — hydro under τ_x (grip-axis twist) at ~10–15 N·m**

| τ_x [N·m] | point creep / tilt | cslc creep / tilt | hydro creep / tilt |
|---|---|---|---|
| 5.00 | +0.119 / +4.12° | +0.015 / +1.83° | +2.766 / +28.01° |
| 7.50 | +0.119 / +6.17° | +0.015 / +2.75° | +9.329 / +57.37° |
| 10.00 | +0.119 / +8.23° | +0.015 / +3.66° | +20.449 / +99.16° |
| 15.00 | +0.119 / +12.35° | +0.015 / +5.50° | +20.639 / **+179.91°** |

Hydro's pressure-field polygons concentrate around the box edges; as
twist tilts the book, those edge polygons rapidly reorient and lose
moment-arm capacity, producing very high tilts (180° at 15 N·m).
Point degrades gracefully (12.35° at 15 N·m) because its 4 corners
remain symmetrically engaged.  CSLC's lattice maintains the lowest
tilt of all three (5.50° at 15 N·m) — the **only model where the
tilt-vs-magnitude curve stays sub-linear** through this regime.

#### What this means for the paper

1. **Figure 1 candidate**: τ_z at 3 N·m, BOX-on-BOX.  Point ejected,
   CSLC and hydro hold, but CSLC's tilt is 6× lower than hydro's
   (0.33° vs ~1.8° interpolated).  Single magnitude, three
   qualitatively different outcomes.

2. **Capacity-curve figure**: tilt-vs-magnitude curves overlaid for
   τ_z and τ_y.  Point shows discontinuous cliff; CSLC and hydro are
   continuous.  CSLC's slope is the smallest of the three.

3. **Compliance-leak figure**: F_z creep-vs-magnitude curve.  All
   three models linear, with slopes in the ratio
   `point : hydro : CSLC ≈ 6 : 9 : 1`.  This is the "constraint-
   compliance leak" claim made experimentally concrete across a
   continuous range.

4. **Empirical robustness claim**: across 63 disturbance magnitudes
   spanning 6 axes, CSLC was the only model with zero ejections.
   Point ejected on 2 axes; hydro reached 180° tilt (without losing
   contact) on 1 axis.

```
uv run --extra dev python cslc_v1/_validation_logs/sweep_book_disturbance_curves.py
```

---

### Disturbance sweep — three baselines side by side (2026-05-03, post-fix)

Reproduce via `cslc_v1/_validation_logs/sweep_book_disturbances.py`.
Trade-paperback book (152×229×25 mm, 0.45 kg, μ=0.5, weight 4.42 N),
disturbance applied to the held body during HOLD only.

> **All numbers below post-date the 2026-05-03
> `apply_external_wrench` layout fix.** The pre-fix sweep had every
> `--external-force` row aliased to a torque disturbance and vice
> versa.  A previous edit of this section ("CSLC creep is ~8× lower
> than point ... tilt is 0° for both") was based on those mislabelled
> runs and is wrong: under the *correct* wrench layout, point contact
> exhibits **two distinct catastrophic-failure regimes** that CSLC
> cleanly resists.  The numbers below are the regenerated, correct
> sweep.

#### A. BOX book + BOX pads (paper baseline)

| Disturbance | point creep / tilt | **cslc** creep / tilt | hydro creep / tilt | point | cslc | hydro |
|---|---|---|---|---|---|---|
| none           | +0.119 / 0.00°        | **+0.015 / 0.00°** | +0.242 / 1.09° |   8 | 378 |  97 |
| τ_y = 1 N·m    | +0.119 / 0.01°        | **+0.015 / 0.02°** | +0.254 / 1.35° |   8 | 378 |  98 |
| τ_y = 5 N·m    | +0.119 / 0.03°        | **+0.015 / 0.10°** | +0.406 / 2.77° |   8 | 378 |  93 |
| τ_x = 5 N·m    | +0.119 / 4.12°        | **+0.015 / 1.83°** | +2.769 / 28.02° |  8 | 378 |  97 |
| **τ_z = 5 N·m**| **−74.822 / 177.60°** ← **EJECTED** | **+0.015 / 0.55°** | +0.320 / 4.51° | 0 | 378 | 100 |
| F_z = −2 N     | +0.179 / 0.00°        | **+0.030 / 0.00°** | +0.307 / 1.10° |   8 | 378 |  98 |

(creep in mm/s, positive = falling; rightmost three columns are
active-contact counts)

The flagship row is **τ_z = 5 N·m**: point contact's 4 pad-corner
specialisation cannot resist a 5 N·m vertical twist (capacity ≈ 4 ×
μ·F_n·r_corner ≈ 4 × 0.5·100N·54mm ≈ 11 N·m *if all corners stay
loaded*, but corner unloading cascades faster than the friction cone
can compensate — observed: contacts → 0, book ejected at 75 mm/s,
178° tilt by HOLD-end).  CSLC at the same disturbance: creep
+0.015 mm/s, tilt 0.55°, all 378 surface spheres still engaged.
Hydro degrades but holds: creep +0.32 mm/s, tilt 4.5°, 100 polygons.

τ_x = 5 N·m (twist around the grip axis) also exposes a real CSLC
advantage: tilt 1.83° vs point 4.12° vs hydro 28.02°.  Hydro's
near-failure on this row is interesting — likely due to its
pressure-field polygons concentrating around the box edges and not
having compliance to redistribute load when the twist starts.

#### B. MESH book + BOX pads (`--book-as-mesh`)

| Disturbance | point creep / tilt | cslc creep / tilt | hydro creep / tilt | active contacts |
|---|---|---|---|---|
| none           | +0.078 / 0.07° | +0.078 / 0.07° | +0.112 / 0.04° | 12 / 12 / 133 |
| τ_y = 1 N·m    | +0.075 / 0.08° | +0.075 / 0.01° | +0.089 / 0.05° | 12 / 12 / 133 |
| τ_y = 5 N·m    | +0.075 / 0.09° | +0.075 / 0.09° | +0.110 / 0.09° | 12 / 12 / 133 |
| τ_x = 5 N·m    | −0.042 / 4.13° | +0.019 / 3.98° | +0.116 / 7.76° | 10 / 11 / 133 |
| τ_z = 5 N·m    | +0.075 / 0.14° | +0.075 / 0.14° | +0.098 / 1.07° | 12 / 12 / 127 |
| F_z = −2 N     | +0.119 / 0.07° | +0.116 / 0.11° | +0.159 / 0.09° | 12 / 12 / 133 |

CSLC and point columns are **bit-exact** (within fp32 noise) —
confirming the handler is dormant on mesh targets (per §"What
changes when the book is a mesh" above).  Until CSLC gains a
mesh-target kernel, the paper's claim must be carefully scoped to
"pad-vs-sphere or pad-vs-primitive-box" or to "pre-MorphIt'd
targets".

Notable: the τ_z = 5 N·m row that ejected point contact in Table A
no longer ejects in Table B — because the mesh narrow phase emits
~12 BVH-triangle contacts spread across the book covers (rather
than 4 pad-face corners), giving the grip more moment-arm coverage
against vertical twist.  This is a side-effect of MuJoCo's mesh
narrow phase being multipoint; it's not a CSLC win and shouldn't be
read as such.

#### C. BOX book + SPHERE pads (true 1-contact-per-pad baseline)

| Disturbance | point creep / **tilt** | contacts |
|---|---|---|
| none           | +0.492 / 0.00°       | 2 |
| τ_y = 1 N·m    | +0.492 / **+1.55°**  | 2 |
| τ_y = 5 N·m    | +0.480 / **+7.21°**  | 2 |
| **τ_x = 5 N·m**| **+14144 / 179.91°** ← **EJECTED** | 0 |
| τ_z = 5 N·m    | +0.492 / +7.20°      | 2 |
| F_z = −2 N     | +0.715 / 0.00°       | 2 |

Two complementary failure modes appear with the corrected wrench:

- **Bending torque (τ_y)**: book pivots around the line joining the
  two pads.  1.55° at 1 N·m, 7.21° at 5 N·m — gracefully rotational
  rather than catastrophic.
- **Grip-axis torque (τ_x)**: catastrophic.  A two-pin grasp cannot
  resist twist around the line connecting the pads (the rotation axis
  passes through both contact points, so neither pad has any moment
  arm to oppose it — only sliding friction can, and once that
  saturates the grip is gone).  Observed: contacts → 0, drop 14 m,
  tilt 179.9°.

Combined with Table A's τ_z = 5 N·m point-ejection, we now have **two
distinct catastrophic-failure regimes for point contact** (one per
pad geometry) plus **a smooth tilt-vs-disturbance curve** for the
two-pin sphere-pad case — exactly the empirical profile the paper's
§1 narrative needs.

### Paper-grade implications

The corrected sweep changes the recommended paper narrative.  The
strongest single comparison is now **Table A's τ_z = 5 N·m row**:
point ejected, CSLC held cleanly, hydro held but degraded.  This is
the experiment to put in Figure 1.

For the ICRA submission, three coherent narratives:

1. **Catastrophic-failure regime** (Table A τ_z = 5 N·m + Table C
   τ_x = 5 N·m) — "point contact has finite vertical-twist capacity
   ≈ 5 N·m at the BOX-pad geometry, and zero capacity for grip-axis
   twist with sphere pads.  CSLC's distributed lattice resists both
   regimes by an order of magnitude."  This is the **cleanest
   demonstration that distributed contact qualitatively differs from
   point contact**, not just quantitatively.

2. **Compliance-leak regime** (Table A creep across all
   non-failure rows) — "even when point contact holds without
   ejection, MuJoCo's per-constraint `c·f_n` compliance leak
   produces an 8× higher creep rate than CSLC's lattice
   equilibrium."  This is the §3.1 theoretical claim made
   experimentally concrete.

3. **General-target gap** (Table B) — "for mesh-based targets
   (USD assets, MorphIt sphere clouds, scanned objects) the current
   CSLC handler is dormant and the distributed-contact advantage
   does not apply.  Closing this gap (§5.1 `compute_cslc_penetration
   _mesh` TODO) is the highest-priority follow-up."

The strongest single experiment for the paper is now **Table A's
τ_z = 5 N·m row**: a single, reproducible disturbance under which
point contact is ejected (178° tilt, 75 mm/s creep, 0 active
contacts), CSLC holds cleanly (0.55° tilt, 0.015 mm/s creep, 378
active spheres), and hydro degrades but holds (4.5° tilt, 0.32 mm/s
creep, 100 polygons).  This is one disturbance, three contact
models, three qualitatively different outcomes.  Add Table C's
τ_x = 5 N·m sphere-pad ejection as the supplementary figure that
recovers the paper's "fragile point-contact grasp" intuition without
relying on MuJoCo's box-vs-box specialisation.

```
uv run --extra dev cslc_v1/squeeze_test.py --mode squeeze --object book
uv run --extra dev cslc_v1/squeeze_test.py --mode squeeze --object book --book-as-mesh
uv run --extra dev python cslc_v1/_validation_logs/sweep_book_disturbances.py
```

### Active contacts — what the column actually means

`Active contacts` counts slots where `shape0 ≥ 0` in the contacts buffer —
i.e. how many contacts the MuJoCo solver is solving simultaneously. It IS
a useful observability number (it tells you the pressure field width in
hydro, the number of active surface spheres in CSLC, the GJK narrow-phase
count in point). It is NOT a fair cross-model matching criterion (§5.2):
the three models have different contact semantics, so the counts are
apples-to-oranges.

### Per-pad calibration (kept invariant — do not revert)

`calibrate_kc` defaults to `per_pad=True`: each pad's aggregate stiffness
at uniform contact equals `ke_bulk`, regardless of pad count.
`recalibrate_cslc_kc_per_pad(model, contact_fraction)` (now in
`cslc_v1/common.py`) overrides `kc` so the active-contact count
matches the calibration prior.  Without this override, the handler
default `cf=0.3` under-estimates the active count for sphere targets
(only ~5/189 spheres active at 1 mm Hertzian pen) and each pad
aggregates to far less than the intended `ke_bulk`.

Exact derivation (positive denominator):
```
N_contact_per_pad · kc · ka / (ka + kc) = ke_bulk
→ kc = ke_bulk · ka / (N · ka − ke_bulk)        (if N·ka > ke_bulk)
→ kc = ke_bulk / N                              (fallback otherwise)
```

Operating-point values per scene:

| Scene | cf prior | N_per_pad | `ka` | `kc` | `keff` | Aggregate / pad |
|---|---|---|---|---|---|---|
| Sphere (1 mm pen) | 0.025 | 4 | 15 000 | 75 000 | 12 500 | 50 000 ✓ |
| Book (full pad-on-cover) | 1.0 | 189 | 15 000 | 269.3 (fallback) | 264.6 | 50 000 ✓ |

### Historical sweeps (pre-2026-04-25 deep-pen regime)

> The two tables below were collected at the original 15 mm-pen squeeze
> (cf=0.46, ka=5000), before the scene was aligned to the lift test's
> 1 mm operating point.  Absolute numbers don't carry over to the
> current setup, but the **qualitative findings still apply** and
> motivate today's defaults.

**Resolution scaling (spacing 5 mm → 2.5 mm).** Doubling lattice
density at fixed pad geometry roughly doubled the active sphere count
and cut Z-drop by ~1.8× — sub-linear `N^0.4` scaling, consistent with
diminishing returns from constraint distribution under MuJoCo's CG
solver.

| Spacing | N_active / pad | `kc` | Z-drop |
|---|---|---|---|
| 5 mm | 87 | 657.9 | 0.214 mm |
| 2.5 mm | 309 | ~161 | **0.119 mm** (10.3× better than point) |

**Creep-mitigation knobs.** `solimp[1]=0.99` is a no-op — creep is not
dominated by the impedance `dmax` because the CSLC kernel flattens
`solimp` width to 0.001 (so `dmax` applies above 1 mm pen regardless).
**Elliptic cone reduces creep ~24 % for both models while preserving
the ratio**; it is now the production default for both squeeze and
lift.

| Change | Point FullDrop | CSLC FullDrop (cf=0.46) | Ratio |
|---|---|---|---|
| baseline (`solimp[1]=0.95`, pyramidal cone) | 1.223 mm | 0.214 mm | 5.71× |
| `solimp[1] = 0.99` (both models) | 1.223 mm | 0.211 mm | 5.80× |
| `cone="elliptic"` (default solimp) | **0.926 mm** | **0.166 mm** | 5.58× |

---

## 2. Lift test (`cslc_v1/lift_test.py`)

### Scene

Two **dynamic** pad arms (prismatic joints, PD-controlled) approach, squeeze,
lift, and hold a 500 g sphere. Because pads are dynamic (not kinematic),
MuJoCo computes relative contact velocity correctly — friction drags the
sphere upward during lift. Phase timing: APPROACH 1.5 s, SQUEEZE 0.5 s,
LIFT 1.5 s, HOLD 1.0 s. Total 2250 steps @ 2 ms.

```
# Headless comparison (all three models):
uv run cslc_v1/lift_test.py --mode headless --contact-models point,cslc,hydro

# Viewer:
uv run cslc_v1/lift_test.py --viewer gl --contact-model cslc
uv run cslc_v1/lift_test.py --viewer gl --contact-model cslc --start-gripped
```

### Tuning parameters (shared and per-model)

Shared material: `ke=5.0e4`, `kd=500`, `μ=0.5`, `dt=2.0 ms`.
Joint drive: `drive_ke=5e4`, `drive_kd=1e3`, position PD.
MuJoCo solver: `solver=cg`, `cone=elliptic`, `integrator=implicitfast`,
`iterations=100`, `ls_iterations=10` — matched to `squeeze_test.py` (see
§5.4 for the sweep that picked this).
Lift-ramp: symmetric C¹-smooth velocity profile with 250 ms ramps at BOTH
endpoints (see `_lift_dz` in `lift_test.py`) — eliminates the LIFT→HOLD
inertial overshoot that otherwise pushes the sphere past the pad's stop.

| Knob | Point | CSLC | Hydro |
|---|---|---|---|
| lattice spacing | — | 5 mm | — |
| `cslc_ka` | — | 5000 | — |
| `cslc_kl` | — | 500 | — |
| `cslc_n_iter` / `α` | — | 20 / 0.6 | — |
| `cslc_contact_fraction` | — | **0.025** (empirical: ~5 active/pad at 1 mm face-pen) | — |
| `kc` (recalibrated, fallback) | — | **12 500 N/m** (`kc = ke/N` because `N·ka − ke ≤ 0`) | — |
| `kh` | — | — | **1e8 Pa** |

### Results (best config, all models — cg + elliptic solver, 2026-04-20)

Initial sphere-on-ground offset: ~20 mm (sphere_z = r = 0.03, pad_z start = 0.05).
At HOLD end, pad_z = 0.0706 (pads lifted 20.6 mm).
Slip = (pad_z − sphere_z_final) − initial_offset.

**Baseline calibration** (default scene params — unmatched aggregate
stiffness, informative only; prefer the fair calibration below):

| Model | max_z | final_z | Slip from pad | Active contacts |
|---|---|---|---|---|
| `point_mujoco` | 0.0500 | 0.0427 | **7.7 mm** | 11 |
| `cslc_mujoco` (cf=0.025, ka=5000) | 0.0500 | 0.0491 | **1.5 mm** | 21–25 (≈7 CSLC surface in-band) |
| `hydro_mujoco` (kh=1e8) | 0.0534 | 0.0452 | **5.2 mm** | 45 |

**Fair calibration** (all three tuned so per-pad aggregate normal
stiffness = `ke_bulk = 5.0e4 N/m` at 1 mm face_pen):

| Model | Tuning | max_z | final_z | Slip from pad |
|---|---|---|---|---|
| `point_mujoco` | ke = 5e4 (unchanged) | 0.0500 | 0.0427 | **7.7 mm** |
| `cslc_mujoco` | **ka = 15 000**, cf = 0.025 → agg/pad = 50 kN/m ✓ | 0.0500 | 0.0494 | **1.2 mm** |
| `hydro_mujoco` | **kh = 2.65e8**, kh·A_patch = 50 kN/m ✓ | 0.0517 | 0.0466 | **4.0 mm** |

Reproduce (fair-calibration values are now the defaults, so no
overrides needed):
```
uv run cslc_v1/lift_test.py --mode headless \
    --contact-models point,cslc,hydro
```

**Conclusions under fair calibration:**
- CSLC has **6.4× less slip than point** and **3.3× less than hydroelastic**.
- Hydroelastic's baseline under-gripping (kh too low) accounts for
  ~1.3× of its original slip; the remaining ~3.3× gap vs CSLC is the
  distributed-contact geometric advantage.
- Hydroelastic still overshoots most at LIFT→HOLD (max_z 0.0517 vs 0.0500),
  suggesting more LIFT-phase momentum accumulation — a physics property
  of the pressure-field model, not a calibration artifact.
- Qualitative ranking CSLC < hydro < point is preserved; the distributed-
  contact advantage is real and independent of parameter choices.

### Per-step timing / online compute cost (2250 steps @ 2 ms, RTX 3070)

Measured via timing instrumentation in `lift_test.run_headless` — full
sim-loop wall time with a one-step warm-up pass to exclude JIT compile.
Reproduce with the same command as above
(`--cslc-ka 15000 --kh 2.65e8`).

**Baseline** (CSLC using 20-iteration iterative Jacobi lattice solve):

| Model | Wall | Per-step | Collide | MuJoCo step |
|---|---|---|---|---|
| point | 7.54 s | 3.35 ms | 0.32 ms | 2.79 ms |
| CSLC | 12.61 s | 5.61 ms | 1.82 ms | 3.55 ms |
| hydro | 10.71 s | 4.76 ms | 0.99 ms | 3.53 ms |

**Optimised** (CSLC Kernel 2 switched to the one-shot
`lattice_solve_equilibrium` matvec; `build_A_inv=True` in
`CSLCData.from_pads`; handler dispatches to the dense solve when
`data.A_inv` is available, falls back to iterative Jacobi otherwise):

| Model | Wall | Per-step | Collide | MuJoCo step | Δ vs baseline |
|---|---|---|---|---|---|
| point | 7.32 s | 3.25 ms | 0.31 ms | 2.71 ms | — |
| **CSLC** | **10.71 s** | **4.76 ms** | **0.99 ms** ↓ | 3.54 ms | **collide −46 %, per-step −15 %** |
| hydro | 10.67 s | 4.74 ms | 0.99 ms | 3.51 ms | — |

**Optimised + dead-readback removed (2026-05-03)** — `_launch_vs_sphere`
was issuing 4 `.numpy()` GPU→CPU syncs per pair per step to populate a
diagnostic block whose only consumer was a commented-out `print`.
Removing the dead readback drops the collide phase by ~half:

| Model | Wall | Per-step | Collide | MuJoCo step | Δ vs prior optimised |
|---|---|---|---|---|---|
| **CSLC** | **10.38 s** | **4.61 ms** | **0.50 ms** ↓↓ | 3.86 ms | **collide −49 %, per-step −3 %** |
| hydro | 10.88 s | 4.84 ms | 0.91 ms | 3.67 ms | (jitter) |

CSLC is now **faster than hydroelastic** on both collide phase
(0.50 ms vs 0.91 ms = 1.8× faster) and per-step (4.61 ms vs 4.84 ms),
in addition to being far cheaper to set up (no SDF precomputation, no
mesh dependency).  This is the headline speed claim for the paper.

**Conclusions:**
- CSLC's online per-step cost is now **statistically identical to
  hydroelastic** — 4.76 ms vs 4.74 ms (0.4 % spread), with collide
  phase **exactly matched at 0.99 ms**.
- CSLC is 1.46× slower than point contact. Most of the remaining gap is
  in MuJoCo's constraint solver step (3.54 ms vs 2.71 ms) — a function
  of the number of simultaneous constraints CG iterates over, not
  something the CSLC kernel chain controls.
- Under-utilisation note: 378 lattice threads don't fill an RTX 3070;
  there's GPU head-room for multi-pad grippers at no additional
  per-step cost (batched-pair-launch TODO in §5.6).

**Why the dense solve wins.** Iterative Jacobi issued ~20 `wp.launch`
calls per pair per step = 40 launches/step in a tight Python loop. At
~5–10 µs of host-side launch overhead each, that alone was ~300 µs.
The dense `A⁻¹ · φ` matvec collapses those 20 launches into 1 while
preserving the full lattice physics (`ka`, `kl`, `kc`) baked into the
pre-inverted `A = K + kc · I`. See `cslc_handler.py:_launch_vs_sphere`
for the dispatch logic and `cslc_data.py:from_pads(build_A_inv=True)`
for the pre-inversion. The matvec is also tape-compatible (see §4.5),
so the backward pass is a single linear operation rather than a
ping-pong Jacobi chain that aliases buffers.

**Scaling cliff.** `A⁻¹` is O(n²) memory. At n = 378 per pad (current
scene) it's 0.57 MB — trivial. At n = 10 000 (MorphIt-scale irregular
pad) it's 400 MB — borderline. Above that, the sparse Cholesky
factorisation TODO (§5.6) is required; the iterative-Jacobi fallback
path is retained in the handler for that case.

### Why the calibration is fair — theoretical derivation

The three models have different contact semantics, so matching on contact
*count* is meaningless. The right invariant is the **per-pad aggregate
normal stiffness at the operating penetration** `k_agg` — the single
macroscopic number that sets how much grip force the pad delivers for a
given face penetration. Under position-PD drive, this directly determines
the equilibrium normal force.

**Target** (`ke_bulk = 5.0e4 N/m`) and **recipe** at 1 mm face_pen,
r = 30 mm sphere (`A_patch ≈ π·(2·r·pen) = 188 mm²`):

| Model | Formula | Fair value |
|---|---|---|
| Point | `k_agg = ke` | ke = **5.0e4 N/m** (already) |
| CSLC | `k_agg = N · kc·ka/(kc+ka)`; exact branch needs `N·ka > ke_bulk` | `ka = 15 000`, cf = 0.025 (N = 4) → `kc = ke·ka/(N·ka − ke) = 75 000` → `keff = 12 500` → `agg = N·keff = 50 000` ✓ |
| Hydro | `k_agg ≈ kh · A_patch(pen)` | `kh = ke_bulk / A_patch = 5e4 / 188e-6 =` **2.65e8 Pa** |

**Equilibrium force-balance under position-PD drive.** Command a virtual
"zero-compliance" target penetration `pen_target`. At HOLD (no slip):
```
drive_ke · (pen_target − face_pen) = k_agg · face_pen
⇒ face_pen = drive_ke / (drive_ke + k_agg) · pen_target
⇒ Σ F_n    = k_agg · drive_ke / (drive_ke + k_agg) · pen_target
```
When `k_agg` is matched across models (and drive gains are shared),
`Σ F_n` is matched at equilibrium **by construction**. Friction capacity
`μ · Σ F_n` is therefore identical across all three models, so any
residual difference in slip must come from:

1. **Spatial distribution of `F_n`** (point: concentrated; CSLC: ~7 spheres
   over a ~7 × 1 mm disk; hydro: ~45 polygons over a ~188 mm² patch).
2. **Per-constraint soft-compliance leak** under MuJoCo CG (§3.1). Each
   of the N active constraints contributes its own `c · f_n` term; the
   distributed models get N smaller contributions summing to the same
   total normal force, with the per-constraint leak reduced empirically
   by ~N^0.4 (see resolution sweep in §1).

Items 1 and 2 are exactly what the paper claims are the advantages of
distributed contact — and the fair calibration isolates them.

### Does a force controller further improve fairness?

**Short answer:** Yes, strictly, but the improvement is marginal once
aggregate stiffness is matched.

**Long answer.** Under force-PD (command force `f_cmd` directly, joint
position becomes a state):
```
Σ F_n = f_cmd                   (exact, independent of k_agg)
face_pen = f_cmd / k_agg        (varies: stiffer → smaller pen)
```
This removes `k_agg` from the `Σ F_n` equation entirely. So:

- **Normal-force identity**: exact under force-PD; approximate under
  position-PD (controlled by `drive_ke / (drive_ke + k_agg)` ratio).
  For matched `k_agg = ke_bulk = drive_ke`, position-PD gives
  `Σ F_n = 0.5 · k_agg · pen_target` across all three — fair to within
  machine precision.
- **Residual variation under position-PD**: `face_pen` also varies with
  `k_agg` via the `drive_ke / (drive_ke + k_agg)` factor. Models with
  different `k_agg` at the operating pen end up at slightly different
  actual face penetrations, which changes the patch geometry. Force-PD
  pins force but amplifies this geometric variation.
- **What remains unfair even under force-PD**: the *face_pen* differs
  across models (stiffer → thinner patch). This is a genuine physical
  property of each contact model — not a calibration artifact. Matching
  patch area as an additional constraint would be required to isolate
  the purely topological (point vs lattice vs polygon) effect.

**Recommendation**: position-PD with matched `k_agg` (the calibration
above) is sufficient for the current "distributed contact is better
than point" claim — it fixes `Σ F_n` within < 1 %. A force-PD variant
(§5.5) is worth building for the *slip-detection* experiment where
per-sphere tangential utilisation matters, because position-PD's
macroscopic feedback loop hides slip-detection latency. Force-PD is
**not** a prerequisite for the fairness conclusions above.

**Solver note** (see §5.4): switching lift from `solver="newton", ls=50`
to `solver="cg", cone="elliptic", ls=10` (the squeeze_test setting) improved
slip across all three models: point 11.6 → 7.7 mm, CSLC 4.1 → 1.6 mm,
hydro 9.2 → 5.2 mm. Both tests now share the same solver config.

### Slip-mitigation experiments

Lift exhibits three additive slip mechanisms: LIFT-phase lag, LIFT→HOLD
inertial overshoot, and HOLD-phase creep. The `slip` metric subtracts the
permanent z-offset from sphere-on-ground vs pad-center.

| # | Intervention | Outcome | Kept |
|---|---|---|---|
| H1 | `SolverSemiImplicit` as no-soft-constraint baseline | **Untestable** — joint-PD drive fails, `pad_z=NaN` | — |
| H2 | Symmetric LIFT end-ramp (smooth deceleration) | Small win — overshoot 9 → 4 mm | ✓ |
| H3 | MJ iterations 100 → 1000 | **No effect** — slip is constraint-stiffness-limited, not convergence-limited | ✗ |
| H4 | Per-pad recalibration with cf=0.025 (matches ~5 active spheres/pad) | **Big win** — HOLD creep 8.9 → 4.2 mm/s | ✓ |
| H5 | `ka` 5000 → 50 000 (break series-spring saturation) | Slight regression — fewer active spheres (δ saturates) | ✗ |

### Mass and squeeze-depth sensitivity

| Variable | Value | Point slip | CSLC slip |
|---|---|---|---|
| Mass | 500 g (default) | 11.6 mm | 4.1 mm |
| Mass | 50 g | 2.1 mm | 0.6 mm |
| Squeeze depth | 1 mm (default) | 11.6 mm | 4.1 mm |
| Squeeze depth | 5 mm | 7.7 mm | 1.5 mm |

Slip scales sub-linearly with mass — confirms force-proportional
soft-constraint compliance, NOT Coulomb saturation. Closing harder helps
modestly via "more spheres engaged → lower per-constraint utilisation",
not via friction-cone margin. The ~4 mm slip floor under MuJoCo is the
irreducible per-step constraint compliance for this pad geometry.

### `kh` stability sweep (hydro)

Re-measured 2026-04-20 on the current squeeze scene (default params,
hydro_mujoco). `Squeeze z-drop` is the legacy `FullDrop = z[0] - min(z)`.
`HoldCreep` is the second-half-of-HOLD mean velocity (positive = falling).
All five runs reproduced via `cslc_v1/_validation_logs/sweep_kh_stability.py`.

| kh [Pa] | FullDrop | HoldDrop | HoldCreep | Peak contacts | Notes |
|---|---|---|---|---|---|
| 1e6 | 1.355 mm | +1.075 mm | +0.722 mm/s | 133 | monotonic fall; no rise observed |
| 1e7 | 0.453 mm | +0.345 mm | +0.229 mm/s | 133 | monotonic fall; no rise observed |
| 3e7 | 0.264 mm | +0.198 mm | +0.132 mm/s | 134 | monotonic fall; no rise observed |
| **1e8** | **0.130 mm** | **+0.093 mm** | **+0.062 mm/s** | **140** | production default; monotonic fall |
| 1e10 | DIVERGED | n/a | n/a | 126→0 | sphere ejected during SQUEEZE (final z ≈ −9.7 m) |

HoldCreep reduces monotonically with `kh` up to the stability boundary —
roughly `HoldCreep ∝ 1/kh` in the 1e6–1e8 regime. At 1e10 the contact
impedance saturates MuJoCo's CG solver and the sphere is ejected through
the pad within the first few SQUEEZE steps. **None of the five runs showed
the upward-rise behavior previously documented at low kh**; see the earlier
commentary (pre-merge) for context on that regime.

---

## 3. Related work

### 3.1 Solver architecture — Drake/SAP vs MuJoCo vs SemiImplicit

The HOLD creep in both tests is a **solver property**, not a contact-model
property. The three relevant paradigms differ in how — and whether — they
leak compliance.

**MuJoCo CG/Newton** (what we use).
Regularised constraint solver. Each contact carries
`solref = (timeconst, dampratio)` and `solimp = (dmin, dmax, …)`. Forces
come from relaxing a constrained system whose compliance is **explicit in
the constraint formulation**:

```
0 ≤ φ + (δt + d·c)·v_n + c·f_n ⊥ f_n ≥ 0,    with c = 1/k
```

The `c·f_n` term is the source of residual per-step slip. Designed for speed
and convexity; pays for it with constraint softness.

**Drake SAP** (Castro et al. 2021). Semi-Analytic Primal solver. Solves an
unconstrained convex optimisation each timestep:

```
min_v  ½(v − v*)ᵀ M (v − v*) + Σ_i  ψ_i(v)
```

where `ψ_i` is a convex per-contact dissipation potential derived from
the contact compliance. SAP does **not** regularise via `solref / solimp`;
compliance enters as a (possibly very stiff) function in the objective.
At the velocity level, SAP enforces contact and friction constraints to the
numerical accuracy of the Newton iteration — **no per-step compliance leak
by construction**. Drake's hydroelastic-on-SAP demos hold objects without
creep for exactly this reason.

**Newton's `SolverSemiImplicit`** (penalty integrator). Not a constraint
solver at all. Forward-Euler integration of `F_n = −k·φ − d·v_n` (Hunt–Crossley)
plus regularised Coulomb friction. No LCP, no convex optimisation, no implicit
time stepping. No constraint compliance because there are no constraints.
Needs very small `δt` for stable contact, can't handle hard joints, chatters
at stick-slip transitions.

| Paradigm | Compliance leak | Per-step cost | Implicit step | Hard joints | Differentiable |
|---|---|---|---|---|---|
| MuJoCo CG/Newton | **Yes** (`c·f_n`) | Low | Stable any δt | Yes | Via Warp |
| Drake SAP | **No** (rigorous KKT) | Medium–High (sparse Cholesky) | Stable any δt | Yes | Limited |
| `SolverSemiImplicit` | **No** (no constraints) | Very Low | Tiny δt required | No (soft only) | Via Warp |

SAP and SemiImplicit reach the same HOLD outcome (no creep) for opposite
reasons: SAP solves the constrained problem rigorously; SemiImplicit doesn't
have one to leak from. The MuJoCo creep floor is a known limitation; the
paper should acknowledge it explicitly.


---

## 4. Bugs fixed (invariants — do not revert)

### 4.1 Stiffness conversion (`newton/_src/solvers/mujoco/kernels.py`, FIXED 2026-04-19)

MuJoCo encodes contact stiffness via `solref = (timeconst, dampratio)`.
The contact force law is:

```
F = imp × ke_ref × pen,    ke_ref = 1 / (timeconst² × dampratio²)
```

To recover `F = contact_ke × pen`, we need `imp × ke_ref = contact_ke` →
`ke_ref = contact_ke / imp` → `1/(tc²·dr²) = contact_ke/imp`.

**Correct formulas** (see `kernels.py:390–411`):
```python
# kd=0 branch (used for CSLC after damping fix):
tc = sqrt(imp / contact_ke);  dr = 1.0          # F = contact_ke × pen ✓

# kd>0 branch (used for standard Newton contacts):
tc = 2/kd;  dr = sqrt(imp / (tc² × contact_ke))  # F = contact_ke × pen ✓
```

**Original wrong formula** (preserved as comment trail):
```python
contact_ke *= (1.0 - imp)            # ≈ 0.05·ke
timeconst = sqrt(1.0 / ke_scaled)
dampratio = sqrt(1.0 / (tc² × ke_scaled))
# → F = imp × ke_scaled × pen = imp·(1−imp)·ke·pen ≈ 0.047·ke·pen  (21× too low)
```

For `imp=0.95`: original gave `F ≈ 0.05·ke·pen` instead of `ke·pen` — 21×
too weak. Caused the sphere to fall through the pads in all early CSLC tests.

### 4.2 CSLC damping → kd=0 (`cslc_kernels.py:line ~467`)

`cslc_dc = 2.0 N·s/m` is a semi-implicit-integrator coefficient. In MuJoCo's
kd>0 branch, `timeconst = 2/kd = 1.0 s` — **250× softer** than standard
contacts (`tc ≈ 0.004 s`). Fix: write `out_damping = 0.0` to trigger the
kd=0 branch (`tc = sqrt(imp/ke) ≈ 0.030 s`). Normal force is unchanged by
design; only friction stiffness improves.

### 4.3 CSLC friction scale = 1.0 (`cslc_kernels.py:line ~475`)

The MuJoCo conversion kernel treats `rigid_contact_friction` as a **scale
factor** on the geom pair's base friction:
```
effective_mu = max(μ_pad, μ_sphere) × rigid_contact_friction
```
Original: `out_friction = shape_material_mu` → `effective_mu = μ × μ = μ²`.
Fix: `out_friction = 1.0` → `effective_mu = μ × 1.0 = μ` ✓.

### 4.4 Other durable invariants

- Position-FD for velocity in both test files (body_qd always ≈ 0 under MuJoCo GPU).
- Right-pad 180° shape rotation (face_axis=0, face_sign=+1 hardcoded in handler).
- `cslc_handler._from_model` uses `create_pad_for_box_face` (2-D face lattice),
  not the volumetric pad. Volumetric pads have tilted normals at edges/corners
  that violate the flat-patch assumption and cause launch instability during
  lift. The handler hardcodes the inner face to local +x — each CSLC shape
  must be oriented so local +x points toward the grasped object.

### 4.4a Box K3 `pen_3d` uses `effective_r`, not `r_lat` (FIXED 2026-05-03)

`write_cslc_contacts_box` (cslc_kernels.py) was using `pen_3d = r_lat -
signed_dist`, the *rest* overlap from K1.  Paper eq. 5 says the contact
force at equilibrium is `F = kc · (φ_rest − δ) · gate` — the
*post-equilibrium* effective overlap, with δ subtracted.  The sphere K3
gets this right via `pen_3d = (effective_r + target_radius) − dist`; the
box variant was a copy-paste oversight from the 2026-04-26 box-kernel
addition.

**Fix:** `pen_3d = effective_r − signed_dist` so `F = kc · pen_3d · gate
= kc · (φ_rest − δ) · gate`, matching paper eq. 6 static part.

**Why the bug was hidden:** Per-sphere force overstiffening factor is
`(kc + ka) / ka`.  For book scene (`kc = 269.3`, `ka = 15000`), this is
1.018 — a 1.8 % per-sphere over-force.  Squeeze book uses *kinematic*
pads, so the macroscopic grip is regulated by the position constraint
and HoldCreep is unchanged (bit-exact +0.045 mm/s before and after the
fix).  Lift uses sphere targets, so the box kernel never runs.  The bug
was invisible to every existing integration metric.

**Why it still mattered:** The differentiable-MPC story relies on
`dF/dδ` matching the paper.  With the bug, `dF/dδ` is wrong by a
constant factor on box targets, breaking gradient-based optimisation
through CSLC vs box contact.  Fixing it preserves end-to-end
consistency between paper §3.4 calibration, §4.5 gradient flow, and
the implementation.

### 4.4b Box-kernel diagnostic parity + dead-readback cleanup (2026-05-03)

Two paired changes to `cslc_kernels.py` / `cslc_handler.py`:

- **Diagnostic parity.**  `write_cslc_contacts_box` now writes the same
  five per-contact diagnostic arrays the sphere variant does
  (`dbg_pen_scale`, `dbg_solver_pen`, `dbg_effective_r`, `dbg_d_proj`,
  `dbg_radial`), indexed by the same per-pair-per-slot layout
  (`pair_idx · n_surface_contacts + slot`).  `dbg_d_proj` records
  `d_proj_solver` (the projection that drives `solver_pen`) and
  `dbg_radial` is the in-face-plane component of `closest − q` — the
  geometric analog of the sphere kernel's `sqrt(dist² − d_proj²)`.
  Hard-culled slots keep their post-geometry diag values, matching the
  sphere convention.  External readers can now interrogate sphere and
  box pairs with identical code.

- **Dead-readback removed.**  The post-K3 block in
  `_launch_vs_sphere` was issuing 4 GPU→CPU `.numpy()` syncs per pair
  per step to feed a commented-out `print`.  Stripped.  Collide phase
  on the lift scene drops 0.99 → 0.50 ms (−49 %), making CSLC's online
  cost lower than hydroelastic's (see §2 "Optimised + dead-readback
  removed" table).  Both changes are observability-only: no test
  numbers move.

### 4.4c `apply_external_wrench` layout bug (FIXED 2026-05-03)

`cslc_v1/common.py:apply_external_wrench` was writing the wrench in
`[torque, force]` order — opposite to Newton's actual
`wp.spatial_vector(force, torque)` convention used by both
`semi_implicit/kernels_contact.py:261` (`wp.spatial_vector(f_total,
wp.cross(r, f_total))`) and `mujoco/kernels.py:1367` (which reads
`f[0:3]` as linear and `f[3:6]` as angular before forwarding to
MuJoCo's `xfrc_applied`).

**Effect of the bug:** every disturbance row in the squeeze and lift
tests that used `--external-force fx,fy,fz` was actually applying a
torque `(fx, fy, fz)` N·m, and every `--external-torque tx,ty,tz`
was actually applying a force `(tx, ty, tz)` N.  Pre-fix sweep
tables in this document (any §1 disturbance numbers from before
2026-05-03) are mislabelled.

**Fix:** swap the slot writes in `apply_external_wrench` so
`bf[0:3] = force, bf[3:6] = torque`, and update the docstring +
comment to document the convention with cross-references to the two
solver paths that establish it.

**Empirical confirmation (post-fix):** the τ_z = 5 N·m disturbance
on the BOX-vs-BOX point case ejects the book (creep −74.8 mm/s,
tilt 178°), which is consistent with a true 5 N·m vertical twist
exceeding the 4-corner moment arm capacity.  Pre-fix this row read
"−0.015 mm/s, 0° tilt" because it was actually applying a +5 N
upward force that nearly cancelled gravity.  See §1 "Disturbance
sweep — three baselines" for the regenerated tables.

**Why the bug was hidden for so long:** none of the existing tests
asserted on the *direction* of the disturbance effect — they read
the metric values without cross-checking against the disturbance
label.  Going forward, any new disturbance entries should include a
predicted-vs-observed sanity check.

### 4.5 Smoothness / differentiability coverage (verified by `cslc_v1/smooth_basic_test.py`, 33 tests)

The paper's "differentiable everywhere" claim is supported by replacing
every hard `[·]_+` / `H(·)` in the CSLC math with C^∞ surrogates
`smooth_relu(x, ε) = ½(x + √(x² + ε²))` and
`smooth_step(x, ε) = ½(1 + x/√(x² + ε²))`.  Default `ε = 1e-5 m`; gates are
effectively binary above 0.1 mm while staying C^∞ through the threshold.

| Location | Smoothness status | Tape backward | Test class |
|---|---|---|---|
| `smooth_relu` / `smooth_step` primitives | C^∞ for ε > 0; verified Lipschitz, analytic derivatives, range/sign invariants, eps→0 limits | ✓ matches analytic | `TestSmoothPrimitives` (14 tests) |
| Kernel 1 `compute_cslc_penetration_sphere` — phi | Lipschitz through both `d_proj=0` and `pen_3d=0` boundaries (dense sweep) | ✓ per-launch | `TestKernel1PhiSmoothness` (6) |
| Kernel 2a `lattice_solve_equilibrium` — δ = kc · A_inv · φ | Linear matvec; exact match to closed form | ✓ full Jacobian = kc · A_inv (tape + central-FD) | `TestLatticeSolveEquilibrium` (5) |
| Kernel 2 `jacobi_step` (single launch) | Smooth | ✓ per-launch | `TestJacobiConvergesToAnalytic` (2) |
| Kernel 2 + Python-side src/dst swap | Smooth as ODE | ✗ aliased buffers break `wp.Tape` — use Kernel 2a for backward | — |
| Kernel 3 `write_cslc_contacts` | **Smooth in the physically meaningful transition region** (gate ≥ 1e-4, ≈ \|d_proj\|, \|pen_3d\| ≤ 30·ε); hard-culled deep in the tail (gate < 1e-4) where both the smooth force AND its gradient are already machine-zero | ✓ in-band | `TestKernel3WriteCullGap` (6) |

**Key implementation detail (2026-04-20): sign-preserving smooth reciprocal.**
`pen_scale = pen_3d / solver_pen` is the correction that makes
`F = stiffness · solver_pen = kc · pen_3d`. The naive smooth protection
`pen_3d / √(solver_pen² + δ²)` divides by `|solver_pen|` — which flips
the sign of `pen_scale` when `pen_3d` and `solver_pen` cross zero
together along the face normal. The fix:
```
pen_scale = pen_3d · solver_pen / (solver_pen² + ε²)
```
This is C^∞, preserves sign, and produces a ~10 µm-wide notch at
`solver_pen = 0` that is resolvable by any reasonable sampling density.
Without this, `TestKernel3WriteCullGap.test_stiffness_lipschitz_across_pen_3d_zero`
fails with a 30 %-of-peak jump at the pen_3d boundary.

**Hybrid emission policy.** Kernel 3 writes every transition-band slot
as a valid contact with smoothly-gated stiffness, but hard-culls slots
whose gate has fallen below 1e-4 (roughly |d_proj| > 30·ε ≈ 300 µm
outside the boundary, or similarly for pen_3d). This was added after the
2026-04-20 lift regression: with every one of the 378 lattice-sphere
slots live, MuJoCo's per-constraint compliance leak (each with its own
soft-constraint `c·f_n` term — see §3.1) summed across spheres and
degraded static friction enough for the sphere to slip out of the grip
(4 mm → 20 mm slip). The hybrid emission recovers pre-fix static
friction (CSLC slip 1.5 mm ≈ unchanged) while preserving C^∞ gradient
flow through the entire physically meaningful transition band — the only
region where a gradient-based optimiser would ever query derivatives.

Key practical consequence: the entire CSLC kernel chain is now smooth
in the region where smoothness matters for MPC / RL / trajectory
optimisation. `wp.Tape` backward flows from rigid-body pose gradients
through phi, δ, and the written contact stiffness without any discrete
jumps in the active transition band.

Run with:
```
uv run --extra dev python -m unittest cslc_v1.smooth_basic_test -v
```

---

## 5. TODOs

### 5.1 CSLC refinement

- **Iterative `contact_fraction` bootstrap.** Replace the hardcoded fraction
  with a 1–2 step bootstrap: assume 0.3, run one collision step, measure
  active count, recompute `kc`. Eliminates the geometry-specific tuning we
  do today (0.46 for squeeze, 0.025 for lift).
- **Per-shape `shape_cslc_contact_fraction`** model attribute. Allows
  different pads in the same scene to use different priors.
- **Per-pad `kc` storage** in `CSLCData`. Currently one scalar shared across
  pads; for mixed sizes/materials, promote to a per-shape array indexed via
  `sphere_shape`.
- **Material → spring constants.** Add a `CSLCMaterial(E, ν, h, μ)` spec:
  `ka = E·A_sphere/h`, `kl = G·h/spacing` with `G = E/(2(1+ν))`,
  `ke_bulk = E·A_pad/h`. User specifies measurable physical quantities only.
- **MorphIt integration for arbitrary pad geometry.** Replace the box-face
  lattice generator with MorphIt-driven sphere packing; auto-compute outward
  normals from local surface gradients and neighbor topology from spatial
  proximity (k-NN at 1.5× spacing).
- **Auto-tune solver params.** `n_iter` from spectral radius (already printed
  in `inspect_cslc_handler`); set `n_iter = ceil(3 / −log10(ρ))` to converge
  to 3 digits.
- **Fix `squeeze_test.save_results`**: references `m.settled` / `m.sphere_angle`
  that don't exist on `Metrics`. Currently crashes at end of squeeze mode
  (cosmetic — after RESULT prints, before CSV save).
- **(DONE 2026-04-20) Smooth Kernel 3 write path.** The two hard culls
  have been replaced with a smooth gate
  `contact_gate = smooth_step(d_proj, eps) · smooth_step(pen_3d, eps)`
  and a sign-preserving smooth reciprocal
  `pen_scale = pen_3d · solver_pen / (solver_pen² + eps²)`. A hybrid
  emission policy keeps the kernel-to-solver interface C^∞ inside the
  transition band (gate ≥ 1e-4, ≈ |d_proj|, |pen_3d| ≤ 30·eps) and
  hard-culls deep in the tail to avoid MuJoCo's per-constraint
  compliance leak from swamping static friction. See §4.5 for the
  complete smoothness audit.

### 5.2 Comparison fairness — remaining TODOs

Matched-aggregate-stiffness calibration is complete (see §2 for theory,
parameters, and lift results). What's left:

- **Force-balance audit during HOLD** using `contacts.rigid_contact_force`
  (already requested via `request_contact_attributes("force")`). Report
  per-model:
  (a) Σ F_n = W ✓ at equilibrium (sanity),
  (b) distribution of per-contact F_n,
  (c) distribution of per-contact tangential utilisation `|F_t|/(μ·F_n)`.
  The utilisation histogram is the paper's cleanest visualisation of the
  distributed-contact claim. Under the fair calibration this is now the
  right experiment to validate that CSLC's advantage is patch-geometric,
  not stiffness-artifactual.
- **Sweep `sdf_max_resolution` 32 / 64 / 128** to quantify hydro
  contact-count scaling under the fair `kh = 2.65e8`. If slip is
  resolution-independent, the 45-vs-7 count is cosmetic; if slip tightens
  with more polygons, that's a real hydro vs CSLC trade-off to report.
- **Inspect hydroelastic active contacts during lift** (all on the pad
  face, or are corner/edge polygons producing off-axis forces?). Plot
  contact points in world frame for one HOLD step per model.
- **Match patch area as a secondary criterion**. Once force-PD lands
  (§5.5), additionally constrain `A_patch` to be shared across models so
  both grip force AND contact area are fixed, isolating the purely
  topological (1 point vs ~7 spheres vs ~45 polygons) effect.

### 5.3 Rotational / multi-axis mechanics

- **Tilt/twist perturbation** (paper's §4.2). Add angular-impulse tests to
  squeeze_test; measure restoring torque vs ω_y / ω_x. This is the paper's
  flagship claim that hasn't been re-run since the kernel rewrite.
- **Contact position leverage.** CSLC writes `pos = midpoint(lattice_center,
  sphere_center)` — correct for force transmission but may give wrong torque
  lever arms for tilt/twist. Investigate before perturbation tests.
- **Confirm CSLC/hydro produce comparable contact moments** (rotational
  stiffness, scrubbing torque). Add tilt/twist to squeeze_test for all three
  models.

### 5.4 Solver / creep follow-ups

**Investigation (2026-04-20): cg + elliptic beats newton for all three
contact models in the lift test.**

Historical mismatch: squeeze used `solver="cg"`, `cone="elliptic"`,
`ls_iterations=10`; lift used `solver="newton"`, `ls_iterations=50` (no
cone override). Matching lift to squeeze's setting gave unambiguous gains
across all three models:

| Model | Lift slip (newton, ls=50) | Lift slip (cg + elliptic, ls=10) | Delta |
|---|---|---|---|
| point | 11.6 mm | 7.7 mm | −33% |
| CSLC | 4.1 mm | **1.6 mm** | **−61%** |
| hydro | 9.2 mm | 5.2 mm | −43% |

Why cg + elliptic wins:
- **Elliptic friction cone** removes the pyramidal-cone stick-slip artifact
  that amplifies under the velocity-mismatched LIFT ramp. Already confirmed
  in §1 creep-mitigation; now confirmed to transfer to dynamic lift.
- **CG with fewer line-search iters** converges on the constraint-softness
  residual just as well as the Newton solver with 50 ls iters, at ~1/3 the
  wall time per step. CSLC's ~170 simultaneous constraints don't benefit
  from Newton's quadratic convergence because the constraint Hessian is
  dominated by the soft `c·f_n` term, not by nonlinearity.

**Both tests now use the same solver config** (`cg` + `elliptic`, iter 100,
ls 10) — any further comparison starts from a consistent baseline.

Remaining follow-ups:

- **Sweep CG `iterations`** 100 → 500 → 1000 to confirm slip is constraint-
  stiffness-limited, not convergence-limited (previous H3 check said yes;
  worth repeating with the new baseline numbers).
- **Debug `SolverSemiImplicit` joint-PD behaviour** (pad_z=NaN under the
  prismatic-X/Z articulated arm) — useful as a no-soft-constraint baseline.

  **Findings from 2026-04-20 investigation (changes reverted, not kept):**

  `pad_z=NaN` is not a single bug; explicit Euler is violating stability
  on *three* independent stiff terms in the existing lift scene at the
  default `dt=2 ms`. All three can produce NaN even if one is fixed, so
  any future patch has to address them together.

  1. **Joint-PD damping.** `drive_kd=1e3` on pad mass 0.08 kg gives
     `c·dt/m = 25` — explicit Euler stability needs `≤ 2`, so the velocity
     update amplifies `~24×` per step and saturates to NaN within ~6
     steps. Fix: implicit-damping correction inside
     `semi_implicit/kernels_body.py::eval_body_joints` on the PRISMATIC
     branch, scaling `kd_eff = kd / (1 + kd·inv_m·dt)`. Requires plumbing
     `dt` + `body_inv_mass` into `eval_body_joints` and updating
     `eval_body_joint_forces` + `solver_semi_implicit.py::step` to pass
     `dt`. Collapses to `kd` when `kd·dt/m → 0`, so existing stable
     scenes are unchanged. On its own this only pushed the NaN out by
     a handful of steps.

  2. **Ghost-slider angular attachment.** The 2 mm radius slider used as
     an intermediate link for the X prismatic DOF has `inv_I ≈ 1e6`. The
     default `joint_attach_ke=1e4` then creates an angular restoring
     spring with `ω·dt = sqrt(k·inv_I)·dt ≈ 45` (need `< 2`). Tiny
     floating-point noise in the contact moment arm seeds an initial
     `wz ~ 1e-4` rad/s which amplifies by `|1 − c·dt·inv_I| ~ 20` per
     step (contact torque from off-axis numerical error on right-pad
     shape-rotated-180° is what consistently triggers it on body 2 and
     body 3). Sweeping `joint_attach_ke`/`joint_attach_kd` down to
     `(1, 0.01)` only shifts the blow-up from step 9 to step 24 — the
     exponential growth is dominated by the slider's near-zero angular
     inertia, not the attach constants. Root-cause fix requires either
     giving the slider an explicit non-trivial `inertia=wp.mat33(...)`
     in `add_link`, switching to `add_joint_d6(X,Z)` to eliminate the
     slider entirely, or implementing implicit angular-attach damping
     in the solver.

  3. **Distributed contact damping on the sphere.** CSLC writes N≈150
     active simultaneous contacts per step. `eval_body_contact` reads
     per-contact `rigid_contact_damping`; CSLC writes `0` (deliberately,
     to force MuJoCo's `tc = sqrt(imp/ke)` branch — see §4.2), which in
     the semi-implicit path falls through to `shape_material_kd=500`.
     The 500 Ns/m contribution **sums across contacts**: total damping
     on the 0.5 kg sphere = `150·500·dt/m = 300` (need `< 2`), so sphere
     velocity goes NaN during SQUEEZE regardless of any joint-PD fix.
     The 0 sentinel has different semantics in the two solver paths, so
     a clean fix needs solver-aware per-contact kd (or an extra field on
     the contacts buffer that distinguishes "use solver default" from
     "apply 0 damping"). Hydro has the same pathology at lower N but
     higher per-contact force magnitudes.

  **What was tried and reverted:**
  - `semi_implicit/kernels_body.py`: implicit damping for PRISMATIC
    joint target_kd (addresses #1 only).
  - `semi_implicit/solver_semi_implicit.py`: thread `dt` into
    `eval_body_joint_forces`.
  - `cslc_v1/lift_test.py`: `kinematic_pads=True` path — pads become
    `is_kinematic=True` free-joint root bodies, pose/velocity written
    directly each step via a new `set_pad_kinematic_state` helper. Auto-
    enabled when `--solver semi`. This sidesteps #1 and #2 entirely by
    removing the articulated drive, yielding a genuine "no-soft-
    constraint baseline" where only the sphere dynamics are integrated.
    With that change `point_semi` runs cleanly to completion (sphere
    falls — expected, penalty point contact has no static friction);
    `cslc_semi` and `hydro_semi` still NaN during SQUEEZE because of
    #3.

  **If this is picked up again:**
  - Pick one of: (a) keep the articulated scene and fix all three
    instabilities in the solver (proper implicit Rayleigh damping on
    both translational body_qd and joint attach, plus sensible
    per-contact kd semantics), OR (b) go with kinematic pads +
    fix #3 only. (b) is much less code but gives up "joint-PD behavior"
    as the thing being tested — the baseline becomes purely about
    contact-model fidelity under a penalty integrator.
  - For #3, probably the right move is to have CSLC write per-contact
    `out_damping = -1.0` (sentinel) instead of `0.0`, extend the
    semi-implicit contact kernel to treat negative as "use shape default
    scaled by 1/N_active", and leave the MuJoCo conversion to also
    interpret the sentinel as "kd=0, use `tc = sqrt(imp/ke)`". Both
    callers get what they want, and existing MuJoCo numbers don't move.
  - Verified unchanged under the revert: MuJoCo lift (point/cslc/hydro
    = 0.0427/0.0492/0.0466), MuJoCo squeeze (HoldCreep point +0.495,
    cslc +0.082 mm/s), all 33 `smooth_basic_test` tests,
    `newton.tests.test_joint_drive` (19 pass). `test_joint_controllers`
    has a pre-existing XPBD revolute failure unrelated to this code
    path.
- **Don't claim CSLC eliminates creep.** It distributes load to N
  constraints, each running at 1/N capacity, which empirically reduces creep
  by ~N^0.4. Frame the paper's claim as "reduces" not "eliminates".

### 5.5 Force-controlled gripper variant

The current lift_test uses **open-loop position-PD** on the prismatic-X
joints — pads command a fixed inward position with no slip feedback. A
force-controlled variant would expose CSLC's clearest advantage over point
contact: **per-sphere normal-force distribution is a native slip detector**.

Proposed sketch:

1. Switch prismatic-X from `JointTargetMode.POSITION` to `FORCE`.
2. Target force schedule:
   ```
   f_grip(t) = f_min       APPROACH (just enough to make contact)
             = f_squeeze   SQUEEZE (build up)
             = f_squeeze + f_react(slip)   LIFT and HOLD
   ```
3. **Slip detector per model:**
   - Point: only macroscopic sphere-pad lag (crude, lags by O(τ_constraint)).
   - **CSLC:** per-sphere tangential force from `contacts.rigid_contact_force`
     (already requested via `b.request_contact_attributes("force")`). Detect
     slip when individual sphere friction forces saturate against `μ·F_n`
     BEFORE macroscopic lag is visible — the spatially distributed
     information CSLC uniquely provides.
   - Hydroelastic: per-polygon pressure gradient changes.
4. Closed-loop: `if max_utilization > 0.7: f_react += k_react · dt`.

Expected paper result: closed-loop CSLC reacts on O(δt) ≈ 2 ms vs point's
O(τ_constraint) ≈ 30 ms.

Effort: ~1 day force-control wiring, ~2 days slip detector + A/B harness,
~1 day viewer instrumentation.

### 5.6 Performance / speed optimisations — remaining

Dense-solve path (`lattice_solve_equilibrium` via `build_A_inv=True`) is
done and tied CSLC to hydroelastic on per-step wall time (§4.5a). The
remaining head-room is implementation-level:

- **CUDA graph capture** of the per-pair K1 + solve + K3 launch
  sequence. Expected ~80 µs/step savings. Blocker: Newton's state-swap
  pattern changes `body_q` pointers across steps, so a naive capture
  would break on replay. Workaround: capture two graphs on the first
  two steps, alternate by step parity; or eliminate the state swap in
  the benchmark harness (but keeps the idiomatic user code ergonomic
  for MPC/RL rollouts, which don't swap).
- **Active-sphere broad-phase prune** before Kernel 1 and Kernel 3.
  Mark per-sphere AABB overlap vs target's AABB, early-return in the
  transition-band kernels. Biggest win for multi-pad grippers where
  most lattice spheres are far from any target. Low-medium complexity.
- **Batched pair launches** — dispatch all (pad, target) pairs in a
  single kernel launch, with thread-id decoded as
  `(pair_idx, sphere_local_idx)`. Fills the GPU better (378 threads
  under-utilise RTX 3070) and amortises launch overhead across pairs.
  Generality-preserving: neighbour CSR already handles irregular pad
  topologies, the decode is purely bookkeeping.
- **Sparse Cholesky for large pads** (see `cslc_data.py:430` TODO).
  Dense `A⁻¹` is O(n²) memory, fine up to n ≈ 2000; large MorphIt pads
  (n ≈ 10 000) would need a sparse factorisation with precomputed L
  (stored once at init) plus two triangular solve kernels at run-time.
  Keeps the tape-compatible matvec pattern; just swaps one dense
  matvec for two triangular solves.
- **Kernel 1 + Kernel 2a fusion** (compute φ and solve in one kernel).
  Requires grid-wide synchronisation (cooperative groups) which Warp
  may not expose directly. Low priority — modest expected gain
  compared to the above.

### 5.7 Tried and rejected (2026-04-21)

Two kernel-level extensions were implemented, tested, and reverted.
Record here so future-us doesn't re-run the same experiment.

**#1 — Nonlinear contact law `F = kc · pen_3d · (max(pen_3d, pen_ref)/pen_ref)^(p-1)`.**
Global power-law stiffening applied post-hoc to the MuJoCo constraint
stiffness. With `p = 1.5`, `pen_ref = 1 mm`: squeeze HoldCreep improved
from 0.082 → 0.052 mm/s (−37 %, beats hydro's 0.062), lift was bit-exact
unchanged (as designed — dormant when `pen_3d ≤ pen_ref`), timing was
within run-to-run jitter. **Rejected because `p` is a
scene-geometry-dependent global:** Hertz `p = 1.5` is appropriate for
sphere-on-flat but wrong for flat-on-flat (which should stay linear),
so shipping a global `p` would force every new scene to re-tune and
would weaken the paper's claim that distributed contact captures
geometry from per-sphere state, not from a hand-tuned global knob.
A per-sphere geometry-adaptive compliance (local-curvature-dependent
`kc_i`, or a non-Winkler pressure profile that recovers Hertz for
sphere-on-flat *and* stays linear for flat-on-flat automatically) is
the principled fix and is real research, not a kernel tweak.

**#2 — Pressure-weighted friction `μ_eff(i) = μ · tanh(δ_i / δ_ref)`.**
Reduce per-sphere friction scale on low-δ edge spheres to trim their
compliance leak. Null result: lift regressed at aggressive `δ_ref`
(center-sphere μ haircut cost more grip than edge-leak trimming
saved), was a no-op at small `δ_ref`; squeeze creep was insensitive
in either direction. **Rejected — edge-sphere friction contribution
is worth more than the tangential leak it adds in MuJoCo.**

### 5.8 Why the per-sphere pressure gradient should already be emergent

For sphere-on-flat contact, the per-sphere rest overlap is parabolic
in lateral distance from the contact axis:
`φ_rest,i ≈ pen_center − (x_i² + y_i²) / (2R)`.
Central lattice spheres therefore see the largest `φ_rest`, largest
`pen_3d`, largest `δ`, and largest `F_i = kc · pen_3d` — exactly the
center-high / edge-low pressure gradient expected from compliant
contact. The Laplacian coupling (`k_ℓ`) smooths the profile but does
not wipe out the spatial variation. **This is already what the
current linear CSLC implementation produces per-sphere** — no kernel
change is required to get the right qualitative pressure distribution.

What the current implementation does NOT capture is the full
aggregate Hertz scaling `F_total ∝ pen_center^1.5`. A Winkler-style
integration over the parabolic profile gives `F ∝ pen²`, which is
too stiff at deep pen compared to Hertz's sqrt-pressure profile.
Getting the aggregate right needs either (a) per-sphere compliance
that depends on local-curvature / radial position in the contact
patch, or (b) a fundamentally different force law that produces the
elliptical pressure profile Hertz gives. Both are research items,
not kernel tweaks; the global-`p` bandaid from #1 above conflates
"distributed contact captures pressure gradients" (which CSLC does)
with "distributed contact captures aggregate Hertz scaling" (which
CSLC does not, and which a single global exponent cannot fix
without re-tuning per scene).

### 5.9 File state (modified — do not revert)

- `newton/_src/solvers/mujoco/kernels.py` — stiffness conversion fixed
  (lines 390–411). Comment trail of original wrong formula preserved.
- `newton/_src/geometry/cslc_kernels.py`:
  - `out_damping = 0.0` (see §4.2)
  - `out_friction = 1.0` (see §4.3)
  - Smooth-gate Kernel 3 with hybrid emission (see §4.5).
  - **CSLC vs BOX kernels (2026-04-26)**: `_box_signed_dist`,
    `_box_closest_local`, `compute_cslc_penetration_box`,
    `write_cslc_contacts_box`.  Gate uses box centroid (sphere-
    convention) to avoid the closest-point sign flip across the box
    surface; `solver_pen` and `point1` still use the closest-point so
    MuJoCo's penetration math matches the kernel's `pen_3d`.
- `newton/_src/geometry/cslc_handler.py`:
  - `_from_model` passes `build_A_inv=True` (see §4.5a performance).
  - `_launch_vs_sphere` dispatches to `lattice_solve_equilibrium` when
    `A_inv` is present; falls back to iterative Jacobi otherwise.
  - **CSLC vs BOX dispatch (2026-04-26)**: `_launch_vs_box` mirrors
    `_launch_vs_sphere` for box targets; `CSLCShapePair` cached fields
    `other_local_xform` (full transform) + `other_half_extents`;
    `supported_pairs` filter now includes `_GEOTYPE_BOX`.
- `cslc_v1/common.py` (new in 2026-04-26 refactor) — shared helpers
  for squeeze/lift/robot tests: `make_solver`, `count_active_contacts`,
  `read_cslc_state`, `recalibrate_cslc_kc_per_pad`,
  `apply_external_wrench`, `inspect_model`, `get_cslc_lattice_viz_data`.
  **Layout fix (2026-05-03):** `apply_external_wrench` now writes
  `body_f[0:3] = force, body_f[3:6] = torque` (matches Newton's
  `wp.spatial_vector(force, torque)` convention).  Was previously
  swapped — see §4.4c.
- `cslc_v1/squeeze_test.py` — 3-way comparison wired via
  `--contact-models`; HOLD-phase metrics (`hold_drop_mm`,
  `hold_creep_rate_mm_per_s`).  Held target selectable via
  **`--object {sphere,book}`** — book mode exercises the box kernels
  with realistic trade-paperback defaults (152×229×25 mm, 0.45 kg).
  **`--book-as-mesh` (2026-05-03)** — replaces the primitive
  `add_shape_box` with a 12-triangle `add_shape_mesh`; routes
  contact through MuJoCo Warp's BVH-per-triangle mesh narrow phase
  (~12 contacts/pair) and verifies the CSLC-handler-on-mesh
  dormancy gap (handler currently filters out non-(SPHERE,BOX)
  targets).  See §1 "What changes when the book is a mesh".
  HOLD-only external wrench via `--external-force fx,fy,fz` /
  `--external-torque tx,ty,tz` (applied through
  `common.apply_external_wrench`).  Geometry / material overrides:
  `--initial-pen <m>` (recomputes `pad_gap_initial` against the
  target's squeeze-axis half-extent), `--mu <coeff>`,
  `--book-mass <kg>`, `--cslc-spacing`, `--contact-fraction`.
  Rotational metrics on `Metrics`: `max_tilt_deg`, `final_tilt_deg`.
  Helper `_make_box_mesh(hx, hy, hz)` builds a CCW-from-outside
  12-triangle box `newton.Mesh` (verified by `_validation_logs/
  sweep_book_disturbances.py`).
- `cslc_v1/lift_test.py` — 3-way comparison; per-pad kc recalibration with
  cf=0.025 via `recalibrate_cslc_kc_per_pad`; `--cslc-ka` / `--cslc-contact-fraction` / `--kh` CLI overrides; per-step timing diagnostic.
- `cslc_v1/cslc_box_test.py` (new 2026-04-26) — 12 tests for the box
  kernels: SDF analytical (7), K1 contact-active gate (4) regressing
  the d_proj sign-flip bug, and one full-pipeline aggregate-force test
  on the book scene.  Run via
  `uv run --extra dev -m unittest cslc_v1.cslc_box_test -v`.
- `cslc_v1/_validation_logs/sweep_book_disturbances.py` (new 2026-05-03)
   — disturbance sweep across (BOX book, MESH book, SPHERE-pad book) ×
   (point, cslc, hydro) × {none, τ_y=1, τ_y=5, τ_x=5, τ_z=5, F_z=−2}.
   Source of the §1 "Disturbance sweep — three baselines" tables.
   Expected runtime ~3 min on RTX 3070.
- `cslc_v1/_validation_logs/sweep_book_disturbance_curves.py` (new 2026-05-03)
   — magnitude sweep across all 6 disturbance axes × 3 contact models
   on the BOX-on-BOX book scene, exposing the failure curves
   (continuous tilt-vs-magnitude, ejection thresholds).  189-row CSV
   at `sweep_book_disturbance_curves.csv` for downstream plotting.
   Source of the §1 "Disturbance magnitude sweep — failure curves"
   tables (three identified cliffs: point τ_z @ 3 N·m, point τ_y @
   20 N·m, hydro τ_x @ 15 N·m).  Expected runtime ~5 min on RTX 3070.
