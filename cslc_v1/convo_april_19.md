# CSLC Integration Notes — April 19, 2026

Newton + MuJoCo integration of the CSLC (Compliant Sphere Lattice Contact) model.
Goal: validate that CSLC outperforms point contact in squeeze and lift tasks for the ICRA paper.

---

## 1. Squeeze test (`cslc_v1/squeeze_test.py`)

### Scene

Two kinematic box pads squeeze a dynamic sphere under gravity. The sphere starts
with 10 mm penetration per side; pads squeeze an additional 2.5 mm over 0.5 s,
then hold for 1.5 s. The metric is sphere z-drop during the hold phase.

```
uv run cslc_v1/squeeze_test.py --solver mujoco --mode squeeze --steps 1000
```

### Results (after all fixes)

| Model | Z-drop (1 s hold) | Slip rate | Contacts |
|---|---|---|---|
| `point_mujoco` | 1.223 mm | 0.611 mm/s | 2 |
| `cslc_mujoco` | 0.168 mm | 0.082 mm/s | 174 |

CSLC drops 7.3× less than point contact. Both rates are constant throughout
the hold phase (MuJoCo soft-constraint creep — not Coulomb slip).

### Fairness analysis

Both models use `ke=50000`, `mu=0.5`, `kd=500`. The calibration, however,
is not perfectly matched:

`calibrate_kc` targets `contact_fraction=0.15` → 56 spheres expected in contact.
The squeeze test activates 174 (46%). With `kc=1087 N/m` and `keff=892.9 N/m`,
this gives:

- **CSLC** aggregate `Fn = 174 × 892.9 × 15 mm = 2330 N` → friction cap = **1165 N**
- **Point** aggregate `Fn = 2 × 50000 × 15 mm = 1500 N` → friction cap = **750 N**

CSLC has **1.55× more friction capacity** from calibration mismatch alone.
Decomposing the 7.3× result: ~1.55× from excess normal force, ~4.7× from
distributed contact dynamics.

**To make the comparison fully fair**: set `contact_fraction ≈ 0.46` so `kc` is
calibrated to the actual contact count, or explicitly match total normal force
when comparing. For the paper, acknowledge that CSLC's larger contact area is
part of its design — a real compliant pad naturally engages more surface area
than a rigid point.

---

## 2. Lift test (`cslc_v1/lift_test.py`)

### Scene

Two dynamic pad arms (prismatic joints, PD-controlled) approach, squeeze, lift,
and hold a sphere. Because the pads are dynamic (not kinematic), MuJoCo computes
relative contact velocity correctly — friction drags the sphere upward during lift.

```
# Full cycle:
uv run cslc_v1/lift_test.py --viewer gl --contact-model cslc

# Skip to lift (pads already gripping):
uv run cslc_v1/lift_test.py --viewer gl --contact-model cslc --start-gripped

# Headless comparison (both models):
uv run cslc_v1/lift_test.py --mode headless
```

### Parameter matching (April 19)

The original lift_test used `ke=1000`, `kd=50`, `cslc_spacing=10mm` — tuned for
an older volumetric CSLC lattice. These were matched to squeeze_test values:

| Parameter | Before | After (matched) |
|---|---|---|
| `ke` | 1000 | 50000 |
| `kd` | 50 | 500 |
| `cslc_spacing` | 10 mm | 5 mm |
| `cslc_ka` | 2000 | 5000 |
| `cslc_kl` | 100 | 500 |
| `cslc_n_iter` | 40 | 20 |

**Before**: sphere launched upward at lift start (friction insufficient to support
weight, then overshot when constraint caught up). **After**: smooth monotonic lift.

### Results after parameter matching

| Model | Final z | Lifted | Held |
|---|---|---|---|
| `point_mujoco` | 38.9 mm | YES | YES |
| `cslc_mujoco` | 44.0 mm | YES | YES |

Sphere bottom at end of hold: `44.0 - 30 = 14 mm` above ground for CSLC,
`38.9 - 30 = 8.9 mm` for point. Both held in air; CSLC holds 5 mm higher.
Both still creep downward during hold at ~3–5 mm/s (same soft-constraint
effect as squeeze test — not Coulomb slip).

### Proposed next steps

1. **Fix calibration mismatch**: set `contact_fraction` to match actual active
   sphere count in lift test, so CSLC and point have equal aggregate normal force.
   Only then is the comparison purely about distributed friction geometry.

2. **Reduce hold creep**: the 3–5 mm/s downward drift during HOLD is MuJoCo
   constraint regularization (friction timeconst ≈ 30 ms). Options:
   - Tighten `mu` for CSLC contacts to compensate (but this is cheating).
   - Accept it as a known limitation of soft-constraint solvers and report
     it as a drop rate, not a binary "held / fell".

3. **Tilt/twist perturbation**: with basic hold confirmed, add angular impulse
   tests to check rotational stability — the paper's main CSLC claim.

4. **Contact position leverage**: CSLC writes `pos = midpoint(lattice_center,
   sphere_center)` — this is correct for force transmission but may give wrong
   torque lever arms for tilt/twist. Investigate before perturbation tests.

---

## 3. Bugs fixed in `newton/_src/solvers/mujoco/kernels.py`

### 3.1 Stiffness conversion bug (FIXED 2026-04-19)

**File**: `newton/_src/solvers/mujoco/kernels.py`, lines 371–411.

MuJoCo encodes contact stiffness via `solref = (timeconst, dampratio)`.
The contact force law is:

```
F = imp × ke_ref × pen
ke_ref = 1 / (timeconst² × dampratio²)
```

To recover `F = contact_ke × pen` (Newton's calibrated stiffness), we need:

```
imp × ke_ref = contact_ke
→ ke_ref = contact_ke / imp
→ 1 / (tc² × dr²) = contact_ke / imp
```

**Correct formula** (kd=0 branch, used for CSLC after damping fix):
```
tc = sqrt(imp / contact_ke)   → ke_ref = contact_ke/imp   → F = contact_ke × pen  ✓
dr = 1.0  (critically damped)
```

**Correct formula** (kd>0 branch, used for standard Newton contacts):
```
tc = 2/kd  (unchanged — Newton convention for viscous damping)
dr = sqrt(imp / (tc² × contact_ke))  → ke_ref = contact_ke/imp  → F = contact_ke × pen  ✓
```

**Original wrong formula** (preserved as comment trail in the code):
```python
contact_ke *= (1.0 - imp)          # ke_scaled = ke*(1−imp) ≈ 0.05·ke
timeconst = sqrt(1.0 / ke_scaled)
dampratio = sqrt(1.0 / (tc² × ke_scaled))   # = 1.0
# → F = imp × ke_scaled × pen = imp*(1−imp)*ke*pen ≈ 0.047·ke·pen  (21× too low)
```

For `imp=0.95`: original gave `F ≈ 0.05 × ke × pen` instead of `ke × pen` — 21×
too weak. This caused the sphere to fall through the pads in all early CSLC tests.

**Why the original was wrong**: the intent was to "compensate for impedance scaling"
but multiplying `ke` by `(1−imp)` before computing `tc` and `dr` gives a doubly-
reduced force. The correct derivation works backward from the desired `F = ke × pen`
and solves for `tc` and `dr` analytically.

### 3.2 Damping conversion bug (`cslc_kernels.py`, FIXED 2026-04-19)

**File**: `newton/_src/geometry/cslc_kernels.py`, line 377.

`cslc_dc = 2.0 N·s/m` is a viscous damping coefficient calibrated for Newton's
semi-implicit integrator (`F_damp = −kd · v_pen`). In MuJoCo's constraint solver,
the conversion formula for the kd>0 branch is `timeconst = 2/kd`, giving:

```
timeconst = 2 / 2.0 = 1.0 s
```

Standard rigid contacts use `timeconst ≈ 0.004 s` (from `ke=50000, kd=500`).
CSLC's friction constraints were **250× softer**, allowing Coulomb creep at high
velocity throughout the hold phase.

**Fix**: write `out_damping = 0.0` instead of `cslc_dc * pen_scale`. This
triggers the kd=0 branch:

```
timeconst = sqrt(imp / contact_ke) ≈ 0.030 s
```

This is comparable to standard contacts (within 1 order of magnitude), and
**normal force is unchanged** — both branches solve for `ke_ref = contact_ke/imp`,
so `F = contact_ke × pen` regardless. Only friction stiffness improves.

**Why this is theoretically correct**: `cslc_dc` has no valid analog in MuJoCo's
constraint relaxation framework. It's a semi-implicit integrator concept. Setting
`kd=0` tells MuJoCo "use critically-damped constraint enforcement with timeconst
set by stiffness alone" — the appropriate default when the Newton damping
coefficient cannot be meaningfully translated.

### 3.3 Friction double-count bug (`cslc_kernels.py`, FIXED 2026-04-19)

**File**: `newton/_src/geometry/cslc_kernels.py`, line 385.

The MuJoCo conversion kernel treats `rigid_contact_friction` as a **scale factor**
applied to the geom pair's base friction:

```
effective_mu = max(mu_pad, mu_sphere) × rigid_contact_friction
```

Original code wrote `out_friction = shape_material_mu` → `effective_mu = mu × mu = mu²`.
Fix: write `out_friction = 1.0` → `effective_mu = mu × 1.0 = mu`. ✓

---

## 4. File state

**Modified files:**

- `newton/_src/solvers/mujoco/kernels.py` — stiffness conversion fixed (lines 390–411).
  Comment trail of original wrong formula preserved for reference.

- `newton/_src/geometry/cslc_kernels.py`:
  - `out_damping = 0.0` (line 377) — was `cslc_dc * pen_scale`
  - `out_friction = 1.0` (line 385) — was `shape_material_mu`

- `cslc_v1/squeeze_test.py` — squeeze-only diagnostic:
  - Runs point and CSLC back-to-back; reports drop, rate, FD-inferred force
  - Position-FD velocity (body_qd stale under MuJoCo GPU)
  - MuJoCo contact.dist readout at selected steps

- `cslc_v1/lift_test.py` — full lift cycle:
  - Parameters matched to squeeze_test (ke=50000, spacing=5mm)
  - Viewer + headless modes; --start-gripped to isolate lift phase

**Known invariants (do not revert):**
- `out_damping = 0.0` for CSLC contacts in MuJoCo
- `out_friction = 1.0` for CSLC contacts in MuJoCo
- Position-FD for velocity in both test files (body_qd always ≈0 under MuJoCo GPU)
- Right-pad 180° shape rotation in both test files (face_axis=0, face_sign=+1)

---

## 5. TODOs — Generic pad calibration for arbitrary soft pads

To let users define a pad once (geometry + material) and have CSLC pick all
discrete spring constants automatically:

- **Material → spring constants**. Add a `CSLCMaterial(E, nu, h, mu)` spec
  and derive: `ka = E·A_sphere/h`, `kl = G·h/spacing` with `G = E/(2(1+nu))`,
  `ke_bulk = E·A_pad/h`. User specifies measurable physical quantities only.
- **Iterative `contact_fraction` bootstrap**. Replace the hardcoded fraction
  with a 1–2 step bootstrap: assume 0.3, run one collision step, measure
  active count, recompute `kc`. Eliminates the geometry-specific tuning we
  do today (0.46 for the squeeze scene).
- **Per-shape `shape_cslc_contact_fraction`** model attribute, following the
  pattern of the other `shape_cslc_*` arrays. Allows different pads in the
  same scene to use different priors.
- **Per-pad `kc` storage** in `CSLCData`. Currently `kc` is one scalar shared
  across all pads. For mixed pad sizes/materials, promote to a per-shape
  array indexed via `sphere_shape`. Today's per-pad calibration assumes
  uniform pads (`np.mean(n_surface)` across pads).
- **MorphIt integration for arbitrary pad geometry**. Replace the box-face
  lattice generator with a MorphIt-driven sphere packing; auto-compute
  outward normals from local surface gradients and neighbor topology from
  spatial proximity (k-NN at 1.5× spacing).
- **Auto-tune solver params**. `n_iter` from spectral radius (already
  printed in `inspect_cslc_handler`); set `n_iter = ceil(3 / -log10(rho))`
  to converge to 3 digits.

---

## 6. MuJoCo creep mitigation — what works, what doesn't

The ~constant z-drop rate during HOLD is **not Coulomb slip** (`Fz ≈ W`
exactly throughout) but soft-constraint compliance from MuJoCo's CG
relaxation. Tested two knobs:

| Change | point z-drop | CSLC z-drop (cf=0.46) | Ratio |
|---|---|---|---|
| baseline (`solimp[1]=0.95`, pyramidal cone) | 1.223 mm | 0.214 mm | 5.71× |
| `solimp[1] = 0.99` (both models) | 1.223 mm | 0.211 mm | 5.80× |
| `cone="elliptic"` (default solimp) | **0.926 mm** | **0.166 mm** | 5.58× |

- **`solimp[1] = 0.99` does nothing** — creep is not dominated by the
  impedance dmax. (The CSLC kernel flattens solimp width to 0.001 anyway,
  so dmax effectively applies for any penetration > 1 mm.)
- **Elliptic friction cone reduces creep by ~24%** for both models; the
  CSLC/point ratio is preserved. Cone change is now wired into
  `_make_solver` in `squeeze_test.py`.

### TODO — creep mitigation follow-ups

- **Compare against `solver="newton"`** instead of CG. Newton solver
  converges per-step rather than approximating; should reduce residual
  per-step displacement further.
- **Try `SolverSemiImplicit`** as a non-MuJoCo baseline — it has no
  constraint compliance, only Coulomb friction, so any "slip" would be
  real Coulomb slip. Useful for the paper's Newton-solver-side comparison.
- **Sweep `iterations` (CG)** from 100 → 500 → 1000 to test if higher
  iteration count further reduces creep at the cost of step time.
- **Don't claim CSLC eliminates creep.** It distributes load to N
  constraints, each running at 1/N capacity, which empirically reduces
  creep by ~N^0.4. Frame the paper's claim as "reduces" not "eliminates".

---

## 7. Per-pad calibration promotion (2026-04-19, late)

`calibrate_kc` now defaults to `per_pad=True`: each pad's aggregate
stiffness at uniform contact equals `ke_bulk` (the per-shape material
property), regardless of pad count. The handler default
`contact_fraction=0.3` is a moderate generic prior; geometry-specific
overrides go via `recalibrate_cslc_kc_per_pad` in `squeeze_test.py`
(the squeeze pad/sphere pair uses 0.46).

| Quantity | Old (cf=0.15, total) | New (cf=0.3, per-pad) | Override (cf=0.46, per-pad) |
|---|---|---|---|
| `kc` | 1087 N/m | 1087 N/m (coincidence) | 657.9 N/m |
| Per-pad aggregate stiffness | 25 000 N/m (split) | **50 000 N/m** ✓ | **50 000 N/m** ✓ |
| Squeeze z-drop | 0.168 mm | 0.214 mm | 0.214 mm |

The numerical `kc` matches the old default by accident
(`0.15·N_total = 0.3·N_per_pad`); semantics now reflect per-pad
calibration so multi-pad grasps no longer silently halve each pad's
aggregate stiffness.

### Resolution scaling experiment (spacing 5 mm → 2.5 mm)

| Spacing | N_active/pad | kc | z-drop |
|---|---|---|---|
| 5 mm | 87 | 657.9 | 0.214 mm |
| 2.5 mm | 309 | ~161 | **0.119 mm** (10.3× better than point) |

Active count scaled 3.55×; z-drop reduced 1.79× — sub-linear N^0.4
scaling, consistent with diminishing returns from constraint distribution
under MuJoCo's CG solver.

---

## 8. Lift test — slip mitigation findings (2026-04-19)

The lift_test exhibits three additive slip mechanisms during a pick-and-hold:
LIFT-phase lag, LIFT→HOLD inertial overshoot, and HOLD-phase creep.  Re-using
the `slip = (pad_z - sphere_z) - initial_offset` metric (subtracts the
permanent z-offset from sphere-on-ground vs pad-center).

### Hypotheses tested

| # | Intervention | Outcome | Kept |
|---|---|---|---|
| H1 | `SolverSemiImplicit` | **Untestable** — joint-PD drive doesn't work, pad_z=NaN | — |
| H2 | Symmetric LIFT end-ramp (smooth deceleration) | Small win — overshoot 9 → 4 mm | ✓ |
| H3 | MJ iterations 100 → 1000 | **No effect** — confirms slip is constraint-stiffness-limited, not convergence-limited | ✗ |
| H4 | Per-pad recalibration with cf=0.025 (matches actual ~5 active spheres/pad) | **Big win** — final_z 0.0440 → 0.0463; HOLD creep 8.9 → 4.2 mm/s; recalibration helper now in lift_test.py with `cslc_contact_fraction` SceneParam | ✓ |
| H5 | `ka` 5000 → 50000 (break series-spring saturation) | Slight regression — fewer active spheres (delta saturates) | ✗ |

### Final lift-test slip (best config: H2 + H4)

| Config | slip from pad |
|---|---|
| point baseline | 10.1 mm |
| **cslc + H2 + H4** | **2.6 mm** (3.9× better than point) |

### Mass and squeeze-depth sensitivity (Q1/Q2)

| Variable | Value | point slip | cslc slip |
|---|---|---|---|
| Mass | 500 g (default) | 10.1 mm | 2.6 mm |
| Mass | 50 g | 2.1 mm | 0.6 mm |
| Squeeze depth | 1 mm (default) | 10.1 mm | 2.6 mm |
| Squeeze depth | 5 mm | 7.7 mm | 1.5 mm |

**Conclusions:**
- Slip scales sub-linearly with mass — confirms force-proportional
  soft-constraint compliance, NOT Coulomb saturation.
- Closing harder (deeper squeeze) helps modestly via "more spheres engaged →
  lower per-constraint utilisation", not via friction-cone margin.
- The 2.6 mm slip floor under MuJoCo is the irreducible per-step constraint
  compliance.  Cannot eliminate without changing the solver formulation.

### TODO — lift test follow-ups

- **Force-controlled gripper variant.** Replace position-PD drive with
  force-target drive that closes harder when slip detected.  This is the
  realistic robotics setup and should show CSLC's per-sphere pressure
  distribution as a slip detector.
- **Debug `SolverSemiImplicit` joint-PD behavior** (pad_z=NaN under the
  prismatic-X/Z articulated arm); useful as a no-soft-constraint baseline.
- **Cross-validate with hydroelastic** (PFC) — if PFC under MuJoCo also
  exhibits the same slip floor, confirms the bottleneck is solver, not
  contact model.

---

## 9. Hydroelastic comparison in squeeze (2026-04-19)

Newton's hydroelastic backend works (`is_hydroelastic=True`, `kh=...`,
`sdf_max_resolution`).  `kh=1e10` (steel) is unstable and ejects the sphere
after squeeze; `kh=1e8` is stable.  Now wired into `squeeze_test.py` via
`--contact-models point,cslc,hydro`.

### Three-way result (1 s hold, elliptic cone, default solimp)

| Model | z-drop (RESULT metric) | steady creep rate | active contacts |
|---|---|---|---|
| `point_mujoco` | 0.926 mm | 0.246 mm/s | 2 |
| `cslc_mujoco` (cf=0.46) | 0.165 mm | **0.082 mm/s** | 174 |
| `hydro_mujoco` (kh=1e8) | 0.029 mm (rose) | **0.082 mm/s** | 106 |

**Key finding: CSLC and hydroelastic produce IDENTICAL creep rate (0.082 mm/s).**
Both are 3.0× better than point contact under MuJoCo's CG/elliptic solver.
Z-drop differs because hydroelastic settles slightly above the start position
during squeeze (pressure-field equilibrium is asymmetric), but the steady-state
HOLD compliance per step is the same.

This **validates the paper's central comparison**: CSLC achieves the same
distributed-contact behavior as PFC under the same velocity-level MuJoCo
solver, at the cost of sphere queries instead of mesh+SDF+pressure-field
precompute.  The remaining difference is computational and architectural,
not physical.

### kh stability sweep (squeeze, 1 s hold)

| kh [Pa] | z-drop | peak contacts | notes |
|---|---|---|---|
| 1e6 | 0.94 mm | 106 | sphere ROSE 2.3 mm during squeeze |
| 1e7 | 0.078 mm | 103 | rose 3.7 mm |
| 3e7 | 0.029 mm | 103 | rose 1.1 mm |
| **1e8** | **0.033 mm** | **111** | basically static (good default) |
| 1e10 | DIVERGED | n/a | sphere ejected at HOLD start |

Default `kh` in `SceneParams` updated to 1e8.

### TODO — hydroelastic follow-ups

- **Match hydroelastic and CSLC on aggregate normal force**, not on
  per-shape stiffness parameter.  Currently `ke=50000` (point/CSLC) and
  `kh=1e8` (hydro) yield different total grip forces (~1500 N point vs
  ~1100 N hydro by rough estimate); the creep-rate match could partly
  reflect this mismatch.  Sweep kh to find the value where total normal
  force matches point's `ke·pen·N_contacts`.
- **Confirm CSLC/hydro produce comparable contact moments** (rotational
  stiffness, scrubbing torque) — that's where distributed contact really
  matters and where the paper's central claim lies.  Add tilt/twist
  perturbations to squeeze_test.
- **Port to lift_test** — see Section 10.

---

## 10. Three-way lift comparison (point/CSLC/hydro)

`build_hydro_scene` now exists in `lift_test.py`; pad and sphere both carry
`is_hydroelastic=True` (PFC requires both bodies to have pressure fields).
Run with `uv run cslc_v1/lift_test.py --mode headless --contact-models point,cslc,hydro`.

| Model | max_z | final_z | slip from pad | active contacts |
|---|---|---|---|---|
| `point_mujoco` | 0.0500 | 0.0388 | **10.1 mm** | 11 |
| `cslc_mujoco` (cf=0.025) | 0.0500 | 0.0463 | **2.6 mm** | 15 (7 cslc surface) |
| `hydro_mujoco` (kh=1e8) | 0.0529 | 0.0412 | 7.7 mm | 45 |

**Asymmetry from squeeze:** hydroelastic ties CSLC on static squeeze creep
(0.082 mm/s vs 0.082 mm/s) but loses to CSLC on dynamic lift slip
(7.7 mm vs 2.6 mm).  Hypotheses:

1. CSLC's per-pad recalibration fixes a mismatch (~5 active spheres vs the
   default cf=0.3 prior of ~56) that gives CSLC effectively stiffer grip
   than hydroelastic at the small ~1 mm face_pen typical of pick tasks.
   Hydroelastic's per-shape kh prior is fixed; we did not equivalent-tune.
2. CSLC overshoots less at LIFT→HOLD (max_z 0.0500) than hydroelastic
   (max_z 0.0529 — a 5 mm overshoot), suggesting hydroelastic accumulates
   more LIFT-phase momentum.
3. CSLC has 7 active spheres but the grip is concentrated; hydroelastic
   has 45 contacts but each is weaker (pressure × small face_pen × small
   per-polygon area).

For the paper: CSLC's dynamic grip stability is a *new* result not implied
by PFC's static benchmarks — worth highlighting.

### TODO — lift comparison follow-ups

- **Tune hydroelastic kh equivalently** (per-pad aggregate normal force
  matched to point's `ke·pen`).  The current `kh=1e8` is the default from
  the squeeze test; lift-relevant face_pen is 1 mm vs squeeze's 15 mm.
- **Inspect hydroelastic active contacts during lift** (45 vs CSLC's 7).
  Are these on the pad face only, or are corner/edge polygons producing
  off-axis forces?
- **Force-balance audit during HOLD**: read MuJoCo applied normal forces
  per contact; verify total = weight for all three.  Differences would
  expose calibration mismatches.

---

## 11. Force-controlled gripper variant — design (TODO)

The current lift_test uses **open-loop position-PD** on the prismatic-X
joints — pads command a fixed inward position throughout the test, with no
reaction to slip.  This is *unrepresentative* of real robot grippers, which
use **force/current control with feedback**: detect slip → close harder.

A force-controlled lift_test variant would expose CSLC's *clearest* clinical
advantage over point contact: per-sphere normal-force distribution is a
**native slip detector**, while point contact has a single contact and no
spatial information.

### Proposed implementation sketch

1. **Switch joint drive mode** for prismatic-X from `JointTargetMode.POSITION`
   (current) to `JointTargetMode.FORCE` (or torque target on the slider).
   The pad commands an inward FORCE; the slider position becomes a state.

2. **Define a target grip force schedule**:
   ```
   f_grip(t) = f_min  during APPROACH (just enough to make contact)
             = f_squeeze  during SQUEEZE (build up to nominal)
             = f_squeeze + f_react(slip)  during LIFT and HOLD
   ```
   where `f_react` is a reactive term that increases when slip is detected.

3. **Slip detector**:
   - **Point contact**: only signal is `sphere_z - pad_z` lag relative to
     initial offset.  Crude — lags by O(constraint timeconst).
   - **CSLC**: per-sphere tangential force distribution can be read from
     `contacts.rigid_contact_force` (we already request the `force`
     attribute via `b.request_contact_attributes("force")`).  Detect slip
     when individual sphere friction forces start saturating against
     `μ·F_n` BEFORE the macroscopic sphere-pad lag becomes visible.  This
     is the *spatially distributed information* CSLC uniquely provides.
   - **Hydroelastic**: per-polygon pressure is similar; could detect slip
     via pressure gradient changes across the patch.

4. **Closed-loop control law** (simplest version):
   ```
   if max_per_sphere_friction_utilization > 0.7:
       f_react += k_react * dt
   ```
   I.e. close harder whenever any individual contact is approaching its
   friction limit, regardless of macroscopic slip.

### Expected paper result

| Setup | Slip detection latency | Slip mitigation |
|---|---|---|
| Open-loop position | n/a | none — slip accumulates |
| Closed-loop point | O(τ_constraint) ≈ 30 ms | grip adjusts AFTER slip starts |
| **Closed-loop CSLC** | **O(δt) ≈ 2 ms** | **grip pre-adjusts BEFORE slip** |
| Closed-loop hydro | similar to CSLC if per-polygon forces accessed | similar |

This would demonstrate CSLC's value in a **realistic robotics control loop**,
not just in passive simulation — strengthening the paper's transfer story.

### Estimated effort

- ~1 day: switch joint drive mode + write force-target schedule + verify
  open-loop force-controlled gripper still lifts/holds.
- ~2 days: implement slip detector for each contact model + closed-loop
  reactive controller + headless A/B comparison.
- ~1 day: viewer instrumentation to visualise per-sphere friction utilisation.


---

## 12. Solver architecture — Drake/SAP vs MuJoCo vs SemiImplicit

The HOLD creep we see in both squeeze and lift is a **solver property**, not
a contact-model property.  The three relevant paradigms differ in how — and
whether — they leak compliance:

### MuJoCo CG/Newton (what we use today)

Regularised constraint solver.  Each contact carries
`solref = (timeconst, dampratio)` and `solimp = (dmin, dmax, …)`.  Forces
come from relaxing a constrained system whose compliance is **explicit in
the constraint formulation**:

```
0  ≤  φ + (δt + d·c)·v_n + c·f_n   ⊥   f_n  ≥  0,    with  c = 1/k
```

The `c·f_n` term is the source of the residual per-step slip.  Designed for
speed and convexity; pays for it with constraint softness.

### Drake SAP — Castro et al. 2021

**Semi-Analytic Primal** solver.  Solves an unconstrained convex
optimisation problem each timestep:

```
min_v  ½(v − v*)ᵀ M (v − v*)  +  Σ_contacts  ψ_i(v)
```

where `ψ_i` is a convex per-contact dissipation potential derived from the
contact compliance.  Crucially, SAP does **not** regularise via
`solref / solimp`; compliance enters as a (possibly very stiff) function in
the objective.  At the velocity level, SAP enforces contact and friction
constraints to the numerical accuracy of the Newton iteration on the
convex objective — **no per-step compliance leak by construction**.
Drake's hydroelastic-on-SAP demos hold objects without creep for exactly
this reason.

### Newton's `SolverSemiImplicit` (penalty integrator)

Not a constraint solver at all.  Forward-Euler integration of penalty force
laws: `F_n = −k·φ − d·v_n` (Hunt–Crossley) plus regularised Coulomb
friction, then `v_{n+1} = v_n + δt · M⁻¹ · F_total`.  No LCP, no convex
optimisation, no implicit time stepping — every contact force is computed
explicitly per step.

No constraint compliance because there are no constraints in the
optimisation sense.  But it needs **very small `δt`** for stable contact
(Hunt–Crossley with high `k` is stiff), **can't handle hard constraints**
(joints with PD must be soft), and can **chatter at stick-slip
transitions**.

### Side-by-side

| Paradigm | Compliance leak | Per-step cost | Implicit step size | Hard joints | Differentiable |
|---|---|---|---|---|---|
| MuJoCo CG/Newton | **Yes** (`c·f_n` term) | Low | Stable for any δt | Yes | Via Warp |
| Drake SAP | **No** (rigorous KKT) | Medium–High (sparse Cholesky) | Stable for any δt | Yes | Limited |
| `SolverSemiImplicit` | **No** (no constraints) | Very Low | Tiny δt required | No (soft only) | Via Warp |

SAP and SemiImplicit reach the **same HOLD outcome (no creep) for opposite
reasons**:

- **SAP**: solves the constrained problem rigorously each step, no leak by
  construction.
- **SemiImplicit**: doesn't have a constrained problem to leak from; it's
  just integrating forces.

### Implications for this thesis

- **Don't port SAP to Newton** — that's a 3–6 month engineering project of
  its own, not a step in this paper.  See section 5 TODOs for the
  CSLC-specific work that's a higher priority.
- **Use Drake directly** for any solver-side comparison the paper needs
  (CSLC under MuJoCo vs PFC under SAP).  Maintain a parallel Drake test
  harness rather than wrapping SAP into Newton.
- **Newton/MuJoCo stays the primary platform** for differentiable CSLC.
  The MuJoCo creep floor is a known limitation; the paper should
  acknowledge it explicitly rather than try to hide it.
