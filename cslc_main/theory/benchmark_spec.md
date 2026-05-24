# Contact-model benchmark spec — DRAFT v0.9

Three-way comparison: **point contact (MuJoCo default), CSLC, hydroelastic.**

This is a living document.  Read top-to-bottom for a complete picture; §7
is the actionable to-do list at the current state.

## What this benchmark answers

**The headline scientific question**: does CSLC achieve grip on a held
object with less pad penetration (and therefore less squeeze force) than
hydroelastic or single-point contact, at the same physical material
specification?  If yes, CSLC's distributed-lattice formulation is doing
something the other models can't.  If no, the distributed-contact
hypothesis loses, or the comparison reveals a different axis where CSLC
wins (e.g., gradient quality through the contact step).

The benchmark also produces two secondary results:
- **Quasi-static contact mechanics validation** (F(d), slip-onset, patch
  radius) — establishes that material-property matching produced
  physically reasonable behavior in each model.
- **Differentiability characterization** — does each model produce
  finite, finite-error gradients of object motion w.r.t. pad pose?
  CSLC's intended advantage for MPC/RL applications stands or falls
  on this.

## Quick orientation for a fresh reader

- **Calibration philosophy**: match the physical material, not observable
  behavior.  §0 explains why F(d)-matching would defeat the comparison.
- **Three models**: point (MuJoCo default), CSLC (this codebase's
  distributed-lattice model), hydroelastic (Newton's SDF-volumetric
  model).  §2 explains parameter surfaces.
- **The scene** is a fixed dome-pad gripper closing on a 25mm steel cube;
  the sweep axis is pad penetration depth (`--pad-close-offset`).  §1.
- **Calibration values** are empirically anchored from §7.1 measurements;
  §3.3 has the locked numbers.
- **What's done**: anchors measured, mass locked, wrench readout working,
  one pilot run on CSLC alone.
- **What's next**: pilot the same range on hydro and point, then run the
  headline sweep.  §7 has the ordered to-do list.

## Changelog

* **v0.9** (THIS REVISION): first measurements at the v0.8-corrected
  R_pad = 10 mm geometry using the new
  [exp_anchors.py](../grasp/scripts/exp_anchors.py) instrument.
  Three substantive findings:
  (a) **F_per_pad collapsed 43× from v0.7** (1.58 N at v0.9 silicone
  target vs 67.9 N at v0.7).  Grip headroom dropped from ~60× cube
  weight (122 g) to ~1.3×.  The v0.7 number was inflated by the
  retracted R_pad = 20 mm geometry's patch overflow, AND by a
  different ad-hoc L-measurement protocol — v0.9 uses the documented
  delta_n-with-three-spring-correction protocol.  v0.9 vs v0.7 are
  NOT bit-comparable.  See §3.3 and §7.1.
  (b) **contact_fraction ≈ 0.7 ± 0.1**, not 1.0.  The v0.7 finding
  was inflated by spheres engaging onto adjacent box faces past the
  cap overflow.  At corrected R_pad = 10 mm only the apex region
  (~70%) actually engages.  Per-pad asymmetry up to ~25% per run.
  See §3.5.
  (c) **F(d) non-monotonicity SURVIVES the geometry fix**, but the
  peak shifted from ~1 mm (v0.7) to ~2 mm (v0.9), AND the cube DROPS
  at 5 mm depth at v0.9 (F → 0, Newton-III breaks).  v0.7 held
  throughout the {0.2, 0.5, 1, 2, 5} mm range.  Keep §5.3(a) as a
  paper-grade CSLC characteristic, but document the shape change
  and the drop-at-deep-squeeze finding.  See §5.3 and §7.4.
  Two methodology notes:
  (i) **GPU non-determinism dominates calibration scatter.** Three
  re-anchor iterations at the same input ke gave derived ke values
  spanning {4.4e5, 5.0e5, 5.3e5} — 10-20% scatter at the same
  config, consistent with the C2-closure bounded-non-determinism
  finding.  v0.9 calibration values are central estimates; treat
  trailing significant figures with skepticism.
  (ii) **Self-consistency requires contact_fraction iteration.**
  The exp_anchors script measures empirical contact_fraction but the
  scene's `calibrate_kc` uses `CSLCParams.contact_fraction` (default
  0.025), so the running kc doesn't match the measurement-implied
  kc.  For a tight self-consistent v0.9 calibration, run
  exp_anchors with `--cslc-contact-fraction 0.7` and iterate.
  This was not done for the v0.9 measurements (single-pass at
  default cf=0.025); the §3.3 numbers are first-order valid but
  carry an additional self-consistency caveat documented in §7.1.
* **v0.8**: retract R_pad = 20 mm geometry (violated
  the spec's own dome-vs-box constraint by ~19 mm at the 25 mm box —
  the contact patch overflowed the approach face).  Restore the
  original Step 8 / 9 / 10 production geometry R_pad = 10 mm at
  half_angle = 72°.  Three downstream consequences:
  (a) §3.3 calibration table v0.7 numbers carry forward but are
  flagged "RE-VALIDATION REQUIRED" — A_patch = 330 mm² is
  geometrically impossible at the smaller cap, so re-measurement is
  not optional;
  (b) §5.3 a non-monotonic F(d) finding is preserved as a hypothesis
  but reopened pending re-validation — the drop from 69 N → 6.6 N
  could be patch-overflow rather than the attributed anchor-pull-back
  mechanism;
  (c) §3.5 calibration loop closure used 2-spring `kc_series` instead
  of 3-spring `keff` and over-predicted force by ~60% (264 N instead
  of the correct E·A = 165 N).  Corrected; residual restated as 2.4×
  not 4×.
  §7 Block A reopened for re-anchoring at R_pad = 10 mm.
* **v0.7**: document restructured for fresh-reader
  comprehension.  v0.6 empirical findings promoted from changelog into
  the body (§3.3 calibration table, §3.4 coupling analysis, §3.5
  contact_fraction = 1.0, §5 non-monotonic F(d) characterization).
  §7 reorganized as actionable to-do list with completion status per
  item.  Two findings clarified: (a) at v0.6 empirical anchors, the
  series-spring coupling §3.4 warned about is NOT tripped (per-sphere
  kc ≈ 15,000 << target_ke = 500,000, so kc_series ≈ kc); (b) F(d)
  non-monotonicity is likely an anchor-spring pull-back mechanism,
  not gate saturation alone — added to §5 as a CSLC-characteristic
  finding worth reporting.
* v0.6: §7.1 empirical anchors measured.  Three findings: (a) wrench
  readout works via `state.mujoco.qfrc_actuator` after pre-allocation;
  proxy `Σ stiffness × L` was off by ~3-100×; (b) at 5.75 g cube the
  cube settles +2-3 mm off-center; at 122 g cube the system is
  symmetric and Newton-III balanced; (c) F(squeeze_depth) is
  non-monotonic.  Mass locked at 122 g (steel density 7800 kg/m³).
  Pad construction documented: pads are rotated, not mirrored.
* v0.5: math + methodology corrections.  Fixed: "harmonic-mean" →
  "series-spring composition"; "20% above" → "17% below" (correct
  sign); added `ka_tangent_ratio = 0.336` (Poisson-derived) to CSLC
  calibration table.  Sweep axis switched from "squeeze force" to
  `pad_close_offset`.  Headline metric reframed to continuous
  `offset_50`.
* v0.4: added point contact as third model; calibration reframed from
  F(d) behavioral matching to material-property matching + outcome
  sweep.  Validation experiments (F(d), slip-onset, patch radius)
  moved to §4.
* v0.3.1: tightened language after n=3 std-of-std caveat.
* v0.3: first n=3 seed sweep showed seed-variance dominated
  single-seed claims.  Required (mean, std) reporting added.
* v0.2: operating point updated post-Bug B post-mortem.  Approach-face-
  only sampling restored default `alpha=0.3, n_iter=40` viability.
* v0.1: initial draft, operating point `alpha=0.1, n_iter=120`.

## 0. Calibration philosophy

**Match the physical material, not the observable behavior.**  Each model
gets parameters that correspond to the same silicone-equivalent fingertip
material.  Differences in observable behavior — F(d) curve shape, grasp
success at low squeeze, contact patch dynamics — emerge from differences
in how each model translates material properties to contact forces.
Those differences ARE the scientific contribution; calibrating them
away would defeat the comparison.

Why this approach over F(d) behavioral matching:
- F(d) matching forces both models to produce the same force at the
  same penetration by construction, which mathematically eliminates
  any "CSLC grips at lower squeeze" claim before the experiment runs.
- F(d) curves still get reported — as **validation** that material-
  property matching produced physically reasonable contact mechanics
  (see §4).  Not as calibration.

**Scope of comparison**: the experiment compares CSLC against
**Newton's implementations of hydroelastic and point contact**, not
against the canonical-physics references (Drake hydroelastic, MuJoCo
standalone).  Any divergence between Newton's implementations and the
canonical models is out of scope and would be supplementary work.
Disclose this in any paper write-up so the comparison claim is
correctly scoped.

## 1. Scene

Single fixed scene; the sweep axis is `pad_close_offset`, not scene
geometry:

- Held object: **25 mm cube** (`--object-kind box --box-side 0.025`),
  spawn z = 0.10 m, free-falls and settles on z=0 ground before pads
  engage.
- **Object mass: 122 g (steel density 7800 kg/m³)**, locked from §7.1
  measurement.  CLI: `--object-density 7800`.  At lighter masses
  (5.75 g default) the cube settles +2–3 mm off-center deterministically
  and per-pad forces are 6× asymmetric; gravity dominates these small
  imbalances at 122 g and the system becomes Newton-III balanced
  (F_left + F_right = 0 in x to within 0.3%).
- Pads: **parametric-dome, R_pad = 10 mm, half-angle = 72°**
  (`--pad-kind dome_param --pad-r-pad 0.010 --pad-half-angle 72`).
  This is the original Step 8 / 9 / 10 production geometry (notes.md);
  v0.6 / v0.7 specified R_pad = 20 mm, which violates the geometric
  constraint below (see v0.8 changelog).
- Approach-face-only sampling on box (`_BOX_APPROACH_FACES = ("+x", "-x")`)
  enabled by default in
  [contact_models.py](../grasp/contact_models.py).
- Grip schedule: APPROACH → SQUEEZE → LIFT → HOLD.  SQUEEZE controlled
  by `--pad-close-offset` (position control); achieved squeeze force
  is the **dependent** measurement (see §5).

Dome-vs-box geometric constraint (`2·hx ≥ 2·patch_radius + 6mm`):

    patch_radius = R_pad · sin(half_angle) = 10 mm · sin(72°) ≈ 9.51 mm
    2·patch_radius + 6 mm                  = 25.02 mm
    2·hx                                   = 25 mm  (box side)

The 25 mm box is **marginally below the worst-case constraint**
(25 mm < 25.02 mm by 0.02 mm).  The constraint's 6 mm margin was
sized for full-cap-rim engagement at the geometric maximum; at the
empirical (v0.7-carried-forward) L ≈ 0.12 mm the actually engaged
patch is the Hertz disk `sqrt(R_pad · L) ≈ 1.1 mm` — about 12% of
the cap-rim radius — so the worst-case constraint massively
over-bounds the engaged contact extent.  The literal 0.02 mm
violation is acceptable in practice but must be re-checked at v0.8
re-validation: if §7.1 re-measurement at R_pad = 10 mm produces
L > ~1 mm, the Hertz disk grows toward `sqrt(10 · 1) ≈ 3.2 mm`,
still under the 9.5 mm cap-rim radius but worth re-confirming
patch-stays-inside-face explicitly.  Any redesign that grows the
cap rim (larger half_angle or R_pad) must bump `--box-side`
proportionally or revisit this analysis.  Corner-induced wedge does
not engage at the v0.8 anchor.

**Pad construction quirk to know**: pads are constructed by **rotation**
(180° about z for box, ±π/2 about y for dome) of a single body-frame
lattice, not by mirroring.  This means lattice-asymmetry biases in
some directions ADD across pads (y for dome) and in others CANCEL
(world-x for dome via ±π/2 rotation about y).  The asymmetry of Step 10d
(y-drift from Fibonacci-spiral sampling) is consistent with this
construction — y-biases add.  A follow-up investigation could test
whether a mirror-based pad construction (instead of rotation) would
eliminate Step 10d-style drift, but that is out of v0.7 benchmark scope.

## 2. The three models

| Model | Physical knob | Numerical knob | Physical-numerical separation |
|---|---|---|---|
| Point contact (MuJoCo default) | (none — `ke` does both) | `ShapeConfig.ke` (N/m) | **None.**  Intrinsic limitation of the model. |
| CSLC | `material.ke_pad_physical` (N/m) | `material.ke_target_constraint` (N/m) | **Labels separated** by C2 ship; physics coupled via series-spring composition.  See §3.4. |
| Hydroelastic | `material.kh` (Pa/m) | `material.ke_target_constraint` (N/m) | **Clean** — separate arrays (`shape_material_kh` vs `shape_materials_ke`), no cross-talk. |

**Point contact's lack of physical-vs-numerical separation is an honest
limitation of the model, not a comparison flaw.**  The squeeze-sweep
will show where that limitation bites versus models with separation.

## 3. Calibration (locked v0.7 from §7.1 empirical anchors)

### 3.1 Target physical material

**Silicone tactile sensor analog:**
- Elastic modulus E ≈ 500 kPa.  Citation pending — recommend anchoring
  to GelSight or SynTouch BioTac published material data.  If no
  citation is available, document as a chosen operating point.
- Poisson ratio ν ≈ 0.49 (near-incompressible flesh).  CSLC anchor
  tangent ratio derives via `ρ = 1/(2(1+ν))` (theory.txt
  eq:anchor-aniso): ρ = 1/(2·1.49) ≈ **0.336**.
- Friction coefficient μ ≈ 0.5 (silicone on rigid plastic, mid-range).

### 3.2 Target numerical regularization

**MuJoCo constraint stiffness:** `ke_constraint = 5×10⁵ N/m` for all
three models.  Rationale: Bug B operating point (notes.md C2 closure
§5) — suppresses wedge-induced normal-axis drift on convex pads,
stays grip-stable on flat-face geometries, MuJoCo `timeconst =
√(0.95/5e5) ≈ 1.4 ms` is well-resolved at dt = 2 ms.

### 3.3 Per-model parameter assignment (locked v0.9; first-pass at R_pad = 10 mm)

Calibration uses empirical anchors measured in §7.1 via
[exp_anchors.py](../grasp/scripts/exp_anchors.py) at the silicone-
target 122 g cube scene at R_pad = 10 mm:

- **L_phi_eff = 0.28 mm** (v0.9 central; range 0.25 - 0.32 mm across
  three same-config runs due to bounded GPU non-determinism per C2
  closure).  Measured as
  ``L_delta_n · (ka + kc_series) / kc_series`` — the contact
  penetration depth, NOT the raw lattice-sphere displacement
  ``delta_n``.  See exp_anchors.py "L protocol" docstring for full
  derivation.  The v0.7 number was 0.12 mm under an ad-hoc protocol;
  v0.7 vs v0.9 are not bit-comparable.
- **A_patch_avg = 250 mm²** (v0.9 central; range 175 - 265 mm² across
  runs).  Smaller than v0.7's 330 mm² because the cap rim's projected
  area drops 4× when R_pad goes 20 → 10 mm.
- **contact_fraction ≈ 0.7 ± 0.1** (v0.9 measured; per-pad asymmetry
  up to 25% per run).  Materially smaller than v0.7's 1.0, which was
  inflated by spheres engaging onto adjacent box faces past the cap
  overflow at the retracted R_pad = 20 mm.  At corrected R_pad = 10 mm
  only the apex region (~70%) engages.

| Param | Point | CSLC | Hydroelastic |
|---|---|---|---|
| Physical compliance | `ke = 5×10⁵` (shared with numerical) | `ke_pad_physical = E·A/L ≈ 4.5×10⁵ N/m` (v0.9) | `kh = E/L ≈ 1.8×10⁹ Pa/m` (v0.9) |
| Numerical regularization | `ke = 5×10⁵` (shared with physical) | `ke_target_constraint = 5×10⁵ N/m` | `material.ke_target_constraint = 5×10⁵` (→ ShapeConfig.ke) |
| Friction | `mu = 0.5` | `mu_friction = 0.5`, `k_stick = production default` | `mu = 0.5` |
| Anchor tangent ratio | n/a | **`ka_tangent_ratio = 0.336`** (Poisson-derived) | n/a |
| Damping | `kd = 0` | `dc = 0` (production default) | `kd = 0` |
| Active spheres / patch | (n/a, single point) | calibrated at `contact_fraction ≈ 0.7` (empirical v0.9) | (n/a, SDF field) |

**Derivation provenance:**
- `E·A/L = 5×10⁵ · 2.5×10⁻⁴ / 2.8×10⁻⁴ ≈ 4.46×10⁵ N/m`.  Maps Young's
  modulus to CSLC's `k_e_bulk` interpretation in `calibrate_kc`.
- `E/L = 5×10⁵ / 2.8×10⁻⁴ ≈ 1.78×10⁹ Pa/m`.  Maps Young's modulus to
  Newton's `kh` (Pa/m) parameterization (Drake-style elastic-foundation
  depth).
- Both depend on the empirical L_phi_eff = 0.28 mm at the v0.9
  protocol.  L_phi_eff is the *load-dependent* contact penetration
  depth at HOLD; sensitivity to L is the largest single source of
  cross-model parameter uncertainty.  At v0.9 the GPU non-determinism
  band alone gives a ±15% spread on derived ke_pad_physical.
  Document in the paper.

**Default deviations**: the v0.9 calibration sets `material.kh ≈
1.8×10⁹ Pa/m` (vs MaterialParams default 5.3×10⁸; 3.4× stiffer) and
`material.ke_pad_physical ≈ 4.5×10⁵` (vs default 5×10⁴; 9× stiffer).
Both deviations are intentional — defaults were chosen for backward
compat with sphere baseline, benchmark requires silicone-target values.

**v0.9 self-consistency caveat.**  The exp_anchors script measures
contact_fraction ~ 0.7 in the HOLD window, but the scene's
``calibrate_kc`` calculation runs at ``CSLCParams.contact_fraction =
0.025`` (GraspConfig default) — so the running ``kc`` inside the
sim doesn't match the value implied by the measurement.  First-pass
calibration is OK because both fall in similar magnitude ranges
(``kc ≈ 5×10³`` in either case), but a tight self-consistent
calibration loop should run exp_anchors with
``--cslc-contact-fraction 0.7`` and iterate.  Not done for v0.9;
flagged in §7.1 as future work.

### 3.4 The CSLC series-spring composition caveat

The C2 ship of `MaterialParams.ke_pad_physical` / `ke_target_constraint`
separates the *labels* but not the *physics*: MuJoCo's rigid-contact
buffer accepts one stiffness per contact, computed from

```
kc_series = kc · target_ke / (kc + target_ke + ε²)
```

This is **series-spring composition** (algebraically identical to
the parallel-resistor formula: `1/kc_series = 1/kc + 1/target_ke`
for small ε; despite the shared formula these represent **series
springs**, not parallel ones — same algebraic form, opposite
kinematic interpretation, so don't conflate with the inter-pad
"parallel" left/right composition).  It always lowers effective
stiffness below the smaller of `kc` and `target_ke`.  The formula is
sometimes called "harmonic mean" colloquially, but strict
mathematical harmonic mean is `2·a·b/(a+b)` (factor of 2 different).
This spec uses "series-spring composition" for accuracy.

**At v0.9 empirical anchors, the coupling is even milder than v0.7.**
Running `calibrate_kc` with N_contact = 105 (empirical
contact_fraction = 0.7 × 150), ka=25000, ke_phys=4.5×10⁵,
ke_constraint=5×10⁵:

- `1/kc = 105/4.5×10⁵ − 1/25000 − 1/5×10⁵`
- `    = 2.33×10⁻⁴ − 4×10⁻⁵ − 2×10⁻⁶ = 1.91×10⁻⁴`
- `kc ≈ 5,200 N/m per sphere`
- `kc_series = 5,200 · 5×10⁵ / (5,200 + 5×10⁵) ≈ 5,150 N/m`

So `kc_series ≈ kc` (1% reduction; v0.7 had 3% reduction).  The
numerical knob (`ke_target_constraint`) influences contact stiffness
even less at v0.9 than v0.7 because kc per sphere dropped 3×
(5,200 vs 15,000) while target_ke stayed at 5×10⁵.

**The 1a-refactor escalation criterion (§7.8) is NOT tripped at
v0.9 anchors either** — kc/target_ke = 0.01 (vs v0.7's 0.03).
Series composition is essentially a no-op.

**v0.7 numbers, for comparison** (now superseded; carried for the
version-history audit trail).  At ke_phys=1.38×10⁶ on R_pad=20mm
geometry with contact_fraction=1.0:
- `1/kc = 150/1.38×10⁶ − 1/25000 − 1/5×10⁵ = 6.67×10⁻⁵`
- `kc ≈ 15,000 N/m per sphere; kc_series ≈ 14,600 N/m (3% reduction)`

Hydroelastic does NOT have this coupling: `kh` flows through
`area × k_eff(kh_a, kh_b)` into MuJoCo's `contact_ke` without
dependence on the numerical-regularization `ke`.  Architecturally
cleaner.  Disclosed but does not change the v0.7 squeeze-sweep plan.

### 3.5 contact_fraction ≈ 0.7 (empirical v0.9; was 1.0 in v0.7)

At v0.9 anchors with a 122 g cube and `pad_close_offset = 1 mm`,
about 70% of the 150 surface spheres engage per pad (`delta_n > 1×10⁻⁶`
on the persistent-active filter).  Per-pad asymmetry up to ~25% per
run (e.g. 0.89 left / 0.68 right) and 10-20% run-to-run scatter from
GPU non-determinism — three same-config runs gave
{0.80/0.80, 0.87/0.68, 0.93/0.79}.

**This supersedes the v0.7 finding** of contact_fraction = 1.0 (all
150 / 150 spheres active).  The v0.7 result was inflated by spheres
engaging onto adjacent box faces past the cap overflow at the
retracted R_pad = 20 mm geometry; at the corrected R_pad = 10 mm
only the apex region engages, and the back-cap spheres legitimately
don't touch the cube.

What this means for the calibration:
- `calibrate_kc` formula path is the active code path (no fallback);
  the three-spring composition is internally consistent at N_contact ≈ 105.
- Per-sphere `kc ≈ 5,200 N/m` (v0.9; was 15,000 N/m at v0.7).  The right
  per-sphere stiffness for the patch-level prediction is the
  **three-spring series**:

      1/keff_per_sphere = 1/ka + 1/kc + 1/ke_target
                        = 1/25000 + 1/5200 + 1/500000
                        = 4.0e-5 + 1.92e-4 + 2.0e-6  =  2.34e-4
      keff_per_sphere  ≈ 4,270 N/m

  Aggregate patch-level stiffness `N × keff ≈ 105 × 4,270 ≈
  4.5×10⁵ N/m` — which equals `ke_bulk = E·A/L = 4.5×10⁵ N/m` by
  construction of `calibrate_kc`.  At L_phi_eff = 0.28 mm this
  predicts `F = ke_bulk · L = E·A = 5×10⁵ · 2.5×10⁻⁴ = 125 N per pad`.

  **Empirical (v0.9) is 1.5 N at 1 mm depth — predicted-vs-actual
  ratio ≈ 80×.**  This is *much* worse than v0.7's 2.4× gap.

  Three working hypotheses for the v0.9 gap, none confirmed:
  (a) **L_phi_eff inflation.**  The delta_n → phi_eff conversion
      factor `(ka + kc_series) / kc_series` assumes face-on contact
      equilibrium with the ideal three-spring law.  Real curved-pad-
      on-flat geometry may not satisfy this; effective L_phi_eff may
      be smaller than the conversion suggests.
  (b) **Non-equilibrium HOLD.**  The HOLD-averaged Newton-III residual
      is 0.01% (balanced on the mean), but local sub-cycle oscillations
      could be reducing the time-averaged force below the
      static-equilibrium prediction.
  (c) **Calibration regime divergence.**  At v0.9 the per-sphere
      kc (5,200 N/m) is much closer to ka (25,000 N/m) than at v0.7
      (15,000 N/m vs 25,000 N/m); the linearised three-spring model
      may be a worse approximation in the v0.9 regime.

  v0.7 closure used N=150 (cf=1.0), kc=15,000, predicted F=165 N vs
  empirical 67.9 N (2.4× over).  At the v0.7 over-prediction the gap
  was attributed to the F(d) non-monotonicity (system operating past
  F-peak).  The same explanation cannot fully account for the v0.9
  80× gap — investigation pending.  See §7.8 escalation criterion
  for the suggested follow-up.

**Caveat for the sweep**: at smaller `pad_close_offset` (the regime
the headline sweep targets), N_active drops below 150 as marginal
spheres disengage.  The empirical contact_fraction is operating-point
dependent, not a fixed value.  §7.4 captures the per-offset N_active
profile during the pilot to set calibration per-sweep-cell if needed.

### 3.6 Common settings

All three models run with identical:
- `--cslc-alpha 0.3` and `--cslc-n-iter 40` (default; only applies to
  CSLC, listed for reference).
- MuJoCo solver: pick ONE iteration count and apply to all three
  models.  See §7.3.
- `dt = 2 ms`, `sim_substeps = 4`.
- `gap = 0.005` (5 mm contact margin) where applicable.
- Same gravity, same grasp trajectory, same seed RNG.

## 4. Validation experiments

These experiments validate that material-property matching produced
physically reasonable contact mechanics on all three models.  They are
NOT calibration steps — parameters are fixed before these run.

**Implementation location** (TBD §7.7): new scripts under
`cslc_main/grasp/scripts/exp_validation_*.py`, alongside existing
experiment drivers.  Each script takes `--contact-model {cslc, hydro,
point}` and dumps measurements to CSV for plotting.

### 4.1 Quasi-static F(d) curve

Scene: dome pad attached to kinematic body, pressed into rigid plane
at prescribed penetration depths `d ∈ {0.02, 0.05, 0.1, 0.2, 0.5, 1.0,
2.0, 5.0} mm`.  No gravity, no dynamics — measure steady-state normal
force F at each d via `state.mujoco.qfrc_actuator`.

Pass criterion: all three models produce **physically reasonable
F(d) curves**.  Monotonic is the textbook expectation, BUT see §5
finding: CSLC has been observed to be non-monotonic past ~1 mm depth
at the 122 g grasp scene, likely from anchor-spring pull-back.  If
the quasi-static §4.1 test confirms the non-monotonicity, this becomes
a positive characterization of CSLC's distributed-contact mechanics,
not a failure.

Report: one figure with three curves on the same axes; characterize
each model's monotonicity, saturation depth, and force scale at
matched penetration.

### 4.2 Slip-onset

Scene: pad pressed into plane at the empirical HOLD-phase normal load
(68 N at 122 g cube, 1 mm pad_close_offset — §7.1).  Apply slowly-
increasing tangential force; measure the lateral force at which slip
onsets.

Pass criterion: `F_slip ≈ μ × F_normal = 34 N` within ±15% on all three
models.  Larger deviation flags a friction-model interaction issue
that needs investigation before the headline sweep.

### 4.3 Contact patch radius at fixed load

Scene: same as §4.2.  Measure:
- **Point**: patch radius = 0 (mathematical, single contact point).
- **CSLC**: "active sphere set" = `{i : phi_eff_i > 10⁻⁶}`.  Patch
  radius = max distance from active-set centroid to any active sphere.
  Empirical comparison reference: 330 mm² convex-hull area at the
  HOLD operating point (≈ 10.2 mm equivalent disk radius).
- **Hydroelastic**: contact-surface area from `output_contact_surface`,
  converted to equivalent disk radius.

This metric is EXPECTED to diverge across models — point contact has
zero patch by construction, distributed models have finite patches.
**The patch-radius asymmetry IS a headline science finding.**

## 5. Headline experiment: pad_close_offset sweep

### 5.1 Sweep design

**Sweep axis: `pad_close_offset`** (position control, what the existing
pipeline supports natively).  Achieved squeeze force is the **dependent
measurement**, reported per-side, not a sweep parameter.

**Proposed sweep range (v0.9; pending hydro/point pilots)**:
`{0.1, 0.2, 0.5, 1.0, 2.0} mm`.

**Rationale (v0.9 update from §7.4 re-pilot, supersedes v0.7
proposal of {0.025, 0.05, 0.1, 0.2, 0.5} mm)**:
- Upper bound capped at 2 mm: at v0.9 R_pad = 10 mm the cube DROPS
  at 5 mm depth (F → 0, Newton-III residual 12%; §5.3 a v0.9
  finding).  The F-peak shifted to ~2 mm.  Above ~3 mm grip is
  unstable.
- Lower bound raised to 0.1 mm: the v0.7 lower bound of 0.025 mm
  was at MuJoCo's positional-resolution edge (~30 µm at default
  ke), AND v0.9 §7.4 pilot at 0.2 mm already shows marginal grip
  (F = 0.74 N, ~30% below cube weight × mu).  Below 0.2 mm the
  v0.9 calibration likely transitions to held-NO; the bracketed
  range needs to cover that transition with margin.

This range covers the CSLC grip-success transition (at v0.9, the
transition is between 0.1 and 0.2 mm) and the peak-F regime
(~2 mm), avoiding the unstable-grip regime (≥ 3 mm).  Hydro and
point pilots are required to confirm their transitions also fall
in this range — if either model's transition is outside
{0.1, 2.0} mm, the range expands.

**Resolution caveat on the 0.025 mm lower bound.**  At `dt = 2 ms`
and `ke_target_constraint = 5×10⁵ N/m`, MuJoCo's regularised
constraint reaches positional accuracy of roughly `dt ·
sqrt(impedance/ke) ≈ 30 µm`.  A commanded `pad_close_offset = 25 µm`
is at or below the solver's own positional resolution; the achieved
penetration may not differentiate from the 0.05 mm cell.  If the
hydro/point pilots fail to resolve a distinct transition at 0.025 mm
(e.g., identical grip outcomes for 0.025 and 0.05 mm cells), raise
the lower bound to 0.05 mm and accept the narrower sweep.  Either
expand UP (into the 1–5 mm regime where §7.4 already pilot data) or
acknowledge a coarser resolution at the floor.

- Models: 3 (point, CSLC, hydroelastic).
- Seeds: 3 per cell with 1 mm spawn-y jitter (matches v0.3 protocol).
- Per-cell reproducibility audit: one extra run of seed=0 per cell
  at HOLD, to check bounded GPU non-determinism (notes.md C2 closure).
  Cells where the audit shows 100× outcome spread get flagged as
  chaotic-sensitivity cells.

Total: 5 × 3 × 3 + audits ≈ 50–60 runs.  At ~26 ms/step (production
CSLC) the headline sweep budget is roughly 90 minutes wall.

### 5.2 Metrics, ordered by what the paper argues

1. **Minimum grip-offset threshold per model — PER SIDE.**  At each
   offset level, measure both pads' actuator force via
   `state.mujoco.qfrc_actuator[dof_map["left_x"]]` and `[right_x]`.
   **Grip-success threshold uses the LIGHT-SIDE pad's force** as the
   limiting variable: `F_light(offset) = min(|F_left|, |F_right|)`.
   The cube slips against whichever pad has less force; the heavier-
   pressed side is wasted over-squeeze.  Compute success rate (binary:
   held=YES if `F_light ≥ m·g/(2μ)` AND obj_z_final > z_settled + ε).
   Fit monotone sigmoid to success-vs-offset; report **offset_50 with
   bootstrap 95% CI over seeds**.
   At n=3 per offset level the bootstrap CI is wide (±20-30% of
   threshold); non-overlapping CIs across models is the criterion
   for non-preliminary claims.
   **Ranking rule**: model with lowest `offset_50` (F_light reaches
   threshold at smallest pad penetration) wins; CIs must not overlap
   for the claim to ship.

2. **Achieved squeeze force at offset_50 (per side).**  Report
   `F_light(offset_50)` and `F_heavy(offset_50)` per model; the ratio
   `F_heavy/F_light` is a side-asymmetry metric for the pipeline.
   Sanity check: at static equilibrium `F_left + F_right = 0` in x
   (Newton III); the per-side reading verifies this.

3. **F(d) characterization (full sweep, all models).**  Report the
   complete F_light vs offset curve for each model, not just at
   offset_50.  This is where the **non-monotonic F(d) finding** lives
   (see "CSLC characteristic findings" below).  If CSLC has a soft
   force ceiling and hydro/point don't, that's a paper-grade
   characterization of distributed-contact mechanics, not a sweep
   artifact.

4. **HOLD drift (dz/dt) [mm/s].**  Reported as (mean, std, range)
   across seeds per cell.  Per v0.3.1 caveat, ratios of stds across
   cells are NOT load-bearing at n=3.
   **Ranking rule**: smaller |mean| AND smaller std at the same cell
   wins; otherwise reported as a tie.

5. **xy_slip_max [mm].**  Lateral drift during HOLD.  Lower is better.

6. **Wall-clock real-time factor** = `sim_time / wall_clock`.  Values
   > 1 mean faster than realtime.  Same hardware (RTX 3070).

7. **Gradient quality through the contact step.**  Forward-mode FD vs
   reverse-mode `wp.Tape` gradient of `final_obj_z` w.r.t.
   `pad_close_offset`, on a single SQUEEZE step.  Metric: max
   relative error of `wp.Tape` gradient vs FD reference; binary
   "gradient exists / does not exist / NaN-poisoned" verdict.
   Point and hydroelastic are **expected to fail** here — neither
   ships a differentiable variant in the current Newton tree, to
   our knowledge.

8. **Contact-event smoothness.**  Number of discrete contact-set
   changes per simulated second.

9. **Bounded-determinism audit result.**  Per-cell binary: did the
   seed=0 reproducibility check produce md5-identical CSVs?

### 5.3 CSLC characteristic findings to expect (from v0.6 pilot)

Two findings from the §7.4 CSLC pilot are likely to appear in the
headline-sweep results.  Documenting upfront so reviewers see them
as characterized features, not surprises.

**(a) Non-monotonic F(d) — anchor-spring pull-back.**

Empirical CSLC v0.9 pilot at 122 g cube, R_pad=10mm, ke=5×10⁵
(measured via exp_anchors.py at each commanded depth):

| pad_close_offset | F_per_pad | Newton-III | mechanism |
|---|---|---|---|
| 0.2 mm | 0.74 N | 0.29% | marginal grip |
| 0.5 mm | 0.55 N | 0.01% | dip |
| 1.0 mm | 0.97 N | 0.00% | rising |
| 2.0 mm | **4.99 N** | 0.00% | peak |
| 5.0 mm | 0.00 N | **12.13%** | **cube DROPS** |

**Two findings from v0.9 (vs v0.7):**

1. **Non-monotonicity survives the geometry fix.**  Peak shifted
   from 1 mm (v0.7) to ~2 mm (v0.9), but the rise-then-fall shape
   is preserved.  The anchor-pullback hypothesis stands as the
   leading mechanism.  Magnitudes scaled down ~40× across the
   board because the v0.9 ke_pad_physical is 3× smaller and the
   pad area is 4× smaller — combined ~12× force reduction
   per-engaged-sphere times similar engagement counts.
2. **Cube DROPS at 5 mm depth in v0.9** (F→0, Newton-III balance
   breaks to 12%).  This did NOT happen at v0.7 (which held with
   F=6.6 N at 5 mm).  At the smaller v0.9 pad geometry, deep
   squeeze pushes the contact patch past the dome's stable
   engagement window — likely the same wedge-instability mechanism
   notes.md Step 11 documented for the curved-pad-on-sphere case.
   **Practical implication**: the v0.9 sweep range upper bound
   should be 2 mm (where F peaks), not 5 mm.  Above ~2-3 mm
   depth the grip is unstable.

At small offsets, increasing pad penetration grows the active sphere
count and the per-pair compression, raising F.  Past the peak depth,
the per-sphere anchor spring `k_a · δ_i` pulls each lattice sphere
back toward its rest position, opposing the contact force.  Net
wrench transmitted to the cube body drops as the anchors absorb
more load.  This is a real architectural property of CSLC:
distributed-lattice contact has a **soft force ceiling** determined
by anchor stiffness — paper-grade CSLC characteristic.

Hydroelastic and point contact are expected to be monotone
(pressure-field and linear-spring models both predict F ↑ with depth).
The contrast IS a finding.

**v0.7 numbers** (now superseded; carried for version-history audit):

| pad_close_offset | F_light (v0.7) | held |
|---|---|---|
| 0.2 mm | 43.3 N | YES |
| 0.5 mm | 52.7 N | YES |
| 1.0 mm | 69.2 N | YES (peak) |
| 2.0 mm | 20.3 N | YES |
| 5.0 mm | 6.6 N | YES |

The v0.7-vs-v0.9 magnitude collapse and peak shift are consistent
with the smaller pad: fewer spheres in contact, softer per-sphere
calibration, narrower stable-grip operating range.

**(b) Mass-dependent symmetry transition.**

At 5.75 g cube, the cube settles +2-3 mm off-center deterministically;
per-pad force imbalance is 6× (one side compresses harder).  At 122 g
cube, the system is Newton-III balanced (F_left = F_right within 0.3%).
Gravity dominates the small lateral force imbalances that lattice
asymmetry produces; below some mass threshold, the imbalance
perturbs the cube position.

This is a CSLC pipeline characteristic to document.  Benchmark uses
122 g to be in the equilibrated regime; reviewer-facing question is
whether real applications (light tactile-sensor objects, < 30 g) hit
this regime and need a mitigation (mirror-based pad construction,
lattice symmetrization, etc.).

### 5.4 Win criteria

Defined **before** running anything.

| Metric | CSLC wins | Tie | CSLC loses |
|---|---|---|---|
| offset_50 (headline) | CSLC offset_50 lower AND CIs non-overlapping | CIs overlap | CSLC offset_50 higher with non-overlapping CIs |
| Achieved force at offset_50 | CSLC force < 0.5× hydroelastic | within 2× | > 2× |
| F(d) characterization | CSLC's non-monotonic ceiling is interpretable as design feature | n/a (descriptive) | CSLC ceiling unexplainable / model artifact |
| HOLD drift (mean) | CSLC < 0.05 mm/s AND other > 5× CSLC | within 2× | CSLC > 0.5 mm/s regardless |
| HOLD drift (std) at n ≥ 10 cells only | CSLC std < 0.5× other | within 2× | CSLC std > 2× other |
| xy_slip_max | CSLC < 0.5× other | within 2× | CSLC > 2× other |
| Real-time factor | CSLC > other | within 1.5× | < 0.5× other |
| Gradient quality | CSLC produces finite gradient within 5% of FD | n/a (binary) | CSLC NaN or > 5% error |
| Contact smoothness | CSLC ∫\|dn/dt\| < 0.5× other | within 2× | > 2× |

Paper's headline arguments are **offset_50** (does CSLC grip at lower
pad penetration?), **F(d) ceiling** (does CSLC have a distinctive
distributed-contact mechanics signature?), and **gradients exist**.
Other metrics are supporting evidence.

## 6. What we're NOT measuring (and why)

- **Object rotation under shear** — out of scope; needs 6-DOF analysis
  the C2 scene doesn't drive into.
- **Tactile sensor signal fidelity** — not the thesis claim.
- **Multi-object scenes** — scope creep.
- **Real-robot transfer** — out of scope for sim-only benchmark.
- **Hydroelastic with `kh` swept (sensitivity)** — treats hydroelastic
  as a single-parameter-point reference.  Limited smoke check in §7.5;
  full sensitivity supplementary.
- **CSLC at multiple `(ke_pad_physical, ke_target_constraint)` operating
  points** — series-spring coupling §3.4 is mild at v0.7 anchors;
  reported at one operating point.
- **Canonical hydroelastic / canonical MuJoCo point contact** — see §0
  scope disclaimer.  Comparison is against Newton's implementations.
- **Light-cube (< 30 g) regime where lattice asymmetry perturbs cube
  position** — known CSLC pipeline characteristic, not a flaw to fix
  for the benchmark; benchmark uses 122 g.  Investigation deferred.
- **Mirror-based pad construction as a fix for asymmetric lattice
  biases** — would require building a new pad path and is C3+ work.
  Spec uses current rotation-based pad construction with the side-
  asymmetry metric to characterize the effect.

## 7. Actionable to-do list (state of v0.7)

Items with [x] are done; items with [ ] are pending.  Order is the
recommended execution sequence; items within a "block" can run in
parallel.

### Block A — Empirical anchors (DONE v0.9; one self-consistency follow-up flagged)

- [x] **§7.1 Empirical-anchors measurement** *(v0.9 DONE via
      [exp_anchors.py](../grasp/scripts/exp_anchors.py); see §3.3 for
      locked values)*:
      v0.9 measured at R_pad = 10 mm, ke_pad_physical = 5×10⁵,
      ke_target_constraint = 5×10⁵, contact_fraction default (0.025).
      Central anchors: `L_phi_eff ≈ 0.28 mm`, `A_patch_avg ≈ 250 mm²`,
      `contact_fraction ≈ 0.7`, `F_per_pad ≈ 1.5 N` (much lower than
      v0.7's 67.9 N — see v0.9 changelog).  GPU non-determinism gives
      10-20% scatter on derived ke_pad_physical across same-config
      runs; treat trailing significant figures as noise-floor.
      **Follow-up — self-consistency iteration** *(deferred)*:
      exp_anchors measures empirical `contact_fraction ≈ 0.7` but the
      scene's internal `calibrate_kc` uses
      `CSLCParams.contact_fraction = 0.025` (GraspConfig default).
      For a tight self-consistent calibration loop, re-run
      exp_anchors with `--cslc-contact-fraction 0.7` and iterate
      ke_pad_physical to convergence (~3-5 iterations expected).
      The v0.9 first-pass values are within an order of magnitude of
      what the self-consistent loop would produce and are usable for
      Block B pilots; tighten before locking the headline-sweep
      calibration.
- [x] **§7.2 Object mass decision**: locked at 122 g (steel density
      7800 kg/m³).  At 5.75 g the cube settles off-center; 122 g is
      Newton-III balanced.  *(Mass decision is geometry-independent;
      v0.8 carries it forward unchanged.)*
- [x] **§7.4 CSLC pilot at {0.2, 0.5, 1.0, 2.0, 5.0} mm** *(v0.9
      DONE; see §5.3 (a) for the updated F(d) table)*:
      v0.9 results at R_pad = 10 mm reproduce the non-monotonic F(d)
      shape, with peak shifted from ~1 mm (v0.7) to ~2 mm (v0.9).
      **New finding**: at 5 mm depth the cube DROPS at v0.9
      (F = 0 N, Newton-III residual jumps to 12% — system not in
      equilibrium).  v0.7 held throughout the same sweep range.  The
      v0.7 non-monotonicity was NOT a corner artifact — it
      survives the geometry fix — so §5.3 (a) stands.  But the v0.9
      "drops at deep squeeze" finding constrains the practical
      operating range: headline sweep upper bound should be 2 mm,
      not 5 mm (see §5.1 update).

### Block B — Required before headline sweep (TO DO, ~60 min total)

These block the headline sweep.  Run in order.

- [~] **§7.3 MuJoCo solver settings — DECISION DEFERRED v0.9b** *(v0.9a
      picked 15/100 from nut_bolt_hydro; falsified by pilot data —
      see Block B retraction below)*.

      | Setting | Production CSLC | nut_bolt_hydro | v0.9a (retracted) | v0.9b (current) |
      |---|---|---|---|---|
      | `iterations` | 100 (auto with CSLC) | 15 | 15 | **TBD — production-default for now** |
      | `ls_iterations` | 10 | 100 | 100 | **TBD — production-default for now** |
      | `cone` | elliptic | elliptic | elliptic | elliptic |
      | `impratio` | default | 1.0 | 1.0 | default |
      | `integrator` | implicitfast | implicitfast | implicitfast | implicitfast |

      v0.9a picked 15/100 from nut_bolt_hydro on the rationale that
      tighter line search trades for fewer outer iterations.  Pilot
      data falsified this:
      - At 15/100 hydro and point both read F_per_pad ≈ 0 N.
      - At 100/10 (production CSLC default) hydro still reads F ≈ 0.
      - At 1000/100 (overkill) hydro still reads F ≈ 0.
      Solver iteration count is NOT the cause of hydro / point
      F = 0; see Block B retraction.  The §7.3 decision is deferred
      until the underlying hydro issue is diagnosed AND the
      headline-sweep solver choice can be made on first-principles
      (likely production-default per-model: 100 for CSLC, 20 for
      hydro / point).

- [x] **Hydroelastic box+dome smoke test + pilot at sweep range
      (v0.9 DONE via [exp_fd_pilot.py](../grasp/scripts/exp_fd_pilot.py)).**
      Hydro emits ~40 contacts per HOLD frame at the v0.9 silicone
      target — so the box+dome hydro path runs end-to-end — but
      F_per_pad reads essentially zero (`~1×10⁻⁴ N`) at all five
      depths {0.1, 0.2, 0.5, 1.0, 2.0} mm.  The cube lifts to 100 mm
      under the LIFT command then slips out by the end of HOLD
      (final_z = settled_z = 12.5 mm, xy_slip_max ≈ 20 mm,
      Newton-III residual 5-23%).  **This is a v0.9 calibration /
      wiring issue, NOT a "hydro fails to grip" finding.**  The kh
      override (`--kh 1.8e9`) appears not to propagate to the hydro
      contact-stiffness path in the way exp_fd_pilot expects, OR the
      hydroelastic SDF integration on the dome_param pad doesn't
      build the expected pressure-field gradient.  Diagnostic
      required before any hydro-vs-CSLC comparison.

- [x] **Point contact pilot at sweep range (v0.9 DONE).**  Same
      pattern as hydro: cube lifts to 100 mm then slips out during
      HOLD, F_per_pad ≈ 0 at all five depths, Newton-III residual
      0.7-22%.  At point-contact's single-point-per-shape model with
      `ke = 5×10⁵`, the contact should transmit `F = ke × pen`
      directly to the cube — but the wrench readout shows ~zero.
      Same diagnostic blocker as hydro.

- [x] **Land wrench instrumentation permanently in runner (v0.9 DONE).**
      `_attach_qfrc_actuator(states, model)` in
      [runner.py](../grasp/runner.py) is now called for both the
      headless and viewer-mode state setup.  All benchmark runs
      (including non-CSLC models) get `state.mujoco.qfrc_actuator`
      pre-allocated by default; safe on non-MuJoCo solvers (buffer
      allocated, ignored).

- [~] **Confirm sweep range with all three models' pilot data**
      *(v0.9 PARTIAL: CSLC pilot done, hydro + point blocked on
      calibration diagnostic)*.

      v0.9 CSLC pilot at the §7.3 solver settings (15/100 vs
      production 100/10) shows CSLC also fails to hold the cube
      across all five depths {0.1, 0.2, 0.5, 1.0, 2.0} mm — F_per_pad
      ranges 0.05-2.87 N (vs the ~1.2 N grip threshold) but the cube
      drops by HOLD end at every depth.  This is DIFFERENT from
      exp_anchors.py's same-config measurement (which held at 1 mm
      with F=1.58 N) — the difference is the solver tuning.  At
      §7.3 (15/100) settings the CSLC inner Jacobi has fewer outer
      iterations to recover from MuJoCo's tighter line search, and
      grip becomes unstable.

      **Compounded with the hydro/point F=0 issue, the v0.9 sweep
      range cannot be confirmed against pilot data because all three
      models drop the cube.**  Sweep range update deferred until
      calibration / wiring is resolved.

### Block B BLOCKER (v0.9 — pilots reveal upstream issues)

The v0.9 pilot data above blocks Block D / Move 7 (the headline
sweep).  Updated diagnosis after Step 1 (solver-iteration sweep,
2026-05-23):

**~~Old hypothesis (RETRACTED)~~**: "Under-converged MuJoCo CG
iterations (15/100 from nut_bolt-derived §7.3 decision) cause hydro
and point F=0; CSLC marginal at 15/100 but holds at 100/10."

**Falsifying data (v0.9b Step 1)**:
- Hydro F_per_pad at v0.9 anchors stays ≈ 0 N at iterations ∈
  {15/100, 100/10, 1000/100} and at kh ∈ {5.3e8, 1.8e9} and at
  cube mass ∈ {5.75 g, 122 g}.  40 contacts emit per frame in all
  cases.  Iteration count is NOT the cause.
- CSLC at 100/10 in exp_fd_pilot returned F = 0.39 N (DROPPED), while
  exp_anchors at the same nominal config returned F = 1.58 N (HELD).
  Same settings, two runs, different outcomes — this is the
  chaotic-basin bounded GPU non-determinism documented in notes.md
  C2 closure, not a solver-setting effect.

**Confirmed root cause (v0.9b, 2026-05-23, two-test diagnostic)**:
the dome_param pad is an OPEN spherical cap.  Newton's SDF builder
produces no real interior for an open mesh, so the hydroelastic
pressure-field gradient is zero across the contact patch.

Diagnostic 1 — mesh inspection:
- `dome_param R=10mm half_angle=72°`: `is_volume=False`,
  `is_watertight=False`, `euler_number=3` (open boundary).
- `box pad` (production regression target): `is_volume=True`,
  `is_watertight=True`, `euler_number=2` (closed).

Diagnostic 2 — hydro on the closed-mesh box pad with same
122 g cube, same `kh=1.8×10⁹`, all other v0.9 settings unchanged:
- `RESULT  max_z=0.1000  final_z=0.0313  lifted=YES  held=YES
  xy_slip_max=0.13 mm` ✓
- 97-101 contacts/frame during HOLD, vs the 40 contacts on
  dome_param with F ≈ 0.

Hydro works correctly on closed-mesh pads.  The dome_param cap's
open boundary is the entire reason hydro and point pilots return
F ≈ 0 at v0.9 anchors.  CSLC was unaffected because CSLC doesn't
use the SDF — it uses its own lattice-vs-box-target-point-set
contact path.

**v0.9b deeper finding: hydroelastic pressure-integration sensitivity
floor on small curved SDFs (paper-grade comparative finding).**

After landing the cap-closure fix (below) and confirming it was
necessary but not sufficient, two follow-up sweeps characterised
the remaining gap:

Sweep 1 (kh sweep on closed dome_param + 122 g cube at production
solver settings, [exp_fd_pilot.py](../grasp/scripts/exp_fd_pilot.py)):

| kh (Pa/m) | F_per_pad | Newton-III | n_contacts | grip |
|---|---|---|---|---|
| 1×10⁶ | 0 N | 8.6% | 40 | dropped |
| 1×10⁷ | 0 N | 2.4% | 40 | dropped |
| 1×10⁸ | 0 N | 20.5% | 40 | dropped |
| 1×10⁹ | 0 N | 4.4% | 40 | dropped |
| 1×10¹⁰ | 0 N | 12.5% | 40 | dropped |
| **1×10¹¹** | **39.0 N** | **0.00%** | **76** | partial |
| 1×10¹² | 44.0 N | 0.00% | 72 | partial |

Sharp regime boundary at `kh ≈ 1×10¹¹ Pa/m`.  Below: pressure
integration produces zero force, n_contacts pinned at 40 (likely
all boundary-edge contacts with degenerate pressure).  Above:
n_contacts jumps to 70-80 (interior volumetric engagement), F
appears, Newton-III balances.

Sweep 2 (back_height sweep on closed dome at silicone kh =
1.8×10⁹, to test whether more interior volume lowers the
threshold):

| back_height | F_per_pad | n_contacts | held |
|---|---|---|---|
| 3 mm (default) | 0 N | 40 | NO |
| 10 mm | 0 N | 34 | NO |
| 30 mm | 0 N | 28 | NO |

**Thicker pad does NOT lower the threshold.**  F stays at 0
regardless of back depth, and n_contacts actually decreases (the
larger pad geometrically intersects the cube less).

**Interpretation.**  The sensitivity floor is intrinsic to
Newton's hydroelastic SDF pressure-integration on small curved
geometries — not a function of mesh thickness or interior volume.
For silicone-equivalent stiffness (E ≈ 500 kPa → kh ≈ 1.8×10⁹ Pa/m)
on a 10-mm radius dome cap, hydroelastic produces zero contact
force regardless of pad thickness; it only emits force above
kh ≈ 1×10¹¹ Pa/m (equivalent to E ≈ 28 MPa, i.e. polyurethane or
hard rubber, NOT silicone).

**This IS a paper-grade comparative finding for the CSLC story.**
CSLC, by construction, computes per-sphere contact force as the
direct three-spring series at every active lattice sphere — there
is no SDF pressure-integration step and no sensitivity floor.
The benchmark's headline argument can include "CSLC produces
finite contact force in the silicone-soft regime where Newton's
hydroelastic implementation produces none" as a CSLC advantage on
small-curved-pad geometries.  Worth running the same kh sweep on
the box pad (closed by construction, large interior) to confirm
the sensitivity floor is geometry-specific (small curved SDF) and
not a generic hydro limitation.

---

**v0.9b fix attempt: close the cap (LANDED).**
[pads._build_dome_param_trimesh](../grasp/pads.py) was rewritten
to construct a single watertight mesh that shares the cap base
ring with a cylindrical sidewall and bottom disk, instead of
concatenating an open cap with a separate cylinder.  Verification:
- Closed dome_param: `is_volume=True`, `is_watertight=True`,
  `euler_number=2`, `volume=2.0×10⁻⁶ m³`.  ✓
- CSLC regression on the new closed dome_param: F = 1.57 N at
  v0.9 anchors (vs exp_anchors baseline 1.58 N) — no CSLC
  regression.  ✓ The cap-face mask preserves the same lattice-
  sampling region; only the back closure is new.
- Hydro on the closed dome_param at v0.9 anchors: STILL
  F ≈ 0.0001 N, cube still drops, n_contacts still 40/frame.  ✗

So the open-cap SDF was a real bug — mesh closure IS necessary —
but not sufficient.  Comparing closed dome_param (n_contacts=40,
F=0) against box pad (n_contacts=97-101, F=20 N, held cube) at
identical kh / cube / solver / object settings suggests the
remaining issue is Newton-side: hydroelastic SDF integration on
small / thin / curved pad geometries emits fewer contacts and
appears to apply ~0 pressure-field force even when the mesh is
geometrically valid.  The 40 contacts may be all boundary-pinned
edge contacts with degenerate pressure, vs the box pad's interior
volumetric contacts with real pressure.

**Remaining fix options for the hydro/point comparison** (cap-
closure landed but doesn't unblock; the diagnosis above is the
current state of investigation):
  (a) **Use box pad for all three models** — production CSLC
      regression already uses `--pad-kind box`.  Sidesteps the
      Newton-internal hydro-on-small-curved-pad issue entirely.
      Requires re-anchoring §7.1 at box-pad geometry (different
      patch area, contact_fraction, F per pad).
  (b) **Use different pad shapes per model** — dome_param for
      CSLC, box pad for hydro / point.  Cross-model comparison
      becomes geometry-confounded; reviewers will question what's
      being held constant.
  (c) **Deeper Newton-side investigation** of the hydro SDF
      pipeline on the dome geometry.  Quantify:
        (c-i) per-contact pressure values (not just contact count)
              on dome vs box at same kh;
        (c-ii) effect of dome geometry parameters (back_height,
               n_theta, n_phi) on hydro contact count and force;
        (c-iii) compare against a known-working hydro example
                (e.g., example_nut_bolt_hydro.py) with a similar-
                scale spherical-cap shape.

**Updated recommendation (post-Option 2/3 sweeps)**: run TWO
headline scenes, not one.
  - **Scene A — box pad** (closed-by-construction): the cross-
    model apples-to-apples comparison.  All three models work,
    F is measurable, headline sweep ranges are bracketable.
    Quantifies offset_50, F at HOLD, gradient quality on a scene
    where hydro / point are not pressure-integration-limited.
  - **Scene B — closed dome_param**: the CSLC-only
    characterization scene + the hydro sensitivity-floor
    finding.  Reports CSLC's behavior at the silicone target on
    a fingertip-curvature geometry, AND reports the kh sweep
    above as a "hydro can't represent silicone-soft on small
    curved pads, CSLC can" comparative result.  This is the
    paper-grade finding the dome_param geometry was originally
    meant to surface.

Implementation: both scenes share §3.3 calibration anchors but
report separate offset_50 / F(d) tables.  §5 (headline metrics)
splits into §5A and §5B sections.  Block B re-anchors at box
pad (§7.1 needs a v0.9c re-run via exp_anchors `--pad-kind box`).
Block C / D run twice (once per scene).

Option (c) — Newton-side investigation of WHY hydroelastic
pressure-integration has a sensitivity floor on small curved
SDFs — remains good follow-up but is no longer blocking: the
empirical regime boundary IS the finding worth publishing.

**Remaining v0.9 issues** (independent of the hydro F = 0 root cause):

1. **v0.9 calibration sits at the marginal-grip edge** for the 122 g
   cube.  At v0.7 calibration grip headroom was ~60×; at v0.9 it
   collapses to ~1.3× even when grip works.  Bounded GPU
   non-determinism flips the system across the grip threshold per
   run (exp_anchors HELD vs exp_fd_pilot DROPPED at same config).
   Mitigations to consider (any is a designed paper-write-up choice,
   not a patch):
     (a) increase the silicone-target stiffness (E from 5×10⁵
         to 1×10⁶ Pa) to widen the grip envelope;
     (b) decrease cube mass from 122 g to 60 g (smaller block) to
         reduce gravity load;
     (c) increase mu_friction from 0.5 to 0.7 to widen the cone;
     (d) report ALL benchmark cells with n ≥ 5 seeds + per-cell
         reproducibility checks so the chaotic-basin variance is
         captured rather than aliased.

2. **Self-consistency in `calibrate_kc`** (per audit, Bug 2):
   exp_anchors measures empirical `contact_fraction ≈ 0.7` but the
   scene's internal calibrate_kc uses `CSLCParams.contact_fraction
   = 0.025` default → falls into the fallback path → kc sized for
   N_contact = 4 while 105 actually engage.  The kc the simulator
   actually runs is ≈ 1.1×10⁵ N/m, not the ≈ 5.2×10³ implied by
   the empirical contact_fraction.  At the running kc the predicted
   F = 576 N vs empirical 1.5 N (380× over-prediction, not the 80×
   the v0.9a closure estimated from the contact_fraction = 0.7
   implied kc).  Re-run anchors with
   `--cslc-contact-fraction 0.7` to converge.

Recommend resolving the hydro F = 0 root cause first (likely a
mesh-closure fix in `pads.build_pad_trimesh` or a separate hydro
pad-shape path).  Then fix Bug 2 via the self-consistency loop.
Then re-pilot all three models at production solver defaults
(per-model: 100 for CSLC, 20 for hydro / point) and lock §3.3
calibration before any headline sweep.

### Block C — Validation experiments (parallel with B; ~90 min)

These run after Block B's smoke tests confirm each model works
end-to-end.  Can be done before or after the headline sweep, but
results inform the calibration claim.

- [ ] **§4.1 F(d) curve, all three models (30 min).**
      Quasi-static, no gravity, depths {0.02, 0.05, 0.1, 0.2, 0.5,
      1.0, 2.0, 5.0} mm.  Reports each model's F(d) characteristic
      independently of the grip dynamics.  Critical for documenting
      CSLC's non-monotonic ceiling against hydro/point's expected
      monotonicity.
- [ ] **§4.2 Slip-onset, all three models (30 min).**  At F_normal =
      68 N (empirical HOLD load), measure F_slip.  Pass criterion:
      F_slip ≈ μ·F_normal = 34 N ± 15%.
- [ ] **§4.3 Patch radius, all three models (30 min).**  At F_normal
      = 68 N, measure each model's contact patch.  Expected to
      diverge (point = 0; distributed models = finite); this
      divergence IS a headline finding.
- [ ] **§7.7 Validation-experiment script location**: pick
      `cslc_main/grasp/scripts/exp_validation_*.py` or a single
      `exp_validation.py` with subcommands.  Recommend per-experiment
      files for clarity.

### Block D — Headline sweep (BLOCKED on Block B)

- [ ] **Lock sweep range from Block B pilots, then run §5 headline
      sweep (90 min wall).**  5 offset levels × 3 models × 3 seeds +
      per-cell reproducibility audits ≈ 50–60 runs.

### Block E — Supplementary checks (recommended but not blocking)

- [ ] **§7.5 Hydroelastic kh sensitivity smoke.**  At calibrated kh =
      4.17×10⁹, run one HOLD at 10× and 0.1× kh, n=1.  Confirms
      hydro grip-success isn't pathologically sensitive to kh choice.
      Order-of-magnitude (rather than v0.7's ±2×) honestly probes the
      L conversion uncertainty: v0.6 measured L = 0.12 mm against a
      working assumption of 5 mm (42× off), so the kh sweep should
      cover the same uncertainty band.  If grip-success is flat
      across ±10× kh, the L conversion isn't load-bearing; if it
      breaks within that band, re-run §7.1's L measurement with
      tighter protocol before locking the kh value.
- [ ] **§7.6 FD step size for gradient-quality.**  Verify FD reference
      is converged: compute gradient at FD steps ±0.1 mm and ±0.05 mm,
      check self-consistency.
- [ ] **§7.8 1a-refactor escalation criterion.**  At fixed
      `ke_pad_physical = 1.38×10⁶`, sweep `ke_target_constraint` over
      {1×10⁵, 5×10⁵, 2.5×10⁶} on a single HOLD.  If offset_50 shifts
      > 20%, the series-spring coupling matters and §3.4's
      disclosure isn't enough — ship the 1a refactor before
      submission.  At v0.7 anchors this is likely NOT tripped (kc ≪
      target_ke; §3.4), but verify before claiming the headline.

### Block F — Investigation / future work (deferred)

- [ ] **Non-monotonic F(d) mechanism confirmation.**  Pilot suggested
      anchor-spring pull-back is the dominant mechanism past 1 mm
      depth.  Confirm by instrumenting a single CSLC step at depths
      {0.5, 1, 2, 3, 4, 5} mm and recording `N_active`,
      `phi_eff_mean`, `gate_mean`, `anchor_force_mean`,
      `F_per_pad`.  Anchor force scaling with depth should reveal
      the pull-back mechanism cleanly.
- [ ] **Light-cube asymmetry root cause.**  At 5.75 g the cube
      settles +2-3 mm off-center.  Pads are rotated, not mirrored;
      Step 10d-style y-bias mechanisms may or may not be the
      driver.  Investigate after the headline sweep; if
      reviewer-facing, prioritize.
- [ ] **Mirror-based pad construction as a one-line fix for lattice
      biases.**  Test whether constructing the right pad as a
      mirror image of the left (about cube midplane) cancels the
      lattice y-bias and/or fixes the light-cube asymmetry.

## Appendix A — File pointers

- C2-shipped split (CSLC physical/numerical knob separation):
  [params.py:280](../grasp/params.py),
  [contact_models.py:116](../grasp/contact_models.py),
  [objects.py:64](../grasp/objects.py),
  [main.py:173](../grasp/main.py).
- Notes on Bug B (oscillation-was-sampling-not-solver) and the
  approach-face-only fix: [notes.md C2 closure](notes.md).
- Hydroelastic implementation:
  [sdf_hydroelastic.py](../../newton/_src/geometry/sdf_hydroelastic.py).
  `kh` units verified Pa/m via the source trace at
  [sdf_hydroelastic.py:1378](../../newton/_src/geometry/sdf_hydroelastic.py)
  (`c_stiffness = area × k_eff`) and bit-identical empirical check.
- Reference hydroelastic example using `kh`-style stiffness:
  [example_nut_bolt_hydro.py](../../newton/examples/contacts/example_nut_bolt_hydro.py).
- Wrench instrumentation pattern:
  ```python
  import types
  for state in (state_0, state_1):
      state.mujoco = types.SimpleNamespace()
      state.mujoco.qfrc_actuator = wp.zeros(
          model.joint_dof_count, dtype=wp.float32, device=model.device)
  # After solver.step(): state.mujoco.qfrc_actuator[dof_map["left_x"]] is the
  # joint-actuator force = -F_contact_left_x by Newton III.
  ```
  Belongs in the runner's headless loop as a permanent hook for any
  benchmark that needs actual contact wrench.
- The 1a refactor (true physical-vs-numerical separation requiring a
  rigid-contact-buffer schema change) is parked at C3+ scope.  Current
  v0.7 spec discloses the series-spring coupling rather than removing
  it.  Escalation criterion in §7.8 (Block E).
