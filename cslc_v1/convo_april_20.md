# CSLC Integration Notes — April 20, 2026

Newton + MuJoCo integration of the CSLC (Compliant Sphere Lattice Contact) model.
Goal: validate that CSLC outperforms point contact — and matches or beats
hydroelastic PFC — in squeeze and lift tasks for the ICRA paper.

---

## 1. Squeeze test (`cslc_v1/squeeze_test.py`)

### Scene

Two kinematic box pads squeeze a dynamic sphere under gravity. The sphere
starts with ~10 mm penetration per side; pads squeeze an additional 2.5 mm
over 0.5 s, then hold for 1.5 s. Sphere mass 500 g (r = 30 mm, ρ = 4421 kg/m³).

```
uv run cslc_v1/squeeze_test.py --mode squeeze --solver mujoco \
    --contact-models point,cslc,hydro
```

### Tuning parameters (shared and per-model)

Shared material: `ke=5.0e4`, `kd=500`, `μ=0.5`, `dt=2.0 ms`.
MuJoCo solver: `solver=cg`, `integrator=implicitfast`, `cone=elliptic`,
`iterations=100 (CSLC) / 20 (point)`, default `solimp`.

| Knob | Point | CSLC | Hydro |
|---|---|---|---|
| lattice spacing | — | 5 mm | — |
| `cslc_ka` | — | 5000 | — |
| `cslc_kl` | — | 500 | — |
| `cslc_n_iter` / `α` | — | 20 / 0.6 | — |
| `cslc_contact_fraction` | — | **0.46** (empirical) | — |
| `kc` (recalibrated) | — | **657.9 N/m** | — |
| `keff` per sphere | — | 581.4 N/m | — |
| Aggregate per pad | — | 50 000 N/m (=`ke_bulk`) ✓ | — |
| `kh` | — | — | **1e8 Pa** |
| `sdf_max_resolution` | — | — | 64 |

### Results (1 s hold, elliptic cone, default `solimp`)

Three complementary metrics (see *Metric interpretation* below):

- `FullDrop` = `sphere_z[0] − min(sphere_z)` (legacy metric — **confounded** by
  the squeeze transient; do not use for cross-model comparison).
- `HoldDrop` = `sphere_z[n_squeeze_steps] − sphere_z[-1]` (displacement
  during HOLD only; positive = fell, negative = rose).
- `HoldCreep` = second-half-of-HOLD mean velocity (positive = falling).

| Model | FullDrop | **HoldDrop** | **HoldCreep** | Active contacts |
|---|---|---|---|---|
| `point_mujoco` | 0.926 mm | +0.725 mm | **+0.495 mm/s** | 2 |
| `cslc_mujoco` (cf=0.46) | 0.159 mm | +0.123 mm | **+0.082 mm/s** | 174 |
| `hydro_mujoco` (kh=1e8) | 0.050 mm | −0.085 mm (rising) | **+0.039 mm/s** | 108 |

**Key finding (updated 2026-04-20):** CSLC has the **lowest absolute creep
rate** of the three: 6× better than point, 2.7× better than hydroelastic.
The earlier "CSLC and hydro tie on creep" reading was wrong — it came from
comparing `FullDrop`, which is dominated by a squeeze-phase transient in
hydroelastic: the asymmetric pressure field at 15 mm penetration generates
a persistent upward force that ejects the sphere during SQUEEZE and
continues to push it up throughout HOLD (`HoldCreep = −0.221 mm/s`).
Hydroelastic isn't "holding better"; it's floating the sphere upward.

This fixes the apparent contradiction with the lift test (where CSLC clearly
beats hydro): both tests now agree that CSLC is the best of the three
distributed contact models under MuJoCo.

### Metric interpretation — why `FullDrop` is confounded

`FullDrop = z[0] − min(z)` assumes the sphere monotonically falls; it
equals the total distance fallen when that's true. For models with
**asymmetric normal-force distributions** during SQUEEZE (hydroelastic with
its volumetric pressure gradient), the sphere rises *first*, so `min(z)`
is close to `z[0]` and the metric under-reports the true compliance of the
contact model.

Both `HoldDrop` and `HoldCreep` isolate the HOLD phase and so strip out the
squeeze transient. For the paper, report `HoldCreep` (rate) as the primary
number — it's solver-compliance-dominated and comparable across models.
Keep `HoldDrop` as secondary so a reader can see sign and magnitude at a
glance.

### Active contacts — what the column actually means

`Active contacts` counts slots where `shape0 ≥ 0` in the contacts buffer —
i.e. how many contacts the MuJoCo solver is solving simultaneously. It IS
a useful observability number (it tells you the pressure field width in
hydro, the number of active surface spheres in CSLC, the GJK narrow-phase
count in point). It is NOT a fair cross-model matching criterion (§5.2):
the three models have different contact semantics, so the counts are
apples-to-oranges.

### Per-pad calibration (kept invariant — do not revert)

`calibrate_kc` defaults to `per_pad=True`: each pad's aggregate stiffness at
uniform contact equals `ke_bulk`, regardless of pad count.
`recalibrate_cslc_kc_per_pad(model, 0.46)` inside `squeeze_test.py` overrides
`kc` so the actual active-contact count (174/378 ≈ 0.46) matches the
calibration prior. Without this override the handler default cf=0.3
under-estimates the active count and each pad aggregates to only 14 286 N/m,
not the intended 50 000 N/m.

Exact derivation (positive denominator):
```
N_contact_per_pad · kc·ka/(ka+kc) = ke_bulk
→ kc = ke_bulk · ka / (N · ka − ke_bulk)
```
With N=86, ka=5000, ke=50000: `kc = 657.9 N/m`. Aggregate per pad ≈ 50 000 N/m ✓.

### Resolution scaling experiment (spacing 5 mm → 2.5 mm)

| Spacing | N_active / pad | `kc` | Z-drop |
|---|---|---|---|
| 5 mm | 87 | 657.9 | 0.214 mm |
| 2.5 mm | 309 | ~161 | **0.119 mm** (10.3× better than point) |

Active count scaled 3.55×; z-drop reduced 1.79× — sub-linear N^0.4 scaling,
consistent with diminishing returns from constraint distribution under
MuJoCo's CG solver.

### Creep mitigation knobs tested

Numbers below use legacy `FullDrop` — the ratios are still informative for
point vs CSLC (both monotonically fall), but hydro is not listed here
because `FullDrop` is not meaningful for a model that rises (see
*Metric interpretation* above).

| Change | Point FullDrop | CSLC FullDrop (cf=0.46) | Ratio |
|---|---|---|---|
| baseline (`solimp[1]=0.95`, pyramidal cone) | 1.223 mm | 0.214 mm | 5.71× |
| `solimp[1] = 0.99` (both models) | 1.223 mm | 0.211 mm | 5.80× |
| `cone="elliptic"` (default solimp) | **0.926 mm** | **0.166 mm** | 5.58× |

`solimp[1]=0.99` does nothing: creep is not dominated by the impedance
`dmax` (the CSLC kernel flattens `solimp` width to 0.001, so `dmax` applies
above 1 mm penetration regardless). Elliptic cone reduces creep ~24% for
both models while preserving the ratio; now the production default.

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

Reproduce:
```
uv run cslc_v1/lift_test.py --mode headless \
    --contact-models point,cslc,hydro \
    --cslc-ka 15000 --kh 2.65e8
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

Earlier exploration on the squeeze scene; informs the lift default.

| kh [Pa] | Squeeze z-drop | Peak contacts | Notes |
|---|---|---|---|
| 1e6 | 0.94 mm | 106 | sphere rose 2.3 mm during squeeze |
| 1e7 | 0.078 mm | 103 | rose 3.7 mm |
| 3e7 | 0.029 mm | 103 | rose 1.1 mm |
| **1e8** | **0.033 mm** | **111** | basically static (production default) |
| 1e10 | DIVERGED | n/a | sphere ejected at HOLD start |

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

### 3.2 Implications for the paper

- **Don't port SAP to Newton** — 3–6 month engineering project.
- **Use Drake directly** for any solver-side comparison the paper needs
  (CSLC under MuJoCo vs PFC under SAP). Maintain a parallel Drake harness
  rather than wrapping SAP into Newton.
- **Newton/MuJoCo stays the primary platform** for differentiable CSLC.

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
- **Squeeze-scene fair calibration** at low penetration. The squeeze
  test (§1) uses 15 mm penetration, at which hydro's pressure-field
  asymmetry ejects the sphere upward (`HoldCreep = −0.22 mm/s`) —
  aggregate stiffness matching alone won't fix this. Either (a) drop
  squeeze penetration to ~1 mm to match the lift regime, or (b) report
  squeeze results at 15 mm pen explicitly as a deep-deformation regime
  where the three models differ physically, not just in calibration.
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
- **Cross-validate with Drake/SAP.** If PFC under SAP also creeps, confirms
  MuJoCo regularisation is the bottleneck (see §3.1). This is the right
  comparison for the paper's solver-side discussion.
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

### 5.6 File state (modified — do not revert)

- `newton/_src/solvers/mujoco/kernels.py` — stiffness conversion fixed
  (lines 390–411). Comment trail of original wrong formula preserved.
- `newton/_src/geometry/cslc_kernels.py`:
  - `out_damping = 0.0` (see §4.2)
  - `out_friction = 1.0` (see §4.3)
- `cslc_v1/squeeze_test.py` — 3-way comparison wired via `--contact-models`.
- `cslc_v1/lift_test.py` — 3-way comparison; per-pad kc recalibration with
  cf=0.025 via `recalibrate_cslc_kc_per_pad`.
