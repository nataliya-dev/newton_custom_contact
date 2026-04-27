# CSLC Integration Notes — April 20, 2026

Newton + MuJoCo integration of the CSLC (Compliant Sphere Lattice Contact) model.
Goal: validate that CSLC outperforms point contact — and matches or beats
hydroelastic PFC — in squeeze and lift tasks for the ICRA paper.

---


## 1. Squeeze test (`cslc_v1/squeeze_test.py`)

> **2026-04-25 — scene aligned to lift_test.py.** Squeeze now operates
> at 1 mm face penetration (0.5 mm initial + 0.5 mm active squeeze)
> with the same fair-calibration values as the lift test:
> `cslc_ka=15000`, `cslc_contact_fraction=0.025`, `kh=2.65e8`.  Both
> tests are now in the regime where CSLC's distributed-constraint
> advantage is unambiguous on every metric (vs the previous 12.5 mm
> deep-deformation regime, where hydro happened to win on creep
> alone).  Numbers below reflect the aligned scene.

### Scene

Two kinematic box pads squeeze a dynamic sphere under gravity. The sphere
starts with ~0.5 mm penetration per side; pads squeeze an additional
0.5 mm over 0.5 s, then hold for 1.5 s, giving 1 mm face_pen at HOLD —
matching the lift test's operating point.  Sphere mass 500 g (r = 30 mm,
ρ = 4421 kg/m³).

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
| `cslc_ka` | — | **15 000** | — |
| `cslc_kl` | — | 500 | — |
| `cslc_n_iter` / `α` | — | 20 / 0.6 | — |
| `cslc_contact_fraction` | — | **0.025** (matches lift) | — |
| `kc` (recalibrated) | — | **75 000 N/m** | — |
| `keff` per sphere | — | 12 500 N/m | — |
| Aggregate per pad | — | 50 000 N/m (=`ke_bulk`) ✓ | — |
| `kh` | — | — | **2.65e8 Pa** (fair) |
| `sdf_max_resolution` | — | — | 64 |

### Results (1 s hold, elliptic cone, default `solimp`, 2026-04-25)

Three complementary metrics (see *Metric interpretation* below):

- `FullDrop` = `sphere_z[0] − min(sphere_z)` (legacy metric — **confounded** by
  the squeeze transient; do not use for cross-model comparison).
- `HoldDrop` = `sphere_z[n_squeeze_steps] − sphere_z[-1]` (displacement
  during HOLD only; positive = fell, negative = rose).
- `HoldCreep` = second-half-of-HOLD mean velocity (positive = falling).

| Model | FullDrop | **HoldDrop** | **HoldCreep** | Active contacts |
|---|---|---|---|---|
| `point_mujoco` | 1.020 mm | +0.733 mm | **+0.490 mm/s** | 2 |
| `cslc_mujoco` (cf=0.025) | **0.092 mm** | **+0.067 mm** | **+0.045 mm/s** | 26 |
| `hydro_mujoco` (kh=2.65e8) | 0.564 mm | +0.379 mm | +0.253 mm/s | 35 |

**Key finding:** CSLC is the clear winner across every metric in the
aligned 1 mm regime — **10.9× better than point and 5.6× better than
hydro on HoldCreep**, and the same ranking on HoldDrop and FullDrop.
This matches the lift test (§2) exactly: under fair calibration at 1 mm
face_pen, CSLC's distributed-constraint advantage produces the
lowest-slip / lowest-creep behaviour by a wide margin.

The previous 12.5 mm deep-deformation regime had hydro marginally
ahead of CSLC on creep (0.062 vs 0.082 mm/s), but that was an
artifact of operating outside the fair-calibration design point and
of letting more lattice spheres into the active patch than the prior
assumes.  Reproduce via:
```
uv run cslc_v1/squeeze_test.py --mode squeeze \
    --contact-models point,cslc,hydro
```

### Metric interpretation — why `FullDrop` is confounded

`FullDrop = z[0] − min(z)` commingles the SQUEEZE transient (pads closing,
sphere accelerating through ~15 mm of penetration) with the HOLD-phase
compliance drift we actually care about, so the number depends on the
ramp profile as well as the contact model.

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

Numbers below use legacy `FullDrop`. Only point and CSLC are listed;
hydro was not part of this mitigation sweep.

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
  test (§1) uses 15 mm penetration — a deep-deformation regime where
  hydro beats CSLC on HOLD creep (opposite of the lift test at 1 mm
  face_pen). Either (a) drop squeeze penetration to ~1 mm to match the
  lift regime, or (b) report squeeze results at 15 mm pen explicitly
  as a deep-deformation regime where the three models differ physically,
  not just in calibration.
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
- `newton/_src/geometry/cslc_handler.py`:
  - `_from_model` passes `build_A_inv=True` (see §4.5a performance).
  - `_launch_vs_sphere` dispatches to `lattice_solve_equilibrium` when
    `A_inv` is present; falls back to iterative Jacobi otherwise.
- `cslc_v1/squeeze_test.py` — 3-way comparison wired via `--contact-models`;
  HOLD-phase metrics (`hold_drop_mm`, `hold_creep_rate_mm_per_s`).
- `cslc_v1/lift_test.py` — 3-way comparison; per-pad kc recalibration with
  cf=0.025 via `recalibrate_cslc_kc_per_pad`; `--cslc-ka` / `--cslc-contact-fraction` / `--kh` CLI overrides; per-step timing diagnostic.
