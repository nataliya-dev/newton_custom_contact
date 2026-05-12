# CSLC Validation Plan — Math → Kernel → Indenter → FEM → Grasp

**Status:** Plan
**Scope:** Establish CSLC as a principled Galerkin discretization of elastic-skin contact, verified against FEM ground truth, with each tier producing a publishable figure or table.
**Anchor claim:** CSLC's lattice physics is the discrete graph-Laplacian approximation of the elastic-skin PDE; mode truncation error is controlled by sphere count; convergence to FEM is provable.

---

## 0. Why a tiered plan

Jumping straight to FEM is fragile — if the lattice math is wrong, FEM comparison fails for opaque reasons. The ladder isolates one assumption per tier so failures land at a known step:

| Tier | Isolates | Tooling | Cost |
|---|---|---|---|
| T0 | Lattice math (K, eigenstructure, modal response) | numpy | 1 day |
| T1 | CSLC kernel correctness vs numpy reference | Newton (no contact) | 1 day |
| T2 | Single-indenter force and pressure profile | Newton CSLC | 3 days |
| T2.5 | Same scene vs FEM | FEniCS + Newton | 1–2 weeks |
| T3 | Macroscopic grasp consequences | Newton CSLC + squeeze_test | 3 days |
| T4 | Real-world (optional) | Pressure film or tactile sensor | Hardware-dependent |

Each tier consumes the previous tier's output. If T0 mode shapes are wrong, T1 fails; if T1 kernel is wrong, T2 fails; etc.

---

## Tier 0 — Modal mathematics, pure numpy

### Scientific claim
The CSLC lattice stiffness matrix `K_ii = k_a + k_ℓ|N(i)|`, `K_ij = -k_ℓ if j ∈ N(i)` is the graph-Laplacian discretization of the continuous Helmholtz operator `k_a·I − k_ℓ·∇²` on the contact surface. Its eigenvectors are the deformation modes of the discrete elastic skin.

### Hypotheses
- **H0.1** The first six eigenvectors of K on a uniform N×N grid pad correspond to: bulk compression, x-gradient, y-gradient, xy-saddle, x²−y², and second-order asymmetric modes.
- **H0.2** Under a localized unit load at node i, the response δ = K⁻¹·eᵢ decays radially from i with a characteristic length `ℓ_c = √(k_ℓ / k_a)`. This is the elastic-skin's natural correlation length.
- **H0.3** Modal energy under realistic non-uniform loads concentrates in the first 5–10 modes, with the tail decaying faster than `1/k²`.

### Method
- Build K for an N×N grid (N = 15) as a sparse scipy matrix, with `k_a = 15000`, `k_ℓ = 500` (matching squeeze_test calibration).
- `eigs, vecs = scipy.linalg.eigh(K.toarray())`. Render the first 6 vecs as heatmaps on the grid.
- For H0.2: solve `K δ = eᵢ` for the centre node i. Fit `‖δ(r)‖ = A·exp(-r/ℓ_c)` to the response; verify `ℓ_c ≈ √(k_ℓ/k_a)` within 10%.
- For H0.3: apply three load patterns (uniform, linear gradient, twist), decompose `f = Σ cₖ·vₖ`, plot |cₖ|² vs k.

### Success criteria
- Mode 0 is uniform (variance < 1% of mean).
- Modes 1–2 are orthogonal linear gradients (correlation with x and y axes > 0.99).
- Mode 3 is a saddle (positive on one diagonal, negative on the other).
- ℓ_c matches √(k_ℓ/k_a) within 10%.
- Top 5 modes carry > 95% of energy for the three test loads.

### Deliverable
- `cslc_v1/validation/t0_modal_analysis.py` (standalone numpy script, no Newton dep)
- Figure: 6-panel mode-shape gallery + impulse-response radial profile + modal energy bar chart
- This figure goes in the paper as **the lateral-coupling explanation figure** — the impulse-response panel alone shows what hydroelastic cannot capture (no lateral coupling → no spread).

### Why this matters scientifically
Mode shapes are the dictionary the rest of the paper speaks in. Bending, twist, and load-spreading are not metaphors — they are specific eigenvectors with computable wavelengths. The Newton paper has no analog: hydroelastic has no operator whose eigenvectors are meaningful.

---

## Tier 1 — Kernel sanity vs numpy reference

### Scientific claim
The Newton CSLC kernel (`jacobi_step` iterative path and `lattice_solve_equilibrium` closed-form path) computes the same `δ` as numpy's `K⁻¹ f` to floating-point precision.

### Hypotheses
- **H1.1** With contact forcing zeroed and arbitrary node loads applied directly, both kernel paths converge to the numpy reference within 1e-5 relative error.
- **H1.2** Warm-start from previous-step δ reduces iteration count by ≥ 2× compared to cold start, for sequential loads that change smoothly.

### Method
- Build a CSLC pad via `create_pad_for_box_face`, identical topology to T0's numpy K.
- Synthetically inject a per-sphere forcing vector by setting `phi_target` directly (bypassing collision detection).
- Run `lattice_solve_equilibrium` and `jacobi_step` (n_iter sweep: 5, 10, 20, 40); record `‖δ_kernel − δ_numpy‖ / ‖δ_numpy‖`.
- For H1.2: run two consecutive scenes with small phi perturbation, measure residual at each iteration with cold vs warm start.

### Success criteria
- Both kernel paths achieve < 1e-5 relative error at n_iter ≥ 20 (Jacobi) or single-shot (closed form).
- Warm-start convergence: residual at iteration 5 is ≥ 2× lower than cold start.

### Deliverable
- `cslc_v1/validation/t1_kernel_sanity.py`
- Table: relative error vs n_iter for both solver paths
- This tier is a regression-test foundation. Failures here block everything downstream.

### Open question
The closed-form `A_inv = (K + kc·I)⁻¹` assumes saturated active contact (Σ_ε ≈ 1 everywhere). The paper's full formulation uses per-sphere gates `Σ_ε(φ^rest − δ)`. T1 should measure the error between the saturated-limit closed form and the iterative (per-sphere gated) solve under partial-contact scenes. If the error is > 5% at gate ≈ 0.5, the closed-form path is unfit for differentiation through contact onset and should be disabled.

---

## Tier 2 — Single-indenter robotics scenario

### Scientific claim
For a rigid hemispherical indenter pressed into a CSLC pad, the force-vs-penetration curve and the pressure distribution under the indenter match physical expectations for an elastic skin layer: roughly Hertzian (`F ∝ δ^{3/2}` at moderate depth), with Gaussian-decaying pressure radially around the contact axis.

### Hypotheses
- **H2.1** At small δ (single sphere in contact), the response is linear: `F ≈ k_eff · δ` where `k_eff = k_a·k_c/(k_a+k_c)`.
- **H2.2** As δ grows and the active patch expands, the curve transitions toward Hertzian scaling `F ∝ δ^{3/2}`.
- **H2.3** The radial pressure profile under the indenter is approximately Gaussian with width determined by the lattice spacing and `ℓ_c = √(k_ℓ/k_a)`. Hydroelastic's profile would be sharply bounded by the indenter footprint with no Gaussian decay outside the footprint.
- **H2.4** Convergence in N (sphere count): doubling N decreases the residual between CSLC and the asymptotic curve by approximately 2× (first-order convergence for graph-Laplacian Galerkin).

### Method

**Scene:** Flat CSLC pad (face-lattice, 100 mm × 100 mm), rigid hemispherical indenter (radius R = 10 mm) above the pad's centre, descended by a kinematic joint along the pad normal.

Sweep parameters:
- δ ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 4.0} mm
- N ∈ {8×8, 12×12, 16×16, 20×20, 25×25} pad sphere counts (keep pad size fixed; vary spacing)

For each (δ, N) record:
- Aggregate normal force F (sum over surface spheres' f_n,i)
- Per-sphere normal force f_n,i and position (x_i, y_i)
- Active-contact count

Run three contact models on the same scene:
1. CSLC (this work)
2. Point contact (single point at indenter axis)
3. Hydroelastic (Newton's existing HydroelasticSDF on the same indenter and pad geometry, kh tuned to match macroscopic k_e)

### Success criteria
- H2.1: linear regime F-vs-δ slope matches `k_eff` within 5% at δ < 0.5 mm and N ≥ 12.
- H2.2: log-log slope of F-vs-δ transitions from ~1.0 to ~1.5 across the sweep; the transition δ matches the geometric estimate `δ_transition ≈ spacing²/R` within a factor of 2.
- H2.3: CSLC's radial profile p(r) has FWHM at least 1.5× the indenter footprint at δ = 1 mm. Hydroelastic's profile has FWHM ≤ 1.1× the footprint.
- H2.4: error vs asymptote (largest-N CSLC result) at fixed δ decreases as N is refined; fitted convergence order > 0.8.

### Deliverable
- `cslc_v1/validation/t2_indenter.py`
- Figure A: F-vs-δ on log-log axes, three models, with Hertzian asymptote line
- Figure B: pressure profile p(r) at δ = 1 mm, three models overlaid; **this is the lateral-coupling-vs-hydroelastic comparison figure**
- Figure C: convergence-with-N plot

### Why this tier is the keystone
T2 is where CSLC's physics differentiates from hydroelastic empirically. Figure B in particular is the **"hydroelastic is a fluid layer, CSLC is an elastic skin"** demonstration: the radial pressure spread outside the indenter footprint exists in real elastomers and in CSLC, but cannot exist in hydroelastic by construction.

---

## Tier 2.5 — FEM ground truth

### Scientific claim
CSLC's force-vs-penetration curve and pressure profile from T2 converge to the FEM solution of the elastic-skin contact problem as N → ∞, at the first-order rate predicted by graph-Laplacian Galerkin theory.

### Hypotheses
- **H2.5.1** Pointwise pressure error `‖p_CSLC(r) − p_FEM(r)‖_∞ / max(p_FEM)` decreases as `O(h)` where `h` is sphere spacing.
- **H2.5.2** Integrated force error `|F_CSLC − F_FEM| / F_FEM` decreases as `O(h²)` (superconvergence of integral quantities).
- **H2.5.3** With matched material parameters (E, ν, pad thickness), CSLC's `k_a` and `k_ℓ` map to the FEM elastic moduli via a derivable relation:
  - `k_a ≈ (E · A_sphere) / t_pad` where t_pad is the pad thickness
  - `k_ℓ ≈ G · A_sphere / spacing` where G is the shear modulus
  - These mappings should be verifiable to within 30% from the convergence study.

### FEM toolchain — FEniCS
**Choice:** [FEniCS (v0.7+)](https://fenicsproject.org/) — free, Python-native, well-suited for small elasticity problems.

**Alternative:** Drake's tet-mesh hydroelastic with the volumetric pressure field as ground truth. Easier to integrate with Newton scenes but Drake's discretization is not standard linear elasticity FEM — it's a finite-volume pressure-field model. **Don't use Drake here**; you'd be comparing one discretization against another, not against PDE ground truth.

**Geometry:** Same as T2 — 100×100×3 mm elastic layer (E = 1 MPa, ν = 0.3), rigid hemispherical indenter (R = 10 mm) descended kinematically.

**Mesh:** Tetrahedral, characteristic element size 0.5 mm, refined to 0.2 mm under the indenter (verify mesh-independence by halving element size, error must change by < 5%).

**Contact formulation:** FEniCS contact with `mu = 0.5` Coulomb friction. Outputs:
- `σ_nn(x, y, 0)` on the top surface — normal pressure field
- `δ_z(x, y, 0)` — vertical displacement
- F = ∫ σ_nn dA — aggregate force

**Validation of FEM:** Compare FEM at small δ against analytical Hertzian half-space contact `F = (4/3)·E*·√R·δ^{3/2}` where `E* = E/(1−ν²)`. Must match within 3% at δ = 0.1 mm to certify the FEM as ground truth before using it to validate CSLC.

### Method
1. Run FEM at δ ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 4.0} mm. Save (F, p(r), δ_z(x,y)) per case.
2. For each δ, run CSLC at N = {8×8, 12×12, 16×16, 20×20, 25×25, 30×30}.
3. Interpolate CSLC's per-sphere pressure onto the FEM grid via Voronoi area weighting:
   `p_CSLC,i ≈ f_n,i / A_voronoi(i)`
4. Compute the four error metrics per (δ, N) combination:
   - `e_force = |F_CSLC − F_FEM| / F_FEM`
   - `e_pressure_L2 = ‖p_CSLC − p_FEM‖_L2 / ‖p_FEM‖_L2`
   - `e_pressure_Linf = max|p_CSLC − p_FEM| / max(p_FEM)`
   - `e_profile_radial = ‖p_CSLC(r) − p_FEM(r)‖_L2 / ‖p_FEM(r)‖_L2` after radial averaging

5. Plot error vs h on log-log axes. Fit slope; verify convergence orders match H2.5.1 and H2.5.2.

### Success criteria
- FEM validates against Hertz analytic to < 3% at small δ.
- CSLC convergence to FEM: at N = 30×30, all four error metrics are < 10%.
- Convergence orders: force error slope ≥ 1.5 (consistent with O(h²)); pressure L∞ error slope ≥ 0.8 (consistent with O(h)).
- Material-parameter mapping (H2.5.3) recovered to within 30% by fitting `k_a` and `k_ℓ` to the convergence-extrapolated CSLC results.

### Deliverable
- `cslc_v1/validation/t2_5_fem_comparison.py`
- FEniCS scripts in `cslc_v1/validation/fem/`
- Convergence table: rows = N, columns = (e_force, e_pressure_L2, e_pressure_Linf)
- Figure: error vs h log-log plot with theoretical slopes overlaid
- **This is the scientific anchor of the paper.** Newton's hydroelastic paper has no equivalent — they do not show convergence to PDE ground truth.

### Why this matters
This tier proves CSLC is not a heuristic. It is a discrete Galerkin approximation of a well-posed PDE, with computable error bounds tied to mode truncation. The paper can claim: **"CSLC's contact model has the same theoretical standing as FEM, with reduced computational cost proportional to mode truncation."** That is a much stronger scientific statement than slip-creep numbers.

### Risk and mitigation
- **Risk:** FEniCS contact formulations differ. Use FEniCS-X penalty contact if standard contact is unavailable; verify against Hertz before trusting.
- **Risk:** Pad thickness affects the result substantially. Either fix t_pad as a known parameter and report the mapping H2.5.3, or treat the pad as a half-space and skip the thickness-dependent claim.
- **Risk:** ν dependence not captured by CSLC (no Poisson coupling). The paper limitations section explicitly cites this — confirm experimentally that error grows with ν > 0.4 and report as known limitation.

---

## Tier 3 — Grasp consequences

### Scientific claim
The validated lattice physics from T2.5 produces macroscopic grasp behaviour (bending stiffness, torsional stiffness, slip threshold) that differs from point contact and hydroelastic in predictable, measurable ways tied to the mode structure from T0.

### Hypotheses
- **H3.1** Bending stiffness `K_bend = ∂τ_y/∂θ_y` of a two-pad grasp on a flat object scales as `K_bend ∝ k_a + k_ℓ · λ_1` where `λ_1` is the first non-trivial Laplacian eigenvalue from T0. Point contact: K_bend determined only by patch geometry. Hydroelastic: K_bend determined by patch geometry alone (no lateral coupling contribution).
- **H3.2** Torsional stiffness `K_twist = ∂τ_z/∂θ_z` scales as `K_twist ∝ k_a + k_ℓ · λ_2`. Point contact: zero (single line). Hydroelastic: nonzero from patch geometry but smaller than CSLC for the same k_a.
- **H3.3** Under increasing external torque, point contact fails discontinuously (corners unload cascade), hydroelastic degrades gracefully but reaches > 90° tilt before recovery, CSLC degrades most gracefully via load redistribution through `k_ℓ`.

### Method
Reuse `squeeze_test.py` infrastructure. Sweep external torque (τ_y for bending, τ_z for twist) on a held book at fixed grip penetration. Record tilt angle and per-sphere force distribution at HOLD.

For each torque axis, fit `K = τ / θ` in the linear regime. Compare against the T0 prediction `K = (k_a + k_ℓ · λ_k) · L²·N` for the appropriate mode k.

### Success criteria
- Linear-regime K_bend and K_twist match T0 modal predictions within 25%.
- CSLC's tilt-vs-torque curve stays sub-linear across the full sweep where point and hydroelastic do not.
- Per-sphere force distribution under bending visibly activates the linear-gradient mode (Mode 1) identified in T0.

### Deliverable
- `cslc_v1/validation/t3_grasp_consequences.py`
- Three plots: τ_y vs θ_y, τ_z vs θ_z, per-sphere force heatmaps at three torque levels
- Connection back to T0: caption each heatmap with the dominant mode index by projection.

### Why this matters
T3 ties the abstract modal analysis to the practical grasp behaviour roboticists measure. The headline argument: **"Macroscopic grasp stiffness is the lattice's K matrix projected onto rigid-body motion modes."** This connects the paper's contribution to grasp-quality metrics any roboticist already uses.

---

## Tier 4 — Real-world (optional, journal scope)

### Scientific claim
CSLC's per-sphere pressure distribution matches the readout of a physical pressure-sensitive film or tactile sensor under controlled indenter loading, with material parameters calibrated from independent measurements (durometer or stress-strain).

### Path A — Pressure film (~$200 hardware, 1 week)
- FujiFilm Prescale Ultra Low (0.05–0.2 MPa range) or Tactilus matrix sensor
- Press a calibrated hemispherical indenter onto a known elastomer pad with film between
- Scan the developed film, extract pressure profile p(r) by colour calibration
- Compare against CSLC and FEM predictions

### Path B — Tactile sensor (more upside, harder)
- GelSight Mini or DIGIT on a fingertip
- Press the sensor against known indenters, record per-pixel deformation
- Calibrate CSLC parameters (k_a, k_ℓ) from the sensor's known elastomer durometer
- Compare per-pixel CSLC δ_i against per-pixel sensor displacement
- This is **the sim-to-real bridge demonstration** — sensor data and simulator data have the same shape

### Success criteria
- Path A: ‖p_film − p_CSLC‖_L2 / ‖p_film‖_L2 < 25% with parameters fitted independently
- Path B: per-pixel displacement correlation r > 0.85 between sensor and CSLC under three test indenters

### Why this matters
Closes the loop from PDE → discretization → simulation → real measurement. Newton's hydroelastic paper has nothing comparable; their tactile demo (§IV) uses raycasting on the simulated isosurface and is not validated against real sensor data.

---

## Cumulative deliverables and dependencies

```
T0 ──→ T1 ──→ T2 ──→ T2.5 ──→ T3 ──→ T4
        │      │       │        │
        │      └───────┴────────┴──→ Paper figures B, C, D
        │
        └────→ Paper figure A (modal analysis)
```

Each tier produces a self-contained validation script in `cslc_v1/validation/tN_*.py` and a figure or table directly usable in the paper.

## What this plan does NOT cover

- Multi-CSLC contacts (CSLC pad on CSLC pad). Requires H1 implementation first — see `h1_gstack_summary.md`.
- Mesh targets. Requires the uniform-surface-sampling extension.
- Dynamics-dominated regimes (impacts, fast slip). Quasistatic skin assumption may not hold; out of scope for this plan.

## Estimated total effort

| Tier | Person-days |
|---|---|
| T0 | 1 |
| T1 | 1 |
| T2 | 3 |
| T2.5 | 7–10 |
| T3 | 3 |
| T4 (optional) | 5–10 |
| **Sum (T0–T3)** | **15–18 days** |

T0–T3 produces a defensible workshop paper backbone. T2.5 is the journal upgrade. T4 is the sim-to-real claim closure.
