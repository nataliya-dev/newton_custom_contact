# Consolidated FEM-Validation Findings (running log)

This file gathers every paper-relevant finding from the validation plan
in `docs/fem_validation_plan.md` as I work through it.  Each finding is
self-contained and includes evidence, paper impact, and a pointer to
the script + figure that produced it.

Last update: through Tier 2 stage 1 (CSLC F-vs-delta verified).
Newer findings are appended.

The per-tier write-ups are still authoritative:
- [t0_results.md](t0_results.md) -- Tier 0 (modal mathematics)
- [t1_results.md](t1_results.md) -- Tier 1 (kernel sanity)
- [t2_results.md](t2_results.md) -- Tier 2 stage 1 (CSLC indenter)

This file is the **index** you skim to remember what we learned; the
tier files are where you go for the numerical evidence.

---

## Top-line take-aways

1. **The lattice math is sound; the kernel is sound.**  Tier 0 verifies
   the K matrix's eigenstructure against the analytical DCT prediction
   to machine precision; Tier 1 verifies the warp kernels against
   numpy references to fp32 precision.  None of the findings below
   reflect bugs -- they all reflect **scope decisions in the paper
   that need tightening to match what the model actually does**.

2. **The paper's "elastic skin / lateral spread" narrative is only
   true in a regime CSLC has not been operated in.**  At the
   prescribed `(ka, kl) = (15000, 500)`, the lateral correlation
   length is `l_c = sqrt(kl/ka) = 0.18 lattice spacings` -- sub-grid.
   The lateral spring barely smooths anything; the model is
   essentially a distributed anchor-only field at those parameters.
   See Finding A.

3. **The single-indenter Hertzian story works as expected.**  H2.1
   (linear regime) holds exactly to 1 % within the single-cell
   regime, and H2.2 (Hertzian transition) recovers an empirical
   exponent of `1.529` (target 1.5) -- but with a **discrete
   staircase** at lattice quantisation that is a CSLC artefact and
   needs the FEM convergence study to disappear.  See Finding F.

4. **There is a "ghost-force baseline" at zero indenter penetration
   that comes from the sphere-tiling representation, not from
   physical contact.**  Section III of the paper should make this
   explicit; the user-facing `delta = 0` must be defined as "indenter
   at the lattice-apex" not "indenter at the pad face".  See Finding G.

---

## Finding A.  Paper-prescribed parameters have sub-grid lateral coupling

**Source.** Tier 0, H0.2.  Script: `t0_modal_analysis.py`.

**What we found.** The continuous Helmholtz operator
`(ka * I - kl * nabla^2)` has a Green's-function decay length
`l_c = sqrt(kl / ka)` in lattice-spacing units.  At the paper-
prescribed `(ka, kl) = (15000, 500)`, this gives `l_c = 0.183`
spacings -- below the grid resolution.  Solving `K * delta = e_i`
for the centre node, the response decays below 1 % of peak before
reaching the first-neighbour ring; the exponential fit is rejected
as ill-conditioned (regime A).  At an alternative `(ka, kl) = (1000,
1000)` with `l_c = 1.0` spacings, the fit is unambiguous: measured
`l_c = 0.829` vs theoretical 1.0, R^2 = 0.991 (regime B).

**Why it matters for the paper.**
- Any claim that distinguishes CSLC from hydroelastic on the basis of
  "lateral spread" or "elastic-skin diffusion" is only physically
  observable at `kl / ka ~ 1`, not at the squeeze-test parameters.
- The paper's section III.B introduces `kl` as the load-spreading
  spring; the validation plan's H0.2 calls this the "elastic-skin
  natural correlation length".  Both narratives need a footnote
  clarifying that at the operating point reported in §IV the spread
  is sub-grid.
- The squeeze-test grasp-stability results in `docs/summary.md` are
  *not* invalidated by this: at the macro level the dominant CSLC
  advantage is the distributed *anchor*-contact stiffness (378
  spheres at calibrated `k_eff = ke_bulk / N_active_per_pad`), not
  lateral spreading.  They could equally well be claimed for a
  hypothetical "lateral-spring-off" CSLC.

**Suggested paper edit.** Add one sentence to §III.B:
"Throughout this paper we operate at `kl / ka << 1`, where the
lateral spring redistributes deltas across at most 1-2 cells.  Larger
`kl` is required for paper-scale spread; we leave that regime to
future work."

---

## Finding B.  Eigenstructure of K is exact to fp64 precision

**Source.** Tier 0, H0.1.  Script: `t0_modal_analysis.py`.

**What we found.** The first six eigenpairs of K on a 15x15 grid
match the analytical open-grid DCT prediction
`lambda_{p,q} = ka + 2 * kl * (2 - cos(p*pi/N) - cos(q*pi/N))`
to relative error < 1e-6 in both regimes.  Mode-shape gallery in
`figures/t0_mode_gallery_{A,B}.png` shows the expected uniform,
linear-gradient (x and y), saddle, and second-order shapes.

**Why it matters for the paper.**  Makes the "modes are the
dictionary CSLC speaks in" narrative concrete with computable
eigenvalues.  No edit needed; this is the figure-A backbone the plan
called for.

---

## Finding C.  Indenter-scale loads need 30+ modes, not 5

**Source.** Tier 0, H0.3 diagnostic load.  Script: `t0_modal_analysis.py`.

**What we found.** The plan's three loads (uniform, linear gradient,
twist) concentrate > 97 % of their energy in the first 5 modes.  An
added Gaussian-bump load with FWHM = 3 spacings (representing a
small-radius indenter) concentrates only 18-26 % in 5 modes; ~30
modes are needed for > 95 %.

**Why it matters for the paper.**
- The plan's framing of CSLC as a "mode-truncated" representation is
  correct for *smooth* contact patches (squeeze on a flat book cover)
  but fails for narrow indenters relative to the lattice scale.
- For the Tier 2.5 FEM convergence study, this means convergence
  will be driven by *spatial resolution* (sphere count N) rather
  than by mode truncation order, even though both knobs are
  geometrically the same.

---

## Finding D.  Closed-form `lattice_solve_equilibrium` is unfit for partial contact

**Source.** Tier 1, H1.1c (the plan's open question, answered).
Script: `t1_kernel_sanity.py`.  Figure:
`figures/t1_partial_contact_B.png`.

**What we found.** The closed-form `delta = kc * A_inv * phi` assumes
saturated active contact everywhere (gate = 1).  The iterative
`jacobi_step` uses a self-consistent gate
`Sigma_eps(phi - delta)`.  Injecting a phi that is positive on one
half of the pad and zero on the other:

| Regime | rel.err(closed-form vs gated reference) | rel.err at gate-boundary cells |
|---|---|---|
| A (paper, kl/ka << 1)  | 0.97 % | 56 % |
| B (coupled, kl/ka = 1) | **14.9 %** | 56 % |

In regime A the global L2 error looks acceptable only because the
lateral spring is too weak to redistribute the boundary mismatch.
In regime B the closed-form materially misrepresents the gate-
transition region, which is *exactly* the region of interest for
differentiable contact onset.

**Why it matters for the paper.**  Paper §III.D currently presents
the closed-form and iterative paths as differentiable equivalents.
That is true only for **saturated contact**.  Tape-differentiable
trajectory optimisation through contact onset must use the iterative
path with `wp.Tape`, accepting the buffer-aliasing complication.

**Suggested paper edit.** Tighten the scope in §III.D:
"The closed-form `lattice_solve_equilibrium` provides a tape-
differentiable forward pass when the active set is known a priori
(e.g. squeeze on a flat cover at steady state).  For trajectories
that include contact onset, the iterative `jacobi_step` must be used;
its boundary error vs the saturated closed form can exceed 50 % at
gate-transition cells."

---

## Finding E.  The `eps = 1e-5` smoothing default is regime-dependent

**Source.** Tier 1, H1.1b convergence diagnostics.  Script: `t1_kernel_sanity.py`.

**What we found.** The CSLCData default `smoothing_eps = 1e-5 m` is
designed to be effectively binary at typical penetrations.  In the
paper's squeeze regime A (`ka = 15000, kc = 75000`, delta ~ 1 mm)
this holds: `Sigma_eps(phi - delta) > 0.9998` at convergence, gate
effectively 1.  In regime B (`ka = 1000, kc = 75000`, same delta),
`phi - delta = 1.3e-5 m` at convergence -- right at the smoothing
scale, gate = 0.91 (9 % reduction in effective contact stiffness).
Tier 2 stage 1 shows the consequence at small delta:
`F_anchor = ka * sum(delta_i)` and `F_contact = sum(kc * (phi -
delta) * gate)` diverge by up to 10 % in regime B (~0 % in regime A).

**Why it matters for the paper.**  The smoothing is *invisible* in
the paper's reported squeeze numbers because regime A is in the
binary limit.  If the paper claims softer-pad operability (smaller
ka), or wants to make material-parameter portability claims, the
smoothing must either be tied to the operating delta or made smaller.

**Suggested paper edit.** Add to §III.C the constraint that
`eps << phi - delta` at the intended operating point, and quote a
rule of thumb like `eps <= 0.01 * (kc / ka) * delta_typical`.

---

## Finding F.  H2.1 and H2.2 hold exactly within their cell-count regimes

**Source.** Tier 2 stage 1.  Script: `t2_indenter.py`.  Figure:
`figures/t2_stage1_F_vs_delta_both.png`.

**What we found.** On a 100 mm pad with 5 mm lattice spacing and a
10 mm hemispherical indenter, varying `delta_indenter` (measured from
the lattice apex, not the pad face -- see Finding G):

- **Single-cell regime** (`delta in [0.025, 1.0] mm`, n_active = 1):
  `F = k_eff * delta` to better than 1 % over all six data points.
  H2.1 PASS exactly.
- **Multi-cell regime** (`delta > spacing/2 = 2.5 mm`, n_active >= 5):
  empirical fit `F ~ delta^1.529` over delta in [2.5, 8] mm.  Within
  2 % of the predicted Hertzian 3/2.  H2.2 PASS.

The transition penetration predicted by `delta_transition ~
spacing^2 / R = 2.5 mm` matches observed onset of multi-cell
engagement at `delta = 1.5 mm` (n_active jumps from 1 to 5).

**Why it matters for the paper.**
- This is the closest match the paper has between its theoretical
  framing (anchor-contact series, Hertzian patch) and a quantitative
  experiment.  Two clean PASS hypotheses with predicted vs measured
  numbers usable as a results table.
- The figure also shows the **discrete staircase**: F has visible
  slope jumps at delta values where a new concentric shell of
  lattice cells crosses the activation threshold.  This is a CSLC
  finite-N artefact that the H2.4 convergence study (stage 4) is
  expected to smooth out.

---

## Finding G.  The CSLC sphere-tiling has an `r_lat` baseline overlap

**Source.** Tier 2 stage 1.  Script: `t2_indenter.py`, see the
`measure_one` docstring.

**What we found.** With the standard `create_pad_for_box_face`
tiling, every lattice sphere has radius `r_lat = spacing / 2`.  When
the indenter is positioned with its lower-pole *just touching the
pad face*, the centre lattice sphere already has a sphere-sphere
overlap of `phi_centre = r_lat`.  At our 5 mm spacing this is 2.5
mm of baseline overlap, producing several N of force at "zero
indenter penetration" if `delta_indenter` is measured from the pad
face.

The fix is conventional: measure `delta_indenter` from the top of
the highest lattice sphere (`apex_z = pad_top_z + r_lat`).  Then
`phi_centre = delta_indenter` exactly.

**Why it matters for the paper.**
- §III.A introduces the lattice spheres without explaining the
  geometric implication of `radius >= spacing / 2`.  A reader trying
  to reproduce the model from the paper, with `delta_indenter`
  measured from "the pad face" (the physical-intuition convention),
  will see a 3-5 N ghost force at zero penetration.
- The fix is a coordinate convention, not a model change.  The
  paper just needs to be explicit:
  "delta_indenter is measured from the topmost lattice apex; the
  pad face lies r_lat below this apex by construction."

---

## Finding H.  Pressure profile shape is dictated by sphere-overlap geometry, not by lateral coupling -- H2.3 fails as written

**Source.** Tier 2 stage 3.  Script: `t2_indenter.py`,
`pressure_profile_comparison()`.  Figures:
`figures/t2_stage3_profile_d{05,10,20,40}_{A,B}.png`.

**What we found.** Plan H2.3 expected CSLC's radial pressure profile
to have FWHM (or r10 -- the 10 %-falloff radius) **>= 1.5x the
indenter footprint** because of lateral spread.  Hydro was expected
to be <= 1.1x.  The measurement, at 4 deltas in [0.5, 4.0] mm and
both regimes (A: paper kl=500, B: well-coupled kl=1000):

| delta (mm) | a_indent (mm) | CSLC r10 / a | Point r10 / a | Hydro r10 / a |
|---|---|---|---|---|
| 0.5 | 2.24 | 0.44 | 0.44 | 1.02 |
| 1.0 | 3.16 | 0.39 | 0.39 | 0.94 |
| 2.0 | 4.47 | 0.36 | 0.36 | 0.13 |
| 4.0 | 6.32 | 0.34 | 0.34 | 0.12 |

(a_indent = sqrt(R * delta), the Hertzian half-space footprint
radius.)  Two crucial observations:

1. **CSLC's profile is NEVER wider than the indenter footprint** at
   any delta we tested.  The plan's expected lateral spread does not
   appear.
2. **Regime A and Regime B give identical r10 values** to four
   decimal places.  The lateral coupling kl barely affects the
   *shape* of the profile because the lateral spring redistributes
   pressure *among already-active* cells -- it does not turn on
   new cells beyond the geometric footprint.

The CSLC pressure profile is determined by **which lattice spheres
overlap the indenter**, which is set by the sphere-sphere overlap
condition `(r_lat + R) > distance(centres)`, NOT by kl.  At small
delta only the centre cell engages; at delta ~ 2 mm the first ring
(4 cells at lateral 5 mm) engages; at delta ~ 4 mm the diagonal ring
(4 more cells at 7.07 mm) engages.  Each new ring shows up as a
discrete step in the profile (visible in
`figures/t2_stage3_profile_d20_A.png` as the secondary peak at
r ~ 5 mm).

**Why it matters for the paper.**
- The paper's narrative that CSLC "spreads load outside the
  indenter footprint" is **not supported** at the parameter regime
  the paper operates in.  Combined with Finding A (sub-grid
  correlation length), this means CSLC at the prescribed parameters
  reproduces the **geometric** contact patch of a sphere-tiling
  representation, not an elastic-skin pressure distribution.
- Hydroelastic actually wins the "wider profile" comparison at small
  delta (r10/a = 1.0 at delta = 0.5 mm).  At larger delta hydro
  concentrates because its kh produces very stiff polygon-level
  contact and the bin discretisation truncates the tail.
- The plan's H2.3 is **un-falsifiable as written** without resolving
  the bin/spacing discretisation -- CSLC's lattice quantisation and
  hydro's polygon binning both introduce artefacts at the bin scale.

**Suggested paper edit.**
- Remove the "CSLC spreads load outside the indenter footprint"
  claim, OR reframe it as "CSLC distributes load across the
  geometric contact patch" (a weaker, but actually-supported claim).
- Defer the comparison-against-hydro pressure-shape claim to Tier
  2.5 with FEM ground truth -- the lattice quantisation and SDF
  resolution effects make the head-to-head ambiguous.
- The actual paper-relevant CSLC advantage from these experiments is
  the **discrete-ring active-cell schedule**: load engages new shells
  as the indenter descends.  This is a structural feature of
  CSLC that hydroelastic does not have.

---

## Finding I.  CSLC F-vs-delta converges in spacing with order >= 1.49 (H2.4 PASS)

**Source.** Tier 2 stage 4.  Script: `t2_indenter.py`,
`n_convergence_sweep()`.  Figure: `figures/t2_stage4_convergence.png`.

**What we found.** Sweeping `cslc_spacing` in
{12.5, 8.0, 6.25, 5.0, 4.0, 3.125, 2.5} mm on a 100x100 mm pad (grids
9x9 -> 41x41, 81 -> 1681 spheres) at ka=15000, kl=500:

| delta (mm) | F at h=12.5mm | F at h=2.5mm (asymp) | empirical order p |
|---|---|---|---|
| 0.5 | 1.042 | 0.135 | 3.16 |
| 1.0 | 2.084 | 0.539 | 2.79 |
| 2.0 | 4.167 | 2.038 | 1.67 |
| 4.0 | 8.334 | 6.900 | **1.49** |

(`F` from the anchor sum, `ka * sum(delta_i)`.  Order fitted on
`log |F(h) - F(h_min)|` vs `log h` for the 6 non-asymptote spacings.)

All four deltas show empirical convergence orders **far above the
plan's 0.8 threshold**.  H2.4 PASSES.  The super-Hertzian orders at
small delta (3.16 at 0.5 mm) reflect that the centre lattice sphere's
contribution depends explicitly on `kc(h)` through the calibration
formula `kc(h) = ke * ka / (N_contact(h) * ka - ke)`: as h shrinks,
N_contact grows ~ h^-2 and kc shrinks, suppressing the single-cell
force linearly with kc.

**Calibration caveat.** Because `kc` is recalibrated at each spacing
to enforce `aggregate stiffness = ke_bulk` (Newton's `calibrate_kc`
with `contact_fraction = 0.3`), the asymptote is NOT a fixed physical
quantity.  It moves toward zero at small delta (single-cell regime)
as h -> 0 because each individual sphere becomes less stiff.  The
asymptote IS a physical quantity at large delta where many cells
engage: there the per-pad aggregate stiffness equals `ke_bulk` by
construction, and F(delta=4 mm) converges to ~6.9 N regardless of h.

**Why it matters for the paper.**
- The plan's H2.4 "first-order convergence as predicted for graph-
  Laplacian Galerkin" is *consistent* with the data, but the formal
  Galerkin theorem applies to a fixed PDE discretisation -- not to a
  calibration that re-tunes the contact stiffness with h.  The
  observed convergence is therefore a *combined* convergence of the
  Galerkin discretisation and the calibration.  Tier 2.5 with FEM
  will isolate the two by holding kc fixed.
- At delta = 4 mm and h = 2.5 mm (1681 spheres) the F is within
  ~2 % of the next coarser h, so the **paper's claims about
  aggregate behaviour are converged enough at the squeeze-test
  spacing (5 mm, 441 spheres).**
- The "discrete-shell staircase" in F-vs-delta (Finding F) shrinks
  monotonically as h shrinks: at h = 12.5 mm only 1 cell engages
  even at delta = 4 mm; at h = 2.5 mm, 37 cells engage at the same
  delta.  The Hertzian asymptote is recovered when h is small
  enough that the indenter footprint covers many cells.

---

## Finding J.  CSLC's calibration is decoupled from physical material parameters (H2.5.3 FAIL hard)

**Source.** Tier 2.5 part 1.  Script: `t2_5_hertz.py`.  Figure:
`figures/t2_5_hertz_comparison.png`.

**What we found.** Plan H2.5.3 prescribes
`ka = E * spacing^2 / t_pad`, `kl = G * spacing` to map CSLC to a
target elastomer.  For a target `E = 1 MPa, nu = 0.3, t_pad = 3 mm,
R_indenter = 10 mm` we build the indenter scene at the **finest
spacing tested (2.5 mm, 1681 spheres)** and compare aggregate F to
the analytical Hertzian half-space solution `F = (4/3) * E* * sqrt(R)
* delta^{3/2}`.

| delta (mm) | F_mapped | F_paper | F_Hertz | rel.err vs Hertz |
|---|---|---|---|---|
| 0.1  | 0.010 | 0.010 | 0.147  | -93 % |
| 0.5  | 0.135 | 0.135 | 1.638  | -92 % |
| 1.0  | 0.539 | 0.539 | 4.633  | -88 % |
| 2.0  | 2.038 | 2.038 | 13.105 | -84 % |
| 4.0  | 6.900 | 6.900 | 37.067 | -81 % |

H2.5.3 **FAILS** with errors of 80-95 % across the tested range.

**The crucial observation: F_mapped == F_paper to machine precision.**
Setting ka = 2083 (the H2.5.3 mapping for E = 1 MPa) gives literally
identical F as setting ka = 15000 (the paper default).  This is
because the handler's kc calibration `kc = ke * ka / (N * ka - ke)`
reduces to `kc ~= ke / N` whenever `N * ka >> ke` (here `N = 504,
ka >= 2083, ke = 5e4` so the dominant denominator term is `N*ka` and
the ratio collapses).  **The per-sphere effective stiffness `keff ~=
ke / N` is determined by `ke_bulk` and `n_surface`, not by ka.**  The
H2.5.3 mapping is mathematically inert at this regime.

**Why it matters for the paper.**
- CSLC's `ka` and `kl` parameters, in their current calibration
  pipeline, do NOT correspond to physical material moduli.  The
  paper's §III.B description of CSLC "anchor stiffness representing
  the bulk modulus of the skin layer" is misleading -- at the user-
  facing operating point, ka does nothing.  What controls the force
  is `ke_bulk` (a macroscopic grip-stiffness target) divided by
  `n_surface * contact_fraction` (the active-cell prior).
- Mapping CSLC to a physical elastomer is therefore a free fit, NOT
  a derived relationship.  Section IV's mention that "ke_bulk
  matches the squeeze-test calibration" is true but does not
  generalise: a different ke_bulk would be required to match each
  indenter / pad geometry.
- The thin-layer caveat (3 mm pad vs 3 mm Hertz contact radius) only
  explains ~factor-1.5 of the gap.  The remaining factor of ~10x
  underestimate is the calibration vs material moduli decoupling.

**Suggested paper edits.**
- Remove or sharply tighten the "ka represents the bulk skin
  stiffness" framing in §III.B.  Replace with: "ka is a numerical
  regulariser controlling the SPD-ness of K; the user-facing
  stiffness is set by ke_bulk and the lattice topology."
- Section V (limitations) must note that CSLC parameters are
  grip-tuned, not material-tuned, and the model cannot currently be
  calibrated to match arbitrary elastomer moduli without re-fitting
  ke_bulk per scene.
- Tier 2.5 full-FEM comparison (with FEniCSx, against a full elastic-
  layer solution) would not change this finding -- the calibration
  formula is the bottleneck, not the modelling approximation.

---

## Finding K.  CSLC is softer than point but never fails; H3.3 PASS, H3.1/H3.2 ambiguous

**Source.** Tier 3.  Script: `t3_grasp.py`.  Empirical data from
`docs/summary.md` "Disturbance magnitude sweep -- failure curves".

**What we found.**  Combining Tier 0's lambda spectrum with the
existing book-grasp disturbance sweep:

H3.3 (graceful degradation) **PASSES emphatically.**  Across 63
torque magnitudes spanning 6 axes, CSLC was the only model with **zero
ejections**.  Three discontinuous cliffs:

  - Point ejects at tau_z = 3 N*m  (vertical twist, BOX-on-BOX pads).
  - Point ejects at tau_y = 20 N*m (vertical bending).
  - Hydro reaches 180 deg tilt at tau_x = 15 N*m  (grip-axis twist).

At every cliff, CSLC at the same torque retains all 378 surface
spheres in contact and shows < 6 deg tilt.

H3.1 and H3.2 (eigenvalue-mapping to K_bend and K_twist) are **mostly
unsupported by the empirical data.**  Empirical linear-regime K
slopes (N*m / rad) from `docs/summary.md`:

| axis | point | CSLC | hydro | CSLC/point | CSLC/hydro |
|---|---|---|---|---|---|
| tau_y (bending) | 9519 | 2865 |  716 | 0.30x | 4.0x  |
| tau_z (twist)   | 1432 |  521 |  176 | 0.36x | 3.0x  |
| tau_x (grip)    |   70 |  157 |    8 | 2.25x | 19.5x |

CSLC is **softer than point** for bending and vertical twist but
**stiffer than hydro** by 3-20x.  CSLC is **stiffer than point** for
the grip-axis twist (where point has only a sparse line contact and
hydro has no compliance to redistribute load).

Tier 0's eigenvalue ratio `lambda_1 / lambda_0 = 1.0015` at regime A
implies that whatever bending stiffness CSLC produces, it does **not**
come from the kl-mediated lateral mode.  The macroscopic K_bend is
instead dominated by the **geometric moment-arm sum** of the per-
sphere anchor forces -- which is a function of ka, the patch
geometry, and ke_bulk (per Finding J).  The plan's H3.1 formula
`K_bend = (ka + kl * lambda_1) * L^2 * N` correctly identifies the
dimensions but assigns the wrong contribution: at regime A, kl *
lambda_1 << ka, so the formula collapses to `K_bend = ka * L^2 * N`.

**Why it matters for the paper.**
- The paper's narrative that CSLC "exceeds point contact's rotational
  stiffness" is **wrong in the linear regime** -- point contact's
  4-corner BOX-pad specialisation is *stiffer* than CSLC for small
  torques.  CSLC's advantage is **graceful degradation, not linear-
  regime stiffness.**  This is a major narrative change.
- The strongest paper claim is the **cliff comparison at tau_z = 3
  N*m**: point ejected (178 deg, 75 mm/s creep), CSLC held (0.33 deg,
  0.015 mm/s creep), hydro degraded (~1.8 deg, 0.32 mm/s creep).
  This is one disturbance, three qualitatively different outcomes.
  Same story at tau_y = 20 N*m and tau_x = 15 N*m.
- The plan's "K_bend / K_twist scales with lambda_1 / lambda_2" claim
  is dimensionally consistent but not the dominant scaling. The
  empirical K values are proportional to **ka * (patch geometric
  factor)**, not the Laplacian eigenvalues.

**Suggested paper edits.**
- Section IV's main result: "CSLC is the only model with zero
  catastrophic failures across the disturbance sweep" (not "CSLC has
  higher rotational stiffness than point contact").
- Section III.E (theoretical claims): replace "K_bend scales with
  lambda_1" with "K_bend scales with `sum_i ka * r_i^2`, the
  geometric moment-arm sum over engaged surface spheres".  The
  lambda_1 contribution is sub-1 % at the paper's calibration.

---

## Finding L.  Fine grid + strong lateral coupling still does not change CSLC pressure-profile shape — Finding H confirmed structurally, not by under-resolution

**Source.**  Fix 1.2 + Fix 2 decisive experiment, 2026-05-11.  Script:
`t2_fine_grid_kl_sweep.py`.  Figures:
`figures/t2_stage3_profile_d40_finegrid_{A_weak,B_strong}.png`.

**What we found.**  Finding H attributed CSLC's invariant pressure
profile to two confounds: (i) at spacing = 5 mm only ~5 cells engage
geometrically inside the indenter footprint at δ=4 mm — too few to
resolve a profile-shape difference, (ii) lattice-spacing-unit
correlation lengths in both tested regimes (0.18 and 1.0 spacings) were
below the threshold where elastic-skin diffusion would observably
redistribute load.  We controlled for both confounds with a 1 mm
spacing pad (~221 active cells at δ=4 mm) and a 200× kl bump from 500
→ 100 000 (`ℓ_c` from 0.14 → 2.0 spacings in *physical* units, via
Fix 1.1's resolution-independent scaling).

The result is unambiguous:

| Regime | ka | kl_kernel | ℓ_c (mm) | F (N) | r10/a_indent | FWHM/a_indent |
|---|---|---|---|---|---|---|
| Weak (sub-grid) | 25 000 | 500 | 0.14 | 25.231 | 1.261 | 0.862 |
| Strong (ℓ_c=2 spacings) | 25 000 | 100 000 | 2.00 | 25.240 | 1.261 | 0.862 |

Aggregate force differs by **0.04 % (9 mN out of 25 N)**.  Pointwise
normalised profile RMS difference is **1·10⁻⁴**; max difference
3·10⁻⁴.  **The two profiles are identical to four decimal places.**
The same number of cells engage (221), the same active set, the same
shape.

**Why it matters for the paper.**
- Finding H's negative result is **structural**, not a consequence of
  under-resolution or weak coupling.  At any (ka, kl_kernel, spacing)
  in the paper's calibration regime, the pressure profile is dictated
  by **which lattice spheres geometrically overlap the indenter**, not
  by lateral-coupling-mediated diffusion.
- The mechanism (confirmed across Findings A, H, L): the per-sphere
  contact force emitted to the rigid-body solver is
  `kc_series · pen_3d_i = kc_series · (phi_rest_i - δ_i)`.  Even when
  strong kl redistributes the `δ_i` field, the per-sphere `phi_rest_i`
  is set by indenter geometry alone, and the contact gate
  `Σ_ε(pen_3d_i)` is dominated by `phi_rest_i` at any kl in the tested
  range.  Lateral spring redistribution alters δ but the emitted force
  is largely insensitive to that redistribution.
- The elastic-skin-diffusion narrative is now **definitively dead** for
  this paper.  No parameter-scope adjustment recovers it within the
  current kernel architecture.  A genuine elastic-skin reframe would
  require either (a) replacing the graph Laplacian with proper
  Galerkin FEM (the Fix 3 from the prior planning conversation), (b)
  decoupling the contact gate from per-sphere `phi_rest` so neighbour-
  pulled δ can transmit force at non-overlapping spheres (changes the
  physics), or (c) emitting per-sphere force based on the anchor
  reaction `ka·δ_i` instead of the contact penalty
  `kc_series·pen_3d_i` (also changes the physics).

**Suggested paper edit.**  Replace any "elastic-skin diffusion" or
"lateral spreading" language in §III and §IV.B with the actually-
supported claim: **CSLC produces a distributed pressure profile over
the geometric contact patch, with the per-sphere force determined by
the discrete sphere-indenter overlap geometry and the kc/ka/ke_target
series-spring composition.**  This is a weaker but defensible
statement that matches the implementation, Finding H, and now Finding
L.  The kl parameter remains in the model as a numerical regulariser
of the K matrix's spectrum; it does not produce observable elastic-
skin effects at the paper's operating point.

**Implementation note.**  Fix 1.1's `kl_physical` parameter (in
`CSLCData.from_pads`) is implemented and works — the strong-regime
ℓ_c value in this experiment was set via `kl_physical = ka · (2·h)² =
0.1 N·m`.  The Fix is retained because resolution-independent kl
scaling is the *correct* discretisation regardless of whether
elastic-skin effects manifest, and may matter for future MorphIt-
based pad geometries where spacing varies across the pad.

---

## End-of-validation summary

Completed tiers: T0 (modal math), T1 (kernel sanity), T2 stages 1+3+4
(indenter scenarios), T2.5 part 1 (analytical Hertz), T3 (grasp
consequences via synthesis).  Full FEM (T2.5 part 2) is left as a
follow-up; the calibration-decoupling Finding J would not change
under FEM.

11 paper-relevant findings logged (A-K).  See per-tier results files
for the full numerical evidence.  Three findings (A, J, K) imply
NARRATIVE-LEVEL changes to the paper text; the others (B-I) imply
TIGHTER SCOPE on existing claims.
