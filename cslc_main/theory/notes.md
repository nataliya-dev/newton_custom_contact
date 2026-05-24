# cslc_main.theory — running notes

A scientific verification of the CSLC compliant-skin contact model,
built up from first principles in pure numpy + scipy.  The point is not
to ship code that runs in production — that's what `newton/_src/geometry/cslc_kernels.py`
is for.  The point is to have a transparent reference where every line
of code traces to a printed equation, and every numerical answer can
be hand-checked against a closed form.  When the Warp kernels diverge
from this theory, we know which one is wrong.

## Big-picture goal

The "ideal" CSLC model uses:

* **Deformed-centre formulation** — the lattice sphere physically
  moves to `q = p - δ`, its radius stays at `r`.  ``jacobi_step`` and
  ``write_cslc_contacts`` both emit at ``q_def = p - δ`` with ``r_lat``
  untouched (Step 7 D2 / D4).

* **Vec3 δ** — displacement has 3 components per sphere, not just a
  scalar along the rest normal.  This is what lets the sphere "slide"
  toward an off-axis contact direction and what lets perpendicular
  shear deform the skin (precondition for friction-via-tangential-δ).

* **Distance-preserving lateral spring** —
  `f_lat(i,j) = -k_l·(‖q_j-q_i‖ - L_ij) · ê_ij` instead of the paper's
  graph-Laplacian `-k_l·(δ_i-δ_j)`.  The former is a real physical
  spring on the deformed centres; the latter is a penalty on the
  displacement-difference vector.  They agree at small δ along each
  edge direction, diverge elsewhere.

* **Series-spring contact** — `f_contact = k_c·(φ_rest - δ_n)` (paper
  eq. 12), the Hookean spring on the contact-layer deformation.

This theory module is the **gold reference** the in-tree kernels are
verified against in [test_07_kernel_bridge.py](test_07_kernel_bridge.py)
(see Step 7).  Step 7 landed; the kernels now obey this model.

## How to read this document

The notes are roughly chronological — each section captures what landed
when, with the most-recent findings at the bottom.  A fresh reader
should know:

* **Steps 1-9** (PASS): the theoretical foundations (single-sphere
  primitives, lattice, friction, anisotropic anchor, dome contact, grip
  budget).  These are the building blocks the kernels are verified
  against.  Skip if you only care about current status.
* **"Where we are now" / "Where we are going"**: the current
  cross-cutting status summary.  Read first if you're orienting.
* **Step 11 — Dome stability investigation**: the wedge-climb mystery
  on dome+sphere grasps and the four-script falsification chain
  (`exp_slow_lift`, `exp_friction_sweep`, `exp_emitted_friction_sweep`,
  `exp_constraint_softening_sweep`) that closed it to a MuJoCo
  regularization artifact.  See "The falsification chain summary"
  subsection.
* **C2 closure**: the n=3 seed-sweep finding that single-seed claims
  on this scene don't replicate, plus the bounded-GPU-non-determinism
  characterization.
* **§7.1 / §7.4 / §3.4**: benchmark-prep measurements (empirical
  anchors, squeeze-depth pilot, series-spring coupling check).  These
  reference [benchmark_spec.md](benchmark_spec.md) and were originally
  locked at v0.6/v0.7; v0.8 of the spec retracted the R_pad = 20 mm
  geometry and the sections are pending re-validation at R_pad = 10 mm.

**Related documents**:
* [benchmark_spec.md](benchmark_spec.md) — the contact-model benchmark
  protocol (point vs CSLC vs hydroelastic).  Currently v0.8.  Reads
  top-to-bottom for the calibration + sweep plan; §7 is the actionable
  to-do list.
* [test_07_kernel_bridge.py](test_07_kernel_bridge.py) — the
  kernel-vs-theory regression harness.  Currently 27/27 PASS.
* `../grasp/scripts/exp_*.py` — the falsification scripts named above,
  plus the in-flight `exp_anchors.py` for empirical-anchors measurement.
* `../../cslc_mujoco/docs/summary.md` — earlier squeeze/lift work on
  box pads with sphere/book objects.  Separate work thread; not
  cross-referenced from here because the scene and pad construction
  differ.

## Sign conventions (keep these handy)

| Quantity | Definition | Sign |
|---|---|---|
| δ | displacement of lattice centre | δ_n > 0 ⇒ sphere compressed INWARD (q on body side of p) |
| q | deformed centre | `q = p - δ` |
| n̂ | rest outward normal | unit, pointing from body into vacuum |
| φ_rest | rest overlap | (r + R) − ‖p − t‖, positive when rest spheres overlap |
| φ_eff | active deformation | max(0, φ_rest − δ_n), the contact-spring squish |
| f_anchor (on q) | restores q toward p | +k_a · δ |
| f_contact (on q) | pushes q away from target | +k_c · φ_eff · (q − t)/‖q − t‖ |
| dE/dδ | gradient of stored energy | same sign as f_anchor for q = p − δ (chain rule absorbs a -1) |

## Step 1 — Single sphere (PASS)

**File:** [test_01_single_sphere.py](test_01_single_sphere.py)

**Math.**  One sphere, one rigid target, no neighbours.  Forces:
`f_anchor + f_contact = 0`.  For face-on contact (target on n̂):

```
δ_n* = k_c · φ_rest / (k_a + k_c)
|F*| = k_a · k_c / (k_a + k_c) · φ_rest = k_eff · φ_rest    (series spring)
```

Series-spring composition `1/k_eff = 1/k_a + 1/k_c` comes from the
**two distinct physical compliances** (bulk tissue + surface contact)
sitting end-to-end between rigid body and rigid target.  Same force
flows through both, deformations add up to `φ_rest`.

**Tests.**
| Part | Test | Pass |
|---|---|---|
| A | Numerical L-BFGS-B on `E_total(δ)` matches closed form to 2e-10 rel err across 15 (φ, k_c) pairs | ✓ |
| B | F* vs k_c saturates at k_a·φ_rest (rigid-contact limit), vanishes at k_c → 0 | ✓ |
| C | F* linear in φ_rest with slope k_eff, residual ~1e-13 N | ✓ |
| D | Off-axis target: δ tilts to follow contact line; ‖δ‖ = 750 µm conserved across angles | ✓ |

**Figures:** `figures/01a_face_on_table.txt`, `figures/01b_series_spring_kc.png`, `figures/01c_force_vs_phi.png`, `figures/01d_off_axis_geometry.png`.


## Step 2 — Chain with lateral coupling (PASS)

**Files:** [cslc_lattice.py](cslc_lattice.py), [test_02_chain.py](test_02_chain.py)

**Math.**  N spheres along x with anchor `k_a`, lateral `k_l`, no contact.
Lattice stiffness matrix (paper eq. 10):

```
K_ii = k_a + k_l · |N(i)|,    K_ij = -k_l  for (i,j) in edges
```

Symmetric, SPD for k_a > 0, tridiagonal for a chain.  Smallest
eigenvalue = k_a (uniform translation, the only mode lateral cannot
resist).

**Two lateral laws compared:**

1. **Graph-Laplacian** (paper): `f_lat(i,j) = -k_l·(δ_i - δ_j)` —
   isotropic in 3D, equivalent energy `(1/2) k_l ‖δ_i - δ_j‖²`.

2. **Distance-preserving** (deformed-centre, ideal):
   `f_lat(i,j) = -k_l·(‖q_j-q_i‖ - L_ij)·ê_ij` —
   linearises around δ = 0 to a rank-1 projector
   `[ê^rest ⊗ ê^rest] · (δ_i - δ_j)` that picks only the rest-edge
   direction component.

**Discrete vs continuum decay length:** the chain's Green's function
decays as `g_i ∝ z_-^|i|` where

```
z_- = (1 + α) - sqrt(α(α+2)),    α = k_a / (2k_l)
ℓ_c^discrete = -1 / ln(z_-)        (in spacings)
```

The continuum approximation `ℓ_c = √(k_l/k_a)` is correct only in the
`k_l >> k_a` limit.  Production CSLC defaults `k_l/k_a = 0.2` give
`ℓ_c^discrete = 0.520` spacings, not the continuum `0.447`.  This is
the validation FINDINGS.md Finding A, rediscovered here from first
principles.

**Tests.**
| Part | Test | Pass |
|---|---|---|
| A | K matrix matches hand-built tridiagonal to machine precision; smallest eigenvalue = k_a | ✓ |
| B | All 20 eigenvalues match DCT-II formula `λ_k = k_a + 2k_l(1 - cos(kπ/N))` to 4e-16 rel err | ✓ |
| C | Green's function fits the DISCRETE decay formula to ≤1% across k_l/k_a ∈ {0.2, 1, 10} | ✓ |
| D | Perpendicular load on chain: graph-Laplacian spreads δ_y to neighbours; distance-preserving leaves it localised at the loaded sphere (`δ_y = F/k_a`, neighbours = 0 to leading order) | ✓ |

**Figures:** `02a_K_matrix.txt`, `02b_eigenvalue_spectrum.png`, `02c_greens_function_axial.png`, `02d_perpendicular_diagnostic.png`.

## Step 3 — Chain + contact at one sphere (PASS)

**File:** [test_03_chain_contact.py](test_03_chain_contact.py)

**Math.**  Add a rigid target above the centre sphere k.  Solve
`(K + k_c·e_k e_k^T)·δ_n = k_c·φ_rest·e_k`.  Sherman-Morrison gives

```
δ_n_k = k_c · φ_rest · g_kk / (1 + k_c · g_kk),    g_kk = (K^{-1})_kk
```

i.e. the single-sphere series-spring formula with `k_a` replaced by
`k_a^eff = 1/g_kk`.  Lateral coupling STIFFENS the centre's response
(neighbours share the load).  For the production chain `k_l/k_a = 0.2`,
`k_a^eff = 1.342·k_a`.

The δ profile away from k follows the same chain Green's function as
step 2: `δ_n_j ∝ z_-^|j-k|`.

By Newton's 3rd law:

```
Σ_i F_i  =  k_a · Σ_i δ_n_i  =  k_c · (φ_rest - δ_n_k) = F_contact
```

The per-sphere force `F_i = k_a · δ_n_i` is the contact-patch
pressure distribution.

**k_l limits:**
* `k_l = 0`: δ_n_k = k_c φ / (k_a + k_c) (isolated, single-sphere step-1 answer).
* `k_l → ∞`: chain becomes rigid; all spheres yield equally; δ → 0, F_total → k_c·φ.

**Distance-preserving on 1D chain + perpendicular contact** is
*quasi-localised*, not the exactly-isolated single-sphere answer I
first predicted:

* Centre δ_k drops to ~498 µm (below isolated 500): the stretched
  lateral spring resists compression at k via its y-component.
* Neighbours yield by ~1.6 µm in the **same direction as k** (no
  outward bulge).
* Both effects are third-order in δ: stretch ≈ δ²/(2h), y-projection
  ≈ δ/h, so force ≈ k_l · δ³ / (2h²).

True outward bulging (negative δ at neighbours) requires 2D or 3D
curvature — neighbours whose rest edges aren't purely transverse to
the contact direction.  The 1D chain doesn't have that.

**Tests.**
| Part | Test | Pass |
|---|---|---|
| A | Closed-form chain contact equilibrium matches Sherman-Morrison to 1e-16; force balance holds to 1.8e-15 N | ✓ |
| B | Sweep k_l/k_a ∈ {0, 0.2, 1, 10, 100}: peak δ_k drops monotonically, FWHM grows from 2 to 20 spacings, F_total grows toward k_c·φ | ✓ |
| C | Distance-pres centre δ matches third-order formula to 5%, neighbour δ to 25%, signs all match | ✓ |

**Figures:** `03a_chain_contact_profile.png`, `03b_kl_sweep_patch.png`, `03c_gl_vs_dp_chain_contact.png`.

## Step 4 — Stick-slip friction on a single sphere (PASS)

**File:** [test_04_friction.py](test_04_friction.py)

**Math.**  Add a tangential external load `F` to the step-1 sphere, plus
a friction spring `k_stick` between q and the contact patch (which is
bonded to the target's surface in stick mode).  Anchor + friction are
**two parallel springs** in the tangent plane.  Two regimes:

```
stick (k_stick * s <= mu * f_n):  s = F / (k_a + k_stick),    F_fric = k_stick * s
slip  (k_stick * s >  mu * f_n):  s = (F - mu * f_n) / k_a,    F_fric = mu * f_n
F_thresh = mu * f_n * (k_a + k_stick) / k_stick               (transition)
```

Limits:
* `k_stick -> 0`: no friction, `F_thresh -> infinity`, system never slips.
* `k_stick -> infinity`: bare Coulomb, `F_thresh -> mu * f_n`.

**Tests.**
| Part | Test | Pass |
|---|---|---|
| A | Stick formula `s = F / (k_a + k_stick)` vs hard piecewise scipy minimize_scalar, machine precision | ✓ |
| B | Slip formula `s = (F - mu * f_n) / k_a` and Coulomb plateau `F_fric = mu * f_n`, to scipy bounded method precision (~1e-8) | ✓ |
| C | Continuous transition at `F_thresh`; slope ratio `(k_a+k_stick)/k_a` exact; analytical == hard-piecewise to picometer level | ✓ |
| D | `k_stick` sweep showing `F_thresh -> mu * f_n` as `k_stick -> infinity` (bare Coulomb limit) | ✓ |

**Smoothing-zone observation.**  The kernel's `harmonic-mean` smooth
surrogate `F(s) = K M s / (K s + M)` has a transition zone of width
`~mu*f_n/k_stick` (about a factor of ten on each side of
`s_thresh`).  Inside that zone the smooth surrogate differs from the
hard law by up to 41% at our default parameters.  Deep stick and deep
slip both recover the hard law exactly.  This is the kernel's price
for C-infinity differentiability; the ideal physics is the hard
piecewise law.

**Figures:** `04c_full_stick_slip.png` (transition diagram with both
hard and smooth curves), `04d_k_stick_sweep.png` (Coulomb limit).

## Step 5 -- Curvature unlocks outward bulging (PASS)

**Files:** [cslc_lattice.py](cslc_lattice.py) (added `make_arc`,
`solve_lattice_contact_linear` general-normal solver),
[test_05_arc_contact.py](test_05_arc_contact.py).

**Geometry.**  An arc of N spheres on a circle of radius `R_pad` with
arc-length spacing `h`:

    p_i = R_pad * (cos theta_i, sin theta_i, 0),    n_i = p_i / R_pad
    delta_theta = h / R_pad        alpha = h / R_pad (angular spacing)

Each sphere's outward normal is radial.  Edges nearest-neighbour
along the arc.  Neighbouring sphere is at curvature drop  `R_pad * alpha^2 / 2`
below the apex in the y-direction (apex on +y).

**Geometric Poisson math.**  Compress the apex by `delta_y` (q_apex
at y = R_pad - delta_y).  Right-neighbour edge:

    v = q_right - q_apex = (R_pad*alpha, delta_y - R_pad*alpha^2/2, 0)
    L_def^2 - L_rest^2 = delta_y * (delta_y - R_pad * alpha^2)

so the spring is **compressed** (L_def < L_rest) in the window

    0  <  delta_y  <  R_pad * alpha^2

A compressed lateral spring pushes endpoints **apart** -- the neighbour
is pushed outward along its own local outward normal, producing
**outward bulge (local delta_n < 0)**.  At `delta_y = R_pad * alpha^2`
the chord recovers its rest length and the bulge crosses zero.
Beyond that, the spring is stretched and we recover same-sign
spreading (the 1D-chain story).

**Graph-Laplacian** has no such window -- it sees only `(delta_i - delta_j)`
and is geometry-blind, spreading load in same-sign neighbours always.

**Tests.**
| Part | Test | Pass |
|---|---|---|
| A | Build the arc; verify radial normals, rest lengths match `2 R_pad sin(delta_theta/2)`, K is symmetric PD with smallest eigenvalue = ka | ✓ (residuals < 1e-12) |
| B | Face-on contact at apex, phi inside bulge window.  Distance-preserving local delta_n is **negative at perimeter** (bulge), graph-Laplacian stays positive | ✓ (DP perim = -33 nm at +1 vs GL = +3.52 um) |
| C | Phi sweep across the bulge window; DP perim delta_n crosses zero near `delta_apex = R_pad*alpha^2`. Measured crossover phi = 199.5 um, expected `2 * R_pad*alpha^2 = 200 um` (apex sinkage is half of phi for kc = ka) | ✓ |
| D | Curvature sweep at fixed phi; bulge magnitude scales with curvature: R_pad = 2.5 mm gives 0.47 um bulge, R_pad = 100 mm gives < 1 nm | ✓ |

**Figures:** `05a_arc_geometry.png` (the arc + radial normals),
`05b_face_on_local_delta.png` (full profile + DP-zoomed perimeter showing the bulge),
`05c_bulge_window_phi_sweep.png` (DP crosses zero near 2*R*alpha^2, GL doesn't),
`05d_curvature_sweep.png` (bulge -> 0 as R -> inf).

**Subtle but important.**  In test_05 we use `kl/ka = 1` (well-coupled,
lc = 1 spacing) for the default scene so the bulge propagates a few
spacings and shows on plots.  Production CSLC uses `kl/ka = 0.2`
(sub-grid lc = 0.45 spacings) -- the bulge is real there too but
concentrated at offset +-1, with magnitude ~10x smaller per the
chain Green's function.  See the bug-#1 finding (continuum vs
discrete decay length) for why.

## Step 6 -- Anisotropic anchor (PASS)

**File:** [test_06_anisotropic_anchor.py](test_06_anisotropic_anchor.py).

**Math.**  The compliant skin is closer to incompressible flesh than
to an isotropic spring.  Per-axis anchor with normal `ka` and tangent
`ka_t = ka * ka_t_ratio`.  For a uniform isotropic solid,

    ka_t / ka  =  1 / (2 (1 + nu))     (nu Poisson;
                                         0.5 -> 1/3 = incompressible).

Affects only TANGENT motion: face-on contact and normal equilibrium
are unchanged because they live entirely on the n_hat axis where ka
dominates.  Three observable consequences:

1. **Off-axis tilt.**  In the local {n_hat, t_hat} frame with target
   along  u = cos(theta) n_hat + sin(theta) t_hat,

       delta_t / delta_n  =  (ka - A) / (ka_t - A) * tan(theta)

   where  A = kc * phi_eff / L  is the "effective contact coupling"
   (~347 N/m at our defaults).  The naive linear form
   (ka/ka_t) tan(theta) drops A and is wrong by O(A/ka_t): at
   ka_t/ka = 1/3 the correction is ~3%, at ka_t/ka = 0.1 it's ~14%.
   Test verifies the EXACT formula self-consistently from the
   numerical equilibrium.

2. **Stick-slip threshold shift.**

       F_thresh  =  mu * f_n * (ka_t + k_stick) / k_stick

   Softer tangent anchor -> earlier slip.  The Coulomb plateau
   mu * f_n is unchanged because f_n depends only on ka (normal).

3. **Stick-mode slope.**

       dF_friction / dF  =  k_stick / (ka_t + k_stick).

   Softer ka_t -> more of the applied force flows directly into
   friction (in the limit ka_t -> 0, all of F becomes friction).

**Tests.**
| Part | Test | Pass |
|---|---|---|
| A | Off-axis delta tilt verified against the EXACT formula  (ka-A)/(ka_t-A) tan(theta)  across ka_t/ka in {1, 1/2, 1/3, 0.1} at theta = 1 deg | ✓ (max rel err 6e-8) |
| B | F_thresh matches closed form for ratios {2, 1, 1/3, 0.1} (bisection on regime flip via the hard-piecewise solver) | ✓ (7e-9) |
| C | Stick slope dF_friction/dF tracks k_stick/(ka_t+k_stick) for ratios spanning 0.05 to 10 | ✓ (5e-16) |
| D | Poisson sweep nu in {0, 0.25, 0.4, 0.5}; visual confirmation of incompressible-flesh limit ka_t = ka/3 | ✓ (visual) |

**Sign-regression guard.**  Cross-check that all three friction
solvers (analytical / hard-piecewise scipy / smooth L-BFGS-B) agree
on the sign of `delta_t` at `ka_t_ratio = 1/3`.  Same discipline as
the step-4 audit: magnitude-only checks miss sign flips.

**Figures:** `06a_off_axis_tilt.png`, `06b_stick_slip_threshold.png`,
`06c_stick_slope.png`, `06d_poisson_sweep.png`.

**Scope note.**  Step 6 stays single-sphere.  `Lattice` does not yet
expose `ka_t_ratio`; once steps 7/8 need anisotropic anchors on
chains or arcs, `anchor_force_all` / `anchor_energy` and the
gradient in `solve_lattice_contact_numerical` will need the per-axis
decomposition that `cslc_theory.anchor_force` already implements.

## Step 8 -- 3D dome contact under a sphere indenter (PASS)

**Files:** [cslc_lattice.py](cslc_lattice.py) (added `make_dome`,
`SphereIndenter`, `solve_lattice_sphere_indenter`),
[test_08_dome_contact.py](test_08_dome_contact.py).

**Motivation.**  Production drops the tennis ball from the dome pad
even though the box pad of the same dimensions holds it (`held=YES,
xy_slip=0.05 mm` in the box regression).  Step 5 proved the 1D arc
bulge analytically, but the production dome is 3D and the static
question -- *what does the dome lattice actually do when a sphere
indents it?* -- had no theory baseline.  Step 8 lifts the arc to a
quasi-uniform Fibonacci-spiral spherical cap (N = 150 spheres on
R_pad = 10 mm at theta_max ~72 deg, matching the production OBJ in
`assets/pad/pad.obj`), and presses a tennis-ball indenter (R_obj =
33.5 mm) into it.

**New machinery in `cslc_lattice.py`:**

* `make_dome(N, R_pad, half_angle, ka, kl, k_neighbors)` -- builds a
  Fibonacci-spiral cap with equal-area-per-sample placement (deterministic
  analog of the production Lloyd / CVT sampler in
  `make_cslc_pad_from_samples`).  Returns `(Lattice, spacing, cap_area)`.
  Spacing = mean nearest-neighbour distance.  Edges via 3D k-NN with
  the apex (densest sample) permuted to index 0 for deterministic
  test driving.  Average degree = 6.72 at k_neighbors = 6 (boundary
  spheres have fewer neighbours; cap interior at ~12 like a closed
  manifold).

* `SphereIndenter(t, R, kc)` -- multi-contact analog of `ContactTarget`.
  No `sphere_idx`: every lattice sphere may engage independently.

* `solve_lattice_sphere_indenter(lat, indenter, r_lat, lateral=..., eps=..., ...)`
  -- L-BFGS-B over the multi-contact energy
  `E_total = E_anchor + E_lateral + sum_i 0.5 * kc * sigma_eps(raw_i)^2`
  with `raw_i = (r_lat[i] + R) - ||q_i - t||`, `q_i = p_i - delta_i`.
  Per-sphere gradient picks up the `smooth_step` factor explicitly
  (theory.txt eq. contact-grad).  Default `eps = 5e-4 m` matches the
  production `CSLCParams.smoothing_eps`; pass `eps ~ 1e-9` for
  closed-form comparisons against the hard `max(0, .)`.

**The four PARTs.**

| Part | What it verifies | Status |
|---|---|---|
| A | Geometry: N = 150, radial outward normals to machine precision, K SPD with smallest eig = ka, cap area = `2 pi R^2 (1 - cos theta_max)` to 2.5e-16, degree distribution in [k, 2k] (boundary effects) | PASS |
| B | Apex sinkage: at phi_apex from 50 um to 3 mm, measure `delta_n` at the apex sphere.  Compare to two predictions: **isolated** `kc phi / (ka + kc)` and **Green's-stiffened** `kc phi / (1/g_00 + kc)`.  Production geometry favours the isolated regime to 0.6 %; Green's is 76 % off | PASS |
| C | Patch flattening + Hertz prediction: at four penetrations, count N_active, measure patch radius, plot 3D scatter + top-down `delta_n` heatmap with Hertz patch overlay.  N_active log-log slope vs phi = 1.020 (predicted 1.0 from `N ~ pi a^2 / cell_area`, `a = sqrt(R_eff phi)`).  F_total log-log slope = 2.02 (deep-sat regime, not Hertz 1.5 -- each engaged sphere is in deep compression) | PASS |
| D | `kl / ka` sweep at fixed phi = 1 mm.  100x change in kl/ka changes apex by 5 %, N_active by 0, F_total by 4 %.  **Production is in the localised regime** where lateral coupling does almost nothing | PASS |

**The diagnostic finding** (Part B + Part D): at production geometry,
the dome lattice acts like 17 independent normal springs.  The apex
sphere's lateral neighbours all compress by similar amounts so
`q_j - q_i` stays close to `p_j - p_i` (the rest configuration), the
distance-preserving spring is unstrained, and each sphere just sees
its own anchor + contact balance.  `kl` is therefore not the knob to
tune for grip; `kc`, `contact_fraction` (which sets the per-pad
N_contact in the calibration), and `mu_friction` are.

**Production-relevant numbers at default kc = 1e4:**

| phi_apex | N_active | F_n^total |
|---|---|---|
| 0.5 mm | 8 | 14.8 N |
| 1.0 mm | 17 | 60.5 N |
| 2.0 mm | 35 | 243 N |
| 3.0 mm | 49 | 549 N |

The 17 active spheres at 1 mm penetration is ~2x the Hertz prediction
of 8.4 -- the rest are at the patch boundary where the indenter
geometrically intersects them but at small `f_n`.

**Figures:** `08a_dome_geometry.png` (3D scatter + normal quivers),
`08b_apex_sinkage.png` (isolated vs Green's-stiffened),
`08c_flattening_3d.png` (3D heatmap + top-down with Hertz overlay),
`08d_kl_sweep.png` (4-panel diagnostic),
`08e_flattening_profile.png` (side-view `(r, z)` cross-section at
phi in {0, 0.5, 1, 2, 3} mm overlaid -- the canonical "dome flattens
under squeeze" picture: each sphere plotted as `(r_radial,
z_axial)` in the apex frame, with the rest cap arc and indenter
circles overlaid).  The dome doesn't go fully flat -- it conforms
to the indenter's shallower curvature (apex zone tracks `R_obj`,
the perimeter stays on the rest `R_pad`).  Active-patch radius
grows ~`sqrt(R_eff * phi)` (Hertz law).

## Step 9 -- Dome grip budget under tangential load (PASS)

**File:** [test_09_dome_grip.py](test_09_dome_grip.py).

**Motivation.**  Step 8 quantified normal load (`F_n^total` as a
function of phi); Step 9 quantifies how much *tangential* force the
patch can resist before slipping -- the actual grip budget.

**Model.**  Three already-verified pieces fit together:

1. **Step 8 normal equilibrium** gives per-sphere `f_n,i = kc *
   phi_eff_i` (heterogeneous across the patch: apex `f_n` is ~70x the
   perimeter `f_n` at phi = 1 mm).
2. **Step 4 hard piecewise stick-slip** runs independently per sphere
   with each sphere's own `f_n,i`.  Sphere `i` flips to slip when
   `k_stick * s > mu * f_n,i`, so the perimeter spheres slip first
   and the apex slips last.
3. **Step 6 anisotropic anchor** lifts the trade between
   force-domain `F_thresh` and displacement-domain `s_first_slip`.

We treat the rigid indenter as imposing a uniform tangential
displacement `s` on every active sphere (the spheres' contact patches
are bonded to the indenter while stuck, so they translate together).
The aggregate grip is
`F_grip(s) = sum_{i active} min(k_stick * s, mu * f_n,i)`.

**The four PARTs.**

| Part | What it verifies | Status |
|---|---|---|
| A | Normal-only baseline: histogram of `f_n,i` at phi = 1 mm.  `F_n^total = 60.5 N`, `mu F_n^total = 18.1 N` against tennis-ball weight `0.57 N` -- 31.9x headroom.  **Static theory says the dome SHOULD grip** -- the production failure mode therefore isn't a fundamental friction budget limit | PASS |
| B | Tangential displacement sweep at phi = 1 mm.  Plateau matches `mu F_n^total` to 0.000 %; low-s slope matches `N_active * k_stick` to 0.000 % (after correcting for stick-only sampling window); first-slip `s` matches `mu f_n,min / k_stick` within one grid step | PASS |
| C | `F_grip^max` vs operating phi.  Slope of `F_n^total vs phi` = 1.96 in log-log, agreeing with Step-8 PART C's deep-sat slope 2.0.  At phi = 0.1 mm grip headroom drops to 0.38x weight -- below the ball's weight; the dome cannot hold the ball at sub-mm penetration | PASS |
| D | Anisotropic anchor sweep: `ka_t_ratio in {1.0, 1/2, 1/3, 0.1}`.  Displacement-domain plateau identical across ratios (set by `k_stick` and `f_n`, not anchor).  Force-domain `F_thresh` shifts by `(k_at + k_stick) / k_stick` -- softer skin engages friction earlier in the force-input form | PASS |

**Action-item finding for the production grasp:**

Initial hypothesis (now disproved): the headroom collapse at small phi
(PART C) means the production failure mode is operating-phi
sensitivity, and bumping `contact_fraction` should fix it.

**Empirical disproof of that hypothesis** (`outputs/grasp/_baseline_cf0p025`
vs `_experiment_cf0p11`).  Two end-to-end dome grasp runs, identical
trajectory, only `contact_fraction` changed.  Both reach
SQUEEZE `max_pen ≈ 1.2-1.4 mm` (more than the 0.5 mm headroom
threshold from PART C).  **Both drop the ball at the start of LIFT.**
`max_pen` collapses from > 1 mm to ~0 within ~50 ms of LIFT
beginning, well before the pad has moved meaningfully.

The right diagnosis (see [`outputs/grasp/dome_grasp_diagnostic.png`](../../outputs/grasp/dome_grasp_diagnostic.png)):
the dome shape itself is geometrically unstable under any vertical
pad motion at this `R_pad / R_obj` ratio.

* Dome cap apex is concentrated at one point along the pad's outward
  normal; the rest of the cap is RECESSED radially.  The active
  contact patch at SQUEEZE is ~3.85 mm wide (Step 8 PART C), centred
  on the apex, hitting the ball's *equator*.
* The instant LIFT moves the pad up, the apex moves above the
  equator.  The ball's horizontal radius shrinks with `z` away from
  the equator (sphere geometry).  The recessed region behind the
  apex never reaches the narrower ball -- phi drops to zero.
* A box pad doesn't have this failure because its contact face is
  flat in z; vertical pad motion preserves contact area.  This is
  why box-pad regression passes and dome-pad does not.

This is a **geometric / design** issue, not a CSLC tuning or
friction-budget issue.  The CSLC contact model is doing exactly what
it should; the dome shape simply has insufficient vertical contact
extent for the LIFT trajectory.

Possible fixes, ranked by leverage:

1. **Pad geometry (asset change only, no kernel work).**  Less
   curved dome (R_pad ~ 33 mm matching the ball, so the cap's apex
   region stays close to the ball through more vertical motion), or
   a taller cap (larger half-angle so the protruding region has more
   vertical extent).
2. **Trajectory change (controller only).**  Lift while still
   squeezing inward, so the pads track the ball's narrowing
   horizontal radius as they move up the ball's surface.  Or drop
   LIFT entirely for dome-grasp validation -- approach + squeeze +
   hold only.
3. **Algorithm extension (real new theory).**  A CSLC variant where
   the lateral spring also generates an in-plane "wrap-around"
   tension, letting the dome actively conform to the ball rather
   than just locally indent.  This is the genuine extension that
   would let curved CSLC fingertips outperform hydroelastic on
   smaller objects.

The original line in this notes file, *"dome pad's hold failure is
a separate calibration issue (object slips out even at fix-point
delta, independent of Step 7)"*, was a **wrong diagnosis**.  Step 9
empirics replaced it with the geometric-instability finding above.

**Figures:** `09a_normal_distribution.png` (per-sphere f_n histogram
with min / mean / max markers), `09b_grip_curve.png` (grip(s) with
plateau and first/last slip markers), `09c_grip_vs_phi.png` (log-log,
weight reference, Hertz and deep-sat slope guides),
`09d_anisotropic_sweep.png` (4 ratios overlaid).

## Bugs encountered + fixes

1. **Continuum vs discrete ℓ_c (step 2 Part C, first run).**  Compared
   the Green's function fit to `√(k_l/k_a)` and got 16% error at
   sub-grid `k_l/k_a = 0.2`.  Fit was right; the reference was the
   continuum approximation rather than the exact discrete formula.
   Fix: added `chain_discrete_decay_length` to `cslc_lattice.py`, used
   it as the reference, error dropped to ≤1%.  **Lesson:** when
   comparing numerics on a discrete lattice to a "well-known" formula,
   check whether that formula is the discrete or continuum form.

2. **Gradient sign mismatch (step 3 Part C, first run).**  Two helpers
   in `cslc_lattice.py` had opposite sign conventions:
   `lateral_force_graph_laplacian` returned `-dV/dδ` (physical force),
   `lateral_force_distance_preserving` returned `+dV/dδ` (gradient).
   The L-BFGS-B `jac` callable subtracted both, producing a wrong-sign
   gradient for distance-preserving.  Step 2 Part D missed it because
   the equilibrium happened to sit at near-zero lateral gradient.
   Step 3 surfaced it as 9% finite-difference vs analytical
   gradient disagreement, and the solver bailed with "ABNORMAL"
   termination.  Fix: rewrote the `jac` inline with explicit `dE/dδ`
   terms; FD agreement now 4e-10.  **Lesson:** any time two helpers
   could disagree on sign convention, write the gradient inline.  Add
   a finite-difference vs analytical check as a unit test.

3. **Initial Part C prediction was naive (step 3 first cleanup).**
   Predicted distance-pres on chain + perpendicular contact would be
   exactly the isolated answer (zero neighbour response).  Correct to
   *linear* order, but the third-order correction is observable
   (`~kl·δ³/(2h²)`).  After deriving the third-order formula by hand,
   numerical matched it to 5% (centre) and 18% (neighbour).

4. **Missing `smooth_step` factor in contact gradient (post-step-4 audit).**
   In `cslc_theory.equilibrium_numerical.jac`, `equilibrium_with_friction_smooth_numerical.jac`,
   and `cslc_lattice.solve_lattice_contact_numerical.jac`, the contact
   contribution to the gradient was written as `kc * phi_eff * e_hat`
   (or `kc * phi_eff * n_k` for the face-on lattice case).  That's the
   spring **force**, not the gradient of the smoothed energy.  The
   energy `E_contact = (1/2) kc phi_eff^2` with `phi_eff = sigma_eps(raw)`
   picks up a chain-rule factor:

       dE_contact / d delta = kc * phi_eff * smooth_step(raw, eps) * e_hat

   where `smooth_step = sigma_eps'(raw) = 0.5 (1 + raw/sqrt(raw^2 + eps^2))`.

   At `raw >> eps` (deep saturated contact), `smooth_step ≈ 1` and the
   factor is invisible.  At `raw ~ eps` (contact onset / margin), the
   factor swings between 0 and 1 and the gradient is up to **2x wrong**.

   All four existing tests (steps 1-4) operate in deep saturated contact
   (`raw / eps >= 10^4`), so the bug was masked.  A direct FD check at
   `raw / eps = 0.5` showed `|jac - FD| / |FD| = 22.4` (262% error in
   the buggy form) collapsing to `~8e-9` after the fix.

   Fix: factored out `smooth_step` and `contact_raw_overlap` helpers in
   `cslc_theory.py` and multiplied them through every smoothed contact
   jac.  All four tests re-pass.  **Lesson:** when the energy uses a
   smooth surrogate, the "spring force" magnitude is NOT the gradient
   of the energy -- the chain rule contributes a Heaviside-surrogate
   factor that is unity only in the deep-saturated regime.

6. **Sign error in `equilibrium_with_friction_smooth_numerical` and
   incomplete `k_stick = 0` guard.**  Second-pass audit found the
   smooth-friction L-BFGS-B was returning `delta_t` with the WRONG
   SIGN -- previous step-4 tests checked only `|delta_t|` so the bug
   was invisible there but would have surfaced as soon as any vector
   comparison happened downstream (step 7 kernel bridge, step 8 grasp
   integration).

   Root cause: with `q = p - delta`, the displacement of q from rest
   is `(q - p) = -delta`, so an external force `f_ext` does work
   `W = f_ext . (-delta) = -f_ext . delta`, and its potential is
   `V_ext = -W = +f_ext . delta`.  The code had  `E_ext = -f_ext . delta`
   (sign flipped) and the corresponding `g -= f_ext` in the jac.
   Both flipped:

       E_ext = +float(np.dot(f_ext_tangent, d))         # was -
       g = g + f_ext_tangent                            # was -

   Verification: with `F_ext = (+1, 0, 0)` and stick mode, analytical
   and hard-piecewise scipy both return `delta_x = -2.0e-5`; smooth
   numerical previously returned `+2.13e-5`, now returns `-2.13e-5`
   (magnitude discount is the smoothing zone, sign is correct).

   Same audit-pass also caught an incomplete `k_stick = 0` guard in
   the PRODUCTION analytical solver: `F_thresh = mu*f_n*(ka+k_stick)/max(k_stick, 1e-30)`
   returned `~3.75e30 N` instead of `+inf`, and the regime was
   advertised as "stick" instead of "no_friction".  Replaced with an
   explicit branch (`if k_stick > 0 and mu > 0` else `inf`,
   `regime = "no_friction"`).  This makes `info["F_thresh"]` semantically
   honest and lets downstream callers test `is_finite(F_thresh)` for
   the "system can slip" check.

   Test_04 Parts A and C now carry explicit sign-regression guards:
   for `F_ext` in +x they assert `delta_x < 0` (per the `q = p - delta`
   convention).  Magnitude-only checks miss sign flips; vector checks
   don't.

   **Lesson:** test pass criteria that ignore signs are a trap.  When
   the sign of a vector quantity is physically meaningful (and it
   almost always is), check the vector, not the magnitude.  Also:
   keep solver conventions consistent across the whole module -- if
   `equilibrium_face_on_analytical` defines `V_ext = +f_ext . delta`
   in its derivation, every other solver in the module had better
   agree.

5. **Friction force/energy inconsistency, plus dead code in F_thresh helper.**
   Post-step-4 audit pass surfaced four small issues:

   * `friction_force_smooth` had a stray `+ eps * s` term in the
     denominator (kernel-parity inheritance).  With `K > 0` and the
     `s = 0` early-return, that term is unnecessary -- and it made
     `friction_energy_smooth` (closed-form integral assuming eps = 0)
     fail to be the antiderivative of the force.  At eps = 1e-12 the
     deviation was below test resolution; at kernel-scale eps = 1e-5
     it would matter.  Fix: drop the eps term from the force (denom is
     now exactly `K*s + M`), keep the energy formula; FD verification
     across 5 decades of s gives 1e-10 relative agreement.

   * `friction_force_smooth` is algebraically equivalent to the
     kernel's `(scale_used * |delta_t|)` with the `(K + cone_scale + eps)`
     denominator (multiply numerator and denominator by s).  Docstring
     now states this explicitly and points to
     `cslc_kernels.py:491-495`.

   * `F_thresh_analytical` in `test_04_friction.py` had a dead call
     to `equilibrium_face_on_analytical` with a dummy target before
     re-computing `f_n` from the series-spring closed form.  Deleted.

   * `F_thresh_analytical` returned `~3.75e30` at `k_stick = 0`
     (numerical guard `max(k_stick, 1e-30)`).  The mathematically
     correct value is `+inf` (no friction -> never slips).  Returning
     `float("inf")` now.

   **Lesson:** when defining a smooth surrogate, the integral form
   should be the EXACT antiderivative of the differential form.
   Carrying over a kernel-parity regulariser into theory code where
   it has no role is a smell -- the kernel had reasons (per-step
   numerical safety for in-flight Jacobi residuals); the theory does
   not.

6. **Friction smooth surrogate has a wide smoothing zone (step 4).**
   First run used the kernel's harmonic-mean smooth surrogate inside
   L-BFGS-B and compared it to the analytical hard piecewise law.
   They disagreed by 10-40% across the stick-slip transition zone.
   That's not a bug — the harmonic-mean form rounds the corner over
   a width of `mu*f_n/k_stick`, which IS the entire stick range.
   Fix: switch the numerical reference to scipy `minimize_scalar` on
   the hard piecewise 1D energy (independent code path, hard law),
   show the smooth surrogate alongside for visualisation only.
   **Lesson:** any smooth surrogate for a hard corner has a known
   smoothing zone proportional to the corner's characteristic scale;
   tests should either avoid that zone or accept its known error.

7. **Friction solvers used `sphere.ka` for tangent dynamics (pre-step-6).**
   `LatticeSphere.ka_t_ratio` had been exposed and wired through
   `anchor_force` / `anchor_energy` since step 1, but the three
   friction solvers (`equilibrium_with_friction_analytical`,
   `equilibrium_with_friction_hard_numerical`,
   `equilibrium_with_friction_smooth_numerical`) all used
   `sphere.ka` for the tangent direction.  The smooth solver's
   `jac` wrote `g = sphere.ka * d` directly while the matching
   `fun` called `anchor_energy(sphere, d)` (which DOES honour
   `ka_t_ratio`) -- so jac and fun silently disagreed for
   `ka_t_ratio != 1`.  Step 4 didn't catch it because every
   step-4 scene ran at `ka_t_ratio = 1.0` (default).  Fix:

   * Analytical: replace `sphere.ka` with `ka_t = sphere.ka * sphere.ka_t_ratio`
     in `s`, `F_thresh`, and the "no_friction" branch.
   * Hard numerical: same for `s_opt`, `E_anc`, and the upper bound.
   * Smooth numerical: replace `g = sphere.ka * d` with
     `g = anchor_force(sphere, d)` so jac uses the same path as the
     energy.

   Verification: step 6 PART B (threshold shift) and PART C (stick
   slope) match the closed forms to 1e-8 and 5e-16 respectively
   across `ka_t_ratio` in [0.05, 10] -- impossible to achieve with
   the pre-fix isotropic shortcut.  **Lesson:** when a property is
   plumbed via a dataclass field and only consumed by a few
   helpers, the rest of the module silently inherits the default.
   Anywhere `sphere.ka` appeared on a tangent axis was a bug
   waiting for someone to flip the ratio.

## Where we are now

* Theory module: single sphere + chain + chain-contact + stick-slip
  friction + curved-arc with geometric Poisson bulging + anisotropic
  anchor + **3D dome lattice + sphere indenter + multi-contact grip**
  (Steps 8 / 9).  All gradients FD-verified, sign-regression guards on
  every vector-quantity test, 8 driver scripts that rerun with
  `uv run -m cslc_main.theory.test_NN_*`.
* Step 7 (theory-align the kernels) **landed**: both `jacobi_step`
  (lattice equilibrium solve) and `write_cslc_contacts` (MuJoCo
  emission) now obey the series-spring + deformed-centre formulation
  this theory verified.
* Step 8 (3D dome contact) and Step 9 (dome grip budget) **landed**:
  production-equivalent fingertip dome (R_pad = 10 mm, 72 deg cap,
  N = 150) pressed by a tennis-ball indenter (R_obj = 33.5 mm); all
  four parts of each driver pass.  Key scientific finding: at
  production geometry the dome lattice behaves like a **parallel
  array of independent normal springs**, not a coupled mesh.
  **`kl` is not the right knob for tuning the dome's grip.**
* Step 10 / Step 11 dome+sphere stability characterised (NOT fixed);
  see C2 closure below for the n=3 revision of the
  "+3 mm/s wedge climb" headline.
* **C2 [LANDED]** — point-set contact path for box-target / mesh-
  target shapes.  Three sub-ships:
  - **C2a-c**: `jacobi_step_point_set` + `write_cslc_contacts_point_set`
    + `compute_pad_force_vs_point_set` (force-only verification kernel)
    in `newton/_src/geometry/cslc_kernels.py`.  Shared
    `INACTIVE_RAW_EPS_FACTOR = -50.0` constant in
    `cslc_main/theory/cslc_theory.py`.  Verified against the gold
    reference at <1e-5 rel err for scenes H (single pad vs box) and
    I (chain vs box) in test_07.
  - **C2d**: handler dispatch.  `compute_cslc_penetration_point_set`
    (argmax-overlap warm-start), `_launch_vs_point_set` mirror of
    `_launch_vs_sphere`, per-pair `is_point_set` flag, K_max sizing
    via the shared `INCLUSION_FACTOR = 50.0` constant, per-pair
    truncation counter with RuntimeWarning.
  - **C2e**: grasp-pipeline wiring.  `--object-kind box --box-side`
    / `--box-face-pitch` CLI, `make_box_target` 6-face sampler with
    optional `faces=(...)` selection, **approach-face-only sampling**
    (`_BOX_APPROACH_FACES = ("+x", "-x")`) as the default for
    grasp-pipeline box-held-object scenes.  Geometric-constraint
    warning when box edge would clip a dome patch.
* **Bug B [DIAGNOSED + FIXED via C2e]**: at ke=5e5 the lattice
  showed wild max_delta oscillation.  Initially looked like a
  damped-Jacobi convergence issue.  Profiling revealed MuJoCo
  solver was 3% of cost, Jacobi loop dominated, and the K_max
  truncation counter was firing on every active pad sphere.  Root
  cause was **adjacent-face wraparound contacts** in the 6-face
  box sampler -- pad spheres engaged with +z / -z / +/-y faces
  that were physically unreachable.  Approach-face-only sampling
  drops target_count 3750 → 1250 (3× speedup), eliminates the
  K_max overflow, and stabilises max_delta at default
  alpha=0.3, n_iter=40.  See C2 closure for full diagnostic chain.
* **ke split [LANDED]** -- `MaterialParams.ke` split into
  `ke_pad_physical` (drives `calibrate_kc` on the CSLC pad) and
  `ke_target_constraint` (drives `kc_series` target_ke in the
  emission kernel = MuJoCo rigid-contact stiffness).  Hydroelastic
  `kh` moved from `HydroParams` into `MaterialParams` so all
  material knobs share one config object.  Legacy `--material-ke`
  CLI preserved as alias that writes both `ke` fields; legacy
  `material.ke` property + setter route through both.  The split
  decouples *labels* (per-role knobs) but not *physics* -- the
  series-spring composition couples them; the cleanest comparison
  with hydro uses fixed `ke_target_constraint` and swept
  `ke_pad_physical` / `kh` as the physical-material axis.
* **Bridge passes 27/27** at per-scene tol `{5e-06, 0.001, 0.02}`:
  scenes A-G (single sphere + friction) at tol 1e-3, scene G slip
  at 2e-2, scenes H (single pad vs box) and I (chain vs box) at
  tol 5e-6.  Plus constant-discipline guard (4 kernel literal
  sites match `INACTIVE_RAW_EPS_FACTOR`) and N-pad sweep
  regression (chain×box at N in {50, 200, 800}) both PASS.
* **§7.1 empirical anchors measured [LOCKED v0.6]** -- at the
  silicone-target box-grasp (122 g cube), HOLD-averaged:
  L = 0.12 mm, contact_fraction = 1.0, A_patch = 330 mm²,
  F_per_pad = 67.9 N (qfrc_actuator real wrench, Newton-III
  balanced).  See benchmark_spec.md §7 for full delta vs working
  assumptions.  Wrench measurement instrumentation in
  `state.mujoco.qfrc_actuator` (pre-allocate, then read after
  each `solver.step`) -- the conventional `Contacts.rigid_contact_force`
  is allocated but never populated by MuJoCo's solver path.
  Prior `Σ stiffness × L` proxy was off by 3-100× because it
  counts lattice-internal spring forces, not net contact wrench.
* End-to-end dome grasp stability: **n=3 seed sweep showed
  single-seed conclusions don't replicate** at this scene scale.
  Step 11's "+3 mm/s climbing" was one realization in a
  distribution spanning -4.82 to +3.36 mm/s.  Mass-dependent
  asymmetry: at 5.75 g the cube settles +2-3 mm off-center
  reproducibly; at 122 g (steel-density, locked v0.6) the system
  is symmetric and Newton-III balanced.  See C2 closure for
  detail.

## Where we are going

### Step 7 — Theory-align the kernels [LANDED]

**Goal.**  Bring `newton/_src/geometry/cslc_kernels.py` to convergence
on the equilibrium this theory module validated -- in-place, no
side-by-side flag.  Cleanup pass first made the CSLC stack generic
(arbitrary `(positions, normals, edges)` lattice from any sampling
pipeline, no shape-specific kernels or "pad" terminology); the physics
rewrite then replaced the constant-load contact law with the
series-spring + deformed-centre formulation.

**What landed in `jacobi_step` (lattice solve).**

1. **Series-spring contact, deformed direction, exact deformed
   overlap.**  Each iteration recomputes `n_eff = (t_world - q_def) /
   ||t_world - q_def||` from the current `q_def = p - delta`, and uses
   `raw = (r_lat + R_target) - ||t_world - q_def||` directly instead
   of the linear approximation `phi_rest - dot(delta, n_eff)`.  The
   linear form collapsed to ~0 at non-contact spheres (kernel 1
   clamps `phi_rest` via `smooth_relu`), making `smooth_relu(0, eps) =
   eps/2` leak a spurious force; the exact form gives strongly negative
   `raw` far from contact, killing the leak structurally.
2. **`smooth_step` gradient factor on `f_contact`.**  The load on δ
   from the contact energy is `kc * phi_eff * smooth_step(raw) *
   n_eff` (theory bug #4 applied to the kernel).  Forgetting it gives
   the correct answer at deep saturated contact (`step → 1`) but
   leaves a residual force at non-contact spheres that sign-flipped
   the arc-bulge perimeter in scene D.  The implicit diagonal
   `S_n = kl*|N| + kc*gate` is unchanged -- still an upper bound on
   `|d(f_load)/d(delta_n)|`, so the damped Jacobi stays contracting.
3. **Friction `f_n`** inherits the same `gate` factor, so non-contact
   spheres carry no friction either.

**What landed in `write_cslc_contacts` (MuJoCo emission).**

4. **Deformed-centre emission.**  `point0 = q_def = q_world - delta`
   instead of the rest sphere centre; `margin0 = r_lat` (untouched,
   no radius-reduction shim); `normal_ab = (t - q_def)/||t - q_def||`
   (matches jacobi_step's D2).  MuJoCo reconstructs `solver_pen =
   (r_lat + R) - ||t - q_def|| = phi_def` exactly -- the same
   deformed overlap the lattice solve converges on.
5. **`pen_scale ≡ 1` by construction.**  The radius-reduction shim
   was the only thing producing pen_3d ≠ solver_pen; without it, the
   pen_scale workaround collapses to a constant.  Diagnostic field
   kept for back-compat (now always reads `contact_gate`).
6. **Stiffness handed to MuJoCo.**  `out_stiffness = kc_series ·
   contact_gate` (no pen_scale factor).  Per-contact force MuJoCo
   applies = `stiffness · solver_pen = kc_series · gate · phi_def`,
   the series-spring law jacobi_step obeys.

**kc calibration.**  The existing `calibrate_kc` helper
([cslc_data.py](../../newton/_src/geometry/cslc_data.py)) already used
the series-spring identity `1/keff = 1/ka + 1/kc + 1/ke_target` --
which is now an exact match for the post-Step-7 kernel.  No code
migration needed for the auto-calibrated production path; the pre-fix
kernel was the broken side (constant-load `kc * phi_rest`),
numerically agreeing with the series-spring formula only at
`kc << ka` (production today).  Hand-tuned scenes that set ``kc`` to
hit a specific force under the old law would need
`kc_new = kc_old * (ka + kc_old)/ka` to approximate the same force
back -- the auto-calibrated helper is exempt.

**Verification harness.**  [test_07_kernel_bridge.py](test_07_kernel_bridge.py)
drives the kernel and [kernel_bridge.py](kernel_bridge.py) (theory's
gold reference, routed through the multi-contact L-BFGS-B path so it
applies the same eps-smoothed gate at every surface sphere) on the
same `KernelScene` and compares per-sphere δ vectors -- magnitude AND
sign.  Seven scenes × `kc/ka ∈ {0.1, 1, 10}`:

| Scene | Verifies | Tol | Worst rel err |
|---|---|---|---|
| A. single sphere face-on | series spring | 1e-3 | 2e-6 |
| B. single sphere off-axis 30° | deformed contact direction | 1e-3 | 4e-7 |
| C. chain face-on at centre | lateral + contact | 1e-3 | 2e-5 |
| D. arc face-on at apex (inside bulge window) | curvature bulge sign | 1e-3 | 5e-4 |
| E. anisotropic anchor `ka_t = ka/3` off-axis | per-axis anchor | 1e-3 | 3e-6 |
| F. single sphere + tangential load, STICK | friction stick spring | 1e-3 | 2e-4 |
| G. single sphere + tangential load, SLIP | Coulomb plateau | 2e-2 | 2e-2 |

**Result: 21/21 PASS** (Step 7 ship; **now 27/27** with C2's added
scenes H + I — single pad and chain vs box, point-set kernel — plus
front-loaded constant-discipline and N-sweep regression guards).
G uses a relaxed 2% tolerance to absorb the
known geometric divergence between the kernel's deformed-direction
contact normal and the theory's analytical slip solver, which uses
the rest direction (slip requires δ_t > F_thresh/ka ≈ 0.3 mm at
production scales, where the rest assumption breaks at the 1-2%
level).  Kernel iteration converges to `||δ^(k+1) - δ^(k)||_∞ < 1e-10`
(typical 40-100 iters, alpha=0.3; B@kc=10·ka and friction scenes
need ~190-2000 iters).  Grasp `--pad-kind box` end-to-end regression
unchanged (`lifted=YES held=YES xy_slip_max=0.05mm`) through both
the jacobi-side (D1-D3+smooth_step) and emission-side (D4-D6)
physics rewrites -- the production calibration at `kc << ka` is in
the regime where the new series-spring law and the old constant-load
law numerically agree.

**Out of scope (step 7b, deferred).**

* **Non-sphere target geometries.**  Re-adding box / mesh / SDF
  targets is a mechanical port of the series-spring + deformed-centre
  emission onto a target-specific penetration kernel and the matching
  contact-emit.  Same divergence ledger; not blocked by anything
  Step 7 left undone.
* **`lattice_solve_equilibrium` triage.**  Reclassified in
  [cslc_kernels.py](../../newton/_src/geometry/cslc_kernels.py) as a
  **linear warm-start** (solves `(K + kc·I) δ = kc · phi_rest`, the
  series-spring law's tangent-space approximation at δ = 0).
  jacobi_step always runs after it and corrects the nonlinear
  residual.  Kept because production saves 10-20 jacobi iterations
  per step and the dense matvec preserves wp.Tape backward through
  the lattice solve.  Could be replaced with a sparse Cholesky for
  n ≳ 1000.

### Step 8 -- 3D dome contact [LANDED]

See "Step 8" section above.  Outcome: `kl` is irrelevant at production
geometry; the dome is a parallel array of independent springs.
Aggregate normal force `F_total` is set by `kc * sum(phi_eff_i)`, so
`kc` (via `contact_fraction` in the calibration) IS the right knob
*for the static normal-force budget* -- but Step 10 below disproves
the assumption that the static budget is what's limiting the dome
grasp.

### Step 9 -- Dome grip budget [LANDED]

See "Step 9" section above.  Outcome: friction headroom is 31.9x ball
weight at phi = 1 mm.  Initial hypothesis (operating-phi sensitivity)
was disproved empirically in Step 10.

### Step 10 -- Grasp integration: hypothesis disproved [LANDED-NEGATIVE]

Two end-to-end dome runs with the production trajectory:

| run | `contact_fraction` | SQUEEZE `max_pen` | held? | xy_slip |
|---|---|---|---|---|
| `_baseline_cf0p025` | 0.025 (production default) | 1.21 mm | NO | 723 mm |
| `_experiment_cf0p11` | 0.11 (matches Step-8 measured N_active/N) | 1.40 mm | NO | 630 mm |

Both reach more than 2x the 0.5 mm phi where Step 9 said headroom
collapses, and both still drop the ball, **in the same way at the
same time** (LIFT step ~1600, ~50 ms after LIFT begins).  The
`contact_fraction` hypothesis is therefore disproved: making the
dome softer per-sphere (lower `kc`) shifts SQUEEZE depth as
predicted but doesn't fix the held=NO outcome.

The real diagnosis (see "Action-item finding" under Step 9 above):
**geometric contact loss under LIFT.**  `max_pen` collapses to ~0
within ~50 ms of LIFT beginning -- before the pad has moved
meaningfully -- because the dome's apex rides up off the ball's
equator and the recessed cap region behind the apex can't reach the
ball's narrower upper surface.  Box pad doesn't fail this way
because its contact face is flat in z.  See
[`outputs/grasp/dome_grasp_diagnostic.png`](../../outputs/grasp/dome_grasp_diagnostic.png).

### Step 11 -- Dome stability investigation [LANDED]

**The falsification chain (one-paragraph orientation).**  Step 11
diagnosed a mysterious +3 mm/s upward drift of the held ball during
HOLD on the dome+sphere grasp.  Static physics said `tan(θ) < μ`
should be self-locking, but empirically the ball climbed and
eventually slipped out.  Four scripts in
[`../grasp/scripts/`](../grasp/scripts/) tested the four candidate
mechanisms in sequence:

1. [`exp_slow_lift.py`](../grasp/scripts/exp_slow_lift.py) tested
   whether LIFT-trajectory dynamics caused the drift (slow LIFT down
   to 15 s vs production 1.5 s).  **Falsified**: slow LIFT made it
   *catastrophically worse*, not better.  The drift is not transient.
2. [`exp_friction_sweep.py`](../grasp/scripts/exp_friction_sweep.py)
   tested whether CSLC's in-kernel `mu_friction` and `k_stick` could
   stop the drift (2×2 sweep over those two knobs).  **Falsified**:
   all 4 settings gave the same +3 mm/s drift.  In-kernel friction is
   invisible to the climb.
3. [`exp_emitted_friction_sweep.py`](../grasp/scripts/exp_emitted_friction_sweep.py)
   tested whether MuJoCo's emitted contact friction (`material.mu` at
   the geom-pair level, distinct from CSLC's in-kernel knob) could
   stop the drift (10× sweep μ ∈ {0.5, 5.0}).  **Falsified**: μ-
   invariant.  Friction at *any* layer can't fix the wedge.  This
   ruled out the entire friction-resistance hypothesis class.
4. [`exp_constraint_softening_sweep.py`](../grasp/scripts/exp_constraint_softening_sweep.py)
   tested whether the drift scales with MuJoCo's regularized-contact
   stiffness (`material.ke` over 4 orders of magnitude).  **Confirmed**:
   10× tighter ke reduced drift 67×.  Mechanism identified:
   MuJoCo's regularized-cone constraint admits per-step normal-axis
   position drift sized by `timeconst = √(0.95/ke)`; on convex pad
   geometry that drift projects onto +z and integrates over HOLD.

Each script answers a yes/no question about one hypothesis class.  The
collective rules out everything except regularization-stiffness as the
lever.  Subsections below give the empirical data for each.

**Implementation.**  Added `pad.kind = "dome_param"` in
[cslc_main/grasp/pads.py](../grasp/pads.py), a parametric
spherical-cap pad generated in-code from `R_pad` + `half_angle` (no
asset file).  The cap math reuses
[`cslc_main.theory.cslc_lattice.make_dome`](cslc_lattice.py) so the
grasp pad and the theory dome lattice share their geometry.  Plumbed
through `PadParams.dome_param_*` and CLI flags `--pad-r-pad` /
`--pad-half-angle`.

For `half_angle > pi/2` (wrap-around / mushroom caps) the n_z > 0.3
heuristic the OBJ-loading path uses to pick contact faces would have
cut off the wrap-around region.  Replaced it with an explicit
cap-face mask returned by the builder -- the mask captures the full
cap regardless of half_angle, AND excludes the back cylinder's end
caps which the n_z heuristic would have leaked.

Smoke test at production-equivalent geometry (R = 10 mm,
half_angle = 72 deg) reproduces the OBJ-based dome's failure
(`lifted=YES held=NO xy_slip=723 mm`).

#### Initial sweep -- `held=YES` was a too-coarse criterion

Six runs at `contact_fraction = 0.025`:

| R_pad | half_angle | held? (`final_z > 50 mm`) | final z [mm] | xy drift [mm] |
|---|---|---|---|---|
| 10 mm | 72°  | NO -- prod baseline | 33.3 | 724 |
| 20 mm | 72°  | **YES** | 67.8 | 2.5 |
| 33 mm | 72°  | **YES** | 54.7 | 4.9 |
| 20 mm | 90°  | NO (delayed drop ~6.2 s) | 33.3 | 56.0 |
| 33 mm | 90°  | NO (delayed drop ~6.1 s) | 33.3 | 57.4 |
| 20 mm | 110° | **YES** | 71.3 | 1.6 |
| 33 mm | 110° | NO (delayed drop ~6.0 s) | 33.3 | 48.2 |

Figure: [`outputs/grasp/dome_geometry_sweep.png`](../../outputs/grasp/dome_geometry_sweep.png).

Naive reading: "doubling R_pad fixes the grasp".  Wrong.  Looking at
the per-step ball trajectory inside the three `held=YES` runs shows
**none of them is actually stable** -- the ball is slowly slipping
out of position during HOLD.  Figure
[`outputs/grasp/cslc_held_trajectory_diagnostic.png`](../../outputs/grasp/cslc_held_trajectory_diagnostic.png):

| run | HOLD `dz/dt` [mm/s] | HOLD `dy/dt` [mm/s] |
|---|---|---|
| R = 20 mm / 72°  | **+3.04** (climbing up the dome) | -0.73 |
| R = 33 mm / 72°  | -0.03 (z stable) | **+1.37** (lateral) |
| R = 20 mm / 110° | **+3.84** (climbing up the dome) | +0.51 |

`held=YES` only meant the ball hadn't escaped within the 3 s HOLD.
The mechanism that's making the ball move is genuinely unstable and
would lose the ball given a longer HOLD.

#### The vertical climb is a geometric wedge instability

At HOLD start for R = 20 mm / 72° the ball center sits at world
`(0, 0, 58.7) mm` while the left-pad apex sits at world
`(-22.5, 0, 54.1) mm`.  The line from apex to ball center is
`(22.5, 0, 4.6)`; the contact normal therefore points
`+x +z` for the left pad and `-x +z` for the right pad.  The `±x`
components cancel; the `+z` components add to a **net upward force
on the ball** equal to `2 N sin(theta)` where
`theta = arcsin(4.6 / 22.97) ≈ 11.5°` at HOLD start.  The ball
climbs.  As it climbs, `dz_apex` grows, `theta` grows, the net
upward force grows -- runaway.

Two control experiments confirm this is geometric, not dynamic:

* **Slow-LIFT sweep** ([`outputs/grasp/wedge_force_diagnostic.png`](../../outputs/grasp/wedge_force_diagnostic.png),
  driver [`cslc_main/grasp/scripts/exp_slow_lift.py`](../grasp/scripts/exp_slow_lift.py)).
  Slowing LIFT (1.5 s -> 4.5 s -> 15 s, holding total distance
  constant) makes the failure **catastrophically worse**, not
  better.  At LIFT = 15 s the ball tracks the apex perfectly during
  lift (`dz_apex ≈ 0`) and then, at t ~ 9 s, undergoes a sudden fall
  to `dz_apex = -22 mm` ≈ `R_pad sin(half_angle)`, i.e. the cap rim.
  Reading: the apex axis is an unstable equilibrium; any
  perturbation is amplified geometrically; slow LIFT lets the
  perturbation grow until the ball falls off the cap rim.
* **Friction sweep** ([`outputs/grasp/friction_sweep_diagnostic.png`](../../outputs/grasp/friction_sweep_diagnostic.png),
  driver [`cslc_main/grasp/scripts/exp_friction_sweep.py`](../grasp/scripts/exp_friction_sweep.py)).
  Sweeping `CSLCParams.mu_friction in {0.3, 1.0}` and
  `CSLCParams.k_stick in {2.5e4, 2.5e5}` at R = 20 mm / 72°.  All
  four show the same `dz/dt = 3.0-3.2 mm/s` vertical climb during
  HOLD.  Friction tuning is **invisible** to the vertical climb.
  Static math `tan(theta) < mu` predicts stability at mu = 1.0
  through the wedge tilt range observed during HOLD (theta starts
  at 11.5° at HOLD entry, grows to ~22° once dz_apex reaches
  ~9 mm midway through HOLD).  Yet the empirical climb rate
  doesn't change with mu, so something in the friction pathway is
  not enforcing the static cone.

  **Resolved (two-experiment closure, 2026-04-24).**  Falsified
  through one sweep, attributed to the correct mechanism through a
  second, with one explicit retraction of an earlier mistake.

  **Experiment 1 -- emitted-friction sweep.**  Driver:
  [exp_emitted_friction_sweep.py](../grasp/scripts/exp_emitted_friction_sweep.py),
  figure: [`outputs/grasp/emitted_friction_diagnostic.png`](../../outputs/grasp/emitted_friction_diagnostic.png).
  Sweep ``cfg.material.mu`` over ``{0.5, 1.0, 2.0, 5.0}`` (10x range,
  comfortably past the static-stability threshold ``tan(22°)≈0.40``)
  while pinning ``cfg.cslc.mu_friction=0.3``.  HOLD ``dz/dt`` is
  essentially invariant:

  | ``material.mu`` | HOLD ``dz/dt`` | HOLD ``dy/dt`` |
  |-----------------|----------------|----------------|
  | 0.5 (baseline)  | +3.33 mm/s     | -0.72 mm/s     |
  | 1.0             | +3.22 mm/s     | +0.07 mm/s     |
  | 2.0             | +3.01 mm/s     | -0.70 mm/s     |
  | 5.0             | +3.05 mm/s     | -0.17 mm/s     |

  This **falsifies** "the gap is friction-coefficient mistuning."
  A 10x friction sweep moving ``dz/dt`` by < 10% rules out the
  static ``tan(theta) < mu`` story regardless of which friction
  layer (in-kernel ``mu_friction`` or emitted ``material.mu``)
  carries the parameter.

  **Earlier incorrect framing -- retracted.**  An initial
  interpretation argued "the wedge force enters through the contact
  normal, friction has no leverage on normal forces."  This is the
  WRONG textbook physics for this geometry.  At apex tilt
  ``theta ≈ 11.5°`` the contact normal is
  ``n ≈ (cos theta, 0, sin theta) ≈ (0.98, 0, 0.20)``; for pure +z
  ball motion the tangential slip component is
  ``|v_t| = |v| cos theta ≈ 0.98|v|`` -- ~96% of the climb energy
  IS tangential to the contact normal, which is exactly what
  friction is supposed to resist.  Classical analysis predicts
  ``F_friction_z = mu F_pad cos theta`` per pad
  comfortably overpowering the wedge ``F_wedge_z = F_pad sin theta``
  at any ``mu > tan theta`` -- yet the data shows it doesn't.
  The discrepancy is what motivated experiment 2 below.

  **Experiment 2 -- constraint-softening sweep.**  Driver:
  [exp_constraint_softening_sweep.py](../grasp/scripts/exp_constraint_softening_sweep.py),
  figure: [`outputs/grasp/constraint_softening_diagnostic.png`](../../outputs/grasp/constraint_softening_diagnostic.png).
  Hold ``mu_pad = 0.5``, ``mu_friction = 0.3`` fixed.  Sweep
  ``cfg.material.ke`` over four orders of magnitude (5e3 .. 5e6
  N/m), which through ``kc_series = (cslc_kc · target_ke) /
  (cslc_kc + target_ke + eps^2)`` in
  [cslc_kernels.py:933](../../newton/_src/geometry/cslc_kernels.py#L933)
  controls the per-contact stiffness MuJoCo applies -- and through
  it the regularised-constraint relaxation time via
  ``timeconst = sqrt(imp/ke)``.  This formula is the runtime
  contact-stiffness override path verified in
  [newton/_src/solvers/mujoco/kernels.py:429](../../newton/_src/solvers/mujoco/kernels.py#L429)
  (NOT the geom-level ``convert_solref`` path at line 192, which
  is a fallback for shapes that did not override their stiffness).
  Higher ``material.ke`` -> shorter ``timeconst`` -> tighter per-step
  constraint enforcement.  Result:

  | ``material.ke`` | HOLD ``z_start -> z_end``      | HOLD ``dz/dt``  | regime            |
  |-----------------|--------------------------------|-----------------|-------------------|
  | 5e3 (10x soft)  | 54.6 mm -> 50.9 mm             | **-1.00 mm/s**  | LIFT-incomplete   |
  | 5e4 (baseline)  | 58.6 mm -> 68.1 mm             | **+3.38 mm/s**  | the wedge climb   |
  | 5e5 (10x stiff) | 54.9 mm -> 55.0 mm             | **+0.05 mm/s**  | climb suppressed  |
  | 5e6 (100x stiff)| 33.4 mm -> 33.4 mm             | 0 (grip fails)  | constraint too tight |

  Two operating regimes flank the wedge climb on either side:

  *  ``ke = 5e3`` is **NOT the wedge mechanism**: at very low
     constraint stiffness the LIFT trajectory under-tracks -- HOLD
     starts at ``z_start = 54.6`` mm instead of the baseline's
     ``58.6`` mm because the soft contact can't transmit the LIFT
     velocity command to the ball.  During HOLD the ball is below
     the wedge-amplification regime; gravity-induced settling toward
     a lower equilibrium dominates the per-step normal-axis leak.
     This regime tells us "constraint too soft to track LIFT", not
     "wedge reversed."  The wedge mechanism is genuinely studied
     between ``ke = 5e4`` and ``ke = 5e5`` where LIFT completes
     properly.
  *  ``ke = 5e6`` is also not the wedge mechanism: at very stiff
     constraint MuJoCo's solver can't converge the per-step contact
     equation at production timestep, so grip fails entirely and the
     ball never lifts.  This bounds the operational ceiling on the
     "tighten ke" lever.

  Between those endpoints, the trend is monotonic and decisive: a
  10x stiffening from baseline ``5e4 -> 5e5`` reduces ``dz/dt`` by
  67x.  The mechanism, correctly stated:

  > MuJoCo's regularised contact constraint admits per-step
  > position drift along the contact normal direction with
  > magnitude proportional to ``timeconst·dt`` (set by the emitted
  > stiffness, via ``timeconst = sqrt(imp/ke)``), independent of
  > ``mu``.  At convex-pad-on-sphere geometry that normal direction
  > tilts upward as the ball climbs (eq:emit-normal in theory.txt),
  > so the per-step drift projects onto +z and integrates over
  > the 3 s HOLD into mm-scale climb.  Friction acts on slip
  > velocity within the cone, but cannot reduce the cone-interior
  > drift -- the leak is in the NORMAL constraint, where the cone
  > doesn't apply.

  **z-climb and y-drift are distinct mechanisms** (retracting an
  earlier overunification claim).  Looking at the two sweeps side
  by side:

  | sweep             | range  | ``dz/dt`` response | ``dy/dt`` response |
  |-------------------|--------|--------------------|--------------------|
  | mu (Experiment 1) | 10x    | <10% change (3.33 -> 3.05)  | up to 10x (-0.72 -> -0.07 @ mu=1) |
  | ke (Experiment 2) | 100x   | **67x change** (3.38 -> 0.05) | 2x change at most (-0.16 to -0.07) |

  ``dz/dt`` responds to ``ke`` and not to ``mu``; ``dy/dt``
  responds to ``mu`` and only weakly to ``ke``.  Two mechanisms
  acting on orthogonal parameters:

  *  **z-climb**: normal-axis regularised-cone leak, scales with
     ``timeconst`` (= 1/sqrt(ke)), independent of ``mu``.  Sources
     from contact-normal +z components on convex-on-sphere geometry
     (eq:emit-normal).  In a symmetric two-pad face-on grasp, left
     and right per-sphere normals have +z components by the
     uniformly-upward apex-to-ball line; these ADD, producing the
     systematic wedge.
  *  **y-drift**: lattice rest-position asymmetry produces a small
     net lateral wrench (~0.085% of normal at production N -- see
     the Closure section just below).  Friction can resist this
     when ``mu`` is high enough relative to the y-fraction of the
     normal force.  This cannot come from the normal-axis leak,
     because each pad's per-sphere normals integrate to zero
     azimuthally around the apex axis on a perfectly
     azimuthally-symmetric cap.  The Fibonacci spiral breaks that
     per-pad azimuthal symmetry sphere-by-sphere (the 0.085%
     residual at N=150 measured in Step 10d); summed across the
     two-pad mirror-pair construction those per-pad residuals add
     rather than cancel, but the existence of a nonzero y-component
     in the first place is a per-pad azimuthal-symmetry-breaking
     phenomenon, not a property of the inter-pad x-mirror.

  These two mechanisms remain experimentally separable: a future
  sweep that varied ``ke`` and ``mu`` together would show a
  product structure (``dz/dt`` factor varies with ``ke``,
  ``dy/dt`` factor varies with ``mu``), not the additive
  interactions that a single mechanism would produce.

  **Operational resolution**.  Three independent levers, in order
  of preference:

  1. **Use flat-faced contact pairs.**  For face-on grasps where the
     contact patch is INSIDE a flat face (e.g. box pad on box object
     interior face, cylinder pad on cylinder face), the contact
     normals are pose-independent in the body frame: a pose
     perturbation that shifts the patch within the face doesn't
     rotate the normal direction, so the per-step constraint drift
     does not project onto an out-of-plane direction.  The wedge
     mathematically vanishes.  **Geometric requirement.**  The
     contact patch's lateral extent at HOLD penetration must sit
     inside one face with margin -- a box edge or corner reintroduces
     mixed-normal contributions (corners behave like spheres with
     R -> 0) and a local wedge can reappear.  For the production
     dome lattice (R_pad = 10 mm, half_angle = 72 deg), the engaged
     contact patch radius at HOLD penetration is approximately
     ``R_pad · sin(half_angle) ≈ 9.5 mm``.  C2 box geometry should
     satisfy

         box_side  ≥  2 · patch_radius  +  2 · margin
                   ≈  2 · 9.5 mm  +  2 · 3 mm   ≈  25 mm

     to keep the patch comfortably inside one face.  Smaller boxes
     would transfer the wedge to the box edges and confound the
     box-vs-dome comparison.  **Recommendation for the C2 box
     test**: run the box-grasp scene at TWO ``ke`` values (production
     ``5e4`` and the wedge-suppressing ``5e5``).  If the box
     ``dz/dt`` is also ``ke``-sensitive but with a much smaller
     baseline magnitude, the closure transfers cleanly.  If
     ``ke``-insensitive at the box, flat-face geometry has revealed
     a new mechanism -- still informative, but a different scientific
     story.
  2. **Tighten the contact constraint.**  ``material.ke = 5x10^5``
     reduces the climb 70x without numerical instability.  Above
     1x10^6 the grip starts failing because MuJoCo's solver can't
     converge the tighter constraint at the production timestep --
     so this lever has a ceiling and isn't a universal fix.
  3. **Move fast.**  Slow LIFT lets the leak integrate longer;
     fast LIFT escapes before the wedge can amplify.  Already
     consistent with the slow-LIFT diagnostic showing slower lift =
     catastrophically worse.

  Step 11's vertical-wedge climb is now closed at the level of
  mechanism (regularised-cone normal-axis drift), parameter
  sensitivity (scales with ``timeconst``, not ``mu``), and
  resolution (flat-face contact pair, or higher emitted stiffness
  bounded by integrator stability).

#### The y-drift is a separate lattice-asymmetry issue

The `mu_friction = 1.0` run does reduce the lateral drift at
R = 20 mm/72° from 0.73 to 0.02 mm/s (36x improvement) -- but only
at the SMALL pad.  At R = 33 mm with the same `mu_friction = 1.0`,
the y-drift is unchanged at 1.4 mm/s.

Reading: the y-drift comes from the **Fibonacci-spiral lattice not
being symmetric about y = 0**.  The spiral places samples at
golden-angle-spaced phi, which is uniform in expectation but biased
sample-by-sample.  Each lattice sphere's contact force has a small
y component; summed over the patch the bias produces a net y force
on the ball.  At small R_pad / dense lattice the bias is small per
sphere and friction can hold (mu = 1.0 fixes it).  At larger R_pad
with the same N = 150 samples the per-sample spacing grows, the
bias per sphere grows, and friction can't.

The fix here is sampler-side: replace the Fibonacci spiral with a
sampling pattern that is exactly symmetric about y = 0 (e.g.
mirror-pair the spiral, or use a regular hexagonal lattice mapped
onto the cap).  Deferred -- not a CSLC kernel concern.

#### What this means for the dome grasp

* The CSLC kernel is doing its job.  The instability is not a
  contact-model bug.
* The vertical wedge instability is **intrinsic to a curved convex
  pad pressing a sphere from the side under vertical LIFT**.  No
  CSLC parameter sweep fixes it.  Friction can't because the
  geometric wedge is generated by the contact normal direction,
  not by tangential motion.
* The wedge force at HOLD scales with `dz_apex / dist`.  At
  R_pad = 33 mm `dist = sqrt(33² + dz_apex²)` is large enough that
  the wedge is negligible (~2 % tilt at the same `dz_apex`).
  R = 33 mm / 72° therefore has near-zero z-climb -- it's the most
  stable dome geometry in the sweep -- but y-drift persists due to
  the separate lattice-asymmetry mechanism.
* A hypothetical box pad (flat-face shape with regular-grid sampling)
  would have neither problem: flat face = zero wedge by construction;
  regular grid = zero lattice bias.  **Production caveat**: the actual
  grasp pipeline always uses DOME pads (R_pad, half_angle); "box" in
  the production CLI refers to the held OBJECT, not the pad.  Dome
  pads grasping a box object still carry Fibonacci-spiral lattice
  bias.  v0.6 §7.1 confirmed this: at 5.75 g cube the deterministic
  +2-3 mm cube x-offset is driven by lattice asymmetry (Step 10d's
  y-drift mechanism, projected onto x via the dome-pad rotation
  scheme; see §7.1 "Pad construction note" for the geometry).  At
  122 g cube the asymmetry disappears because gravity dominates the
  small lateral imbalances.

#### How to reproduce

All commands write to `outputs/grasp/<run-label>_<contact_model>/`.
The `_<contact_model>` suffix is appended automatically (see
[`GraspConfig.run_dir_name`](../grasp/params.py)) so the same
`--run-label` can be reused across `--contact-model {cslc,hydro,point}`
without collision.

Each run takes about 20 s on a single GPU; expected end-of-run
verdict is shown in the `# RESULT` comment.

##### Reproducing the original failure modes

```bash
# Production fingertip dome from the shipped OBJ asset.
# Ball lifts briefly, slips off the cap, flies sideways.
uv run --extra importers -m cslc_main.grasp.main \
    --pad-kind dome --no-timestamp --run-label dome_original
# RESULT  lifted=YES held=NO xy_slip_max ~ 723 mm
```

```bash
# Parametric dome at the same geometry -- reproduces the OBJ failure
# to within sampler resolution.
uv run --extra importers -m cslc_main.grasp.main \
    --pad-kind dome_param --pad-r-pad 0.010 --pad-half-angle 72 \
    --no-timestamp --run-label dome_param_R10_a72
# RESULT  lifted=YES held=NO xy_slip_max ~ 724 mm
```

##### The "seemingly fixed" R = 20 mm / 72 deg dome (still slips)

```bash
uv run --extra importers -m cslc_main.grasp.main \
    --pad-kind dome_param --pad-r-pad 0.020 --pad-half-angle 72 \
    --no-timestamp --run-label dome_curved_flat
# RESULT  lifted=YES held=YES (coarse) ; HOLD dz/dt = +3.0 mm/s climbing
```

```bash
# Hydroelastic at the SAME geometry (note: kh is uncalibrated for the
# mesh-pad path, see grasp/README.md).  Writes to dome_curved_flat_hydro/.
uv run --extra importers -m cslc_main.grasp.main \
    --contact-model hydro \
    --pad-kind dome_param --pad-r-pad 0.020 --pad-half-angle 72 \
    --no-timestamp --run-label dome_curved_flat
# RESULT  lifted=YES held=NO xy_slip ~ 6 mm  (drops at LIFT->HOLD)
```

##### Most z-stable dome geometry in the sweep

```bash
# R = 33 mm / 72 deg.  No vertical climb during HOLD but the
# Fibonacci-spiral lattice-asymmetry y-drift still runs at ~ 1.4 mm/s.
uv run --extra importers -m cslc_main.grasp.main \
    --pad-kind dome_param --pad-r-pad 0.033 --pad-half-angle 72 \
    --no-timestamp --run-label dome_R33_a72
# RESULT  lifted=YES held=YES (coarse) ; HOLD dz/dt ~ 0, dy/dt = +1.4 mm/s
```

##### Full 6-cell geometry sweep

Each combination of `R_pad in {20, 33} mm` x
`half_angle in {72, 90, 110}°`:

```bash
for combo in "20 72" "33 72" "20 90" "33 90" "20 110" "33 110"; do
    R_mm=${combo% *}; angle=${combo#* }
    R_m=$(python3 -c "print($R_mm / 1000.0)")
    uv run --extra importers -m cslc_main.grasp.main \
        --pad-kind dome_param --pad-r-pad ${R_m} --pad-half-angle ${angle} \
        --no-timestamp --run-label dome_param_R${R_mm}mm_a${angle}deg
done
```

##### Control experiments (rule out non-geometric causes)

```bash
# Slow-LIFT sweep: 1.5 s / 4.5 s / 15 s LIFT duration, total distance
# held constant.  Slower LIFT makes failure CATASTROPHICALLY worse,
# proving the wedge instability is not dynamics-induced.
uv run --extra importers python \
    cslc_main/grasp/scripts/exp_slow_lift.py
# RESULT  fast: xy_slip = 2.5 mm   medium: 311 mm   slow: 322 mm
```

```bash
# CSLC friction sweep on R = 20 mm / 72 deg: mu_friction in {0.3, 1.0},
# k_stick in {2.5e4, 2.5e5}.  All four runs show the SAME ~3 mm/s
# HOLD z-climb -- friction tuning cannot resist the geometric wedge.
uv run --extra importers python \
    cslc_main/grasp/scripts/exp_friction_sweep.py
# RESULT  HOLD dz/dt unchanged across all 4 friction settings
#         (mu = 1.0 does reduce y-drift at this small pad: 0.73 -> 0.02 mm/s)
```

##### Diagnostic plots

After running the experiments above, regenerate the figures referenced
in this Step 11 with:

```bash
# Per-run obj(t) trajectories side-by-side (z, x, y, xy radial drift).
# Reads outputs/grasp/dome_param_R20mm_a72deg_cslc/ etc.
uv run --extra dev python -c "
import csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
runs = [
    ('CSLC R=20mm/72',  'dome_param_R20mm_a72deg_cslc'),
    ('CSLC R=33mm/72',  'dome_param_R33mm_a72deg_cslc'),
    ('CSLC R=20mm/110', 'dome_param_R20mm_a110deg_cslc'),
]
fig, axes = plt.subplots(3, len(runs), figsize=(5*len(runs), 9), sharex=True)
for col, (name, label) in enumerate(runs):
    rows = list(csv.DictReader(open(Path('outputs/grasp') / label / 'timeseries.csv')))
    t = np.array([float(r['t']) for r in rows])
    x = np.array([float(r['obj_x']) for r in rows]) * 1000
    y = np.array([float(r['obj_y']) for r in rows]) * 1000
    z = np.array([float(r['obj_z']) for r in rows]) * 1000
    axes[0, col].plot(t, z); axes[0, col].set_title(name)
    axes[1, col].plot(t, x, 'r'); axes[1, col].plot(t, y, 'g')
    axes[2, col].plot(t, np.hypot(x, y))
plt.tight_layout()
fig.savefig('outputs/grasp/cslc_held_trajectory_diagnostic.png', dpi=140)
print('saved')
"
```

The other diagnostic figures (`wedge_force_diagnostic.png`,
`friction_sweep_diagnostic.png`, `dome_geometry_sweep.png`,
`dome_grasp_diagnostic.png`) are produced by inline Python snippets
that read the same CSVs; see this notes file's git history for the
exact one-liners.

#### Closure -- finite-N lattice asymmetry, characterised not fixed [LANDED]

The y-drift mechanism above was investigated at the theory level in
follow-up Steps 10 / 10b / 10c / 10d.  Punchline: **the CSLC static
wrench residual from Fibonacci-spiral lattice asymmetry converges as
N^(-1.25) -- faster than the random-walk N^(-1/2) rate -- and is
0.085% of the normal wrench at the production N = 150 single-pad
geometry tested in Step 10d**.  At that level the distributed
contact thesis is preserved at the wrench level for production-N
single-pad scenes; no per-sphere or wrench-level projection is
warranted for symmetry reasons.

**Caveat on the single-pad -> two-pad bound.**  The 0.085% figure
was measured on ONE dome lattice pressed face-on against a ball.
Production has LEFT + RIGHT pads, each its own Fibonacci-spiral
instantiation.  Two pads constructed from the same Fibonacci formula
are NOT in general mirror-symmetric.  Production right-pad
construction uses **rotation, not mirroring** (v0.6 §7.1 confirmation):
box-pads get a 180° rotation about z, dome-pads get ±π/2 about y.
Both schemes preserve the body-frame y-direction when mapped to
world (180° about z preserves y; ±π/2 about y preserves y), so
per-pad y-biases ADD rather than cancel across pads.  The two-pad
lateral wrench at production N can therefore range from ~0 (lucky
cancellation) up to ~2x the per-pad residual (additive worst case),
i.e. up to ~0.17% of the normal wrench.  Still small in absolute
terms; v0.6 §7.1 measurement at 122 g cube shows F_left = F_right
within 0.3%, so at production mass the residual is even smaller
than the theoretical worst case (gravity-driven equilibration helps).
Verifying which of the two regimes the production scene sits in is
part of suspect #2 below.

How that conclusion was reached (all theory-side, no kernel work):

* **Step 10** ([test_10_pad_vs_box.py](test_10_pad_vs_box.py)).
  Generalised the target from `RigidTarget` (single sphere) to
  `PointSetTarget` (arbitrary surface sampling, e.g. a trimesh box).
  Verified the multi-point energy + gradient + solver reduce
  exactly to the single-sphere series-spring law at M = 1.  Part C's
  sample-density sweep on a 50 mm sampled box showed the finite-N
  lateral residual dropping 17x from N = 150 to N = 3000 -- first
  hint that what production observed is a discretisation artifact,
  not a model bug.

* **Step 10b** ([test_10b_symmetry_projection.py](test_10b_symmetry_projection.py)).
  Built a per-pad-sphere "patch-resultant projection" variant
  (`equilibrium_point_set_projected_numerical`, in the experimental
  sidecar [cslc_projection_experiment.py](cslc_projection_experiment.py)).
  The variant constrains each sphere's contact energy to its own
  outward-normal axis, killing per-sphere lateral residuals to
  machine zero by construction.  Showed the variant agrees with the
  canonical solver to ~0.5% at dense N and *redistributes*
  lattice-asymmetry-induced lateral motion back into normal
  compression at sparse N.

* **Step 10c** ([test_10c_lattice_projection.py](test_10c_lattice_projection.py)).
  Multi-pad-sphere generalisation
  (`solve_lattice_sphere_indenter_projected`) on a production-shaped
  dome lattice (R_pad = 10 mm, half_angle = 72 deg, N = 150) face-on
  against a 30 mm ball.  **Per-sphere projection reduces
  lattice-internal lateral residual ~5x (199 nm -> 40 nm median per
  sphere) but does NOT reduce the net wrench on the ball** (1.01x
  ratio).  Diagnosis: the ball-side y-drift is driven by
  rest-position asymmetry ``p_i``, not by per-sphere ``delta_i`` --
  killing ``delta_t_i`` shifts the contact direction by ~0.01%,
  invisible at the wrench level.  This is what motivated 10d.

* **Step 10d** ([test_10d_wrench_convergence.py](test_10d_wrench_convergence.py)).
  Swept N in {50, 150, 500, 1500, 5000} on the same dome + ball
  scene, baseline solver only.  Power-law fit
  ``log10(|F_t|/|F_n|) = -0.564 + (-1.249) * log10(N)``.  Exponent
  -1.25, much faster than the random-walk -0.5 (CLT) rate -- the
  Fibonacci spiral has structured self-cancellation, not random
  noise.  Measured ratios:

  | N    | spacing | F_normal  | \|F_tangent\| | \|F_t\| / \|F_n\| |
  |------|---------|-----------|---------------|---------|
  | 50   | 2.77 mm | 6.39 N    | 7.4e-3 N      | 0.116%  |
  | 150  | 1.61 mm | 18.78 N   | 1.6e-2 N      | 0.085%  |
  | 500  | 0.89 mm | 62.24 N   | 1.0e-2 N      | 0.016%  |
  | 1500 | 0.52 mm | 186.4 N   | 6.0e-3 N      | 0.003%  |
  | 5000 | 0.28 mm | 621.2 N   | 2.8e-3 N      | 0.00045%|

  See [`figures/10d_wrench_convergence.png`](figures/10d_wrench_convergence.png)
  for the log-log convergence plot + the residual-direction stability
  diagnostic.

**What this means for the production y-drift.**  The static wrench
residual cannot account for the ~1.4 mm/s drift during HOLD
(0.085% of 18 N is 15 mN -- below typical solver / friction noise).
Suspects, ranked by likelihood, all owned by the grasp pipeline /
integrator layer rather than the CSLC contact model:

1.  **MuJoCo solver coupling.**  ~~The soft constraint compliance +
    integrator-level dynamics can amplify per-step residuals over the
    3 s HOLD interval.~~  **PARTIALLY resolved 2026-04-24**: the
    constraint-leak mechanism IS real on the vertical-wedge axis
    (see the friction-vs-wedge sweep result inside this Step 11),
    but it is a *consequence* of the wedge geometry, not an
    independent y-drift source.  The y-drift specifically is
    laterally driven and IS friction-resistible (``mu_friction = 1.0``
    reduces it 10x in both notes.md Step 11 original sweep and the
    new emitted-friction sweep).
2.  **Two-pad lattice mismatch.**  Production has left + right pads,
    each a different Fibonacci-spiral instantiation; per-pad
    residuals don't pair-cancel.  Quantifiable by sweeping the
    right-pad seed against a fixed left pad, but requires the grasp
    pipeline (not theory-pure).  Now the leading remaining suspect.
3.  **Gravity-driven settling.**  Gravity-induced ball settling
    during HOLD can express small static residuals as visible
    kinematics through the constraint solver's compliance.

None of these require modifying the CSLC contact model.  Further
y-drift diagnostic belongs at the grasp pipeline / MuJoCo integrator
layer, not here.

The **vertical wedge climb** (the OTHER mode -- separate from y-drift)
IS now closed: see the two-experiment "Resolved" panel in this
Step 11.  Mechanism is regularised-cone normal-axis drift; resolution
is flat-face contact pair OR higher emitted constraint stiffness
(bounded by integrator stability).  C2's box-target work will exhibit
the wedge much less SO LONG AS the contact patch fits inside a single
box face with margin -- a box edge or corner reintroduces mixed-normal
contributions and a local wedge can reappear.  Test geometry for C2
should specify ``box_side >= 2 * pad_diameter`` plus the lattice
neighbourhood radius so the contact patch sits cleanly inside one face.

**Files added by Steps 10 / 10b / 10c / 10d** (canonical kept,
sidecars removable):

* Canonical (kept):
  * [cslc_theory.py](cslc_theory.py) -- gained `PointSetTarget` +
    multi-point primitives
  * [cslc_box.py](cslc_box.py) -- trimesh box surface sampler
  * [test_10_pad_vs_box.py](test_10_pad_vs_box.py) -- multi-point
    verification
* Experimental sidecars (deletable to revert):
  * [cslc_projection_experiment.py](cslc_projection_experiment.py)
    -- both projection solvers
  * [test_10b_symmetry_projection.py](test_10b_symmetry_projection.py)
    -- box-target ablation
  * [test_10c_lattice_projection.py](test_10c_lattice_projection.py)
    -- dome + ball ablation
  * [test_10d_wrench_convergence.py](test_10d_wrench_convergence.py)
    -- convergence diagnostic
  * `figures/10[bcd]_*.png`

Deleting the four sidecar `.py` files and their figures restores the
canonical Step 10 baseline with no side effects.

#### Open follow-up: projection stress-test at high curvature / corners

If the projection variants are ever revived (e.g. for a different
target geometry where rest-position asymmetry is somehow controllable),
the next test to write is an **over-constrain check** on contact
patches where the underlying surface normal genuinely varies across
the patch (box edge or corner, sharply curved mesh, etc.).  The
frozen-tangent simplification assumes the patch normal is
approximately constant per pad sphere -- on a box face this is exact,
on a sphere indenter it's ~radial, but on a corner the per-sphere
projection could over-constrain the deformation.  This is left as a
TODO for if/when the projection path is revisited.

## C2 closure -- single-seed claims do not replicate [n=3, INSUFFICIENT FOR FINDINGS]

The C2 day-1 ke-sweep falsification on the box-grasp scene
(`--object-kind box --box-side 0.025 --pad-kind dome_param
--pad-r-pad 0.020 --pad-half-angle 72`, ke in {5e4, 5e5}) was
run after C2e wired the point-set contact path.  Three-stage
discovery, each stage walking back the previous narrative:

**Stage 1 (n=1, 6-face sampling):** apparent ~36x ke-sensitivity
of HOLD `dz/dt`.  Read as confirmation that the dome+sphere
"regularization-cone leak" mechanism transferred cleanly to flat
geometry.  *Wrong: confounded by spurious +z-face contacts
(see Bug B post-mortem below).*

**Stage 2 (n=1, 2-face sampling -- post-Bug B):** sign of
ke-sensitivity reversed.  Read as a geometry-dependent sign of
the leak's projection onto the body's translational DOF.
*Wrong: n=1 misread of high-variance data.*

**Stage 3 (n=3 with 1mm spawn-y jitter, 2-face sampling -- this
section):** the Stage 2 sign-inversion does not survive
replication.  Seed-to-seed variability dominates the ke-shift on
both geometries.

### n=3 seed-sweep result (defaults `alpha=0.3, n_iter=40`)

| cell | seed=0 | seed=+1mm | seed=-1mm | mean | std |
|---|---|---|---|---|---|
| sphere + ke=5e4 | +3.36 | -0.06 | -4.82 | -0.51 | 3.35 |
| sphere + ke=5e5 | -0.025 | -0.087 | -0.041 | -0.051 | 0.027 |
| box + ke=5e4 | -0.007 | +0.011 | +0.018 | +0.008 | 0.010 |
| box + ke=5e5 | -0.095 | **-1.15** | -0.025 | -0.42 | 0.51 |

All values are HOLD `dz/dt` in mm/s.  Raw data:
`outputs/grasp/seed_sweep__*/timeseries.csv`.

### The one finding the n=3 data DOES support

**Single-seed conclusions from this scene do not replicate.**
With n=3 at 1mm spawn-y jitter, the seed-to-seed std exceeds
the mean ke-shift on both geometries (ke-shift z-scores: sphere
0.14, box 0.83 -- both well under 1-sigma).  Future
falsifications on this scene class need n>=3 minimum, with at
least n>=10 to characterise distributional structure.  Step
11's `+3.0 mm/s climbing` headline was one realization in a
distribution that ranges -4.82 to +3.36 mm/s; the wedge mechanism
may still be the correct explanation of any individual
trajectory but the population-mean direction is unresolved.

### Hypotheses suggested by the data (NOT supported at n=3)

The following patterns appear in the n=3 table but are
underpowered to claim as findings.  With n=3, the standard
deviation estimator's own 95% CI spans roughly [0.5x, 4x] of the
point estimate; ratios of stds are even noisier.  All are
hypotheses for follow-up at n>=10:

* **H1: ke acts as a variance regulator on dome+sphere.**  Sphere
  drift std appears to drop from 3.35 to 0.027 mm/s (~120x
  apparent ratio).  Suggests at low stiffness the contact patch
  may have multiple near-equilibria with 1mm jitter selecting
  between them; high stiffness may collapse to a single
  attractor.  Would require n>=10 per cell to claim a >10x
  variance ratio with confidence.

* **H2: ke acts as an anti-variance regulator on box+dome.**
  ~~Initially suggested by the rise from std 0.010 to 0.51 mm/s
  driven by one outlier seed (ke=5e5, +1mm jitter, -1.15 mm/s).~~
  **REFUTED by reproducibility check.**  Re-running that exact
  config gave dz/dt = +0.013 mm/s -- 100x different outcome
  from the same seed/config, MD5-different timeseries.csv.  The
  cell sits on the edge of a stability basin where bounded
  floating-point non-determinism (GPU atomic-reduction order
  etc.) flips the macroscopic outcome.  This is a "chaotic
  basin" cell, not a bistable cell -- there isn't a second
  stable mode the trajectory consistently lands in, just
  sensitivity to numerical noise.  Excluding the artifact,
  box+ke=5e5 has mean ~-0.04 mm/s, std ~0.05 mm/s -- comparable
  to box+ke=5e4 (no strong ke effect on box drift at all).

  Implication for the benchmark: at least one cell in the
  CSLC C2 grid is in a numerically-sensitive regime.  Cells
  this close to a stability boundary need a run-to-run
  reproducibility check (run the same seed twice, compare),
  not just seed-to-seed jitter.  This is a real CSLC
  characteristic, not a bug to fix.

* **H3: Step 11's wedge mechanism is one of several outcomes.**
  At n=3 the dome+sphere case spans -4.82 to +3.36 mm/s.  Three
  possible explanations -- the n=3 data does not discriminate:
  * (a) the geometric wedge is real but only one of several
    competing forces, and gravity dominates in some seed
    configurations;
  * (b) Step 11's wedge analysis is correct only for a subset of
    initial conditions (e.g., seeds that land the dome's apex
    close to the sphere's pole);
  * (c) the wedge IS the dominant effect on average but the seed
    variance dwarfs it at this scene scale.

### Bounded GPU non-determinism characterised

Reproducibility re-runs of two extreme outliers (same seed,
same CLI args, separated runs):

| Cell | seed | Run 1 dz/dt | Run 2 dz/dt | md5 same? |
|---|---|---|---|---|
| sphere ke=5e4 | -1mm | -4.82 mm/s | -4.84 mm/s | NO -- but outcomes agree to 0.5% |
| box ke=5e5 | +1mm | -1.15 mm/s | +0.013 mm/s | NO -- outcomes differ by 100x |

The pipeline has bounded GPU non-determinism (Warp atomic
reductions, MuJoCo CG solver iteration order on parallel
hardware) that produces different float32 traces on
bit-identical inputs.  In MOST regimes this noise stays
microscopic and the macroscopic outcome is reproducible to
0.1-0.5%.  In chaotic-basin regimes (one cell found so far)
the noise amplifies into 100x outcome differences.

**Methodological implication:** any benchmark cell suspected
to be near a stability boundary needs a run-to-run check
(run same seed twice), not just seed-to-seed jitter.  This
gives a "numerical reproducibility" axis orthogonal to
"initial-condition reproducibility".

### Open questions before benchmark execution

* **n>=10 sweep**, IF the benchmark headline metric ends up
  needing distributional structure.  Cost: ~85 min wall.  Defer
  unless reviewers ask.
* **Smaller jitter (0.25mm) sweep**, IF dome variance turns out
  to be a 1mm-jitter artifact rather than intrinsic.  Side
  quest; defer.
* **Run-to-run reproducibility audit** of each benchmark cell
  (2 runs at same config, compare outcomes).  Adds n_cells
  extra runs to the benchmark spec.  Recommended.

### Bug B post-mortem (linkage)

The full diagnostic chain that produced this closure:

1. Day-1 falsification on 6-face box at n=1: apparent Outcome A
   (mechanism transfers), but max_delta oscillation at `ke=5e5`
   raised a red flag (8-10 mm spikes vs ~0.5 mm on dome+sphere).
2. Profiled the scene -- found MuJoCo solver was 3% of cost (not
   the bottleneck), Jacobi loop at ~2.4 ms/iter dominated,
   K_max truncation firing on every pad sphere every step.
3. Implemented approach-face-only sampling
   (`_BOX_APPROACH_FACES = ("+x", "-x")` in
   `cslc_main/grasp/contact_models.py`): target_count 3750 ->
   1250, K_max overflow drops 150/150 -> 7-13 pad spheres,
   2.3-2.9x per-step speedup, max_delta oscillation disappears.
4. Bug B itself was a sampling artifact, not a solver
   instability.  Default `alpha=0.3, n_iter=40` works on the
   2-face scene.
5. Re-falsified at n=1: apparent sign-inversion (Stage 2).
6. Re-falsified at n=3: sign-inversion is within seed noise.
   Stage 3 finding: single-seed claims on this scene do not
   replicate.

The earlier `[LANDED]` closure on finite-N lattice asymmetry
remains correct -- that's a separate analytical claim about the
Fibonacci-spiral lattice's y-drift, independent of the ke
question.  C2 adds:

* **infrastructure**: point-set contact path, approach-face
  sampling, geometry-derived K_max, profile harness,
  multi-seed sweep mechanism.
* **methodological finding**: single-seed conclusions on this
  scene scale are unreliable -- the variance is too high.
  Future falsifications must report (mean, std) across n>=3
  seeds.  This is the one robust takeaway from the C2 cycle.
* **hypotheses for follow-up** (H1-H3 above): variance
  regulator behaviour, box bistability candidate, Step 11
  wedge interpretation.  None are claimed as findings at n=3.

### §7.1 empirical anchors + wrench-readout breakthrough [LOCKED v0.6, PENDING v0.8 RE-VALIDATION]

> **v0.8 status note (2026-05-23)**: the v0.6 measurements below were
> taken at R_pad = 20 mm, which v0.8 of [benchmark_spec.md](benchmark_spec.md)
> retracted as geometrically invalid (contact patch overflows the 25 mm
> box face by ~19 mm).  The spec restored R_pad = 10 mm (the original
> Step 8/9/10 production geometry).  Numbers below are correct for
> their measurement geometry but are being re-measured at R_pad = 10 mm
> via [`exp_anchors.py`](../grasp/scripts/exp_anchors.py); the
> wrench-readout breakthrough and the mass-dependence-of-asymmetry
> finding are geometry-agnostic and carry forward.  See
> [benchmark_spec.md §7.1](benchmark_spec.md) for the re-validation
> status.

Benchmark prep (see [benchmark_spec.md](benchmark_spec.md) §7.1)
measured the four anchors that unblocked §3.3 calibration:

| Anchor | Pre-v0.6 working assumption | v0.6 empirical (122 g cube, HOLD-avg) |
|---|---|---|
| L (mean pen depth) | 5.0 mm | **0.12 mm** (÷42) |
| contact_fraction | 0.3 | **1.0** (×3.3) |
| A_patch | 1140 mm² | **330 mm²** (÷3.4) |
| F_per_pad | 1.5 N estimate | **67.9 N** (real wrench) |

The empirical numbers drive `ke_pad_physical = E·A/L ≈ 1.38×10⁶`
(×12 from working assumption) and `kh = E/L ≈ 4.17×10⁹ Pa/m`
(×42).  Calibration locks at these values for the squeeze-sweep
benchmark.

**Wrench-readout breakthrough.**  `state.mujoco.qfrc_actuator`
(pre-allocated; see the boilerplate in benchmark_spec.md §7.1) is
the route to per-pad actual contact force at the joint level.
`Contacts.rigid_contact_force` exists in the buffer but is NEVER
populated by MuJoCo's solver path -- a pitfall worth flagging for
anyone debugging force measurement.  With qfrc_actuator the
prior `Σ stiffness × L` proxy is documented as wrong: it counts
lattice-internal spring forces (anchor + lateral), not the NET
contact wrench transmitted to the cube.  Newton III is the test:
the proxy violated it by 26× across pads at the same physical
equilibrium; qfrc_actuator gives balanced forces (0.3% residual).

**Mass dependence of asymmetry.**  At 5.75 g cube (density
368 kg/m³) the cube deterministically settles +2-3 mm off-center
across seeds -- both pads' lattices have the same Fibonacci-spiral
geometry (rotated, not mirrored; see "Pad construction note"
below), and the small force imbalances perturb the lightweight
cube enough to bias its rest position.  At 122 g (steel,
7800 kg/m³) the asymmetry disappears: obj_x ≈ 0, L_left = L_right,
F_left = F_right within 0.3%.  Benchmark mass is locked at 122 g
for this reason.

**Pad construction note.**  Pads are constructed via ROTATION,
not mirroring -- box-pad right shape gets a 180° rotation about
z; dome-pad left/right get ±π/2 about y.  Consequence for dome
pads: lattice y-biases preserve direction under both ±π/2
rotations and **ADD across pads** (this is Step 10d's y-drift
mechanism); lattice x-biases map to opposite world-z under the
two rotations and CANCEL in world x.  The deterministic +2-3 mm
cube x-bias at light mass therefore has a non-trivial origin
(not a simple lattice x-bias added across pads).  Root-cause
investigation deferred to C3+.

### §7.4 squeeze-depth pilot — non-monotonic F(depth) [LANDED, PENDING v0.8 RE-VALIDATION]

> **v0.8 status note (2026-05-23)**: pilot below ran at R_pad = 20 mm
> (same retracted geometry as §7.1).  The non-monotonic F(d) shape —
> peak ~69 N at 1 mm depth, falling to 6.6 N at 5 mm — may be a
> patch-overflow artifact rather than the anchor-pull-back mechanism
> originally attributed.  At the retracted geometry the contact patch
> exceeded the box face, so spheres at deep squeeze were sliding off
> the face edge (losing contact) rather than over-compressing.  The
> v0.8 re-pilot at R_pad = 10 mm will discriminate between three
> outcomes: (a) non-monotonicity survives → anchor-pull-back confirmed
> as a real CSLC characteristic; (b) F(d) becomes monotonic → the
> v0.6 finding was geometric artifact; (c) shape changes substantially
> → mixed mechanism, refine the story.  Defer interpretation until
> the re-pilot lands.

Pilot at 122 g cube, n=1, squeeze depths {0.2, 0.5, 1.0, 2.0,
5.0} mm.  Light-side `qfrc_actuator` at HOLD:

| depth (mm) | F_light (N) | obj_x (mm) | F/F_min | held |
|---|---|---|---|---|
| 0.2 | 43.3 | 0.0 | 36× | YES |
| 0.5 | 52.7 | 0.0 | 44× | YES |
| 1.0 | **69.2** (peak) | 0.0 | 58× | YES |
| 2.0 | 20.3 | +2.7 | 17× | YES |
| 5.0 | 6.6 | −0.8 | 5.5× | YES |

**Two findings**:

1. **F is non-monotonic in squeeze depth.**  Peaks around 1 mm
   then drops.  Mechanism (v0.7 refinement): likely the **anchor-
   spring pull-back** -- at deeper pad penetration the lattice
   anchors stretch past their effective stiffness range and start
   pulling pad spheres BACK toward their rest positions, which
   reduces the net contact wrench transmitted to the cube.  The
   `smooth_step(pen)` gate-saturation effect alone is too small to
   explain the >10× force collapse; the anchor-pullback story
   matches the magnitude better.  Practical implication: "more
   squeeze = more force" intuition fails past ~1 mm; the benchmark
   must report squeeze-depth → force curves per model, not assume
   monotonicity.  This is a CSLC-characteristic finding worth
   reporting as a paper-grade observation.

2. **Asymmetry returns at depth 2 mm** even at 122 g cube
   (obj_x = +2.7 mm).  An intermediate-squeeze stability mode
   where the cube gets pushed off-center.  A real CSLC
   characteristic, not an artifact.

For the headline benchmark: at 122 g cube, all sweep cells in
{0.2 .. 5.0 mm} succeed; the grip-success transition is below
0.2 mm.  Revised proposed range: `{0.025, 0.05, 0.1, 0.2, 0.5}` mm.

### §3.4 series-spring coupling — empirically does NOT trip at the operating point [v0.7]

v0.5/v0.6 flagged a worry that `kc_series = kc · target_ke /
(kc + target_ke)` would couple the "physical" and "constraint"
ke knobs even after the ke-split.  At the v0.6 empirical anchors
(silicone target, 122 g cube), per-sphere `kc ≈ 15,000 N/m` and
`target_ke = 500,000 N/m`, so `kc_series = kc · target_ke /
(kc + target_ke) ≈ kc · 0.97`.  **The coupling reduces kc_series
by ~3% at the operating point** -- well below measurement noise
and not material to the benchmark conclusions.  The §3.4 caveat
is technically true (the physics IS coupled) but practically
inert at this operating point.  Document in the paper write-up
as "series-spring coupling exists but is small in the studied
regime"; defer the 1a refactor (true physical-vs-numerical
separation requiring a rigid-contact-buffer schema change) until
empirical data demands it.

## File map

```
cslc_main/theory/
├── __init__.py
├── cslc_theory.py                  # single-sphere primitives + friction (steps 1, 4); INACTIVE_RAW_EPS_FACTOR, INCLUSION_FACTOR
├── cslc_lattice.py                 # lattice + chain + contact (steps 2, 3, 5, 8)
├── cslc_box.py                     # box-target sampler used by point-set scenes (test_07 H/I, C2e tests)
├── cslc_projection_experiment.py   # exploratory tangent-projection variants (Step 10 follow-up; sidecar)
├── kernel_bridge.py                # theory-side shim feeding test_07 (steps 7 + C2a/c)
├── test_01_single_sphere.py        # step 1 driver (PASS)
├── test_02_chain.py                # step 2 driver (PASS)
├── test_03_chain_contact.py        # step 3 driver (PASS)
├── test_04_friction.py             # step 4 driver (PASS)
├── test_05_arc_contact.py          # step 5 driver (PASS)
├── test_06_anisotropic_anchor.py   # step 6 driver (PASS)
├── test_07_kernel_bridge.py        # kernel-vs-theory bridge (27/27 PASS) + constant-discipline + N-sweep
├── test_08_dome_contact.py         # step 8 dome + sphere indenter (PASS)
├── test_09_dome_grip.py            # step 9 dome friction budget (PASS)
├── test_10_pad_vs_box.py           # step 10 box vs dome pad characterisation
├── test_10b_symmetry_projection.py # step 10b tangent-projection variants
├── test_10c_lattice_projection.py  # step 10c lattice asymmetry projection
├── test_10d_wrench_convergence.py  # step 10d Fibonacci y-drift characterisation
├── benchmark_spec.md               # v0.7 contact-model benchmark spec (CSLC vs hydro vs point)
├── notes.md                        # this file
└── figures/                        # PNGs + diagnostic text tables
```

Run any test with:
```
uv run -m cslc_main.theory.test_NN_*
```

## Regression sweep

Exact commands used to verify Step 7 landed cleanly.  Three independent
checks; each is a self-contained no-arg invocation that prints a
verdict on its final line.

**1. Kernel-vs-theory bridge** — 9 scenes × 3 kc/ka = 27 runs against
the theory's L-BFGS-B / per-pad-force gold references, plus two
front-loaded guards (constant-discipline + N-pad sweep regression).

```bash
uv run --extra dev -m cslc_main.theory.test_07_kernel_bridge
```
Expect: `SUMMARY: 27/27 scenes pass at per-scene tol in {5e-06, 0.001, 0.02}`
(scenes A-G at tol 1e-3, scene G slip at 2e-2, scenes H + I (point-set
kernels) at tol 5e-6).  Constant-discipline guard reports 4 kernel
literal sites match `INACTIVE_RAW_EPS_FACTOR`.  N-sweep regression
reports active count constant at ~31 over N in {50, 200, 800}.

**2. Theory steps 1-6 + 8-9** — no-regression check on the gold reference
itself.

```bash
for t in 01_single_sphere 02_chain 03_chain_contact \
         04_friction 05_arc_contact 06_anisotropic_anchor \
         08_dome_contact 09_dome_grip; do
    uv run --extra dev -m cslc_main.theory.test_${t}
done
```
Each prints `Step N <name> test: PASS` on its final line.  Step 8
takes ~10 s; Step 9 takes ~6 s (both run multi-contact L-BFGS-B at
N = 150 spheres at multiple phi values).

**3. Grasp box pad end-to-end** — 3250-step APPROACH → SQUEEZE →
LIFT → HOLD on a tennis-ball-sized sphere with box pads + CSLC +
MuJoCo solver.

```bash
uv run --extra importers -m cslc_main.grasp.main \
    --pad-kind box --no-timestamp --run-label _regression
```
Expect: `RESULT  max_z=0.1000  final_z=0.0534  lifted=YES  held=YES  xy_slip_max=0.05mm`.

**4. C2 sphere-target bit-identical** — dome+sphere production grasp
must produce bit-identical CSVs before and after C2 ships.  Used
during C2d/C2e/ke-split landings to confirm the sphere-target
dispatch path stayed structurally untouched.

```bash
uv run --extra importers -m cslc_main.grasp.main \
    --pad-kind dome_param --pad-r-pad 0.020 --pad-half-angle 72 \
    --no-timestamp --run-label dome_curved_flat
md5sum outputs/grasp/dome_curved_flat/*.csv
```
Expect (post-C2 stable hashes):
`dd85b4ec8638437d6d4f34a18fca1912  cslc_state.csv`
`76e486638fb2fa538b786b3233880c60  timeseries.csv`
Hashes drift only if the sphere-target dispatch path changes; if
they shift unexpectedly after a refactor, the change broke
sphere-target physics.

All four together take ~6-8 minutes on a single GPU.  Outputs land in
`outputs/grasp/_regression/` (run 3) and `outputs/grasp/dome_curved_flat/`
(run 4); the theory tests write figures to [figures/](figures/).
