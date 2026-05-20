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
  anchor.  All gradients FD-verified, sign-regression guards on every
  vector-quantity test, 6 driver scripts that rerun with
  `uv run -m cslc_main.theory.test_NN_*`.
* Step 7 (theory-align the kernels) **landed**: both `jacobi_step`
  (lattice equilibrium solve) and `write_cslc_contacts` (MuJoCo
  emission) now obey the series-spring + deformed-centre formulation
  this theory verified.  Bridge passes **21/21** (face-on,
  off-axis, chain, arc-bulge, anisotropic anchor, friction stick,
  friction slip × kc/ka ∈ {0.1, 1, 10}); grasp box pad regression
  unchanged.

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

**Result: 21/21 PASS.**  G uses a relaxed 2% tolerance to absorb the
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

### Step 8 — Grasp integration [NEXT]

Run the two-finger grasp harness
([cslc_main/grasp/README.md](../grasp/README.md)) on the
theory-aligned kernels and compare slip / hold metrics against the
pre-fix baseline.  Box pad already verified (held=YES,
xy_slip=0.06mm); dome pad's hold failure is a separate calibration
issue (object slips out even at fix-point delta, independent of
Step 7).

## File map

```
cslc_main/theory/
├── __init__.py
├── cslc_theory.py                  # single-sphere primitives + friction (steps 1, 4)
├── cslc_lattice.py                 # lattice + chain + contact (steps 2, 3, 5)
├── kernel_bridge.py                # theory-side shim feeding test_07 (step 7)
├── test_01_single_sphere.py        # step 1 driver (PASS)
├── test_02_chain.py                # step 2 driver (PASS)
├── test_03_chain_contact.py        # step 3 driver (PASS)
├── test_04_friction.py             # step 4 driver (PASS)
├── test_05_arc_contact.py          # step 5 driver (PASS)
├── test_06_anisotropic_anchor.py   # step 6 driver (PASS)
├── test_07_kernel_bridge.py        # step 7 kernel-vs-theory bridge (21/21 PASS)
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

**1. Kernel-vs-theory bridge** — 7 scenes × 3 kc/ka = 21 runs against
the theory's L-BFGS-B gold reference.

```bash
uv run --extra dev -m cslc_main.theory.test_07_kernel_bridge
```
Expect: `SUMMARY: 21/21 scenes pass at per-scene tol in {0.001, 0.02}`
(the 0.02 is scene G slip-mode; see [test_07 docstring](test_07_kernel_bridge.py)
for why).

**2. Theory steps 1-6** — no-regression check on the gold reference
itself.

```bash
for t in 01_single_sphere 02_chain 03_chain_contact \
         04_friction 05_arc_contact 06_anisotropic_anchor; do
    uv run --extra dev -m cslc_main.theory.test_${t}
done
```
Each prints `Step N <name> test: PASS` on its final line.

**3. Grasp box pad end-to-end** — 3250-step APPROACH → SQUEEZE →
LIFT → HOLD on a tennis-ball-sized sphere with box pads + CSLC +
MuJoCo solver.

```bash
uv run --extra importers -m cslc_main.grasp.main \
    --pad-kind box --no-timestamp --run-label _regression
```
Expect: `RESULT  max_z=0.1000  final_z=0.0534  lifted=YES  held=YES  xy_slip_max=0.05mm`.

All three together take ~5-7 minutes on a single GPU.  Outputs land in
`outputs/grasp/_regression/` (run 3 only); the theory tests write
figures to [figures/](figures/).
