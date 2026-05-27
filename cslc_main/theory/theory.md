# CSLC Contract v2 — Unified Half-Space Contact

## 1. Scope and motivation

The pre-v2 CSLC model represented every contactable shape as one or more
**spheres** and computed contact via the sphere-vs-sphere overlap

```
raw = (r + R) − ||q − t||                                   (v1, deprecated)
```

This form has two related problems:

1. **Non-monotone under deep penetration.**  When the pad sphere centre
   `q` crosses the target surface and travels into the body interior,
   `||q − t||` first shrinks then grows, so `raw` is non-monotone and
   eventually flips sign.  The model bakes in an *outside-only*
   assumption that is silently violated by any solver that takes a step
   past the surface.

2. **`R` (target sample radius) has no physical meaning on a mesh.**  A
   mesh sample is a `(position, normal, area)` triple representing a
   patch of surface.  There is no "ball" around it — `R` was a
   discretisation hack (typically `R = pad_spacing/2`) to soften the
   point-vs-sphere contact.

v2 replaces this with the **signed half-space overlap**

```
raw = r − n̂_face · (q − t)                                  (v2)
```

monotone in penetration depth at any depth (no sign flip), and naturally
generalises across pad and target geometries:

* The pad lattice is *any* sampling of contact spheres on the pad
  surface (dome cap, flat grid, mesh — same code path).
* The target is *any* `PointSetTarget` with face normals (sphere,
  box, mesh — same code path).
* The contact is between pad spheres of radius `r_i` (compliance skin
  thickness) and target half-space face elements.

This also lets us delete:
- the **distance-preserving lateral law** and the **Poisson bulge
  window** (§5 of theory.txt) — the production code already uses graph-
  Laplacian everywhere; the only consumer was an academic test;
- the sphere-target Warp kernels (`jacobi_step`, `write_cslc_contacts`,
  `compute_cslc_penetration`) — superseded by their `*_point_set`
  twins after renaming.

---

## 2. Symbols and conventions

| Symbol | Meaning | Notes |
|---|---|---|
| `p_i` ∈ ℝ³ | Rest position of pad lattice sphere `i` | body-local |
| `r_i` ∈ ℝ₊ | Pad sphere radius (compliance skin thickness) | typically `pad_spacing/2` |
| `n̂_i` ∈ ℝ³ | Pad sphere outward unit normal | rest, body-local |
| `δ_i` ∈ ℝ³ | Displacement of pad sphere `i`'s centre | `δ_n > 0` ⇒ compressed inward |
| `q_i` ∈ ℝ³ | Deformed pad centre | `q_i = p_i − δ_i` |
| `t_j` ∈ ℝ³ | Target sample point `j` | target body-local |
| `n̂_j` ∈ ℝ³ | Target sample outward face normal | target body-local, unit |
| `A_j` ∈ ℝ₊ | Target sample Voronoi area on underlying surface | [m²] |
| `k_a`, `k_l`, `k_c` | Anchor, lateral, contact stiffness | [N/m] (k_c units depend on area conv., see §11) |
| `ρ = k_{a,t}/k_a` | Tangent anchor ratio | 1.0 isotropic, 1/3 incompressible |
| `μ`, `k_stick` | Coulomb coefficient, stick spring stiffness | unchanged from v1 |
| `ε` | Smoothing width | production default `5×10⁻⁴ m` |
| `N(i)` | Lattice neighbours of `i` | undirected edge graph |

**Sign convention.**  `q_i = p_i − δ_i`, so for any energy `E(δ)` the
physical force on pad sphere `i` (acting at `q_i`) equals `+∂E/∂δ_i`,
and the kernel-side *load* on `δ_i` equals `−∂E/∂δ_i`.  These two
differ by sign; we flag the sign at each appearance.

---

## 3. Geometry primitives

### 3.1 Deformed centre

```
q_i = p_i − δ_i                                              (eq:def-centre)
```

### 3.2 Half-space raw overlap (the v2 core)

For pad sphere `i` and target sample `j` with outward face normal `n̂_j`:

```
raw_ij = r_i − n̂_j · (q_i − t_j)                            (eq:raw)
```

Equivalently `raw_ij = r_i + n̂_j · (t_j − q_i)` — useful when reasoning
about MuJoCo's solver_pen reconstruction in §10.

**Monotonicity (regression guarded by T-A, T-B).**  Fix `n̂_j` and `t_j`;
let `q_i` vary along the `−n̂_j` direction (i.e. deeper into the target
body).  Then `n̂_j · (q_i − t_j)` strictly decreases, so `raw_ij`
strictly increases.  No sign flip ever; deep-penetration is well-defined.

### 3.3 Tangential distance

For the locality kernel `w_t` below:

```
d_t_ij = || (q_i − t_j) − [n̂_j · (q_i − t_j)] n̂_j ||         (eq:dt)
```

i.e. the in-face-plane distance between `q_i` and `t_j`.

### 3.4 Smooth surrogates

```
σ_ε(x) = ½ (x + √(x² + ε²))                                  (eq:sigma)   smooth ReLU
Σ_ε(x) = ½ (1 + x / √(x² + ε²))                              (eq:Sigma)   smooth step
```

with the chain-rule identity `σ'_ε = Σ_ε`.

### 3.5 Smoothed quantities

```
φ_eff_ij = σ_ε(raw_ij)                                       (eq:phi-eff)
gate_ij  = Σ_ε(raw_ij)                                       (eq:gate)
w_t_ij   = Σ_ε(3 r_i − d_t_ij)                               (eq:w_t)
```

`w_t_ij` is the **tangential locality kernel**.  Kernel half-width
`3 r_i` exceeds typical pad spacing so the contact patch is always
sampled by at least a handful of target points; below that `w_t` falls
smoothly to zero, preventing a pad sphere from coupling to far-away
samples (e.g. samples on the opposite face of a box).

### 3.6 Active-set culling and the alignment gate

Each `(i, j)` pair enters the contact energy with a multiplicative
**alignment gate** `a_ij` ∈ [0, 1]:

```
α_ij  =  −(n̂_face_j · n̂_pad_i)                              (>0 = opposing)
a_ij  =  smoothstep(α_ij; ε_align)                           (eq:align-gate)
```

where `smoothstep(α; ε)` is the C¹ cubic Hermite step on the
**one-sided** band `[0, +ε_align]`

```
smoothstep(α; ε)  =  3 t² − 2 t³,
   t  =  clip( α / ε,  0,  1 )                               (eq:smoothstep)
```


The default `ε_align = 0.05` (≈ 2.87° angular full-transition from
α = 0 to α = +ε_align) is matched to typical pad-lattice and
target-sampling angular resolutions.  The energy / force from §4 carries the gate as a
per-pair factor:

```
E_ij  =  ½ k_c · A_j · w_t_ij · a_ij · φ_eff_ij²
F_ij  =  + k_c · A_j · w_t_ij · a_ij · φ_eff_ij · gate_ij · n̂_face_j
```

(see §4 for sign conventions).  Setting `ε_align → 0` recovers the
hard step `a_ij = 1` if `α > 0` else `0`, which is **discontinuous**
at `α = 0` and breaks the smooth-energy structure of the contract;
this mode is supported for ablation testing only.

A pair `(i, j)` is **inactive** (skipped without numerical error) iff
EITHER:

```
raw_ij  <  −50 ε                                             (eq:inactive)
```

OR the alignment gate is exactly zero (`α_ij ≤ 0`, equivalent to
`a_ij = 0` by eq:align-gate's one-sided compact support — covers
both perpendicular and back-to-back orientations).

**Why the alignment gate is required.**  The half-space overlap
`raw_ij = r − n̂_face · (q − t)` treats every target sample as a
half-plane stretching to infinity in the −n̂_face direction.  For a
closed convex target (sphere, box, mesh), far-side AND side-face
samples — whose outward face normals point AWAY from or perpendicular
to the pad — would otherwise be treated as deeply in contact: the
pad's deformed centre IS inside their half-plane, but those samples
are on the *wrong side* of (back-face) or *parallel to* (side-face)
the target body and do not represent surfaces facing the pad.  The
one-sided gate `a_ij` suppresses these spurious contacts: perpendicular
and back-side hard-culled, with smooth C¹ taper only on the face-on
edge of the band.  On flat targets and near-faces of convex targets
the pad sees, `α_ij` is well above `+ε_align` everywhere, `a_ij = 1`
identically, and the gate is invisible.

At this threshold `σ_ε(raw) ≈ 0.005 ε` and `Σ_ε(raw) ≈ 1.0×10⁻⁴`
*individually*; neither factor is small on its own.  What justifies
the cull is the **product** that enters every force / energy term:

```
σ_ε(−50ε) · Σ_ε(−50ε) ≈ 5×10⁻⁷ ε                            (eq:cull-product)
```


---

## 4. Single-pair contact

For one (pad sphere `i`, target sample `j`) pair, the **contact energy
contribution** is

```
E_ij = ½ k_c · A_j · w_t_ij · a_ij · φ_eff_ij²               (eq:E-ij)
```

where `a_ij` is the alignment gate from eq:align-gate.  The **load**
on `δ_i` (kernel convention, `load = −∂E/∂δ`) is

```
f_load_ij = −k_c · A_j · w_t_ij · a_ij · φ_eff_ij · gate_ij · n̂_j  (eq:f-load)
```

i.e. opposite to the target's outward face normal.  The corresponding
**physical force on pad sphere `i`** (force on `q_i`) is

```
f_phys_ij = +k_c · A_j · w_t_ij · a_ij · φ_eff_ij · gate_ij · n̂_j
```

i.e. along `+n̂_j` — the target pushes the pad outward along the target
face's outward normal, regardless of where the pad's own normal points
or whether the pad center has crossed the face.  This is the
half-space's signature: contact direction is **face-defined, not
line-of-centres**.

**Why `gate_ij` appears in the load.**  `φ_eff_ij = σ_ε(raw_ij)`, so
the chain rule gives

```
∂E_ij/∂δ_i  =  k_c · A_j · w_t_ij · a_ij · φ_eff_ij · σ'_ε(raw_ij) · ∂(raw_ij)/∂δ_i
             =  k_c · A_j · w_t_ij · a_ij · φ_eff_ij · Σ_ε(raw_ij) · n̂_j
```

---

## 5. Total contact on a pad sphere

Per pad sphere `i`, sum over all target samples:

```
F_contact_i = Σ_j  k_c · A_j · w_t_ij · a_ij · φ_eff_ij · gate_ij · n̂_j     (eq:F-contact-i)
```

(physical force; load form is `−F_contact_i`).  Active-set culling
(eq:inactive plus `a_ij > 0`) restricts the sum to pairs that are
both geometrically nearby and on the correct side of the target face.

---

## 6. Lattice equilibrium

The total energy on the pad lattice is

```
E_tot(δ_{1..N})  =  Σ_i E_anchor(δ_i)
                  +  E_lateral(δ_{1..N})
                  +  Σ_{i in S} Σ_j E_ij(δ_i)
                  +  Σ_{i in S} E_friction_i(δ_i)
```

where `S` is the set of surface pad spheres.  Quasistatic equilibrium:
`∂E_tot/∂δ_i = 0 ∀i`.

### 6.1 Anchor (anisotropic)

In pad sphere `i`'s local `{n̂_i, n̂_i^⊥}` frame:

```
E_anchor_i  =  ½ k_a δ_n²  +  ½ (k_a ρ) ||δ_t||²            (eq:E-anchor)
∂E_anchor_i/∂δ_i  =  k_a δ_n n̂_i  +  (k_a ρ) δ_t           (eq:f-anchor)

  where  δ_n = δ_i · n̂_i,  δ_t = δ_i − δ_n n̂_i
```

`ρ = 1` recovers isotropic.

### 6.2 Lateral (graph-Laplacian only)

```
E_lateral  =  ½ k_l Σ_{(i,j) ∈ E} ||δ_i − δ_j||²            (eq:E-lat)
∂E_lateral/∂δ_i  =  k_l Σ_{j ∈ N(i)} (δ_i − δ_j)           (eq:f-lat)
```
Consequence: curvature-driven Poisson bulging (v1 §5, test_05's
*outward bulge window*) **does not occur**.  Curved pad lattices
(dome) still *flatten* against a target — that's the kinematic
consequence of the anchor+contact balance, independent of the lateral
law — but no surface sphere develops `δ_n < 0` under static contact.
T-G is the regression guard.

### 6.3 Contact

See §5.  Per-sphere load: `−F_contact_i`.

### 6.4 Friction

Unchanged (stick-slip via smooth `f_t = K M s /
(K s + M)` with `K = k_stick`, `M = μ f_n`).  Two refinements:

* `f_n = |F_contact_i · n̂_i|`  — the aggregate normal-axis magnitude
  of `F_contact_i` projected on the pad's *own* outward normal.
  Unchanged from the existing point-set kernel.
* The local frame for `δ_t` is the pad sphere's `n̂_i`, not the
  per-target `n̂_j` — friction is a property of the pad's compliant
  skin, which has one outward normal per lattice sphere.

### 6.5 Equilibrium per sphere

Written as `∂E_tot/∂δ_i = 0` — every term is `+∂E_X/∂δ_i` (physical
force on `q_i` per §2), all signs positive:

```
0  =  + k_a δ_{n,i} n̂_i + (k_a ρ) δ_{t,i}            (anchor,  eq:f-anchor)
      + k_l Σ_{j∈N(i)} (δ_i − δ_j)                    (lateral, eq:f-lat)
      + F_contact_i                                   (contact, eq:F-contact-i)
      + f_friction_i                                  (friction, §6.4 grad form)
                                                         (eq:equilibrium)
```


---

## 7. Target abstraction

A target is a `PointSetTarget(positions, normals, areas)`.  No radii.
Per-sample area `A_j` is the Voronoi area of sample `j` on the
underlying surface; it absorbs sampling density so the per-pair sum in
§5 approximates the surface integral `∫ k_c φ n̂_face dA`.

### 7.1 Concrete samplers (in `cslc_main/theory/cslc_targets.py`)

```python
make_flat_face_target(centre, normal, span_u, span_v, pitch)
    # Regular grid on a flat face.  Used by every theory test below.

make_sphere_target(t_centre, R, n_samples)
    # Fibonacci spiral on the sphere surface.  Each sample's normal
    # is the outward radial direction at the sample point.  Each
    # sample's area is 4πR²/n_samples (uniform-area spiral).

make_box_target(extents, n_samples, *, center=None, seed=0)
    # Per-face area-weighted random sampling via trimesh.
    # Per-sample normal is the source face's outward normal.
    # Per-sample area is total_box_surface_area / n_samples.

make_mesh_target(mesh, n_samples, *, seed=0)
    # Generic trimesh sampler.  Per-sample normal = face normal at
    # the triangle the sample landed on.  Voronoi areas approximated
    # as total_mesh_surface_area / n_samples (uniform-density limit).
```

**No `radii` arg anywhere.**  This is the v2 break with v1.

## 9. Sample numbers (for grounding)

At production defaults (`r_pad = 1.5 mm`, `pad_spacing = 3 mm`,
`R_object = 33.5 mm` tennis ball, `n_samples_object = 1500` Fibonacci):

```
A_j_sphere   =  4π R² / n_samples   ≈  4π (33.5e-3)² / 1500 ≈ 9.4e-6 m²
mean spacing ≈  √A_j_sphere         ≈  3.1e-3 m  ≈ pad_spacing  ✓ (matched)
```

so the sample-density choice is "match target spacing to pad spacing".
This is a configuration default in `cslc_main/grasp/params.py`, not
something the contract pins down.

---

## 10. Calibration

Unchanged from v1.  The series-spring identity

```
1/k_c  =  N_contact / k_e_bulk  −  1/k_a  −  1/k_e_target    (eq:calibration)
```

(implemented in `cslc_data.calibrate_kc`) holds because `R` and the
target geometry do not appear in it — `k_c` is per-pair-stiffness,
`N_contact` is the number of *pad* spheres engaged.  Half-space vs
sphere-vs-sphere doesn't change which pad spheres engage at a given
penetration, so `N_contact` is unchanged.  Sanity-checked by T-Q.

---

## 11. What is preserved exactly

Implementation may not assume these change:

* All sign conventions (q = p − δ; load = −∂E/∂δ).
* `INACTIVE_RAW_EPS_FACTOR = −50.0`, `INCLUSION_FACTOR = 50.0`,
  `ε_default = 5e-4`.
* Anisotropic anchor decomposition (eq:E-anchor).
* Friction smooth surrogate (`f_t = K M s / (K s + M)`,
  `E_f = M s − M²/K · ln(1 + K s / M)`).
* Damped Jacobi α = 0.3 default; warm-start `(K + k_c·I) δ = k_c φ_rest`.
* Lattice graph topology (`neighbor_indices`, `neighbor_counts`).
* `kc_series = kc · ke_target / (kc + ke_target + ε²)` composition in
  the emitted MuJoCo stiffness.<sup>†</sup>

<sup>†</sup> **Dimensional caveat (legacy v1).**  `kc` and `ke_target`
are stiffnesses `[N/m]` but `ε` is a length `[m]`, so the `+ ε²` term
adds `[m²]` to a `[N/m]` denominator — dimensionally inconsistent.  In
v1 this was a small-`ε` numerical guard against `kc + ke_target → 0`;
at production scales the term is negligible (`ε² = 2.5×10⁻⁷ m²` vs
`kc + ke_target ~ 10⁴ N/m`) and the units mismatch is masked.  v2
preserves the formula verbatim to avoid changing kernel output, but
a follow-up could replace `ε²` with an explicitly typed
`ε_stiffness² [N²/m²]` regulariser without changing any numerical
result at production parameters.

---

## 12. Test catalog (gates for each phase)

Phases run sequentially; a phase is *done* when all its tests pass at
the listed tolerance.  Tolerances are absolute on δ vectors (per-sphere
worst case), relative on aggregate forces.

### Phase 1 — Half-space primitives + single-pad-sphere tests  **[COMPLETE, 4/4 PASS]**

| ID | File | Asserts | Tol | Status |
|---|---|---|---|---|
| **T-A** | `test_half_space_monotonicity.py` (new) | Sweep `q` from `+10r` outside to `−10r` inside a flat face.  `raw(q)` strictly increasing; `phi_eff(q)` non-decreasing; `gate(q)` non-decreasing.  No sign flip. | exact | **PASS**.  Linear-fit deviation `6.8×10⁻¹⁸ m`; 1 sign crossing on v2 (none on v1, which has the inverted-V regression). |
| **T-B** | `test_half_space_deep_penetration.py` (new) | `q` at `−10r` inside body.  Force finite, monotone in depth; gate = 1; no NaN. | exact | **PASS**.  v2 force monotone to depth = 10·r (= 25 mm); gate saturated at 0.999900 by raw ≥ 50·ε; v1 force drops to 0 past r+R cliff (regression target). |
| **T-C** | `test_face_on_series_spring.py` (new, replaces `test_01_single_sphere.py`) | Pad sphere vs one flat point.  Closed form `δ_n = k_c d / (k_a + k_c)` where `d = r − n̂·(p−t)` at δ=0.  Series-spring linear in `d`.  **Part D (added):** production-eps precision floor as function of `raw_eq / ε`. | δ: `2e-10`; F: `1e-13 N` | **PASS**.  Part A worst: `1.1×10⁻¹⁹ m`, `1.4×10⁻¹⁴ N`.  Parts B/C verified asymptotes + linearity.  Part D: 1% rel err bound holds at `raw_eq ≥ 10·ε`. |
| **T-D** | `test_tilted_face.py` (new, replaces `test_06_anisotropic_anchor.py`) | Same pad sphere, face tilted by `θ ∈ {5°, 15°, 30°, 45°}`.  **v2 anisotropic tilt formula** (exact, no `A` correction): `\|δ_t/δ_n\| = (1/ρ) tan(θ)` for `ρ ∈ {1, 1/2, 1/3, 0.1}`.  The v1 (sphere-vs-sphere) form had an `A = k_c φ_eff / L` correction because the line-of-centres direction rotated with `δ`; v2's face normal is `δ`-independent, so the correction vanishes.  Empirical check: v1 formula is wrong by up to 21% at `ρ = 0.1`; v2 formula matches numerical to ~1e-9 rel err. | rel `1e-7` | **PASS**.  Worst rel err `2.2×10⁻⁸` across full ρ×θ grid.  v1 (with `A`) confirmed wrong by 13% at `ρ = 0.1`. |

### Phase 2 — Lattice solver + dome-flattens regression  **[COMPLETE, 4/4 PASS]**

| ID | File | Asserts | Tol | Status |
|---|---|---|---|---|
| **T-E** | `test_02_chain.py` (in-place edit) | Chain anchor + graph-Laplacian lateral.  K matrix tridiagonal; eigenvalues match `λ_k = k_a + 2 k_l (1 − cos kπ/N)`; Green's function fits discrete decay.  *DP comparison sections deleted; new Part D' verifies GL axes-decouple at k_l/k_a ∈ {0.2, 1, 10}.* | rel `4e-16` | **PASS**.  Eigenvalues to `4×10⁻¹⁶` rel err; Green's function fits discrete decay; GL axes decouple exactly (`0` off-axis leak at all three k_l/k_a ratios). |
| **T-F** | `test_chain_contact_flat.py` (new, replaces `test_03_chain_contact.py`) | Chain pressed onto flat face.  Sherman-Morrison `δ_n_k = k_c d g_kk / (1 + k_c g_kk)`; force balance `Σ_i F_i = k_c (d − δ_n_k)`. | rel `1e-7` (Phase-2 empirics: L-BFGS-B on multi-sphere problems floors at ~5×10⁻⁸ — earlier `1e-12` assumed a direct linear solve that the v2 nonlinear solver does not perform; direct-linear path retained as reference inside the test for fp-precision comparison) | **PASS**.  Sherman-Morrison worst `5.2×10⁻⁸` (within tol); force-balance worst `5.2×10⁻⁸`; isolated limit and N-anchors-in-parallel limit both within 5%.  Note: Part C's "rigid pad" limit is `k_c·d·(N·k_a)/(N·k_a + k_c)`, NOT `k_c·d` — the chain has finite anchors so contact saturation is bounded. |
| **T-G** | `test_dome_flattens.py` (new, replaces `test_05_arc_contact.py` and `test_08_dome_contact.py`) | Dome lattice (Fibonacci spiral, N=150, R=10 mm, 72° cap) pressed against flat face at penetration depths `{0.1, 0.5, 1.0, 2.0} mm`, areas=None (per-pair k_c).  Assert: (a) apex has `δ_n > 0`; (b) **NO sphere has `δ_n < −1 nm`** anywhere on the lattice (regression: v1 DP gives perimeter `δ_n ≈ −33 nm` to `−1 μm` at production geometry; v2 GL gives effectively zero); (c) cross-section monotonically conforms (inner-quarter `δ_n` > outer-quarter `δ_n`); (d) `N_active_pairs` monotone in depth. | (a),(b),(d): exact; (c): inner-outer monotonicity at each depth | **PASS — centerpiece regression.**  0 bulge spheres at all 4 depths; apex sinks `956 μm` at depth = 1 mm; inner-quarter mean δ_n `304 μm` vs outer-quarter `0.25 μm` (3-decade conformity ratio); N_active monotone 1352 → 42471. |
| **T-H** | `test_dome_vs_sphere.py` (new) | Dome lattice pressed against a sphere target (R = 33.5 mm, 1500-pt Fibonacci spiral, radial normals) at penetration depths `{0.1, 0.5, 1.0, 2.0} mm`.  Verify: (a) equilibrium converges (`info.success`) with `\|∇E\| < 1e-3` at all depths (relaxed from `1e-9` — L-BFGS-B floor at 225k-pair scale is ~1e-4); (b) no NaN; (c) `N_active_pairs` monotone in depth; (d) `F_total` (anchor reaction sum) monotone in depth; (e) `δ_n_apex` monotone in depth.  **Bonus** carry-over from T-G: `min δ_n > −1 nm` at all depths (no bulge even on curved target).  **Part D** alignment-gate continuity regression (§3.6 smooth gate): sweep a single pad's outward normal across the perpendicular-to-target boundary in 1° steps; per-step ΔF < 30% of saturated F (legacy binary cull gives 100% jump at θ = 90°); F monotone non-increasing in θ. | δ converges, `\|∇E\| < 1e-3`; Part D: per-step ΔF < 30% of F_max | **PASS**.  Converges in 11–26 iters at all depths; F_total grows `5 N → 549 N`; δ_n_apex `95 μm → 1425 μm`; no-bulge carry-over verified.  Smooth-gate continuity verified across the alignment boundary. |

### Phase 3 — Friction in unified path  **[COMPLETE, 2/2 PASS]**

| ID | File | Asserts | Tol | Status |
|---|---|---|---|---|
| **T-I** | `test_friction_flat.py` (new, replaces `test_04_friction.py`) | Single pad vs flat face + tangential `F_ext`.  Stick `s = F/(k_a+k_stick)`; slip `s = (F−μf_n)/k_a`; `F_thresh = μf_n (k_a+k_stick)/k_stick`.  4 parts: (A) stick sweep; (B) slip sweep; (C) full Coulomb transition with v2 analytical vs v2 hard-piecewise and smooth-surrogate witness; (D) `k_stick` sweep across `{0.1, 1, 10, 100}·k_a`. | rel `1e-7` (relaxed from contract `1e-8` per the `scipy.optimize.minimize_scalar(method='bounded')` ~7-digit floor — same floor v1 `test_04` hit, NOT a v2-specific regression).  Smooth-surrogate vs hard is informational only (the harmonic-mean form has an inherent smoothing-zone error; cf. v1 `test_04` PART C). | **PASS**.  Stick worst `1.5×10⁻⁸` rel err; slip worst `1.5×10⁻⁸`; analytical vs hard-piecewise (Part C) max `\|s_a − s_h\|` = `8.5×10⁻¹² m` (~9 pm); slope ratio slip/stick = 2.0 exactly; F_thresh tracks predicted across full k_stick sweep. |
| **T-J** | `test_dome_grip_flat.py` (new, replaces `test_09_dome_grip.py`) | Dome (T-G scene) on flat face; tangential displacement sweep.  4 parts: (A) normal-only baseline with per-sphere `f_n,i` distribution via :func:`lattice_contact_normal_forces`; (B) grip-curve invariants — low-s slope = `N_eng · k_stick`, plateau = `μ · F_n^total`, first-slip `s = μ · f_n,min/k_stick`; (C) self-consistency of vectorised aggregation vs sphere-by-sphere loop at 4 probe `s`; (D) grip vs depth sweep `{0.1, 0.5, 1.0, 2.0}` mm. | rel `1e-6` (grip aggregation is closed-form by construction; precision floor comes from per-sphere `f_n,i` via the L-BFGS-B floor of `~5×10⁻⁸`, well inside `1e-6`). | **PASS**.  Plateau matches `μ·F_n^total` to fp precision (rel err = 0); low-s slope rel `1.2×10⁻¹⁶`; first-slip displacement within one grid step (`Δs = 7.3 μm`); sphere-by-sphere aggregation matches vectorised to `1.4×10⁻¹⁴ N` worst; F_grip^max monotone in depth `2 N → 325 N`; log-log slope `F_n^total vs depth = 1.70` (between Hertz 1.5 and parallel-anchors 2.0 — consistent with T-G/T-H regime). |

### Phase 4 — Bridge harness  **[COMPLETE: Phase 4a 6/6 PASS + Phase 4b 12/12 scenes PASS]**

| ID | File | Asserts | Tol | Status |
|---|---|---|---|---|
| **T-L** | `test_07_kernel_bridge.py` (Phase 4a) | Active-set parity at the half-space threshold AND the alignment gate.  Sweep 1: pad at three rest raws `{−60ε, −40ε, +40ε}` — both sides inactive / marginally active / saturated.  Sweep 2: pad outward normal rotates 0°..180° at `raw_rest = +30ε` — both sides smoothly taper through the alignment band `[-eps_align, +eps_align]` with no force jump and monotonic decrease past the perpendicular boundary. | Sweep 1: abs-floor `1 nm` on marginally-active case (sub-pm δ on both sides); rel `1e-3` elsewhere.  Sweep 2: rel `1e-3` + monotone non-increase in θ. | **PASS** (Phase 4a).  Both sweeps clean; smooth-gate band [-0.05, +0.05] verified continuous on both sides; alignment gate added to `jacobi_step_point_set` (contract §3.6 literal `eps_align = 0.05` hardcoded — MUST match theory). |
| **T-K** | `test_07_kernel_bridge.py` (Phase 4a vertical slice: A, D, F, H × kc/ka ∈ {0.1, 1, 10} = 12 cases) | Subset of the contract's full 10×3 matrix:  **A** (single pad face-on flat), **D** (dome lattice vs flat — T-G centerpiece including no-bulge regression), **F** (single pad + stick friction with `F_ext` deep in the stick window), **H** (single pad vs **multi-sample flat face** — Phase 4a reduction from full box; see finding #11). | rel `1e-3` (all four scenes).  Kernel and theory both at production `eps = 5×10⁻⁴` m, `areas = None` (per-pair k_c). | **PASS**.  Worst rel err: scene A `1.7×10⁻⁶`; scene D `6.6×10⁻⁴` (max across N=150 spheres at kc/ka=0.1); scene F `3.8×10⁻⁶` (after live-f_n bridge fix, finding #12); scene H `1.7×10⁻⁶`.  Constant-discipline guard PASS (4 kernel sites match `INACTIVE_RAW_EPS_FACTOR = −50.0`). |
| **T-K** Phase 4b | `test_07_kernel_bridge.py` (full matrix) | Phase 4b adds the remaining 6 scenes (B tilted flat, C chain vs flat, E anisotropic anchor + tilted, G slip friction, I chain vs box, J pad vs sphere-as-point-set) and restores scene H to the full box target (`make_box_target`).  Required contract amendment: §3.6 alignment gate shifted from symmetric `[-eps_align, +eps_align]` to one-sided `[0, +eps_align]` so perpendicular faces are HARD-culled (finding #11).  Required kernel cleanups: (a) legacy v1 sphere-overlap pre-cull `(r_i + R_j) - L < -50·eps` REMOVED from `jacobi_step_point_set` — for closed convex targets this hard-culled samples that theory keeps via the smooth `w_t` tail (finding #13); (b) damped-Jacobi post-refinement added to theory's `solve_lattice_contact` — L-BFGS-B's Wolfe line search is unreliable when `fun` includes `w_t` while truncated `jac` drops `∂w_t/∂δ`, which manifests on sphere targets (finding #14). | rel `1e-3` (scenes B-F, H, I, J); rel `2e-2` (G slip). | **PASS** (Phase 4b, all 30 cases).  Worst rel err: scene A `1.7×10⁻⁶`; scene B `1.7×10⁻⁶`; scene C `2.5×10⁻⁵`; scene D `6.6×10⁻⁴`; scene E `3.9×10⁻⁶`; scene F `3.8×10⁻⁶`; scene G `8.1×10⁻⁶` (slip); scene H `2.3×10⁻⁶`; scene I `2.2×10⁻⁶`; scene J `4.6×10⁻⁶`.  Constant-discipline guard PASS (3 raw-cull sites + 2 align-gate sites after the kernel legacy-cull removal). |

