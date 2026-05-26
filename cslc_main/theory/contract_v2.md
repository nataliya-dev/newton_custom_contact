# CSLC Contract v2 — Unified Half-Space Contact

**Status:** Phases 0–5 complete (Phase 4a vertical slice + Phase 4b
full 30-case T-K matrix, 12 scenes / 32 sub-checks PASS; Phase 5
T-L sweep 3 added — 19-point active-set membership equality probe
across the `raw = −50·eps` threshold, PASS).  Phase 5 deleted every
v1 symbol from the kernels (sphere-target triple), the handler
(``_launch_vs_sphere``, ``CSLCShapePair`` sphere fields), and the
theory module (``RigidTarget``, ``ContactTarget`` /
``SphereIndenter`` / ``PointSetIndenter``, the DP lateral law, every
``equilibrium_with_friction_*`` and ``equilibrium_face_on_analytical``
variant, ``kernel_contact_force_n_axis``, the v1
``PointSetTarget`` + ``point_set_*`` family, the orphaned
``kernel_bridge.py``); renamed the point-set kernels to drop the
``*_point_set`` suffix; dropped ``target_radii`` from every kernel
signature; set ``margin1 = 0`` in MuJoCo emission; unified the
warm-start + emission gates to match ``jacobi_step``'s contract §3.6
gates; and verified active-set membership parity at the contract
threshold via the new T-L sweep.  Phases 6 (data-layer) – 8
pending.  See §17 for the progress snapshot and key empirical
findings.  This
document is the contract every subsequent phase of the rewrite
verifies against.  It supersedes the sphere-vs-sphere formulation in
[`theory.txt`](theory.txt) (the current Overleaf source) once the
rewrite lands.  Correctness of the
rewrite is defined as "every equation here is implemented by exactly
one function, and every test in §12 passes."

---

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

— exactly 1 for `α ≥ +ε_align` (face-on; full contact), exactly 0
for `α ≤ 0` (perpendicular OR back-to-back; HARD-culled), and
smoothly interpolating on `(0, +ε_align)`.

Three key properties:

* **Compact support (back AND perpendicular)**: back-side and
  perpendicular samples contribute *exactly* zero, not an O(ε²/α²)
  polynomial tail (which is what a `Σ_ε`-style surrogate would give
  and which would over-couple far-side pairs on closed convex
  targets — see "Why the gate is required" below).
* **One-sided band**: the smoothstep is on `[0, +ε_align]`, NOT
  `[-ε_align, +ε_align]`.  An earlier symmetric form gave
  `a_ij = 0.5` at perpendicular (α = 0), which over-coupled the side
  faces of closed convex targets — corner samples sat inside the
  locality kernel with half-strength asymmetric in-plane forces that
  drove L-BFGS-B non-convergence at production eps on box/mesh
  targets (Phase 4a finding #11).  The amended one-sided form
  hard-culls perpendicular while preserving C¹ continuity on the
  face-on edge of the band.
* **δ-independent**: `n̂_pad` and `n̂_face` are rest body-local
  geometry, so `∂a_ij/∂δ ≡ 0` and the gate enters the gradient as a
  constant per-pair multiplier (no extra Jacobian term).

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

This is the v2 analogue of the v1 3-D distance gate
`(r + R) − ‖q − t‖ < −50 ε` (which depended on the target sample
radius `R` that v2 drops); the alignment gate is a geometric
reformulation that does not require `R`.

At this threshold `σ_ε(raw) ≈ 0.005 ε` and `Σ_ε(raw) ≈ 1.0×10⁻⁴`
*individually*; neither factor is small on its own.  What justifies
the cull is the **product** that enters every force / energy term:

```
σ_ε(−50ε) · Σ_ε(−50ε) ≈ 5×10⁻⁷ ε                            (eq:cull-product)
```

At production `ε = 5×10⁻⁴ m` this product is 2.5×10⁻¹⁰ m.  Multiplied
by typical `k_c · A_j · w_t ~ 10⁻² N/m` at production sampling
density, the per-pair force contribution from a culled pair is
~10⁻¹¹ N — well below the ~1–100 N range of total contact wrench, so
the cull is numerically harmless.  The constant `−50` is
`INACTIVE_RAW_EPS_FACTOR` in `cslc_theory.py` and the corresponding
kernel literal; unchanged from v1.

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

(using `∂q_i/∂δ_i = −I` and `∂(raw)/∂δ_i = +n̂_j`; `a_ij` is
δ-independent so passes through the derivative unchanged).
Forgetting `Σ_ε` is the silent gradient bug fixed in v1 step 7 and
inherited verbatim here.  Required at the smooth gate's edge; unity
at deep saturated contact.

`w_t_ij` is treated as δ-independent in this gradient.  Its actual
derivative is `∂w_t/∂δ_i = Σ'_ε(3r_i − d_t_ij) · (−∂d_t_ij/∂δ_i)`,
with `Σ'_ε(x) = ε²/(2(x²+ε²)^{3/2})`.  In the well-supported regime
where `x = 3r_i − d_t_ij ≫ ε` (i.e. samples comfortably inside the
locality kernel), `Σ'_ε ≈ ε²/(2x³)` is small — order `1.4 m⁻¹` at
`x = 3r_i = 4.5×10⁻³ m`, `ε = 5×10⁻⁴ m`.  Omitting this term is the
production kernel's convention; we match it so the bridge harness sees
identical gradients on both sides.  The error scales as `O(ε²/r_i³)`
and vanishes in the small-ε limit, so the smooth-saturated equilibrium
is unaffected at converged δ.

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

### 6.2 Lateral (graph-Laplacian only — v2 has no distance-preserving)

```
E_lateral  =  ½ k_l Σ_{(i,j) ∈ E} ||δ_i − δ_j||²            (eq:E-lat)
∂E_lateral/∂δ_i  =  k_l Σ_{j ∈ N(i)} (δ_i − δ_j)           (eq:f-lat)
```

Load form (negative gradient): `f_lat_load_i = −k_l Σ_j (δ_i − δ_j)`.
This is the linearisation of the distance-preserving spring around
`δ = 0`; the *nonlinear* distance-preserving law is **deleted in v2**.

Consequence: curvature-driven Poisson bulging (v1 §5, test_05's
*outward bulge window*) **does not occur**.  Curved pad lattices
(dome) still *flatten* against a target — that's the kinematic
consequence of the anchor+contact balance, independent of the lateral
law — but no surface sphere develops `δ_n < 0` under static contact.
T-G is the regression guard.

### 6.3 Contact

See §5.  Per-sphere load: `−F_contact_i`.

### 6.4 Friction

Unchanged in form from v1 §6 (stick-slip via smooth `f_t = K M s /
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

**Convention note.**  We write the equilibrium in **gradient form**
(all `+`) to match v1 `theory.txt §5.4`.  The kernel implements the
algebraically equivalent **load form** (`load_X = −∂E_X/∂δ`,
equation has all `−`, or equivalently `Σ_X load_X = 0`).  Either
form is correct; mixing them within a single equation flips the
sign of half the terms and breaks the balance.  When citing the
kernel code (which uses `f_load_ij = − k_c · … · n̂_j` etc.), translate
by flipping every term's sign.

Solved by damped Jacobi (unchanged α=0.3 default) with the diagonal
stabilisation `S_n = k_l |N(i)| + k_c Σ_j A_j w_t_ij gate_ij`,
`S_t = k_l |N(i)|`.  Linear warm-start via `(K + k_c·I) δ_n = …` matrix
inversion (unchanged from v1 §7.1, drops the `R` term in the rhs).

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

### 7.2 Sphere-target reduction (approximation quality)

A sphere target of radius `R` sampled via `make_sphere_target` is **not**
geometrically equivalent to a v1 `RigidTarget` of the same `R` — the
half-space form locally replaces the sphere with its tangent plane at
each sample.  The approximation is *exact* at face-on, single-sample
contact (the sample's radial normal coincides with the line of
centres) and degrades smoothly off-axis.

**Geometric error.**  For a pad sphere in contact with target samples
up to lateral distance `d_t` from each sample, the target surface
deviates from the tangent plane by `ε_geom(d_t) = d_t² / (2R)`
(sagitta).  Since `w_t` cuts off contributions beyond `d_t = 3 r_pad`,
the maximum geometric error in `raw_ij` per active pair is

```
ε_geom_max  =  (3 r_pad)² / (2R)  =  4.5 r_pad² / R          (eq:err-geom)
```

i.e. the half-space form *underestimates* the depth a curved target
samples reach at lateral offset, since the true sphere falls away
from the tangent plane.

**Validity regime.**  The approximation is fit-for-purpose when
`ε_geom_max ≪ depth`, i.e. when

```
R  ≫  r_pad² / depth                                          (eq:R-validity)
```

For production values (`r_pad = 1.5 mm`, `depth = 0.5 mm`) this needs
`R ≫ 4.5 mm` — comfortably satisfied by the tennis-ball target
(`R = 33.5 mm`, ratio ≈ 7×).  The worst-case relative error in `raw`
per pair is then `ε_geom_max / depth ≈ r_pad²/(R·depth) ≈ 13%`
**at the kernel's lateral edge**; nearer samples have much smaller
error, and the area-weighted sum suppresses the edge-pair
contribution proportionally.

For *small* targets (`R ≲ a few r_pad`) the approximation breaks down;
in that regime production should either increase target sampling
density (smaller `A_j`, smaller `kernel_w` support per pair) or fall
back to a purpose-built small-sphere kernel.  T-J pins where the
approximation is valid; T-H exercises it at production scale.

---

## 8. Emission to the rigid-body solver

`write_cslc_contacts` (renamed from `write_cslc_contacts_point_set`)
emits one MuJoCo contact per active `(pad_sphere, target_sample)` pair.
For pair `(i, j)`:

```
point0     =  q_i^def  =  p_i − δ_i                   (deformed pad centre)
point1     =  t_j
normal     =  −n̂_j                                    (from target outward to pad)
margin0    =  r_i
margin1    =  0                                        (** v2: was R_j, now zero **)
stiffness  =  k_c_series · A_j · w_t_ij · gate_ij
friction   =  μ
```

**Solver reconstruction.**  MuJoCo computes

```
solver_pen  =  margin0 + margin1 − (point1 − point0) · normal
             =  r_i + 0 − (t_j − q_i^def) · (−n̂_j)
             =  r_i + n̂_j · (t_j − q_i^def)
             =  r_i − n̂_j · (q_i^def − t_j)
             =  raw_ij                                       ✓
```

so the per-contact force MuJoCo applies, `stiffness · solver_pen`,
equals `k_c_series · A_j · w_t_ij · gate_ij · raw_ij` — algebraically
the same as `φ_eff_ij · gate_ij` in the deep-saturated limit
(`φ_eff_ij ≈ raw_ij` when `raw_ij ≫ ε`), matching the lattice solver's
converged force.

**v2 change vs v1.**  `margin1` was `R_j`; now zero.  `solver_pen`
reconstruction loses the `R_j` term; otherwise identical to the
existing point-set emission kernel.

---

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

### Phase 5 — Warp kernel parity

Same harness, run against the cleaned/renamed Warp kernels.  Same
tolerances as T-K.  Plus:

| ID | File | Asserts | Tol |
|---|---|---|---|
| **T-L** | `test_07_kernel_bridge.py` (added scene) | Active-set transition: kernel and theory both report the same set of active `(i, j)` pairs as a pad sphere is swept from `raw = −60ε` to `raw = +60ε`.  Both transitions occur at `raw = −50ε`. | exact set equality |

### Phase 6 — Handler integration

| ID | File | Asserts | Tol |
|---|---|---|---|
| **T-M** | `tests/test_cslc_handler_sphere_target.py` (new, under `newton/_src/geometry/tests/`) | Dome pad lattice + sampled sphere target through the handler.  Equilibrium force `F = k_eff · depth` within `1%` of analytical. | rel `1e-2` |

### Phase 7 — End-to-end grasp

| ID | File | Asserts |
|---|---|---|
| **T-N** | `examples/grasp/main.py --object-kind sphere` | Run to completion, held=YES, `xy_slip_max` within pre-refactor bound (≤5 mm at production defaults). |
| **T-O** | `examples/grasp/main.py --object-kind box` | Same, box object. |

### Phase 8 — Documentation regression

| ID | File | Asserts |
|---|---|---|
| **T-P** | (manual review) | `theory.txt` v2 cites every equation in §3–§10 above; every kernel name in `cslc_kernels.py` matches the names in `theory.txt`; `notes.md` has a Step 12 closure section. |
| **T-Q** | `test_calibration.py` (new) | `calibrate_kc` produces a `k_c` such that the dome-vs-flat equilibrium at the calibration depth gives `F_total = k_e_bulk · depth` within `1%`.  Verifies §10 invariance under the v2 refactor. | rel `1e-2` |

---

## 13. Deletion list

After the rewrite lands, these symbols must not exist anywhere except
in git history.  Verified by a `grep` regression in CI.

### In `cslc_main/theory/cslc_theory.py`

**Deleted outright:**
- `RigidTarget` class
- `effective_penetration`, `contact_raw_overlap`, `contact_direction`
- `contact_force`, `contact_energy`, `total_energy` (sphere-target)
- `equilibrium_face_on_analytical`, `equilibrium_numerical` (sphere-target)
- `kernel_contact_force_n_axis` (sphere-target witness)
- `PointSetTarget.radii` field
- `PointSetTarget.M = 1` reduction guard (no longer meaningful)

**Rewritten (same names, new bodies — listed here for completeness):**
- `equilibrium_with_friction_analytical`,
  `equilibrium_with_friction_hard_numerical`,
  `equilibrium_with_friction_smooth_numerical` — reborn under flat-face
  target form (single `PointSetTarget` sample with face normal).
  Signatures keep `sphere, kc, f_ext_tangent, k_stick, mu`; the old
  `target: RigidTarget` argument becomes
  `face_normal: np.ndarray, face_offset: float` (or equivalent
  single-sample `PointSetTarget`).

### In `cslc_main/theory/cslc_lattice.py`
- `ContactTarget` class
- `SphereIndenter` class
- `PointSetIndenter` class (use `PointSetTarget` directly)
- `lateral_force_distance_preserving`, `lateral_energy_distance_preserving`
- `contact_energy_face_on`, `contact_force_face_on`
- `solve_chain_contact_linear`, `solve_lattice_contact_linear`
- `solve_lattice_contact_numerical`, `solve_lattice_sphere_indenter`
- Rename: `solve_lattice_point_set_indenter` → `solve_lattice_contact`

### In `cslc_main/theory/kernel_bridge.py`
- `_solve_single_sphere`, `_solve_lattice`
- `_solve_lattice_single_overlap_legacy`, `_solve_lattice_multi_contact`
- `compute_contact_force` (sphere-target)
- `KernelScene.target_position`, `KernelScene.target_radius`,
  `KernelScene.is_point_set` (always point-set now)

### In `newton/_src/geometry/cslc_kernels.py`
- Kernels `jacobi_step` (sphere; renamed-from will replace),
  `write_cslc_contacts` (sphere),
  `compute_cslc_penetration` (sphere),
  `compute_pad_force_vs_point_set` (smoke-only helper)
- Rename: `jacobi_step_point_set` → `jacobi_step`
- Rename: `write_cslc_contacts_point_set` → `write_cslc_contacts`
- Rename: `compute_cslc_penetration_point_set` → `compute_cslc_penetration`
- Drop `target_radii` from all kernel signatures (`raw = r_i + R_j − …`
  → `raw = r_i − …`).
- Drop `margin1 = R_j` from emission (set to 0).

### In `newton/_src/geometry/cslc_handler.py`
- `_launch_vs_sphere`; rename `_launch_vs_point_set` → `_launch`
- `CSLCShapePair.is_point_set`, `CSLCShapePair.target_radius`

### Test files
- Delete: `test_05_arc_contact.py` (bulge — physics removed)
- Rename + rewrite: `test_01_single_sphere.py` → `test_face_on_series_spring.py`
- Rename + rewrite: `test_03_chain_contact.py` → `test_chain_contact_flat.py`
- Rename + rewrite: `test_04_friction.py` → `test_friction_flat.py`
- Rename + rewrite: `test_06_anisotropic_anchor.py` → `test_tilted_face.py`
- Rename + rewrite: `test_08_dome_contact.py` + `test_09_dome_grip.py` →
  `test_dome_flattens.py` + `test_dome_grip_flat.py`
- Edit in place: `test_02_chain.py` (drop DP sections),
  `test_07_kernel_bridge.py` (rewrite scene list), `test_10*.py` (drop `R`)
- New: `test_half_space_monotonicity.py`,
  `test_half_space_deep_penetration.py`, `test_dome_vs_sphere.py`,
  `test_calibration.py`

### Documentation
- `theory.txt`: rewrite §1–§7; delete §3.2 (DP), §5 (bulge); replace
  §10 emission; new §12 (test catalog).
- `notes.md`: add Step 12 closure section.
- This file (`contract_v2.md`): retained as the historical spec the
  rewrite was built to.

---

## 14. Out of scope (explicitly)

These are *not* changed by the v2 rewrite; if you find yourself
modifying them, stop and check.

- Hydroelastic contact (`contact_reduction_hydroelastic.py`, etc.)
- Point contact (`contact_reduction.py` baseline)
- Newton's broad-phase, BVH, SAP
- Any non-CSLC kernel in `newton/_src/geometry/`
- `cslc_mujoco/*` (separate work thread per notes.md)
- The friction *form* (smooth `K M s / (K s + M)` stays; only the
  `f_n` source changes from sphere-vs-sphere to half-space-aggregate)
- The Jacobi solver topology, damping, warm-start
- `calibrate_kc` formula

---

## 15. Glossary of v1 → v2 changes (one-line summary each)

| Concept | v1 | v2 |
|---|---|---|
| Overlap formula | `(r + R) − ‖q − t‖` | `r − n̂_face · (q − t)` |
| Target sample | `(position, radius, normal, area)` | `(position, normal, area)` |
| Contact direction | line-of-centres `(q − t)/‖·‖` | face normal `n̂_face` |
| Lateral law | DP or graph-Laplacian (choice) | graph-Laplacian only |
| Poisson bulge | physical phenomenon | removed; flatness is anchor+contact |
| Sphere target | first-class via `RigidTarget` | sampled as `PointSetTarget` |
| Kernel paths | sphere + point-set (two) | point-set only (one) |
| MuJoCo `margin1` | `R_target` | 0 |
| `target_radii` array | per-sample | deleted |

---

## 16. Sign-off checklist

Before phase 1 began, every reader of this document confirmed:

- [x] The half-space formula `raw = r − n̂·(q−t)` is what they want.
- [x] The single contact direction (face normal) is what they want.
- [x] No code path uses distance-preserving lateral or expects bulge.
- [x] Sphere targets are acceptable as sampled point-sets (with
      `O(r_pad² / R)` approximation error vs analytic — note: §7.2
      corrected from the original `O(R · κ_pad · depth)` framing,
      which had the wrong limit).
- [x] `R` is dropped from kernels, data, theory, samplers, tests.
- [x] Tests in §12 cover every claim in §3–§10.
- [x] Deletion list in §13 is exhaustive (grep regression in CI).

---

## 17. Progress snapshot

### Completed phases

| Phase | Description | Date | Status |
|---|---|---|---|
| 0 | Write `contract_v2.md` spec | 2026-05-24 | ✓ Complete |
| 1 | Half-space primitives + foundational tests (T-A through T-D) | 2026-05-24 | ✓ Complete — 4/4 PASS |
| 2 | Unified lattice solver + dome-flattens regression (T-E through T-H) | 2026-05-25 | ✓ Complete — 4/4 PASS |
| 3 | Friction in unified path (T-I, T-J) | 2026-05-25 | ✓ Complete — 2/2 PASS.  Worst measured: T-I stick/slip `1.5×10⁻⁸` rel err (scipy `minimize_scalar` floor); T-J grip plateau exact (rel err = 0); T-J aggregation self-consistency `1.4×10⁻¹⁴ N` worst absolute. |
| 4a | Bridge harness vertical slice (T-L + T-K{A, D, F, H}) | 2026-05-25 | ✓ Complete — 6/6 PASS at production `eps = 5×10⁻⁴` m on both sides.  Required: drop ∂w_t/∂δ from `solve_lattice_contact` (match kernel's truncated gradient); add anisotropic anchor to `solve_lattice_contact`; add smooth `align_gate` to `jacobi_step_point_set` (`eps_align = 0.05` hardcoded); rewrite kernel friction to contract form `f = K·M·s/(K·s+M)` (both `jacobi_step` and `jacobi_step_point_set`); add `f_n_override` bridge hook to theory's smooth friction.  Worst rel err 6.6×10⁻⁴ (scene D max across N=150 spheres). |
| 4b | T-K full 30-case matrix (B/C/E/G/I/J) + §3.6 amendment + kernel legacy-cull removal + theory Jacobi refinement | 2026-05-25 | ✓ Complete — all 12 scenes (T-L + T-K{A..J} × kc/ka ∈ {0.1, 1, 10}) PASS at production `eps = 5×10⁻⁴` on both sides.  Contract changes: §3.6 alignment-gate amendment (one-sided smoothstep `[0, +eps_align]`; finding #11); kernel legacy v1 sphere-overlap pre-cull removed from `jacobi_step_point_set` (finding #13); damped-Jacobi post-refinement added to `solve_lattice_contact` for L-BFGS-B fragility on sphere targets (finding #14).  T-H Part D continuity sweep refined to 0.25° steps (one-sided band is half the angular width — coarse sampling aliased an apparent jump that was actually a sampling artifact).  All 10 Phase 1–3 regression tests PASS unchanged.  Worst rel err 6.6×10⁻⁴ (scene D, same as Phase 4a). |
| 5 | Warp kernel cleanup (delete sphere kernels, rename point-set → unified) + handler unification + theory v1 deletion + T-L sweep 3 (active-set membership equality) | 2026-05-25 | ✓ Complete.  Kernels: v1 sphere kernels deleted (`jacobi_step`, `write_cslc_contacts`, `compute_cslc_penetration` sphere variants; `compute_pad_force_vs_point_set` smoke helper); point-set kernels renamed (`*_point_set` suffix dropped); `target_radii` dropped from every kernel signature; `margin1 = R_j` → `0` in emission; the legacy `(r+R)-L < -50·eps` pre-cull replaced by the contract-conformant raw + one-sided alignment gates in `compute_cslc_penetration` and `write_cslc_contacts` (matches `jacobi_step` exactly).  Handler (Phase 6 §13 deletions completed concurrently per user direction): `_launch_vs_sphere` deleted; `_launch_vs_point_set` renamed to `_launch`; `CSLCShapePair.is_point_set` / `other_local_pos` / `other_radius` / `target_radii` dropped; `from_model_with_lattices` simplified (single dispatch path, caller-populated target arrays).  Theory: `RigidTarget`, `PointSetTarget` (v1), the sphere-target equilibria (`equilibrium_face_on_analytical`, `equilibrium_numerical`, `equilibrium_with_friction_*`), the `point_set_*` family, `kernel_contact_force_n_axis`, `lateral_force/energy_distance_preserving`, `ContactTarget` / `SphereIndenter` / `PointSetIndenter`, and the v1 lattice solver chain (`solve_chain_contact_linear`, `solve_lattice_contact_linear`, `solve_lattice_contact_numerical`, `solve_lattice_sphere_indenter`, `solve_lattice_point_set_indenter`) deleted.  `kernel_bridge.py` fully orphaned post-cleanup, deleted entirely.  Test files: `test_01_single_sphere`, `test_03_chain_contact`, `test_04_friction`, `test_05_arc_contact`, `test_06_anisotropic_anchor`, `test_08_dome_contact`, `test_09_dome_grip`, `test_10*`, `test_11_recalibrate_kc_units` deleted (all superseded).  Additional v1-only files deleted: `cslc_box.py`, `cslc_projection_experiment.py`, `cslc_mujoco/validation/t1_kernel_sanity.py`.  Grasp: `contact_models.py` collapsed to the unified point-set path; sphere-target objects now raise `NotImplementedError` until Phase 7 wires a sphere-as-point-set sampler.  Constant-discipline guard: 3 raw-cull sites + 4 align-gate sites (Phase 4b had 3 + 2; Phase 5 adds matching align gate to the emission kernel).  Bridge harness gains Phase 5 T-L sweep 3: 19-point active-set membership probe across the `raw_factor = -50.0` threshold, asserting EXACT set equality on both sides; transitions coincide at `raw_factor ∈ (-50.001, -49.999)` on both sides (last sub-threshold probe gives δ = 0 exactly; first super-threshold probe gives δ ≠ 0; smooth-tail contributes on both as expected past the threshold).  All 12/12 bridge scenes PASS at the same numerics as Phase 4b; all 10/10 Phase 1–3 regression tests PASS unchanged. |
| 6 | Handler + data layer | 2026-05-25 (handler portion folded into Phase 5) | Handler portion complete in Phase 5 per user direction (full `§13` handler-side deletions landed alongside the kernel cleanup); remaining data-layer work (e.g. `CSLCData` shape conventions, sphere-as-point-set sampling helpers) deferred. |
| 7 | Examples / grasp scripts | — | Pending |
| 8 | Documentation rewrite (`theory.txt` v2, `notes.md` Step 12 closure) | — | Pending |

### Files created / modified in Phases 1–4b (still in place)

```
cslc_main/theory/
├── contract_v2.md                              # this document
├── cslc_targets.py                             # NEW: PointSetTargetV2 + 4 samplers (Phase 2a)
├── cslc_theory.py                              # MODIFIED: half-space primitives (Phase 1a)
│                                               #         + 3 v2 friction equilibria (Phase 3a)
│                                               #         + ``f_n_override`` on smooth friction (Phase 4a)
├── cslc_lattice.py                             # MODIFIED: solve_lattice_contact (Phase 2b)
│                                               #         + lattice_contact_normal_forces (Phase 3b)
│                                               #         + ka_t_ratio kwarg + ∂w_t/∂δ DROPPED (Phase 4a)
│                                               #         + one-sided align gate (Phase 4b, finding #11)
│                                               #         + _jacobi_refine post-step (Phase 4b, finding #14)
├── test_half_space_monotonicity.py             # NEW (T-A, Phase 1b)
├── test_half_space_deep_penetration.py         # NEW (T-B, Phase 1c)
├── test_face_on_series_spring.py               # NEW (T-C, Phase 1d) — replaces test_01
├── test_tilted_face.py                         # NEW (T-D, Phase 1e) — replaces test_06
├── test_02_chain.py                            # EDITED (T-E, Phase 2c) — DP sections removed
├── test_chain_contact_flat.py                  # NEW (T-F, Phase 2d) — replaces test_03
├── test_dome_flattens.py                       # NEW (T-G, Phase 2e) — replaces test_05 + test_08
├── test_dome_vs_sphere.py                      # NEW (T-H, Phase 2f)
├── test_friction_flat.py                       # NEW (T-I, Phase 3c) — replaces test_04
├── test_dome_grip_flat.py                      # NEW (T-J, Phase 3d) — replaces test_09
├── test_07_kernel_bridge.py                    # REWRITTEN IN PLACE (T-K + T-L, Phase 4a)
│                                               # EXTENDED (Phase 4b): + scenes B, C, E, G, I, J;
│                                               #   scene H restored to full box; delta0 warm-start
│                                               #   in run_kernel for scene G's slip warm-start.
├── test_dome_vs_sphere.py                      # TWEAKED (Phase 4b): T-H Part D angular sweep
│                                               #   re-sampled at 0.25° (was 1°) — the amended
│                                               #   one-sided gate band is half the angular width,
│                                               #   so coarse sampling aliased a non-existent jump.
└── figures/{ta..th,ti,tj}_*.png                # 16 figures, 1 text table

newton/_src/geometry/
└── cslc_kernels.py                             # MODIFIED (Phase 4a):
                                                #   + smooth align_gate (eps_align = 0.05) in jacobi_step_point_set
                                                #   + friction rewritten to f = K·M·s/(K·s+M) in BOTH
                                                #     jacobi_step AND jacobi_step_point_set
                                                # MODIFIED (Phase 4b):
                                                #   + one-sided align gate (band [0, +eps_align]; finding #11)
                                                #   + legacy v1 sphere-overlap pre-cull REMOVED from
                                                #     jacobi_step_point_set (finding #13)
```

The v1 tests `test_01`, `test_03`–`test_06`, `test_08`, `test_09` are **untouched** — they continue to pass, and will be deleted in Phase 5+ once their replacements are propagated through the full stack.  Regression sweep at end of Phase 4a (`test_half_space_monotonicity`, `test_half_space_deep_penetration`, `test_face_on_series_spring`, `test_tilted_face`, `test_02_chain`, `test_chain_contact_flat`, `test_dome_flattens`, `test_04_friction`, `test_friction_flat`, `test_dome_grip_flat`): all 10 exit 0.

### Key Phase 1–4a findings (empirical)

1. **L-BFGS-B floor on multi-sphere problems.**  Convergence floors at
   `~5×10⁻⁸ rel err` on chain/dome scenes even with `eps = 1e-12` and
   `gtol = 1e-16` — fp roundoff in the `np.add.at` lateral assembly
   propagates through scipy's stopping criteria.  Phase 2 tolerances
   set against this floor; contract §12 T-F tolerance updated from
   the original `1e-12` to `1e-7`.

2. **Production-eps precision floor.**  At production `ε = 5×10⁻⁴ m`
   the smooth-surrogate correction scales as `~1/(raw_eq/ε)⁴` (the
   `σ_ε` and `Σ_ε` corrections cancel to leading order in the product
   `φ_eff · gate`).  Force matches analytical to <1% when
   `raw_eq ≥ 10·ε`.  Below that the smoothing dominates.  This is the
   precision budget every Phase 2+ test inherits.  Quantified in
   T-C Part D.

3. **Alignment gate required for closed convex targets.**  The
   half-space form alone gives spurious "contact" on the far side of
   closed bodies (sphere, box).  v1 implicitly culled these via the
   3-D distance gate `(r + R) − ‖q − t‖ < threshold`, but in v2 with
   `R = 0` that gate collapses.  The replacement is `n̂_face · n̂_pad
   < 0` — a parameter-free geometric test for "surfaces facing each
   other".  Added in Phase 2f after T-H reported `min δ_n ≈ −7 mm`
   on the tennis ball; with the gate, `min δ_n > 0` at all depths.

4. **`w_t` δ-derivative included for L-BFGS-B fun/jac consistency.**
   The v2 solver computes the FULL gradient including
   `∂w_t/∂δ_i · ½ φ_eff² · A_j`, even though contract §4 says the
   kernel ignores this term.  Empirically the term contributes
   `O(ε²/r_i³)` to the gradient (~1.4 m⁻¹ at production) — negligible
   for convergence direction but matters for fun/jac match at L-BFGS-B
   line searches.  Phase 4 bridge tests will quantify the theory-vs-
   kernel gap from the kernel's truncated gradient.

5. **`v2 anisotropic tilt` is simpler than `v1`.**  Contract §12 T-D
   originally inherited the v1 formula `|δ_t/δ_n| = (k_a − A)/(k_a ρ
   − A) tan θ` with `A = k_c φ_eff / L`.  v2's face normal is
   δ-independent, so `∂(raw)/∂δ = +n̂_face` is constant and the `A`
   correction vanishes.  Corrected formula: `|δ_t/δ_n| = (1/ρ) tan θ`
   exactly.  Verified empirically: v1 wrong by up to 21% at `ρ =
   0.1`; v2 matches numerical to ~`1e-9 rel err`.

6. **v2 friction is one-line different from v1 friction.**  Contract §6.4
   says only the `f_n` source changes between v1 and v2 friction; the
   stick-slip law is unchanged.  Empirical confirmation: T-I PART C max
   `|s_a − s_h|` ~ `9 pm` (`8.5×10⁻¹² m`) — the same `scipy.optimize.
   minimize_scalar(method='bounded')` 7-digit floor v1 `test_04` hit,
   NOT a v2-specific artefact.  The smooth-surrogate smoothing-zone
   deviation (peaks at ~40% rel near the stick-slip kink for the
   production scene) is identical in shape and magnitude to v1; the
   harmonic-mean form is structurally unchanged.  Practical
   consequence: Phase 4 bridge tests for friction inherit v1's smooth-
   zone budget verbatim.

7. **T-J grip aggregation needs no new solver.**  Contract §12 T-J asks
   for `F_grip(s) = Σ_i min(k_stick·s, μ·f_n,i)` under uniform applied
   `s` — a displacement-driven aggregation, not a force-driven
   equilibrium.  Per-sphere `f_n,i` extracted from
   `solve_lattice_contact` plus the new
   `lattice_contact_normal_forces` helper (which projects the
   half-space contact force on each pad's own `n̂_i` per §6.4) is
   sufficient.  Plateau matches `μ · F_n^total` to exact fp precision
   (rel err = 0); the only error budget is the per-sphere `f_n,i`
   precision inherited from the L-BFGS-B floor of finding #1.  No
   coupled lattice-friction solver added in Phase 3 — kept minimal per
   the contract's "simplest path that proves the claim" stance.  Phase
   4 bridge harness will still need this same helper to extract per-
   sphere `f_n,i` for kernel-vs-theory comparison.

8. **T-I `k_stick` sweep needs F_grid spanning past max F_thresh.**
   v1 `test_04` PART D used `F_grid ∈ [0, 6 μ f_n]` and got away
   without asserting on F_thresh values directly.  T-I PART D adds an
   explicit "F_friction equals plateau just past F_thresh" check —
   which fails at `k_stick/k_a = 0.1` (`F_thresh = 11 μ f_n`, far past
   `6 μ f_n`) unless the grid spans past the WORST-case `F_thresh`
   among the swept ratios.  Solved by sizing
   `F_grid_max = 1.5 · F_thresh(min(k_stick))`.  Trap to remember when
   extending the sweep beyond the four ratios currently tested.

9. **w_t-derivative dropped from `solve_lattice_contact` (Phase 4a commit).**
   Phase 2 finding #4 documented that theory's solver included the
   ∂w_t/∂δ term (= ½ k_c A_j a φ_eff² · Σ'_ε(3r − d_t) · ẑ_ij) for
   fun/jac consistency; the kernel ignored it.  Phase 4a's "theory =
   kernel, never both ways" rule forced a commit.  Dropped from theory
   side ⇒ ``solve_lattice_contact``'s ``jac`` is now a truncated
   approximation of the full energy gradient; the converged δ is the
   point where the **truncated** gradient is zero, NOT a stationary
   point of the smooth-surrogate energy.  **Bridge parity holds**:
   theory and kernel both run the truncated gradient, so both
   converge to the same δ.

   The truncated-vs-full shift is small but measurable:
   quantified at the Phase 4a scene-D converged iterate (dome on
   flat, production eps), the dropped term contributes
   ``||Δg|| ≈ 0.34 N`` total (max ~0.1 N per sphere) — ~1.1% of the
   anchor gradient scale ``ka·max|δ|``, but ~1500× larger than the
   converged truncated ``|∇E|`` the solver reports.  In other words:
   the truncated solver's "converged" point sits ~1% off the true
   energy minimum.  Acceptable for bridge parity (which is the
   Phase 4 deliverable) but worth knowing for anyone interpreting
   `info["energy"]` as a minimised value.  All 10 Phase 1–3 theory
   tests pass unchanged after the drop.

10. **Kernel friction had a hidden bias at production eps (Phase 4a
    finding).**  The original kernel computed
    ``inv_dt_mag = s / (s² + eps²)`` as a regularised ``1/s``,
    suitable when ``s ≫ eps`` but at production ``eps = 5×10⁻⁴ m``
    and stick-mode ``s ~ μm``, ``eps²`` dominates and
    ``inv_dt_mag ≈ s/eps²`` is 3-4 orders of magnitude too small —
    friction effectively vanishes.  Discovery: T-K scene F kernel
    converged to ``s = F/k_a`` (no friction) while theory gave
    ``s = F/(k_a + k_stick)`` — a clean 2× ratio that pinpointed the
    bug.  **Fix:** rewrite the kernel's friction to the contract form
    ``scale = K·M / (K·s + M + 1e-30)``, ``f_friction = -scale · δ_t``,
    which avoids ``1/s`` entirely and is the algebraic identity for
    the contract §6.4 ``f = K·M·s / (K·s + M)`` form.  Landed in both
    ``jacobi_step`` (v1) and ``jacobi_step_point_set`` (v2) — the two
    sites MUST stay in sync.

11. **Alignment gate insufficient for closed convex targets with
    flat side faces — RESOLVED in Phase 4b.**  Contract §3.6
    originally defined the smooth gate as ``smoothstep`` on
    ``[-eps_align, +eps_align]``.  At ``align_arg = 0`` (perpendicular
    faces) ``align_w = 0.5``.  For a box target whose side faces are
    perpendicular to the pad normal: side-face samples at corners ARE
    in the locality kernel (``d_t ≈ box_bottom_height``, well inside
    ``3·r_pad``), have raw very large positive (deeply "inside" their
    half-plane), and contributed half-strength asymmetric in-plane
    forces that prevented L-BFGS-B convergence at production eps.
    T-H (sphere target) didn't surface this because equatorial
    samples are geometrically far (``d_t > 3·r_pad``) and
    ``w_t``-culled.  **Phase 4b amendment:** shifted ``§3.6`` to the
    one-sided form ``smoothstep`` on ``[0, +eps_align]``
    (perpendicular AND back-to-back HARD-culled).  Landed in
    ``solve_lattice_contact``, ``lattice_contact_normal_forces``, and
    ``jacobi_step_point_set``.  T-K scene H restored to full box
    target (``make_box_target``); scene I (chain vs box) added under
    the same amendment.  Regression: T-H Part D continuity sweep
    needed finer angular sampling (0.25° instead of 1°) — the
    one-sided band is half the angular width of the symmetric band,
    so coarse sampling aliased an apparent ~50% per-step jump that
    was actually a sampling artifact, not a discontinuity.  Worst
    actual per-step ΔF/F_max at 0.25° resolution: 13.0% (well inside
    the 30% C¹-gate bound).

12. **Theory friction f_n must be LIVE for bridge parity (Phase 4a
    finding).**  Contract §6.4 specifies ``f_n = |F_contact_i · n̂_i|``
    — i.e., evaluated at the iterate's current δ.  Phase 3's
    ``equilibrium_half_space_friction_smooth_numerical`` freezes f_n
    at the analytical face-on value (``ka·δ_n_analytic``) for hard-law
    decoupling — exact at ``eps_contact = 1e-9``, but at production
    ``eps_contact = 5×10⁻⁴`` the smooth equilibrium δ_n drifts
    ~15-20% from analytical (cf. T-C Part D production-eps floor),
    so f_n_smooth ≠ f_n_analytic.  This shifted scene F by ~6×10⁻³
    relative — kernel and theory each consistent with themselves
    but inconsistent with each other.  **Fix:** added
    ``f_n_override`` kwarg to the Phase 3 primitive; bridge harness
    pre-computes ``f_n_smooth = ka · |δ_n_smooth|`` from a
    normal-only ``solve_lattice_contact`` run at the same eps, then
    passes it in.  Scene F rel err dropped from `6×10⁻³` to `4×10⁻⁶`.
    Phase 3 tests unchanged because the default still uses analytical
    f_n at tight eps.

13. **Legacy v1 sphere-overlap pre-cull in kernel — REMOVED Phase
    4b.**  The Phase 4a kernel kept a two-gate active-set check in
    ``jacobi_step_point_set``:

    ```
    (1)   (r_i + R_j) - L  < -50·eps    ⇒ skip   (sphere-overlap pre-cull)
    (2)   raw_half         < -50·eps    ⇒ skip   (contract §3.6 raw cull)
    ```

    Gate (1) is a v1 sphere-vs-sphere overlap test inherited from
    pre-v2 code; with ``R_j = 0`` it becomes ``L > r_i + 50·eps`` —
    a 3-D distance hard-cull at ~26.5 mm at production eps.  For
    flat / box targets the cull is harmless (all relevant samples
    have small ``L``).  For sphere targets (R = 33.5 mm) it removed
    ~50% of southern-hemisphere samples from the active set,
    samples that theory keeps with a small but nonzero ``w_t``
    polynomial tail.  Empirically this shifted scene J equilibrium
    ``δ_n`` by ~10%.  **Fix:** removed gate (1) from
    ``jacobi_step_point_set``; only the contract-conformant gate (2)
    plus the §3.6 alignment gate remain.  Other kernels in
    ``cslc_kernels.py`` (the v1 sphere kernels, the
    ``compute_cslc_penetration_point_set`` emission helper) still
    carry the legacy gate and will be addressed in Phase 5+ per
    contract §13.  Performance impact: the per-target loop now
    visits every sample regardless of distance — at production
    point-set sizes (~1500 samples per target) this is a small
    per-step cost; bridge correctness takes priority for Phase 4b.

14. **Theory's L-BFGS-B fragile on sphere targets — damped-Jacobi
    post-refinement added Phase 4b.**  ``solve_lattice_contact``
    minimises ``fun`` (full E, including the ``w_t``-weighted contact
    energy) using ``jac`` that DROPS ``∂w_t/∂δ`` (Phase 4a
    finding #9 commit).  L-BFGS-B's Wolfe-condition line search
    compares ``fun`` against directions from ``jac``; with the
    inconsistency, line search can fail to make progress when the
    dropped term is large.  For flat / box / chain targets the
    dropped term is ``O(ε²/r³)`` per pair and L-BFGS-B converges
    fine.  For sphere targets — many off-axis samples sit
    simultaneously in the ``w_t`` transition zone, so the dropped
    term is the dominant gradient contribution at the iterate —
    L-BFGS-B stalls at a non-stationary point (10–30% off the
    true truncated-gradient zero, depending on R and sample
    density).  **Fix:** added ``_jacobi_refine`` post-step that
    mirrors the kernel's damped-Jacobi iteration (contract §6.5)
    on theory's truncated gradient, with the same ``α = 0.3`` and
    ``tol = 1e-10`` as the kernel.  On L-BFGS-B-converged scenes
    (A, B, C, D, E, F, H, I) Jacobi exits in O(10–100) iterations;
    on stalled scenes (J) it runs the full budget.  Theory now
    reaches the kernel's truncated-gradient fixed point to
    ``|grad| ~ 1e-5`` on all Phase 4b scenes; all 10 Phase 1–3
    regression tests unchanged (PASS).  ``info`` dict gains the
    ``jacobi_refine_iters`` key for diagnosis.

### Contract amendments applied during Phases 1–4b

| Section | Change | Reason |
|---|---|---|
| §3.6 | Added alignment-cull `n̂_face · n̂_pad ≥ 0` as a second active-set condition. | Required for closed convex targets (T-H surfaced the need). |
| §7.2 | Rewrote sphere-target reduction quality discussion: correct `O(r_pad² / R)` error bound; correct validity regime `R ≫ r_pad²/depth`. | Original `O(R · κ_pad · depth)` framing had the wrong limit (caught in review). |
| §6.5 | Equilibrium equation now in pure gradient form (all `+`); convention note added explaining gradient ↔ load translation. | Original mixed signs (anchor as load, contact as physical) was algebraically inconsistent (caught in review). |
| §11 | Footnote `†` added flagging the `kc_series = kc · ke / (kc + ke + ε²)` dimensional mismatch. | Inherited v1 quirk; flagged for future cleanup, preserved as-is to avoid changing kernel output. |
| §12 T-C | Added Part D (production-eps precision floor). | Provides the tolerance budget Phase 2+ tests inherit. |
| §12 T-D | Rewrote formula and tolerance. | v1's `A` correction does not apply in v2. |
| §12 T-F | Tolerance relaxed `1e-12 → 1e-7`. | L-BFGS-B floor; v1's `1e-12` assumed a direct linear solve. |
| §12 T-G, T-H | Made assertions concrete (1 nm bulge bound, specific monotonicity checks). | Phase 2 empirics filled in the success metrics. |
| §12 T-I | Tolerance relaxed `1e-8 → 1e-7`. | `scipy.optimize.minimize_scalar(method='bounded')` is ~7-digit precise; same floor v1 `test_04` PART B documented.  Independent of the v2 contact swap. |
| §13 (deletion list) | No additions in Phase 3. v1 friction (`equilibrium_with_friction_*` taking `RigidTarget`) remains in `cslc_theory.py` alongside the new v2 versions; deletion deferred to Phase 5+ per the additive-only Phase 3 plan. | Phase 5 owns kernel cleanup and is the natural point to retire v1 friction once Phase 4 bridge tests confirm the v2 path matches. |
| §6 contract `solve_lattice_contact` | (Phase 4a) Added `ka_t_ratio` kwarg (anisotropic anchor — required for T-K scene E in Phase 4b).  Dropped `∂w_t/∂δ` from the gradient (theory now matches kernel's truncated gradient — see finding #9).  Renamed Phase 2 "no anisotropic anchor" caveat to "anisotropic via ka_t_ratio". | Bridge harness needs theory and kernel to compute the same gradient; "theory = kernel, never both ways" rule from Phase 4 user spec. |
| §6.4 friction (kernel form) | (Phase 4a) `jacobi_step_point_set` AND `jacobi_step` rewritten: `inv_dt_mag = s/(s²+eps²); cone_scale = M·inv_dt_mag; scale = K·cone_scale/(K+cone_scale+...)` → `scale = K·M/(K·s+M+1e-30); f_friction = -scale·δ_t`.  Algebraically equivalent to contract eq:friction-force but avoids 1/s entirely — fixes the production-eps friction bias of finding #10. | Bridge harness exposed the bias at production eps where stick-mode `s ~ μm << eps = 5×10⁻⁴ m`.  Contract §6.4 form unchanged; only the kernel's internal computation changed. |
| §12 T-K | (Phase 4a) Split into Phase 4a vertical slice (T-L + T-K A/D/F/H = 6/6 PASS) and Phase 4b full matrix (10×3 = 30 cases pending).  T-K scene H reduced to "multi-sample flat face" for Phase 4a; "single pad vs box" restored in Phase 4b after §3.6 amendment lands. | Phase 4a delivers the harness end-to-end + 4 representative scenes; Phase 4b adds the remaining 6 scenes (B/C/E/G/I/J) after the alignment-gate amendment (finding #11). |
| §3.6 | (Phase 4b) Alignment gate **shifted** from `smoothstep` on `[-eps_align, +eps_align]` (symmetric; align_w = 0.5 at perpendicular) to `smoothstep` on `[0, +eps_align]` (one-sided; align_w = 0 at perpendicular AND back-to-back).  Perpendicular faces of closed convex targets HARD-culled; smooth band only near face-on for jitter robustness.  Landed in `solve_lattice_contact`, `lattice_contact_normal_forces`, and `jacobi_step_point_set`.  T-H Part D continuity test re-sampled at 0.25° (was 1°) to resolve the now-half-width band; max per-step ΔF/F_max = 13% at the new resolution (well inside the 30% C¹-gate bound). | Required for scenes H, I (box targets) at production eps — see finding #11.  Landed Phase 4b. |
| §12 T-K | (Phase 4b) All 30 cases of the full matrix complete: scenes A, B, C, D, E, F, G, H, I, J × kc/ka ∈ {0.1, 1, 10}.  Scene H restored to full box target; scene I added (chain vs box); scenes B, C, E, G, J added per contract.  Constant-discipline guard PASS with the amended literal `align_arg <= 0.0`. | Phase 4b completes the bridge matrix on the unified path. |
| §13 (deletion list) | (Phase 5) Every v1 symbol on the list removed from `cslc_kernels.py` (sphere-target triple + `compute_pad_force_vs_point_set`), `cslc_handler.py` (sphere-target launch path + `CSLCShapePair` sphere fields), `cslc_theory.py` (RigidTarget + sphere-target equilibria + sphere-target friction equilibria + `point_set_*` family + DP-fallback witness), `cslc_lattice.py` (DP lateral law + `ContactTarget` / `SphereIndenter` / `PointSetIndenter` + the v1 solver chain).  Point-set kernels renamed (`*_point_set` suffix dropped); `target_radii` dropped from every kernel signature; `margin1 = R_j` → `0` in emission.  Emission + warm-start gates (`compute_cslc_penetration`, `write_cslc_contacts`) brought in line with `jacobi_step`'s contract §3.6 raw + one-sided alignment gates (the legacy `(r+R)-L < -50·eps` cull deleted).  `cslc_main/theory/kernel_bridge.py` was orphaned post-cleanup; deleted entirely.  Phase-1–3 v1 test files (`test_01`, `test_03`, `test_04`, `test_05`, `test_06`, `test_08`, `test_09`) and the v1-only `test_10*` / `test_11_recalibrate_kc_units` / `cslc_box.py` / `cslc_projection_experiment.py` / `t1_kernel_sanity.py` deleted.  Grasp pipeline collapsed to the unified point-set path (sphere-object grasp scenes raise `NotImplementedError` until Phase 7 wires `make_sphere_target` through `cslc_main/grasp/contact_models.py`).  Constant-discipline guard: 3 raw-cull sites + 4 align-gate sites (Phase 4b had 3 + 2; Phase 5 adds the alignment gate to the emission kernel for solver/emission parity).  All bridge-harness and Phase 1–3 numerics PASS unchanged (worst rel err 6.6×10⁻⁴, scene D). | Phase 5 lands the §13 deletion list (kernel cleanup + handler unification + theory v1 removal).  Phase 6 (data layer) and Phase 7 (sphere-object grasp sampling) pending. |
| §12 T-L | (Phase 5) Added sweep 3 to `t_l_active_set_parity`: fine `raw_factor` sweep across the contract threshold `raw = -50·eps`.  Asserts EXACT set-membership equality at every probe (kernel and theory must agree on whether the (pad, target) pair is in the active set), with both transitions occurring at exactly `raw_factor = -50.0`.  Operational definition: "active" iff δ ≠ 0 (a hard-skipped pair contributes no force on a 1-sphere, no-neighbor, no-friction scene, so δ stays at exactly zero; a smooth-tail contact past the threshold yields δ ≠ 0 even when the magnitude is sub-pm).  19 probe points, all PASS; the last sub-threshold probe (`raw_factor = -50.001`) gives δ = 0.0 on both sides, the first super-threshold probe (`raw_factor = -49.999`) gives δ ≠ 0 on both sides.  Confirms the kernel's `if raw < -50.0 * eps: continue` and the theory's matching `raws >= INACTIVE_RAW_EPS_FACTOR * eps` cull activate at the same rest geometry, with both using the half-space `raw = r - n_face·(q - t)` form (a legacy 3-D `(r+R)-L` cull would transition at a different `raw_factor`). | Phase 5 T-L extension per contract §12 ("Same harness, run against the cleaned/renamed Warp kernels.  Same tolerances as T-K.  Plus: ... Both transitions occur at raw = -50ε."). |
