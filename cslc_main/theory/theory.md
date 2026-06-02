# CSLC — Unified Half-Space Contact

## 1. Scope

The Compliant Sphere Lattice Contact (CSLC) model represents a deformable
pad as a lattice of compliant spheres anchored to a rigid base body, and
generates contact forces against a target geometry sampled as a point set
with face normals.  This document specifies the contact law, lattice
equilibrium, calibration, and MuJoCo emission convention.

A contactable target is represented as a `PointSetTarget(positions,
normals, areas)` — for every sample `j` the model carries an outward
face normal `n̂_face_j` and a Voronoi-area weight `A_j` on the underlying
surface.  No per-sample radius is consumed: contact is between pad
spheres of radius `r_i` (compliance skin thickness) and target half-space
face elements.

The pad lattice is *any* sampling of contact spheres on the pad surface
(dome cap, flat grid, mesh — same code path) with positions, radii,
outward normals, and a neighbour adjacency.  The target is *any*
`PointSetTarget` with face normals (sphere, box, mesh — same code path).

The per-pair half-space overlap is

```
raw_ij = r_i − n̂_face_j · (q_i − t_j)                       (eq:raw)
```

monotone in penetration depth at any depth, with no sign flip at face
crossing.  The contact law consumes the **face-penetration** scalar

```
raw_face_ij = raw_ij − r_i = − n̂_face_j · (q_i − t_j)       (eq:raw-face)
```

which is zero at face contact and positive at face penetration.
Switching from `raw_ij` to `raw_face_ij` in the force law removes an
`r_i` shelf at face onset: under the half-space form, flat-pad lattice
sphere centres sit at the target plane at face contact and see
`raw_ij ≈ r_i`, summing to ~84 N of spurious force per body on a
~490-sphere box pad even when no face has actually penetrated.

**Hertz-like force law.**  The per-pair force coupling is

```
φ_eff(raw_face) = β_ε(raw_face) · √(β_ε(raw_face) + ε)      (eq:phi-eff)
```

where `β_ε` is a one-sided polynomial blend (Hermite quintic) that is
exactly 0 for `raw_face ≤ 0` and exactly `raw_face` for `raw_face ≥ ε`
with a C² blend in between (§3.4).  Replacing the analytic smooth ReLU
`σ_ε` — which has an unavoidable `σ_ε(0) = ε/2` floor and propagates
an `ε^{1.5}` baseline force through `φ_eff` — with `β_ε` makes
`φ_eff(raw_face = 0) = 0` exactly.  In the saturated regime
`raw_face ≫ ε` the formula reduces to the true Hertz scaling
`F ∝ raw_face^{1.5}` with local stiffness `dF/d(raw_face) ∝ √raw_face`
vanishing at first touch.  Sphere-on-flat integration over the
contact patch picks up one more power of `δ` from area scaling,
giving aggregate `F_total ∝ δ^{2.5}` — one power stiffer than Hertz's
`δ^{1.5}` because the contact patch grows with `δ`.  Consequence:
`k_c` is **not** literally a Young's modulus; see §10 for the
calibration that maps a bulk material stiffness to `k_c`.

**Partition-of-unity area accounting.**  Each per-pair force is
multiplied by a target-side share weight

```
share_ij = (w_t_ij · a_ij) / W_j,   W_j = Σ_k  w_t_kj · a_kj   (eq:share)
```

so `Σ_i share_ij = 1` for every target sample `j` (a true partition
of unity over the pad spheres that reach target `j`).  Total force on
the pad body becomes a Riemann sum over target Voronoi cells
`Σ_j  k_c · A_j · ⟨φ_eff⟩_j · n̂_face_j`, lattice-density-invariant
by construction.  Without `share_ij`, each target's area `A_j` is
counted ~`π·r_pad²/h²` times across the lattice for Option-2 tiling,
making the total force grow with pad density rather than with
physical contact-patch area.

The lateral coupling on the lattice is a graph Laplacian (linear, no
distance-preserving / rest-length term).  See §6.

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
| `A_j` ∈ ℝ₊ | Target sample Voronoi area on the underlying surface | [m²] |
| `k_a`, `k_l` | Anchor, lateral stiffness | [N/m] |
| `k_c` | Per-volume contact stiffness | [Pa · m^(−1/2)] — see §10 |
| `ρ = k_{a,t}/k_a` | Tangent anchor ratio | 1.0 isotropic, 1/3 incompressible |
| `μ`, `k_stick` | Coulomb coefficient, stick spring stiffness | |
| `ε` | Smoothing width | default `5×10⁻⁴ m` |
| `ε_align` | Alignment-gate band width | hard-coded `0.05` |
| `c_lattice` | Lattice velocity-damping rate | [N·s/m] — see §11 |
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

### 3.2 Half-space raw overlap and face-penetration form

For pad sphere `i` and target sample `j` with outward face normal `n̂_j`:

```
raw_ij      = r_i − n̂_j · (q_i − t_j)                       (eq:raw)
raw_face_ij = raw_ij − r_i = − n̂_j · (q_i − t_j)            (eq:raw-face)
```

`raw_ij` is the half-space form used to reconstruct MuJoCo's
`solver_pen` (§8).  `raw_face_ij` is the **face-penetration** form
consumed by the contact law (`φ_eff`, `gate`, active-set culls): it
is zero at face contact (`q_i` on the target's tangent plane through
`t_j`) and positive when the pad sphere centre has crossed the plane
into the target half-space.  The two differ by a constant `r_i` per
pad sphere.

**Why face-penetration form.**  Under the half-space `raw_ij`, a
flat-pad lattice sphere centre at the target plane sees
`raw_ij = r_i` at face contact (not 0), and a Hertz-law `φ_eff(raw_ij)`
emits an `r_i^{1.5}` shelf force per sphere even before any geometric
face penetration.  On a 490-sphere box pad this sums to ~84 N of
spurious force at `δ_face = 0`.  Using `raw_face_ij` in the force
law shifts the on-onset point to the geometric face contact.

**Monotonicity.**  Fix `n̂_j` and `t_j`; let `q_i` vary along the `−n̂_j`
direction (deeper into the target body).  Then `n̂_j · (q_i − t_j)`
strictly decreases, so both `raw_ij` and `raw_face_ij` strictly
increase.  No sign flip ever; deep-penetration is well-defined.

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
β_ε(x) = 0                       if x ≤ 0                    (eq:beta)    one-sided blend
        x · H₅(x/ε)              if 0 < x < ε
        x                        if x ≥ ε
H₅(t) = t³ · (10 − 15 t + 6 t²)                              (Hermite quintic)
```

with the chain-rule identity `σ'_ε = Σ_ε`.

**Smooth-blend obstruction.**  `σ_ε` is C^∞ but `σ_ε(0) = ε/2 ≠ 0`,
so `φ_eff(raw = 0) = (ε/2) · √(3ε/2) ≈ 0.61 · ε^{1.5}` propagates a
fixed-magnitude force floor whose value scales with the smoothing
width.  An IVT argument forbids a C^∞ approximation of `max(x, 0)`
that simultaneously passes through `(0, 0)` and stays non-negative:
if `f` is C¹ with `f'(−∞) = 0` and `f'(+∞) = 1`, then `f'(0) ∈ (0, 1)`
and `f(x) ≈ f'(0) · x < 0` for small `x < 0`, violating
non-negativity.  Trading C^∞ for C² breaks the obstruction: `β_ε`
(Hermite quintic blend) gives `β_ε(x ≤ 0) = 0` exactly, `β_ε(x ≥ ε) = x`
exactly, and is C² at both endpoints (`H₅` and its first two
derivatives vanish at `t ∈ {0, 1}`).  C² is sufficient for reverse-
mode autodiff through the lattice solve — the loss of C^∞ is invisible
to downstream gradients.

**Substitution rule.**  Anywhere the smooth-energy form would call
for `σ_ε(raw_face)`, the implementation uses `β_ε(raw_face)`.  Smooth
steps `Σ_ε` are unchanged and still apply to `gate_ij`, `w_t_ij`, and
the alignment-gate smoothstep.

### 3.5 Smoothed quantities

```
φ_eff_ij = β_ε(raw_face_ij) · √(β_ε(raw_face_ij) + ε)        (eq:phi-eff)
gate_ij  = Σ_ε(raw_face_ij)                                  (eq:gate)
w_t_ij   = Σ_ε(r_i − d_t_ij)                                 (eq:w_t)
```

**Hertz-like `φ_eff`.**  `φ_eff` is the contact-force coupling that
appears in the per-pair load (§4).  The factor `√(β_ε + ε)` lifts the
linear blend to a `raw_face^{1.5}` law in the saturated regime
(`β_ε(raw_face) = raw_face` for `raw_face ≥ ε`), so the per-pair force
`F ∝ raw_face^{1.5}` and its local derivative
`dF/d(raw_face) ∝ √raw_face` vanishes at face contact.  Below face
contact `β_ε(raw_face ≤ 0) = 0`, so `φ_eff = 0` exactly — no shelf,
no baseline force.  The `+ε` inside the square root is a smoothing
guard so the derivative stays bounded as `raw_face → 0⁺`; since
`β_ε(0) = 0` it multiplies to zero and contributes no floor.

**Tangential locality kernel (tiling).**  Kernel half-width `r_i` (the
pad sphere radius itself).  Discs of radius `r_pad` tile the pad face
without overlap when pads are CVT/Lloyd-sampled at spacing `2·r_pad`,
which makes the discrete sum in §5 a clean Riemann-style quadrature of
the surface integral `∫ k_c · φ · n̂_face dA` over the contact patch.
This makes the identity `k_c_per_volume · π · r_pad² = k_c_per_sphere`
exact (see §10).

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

With `ε_align = 0.05` (≈ 2.87° angular full-transition from
α = 0 to α = +ε_align), matched to typical pad-lattice and
target-sampling angular resolutions.  The force from §4 (eq:f-phys)
carries the gate as a per-pair factor:

```
F_ij  =  + k_c · A_j · w_t_ij · a_ij · φ_eff_ij · gate_ij · n̂_face_j
```

(see §4 for sign conventions).  Setting `ε_align → 0` recovers the
hard step `a_ij = 1` if `α > 0` else `0`, which is **discontinuous**
at `α = 0` and breaks the smooth-force structure; this mode is
supported for ablation testing only.

A pair `(i, j)` is **inactive** (skipped without numerical error) iff
ANY of the following hold:

```
(a)  raw_face_ij  <  −50 ε                                   (eq:inactive-raw)
(b)  α_ij         ≤  0                (back-side / perpendicular hard-cull)
(c)  w_t_ij       <  1×10⁻²           (tangential-locality hard-cull)
(d)  ‖q_i − t_j‖  >  3·r_i + 5·ε      (distance-magnitude hard-cull)
                                                             (eq:dist-cull)
```

(b) is equivalent to `a_ij = 0` by eq:align-gate's one-sided compact
support.  (c) — present in all four contact kernels
(`compute_cslc_penetration`, `jacobi_step`, `compute_target_W`,
`write_cslc_contacts`) — is **not** in the theoretical smooth-energy
form but is required for closed convex targets: the half-space form
admits samples on the far side of a sphere or box whose face normals
satisfy (a) and (b) yet sit geometrically distant from the pad sphere.
The smooth-tail `w_t ≈ ε / (2·d_t)` at `d_t ≫ r_i` is small per sample
but, multiplied by the unbounded `phi_eff = raw_face^{1.5}` on a curved
target, can sum to spurious force contributions of order tens of
newtons.  Threshold `1×10⁻²` corresponds to `d_t ≈ r_pad + 5ε`; past
that, `w_t · phi_eff` is below MuJoCo's solver resolution.

(d) is the **distance-magnitude cull**.  The half-space form
`raw_face = −n̂_face · (q − t)` is a *local* approximation — it treats
the target's tangent plane at `t_j` as an infinite plane extending in
the `−n̂_face` direction.  For a closed convex target, a pad sphere
on the OPPOSITE side of the target body from `t_j` can pass culls
(a)–(c): their outward normals end up antiparallel in world frame so
`α_ij = +1` passes (b); they lie on the same axis through the body
so `d_t = 0` and `w_t ≈ 1` passes (c); and the half-space form gives
`raw_face = ‖q − t‖` (positive 50-100 mm), passing (a).  The
resulting phantom contact emits `(stiffness · raw_face) ≈ 5×10⁴ ·
0.1 ≈ 5000 N` of spurious force per pair.  Discovered on the dome
scene where dome-rim spheres paired with ball back-side targets
across the scene during APPROACH; the box scene is immune because
all its surface samples share a single outward normal direction.
Bounding `‖q − t‖ ≤ 3·r_i + 5·ε` admits typical legitimate
configurations (deepest plausible `‖q − t‖ ≤ √5 · r_i ≈ 2.24·r_i`
at `δ ≤ r_i` with `d_t ≤ r_i`) while culling scene-scale phantoms
(`‖q − t‖ ~ 50-100·r_i`).

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

At the `raw = −50ε` cull threshold `σ_ε(raw) ≈ 0.005 ε`,
`φ_eff = σ_ε · √(σ_ε + ε) ≈ 0.005 ε^{1.5}`, and `Σ_ε(raw) ≈ 1.0×10⁻⁴`
*individually*; neither factor is small on its own.  What justifies
the cull is the **product** that enters every force term:

```
φ_eff(−50ε) · Σ_ε(−50ε)  ≈  5×10⁻⁷ · ε^{1.5}                (eq:cull-product)
```

(seven orders of magnitude below the saturated-contact product at
`ε = 5×10⁻⁴ m`).

---

## 4. Single-pair contact

For one (pad sphere `i`, target sample `j`) pair with the Hertz-like
`φ_eff` of eq:phi-eff and the partition-of-unity share `share_ij`
of eq:share, the kernel applies the **physical force on the pad
sphere** (acting at `q_i`) as

```
f_phys_ij = + k_c · A_j · share_ij · φ_eff_ij · gate_ij · n̂_face_j   (eq:f-phys)
```

with `share_ij = (w_t_ij · a_ij) / W_j` and
`W_j = Σ_k  w_t_kj · a_kj` summed over the pad spheres that pass the
§3.6 active-set culls for target `j`.  The force is along `+n̂_face_j` —
the target pushes the pad outward along the target face's outward
normal, regardless of where the pad's own normal points or whether
the pad centre has crossed the face.  This is the half-space's
signature: contact direction is **face-defined, not line-of-centres**.
The corresponding **load on `δ_i`** (kernel convention
`load = −∂E/∂δ`) is

```
f_load_ij = − k_c · A_j · share_ij · φ_eff_ij · gate_ij · n̂_face_j   (eq:f-load)
```

In the saturated regime `raw_face_ij ≫ ε` we have `β_ε ≈ raw_face`
and `Σ_ε ≈ 1`, so

```
f_phys_ij  ≈  k_c · A_j · share_ij · raw_face_ij^{1.5} · n̂_face_j   (Hertz-like)
```

with `dF/d(raw_face) ∝ √raw_face → 0` at face contact.

**Partition-of-unity invariant.**  Summing the per-pair force over all
pad spheres that reach target `j`:

```
Σ_i  f_phys_ij  =  k_c · A_j · ⟨φ_eff_j⟩ · gate_j · n̂_face_j        (eq:per-target)
```

where `⟨φ_eff_j⟩ = Σ_i share_ij · φ_eff_ij` is the share-weighted
average per-pair coupling on target `j`, and `Σ_i share_ij = 1` by
construction (eq:share).  The per-target sum is independent of the
number of pad spheres covering the target — refining the pad lattice
redistributes `share_ij` across more spheres but does not change the
sum.  Total force on the pad body is then a Riemann sum

```
F_body  =  Σ_j  k_c · A_j · ⟨φ_eff_j⟩ · gate_j · n̂_face_j           (eq:F-body)
       =  Σ_i Σ_j  f_phys_ij                                       (consistency)
```

over target Voronoi cells, matching the hydroelastic structure
`∫_Ω k_h · φ · n̂_face dA` (eq:per-target is the discrete sample of
the integrand at `t_j` with weight `A_j`).

**W_j stability across the sweep.**  `W_j` depends on `δ` through
`w_t_ij(δ)` and `a_ij(δ)`, so the partition normalises against a
state-dependent quantity.  In practice the active set stabilises
within ~5 damped-Jacobi sweeps after warm-start (sub-mm `δ` changes
versus mm-cm target spacing leave `d_t` and `α` nearly invariant), so
`W_j` is updated per sweep but Picard-style — each iteration sees
the previous iteration's `W_j` in its denominator — without a
performance-critical refresh frequency.

**Note on energy form.**  An exact `½·k_c·A_j·share·φ_eff²` quadratic
energy would imply a `gradient = k_c·A_j·share·φ_eff·(dφ_eff/d(raw_face))`
form, which differs from eq:f-phys by the trailing factor and by
the `share_ij` dependence on `δ` (which makes `share_ij` itself a
function of the displacement field).  The kernel writes eq:f-phys
directly (force-form law, not energy-form gradient), so the lattice
equilibrium of §6 should be read as a fixed-point of *forces*, not
the stationary point of a globally-defined potential.  In the
saturated regime the discrepancy is the ratio
`gate · phi_eff'(raw_face) / phi_eff(raw_face)`; for `raw_face ≫ ε`
this is `1.5 / raw_face`, dimensionally a stiffness rescaling, not
a direction error — the contraction property of the damped-Jacobi
sweep (§6.6) still holds.

---

## 5. Total contact on a pad sphere

Per pad sphere `i`, sum over all target samples:

```
F_contact_i = Σ_j  k_c · A_j · share_ij · φ_eff_ij · gate_ij · n̂_face_j   (eq:F-contact-i)
```

with `share_ij = (w_t_ij · a_ij) / W_j` (eq:share) and
`W_j = Σ_k  w_t_kj · a_kj` summing over reaching pad spheres
(physical force; load form is `−F_contact_i`).  Active-set culling
(§3.6: `raw_face_ij ≥ −50·ε`, `a_ij > 0`, `w_t_ij ≥ 10⁻²`,
`‖q_i − t_j‖ ≤ 3·r_i + 5·ε`) restricts the sum to pairs that are
simultaneously geometrically nearby, on the correct side of the
target face, inside the tangential-locality disc, and within
distance of the target sample.  The `(A_j · share_ij)` factor
reconstructs the surface integral
`∫_{contact_patch} k_c · φ_eff · n̂_face dA` from the discrete sample
set: each target Voronoi cell's area `A_j` is split among the pad
spheres reaching it via `share_ij` weights that sum to 1, so the
total body force `Σ_i F_contact_i = Σ_j k_c · A_j · ⟨φ_eff⟩_j ·
gate_j · n̂_face_j` is independent of pad lattice density.

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

Consequence: curvature-driven Poisson bulging does not occur.  Curved
pad lattices (dome) still *flatten* against a target — that's the
kinematic consequence of the anchor + contact balance, independent of
the lateral law — but no surface sphere develops `δ_n < 0` under static
contact.

### 6.3 Contact

See §5.  Per-sphere load: `−F_contact_i`.

### 6.4 Friction

Stick-slip via the smooth form

```
f_t = − scale · δ_t,     scale = (k_stick · M) / (k_stick · ‖δ_t‖ + M + δ_floor)
                         M     = μ · f_n
                         f_n   = |F_contact_i · n̂_i|                (eq:friction)
```

with `K = k_stick`, `M = μ·f_n`.  The `δ_floor = 10⁻³⁰` term in the
denominator is a numerical guard for the `k_stick = 0 ∧ μ·f_n = 0`
corner (no-friction limit); in any active friction regime
`k_stick·‖δ_t‖ + M ≫ δ_floor` so the floor is invisible.  Two
properties:

* `f_n = |F_contact_i · n̂_i|`  — the aggregate normal-axis magnitude
  of `F_contact_i` projected on the pad's *own* outward normal
  (not the per-target `n̂_face_j`).
* The local frame for `δ_t` is the pad sphere's `n̂_i`, not the
  per-target `n̂_face_j` — friction is a property of the pad's
  compliant skin, which has one outward normal per lattice sphere.

### 6.5 Equilibrium per sphere

Written as `∂E_tot/∂δ_i = 0` — every term is `+∂E_X/∂δ_i` (physical
force on `q_i` per §2), all signs positive:

```
0  =  + k_a δ_{n,i} n̂_i + (k_a ρ) δ_{t,i}            (anchor,  eq:f-anchor)
      + k_l Σ_{j∈N(i)} (δ_i − δ_j)                    (lateral, eq:f-lat)
      + F_contact_i                                   (contact, eq:F-contact-i)
      + f_friction_i                                  (friction, eq:friction)
                                                         (eq:equilibrium)
```

### 6.6 Damped-Jacobi diagonal (per-axis stabilisation)

The contact spring `k_c · n̂_face · n̂_face^T` does not act along the
pad sphere's own normal `n̂_i` when the pad is curved.  Decomposing
in pad sphere `i`'s local rest-normal frame with `α_{ij} =
−(n̂_face_j · n̂_i) = cos α`, the per-axis contact-stiffness diagonal
contributions are

```
S_{n,i} = k_l · |N(i)|  +  k_c · Σ_j  A_j · share_ij · gate_ij · cos²α_ij
S_{t,i} = k_l · |N(i)|  +  k_c · Σ_j  A_j · share_ij · gate_ij · sin²α_ij
                                                                          (eq:S-axes)
```

with `share_ij = (w_t_ij · a_ij) / W_j` as in §4.  Pre-partition-of-
unity the same sum used `A_j · w_t_ij · a_ij` directly; under
partition of unity each per-pair contribution to the diagonal is
share-weighted, matching the corresponding factor in the rhs `f_phys`
so the iteration's contraction property carries over unchanged.
Numerically the diagonal is `~π×` smaller than the pre-PoU value in
dense-overlap regions of the contact patch, reflecting that each pad
sphere now bears only a fraction of the local load.

The damped-Jacobi update solves the linearised one-iteration system
in each axis separately:

```
δ_{n,i}^{k+1} = (1−α) · δ_{n,i}^k  +  α · (rhs_{n,i} + S_{n,i}·δ_{n,i}^k)
                                       / (k_a    + S_{n,i} + c_over_dt)
δ_{t,i}^{k+1} = (1−α) · δ_{t,i}^k  +  α · (rhs_{t,i} + S_{t,i}·δ_{t,i}^k)
                                       / (k_a·ρ  + S_{t,i} + c_over_dt)
```

with `α ∈ (0, 1)` the relaxation factor (typical 0.3–0.6) and
`c_over_dt` the §11 lattice-damping term.  On flat pads `cos²α = 1`
everywhere so `S_{t,i}` reduces to the pure lateral-Laplacian diagonal.
On curved pads (dome) off-apex spheres see `α` up to ~70°, so
`sin²α ≈ 0.9` of the contact stiffness lands on the tangent axis;
bracing `S_{t,i}` with this term restores the iteration's contraction
property when the tangent-axis residual is dominant.

---

## 7. Target abstraction

A target is a `PointSetTarget(positions, normals, areas)`.  No radii.
Per-sample area `A_j` is the Voronoi area of sample `j` on the
underlying surface; it absorbs sampling density so the per-pair sum in
§5 approximates the surface integral `∫ k_c φ n̂_face dA`.

### 7.1 Concrete samplers

```python
make_flat_face_target(centre, normal, span_u, span_v, pitch)
    # Regular grid on a flat face.

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

---

## 8. MuJoCo contact emission

Per active `(pad_sphere i, target_sample j)` pair that passes the §3.6
cull AND two emission-only thresholds (below), `write_cslc_contacts`
writes one MuJoCo contact slot with

| Field | Value |
|---|---|
| `point0`   | `q_def_i = p_i_world − δ_i` (deformed pad centre, body-local) |
| `point1`   | `t_j` (target sample, target-body-local) |
| `normal`   | `−n̂_face_j` (face's outward normal, pointing toward pad) |
| `offset0`  | `r_i · (−n̂_face_j)` (body-local) |
| `offset1`  | `0` |
| `margin0`  | `0` |
| `margin1`  | `0` |
| `stiffness`| `k_c · A_j · share_ij · gate_ij · √(β_ε(raw_face_ij) + ε)` (eq:emit-K) |
| `damping`  | `c_d` (per-contact damping, see §11) |
| `friction` | `1.0` (SCALE on geom-pair `μ`, not `μ` itself — see below) |

`margin0 = 0` (was `r_i`) and the matching switch from `raw_ij` to
`raw_face_ij` in the stiffness make MuJoCo reconstruct the
**face-penetration** scalar:

```
solver_pen = margin0 + margin1 − (point1 − point0) · normal
           = 0 + 0 − (t_j − q_def_i) · (−n̂_face_j)
           = − n̂_face_j · (q_def_i − t_j)
           = raw_face_ij                                         ✓
```

so MuJoCo's per-contact force `stiffness · solver_pen` equals

```
F_MJ_ij  =  k_c · A_j · share_ij · gate_ij
            · √(β_ε(raw_face_ij) + ε) · raw_face_ij
         ≈  k_c · A_j · share_ij · raw_face_ij^{1.5}              (eq:F-emit)
         =  F_lat_ij                                              (matches §5)
```

The emitted stiffness is the **secant slope**
`F(raw_face)/raw_face = k_c · A · share · gate · √raw_face` of the
Hertz-like force, not the local derivative.  Newton's MuJoCo binding
back-solves MuJoCo's `solref(timeconst, dampratio)` so that the
emitted force at the constraint is exactly `stiffness · solver_pen`
— a linear-Hookean per-step constraint, not a linearisation of a
nonlinear law.  Using the secant slope makes the per-step linear
constraint deliver the correct Hertz force at the current depth, and
matches `jacobi_step`'s internal per-pair force exactly (eq:f-phys).

**`W_j` shared with the lattice solve.**  The same partition-of-unity
normaliser `W_j` consumed by `jacobi_step` during the n_iter sweeps is
re-evaluated on the converged `δ` and reused inside
`write_cslc_contacts` (via the `share_ij` in eq:emit-K).  This keeps
MuJoCo's per-contact force and the lattice solver's per-pair force
bit-identical at the converged state.

**Emission gates not in the lattice solve.**  Two extra thresholds
apply *only* on emission (the lattice solver §6 sees the full smooth
form):

```
(a)  gate_ij  ≥  0.5   ⇔  raw_face_ij ≥ 0   (positive face penetration only)
(b)  w_t_ij   ≥  10⁻²                       (matches §3.6 cull)
```

(a) drops the negative-raw_face smooth tail from MuJoCo's solver slots
— emitting it floods the CG solver with thousands of near-zero
constraints, turning a 2 ms step into a 45 ms step, without changing
lattice equilibrium (the tail's `φ_eff` is exactly 0 since
`β_ε(raw_face ≤ 0) = 0`).  (b) mirrors the §3.6 cull.  The §3.6
distance-magnitude cull (eq:dist-cull) and raw-cull (eq:inactive-raw)
are inherited from the upstream culls — pairs that fail those never
reach the emission gates.

**Friction scale `1.0`, not `μ`.**  Newton's MuJoCo binding treats
`rigid_contact_friction` as a SCALE multiplied onto the geom-pair
base friction (= `μ`):

```
effective_μ  =  geom_friction_max  ×  rigid_contact_friction
              =  μ                  ×  1.0  =  μ                  ✓
```

Writing `μ` would give `effective_μ = μ²`.

**`K_max` budget.**  Each pad sphere is allocated `K_max` slots
(`K_max = 32` default, sized geometrically by
`cslc_main.grasp.objects.compute_k_max`).  Excess slots are filled
with the `shape_a = −1` sentinel for downstream MuJoCo conversion to
cull.  If a pad sphere has more active emittable pairs than `K_max`,
an atomic counter increments and `CSLCHandler.launch` raises a
`RuntimeWarning` on CPU read after `collide()`.

---

## 9. Sample numbers (for grounding)

At sample defaults (`r_pad = 1.5 mm`, `pad_spacing = 3 mm`,
`R_object = 33.5 mm` tennis ball, `n_samples_object = 1500` Fibonacci):

```
A_j_sphere   =  4π R² / n_samples   ≈  4π (33.5e-3)² / 1500 ≈ 9.4e-6 m²
mean spacing ≈  √A_j_sphere         ≈  3.1e-3 m  ≈ pad_spacing  ✓ (matched)
```

so the sample-density choice is "match target spacing to pad spacing".
This is a configuration default; the model itself does not pin it down.

---

## 10. Calibration

The pad's bulk material stiffness `k_e_bulk` is mapped to the
kernel-side `k_c` in two steps.

**Step 1 — three-spring series identity** (`cslc_data.calibrate_kc`,
called by `CSLCHandler.from_model_with_lattices` with the target
body's stiffness):

```
1/k_c^sphere  =  N_contact / k_e_bulk  −  1/k_a  −  1/k_e_target      (eq:calibration)
```

with `N_contact = ⌈contact_fraction · n_surface⌉` (default
`contact_fraction = 0.3`).  `k_c^sphere` has units `[N/m]` and is the
per-pad-sphere contact stiffness such that the per-pair chain (anchor
`k_a` ⊕ contact `k_c^sphere` ⊕ target modulus `k_e_target`, in series)
aggregated over `N_contact` active pairs reproduces `k_e_bulk` exactly:

```
N_contact · (1/k_a + 1/k_c^sphere + 1/k_e_target)⁻¹  =  k_e_bulk
```

Omitting `k_e_target` (rigid-target limit, `k_e_target → ∞`) recovers
the two-spring form `1/k_c^sphere = N_contact / k_e_bulk − 1/k_a`.

When the analytic formula has no positive solution — anchor or target
too soft for the requested `k_e_bulk` at the given `N_contact`, i.e.
`1/k_a + 1/k_e_target ≥ N_contact / k_e_bulk` — `calibrate_kc` falls
back to the conservative `k_c^sphere = k_e_bulk / N_contact` and emits
a `RuntimeWarning` with the minimum-`k_a` hint.

**Step 2 — per-volume rescale to kernel units** (handler):

```
k_c  =  k_c^sphere  /  A_kernel,     A_kernel = π · r_pad²            (eq:kc-volume)
```

— this is the `k_c` consumed by the kernels (`cslc_data.CSLCData.kc`,
units `Pa · m^(−1/2)`).  Per-volume rescale is required because
`jacobi_step` multiplies by the per-sample Voronoi area `A_j` and the
locality kernel `w_t`; the tiling identity (§3.5)

```
Σ_j A_j · w_t_ij  =  A_kernel  =  π · r_pad²
```

makes `k_c^sphere = k_c · A_kernel` an exact equality on a uniform
target sampling, which is what closes the loop between the per-sphere
calibration target and the per-pair force law of §4.

**Calibration under partition of unity.**  The per-sphere identity
(eq:calibration) was derived treating each engaged pad sphere as an
independent spring at saturation `k_c · A_kernel`.  Under partition
of unity (§4), each pad sphere's contribution to the body force is
share-weighted: per-target Voronoi area `A_j` is split via `share_ij`
summing to 1 over reaching spheres, so the AGGREGATE force on the
body is independent of how many pad spheres cover the patch.  The
N_contact term then represents how many INDEPENDENT target cells the
patch covers, not how many pad spheres engage — `N_contact ≈
contact_area / A_j` with `A_j` the typical target Voronoi area.  In
practice the production calibration (kc_per_volume = 1e10,
contact_fraction = 0.3) still gives reasonable force levels at
δ = 1 mm post-PoU, but the calibration is now lattice-density-
invariant by construction: refining the pad sampling does not require
re-tuning `k_c`.  Empirical box/box at δ=1mm shifts from 125.6 N
(pre-fix shelf inflated) to 43.2 N (post-fix-D1, no shelf) at the
same kc, much closer to the hydroelastic target.

**Anchor/contact balance regime.**  With `k_a = 3.5×10⁴ N/m` and
`k_c = 10¹⁰ Pa · m^(−1/2)`, the per-sphere normal-axis effective
contact stiffness `1.5·k_c·A_j·share_ij·√raw_face` at typical
operating depths is much smaller than `k_a`, so the lattice δ is
anchor-dominated and tracks the local equilibrium.  Sweeping kc
empirically (`1e10 → 1e8 → 1e6`) on the canonical grasp scene shows
the transition between "δ ≈ φ_rest tracking the warm-start" (kc=1e10,
max_δ ≈ 0.6 mm) and "δ governed by anchor balance" (kc=1e8,
max_δ ≈ 0.01 mm).  Both regimes preserve grasp stability; the kc=1e8
regime makes the compliant-lattice machinery load-bearing (anchor
spring matters), while the kc=1e10 regime keeps maximum contact
stiffness at the cost of an anchor-irrelevant lattice.  See `params.py`
docstring for the production tuning rationale.

---

## 11. Lattice velocity damping

The lattice solver is quasi-static (no acceleration term in §6.5);
energy injected by a moving pad body therefore sits in the lattice as
oscillatory `δ` modes until contact / friction dissipates it.  For
grasping at human-hand speeds this is harmless, but on stiff
trajectories (PD-tracked motion, sudden contact ramp) the modes ring
audibly in the emitted contact wrench.  A dissipative force
proportional to the rate of change of `δ` damps these modes:

```
f_damp_i  =  − c_lattice · δ̇_i ≈ − (c_lattice / dt) · (δ_i − δ_i^{prev step})
                                                                  (eq:f-damp)
```

discretised by backward Euler against the previous step's converged
`δ` snapshot.  Substituting into eq:equilibrium gives an
**implicit-Euler** form per axis:

```
(k_a   + S_{n,i} + c_lattice/dt) · δ_{n,i}^{new}  =  rhs_n + (c_lattice/dt) · δ_{n,i}^{prev step}
(k_a·ρ + S_{t,i} + c_lattice/dt) · δ_{t,i}^{new}  =  rhs_t + (c_lattice/dt) · δ_{t,i}^{prev step}
                                                                  (eq:implicit-damping)
```

so `c_lattice / dt` appears on **both sides** — added to the diagonal
*and* added to the RHS as `(c_lattice/dt) · δ^{prev step}`.  This form
is unconditionally stable: increasing `c_lattice` monotonically pulls
`δ^{new}` toward `δ^{prev step}`.  Setting `c_lattice = 0` makes the
term identically zero and recovers §6.5 exactly.

A second damping channel `cslc.dc` ([N·s/m]) lives **inside** the
emitted MuJoCo contact (§8 `damping` field) rather than the lattice
solver, with semantics

```
dc = 0   ⇒  timeconst = √(imp / k_e)  ≈ 0.030 s  (MuJoCo stiffness-derived)
dc > 0   ⇒  timeconst = 2 / dc                  (explicit)
```

**Avoid the dead zone `0 < dc < ~67`**, where the explicit branch
gives a LARGER timeconst than the stiffness-derived default — the
contact becomes SOFTER than `dc = 0` and stiff scenes (dome on tennis
ball) catastrophically diverge.

---

## 12. Fixed constants

Implementation invariants — not configurable at runtime:

* All sign conventions (`q = p − δ`; `load = −∂E/∂δ`).
* `INACTIVE_RAW_EPS_FACTOR = −50.0` (applied to `raw_face`, not `raw`),
  `EPS_ALIGN_DEFAULT = 0.05`, `ε_default = 5×10⁻⁴ m`, tangential cull
  `w_t < 1×10⁻²`, distance cull `‖q − t‖ > 3·r_i + 5·ε`.
* Face-penetration scalar `raw_face = raw − r_i = −n̂_face · (q − t)`
  (eq:raw-face) consumed by `φ_eff`, `gate`, and the raw cull.
* Smooth surrogate `β_ε` (Hermite quintic, eq:beta) for the `φ_eff`
  positive-part clamp — C² with `β_ε(0) = 0` exactly, replacing the
  C^∞ `σ_ε` whose `σ_ε(0) = ε/2` floor propagated through `φ_eff`.
* Partition-of-unity normaliser `W_j = Σ_k w_t_kj · a_kj` (eq:share),
  computed per damped-Jacobi sweep on the current `δ` (per-iter
  refresh; see §4 note on stability) and reused inside
  `write_cslc_contacts` for MuJoCo parity.
* Anisotropic anchor decomposition (eq:E-anchor).
* Friction smooth surrogate (eq:friction; `K = k_stick`, `M = μ·f_n`).
* Warm-start: `(K_n + k_c·I) δ_n = k_c · φ_rest` on the normal axis,
  `K_t · δ_t = k_c · f_t` on the tangent axis, both per §6.6.
* Lattice graph topology (`neighbor_indices`, `neighbor_counts`).
* Three-spring calibration `1/k_c^sphere = N_contact/k_e_bulk − 1/k_a
  − 1/k_e_target` done UPFRONT in `CSLCHandler.from_model_with_lattices`
  (eq:calibration); see §10 for the partition-of-unity reinterpretation
  of `N_contact` as the count of independent target Voronoi cells the
  patch covers.
* Per-volume rescale `k_c = k_c^sphere / (π · r_pad²)` in the handler
  (eq:kc-volume) — kernels expect per-volume `k_c`, not per-sphere.
* MuJoCo emission convention (§8): `margin0 = 0` so `solver_pen =
  raw_face`; emission gate `gate_ij ≥ 0.5` ⇔ `raw_face ≥ 0`.
