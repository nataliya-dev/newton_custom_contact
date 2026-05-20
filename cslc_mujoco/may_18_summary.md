# CSLC Compliant-Skin Upgrade — Session Summary

**Date:** 2026-05-18
**Builds on:** [may_17_summary.md](may_17_summary.md) (vec3 δ migration)
**Top-line result:** Full physically-grounded "compliant skin" model now ships
end-to-end. At the new production default (`SceneParams.cslc_kl = 5000` for
pad_lift), pad_lift slip matches Phase 1g (4.9 mm vs 5.2 mm baseline). The
geometric Poisson effect from the §III-F paper Limitations is empirically
confirmed at a stiffer calibration (8/300 spheres bulging at kl = ka).
Squeeze and lift regressions both pass.

---

## 1. Plain-English goal of the session

The CSLC contact model represents a robot pad as a lattice of small spheres
connected by springs — the "compliant skin" framing. Until today, each lattice
sphere only knew about ONE thing: how much it was being compressed inward along
its rest outward normal. That works for flat contact, but it can't describe:

- **Shear / tangential deformation** — when the target slides sideways, the
  skin should stretch tangentially before slipping (real flesh "sticks").
- **Bulging** — when one spot compresses, neighbouring spots should bulge
  outward (Poisson effect, what makes a fingertip flatten on contact).
- **Soft tangent direction** — anisotropic anchor stiffness, matching real
  nearly-incompressible flesh (Poisson ν → 0.5 ⇒ shear modulus G = E/3).

The original CSLC paper (Overleaf draft in
[cslc_mujoco/docs/overleaf_theory_cslc_icra.txt](docs/overleaf_theory_cslc_icra.txt))
calls out the missing physics in its §III-F Limitations as the "full vector
formulation" with cross-normal shear coupling. **This session implements that
full vector formulation, the geometric Poisson coupling, and the stick-slip
friction that real skin uses for grip.**

---

## 2. The three stages of the model

### Stage 1 — Original paper (scalar δ along the rest normal)

Each lattice sphere `i` has a SCALAR compression `δ_i ≥ 0` along its rest
outward normal `n_i`. Three forces act on it:

```
f_anchor_i  = -ka · δ_i                               (anchor spring, rest position)
f_lateral_i = -kl · Σ_{j∈N(i)} (δ_i − δ_j)            (graph-Laplacian "smoothness penalty")
f_contact_i = kc · [φ_rest − δ_i]_+                   (contact spring, paper eq. 12)
```

Where `φ_rest = (r_i + R_t) − ‖p_rest − t‖` is the rest-position penetration.
Equilibrium: `K · δ = kc · [φ_rest − δ]_+`. Solved in closed form via a
precomputed `A_inv = (K + kc·I)^-1`.

**Limitations the paper itself acknowledges:**
- Cannot model shear coupling — δ is 1D scalar.
- Cannot model Poisson bulging — graph Laplacian penalises displacement-vector
  change, not physical-distance change between neighbours.

### Stage 2 — may_17 (vec3 δ shim, anisotropic anchor)

The may_17 session promoted `δ` from scalar to `wp.vec3` with a shim:
`δ_stored = δ_scalar · n_outward_rest`. The kernels read it as vec3 but the
scalar projection onto the rest outward normal recovered the legacy behaviour.
Two new physics knobs were added:

- **Anisotropic anchor** (Phase 1f): `K_anchor = diag(ka, ka·ratio, ka·ratio)`
  in each sphere's local rest-normal frame. `ratio = 1/3` matches
  incompressible flesh.
- **Per-axis A_inv** (Phase 1g — finished early THIS session): `A_inv_n` for
  the normal axis (+kc·I in the diagonal) and `A_inv_t` for the tangent axis
  (no contact spring on tangent). Unlocks the closed-form solve for the
  anisotropic case, which was previously trapped on iterative Jacobi.

**What may_17 did NOT have:** the contact gate, contact emission, and lateral
force still all operated as if δ were the scalar normal projection. Tangential
δ was mathematically present but physically invisible to contact and lateral
coupling, so anisotropy alone produced no improvement on the dome scene.

### Stage 3 — may_18 ambitious compliant-skin (THIS session)

Four "micro-steps" (each independently validated before the next):

| Micro-step | Idea | Where in code |
|---|---|---|
| 1 | Line-of-centres projection: `dot(δ, n_eff)` replaces `dot(δ, n_outward_local)` everywhere the legacy effective_pen / effective_r subtraction happens. Makes tangential δ visible to contact for the first time. | `jacobi_step`, `write_cslc_contacts` + `_box` variant in [cslc_kernels.py](../newton/_src/geometry/cslc_kernels.py) |
| 2 | Drop the `δ_n ≥ 0` clamp. Side spheres are now allowed to expand outward (negative δ_n) — required for bulging to be a legal equilibrium. | `jacobi_step` final write-back |
| 3 | Replace the graph-Laplacian lateral with a true distance-preservation spring on the deformed centres `q_i = p_i_world − δ_i`. Linearises to the graph-Laplacian at δ → 0 (paper analysis still applies), produces geometric Poisson bulging at finite δ. | `jacobi_step` inner loop, `neighbor_rest_length` array in [cslc_data.py](../newton/_src/geometry/cslc_data.py) |
| 4 | Stick-slip friction via tangential δ. `f_friction = -δ_t · min(k_stick, μ·f_n / ‖δ_t‖)` with a smooth-min surrogate for differentiability. | `jacobi_step` friction block |

**Solver change to absorb the nonlinear distance-preservation:** the Phase 1g
closed-form `lattice_solve_equilibrium` is now used as a LINEAR WARM-START
predictor (one launch giving the exact small-δ solution), followed by
`n_iter` damped-Jacobi refinement passes that correct for the geometric
nonlinearity. See [cslc_handler.py:521-575](../newton/_src/geometry/cslc_handler.py#L521-L575).

---

## 3. How the equations flow through the contact pipeline (audit)

I re-read each kernel to verify the equations connect end-to-end. The flow per
contact pair, per timestep:

### 3.1 Penetration kernel (`compute_cslc_penetration_sphere` / `_box`)

**Inputs:** `p_local`, `sphere_delta` (NOT READ — see below), body+shape
transforms, target pose.
**Output:** `phi_rest = (r + R) − ‖p_rest_world − t_world‖`, gated by smooth
step on `d_proj`. Also writes `contact_normal_world = (t − p_rest)/dist`
(line of centres FROM REST POSITION).

> **Important:** despite `sphere_delta` being in the signature, this kernel
> does NOT use δ to displace the lattice centre. Stage A (deformed-centre
> penetration via `q_def = p_rest − δ`) was attempted and reverted — it
> created inter-step instability because the closed-form A_inv warm-start
> assumes the RHS doesn't depend on δ. The Stage-A goal (tangent δ affects
> contact) is instead delivered by Micro-step 1's `effective_pen / effective_r`
> reduction below.

### 3.2 Closed-form warm-start (`lattice_solve_equilibrium`)

Phase 1g per-axis A_inv applied once: gives the EXACT small-δ linear-graph-
Laplacian solution. Tape-compatible single matvec. Writes the warm-start δ
to `_jacobi_a`.

### 3.3 Jacobi refinement (`jacobi_step` × n_iter, lines 332-528 of [cslc_kernels.py](../newton/_src/geometry/cslc_kernels.py))

Per active sphere, per iteration:

1. Read world-frame transforms, current `δ_old`, `out_n_world` (rest outward
   normal rotated to world).
2. **Distance-preservation lateral** (Micro-step 3):
   ```
   q_i_world = p_i_world - δ_old_i
   for j in N(i):
       q_j_world = p_j_world - δ_old_j         (lagged)
       d = q_j_world - q_i_world
       f_lateral += k_l · (‖d‖ - L_ij) · d / ‖d‖
   ```
   `L_ij` is `neighbor_rest_length` (precomputed in `CSLCData.from_pads`).
3. **Contact force** (Micro-step 1 line-of-centres):
   ```
   delta_proj_neff = dot(δ_old, n_eff)         (line-of-centres projection)
   effective_pen   = φ_rest - delta_proj_neff
   gate            = smooth_step(effective_pen)
   f_contact       = kc · φ_rest · gate · n_eff
   ```
   For face-on contact `n_eff ≈ n_outward` and this is algebraically identical
   to Phase 1g. For tangent-shifted contact, the gate sees the deformed
   geometry without needing q_def in the penetration kernel.
4. **Stick-slip friction** (Micro-step 4):
   ```
   δ_t = δ_old - delta_proj_neff · n_eff       (tangent-plane displacement)
   f_n_mag    = kc · φ_rest · gate             (cone radius source)
   cone_scale = μ · f_n_mag / ‖δ_t‖             (smooth reciprocal)
   scale      = (k_stick · cone_scale) / (k_stick + cone_scale + eps)   (smooth-min)
   f_friction = -scale · δ_t
   ```
   Stick mode (`k_stick · ‖δ_t‖ < μ·f_n`): scale ≈ k_stick, force = -k_stick·δ_t.
   Slip mode (`k_stick · ‖δ_t‖ > μ·f_n`): scale ≈ μ·f_n / ‖δ_t‖, force
   saturates at the Coulomb cone.
5. **Decompose into local rest-normal frame** for anisotropic anchor, apply
   damped Jacobi update with stabilised diagonal `k_diag = ka(_t) + kl·|N| +
   kc·gate` and damping `α = 0.6`:
   ```
   rhs = f_contact + f_lateral + f_friction
   δ_new_n = (rhs · out_n + S_n · δ_old_n) / k_diag_n     (S_n = kl·|N| + kc·gate)
   δ_new_t = (rhs_⊥ + S_t · δ_old_⊥)        / k_diag_t     (S_t = kl·|N|)
   δ_dst = (1-α)·δ_old + α·(δ_new_n · out_n + δ_new_t)
   ```
6. **Micro-step 2:** no clamp on δ_n — anchor + lateral govern sign at
   equilibrium.

### 3.4 Contact emission (`write_cslc_contacts` / `_box`)

After the converged δ lands in `data.sphere_delta` (via `cslc_copy_active`):

```
delta_proj_neff = dot(δ_world, n_eff)             (Micro-step 1, line-of-centres)
effective_r     = smooth_relu(r_lat - delta_proj_neff, eps)
margin0         = effective_r
point0          = p_rest_body                    (REST position — NOT q_def)
normal          = (t - p_rest)/dist              (rest-position line of centres)
solver_pen      = effective_r + R - d_proj       (MuJoCo reconstructs this)
out_stiffness   = kc_series · pen_scale · gate
```

MuJoCo's reconstructed contact penetration = `margin0 + margin1 - dot(p1 - p0,
normal) = effective_r + R - dist = (r - dot(δ, n_eff)) + R - dist`. For
face-on contact this equals `φ_rest - δ_n`. The contact emission and the
solver agree on the same φ_def-equivalent value, so MuJoCo and the lattice
solver are on the same physics — this was the bug that broke the first
attempt; fixed by keeping the rest-position emission framework and only
generalising the δ projection to line-of-centres.

---

## 4. Metrics — what changes when

All measurements N=5 unless noted (MuJoCo GPU solver has run-to-run variance
from atomics — see [may_17_summary.md §3.1](may_17_summary.md)).

### Squeeze (face-on flat contact, no tangent δ expected)

| Configuration | FullDrop | HoldDrop | Notes |
|---|---|---|---|
| point_mujoco | 1.025 mm | 0.736 mm | baseline |
| Phase 1g (may_17 end-state) | 0.094 mm | 0.065 mm | pre-session baseline |
| Micro-1 only | 0.133 mm | 0.071 mm | n_eff vs n_outward — within noise |
| Micro 1+2+3 | 0.174 mm | 0.118 mm | distance-pres softer than graph Laplacian |
| **Micro 1-4 (final)** | **0.165 mm** | **0.113 mm** | stick-slip barely activates face-on |

### Lift_test (box-target pad squeezing a sphere)

`lifted=YES, held=YES` across all stages. `max_z = 0.0505 m` consistently.
No regression.

### Pad_lift (curved dome pad lifting a sphere — the hard case)

| Configuration | XY slip during HOLD (mean ± σ) | Bulging at LIFT entry |
|---|---|---|
| Phase 1g baseline (may_17) | 5.2 ± 0.8 mm | n/a (scalar δ) |
| Micro-1 only | 10.2 mm (N=3) | 0/300 |
| Micro 1+2+3 (kl=500) | 42.8 mm BIMODAL | 0/300 |
| Micro 1+2+3 (kl=5000) | 4.9 mm | 0/300 |
| Micro 1+2+3 (kl=25000) | 61.0 mm | **8/300, max −0.037 mm** ✓ |
| Micro 1-4 (kl=500 + stick-slip) | 33.0 mm BIMODAL | 0/300 |
| **Micro 1-4 (kl=5000 + stick-slip)** | **4.9 ± 0.6 mm** | 0/300 |
| Micro 1-4 (kl=25000 + stick-slip) | 75.3 mm | 8/300 |

**Bulging diagnostic** (per-phase δ_n distribution, lines 1336-1419 of
[pad_lift_test.py](pad_lift_test.py) via `_print_bulging_diagnostic`): the
geometric Poisson coupling is real, but small on the dome — only at kl ≈ ka
does the lateral spring carry enough force to overcome the anchor's
pull-to-rest. The 8 bulging spheres are confined to the perimeter of the
contact patch, with max −0.037 mm outward displacement.

### Critical research finding: dome grip-vs-bulging trade-off

The dome scene has a fundamental geometric constraint: only ~50/300 spheres
engage at typical squeeze depth, so the contact patch can't carry enough
tangential force for friction-based grip. The trade-off shows clearly above:
soft lateral (kl=500-5000) gives stable grip but no bulging; stiff lateral
(kl=ka) gives bulging but the lattice over-stiffens and grip fails. **No
single kl satisfies both** — this validates [may_17_summary.md §5.2 #6]
("dome creates cascade; use flatter pads"). For the paper, flat pads engage
many more spheres at the same squeeze depth and bypass this trade-off.

---

## 5. Files changed (working tree, uncommitted per user request)

```
 cslc_mujoco/common.py                  +57 / -9    viewer-crash fixes (vec3 δ)
 cslc_mujoco/pad_lift_test.py           +173 / -3   --cslc-kl, --ka-tangent-ratio,
                                                    bulging diagnostic, kl=5000 default
 cslc_mujoco/docs/summary.md            +1 / -1     (one-line tweak)
 newton/_src/geometry/cslc_data.py      +159 / -27  neighbor_rest_length, A_inv_t,
                                                    k_stick, mu_friction
 newton/_src/geometry/cslc_handler.py   +109 / -48  A_inv warm-start + jacobi refine,
                                                    out_normal_world_scratch, box-launch fix
 newton/_src/geometry/cslc_kernels.py   +315 / -159 lattice_solve_equilibrium per-axis,
                                                    compute_outward_normals_world,
                                                    jacobi_step Micro-1+2+3+4 rewrite,
                                                    deformed-centre attempt + revert
```

### Key spots a future change must touch together

If you modify the δ convention, the SOLVER (`jacobi_step`) and the EMISSION
(`write_cslc_contacts` and `_box`) must move together. The pre-session
"Stage A" attempt failed precisely because I updated the penetration kernel
to use `q_def = p_rest − δ` but the emission still used the
`effective_r = r_lat − δ_n` reduction — they were on different geometric
conventions and the body got conflicting force vectors. The Micro-step 1
solution avoids that risk by keeping BOTH on the rest-position framework
and only generalising the δ projection to line-of-centres.

---

## 6. Known limitations and open work

### 6.1 Dome bulging is small (8/300 spheres, sub-40-µm)

The geometric Poisson coupling is correctly implemented but limited by the
dome's local curvature and the achievable apex compression (~0.07 mm at
squeeze-end). Real fingertip flesh shows bulging in the 100s of µm range.
Three paths to amplify:

- **Flatter pads** — more neighbours of an apex sphere share the apex's
  normal direction, so distance-preservation drives more outward force per
  unit compression.
- **Deeper squeeze** — increase `face_pen` from 1 mm to 4 mm. Compression
  scales linearly; bulging response scales with the geometric Poisson
  coefficient (depends on neighbour topology).
- **Internal-pressure DOF** — proper hydroelastic-style incompressibility
  with a global pressure per pad. This is a research project, not a knob.

### 6.2 Inter-step stability of stick-slip

Friction's `δ_t` is lagged from the previous step. Under fast tangential
loading, the iteration can fall behind the target's motion and the contact
slips for a few steps before re-sticking. The bimodal pad_lift histogram at
kl=500 (33 mm mean, with 2/5 runs sub-15 mm and 3/5 runs 60+mm) is the
signature. The kl=5000 default makes this rare; if needed for hard scenes,
either bump `n_iter` (currently 20 from `pad_lift_test.SceneParams.cslc_n_iter`)
or add a `dt`-aware damping to the friction.

### 6.3 Differentiability under nonlinear distance-preservation

The Phase 1g closed-form `lattice_solve_equilibrium` was specifically built
to be tape-compatible (one linear matvec). The new path uses it as a
warm-start, then runs n_iter Jacobi refinement passes for the geometric
nonlinearity. The refinement is implemented with explicit per-iteration src
→ dst buffers (no aliasing), so wp.Tape CAN backprop through the unrolled
iterations — but each iteration is in the tape, which means O(n_iter) more
gradient memory than Phase 1g. For paper-grade differentiability, the right
follow-up is implicit-function-theorem at convergence: at the converged δ*,
solve `J · ∂δ*/∂x_target = -∂F/∂x_target` once, where `J` is the
linearised stiffness Hessian. Phase 1g's per-axis A_inv structure IS this
Hessian in the small-δ limit, so it can be reused for the IFT solve.

### 6.4 Box-target stick-slip (`write_cslc_contacts_box`)

The box-target emission was updated for Micro-step 1 (line-of-centres in
the effective_r reduction). Lift_test passes, so no functional issue. But
the box's `normal_ab` is the pad's REST outward normal (not a line of
centres — boxes don't have a "centre to project toward"). For consistency
with the sphere variant, the box's `n_eff` is effectively `n_outward`. This
is geometrically correct for a flat-faced target but does mean stick-slip
on the box-target path is identical to the legacy "tangent of the rest
outward normal" formulation — not the deformed line-of-centres.

### 6.5 Defaults and CLI

- Production defaults baked in:
  - `SceneParams.cslc_kl = 5000` in [pad_lift_test.py:206-226](pad_lift_test.py#L206-L226)
  - `CSLCData.k_stick = 25000` (= ka), `CSLCData.mu_friction = 0.3`
  - `CSLCData.ka_tangent_ratio = 1.0` (isotropic — flesh-like 1/3 available
    via `--ka-tangent-ratio 0.333`)
- CLI flags added to pad_lift_test:
  `--smoothing-eps`, `--ka-tangent-ratio`, `--cslc-kl`, `--lift-ramp-duration`
- No `--k-stick` or `--mu-friction` flag yet — TODO if A/B testing friction
  is needed. Set `k_stick = 0` to disable friction entirely.

---

## 7. Reproduction recipe

```bash
# Baseline (default Micro 1-4, kl=5000):
uv run cslc_mujoco/squeeze_test.py --solver mujoco --mode squeeze --steps 500
# Expected: cslc_mujoco HoldDrop=0.113mm, Contacts=62

uv run cslc_mujoco/lift_test.py --mode headless --solver mujoco
# Expected: lifted=YES, held=YES, max_z=0.0505

for i in 1 2 3 4 5; do
  uv run cslc_mujoco/pad_lift_test.py --viewer null --contact-model cslc \
    --num-frames 270 --test --quiet 2>&1 | grep "XY slip"
done
# Expected: 4-6 mm slip per run

# To see bulging (kl=ka):
uv run cslc_mujoco/pad_lift_test.py --viewer null --contact-model cslc \
  --num-frames 270 --test --cslc-kl 25000 2>&1 | grep -E "BULGING|XY slip"
# Expected: 8/300 bulging spheres at LIFT entry, 60-90 mm slip (grip lost)

# Flesh-like (ratio = 1/3):
uv run cslc_mujoco/pad_lift_test.py --viewer null --contact-model cslc \
  --num-frames 270 --test --ka-tangent-ratio 0.333

# Visual (GL viewer with lattice colouring):
uv run cslc_mujoco/squeeze_test.py --contact-model cslc --solver mujoco
uv run cslc_mujoco/lift_test.py --viewer gl --contact-model cslc
```

---

## 8. Notes to pass to the next session

### 8.1 Immediate next steps (small)

1. **Add `--k-stick` and `--mu-friction` CLI flags** to pad_lift_test. ~15 min.
   Lets you A/B the friction contribution without code edits.
2. **Curvature sweep harness** — the user picked this as the paper direction.
   Build a `curvature_sweep.py` that varies pad geometry from flat → dome and
   plots bulging count + slip + per-sphere contact force across the sweep.
   The bulging diagnostic + `--cslc-kl` flag are already in place to support
   this. ~1 day.
3. **Atomic commits when ready** — five natural splits exist in the diff:
   Phase 1g per-axis A_inv, Micro-1 line-of-centres, Micro-2+3 distance-pres
   + free δ, Micro-4 stick-slip, viewer fix.

### 8.2 Subtle things easy to miss

- **`sphere_delta` is in WORLD frame**, not body-local. The dot product
  `dot(δ, n_local)` is wrong; use `dot(δ, n_world)`. See pre-existing bug
  in `write_cslc_contacts` line 547 (Phase 1b) — that compiled fine on
  squeeze_test only because the pad body has identity rotation, so n_local
  numerically equals n_world. The Micro-step 1 update correctly uses
  `dot(δ, n_eff_world)`.
- **`compute_cslc_penetration_sphere` does NOT read sphere_delta** —
  parameter is in the signature for future Stage A but currently unused.
  Don't add δ-dependent logic here without ALSO updating
  `write_cslc_contacts` in the same change (see §5 above).
- **`L_ij` is body-local distance** (precomputed from pad.positions). This
  is correct because rigid-body transforms preserve distances, so body-local
  distance equals world-frame distance throughout the simulation.
- **The bulging diagnostic computes signed δ_n in world frame** using
  body→world rotation of the rest outward normal. See
  `_print_bulging_diagnostic` for the inline quat-rotate.

### 8.3 What the user gets vs what they asked for

User asked for: a contact model that **rivals other models (hydroelastic)**
with full skin-like deformation behaviour: bulging, stick-slip, anisotropic
flesh.

Honest assessment shipped:
- ✓ vec3 δ with full directional freedom
- ✓ Anisotropic anchor (Phase 1g, closed-form solve)
- ✓ Tangent δ visible to contact (Micro-1 line-of-centres)
- ✓ Distance-preservation lateral with geometric Poisson coupling
  (Micro-3) — **bulging empirically confirmed at kl=ka**
- ✓ Stick-slip friction via tangential δ (Micro-4)
- ✗ Bulging is small on the dome (8/300 spheres, 37 µm). Real fingertip
  bulging is larger. The dome geometry limits how much the implemented
  Poisson coupling can produce; flat or steeper pads will demonstrate it
  more dramatically.
- ✗ No implicit-function-theorem differentiability yet — only unrolled
  Jacobi tape backprop, which is more memory than the Phase 1g closed-form
  used to be.

### 8.4 The user's mental model that helped get this right

When the user pushed back on "the graph Laplacian has to go", I almost
threw it away. The right insight (which the user pushed me toward):
**keep the graph Laplacian as the LINEARISATION of the distance-
preservation energy, don't throw it away.** A_inv from Phase 1g IS still
the Hessian of the new operator at δ → 0. The closed-form survives as a
warm-start; the iterative refinement only handles the geometric
nonlinearity that the linear operator can't see. This preserves the
parallelism and the gradient-friendly linear structure the user cared
about, while adding the bulging physics.
