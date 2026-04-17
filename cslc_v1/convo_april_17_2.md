# CSLC × Newton integration — debug handoff (2026-04-17)

> **Read this first.** This document is the accumulated state of the CSLC-in-Newton
> integration after four rounds of iterative debugging. It tells you what was tried,
> what we confirmed, what's still broken, and — critically — what the remaining
> launch behaviour is **not** caused by. If you're the next agent picking this up,
> don't repeat the parameter-tuning loop; read the "What remains unexplained" and
> "Next experiments" sections and set up the crucial decoupling tests **first**.

---

## 1. Project context

- **Goal**: integrate the paper's Compliant Sphere Lattice Contact (CSLC) model into
  the Newton physics simulator as a fast, differentiable, sphere-native distributed
  contact model. Target: RSS best paper.
- **Reference test**: `lift_test.py`. Two driven pads approach, squeeze, and lift a
  30 mm, 0.5 kg sphere through friction.
- **Phases**: APPROACH (pads reach sphere surface) → SQUEEZE (pads compress ~1 mm
  into sphere) → LIFT (pads rise; friction must carry the sphere) → HOLD.
- **Acceptance criterion**: sphere tracks the pads vertically during LIFT with bounded
  Δz (sphere_z − pad_z); no launch, no drop.

---

## 2. The four-round history — what was tried and the outcome

All rounds kept the paper's Stage-1 CSLC (ReLU activation, face-normal formulation
with `pen_scale` stiffness correction in `write_cslc_contacts`).

### Round 1 — scene parameter retune

**Hypothesis**: the squeeze depth was 12.5 mm (way outside small-deformation regime
since the paper uses 0.4–1 mm), producing degenerate kinematics.
- `approach_speed` set so APPROACH ends at the sphere surface (not 10 mm past it).
- `squeeze_speed` set to 2 mm/s so SQUEEZE compresses by exactly 1 mm.
- `ke` raised from 5e3 → 2.5e4 to give ~24 N per-pad grip force (paper-like).

**Outcome**: SQUEEZE became clean. `max pen_3d` dropped from 17 mm → 5 mm as
predicted. Grip force became ~24 N per pad. **But**: sphere launched at LIFT onset,
worse than before. `sphere_z` went from 0.03 m to 0.126 m — a 96 mm jump.

### Round 2 — per-pair `raw_penetration` buffers + smoothstep lift ramp

Per-pair buffers in `cslc_handler.py` were uncontroversially good: they fixed the
diagnostic artifact where pad 2 always reported `kernel1 phi>0: 0` after pair (4,5)
ran. **Do not revert this.**

A smoothstep ramp on `lift_speed` was added to `_pad_state` in `lift_test.py`.
It's C¹ continuous, verified numerically.

**Outcome**: SQUEEZE clean, per-pair diagnostic clean. LIFT onset softened but
sphere still launched — just more slowly. At this point the diagnosis was
"friction margin too high" (5× weight). We could see the mechanism: regularized
Coulomb saturates at μ·Fn as soon as slip > ε, applying 24 N upward force to a
4.9 N sphere.

### Round 3 — reduce friction reserve

`ke` reduced to 1e4, `lift_speed` reduced to 0.015 m/s, ramp extended to 0.25 s.
Expected: μ·Fn ≈ 10 N, margin ≈ 2× weight, sphere tracks pad with tiny lag.

**Outcome**: sphere still launched. Even at `ke=1000` (friction < weight per the
ke·pen calibration) the sphere **still rose ~50 mm before falling back**. That is
the observation that broke the "friction margin" hypothesis: if friction were
responsible, lowering it to below weight should have caused the sphere to **sink**,
not rise. The upward motion was not scaling with friction, so friction alone could
not explain the launch.

### Round 4 — 2-D face lattice (paper-faithful flat patch)

**Hypothesis (wrong)**: the volumetric box lattice has edge/corner spheres with
averaged normals (e.g. `(1,0,1)/√2`); when the target moves off-centre, these
tilted-normal spheres inject a vertical force component that amplifies any
vertical drift into a launch.

**Fix applied**: `cslc_handler._from_model` now calls `create_pad_for_box_face` with
`face_axis=0, face_sign=+1`. Right-pad's box shape rotated 180° around z so its
local +x maps to world -x (both pads present inward-facing local +x). `ke=1000` to
compensate for the more concentrated effective stiffness of the 2-D lattice.

**Outcome**: sphere still launches. `Δz` reaches +33 mm at step 1440, then the
sphere falls back and oscillates before eventually coming to rest on the ground
while the pads continue to hover 20 mm above.

### Post-hoc verification that the face-normal hypothesis was wrong

After the face-lattice change failed, I numerically verified whether face-normal
vs sphere-sphere normal produces a vertical restoring force when the target
drifts. It does **not**, under either convention. The active set is the disk of
lattice points within `r_lat + R` of the target's projection onto the face; that
disk moves *with* the target, staying symmetric around it. Net vertical force is
~0 by symmetry regardless of which normal is used. So neither face-normal nor
volumetric edge spheres are the root cause of the launch.

---

## 3. What we have confirmed works (don't touch)

- **Per-pair `raw_penetration` buffers** in `cslc_handler.py` (`raw_penetration_pairs`
  list + `get_phi_for_cslc_shape` helper). Diagnostic now shows each pad's phi
  correctly.
- **`verify_cslc_diag_active_set.py`** now reads from the per-pair buffers.
- **Smoothstep lift ramp** in `_pad_state`. C¹ verified; does not cause discontinuity.
- **2-D face lattice** `create_pad_for_box_face` in `cslc_handler`. SQUEEZE is clean
  with this lattice: `cslc=9/pad`, `max pen_3d ≈ 5 mm`, no edge-sphere contribution.
- **Right-pad 180°-around-z shape xform** in `lift_test.py`. Keeps joint kinematics
  unchanged (rotates shape, not body); makes both pads present local +x inward.
- **Extended telemetry** (`Δz`, `vz`, per-100-ms sampling) in `Example.step`.

---

## 4. What remains broken — the key observation

**In the most recent run (ke=1000, 2-D face lattice, ramped lift):**

| Step  | Phase  | sphere_z | pad_z | Δz      | Interpretation                          |
|-------|--------|----------|-------|---------|-----------------------------------------|
| 960   | LIFT 0 | 0.02947  | 0.02998 | -0.52 | Just before LIFT starts. Stable grip.   |
| 1008  | LIFT+80ms | 0.02950 | 0.03009 | -0.59 | Sphere still on ground (pen ~0.5 mm). Tracking pad. |
| 1056  | LIFT+200ms | 0.03032 | 0.03084 | -0.53 | Sphere just left ground. Still tracking. |
| 1104  | LIFT+300ms | **0.04027** | 0.03226 | **+8.01** | **Sphere leapt 10 mm in 100 ms. Pad rose 1.4 mm.** |
| 1200  | LIFT+500ms | 0.05795 | 0.03526 | +22.69 | Sphere 23 mm above pad.                 |
| 1440  | LIFT+900ms | 0.07545 | 0.04276 | +32.69 | Peak rise.                              |
| 1728  | HOLD   | 0.02478  | 0.05064 | -25.86 | Sphere has fallen back through and is oscillating on ground. |
| 3024+ | HOLD (late) | 0.02947 | 0.05061 | -21.14 | Sphere resting on ground; pads hover 20 mm above. Grip lost permanently. |

### Facts that narrow the diagnosis

1. **The launch is NOT proportional to ke**. Sphere rises ~30–50 mm at ke=1000,
   5000, 10000, 25000. Friction/normal force should scale with ke; the launch
   magnitude does not. **This rules out "friction margin too high" as the root cause.**

2. **The launch happens the instant the sphere leaves the ground** (between
   step 1056 and step 1104 — exactly when `sphere_z` crosses the ground plane).
   Before leaving ground: sphere tracks pad with Δz ≈ -0.52 mm. After: Δz diverges
   at 100 mm/s.

3. **At `ke=1000` the predicted friction is ≈ 1.7 N, below the 4.9 N weight.**
   The paper's calibration says the sphere should **drop** under gravity. Instead it
   rises. This is not compatible with the `F = kc·pen_3d` picture — some other
   force is present.

4. **Face-normal vs sphere-sphere normal is symmetric around the target.** The net
   vertical force contribution from the active set is 0 by construction under either
   convention (verified numerically). Neither is the launch mechanism.

5. **`vz = +0.0000` m/s in every telemetry line** even when `sphere_z` is clearly
   changing. `body_qd[SPHERE_BODY, 5]` is zero throughout. Either the MuJoCo solver
   doesn't populate `body_qd` on the state object (joint-space only), or the
   telemetry is reading the wrong index. **This means our velocity observations are
   meaningless; all velocity analysis in earlier rounds was based on finite
   differences of position, not this field.**

6. **Ground contact release is the trigger.** Sphere on ground is compressed by
   ~0.5 mm (ground penetration). Ground provides some portion of the 4.9 N support.
   At liftoff, that ground force transitions from ~4 N to 0 N. The sphere was at the
   edge of friction-only support; the transition creates a dynamic event we don't
   understand.

---

## 5. Suspected actual causes (not yet ruled in or out)

Ranked by how compatible they are with the facts in §4:

### (A) MuJoCo constraint solver with `use_mujoco_contacts=False`

Newton passes our `out_stiffness` and `out_damping` values into the MuJoCo
warp-backend solver. We have been assuming these map onto a Hunt–Crossley-like
penalty spring with `F = kc · pen_3d`. That assumption is unverified.

MuJoCo normally uses `solref` (time constant + damping ratio — units of seconds,
not N/m) and `solimp` (impedance) to define contact impedance. Contact forces under
that formulation are **not** `k · pen`; they are whatever the iterative constraint
solver computes to enforce non-penetration with the specified impedance. Under this
interpretation, `out_stiffness` values in the hundreds of N/m either map to
timeconst values (meaning a *slow* contact response, not a weak one), or are
rescaled by the solver before use, or are ignored entirely.

**If this is the cause**, tuning `ke` will never fix the behaviour — the solver is
producing whatever force it needs to stop penetration, independent of our
calibration. The launch magnitude being independent of `ke` is consistent with this.

### (B) State snapshot / integration mismatch

`vz = +0.0000` suggests the `body_qd` array is not what we think it is. If the
solver operates in joint space and `body_qd` is a stale FK result, then any
decision the physics pipeline makes based on sphere body velocity — e.g. contact
damping terms `dc · v_normal` — would see zero velocity and produce pathological
forces.

### (C) Ground-contact elastic rebound

With a 0.5 mm penetration into the ground and an unknown `ke_ground`, the stored
elastic energy could launch the sphere if the lift-off is sudden enough. A quick
back-of-envelope: `PE = ½ · ke · pen² = ½ · 1.2e6 · (0.5e-3)² = 0.15 J → h = 0.031 m`.
The observed rise is 0.03 m. That's a numerical coincidence worth taking seriously
but also not conclusive.

### (D) The `pen_scale` correction in `write_cslc_contacts`

`out_stiffness = cslc_kc · pen_3d / solver_pen`. For the face lattice with
`solver_pen ≈ pen_3d` at the patch centre, `pen_scale ≈ 1`. But at lattice spheres
near the edge of activation (small `pen_3d`, larger `solver_pen`), `pen_scale < 1`
and the per-contact stiffness is reduced. Under MuJoCo's constraint solver this
scaling might interact unpredictably with the impedance model.

---

## 6. Next experiments (the path forward)

**Do these in order, and do NOT tune any parameters until you've done the first
two — they're decoupling tests, not parameter searches.**

### 6.1 Solver-swap test (1 line change, ~5 min)

Run the lift_test with `--solver semi` (SolverSemiImplicit) instead of
`--solver mujoco`. Semi-implicit Euler is a straightforward penalty-based
integrator; if CSLC is correct, the semi solver should produce a clean lift.

- **If the sphere tracks the pads under `semi`**: the MuJoCo solver is the root
  cause. The fix is either (i) use `semi` for validation runs, (ii) configure MuJoCo
  to use `use_mujoco_contacts=True` and match MuJoCo's contact convention, or
  (iii) manually set per-contact `solref`/`solimp` instead of just `stiffness`/
  `damping`.
- **If the sphere still launches under `semi`**: the bug is in CSLC itself. Move to §6.2.

### 6.2 Remove the ground (2 line change, ~5 min)

Set `sphere_start_z = 0.05` (20 mm air gap) and reduce `sphere_density` so the
sphere doesn't fall too fast during APPROACH (or start the pads touching the sphere
immediately). This eliminates the ground-contact release as a confounder.

- **If no launch without ground**: hypothesis (C) is confirmed. Either use a softer
  ground (set `ke_ground` explicitly) or accept that lift tests require an air-gap
  start.
- **If still launches without ground**: the mechanism is internal to the
  CSLC-vs-MuJoCo interaction.

### 6.3 Point-contact control (no code change, already supported)

Run `python -m cslc_v1.lift_test --contact-model point`. This uses the same scene
geometry but with standard point-contact on the pad box (no CSLC).

- **If point-contact also launches**: the problem is the scene/solver, not CSLC.
- **If point-contact tracks cleanly**: the problem is specifically in how CSLC
  feeds contacts into the solver.

### 6.4 Read the ACTUAL contact forces

Replace the current `F ≈ kc · Σφ` estimate in the `CSLC_SUMMARY` print with the
real normal force the solver applied. The value is in
`contacts.rigid_contact_force` (or whatever Newton names it in this version —
grep `rigid_contact_force` in the narrow-phase or solver kernels). Compare to
our prediction. A big discrepancy confirms hypothesis (A).

### 6.5 Fix the velocity telemetry

The `vz = +0.0000` reading is not credible. In `Example.step`, compute sphere
velocity by finite difference: cache `sphere_z` from last call and divide by
`frame_dt`. The current `body_qd[SPHERE_BODY, 5]` reading may be stale or wrong
index; either diagnose (try indices 0–5, see which one changes during lift) or
switch to finite differences. Without reliable velocity, we can't see the launch
onset clearly.

### 6.6 If all above point to CSLC-solver interaction: Stage 2 smooth activation

Stage 2 (softplus + compact-support kernel) is already sketched in
`convo_april_17.md`. Once the forward dynamics is clean, Stage 2 gives the
differentiability the paper needs. Don't do Stage 2 until §6.1–6.5 are resolved;
smoothing activations on top of a broken forward pass won't help.

---

## 7. Pitfalls we fell into — don't repeat

1. **Diagnostic confusion about pad 2 vs pad 4**. The old shared `raw_penetration`
   buffer made it look like kernel 1 was broken for pad 2, because kernel 1's
   active-pad filter zeros the other pad. *Fixed* by per-pair buffers, but if you
   rewrite the handler, keep this in mind.

2. **"max pen_3d = 17 mm is fine because the lattice centers at ±hx protrude".**
   True, but the *effective stiffness* interpretation changes. The paper's
   calibration assumes `ke_bulk = effective stiffness at equilibrium`, which depends
   on how `pen_3d_avg` relates to face compression. With volumetric lattice and
   `pen_3d` including the `r_lat` protrusion, the user-specified `ke` produces
   different real-world behaviour than expected. If documenting the model in the
   paper, clarify whether `ke_bulk` is "per face compression" or "per lattice
   overlap".

3. **Chasing friction strength**. Three rounds of `ke` tuning changed the launch
   magnitude only slightly. Stop tuning once a parameter sweep at extreme values
   (ke=1000 vs ke=25000) doesn't produce a *qualitative* change.

4. **Believing `vz` in the telemetry**. See §6.5. The reported velocity is zero
   throughout. Don't build arguments on it.

5. **Hypothesising about off-centre contact force asymmetry** (face-normal vs
   sphere-sphere normal). The active set is centred on the target by construction;
   net vertical force is ~0 by symmetry under either convention. See the
   verification script at the end of §2 in the conversation.

6. **Assuming MuJoCo treats `out_stiffness` as a spring constant**. We don't have
   evidence either way. See §6.4.

---

## 8. Files and state

Production files in `newton/_src/geometry/`:
- `cslc_data.py` — unmodified; `create_pad_for_box_face` is the correct function
  for the current setup.
- `cslc_handler.py` — uses `create_pad_for_box_face(face_axis=0, face_sign=+1)`.
  Has per-pair `raw_penetration_pairs` list + `get_phi_for_cslc_shape` helper.
  **TEMP**: face axis/sign is hardcoded. A principled fix would add
  `model.shape_cslc_face_axis[i]` / `shape_cslc_face_sign[i]` via ShapeConfig.
- `cslc_kernels.py` — unmodified since the original Stage 1 fix. Uses face-normal
  formulation with `pen_scale` stiffness correction.
- `collide.py` — unmodified.

Scene / test files in `cslc_v1/`:
- `lift_test.py` — `SceneParams`: `ke=1.0e3`, `lift_speed=0.015`, ramp=0.25 s,
  `sphere_radius=0.03`, `sphere_density=4421` (mass 0.5 kg). Right pad's box shape
  rotated 180° around z. Extended per-100-ms telemetry.
- `verify_cslc_diag_active_set.py` — uses per-pair buffers.

Key scene-level invariants the current files assume:
- Both CSLC box shapes expose their local +x face inward (via body or shape rotation).
- Inward direction is aligned with world ±x (right pad rotated 180° around z).
- Face spacing = 10 mm (unchanged from original).

---

## 9. TL;DR for the next agent

You're handed a clean SQUEEZE, broken LIFT, and a working diagnostic. The remaining
failure is **not** a parameter-tuning problem — we've verified that. Two cheap
experiments (§6.1 solver swap, §6.2 no-ground) will narrow the cause in 10 minutes.
Do them before writing any more code. Then instrument actual contact forces (§6.4)
so you're reasoning about measured behaviour, not predicted behaviour. Only after
the forward dynamics is clean should you move to Stage 2 smooth activation for the
paper's differentiability claims.

## from the user on above two cheap experiments
note --solver semi -> produces nan
sphere_start_z = 0.05 -> this doesn't do anything because the sphere is under gravity
--contact-model point -> this is interesting because there is still a small jump during squeeze but the jump is smaller than the cslc one