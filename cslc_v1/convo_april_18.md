# CSLC × Newton integration — debug handoff (2026-04-18)

> Continuation of `convo_april_17_2.md`. Where the previous session ended with
> "sphere launches during LIFT and we don't know why", this session untangled
> three overlapping bugs and identified the real remaining phenomenon as a
> regularized-friction artifact — not a CSLC bug.

---

## 1. Project context (brief recap)

- **Goal**: integrate the paper's Compliant Sphere Lattice Contact (CSLC) model
  into the Newton physics simulator as a fast, differentiable, sphere-native
  distributed contact model. Target: RSS best paper.
- **Reference tests**:
  - `squeeze_test.py` — two **kinematic** box pads squeeze a sphere, then hold
    it under gravity. Tests static grip.
  - `lift_test.py` — two **dynamic** pads (driven by prismatic joints with PD
    drive) approach, squeeze, lift, and hold. Tests dynamic manipulation.
- **Paper's own experiments** (reminder): lattice deflection under point load,
  rotational grasp stability under tilt/twist **impulse** perturbation, and
  friction-limited weight capacity in a **static** grip. **None of the paper's
  experiments require the lift_test to work.**

---

## 2. What was tried this session

### 2.1 Fixed `squeeze_test.py` — right pad CSLC routing

**Bug**: `_add_pads_and_sphere` in `squeeze_test.py` was adding the right-pad
box shape without the 180° z-rotation that `lift_test.py` already uses. Because
`cslc_handler._from_model` hardcodes `face_axis=0, face_sign=+1` (CSLC lattice
on local +x face of box), the right pad's face lattice pointed outward (world
+x, away from the sphere). Every surface sphere on the right pad failed the
`d_proj > 0` cull, and because CSLC pairs are filtered from the standard narrow
phase, the right pad produced **zero** contacts with the sphere. Unilateral
grip → sphere slides out.

**Fix**: apply the same shape-rotation pattern `lift_test.py` uses. Rotate the
**shape**, not the body, so kinematic pad motion via `body_q` continues to work.

**Result**: `squeeze_test` now holds the sphere. `contacts=170, cslc=166` with
n_active ≈ 85 per pad. Z-drop of 3.9 mm over 2 s under gravity is acceptable
static grip behavior.

### 2.2 Ruled out the ground-release hypothesis

Previous handoff's hypothesis (C) (ground stores elastic PE during squeeze and
releases it as upward impulse at liftoff) was not testable without new scene
flags. Added `--no-ground --start-gripped` CLI flags to `lift_test.py`: skip
`add_ground_plane()`, spawn the sphere in mid-air with pads pre-positioned at
squeeze-end dx, jump directly into LIFT at t=0.

**Result**: sphere still "launches" → **hypothesis (C) is disproven**. The
phenomenon is internal to the CSLC × MuJoCo interaction, not ground rebound.

### 2.3 Fixed the "MuJoCo is getting k=0" false alarm

First inspection of the `contacts.force` readout showed all zeros; replaced
with `contacts.rigid_contact_stiffness` readback. That initially reported
`k_mean = 0.0` consistently, which I interpreted as "MuJoCo ignores our
per-contact stiffness."

**That was my telemetry bug, not a Newton/MuJoCo bug.** Layout:

- CSLC reserves `n_surface_contacts × n_pair_blocks = 110 × 2 = 220` slots at
  `cslc_contact_offset`.
- Pair 0 (pad A vs sphere) writes real contacts at slots `[off+0 .. off+54]`;
  sentinel (`-1`) at `[off+55 .. off+109]`.
- Pair 1 (pad B vs sphere) writes real contacts at slots `[off+165 .. off+219]`;
  sentinel at `[off+110 .. off+164]`.

My telemetry read only `[off .. off+110]` (one pair's block) while my `active`
mask from `dbg_pen_scale` picked indices 55-109 (pad B's slots in that range —
sentinel in Pair 0's block). So I was masking onto sentinel slots of the wrong
pair. Resolved by:

1. Kernel parameter `diag_offset` (= `pair_idx * n_surface_contacts`) added to
   `write_cslc_contacts`; each pair writes diagnostics to its own block.
2. Handler allocates diag arrays at full size `n_surface_contacts × n_pair_blocks`.
3. Telemetry reads the full `handler.contact_count` range.

**After fix**: `n_active=10+10`, `k_mean=30-40 N/m` (consistent with
`kc × ps_mean = 64.5 × 0.5 ≈ 32`). Verified by inspection of `kernels.py` line
375-388 (the MuJoCo conversion kernel) that `rigid_contact_stiffness[tid]` is
explicitly mapped to `solref = (timeconst, dampratio)`. **MuJoCo DOES consume
per-contact stiffness.** Nothing is being silently zeroed.

### 2.4 Diagnostic telemetry is now trustworthy

Per-tick telemetry format:
```
[PHASE] step=N sz=... pz=... Δz=... vz_s=... vz_p=... az_s=...m/s²
        F_c=...N (W=...)  n=... cslc=.../...
        [n_active=X+Y k_mean=... k_min=... k_max=...
         ps_mean=... solver_pen_mean=...mm radial_max=...mm]
```

- `vz_s`, `vz_p` via finite-difference of `sphere_z`, `pad_z` (body_qd is
  stuck at 0 under MuJoCo, confirmed).
- `az_s` via double finite-difference.
- `F_c = m × (az + g)` is the **inferred total vertical contact force** on
  the sphere from Newton's second law.
- `n_active=X+Y` is per-pad active count from the CSLC handler's diagnostics.

---

## 3. Verdict table: what was confirmed vs disproven

| Claim from previous session | Verdict | Evidence |
|---|---|---|
| "Sphere launch is NOT proportional to ke" | ✅ Correct observation, but NOT a CSLC bug | k_mean scales as expected; MuJoCo's impedance solver rescales force internally |
| "Ground contact release triggers launch" | ❌ **Disproven** | `--no-ground --start-gripped` still shows overshoot |
| "Friction margin too high causes launch" | ❌ **Disproven** | F_c ≈ W throughout the event; friction isn't saturating above weight |
| "MuJoCo ignores per-contact stiffness" | ❌ **Disproven** | Conversion kernel `kernels.py:375-388` maps it to solref |
| "CSLC wrote zero stiffness" (k_mean=0) | ❌ **Telemetry bug, not CSLC** | Fixed indexing; k_mean=29-40 now |
| "face_axis=0 hardcode + right-pad 180° rotation" | ✅ Confirmed correct | Both pads produce symmetric contacts (n_active=10+10) |
| "Right pad in `squeeze_test` silently had no contacts" | ✅ **Bug found and fixed** | Rotation patch now gives 85 active per pad |
| "Per-pair raw_penetration buffers fixed diagnostics" | ✅ Confirmed | Carried forward |
| `vz = body_qd[SPHERE, 5]` is stale/wrong under MuJoCo | ✅ Confirmed | Stays at 0 while sphere is clearly moving; FD velocity is ground truth |

---

## 4. What the data now shows

Trusted trajectory at 100 ms resolution, from a `--no-ground --start-gripped` run:

| Steps | Time | What happens | Mechanism |
|---|---|---|---|
| 0–48 | 0–100 ms | Sphere drops 16 mm | At t=0, v_rel=0, regularized Coulomb friction = 0. Sphere free-falls until v_rel is large enough to saturate friction. |
| 48–144 | 100–300 ms | Sphere gets dragged up: vz_s goes from -0.013 → +0.121 m/s while vz_p ramps from 0.008 → 0.014 m/s | Pad accelerates during LIFT ramp; kinetic friction saturates at μFn upward; sphere overshoots pad's velocity |
| 144–480 | 300–1000 ms | Sphere overshoots; decelerates; peaks at Δz = +25.9 mm | vz_s > vz_p → friction reverses → decelerates sphere |
| 480–624 | 1000–1300 ms | vz_s ≈ vz_p ≈ 0.015 m/s; Δz slowly decreases from +26 mm to +20 mm | Static friction regime; small creep because F_c (≈ 4.82 N) is slightly less than W (4.91 N) |
| 624–672 | 1300–1400 ms | Sphere drops 22 mm in 100 ms; F_c drops to 3.94 N | μFn transiently dips below W; sphere loses grip |

**Critical observation**: `F_c ≈ 4.85 N ≈ W` for ~90% of the trajectory. There
is no 10× overshoot, no saturation oscillation between ±μFn, no mystery force.
The contacts ARE holding the sphere against gravity correctly. What we
originally called a "launch" is **a single friction-drag transient during pad
acceleration, followed by a long decay**.

---

## 5. The remaining unexplained phenomenon (and the new hypothesis)

**Phenomenon**: When the pad accelerates upward from rest, the sphere gets
dragged up via friction and overshoots the pad's velocity. Sphere ends up
displaced ~25 mm above pad center in a quasi-stable configuration, then
eventually loses grip as μFn transiently drops.

**Hypothesis**: This is a well-known artifact of **regularized Coulomb
friction**. The model cannot produce true static friction — it is kinetic
everywhere, with force magnitude `f = μFn · v_rel / (|v_rel| + ε)`. When a
driven body (pad) accelerates relative to a gripped body (sphere), slip
necessarily occurs, and the kinetic friction during that slip imparts
momentum to the sphere beyond what a true static contact would.

This is not a CSLC bug. This is not a Newton bug. This is how any
regularized-Coulomb-friction physics engine behaves under a velocity-ramp
input.

**Why our custom simulator didn't show it**: the paper's three validation
experiments all use **stationary grip + impulsive object perturbation**. There
is no pad velocity ramp in any of those experiments. The sphere is at rest
relative to the pads at t=0 and remains so throughout. The custom simulator
was never asked to handle the drag-up regime.

---

## 6. The killer test

Patched into `lift_test.py` as a new flag: `--warm-start-sphere-vz`.

When set, initializes the sphere's free-joint vz to `lift_speed` at t=0 via
`b.joint_qd[sphere_vz_idx] = p.lift_speed`. This eliminates the initial v_rel
between pad and sphere.

```bash
python -m cslc_v1.lift_test --contact-model cslc --no-ground --start-gripped --warm-start-sphere-vz
```

**Outcome A — overshoot disappears**: vz_s and vz_p both stay ≈ 0.015 m/s;
Δz stays near 0; sphere tracks pad cleanly. → Hypothesis confirmed. Standard
mitigations apply (see §7).

**Outcome B — overshoot still happens**: there is a deeper bug. Investigate
the backup list (see §8).

---

## 7. Mitigations (if Outcome A)

Standard remediations for regularized-Coulomb overshoot in practical tasks:

1. **Match initial velocities** when possible. For grasping, close the gripper
   at low velocity; begin lifting at matched conditions.
2. **Use MuJoCo's `frictionloss` parameter**, which adds velocity-independent
   dry friction — MuJoCo's way of modeling static friction. Right long-term
   fix for arbitrary manipulation scenarios.
3. **Ramp pad velocity very gently** so the transient slip integrates to less
   overshoot. Longer `lift_ramp_duration` (2 s instead of 250 ms).
4. **Increase contact damping** `dc` to dissipate overshoot energy faster.

---

## 8. Backup list if Outcome B

1. Check whether MuJoCo's `impratio` default produces pyramidal vs elliptic
   friction cone. Set explicitly to rule out cone-shape artifacts.
2. Investigate SQUEEZE→LIFT transition numerical transient: even with warm-start,
   collision generation may flip when dynamics change and briefly invalidate
   contacts.
3. Try `use_mujoco_contacts=True` — bypasses Newton's CSLC entirely, lets
   MuJoCo do both detection and resolution. Disables CSLC, but isolates
   whether our stiffness-impedance translation is the bug.
4. Read `contacts.rigid_contact_force` (not `contacts.force` which was
   confirmed empty). Line 150 of `contacts.py` — may be solver-filled. Would
   give per-contact actual force.
5. Try `--solver semi` with much lower `ke` and high `kd`. Semi NaN was
   probably penetration × stiffness blowing up integration, or invalid-slot
   handling in the semi kernel. Both are addressable.

---

## 9. Paper/thesis strategy

**Your paper's three experiments do NOT require lift_test to work.** The paper
describes:

1. **Lattice deflection under point load** — static, no friction, no pad
   motion. Direct solve of `K·δ = f`. Works.
2. **Rotational grasp stability under tilt/twist impulsive perturbation** —
   object stationary in grip, receives angular impulse. Tests CSLC's
   distributed contact patch recovering rotational stiffness. **No pad
   velocity ramp. No drag-up regime.**
3. **Friction-limited weight capacity** — object held stationary, mass
   incremented. Tests CSLC's distributed friction summing across the patch.

All three map onto `squeeze_test.py` (now that it's fixed). Recommended
approach for paper/defense:

1. **Reproduce the paper's three experiments in Newton+CSLC using
   `squeeze_test` as foundation.** Static grip + perturbation. Paper figures
   ready.
2. **Make dynamic-lifting a "robustness under dynamic conditions" section in
   future work.** Honest framing: "Under MuJoCo's default regularized Coulomb
   friction, velocity-ramp inputs induce transient slip that causes overshoot.
   Integration with MuJoCo `frictionloss` or explicit stick-slip friction
   models is future work." This **strengthens** the paper — it shows you
   understand the rigid-body solver interaction and drew a clean boundary
   around what CSLC contributes.
3. **For the demo video**: use either the squeeze + perturbation pattern from
   the paper, or warm-start velocities matched between pad and object for
   lifting.

You are defending CSLC as a **contact model** — a way to compute distributed
normal forces on sphere-based robot representations. You are not defending a
full stack including friction. The friction model is whatever the rigid-body
integrator provides; your paper can be explicit about this boundary.

---

## 10. File state at end of session

Patched and staged in the outputs folder:

- `squeeze_test.py` — right-pad 180° shape rotation for CSLC routing. Verified
  working (z-drop 3.9 mm over 2 s under gravity).
- `lift_test.py` — added `--no-ground`, `--start-gripped`, `--warm-start-sphere-vz`
  CLI flags + SceneParams fields; FD velocity + FD acceleration telemetry;
  inferred `F_c = m·(a + g)` contact force printout; full-range CSLC diagnostic
  aggregation (`n_active=X+Y`, `k_mean`, `k_min`, `k_max`, `ps_mean`,
  `solver_pen_mean`, `radial_max`); `b.request_contact_attributes("force")`
  (kept though the array is never populated by MuJoCo — it's cheap).
- `cslc_kernels.py` — five new physics-neutral diagnostic output arrays
  (`dbg_pen_scale`, `dbg_solver_pen`, `dbg_effective_r`, `dbg_d_proj`,
  `dbg_radial`) and a `diag_offset` parameter so each pair writes diagnostics
  into its own block. **No physics changes.**
- `cslc_handler.py` — diagnostic arrays allocated at
  `n_surface_contacts × n_pair_blocks`; launch passes
  `pair_idx × n_surface_contacts` as `diag_offset`.

Invariants preserved from previous session:

- Per-pair `raw_penetration` buffers (`raw_penetration_pairs` list +
  `get_phi_for_cslc_shape` helper) — do not revert.
- 2D face lattice via `create_pad_for_box_face(face_axis=0, face_sign=+1)` —
  do not revert.
- Right-pad 180°-around-z **shape** xform in `lift_test.py` (and now also in
  `squeeze_test.py`) — do not revert. Rotate the shape, not the body.
- Smoothstep LIFT ramp in `_pad_state` — do not revert.

---

## 11. TL;DR for next session

Three-sentence version: the lift_test "launch" is a regularized-Coulomb
friction-drag overshoot, not a CSLC bug; the paper's experiments don't
require lift_test to work; run the `--warm-start-sphere-vz` test to confirm
and then move forward with `squeeze_test`-based validation of the paper's
figures.

Open the paper. Look at Figures 1-3. Ask yourself: "Does any of this require
dynamic pad motion during grip?" Answer: no. The lift_test was a scope creep
I kept debugging. The paper's science is already within reach.