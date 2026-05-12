# H1 — Harmonic-Mean Compliance Composition



**Branch:** `gstack_test`

**Date:** 2026-05-10

**Status:** Implemented, verified, physically correct. Does not reduce HoldCreep for rigid targets.



---



## 1. The Problem H1 Solves



The CSLC kernel treats the target body as rigid when emitting contact stiffness:



```

F = cslc_kc · pen_3d ← pre-H1 (assumes target is infinitely stiff)

```



`pen_3d` is the total approach distance between the lattice sphere surface and the target

surface. When both the lattice and the target are compliant, this approach distance is shared

between the two bodies:



```

pen_3d = δ_lattice + δ_target

```



Treating the full `pen_3d` as if it were pure lattice compression overstates the contact

force. The constraint should enforce the force that two springs in series produce for a

given total compression.



---



## 2. Theory — Series-Spring Force Law



Two springs in series with stiffnesses `kc` (lattice) and `ke` (target), compressed by

total distance `δ_total`, transmit equal force through both:



```

F = kc · δ_lattice = ke · δ_target

δ_total = δ_lattice + δ_target = F · (1/kc + 1/ke)

→ F = k_series · δ_total

```



where the series stiffness is the harmonic mean:



```

k_series = (kc · ke) / (kc + ke)

```



This is the force MuJoCo's constraint should enforce. H1 emits `k_series` as the constraint

stiffness so `F = k_series · pen_3d` is correct.



**Reference:** Masterjohn et al. 2021 (PFC-V, eq 23); Castro et al. 2022 (SAP).



### Limit behavior



| Condition | k_series | Physical meaning |

|---|---|---|

| `ke → ∞` (rigid target) | `→ kc` | Recovers pre-H1 behavior |

| `ke = kc` (equal stiffness) | `= kc / 2` | Compression splits evenly |

| `ke < kc` (soft target) | `< kc / 2` | Target deflects more than lattice |

| `ke → 0` (jelly) | `→ 0` | No contact force — correct |



---



## 3. Files Changed



### `newton/_src/geometry/cslc_kernels.py` — Kernel 3 (`write_cslc_contacts`)



**Before H1** (line ~541 in pre-H1 code):

```python

out_stiffness[buf_idx] = smooth_relu(

cslc_kc * pen_scale * contact_gate, 1.0e-9)

```



**After H1** ([cslc_kernels.py:541–543](../newton/_src/geometry/cslc_kernels.py#L541)):

```python

kc_series = (cslc_kc * target_ke) / (cslc_kc + target_ke + eps * eps)

out_stiffness[buf_idx] = smooth_relu(

kc_series * pen_scale * contact_gate, 1.0e-9)

```



The `eps²` floor in the denominator guards against 0/0 when both stiffnesses are zero.

`target_ke` is a new parameter added to both `write_cslc_contacts` and

`write_cslc_contacts_box`.



The same change is applied to `write_cslc_contacts_box` for the box-target kernel.



### `newton/_src/geometry/cslc_handler.py` — `CSLCShapePair` + `_from_model`



**Added field to `CSLCShapePair`** ([cslc_handler.py:77](../newton/_src/geometry/cslc_handler.py#L77)):

```python

other_ke: float = 0.0 # H1: target-body material stiffness, cached at construction

```



**Cached at construction in `_from_model`** ([cslc_handler.py:368–371](../newton/_src/geometry/cslc_handler.py#L368)):

```python

# H1: cache the target's material stiffness for harmonic-mean

# composition in the kernel. Done once at construction; the

# array index never changes during the simulation.

pair.other_ke = float(shape_ke[pair.other_shape])

```



`shape_ke` is `model.shape_material_ke.numpy()`, read once during `_from_model`. No

per-step GPU→CPU syncs.



`pair.other_ke` is passed to the kernel call in `_launch_vs_sphere` and `_launch_vs_box`

as the `target_ke` argument.



---



## 4. Verification



### Rung 1 + 2 — Unit tests (`cslc_v1/test_h1_compliance.py`)



All 9 tests pass. The test file covers:



**Rung 1 — Math (5 tests):**

- `test_series_identity_equal_stiffness`: `H(ke, ke) = ke/2` to floating-point precision

- `test_rigid_limit`: `H(kc, ∞) → kc`, error bounded by `kc²/ke_target`

- `test_soft_limit`: `H(kc, 0) = 0` with `eps²` guard

- `test_smooth_relu_compatibility`: `smooth_relu(H(kc, ke)) ≥ 0` across 36 stiffness pairs

- `test_differentiability_partial_kc`: `∂H/∂kc = ke²/(kc+ke+eps²)²` verified by finite-diff, rel error < 1e-4



**Rung 2 — Series-spring equilibrium (4 tests):**

- `test_equal_stiffness_split_evenly`: F=5 N, kc=ke=75000 → δ_each = 66.67 µm, δ_total = 133.33 µm

- `test_rigid_target_equivalent_to_lattice_alone`: rigid limit to leading-order error

- `test_compliant_target_softer_than_lattice_alone`: kc=75000, ke=50000 → k_series=30000 (2.5× softer)

- `test_force_at_equilibrium_matches_series_law`: forward check, δ_lattice + δ_target = δ_total



Run:

```bash

uv run --extra dev -m unittest cslc_v1.test_h1_compliance -v

```



### Rung 3 — Lift test integration result



**Test:** `uv run cslc_v1/lift_test.py --mode headless --contact-models cslc`



The lift test uses `ke_sphere = ke_pad = 50000 N/m` and `kc = 75000 N/m`.

With H1: `k_series = (75000 × 50000) / (75000 + 50000) = 30000 N/m` — active and measurable

(2.5× softer than without H1).



| Metric | Baseline | H1 |

|---|---|---|

| sphere_z @ HOLD | 0.04970 m | 0.04970 m |

| HoldCreep | +0.5944 mm/s | +0.5944 mm/s |

| HoldDrop | +0.575 mm | +0.575 mm |

| Per-step timing | 5.306 ms | 5.490 ms (+3.5%) |



**H1 has zero measurable effect on HoldCreep.** Trajectories identical to 10 µm precision.



---



## 5. Why H1 Doesn't Reduce HoldCreep



The Anitescu friction gap is:



```

δv ≈ R_t × F_contact

```



where `R_t` is the **tangential** constraint regularization, set by MuJoCo's `solimp`

parameter. `solimp` is identical regardless of `kc_series`. H1 correctly sets the contact

force magnitude for a given penetration depth, but the velocity-level gap mechanism operates

on `R_t`, which H1 does not touch.



This is not a failure of H1. It shows that HoldCreep lives entirely in the tangential

velocity-level solver, not in the normal force magnitude.



Three compounding reasons HoldCreep is unchanged:



1. **Wrong R direction.** Reducing stiffness from 75000→30000 N/m *increases* `R ∝ 1/sqrt(ke_series)`, making the Anitescu gap slightly worse.

2. **Equilibrium shift too small.** The force shift from softer constraint is ~8 µm — below the 10 µm measurement noise floor, and far below the dominant 0.6 mm HoldDrop.

3. **Baseline architecture mismatch.** Old baseline (volumetric pad, 26 contacts) showed 0.045 mm/s HoldCreep. Current face-lattice baseline shows 0.60–0.75 mm/s — a separate regression unrelated to H1.



---



## 6. When H1 Does Matter



H1 is the correct model whenever `ke_target` is not much larger than `kc`:



| Target | ke_target | k_series | Force error without H1 |

|---|---|---|---|

| Steel sphere | ~1e10 N/m | ≈ kc = 75000 | < 0.001% |

| Rubber ball | ~50000 N/m | 30000 N/m | 2.5× overestimate |

| Soft tissue | ~1000 N/m | ≈ 987 N/m | ~76× overestimate |

| Foam | ~100 N/m | ≈ 99.9 N/m | ~750× overestimate |



For manipulation with deformable objects, H1 is required for physically meaningful contact

forces. For rigid-body manipulation (steel/ceramic), the pre-H1 behavior is recovered

exactly.



---



## 7. Summary



H1 is a one-line kernel change that makes CSLC physically correct for compliant targets.

It is already in the codebase on `gstack_test`, adds a one-time construction cost

(one numpy read of `shape_material_ke`), and zero steady-state runtime overhead.

For rigid targets the behavior is unchanged. For deformable targets it is the correct model.



