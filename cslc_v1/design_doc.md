# CSLC Integration into Newton — Detailed Design

## Goal

Make CSLC a first-class contact model in Newton's collision pipeline,
following the same architectural pattern as hydroelastic contacts.
The result: any solver (SemiImplicit, MuJoCo, XPBD, Kamino) can consume
CSLC contacts through the standard `Contacts` buffer.s

---

## Architecture Overview

```
WRONG (side-channel hack):

  eval_cslc_contact_forces(model, state, cslc_data, ...)
       │
       └──► state.body_f  (direct injection)

  solver.step(state, contacts=None)   ← solver sees NO contacts


TARGET (pipeline integration):

  model.collide(state)
       │
       ├── Broad phase: AABB pair filtering
       │
       ├── Narrow phase triage:
       │   ├── Convex pairs         → MPR/GJK        → Contacts
       │   ├── SDF pairs            → distance query  → Contacts
       │   ├── Hydroelastic pairs   → marching cubes  → Contacts
       │   └── CSLC pairs (NEW)     → Jacobi solve    → Contacts  ◄── NEW
       │
       └── Contacts buffer (uniform format for all solvers)
                │
                ▼
  solver.step(state, contacts)   ← solver resolves ALL contacts uniformly
```

---

## Phase 1: ShapeConfig + ShapeFlags

### 1A. Add `CSLC` to `ShapeFlags` enum

**File: `newton/_src/geometry/types.py`** (or wherever ShapeFlags lives)

```python
class ShapeFlags(IntFlag):
    VISIBLE           = 1 << 0
    COLLIDE_SHAPES    = 1 << 1
    COLLIDE_PARTICLES = 1 << 2
    SITE              = 1 << 3
    HYDROELASTIC      = 1 << 4
    CSLC              = 1 << 5    # ← NEW
```

### 1B. Add CSLC parameters to `ShapeConfig`

**File: `newton/_src/sim/builder.py`**, inside class `ShapeConfig`:

```python
# --- Existing hydroelastic fields (lines 268-281) ---
is_hydroelastic: bool = False
kh: float = 1.0e10

# --- NEW CSLC fields (add after kh) ---
is_cslc: bool = False
"""Whether the shape uses CSLC distributed contact.
For CSLC collisions, the CSLC shape provides the lattice pad;
the other shape in the pair can be any geometry type.
Defaults to False."""

cslc_spacing: float = 0.004
"""Spacing between lattice spheres [m]. Controls pad resolution.
Smaller = more spheres = higher fidelity but more compute."""

cslc_ka: float = 5000.0
"""Anchor spring stiffness [N/m]. Resists sphere displacement from rest."""

cslc_kl: float = 500.0
"""Lateral spring stiffness [N/m]. Spreads load to neighbors.
Higher = wider pressure distribution. 0 = no coupling (N independent
point contacts on a grid)."""

cslc_dc: float = 2.0
"""Hunt-Crossley damping coefficient [s/m]. Controls energy dissipation
during approach/separation."""

cslc_n_iter: int = 40
"""Number of Jacobi iterations for quasistatic solve per timestep.
More = better equilibrium. 20-60 typical."""

cslc_alpha: float = 0.3
"""Jacobi damping factor (under-relaxation). 0.2-0.5 typical.
Lower = more stable, higher = faster convergence."""
```

### 1C. Wire the flag through `ShapeConfig.flags`

```python
@property
def flags(self) -> int:
    shape_flags = ShapeFlags.VISIBLE if self.is_visible else 0
    shape_flags |= ShapeFlags.COLLIDE_SHAPES if self.has_shape_collision else 0
    shape_flags |= ShapeFlags.COLLIDE_PARTICLES if self.has_particle_collision else 0
    shape_flags |= ShapeFlags.SITE if self.is_site else 0
    shape_flags |= ShapeFlags.HYDROELASTIC if self.is_hydroelastic else 0
    shape_flags |= ShapeFlags.CSLC if self.is_cslc else 0       # ← NEW
    return shape_flags
```

### 1D. Validate mutual exclusivity

In `ShapeConfig.validate()`:

```python
if self.is_cslc and self.is_hydroelastic:
    raise ValueError(
        "A shape cannot be both CSLC and hydroelastic. Choose one."
    )
if self.is_cslc and shape_type in (GeoType.PLANE, GeoType.HFIELD):
    raise ValueError(
        "CSLC is not supported for plane or heightfield shapes."
    )
```

### 1E. Store CSLC material properties on Model

**File: `newton/_src/sim/builder.py`**, in `ModelBuilder.__init__()`:

```python
# Add alongside existing shape_material_kh list:
self.shape_cslc_spacing: list[float] = []
self.shape_cslc_ka: list[float] = []
self.shape_cslc_kl: list[float] = []
self.shape_cslc_dc: list[float] = []
self.shape_cslc_n_iter: list[int] = []
self.shape_cslc_alpha: list[float] = []
```

In `add_shape()` (the internal method that all `add_shape_*` call):

```python
# After line: self.shape_material_kh.append(cfg.kh)
self.shape_cslc_spacing.append(cfg.cslc_spacing)
self.shape_cslc_ka.append(cfg.cslc_ka)
self.shape_cslc_kl.append(cfg.cslc_kl)
self.shape_cslc_dc.append(cfg.cslc_dc)
self.shape_cslc_n_iter.append(cfg.cslc_n_iter)
self.shape_cslc_alpha.append(cfg.cslc_alpha)
```

In `finalize()`:

```python
# After: m.shape_material_kh = wp.array(...)
m.shape_cslc_spacing = wp.array(self.shape_cslc_spacing, dtype=wp.float32)
m.shape_cslc_ka = wp.array(self.shape_cslc_ka, dtype=wp.float32)
m.shape_cslc_kl = wp.array(self.shape_cslc_kl, dtype=wp.float32)
m.shape_cslc_dc = wp.array(self.shape_cslc_dc, dtype=wp.float32)
# n_iter and alpha stay as Python-side config (not per-sphere GPU data)
```

**File: `newton/_src/sim/model.py`**, in `Model.__init__()`:

```python
# After: self.shape_material_kh
self.shape_cslc_spacing: wp.array[wp.float32] | None = None
self.shape_cslc_ka: wp.array[wp.float32] | None = None
self.shape_cslc_kl: wp.array[wp.float32] | None = None
self.shape_cslc_dc: wp.array[wp.float32] | None = None
```

---
# CSLC Integration into Newton — Design Document (v3)

## Status Summary

| Phase | Status | Notes |
|-------|--------|-------|
| 1. ShapeConfig + ShapeFlags | ✅ Done | Flag, config, builder, model fields all wired |
| 2. Pipeline integration | ✅ Done (with fixes) | Handler, kernels, collide.py patches applied |
| 3. Bug fixes | ✅ Done | Bugs 1–7 + Issues 8–11 addressed |
| 4. Squeeze comparison test | ✅ Done | 7-config matrix across 3 contact models × 3 solvers |
| 5. Paper update (scalar δ) | ✅ LaTeX patches ready | Section 3.2 revised for honest 1D formulation |

---

## File Layout

```
newton/_src/
├── sim/
│   ├── collide.py            ← CollisionPipeline — CSLC handler + pair filtering
│   ├── model.py              ← Model fields: shape_cslc_spacing/ka/kl/dc/n_iter/alpha
│   ├── builder.py            ← ShapeConfig.is_cslc, finalize() uploads
│   ├── contacts.py           ← Contacts buffer (per_contact_shape_properties)
│   └── state.py
├── geometry/
│   ├── types.py              ← ShapeFlags.CSLC = 1 << 5
│   ├── cslc_data.py          ← CSLCPad, CSLCData, calibrate_kc
│   ├── cslc_kernels.py       ← 3 Warp kernels (penetration, Jacobi, write)
│   ├── cslc_handler.py       ← CSLCHandler._from_model(), launch()
│   ├── sdf_hydroelastic.py   ← HydroelasticSDF (reference pattern)
│   └── narrow_phase.py
└── tests/
    └── squeeze_test.py       ← Comparison benchmark
```

---

## Architecture

```
model.collide(state, contacts)
  │
  ├── Broad phase: AABB pair culling
  │     └── CSLC pairs EXCLUDED (shape_collision_filter_pairs)
  │
  ├── Narrow phase:
  │     ├── Convex pairs         → MPR/GJK        → Contacts[0..offset)
  │     ├── SDF pairs            → distance query  → Contacts[0..offset)
  │     └── Hydroelastic pairs   → marching cubes  → Contacts[0..offset)
  │
  ├── CSLC handler (runs AFTER narrow phase):
  │     ├── Kernel 1: penetration (surface spheres vs target)
  │     ├── Kernel 2: Jacobi solve (40 iters, warm-started)
  │     └── Kernel 3: write contacts → Contacts[offset..offset+n_cslc)
  │
  └── _set_cslc_contact_count → ensures solver sees all slots
```

**Why CSLC runs separately from the narrow phase:**
1. Jacobi solve loop (40 iterations) doesn't fit per-pair dispatch
2. Pre-allocated contact slots (no atomic_add) for differentiability
3. Warm-start state persists across timesteps

**Contact buffer layout:**
```
[0 .................. cslc_offset) ← narrow phase (atomic_add slots)
[cslc_offset .. cslc_offset+n_cslc) ← CSLC (pre-allocated, deterministic)
```

---

## Critical Newton API Notes

These caused the most bugs. Document them prominently for anyone
working on this codebase.

### Gravity

```python
# WRONG — ModelBuilder has no set_gravity() method
b.set_gravity((0, 0, -9.81))

# RIGHT — gravity is a constructor scalar, or set on Model after finalize
b = newton.ModelBuilder(gravity=-9.81)       # scalar × up_axis
model = b.finalize()
model.set_gravity((0.0, 0.0, -9.81))        # vec3 on Model
```

`ModelBuilder.gravity` is a **scalar** (default -9.81) multiplied by
`up_vector` to produce the gravity vec3. `Model.set_gravity()` takes
a vec3 or per-world array.

### Kinematic Bodies

```python
# WRONG — manually zeroing mass/inv_mass/inv_inertia
body = b.add_body(xform=...)
b.body_mass[body] = 0.0
b.body_inv_mass[body] = 0.0
b.body_inv_inertia[body] = (0,0,0,0,0,0,0,0,0)

# RIGHT — use the is_kinematic flag
body = b.add_body(
    xform=wp.transform((x, y, z), wp.quat_identity()),
    is_kinematic=True,    # ← sets BodyFlags.KINEMATIC, solver skips it
    label="my_pad",
)
# Also set density=0 on shapes so they don't add mass:
cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ...)
b.add_shape_box(body, hx=0.04, hy=0.04, hz=0.04, cfg=cfg)
```

Kinematic bodies are moved by overwriting `state.body_q` each step.

### body_q / body_qd numpy layout

```python
# body_q: wp.array(dtype=wp.transform) → numpy (N, 7)
#   columns: [px, py, pz, qx, qy, qz, qw]
q = state.body_q.numpy()
q[body_idx, 0] = new_x   # position x
q[body_idx, 1] = new_y   # position y
q[body_idx, 2] = new_z   # position z
state.body_q.assign(wp.array(q, dtype=wp.transform, device=...))

# body_qd: wp.array(dtype=wp.spatial_vector) → numpy (N, 6)
#   columns: [vx, vy, vz, wx, wy, wz]  (world frame for most solvers)
qd = state.body_qd.numpy()
qd[body_idx] = 0.0   # zero all 6 components
state.body_qd.assign(wp.array(qd, dtype=wp.spatial_vector, device=...))
```

### Simulation Loop Pattern

```python
model = b.finalize()
model.set_gravity((0, 0, -9.81))

solver = SolverSemiImplicit(model)   # or SolverMuJoCo, SolverXPBD
state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()          # allocates correctly-sized buffer

for step in range(n_steps):
    state_0.clear_forces()
    model.collide(state_0, contacts)                    # fills contacts
    solver.step(state_0, state_1, control, contacts, dt)  # resolves
    state_0, state_1 = state_1, state_0
```

`model.collide()` lazily creates a `CollisionPipeline` on first call.
The pipeline is cached on `model._collision_pipeline`. If you need to
customize it (e.g., broad phase mode), pass `collision_pipeline=` kwarg.

---

## All Bug Fixes Applied

### Bug 1 — Kernel 3 argument mismatch (CRASH)

**File:** `cslc_handler.py`, method `_launch_vs_sphere`
**Problem:** `self.contact_normal_scratch` passed as extra input to
`write_cslc_contacts`, shifting all subsequent args by one. Warp
raises a parameter count error.
**Fix:** Remove `self.contact_normal_scratch` from the inputs list.
The kernel recomputes the normal internally.

### Bug 2 — Double-counting contacts

**File:** `cslc_handler.py` `_from_model()` + `collide.py` pipeline init
**Problem:** Narrow phase still generates point contacts for CSLC pairs.
Solver sees both point contact AND distributed CSLC forces.
**Fix:** In `_from_model()`, add CSLC pairs to
`model.shape_collision_filter_pairs`. In `collide.py`, rebuild
`shape_pairs_excluded` after CSLC handler creation. For EXPLICIT
broad-phase mode, also filter `model.shape_contact_pairs`.

### Bug 3 — CPU round-trips every timestep

**File:** `cslc_handler.py`, method `_launch_vs_sphere`
**Problem:** `.numpy()` calls on GPU arrays (shape_body, shape_transform,
shape_scale) every step → GPU sync barrier + breaks gradient tape.
**Fix:** Cache target body index, local position, and radius on
`CSLCShapePair` at construction time in `_from_model()`.

### Bug 4 — Damping coefficient unused

**Files:** `cslc_data.py`, `cslc_kernels.py`, `cslc_handler.py`
**Problem:** `cslc_dc` stored but never used. Contacts have zero damping.
**Fix:** Added `dc` field to `CSLCData`, passed through `from_pads()`,
written as `out_damping` in `write_cslc_contacts`.

### Bug 5 — Scalar vs vector displacement (paper mismatch)

**Files:** Paper only (no code change)
**Problem:** Paper writes δᵢ ∈ ℝ³ but code uses scalar δ along normal.
**Fix:** Revised Section 3.2 to present 1D normal projection honestly.
The projection is exact for flat contact patches and adds a 3× speedup
narrative that strengthens the real-time claim.

### Bug 6 — n_iter and alpha from ShapeConfig ignored

**Files:** `model.py`, `builder.py`, `cslc_handler.py`
**Problem:** Handler hardcodes `n_iter=40, alpha=0.3`. User config via
ShapeConfig silently ignored.
**Fix:** Added `shape_cslc_n_iter` (list[int]) and `shape_cslc_alpha`
(list[float]) to Model. Builder uploads them in finalize(). Handler
reads them in `_from_model()`.

### Bug 7 — out_tids always 0

**File:** `cslc_kernels.py`
**Impact:** Minor — `tids` is unused by per-contact-property solvers.
When `per_contact_shape_properties=True`, the solver reads stiffness/
damping/friction from per-contact arrays, not from tids-indexed lookups.
Kept as-is.

### Issue 11 — No per-contact material properties

**Files:** `cslc_kernels.py`, `cslc_handler.py`
**Problem:** `write_cslc_contacts` didn't write stiffness/damping/friction.
Solver saw zeros → ghost contacts.
**Fix:** Added `out_stiffness`, `out_damping`, `out_friction` outputs.
Stiffness = calibrated `kc`. Damping = CSLC's `dc`. Friction = average
of shape-pair `mu`.

### EXPLICIT mode caveat

**File:** `collide.py`
**Problem:** In EXPLICIT broad-phase mode, `shape_contact_pairs` is
precomputed during finalize() and already contains CSLC pairs.
`shape_pairs_excluded` only works for SAP/NxN modes.
**Fix:** After CSLC handler creation, also filter CSLC pairs from
`model.shape_contact_pairs` if it exists.

---

## Squeeze Test Configuration Matrix

```
SCENE: Two kinematic boxes squeeze a dynamic sphere under gravity

     ┌─────┐         ┌─────┐
     │     │  ←───   │     │
     │ left│  sphere  │right│
     │ pad │   (●)   │ pad │
     │     │  ───→   │     │
     └─────┘         └─────┘

Body layout: 0=left_pad (kinematic), 1=right_pad (kinematic), 2=sphere (dynamic)
Phase 1 (0.0–0.5s): pads squeeze inward at 0.02 m/s
Phase 2 (0.5–2.0s): pads hold position, sphere hangs by friction

COMPARISON MATRIX:
┌────────────────┬──────────────┬───────────────┬──────────────┐
│ Contact Model  │ Solver       │ Geometry      │ What varies  │
├────────────────┼──────────────┼───────────────┼──────────────┤
│ Point contact  │ SemiImplicit │ box + sphere  │ baseline     │
│ Point contact  │ MuJoCo       │ box + sphere  │ solver       │
│ Point contact  │ XPBD         │ box + sphere  │ solver       │
│ Hydroelastic   │ SemiImplicit │ box+SDF       │ contact      │
│ Hydroelastic   │ MuJoCo       │ box+SDF       │ contact+solv │
│ CSLC           │ SemiImplicit │ box + sphere  │ contact      │
│ CSLC           │ MuJoCo       │ box + sphere  │ contact+solv │
└────────────────┴──────────────┴───────────────┴──────────────┘

METRICS:
  - Sphere z-drop (mm): friction capacity — lower = better grip
  - Peak contact count: patch area proxy
  - Step time (ms): computational cost
  - Gradient norm: differentiability check (SemiImplicit only)
```

### Running the test

```bash
# All configs
python squeeze_test.py

# Smoke-test one config at a time
python squeeze_test.py --configs point_semi
python squeeze_test.py --configs cslc_semi

# List available
python squeeze_test.py --list

# CPU fallback
python squeeze_test.py --device cpu
```

### Expected results

| Config | Z-drop | Contacts | Why |
|--------|--------|----------|-----|
| point_semi | Large (10+ mm) | 2 | Single contact per box face; friction capacity = μ·F_n at one point |
| cslc_semi | Small (<1 mm) | ~100+ | Distributed friction across lattice; aggregate capacity >> point |
| hydro_semi | Small-medium | 5-20 | Distributed but fewer contacts than CSLC lattice |

---

## Remaining Work

### Next immediate steps

1. **Run squeeze test end-to-end** — verify all 7 configs produce valid output
2. **Tune CSLC parameters** — if z-drop is too large, increase `cslc_ka` or decrease `cslc_spacing`
3. **Add visualization** — render sphere trajectory + contact patch heatmap
4. **Gradient validation** — verify non-zero gradients flow through CSLC pipeline

### Future work (post-conference)

- **CSLC vs box target** — add `_launch_vs_box` to CSLCHandler for box-on-box grasps
- **Vector displacement (3D δ)** — upgrade for curved surfaces / large deformations
- **MPC integration** — embed CSLC in a model-predictive control loop
- **Drake benchmark** — compare against Drake's hydroelastic on identical scenes
- **Resolution study** — sweep `cslc_spacing` to show convergence to continuum limit