# CSLC × Newton integration — running notes

Purpose: keep one document that tracks what we found, what we changed, and why,
so anyone (including future-you) can reconstruct the reasoning without re-reading
the whole chat.

---

## 1. Problem statement

Goal: integrate Compliant Sphere Lattice Contact (CSLC) into the Newton
simulator as a fast, differentiable, sphere-native distributed contact model
suitable for an RSS-quality paper.

Reference test: `lift_test.py`. Two driven pads approach, squeeze, and lift a
30 mm sphere sitting on the ground plane. Friction must hold the sphere against
gravity through the LIFT phase.

Observed failure (before any fix):

- As LIFT begins, the sphere jumps upward, then slips out of the grip.
- `CSLC_SUMMARY` shows only `cslc=1 tids=[...]` per pad — the lattice is
  producing **point contact**, not a distributed patch.

---

## 2. Root causes (confirmed from logs + kernel code)

### Cause A — radial gate in both kernels

`CSLC_RADIAL_CUTOFF = 2 mm`. Lattice spacing is `5–10 mm`, so the first
neighbor of the closest lattice sphere sits outside the gate. Consequence:
at most **one sphere per pad** ever qualifies as "in contact". Source lines:

- `cslc_kernels.py :: compute_cslc_penetration_sphere` — radial test around line 80.
- `cslc_kernels.py :: write_cslc_contacts` — radial test around line 307.

### Cause B — writer uses projected penetration but the lattice solves 3D overlap

After the Stage 1 kernel-1 patch, the internal solver uses `phi = pen_3d =
(r_lat + R) − dist`. But the writer still hands the rigid-body solver
`margin0 = effective_r`, `margin1 = R`, `normal = face_normal`. The rigid-body
solver therefore evaluates

```
solver_pen = effective_r + R − d_proj
```

which over-estimates penetration away from the contact axis by a factor of
`dist / d_proj`. Internal phi and solver-phi disagree.

### Cause C — Jacobi has no pad filter; warm-start is destroyed

`_launch_vs_sphere` launches `jacobi_step` with `dim=n_spheres` and no pair
filter. During pair (A,tgt) the solver iterates over pad B's spheres too.
Kernel 1 has zeroed pad B's `raw_penetration`, so their Laplacian coupling
decays their delta toward 0. Pair (B,tgt) then starts from a destroyed warm
start.

### Cause D — `kc` calibrated for the wrong active-set size

`cslc_handler.py :: _from_model` calls `calibrate_kc(..., contact_fraction=0.1)`.
With 138 surface spheres per pad and only 1 active, the actual fraction is
`~0.007`, not `0.1`. Per-sphere `kc` is therefore tuned as if the load is
shared ~14 ways but is actually carried alone → wrong force magnitude.

---

## 3. Fix strategy — staged

### Stage 1 — correct the active set, keep ReLU activation

1. Kernel 1: drop radial gate, emit `phi = pen_3d`.
2. Kernel 3: drop radial gate; cull only on `d_proj ≤ 0` (wrong side) and
   `pen_3d ≤ 0` (no real overlap); scale `out_stiffness` by
   `pen_3d / solver_pen` so the rigid-body solver's force equals
   `k_c · pen_3d`.
3. Kernel 2: add `active_cslc_shape_idx` argument; non-matching tids pass
   `delta_dst[tid] = delta_src[tid]` (preserve warm start) and return.
4. Handler: pass `pair.cslc_shape` to the Jacobi launch. Re-measure active
   count and set `contact_fraction` accordingly (empirical, not default).

After Stage 1 the active set should jump from `~1` to `~20–30` per pad in the
lift_test geometry, depending on spacing.

### Stage 2 — smooth activation for gradient quality

Replace the remaining hard conditions with smooth approximations. Only
pursue this after Stage 1 is confirmed working forward-wise.

1. Helper `@wp.func` definitions: `softplus`, `sigmoid`, `compact_weight`
   (quartic compact-support kernel).
2. Kernel 1 also writes a per-sphere `contact_weight ∈ [0,1]` combining
   `sigmoid(d_proj)` and `compact_weight(pen_3d / support)`.
3. Kernel 2 uses weighted softplus contact law:
   `f_contact = kc · w · softplus(phi − delta)` and `k_diag += kc · w`.
4. Kernel 3 scales stiffness/damping by `w`, writes only when
   `w · softplus(pen_3d) > ε`.
5. Handler allocates `contact_weight` scratch and threads it through.

This eliminates ReLU-induced gradient spikes at the active-set boundary,
which is what you need for MPC and learning.

---

## 4. Choice: face normal vs sphere-sphere normal

Keeping **face normal** (the paper's §3 formulation). Force magnitude is
corrected via Option A stiffness scaling. The cleaner-physics alternative
(`normal = (target − q_lattice) / dist`) is noted as future work for
non-flat pad geometries. See §7 of the chat for the comparison equations.

---

## 5. Debug checkpoints

The `CSLC_SUMMARY` handler print is extended to include:

- `delta[mm] max/mean`
- `pen[mm] max/mean`
- `F_sum ≈ Σ kc · phi` — aggregate patch force estimate

Expected values after each stage (lift_test, late SQUEEZE phase):

| Stage | cslc per pad | F_sum per pad | delta max |
|-------|-------------:|--------------:|----------:|
| Baseline (as-is) | 1 | ≈ 6 N | ≈ 1 mm |
| Stage 1 applied  | 20–30 | 5–15 N * | patch-shaped |
| Stage 2 applied  | same shape, smoother transitions | same | same |

\* F_sum depends on the new `contact_fraction` you pick. Target:
`F_sum ≈ ke_bulk · pen_rest` where `pen_rest` is the grip penetration.
For ke_bulk = 5e3, pen = 12 mm → 60 N; for a shallower grip (1–2 mm
effective) the number is a few N.

---






  # ── Stage 2: smooth contact helpers ──

# Tunables.  These are starting values; sweep later.
CSLC_SUPPORT_RADIUS = 3.0e-3   # radial compact-support half-width [m]
CSLC_SMOOTH_BETA    = 4.0e3    # softplus sharpness [1/m]; larger = sharper
CSLC_SIDE_BETA      = 2.0e3    # d_proj sigmoid sharpness [1/m]
CSLC_WRITE_EPS      = 1.0e-6   # skip-write threshold on w · softplus(pen)


@wp.func
def _softplus(x: wp.float32, beta: wp.float32):
    """Smooth ReLU: log(1 + exp(beta·x)) / beta.  Equals x for large beta·x."""
    bx = beta * x
    if bx > 30.0:
        return x
    if bx < -30.0:
        return wp.float32(0.0)
    return wp.log(1.0 + wp.exp(bx)) / beta


@wp.func
def _sigmoid(x: wp.float32, beta: wp.float32):
    """Smooth step: 1/(1 + exp(−beta·x)).  1 on the front side, 0 behind."""
    return wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-beta * x))


@wp.func
def _compact_weight(radial: wp.float32, support: wp.float32):
    """Quartic compact-support kernel (C² smooth, strict zero at support).

    w(ρ) = (1−ρ/R)^4 · (1 + 4·ρ/R)   for ρ ≤ R
         = 0                           for ρ > R
    """
    if radial >= support:
        return wp.float32(0.0)
    rho = radial / support
    t   = wp.float32(1.0) - rho
    return t * t * t * t * (wp.float32(1.0) + wp.float32(4.0) * rho)


    @wp.kernel
def compute_cslc_penetration_sphere(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    target_shape_idx: int,
    target_local_pos: wp.vec3,
    target_radius: float,
    raw_penetration: wp.array(dtype=wp.float32),
    contact_weight:  wp.array(dtype=wp.float32),   # NEW OUTPUT
    contact_normal_out: wp.array(dtype=wp.vec3),
):
    """Stage 2: true 3-D overlap + smooth geometric weight.

    Outputs per sphere:
        raw_penetration[tid] = pen_3d = (r_lat + R) − dist   (may be negative)
        contact_weight[tid]  = σ(d_proj) · compact(radial)   ∈ [0, 1]

    contact_weight is 1 deep in the contact (front side, near axis) and
    smoothly decays to 0 when the sphere is far off-axis or on the wrong
    side of the face.  Replaces the binary radial gate.
    """
    tid = wp.tid()
    raw_penetration[tid] = 0.0
    contact_weight[tid]  = 0.0
    contact_normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)

    if sphere_shape[tid] != active_cslc_shape_idx:
        return
    if is_surface[tid] != 1:
        return

    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]

    p_local = sphere_pos_local[tid]
    r_lat   = sphere_radii[tid]
    out_n   = sphere_outward_normal[tid]

    q_body  = wp.transform_point(X_ws, p_local)
    q_world = wp.transform_point(X_wb, q_body)

    X_tb    = body_q[target_body_idx]
    t_world = wp.transform_point(X_tb, target_local_pos)

    diff = t_world - q_world
    dist = wp.length(diff)

    n_body  = wp.transform_vector(X_ws, out_n)
    n_world = wp.transform_vector(X_wb, n_body)
    d_proj  = wp.dot(diff, n_world)

    pen_3d = (r_lat + target_radius) - dist

    radial_sq = dist * dist - d_proj * d_proj
    if radial_sq < 0.0:
        radial_sq = 0.0
    radial = wp.sqrt(radial_sq)

    w_side = _sigmoid(d_proj, CSLC_SIDE_BETA)
    w_rad  = _compact_weight(radial, CSLC_SUPPORT_RADIUS)
    w      = w_side * w_rad

    raw_penetration[tid] = pen_3d   # may be negative; Jacobi uses softplus
    contact_weight[tid]  = w
    contact_normal_out[tid] = n_world



    @wp.kernel
def jacobi_step(
    delta_src: wp.array(dtype=wp.float32),
    delta_dst: wp.array(dtype=wp.float32),
    raw_penetration: wp.array(dtype=wp.float32),
    contact_weight:  wp.array(dtype=wp.float32),   # NEW
    is_surface: wp.array(dtype=wp.int32),
    sphere_shape: wp.array(dtype=wp.int32),
    active_cslc_shape_idx: int,
    neighbor_start: wp.array(dtype=wp.int32),
    neighbor_count: wp.array(dtype=wp.int32),
    neighbor_list: wp.array(dtype=wp.int32),
    ka: float,
    kl: float,
    kc: float,
    alpha: float,
):
    """Stage 2: weighted softplus activation.

    Replaces the hard 'if effective_pen > 0' with
        f_contact = kc · w · softplus(effective_pen, β)
    which is C^∞ in δ, giving clean gradients.
    """
    tid = wp.tid()

    if sphere_shape[tid] != active_cslc_shape_idx:
        delta_dst[tid] = delta_src[tid]
        return

    delta_old   = delta_src[tid]
    n_neighbors = neighbor_count[tid]

    neighbor_sum = float(0.0)
    start = neighbor_start[tid]
    for n in range(n_neighbors):
        j = neighbor_list[start + n]
        neighbor_sum = neighbor_sum + delta_src[j]

    f_contact = float(0.0)
    k_diag    = ka + kl * float(n_neighbors)

    if is_surface[tid] == 1:
        phi = raw_penetration[tid]
        w   = contact_weight[tid]
        effective_pen = phi - delta_old
        pen_soft = _softplus(effective_pen, CSLC_SMOOTH_BETA)
        f_contact = kc * w * pen_soft
        k_diag    = k_diag + kc * w   # same sign+shape as Stage 1 for PD-ness

    delta_jacobi = (f_contact + kl * neighbor_sum) / k_diag
    delta_new    = (1.0 - alpha) * delta_old + alpha * delta_jacobi
    if delta_new < 0.0:
        delta_new = 0.0
    delta_dst[tid] = delta_new



    @wp.kernel
def write_cslc_contacts(
    sphere_pos_local: wp.array(dtype=wp.vec3),
    sphere_radii: wp.array(dtype=wp.float32),
    sphere_delta: wp.array(dtype=wp.float32),
    sphere_shape: wp.array(dtype=wp.int32),
    is_surface: wp.array(dtype=wp.int32),
    sphere_outward_normal: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    active_cslc_shape_idx: int,
    target_body_idx: int,
    target_shape_idx: int,
    target_local_pos: wp.vec3,
    target_radius: float,
    contact_offset: int,
    surface_slot_map: wp.array(dtype=wp.int32),
    raw_penetration: wp.array(dtype=wp.float32),
    contact_weight:  wp.array(dtype=wp.float32),   # NEW
    out_shape0: wp.array(dtype=wp.int32),
    out_shape1: wp.array(dtype=wp.int32),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_margin0: wp.array(dtype=wp.float32),
    out_margin1: wp.array(dtype=wp.float32),
    out_tids: wp.array(dtype=wp.int32),
    shape_material_mu: wp.array(dtype=wp.float32),
    cslc_kc: float,
    cslc_dc: float,
    out_stiffness: wp.array(dtype=wp.float32),
    out_damping: wp.array(dtype=wp.float32),
    out_friction: wp.array(dtype=wp.float32),
    debug_reason: wp.array(dtype=wp.int32),
):
    """Stage 2: write with weighted softplus force, no hard gates."""
    tid = wp.tid()

    slot = surface_slot_map[tid]
    if slot < 0:
        return
    buf_idx = contact_offset + slot

    if sphere_shape[tid] != active_cslc_shape_idx:
        out_shape0[buf_idx] = -1
        debug_reason[slot]  = 4
        return

    s_idx = sphere_shape[tid]
    b_idx = shape_body[s_idx]
    X_ws  = shape_transform[s_idx]
    X_wb  = body_q[b_idx]

    p_local   = sphere_pos_local[tid]
    out_n     = sphere_outward_normal[tid]
    delta_val = sphere_delta[tid]
    r_lat     = sphere_radii[tid]

    q_body  = wp.transform_point(X_ws, p_local)
    q_world = wp.transform_point(X_wb, q_body)

    effective_r = r_lat - delta_val
    if effective_r < 0.0:
        effective_r = 0.0

    X_tb    = body_q[target_body_idx]
    t_world = wp.transform_point(X_tb, target_local_pos)

    diff = t_world - q_world
    dist = wp.length(diff)

    n_body    = wp.transform_vector(X_ws, out_n)
    normal_ab = wp.transform_vector(X_wb, n_body)
    d_proj    = wp.dot(diff, normal_ab)

    # True 3-D overlap and smooth activation.
    pen_3d     = (effective_r + target_radius) - dist
    w          = contact_weight[tid]
    pen_soft   = _softplus(pen_3d, CSLC_SMOOTH_BETA)

    write_strength = w * pen_soft
    if write_strength < CSLC_WRITE_EPS:
        out_shape0[buf_idx] = -1
        debug_reason[slot]  = 1
        return

    # Solver-view projected penetration (needed for stiffness scaling).
    # Guard for edge cases where d_proj is very small but positive.
    solver_pen_raw = (effective_r + target_radius) - d_proj
    solver_pen     = _softplus(solver_pen_raw, CSLC_SMOOTH_BETA)
    # Scale so solver force = kc · w · pen_soft.
    pen_scale = pen_soft / wp.max(solver_pen, 1.0e-8)

    X_wb_inv = wp.transform_inverse(X_wb)
    X_tb_inv = wp.transform_inverse(X_tb)
    p0_body      = wp.transform_point(X_wb_inv, q_world)
    p1_body      = wp.transform_point(X_tb_inv, t_world)
    offset0_body = wp.transform_vector(X_wb_inv,  effective_r   * normal_ab)
    offset1_body = wp.transform_vector(X_tb_inv, -target_radius * normal_ab)

    out_shape0[buf_idx]    = s_idx
    out_shape1[buf_idx]    = target_shape_idx
    out_point0[buf_idx]    = p0_body
    out_point1[buf_idx]    = p1_body
    out_offset0[buf_idx]   = offset0_body
    out_offset1[buf_idx]   = offset1_body
    out_normal[buf_idx]    = normal_ab
    out_margin0[buf_idx]   = effective_r
    out_margin1[buf_idx]   = target_radius
    out_tids[buf_idx]      = 0

    out_stiffness[buf_idx] = cslc_kc * w * pen_scale
    out_damping[buf_idx]   = cslc_dc * w * pen_scale
    out_friction[buf_idx]  = shape_material_mu[s_idx]

    debug_reason[slot] = 0


in handler init

    n = cslc_data.n_spheres
        self.raw_penetration     = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.contact_weight      = wp.zeros(n, dtype=wp.float32, device=self.device)   # NEW
        self.contact_normal_scratch = wp.zeros(n, dtype=wp.vec3,   device=self.device)
        self._jacobi_a = wp.zeros(n, dtype=wp.float32, device=self.device)
        self._jacobi_b = wp.zeros(n, dtype=wp.float32, device=self.device)



  Thread self.contact_weight into each of the three kernel launches in _launch_vs_sphere:

Kernel 1 outputs: [self.raw_penetration, self.contact_weight, self.contact_normal_scratch]
Kernel 2 inputs (after raw_penetration): add self.contact_weight
Kernel 3 inputs (after self.raw_penetration): add self.contact_weight

Stage 2 — what you should see

Forward behavior: patch size and F_sum roughly the same as Stage 1 (maybe a hair smaller because softplus rounds the edges).
The handoff "cliff" in the lift_test log — pads transitioning from tid=192 to tid=191 — should now be a gentle transfer, not a cslc=0 dropout between them. Watch the δ mean over the lift phase: with Stage 1 it still has small jumps when a new sphere becomes dominant; with Stage 2 it should be visually smooth.
The real payoff is gradient quality. Save that benchmark for its own session once forward is solid.

Stage 2 — what to toggle when debugging

CSLC_SMOOTH_BETA: larger (e.g. 10 000) → closer to hard ReLU, less smoothing; smaller (e.g. 500) → noticeably soft forces at small penetrations. If Stage 2 forces feel too weak at small grip pen, reduce BETA.
CSLC_SUPPORT_RADIUS: think of this as the "blur radius" of the patch. At 3 mm with 10 mm spacing, only nearest-neighbor spheres get appreciable weight. Widen to 8–10 mm to engage a full 5×5 patch.
CSLC_SIDE_BETA: almost never needs tuning. If you see lattice spheres on the back face of the pad getting nonzero weight, bump this up.