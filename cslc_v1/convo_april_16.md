CSLC Debug Summary — April 16, 2026
Goal

Today’s goal was to debug the current CSLC implementation for a differentiable, sphere-based contact model and understand why the grasped sphere was jumping or losing contact during lift. The investigation focused on the CSLC handler, the three-kernel pipeline, per-pair contact allocation, and the contact geometry used inside the solver and writer.
What we learned

The main instability was not a simple variable-naming bug in the Jacobi step. The active-contact branch in jacobi_step() was internally consistent with the current linearized contact solve, but the overall pipeline was inconsistent because contact selection, penetration definition, and final contact writing were not using the same smooth geometric model.

The original implementation mixed two different ideas:

    a flat-patch-style projected penetration based on d_proj, and

    a true sphere-sphere overlap check based on dist.

That mismatch caused the radial cutoff to behave badly:

    Large cutoff admitted off-axis spheres, which made the projected penetration too strong and increased the jump.

    Small cutoff reduced the jump, but created handoff gaps where no sphere qualified, so contact disappeared.

The hard binary rule radial <= CSLC_RADIAL_CUTOFF is therefore not a good main design choice for a differentiable contact model. It creates either exaggerated forces or contact dropout depending on the threshold.
Bugs fixed today
1. Handler initialization order

A crash occurred because self.debug_reason was allocated before self.device existed in CSLCHandler.__init__(). The fix was to assign self.device = device or wp.get_device() before any wp.zeros(...) allocations.
2. Undefined kernel variable

write_cslc_contacts() referenced pen_3d before it had been defined, which caused Warp code generation to fail. The fix was to compute geometry quantities in a consistent order before debug prints and before any rejection logic.
3. Per-pair slot overwrite

Multiple CSLC-vs-sphere pairs were initially writing into the same slot range, so later launches could overwrite earlier pair results. The fix was to allocate a separate block of contact slots per supported CSLC pair and use a per-pair contact_offset during launch.
4. Cross-pad mixing

Each pair launch was initially processing the combined lattice rather than only the spheres belonging to the active CSLC shape. The fix was to filter kernels by pair.cslc_shape, so pair=(2,5) and pair=(4,5) no longer reported the same active lattice tids.
5. Better summary debugging

Per-pair summaries were improved so the logs now identify which CSLC pair produced which active tids. This made it possible to distinguish overwrite bugs from real contact dropout.
Current diagnosis

At the end of today’s debugging, the pair-isolation problem looked fixed, but the physical issue remained: each pad could drop from one surviving lattice contact to zero and then later recover on a neighboring tid. That behavior is consistent with a discrete lattice handoff problem caused by the hard radial gate in write_cslc_contacts() and the projected-penetration formulation used in the current kernels.

In other words, the remaining failure is now a real modeling issue, not mainly a handler bookkeeping issue. The solver path is still too discontinuous for stable differentiable contact.
Recommended direction

The recommended direction is:

    Keep the sphere lattice.

    Keep the scalar normal displacement state per surface sphere.

    Use true 3D sphere-sphere overlap as the base contact geometry.

    Replace hard radial cutoff with a smooth compact support kernel.

    Replace hard ReLU-style contact activation with a smooth approximation such as softplus in the solver path.

    Evaluate gradient quality, not only forward simulation stability.

This moves CSLC toward a sphere-native smooth distributed contact model, which is much better aligned with differentiable robotics simulation than the current binary patch-membership approach.
Concrete implementation plan
1. Update kernel 1: smooth contact geometry

File: newton/_src/geometry/cslc_kernels.py
Current issue

compute_cslc_penetration_sphere() currently computes:

    pen_3d = (r_lat + target_radius) - dist

    d_proj

    radial

but then discards pen_3d and sets phi from projected distance only when the hard radial cutoff passes.
Change

Use:

    phi = pen_3d as the base penetration,

    a new smooth contact_weight based on radial and d_proj.

Suggested implementation

Add helper functions:

python
@wp.func
def softplus_beta(x: float, beta: float):
    bx = beta * x
    if bx > 30.0:
        return x
    return wp.log(1.0 + wp.exp(bx)) / beta

@wp.func
def sigmoid_beta(x: float, beta: float):
    return 1.0 / (1.0 + wp.exp(-beta * x))

@wp.func
def compact_weight(radial: float, support_radius: float):
    rho = radial / support_radius
    if rho >= 1.0:
        return 0.0
    t = 1.0 - rho
    return t * t * t * t * (1.0 + 4.0 * rho)

Then in compute_cslc_penetration_sphere() compute:

python
pen_3d = (r_lat + target_radius) - dist
front_w = sigmoid_beta(d_proj, SIDE_BETA)
radial_w = compact_weight(radial, SUPPORT_RADIUS)

phi = pen_3d
w = front_w * radial_w

Write both:

    raw_penetration[tid] = phi

    contact_weight[tid] = w

2. Update kernel 2: smooth solver activation

File: newton/_src/geometry/cslc_kernels.py
Current issue

jacobi_step() uses a hard branch:

python
if effective_pen > 0.0:
    f_contact = kc * phi
    k_diag = k_diag + kc

This creates a discontinuous active set.
Change

Pass contact_weight into jacobi_step() and replace the branch with a smooth contact law.
Suggested implementation

python
phi = raw_penetration[tid]
w = contact_weight[tid]

effective_pen = phi - delta_old
phi_soft = softplus_beta(effective_pen, SMOOTH_BETA)

f_contact = kc * w * phi_soft
k_diag = k_diag + kc * w

This keeps the scalar displacement state and lattice solve structure, but removes the hard on/off transition.
3. Update kernel 3: remove hard radial cull

File: newton/_src/geometry/cslc_kernels.py
Current issue

write_cslc_contacts() still:

    uses hard radial rejection,

    uses projected penetration in the writer path,

    and only writes contact when the binary admissibility test passes.

Change

Use:

    pen_3d = (effective_r + target_radius) - dist

    normal_ab = diff / max(dist, eps)

    write_strength = contact_weight[tid] * softplus_beta(pen_3d, SMOOTH_BETA)

Only skip writing when write_strength is numerically tiny.
Suggested implementation

python
pen_3d = (effective_r + target_radius) - dist
normal_ab = diff / wp.max(dist, 1.0e-8)

w = contact_weight[tid]
pen_soft = softplus_beta(pen_3d, SMOOTH_BETA)
write_strength = w * pen_soft

if write_strength < 1.0e-6:
    out_shape0[buf_idx] = -1
    return

out_normal[buf_idx] = normal_ab
out_margin0[buf_idx] = effective_r
out_margin1[buf_idx] = target_radius
out_stiffness[buf_idx] = cslc_kc * w
out_damping[buf_idx] = cslc_dc * w

This removes the radial handoff discontinuity from the writer.
4. Update handler: add smooth weight scratch array

File: newton/_src/geometry/cslc_handler.py
Current issue

The handler currently allocates:

    raw_penetration

    contact_normal_scratch

    Jacobi scratch buffers

but no smooth weight field.
Change

Add:

python
self.contact_weight = wp.zeros(n, dtype=wp.float32, device=self.device)

Then update _launch_vs_sphere() so:

    kernel 1 outputs raw_penetration, contact_weight, and normal scratch,

    kernel 2 receives contact_weight,

    kernel 3 receives contact_weight.

5. Clean up handler construction

File: newton/_src/geometry/cslc_handler.py

There are still structural cleanup tasks to keep:

    only one contact_count property,

    n_pair_blocks stored explicitly on the handler,

    per-pair slot offsets preserved,

    per-pair shape filtering preserved.

These were part of today’s debugging and should remain in the cleaned-up version.
Suggested parameter starting point

Use these only as starting values:

python
SUPPORT_RADIUS = 2.5e-3
SMOOTH_BETA = 4000.0
SIDE_BETA = 2000.0
WRITE_THRESHOLD = 1.0e-6

These should be tuned empirically after the smooth model is in place. The important point is that support width should now control smooth patch weighting, not binary contact membership.
Benchmark plan

Forward stability is not enough. The new model should be evaluated on both trajectory quality and gradient quality. This is especially important because the end goal is differentiable robotics contact, not only visually plausible simulation.
Forward metrics

Track:

    object lift height,

    slip distance,

    final object pose,

    net grasp wrench,

    number of active CSLC contacts written.

Gradient metrics

Measure sensitivity of:

    lift height,

    object pose,

    grasp wrench

with respect to:

    finger closing displacement,

    ka,

    kl,

    kc,

    object pose perturbation.

Use finite differences first:

    run at p−ϵp−ϵ,

    run at pp,

    run at p+ϵp+ϵ,

    compare centered differences across time.

Success looks like:

    no large spikes when contact shifts from one lattice sphere to the next,

    no long zero-contact gaps during lift,

    smoother derivative curves with respect to finger pose and stiffness.

Longer-term note

Even after these changes, the final exported contact set is still a variable set of rigid contacts in Newton’s contact buffer. That means the internal CSLC solve can become much smoother than the current version, but the overall pipeline is still not perfectly end-to-end smooth as long as contact emission itself is discrete.

A stronger long-term direction is to accumulate a smooth net wrench directly from the lattice onto the rigid body instead of representing the lattice through a changing discrete contact manifold. That would be the most natural next step once the smoothed CSLC prototype is working.
Priority order for next session

    Clean up cslc_handler.py so per-pair offsets and contact_count are correct and consistent.

    Add contact_weight scratch storage and thread it through all three kernel launches.

    Rewrite compute_cslc_penetration_sphere() to output true pen_3d and smooth contact_weight.

    Rewrite jacobi_step() to use weighted softplus contact activation.

    Rewrite write_cslc_contacts() to use true 3D overlap and weighted contact export instead of hard radial culling.

    Run the lift test again and compare forward behavior with the current baseline.

    Add a small benchmark script for finite-difference gradient smoothness.

Bottom line

Today’s debugging fixed the bookkeeping issues and revealed the real modeling problem. The current CSLC prototype is no longer mainly failing because of slot allocation or pad mixing; it is failing because the contact model still depends on hard binary membership and projected penetration.

The next version should become a smooth, sphere-native distributed contact model:

    true 3D overlap,

    smooth compact support weighting,

    softplus solver activation,

    and gradient-based evaluation.

If you want, I can also format this into a shorter “meeting notes” version or a more implementation-heavy developer handoff version.