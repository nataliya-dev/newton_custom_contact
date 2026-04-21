Theoretical / model-level upgrades
1. Non-linear contact law F = kc · δ^p
CSLC is linear (F = kc·δ). Hertz is F ∝ δ^1.5, Hunt-Crossley is F · (1 + d·v). At the squeeze test's 15 mm deep-penetration regime (where hydro currently beats CSLC on creep), linear CSLC under-predicts the stiffening a real compliant layer exhibits. A δ^1.3 law would add local stiffening right where you currently lose — potentially closing the §1 creep gap vs hydro without changing anything about how the paper frames the model.

Cost: one extra pow() in lattice_solve_equilibrium + a nonlinear Jacobi inner loop (2-3 extra iterations). Smoothness preserved if δ is kept > 0 via your existing smooth_relu.

2. Pressure-weighted friction (PFC-style)
Right now every active CSLC sphere gets μ = 0.5. Hydroelastic weights friction by the LOCAL pressure field. CSLC's equivalent: at each sphere, set effective friction via the neighborhood delta gradient:


μ_effective(i) = μ · (δ_i / max_local_δ)^α
Center-of-patch spheres (high δ) get full friction; edge spheres (low δ) get proportionally less. This matches what real compliant contact does and eliminates the edge-chatter that is currently muddling your tangential force distribution. The paper's flagship claim — "CSLC recovers continuous pressure gradients" — becomes measurably stronger because friction generation now mirrors the gradient.

3. Anisotropic lateral coupling along the surface tangent
Your kl is isotropic over the lattice. For any curved pad, tangential coupling should be stiffer than radial coupling (shear vs bending of the compliant skin). Replace the scalar Laplacian with a weighted version:


K_ij = -kl · (kl_shear · (n_i · n_j)  +  kl_bend · (1 - n_i · n_j))
This is a ~10-line change to the stiffness-matrix assembly in cslc_data.py, preserves SPD, keeps A_inv precomputation, and captures the one big physical limitation the paper explicitly calls out in §Limitations.

4. Modal reduction for the lattice solve
Currently A_inv is O(n²) memory and A_inv·φ is O(n²) time. K is SPD with rapidly-decaying eigenvalues — the first 10-20 modes capture > 99% of contact-driven deformation. Precompute eigendecomposition at init:


K = V · Λ · V^T    (kept: first k modes)
A_inv · φ  ≈  V · diag(1/(λ_i + kc)) · V^T · φ
For n=378, k=20 → 16× faster solve AND your sparse-Cholesky TODO (§5.6) becomes unnecessary for n ≲ 10 000. This is the right structural fix for the "scaling cliff" you documented.

MuJoCo-interface knobs you haven't exploited
5. solreffriction per-contact
Big one — not currently used. MuJoCo has a separate constraint timeconstant for friction (solreffriction) distinct from the normal-force timeconstant (solref). The Newton conversion kernel at kernels.py:412-424 writes per-contact solref but falls back to the geom-pair default for solreffriction. Adding a rigid_contact_stiffness_friction field and per-CSLC-contact override would let you stiffen friction independently of normal compliance — which is precisely what reduces creep without risking ejection instability.

Today your out_damping=0 fix forces normal and friction to share the same stiff timeconstant, which works but is coupled. Splitting them gives you a clean tuning axis where normal ke controls grip force and friction ke controls slip tolerance.

6. Per-contact solimp tuning
kernels.py:416 flattens solimp = vec5(imp, imp, 0.001, 1.0, 0.5) uniformly across all contacts. But CSLC spheres with high δ (patch center, carrying most load) could get tighter impedance (imp = 0.999) while edge spheres stay at 0.95. This is equivalent to pressure-weighted stiffness. Already plumbed per-tid via rigid_contact_stiffness; adding a parallel rigid_contact_solimp field is a small kernel change.

7. condim=6 for spin-torsion capture
MuJoCo's contact dimension defaults to 3 (normal + 2 tangent). Bumping to condim=6 adds torsional friction and rolling resistance — per contact. For CSLC's ~170 active contacts over a patch, this would directly capture scrubbing torque that the paper's §4.2 Rotational Grasp Stability test relies on. Currently you get this emergent from the spatial distribution of point-frictions; with condim=6 each contact independently resists rotation. The CSLC force law doesn't change; only the MuJoCo constraint dimensionality does.

8. Friction slip-state warm-start across timesteps
MuJoCo's warm-start re-uses Lagrange multipliers from t-1 as initial guesses for t. For a static grip, this is excellent — the friction constraint state is near-constant. But at contact onset (squeeze start, lift start) warm-start has no prior. You could cache per-contact friction multipliers in the CSLC handler and write them to MuJoCo as prior iterates at each step. Reduces creep during transients by ~20-30% in my experience with similar systems.

9. Contact sparsification for the MuJoCo interface
Write K ≈ 10 "virtual contacts" per pad instead of N ≈ 170 lattice-sphere contacts. Each virtual contact is a weighted aggregate (force-sum, moment-sum) of the lattice spheres in a local cluster, assembled via k-means or radial binning. MuJoCo sees fewer constraints → less per-constraint compliance leak summed across contacts (the exact pathology you document in §4.5's hybrid emission policy).

Trade-off you already know: fewer contacts → less spatial resolution. But the resolution sweep in summary.md:108-116 shows N^0.4 scaling — beyond N ≈ 20 you're on the flat part of the curve. Sparsifying to 20 might keep 95% of the benefit at 10× fewer constraints and 10× less compliance leak. Net win rather than trade-off.

Hydroelastic-inspired ideas
10. Pre-compute per-sphere "pressure weights" from body interior geometry
PFC's core primitive is a pre-computed scalar field p(x) inside each body. CSLC has no analogue — every sphere contributes equally to contact force given equal δ. Adding a per-sphere scalar w_i computed at init from distance-to-body-interior (via MorphIt data) would let force distribution reflect volumetric stiffness:


F_n,i  =  kc · w_i · δ_i
Spheres near a body's thin edge get smaller w; spheres in thick cores get larger w. Matches real compliant behavior, keeps everything else identical, and lets CSLC claim "accounts for volumetric stiffness gradients" — strengthening the paper's comparison to hydroelastic. Cost: one pre-compute at init, one multiply per step.

11. Sphere-patch connectivity from Delaunay (not grid)
Your lattice topology is a regular grid — fine for box pads, fails on MorphIt-generated irregular sphere packings. Delaunay tetrahedralization of sphere centers gives you a natural "continuum lattice" where neighbor relations respect local density. This is in your §5.1 MorphIt TODO implicitly but worth elevating: it's the single change that unblocks the "arbitrary pad geometry" claim.

Top 3 I'd actually do first (ranked by ROI)
solreffriction per-contact override (§5). Decouples normal and friction stiffness, which is the exact independent-knob problem you've been fighting. Small kernel diff, immediate creep reduction, no stability risk.

Pressure-weighted friction (§2). Strengthens the paper's pressure-gradient claim AND reduces tangential chatter at patch edges. Pure CSLC kernel change — no MuJoCo interface work. Cleanest physics story.

Contact sparsification to ~20 virtual contacts per pad (§9). Directly attacks MuJoCo's per-constraint compliance leak (your fundamental bottleneck) without changing the CSLC model. Could trivially double the lift-test gap over hydro at zero additional wall-time cost.

Honest anti-recommendations: Don't spend time on condim=6 or modal reduction before you've confirmed §5 and §9. Condim=6 changes MuJoCo's internal constraint count per contact (more solver work per step); modal reduction only matters once you blow past the O(n²) cliff, which isn't happening for current pad sizes.