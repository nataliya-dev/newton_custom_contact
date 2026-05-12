# Tier 0 Results — Modal mathematics of the CSLC lattice

**Script:** [t0_modal_analysis.py](t0_modal_analysis.py)
**Figures:** [figures/](figures/)
**Status:** Complete. H0.1 PASS in both regimes; H0.2 PASS at coupled
regime, FAIL at paper regime (sub-grid `l_c`); H0.3 PASS on the plan's
three loads in both regimes.

## Summary table

| Hypothesis | Regime A (ka=15000, kl=500) | Regime B (ka=1000, kl=1000) |
|---|---|---|
| H0.1 eigenstructure | PASS (relative error < 1e-6 on first 6 eigvals) | PASS |
| H0.2 `l_c = sqrt(kl/ka)` | FAIL — `l_c_theory = 0.18` spacings is sub-grid | PASS — measured 0.829 vs theory 1.000 (17 % err), fit R^2 = 0.991 |
| H0.3 top-5 energy on plan loads | PASS — uniform 100 %, x-gradient 98.6 %, twist 97.3 % | PASS — same numbers |

## What the eigenstructure confirms (H0.1)

The CSLC lattice stiffness matrix `K = ka * I + kl * L` (where `L` is the
4-neighbour grid Laplacian) has eigenvectors equal to the open-grid
discrete cosine basis. The first six eigenpairs match the analytical
prediction `lambda_{p,q} = ka + 2 * kl * (2 - cos(p * pi / N) - cos(q * pi / N))`
to floating-point precision in both regimes. The first non-trivial mode
pair (p=1,q=0) / (0,1) is linear gradients; mode 3 is the (1,1) saddle;
modes 4,5 are the degenerate (2,0)/(0,2) pair (the eigensolver returns
some orthogonal combination — for regime B it's a `cos(2y)` band plus a
`cos(2x)+cos(2y)` bowl).

**Implication for the paper.** Section III.B (and the modal narrative in
the validation plan) speaks of "mode shapes as the dictionary the rest of
the paper speaks in". Tier 0 makes that concrete: bending = mode 1 or 2,
twist = mode 3, second-order curvature = modes 4-5, and the eigenvalues
follow a closed-form formula in `(ka, kl, N)`. No paper edit needed; this
is the figure-A backbone the plan called for.

## What the impulse response says about lateral coupling (H0.2)

The continuous Helmholtz operator `(ka - kl * nabla^2)` has a Green's
function that decays as `exp(-r / l_c)` with `l_c = sqrt(kl / ka)` in
lattice-spacing units. On the discrete lattice, solving `K * delta = e_i`
for the centre node and fitting `log |delta(r)|` to a line gives an
estimate of `l_c`.

**Regime B (ka=1000, kl=1000, `l_c_theory = 1.0` spacing):** measured
`l_c = 0.829`, fit R^2 = 0.991 across r in [1, 5] spacings. 17 % under-
shoot is consistent with finite-size effects from the open-grid boundary
within ~2 `l_c` of the source. The exponential fit is unambiguous.

**Regime A (ka=15000, kl=500, `l_c_theory = 0.18` spacing):** the
response decays to below 1 % of peak before reaching the first neighbour
ring, so the fit is rejected as ill-conditioned (< 4 valid points). This
is the scientifically important finding:

> The (ka=15000, kl=500) calibration prescribed by `squeeze_test.py`
> sets the lateral coupling to less than the grid resolution. **At
> these parameters the lattice's lateral coupling is decorative; the
> matrix is approximately diagonal anchor-spring response.** Any
> claim in the paper about lateral spread distinguishing CSLC from
> hydroelastic must either be made at parameters where `kl / ka` is
> close to 1, or be reframed as a topological argument about active
> contact accounting rather than an elastic-skin spreading claim.

This does not invalidate the squeeze/lift results — those measure grasp
behaviour macroscopically and the dominant CSLC advantage there is the
distributed anchor-contact stiffness (378 spheres × `keff` ~= `ke_bulk`
per pad), not lateral spreading. But the *theoretical* advertising of
CSLC's lateral spring should be tightened.

**Recommended follow-up before Tier 2:** sweep `kl / ka` in
{0.001, 0.01, 0.1, 1.0, 10.0} and pick the regime that matches whatever
real elastomer the paper wants to claim. Tier 2.5's H2.5.3 maps `ka`,
`kl` to `(E, G)` from FEM — that pinning is the rigorous way to fix the
ratio.

## What the modal projection says about indenter loads (H0.3)

The plan's three loads (uniform, linear gradient, twist) concentrate
> 97 % of their energy in the first five modes in *both* regimes, as
predicted. This is hypothesis H0.3 as written, and it passes.

A fourth load — a Gaussian bump with FWHM ~ 3 spacings, representing a
small-radius indenter — concentrates only 18-26 % of its energy in the
first five modes. **This is a useful diagnostic, not a failure:** it
quantifies how much spectral content sits in high modes when the load is
narrow relative to the lattice spacing. For Tier 2 this means a 10 mm-
radius indenter on a 5 mm-spacing pad will *not* be approximable by a
handful of modes; convergence to FEM (Tier 2.5) will be driven by
correctly resolving the indenter footprint rather than by mode
truncation.

The paper's "mode truncation" framing is therefore correct for *smooth*
contact patches (squeeze on a book cover, full-pad-on-cover overlap) but
needs to acknowledge that indenter-scale features require either many
modes or proper spatial resolution.

## Reproducibility

```
python cslc_v1/validation/t0_modal_analysis.py
```

Outputs:
- `cslc_v1/validation/figures/t0_mode_gallery_{A,B}.png` — 6-panel mode heatmap
- `cslc_v1/validation/figures/t0_impulse_response_{A,B}.png` — radial decay + fit
- `cslc_v1/validation/figures/t0_modal_energy_{A,B}.png` — |c_k|^2 vs k for all loads
- stdout: PASS/FAIL with all numerical evidence

## Next: Tier 1 — kernel sanity vs numpy reference

Tier 1 verifies that Newton's `lattice_solve_equilibrium` and `jacobi_step`
kernels reproduce the numpy reference `K^-1 * f`. With Tier 0 establishing
that the K matrix is well-formed and its physics interpretable, Tier 1
isolates the *kernel implementation*.
