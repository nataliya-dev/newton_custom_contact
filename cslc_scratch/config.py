"""
config.py — All physical and simulation parameters for CSLC 3D.

Edit this file to change the experiment.  Nothing else imports constants
from anywhere else — this is the single source of truth.
"""

# ── Contact force law (Hunt & Crossley + regularised Coulomb) ──
K_CONTACT = 2_000      # N/m   — contact spring stiffness  (k_c)
D_CONTACT = 15.0       # s/m   — Hunt & Crossley damping   (d_c)
FRICTION_EPS = 0.005      # m/s   — Coulomb regularisation    (ε)
MU = 0.5  # — friction coefficient       (μ)

# ── Lattice spring constants ──
K_ANCHOR = 15_000     # N/m   — anchor spring stiffness   (k_a)
K_LATERAL = 5_000      # N/m   — lateral spring stiffness  (k_ℓ)

# ── Lattice solver ──
N_ITER = 60         # Jacobi iterations per timestep
ALPHA = 0.30       # Jacobi relaxation factor

# ── Finger pad geometry ──
PAD_NY = 11         # spheres along y  (width of pad)
PAD_NZ = 11         # spheres along z  (height of pad)
PAD_SPACING = 0.002      # m — centre-to-centre distance (2 mm)
SPHERE_RADIUS_FACTOR = 0.55  # radius = spacing * this → overlapping spheres

# ── Object geometry (rigid box) ──
BOX_HALF_X = 0.020      # m — half-width  along grip axis (20 mm)
BOX_HALF_Y = 0.015      # m — half-depth  (15 mm)
BOX_HALF_Z = 0.010      # m — half-height (10 mm)
BOX_MASS = 0.200      # kg  (200 g)

# ── Grip setup ──
NOMINAL_PEN = 0.0004     # m — nominal penetration at rest (0.4 mm)

# ── Dynamics ──
DT = 5e-5       # s — integration timestep
T_SETTLE = 0.05       # s — grip settling time before perturbation
T_AFTER = 0.40       # s — simulation time after perturbation
GRAVITY = 9.81       # m/s²

# ── Perturbation ──
TILT_IMPULSE = 0.5        # rad/s — initial angular velocity about z
