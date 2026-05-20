# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""cslc_main.theory --- pure-numpy reference for the IDEAL CSLC model.

The kernels in ``newton/_src/geometry/cslc_kernels.py`` are a GPU
discretisation of CSLC, tuned for Warp launches inside Newton's
collision pipeline.  This package is the opposite end of the lever: a
single-process numpy implementation of the same physics, written so
each line traces back to a printed equation in this module's
docstrings or to ``cslc_mujoco/docs/overleaf_theory_cslc_icra.txt``.

Why a separate reference?

1. We can write the equations the way they are derived, without the
   "stabilised diagonal + active-contact gate" surgery the GPU kernel
   needs for stability.  Numpy + scipy.optimize handles the non-
   linearity for us.

2. Every test in this package can be reproduced from first principles
   on a piece of paper (single-sphere closed forms, small-lattice
   eigenmodes, etc.), so when a kernel disagrees we have somewhere to
   stand.

3. It costs nothing to run.  No CUDA, no Newton model, no MuJoCo --
   ``uv run -m cslc_main.theory.test_01_single_sphere`` produces a
   PNG and a numerical table in seconds.

The ideal model uses the **deformed-centre** formulation throughout:
the lattice sphere physically moves to ``q = p - delta``, its radius
stays at ``r``, and the contact spring sees the actual squish of the
compliant layer.  The radius-reduction shim
(``effective_r = r - delta_n`` in ``write_cslc_contacts``) is
algebraically equivalent for face-on contact but not for off-axis
contact, so we adopt the deformed-centre form as the primary
definition.
"""
