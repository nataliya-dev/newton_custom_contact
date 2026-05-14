# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels specific to UXPBD: lattice projection and lattice-aware contact.

The remaining kernels used by the solver (joint resolution, body integration,
restitution, body_parent_f reporting) live in
:mod:`newton._src.solvers.xpbd.kernels` and are imported there.
"""
