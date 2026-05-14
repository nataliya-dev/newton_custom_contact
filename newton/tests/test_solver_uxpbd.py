# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UXPBD solver (Phase 1: articulated rigid + lattice + static contact)."""

import unittest

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_uxpbd_solver_rejects_enable_cslc(test, device):
    """Phase 1: enable_cslc=True must raise NotImplementedError (reserved for v2)."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    with test.assertRaises(NotImplementedError):
        newton.solvers.SolverUXPBD(model, enable_cslc=True)


def test_uxpbd_solver_instantiates_with_empty_model(test, device):
    """SolverUXPBD can be constructed from an empty Model without errors."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverUXPBD(model)
    test.assertIs(solver.model, model)


class TestSolverUXPBD(unittest.TestCase):
    pass


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_solver_rejects_enable_cslc",
    test_uxpbd_solver_rejects_enable_cslc,
    devices=get_test_devices(),
)
add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_solver_instantiates_with_empty_model",
    test_uxpbd_solver_instantiates_with_empty_model,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
