# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UXPBD Phase 2: free SM-rigid, cross-substrate contact, PBD-R benchmarks."""

import os
import unittest

import newton
import newton.examples
from newton.tests.unittest_utils import add_function_test, get_test_devices

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "uxpbd")


def test_uxpbd_link_lattice_csr_offsets(test, device):
    """builder.add_lattice emits per-link CSR offsets used for per-link iteration."""
    builder = newton.ModelBuilder()
    link_a = builder.add_body(label="a")
    link_b = builder.add_body(label="b")
    builder.add_lattice(link=link_a, morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"), total_mass=1.0)
    builder.add_lattice(link=link_b, morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"), total_mass=1.0)
    model = builder.finalize(device=device)

    counts = model.link_lattice_sphere_count.numpy()
    starts = model.link_lattice_sphere_start.numpy()
    test.assertEqual(counts.shape[0], model.body_count)
    test.assertEqual(int(counts[link_a]), 5)
    test.assertEqual(int(counts[link_b]), 5)
    test.assertEqual(int(starts[link_a]), 0)
    test.assertEqual(int(starts[link_b]), 5)


class TestSolverUXPBDPhase2(unittest.TestCase):
    pass


add_function_test(
    TestSolverUXPBDPhase2,
    "test_uxpbd_link_lattice_csr_offsets",
    test_uxpbd_link_lattice_csr_offsets,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
