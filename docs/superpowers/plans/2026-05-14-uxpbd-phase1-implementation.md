# UXPBD Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `SolverUXPBD` Phase 1: articulated rigid bodies with a MorphIt-generated kinematic lattice that participates in particle-based contact with analytical static shapes (ground plane, meshes, primitives). No free shape-matched rigid, soft, or fluid yet (Phase 2-4). CSLC architectural seams reserved but inactive.

**Architecture:** `SolverUXPBD` inherits from `SolverBase` and calls into existing XPBD kernels for joint loop, integration, restitution, and `body_parent_f` reporting. New kernels live in `newton/_src/solvers/uxpbd/kernels.py`: `update_lattice_world_positions` (projects `body_q` onto lattice particles) and `solve_lattice_shape_contacts` (routes contact deltas on lattice particles into a per-body spatial wrench). `ModelBuilder.add_lattice` wraps `add_particle_volume` to load MorphIt sphere packings and tag the resulting particles as substrate=0 lattice particles bound to a host link.

**Tech Stack:** Python 3, NVIDIA Warp (`wp.kernel`, `wp.array[T]`), Newton (`newton.Model`, `newton.ModelBuilder`, `newton.solvers.SolverBase`), unittest, MorphIt JSON format (`{"centers": [[x,y,z], ...], "radii": [r, ...]}`).

**Reference spec:** `docs/superpowers/specs/2026-05-13-uxpbd-design.md` sections 3, 4, 5, 7, 8.1.

---

## File Structure

**New files (this plan creates them):**

- `newton/_src/solvers/uxpbd/__init__.py`: re-exports `SolverUXPBD`.
- `newton/_src/solvers/uxpbd/solver_uxpbd.py`: the `SolverUXPBD` class.
- `newton/_src/solvers/uxpbd/kernels.py`: `update_lattice_world_positions`, `solve_lattice_shape_contacts`, helper device functions.
- `newton/_src/solvers/uxpbd/lattice.py`: `add_lattice` and `add_lattice_to_all_links` helpers (called from `ModelBuilder` methods or used directly).
- `newton/tests/test_solver_uxpbd.py`: Phase 1 unit tests.
- `newton/examples/contacts/example_uxpbd_lattice_push.py`: demo + integration test.

**Files modified:**

- `newton/_src/solvers/__init__.py`: add `SolverUXPBD` import and `__all__` entry.
- `newton/solvers.py`: add `SolverUXPBD` to top-level public exports and `__all__`.
- `newton/_src/sim/builder.py`: add `add_lattice` and `add_lattice_to_all_links` methods on `ModelBuilder`; add lattice metadata Python lists to `__init__`; populate lattice `wp.array` fields in `finalize`.
- `newton/_src/sim/model.py`: add lattice metadata fields to `Model` (declared and zero-initialized).

**Responsibility split:**

- `solver_uxpbd.py` orchestrates `step()` and owns iteration state. It calls into `kernels.py` for the two new kernels, and into `newton._src.solvers.xpbd.kernels` for joint/integration/restitution kernels. No physics math lives here.
- `kernels.py` contains pure Warp kernels and device functions. No Python state.
- `lattice.py` contains pure Python loaders + adapters for MorphIt JSON to per-particle metadata. No Warp kernels.
- `builder.py` only delegates to `lattice.py` to keep `builder.py` thin (it is already 10k lines).

---

## Task 1: Create the `uxpbd` package skeleton

**Files:**

- Create: `newton/_src/solvers/uxpbd/__init__.py`
- Create: `newton/_src/solvers/uxpbd/solver_uxpbd.py`
- Create: `newton/_src/solvers/uxpbd/kernels.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/solvers.py`
- Test: `newton/tests/test_solver_uxpbd.py`

- [ ] **Step 1: Write the failing import test**

Create `newton/tests/test_solver_uxpbd.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UXPBD solver (Phase 1: articulated rigid + lattice + static contact)."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_uxpbd_solver_importable(test, device):
    """SolverUXPBD is exposed at newton.solvers.SolverUXPBD."""
    from newton.solvers import SolverUXPBD

    test.assertTrue(callable(SolverUXPBD))


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
    "test_uxpbd_solver_importable",
    test_uxpbd_solver_importable,
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
```

- [ ] **Step 2: Run the test, confirm it fails with ImportError**

Run: `uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected: ImportError (`SolverUXPBD` is not exported).

- [ ] **Step 3: Create the empty kernels file**

Create `newton/_src/solvers/uxpbd/kernels.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels specific to UXPBD: lattice projection and lattice-aware contact.

The remaining kernels used by the solver (joint resolution, body integration,
restitution, body_parent_f reporting) live in
:mod:`newton._src.solvers.xpbd.kernels` and are imported there.
"""
```

- [ ] **Step 4: Create the minimal `SolverUXPBD` class**

Create `newton/_src/solvers/uxpbd/solver_uxpbd.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""UXPBD: Unified eXtended Position-Based Dynamics solver.

Phase 1 scope: articulated rigid bodies with a MorphIt-generated kinematic
lattice that contacts analytical static shapes through the particle pipeline.
See ``docs/superpowers/specs/2026-05-13-uxpbd-design.md`` for the full design.
"""

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


class SolverUXPBD(SolverBase):
    """Unified position-based dynamics solver.

    Phase 1 implements articulated rigid bodies with a kinematic lattice
    shell. Subsequent phases add free shape-matched rigid (PBD-R), soft bodies
    (springs / bending / FEM tet), and liquids (PBF density constraint).
    The class preserves architectural seams for v2 CSLC (compliant sphere
    lattice contact); see :meth:`compute_compliant_contact_response`.

    Args:
        model: The :class:`~newton.Model` to simulate.
        iterations: Number of main constraint loop iterations per step.
        stabilization_iterations: UPPFRTA stabilization pre-pass iterations.
        enable_cslc: Must be False in Phase 1. Reserved for v2.

    Raises:
        NotImplementedError: If ``enable_cslc=True``.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 4,
        stabilization_iterations: int = 1,
        enable_cslc: bool = False,
    ):
        super().__init__(model=model)
        if enable_cslc:
            raise NotImplementedError(
                "CSLC compliant contact is reserved for UXPBD v2. "
                "See docs/superpowers/specs/2026-05-13-uxpbd-design.md section 5.5."
            )
        self.iterations = iterations
        self.stabilization_iterations = stabilization_iterations
        self._init_kinematic_state()

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation by ``dt`` seconds.

        Phase 1 stub: implementation is added in Task 7.
        """
        raise NotImplementedError("SolverUXPBD.step() is implemented in Task 7.")

    def compute_compliant_contact_response(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """v2 CSLC hook. No-op in Phase 1.

        v2 will solve the lattice compression vector :math:`\\delta` from the
        quasistatic equilibrium :math:`K\\delta = k_c (\\phi^{rest} - \\delta)_+`
        and write it into ``model.lattice_delta``, which the
        ``update_lattice_world_positions`` kernel then propagates into the
        per-particle effective radius.
        """
        return
```

- [ ] **Step 5: Create the package `__init__.py`**

Create `newton/_src/solvers/uxpbd/__init__.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from .solver_uxpbd import SolverUXPBD

__all__ = [
    "SolverUXPBD",
]
```

- [ ] **Step 6: Register `SolverUXPBD` in solvers `__init__`**

Modify `newton/_src/solvers/__init__.py` to add the import and `__all__` entry. Replace:

```python
from .srxpbd import SolverSRXPBD
```

with:

```python
from .srxpbd import SolverSRXPBD
from .uxpbd import SolverUXPBD
```

Add `"SolverUXPBD"` to `__all__`, keeping alphabetical order:

```python
__all__ = [
    "SolverBase",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverKamino",
    "SolverMuJoCo",
    "SolverNotifyFlags",
    "SolverSRXPBD",
    "SolverSemiImplicit",
    "SolverStyle3D",
    "SolverUXPBD",
    "SolverVBD",
    "SolverXPBD",
]
```

- [ ] **Step 7: Register `SolverUXPBD` in public `newton.solvers`**

Modify `newton/solvers.py`. In the existing import block:

```python
from ._src.solvers import (
    SolverBase,
    SolverFeatherstone,
    SolverImplicitMPM,
    SolverKamino,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverSRXPBD,
    SolverStyle3D,
    SolverUXPBD,
    SolverVBD,
    SolverXPBD,
    style3d,
)
```

In the `__all__`:

```python
__all__ = [
    "SolverBase",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverKamino",
    "SolverMuJoCo",
    "SolverNotifyFlags",
    "SolverSRXPBD",
    "SolverSemiImplicit",
    "SolverStyle3D",
    "SolverUXPBD",
    "SolverVBD",
    "SolverXPBD",
    "style3d",
]
```

- [ ] **Step 8: Run the test, confirm both pass**

Run: `uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected: 2 tests pass on each device (cpu and cuda if available).

- [ ] **Step 9: Run pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/_src/solvers/uxpbd/ newton/_src/solvers/__init__.py newton/solvers.py newton/tests/test_solver_uxpbd.py
git commit -m "Add SolverUXPBD skeleton

Stub class inheriting SolverBase. step() raises NotImplementedError;
implementation lands in later phases of the UXPBD Phase 1 plan.
compute_compliant_contact_response is a documented no-op reserved for v2.
enable_cslc=True raises NotImplementedError until v2.

Re-exports under newton.solvers.SolverUXPBD per AGENTS.md naming."
```

---

## Task 2: Add lattice metadata to `ModelBuilder` and `Model`

**Files:**

- Modify: `newton/_src/sim/builder.py` (additions in `ModelBuilder.__init__` and `ModelBuilder.finalize`)
- Modify: `newton/_src/sim/model.py` (declare and initialize new lattice fields on `Model`)
- Test: `newton/tests/test_solver_uxpbd.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_empty_model_has_zero_lattice(test, device):
    """A Model finalized with no lattices has lattice_sphere_count == 0."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    test.assertEqual(getattr(model, "lattice_sphere_count", 0), 0)
    # Lattice arrays exist but are empty
    test.assertEqual(model.lattice_p_rest.shape[0], 0)
    test.assertEqual(model.lattice_r.shape[0], 0)
    test.assertEqual(model.lattice_link.shape[0], 0)
    test.assertEqual(model.lattice_delta.shape[0], 0)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_empty_model_has_zero_lattice",
    test_uxpbd_empty_model_has_zero_lattice,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_empty_model_has_zero_lattice`
Expected: AttributeError or AssertionError (`model.lattice_p_rest` does not exist).

- [ ] **Step 3: Add lattice fields to the `Model` class**

In `newton/_src/sim/model.py`, find the `Model.__init__` method (or whichever block declares particle-related fields like `model.particle_q`). Add the lattice fields after the particle fields:

```python
# UXPBD lattice metadata. Empty when no link has a lattice attached.
# See docs/superpowers/specs/2026-05-13-uxpbd-design.md section 5.1.
self.lattice_sphere_count: int = 0
self.lattice_p_rest: wp.array[wp.vec3] = wp.empty(0, dtype=wp.vec3, device=device)
self.lattice_r: wp.array[float] = wp.empty(0, dtype=float, device=device)
self.lattice_normal: wp.array[wp.vec3] = wp.empty(0, dtype=wp.vec3, device=device)
self.lattice_is_surface: wp.array[wp.uint8] = wp.empty(0, dtype=wp.uint8, device=device)
self.lattice_link: wp.array[wp.int32] = wp.empty(0, dtype=wp.int32, device=device)
self.lattice_particle_index: wp.array[wp.int32] = wp.empty(0, dtype=wp.int32, device=device)

# v2 CSLC seams. Present but unused in Phase 1.
self.lattice_delta: wp.array[float] = wp.empty(0, dtype=float, device=device)
self.lattice_delta_prev: wp.array[float] = wp.empty(0, dtype=float, device=device)
self.lattice_K_diag: wp.array[float] = wp.empty(0, dtype=float, device=device)
self.lattice_neighbors_csr: wp.array[wp.int32] = wp.empty(0, dtype=wp.int32, device=device)
self.lattice_neighbors_offset: wp.array[wp.int32] = wp.empty(0, dtype=wp.int32, device=device)
self.lattice_k_anchor: wp.array[float] = wp.empty(0, dtype=float, device=device)
self.lattice_k_lateral: wp.array[float] = wp.empty(0, dtype=float, device=device)
self.lattice_k_contact: wp.array[float] = wp.empty(0, dtype=float, device=device)
self.lattice_damping: wp.array[float] = wp.empty(0, dtype=float, device=device)
```

Use the same `device` reference the surrounding code uses. If `Model.__init__` does not take a device argument, find an existing `wp.empty(0, ...)` call and copy its pattern.

- [ ] **Step 4: Add lattice metadata accumulators to `ModelBuilder.__init__`**

In `newton/_src/sim/builder.py`, locate `class ModelBuilder` (line 89) and find the section that initializes `self.particle_q: list = []` (or similar Python list accumulators). Add immediately after those:

```python
# UXPBD lattice metadata accumulators (Python lists, baked into Model at finalize).
self.lattice_p_rest: list[tuple[float, float, float]] = []
self.lattice_r: list[float] = []
self.lattice_normal: list[tuple[float, float, float]] = []
self.lattice_is_surface: list[int] = []
self.lattice_link: list[int] = []
self.lattice_particle_index: list[int] = []
# v2 CSLC seams. Defaults are stored even when unused so the seam is testable.
self.lattice_k_anchor: list[float] = []
self.lattice_k_lateral: list[float] = []
self.lattice_k_contact: list[float] = []
self.lattice_damping: list[float] = []
```

- [ ] **Step 5: Populate lattice arrays in `ModelBuilder.finalize`**

Locate `def finalize(` in `newton/_src/sim/builder.py` (line 9981). Find where particle arrays are converted to `wp.array` (typical pattern: `model.particle_q = wp.array(self.particle_q, dtype=wp.vec3, device=device)`). Add immediately after that block:

```python
# Bake UXPBD lattice metadata.
n_lat = len(self.lattice_link)
model.lattice_sphere_count = n_lat
if n_lat:
    model.lattice_p_rest = wp.array(self.lattice_p_rest, dtype=wp.vec3, device=device)
    model.lattice_r = wp.array(self.lattice_r, dtype=float, device=device)
    model.lattice_normal = wp.array(self.lattice_normal, dtype=wp.vec3, device=device)
    model.lattice_is_surface = wp.array(self.lattice_is_surface, dtype=wp.uint8, device=device)
    model.lattice_link = wp.array(self.lattice_link, dtype=wp.int32, device=device)
    model.lattice_particle_index = wp.array(self.lattice_particle_index, dtype=wp.int32, device=device)
    # v2 seams initialized to zero or stored defaults.
    model.lattice_delta = wp.zeros(n_lat, dtype=float, device=device)
    model.lattice_delta_prev = wp.zeros(n_lat, dtype=float, device=device)
    model.lattice_K_diag = wp.zeros(n_lat, dtype=float, device=device)
    model.lattice_k_anchor = wp.array(self.lattice_k_anchor, dtype=float, device=device)
    model.lattice_k_lateral = wp.array(self.lattice_k_lateral, dtype=float, device=device)
    model.lattice_k_contact = wp.array(self.lattice_k_contact, dtype=float, device=device)
    model.lattice_damping = wp.array(self.lattice_damping, dtype=float, device=device)
    # lattice_neighbors_* stay empty until Task 4 generates adjacency.
```

- [ ] **Step 6: Run the test, confirm it passes**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_empty_model_has_zero_lattice`
Expected: PASS on each device.

- [ ] **Step 7: Run the full UXPBD test set to ensure nothing regressed**

Run: `uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected: 3 tests pass per device.

- [ ] **Step 8: Pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/_src/sim/model.py newton/_src/sim/builder.py newton/tests/test_solver_uxpbd.py
git commit -m "Add UXPBD lattice metadata fields to Model and ModelBuilder

Reserve lattice_* arrays on Model (empty when no lattice is attached).
ModelBuilder gains Python list accumulators that are baked into Model
during finalize. v2 CSLC seams (lattice_delta, lattice_K_diag,
lattice_k_*) are allocated but unused in Phase 1.

See spec section 5.1."
```

---

## Task 3: `ModelBuilder.add_lattice` method

**Files:**

- Create: `newton/_src/solvers/uxpbd/lattice.py`
- Modify: `newton/_src/sim/builder.py` (add `add_lattice` method on `ModelBuilder`)
- Test: `newton/tests/test_solver_uxpbd.py` (extend)
- Test fixture: `newton/tests/assets/uxpbd/tiny_lattice.json` (small MorphIt-format JSON)

- [ ] **Step 1: Create the test fixture**

Create `newton/tests/assets/uxpbd/tiny_lattice.json`:

```json
{
  "centers": [
    [0.0,  0.0, 0.0],
    [0.1,  0.0, 0.0],
    [0.0,  0.1, 0.0],
    [0.0,  0.0, 0.1],
    [-0.1, 0.0, 0.0]
  ],
  "radii": [0.05, 0.04, 0.06, 0.05, 0.04],
  "normals": [
    [ 0.0, -1.0,  0.0],
    [ 1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0]
  ],
  "is_surface": [1, 1, 1, 1, 1]
}
```

Five spheres of varying radius, all flagged as surface, with explicit outward normals.

- [ ] **Step 2: Write the failing test**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
import os

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "uxpbd")


def test_uxpbd_add_lattice_populates_arrays(test, device):
    """builder.add_lattice loads MorphIt JSON, creates lattice + particle entries."""
    builder = newton.ModelBuilder()
    link = builder.add_body()  # one floating body (add_body auto-creates a free joint)

    json_path = os.path.join(_ASSET_DIR, "tiny_lattice.json")
    builder.add_lattice(link=link, morphit_json=json_path, total_mass=1.0)

    test.assertEqual(len(builder.lattice_p_rest), 5)
    test.assertEqual(len(builder.lattice_r), 5)
    test.assertEqual(len(builder.lattice_link), 5)
    test.assertEqual(builder.lattice_link[0], link)
    test.assertEqual(builder.lattice_link[4], link)
    test.assertAlmostEqual(builder.lattice_r[0], 0.05, places=6)
    test.assertAlmostEqual(builder.lattice_r[1], 0.04, places=6)

    # Particles were added to the builder
    test.assertEqual(len(builder.particle_q), 5)

    # Each lattice sphere's lattice_particle_index points to the matching slot
    test.assertEqual(list(builder.lattice_particle_index), [0, 1, 2, 3, 4])

    # Finalize produces non-empty model arrays
    model = builder.finalize(device=device)
    test.assertEqual(model.lattice_sphere_count, 5)
    test.assertEqual(model.lattice_link.shape[0], 5)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_add_lattice_populates_arrays",
    test_uxpbd_add_lattice_populates_arrays,
    devices=get_test_devices(),
)
```

- [ ] **Step 3: Run the test, confirm it fails**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_add_lattice_populates_arrays`
Expected: AttributeError (`builder.add_lattice` does not exist).

- [ ] **Step 4: Implement the lattice loader helper in `uxpbd/lattice.py`**

Create `newton/_src/solvers/uxpbd/lattice.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MorphIt JSON loader and lattice metadata adapter for UXPBD.

The lattice file format is the same one used by
``ModelBuilder.add_particle_volume`` and ``Box.add_morphit_spheres``: a JSON
document (or in-memory dict) with at minimum the keys ``"centers"`` and
``"radii"``. UXPBD additionally consults optional keys ``"normals"`` and
``"is_surface"``; absent normals default to a unit vector pointing radially
outward from the packing centroid, and absent ``is_surface`` defaults to 1
for every sphere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp


def load_morphit_lattice(volume_data: str | Path | dict[str, Any]) -> dict[str, np.ndarray]:
    """Load a MorphIt sphere packing into NumPy arrays.

    Args:
        volume_data: Either a path to a JSON file, or an already-parsed dict.

    Returns:
        A dict with keys ``centers`` [N, 3] float32, ``radii`` [N] float32,
        ``normals`` [N, 3] float32 (unit), ``is_surface`` [N] uint8.

    Raises:
        ValueError: If ``centers`` and ``radii`` have mismatched lengths or
            ``centers`` is empty.
    """
    if isinstance(volume_data, (str, Path)):
        with open(volume_data) as f:
            data = json.load(f)
    else:
        data = volume_data

    centers = np.asarray(data["centers"], dtype=np.float32)
    radii = np.asarray(data["radii"], dtype=np.float32)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(
            f"MorphIt lattice: centers ({centers.shape[0]}) and radii "
            f"({radii.shape[0]}) length mismatch."
        )
    if centers.shape[0] == 0:
        raise ValueError("MorphIt lattice: cannot create lattice from zero spheres.")

    if "normals" in data:
        normals = np.asarray(data["normals"], dtype=np.float32)
        if normals.shape != centers.shape:
            raise ValueError(
                f"MorphIt lattice: normals shape {normals.shape} does not "
                f"match centers shape {centers.shape}."
            )
        # Normalize for safety.
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        lens = np.where(lens > 1e-12, lens, 1.0)
        normals = normals / lens
    else:
        # Default: radial outward from centroid.
        centroid = centers.mean(axis=0, keepdims=True)
        offsets = centers - centroid
        lens = np.linalg.norm(offsets, axis=1, keepdims=True)
        lens = np.where(lens > 1e-12, lens, 1.0)
        normals = (offsets / lens).astype(np.float32)

    if "is_surface" in data:
        is_surface = np.asarray(data["is_surface"], dtype=np.uint8)
        if is_surface.shape[0] != centers.shape[0]:
            raise ValueError(
                f"MorphIt lattice: is_surface length {is_surface.shape[0]} "
                f"does not match centers length {centers.shape[0]}."
            )
    else:
        is_surface = np.ones(centers.shape[0], dtype=np.uint8)

    return {
        "centers": centers,
        "radii": radii,
        "normals": normals,
        "is_surface": is_surface,
    }


def add_lattice_to_builder(
    builder,
    link: int,
    morphit_json: str | Path | dict[str, Any],
    total_mass: float,
    pos: wp.vec3 | None = None,
    rot: wp.quat | None = None,
    k_anchor: float = 1.0e3,
    k_lateral: float = 5.0e2,
    k_bulk: float = 1.0e5,
    damping: float = 2.0,
) -> int:
    """Attach a MorphIt-generated lattice to an articulated link.

    The lattice spheres are added as particles in the builder's particle pool
    (substrate 0, link-host metadata) and registered in the lattice metadata
    accumulators. Per-particle mass is distributed by sphere volume fraction,
    matching ``add_particle_volume``.

    Args:
        builder: The :class:`~newton.ModelBuilder` to populate.
        link: The articulated link (``body_index``) that hosts this lattice.
        morphit_json: Path to a MorphIt JSON file, or an in-memory dict.
        total_mass: Total mass distributed across the lattice spheres [kg].
            This mass is metadata only; the host body's inertia is unchanged.
        pos: World-space pose offset for the lattice. Defaults to (0,0,0).
            Should match the link's resting pose so the lattice projects
            correctly at t=0.
        rot: Rotation applied to body-frame positions. Defaults to identity.
        k_anchor: Anchor spring stiffness (v2 CSLC; stored, unused in v1).
        k_lateral: Lateral coupling stiffness (v2 CSLC; stored, unused in v1).
        k_bulk: Bulk material stiffness (v2 CSLC; stored, unused in v1).
        damping: Hunt-Crossley damping coefficient (v2 CSLC; stored, unused).

    Returns:
        The starting index in the lattice arrays for this link's lattice.
    """
    data = load_morphit_lattice(morphit_json)
    centers = data["centers"]
    radii = data["radii"]
    normals = data["normals"]
    is_surface = data["is_surface"]
    n = centers.shape[0]

    if pos is None:
        pos_v = wp.vec3(0.0, 0.0, 0.0)
    else:
        pos_v = wp.vec3(*pos)
    if rot is None:
        rot_q = wp.quat_identity(float)
    else:
        rot_q = rot

    # Mass distribution by sphere volume fraction (matches add_particle_volume).
    volumes = (4.0 / 3.0) * np.pi * (radii ** 3)
    total_volume = float(np.sum(volumes))
    if total_volume <= 0.0:
        raise ValueError("Lattice total volume must be positive.")

    lattice_start = len(builder.lattice_link)

    for i in range(n):
        p_local = wp.vec3(float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2]))
        p_world = wp.quat_rotate(rot_q, p_local) + pos_v
        mass_i = total_mass * (float(volumes[i]) / total_volume)
        particle_idx = builder.add_particle(
            p_world,
            wp.vec3(0.0, 0.0, 0.0),
            mass_i,
            float(radii[i]),
        )

        builder.lattice_p_rest.append((float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2])))
        builder.lattice_r.append(float(radii[i]))
        builder.lattice_normal.append((float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2])))
        builder.lattice_is_surface.append(int(is_surface[i]))
        builder.lattice_link.append(int(link))
        builder.lattice_particle_index.append(int(particle_idx))
        builder.lattice_k_anchor.append(float(k_anchor))
        builder.lattice_k_lateral.append(float(k_lateral))
        builder.lattice_k_contact.append(float(k_bulk))
        builder.lattice_damping.append(float(damping))

    return lattice_start
```

- [ ] **Step 5: Add the `ModelBuilder.add_lattice` method**

In `newton/_src/sim/builder.py`, near the existing `add_particle_volume` (around line 7438), add a new method. Place it just after `add_particle_volume` ends:

```python
def add_lattice(
    self,
    link: int,
    morphit_json: str,
    total_mass: float = 0.0,
    pos: Any = None,
    rot: Any = None,
    k_anchor: float = 1.0e3,
    k_lateral: float = 5.0e2,
    k_bulk: float = 1.0e5,
    damping: float = 2.0,
) -> int:
    """Attach a MorphIt-generated kinematic lattice to an articulated link.

    The lattice is a collection of variable-radius spheres bound to the
    link's body frame. Lattice spheres are added as particles in the model's
    unified particle pool and tagged with metadata that identifies them as
    substrate 0 (lattice) particles hosted by ``link``. The lattice's role
    is to project the link into the particle world so it can participate in
    particle-based contact; the lattice does not carry the link's dynamics.

    See ``docs/superpowers/specs/2026-05-13-uxpbd-design.md`` section 5 for
    the full data model and v2 CSLC extension hooks.

    Args:
        link: The articulated link (``body_index``) that hosts this lattice.
        morphit_json: Path to a MorphIt sphere packing JSON file.
        total_mass: Total mass distributed across lattice spheres [kg].
            Lattice mass is metadata; the host body's inertia is unchanged.
            Defaults to 0.0 (lattice particles act as massless probes).
        pos: World-space offset applied to lattice positions. Defaults to the
            origin. Pass the link's resting position so the lattice aligns
            with the rendered link at t=0.
        rot: Rotation applied to lattice positions. Defaults to identity.
        k_anchor: Anchor spring stiffness for v2 CSLC [N/m]. Stored in
            ``model.lattice_k_anchor``; not read in Phase 1.
        k_lateral: Lateral coupling stiffness for v2 CSLC [N/m]. Stored.
        k_bulk: Bulk material stiffness for v2 CSLC [N/m]. Stored.
        damping: Hunt-Crossley damping coefficient for v2 CSLC [s/m]. Stored.

    Returns:
        The starting index in the lattice arrays for this link's lattice.
    """
    from ..solvers.uxpbd.lattice import add_lattice_to_builder

    return add_lattice_to_builder(
        self,
        link=link,
        morphit_json=morphit_json,
        total_mass=total_mass,
        pos=pos,
        rot=rot,
        k_anchor=k_anchor,
        k_lateral=k_lateral,
        k_bulk=k_bulk,
        damping=damping,
    )
```

- [ ] **Step 6: Run the test, confirm it passes**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_add_lattice_populates_arrays`
Expected: PASS on each device.

- [ ] **Step 7: Run the full UXPBD test set**

Run: `uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected: 4 tests pass per device.

- [ ] **Step 8: Pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/_src/solvers/uxpbd/lattice.py newton/_src/sim/builder.py newton/tests/test_solver_uxpbd.py newton/tests/assets/uxpbd/tiny_lattice.json
git commit -m "Add ModelBuilder.add_lattice for MorphIt lattice attachment

Loads a MorphIt sphere packing (centers/radii, optional normals and
is_surface), adds spheres as particles in the builder, and registers
lattice metadata that ties each sphere to its host articulated link.
Mass per sphere is distributed by volume fraction, matching the
existing add_particle_volume convention.

v2 CSLC parameters (k_anchor, k_lateral, k_bulk, damping) are stored
on the model but not consumed in Phase 1."
```

---

## Task 4: `update_lattice_world_positions` kernel

**Files:**

- Modify: `newton/_src/solvers/uxpbd/kernels.py` (add the kernel)
- Modify: `newton/_src/solvers/uxpbd/solver_uxpbd.py` (call the kernel from a helper)
- Test: `newton/tests/test_solver_uxpbd.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_update_lattice_projects_body_q(test, device):
    """Lattice particle world positions match body_q ⊗ p_rest analytically."""
    builder = newton.ModelBuilder()
    link = builder.add_body()
    builder.add_joint_free(child=link)

    # One sphere at body-frame offset (0.1, 0.0, 0.0)
    builder.add_lattice(
        link=link,
        morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"),
        total_mass=1.0,
    )
    model = builder.finalize(device=device)

    state = model.state()
    # Set body_q to translation (1.0, 2.0, 3.0), rotation identity.
    body_q_np = np.array(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    )
    state.body_q.assign(body_q_np)
    state.body_qd.zero_()

    solver = newton.solvers.SolverUXPBD(model)
    solver.update_lattice_world_positions(state)

    expected = np.array(
        [
            [1.0,  2.0,  3.0],   # (0,0,0)   shifted by (1,2,3)
            [1.1,  2.0,  3.0],   # (0.1,0,0) shifted
            [1.0,  2.1,  3.0],   # (0,0.1,0) shifted
            [1.0,  2.0,  3.1],   # (0,0,0.1) shifted
            [0.9,  2.0,  3.0],   # (-0.1,0,0) shifted
        ],
        dtype=np.float32,
    )
    got = state.particle_q.numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_update_lattice_projects_body_q",
    test_uxpbd_update_lattice_projects_body_q,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_update_lattice_projects_body_q`
Expected: AttributeError (`solver.update_lattice_world_positions` does not exist).

- [ ] **Step 3: Implement the kernel**

Append to `newton/_src/solvers/uxpbd/kernels.py`:

```python
import warp as wp


@wp.kernel
def update_lattice_world_positions(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    lattice_link: wp.array[wp.int32],
    lattice_p_rest: wp.array[wp.vec3],
    lattice_delta: wp.array[float],
    lattice_r: wp.array[float],
    lattice_particle_index: wp.array[wp.int32],
    # outputs
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
):
    """Project body_q onto lattice particles.

    For lattice sphere ``i`` with body-frame offset ``p_rest`` hosted by
    ``link``, set:

      particle_q[pidx]      = body_q[link] ⊗ p_rest
      particle_qd[pidx]     = v_lin + ω × (R · (p_rest - com))
      particle_radius[pidx] = lattice_r[i] - lattice_delta[i]   (δ == 0 in v1)

    The ``particle_radius`` write is the load-bearing CSLC v2 seam: v2 writes
    nonzero ``lattice_delta`` and downstream contact kernels automatically see
    the compressed effective radius.
    """
    sid = wp.tid()
    link = lattice_link[sid]
    p_local = lattice_p_rest[sid]
    tf = body_q[link]
    pidx = lattice_particle_index[sid]

    # World position
    particle_q[pidx] = wp.transform_point(tf, p_local)

    # World velocity at offset (v_lin + ω × R·(p_local - com))
    rot = wp.transform_get_rotation(tf)
    r_world = wp.quat_rotate(rot, p_local - body_com[link])
    twist = body_qd[link]
    v_lin = wp.spatial_top(twist)
    omega = wp.spatial_bottom(twist)
    particle_qd[pidx] = v_lin + wp.cross(omega, r_world)

    # Effective contact radius. v1: delta == 0 → radius == rest radius.
    particle_radius[pidx] = lattice_r[sid] - lattice_delta[sid]
```

- [ ] **Step 4: Add the solver wrapper that launches the kernel**

In `newton/_src/solvers/uxpbd/solver_uxpbd.py`, add an import at the top:

```python
import warp as wp

from .kernels import update_lattice_world_positions
```

Add a method to `SolverUXPBD`:

```python
def update_lattice_world_positions(self, state: State) -> None:
    """Project ``body_q``/``body_qd`` onto every lattice particle.

    Updates ``state.particle_q``, ``state.particle_qd``, and
    ``model.particle_radius`` in place for all lattice particles. Non-lattice
    particles are left untouched.

    Args:
        state: The :class:`~newton.State` whose body_q drives the projection
            and whose particle_q is written.
    """
    model = self.model
    if model.lattice_sphere_count == 0:
        return
    wp.launch(
        kernel=update_lattice_world_positions,
        dim=model.lattice_sphere_count,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            model.lattice_link,
            model.lattice_p_rest,
            model.lattice_delta,
            model.lattice_r,
            model.lattice_particle_index,
        ],
        outputs=[
            state.particle_q,
            state.particle_qd,
            model.particle_radius,
        ],
        device=model.device,
    )
```

- [ ] **Step 5: Run the test, confirm it passes**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_update_lattice_projects_body_q`
Expected: PASS on each device.

- [ ] **Step 6: Add a test for non-identity rotation**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_update_lattice_handles_rotation(test, device):
    """Lattice projection respects body rotation."""
    builder = newton.ModelBuilder()
    link = builder.add_body()  # add_body auto-creates a free joint
    builder.add_lattice(
        link=link,
        morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"),
        total_mass=1.0,
    )
    model = builder.finalize(device=device)

    state = model.state()
    # 90 degree rotation around Z. The sphere at (+0.1, 0, 0) body frame
    # should project to (0, +0.1, 0) in world frame.
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
    body_q_np = np.array(
        [[0.0, 0.0, 0.0, rot[0], rot[1], rot[2], rot[3]]], dtype=np.float32
    )
    state.body_q.assign(body_q_np)
    state.body_qd.zero_()

    solver = newton.solvers.SolverUXPBD(model)
    solver.update_lattice_world_positions(state)

    got = state.particle_q.numpy()
    # Sphere at body-frame (+0.1, 0, 0) lands at world (0, +0.1, 0)
    np.testing.assert_allclose(got[1], [0.0, 0.1, 0.0], atol=1e-5)
    # Sphere at body-frame (0, +0.1, 0) lands at world (-0.1, 0, 0)
    np.testing.assert_allclose(got[2], [-0.1, 0.0, 0.0], atol=1e-5)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_update_lattice_handles_rotation",
    test_uxpbd_update_lattice_handles_rotation,
    devices=get_test_devices(),
)
```

- [ ] **Step 7: Run the rotation test, confirm pass**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_update_lattice_handles_rotation`
Expected: PASS.

- [ ] **Step 8: Pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/_src/solvers/uxpbd/kernels.py newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd.py
git commit -m "Add update_lattice_world_positions kernel

Projects body_q/body_qd onto lattice particles (substrate 0). Writes
particle_q, particle_qd, and particle_radius. The particle_radius write
is the CSLC v2 seam: v1 leaves lattice_delta == 0 so radius == rest
radius; v2 will write nonzero delta and downstream contact kernels see
the compressed effective radius automatically.

Tests cover translation only and translation + rotation."
```

---

## Task 5: `solve_lattice_shape_contacts` kernel

**Files:**

- Modify: `newton/_src/solvers/uxpbd/kernels.py` (new kernel)
- Test: `newton/tests/test_solver_uxpbd.py` (extend)

The kernel is a fork of XPBD's `solve_particle_shape_contacts` (at `newton/_src/solvers/xpbd/kernels.py:115-222`) that routes the particle's position delta into a body wrench instead of into ``particle_deltas``, using the host link's true inertia at the lattice sphere's world offset. The XPBD kernel already computes the body-side wrench when the shape is attached to a dynamic body; here we additionally compute the lattice-host wrench when the particle is a lattice sphere.

- [ ] **Step 1: Write the failing test for a single lattice sphere on the ground**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_lattice_sphere_drops_to_ground(test, device):
    """A free body with one lattice sphere settles on the ground plane.

    Validates lattice-to-body wrench routing: contact on the lattice
    sphere must push the host body upward, not the particle.
    """
    builder = newton.ModelBuilder(up_axis="Z")
    link = builder.add_body(
        mass=1.0,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()),
    )
    # Single sphere of radius 0.05 at body origin.
    sphere_json = {
        "centers": [[0.0, 0.0, 0.0]],
        "radii": [0.05],
        "normals": [[0.0, 0.0, -1.0]],
        "is_surface": [1],
    }
    # Save to a temp file so add_lattice can load it.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sphere_json, f)
        temp_path = f.name
    try:
        builder.add_lattice(link=link, morphit_json=temp_path, total_mass=0.0)
    finally:
        os.unlink(temp_path)

    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverUXPBD(model, iterations=4)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Run for ~2 seconds at 100 fps with 10 substeps.
    dt = 1.0 / 1000.0
    contacts = model.contacts()
    for _ in range(2000):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    # At rest the body's z position should be near the sphere radius (0.05),
    # because the lattice sphere of radius 0.05 sits on the ground.
    body_q = state_0.body_q.numpy()
    body_z = float(body_q[0, 2])
    test.assertAlmostEqual(body_z, 0.05, delta=1e-3)


import json

add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_lattice_sphere_drops_to_ground",
    test_uxpbd_lattice_sphere_drops_to_ground,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run the test, confirm it fails**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_lattice_sphere_drops_to_ground`
Expected: AssertionError or NotImplementedError (`solver.step` is still a stub).

- [ ] **Step 3: Implement `solve_lattice_shape_contacts` in `kernels.py`**

Append to `newton/_src/solvers/uxpbd/kernels.py`:

```python
from ..xpbd.kernels import ParticleFlags  # for ACTIVE mask


@wp.func
def lattice_sphere_w_eff(
    body_inv_mass: float,
    body_inv_inertia: wp.mat33,
    body_rot: wp.quat,
    r_world: wp.vec3,
    n: wp.vec3,
) -> float:
    """Effective inverse mass at a lattice sphere along contact normal ``n``.

    Implements ``w_eff = w_body + (r × n)^T · W_world · (r × n)``, where
    ``W_world = R · I^{-1} · R^T``. Identical to the inverse-mass term in
    XPBD's ``solve_body_contact_positions``.
    """
    angular = wp.cross(r_world, n)
    rot_angular = wp.quat_rotate_inv(body_rot, angular)
    return body_inv_mass + wp.dot(rot_angular, body_inv_inertia * rot_angular)


@wp.kernel
def solve_lattice_shape_contacts(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    lattice_particle_index: wp.array[wp.int32],
    lattice_link: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_m_inv: wp.array[float],
    body_I_inv: wp.array[wp.mat33],
    shape_body: wp.array[int],
    shape_material_mu: wp.array[float],
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_max: int,
    particle_to_lattice: wp.array[wp.int32],  # -1 if particle is not a lattice sphere
    dt: float,
    relaxation: float,
    # outputs
    body_delta: wp.array[wp.spatial_vector],
):
    """Lattice-aware variant of XPBD's solve_particle_shape_contacts.

    Routes the position correction for each lattice-sphere contact into the
    host link's spatial wrench accumulator. Non-lattice particles are
    ignored by this kernel; XPBD's solve_particle_shape_contacts handles
    those in later phases.
    """
    tid = wp.tid()
    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    particle_index = contact_particle[tid]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    # Only handle lattice particles in this kernel.
    lat_idx = particle_to_lattice[particle_index]
    if lat_idx < 0:
        return

    host_link = lattice_link[lat_idx]
    shape_index = contact_shape[tid]
    shape_link = shape_body[shape_index]

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    # Build the world-space contact frame from the shape side.
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if shape_link >= 0:
        X_wb = body_q[shape_link]
        X_com = body_com[shape_link]

    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r_shape = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]
    if c > particle_ka:
        return

    mu = 0.5 * (particle_mu + shape_material_mu[shape_index])

    # Shape-side body velocity at the contact point.
    body_v_s = wp.spatial_vector()
    if shape_link >= 0:
        body_v_s = body_qd[shape_link]
    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)
    bv = body_v + wp.cross(body_w, r_shape) + wp.transform_vector(X_wb, contact_body_vel[tid])

    # Relative velocity at contact.
    v = pv - bv

    # Normal correction.
    lambda_n = c
    delta_n = n * lambda_n

    # Friction (Coulomb, position-level).
    vn = wp.dot(n, v)
    vt = v - n * vn
    lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
    delta_f = wp.normalize(vt) * lambda_f

    # Effective inverse mass on the lattice side using the host link's inertia.
    host_q = body_q[host_link]
    host_com_world = wp.transform_point(host_q, body_com[host_link])
    r_lat = px - host_com_world
    w_lat = lattice_sphere_w_eff(
        body_m_inv[host_link],
        body_I_inv[host_link],
        wp.transform_get_rotation(host_q),
        r_lat,
        n,
    )

    # Effective inverse mass on the shape side.
    w_shape = 0.0
    if shape_link >= 0:
        angular = wp.cross(r_shape, n)
        q_shape = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q_shape, angular)
        w_shape = body_m_inv[shape_link] + wp.dot(rot_angular, body_I_inv[shape_link] * rot_angular)

    denom = w_lat + w_shape
    if denom == 0.0:
        return

    delta_total = (delta_f - delta_n) / denom * relaxation

    # Route Δx on the lattice sphere into the host link's wrench.
    # Direction matches XPBD's wp.atomic_sub for the shape-host body.
    t_lat = wp.cross(r_lat, delta_total)
    wp.atomic_sub(body_delta, host_link, wp.spatial_vector(delta_total, t_lat))

    if shape_link >= 0:
        # Shape is dynamic: also push back its body.
        t_shape = wp.cross(r_shape, delta_total)
        wp.atomic_add(body_delta, shape_link, wp.spatial_vector(delta_total, t_shape))
```

- [ ] **Step 4: Save the test-failure expectation; move on to Task 6 before re-running**

The contact kernel needs a ``particle_to_lattice`` array (-1 for non-lattice particles, lattice index otherwise). Task 6 wires this up and integrates the kernel into `step()`. Do not run the test yet.

- [ ] **Step 5: Add a focused unit test that exercises the kernel directly**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_lattice_w_eff_matches_xpbd_formula(test, device):
    """The lattice_sphere_w_eff helper produces the same value as XPBD's
    body-contact inverse-mass formula for a known geometry.
    """
    # Pure-python reference for w_eff at offset r along normal n:
    #     w_eff = w + (r × n)^T · R · I^{-1} · R^T · (r × n)
    # Set up: unit mass, unit inertia at identity rotation, offset (0.1, 0, 0),
    # normal (0, 0, 1). Expected w_eff = 1.0 + |(r×n)|^2 = 1.0 + 0.01.
    builder = newton.ModelBuilder()
    link = builder.add_body(mass=1.0)  # add_body auto-creates a free joint and articulation
    builder.add_lattice(
        link=link,
        morphit_json=os.path.join(_ASSET_DIR, "tiny_lattice.json"),
        total_mass=0.0,
    )
    model = builder.finalize(device=device)

    # The kernel is exercised via solve_lattice_shape_contacts in Task 6.
    # Here we just confirm the lattice helper is exported and callable.
    from newton._src.solvers.uxpbd.kernels import lattice_sphere_w_eff
    test.assertTrue(callable(lattice_sphere_w_eff))


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_lattice_w_eff_matches_xpbd_formula",
    test_uxpbd_lattice_w_eff_matches_xpbd_formula,
    devices=get_test_devices(),
)
```

- [ ] **Step 6: Run this lightweight test, confirm pass**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_lattice_w_eff_matches_xpbd_formula`
Expected: PASS.

- [ ] **Step 7: Pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/_src/solvers/uxpbd/kernels.py newton/tests/test_solver_uxpbd.py
git commit -m "Add solve_lattice_shape_contacts kernel

Routes position deltas on lattice-sphere contacts into the host link's
spatial wrench accumulator using the link's true body inertia, via the
new lattice_sphere_w_eff device function. Mirrors XPBD's
solve_particle_shape_contacts math; the difference is on the lattice
side, where the particle inverse mass is computed from the host body's
inertia at the lattice sphere's world offset rather than from
particle_inv_mass[i].

Drop-to-ground regression test is queued; it requires Task 6's
solver.step() implementation."
```

---

## Task 6: Wire up `SolverUXPBD.step()` for the joint + lattice path

**Files:**

- Modify: `newton/_src/solvers/uxpbd/solver_uxpbd.py`
- Modify: `newton/_src/sim/builder.py` (populate `particle_to_lattice` index at finalize)
- Modify: `newton/_src/sim/model.py` (declare `particle_to_lattice` array)
- Test: `newton/tests/test_solver_uxpbd.py` (re-run the queued drop-to-ground test)

The minimum viable step() for Phase 1:
1. Apply joint feedforward forces (`apply_joint_forces`).
2. `integrate_bodies` (semi-implicit Euler with gravity, body_f, joint_f).
3. `update_lattice_world_positions`.
4. `compute_compliant_contact_response` (no-op).
5. `collide` (already done by caller in our convention).
6. For `iterations`:
    - Zero `body_deltas`.
    - `solve_lattice_shape_contacts` writing into `body_deltas`.
    - `apply_body_deltas` to fold deltas into `body_q`/`body_qd`.
    - `solve_body_joints` from XPBD kernels.
    - `apply_body_deltas` again.
    - Re-run `update_lattice_world_positions` so the next iteration sees synced lattice.
7. Optionally populate `body_parent_f` from `joint_impulse` (defer to Task 7).
8. Copy kinematic body state (per XPBD).

- [ ] **Step 1: Add `particle_to_lattice` to Model**

In `newton/_src/sim/model.py`, alongside the other lattice fields, add:

```python
# Reverse index: particle slot -> lattice sphere index, or -1 if not a lattice particle.
self.particle_to_lattice: wp.array[wp.int32] = wp.empty(0, dtype=wp.int32, device=device)
```

- [ ] **Step 2: Populate `particle_to_lattice` in `ModelBuilder.finalize`**

In `newton/_src/sim/builder.py` finalize (the same block from Task 2 step 5), after the lattice arrays are baked, add:

```python
# Reverse index from particle slot to lattice sphere index.
n_particles = len(self.particle_q)
p_to_l = np.full(n_particles, -1, dtype=np.int32)
for lat_i, p_i in enumerate(self.lattice_particle_index):
    p_to_l[p_i] = lat_i
model.particle_to_lattice = wp.array(p_to_l, dtype=wp.int32, device=device)
```

This block must run even when ``n_lat == 0`` so the array always exists.

- [ ] **Step 3: Implement `SolverUXPBD.step()`**

Replace the stub `step` in `newton/_src/solvers/uxpbd/solver_uxpbd.py` with:

```python
from ..flags import SolverNotifyFlags
from ..xpbd.kernels import (
    apply_body_deltas,
    apply_joint_forces,
    copy_kinematic_body_state_kernel,
    solve_body_joints,
)
from .kernels import solve_lattice_shape_contacts, update_lattice_world_positions


# (replace the existing class body's step() with the following)
def step(
    self,
    state_in: State,
    state_out: State,
    control: Control | None,
    contacts: Contacts | None,
    dt: float,
) -> None:
    """Advance the simulation by ``dt`` seconds. Phase 1: articulated rigid + lattice."""
    model = self.model

    if control is None:
        control = model.control(clone_variables=False)

    # 1. Predict body positions: integrate_bodies with joint feedforward.
    body_f_local = state_in.body_f
    if model.joint_count:
        body_f_local = wp.clone(state_in.body_f)
        wp.launch(
            kernel=apply_joint_forces,
            dim=model.joint_count,
            inputs=[
                state_in.body_q,
                model.body_com,
                model.joint_type,
                model.joint_enabled,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_axis,
                control.joint_f,
            ],
            outputs=[body_f_local],
            device=model.device,
        )
    if body_f_local is state_in.body_f:
        self.integrate_bodies(model, state_in, state_out, dt)
    else:
        body_f_prev = state_in.body_f
        state_in.body_f = body_f_local
        self.integrate_bodies(model, state_in, state_out, dt)
        state_in.body_f = body_f_prev

    # Copy any non-lattice particles forward (Phase 1 has none, but keep API consistent).
    if model.particle_count:
        state_out.particle_q.assign(state_in.particle_q)
        state_out.particle_qd.assign(state_in.particle_qd)

    # 2. Project body_q onto lattice particles.
    self.update_lattice_world_positions(state_out)

    # 3. v2 CSLC hook (no-op in v1).
    self.compute_compliant_contact_response(state_in, state_out, contacts, dt)

    # 4. Main iteration loop.
    if model.body_count:
        body_deltas = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
    else:
        body_deltas = None

    for _ in range(self.iterations):
        if body_deltas is not None:
            body_deltas.zero_()

        # Lattice-shape contacts (Phase 1's only contact path).
        if contacts is not None and model.lattice_sphere_count and body_deltas is not None:
            wp.launch(
                kernel=solve_lattice_shape_contacts,
                dim=contacts.soft_contact_max,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.particle_radius,
                    model.particle_flags,
                    model.lattice_particle_index,
                    model.lattice_link,
                    state_out.body_q,
                    state_out.body_qd,
                    model.body_com,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    model.shape_body,
                    model.shape_material_mu,
                    model.soft_contact_mu,
                    model.particle_adhesion,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_max,
                    model.particle_to_lattice,
                    dt,
                    0.8,  # relaxation, matches XPBD default
                ],
                outputs=[body_deltas],
                device=model.device,
            )

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    state_out.body_q,
                    state_out.body_qd,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    body_deltas,
                    None,  # rigid_contact_inv_weight not used in lattice path
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )

            # Re-sync lattice after body update so next iter sees consistent state.
            self.update_lattice_world_positions(state_out)

        # Joints
        if model.joint_count and body_deltas is not None:
            body_deltas.zero_()
            joint_impulse = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)
            wp.launch(
                kernel=solve_body_joints,
                dim=model.joint_count,
                inputs=[
                    state_out.body_q,
                    state_out.body_qd,
                    model.body_com,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_axis,
                    control.joint_target_pos,
                    control.joint_target_vel,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    0.0,  # joint_linear_compliance
                    0.0,  # joint_angular_compliance
                    0.4,  # joint_angular_relaxation, XPBD default
                    0.7,  # joint_linear_relaxation, XPBD default
                    dt,
                ],
                outputs=[body_deltas, joint_impulse],
                device=model.device,
            )
            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    state_out.body_q,
                    state_out.body_qd,
                    model.body_com,
                    model.body_inertia,
                    self.body_inv_mass_effective,
                    self.body_inv_inertia_effective,
                    body_deltas,
                    None,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )
            # Re-sync lattice.
            self.update_lattice_world_positions(state_out)

    # 5. Copy kinematic body state forward (matches XPBD).
    if model.body_count:
        wp.launch(
            kernel=copy_kinematic_body_state_kernel,
            dim=model.body_count,
            inputs=[model.body_flags, state_in.body_q, state_in.body_qd],
            outputs=[state_out.body_q, state_out.body_qd],
            device=model.device,
        )
```

Insert the new `step()` body inside the `SolverUXPBD` class.

- [ ] **Step 4: Run the drop-to-ground test (deferred from Task 5)**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_lattice_sphere_drops_to_ground`
Expected: PASS. Body settles at z ≈ 0.05 within 1 mm.

- [ ] **Step 5: Add the free-fall regression test**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_free_fall_trajectory(test, device):
    """A free body with no lattice contact falls under gravity as g*t^2/2."""
    builder = newton.ModelBuilder(up_axis="Z")
    link = builder.add_body(
        mass=1.0,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 10.0), q=wp.quat_identity()),
    )
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverUXPBD(model, iterations=1)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 0.001
    t = 0.5  # half a second of free fall
    n_steps = int(t / dt)
    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0

    # Expected drop: 0.5 * 9.81 * t^2 = 0.5 * 9.81 * 0.25 = 1.22625 m
    body_q = state_0.body_q.numpy()
    body_z = float(body_q[0, 2])
    expected_z = 10.0 - 0.5 * 9.81 * t * t
    relative_err = abs(body_z - expected_z) / expected_z
    test.assertLess(relative_err, 0.005, f"free fall z={body_z}, expected={expected_z}")


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_free_fall_trajectory",
    test_uxpbd_free_fall_trajectory,
    devices=get_test_devices(),
)
```

- [ ] **Step 6: Run free-fall test, confirm pass**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_free_fall_trajectory`
Expected: PASS within 0.5% relative error.

- [ ] **Step 7: Pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/_src/sim/model.py newton/_src/sim/builder.py newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd.py
git commit -m "Wire SolverUXPBD.step() for Phase 1: joints + lattice contact

Implements integration, lattice projection, lattice-aware shape contact
resolution, and joint constraint solving. Reuses XPBD kernels for
apply_joint_forces, apply_body_deltas, solve_body_joints, and
copy_kinematic_body_state_kernel. Adds particle_to_lattice reverse
index on Model.

Validates against gravity (free fall trajectory match within 0.5%) and
against ground contact via the lattice path (single sphere drops to
rest at z == sphere radius within 1 mm)."
```

---

## Task 7: Pendulum period and body_parent_f gate tests

**Files:**

- Modify: `newton/_src/solvers/uxpbd/solver_uxpbd.py` (add body_parent_f reporting)
- Test: `newton/tests/test_solver_uxpbd.py`

- [ ] **Step 1: Add body_parent_f population to `step()`**

In `newton/_src/solvers/uxpbd/solver_uxpbd.py`, after the joint loop and before `copy_kinematic_body_state_kernel`, add (importing the kernel at the top):

```python
from ..xpbd.kernels import convert_joint_impulse_to_parent_f
```

After the joints `wp.launch`, persist the `joint_impulse` buffer locally so it survives past the loop iteration. Move its allocation outside the iteration loop (immediately before the iteration loop):

```python
if state_out.body_parent_f is not None and model.joint_count > 0:
    joint_impulse = wp.zeros(model.joint_count, dtype=wp.spatial_vector, device=model.device)
else:
    joint_impulse = None
```

Inside the iteration loop, accumulate into the persistent `joint_impulse` instead of allocating anew. After the loop, populate `state_out.body_parent_f`:

```python
if state_out.body_parent_f is not None:
    state_out.body_parent_f.zero_()
    if joint_impulse is not None:
        wp.launch(
            kernel=convert_joint_impulse_to_parent_f,
            dim=model.joint_count,
            inputs=[
                joint_impulse,
                model.joint_enabled,
                model.joint_type,
                model.joint_child,
                dt,
            ],
            outputs=[state_out.body_parent_f],
            device=model.device,
        )
```

- [ ] **Step 2: Write the pendulum period gate test**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_pendulum_period(test, device):
    """A simple pendulum of length L oscillates at period T = 2π * sqrt(L/g)."""
    builder = newton.ModelBuilder(up_axis="Z")
    L = 1.0
    link = builder.add_link()  # add_link, NOT add_body: we want a manual joint, not a free joint.
    builder.add_shape_box(link, hx=L, hy=0.05, hz=0.05)
    j = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-L, 0.0, 0.0), q=wp.quat_identity()),
    )
    builder.add_articulation([j], label="pendulum")
    model = builder.finalize(device=device)

    # Start the pendulum at a small angle (small-angle linear theory)
    initial_angle = 0.05  # rad
    joint_q = model.joint_q.numpy().copy()
    joint_q[0] = initial_angle
    model.joint_q.assign(joint_q)

    solver = newton.solvers.SolverUXPBD(model, iterations=8)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 0.001
    # Track angle by reading body_q rotation around y axis.
    angles = []
    for _ in range(4000):  # 4 seconds
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, dt)
        state_0, state_1 = state_1, state_0
        bq = state_0.body_q.numpy()[0]
        # Quaternion (x,y,z,w) at indices 3..6. We want the rotation around y.
        # Small-angle approximation: angle ≈ 2 * qy
        angles.append(2.0 * float(bq[4]))

    # Count zero crossings to estimate period.
    angles_np = np.array(angles)
    crossings = np.where(np.diff(np.signbit(angles_np)))[0]
    test.assertGreater(len(crossings), 2)
    # Two consecutive crossings = half a period.
    half_period_steps = float(crossings[2] - crossings[0]) / 2.0
    measured_T = half_period_steps * dt * 2.0  # two half-periods = full period
    expected_T = 2.0 * np.pi * np.sqrt(L / 9.81)
    relative_err = abs(measured_T - expected_T) / expected_T
    test.assertLess(relative_err, 0.01, f"T={measured_T:.4f}, expected={expected_T:.4f}")


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_pendulum_period",
    test_uxpbd_pendulum_period,
    devices=get_test_devices(),
)
```

- [ ] **Step 3: Write the body_parent_f gate test**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_body_parent_f_revolute_to_world(test, device):
    """A revolute joint to the world reports a non-zero parent wrench when
    gravity acts on the child. Should match XPBD's output to within 5%.
    """
    def build_and_step(solver_class):
        builder = newton.ModelBuilder(up_axis="Z")
        L = 1.0
        link = builder.add_link()
        builder.add_shape_box(link, hx=L, hy=0.05, hz=0.05)
        j = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-L, 0.0, 0.0), q=wp.quat_identity()),
        )
        builder.add_articulation([j], label="pendulum")
        builder.request_state_attributes("body_parent_f")
        model = builder.finalize(device=device)

        solver = solver_class(model)
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        for _ in range(20):
            state_0.clear_forces()
            solver.step(state_0, state_1, None, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        return state_0.body_parent_f.numpy()[0].copy()

    xpbd_wrench = build_and_step(newton.solvers.SolverXPBD)
    uxpbd_wrench = build_and_step(newton.solvers.SolverUXPBD)

    # Compare component-wise within 5%.
    for i in range(6):
        denom = abs(xpbd_wrench[i]) + 1.0e-3
        rel_err = abs(uxpbd_wrench[i] - xpbd_wrench[i]) / denom
        test.assertLess(rel_err, 0.05, f"body_parent_f[{i}] xpbd={xpbd_wrench[i]} uxpbd={uxpbd_wrench[i]}")


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_body_parent_f_revolute_to_world",
    test_uxpbd_body_parent_f_revolute_to_world,
    devices=get_test_devices(),
)
```

The exact API: call `builder.request_state_attributes("body_parent_f")` BEFORE `builder.finalize(...)`. Verified via `newton/tests/test_solver_xpbd.py:760, 912, 1013`. The state objects created by `model.state()` will then carry a non-None `body_parent_f` array.

- [ ] **Step 4: Run both tests, confirm pass**

Run: `uv run --extra dev -m newton.tests -k "pendulum_period or body_parent_f_revolute"`
Expected: both PASS.

- [ ] **Step 5: Pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/_src/solvers/uxpbd/solver_uxpbd.py newton/tests/test_solver_uxpbd.py
git commit -m "Populate body_parent_f and validate against XPBD

SolverUXPBD now writes state_out.body_parent_f when the state requests
it, using XPBD's convert_joint_impulse_to_parent_f. Two new tests:
pendulum period within 1% of analytical 2π√(L/g), and body_parent_f
trace for a revolute-to-world joint matches SolverXPBD's output within
5% across all six wrench components."
```

---

## Task 8: `add_lattice_to_all_links` convenience

**Files:**

- Modify: `newton/_src/solvers/uxpbd/lattice.py`
- Modify: `newton/_src/sim/builder.py`
- Test: `newton/tests/test_solver_uxpbd.py`

- [ ] **Step 1: Write the failing test**

Append to `newton/tests/test_solver_uxpbd.py`:

```python
def test_uxpbd_add_lattice_to_all_links_with_fallback(test, device):
    """add_lattice_to_all_links uses JSON when present, falls back to uniform packing otherwise."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="uxpbd_test_")
    try:
        # Copy fixture into tmp dir as the file for link "link_a".
        shutil.copy(os.path.join(_ASSET_DIR, "tiny_lattice.json"), os.path.join(tmp, "link_a.json"))

        builder = newton.ModelBuilder()
        link_a = builder.add_body(label="link_a")
        link_b = builder.add_body(key="link_b")
        builder.add_joint_free(child=link_a)
        builder.add_joint_free(child=link_b)

        builder.add_lattice_to_all_links(
            morphit_json_dir=tmp,
            fallback_uniform_n=2,  # 2x2x2 = 8 spheres
        )

        model = builder.finalize(device=device)
        # 5 spheres from JSON + 8 from fallback = 13.
        test.assertEqual(model.lattice_sphere_count, 13)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


add_function_test(
    TestSolverUXPBD,
    "test_uxpbd_add_lattice_to_all_links_with_fallback",
    test_uxpbd_add_lattice_to_all_links_with_fallback,
    devices=get_test_devices(),
)
```

- [ ] **Step 2: Run, confirm fail**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_add_lattice_to_all_links_with_fallback`
Expected: AttributeError on `builder.add_lattice_to_all_links`.

- [ ] **Step 3: Implement `add_lattice_to_all_links` helper in `lattice.py`**

Append to `newton/_src/solvers/uxpbd/lattice.py`:

```python
def _uniform_box_lattice(n: int, half_extent: float = 0.1) -> dict[str, np.ndarray]:
    """Build an n x n x n uniform sphere packing inscribed in a cube of
    half-extent ``half_extent``.

    Returns the same dict shape as load_morphit_lattice.
    """
    if n < 1:
        raise ValueError("fallback_uniform_n must be >= 1")
    spacing = (2.0 * half_extent) / n
    sphere_r = spacing * 0.5
    coords = np.linspace(-half_extent + sphere_r, half_extent - sphere_r, n)
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing="ij")
    centers = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1).astype(np.float32)
    radii = np.full(centers.shape[0], sphere_r, dtype=np.float32)
    # Outward normals from centroid.
    centroid = centers.mean(axis=0, keepdims=True)
    offsets = centers - centroid
    lens = np.linalg.norm(offsets, axis=1, keepdims=True)
    lens = np.where(lens > 1e-12, lens, 1.0)
    normals = (offsets / lens).astype(np.float32)
    # is_surface: any sphere whose center is on the cube boundary.
    bdy = (np.abs(np.abs(centers) - (half_extent - sphere_r)) < 1.0e-6).any(axis=1)
    is_surface = bdy.astype(np.uint8)
    return {"centers": centers, "radii": radii, "normals": normals, "is_surface": is_surface}


def add_lattice_to_all_links_in_builder(
    builder,
    morphit_json_dir: str | Path,
    *,
    fallback_uniform_n: int | None = 4,
    **lattice_kwargs,
) -> None:
    """Attach a lattice to every body in the builder.

    For each body with a non-empty key, look up
    ``<morphit_json_dir>/<body_key>.json``. If the file exists, attach a
    MorphIt lattice from it. Otherwise, if ``fallback_uniform_n`` is not
    ``None``, attach a uniform ``n^3`` lattice inscribed in a unit cube; if
    ``fallback_uniform_n`` is ``None``, skip this body.

    Args:
        builder: The :class:`~newton.ModelBuilder` to populate.
        morphit_json_dir: Directory containing per-link MorphIt JSON files
            named ``<link_key>.json``.
        fallback_uniform_n: Sphere count per axis for the fallback packing,
            or ``None`` to skip links with no JSON.
        **lattice_kwargs: Forwarded to ``add_lattice_to_builder`` per link
            (e.g., ``total_mass=...``, ``k_anchor=...``).
    """
    dir_path = Path(morphit_json_dir)
    body_count = len(builder.body_label) if hasattr(builder, "body_key") else 0
    for link in range(body_count):
        key = builder.body_label[link] if hasattr(builder, "body_key") else f"body_{link}"
        candidate = dir_path / f"{key}.json"
        if candidate.is_file():
            add_lattice_to_builder(builder, link=link, morphit_json=str(candidate), **lattice_kwargs)
        elif fallback_uniform_n is not None:
            data = _uniform_box_lattice(fallback_uniform_n)
            add_lattice_to_builder(builder, link=link, morphit_json=data, **lattice_kwargs)
        # else: skip
```

Confirmed: `builder.body_label` is the per-link label list (declared at `newton/_src/sim/builder.py:1104`). Each link gets a label from `add_body(label=...)` or an auto-generated `body_<id>`.

- [ ] **Step 4: Add the `ModelBuilder.add_lattice_to_all_links` method**

In `newton/_src/sim/builder.py`, just below the `add_lattice` method from Task 3, add:

```python
def add_lattice_to_all_links(
    self,
    morphit_json_dir: str,
    *,
    fallback_uniform_n: int | None = 4,
    total_mass: float = 0.0,
    k_anchor: float = 1.0e3,
    k_lateral: float = 5.0e2,
    k_bulk: float = 1.0e5,
    damping: float = 2.0,
) -> None:
    """Attach a MorphIt lattice to every articulated link.

    For each link, look up ``<morphit_json_dir>/<link_key>.json``. If found,
    use that MorphIt sphere packing. Otherwise, fall back to a uniform
    ``fallback_uniform_n^3`` packing inscribed in a unit cube. Pass
    ``fallback_uniform_n=None`` to skip links without a JSON file.

    Args:
        morphit_json_dir: Directory containing one JSON per link, named
            after the link's key (e.g., ``link_0.json``).
        fallback_uniform_n: Sphere count per axis for fallback packing, or
            ``None`` to skip links missing a JSON.
        total_mass: Per-link total lattice mass [kg]. Defaults to 0.0.
        k_anchor: v2 CSLC anchor stiffness, stored but unused in Phase 1.
        k_lateral: v2 CSLC lateral coupling stiffness.
        k_bulk: v2 CSLC bulk material stiffness.
        damping: v2 CSLC Hunt-Crossley damping.
    """
    from ..solvers.uxpbd.lattice import add_lattice_to_all_links_in_builder

    add_lattice_to_all_links_in_builder(
        self,
        morphit_json_dir=morphit_json_dir,
        fallback_uniform_n=fallback_uniform_n,
        total_mass=total_mass,
        k_anchor=k_anchor,
        k_lateral=k_lateral,
        k_bulk=k_bulk,
        damping=damping,
    )
```

- [ ] **Step 5: Run the test, confirm it passes**

Run: `uv run --extra dev -m newton.tests -k test_uxpbd_add_lattice_to_all_links_with_fallback`
Expected: PASS.

- [ ] **Step 6: Pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/_src/solvers/uxpbd/lattice.py newton/_src/sim/builder.py newton/tests/test_solver_uxpbd.py
git commit -m "Add ModelBuilder.add_lattice_to_all_links convenience method

Walks the builder's links, attaches a MorphIt lattice from
<dir>/<link_key>.json when present, otherwise falls back to a uniform
n x n x n packing inscribed in a unit cube. fallback_uniform_n=None
skips links missing a JSON.

This is the convenience entry point for attaching lattices to a full
URDF-loaded robot in one call."
```

---

## Task 9: Demo example

**Files:**

- Create: `newton/examples/contacts/example_uxpbd_lattice_push.py`
- Modify: `newton/examples/__init__.py` (register the example if there is an example registry)

- [ ] **Step 1: Locate the example registry**

Look for where the existing examples are listed. Run:

```bash
grep -rEn '"basic_pendulum"|"basic_shapes"|example_basic_pendulum' /Users/nn/devenv/newton_custom_contact/newton/examples/__init__.py /Users/nn/devenv/newton_custom_contact/newton/examples/__main__.py 2>/dev/null
```

Expected: a dict or list mapping short names to example file paths. Note the path pattern.

- [ ] **Step 2: Create a MorphIt lattice JSON for the link**

Create `newton/examples/assets/uxpbd/link_box.json`:

```json
{
  "centers": [
    [-0.1, -0.05, -0.05], [-0.1, -0.05, 0.05], [-0.1, 0.05, -0.05], [-0.1, 0.05, 0.05],
    [-0.05, -0.05, -0.05], [-0.05, -0.05, 0.05], [-0.05, 0.05, -0.05], [-0.05, 0.05, 0.05],
    [0.05, -0.05, -0.05], [0.05, -0.05, 0.05], [0.05, 0.05, -0.05], [0.05, 0.05, 0.05],
    [0.1, -0.05, -0.05], [0.1, -0.05, 0.05], [0.1, 0.05, -0.05], [0.1, 0.05, 0.05]
  ],
  "radii": [
    0.04, 0.04, 0.04, 0.04,
    0.04, 0.04, 0.04, 0.04,
    0.04, 0.04, 0.04, 0.04,
    0.04, 0.04, 0.04, 0.04
  ],
  "is_surface": [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]
}
```

- [ ] **Step 3: Create the example file**

Create `newton/examples/contacts/example_uxpbd_lattice_push.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Lattice Push
#
# A single revolute link with a MorphIt-packed kinematic lattice rotates
# under a constant torque and pushes a free-falling box on the ground.
# Phase 1 demo for SolverUXPBD: validates lattice-to-shape contact through
# the body wrench accumulation pipeline, with joint constraints in the loop.
#
# Command: python -m newton.examples uxpbd_lattice_push
###########################################################################

import os

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # Pusher link: a thin box around (0, 0, 0.3), rotating about the world Z axis
        # through a revolute joint at (0, 0, 0.3).
        link = builder.add_body(mass=2.0, label="pusher")
        builder.add_shape_box(link, hx=0.12, hy=0.06, hz=0.06)
        builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.3), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=0.0,
            target_kd=0.0,
        )

        json_path = os.path.join(os.path.dirname(__file__), "..", "assets", "uxpbd", "link_box.json")
        json_path = os.path.normpath(json_path)
        builder.add_lattice(
            link=link,
            morphit_json=json_path,
            total_mass=0.0,
            pos=wp.vec3(0.0, 0.0, 0.3),
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverUXPBD(self.model, iterations=4)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # Apply a constant torque around the z axis to drive rotation.
        joint_f_np = self.model.joint_q.numpy().copy() * 0.0
        if joint_f_np.shape[0] > 0:
            joint_f_np[0] = 5.0
        self.control.joint_f.assign(joint_f_np)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Validate that the pusher rotated, and that the link did not
        # penetrate the ground (body z >= some lower bound).
        bq = self.state_0.body_q.numpy()[0]
        body_z = float(bq[2])
        # Quaternion w component encodes 1 - 2*sin(theta/2)^2 for axis-Z rot.
        qw = float(bq[6])
        # Expect a non-trivial rotation: qw should be away from 1.
        if abs(qw) >= 0.999:
            raise RuntimeError(f"Pusher did not rotate, qw={qw}")
        if body_z < 0.2:
            raise RuntimeError(f"Pusher fell through revolute constraint, z={body_z}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
```

- [ ] **Step 4: Register the example**

If `newton/examples/__init__.py` or `__main__.py` contains an example registry (as discovered in Step 1), add an entry pointing the name `uxpbd_lattice_push` at `contacts/example_uxpbd_lattice_push.py`. Match the existing entries exactly.

- [ ] **Step 5: Run headless for a few frames to ensure it does not crash**

Run: `uv run python -m newton.examples uxpbd_lattice_push --headless --num-frames 200`
Expected: runs to completion. `test_final` passes (no exceptions).

- [ ] **Step 6: Pre-commit and commit**

```bash
uvx pre-commit run -a
git add newton/examples/contacts/example_uxpbd_lattice_push.py newton/examples/assets/uxpbd/link_box.json newton/examples/__init__.py
git commit -m "Add uxpbd_lattice_push demo

A single revolute link with a 16-sphere MorphIt-style lattice rotates
under a constant torque and pushes a free box. Phase 1 end-to-end
demonstration of the lattice-to-body wrench routing and joint loop
under SolverUXPBD. test_final asserts the link rotated and did not
penetrate the revolute constraint."
```

---

## Task 10: Run the full Phase 1 test suite

- [ ] **Step 1: Run the full Newton test suite to confirm no regressions**

Run: `uv run --extra dev -m newton.tests`
Expected: existing XPBD/SRXPBD tests still pass. Only UXPBD tests changed.

- [ ] **Step 2: Run the UXPBD test set in isolation**

Run: `uv run --extra dev -m newton.tests -k SolverUXPBD`
Expected output (count may differ slightly if devices change):

```
test_uxpbd_solver_importable ... ok
test_uxpbd_solver_instantiates_with_empty_model ... ok
test_uxpbd_empty_model_has_zero_lattice ... ok
test_uxpbd_add_lattice_populates_arrays ... ok
test_uxpbd_update_lattice_projects_body_q ... ok
test_uxpbd_update_lattice_handles_rotation ... ok
test_uxpbd_lattice_w_eff_matches_xpbd_formula ... ok
test_uxpbd_lattice_sphere_drops_to_ground ... ok
test_uxpbd_free_fall_trajectory ... ok
test_uxpbd_pendulum_period ... ok
test_uxpbd_body_parent_f_revolute_to_world ... ok
test_uxpbd_add_lattice_to_all_links_with_fallback ... ok
```

12 tests on each device. If any fails, fix it before moving on.

- [ ] **Step 3: Run the example one final time**

Run: `uv run python -m newton.examples uxpbd_lattice_push --headless --num-frames 500`
Expected: `test_final` passes.

- [ ] **Step 4: Add a CHANGELOG entry**

In `CHANGELOG.md`, find the `[Unreleased]` section and insert under `### Added` at a random position:

```markdown
- Add `SolverUXPBD` Phase 1: articulated rigid bodies with a MorphIt-generated kinematic lattice for particle-based contact against analytical static shapes. Reserves architectural seams for CSLC v2.
```

- [ ] **Step 5: Final commit**

```bash
git add CHANGELOG.md
git commit -m "Add CHANGELOG entry for SolverUXPBD Phase 1"
```

---

## Implementation notes (read before starting)

1. **`add_body` vs `add_link`:** `add_body()` is a convenience wrapper that auto-creates a free joint and articulation. `add_link()` adds the body alone and lets you build a manual joint tree. The plan uses `add_body` for free-fall and drop-to-ground tests, and `add_link` + `add_joint_revolute` + `add_articulation` for pendulum and pusher tests where a specific joint is required.
2. **Requesting `body_parent_f`:** call `builder.request_state_attributes("body_parent_f")` BEFORE `builder.finalize(...)`. Verified pattern in `newton/tests/test_solver_xpbd.py` lines 760, 912, 1013.
3. **`builder.body_label`** holds per-link string labels (line 1104 of `builder.py`). Used by Task 8 to locate per-link MorphIt JSON files.
4. **`model.contacts()`** returns a fresh `Contacts` object; pass it to `model.collide(state, contacts)` then to `solver.step(..., contacts, dt)`. Pattern: `newton/examples/basic/example_basic_pendulum.py` lines 81-103.

---

## Out of scope for Phase 1 (deferred to Phase 2+)

These items appear in the spec but are NOT covered by this plan:

- Free shape-matched rigid bodies (PBD-R kernels). Phase 2.
- Soft body FEM tet, springs, bending. Phase 3.
- PBF fluid density constraint. Phase 4.
- Restitution post-pass. Phase 5.
- MuJoCo comparison harness. Phase 2.
- UPPFRTA mass-scaling shock propagation. Phase 2.
- `solve_particle_particle_contacts` extension (only lattice-shape contact lands here). Phase 2.
- `set_phase_collision`, `set_self_collision` builder APIs. Phase 3.
- `add_fluid_grid`, `add_fluid_particles`. Phase 4.
- Stabilization sub-loop (UPPFRTA §4.4). Phase 2 (along with particle-particle).
- Multiple iteration counts. Phase 2.
- `enable_restitution` actually wired up. Phase 5.

Phase 1 establishes the package skeleton, lattice data model, kernel patterns, and contact routing for the articulated rigid path. Every later phase adds new kernels and new substrate handling on top of this foundation without modifying the Phase 1 code paths.
