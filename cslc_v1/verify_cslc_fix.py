#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CSLC fix verification — inspect contact normals to confirm they're horizontal.

Drop this into your cslc_v1 folder and run it after applying the kernel fix:

    python -m cslc_v1.verify_cslc_fix

It builds a minimal CSLC scene, runs one collision step, and prints:
  - Contact normal direction for each active CSLC contact
  - The Z-component of the normal (should be ~0 for inner face contacts)
  - Vertical component of the friction cone (should be ~1.0 = full capacity)

If the fix is working, all inner-face contact normals should be purely
horizontal (|nz| < 0.01) and friction should be 100% vertical-capable.

If the old code is still in place, you'll see normals with significant
vertical components (|nz| up to ~0.9 for edge contacts).
"""

from __future__ import annotations

import math
import numpy as np
import warp as wp
import newton
from newton import JointTargetMode


def build_minimal_cslc_scene():
    """Single CSLC pad + single sphere, no ground, no joint drive."""
    b = newton.ModelBuilder()

    # Dynamic sphere at origin
    sphere_body = b.add_link(
        xform=wp.transform((0.0, 0.0, 0.03), wp.quat_identity()),
        label="sphere",
    )
    b.add_shape_sphere(
        sphere_body,
        radius=0.03,
        cfg=newton.ModelBuilder.ShapeConfig(
            ke=5000, kd=50, kf=100, mu=0.5, gap=0.002, density=4421.0,
        ),
    )
    b.add_joint_free(sphere_body, label="sphere_free")
    b.add_articulation([b.joint_count - 1], label="sphere_art")

    # CSLC pad positioned so inner face is in contact with sphere
    # Pad at x=-0.0175, so inner face at x=-0.0075, sphere surface at x=-0.03
    # Penetration = 0.03 - 0.0075 = 0.0225 (22.5mm, larger than default)
    pad_body = b.add_link(
        xform=wp.transform((-0.0175, 0.0, 0.03), wp.quat_identity()),
        is_kinematic=True,
        label="pad",
    )
    b.add_shape_box(
        pad_body, hx=0.01, hy=0.02, hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(
            ke=5000, kd=50, kf=100, mu=0.5, gap=0.002, density=1000,
            is_cslc=True,
            cslc_spacing=0.005, cslc_ka=5000, cslc_kl=500, cslc_dc=2.0,
            cslc_n_iter=40, cslc_alpha=0.6,
        ),
    )

    m = b.finalize()
    m.set_gravity((0.0, 0.0, 0.0))  # disable gravity for static test
    return m


def inspect_contacts(model, contacts):
    """Print per-contact normal direction and friction capacity."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    normals = contacts.rigid_contact_normal.numpy()[:n]

    active_mask = shape0 >= 0
    active_normals = normals[active_mask]
    n_active = len(active_normals)

    if n_active == 0:
        print("  ⚠  No active contacts! Check pad position / spacing.")
        return

    # For each contact, compute:
    #   nz = |normal.z|  (vertical component magnitude — should be ~0 for inner face)
    #   vertical_friction_capacity = sqrt(1 - nz²)  (friction perpendicular to normal)
    nz_abs = np.abs(active_normals[:, 2])
    vf_cap = np.sqrt(np.clip(1.0 - nz_abs**2, 0.0, 1.0))

    print(f"  Active CSLC contacts: {n_active}")
    print(f"  Normal Z-component (|nz|):")
    print(f"    mean = {nz_abs.mean():.4f}")
    print(f"    max  = {nz_abs.max():.4f}")
    print(f"    min  = {nz_abs.min():.4f}")
    print(f"  Vertical friction capacity (fraction of μ·Fn usable vertically):")
    print(f"    mean = {vf_cap.mean():.4f}  (1.0 = full, 0.0 = none)")
    print(f"    min  = {vf_cap.min():.4f}")

    # Verdict
    print()
    if nz_abs.max() < 0.05:
        print("  ✅ FIX WORKING: all normals are ~horizontal, full friction for lifting.")
    elif nz_abs.mean() < 0.1:
        print("  ⚠  PARTIAL: most normals horizontal, some edge contacts tilted.")
    else:
        print("  ❌ OLD BEHAVIOR: normals have significant Z-components.")
        print("     Friction will have reduced vertical capacity.")
        print("     Ensure you've copied the fixed cslc_kernels.py and cleared")
        print("     the Warp cache (rm -rf ~/.cache/warp/).")


def main():
    print("━" * 60)
    print("  CSLC FIX VERIFICATION")
    print("━" * 60)

    wp.init()
    model = build_minimal_cslc_scene()
    print(f"\nModel: {model.body_count} bodies, {model.shape_count} shapes")

    # Run one collision step
    state = model.state()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    model.collide(state, contacts)
    wp.synchronize()

    print("\n" + "─" * 60)
    print("  CONTACT NORMAL ANALYSIS")
    print("─" * 60)
    inspect_contacts(model, contacts)

    print()


if __name__ == "__main__":
    main()