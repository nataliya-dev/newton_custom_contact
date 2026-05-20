# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""cslc_main — modular test families for the CSLC contact model.

Each subpackage hosts one family of tests (e.g. ``grasp`` for two-finger
grasps, ``sliding`` for shear-only contact, etc.).  Tests in different
families share neither parameters nor scene code — they are independent
harnesses bound only by the CSLC contact model under test.
"""
