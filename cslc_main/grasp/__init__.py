# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-finger grasp test family for the CSLC contact model.

A simple, modular harness for studying how CSLC's compliant-skin lattice
behaves when two articulated pads (driven by stiff PD joint motors)
APPROACH a held object, SQUEEZE it, optionally LIFT it against gravity,
and HOLD.  Built around a small set of single-purpose modules:

    params.py         all tuning knobs, nested @dataclasses
    pads.py           pad geometry (box | dome) + Poisson-disc sampling
    objects.py        held-object factory (sphere = tennis ball)
    contact_models.py CSLC mesh-pad pipeline + per-shape configs
    solvers.py        MuJoCo / semi-implicit factory, CSLC-aware sizing
    scene.py          builds the Newton model + attaches contact model
    trajectory.py     APPROACH → SQUEEZE → [LIFT] → HOLD pad targets
    logger.py         per-run CSV logger (timeseries + CSLC state + summaries)
    visualization.py  pad-lattice preview PNG + in-sim viewer + post-sim plots
    metrics.py        Metrics dataclass + lifted/held checks
    runner.py         headless runner + viewer Example
    main.py           CLI entry point — minimal args, most config via defaults

Use ``uv run -m cslc_main.grasp.main`` to execute.
"""
