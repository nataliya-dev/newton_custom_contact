# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop test: CSLC sphere onto a fixed rigid sphere.

Standalone study of the CSLC contact model used as a generic deformable-
contact law (no pads, no grasp, no PD drive).  One falling deformable
ball, one immovable rigid ball, gravity.  Exercises the half-space,
alignment-gated, area-weighted form of CSLC end-to-end on a
configuration with no pad/grasp asymmetry::

    +--------------------+
    |  TOP  = CSLC mesh  |  ──►  free body, drops under gravity
    +--------------------+
              ⋆⋆⋆ contact ⋆⋆⋆
    +--------------------+
    |  BOT = rigid Sphere|  ──  pinned to world by a fixed joint
    +--------------------+

Modelling choices
~~~~~~~~~~~~~~~~~

* **Top sphere**: a triangulated icosphere flagged ``is_cslc=True``.  Its
  surface is Lloyd/CVT-sampled (``point_cloud_utils.sample_mesh_lloyd``)
  into a ``CSLCLattice`` with outward-radial normals at every lattice
  sphere — the same surface-sampling pattern used for the grasp pad in
  :mod:`cslc_main.grasp.pads`, just covering the entire sphere instead
  of one contact face.

* **Bottom sphere**: a plain Newton ``SPHERE`` primitive routed through
  CSLC's point-set target path (Fibonacci-spiral surface samples with
  outward-radial normals and uniform Voronoi-area weights, via
  :func:`cslc_main.grasp.objects.make_sphere_target`).

* **Bot is pinned, top is free**.  A fixed joint anchors the bot to
  the world so the only physics in play is the CSLC contact pair —
  any divergence can be attributed unambiguously to CSLC.  The top
  has a 6-DoF free joint and falls under gravity.  No ground plane is
  added; both bodies float in space.

* **Single contact channel.**  The CSLC handler installs its own
  narrow-phase suppression for the (top_mesh ↔ bot_sphere) pair (see
  :func:`cslc_main.grasp.contact_models.build_cslc_handler_with_mesh_pads`),
  so every contact slot in the scene is a CSLC slot.

Reuses (no duplicated machinery)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`~cslc_main.grasp.params.CSLCParams`, ``MaterialParams``,
  ``TimingParams``, ``SolverParams`` — defaults match the grasp scene.
* :func:`~cslc_main.grasp.contact_models.make_pad_shape_cfg` and
  :func:`~cslc_main.grasp.contact_models.make_cslc_pad_from_samples`
  — same lattice / shape-config builder as the grasp pads.
* :func:`~cslc_main.grasp.contact_models.build_cslc_handler_with_mesh_pads`
  and :func:`~cslc_main.grasp.contact_models.patched_cslc_from_model`
  — same handler-attach pipeline as the grasp scene.
* :func:`~cslc_main.grasp.objects.make_object_shape_cfg` — same
  rigid-target shape config.
* :func:`~cslc_main.grasp.solvers.make_solver` — same MuJoCo solver
  factory (auto-sizes ``njmax`` for CSLC slots).
* :func:`~cslc_main.grasp.logger.read_cslc_state` /
  :func:`~cslc_main.grasp.logger.count_active_contacts` — same per-step
  state snapshots.

Calibration caveats (relative to the grasp pad scenario)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSLC's default tuning in :class:`cslc_main.grasp.params.CSLCParams` is
calibrated for *one* topology — a flat hexagonal pad lattice pressing
against a small contact patch.  Two CSLC parameters are overridden in
this script (see :func:`_cslc_for_drop`) for the sphere-on-sphere case:

* ``kc_per_volume`` softened from ``1e10`` to ``5e7``.  The closed
  sphere lattice has ~200 surface spheres covering the full surface;
  on a curved target like the rigid sphere, ~80-200 lattice spheres
  simultaneously lie inside the smooth tail's reach
  (``50·ε ≈ 25 mm``) of the target's near hemisphere, so the
  aggregate force is 10-20× larger per unit penetration than the
  pad case.  Without softening kc, the per-step force is large
  enough to overshoot the Jacobi solver's contraction radius.

* ``build_A_inv = False``.  The dense closed-form warm-start solves
  ``(K_l + K_a + kc·I) δ = kc·φ_rest`` via a precomputed inverse;
  on a closed sphere lattice ``K_l`` has a constant-mode null space
  the inverse handles noisily.  Jacobi iteration from ``δ = 0``
  converges cleanly instead.

Behavioural observations once those overrides are applied:

* The CSLC smooth tail acts as a soft barrier starting ``~50·ε`` (i.e.
  ~25 mm at default ``smoothing_eps=5e-4``) BEFORE geometric contact —
  the top sphere is decelerated noticeably above the rigid-kiss
  height ``z = bot_z + bot_R + top_R``.
* Dynamics are heavily over-damped at the calibrated defaults: the
  top sphere does not "bounce", it slowly asymptotes through the
  kiss height into a quasi-static equilibrium where the smooth-tail
  CSLC force balances its weight.  Plot panels 2 and 4 (top v_z and
  max ‖δ‖) tell the story.
* For more dramatic impact dynamics, raise ``top_start_z`` and/or
  ``top_density`` so the kinetic budget overcomes the tail damping,
  and/or further reduce ``kc_per_volume``.

Run::

    uv run python -m cslc_main.grasp.scripts.drop_sphere_on_sphere
    uv run python -m cslc_main.grasp.scripts.drop_sphere_on_sphere \\
        --top-start-z 0.20 --duration 2.0
    # softer kc for more obvious bounce dynamics:
    uv run python -m cslc_main.grasp.scripts.drop_sphere_on_sphere \\
        --cslc-kc 1e7 --top-start-z 0.18
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import trimesh
import warp as wp

import newton

from cslc_main.grasp import contact_models, objects
from cslc_main.grasp.logger import count_active_contacts, read_cslc_state
from cslc_main.grasp.params import (
    CSLCParams,
    HydroParams,
    MaterialParams,
    ObjectParams,
    PadParams,
    SolverParams,
    TimingParams,
)
from cslc_main.grasp.solvers import make_solver


# Repo root: cslc_main/grasp/scripts/drop_sphere_on_sphere.py → ../../..
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "drop_sphere_on_sphere"


# ── Scene knobs ──────────────────────────────────────────────────────────


@dataclass
class DropConfig:
    """Drop-test configuration.

    Defaults are chosen so the impact stays well within CSLC's calibrated
    operating regime (peak compression of order millimetres, contact
    duration ≫ ``dt``).
    """

    # ── Top (CSLC, "deformable") sphere ──
    # Sampling defaults are tuned for an interactive viewer (~30 fps on
    # a 3070): per-step cost scales as ``top_n_samples × bot_n_samples``
    # so the product is the knob to watch.  60 × 200 = 12k pair checks
    # per step is a ~10× speed-up over the original 200 × 600 = 120k.
    # For headless paper-grade runs, raise both via the CLI:
    #     --top-n-samples 200 --bot-n-samples 600
    top_radius: float = 0.025        # 25 mm
    top_density: float = 500.0       # ~32 g for the 25 mm sphere
    top_n_samples: int = 60          # CSLC lattice spheres on the surface
    top_subdivisions: int = 3        # icosphere (162 verts, 320 tris)
    # Spawn the top sphere just above the bot — the CSLC contact form
    # is calibrated for low/moderate impact velocity (face_pen budget
    # ~mm).  At 30 mm drop, impact velocity ≈ 0.77 m/s, peak compression
    # ~1-3 mm — well inside the calibrated regime.  Larger drops still
    # work but spawn larger transient force spikes that the MuJoCo
    # solver smears out over ~2-3 dt before settling.
    top_start_z: float = 0.165       # 30 mm above the bot kiss height

    # ── Bottom (rigid primitive) sphere ──
    # Pinned to the world via a fixed joint (no ground contact, no
    # ringing) so the only physics in play is the CSLC contact pair.
    # ``bot_center_z`` is the sphere's world centre; default 0.10 m
    # keeps it clear of the ground plane.
    bot_radius: float = 0.030        # 30 mm
    bot_center_z: float = 0.10       # 10 cm above ground (no ground contact)
    bot_density: float = 2000.0      # unused (fixed joint, no dynamics)
    bot_n_samples: int = 200         # Fibonacci-spiral target sample count

    # ── World ──
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Per-pad-sphere contact-slot budget override.  ``compute_k_max``'s
    # default formula (geometry of a flat box face, see
    # :mod:`cslc_main.grasp.objects`) under-counts for sphere-on-sphere:
    # a CSLC lattice sphere near the contact patch sees the curved
    # target's near hemisphere through the 50-ε inclusion radius, so the
    # truncation warning fires on ~5-15 lattice spheres at peak
    # compression.  256 covers it for the default 600-sample target
    # sphere; raise proportionally if you bump ``bot_n_samples``.
    k_max_override: int = 256

    # ── Reused sub-configs ──
    #
    # Two non-default choices vs. the grasp pad's CSLCParams (applied
    # below via ``_cslc_for_drop``):
    #
    # * ``kc_per_volume = 5e7`` (vs. the grasp default ``1e10``).  The
    #   grasp kc is calibrated against a flat hexagonal-lattice pad
    #   pressed against a small contact patch (~10-20 active lattice
    #   spheres at peak compression).  A closed sphere lattice covering
    #   an entire 25 mm ball has ~80-200 lattice spheres simultaneously
    #   inside the smooth tail's reach (``50·ε ≈ 25 mm``) of the curved
    #   target's near hemisphere, so the aggregate
    #   ``F = Σ_i k_c · φ_i`` is 10-20× larger per unit penetration
    #   than the pad case.  Softening kc by ~200× restores a comparable
    #   per-pair force level and keeps the Jacobi sweep within its
    #   contraction radius.
    #
    # * ``build_A_inv = False`` (vs. default ``True``).  The dense
    #   closed-form warm-start solves
    #   ``(K_l + K_a + kc·I) δ = kc·φ_rest`` via a precomputed inverse.
    #   On a closed sphere lattice the graph Laplacian ``K_l`` has a
    #   1-D null space (the constant mode), which the anchor diagonal
    #   ``K_a`` regularises analytically but conditions the inverse on
    #   the ``K_a / K_l`` ratio.  Benign for an open pad lattice;
    #   noisy for a closed sphere — the Jacobi iteration starts from a
    #   clean δ=0 instead and converges cleanly.
    timing: TimingParams = field(default_factory=TimingParams)
    material: MaterialParams = field(default_factory=MaterialParams)
    cslc: CSLCParams = field(default_factory=lambda: _cslc_for_drop())
    hydro: HydroParams = field(default_factory=HydroParams)
    solver: SolverParams = field(default_factory=SolverParams)

    duration_s: float = 1.5

    # ── Output ──
    output_root: Path = field(default_factory=lambda: _DEFAULT_OUTPUT_ROOT)
    run_label: str | None = None
    use_timestamp: bool = True

    @property
    def n_steps(self) -> int:
        return int(self.duration_s / self.timing.dt)

    def run_dir(self) -> Path:
        label = self.run_label or "drop"
        if self.use_timestamp:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            label = f"{ts}_{label}"
        return self.output_root / label


def _cslc_for_drop() -> CSLCParams:
    """CSLCParams overridden for a closed sphere lattice (see ``DropConfig``).

    Three overrides vs. the grasp's CSLCParams:

    * ``kc_per_volume = 5e7`` and ``build_A_inv = False`` per
      ``DropConfig`` — closed-sphere stability.
    * ``n_iter = 20`` (vs. the grasp's 40) — viewer-grade convergence.
      Each Jacobi sweep is one ``wp.launch``, and per-step cost is
      linear in ``n_iter``.  At 60 lattice spheres / 200 target samples
      the dynamics already converge cleanly in 20 sweeps; 40 was
      calibrated for the grasp's dome pad with a much smaller contact
      patch and tighter convergence budget.
    """
    p = CSLCParams()
    p.kc_per_volume = 5.0e7
    p.build_A_inv = False
    p.n_iter = 20
    return p


# ── Sphere mesh + lattice sampling ───────────────────────────────────────


def _build_sphere_lattice(
    radius: float, n_samples: int, subdivisions: int
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Build an icosphere mesh and Lloyd-sample its surface for the lattice.

    The icosphere provides a clean, watertight, near-equilateral
    triangulation of the unit sphere scaled to ``radius``.  Lloyd's
    algorithm (centroidal Voronoi tessellation; Lloyd 1982; see
    :mod:`cslc_main.grasp.pads`) iteratively moves each of ``n_samples``
    points to the centroid of its surface Voronoi cell — converging to
    as-uniform-as-possible spacing on the curved surface.

    Lattice normals are exact analytic outward radial directions
    (``p̂ = p / ‖p‖``).  This makes the CSLC alignment gate
    (eq:align-gate in :doc:`cslc_main/theory/theory.md`) behave
    perfectly: ``α = −n̂_face · n̂_pad`` is identically ``+1`` at the
    contact point on two opposing spheres, so the gate is wide open and
    every lattice sphere "sees" the closest target sample at full
    weight.

    Args:
        radius: sphere radius [m].
        n_samples: number of lattice points to draw from the surface.
        subdivisions: icosphere subdivision count (``trimesh`` convention
            — 4 gives 642 vertices / 1280 triangles, a comfortable
            super-set of typical ``n_samples``).

    Returns:
        ``(mesh, positions, normals)`` — the triangle mesh (passed as
        the Newton collision shape) and the lattice positions / normals
        in the sphere's body-local frame as ``float32`` arrays.
    """
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions,
                                      radius=radius)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)

    pts = np.asarray(pcu.sample_mesh_lloyd(v, f, int(n_samples)))
    pts = pts.astype(np.float32)
    # Lloyd's output drifts off the curved surface by O(triangle edge).
    # Snap each sample to the exact analytic sphere so its distance to
    # the centre is exactly ``radius`` (otherwise the per-sample outward
    # normal direction has a small radial component error, and the
    # alignment gate's α at perfect face-on contact dips slightly below
    # 1).
    pts *= radius / np.linalg.norm(pts, axis=1, keepdims=True)
    normals = (pts / radius).astype(np.float32)
    return mesh, pts, normals


# ── Scene build ──────────────────────────────────────────────────────────


@dataclass
class DropArtifacts:
    """Handles the simulation loop needs after build_drop_scene returns."""

    model: object
    handler: object
    top_body: int
    bot_body: int
    top_shape: int
    bot_shape: int
    bot_obj_params: ObjectParams


def build_drop_scene(config: DropConfig) -> DropArtifacts:
    """Build a Newton model with a CSLC top sphere and a rigid bottom sphere."""

    # ── 1. Top sphere geometry + CSLC lattice. ──
    top_mesh_tm, top_pts, top_normals = _build_sphere_lattice(
        config.top_radius, config.top_n_samples, config.top_subdivisions,
    )
    top_mesh = newton.Mesh(
        top_mesh_tm.vertices.astype(np.float32),
        top_mesh_tm.faces.astype(np.int32).flatten(),
    )

    # ── 2. Shape configs. ──
    # Top: CSLC mesh shape.  ``make_pad_shape_cfg`` only reads ``density``
    # from PadParams (everything else is taken from material / cslc),
    # so a minimal PadParams stand-in is sufficient.
    pad_params = PadParams(kind="box", density=config.top_density)
    top_cfg = contact_models.make_pad_shape_cfg(
        pad_params, config.material, config.cslc, config.hydro,
        contact_model="cslc",
    )

    # Bottom: rigid SPHERE primitive — the CSLC handler will sample
    # its surface as a Fibonacci-spiral point set via
    # :func:`make_sphere_target` when ``obj.kind == "sphere"``.
    bot_obj_params = ObjectParams(
        kind="sphere",
        radius=config.bot_radius,
        density=config.bot_density,
        sphere_n_samples=config.bot_n_samples,
        start_z=config.bot_center_z,
    )
    bot_cfg = objects.make_object_shape_cfg(
        bot_obj_params, config.material, config.hydro,
        contact_model="cslc",
    )

    # ── 3. ModelBuilder. ──
    b = newton.ModelBuilder()

    # Bottom rigid sphere — pinned to the world via a fixed joint.
    # No ground contact, no free-body dynamics: a clean immovable target
    # against which to measure CSLC's force response.  Removes the
    # complication of free-body recoil from the contact pair and lets
    # any divergence be attributed unambiguously to CSLC itself.
    bot_body = b.add_link(
        xform=wp.transform(
            (0.0, 0.0, config.bot_center_z), wp.quat_identity()
        ),
        label="bot_rigid_sphere",
    )
    bot_shape = b.add_shape_sphere(
        bot_body, radius=config.bot_radius, cfg=bot_cfg,
    )
    # Fixed joint anchors the body to the world.  The joint's
    # ``parent_xform`` (in world frame, since parent=-1) is what
    # eval_fk uses to place the child body — passing the spawn position
    # on ``add_link`` alone is NOT sufficient because eval_fk overwrites
    # body_q from the joint chain.  Without this, body_q[bot] resets to
    # the identity (z=0) and the bot sphere ends up at world origin.
    j_bot = b.add_joint_fixed(
        parent=-1,
        child=bot_body,
        parent_xform=wp.transform(
            (0.0, 0.0, config.bot_center_z), wp.quat_identity()
        ),
        label="bot_fixed",
    )
    b.add_articulation([j_bot], label="bot_articulation")

    # Top CSLC sphere — free body, spawned ``top_start_z`` above the ground.
    top_body = b.add_link(
        xform=wp.transform(
            (0.0, 0.0, config.top_start_z), wp.quat_identity()
        ),
        label="top_cslc_sphere",
    )
    top_shape = b.add_shape_mesh(
        body=top_body,
        mesh=top_mesh,
        cfg=top_cfg,
        label="top_cslc_mesh",
    )
    j_top = b.add_joint_free(top_body, label="top_free")
    b.add_articulation([j_top], label="top_articulation")

    b.request_contact_attributes("force")

    model = b.finalize()
    model.set_gravity(config.gravity)

    # ── 4. CSLC handler attach. ──
    cslc_lattice = contact_models.make_cslc_pad_from_samples(
        top_pts, top_normals, top_shape, k_neighbors=6,
    )
    handler = contact_models.build_cslc_handler_with_mesh_pads(
        model,
        mesh_pads_by_shape={top_shape: cslc_lattice},
        cslc=config.cslc,
        obj=bot_obj_params,
        dt=config.timing.dt,
    )
    if handler is None:
        raise RuntimeError(
            "CSLC handler construction failed — no CSLC pair wired.  "
            "Check that the top mesh shape has is_cslc=True and that "
            "the bottom shape is a SPHERE primitive."
        )
    # Override K_max on every CSLC pair.  ``compute_k_max`` sizes the
    # per-pad-sphere slot budget assuming a flat target face; for a
    # curved sphere target a lattice sphere near the contact point sees
    # the full near-hemisphere of the target through its 50-ε inclusion
    # radius, which is many more samples than the flat-face estimate.
    # Bumping the budget up front avoids the runtime truncation warning
    # and the asymmetric forces it introduces.
    for pair in handler.shape_pairs:
        pair.K_max = max(pair.K_max, config.k_max_override)

    # The CollisionPipeline is built inside the patched context so the
    # auto-discovered ``CSLCHandler._from_model`` returns our pre-built
    # handler; after exit, ``model._collision_pipeline`` carries it.
    with contact_models.patched_cslc_from_model(handler):
        _ = model.contacts()

    return DropArtifacts(
        model=model,
        handler=handler,
        top_body=top_body,
        bot_body=bot_body,
        top_shape=top_shape,
        bot_shape=bot_shape,
        bot_obj_params=bot_obj_params,
    )


# ── Headless run + logging ───────────────────────────────────────────────


@dataclass
class DropLog:
    """Per-step timeseries captured during the drop.

    ``bot_z`` is omitted because the bot sphere is pinned via a fixed
    joint and never moves; ``top_vz`` reads the linear-z slot of
    ``state.body_qd`` for the top body — Newton's spatial-velocity
    convention puts linear velocity in the last three slots.
    """

    t: list[float] = field(default_factory=list)
    top_z: list[float] = field(default_factory=list)
    top_vz: list[float] = field(default_factory=list)
    n_contacts: list[int] = field(default_factory=list)
    n_active: list[int] = field(default_factory=list)
    max_delta_mm: list[float] = field(default_factory=list)
    max_pen_mm: list[float] = field(default_factory=list)

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(getattr(self, k)) for k in self.__dataclass_fields__}

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_numpy()
        names = list(data.keys())
        cols = np.stack([data[n] for n in names], axis=1)
        header = ",".join(names)
        np.savetxt(path, cols, delimiter=",", header=header, comments="")


def run_drop(config: DropConfig) -> DropLog:
    """Build the scene, run the drop, and return the per-step log."""
    print(_summary(config))
    artifacts = build_drop_scene(config)
    model = artifacts.model
    solver = make_solver(model, config.solver)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = config.timing.dt
    log = DropLog()

    # Warm-up step — primes Warp JIT + MuJoCo solver so the timed loop
    # sees steady-state per-step cost.  Output state is discarded by
    # the swap convention.
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    # Reset state_0 to the freshly-built joint coords so the warm-up
    # doesn't leak into the timeseries.  Cheap because eval_fk is
    # O(n_bodies).
    state_0 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    wp.synchronize()

    t0 = time.perf_counter()
    print_every = max(1, config.n_steps // 10)
    for step in range(config.n_steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        # ── Read state for logging. ──
        q = state_0.body_q.numpy()
        qd = state_0.body_qd.numpy()
        top_xyz = q[artifacts.top_body, :3]
        # body_qd layout: [omega_xyz, v_xyz] in Newton's spatial
        # convention; the linear velocity sits in the last three slots.
        top_vz = float(qd[artifacts.top_body, 5])

        n_total = count_active_contacts(contacts)
        cs = read_cslc_state(model) or {}

        log.t.append(step * dt)
        log.top_z.append(float(top_xyz[2]))
        log.top_vz.append(top_vz)
        log.n_contacts.append(n_total)
        log.n_active.append(int(cs.get("n_active", 0)))
        log.max_delta_mm.append(float(cs.get("max_delta_mm", 0.0)))
        log.max_pen_mm.append(float(cs.get("max_pen_mm", 0.0)))

        if (step + 1) % print_every == 0 or step == config.n_steps - 1:
            print(
                f"  step={step + 1:5d}/{config.n_steps}  "
                f"t={step * dt:5.3f}s  "
                f"top_z={top_xyz[2] * 1e3:+7.2f}mm  "
                f"vz={top_vz:+6.3f}m/s  "
                f"n={n_total:3d} (active={cs.get('n_active', 0):3d})  "
                f"max_δ={cs.get('max_delta_mm', 0.0):5.2f}mm  "
                f"max_pen={cs.get('max_pen_mm', 0.0):5.2f}mm"
            )

    wall = time.perf_counter() - t0
    rtx = config.n_steps * dt / max(wall, 1e-9)
    print(
        f"  TIMING  wall={wall:.3f}s  "
        f"per-step={1000 * wall / config.n_steps:.3f}ms  "
        f"realtime×={rtx:.2f}"
    )
    return log


def _summary(config: DropConfig) -> str:
    c = config
    return (
        "\n" + "━" * 60 + "\n"
        "  drop_sphere_on_sphere (CSLC as generic contact)\n"
        + "━" * 60 + "\n"
        f"  top : R={c.top_radius * 1e3:.1f}mm  density={c.top_density:.0f}  "
        f"n_lattice={c.top_n_samples}  start_z={c.top_start_z * 1e3:.0f}mm\n"
        f"  bot : R={c.bot_radius * 1e3:.1f}mm  centre_z={c.bot_center_z * 1e3:.0f}mm  "
        f"n_target={c.bot_n_samples}  (fixed to world)\n"
        f"  cslc: kc_per_volume={c.cslc.kc_per_volume:.2e}  "
        f"ka={c.cslc.ka:.0f}  kl={c.cslc.kl:.0f}  "
        f"eps={c.cslc.smoothing_eps:.1e}\n"
        f"  sim : dt={c.timing.dt * 1e3:.2f}ms  "
        f"duration={c.duration_s:.2f}s  ({c.n_steps} steps)\n"
        f"  out : {c.run_dir()}\n"
    )


# ── Post-sim plot ────────────────────────────────────────────────────────


def save_plot(log: DropLog, config: DropConfig, path: Path) -> None:
    """Render a 3-panel summary of the drop dynamics."""
    import matplotlib.pyplot as plt  # local import — keeps headless run fast

    data = log.to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # Panel 1: top-sphere centre height vs. time.
    ax = axes[0]
    ax.plot(data["t"], data["top_z"] * 1e3, label="top (CSLC) centre z",
            color="tab:blue")
    # Reference line: the rigid-rigid "kiss" height = bot_centre + bot_R
    # + top_R, the analytic contact height for two undeformed spheres.
    kiss = (config.bot_center_z + config.bot_radius + config.top_radius) * 1e3
    ax.axhline(kiss, color="gray", linestyle="--", linewidth=0.8,
               label=f"rigid-kiss z = {kiss:.1f} mm")
    ax.set_ylabel("top z  [mm]")
    ax.set_title("Top CSLC sphere dropped onto a fixed rigid sphere")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: top-sphere vertical velocity.  Sign flips at each impact
    # and the steady-state ringing amplitude reads the CSLC restitution.
    ax = axes[1]
    ax.plot(data["t"], data["top_vz"], color="tab:cyan")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("top v_z  [m/s]")
    ax.grid(True, alpha=0.3)

    # Panel 3: contact counts.
    ax = axes[2]
    ax.plot(data["t"], data["n_contacts"],
            label="total contacts emitted", color="tab:green")
    ax.plot(data["t"], data["n_active"],
            label="active lattice spheres (δ·n̂ < 0)",
            color="tab:red", linestyle="--")
    ax.set_ylabel("count")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: peak lattice displacement.  ``read_cslc_state`` also
    # reports ``max_pen_mm`` but in the v2 EXPONENT-EXPERIMENT kernel
    # the underlying ``raw_penetration`` field stores
    # ``φ = raw_pos² · smooth_step`` (units m², not m) — the per-sphere
    # contact-force prefactor, not literal penetration depth — so it is
    # left out of the visual to avoid confusion.  Lattice ‖δ‖ is the
    # one physically interpretable compression signal.
    ax = axes[3]
    ax.plot(data["t"], data["max_delta_mm"],
            label="max ‖δ‖ (lattice displacement)", color="tab:purple")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("lattice compression  [mm]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ── Viewer-mode Example driver ───────────────────────────────────────────


class Example:
    """Viewer-mode driver for ``newton.examples.run``.

    Mirrors :class:`cslc_main.grasp.runner.Example` but tailored to the
    drop test: no PD trajectory, no per-pad targets, single free body.
    Reuses the grasp visualisation helpers
    (:class:`LatticeRenderer`, :class:`TargetPointsRenderer`,
    :class:`ContactNormalRenderer`) so the lattice deformation, target
    point cloud, and contact-normal arrows render exactly as they do in
    the grasp viewer.

    The physics ``dt`` stays at ``config.timing.dt`` (typically 2 ms).
    The viewer ticks at 60 fps and advances the physics by exactly one
    ``sim_dt`` per frame — i.e. simulated time runs at ``sim_dt / frame_dt``
    of real time (~12 % at 2 ms / 60 fps), so the bounce plays back in
    visible slow motion.  This trades wall-clock realism for visual
    smoothness; the alternative (substepping to track real time)
    starves the GPU at ~10 fps and looks worse than slow-mo on a
    millisecond-scale impact event anyway.
    """

    def __init__(self, viewer, args, config: DropConfig):
        self.viewer = viewer
        self.config = config
        self.test_mode = bool(getattr(args, "test", False))
        self._saved_artifacts = False
        # Save CSV + plot on any process exit (normal close, Ctrl-C,
        # exception).  ``newton.examples.run`` only calls ``test_final``
        # in ``--test`` mode, so without this hook a normal viewer
        # session would discard the per-step log.  Idempotent — the
        # hook checks ``_saved_artifacts`` to avoid double-writing
        # when ``test_final`` also fires.
        import atexit
        atexit.register(self._save_artifacts)

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # One sim tick per frame — slow-mo playback (~12 % real-time at
        # 2 ms / 60 fps), but keeps the per-frame physics budget at one
        # ``solver.step`` (~13 ms at the viewer-default sampling), well
        # inside the 16.7 ms frame budget.
        self.sim_substeps = 1
        self.sim_dt = config.timing.dt

        print(_summary(config))
        self.artifacts = build_drop_scene(config)
        self.model = self.artifacts.model
        self.solver = make_solver(self.model, config.solver)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0,
        )

        self.sim_step = 0
        self.sim_time = 0.0
        self.log = DropLog()

        # Reuse the grasp viewer overlays directly.  ``LatticeRenderer``
        # colours each lattice sphere by signed ‖δ·n̂‖ (red = compressed
        # inward, cyan = bulging outward, gray = at rest);
        # ``TargetPointsRenderer`` draws the bot's Fibonacci-spiral
        # samples and highlights the engaged ones;
        # ``ContactNormalRenderer`` draws yellow arrows at each emitted
        # MuJoCo contact, pointing along the target's outward normal.
        from cslc_main.grasp.visualization import (
            ContactNormalRenderer,
            LatticeRenderer,
            TargetPointsRenderer,
        )
        self.lattice = LatticeRenderer(self.model, self.viewer)
        self.target_points = TargetPointsRenderer(self.model, self.viewer)
        self.contact_normals = ContactNormalRenderer(self.model, self.viewer)

        self.viewer.set_model(self.model)
        # Camera pose: look at the contact region from a 3/4 angle.  The
        # scene's interesting volume is the ~50 mm box around the bot
        # sphere centre.
        cam_target_z = config.bot_center_z + 0.03
        self.viewer.set_camera(
            pos=wp.vec3(0.20, -0.20, cam_target_z + 0.05),
            pitch=-15.0,
            yaw=135.0,
        )

    def _step_sim(self) -> None:
        """Advance the physics by one ``sim_dt`` and append to ``self.log``."""
        self.state_0.clear_forces()
        self.model.collide(self.state_0, self.contacts)
        self.solver.step(
            self.state_0, self.state_1, self.control,
            self.contacts, self.sim_dt,
        )
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_step += 1

        q = self.state_0.body_q.numpy()
        qd = self.state_0.body_qd.numpy()
        top_xyz = q[self.artifacts.top_body, :3]
        top_vz = float(qd[self.artifacts.top_body, 5])
        n_total = count_active_contacts(self.contacts)
        cs = read_cslc_state(self.model) or {}
        self.log.t.append(self.sim_step * self.sim_dt)
        self.log.top_z.append(float(top_xyz[2]))
        self.log.top_vz.append(top_vz)
        self.log.n_contacts.append(n_total)
        self.log.n_active.append(int(cs.get("n_active", 0)))
        self.log.max_delta_mm.append(float(cs.get("max_delta_mm", 0.0)))
        self.log.max_pen_mm.append(float(cs.get("max_pen_mm", 0.0)))

    def simulate(self) -> None:
        """Advance ``sim_substeps`` physics ticks per render frame.

        After ``n_steps`` total physics steps, the loop stops advancing —
        the viewer can still spin the camera, but the scene is frozen at
        the final equilibrium.
        """
        for _ in range(self.sim_substeps):
            if self.sim_step >= self.config.n_steps:
                return
            self._step_sim()

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.lattice.update(self.state_0)
        self.target_points.update(self.state_0)
        self.contact_normals.update(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _save_artifacts(self) -> None:
        """Write the CSV + summary PNG of the per-step log; idempotent.

        Registered as an ``atexit`` hook in ``__init__`` so it fires on
        normal viewer close, Ctrl-C, or test_final, without
        double-writing on test runs.  No-op if no steps were logged
        (e.g. viewer opened and closed without stepping).
        """
        if self._saved_artifacts or not self.log.t:
            return
        self._saved_artifacts = True
        run_dir = self.config.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log.save_csv(run_dir / "timeseries.csv")
        save_plot(self.log, self.config, run_dir / "drop_summary.png")
        print(f"\n  wrote {run_dir / 'timeseries.csv'}")
        print(f"  wrote {run_dir / 'drop_summary.png'}\n")

    def test_final(self) -> None:
        """Regression assertion + save artifacts (``--test`` mode).

        ``newton.examples.run`` only calls this when the user passes
        ``--test``; the routine atexit hook handles the artifact write
        for normal viewer runs.
        """
        if self.log.top_z:
            final_z = self.log.top_z[-1]
            # Sanity: the top sphere should sit between the kiss
            # height and a couple of cm below it.  Far above means
            # contact never engaged; far below means CSLC blew up
            # and the top sphere passed through.
            kiss = (
                self.config.bot_center_z
                + self.config.bot_radius
                + self.config.top_radius
            )
            assert kiss - 0.05 < final_z < self.config.top_start_z + 0.01, (
                f"final top_z = {final_z * 1e3:.2f} mm out of expected "
                f"range (~{(kiss - 0.05) * 1e3:.0f}-"
                f"{(self.config.top_start_z + 0.01) * 1e3:.0f} mm)"
            )
        self._save_artifacts()


# ── CLI ──────────────────────────────────────────────────────────────────


def _add_drop_args(parser: argparse.ArgumentParser) -> None:
    """Attach drop-test args to an existing parser.

    Kept separate from :func:`_apply_args_to_config` so the same arg
    surface works for both the headless run and the viewer-mode run
    (which uses ``newton.examples.create_parser`` + ``newton.examples.init``
    and re-parses args after the viewer chooses its backend).
    """
    g = parser.add_argument_group("Drop test")
    g.add_argument("--top-start-z", type=float, default=None,
                   help="Top sphere spawn height [m] (default 0.165).")
    g.add_argument("--top-radius", type=float, default=None,
                   help="Top (CSLC) sphere radius [m] (default 0.025).")
    g.add_argument("--bot-radius", type=float, default=None,
                   help="Bottom (rigid) sphere radius [m] (default 0.030).")
    g.add_argument("--bot-center-z", type=float, default=None,
                   help="Bottom sphere centre height [m] (default 0.10). "
                        "Pinned by a fixed joint, no ground contact.")
    g.add_argument("--k-max-override", type=int, default=None,
                   help="Manual per-pad-sphere contact-slot budget. "
                        "Default 256 covers the 600-sample default bot "
                        "sphere; raise if you see truncation warnings.")
    g.add_argument("--top-n-samples", type=int, default=None,
                   help="CSLC lattice sample count on the top sphere "
                        "(default 200).")
    g.add_argument("--bot-n-samples", type=int, default=None,
                   help="Fibonacci-spiral target sample count on the "
                        "bottom sphere (default 600).")
    g.add_argument("--duration", type=float, default=None,
                   help="Total sim duration [s] (default 1.5).")
    g.add_argument("--cslc-kc", type=float, default=None,
                   help="Override CSLCParams.kc_per_volume.")
    g.add_argument("--cslc-ka", type=float, default=None,
                   help="Override CSLCParams.ka.")
    g.add_argument("--cslc-kl", type=float, default=None,
                   help="Override CSLCParams.kl.")
    g.add_argument("--cslc-dc", type=float, default=None,
                   help="Override CSLCParams.dc (per-contact damping).")
    g.add_argument("--cslc-c-lat", type=float, default=None,
                   help="Override CSLCParams.c_lattice (lattice velocity "
                        "damping coefficient).")
    g.add_argument("--cslc-smoothing-eps", type=float, default=None,
                   help="Override CSLCParams.smoothing_eps [m].  Sharper "
                        "(smaller) values shrink the smooth-tail soft-"
                        "barrier zone (~50·ε wide) but couple with kc to "
                        "increase effective contact stiffness; see "
                        "CSLCParams.smoothing_eps for the tradeoff.")
    g.add_argument("--material-ke", type=float, default=None,
                   help="Set BOTH MaterialParams.ke_pad_physical and "
                        "ke_target_physical to this value.")
    g.add_argument("--run-label", type=str, default=None,
                   help="Override the auto-generated run directory label.")
    g.add_argument("--no-timestamp", action="store_true",
                   help="Drop the timestamp prefix; overwrites the same "
                        "directory on each run.")


def _apply_args_to_config(args, config: DropConfig) -> DropConfig:
    """Mutate ``config`` in place from parsed CLI args; return it."""
    if args.top_start_z is not None:
        config.top_start_z = args.top_start_z
    if args.top_radius is not None:
        config.top_radius = args.top_radius
    if args.bot_radius is not None:
        config.bot_radius = args.bot_radius
    if args.bot_center_z is not None:
        config.bot_center_z = args.bot_center_z
    if args.k_max_override is not None:
        config.k_max_override = args.k_max_override
    if args.top_n_samples is not None:
        config.top_n_samples = args.top_n_samples
    if args.bot_n_samples is not None:
        config.bot_n_samples = args.bot_n_samples
    if args.duration is not None:
        config.duration_s = args.duration
    if args.cslc_kc is not None:
        config.cslc.kc_per_volume = args.cslc_kc
    if args.cslc_ka is not None:
        config.cslc.ka = args.cslc_ka
    if args.cslc_kl is not None:
        config.cslc.kl = args.cslc_kl
    if args.cslc_dc is not None:
        config.cslc.dc = args.cslc_dc
    if args.cslc_c_lat is not None:
        config.cslc.c_lattice = args.cslc_c_lat
    if args.cslc_smoothing_eps is not None:
        config.cslc.smoothing_eps = args.cslc_smoothing_eps
    if args.material_ke is not None:
        config.material.ke = args.material_ke
    if args.run_label is not None:
        config.run_label = args.run_label
    if getattr(args, "no_timestamp", False):
        config.use_timestamp = False
    return config


def _run_headless_and_save(config: DropConfig) -> None:
    """Headless: build, sim, write CSV + PNG."""
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_drop(config)
    log.save_csv(run_dir / "timeseries.csv")
    save_plot(log, config, run_dir / "drop_summary.png")
    print(f"\n  wrote {run_dir / 'timeseries.csv'}")
    print(f"  wrote {run_dir / 'drop_summary.png'}\n")


def main() -> None:
    """Entry point — dispatches to headless or viewer mode based on ``--viewer``.

    Uses ``newton.examples.create_parser()`` so the standard Newton
    viewer flags (``--viewer gl|null|usd``, ``--num-frames``, ``--test``)
    work alongside the drop-test args.  Defaults ``--viewer null`` so
    ``uv run python -m cslc_main.grasp.scripts.drop_sphere_on_sphere``
    stays headless.  Pass ``--viewer gl`` to open the OpenGL viewer.
    """
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="null")
    _add_drop_args(parser)
    args, _ = parser.parse_known_args()

    wp.init()
    print(f"\n{'━' * 60}\n  drop_sphere_on_sphere\n{'━' * 60}")

    config = _apply_args_to_config(args, DropConfig())

    viewer_choice = getattr(args, "viewer", None)
    if viewer_choice in (None, "none", "null"):
        _run_headless_and_save(config)
        return

    # Viewer mode — let newton.examples build the viewer, then drive
    # ``Example`` (defined below) through ``newton.examples.run``.
    viewer, args = newton.examples.init(parser)
    config = _apply_args_to_config(args, config)
    newton.examples.run(Example(viewer, args, config), args)


if __name__ == "__main__":
    main()
