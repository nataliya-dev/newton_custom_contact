#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gripper lift test — point contact vs CSLC.

A sphere sits on a ground plane.  Two articulated pads (driven by prismatic
joints) approach, squeeze, lift, and hold the sphere in the air.

Because the pads are DYNAMIC bodies driven by joint motors (not kinematic),
the solver correctly computes relative velocity at the contact → friction
can drag the sphere upward during the lift phase.

Phases
──────
  APPROACH   Pads move inward from free space toward the sphere.
  SQUEEZE    Pads press inward to build grip force.
  LIFT       Pads rise together; friction must overcome gravity.
  HOLD       Pads stationary in the air; sphere should stay gripped.

Modes
─────
  viewer     Interactive OpenGL visualization (default).
  headless   Point vs CSLC comparison with summary output.

Examples
────────
  python gripper_lift_test.py --mode viewer --contact-model cslc
  python gripper_lift_test.py --mode viewer --contact-model point
  python gripper_lift_test.py --mode headless
  python gripper_lift_test.py --mode headless --solver semi
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.solvers import SolverSemiImplicit

try:
    from newton.solvers import SolverMuJoCo
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    warnings.warn("SolverMuJoCo not available — falling back to semi-implicit.")


# ── Formatting ────────────────────────────────────────────────────────────

_SEP = "─" * 60

def _log(msg, indent=0):
    print(f"  {'  ' * indent}│ {msg}")

def _section(title):
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")


# ── Scene parameters ─────────────────────────────────────────────────────

CSLC_FLAG = 1 << 5


@dataclass
class SceneParams:
    """All knobs for the gripper lift scene."""

    # Sphere
    sphere_radius: float = 0.03
    sphere_density: float = 4421.0
    sphere_start_z: float = 0.03

    # Pads (box half-extents)
    pad_hx: float = 0.01
    pad_hy: float = 0.02
    pad_hz: float = 0.05
    pad_density: float = 1000.0

    # Phase timing
    approach_gap: float = 0.10
    approach_speed: float = 0.02
    approach_duration: float = 1.5

    squeeze_speed: float = 0.005
    squeeze_duration: float = 0.5

    lift_speed: float = 0.05
    lift_duration: float = 1.5

    hold_duration: float = 1.0

    # Material (shared by point + CSLC)
    # Reduced from 5e4 to prevent CSLC force explosion (190 surface spheres × ke = huge).
    # Friction budget check: F_n = ke × pen = 5000 × 0.0125 = 62.5N per pad.
    # F_friction = μ × F_n × 2_pads = 0.5 × 62.5 × 2 = 62.5N >> 4.9N weight. OK.
    ke: float = 5.0e3
    kd: float = 5.0e1
    kf: float = 100.0
    mu: float = 0.5

    # Joint drive stiffness
    drive_ke: float = 5.0e4
    drive_kd: float = 1.0e3

    # CSLC tuning
    cslc_spacing: float = 0.01
    cslc_ka: float = 2000.0
    cslc_kl: float = 100.0
    cslc_dc: float = 2.0
    cslc_n_iter: int = 40
    cslc_alpha: float = 0.6

    # Integration
    dt: float = 1.0 / 500.0
    gravity: tuple = (0.0, 0.0, -9.81)

    @property
    def sphere_mass(self):
        return self.sphere_density * (4 / 3) * math.pi * self.sphere_radius ** 3

    @property
    def approach_steps(self):
        return int(self.approach_duration / self.dt)

    @property
    def squeeze_steps(self):
        return int(self.squeeze_duration / self.dt)

    @property
    def lift_steps(self):
        return int(self.lift_duration / self.dt)

    @property
    def hold_steps(self):
        return int(self.hold_duration / self.dt)

    @property
    def total_steps(self):
        return self.approach_steps + self.squeeze_steps + self.lift_steps + self.hold_steps

    def phase_of(self, step):
        s = step
        for name, dur in [("APPROACH", self.approach_steps),
                          ("SQUEEZE", self.squeeze_steps),
                          ("LIFT", self.lift_steps),
                          ("HOLD", self.hold_steps)]:
            if s < dur:
                return name, s
            s -= dur
        return "HOLD", 0

    def dump(self):
        _section("SCENE PARAMETERS")
        m = self.sphere_mass
        _log(f"Sphere: r={self.sphere_radius*1e3:.1f}mm  mass={m*1e3:.1f}g  weight={m*9.81:.3f}N")
        _log(f"Pads:   hx={self.pad_hx*1e3:.0f}mm  hy={self.pad_hy*1e3:.0f}mm  "
             f"hz={self.pad_hz*1e3:.0f}mm  density={self.pad_density:.0f}")
        _log(f"Phases: approach={self.approach_duration}s  squeeze={self.squeeze_duration}s  "
             f"lift={self.lift_duration}s  hold={self.hold_duration}s")
        _log(f"Material: ke={self.ke:.0f}  kd={self.kd:.0f}  μ={self.mu:.2f}")
        _log(f"Drive: ke={self.drive_ke:.0f}  kd={self.drive_kd:.0f}")
        _log(f"Steps: {self.total_steps} total  dt={self.dt*1e3:.2f}ms")


# ── Metrics ───────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    name: str = ""
    sphere_z: list[float] = field(default_factory=list)
    contacts: list[int] = field(default_factory=list)

    @property
    def max_z(self):
        return max(self.sphere_z) if self.sphere_z else 0.0

    @property
    def final_z(self):
        return self.sphere_z[-1] if self.sphere_z else 0.0

    @property
    def lifted(self):
        return self.max_z > self.sphere_z[0] + 0.005 if len(self.sphere_z) > 1 else False

    @property
    def held(self):
        return self.final_z > 0.01 if self.sphere_z else False


# ── Scene builders ────────────────────────────────────────────────────────
#
#  Each pad is a 2-DOF articulated arm:
#    world ──[prismatic X]──> slider ──[prismatic Z]──> pad
#
#  The sphere has an explicit free joint so MuJoCo tracks it properly.
#
#  Body layout (ground plane creates no body):
#    0: left_slider    2: right_slider    4: sphere
#    1: left_pad       3: right_pad
# ─────────────────────────────────────────────────────────────────────────


def _sphere_cfg(p: SceneParams):
    return newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.sphere_density)


def _build_scene(p: SceneParams, pad_cfg):
    """Build ground + 2 articulated pads + 1 free sphere."""
    b = newton.ModelBuilder()
    ground_shape = b.add_ground_plane()

    lx0 = -(p.approach_gap / 2 + p.pad_hx)
    rx0 = +(p.approach_gap / 2 + p.pad_hx)
    z0 = p.sphere_start_z

    # Ghost config for intermediate slider bodies (no collision)
    ghost_cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False, has_particle_collision=False, density=0.0)

    dof_map = {}
    pad_joints = []

    for label, x0 in [("left", lx0), ("right", rx0)]:
        # Slider body — intermediate link for X translation
        slider = b.add_link(
            xform=wp.transform((x0, 0, z0), wp.quat_identity()),
            mass=0.1, label=f"{label}_slider")
        slider_shape = b.add_shape_sphere(slider, radius=0.002, cfg=ghost_cfg)

        # Pad body — the gripping surface
        pad = b.add_link(
            xform=wp.transform((x0, 0, z0), wp.quat_identity()),
            label=f"{label}_pad")
        b.add_shape_box(pad, hx=p.pad_hx, hy=p.pad_hy, hz=p.pad_hz, cfg=pad_cfg)

        # Prismatic X: world → slider
        j_x = b.add_joint_prismatic(
            parent=-1, child=slider,
            axis=wp.vec3(1, 0, 0),
            parent_xform=wp.transform((x0, 0, z0), wp.quat_identity()),
            child_xform=wp.transform((0, 0, 0), wp.quat_identity()),
            label=f"{label}_x")

        # Prismatic Z: slider → pad
        j_z = b.add_joint_prismatic(
            parent=slider, child=pad,
            axis=wp.vec3(0, 0, 1),
            parent_xform=wp.transform((0, 0, 0), wp.quat_identity()),
            child_xform=wp.transform((0, 0, 0), wp.quat_identity()),
            label=f"{label}_z")

        b.add_articulation([j_x, j_z], label=f"{label}_arm")
        pad_joints.extend([j_x, j_z])

        dof_map[f"{label}_x"] = b.joint_qd_start[j_x]
        dof_map[f"{label}_z"] = b.joint_qd_start[j_z]

        # Filter: ghost slider vs ground (avoid spurious contacts)
        b.add_shape_collision_filter_pair(slider_shape, ground_shape)

    # Set joint drives for pad joints (stiff PD position tracking)
    for ji in pad_joints:
        dof = b.joint_qd_start[ji]
        b.joint_target_ke[dof] = p.drive_ke
        b.joint_target_kd[dof] = p.drive_kd
        b.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        b.joint_armature[dof] = 0.01

    # Dynamic sphere with explicit free joint (required for MuJoCo solver)
    sphere = b.add_link(
        xform=wp.transform((0, 0, z0), wp.quat_identity()),
        label="sphere")
    b.add_shape_sphere(sphere, radius=p.sphere_radius, cfg=_sphere_cfg(p))
    j_free = b.add_joint_free(sphere, label="sphere_free")
    b.add_articulation([j_free], label="sphere")

    m = b.finalize()
    m.set_gravity(p.gravity)

    _log(f"DOF map: {dof_map}")
    _log(f"bodies={m.body_count}  shapes={m.shape_count}  "
         f"joints={m.joint_count}  DOFs={m.joint_dof_count}")

    return m, dof_map


def build_point_scene(p: SceneParams):
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density)
    return _build_scene(p, cfg)


def build_cslc_scene(p: SceneParams):
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=p.ke, kd=p.kd, kf=p.kf, mu=p.mu, gap=0.002, density=p.pad_density,
        is_cslc=True,
        cslc_spacing=p.cslc_spacing, cslc_ka=p.cslc_ka, cslc_kl=p.cslc_kl,
        cslc_dc=p.cslc_dc, cslc_n_iter=p.cslc_n_iter, cslc_alpha=p.cslc_alpha)
    return _build_scene(p, cfg)


BUILDERS = {"point": build_point_scene, "cslc": build_cslc_scene}

SPHERE_BODY = 4
LEFT_PAD = 1
RIGHT_PAD = 3


# ── Pad trajectory ────────────────────────────────────────────────────────

def _pad_state(step, p: SceneParams):
    """Compute (dx_inward, dz_up) at a given step."""
    phase, s = p.phase_of(step)
    t = s * p.dt

    if phase == "APPROACH":
        return p.approach_speed * t, 0.0

    dx_app = p.approach_speed * p.approach_duration
    if phase == "SQUEEZE":
        return dx_app + p.squeeze_speed * t, 0.0

    dx_total = dx_app + p.squeeze_speed * p.squeeze_duration
    if phase == "LIFT":
        return dx_total, p.lift_speed * t

    return dx_total, p.lift_speed * p.lift_duration


def set_pad_targets(control, step, p: SceneParams, dof_map, debug=False):
    """Write joint position targets for both pads."""
    dx, dz = _pad_state(step, p)

    target = control.joint_target_pos.numpy()
    target[dof_map["left_x"]]  = +dx
    target[dof_map["left_z"]]  = +dz
    target[dof_map["right_x"]] = -dx
    target[dof_map["right_z"]] = +dz
    control.joint_target_pos.assign(
        wp.array(target, dtype=wp.float32, device=control.joint_target_pos.device))

    if debug:
        phase, _ = p.phase_of(step)
        gap = p.approach_gap - 2 * dx
        _log(f"[{phase:8s}] step={step:5d}  dx={dx*1e3:.2f}mm  dz={dz*1e3:.2f}mm  "
             f"gap={gap*1e3:.1f}mm  sphere_d={p.sphere_radius*2e3:.0f}mm")


# ── Helpers ───────────────────────────────────────────────────────────────

def count_active_contacts(contacts):
    n = int(contacts.rigid_contact_count.numpy()[0])
    return int(np.sum(contacts.rigid_contact_shape0.numpy()[:n] >= 0)) if n else 0


def read_cslc_state(model):
    pipeline = getattr(model, "_collision_pipeline", None)
    handler = getattr(pipeline, "cslc_handler", None) if pipeline else None
    if handler is None:
        return None
    d = handler.cslc_data
    is_surf = d.is_surface.numpy() == 1
    deltas = d.sphere_delta.numpy()[is_surf]
    pen = handler.raw_penetration.numpy()[is_surf]
    active = pen > 0
    return {
        "n_active": int(active.sum()), "n_surface": int(is_surf.sum()),
        "max_delta_mm": float(deltas.max()) * 1e3 if len(deltas) else 0,
        "max_pen_mm": float(pen.max()) * 1e3 if len(pen) else 0,
    }


def inspect_model(model, label=""):
    GEO = {0: "PLANE", 1: "MESH", 3: "SPHERE", 4: "CAPSULE", 7: "BOX"}
    _log(f"Model '{label}': {model.body_count} bodies, {model.shape_count} shapes, "
         f"{model.joint_count} joints, {model.joint_dof_count} DOFs")
    st, sf, sb = model.shape_type.numpy(), model.shape_flags.numpy(), model.shape_body.numpy()
    for i in range(model.shape_count):
        cslc = " [CSLC]" if sf[i] & CSLC_FLAG else ""
        _log(f"  shape {i}: {GEO.get(int(st[i]),'?')}  body={sb[i]}{cslc}", 1)


# ── Solver creation ───────────────────────────────────────────────────────

def make_solver(model, solver_name, p: SceneParams):
    if solver_name == "mujoco":
        if not HAS_MUJOCO:
            raise RuntimeError("MuJoCo solver not available")
        ncon = 5000
        if model.shape_cslc_spacing is not None:
            spacing = model.shape_cslc_spacing.numpy()
            flags = model.shape_flags.numpy()
            scale = model.shape_scale.numpy()
            for i in range(model.shape_count):
                if not (flags[i] & CSLC_FLAG):
                    continue
                sp = float(spacing[i])
                if sp <= 0:
                    continue
                hx, hy, hz = (float(scale[i][j]) for j in range(3))
                nx, ny, nz = (max(int(round(2*h/sp))+1, 2) for h in [hx, hy, hz])
                interior = max(nx-2,0)*max(ny-2,0)*max(nz-2,0)
                ncon += nx*ny*nz - interior
        return SolverMuJoCo(model, use_mujoco_contacts=False,
                            solver="newton", integrator="implicitfast",
                            iterations=100, ls_iterations=50,
                            njmax=ncon, nconmax=ncon)
    elif solver_name == "semi":
        return SolverSemiImplicit(model)
    raise ValueError(f"Unknown solver: {solver_name}")


# ── Headless runner ───────────────────────────────────────────────────────

def run_headless(name, model, solver, p, dof_map, verbose=True):
    met = Metrics(name=name)
    s0, s1 = model.state(), model.state()
    ctrl, con = model.control(), model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    is_cslc = "cslc" in name.lower()

    for step in range(p.total_steps):
        set_pad_targets(ctrl, step, p, dof_map)
        s0.clear_forces()
        model.collide(s0, con)
        solver.step(s0, s1, ctrl, con, p.dt)
        wp.synchronize()

        q = s1.body_q.numpy()
        sz = float(q[SPHERE_BODY, 2])
        nc = count_active_contacts(con)
        met.sphere_z.append(sz)
        met.contacts.append(nc)
        s0, s1 = s1, s0

        if verbose and ((step + 1) % 200 == 0 or step == p.total_steps - 1):
            phase, _ = p.phase_of(step)
            pad_z = float(q[LEFT_PAD, 2])
            line = (f"  {name:16s} {step+1:5d}/{p.total_steps}  [{phase:8s}]  "
                    f"sphere_z={sz:.5f}  pad_z={pad_z:.4f}  contacts={nc}")
            if is_cslc:
                ci = read_cslc_state(model)
                if ci:
                    line += f"  cslc={ci['n_active']}/{ci['n_surface']}"
            print(line)

    return met


def test_headless(p, solver_name):
    p.dump()
    results = []
    for cm in ["point", "cslc"]:
        label = f"{cm}_{solver_name}"
        _section(label.upper())
        model, dof_map = BUILDERS[cm](p)
        _ = model.contacts()
        inspect_model(model, label)
        solver = make_solver(model, solver_name, p)
        met = run_headless(label, model, solver, p, dof_map)
        results.append(met)
        _log(f"RESULT: max_z={met.max_z:.4f}  final_z={met.final_z:.4f}  "
             f"lifted={'YES' if met.lifted else 'NO'}  held={'YES' if met.held else 'NO'}")

    _section("SUMMARY")
    print(f"  {'Config':<24} {'Max Z':>8} {'Final Z':>8} {'Lifted':>8} {'Held':>8}")
    print(f"  {_SEP}")
    for m in results:
        print(f"  {m.name:<24} {m.max_z:8.4f} {m.final_z:8.4f} "
              f"{'YES':>8 if m.lifted else 'NO':>8} {'YES':>8 if m.held else 'NO':>8}")


# ── Viewer mode ───────────────────────────────────────────────────────────

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.test_mode = args.test
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(self.frame_dt / 0.002))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_step = 0

        self.contact_model = getattr(args, "contact_model", "cslc")
        self.solver_name = getattr(args, "solver", "mujoco")

        self.p = SceneParams(dt=self.sim_dt)
        self.p.dump()

        self.model, self.dof_map = BUILDERS[self.contact_model](self.p)
        inspect_model(self.model, self.contact_model)

        self.solver = make_solver(self.model, self.solver_name, self.p)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.current_z = self.p.sphere_start_z
        self.current_contacts = 0
        self.current_phase = "APPROACH"

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.3, -0.3, self.p.sphere_start_z + 0.15),
            pitch=-15.0, yaw=135.0)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

        q = self.state_0.body_q.numpy()
        _log(f"INITIAL STATE ({self.model.body_count} bodies):")
        for bi in range(self.model.body_count):
            _log(f"  body {bi}: pos=({q[bi,0]:.4f},{q[bi,1]:.4f},{q[bi,2]:.4f})", 1)
        _log(f"sim_substeps={self.sim_substeps}  sim_dt={self.sim_dt*1e3:.3f}ms")

    def simulate(self):
        for _ in range(self.sim_substeps):
            set_pad_targets(self.control, self.sim_step, self.p, self.dof_map,
                            debug=(self.sim_step % 500 == 0))
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_step += 1

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

        q = self.state_0.body_q.numpy()
        qd = self.state_0.body_qd.numpy()
        self.current_z = float(q[SPHERE_BODY, 2])
        self.current_contacts = count_active_contacts(self.contacts)
        self.current_phase, _ = self.p.phase_of(self.sim_step)

        if self.sim_step % (self.sim_substeps * 60) < self.sim_substeps:
            phase, _ = self.p.phase_of(self.sim_step)
            cslc = read_cslc_state(self.model) if self.contact_model == "cslc" else None
            cslc_str = f"  cslc={cslc['n_active']}/{cslc['n_surface']}" if cslc else ""
            _log(f"[{phase:8s}] step={self.sim_step:5d}  "
                 f"sphere_z={q[SPHERE_BODY,2]:+.5f}  "
                 f"vz={qd[SPHERE_BODY,5]:+.5f}  "
                 f"pad_z={q[LEFT_PAD,2]:.4f}  "
                 f"contacts={self.current_contacts}{cslc_str}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _render_ui(self, imgui):
        imgui.text(f"Contact: {self.contact_model}")
        imgui.text(f"Phase:   {self.current_phase}")
        imgui.text(f"Sphere Z: {self.current_z:.4f}")
        imgui.text(f"Contacts: {self.current_contacts}")
        imgui.text(f"Step:    {self.sim_step}")

    def test_final(self):
        if self.contact_model == "cslc":
            assert self.current_z > 0.01, f"Sphere fell: z={self.current_z:.4f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--contact-model", type=str, default="cslc",
                            choices=["point", "cslc"])
        parser.add_argument("--solver", type=str, default="mujoco",
                            choices=["mujoco", "semi"])
        parser.add_argument("--mode", type=str, default="viewer",
                            choices=["viewer", "headless"])
        return parser


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = Example.create_parser()
    args, _ = parser.parse_known_args()

    wp.init()
    print(f"\n{'━' * 60}\n  GRIPPER LIFT TEST — mode: {args.mode}\n{'━' * 60}")

    if args.mode == "viewer":
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
    else:
        args = parser.parse_args()
        sn = args.solver if HAS_MUJOCO or args.solver != "mujoco" else "semi"
        test_headless(SceneParams(), sn)


if __name__ == "__main__":
    main()