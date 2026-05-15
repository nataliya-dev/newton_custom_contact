# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UXPBD Lift Test
#
# Two articulated gripper pads close on a free shape-matched (SM-rigid)
# object, squeeze, then lift it together against gravity. Exercises the
# friction-driven grasp path under SolverUXPBD:
#
#   - Each pad is a rigid body driven by two prismatic joints
#     (X = approach axis, Z = lift axis), with a thin collision box for
#     inertia and a kinematic lattice that does the actual contact work.
#   - The object is a free SM-rigid sphere-packed body added via
#     add_particle_volume; its rigidity is maintained by the SRXPBD
#     shape-matching pass each iteration.
#   - Contact between pad lattice particles and object particles is
#     handled by solve_particle_particle_contacts_uxpbd, which applies
#     position-level Coulomb friction. Slip resistance during the LIFT
#     phase is the test's load-bearing physical behaviour.
#
# Phases (Z up):
#
#       ┌─┐         ┌─┐
#       │L│ ◄────── │R│      APPROACH    pads move inward
#       └─┘         └─┘
#       ┌─┐ ┌───┐ ┌─┐
#       │L│ │obj│ │R│        SQUEEZE     pads press a few mm in
#       └─┘ └───┘ └─┘
#         ↑       ↑
#       ┌─┐ ┌───┐ ┌─┐        LIFT        pads (and the gripped object)
#       │L│ │obj│ │R│                    rise together
#       └─┘ └───┘ └─┘
#       ════════════         HOLD        pads stationary in the air
#
# Adapted from cslc_mujoco/lift_test.py without the MuJoCo / CSLC /
# hydroelastic comparisons. UXPBD-only, single pipeline.
#
# Command: python -m newton.examples uxpbd_lift_test
###########################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode


@dataclass
class SceneParams:
    """All knobs for the gripper lift scene."""

    # --- Object (SM-rigid sphere-packed cube grasped by the pads) ---
    # Identical to the cube in example_uxpbd_lattice_stack: 4x4x4 sphere
    # packing inscribed in a CUBE_HALF_EXTENT cube, mass from default
    # density (1000 kg/m^3 * (2*0.04)^3 = 0.512 kg).
    obj_half_extent: float = 0.04
    obj_sphere_r: float = 0.012
    obj_n: int = 4
    obj_mass: float = 0.512

    # --- Pads (thin boxes; lattice baked into the volume) ---
    # Pad face (Y, Z extents) matches the object's half-extent so the
    # lattice covers the full face. The pad is intentionally thin along
    # the approach axis: pad_lattice_sphere_r > pad_hx so the lattice
    # spheres protrude past the pad's inner X face. This is required
    # functionally (SolverUXPBD has no shape-vs-shape contact path; the
    # collision box is purely for inertia) and visually (the spheres
    # poke through the rendered box surface, so they're easy to see).
    pad_hx: float = 0.01
    pad_hy: float = 0.04
    pad_hz: float = 0.04
    # Slightly bigger than obj_sphere_r so the lattice protrudes past
    # the pad mesh and engages object particles even under tiny squeeze
    # depths.
    pad_lattice_sphere_r: float = 0.014
    # nx=1 puts a single sphere layer at the pad's X center plane; with
    # pad_lattice_sphere_r > pad_hx the sphere extends past both X faces.
    pad_lattice_nx: int = 1
    pad_lattice_ny: int = 4
    pad_lattice_nz: int = 4
    # Pad body spawn height. Picked so the lattice bottom sits ~2 cm
    # above ground (no dragging). With pad_hz = 0.04 and
    # pad_lattice_sphere_r = 0.014:
    #   lattice bottom_z = pad_z0 - (pad_hz - r) - r = pad_z0 - pad_hz.
    pad_z0: float = 0.06

    # --- Phase timing ---
    # APPROACH: pads move inward from `approach_gap` to the object surface.
    # SQUEEZE:  pads press an additional `squeeze_depth` past the surface.
    # LIFT:     pads rise together with a smooth velocity ramp.
    # HOLD:     pads stationary at the lifted height.
    approach_gap: float = 0.14
    approach_duration: float = 1.0
    squeeze_depth: float = 0.002
    squeeze_duration: float = 0.5
    lift_speed: float = 0.015
    lift_duration: float = 1.5
    # Smooth the LIFT velocity transition over this duration. Without
    # ramping, the target velocity jumps 0 -> lift_speed in a single
    # timestep; the high-gain PD drive turns that into a near-impulsive
    # pad velocity which saturates friction against the static object
    # and shoots it upward.
    lift_ramp_duration: float = 0.25
    hold_duration: float = 0.5

    # --- Friction ---
    # mu_eff in the kernels = 0.5 * (particle_mu + shape_material_mu).
    # Setting both ends to the same value yields exactly this mu_eff.
    mu: float = 0.5

    # --- Joint drive ---
    drive_ke: float = 5.0e4   # position stiffness
    drive_kd: float = 1.0e3   # velocity damping

    # --- Integration ---
    fps: int = 100
    sim_substeps: int = 16
    solver_iterations: int = 6

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    @property
    def sim_dt(self) -> float:
        return self.frame_dt / self.sim_substeps

    @property
    def approach_speed(self) -> float:
        # Travel = (approach_gap/2) - obj_half_extent in `approach_duration`.
        travel = (self.approach_gap / 2.0) - self.obj_half_extent
        return travel / self.approach_duration

    @property
    def squeeze_speed(self) -> float:
        return self.squeeze_depth / self.squeeze_duration

    @property
    def total_frames(self) -> int:
        return int((self.approach_duration + self.squeeze_duration
                    + self.lift_duration + self.hold_duration) * self.fps)


def _pad_target_xz(step: int, p: SceneParams) -> tuple[float, float]:
    """Compute (dx_inward, dz_up) at a given (sub)step index.

    dx_inward and dz_up are signed offsets that get mirrored onto each
    pad (left pad uses +dx, right pad uses -dx; both use +dz).
    """
    t = step * p.sim_dt

    if t < p.approach_duration:
        return p.approach_speed * t, 0.0
    t -= p.approach_duration

    dx_app = p.approach_speed * p.approach_duration
    if t < p.squeeze_duration:
        return dx_app + p.squeeze_speed * t, 0.0
    t -= p.squeeze_duration

    dx_total = dx_app + p.squeeze_speed * p.squeeze_duration

    def _lift_dz(t_lift: float) -> float:
        # C1-smooth velocity ramp at start of LIFT: v(s) = lift_speed * (3s^2 - 2s^3),
        # integrated to z(s) = lift_speed * ramp * (s^3 - s^4/2).
        ramp = p.lift_ramp_duration
        if ramp <= 0.0 or t_lift >= ramp:
            return p.lift_speed * max(t_lift - 0.5 * ramp, 0.0) if t_lift >= ramp else 0.0
        s = t_lift / ramp
        return p.lift_speed * ramp * (s ** 3 - 0.5 * s ** 4)

    if t < p.lift_duration:
        return dx_total, _lift_dz(t)
    # HOLD: freeze at end-of-LIFT pose.
    return dx_total, _lift_dz(p.lift_duration)


class Example:
    def __init__(self, viewer, args):
        self.p = SceneParams()
        self.frame_dt = self.p.frame_dt
        self.sim_substeps = self.p.sim_substeps
        self.sim_dt = self.p.sim_dt
        self.sim_time = 0.0
        self.sim_step = 0       # counts substeps, not frames
        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_ground_plane()

        # ----- Object: SM-rigid sphere-packed cube, free-floating ------
        # Spawn at the object's rest height (bottom particle bottom_z = 0
        # touches ground). This avoids a long free fall before the pads
        # close on the object.
        #   bottom particle center_z (body frame) = -(h - r) = -0.028
        #   ground contact at obj_z - 0.028 - r = 0  =>  obj_z = h = 0.04.
        obj_z = self.p.obj_half_extent
        obj_coords = np.linspace(
            -self.p.obj_half_extent + self.p.obj_sphere_r,
            self.p.obj_half_extent - self.p.obj_sphere_r,
            self.p.obj_n,
        )
        oxs, oys, ozs = np.meshgrid(
            obj_coords, obj_coords, obj_coords, indexing="ij")
        obj_centers = np.stack(
            [oxs.flatten(), oys.flatten(), ozs.flatten()], axis=1).astype(np.float32)
        obj_radii = np.full(
            obj_centers.shape[0], self.p.obj_sphere_r, dtype=np.float32)
        self.obj_group = builder.add_particle_volume(
            volume_data={"centers": obj_centers.tolist(),
                         "radii": obj_radii.tolist()},
            total_mass=self.p.obj_mass,
            pos=wp.vec3(0.0, 0.0, obj_z),
        )

        # ----- Two articulated pads -----------------------------------
        # Each pad: world --[prismatic X]--> slider --[prismatic Z]--> pad.
        # Slider is a massless intermediate link (no collision).
        # The pad body carries the contact box AND the kinematic lattice.
        #
        # Pads start at x = +/- (approach_gap/2 + pad_hx) so the pad inner
        # face is at +/- approach_gap/2 at t=0, then prismatic-X moves
        # them inward by `dx` from _pad_target_xz.
        lx0 = -(self.p.approach_gap / 2.0 + self.p.pad_hx)
        rx0 = +(self.p.approach_gap / 2.0 + self.p.pad_hx)
        pad_z0 = self.p.pad_z0

        # Lattice particle layout (in pad body frame). nx=1 places a
        # single sphere layer at the pad's X center (the natural choice
        # when sphere_r >= pad_hx, which is the case here -- np.linspace
        # would produce an inverted interval otherwise).
        if self.p.pad_lattice_nx == 1:
            lx_c = np.array([0.0], dtype=np.float32)
        else:
            lx_c = np.linspace(
                -self.p.pad_hx + self.p.pad_lattice_sphere_r,
                self.p.pad_hx - self.p.pad_lattice_sphere_r,
                self.p.pad_lattice_nx,
            )
        ly_c = np.linspace(
            -self.p.pad_hy + self.p.pad_lattice_sphere_r,
            self.p.pad_hy - self.p.pad_lattice_sphere_r,
            self.p.pad_lattice_ny,
        )
        lz_c = np.linspace(
            -self.p.pad_hz + self.p.pad_lattice_sphere_r,
            self.p.pad_hz - self.p.pad_lattice_sphere_r,
            self.p.pad_lattice_nz,
        )
        lxs, lys, lzs = np.meshgrid(lx_c, ly_c, lz_c, indexing="ij")
        pad_centers = np.stack(
            [lxs.flatten(), lys.flatten(), lzs.flatten()], axis=1).astype(np.float32)
        pad_radii = np.full(
            pad_centers.shape[0], self.p.pad_lattice_sphere_r, dtype=np.float32)

        # Ghost config for the slider's stub geometry (no collision, no mass).
        ghost_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=False,
            has_particle_collision=False,
            density=0.0,
        )

        self.dof = {}            # label -> qd index
        self.pad_bodies = []     # [left_pad_body, right_pad_body]

        for label, x0 in [("left", lx0), ("right", rx0)]:
            slider = builder.add_link(
                xform=wp.transform((x0, 0.0, pad_z0), wp.quat_identity()),
                mass=0.01,
                label=f"{label}_slider",
            )
            builder.add_shape_sphere(slider, radius=0.001, cfg=ghost_cfg)

            # mass=0.0: the shape box density carries both mass and inertia
            # consistently (see example_uxpbd_lattice_stack.py NOTE for the
            # add_body(mass=m) + add_shape_box double-counting gotcha).
            pad = builder.add_link(
                xform=wp.transform((x0, 0.0, pad_z0), wp.quat_identity()),
                mass=0.0,
                label=f"{label}_pad",
            )
            builder.add_shape_box(
                pad,
                hx=self.p.pad_hx, hy=self.p.pad_hy, hz=self.p.pad_hz,
            )
            builder.add_lattice(
                link=pad,
                morphit_json={"centers": pad_centers, "radii": pad_radii},
                total_mass=0.0,
                pos=wp.vec3(x0, 0.0, pad_z0),
            )

            j_x = builder.add_joint_prismatic(
                parent=-1, child=slider,
                axis=wp.vec3(1.0, 0.0, 0.0),
                parent_xform=wp.transform(
                    (x0, 0.0, pad_z0), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                label=f"{label}_x",
            )
            j_z = builder.add_joint_prismatic(
                parent=slider, child=pad,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform_identity(),
                child_xform=wp.transform_identity(),
                label=f"{label}_z",
            )
            builder.add_articulation([j_x, j_z], label=f"{label}_arm")

            self.dof[f"{label}_x"] = builder.joint_qd_start[j_x]
            self.dof[f"{label}_z"] = builder.joint_qd_start[j_z]
            self.pad_bodies.append(pad)

            for ji in (j_x, j_z):
                dof = builder.joint_qd_start[ji]
                builder.joint_target_ke[dof] = self.p.drive_ke
                builder.joint_target_kd[dof] = self.p.drive_kd
                builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()
        # Override friction params (kernel uses
        # mu_eff = 0.5 * (particle_mu + shape_material_mu)).
        self.model.particle_mu = self.p.mu
        self.model.soft_contact_mu = self.p.mu
        self.model.shape_material_mu.assign(
            np.full(self.model.shape_count, self.p.mu, dtype=np.float32))

        self.solver = newton.solvers.SolverUXPBD(
            self.model, iterations=self.p.solver_iterations)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        obj_idx = self.model.particle_groups[self.obj_group]
        if hasattr(obj_idx, "numpy"):
            obj_idx = obj_idx.numpy()
        self._obj_idx = np.asarray(list(obj_idx), dtype=np.int32)

        self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(
            pos=wp.vec3(0.3, -0.3, pad_z0 + 0.10),
            pitch=-15.0, yaw=135.0,
        )

        # Initial state snapshot (used by test_final to measure slip and lift).
        self._obj_z0 = float(self.state_0.particle_q.numpy()[
                             self._obj_idx, 2].mean())
        self._pad_z0 = float(self.state_0.body_q.numpy()
                             [self.pad_bodies[0], 2])

    def _set_pad_targets(self):
        """Write the prismatic-joint position targets for both pads."""
        dx, dz = _pad_target_xz(self.sim_step, self.p)
        target = self.control.joint_target_pos.numpy()
        target[self.dof["left_x"]] = +dx
        target[self.dof["left_z"]] = +dz
        target[self.dof["right_x"]] = -dx
        target[self.dof["right_z"]] = +dz
        self.control.joint_target_pos.assign(
            wp.array(target, dtype=wp.float32,
                     device=self.control.joint_target_pos.device))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._set_pad_targets()
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_step += 1

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # After the full APPROACH -> SQUEEZE -> LIFT -> HOLD cycle, the
        # object should be lifted with the pads. We check:
        #   1. The object did not fall (its z is well above its spawn).
        #   2. Slip between object and pads is bounded (the object tracked
        #      the pad's vertical motion within a tolerance).
        pad_z = float(self.state_0.body_q.numpy()[self.pad_bodies[0], 2])
        obj_z = float(self.state_0.particle_q.numpy()[self._obj_idx, 2].mean())

        pad_lift = pad_z - self._pad_z0
        obj_lift = obj_z - self._obj_z0
        # Slip = pad-frame z drift of the object. Positive means the
        # object lagged behind the pad (slipped down through the grip).
        slip = pad_lift - obj_lift

        # 1. Object did not fall through.
        assert obj_z > self._obj_z0 - 0.005, (
            f"Object dropped: obj_z={obj_z:.4f}, started at {self._obj_z0:.4f}"
        )
        # 2. The pads did move up (sanity check on the drive).
        assert pad_lift > 0.5 * self.p.lift_speed * self.p.lift_duration, (
            f"Pads did not lift: pad_lift={pad_lift:.4f}, "
            f"expected ~{self.p.lift_speed * self.p.lift_duration:.4f}"
        )
        # 3. Slip stays under a few mm (loose tolerance: position-based
        # friction in UXPBD will leak some tangential motion per iteration).
        assert abs(slip) < 0.01, (
            f"Object slipped too much: slip={slip*1e3:.2f} mm "
            f"(pad_lift={pad_lift*1e3:.2f} mm, obj_lift={obj_lift*1e3:.2f} mm)"
        )

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
