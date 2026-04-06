"""
demo_dimple.py — Visible CSLC dimple: sphere pressed into a pad.

A small rigid sphere pushes into the centre of a CSLC pad.
Only centre spheres are compressed; lateral coupling spreads
the deformation outward, creating a visible dimple.

No dynamics needed — just quasi-static deformation updated each frame
as the indenter moves closer.

Usage:
  cd cslc_scratch && uv run python demo_dimple.py --viewer gl
"""

import sys, os
import numpy as np
import warp as wp
import newton
import newton.examples

sys.path.insert(0, os.path.dirname(__file__))
from cslc import solve_contact_local
from sphere_body import create_finger_pad

# ═══════════════════════════════════════════════════════════════════════
#  Parameters — pad lies flat, deformation goes UP
# ═══════════════════════════════════════════════════════════════════════

PAD_NY, PAD_NZ = 15, 15
PAD_SPACING = 0.08
PAD_RADIUS_FACTOR = 0.45   # smaller → visible gaps between spheres
PAD_RADIUS = PAD_SPACING * PAD_RADIUS_FACTOR  # 36mm
N_SPHERES = PAD_NY * PAD_NZ

KA = 200.0
KL = 100.0
KC = 150.0

INDENTER_RADIUS = 0.12
INDENTER_START_Z = 0.3     # starts above pad
INDENTER_END_Z = -0.06     # pushes into pad
INDENT_DURATION = 3.0
HOLD_DURATION = 2.0

INDENTER_START_Y = 0.15   # starts 150mm away from pad surface
INDENTER_END_Y = -0.05    # pushes 50mm past pad surface

DEFORM_SCALE = 5.0

# Pad lies flat in XY plane at z=1.0, deformation along +z
PAD_CENTRE = np.array([0.0, 0.0, 1.0])


class Example:
    def __init__(self, viewer, args):
            self.fps = 30
            self.frame_dt = 1.0 / self.fps
            self.sim_substeps = 1
            self.sim_dt = self.frame_dt
            self.sim_time = 0.0
            self.viewer = viewer
            self.max_time = 1000 #INDENT_DURATION + HOLD_DURATION + 2.0

            self.pad = create_finger_pad(PAD_NY, PAD_NZ, PAD_SPACING, PAD_RADIUS_FACTOR)
            self.pad_local = self.pad['positions_3d'].copy()
            self.prev_delta = None

            # Pad lies flat in XY plane at z = PAD_CENTRE[2]
            self.rest_pos = np.zeros((N_SPHERES, 3))
            self.rest_pos[:, 0] = self.pad_local[:, 1] + PAD_CENTRE[0]  # pad_y → x
            self.rest_pos[:, 1] = self.pad_local[:, 2] + PAD_CENTRE[1]  # pad_z → y
            self.rest_pos[:, 2] = PAD_CENTRE[2]                          # flat at z

            builder = self._build_scene()
            self.model = builder.finalize(device="cpu")
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=1)
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.contacts = self.model.contacts()
            self.viewer.set_model(self.model)

            print(f"  Pad: {PAD_NY}×{PAD_NZ} = {N_SPHERES} spheres")
            print(f"  Sphere radius: {PAD_RADIUS*1000:.0f}mm")
            print(f"  Pad size: {(PAD_NY-1)*PAD_SPACING*1000:.0f}mm × "
                f"{(PAD_NZ-1)*PAD_SPACING*1000:.0f}mm")

    def _build_scene(self):
            builder = newton.ModelBuilder(gravity=0.0)

            # Indenter starts above pad
            ind_pos = [PAD_CENTRE[0], PAD_CENTRE[1],
                    PAD_CENTRE[2] + INDENTER_START_Z + INDENTER_RADIUS]
            self.body_indenter = builder.add_body(
                xform=wp.transform(p=wp.vec3(*ind_pos), q=wp.quat_identity()),
                is_kinematic=True, mass=1.0, label="indenter")
            builder.add_shape_sphere(self.body_indenter, radius=INDENTER_RADIUS)

            # Pad particles
            self.particle_offset = 0
            for i in range(N_SPHERES):
                builder.add_particle(
                    pos=wp.vec3(*self.rest_pos[i].tolist()),
                    vel=wp.vec3(0, 0, 0),
                    mass=0.001, radius=PAD_RADIUS,
                    flags=newton.ParticleFlags.ACTIVE)

            return builder

    def _indenter_y(self):
        """Current indenter surface y position (animated)."""
        if self.sim_time < INDENT_DURATION:
            # Linear approach
            t = self.sim_time / INDENT_DURATION
            return INDENTER_START_Y + t * (INDENTER_END_Y - INDENTER_START_Y)
        else:
            return INDENTER_END_Y
        

    def _indenter_z(self):
        """Current indenter surface z position."""
        if self.sim_time < INDENT_DURATION:
            t = self.sim_time / INDENT_DURATION
            return INDENTER_START_Z + t * (INDENTER_END_Z - INDENTER_START_Z)
        else:
                return INDENTER_END_Z

    def _compute_pen_per_sphere(self, indenter_z):
        """Per-sphere penetration from indenter above pad."""
        indenter_centre = PAD_CENTRE.copy()
        indenter_centre[2] = PAD_CENTRE[2] + indenter_z + INDENTER_RADIUS

        diff = self.rest_pos - indenter_centre
        dist = np.linalg.norm(diff, axis=1)
        pen = (INDENTER_RADIUS + PAD_RADIUS) - dist
        return np.maximum(pen, 0.0)

    def _solve_deformation(self):
        """Run CSLC and return deformed positions."""
        indenter_z = self._indenter_z()
        pen = self._compute_pen_per_sphere(indenter_z)

        sol = solve_contact_local(
            self.pad, KA, KL, KC, pen,
            n_iter=80, alpha=0.30,
            warm_start=self.prev_delta)
        self.prev_delta = sol['delta_x']

        # Deformed: push DOWN (−z) when compressed
        deformed = self.rest_pos.copy()
        deformed[:, 2] -= sol['delta_x'] * DEFORM_SCALE

        self._n_contacts = sol['n_contacts']
        self._max_delta = np.max(np.abs(sol['delta_x']))
        self._total_force = sol['total_force']
        self._pen = pen
        return deformed

    def _update_particles(self, deformed):
        pq = self.state_0.particle_q.numpy()
        pq[self.particle_offset:self.particle_offset + N_SPHERES] = deformed.astype(np.float32)
        self.state_0.particle_q.assign(
            wp.array(pq, dtype=wp.vec3f, device="cpu"))
        # Zero velocities
        pqd = np.zeros((self.state_0.particle_count, 3), dtype=np.float32)
        self.state_0.particle_qd.assign(
            wp.array(pqd, dtype=wp.vec3f, device="cpu"))

    def _update_indenter(self):
        """Move kinematic indenter."""
        indenter_z = self._indenter_z()
        bq = self.state_0.body_q.numpy()
        bq[self.body_indenter][:3] = [
            PAD_CENTRE[0],
            PAD_CENTRE[1],
            PAD_CENTRE[2] + indenter_z + INDENTER_RADIUS]
        self.state_0.body_q.assign(
            wp.array(bq, dtype=wp.transformf, device="cpu"))

    def simulate(self):
        pass  # no real physics — just quasi-static CSLC

    def step(self):
        self.sim_time += self.frame_dt

        # Move indenter
        self._update_indenter()

        # Solve deformation
        deformed = self._solve_deformation()

        # Update particle visuals
        self._update_particles(deformed)

        n_in_contact = np.sum(self._pen > 0)
        if int(self.sim_time * self.fps) % 5 == 0:
            print(f"  t={self.sim_time:.2f}s  "
                  f"indenter_y={self._indenter_y()*1000:.0f}mm  "
                  f"touching={n_in_contact}/{N_SPHERES}  "
                  f"max_delta={self._max_delta*1000:.1f}mm  "
                  f"visual={self._max_delta*1000*DEFORM_SCALE:.0f}mm  "
                  f"F={self._total_force:.1f}N")

        if self.sim_time >= self.max_time:
            self.viewer.close()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)