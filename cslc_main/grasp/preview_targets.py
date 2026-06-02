# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static GL still-life of the CSLC held-objects and their surface sampling.

Spawns the grasp targets — the tennis-ball sphere, the cube, and the
Stanford bunny mesh — resting side by side on the ground, and overlays
every surface sample point used by the CSLC point-set contact path:

  * sphere — Fibonacci-spiral samples via
    :func:`cslc_main.grasp.objects.make_sphere_target`.
  * cube   — uniform per-face tiling via
    :func:`cslc_main.grasp.objects.make_box_target` over **all 6 faces**.
  * bunny  — Lloyd/CVT samples on the mesh surface via
    ``point_cloud_utils.sample_mesh_lloyd`` (the same sampler the pad
    contact face uses), at a count matching the sphere's surface point
    density so the spacing reads consistently across all three objects.

The sample-point markers use the same colour
(``cslc_main.grasp.visualization.TargetPointsRenderer``'s at-rest grey
``[0.6, 0.6, 0.7]``) and size (the default box-pad lattice marker
radius) as the in-sim lift demo, so the still life matches what the
running grasp shows.

Nothing moves: each object is placed at its settled height and the
simulation is never stepped (no solver, no gravity), so the GL window
is a clean still life you can orbit, frame, and screenshot.

Run::

    uv run -m cslc_main.grasp.preview_targets --viewer gl

Tuning knobs (all optional)::

    --sphere-n-samples 600     # denser Fibonacci spiral on the sphere
    --box-face-pitch 0.003     # finer 3 mm tiling on every cube face
    --bunny-height 0.12        # scale the bunny to 12 cm tall [m]
    --separation 0.14          # push the objects further apart [m]
    --point-radius 0.0015      # bigger sample-point markers [m]
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

from . import objects
from .params import GraspConfig, ObjectParams

# Repo root (cslc_main/grasp/preview_targets.py -> ../../.. ), used to
# resolve the default bunny asset path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUNNY_OBJ = _REPO_ROOT / "assets" / "bunny" / "bunny.obj"

# Sample-point marker appearance, copied from the in-sim lift demo so the
# still life matches it:
#   * colour — TargetPointsRenderer's at-rest grey (visualization.py).
#   * radius — that renderer sizes markers at (pad lattice mean
#     nearest-neighbour spacing) x 0.25.  For the default box pad
#     (PadParams, n_samples=100 on the 40x40 mm face) the Lloyd spacing
#     is ~3.9 mm, so the marker radius is ~0.98 mm.  Hard-coded here to
#     keep the preview lightweight (no need to build a pad lattice just
#     to read one number); update if the pad defaults change.
_POINT_COLOR = (0.6, 0.6, 0.7)
_LIFT_DEMO_MARKER_RADIUS_M = 0.00098


class Example:
    """Newton-examples-compatible driver for the static target preview.

    Construction does all the work (build the model, sample every
    surface, pre-bake the marker buffers); :meth:`step` is a no-op and
    :meth:`render` just re-emits the static frame each tick.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        sep = float(getattr(args, "separation", 0.12))
        radius = float(getattr(args, "sphere_radius", 0.0335))
        n_sphere = int(getattr(args, "sphere_n_samples", 300))
        box_side = float(getattr(args, "box_side", 0.067))
        pitch = float(getattr(args, "box_face_pitch", 0.005))
        bunny_obj = Path(getattr(args, "bunny_obj", None) or _DEFAULT_BUNNY_OBJ)
        bunny_height = float(getattr(args, "bunny_height", 0.10))
        bunny_n = int(getattr(args, "bunny_n_samples", 0))  # 0 = auto
        self.point_radius = float(
            getattr(args, "point_radius", _LIFT_DEMO_MARKER_RADIUS_M)
        )

        # Default material / hydro knobs only — used to build the shape
        # configs; none of the dynamics knobs matter since we never step.
        cfg = GraspConfig()

        half = 0.5 * box_side
        sphere = ObjectParams(kind="sphere", radius=radius,
                              sphere_n_samples=n_sphere)
        box = ObjectParams(kind="box",
                           box_half_extents=(half, half, half),
                           box_face_pitch=pitch)

        # Lay the three objects out in a row along Y, sphere — cube —
        # bunny, all resting on the ground (centre at the object's
        # settled height).  Identity spawn rotation, so a body-local
        # point maps to world by a pure translation (used below to lift
        # the surface samples to world frame).
        sphere_spawn = np.array([0.0, -sep, sphere.settled_z_center], np.float32)
        box_spawn = np.array([0.0, 0.0, box.settled_z_center], np.float32)
        bunny_spawn = np.array([0.0, +sep, 0.0], np.float32)

        b = newton.ModelBuilder()
        b.add_ground_plane()

        # Sphere + cube via the tested object factory (free-joint bodies;
        # placed by eval_fk below and never stepped).
        for obj, spawn in ((sphere, sphere_spawn), (box, box_spawn)):
            shape_cfg = objects.make_object_shape_cfg(
                obj, cfg.material, cfg.hydro, "cslc"
            )
            objects.add_object(b, obj, shape_cfg,
                               (float(spawn[0]), float(spawn[1]), float(spawn[2])))

        # Bunny mesh: scale to the requested height, recentre in X/Y, and
        # drop its base onto the ground, then bake those into the mesh
        # vertices so the local frame already has the bunny resting at the
        # origin.  Attach it as static world geometry at ``bunny_spawn``.
        import trimesh  # local: banned at module level (TID253)

        bunny_tm = trimesh.load(str(bunny_obj), force="mesh")
        if not isinstance(bunny_tm, trimesh.Trimesh):
            raise RuntimeError(f"{bunny_obj} did not load as a single mesh")
        bunny_tm.apply_scale(bunny_height / float(bunny_tm.extents[2]))
        mn, mx = bunny_tm.bounds
        bunny_tm.apply_translation(
            [-0.5 * (mn[0] + mx[0]), -0.5 * (mn[1] + mx[1]), -mn[2]]
        )
        bunny_mesh = newton.Mesh(
            bunny_tm.vertices.astype(np.float32),
            bunny_tm.faces.astype(np.int32).flatten(),
        )
        b.add_shape_mesh(
            body=-1,  # world: static, no mass/joint needed
            mesh=bunny_mesh,
            xform=wp.transform(
                (float(bunny_spawn[0]), float(bunny_spawn[1]), float(bunny_spawn[2])),
                wp.quat_identity(),
            ),
            cfg=newton.ModelBuilder.ShapeConfig(),
            label="bunny_mesh",
        )

        self.model = b.finalize()

        # Place the free-joint bodies (sphere, cube) at their spawn poses
        # via forward kinematics and leave them there — the scene is
        # intentionally static.
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # Sample every surface (body-local) and lift to world.  Markers
        # are nudged one marker-radius along the outward normal so the
        # whole marker sits ON the surface instead of half-buried in it.
        sphere_pts = objects.make_sphere_target(radius, n_sphere)
        box_pts = objects.make_box_target((half, half, half), pitch)  # all 6 faces
        if bunny_n <= 0:
            # Match the sphere's surface point density (points / m^2) so
            # the inter-point spacing reads the same across all objects.
            sphere_density = n_sphere / (4.0 * math.pi * radius * radius)
            bunny_n = max(50, int(round(sphere_density * float(bunny_tm.area))))
        bunny_pts = _sample_mesh_surface(bunny_tm, bunny_n)

        self._point_groups = []
        for name, samples, spawn in (
            ("/sphere_points", sphere_pts, sphere_spawn),
            ("/box_points", box_pts, box_spawn),
            ("/bunny_points", bunny_pts, bunny_spawn),
        ):
            world = (samples["positions"]
                     + samples["normals"] * self.point_radius
                     + spawn)
            self._point_groups.append((name, self._make_buffers(world)))

        self.viewer.set_model(self.model)
        # Aim the camera at the centre of the row from the front (-X),
        # slightly above, so all three objects sit across the frame.
        cam_pos = (-0.45, 0.0, 0.24)
        target = (0.0, 0.0, 0.05)
        pitch_deg, yaw_deg = _aim_pitch_yaw(cam_pos, target)
        self.viewer.set_camera(pos=wp.vec3(*cam_pos), pitch=pitch_deg, yaw=yaw_deg)

        print(
            f"  preview: sphere {sphere_pts['positions'].shape[0]} pts, "
            f"cube {box_pts['positions'].shape[0]} pts (all 6 faces), "
            f"bunny {bunny_pts['positions'].shape[0]} pts"
        )

    def _make_buffers(self, world_pts: np.ndarray) -> tuple:
        """Pack world-frame point positions into the (xforms, colors, mats)
        Warp arrays that :meth:`newton.viewer.Viewer.log_shapes` consumes."""
        n = world_pts.shape[0]
        xf = np.zeros((n, 7), np.float32)
        xf[:, :3] = world_pts
        xf[:, 6] = 1.0  # identity quaternion
        colors = np.tile(_POINT_COLOR, (n, 1)).astype(np.float32)
        mats = np.tile([0.4, 0.2, 0.0, 0.0], (n, 1)).astype(np.float32)
        return (
            wp.array(xf, dtype=wp.transform),
            wp.array(colors, dtype=wp.vec3),
            wp.array(mats, dtype=wp.vec4),
        )

    def step(self) -> None:
        pass  # static still-life — no physics

    def render(self) -> None:
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.state)
        for name, (xf, colors, mats) in self._point_groups:
            self.viewer.log_shapes(
                name, newton.GeoType.SPHERE, self.point_radius, xf, colors, mats,
            )
        self.viewer.end_frame()

    def test_final(self) -> None:
        pass  # nothing to assert — this is a visualisation-only scene


def _sample_mesh_surface(tm: trimesh.Trimesh, n: int) -> dict[str, np.ndarray]:
    """Lloyd/CVT-sample a mesh surface; return ``{positions, normals}``.

    Mirrors :func:`cslc_main.grasp.pads.sample_pad_contact_face`: draw
    ``n`` centroidal-Voronoi samples with ``point_cloud_utils`` and
    recover each sample's outward normal from the nearest triangle.
    """
    import point_cloud_utils as pcu  # local: heavy optional dep
    import trimesh  # local: banned at module level (TID253)

    v = np.asarray(tm.vertices, dtype=np.float64)
    f = np.asarray(tm.faces, dtype=np.int32)
    pts = np.asarray(pcu.sample_mesh_lloyd(v, f, int(n)))
    _, _, tri_id = trimesh.proximity.closest_point(tm, pts)
    normals = np.asarray(tm.face_normals)[tri_id]
    return {
        "positions": pts.astype(np.float32),
        "normals": normals.astype(np.float32),
    }


def _aim_pitch_yaw(pos, target) -> tuple[float, float]:
    """Pitch/yaw [deg] that aims a Z-up camera at ``pos`` toward ``target``.

    Matches the GL viewer's Z-up convention (see
    ``newton._src.viewer.camera.Camera.get_front``): the front vector is
    ``(cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))``, so the
    inverse is ``pitch = asin(dz)`` and ``yaw = atan2(dy, dx)``.
    """
    d = np.asarray(target, np.float64) - np.asarray(pos, np.float64)
    d /= np.linalg.norm(d)
    pitch = math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return pitch, yaw


def main() -> None:
    parser = newton.examples.create_parser()
    g = parser.add_argument_group("Target preview")
    g.add_argument("--sphere-radius", type=float, default=0.0335,
                   help="Sphere radius [m] (default 33.5 mm, tennis ball).")
    g.add_argument("--sphere-n-samples", type=int, default=300,
                   help="Fibonacci-spiral sample count on the sphere "
                        "(default 300, matches the grasp/lift-demo default).")
    g.add_argument("--box-side", type=float, default=0.067,
                   help="Cube full side length [m] (default 67 mm, matches "
                        "the sphere's bounding box).")
    g.add_argument("--box-face-pitch", type=float, default=0.005,
                   help="Sample pitch [m] on each cube face (default 5 mm, "
                        "matches the grasp/lift-demo default).")
    g.add_argument("--bunny-obj", type=str, default=str(_DEFAULT_BUNNY_OBJ),
                   help="Path to the bunny mesh OBJ.")
    g.add_argument("--bunny-height", type=float, default=0.10,
                   help="Scale the bunny so it stands this tall [m] "
                        "(default 100 mm).")
    g.add_argument("--bunny-n-samples", type=int, default=0,
                   help="Lloyd sample count on the bunny surface "
                        "(default 0 = auto: match the sphere's point density).")
    g.add_argument("--separation", type=float, default=0.12,
                   help="Spacing [m] between adjacent objects along Y "
                        "(default 120 mm).")
    g.add_argument("--point-radius", type=float,
                   default=_LIFT_DEMO_MARKER_RADIUS_M,
                   help="Rendered sample-point marker radius [m] (default "
                        "matches the lift demo's ~0.98 mm).")

    wp.init()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
