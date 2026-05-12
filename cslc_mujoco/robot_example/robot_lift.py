'''
uv run cslc_v1/robot_example/robot_lift.py --contact-model cslc
uv run cslc_v1/robot_example/robot_lift.py --contact-model hydro --start-gripped
uv run cslc_v1/robot_example/robot_lift.py --headless --contact-models hydro point cslc
'''

import time
import numpy as np
import warp as wp

from cslc_v1.robot_example.utils import  \
                    find_body_in_builder, get_sphere_cfg_not_hyrdo, get_cslc_pad_cfg, get_hydro_pad_cfg, \
                    point_pad_cfg, get_sphere_cfg_hydro, make_solver, \
                    SimDiagnostics, count_active_contacts, read_cslc_state, \
                    inspect_model, recalibrate_cslc_kc_per_pad, \
                    _log, _section

from cslc_v1.robot_example.config import SceneParams, TaskType, LiftMetrics


import newton
import newton.examples
import newton.ik as ik


def make_scene_params(args, sim_dt: float) -> SceneParams:
    return SceneParams(
        dt=sim_dt,
        start_gripped=getattr(args, "start_gripped", False),
    )


def run_headless(name: str, example, verbose: bool = True) -> LiftMetrics:
    """Run an already-built :class:`Example` headless for one full task
    schedule, collecting per-frame sphere z and active contact count.

    The frame budget comes from ``task_time_soft_limits`` summed over the
    (possibly filtered by ``--start-gripped``) schedule, plus a small
    margin so the last task gets a chance to fully reach its target.
    """
    met = LiftMetrics(name=name)
    total_seconds = float(sum(example.task_time_soft_limits.numpy()))
    total_frames = int(total_seconds * example.fps) + 60

    sphere_body_idx = example.sim_diag.sphere_body_idx
    pad_body_idx = example.sim_diag.pad_body_idx

    t_start = time.perf_counter()
    for frame in range(total_frames):
        example.step()

        q = example.state_0.body_q.numpy()
        sz = float(q[sphere_body_idx, 2])
        nc = count_active_contacts(example.contacts)
        met.sphere_z.append(sz)
        met.contacts.append(nc)

        if verbose and ((frame + 1) % 60 == 0 or frame == total_frames - 1):
            pad_z = float(q[pad_body_idx, 2])
            task_idx = int(example.task_idx.numpy()[0])
            if 0 <= task_idx < len(example._schedule_task_types):
                phase = example._schedule_task_types[task_idx].name
            else:
                phase = f"TASK_{task_idx}"
            line = (f"  {name:16s} {frame+1:5d}/{total_frames}  [{phase:8s}]  "
                    f"sphere_z={sz:.5f}  pad_z={pad_z:.4f}  contacts={nc}")
            if example.contact_model == "cslc":
                ci = read_cslc_state(example.model)
                if ci:
                    line += f"  cslc={ci['n_active']}/{ci['n_surface']}"
            print(line)

    wall = time.perf_counter() - t_start
    per_step_ms = 1000.0 * wall / max(total_frames, 1)
    sim_time_s = total_frames * example.frame_dt
    rtx = sim_time_s / wall if wall > 0 else 0.0
    print(
        f"  TIMING {name:16s}  wall={wall:.3f}s  "
        f"per-step={per_step_ms:.3f}ms  realtime×={rtx:.2f}"
    )
    return met


def test_headless(args, contact_models: list[str] | None = None) -> list[LiftMetrics]:
    """Build & run the robot lift scene headlessly for each contact model,
    then print a side-by-side summary.

    Mirrors ``lift_test.test_headless`` end-to-end:
      1. ``SceneParams.dump()`` once before the loop.
      2. ``inspect_model(model, label)`` per iteration.
      3. ``recalibrate_cslc_kc_per_pad(...)`` for the cslc run when
         ``cslc_contact_fraction`` is set.
    """
    # Dump scene params once at the top, using the *exact* same constructor
    # Example will use, so the dump can never drift from the simulated params.
    make_scene_params(args, sim_dt=(1.0 / 60) / 10).dump()

    results: list[LiftMetrics] = []
    for cm in contact_models:
        label = f"{cm}_{args.solver}"
        _section(label.upper())
        args.contact_model = cm
        args.headless = True
        viewer = newton.viewer.ViewerNull()
        example = Example(viewer, args)

        # Body / shape / joint summary, with [CSLC]-flagged shapes annotated.
        inspect_model(example.model, label)

        # CSLC stiffness recalibration (matches lift_test.py): rescale per-
        # sphere kc so each pad's aggregate stiffness equals ke_bulk for the
        # actual active-patch size in this scene.
        if cm == "cslc" and example.scene_params.cslc_contact_fraction is not None:
            recalibrate_cslc_kc_per_pad(
                example.model, example.scene_params.cslc_contact_fraction
            )

        met = run_headless(label, example)
        results.append(met)
        _log(
            f"RESULT: max_z={met.max_z:.4f}  final_z={met.final_z:.4f}  "
            f"lifted={'YES' if met.lifted else 'NO'}  "
            f"held={'YES' if met.held else 'NO'}"
        )
    _section("SUMMARY")
    print(f"  {'Config':<24} {'Max Z':>8} {'Final Z':>8} {'Lifted':>8} {'Held':>8}")
    print(f"  {'─' * 60}")
    for m in results:
        lifted_s = "YES" if m.lifted else "NO"
        held_s = "YES" if m.held else "NO"
        print(
            f"  {m.name:<24} {m.max_z:8.4f} {m.final_z:8.4f} "
            f"{lifted_s:>8} {held_s:>8}"
        )
    return results


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_soft_limits: wp.array[float],
    task_object: wp.array[int],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    task_offset_approach: wp.vec3,
    task_offset_lift: wp.vec3,
    sphere_radius: float,
    penetration_depth: float,
    task_init_body_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ee_index: int,
    robot_body_count: int,
    num_bodies_per_world: int,
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_target_interpolated: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_target_interpolated: wp.array[wp.vec4],
    gripper_target: wp.array2d[wp.float32],
):
    tid = wp.tid()

    idx = task_idx[tid]
    task = task_schedule[idx]
    task_time_soft_limit = task_time_soft_limits[idx]
    sphere_body_index = task_object[idx]

    task_time_elapsed[tid] += task_dt

    # Interpolation parameter t between 0 and 1
    t = wp.min(1.0, task_time_elapsed[tid] / task_time_soft_limit)

    # Get the end-effector position and rotation at the start of the task
    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_id])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_id])
    ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    # Get the current position of the object
    obj_body_id = tid * num_bodies_per_world + sphere_body_index
    obj_pos_current = wp.transform_get_translation(body_q[obj_body_id])
    obj_quat_current = wp.transform_get_rotation(body_q[obj_body_id])

    t_gripper = 0.0

    # Set the target position and rotation based on the task
    if task == TaskType.APPROACH.value:
        ee_pos_target[tid] = obj_pos_current + task_offset_approach
        ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi) * wp.quat_inverse(obj_quat_current)
    elif task == TaskType.REFINE_APPROACH.value:
        ee_pos_target[tid] = obj_pos_current
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value:
        ee_pos_target[tid] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = t
    elif task == TaskType.LIFT.value:
        ee_pos_target[tid] = ee_pos_prev + task_offset_lift
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.HOLD.value:
        ee_pos_target[tid] = ee_pos_prev + task_offset_lift
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0

    ee_pos_target_interpolated[tid] = ee_pos_prev * (1.0 - t) + ee_pos_target[tid] * t
    ee_quat_interpolated = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)

    ee_rot_target[tid] = ee_quat_target[:4]
    ee_rot_target_interpolated[tid] = ee_quat_interpolated[:4]

    # Set the gripper target position. Closing past (sphere_radius -
    # penetration_depth) commands the pads into the sphere by
    # `penetration_depth`, producing a controlled grasp force.
    open_pos = 0.06
    closed_pos = sphere_radius - penetration_depth
    gripper_pos = open_pos * (1.0 - t_gripper) + closed_pos * t_gripper
    gripper_target[tid, 0] = gripper_pos
    gripper_target[tid, 1] = gripper_pos


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_soft_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    num_bodies_per_world: int,
    ee_index: int,
    # outputs
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_init_body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    idx = task_idx[tid]
    task_time_soft_limit = task_time_soft_limits[idx]

    # Get the current position of the end-effector
    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_current = wp.transform_get_translation(body_q[ee_body_id])
    ee_quat_current = wp.transform_get_rotation(body_q[ee_body_id])

    # Calculate the end-effector position error
    pos_err = wp.length(ee_pos_target[tid] - ee_pos_current)

    ee_quat_target = wp.quaternion(ee_rot_target[tid][:3], ee_rot_target[tid][3])

    quat_rel = ee_quat_current * wp.quat_inverse(ee_quat_target)
    rot_err = wp.abs(wp.degrees(2.0 * wp.atan2(wp.length(quat_rel[:3]), quat_rel[3])))

    # Advance the task if the time elapsed is greater than the soft limit,
    # the end-effector position error is less than 0.001 meters,
    # the rotation error is less than 0.5 degrees, and the task index is not the last one.
    # NOTE: These tolerances can be achieved thanks to the gravity compensation enabled via
    # mujoco:gravcomp and mujoco:jnt_actgravcomp custom attributes.
    if (
        task_time_elapsed[tid] >= task_time_soft_limit
        and pos_err < 0.001
        and rot_err < 0.5
        and task_idx[tid] < wp.len(task_time_soft_limits) - 1
    ):
        # Advance to the next task
        task_idx[tid] += 1
        task_time_elapsed[tid] = 0.0

        body_id_start = tid * num_bodies_per_world
        for i in range(num_bodies_per_world):
            body_id = body_id_start + i
            task_init_body_q[body_id] = body_q[body_id]


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.collide_substeps = False
        self.world_count = args.world_count
        self.headless = args.headless

        self.viewer = viewer
        self.contact_model = args.contact_model

        self.scene_params = make_scene_params(args, self.sim_dt)

        self.table_pos = wp.vec3(self.scene_params.table_pos)
        self.table_top_center = wp.vec3(self.scene_params.table_top_center)
        self.robot_base_pos = wp.vec3(self.scene_params.robot_base_pos)
        self.task_offset_approach = wp.vec3(self.scene_params.task_offset_approach)
        self.task_offset_lift = wp.vec3(self.scene_params.task_offset_lift)
        
        franka_with_table = self.build_franka_with_table()
        scene = self.build_scene(franka_with_table)
        self.robot_body_count = franka_with_table.body_count

        self.model_single = franka_with_table.finalize()
        self.model = scene.finalize()
        self.num_bodies_per_world = self.model.body_count // self.world_count
        
        self.solver = make_solver(model=self.model,
                                  solver_name=args.solver,
                                  scene_params=self.scene_params,
                                  )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.joint_target_shape = self.control.joint_target_pos.reshape((self.world_count, -1)).shape
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()
        self.state = self.model.state()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.setup_ik()
        self.setup_tasks()

        if self.headless:
            self.viewer = newton.viewer.ViewerNull()

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False  # Disable interactive GUI picking for this example

        if hasattr(self.viewer, "renderer"):
            self.viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))
        self.episode_steps = 0

        # Sphere is the first body added after the robot bodies (world 0);
        # the pad shape lives on fr3_leftfinger and tracks the contact surface.
        self.sim_diag = SimDiagnostics(
            sphere_body_idx=self.robot_body_count,
            pad_shape_idx=self.left_pad_shape_idx,
            model=self.model,
            sphere_mass=self.scene_params.sphere_mass,
            gravity_z=self.scene_params.gravity[2],
            print_every_frames=6,
        )

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

        if not self.headless:
            self.scene_params.dump()
            inspect_model(self.model, self.contact_model)


    def simulate(self):
        if not self.collide_substeps:
            self.model.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            if self.collide_substeps:
                self.model.collide(self.state_0, self.contacts)

            self.state_0.clear_forces()

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.episode_steps == 1:
            self.start_time = time.perf_counter()

        self.set_joint_targets()

        self.simulate()

        if self.episode_steps > 1:
            self.sim_time += self.frame_dt

        task_idx = int(self.task_idx.numpy()[0])
        phase = self._schedule_task_types[task_idx].name
        cslc_state = read_cslc_state(self.model) if self.contact_model == "cslc" else None
        self.sim_diag.log(
            frame_idx=self.episode_steps,
            phase_name=phase,
            body_q_np=self.state_0.body_q.numpy(),
            active_contacts=count_active_contacts(self.contacts),
            frame_dt=self.frame_dt,
            cslc_state=cslc_state,
        )
        self.episode_steps += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _render_ui(self, imgui):
        # Snapshot lives on self.sim_diag and is refreshed every step().
        d = self.sim_diag
        imgui.text(f"Contact:  {self.contact_model}")
        imgui.text(f"Phase:    {d.phase_name}")
        imgui.text(f"Step:     {d.frame_idx}")
        imgui.text(f"Sphere Z: {d.sphere_z:+.4f}")
        imgui.text(f"Pad Z:    {d.pad_z:+.4f}")
        imgui.text(f"Delta Z:  {d.delta_z * 1e3:+.2f} mm")
        imgui.text(f"Contacts: {d.contact_count}")
        if d.cslc_state is not None:
            imgui.text(f"CSLC:     {d.cslc_state['n_active']}/{d.cslc_state['n_surface']}")


    def build_franka_with_table(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(
                self.robot_base_pos,
                wp.quat_identity(),
            ),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        if self.scene_params.start_gripped:
            arm_q = self._compute_grasp_joint_q()
        else:
            # Franka canonical home pose.
            arm_q = [
                0.0, -1/4 * np.pi, 0.0, -3/4 * np.pi, 0.0,
                1/2 * np.pi, 1/4 * np.pi,
            ]
        finger_q = [0.04, 0.04]

        builder.joint_q[:9] = arm_q + finger_q
        builder.joint_target_pos[:9] = arm_q + finger_q

        # Finger PD gains bumped 5× vs the cube example (which uses 100/10):
        # our sphere is ~500 g (4421 kg/m³ × (4/3)π·r³) vs the cube example's
        # ~50 g cubes, so the tighter grip is needed to produce enough contact
        # normal force for friction to carry the weight during LIFT.
        builder.joint_target_ke[:9] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 500, 500]
        builder.joint_target_kd[:9] = [450, 450, 350, 350, 200, 200, 200, 50, 50]

        # TODO: pad vs gripper ke kd vs driver_ke driver_kd undrestanding
        # builder.joint_target_ke[7:9] = [self.scene_params.drive_ke, self.scene_params.drive_ke]
        # builder.joint_target_kd[7:9] = [self.scene_params.drive_kd, self.scene_params.drive_kd]
            
        builder.joint_effort_limit[:9] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
        builder.joint_armature[:9] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2

        # Enable gravity compensation for the 7 arm joint DOFs
        gravcomp_attr = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_attr.values is None:
            gravcomp_attr.values = {}
        for dof_idx in range(7):
            gravcomp_attr.values[dof_idx] = True

        # Enable body gravcomp on the arm links and hand assembly so MuJoCo
        # cancels their gravitational load.
        # Body 0 = base (root), body 1 = fr3_link0 (fixed to world).
        # Bodies 2-8 = fr3_link1-7 (revolute arm joints).
        # Bodies 9-11 = fr3_link8, fr3_hand, fr3_hand_tcp (hand assembly).
        # Bodies 12-13 = fr3_leftfinger, fr3_rightfinger (gripper).
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(2, 14):
            gravcomp_body.values[body_idx] = 1.0
        
        left_finger_idx = find_body_in_builder(builder, "fr3_leftfinger")
        right_finger_idx = find_body_in_builder(builder, "fr3_rightfinger")
        pad_xform = wp.transform((0.0, -self.scene_params.pad_hy, self.scene_params.pad_local_z))
        
        if self.contact_model == "point":
            pad_cfg = point_pad_cfg(self.scene_params)
        elif self.contact_model == "cslc":
            pad_cfg = get_cslc_pad_cfg(self.scene_params)
        elif self.contact_model == "hydro":
            pad_cfg = get_hydro_pad_cfg(self.scene_params)
        
        self.left_pad_shape_idx = builder.add_shape_box(body=left_finger_idx,
                                                        xform=pad_xform,
                                                        hx=self.scene_params.pad_hx,
                                                        hy=self.scene_params.pad_hy,
                                                        hz=self.scene_params.pad_hz,
                                                        cfg=pad_cfg)

        self.left_pad_shape_idx = builder.add_shape_box(body=right_finger_idx,
                                                        xform=pad_xform,
                                                        hx=self.scene_params.pad_hx,
                                                        hy=self.scene_params.pad_hy,
                                                        hz=self.scene_params.pad_hz,
                                                        cfg=pad_cfg)
        # TABLE
        builder.add_shape_box(
            body=-1,
            hx=1/2 * self.scene_params.table_length,
            hy=1/2 * self.scene_params.table_width,
            hz=1/2 * self.scene_params.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
        )
        return builder

    def build_scene(self, franka_with_table: newton.ModelBuilder):
        scene = newton.ModelBuilder()
        for world_id in range(self.world_count):
            scene.begin_world()
            scene.add_builder(franka_with_table)
            self.add_sphere(
                scene, world_id
            )
            scene.end_world()

        scene.add_ground_plane()
        return scene
    

    def add_sphere(
        self,
        scene: newton.ModelBuilder,
        world_id: int,
    ):
        key = f"world_{world_id}/sphere_0"
        sphere_pos = self.table_top_center + wp.vec3(0, 0, self.scene_params.sphere_radius)
        body_xform = wp.transform(sphere_pos)
        mesh_body = scene.add_body(xform=body_xform)
        sphere_shape_idx = scene.shape_count
        sphere_color = [0.8, 0.2, 0.2]
        
        if self.contact_model in ['cslc', 'point']:
            sphere_cfg = get_sphere_cfg_not_hyrdo(self.scene_params)
        elif self.contact_model in ['hydro']:
            sphere_cfg = get_sphere_cfg_hydro(self.scene_params)
            
        scene.add_shape_sphere(body=mesh_body, radius=self.scene_params.sphere_radius, cfg=sphere_cfg, color=sphere_color, label=key)


    def setup_ik(self):
        self.ee_index = 11
        body_q_np = self.state.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        init_ee_pos = body_q_np[self.ee_index][:3]
        self.home_pos = wp.vec3(init_ee_pos)

        # Position objective
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([self.home_pos] * self.world_count, dtype=wp.vec3),
        )

        # Rotation objective
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.transform_get_rotation(self.ee_tf)[:4]] * self.world_count, dtype=wp.vec4),
        )

        ik_dofs = self.model_single.joint_coord_count

        # Joint limit objective
        self.joint_limit_lower = wp.clone(self.model.joint_limit_lower.reshape((self.world_count, -1))[:, :ik_dofs])
        self.joint_limit_upper = wp.clone(self.model.joint_limit_upper.reshape((self.world_count, -1))[:, :ik_dofs])

        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.joint_limit_lower.flatten(),
            joint_limit_upper=self.joint_limit_upper.flatten(),
        )

        # Variables the solver will update
        self.joint_q_ik = wp.clone(self.model.joint_q.reshape((self.world_count, -1))[:, :ik_dofs])

        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _compute_grasp_joint_q(self):
        """
        Used to calculate joint_q when start-gripped is passed.
        Pre-compute arm joint angles that place the EE at the sphere
        center with gripper-down orientation.
        """
        temp = newton.ModelBuilder()
        temp.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        temp.joint_q[:9] = [
            0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0,
            np.pi / 2, np.pi / 4, 0.04, 0.04,
        ]
        temp_model = temp.finalize()
        # Sphere rest position (same formula add_sphere uses).
        sphere_world_pos = wp.vec3(
            self.table_top_center[0],
            self.table_top_center[1],
            self.table_top_center[2] + self.scene_params.sphere_radius,
        )
        # Gripper-down orientation: 180° around X, quat (x,y,z,w) = (1,0,0,0).
        grip_rot_xyzw = wp.vec4(1.0, 0.0, 0.0, 0.0)
        ee_index = 11  # fr3_hand_tcp
        pos_obj = ik.IKObjectivePosition(
            link_index=ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([sphere_world_pos], dtype=wp.vec3),
        )
        rot_obj = ik.IKObjectiveRotation(
            link_index=ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([grip_rot_xyzw], dtype=wp.vec4),
        )
        ik_dofs = temp_model.joint_coord_count
        joint_q_ik = wp.clone(temp_model.joint_q.reshape((1, -1))[:, :ik_dofs])
        solver = ik.IKSolver(
            model=temp_model,
            n_problems=1,
            objectives=[pos_obj, rot_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        solver.step(joint_q_ik, joint_q_ik, iterations=500)
        return joint_q_ik.numpy()[0, :7].tolist()

    def setup_tasks(self):
        schedule = self.scene_params.task_schedule_time
        if self.scene_params.start_gripped:
            # The arm is already at the grasp pose — skip motion phases.
            schedule = tuple(
                (task, dt)
                for task, dt in schedule
                if task not in (TaskType.APPROACH, TaskType.REFINE_APPROACH)
            )
        self.task_counter = len(schedule)

        # Python-side copy of the scheduled TaskTypes, so the log can map
        # task_idx (an *index* into the schedule) back to the actual task
        # name. Without this, start-gripped mode would print APPROACH for
        # the first task because the raw index collides with TaskType(0).
        self._schedule_task_types = [task for task, _ in schedule]

        self.task_schedule = wp.array(
            [task for task, _ in schedule],
            shape=(self.task_counter,), dtype=wp.int32)

        self.task_time_soft_limits = wp.array(
            [time_limit for _, time_limit in schedule],
            shape=(self.task_counter,), dtype=float)

        task_object = [self.robot_body_count] * self.task_counter
        self.task_object = wp.array(task_object, shape=(self.task_counter), dtype=wp.int32)

        self.task_init_body_q = wp.clone(self.state_0.body_q)
        self.task_idx = wp.zeros(self.world_count, dtype=wp.int32)

        self.task_dt = self.frame_dt
        self.task_time_elapsed = wp.zeros(self.world_count, dtype=wp.float32)

        # Initialize the target positions and rotations
        self.ee_pos_target = wp.zeros(self.world_count, dtype=wp.vec3)
        self.ee_pos_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec3)

        self.ee_rot_target = wp.zeros(self.world_count, dtype=wp.vec4)
        self.ee_rot_target_interpolated = wp.zeros(self.world_count, dtype=wp.vec4)

        self.gripper_target_interpolated = wp.zeros(shape=(self.world_count, 2), dtype=wp.float32)

    def set_joint_targets(self):
        wp.launch(
            set_target_pose_kernel,
            dim=self.world_count,
            inputs=[
                self.task_schedule,
                self.task_time_soft_limits,
                self.task_object,
                self.task_idx,
                self.task_time_elapsed,
                self.task_dt,
                self.task_offset_approach,
                self.task_offset_lift,
                self.scene_params.sphere_radius,
                self.scene_params.penetration_depth,
                self.task_init_body_q,
                self.state_0.body_q,
                self.ee_index,
                self.robot_body_count,
                self.num_bodies_per_world,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_target_interpolated,
                self.ee_rot_target,
                self.ee_rot_target_interpolated,
                self.gripper_target_interpolated,
            ],
        )

        # Set the target position
        self.pos_obj.set_target_positions(self.ee_pos_target_interpolated)
        # Set the target rotation
        self.rot_obj.set_target_rotations(self.ee_rot_target_interpolated)

        # Step the IK solver
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        # Set the joint target positions
        joint_target_pos_view = self.control.joint_target_pos.reshape((self.world_count, -1))
        wp.copy(dest=joint_target_pos_view[:, :7], src=self.joint_q_ik[:, :7])
        wp.copy(dest=joint_target_pos_view[:, 7:9], src=self.gripper_target_interpolated[:, :2])

        wp.launch(
            advance_task_kernel,
            dim=self.world_count,
            inputs=[
                self.task_time_soft_limits,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.num_bodies_per_world,
                self.ee_index,
            ],
            outputs=[
                self.task_idx,
                self.task_time_elapsed,
                self.task_init_body_q,
            ],
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument("--contact-model", type=str, default="cslc",
                            choices=["point", "cslc", "hydro"])
        parser.add_argument("--contact-models", type=str, nargs="+", default=["point", "cslc", "hydro"],
                            help="Space-separated list for headless mode (e.g. point cslc hydro).")
        parser.add_argument("--solver", type=str, default="mujoco",
                            choices=["mujoco", "semi"])
        parser.add_argument("--start-gripped", action="store_true",
                            help="Start at GRASP phase")
        parser.add_argument("--cslc-ka", type=float, default=None,
                            help="Override CSLC anchor stiffness ka [N/m].")
        parser.add_argument("--cslc-contact-fraction", type=float, default=None,
                            help="Override CSLC contact fraction for kc recalibration.")
        parser.add_argument("--kh", type=float, default=None,
                            help="Override hydroelastic modulus [Pa].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    args_peek, _ = parser.parse_known_args()
    headless_mode = args_peek.headless and len(args_peek.contact_models) > 0
    wp.init()
    
    if headless_mode:
        args = parser.parse_args()
        print(f"\n{'━' * 60}\n  ROBOT LIFT — headless\n{'━' * 60}")
        cms = args.contact_models if args.contact_models else [args.contact_model]
        test_headless(args, contact_models=cms)
    else:
        viewer, args = newton.examples.init(parser)
        example = Example(viewer, args)
        newton.examples.run(example, args)
