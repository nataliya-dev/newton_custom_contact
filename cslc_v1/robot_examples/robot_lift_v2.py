# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cube Stacking
#
# Shows how to set up a simulation of a cube stacking task for multiple
# worlds using inverse kinematics to set joint target position references
# for the Franka Emika Franka Panda robot arm.
#
# Command: python -m newton.examples ik_cube_stacking --world-count 16
#
###########################################################################

import enum
import time

import numpy as np
import warp as wp

from cslc_v1.robot_examples.utils import SceneParams, find_body_in_builder, get_sphere_cfg, cslc_pad_cfg, hydro_pad_cfg, point_pad_cfg
import newton
import newton.examples
import newton.ik as ik


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    LIFT = 3
    HOLD = 4


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
    cube_body_index = task_object[idx]
    cube_index = cube_body_index - robot_body_count

    task_time_elapsed[tid] += task_dt

    # Interpolation parameter t between 0 and 1
    t = wp.min(1.0, task_time_elapsed[tid] / task_time_soft_limit)

    # Get the end-effector position and rotation at the start of the task
    ee_body_id = tid * num_bodies_per_world + ee_index
    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_body_id])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_body_id])
    ee_quat_target = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    # Get the current position of the object
    obj_body_id = tid * num_bodies_per_world + cube_body_index
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

    # Set the gripper target position
    gripper_pos = 0.06 * (1.0 - t_gripper)
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
        self.verbose = args.verbose

        self.viewer = viewer
        
        
        self.scene_params = SceneParams(
            dt=self.sim_dt,
            no_ground=getattr(args, "no_ground", False),
            start_gripped=getattr(args, "start_gripped", False),
            warm_start_sphere_vz=getattr(args, "warm_start_sphere_vz", False),
        )
        

        self.table_height = 0.2
        self.table_pos = wp.vec3(0.0, -0.5, 1/2 * self.table_height)
        self.table_top_center = self.table_pos + wp.vec3(0.0, 0.0, 0.5 * self.table_height)
        self.robot_base_pos = wp.vec3(self.table_pos[0], self.table_pos[1], 0.0) + wp.vec3(-0.7, 0.0, 0.0)

        self.task_offset_approach = wp.vec3(0.0, 0.0, 1.0 * self.scene_params.sphere_radius)
        self.task_offset_lift = wp.vec3(0.0, 0.0, 4.0 * self.scene_params.sphere_radius)
        
        # Build scene
        self.use_mujoco_contacts = getattr(args, "use_mujoco_contacts", False)
        
        franka_with_table = self.build_franka_with_table(self.scene_params)
        scene = self.build_scene(franka_with_table, self.scene_params)
        self.robot_body_count = franka_with_table.body_count

        self.model_single = franka_with_table.finalize()
        self.model = scene.finalize()
        self.num_bodies_per_world = self.model.body_count // self.world_count

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=20,
            ls_iterations=100,
            nconmax=1000,
            njmax=2000,
            cone="elliptic",
            impratio=1000.0,
            use_mujoco_contacts=self.use_mujoco_contacts,
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

        tock = time.perf_counter()
        if self.verbose and self.episode_steps > 0:
            print(f"Step {self.episode_steps} time: {tock - self.start_time:.2f}, sim time: {self.sim_time:.2f}")
            print(f"RT factor: {self.world_count * self.sim_time / (tock - self.start_time):.2f}")
            print("_" * 100)

        self.episode_steps += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def build_franka_with_table(self, scene_params: SceneParams):
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

        # Finger joints are prismatic with URDF limits [0, 0.04] m. Starting
        # them at 1.0 sent the fingers (and their attached pads) flying back
        # into the limit at sim start — use the upper limit (fully open).
        builder.joint_q[:9] = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
            0.04,
            0.04,
        ]

        builder.joint_target_pos[:9] = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
            0.04,
            0.04,
        ]

        builder.joint_target_ke[:9] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
        builder.joint_target_kd[:9] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
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
        pad_xform = wp.transform((0.0, -scene_params.pad_hy, scene_params.pad_local_z))
        pad_cfg = point_pad_cfg(scene_params)
        
        builder.add_shape_box(body=left_finger_idx,
                              xform=pad_xform,
                              hx=scene_params.pad_hx,
                              hy=scene_params.pad_hy,
                              hz=scene_params.pad_hz,
                              cfg=pad_cfg)

        builder.add_shape_box(body=right_finger_idx,
                                xform=pad_xform,
                                hx=scene_params.pad_hx,
                                hy=scene_params.pad_hy,
                                hz=scene_params.pad_hz,
                                cfg=pad_cfg)   
                
            
        table_cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, density=1000.0)
        table_cfg.ke = 5.0e4
        table_cfg.kd = 5.0e2
        table_cfg.kf = 1.0e3
        table_cfg.mu = 0.75
        # TABLE
        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5 * self.table_height,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            cfg=table_cfg,
        )

        if self.use_mujoco_contacts:
            # Set condim=4 (torsional friction) on finger shapes
            condim_attr = builder.custom_attributes["mujoco:condim"]
            if condim_attr.values is None:
                condim_attr.values = {}
            for shape_idx in range(builder.shape_count):
                if builder.shape_body[shape_idx] in (12, 13):  # left/right finger bodies
                    condim_attr.values[shape_idx] = 4

        return builder

    def build_scene(self, franka_with_table: newton.ModelBuilder, scene_params: SceneParams):
        scene = newton.ModelBuilder()
        for world_id in range(self.world_count):
            scene.begin_world()
            scene.add_builder(franka_with_table)
            self.add_sphere(
                scene, world_id, scene_params
            )
            scene.end_world()

        scene.add_ground_plane()
        return scene
    

    def add_sphere(
        self,
        scene: newton.ModelBuilder,
        world_id: int,
        scene_params: SceneParams,
    ):

        key = f"world_{world_id}/sphere_0"
        sphere_pos = self.table_top_center + wp.vec3(0, 0, scene_params.sphere_radius)
        body_xform = wp.transform(sphere_pos)
        mesh_body = scene.add_body(xform=body_xform)
        sphere_shape_idx = scene.shape_count
        sphere_color = [0.8, 0.2, 0.2]
        sphere_cfg = get_sphere_cfg(scene_params)    
        scene.add_shape_sphere(body=mesh_body, radius=scene_params.sphere_radius, cfg=sphere_cfg, color=sphere_color, label=key)
            

        if self.use_mujoco_contacts:
            # Set condim=4 (torsional friction) on cube shapes
            condim_attr = scene.custom_attributes["mujoco:condim"]
            if condim_attr.values is None:
                condim_attr.values = {}
            condim_attr.values[sphere_shape_idx] = 4


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

    def setup_tasks(self):
        task_schedule = [
            TaskType.APPROACH,
            TaskType.REFINE_APPROACH,
            TaskType.GRASP,
            TaskType.LIFT,
            TaskType.HOLD,
        ]
        self.task_counter = len(task_schedule)
        self.task_schedule = wp.array(task_schedule, shape=(self.task_counter), dtype=wp.int32)
        self.task_time_soft_limits = wp.array([2.0] * self.task_counter, dtype=float)

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
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
