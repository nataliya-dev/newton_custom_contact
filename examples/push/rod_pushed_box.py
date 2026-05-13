import warp as wp
import numpy as np
import newton
from shapes.box import Box
from shapes.rod import Rod
from examples.push.base_push import BasePush
from examples.base_example import get_base_parser
from utils.data_collector import COLLECTOR_TYPES, collect_data_from_bodies, collect_data_from_spheres
from examples.push.utils import apply_constant_force_to_shape, apply_constant_particle_force_range
from newton import ParticleFlags, ShapeFlags
from examples.solutions import PushBoxRodSolution
from utils.mesh_helper import calculate_box_dimensions_from_sphere_set, calculate_z_lowest_sphere_set

"""
Example usage:

Rod pushes box in a straight line. Compare odd and even number of spheres per box dimension.
uv run -m examples.push.rod_pushed_box -e mujoco --constant-force 0 -0.1 0 --mu 0.4 -t 5
uv run -m examples.push.rod_pushed_box -e srxpbd --num_spheres_rod 1 --num_spheres_box 6 --constant-force 0 -0.1 0 --mu 0.4 -t 5
uv run -m examples.push.rod_pushed_box -e srxpbd --num_spheres_rod 1 --num_spheres_box 7 --constant-force 0 -0.1 0 --mu 0.4 -t 5
python3 scripts/compare_experiments.py -t rod_pushed_box -e rod_pushed_box/mujoco_bx0.000_mu0.400_i10 rod_pushed_box/srxpbd_nr1_nt6_bx0.000_mu0.400_i10 rod_pushed_box/srxpbd_nr1_nt7_bx0.000_mu0.400_i10
"""

"""
Rod pushes box off center, causing it to rotate as well as translate. Compare doubling spheres per box dimension.
uv run -m examples.push.rod_pushed_box -e mujoco --constant-force 0 -0.1 0 --mu 0.4 -t 10 --box-pos 0.01 0.0 0.1
uv run -m examples.push.rod_pushed_box -e srxpbd --num_spheres_rod 1 --num_spheres_box 6 --constant-force 0 -0.1 0 --mu 0.4 -t 10 --box-pos 0.01 0.0 0.1
uv run -m examples.push.rod_pushed_box -e srxpbd --num_spheres_rod 1 --num_spheres_box 12 --constant-force 0 -0.1 0 --mu 0.4 -t 10 --box-pos 0.01 0.0 0.1
python3 scripts/compare_experiments.py -t rod_pushed_box -e rod_pushed_box/mujoco_bx0.010_mu0.400_i10 rod_pushed_box/srxpbd_nr1_nt6_bx0.010_mu0.400_i10 rod_pushed_box/srxpbd_nr1_nt12_bx0.010_mu0.400_i10
"""


class RodPushedBoxExample(BasePush):
    def __init__(self, viewer=None, args=None):
        super().__init__(viewer, args)
        self.example_name = "rod_pushed_box"
        box_x = args.box_pos[0]
        if self.experiment in ["srxpbd"]:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_nr{args.num_spheres_rod}_nt{args.num_spheres_box}_bx{box_x:.3f}_mu{args.mu:.3f}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_bx{box_x:.3f}_mu{args.mu:.3f}_i{self.num_iterations}.npz"
        self.constant_force = wp.vec3(*args.constant_force)

        self.analytical_solution = PushBoxRodSolution(
            x0_rod=self.rod.pos0,
            x0_box=self.box.pos0,
            m_rod=self.rod.mass,
            rod_force=self.constant_force,
            box_hx=self.box.hx,
            box_hy=self.box.hy,
            mu_push=self.mu,
        )

    def build_scene(self, builder):
        mass = self.args.box_mass
        rot = wp.quat_identity()
        if self.experiment == "mrxpbd":
            offset_z = calculate_z_lowest_sphere_set(self.args.morphit_json)
            center = wp.vec3(0.0, 0.0, offset_z)
            box_hx, box_hy, box_hz = calculate_box_dimensions_from_sphere_set(
                self.args.morphit_json)
        elif self.experiment in ["srxpbd", "tetxpbd", "mujoco", "semieuler", "bxpbd"]:
            box_hx = box_hy = box_hz = 0.1
            center = wp.vec3(*self.args.box_pos)
        self.box = Box(pos0=center, rot0=rot, hx=box_hx,
                       hy=box_hy, hz=box_hz, mass=mass)

        self._prev_rot_box = self.box.rot0

        rod_radius = 0.03
        rod_length = 0.4
        rod_z = rod_length / 2.0
        rod_rot = wp.quat_identity()
        rod_pos = wp.vec3(0.0, box_hy + rod_radius*2.0, rod_z)
        self.rod = Rod(
            pos0=rod_pos,
            rot0=rod_rot,
            radius=rod_radius,
            length=rod_length,
            mass=2.0
        )

        # Add objects to the scene based on experiment type
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.box.add_body(builder)
            builder = self.rod.add_body(builder)
        elif self.experiment in ["srxpbd", "bxpbd"]:
            builder = self.box.add_spheres(
                builder, num_spheres=self.args.num_spheres_box)
            builder = self.rod.add_spheres(
                builder, num_spheres=self.args.num_spheres_rod)
        elif self.experiment == "mrxpbd":
            builder = self.box.add_morphit_spheres(
                builder, json_adrs=self.args.morphit_json)
            builder = self.rod.add_morphit_spheres(
                builder, json_adrs=self.args.morphit_json)

        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder.shape_flags[self.rod.shape_idx] = ShapeFlags.INTEGRATE_ONLY | ShapeFlags.VISIBLE | ShapeFlags.COLLIDE_SHAPES

        elif self.experiment in ["srxpbd", "mrxpbd", "bxpbd"]:
            rod_particle_start = self.rod.particle_start_idx
            rod_particle_end = self.rod.particle_end_idx
            num_particles_rod = rod_particle_end - rod_particle_start
            builder.particle_flags[rod_particle_start:rod_particle_end] = [
                ParticleFlags.INTEGRATE_ONLY | ParticleFlags.ACTIVE] * num_particles_rod

        return builder

    def launch_scenario_force_kernels(self):
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            n_bodies = self.state_0.body_f.shape[0]
            wp.launch(
                kernel=apply_constant_force_to_shape,
                dim=n_bodies,
                inputs=[self.state_0.body_f,
                        self.constant_force, self.rod.body_idx],
                device=wp.get_device(),
            )

        elif self.experiment in ["srxpbd", "mrxpbd", "bxpbd"]:
            n_rod = self.rod.particle_end_idx - self.rod.particle_start_idx
            wp.launch(
                kernel=apply_constant_particle_force_range,
                dim=n_rod,
                inputs=[self.state_0.particle_f, self.constant_force,
                        self.model.particle_mass, self.rod.particle_mass_sum,
                        self.rod.particle_start_idx],
                device=wp.get_device(),
            )

    def calculate_analytical_state(self, t):
        return self.analytical_solution.step(self.frame_dt)

    def collect_body_data(self,
                          pos_box_analytic,
                          vel_box_analytic,
                          rot_box_analytic,
                          omega_analytic,
                          L_analytic,
                          p_box_analytic):
        body_q = self.state_0.body_q.numpy().copy()
        body_qd = self.state_0.body_qd.numpy().copy()
        body_f = self.state_0.body_f.numpy().copy()
        I_box = np.asarray(self.box.I_m).reshape((3, 3))

        data_box = collect_data_from_bodies(pos_analytic=pos_box_analytic,
                                            vel_analytic=vel_box_analytic,
                                            rot_analytic=rot_box_analytic,
                                            omega_analytic=omega_analytic,
                                            L_analytic=L_analytic,
                                            p_analytic=p_box_analytic,
                                            body_q=body_q,
                                            body_qd=body_qd,
                                            I=I_box,
                                            mass=self.box.mass,
                                            body_idx=self.box.body_idx,
                                            body_f=body_f)
        for key, value in data_box.items():
            self.collected_data[key].append(value)

    def collect_sphere_packed_data(self,
                                   pos_box_analytic,
                                   vel_box_analytic,
                                   rot_box_analytic,
                                   omega_analytic,
                                   L_analytic,
                                   p_box_analytic):
        particle_q = self.state_0.particle_q.numpy().copy()
        particle_qd = self.state_0.particle_qd.numpy().copy()
        particle_m = self.model.particle_mass.numpy()
        particle_f = self.state_0.particle_f.numpy().copy()

        # Extract particles belonging to the Box
        start_idx_box = self.box.particle_start_idx
        end_idx_box = self.box.particle_end_idx
        particle_q_box = particle_q[start_idx_box:end_idx_box]
        particle_qd_box = particle_qd[start_idx_box:end_idx_box]
        particle_m_box = particle_m[start_idx_box:end_idx_box]
        box_particle_indices = np.arange(start_idx_box, end_idx_box)

        data_box = collect_data_from_spheres(pos_analytic=pos_box_analytic,
                                             vel_analytic=vel_box_analytic,
                                             rot_analytic=rot_box_analytic,
                                             omega_analytic=omega_analytic,
                                             L_analytic=L_analytic,
                                             p_analytic=p_box_analytic,
                                             particles_q=particle_q_box,
                                             particles_qd=particle_qd_box,
                                             particle_m=particle_m_box,
                                             prev_rot_solver=self._prev_rot_box,
                                             dt=self.frame_dt,
                                             particle_f=particle_f,
                                             particle_indices=box_particle_indices)
        for key, value in data_box.items():
            self.collected_data[key].append(value)
        self._prev_rot_box = data_box[COLLECTOR_TYPES.ROT_S]


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "--num_spheres_rod",
        type=int,
        default=1,
        help="Number of spheres per dimension for the rod (SRXPBD only).",
    )
    parser.add_argument(
        "--num_spheres_box",
        type=int,
        default=4,
        help="Number of spheres per dimension for the T-block (SRXPBD only).",
    )
    parser.add_argument(
        "-k",
        "--plot_keys",
        nargs='+',
        type=str,
        default=[
            COLLECTOR_TYPES.POS_L2_E,
            COLLECTOR_TYPES.ROT_E,
            COLLECTOR_TYPES.PARTICLE_FORCE_SUM
        ],
    )
    parser.add_argument("--constant-force",
                        nargs=3,
                        type=float,
                        default=[0.0, -0.1, 0.0])
    parser.add_argument("--mu",
                        type=float,
                        default=0.4,
                        help="Mu between the object and ground (Coulomb friction). Default 0 = frictionless.")

    parser.add_argument("--box-mass", type=float, default=4.0)
    parser.add_argument("--box-pos", nargs=3, type=float,
                        default=[0.0, 0.0, 0.1])

    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = RodPushedBoxExample(viewer, args=args)
    newton.examples.run(example, args)
