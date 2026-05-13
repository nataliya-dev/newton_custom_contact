import warp as wp
import numpy as np
import newton
from shapes.box import Box
from examples.push.base_push import BasePush
from examples.base_example import get_base_parser
from utils.data_collector import COLLECTOR_TYPES, collect_data_from_bodies, collect_data_from_spheres
from examples.push.utils import apply_constant_body_force, apply_constant_particle_force
from examples.solutions import push_object_on_ground_solution
from utils.mesh_helper import calculate_z_lowest_sphere_set, calculate_box_dimensions_from_sphere_set


class PushedBoxExample(BasePush):
    def __init__(self, viewer=None, args=None):
        super().__init__(viewer, args)
        self.example_name = "pushed_box"
        if self.experiment in ["srxpbd", "bxpbd"]:
            self.num_spheres = args.num_spheres
            n = self.num_spheres * self.num_spheres * self.num_spheres
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_n{n}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_i{self.num_iterations}.npz"
        # TODO: extract the number of spheres from morphit
        self.constant_force = wp.vec3(*args.constant_force)

    def build_scene(self, builder):
        mass = 4.0
        rot = wp.quat_identity()
        if self.experiment == "mrxpbd":
            offset_z = calculate_z_lowest_sphere_set(self.args.morphit_json)
            center = wp.vec3(0.0, 0.0, offset_z)
            hx, hy, hz = calculate_box_dimensions_from_sphere_set(
                self.args.morphit_json)
        elif self.experiment in ["srxpbd", "tetxpbd", "mujoco", "semieuler", "bxpbd"]:
            hx = hy = hz = 0.1
            center = wp.vec3(0.0, 0.0, hz)

        self.box = Box(pos0=center, rot0=rot, hx=hx, hy=hy, hz=hz, mass=mass)
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.box.add_body(builder)
        elif self.experiment in ["srxpbd", "bxpbd"]:
            builder = self.box.add_spheres(
                builder, num_spheres=self.args.num_spheres)
        elif self.experiment == "mrxpbd":
            builder = self.box.add_morphit_spheres(
                builder, json_adrs=self.args.morphit_json)
        return builder

    def launch_scenario_force_kernels(self):
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            n_bodies = self.state_0.body_f.shape[0]
            wp.launch(
                kernel=apply_constant_body_force,
                dim=n_bodies,
                inputs=[self.state_0.body_f, self.constant_force],
                device=wp.get_device(),
            )
        elif self.experiment in ["srxpbd", "mrxpbd", "bxpbd"]:
            n = self.state_0.particle_f.shape[0]
            wp.launch(
                kernel=apply_constant_particle_force,
                dim=n,
                inputs=[self.state_0.particle_f, self.constant_force,
                        self.model.particle_mass, self.box.particle_mass_sum],
                device=wp.get_device(),
            )

    def calculate_analytical_state(self, t):
        x0 = self.box.pos0
        rot0 = self.box.rot0
        return push_object_on_ground_solution(x0=x0, rot0=rot0, m=self.box.mass,
                                              mu=self.mu, F=self.constant_force, t=t)

    def collect_body_data(self, pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, p_analytic):
        body_idx = 0
        body_q = self.state_0.body_q.numpy().copy()
        body_qd = self.state_0.body_qd.numpy().copy()
        I_body = np.asarray(self.box.I_m).reshape((3, 3))
        
        data = collect_data_from_bodies(
            pos_analytic=pos_analytic, vel_analytic=vel_analytic, rot_analytic=rot_analytic,
            omega_analytic=omega_analytic, L_analytic=L_analytic,  p_analytic=p_analytic, 
            body_q=body_q, body_qd=body_qd, I=I_body, body_idx=body_idx, mass=self.box.mass)
        for key, value in data.items():
            self.collected_data[key].append(value)

    def collect_sphere_packed_data(self, pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, p_analytic):
        particle_q = self.state_0.particle_q.numpy().copy()
        particle_qd = self.state_0.particle_qd.numpy().copy()
        particle_m = self.model.particle_mass.numpy().copy()
        prev_rot_solver = self.collected_data[COLLECTOR_TYPES.ROT_S][-1] if len(
            self.collected_data[COLLECTOR_TYPES.ROT_S]) > 0 else self.box.rot0

        data = collect_data_from_spheres(pos_analytic=pos_analytic, vel_analytic=vel_analytic, rot_analytic=rot_analytic,
                                         omega_analytic=omega_analytic, L_analytic=L_analytic,  p_analytic=p_analytic, 
                                         particles_q=particle_q, particles_qd=particle_qd, particle_m=particle_m,
                                         prev_rot_solver=prev_rot_solver, dt=self.frame_dt
                                         )
        for key, value in data.items():
            self.collected_data[key].append(value)


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "-n",
        "--num_spheres",
        type=int,
        default=4,
        help="Number of spheres per box dimension for SRXPBD/BXPBD experiment",
    )
    parser.add_argument(
        "-k",
        "--plot_keys",
        nargs='+',
        type=str,
        default=[COLLECTOR_TYPES.POS_L2_E,
                 COLLECTOR_TYPES.ROT_E,
                 COLLECTOR_TYPES.PARTICLE_FORCE_SUM,
                 COLLECTOR_TYPES.POS_E],
    )
    parser.add_argument("--constant-force",
                        nargs=3,
                        type=float,
                        default=[2.0, 0.0, 0.0])
    parser.add_argument("--mu",
                        type=float,
                        default=0.0,
                        help="Mu between the object and ground (Coulomb friction). Default 0 = frictionless.")
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = PushedBoxExample(viewer, args=args)
    newton.examples.run(example, args)
