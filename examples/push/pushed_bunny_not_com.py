import warp as wp
import numpy as np
import newton
from shapes.stanford_bunny import StanfordBunny
from examples.push.base_push import BasePush
from examples.base_example import get_base_parser
from examples.solutions import translate_and_rotate_on_ground_general_I_constrained_ground
from examples.push.utils import compute_lambda_for_torque_particles, apply_constant_force_and_torque_body, apply_constant_force_and_torque_particle
from utils.data_collector import collect_data_from_bodies, collect_data_from_spheres, COLLECTOR_TYPES
from utils.mesh_helper import calculate_z_lowest_sphere_set, calculate_z_lowest_mesh



class PushedBunnyNotCOMExample(BasePush):
    def __init__(self, viewer=None, args=None):
        super().__init__(viewer, args)
        self.example_name = "pushed_bunny_not_com"
        if self.experiment in ["srxpbd", "bxpbd"]:
            self.sphere_radius = args.sphere_radius
            n = len(self.bunny.particle_q_init)
            self.num_spheres = n
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_n{n}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_i{self.num_iterations}.npz"

        self.constant_force = wp.vec3(*args.constant_force)
        self.constant_tau = wp.vec3(*args.constant_torque)

    def build_scene(self, builder):
        rot = wp.quat_identity()
        if self.experiment == "mrxpbd":
            offset_z = calculate_z_lowest_sphere_set(self.args.morphit_json)
            center = wp.vec3(0.0, 0.0, offset_z)
        elif self.experiment in ["srxpbd", "tetxpbd", "mujoco", "semieuler", "bxpbd"]:
            bunny_hz = calculate_z_lowest_mesh(
                StanfordBunny.obj_path, transform=None)
            center = wp.vec3(0.0, 0.0, bunny_hz)

        self.bunny = StanfordBunny(pos0=center, rot0=rot)
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.bunny.add_mesh(builder)
        elif self.experiment in ["srxpbd", "bxpbd"]:
            builder = self.bunny.add_spheres(
                builder, radius=self.args.sphere_radius)
        elif self.experiment == "mrxpbd":
            builder = self.bunny.add_morphit_spheres(
                builder, json_adrs=self.args.morphit_json)
        return builder

    def launch_scenario_force_kernels(self):
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            wp.launch(
                kernel=apply_constant_force_and_torque_body,
                dim=1,
                inputs=[self.state_0.body_f,
                        self.constant_force, self.constant_tau],
                device=wp.get_device(),
            )
        elif self.experiment in ["srxpbd", "mrxpbd", "bxpbd"]:
            pos_ = self.state_0.particle_q.numpy()
            mass_ = self.model.particle_mass.numpy()
            com_world = np.sum(
                pos_ * mass_[:, np.newaxis], axis=0) / np.sum(mass_)

            lambda_ = compute_lambda_for_torque_particles(
                particle_positions=self.state_0.particle_q.numpy(),
                particle_masses=self.model.particle_mass.numpy(),
                com_world=com_world,
                tau_world=np.asarray(self.constant_tau),
            )
            wp.launch(
                kernel=apply_constant_force_and_torque_particle,
                dim=len(self.state_0.particle_q),
                inputs=[self.state_0.particle_f,
                        self.state_0.particle_q,
                        self.model.particle_mass,
                        self.bunny.particle_mass_sum,
                        com_world,
                        self.constant_force,
                        lambda_],
                device=wp.get_device(),
            )

    def calculate_analytical_state(self, t):
        pos, vel, rot, omega, L, p = translate_and_rotate_on_ground_general_I_constrained_ground(
            x0=self.bunny.pos0,
            rot0=self.bunny.rot0,
            I=self.bunny.I_m,
            m=self.bunny.mass,
            mu=self.mu,
            F=self.constant_force,
            tau=self.constant_tau,
            t=t
        )
        return pos, vel, rot, omega, L, p

    def collect_body_data(self, pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, p_analytic):
        body_idx = 0
        body_q = self.state_0.body_q.numpy().copy()
        body_qd = self.state_0.body_qd.numpy().copy()
        I_body = np.asarray(self.bunny.I_m).reshape((3, 3))
        data = collect_data_from_bodies(
            pos_analytic=pos_analytic, vel_analytic=vel_analytic, rot_analytic=rot_analytic,
            omega_analytic=omega_analytic, L_analytic=L_analytic, p_analytic=p_analytic,
            body_q=body_q, body_qd=body_qd, I=I_body, body_idx=body_idx, mass=self.bunny.mass)
        for key, value in data.items():
            self.collected_data[key].append(value)

    def collect_sphere_packed_data(self, pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, p_analytic):
        particle_q = self.state_0.particle_q.numpy().copy()
        particle_qd = self.state_0.particle_qd.numpy().copy()
        particle_m = self.model.particle_mass.numpy().copy()
        prev_rot_solver = self.collected_data[COLLECTOR_TYPES.ROT_S][-1] if len(
            self.collected_data[COLLECTOR_TYPES.ROT_S]) > 0 else self.bunny.rot0
        data = collect_data_from_spheres(pos_analytic=pos_analytic, vel_analytic=vel_analytic, rot_analytic=rot_analytic,
                                         omega_analytic=omega_analytic, L_analytic=L_analytic, p_analytic=p_analytic,
                                         particles_q=particle_q, particles_qd=particle_qd, particle_m=particle_m,
                                         prev_rot_solver=prev_rot_solver, dt=self.frame_dt
                                         )
        for key, value in data.items():
            self.collected_data[key].append(value)

        particle_forces = self.state_0.particle_f.numpy().copy()
        F_net = particle_forces.sum(axis=0)
        self.collected_data[COLLECTOR_TYPES.PARTICLE_FORCE_SUM].append(F_net)

        M = particle_m.sum()
        com_world = np.sum(particle_q * particle_m[:, None], axis=0) / M
        r = particle_q - com_world[None, :]
        tau_total = np.sum(np.cross(r, particle_forces), axis=0)
        self.collected_data[COLLECTOR_TYPES.PARTICLE_TORQUE_SUM].append(
            tau_total)


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "-r",
        "--sphere_radius",
        type=float,
        default=0.005,
        help="radii of spheres",
    )
    parser.add_argument(
        "-k",
        "--plot_keys",
        nargs='+',
        type=str,
        default=[COLLECTOR_TYPES.POS_L2_E,
                 COLLECTOR_TYPES.ROT_E,
                 COLLECTOR_TYPES.OMEGA_E,
                 COLLECTOR_TYPES.ANG_MOM_E,
                 COLLECTOR_TYPES.PARTICLE_FORCE_SUM,
                 COLLECTOR_TYPES.PARTICLE_TORQUE_SUM,
                 COLLECTOR_TYPES.ANG_MOM_A,
                 COLLECTOR_TYPES.ANG_MOM_S,
                 ])
    parser.add_argument("--constant-force",
                        nargs=3,
                        type=float,
                        default=[0.0, 0.0, 0.0])
    parser.add_argument("--constant-torque",
                        nargs=3,
                        type=float,
                        default=[0.0, 0.0, 0.02])
    parser.add_argument("--mu",
                        type=float,
                        default=0.0,
                        help="Mu between the object and the slope (Coulomb friction). Default 0 = frictionless.")
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = PushedBunnyNotCOMExample(viewer, args=args)
    newton.examples.run(example, args)
