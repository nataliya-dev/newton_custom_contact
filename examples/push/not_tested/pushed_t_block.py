import warp as wp
import numpy as np
import newton
from shapes.t_block import TBlock
from examples.push.base_push import BasePush
from examples.base_example import get_base_parser
from utils.data_collector import COLLECTOR_TYPES, collect_data_from_bodies, collect_data_from_spheres
from examples.push.utils import apply_constant_body_force, apply_constant_particle_force


class PushedTBlockExample(BasePush):
    def __init__(self, viewer=None, args=None, plot_sim_time=10.0):
        super().__init__(viewer, args)
        self.example_name = "pushed_t_block"
        if self.experiment in ["srxpbd"]:
            self.num_spheres = args.num_spheres
            n = self.num_spheres * self.num_spheres * self.num_spheres
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_n{n}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_i{self.num_iterations}.npz"
        self.plot_sim_time = plot_sim_time
        self.constant_force = wp.vec3(2.0, 0.0, 0.0)

    def build_scene(self, builder):
        rot = wp.quat_identity()
        mass = 4.0
        
        # T-block dimensions
        # Stem: vertical box
        stem_hx = 0.05
        stem_hy = 0.05
        stem_hz = 0.15
        
        # Crossbar: horizontal box extending from stem
        crossbar_hx = 0.15
        crossbar_hy = 0.05
        crossbar_hz = 0.05
        
        # Position: T-block center starts above ground
        center = wp.vec3(0.0, 0.0, 0.25)
        
        self.t_block = TBlock(
            pos0=center,
            rot0=rot,
            stem_hx=stem_hx,
            stem_hy=stem_hy,
            stem_hz=stem_hz,
            crossbar_hx=crossbar_hx,
            crossbar_hy=crossbar_hy,
            crossbar_hz=crossbar_hz,
            mass=mass
        )
        
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.t_block.add_body(builder)
        elif self.experiment == "srxpbd":
            builder = self.t_block.add_spheres(builder, num_spheres=self.args.num_spheres)
        elif self.experiment == "mrxpbd":
            builder = self.t_block.add_morphit_spheres(builder, json_adrs=self.args.morphit_json)
        
        return builder

    def launch_scenario_force_kernels(self):
        """Apply constant force only to the T-block"""
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            # For rigid bodies, apply force to the single body (index 0)
            wp.launch(
                kernel=apply_constant_body_force,
                dim=1,
                inputs=[self.state_0.body_f, self.constant_force],
                device=wp.get_device(),
            )
        elif self.experiment in ["srxpbd", "mrxpbd"]:
            # For particle-based, apply force proportionally to all T-block particles
            n = self.state_0.particle_f.shape[0]
            wp.launch(
                kernel=apply_constant_particle_force,
                dim=n,
                inputs=[self.state_0.particle_f, self.constant_force,
                        self.model.particle_mass, self.t_block.particle_mass_sum],
                device=wp.get_device(),
            )

    def calculate_analytical_state(self, t):
        """
        Analytical solution for a single T-block being pushed with constant force.
        
        Acceleration: a = F / m
        Position: x = x0 + 0.5 * a * t^2
        Velocity: v = a * t
        """
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            x0 = self.t_block.pos0
        elif self.experiment in ["srxpbd", "mrxpbd"]:
            x0 = self.t_block.particle_com_init
        rot = self.t_block.rot0
        a = self.constant_force / self.t_block.mass
        pos = x0 + 0.5 * a * (t * t)
        vel = a * t

        omega_analytic = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        L_analytic = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray(pos), np.asarray(vel), np.asarray(rot), omega_analytic, L_analytic

    def collect_body_data(self, pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic):
        """Collect data for rigid body case"""
        body_idx = 0
        body_q = self.state_0.body_q.numpy().copy()
        body_qd = self.state_0.body_qd.numpy().copy()
        I_body = np.asarray(self.t_block.I_m).reshape((3, 3))
        
        data = collect_data_from_bodies(
            pos_analytic=pos_analytic, vel_analytic=vel_analytic, rot_analytic=rot_analytic,
            omega_analytic=omega_analytic, L_analytic=L_analytic, body_q=body_q, body_qd=body_qd, I=I_body, body_idx=body_idx)
        for key, value in data.items():
            self.collected_data[key].append(value)

    def collect_sphere_packed_data(self, pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic):
        """Collect data for particle-based case"""
        particle_q = self.state_0.particle_q.numpy().copy()
        particle_qd = self.state_0.particle_qd.numpy().copy()
        particle_m = self.model.particle_mass.numpy().copy()
        prev_rot_solver = self.collected_data[COLLECTOR_TYPES.ROT_S][-1] if len(
            self.collected_data[COLLECTOR_TYPES.ROT_S]) > 0 else self.t_block.rot0

        data = collect_data_from_spheres(pos_analytic=pos_analytic, vel_analytic=vel_analytic, rot_analytic=rot_analytic,
                                         omega_analytic=omega_analytic, L_analytic=L_analytic, particles_q=particle_q,
                                         particles_qd=particle_qd, particle_m=particle_m,
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
        help="Number of spheres per dimension for SRXPBD experiment",
    )
    parser.add_argument(
        "-k",
        "--plot_keys",
        nargs='+',
        type=str,
        default=[COLLECTOR_TYPES.POS_L2_E,
                 COLLECTOR_TYPES.ROT_E,
                 COLLECTOR_TYPES.PARTICLE_FORCE_SUM],
    )
    parser.add_argument("--mu",
                        type=float,
                        default=0.0,
                        help="Mu between the object and ground (Coulomb friction). Default 0 = frictionless.")
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = PushedTBlockExample(viewer, args=args)
    newton.examples.run(example, args)
