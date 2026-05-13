import warp as wp
import numpy as np
import newton
from shapes.box import Box
from examples.tip.base_tip import BaseTip
from examples.base_example import get_base_parser
from utils.data_collector import collect_data_from_bodies, collect_data_from_spheres
import warp as wp


@wp.kernel
def apply_force_at_body_corner(
    body_f: wp.array(dtype=wp.spatial_vector),
    tau: wp.vec3,
):
    i = wp.tid()
    body_f[i] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), tau)


class PushedBoxExample(BaseTip):
    def __init__(self, viewer=None, args=None, plot_sim_time=1.0):
        super().__init__(viewer, args)
        self.example_name = "pushed_box"
        if self.experiment == "srxpbd":
            self.num_spheres = args.num_spheres
            n = self.num_spheres * self.num_spheres * self.num_spheres
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_n{n}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_i{self.num_iterations}.npz"
        self.plot_sim_time = plot_sim_time
        self.tau = wp.vec3(-3.0, 0.0, 0.0) # Torque about X axis

    def build_scene(self, builder):
        hx, hy, hz = 0.1, 0.1, 0.1
        mass = 4.0
        center = wp.vec3(0.0, 0.0, hz)
        rot = wp.quat_identity()
        self.box = Box(pos0=center, rot0=rot, hx=hx, hy=hy, hz=hz, mass=mass)

        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.box.add_body(builder)
        elif self.experiment == "srxpbd":
            builder = self.box.add_spheres(
                builder, num_spheres=self.args.num_spheres)
        return builder

    def launch_scenario_force_kernels(self):
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            n_bodies = self.state_0.body_f.shape[0]
            corner_body = wp.vec3(-self.box.hx, 0.0, -self.box.hz)
            F_world = wp.vec3(1.0, 0.0, 0.0)
            wp.launch(
                kernel=apply_force_at_body_corner,
                dim=n_bodies,
                inputs=[self.state_0.body_f, self.tau],
                device=wp.get_device(),
            )
        # elif self.experiment == "srxpbd":
        #     n = self.state_0.particle_f.shape[0]
        #     wp.launch(
        #         kernel=apply_constant_particle_force,
        #         dim=n,
        #         inputs=[self.state_0.particle_f, self.constant_force,
        #                 self.model.particle_mass, self.box.mass],
        #         device=wp.get_device(),
        #     )


    def calculate_analytical_state(self, t):
        """
        Analytical solution:
        - gravity = 0
        - constant torque about X
        - rotation about fixed ground edge
        """

        m = self.box.mass
        hy = self.box.hy
        hz = self.box.hz
        tau_x = self.tau.x

        # inertia about pivot edge (axis X)
        I_edge = (4.0 / 3.0) * m * (hy * hy + hz * hz)

        # angle and angular velocity
        omega = (tau_x / I_edge) * t
        
        theta = 0.5 * (tau_x / I_edge) * t * t

        # COM position (pivot at y = -hy, z = 0)
        x0 = self.box.pos0[0]

        y = -hy + hy * np.cos(theta) - hz * np.sin(theta)
        z =  hy * np.sin(theta) + hz * np.cos(theta)

        pos = np.array([x0, y, z], dtype=np.float32)

        # COM velocity
        ydot = (-hy * np.sin(theta) - hz * np.cos(theta)) * omega
        zdot = ( hy * np.cos(theta) - hz * np.sin(theta)) * omega

        vel = np.array([0.0, ydot, zdot], dtype=np.float32)

        # orientation (rotation about X)
        rot = np.array([
            np.sin(0.5 * theta),
            0.0,
            0.0,
            np.cos(0.5 * theta),
        ], dtype=np.float32)

        return None, wp.vec3(omega, 0.0, 0.0), None

    
    def collect_body_data(self, pos_analytic, omega, rot_analytic):
        # print(omega)

        body_idx = 0
        body_qd = self.state_0.body_qd.numpy().copy()
        ang_vel = body_qd[body_idx][3:]
        # print("Angular velocity:", ang_vel)
        
        self.collected_data["omega_err"].append(np.linalg.norm(ang_vel - omega))
        
        
        
        # body_q = self.state_0.body_q.numpy().copy()
        # body_qd = self.state_0.body_qd.numpy().copy()
        # data = collect_data_from_bodies(
        #     pos_analytic, vel_analytic, rot_analytic, body_q, body_qd, body_idx)
        # for key, value in data.items():
        #     self.collected_data[key].append(value)
        # pass

    def collect_sphere_packed_data(self, pos_analytic, vel_analytic, rot_analytic):
        # particle_q = self.state_0.particle_q.numpy().copy()
        # particle_qd = self.state_0.particle_qd.numpy().copy()
        # particle_q_init = self.box.particle_q_init
        # data = collect_data_from_spheres(pos_analytic=pos_analytic, vel_analytic=vel_analytic,
        #                                  rot_analytic=rot_analytic, particles_q=particle_q,
        #                                  particles_qd=particle_qd, particle_q_init=particle_q_init,
        #                                  rot_init=self.box.rot0)
        # for key, value in data.items():
        #     self.collected_data[key].append(value)
        pass


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "-n",
        "--num_spheres",
        type=int,
        default=4,
        help="Number of spheres per box dimension for SRXPBD experiment",
    )
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = PushedBoxExample(viewer, args=args)
    newton.examples.run(example, args)
