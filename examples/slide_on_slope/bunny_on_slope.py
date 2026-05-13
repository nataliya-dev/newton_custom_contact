import numpy as np
import warp as wp
import newton
from examples.solutions import obj_on_slope_solution
from utils.mesh_helper import calculate_z_lowest_mesh, calculate_z_lowest_sphere_set
from shapes.stanford_bunny import StanfordBunny
from shapes.slope import Slope
from examples.base_example import get_base_parser
from examples.slide_on_slope.base_slide import BaseSlide
from utils.data_collector import COLLECTOR_TYPES, collect_data_from_bodies, collect_data_from_spheres
from utils.misc import get_object_on_slope_initial_pose


class BunnyOnSlopeExample(BaseSlide):
    def __init__(self, viewer=None, args=None, plot_sim_time=10.0):
        super().__init__(viewer, args)
        self.sphere_radius = args.sphere_radius
        self.example_name = "bunny_on_slope"
        if self.experiment in ["srxpbd", "bxpbd"]:
            n = len(self.bunny.particle_q_init)
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_n{n}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_i{self.num_iterations}.npz"

    def build_scene(self, builder):
        builder = self.add_slope(builder)

        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.add_object_on_slope(builder, type="mesh")
        elif self.experiment in ["srxpbd", "bxpbd"]:
            builder = self.add_object_on_slope(builder, type="spheres")
        elif self.experiment in ["mrxpbd"]:
            builder = self.add_object_on_slope(builder, type="morphit")
        return builder

    def add_slope(self, builder):
        self.slope = Slope(
            pos0=wp.vec3(0.0, -0.5, 0.1),
            rot0=wp.quat_from_axis_angle(
                wp.vec3(1.0, 0.0, 0.0), -wp.pi/8),
            hx=0.2,
            hy=8.0,
            hz=0.0,
        )
        if self.use_sphere_slope:
            if self.experiment in ["srxpbd", "bxpbd"]:
                builder = self.slope.add_spheres(builder, num_spheres_x=self.args.num_spheres_slope_x)
            elif self.experiment == "mrxpbd":
                builder = self.slope.add_morphit_spheres(builder, json_adrs=self.args.morphit_slope_json)
        else:
            builder = self.slope.add_body(builder)
        return builder

    def add_object_on_slope(self, builder, type):
        up_slope_offset = -7.0
        if type == "morphit":
            offset_z = calculate_z_lowest_sphere_set(self.args.morphit_json)
            center, rot = get_object_on_slope_initial_pose(
                slope=self.slope, up_slope_offset=up_slope_offset, object_bottom_offset_z=offset_z)
        else:
            bunny_hz = calculate_z_lowest_mesh(
                StanfordBunny.obj_path, transform=None)
            center, rot = get_object_on_slope_initial_pose(
                slope=self.slope, up_slope_offset=up_slope_offset, object_bottom_offset_z=bunny_hz)

        self.bunny = StanfordBunny(pos0=center, rot0=rot)
        if type == "mesh":
            builder = self.bunny.add_mesh(builder)
        elif type == "spheres":
            builder = self.bunny.add_spheres(
                builder, radius=self.args.sphere_radius)
        elif type == "morphit":
            builder = self.bunny.add_morphit_spheres(
                builder, json_adrs=self.args.morphit_json)
        return builder

    def calculate_analytical_state(self, t):
        x0 = self.bunny.pos0
        rot0 = self.bunny.rot0
        return obj_on_slope_solution(x0=x0, rot0=rot0, mu=self.mu, t=t, slope_rot=self.slope.rot0, m=self.bunny.mass)

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
        "--num_spheres_slope_x",
        type=int,
        default=4,
        help="Number of spheres to use along the minimum slope dimension, only used with sphere slope",
    )
    parser.add_argument(
        "--use_sphere_slope",
        action="store_true",
        help="Represent the slope with packed spheres instead of a mesh (only for SRXPBD/MRXPBD/BXPBD)"
    )
    parser.add_argument(
        "-k",
        "--plot_keys",
        nargs='+',
        type=str,
        default=[COLLECTOR_TYPES.POS_L2_E, COLLECTOR_TYPES.ROT_E])
    parser.add_argument("--mu",
        type=float,
        default=0.0,
        help="Mu between the object and the slope (Coulomb friction). Default 0 = frictionless.")
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = BunnyOnSlopeExample(viewer, args=args)
    newton.examples.run(example, args)
