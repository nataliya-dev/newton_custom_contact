import warp as wp
import numpy as np
import newton
from shapes.torus import TorusB, TorusS
from utils.data_collector import setup_data_collection, COLLECTOR_TYPES
from utils.mesh_helper import calculate_z_lowest_sphere_set, calculate_z_lowest_mesh
from examples.base_example import BaseExample, get_base_parser

class Tori(BaseExample):
    def __init__(self, viewer, args):
        wp.config.cache_kernels = args.cache_kernels
        self.sim_time = 0.0
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.experiment = args.experiment
        self.num_iterations = args.num_iterations
        self.args = args
        self.plot_sim_time = args.plot_sim_time
        self.plot_keys = args.plot_keys
        self.mu = getattr(args, 'mu', 0.0)
        
        self.collected_data = setup_data_collection(self.frame_dt)
        builder = newton.ModelBuilder()
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=True,
            has_particle_collision=True,
        )
        builder.add_ground_plane(cfg=ground_cfg)
        builder.shape_material_mu[0] = 0.0 # Ground
        builder = self.build_scene(builder)
        if self.experiment in ["mujoco", "tetxpbd", "semieuler"]:
            builder.shape_material_mu[1] = 2 * self.mu # Object on the ground
        '''
        The solver calculates the friction between two surfaces as 1/2 (mu1 + mu2)
        By setting ground to 0 and object to 2*mu, the actual mu the solver uses is 1/2(0 + 2 * mu) = mu.
        For SRXPBD, contact mu can be only set after model is finalized.
        '''
        self.model = builder.finalize()

        if self.experiment == "tetxpbd":
            self.solver = newton.solvers.SolverXPBD(self.model,
                                                    iterations=self.num_iterations)
        elif self.experiment in ["srxpbd", "mrxpbd"]:
            self.model.soft_contact_mu = 2 * self.mu
            self.solver = newton.solvers.SolverSRXPBD(self.model,
                                                      iterations=self.num_iterations)
        elif self.experiment == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model,
                                                      njmax=20,
                                                      iterations=self.num_iterations)
        elif self.experiment == "semieuler":
            self.solver = newton.solvers.SolverSemiImplicit(self.model)

        elif self.experiment == "bxpbd":
            self.model.soft_contact_mu = 2 * self.mu
            self.solver = newton.solvers.SolverBXPBD(self.model,
                                                     iterations=self.num_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.example_name = "tori" # Plural of torus :o
        if self.experiment in ["srxpbd", "bxpbd"]:
            self.sphere_radius = args.sphere_radius
            n = len(self.torus_b.particle_q_init)
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_n{n}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_i{self.num_iterations}.npz"


    def build_scene(self, builder):
        rot = wp.quat_identity()
        offset = self.args.torus_offset
        torus_b_hz = calculate_z_lowest_mesh(TorusB.obj_path, transform=None)
        torus_s_hz = calculate_z_lowest_mesh(TorusS.obj_path, transform=None)
        center_b = wp.vec3(0.0, 0.0, torus_b_hz)
        center_s = wp.vec3(0.0, 0.0, torus_b_hz + torus_s_hz + offset)
        self.torus_b = TorusB(pos0=center_b, rot0=rot)
        self.torus_s = TorusS(pos0=center_s, rot0=rot)

        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.torus_b.add_mesh(builder)
            builder = self.torus_s.add_mesh(builder)
        elif self.experiment in ["srxpbd", "bxpbd"]:
            builder = self.torus_b.add_spheres(builder, radius=self.args.sphere_radius)
            builder = self.torus_s.add_spheres(builder, radius=self.args.sphere_radius)
        elif self.experiment == "mrxpbd":
            builder = self.torus_b.add_morphit_spheres(builder, json_adrs=self.args.morphit_json)
            builder = self.torus_s.add_morphit_spheres(builder, json_adrs=self.args.morphit_json)

        return builder

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def calculate_analytical_state(self, t): 
        return [0]

    def collect_body_data(self, ignore):
        pass

    def collect_sphere_packed_data(self, ignore):
        pass

if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "-r",
        "--sphere_radius",
        type=float,
        default=0.05,
        help="radii of spheres",
    )
    parser.add_argument(
        "--torus_offset",
        type=float,
        default=1.0,
        help="vertical offset between the two tori",
    )
    parser.add_argument(
        "-k",
        "--plot_keys",
        nargs='+',
        type=str,
        default=[],
    )
    parser.add_argument("--mu",
                        type=float,
                        default=1e-4,
                        help="Mu between the object and ground (Coulomb friction). Default 1e-4 because of MuJoCo.")

    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = Tori(viewer, args=args)
    newton.examples.run(example, args)
