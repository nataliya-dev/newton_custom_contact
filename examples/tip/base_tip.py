import warp as wp
import newton
from examples.base_example import BaseExample
from utils.data_collector import COLLECTOR_TYPES


class BaseTip(BaseExample):
    def build_scene(self, builder):
        raise NotImplementedError

    def launch_scenario_force_kernels(self):
        raise NotImplementedError

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

        builder = newton.ModelBuilder(gravity=0) # No gravity for tipping scenario
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.0,
            kf=0.0,
            has_shape_collision=True,
            has_particle_collision=True,
        )
        builder.add_ground_plane(cfg=ground_cfg)
        builder = self.build_scene(builder)

        for shape_idx in range(len(builder.shape_material_ke)):
            builder.shape_material_mu[shape_idx] = 1e10 # Large friction to avoid sliding

        self.model = builder.finalize()

        if self.experiment == "tetxpbd":
            self.solver = newton.solvers.SolverXPBD(self.model,
                                                    iterations=self.num_iterations)
        elif self.experiment == "srxpbd":
            self.model.particle_mu = 0.0
            self.model.soft_contact_mu = 0.0
            self.solver = newton.solvers.SolverSRXPBD(self.model,
                                                      iterations=self.num_iterations)
        elif self.experiment == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model,
                                                      njmax=20,
                                                      iterations=self.num_iterations)
        elif self.experiment == "semieuler":
            self.solver = newton.solvers.SolverSemiImplicit(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        self.collected_data = {
            COLLECTOR_TYPES.POS_A: [],
            COLLECTOR_TYPES.VEL_A: [],
            COLLECTOR_TYPES.ROT_A: [],
            COLLECTOR_TYPES.POS_S: [],
            COLLECTOR_TYPES.VEL_S: [],
            COLLECTOR_TYPES.ROT_S: [],
            COLLECTOR_TYPES.POS_E: [],
            COLLECTOR_TYPES.POS_L2_E: [],
            COLLECTOR_TYPES.VEL_E: [],
            COLLECTOR_TYPES.ROT_E: [],
            "omega_err": [],
            COLLECTOR_TYPES.DT: self.frame_dt,
        }
        
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.launch_scenario_force_kernels()
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0


    def plot_data(self, keys=["omega_err"]):
        return super().plot_data(keys)