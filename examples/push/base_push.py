import warp as wp
import newton
import numpy as np
from examples.base_example import BaseExample
from utils.data_collector import setup_data_collection


class BasePush(BaseExample):
    def build_scene(self, builder):
        raise NotImplementedError

    def launch_scenario_force_kernels(self):
        raise NotImplementedError

    def __init__(self, viewer, args):
        np.random.seed(args.seed)
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
        self.mu = args.mu

        self.collected_data = setup_data_collection(self.frame_dt)
        builder = newton.ModelBuilder()
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=True,
            has_particle_collision=True,
        )
        builder.add_ground_plane(cfg=ground_cfg)
        builder = self.build_scene(builder)
        for shape_idx in range(builder.shape_count): # DO NOT USE BODY_COUNT, FOR COMPOSITE SHAPES, SHAPE_COUNT CAN BE GREATER THAN BODY_COUNT
            builder.shape_material_mu[shape_idx] = self.mu
        '''
        The solver calculates the friction between two surfaces as 1/2 (mu1 + mu2)
        By setting one object to mu and the other object to mu, the actual mu the solver uses is 1/2(mu + mu) = mu.
        For SRXPBD, contact mu can be only set after model is finalized: model.particle_mu
        When friction is between two particles the mu solver uses is exactly model.particle_mu thus self.particle_mu = self.mu
        '''
        for i in range(builder.shape_count):
            builder.shape_material_mu_torsional[i] = 0.0
            builder.shape_material_mu_rolling[i] = 0.0

        self.model = builder.finalize()
        if self.experiment == "tetxpbd":
            self.solver = newton.solvers.SolverXPBD(self.model,
                                                    iterations=self.num_iterations)
        elif self.experiment in ["srxpbd", "mrxpbd"]:
            self.model.particle_mu = self.mu # If between particles
            self.model.soft_contact_mu = self.mu # If between shape and particle
            self.solver = newton.solvers.SolverSRXPBD(self.model,
                                                      iterations=self.num_iterations)
        elif self.experiment == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model,
                                                      njmax=50,
                                                      iterations=self.num_iterations,
                                                      cone="elliptic",
                                                      )
        elif self.experiment == "semieuler":
            self.solver = newton.solvers.SolverSemiImplicit(self.model)

        elif self.experiment == "bxpbd":
            self.model.soft_contact_mu = self.mu
            self.solver = newton.solvers.SolverBXPBD(self.model,
                                                     iterations=self.num_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.launch_scenario_force_kernels()
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
