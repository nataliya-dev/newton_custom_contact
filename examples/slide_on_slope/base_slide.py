import numpy as np
import warp as wp
import os
import newton
import newton.examples
from examples.base_example import BaseExample
from examples.push.utils import apply_constant_body_force, apply_constant_particle_force
from utils.data_collector import COLLECTOR_TYPES, setup_data_collection


class BaseSlide(BaseExample):
    def build_scene(self, builder):
        raise NotImplementedError
    
    def add_slope(self, builder):
        raise NotImplementedError

    def add_object_on_slope(self, builder, type):
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
        self.args = args
        self.experiment = args.experiment
        self.num_iterations = args.num_iterations
        self.plot_sim_time = args.plot_sim_time
        self.plot_keys = args.plot_keys
        self.collected_data = setup_data_collection(self.frame_dt)
        self.slope = None
        self.mu = args.mu
        self.use_sphere_slope = args.use_sphere_slope
        if self.use_sphere_slope and self.experiment not in ["srxpbd", "mrxpbd"]:
            print("Warning: sphere slope only supported with srxpbd/mrxpbd. Falling back to mesh.")
            self.use_sphere_slope = False

        builder = newton.ModelBuilder()
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=True,
            has_particle_collision=True,
        )
        builder.add_ground_plane(cfg=ground_cfg)
        builder = self.build_scene(builder)
        for shape_index in range(builder.shape_count):
            builder.shape_material_mu[shape_index] = self.mu
        
        '''
        The solver calculates the friction between two surfaces as 1/2 (mu1 + mu2)
        By setting one object to mu and the other object to mu, the actual mu the solver uses is 1/2(mu + mu) = mu.
        For SRXPBD, contact mu can be only set after model is finalized.
        When friction is between two particles the mu solver uses is exactly model.particle_mu thus this:
        if self.use_sphere_slope: particle-particle contact
            self.model.particle_mu = self.mu
        '''
        for i in range(builder.body_count):
            builder.shape_material_mu_torsional[i] = 0.0
            builder.shape_material_mu_rolling[i] = 0.0

        self.model = builder.finalize()
        if self.experiment == "tetxpbd":
            self.solver = newton.solvers.SolverXPBD(self.model,
                                                    iterations=self.num_iterations)
        elif self.experiment in ["srxpbd", "mrxpbd"]:
            self.model.particle_mu = self.mu
            self.model.soft_contact_mu = self.mu
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
            if self.use_sphere_slope:
                self.model.particle_mu = self.mu
            else:
                self.model.soft_contact_mu = self.mu
            self.solver = newton.solvers.SolverBXPBD(self.model,
                                                     iterations=self.num_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        newton.eval_fk(self.model, self.model.joint_q,
                       self.model.joint_qd, self.state_0)

        if not self.use_sphere_slope:
            self.viewer.update_shape_colors({1: wp.vec3(0.8, 0.2, 0.2)})  # red slope
        
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1,
                             self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
