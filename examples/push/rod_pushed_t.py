import warp as wp
import numpy as np
import newton
from shapes.box import Box
from shapes.t_block import TBlock
from shapes.rod import Rod
from examples.push.base_push import BasePush
from examples.base_example import get_base_parser
from utils.data_collector import COLLECTOR_TYPES, collect_data_from_bodies, collect_data_from_spheres
from examples.push.utils import apply_constant_force_to_shape, apply_constant_particle_force_range
from newton import ParticleFlags, ShapeFlags
from examples.solutions import push_t_rod_solution


class RodPushedTExample(BasePush):
    def __init__(self, viewer=None, args=None):
        super().__init__(viewer, args)
        self.example_name = "rod_pushed_t"
        if self.experiment in ["srxpbd"]:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_nr{args.num_spheres_rod}_nt{args.num_spheres_tblock}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_i{self.num_iterations}.npz"
        self.constant_force = wp.vec3(*args.constant_force)

        # Override collected_data to include per-object and aggregate keys
        self.collected_data = {}
        base_keys = [COLLECTOR_TYPES.POS_A, COLLECTOR_TYPES.VEL_A, COLLECTOR_TYPES.ROT_A, 
                     COLLECTOR_TYPES.OMEGA_A, COLLECTOR_TYPES.ANG_MOM_A, COLLECTOR_TYPES.LIN_MOM_A,
                     COLLECTOR_TYPES.POS_S, COLLECTOR_TYPES.VEL_S, COLLECTOR_TYPES.ROT_S,
                     COLLECTOR_TYPES.OMEGA_S, COLLECTOR_TYPES.ANG_MOM_S, COLLECTOR_TYPES.LIN_MOM_S,
                     COLLECTOR_TYPES.FORCE_S,
                     COLLECTOR_TYPES.POS_E, COLLECTOR_TYPES.VEL_E, COLLECTOR_TYPES.ROT_E, 
                     COLLECTOR_TYPES.OMEGA_E, COLLECTOR_TYPES.ANG_MOM_E, COLLECTOR_TYPES.LIN_MOM_E,
                     COLLECTOR_TYPES.POS_L2_E]

        # Create per-object keys
        for key in base_keys:
            self.collected_data[key + "_rod"] = []
            self.collected_data[key + "_tblock"] = []
        self.collected_data[COLLECTOR_TYPES.DT] = self.frame_dt
        self.custom_rot_keys = [
            ('rot_analytic_rod', 'rot_solver_rod', 'rot_err_rod'),
            ('rot_analytic_tblock', 'rot_solver_tblock', 'rot_err_tblock'),
        ]

    def build_scene(self, builder):

        # T-block dimensions (laid down)
        stem_hx = 0.05
        stem_hy = 0.05
        stem_hz = 0.15
        
        crossbar_hx = 0.15
        crossbar_hy = 0.05
        crossbar_hz = 0.05
        
        # T-block center (laying down flat on ground)
        center_t = wp.vec3(0.0, 0.0, 0.05)
        
        # Rotate T-block 90 degrees around X axis so it lays down flat
        rot_t = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi/2.0)
        self.t_block = TBlock(
            pos0=center_t,
            rot0=rot_t,
            stem_hx=stem_hx,
            stem_hy=stem_hy,
            stem_hz=stem_hz,
            crossbar_hx=crossbar_hx,
            crossbar_hy=crossbar_hy,
            crossbar_hz=crossbar_hz,
            mass=1.0
        )
        rot_pos_t = wp.quat_rotate(rot_t, center_t)

        # Rod pushing against the T-block
        rod_radius = 0.03
        rod_length = 0.4
        # Position rod to contact the t_block
        crossbar_y_offset = rot_pos_t.y + stem_hz + crossbar_hy + rod_radius
        rod_y = crossbar_y_offset
        rod_z = rod_length / 2.0 

        rod_rot = wp.quat_identity()
        rod_pos = wp.vec3(0.0, rod_y, rod_z)
        self.rod = Rod(
            pos0=rod_pos,
            rot0=rod_rot,
            radius=rod_radius,
            length=rod_length,
            mass=2.0
        )

        # Add objects to the scene based on experiment type
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.t_block.add_body(builder)
            builder = self.rod.add_body(builder)
        elif self.experiment in ["srxpbd", "bxpbd"]:
            builder = self.t_block.add_spheres(builder, num_spheres=self.args.num_spheres_tblock)
            builder = self.rod.add_spheres(builder, num_spheres=self.args.num_spheres_rod)

        elif self.experiment == "mrxpbd":
            builder = self.t_block.add_morphit_spheres(builder, json_adrs=self.args.morphit_json)
            builder = self.rod.add_morphit_spheres(builder, json_adrs=self.args.morphit_json)
            
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder.shape_flags[self.rod.shape_idx] = ShapeFlags.INTEGRATE_ONLY | ShapeFlags.VISIBLE | ShapeFlags.COLLIDE_SHAPES

        elif self.experiment in ["srxpbd", "mrxpbd", "bxpbd"]:
            rod_particle_start = self.rod.particle_start_idx
            rod_particle_end = self.rod.particle_end_idx
            num_particles_rod = rod_particle_end - rod_particle_start
            builder.particle_flags[rod_particle_start:rod_particle_end] = [ParticleFlags.INTEGRATE_ONLY | ParticleFlags.ACTIVE] * num_particles_rod
        return builder

    def launch_scenario_force_kernels(self):
        """Apply constant force to the rod (pushing the T-block)"""
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            n_bodies = self.state_0.body_f.shape[0]
            wp.launch(
                kernel=apply_constant_force_to_shape,
                dim=n_bodies,
                inputs=[self.state_0.body_f, self.constant_force, self.rod.body_idx],
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
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            x0_rod = self.rod.pos0
            x0_tblock = self.t_block.pos0
        elif self.experiment in ["srxpbd", "mrxpbd", "bxpbd"]:
            x0_rod = self.rod.pos0
            x0_tblock = self.t_block.pos0
        return push_t_rod_solution(x0_rod=x0_rod,
                                   x0_tblock=x0_tblock,
                                   m_rod=self.rod.mass,
                                   m_tblock=self.t_block.mass,
                                   rod_force=self.constant_force,
                                   t=t)

    def collect_body_data(self, 
                          pos_rod_analytic, 
                          pos_tblock_analytic, 
                          vel_analytic, 
                          rot_analytic, 
                          omega_analytic, 
                          L_analytic, 
                          p_rod_analytic, 
                          p_tblock_analytic):
        """Collect data for both rod and T-block (rigid body case)"""
        body_q = self.state_0.body_q.numpy().copy()
        body_qd = self.state_0.body_qd.numpy().copy()
        body_f = self.state_0.body_f.numpy().copy()
        I_rod = np.asarray(self.rod.I_m).reshape((3, 3))
        I_tblock = np.asarray(self.t_block.I_m).reshape((3, 3))

        # Collect data for rod (index 0)
        data_rod = collect_data_from_bodies(pos_analytic=pos_rod_analytic, 
                                            vel_analytic=vel_analytic, 
                                            rot_analytic=rot_analytic, 
                                            omega_analytic=omega_analytic, 
                                            L_analytic=L_analytic,
                                            p_analytic=p_rod_analytic,
                                            body_q=body_q, 
                                            body_qd=body_qd, 
                                            I=I_rod, 
                                            mass=self.rod.mass,
                                            body_idx=self.rod.body_idx,
                                            body_f=body_f)
        # Compute missing error keys
        data_rod[COLLECTOR_TYPES.OMEGA_E] = omega_analytic - data_rod[COLLECTOR_TYPES.OMEGA_S]
        data_rod[COLLECTOR_TYPES.ANG_MOM_E] = L_analytic - data_rod[COLLECTOR_TYPES.ANG_MOM_S]
        data_rod[COLLECTOR_TYPES.LIN_MOM_E] = p_rod_analytic - (self.rod.mass * data_rod[COLLECTOR_TYPES.VEL_S])
        
        for key, value in data_rod.items():
            self.collected_data[key + "_rod"].append(value)
        
        # Collect data for T-block (index 1)
        data_tblock = collect_data_from_bodies(pos_analytic=pos_tblock_analytic, 
                                                vel_analytic=vel_analytic, 
                                                rot_analytic=rot_analytic,
                                                omega_analytic=omega_analytic, 
                                                L_analytic=L_analytic,
                                                p_analytic=p_tblock_analytic,
                                                body_q=body_q, 
                                                body_qd=body_qd, 
                                                I=I_tblock, 
                                                mass=self.t_block.mass,
                                                body_idx=self.t_block.body_idx,
                                                body_f=body_f)
        # Compute missing error keys
        data_tblock[COLLECTOR_TYPES.OMEGA_E] = omega_analytic - data_tblock[COLLECTOR_TYPES.OMEGA_S]
        data_tblock[COLLECTOR_TYPES.ANG_MOM_E] = L_analytic - data_tblock[COLLECTOR_TYPES.ANG_MOM_S]
        data_tblock[COLLECTOR_TYPES.LIN_MOM_E] = p_tblock_analytic - (self.t_block.mass * data_tblock[COLLECTOR_TYPES.VEL_S])
        
        for key, value in data_tblock.items():
            self.collected_data[key + "_tblock"].append(value)

    def collect_sphere_packed_data(self, 
                                   pos_rod_analytic, 
                                   pos_tblock_analytic, 
                                   vel_analytic, 
                                   rot_analytic, 
                                   omega_analytic, 
                                   L_analytic, 
                                   p_rod_analytic, 
                                   p_tblock_analytic):
        """Collect data for particle-based rod and T-block"""
        particle_q = self.state_0.particle_q.numpy().copy()
        particle_qd = self.state_0.particle_qd.numpy().copy()
        particle_m = self.model.particle_mass.numpy()
        particle_f = self.state_0.particle_f.numpy().copy()
        
        # Extract particles belonging to the rod
        start_idx_rod = self.rod.particle_start_idx
        end_idx_rod = self.rod.particle_end_idx
        particle_q_rod = particle_q[start_idx_rod:end_idx_rod]
        particle_qd_rod = particle_qd[start_idx_rod:end_idx_rod]
        particle_m_rod = particle_m[start_idx_rod:end_idx_rod]
        rod_particle_indices = np.arange(start_idx_rod, end_idx_rod)
        
        data_rod = collect_data_from_spheres(pos_analytic=pos_rod_analytic, 
                                            vel_analytic=vel_analytic, 
                                            rot_analytic=rot_analytic, 
                                            omega_analytic=omega_analytic, 
                                            L_analytic=L_analytic,
                                            p_analytic=p_rod_analytic,
                                            particles_q=particle_q_rod, 
                                            particles_qd=particle_qd_rod, 
                                            particle_m=particle_m_rod,
                                            prev_rot_solver=self.rod.rot0, 
                                            dt=self.frame_dt,
                                            particle_f=particle_f,
                                            particle_indices=rod_particle_indices)
        # Compute missing error keys
        total_mass_rod = particle_m_rod.sum()
        data_rod[COLLECTOR_TYPES.ANG_MOM_E] = L_analytic - data_rod[COLLECTOR_TYPES.ANG_MOM_S]
        data_rod[COLLECTOR_TYPES.LIN_MOM_E] = p_rod_analytic - (total_mass_rod * data_rod[COLLECTOR_TYPES.VEL_S])
        
        for key, value in data_rod.items():
            self.collected_data[key + "_rod"].append(value)
        
        # Extract particles belonging to the T-block
        start_idx_tblock = self.t_block.particle_start_idx
        end_idx_tblock = self.t_block.particle_end_idx
        particle_q_tblock = particle_q[start_idx_tblock:end_idx_tblock]
        particle_qd_tblock = particle_qd[start_idx_tblock:end_idx_tblock]
        particle_m_tblock = particle_m[start_idx_tblock:end_idx_tblock]
        tblock_particle_indices = np.arange(start_idx_tblock, end_idx_tblock)
        
        data_tblock = collect_data_from_spheres(pos_analytic=pos_tblock_analytic, 
                                                vel_analytic=vel_analytic, 
                                                rot_analytic=rot_analytic, 
                                                omega_analytic=omega_analytic, 
                                                L_analytic=L_analytic,
                                                p_analytic=p_tblock_analytic,
                                                particles_q=particle_q_tblock, 
                                                particles_qd=particle_qd_tblock,
                                                particle_m=particle_m_tblock,
                                                prev_rot_solver=self.t_block.rot0, 
                                                dt=self.frame_dt,
                                                particle_f=particle_f,
                                                particle_indices=tblock_particle_indices)
        # Compute missing error keys
        total_mass_tblock = particle_m_tblock.sum()
        data_tblock[COLLECTOR_TYPES.ANG_MOM_E] = L_analytic - data_tblock[COLLECTOR_TYPES.ANG_MOM_S]
        data_tblock[COLLECTOR_TYPES.LIN_MOM_E] = p_tblock_analytic - (total_mass_tblock * data_tblock[COLLECTOR_TYPES.VEL_S])
        
        for key, value in data_tblock.items():
            self.collected_data[key + "_tblock"].append(value)

if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "--num_spheres_rod",
        type=int,
        default=4,
        help="Number of spheres per dimension for the rod (SRXPBD only).",
    )
    parser.add_argument(
        "--num_spheres_tblock",
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
                'pos_l2_err_rod', 
                'rot_err_rod',
                'rot_err_tblock',
                'pos_l2_err_tblock',
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

    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = RodPushedTExample(viewer, args=args)
    newton.examples.run(example, args)
