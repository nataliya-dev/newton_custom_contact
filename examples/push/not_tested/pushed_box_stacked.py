import warp as wp
import numpy as np
import newton
from shapes.box import Box
from examples.push.base_push import BasePush
from examples.base_example import get_base_parser
from utils.data_collector import COLLECTOR_TYPES, collect_data_from_bodies, collect_data_from_spheres
from examples.push.utils import apply_constant_body_force, apply_constant_particle_force
from utils.mesh_helper import calculate_z_lowest_sphere_set, calculate_box_dimensions_from_sphere_set

class PushedBoxStackedExample(BasePush):
    def __init__(self, viewer=None, args=None, plot_sim_time=10.0):
        super().__init__(viewer, args)
        self.example_name = "pushed_box_stacked"
        if self.experiment in ["srxpbd"]:
            self.num_spheres = args.num_spheres
            n = self.num_spheres * self.num_spheres * self.num_spheres
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_n{n}_i{self.num_iterations}.npz"
        else:
            self.exp_address = f"outputs/{self.example_name}/{self.experiment}_i{self.num_iterations}.npz"
        self.plot_sim_time = plot_sim_time
        self.constant_force = wp.vec3(2.0, 0.0, 0.0)

        # Override collected_data to include per-object and aggregate keys
        self.collected_data = {}
        base_keys = [COLLECTOR_TYPES.POS_A, COLLECTOR_TYPES.VEL_A, COLLECTOR_TYPES.ROT_A, 
                     COLLECTOR_TYPES.OMEGA_A, COLLECTOR_TYPES.ANG_MOM_A, COLLECTOR_TYPES.LIN_MOM_A,
                     COLLECTOR_TYPES.POS_S, COLLECTOR_TYPES.VEL_S, COLLECTOR_TYPES.ROT_S,
                     COLLECTOR_TYPES.OMEGA_S, COLLECTOR_TYPES.ANG_MOM_S, COLLECTOR_TYPES.LIN_MOM_S,
                     COLLECTOR_TYPES.POS_E, COLLECTOR_TYPES.VEL_E, COLLECTOR_TYPES.ROT_E, 
                     COLLECTOR_TYPES.OMEGA_E, COLLECTOR_TYPES.ANG_MOM_E, COLLECTOR_TYPES.LIN_MOM_E,
                     COLLECTOR_TYPES.POS_L2_E]
        
        # Create per-object keys
        for key in base_keys:
            self.collected_data[key + "_bottom"] = []
            self.collected_data[key + "_top"] = []
        
        # Add aggregate error keys
        self.collected_data['pos_l2_err_avg'] = []
        self.collected_data['vel_err_avg'] = []
        self.collected_data['rot_err_avg'] = []
        self.collected_data['omega_err_avg'] = []
        self.collected_data['ang_mom_err_avg'] = []
        self.collected_data['lin_mom_err_avg'] = []
        
        # Metadata
        self.collected_data[COLLECTOR_TYPES.DT] = self.frame_dt

        # Define custom rotation keys for unwrapping (per-object rotation errors)
        # This is necessary for experiments with multiple objects to ensure we unwrap the correct
        # rotation keys for each object when plotting
        self.custom_rot_keys = [
            ('rot_analytic_bottom', 'rot_solver_bottom', 'rot_err_bottom'),
            ('rot_analytic_top', 'rot_solver_top', 'rot_err_top'),
        ]

    def build_scene(self, builder):
        rot = wp.quat_identity()
        mass_bottom = 4.0
        mass_top = 2.0
        # Box dimensions
        if self.experiment == "mrxpbd":
            offset_z = calculate_z_lowest_sphere_set(self.args.morphit_json)
            hx, hy, hz = calculate_box_dimensions_from_sphere_set(self.args.morphit_json)
        else:
            hx = hy = hz = 0.1

        # Bottom box (will be pushed)
        center_bottom = wp.vec3(0.0, 0.0, hz)
        self.box_bottom = Box(pos0=center_bottom, rot0=rot, hx=hx, hy=hy, hz=hz, mass=mass_bottom)

        # Top box (sitting on the bottom box)
        center_top = wp.vec3(0.0, 0.0, 3.0 * hz)
        self.box_top = Box(pos0=center_top, rot0=rot, hx=hx, hy=hy, hz=hz, mass=mass_top)

        # Add boxes to the scene based on experiment type
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            builder = self.box_bottom.add_body(builder)
            builder = self.box_top.add_body(builder)
        elif self.experiment == "srxpbd":
            builder = self.box_bottom.add_spheres(builder, num_spheres=self.args.num_spheres)
            builder = self.box_top.add_spheres(builder, num_spheres=self.args.num_spheres)
        elif self.experiment == "mrxpbd":
            builder = self.box_bottom.add_morphit_spheres(builder, json_adrs=self.args.morphit_json)
            builder = self.box_top.add_morphit_spheres(builder, json_adrs=self.args.morphit_json)

        return builder

    def launch_scenario_force_kernels(self):
        """
        Apply constant force only to the bottom box.
        The top box will accelerate through contact forces.
        """
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            # For rigid bodies, apply force only to bottom box (body index 0)
            wp.launch(
                kernel=apply_constant_body_force,
                dim=1,
                inputs=[self.state_0.body_f, self.constant_force],
                device=wp.get_device(),
            )
        elif self.experiment in ["srxpbd", "mrxpbd"]:
            # For particle-based, apply force proportionally to particles of bottom box
            n = self.state_0.particle_f.shape[0]
            wp.launch(
                kernel=apply_constant_particle_force,
                dim=n,
                inputs=[self.state_0.particle_f, self.constant_force,
                        self.model.particle_mass, self.box_bottom.particle_mass_sum],
                device=wp.get_device(),
            )

    def calculate_analytical_state(self, t):
        """
        Analytical solution assuming the two boxes move together as a combined mass.
        This is valid as long as they remain in contact.
        
        Combined acceleration: a = F / (m_bottom + m_top)
        Position: x = x0 + 0.5 * a * t^2
        Velocity: v = a * t
        Omega: w = 0 (no rotation applied)
        Angular momentum: L = 0 (no rotation)
        Linear momentum: p = m * v
        """
        total_mass = self.box_bottom.mass + self.box_top.mass
        a = self.constant_force / total_mass
        
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            x0 = self.box_bottom.pos0
        elif self.experiment in ["srxpbd", "mrxpbd"]:
            x0 = self.box_bottom.particle_com_init
        
        rot = wp.quat_identity()
        pos = x0 + 0.5 * a * (t * t)
        vel = a * t
        omega = np.array([0.0, 0.0, 0.0])
        L = np.array([0.0, 0.0, 0.0])
        p = total_mass * vel
        
        return np.asarray(pos), np.asarray(vel), np.asarray(rot), np.asarray(omega), np.asarray(L), np.asarray(p)

    def collect_body_data(self, pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, p_analytic):
        """Collect data for both boxes (rigid body case)"""
        body_q = self.state_0.body_q.numpy().copy()
        body_qd = self.state_0.body_qd.numpy().copy()
        I_body_bottom = np.asarray(self.box_bottom.I_m).reshape((3, 3))
        I_body_top = np.asarray(self.box_top.I_m).reshape((3, 3))

        # Collect data for bottom box (index 0)
        data_bottom = collect_data_from_bodies(
            pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, 
            body_q, body_qd, I_body_bottom, body_idx=0)
        # Compute missing error keys
        data_bottom[COLLECTOR_TYPES.OMEGA_E] = omega_analytic - data_bottom[COLLECTOR_TYPES.OMEGA_S]
        data_bottom[COLLECTOR_TYPES.ANG_MOM_E] = L_analytic - data_bottom[COLLECTOR_TYPES.ANG_MOM_S]
        data_bottom[COLLECTOR_TYPES.LIN_MOM_E] = p_analytic - (self.box_bottom.mass * data_bottom[COLLECTOR_TYPES.VEL_S])
        
        for key, value in data_bottom.items():
            self.collected_data[key + "_bottom"].append(value)
        
        # Collect data for top box (index 1)
        data_top = collect_data_from_bodies(
            pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic,
            body_q, body_qd, I_body_top, body_idx=1)
        # Compute missing error keys
        data_top[COLLECTOR_TYPES.OMEGA_E] = omega_analytic - data_top[COLLECTOR_TYPES.OMEGA_S]
        data_top[COLLECTOR_TYPES.ANG_MOM_E] = L_analytic - data_top[COLLECTOR_TYPES.ANG_MOM_S]
        data_top[COLLECTOR_TYPES.LIN_MOM_E] = p_analytic - (self.box_top.mass * data_top[COLLECTOR_TYPES.VEL_S])
        
        for key, value in data_top.items():
            self.collected_data[key + "_top"].append(value)
        
        # Compute and store aggregate errors (average across both boxes)
        pos_l2_err_avg = (data_bottom[COLLECTOR_TYPES.POS_L2_E] + data_top[COLLECTOR_TYPES.POS_L2_E]) / 2.0
        vel_err_avg = (np.linalg.norm(data_bottom[COLLECTOR_TYPES.VEL_E]) + np.linalg.norm(data_top[COLLECTOR_TYPES.VEL_E])) / 2.0
        rot_err_avg = (data_bottom[COLLECTOR_TYPES.ROT_E] + data_top[COLLECTOR_TYPES.ROT_E]) / 2.0
        omega_err_avg = (np.linalg.norm(data_bottom[COLLECTOR_TYPES.OMEGA_E]) + np.linalg.norm(data_top[COLLECTOR_TYPES.OMEGA_E])) / 2.0
        ang_mom_err_avg = (np.linalg.norm(data_bottom[COLLECTOR_TYPES.ANG_MOM_E]) + np.linalg.norm(data_top[COLLECTOR_TYPES.ANG_MOM_E])) / 2.0
        lin_mom_err_avg = (np.linalg.norm(data_bottom[COLLECTOR_TYPES.LIN_MOM_E]) + np.linalg.norm(data_top[COLLECTOR_TYPES.LIN_MOM_E])) / 2.0
        
        self.collected_data['pos_l2_err_avg'].append(pos_l2_err_avg)
        self.collected_data['vel_err_avg'].append(vel_err_avg)
        self.collected_data['rot_err_avg'].append(rot_err_avg)
        self.collected_data['omega_err_avg'].append(omega_err_avg)
        self.collected_data['ang_mom_err_avg'].append(ang_mom_err_avg)
        self.collected_data['lin_mom_err_avg'].append(lin_mom_err_avg)

    def collect_sphere_packed_data(self, pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic, p_analytic):
        """Collect data for particle-based boxes"""
        particle_q = self.state_0.particle_q.numpy().copy()
        particle_qd = self.state_0.particle_qd.numpy().copy()
        particle_m = self.model.particle_mass.numpy()
        
        # Extract particles belonging to the bottom box
        start_idx_bottom = self.box_bottom.particle_start_idx
        end_idx_bottom = self.box_bottom.particle_end_idx
        particle_q_bottom = particle_q[start_idx_bottom:end_idx_bottom]
        particle_qd_bottom = particle_qd[start_idx_bottom:end_idx_bottom]
        particle_m_bottom = particle_m[start_idx_bottom:end_idx_bottom]
        
        data_bottom = collect_data_from_spheres(
            pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic,
            particle_q_bottom, particle_qd_bottom, particle_m_bottom,
            self.box_bottom.rot0, self.frame_dt)
        # Compute missing error keys
        total_mass_bottom = particle_m_bottom.sum()
        data_bottom[COLLECTOR_TYPES.ANG_MOM_E] = L_analytic - data_bottom[COLLECTOR_TYPES.ANG_MOM_S]
        data_bottom[COLLECTOR_TYPES.LIN_MOM_E] = p_analytic - (total_mass_bottom * data_bottom[COLLECTOR_TYPES.VEL_S])
        
        for key, value in data_bottom.items():
            self.collected_data[key + "_bottom"].append(value)
        
        # Extract particles belonging to the top box
        start_idx_top = self.box_top.particle_start_idx
        end_idx_top = self.box_top.particle_end_idx
        particle_q_top = particle_q[start_idx_top:end_idx_top]
        particle_qd_top = particle_qd[start_idx_top:end_idx_top]
        particle_m_top = particle_m[start_idx_top:end_idx_top]
        
        data_top = collect_data_from_spheres(
            pos_analytic, vel_analytic, rot_analytic, omega_analytic, L_analytic,
            particle_q_top, particle_qd_top, particle_m_top,
            self.box_top.rot0, self.frame_dt)
        # Compute missing error keys
        total_mass_top = particle_m_top.sum()
        data_top[COLLECTOR_TYPES.ANG_MOM_E] = L_analytic - data_top[COLLECTOR_TYPES.ANG_MOM_S]
        data_top[COLLECTOR_TYPES.LIN_MOM_E] = p_analytic - (total_mass_top * data_top[COLLECTOR_TYPES.VEL_S])
        
        for key, value in data_top.items():
            self.collected_data[key + "_top"].append(value)
        
        # Compute and store aggregate errors (average across both boxes)
        pos_l2_err_avg = (data_bottom[COLLECTOR_TYPES.POS_L2_E] + data_top[COLLECTOR_TYPES.POS_L2_E]) / 2.0
        vel_err_avg = (np.linalg.norm(data_bottom[COLLECTOR_TYPES.VEL_E]) + np.linalg.norm(data_top[COLLECTOR_TYPES.VEL_E])) / 2.0
        rot_err_avg = (data_bottom[COLLECTOR_TYPES.ROT_E] + data_top[COLLECTOR_TYPES.ROT_E]) / 2.0
        omega_err_avg = (np.linalg.norm(data_bottom[COLLECTOR_TYPES.OMEGA_E]) + np.linalg.norm(data_top[COLLECTOR_TYPES.OMEGA_E])) / 2.0
        ang_mom_err_avg = (np.linalg.norm(data_bottom[COLLECTOR_TYPES.ANG_MOM_E]) + np.linalg.norm(data_top[COLLECTOR_TYPES.ANG_MOM_E])) / 2.0
        lin_mom_err_avg = (np.linalg.norm(data_bottom[COLLECTOR_TYPES.LIN_MOM_E]) + np.linalg.norm(data_top[COLLECTOR_TYPES.LIN_MOM_E])) / 2.0
        
        self.collected_data['pos_l2_err_avg'].append(pos_l2_err_avg)
        self.collected_data['vel_err_avg'].append(vel_err_avg)
        self.collected_data['rot_err_avg'].append(rot_err_avg)
        self.collected_data['omega_err_avg'].append(omega_err_avg)
        self.collected_data['ang_mom_err_avg'].append(ang_mom_err_avg)
        self.collected_data['lin_mom_err_avg'].append(lin_mom_err_avg)


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument(
        "-n",
        "--num_spheres",
        type=int,
        default=4,
        help="Number of spheres per box dimension for SRXPBD experiment",
    )
    parser.add_argument(
        "-k",
        "--plot_keys",
        nargs='+',
        type=str,
        # Default to plotting aggregate errors instead of COLLECTOR_TYPES since we've overriden collected_data keys
        default=['pos_l2_err_avg', 
                 'vel_err_avg', 
                 'rot_err_avg'],
    )
    parser.add_argument("--mu",
                        type=float,
                        default=0.0,
                        help="Mu between the object and ground (Coulomb friction). Default 0 = frictionless.")
    viewer, args = newton.examples.init(parser)
    viewer.show_particles = True
    example = PushedBoxStackedExample(viewer, args=args)
    newton.examples.run(example, args)
