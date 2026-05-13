import numpy as np
import warp as wp
import os
import newton
import newton.examples
from utils.plot import plot_all_data
from utils.data_collector import recompute_rot_error_unwrapped


class BaseExample:
    def __init__(self, viewer, args):
        raise NotImplementedError

    def collect_body_data(self, pos_analytic, vel_analytic, rot_analytic):
        raise NotImplementedError

    def collect_sphere_packed_data(self, pos_analytic, vel_analytic, rot_analytic):
        raise NotImplementedError

    def calculate_analytical_state(self, t):
        raise NotImplementedError

    def step(self):
        self.collect_data()
        self.simulate()
        self.sim_time += self.frame_dt
        self.plot_data(self.custom_rot_keys if hasattr(self, 'custom_rot_keys') else None)

    def collect_data(self):
        analytical_state = self.calculate_analytical_state(
            self.sim_time)
        if self.experiment in ["tetxpbd", "mujoco", "semieuler"]:
            self.collect_body_data(*analytical_state)
        elif self.experiment in ["srxpbd", "mrxpbd", "bxpbd"]:
            self.collect_sphere_packed_data(*analytical_state)

    def plot_data(self, custom_rot_keys:list=None):
        """
        Plots data and saves to disk if sim_time exceeds plot_sim_time

        Args:
            custom_rot_keys (Optional list) : Optional tuple of (rot_a_key, rot_s_key, rot_e_key) to specify custom rotation keys 
            for unwrapping and plotting for use in experiments with multiple objects. If None, defaults to standard keys.
        """
        if self.sim_time <= self.plot_sim_time:
            return

        exp_address = self.exp_address
        if self.args.exp_address is not None:
            exp_address = self.args.exp_address

        os.makedirs(os.path.dirname(exp_address), exist_ok=True)
        recompute_rot_error_unwrapped(self.collected_data)
        np.savez(exp_address, **self.collected_data)
        print("Saved data to:", exp_address)
        if not self.args.no_vis_plot:
            png_path = os.path.splitext(exp_address)[0] + ".png"
            plot_all_data(self.collected_data, keys=self.plot_keys, out_path=png_path)
            print("Saved plot to:", png_path)
        self.viewer.close()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def get_base_parser():
    parser = newton.examples.create_parser()
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        choices=["srxpbd", "tetxpbd", "mujoco",
                 "semieuler", "mrxpbd", "bxpbd"],
        default="tetxpbd",
    )
    parser.add_argument(
        "-i",
        "--num_iterations",
        type=int,
        default=10,
        help="Number of iterations for the solver",
    )
    parser.add_argument(
        "--cache_kernels",
        action="store_true",
        default=True,
        help="Cache compiled kernels",
    )
    parser.add_argument(
        "-m",
        "--morphit_json",
        type=str,
        help="Path to MorphIT JSON file for MRXPBD experiment",
    )
    parser.add_argument(
        "-t",
        "--plot_sim_time",
        type=float,
        default=10.0,
        help="Simulation time to plot data and end the simulation",
    )
    parser.add_argument(
        "--no-vis-plot",
        action="store_true",
    )
    parser.add_argument(
        "--exp-address",
        type=str,
        default=None,
        help="Path to override default experiment address for saving data and plots",
    )
    return parser
