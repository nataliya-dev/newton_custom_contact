"""Shared utilities for benchmark scripts."""
import os
import numpy as np
import newton

from examples.push.pushed_box import PushedBoxExample
from examples.push.pushed_bunny import PushedBunnyExample
from examples.slide_on_slope.box_on_slope import BoxOnSlopeExample
from utils.data_collector import COLLECTOR_TYPES, recompute_rot_error_unwrapped
from utils.mesh_helper import calculate_physics_properties_for_obj
from shapes.stanford_bunny import StanfordBunny

BOX_MASS = 4.0  # kg, matches PushedBoxExample.build_scene
GRAVITY = 9.81  # m/s^2

EXAMPLE_CLASSES = {
    'pushed_box': PushedBoxExample,
    'pushed_bunny': PushedBunnyExample,
}


class Args:
    """Lightweight namespace to build args objects for examples."""
    pass


def get_object_mass(example):
    """Return the object mass for the given example type."""
    if example == 'pushed_bunny':
        mass, _, _ = calculate_physics_properties_for_obj(StanfordBunny.obj_path)
        return mass
    elif example == 'pushed_box':
        return BOX_MASS
    else:
        raise ValueError(f"Unknown example '{example}'. "
                         f"Valid choices: {list(EXAMPLE_CLASSES.keys())}")


def static_friction_force(mu, mass=BOX_MASS, g=GRAVITY):
    """Minimum force magnitude to overcome static friction: F = mu * m * g."""
    return mu * mass * g


def run_single(example, solver, mu, num_frames, num_iterations, force_vec, seed):
    """Run a single simulation with given mu and force vector.

    Returns:
        pos_err_l2: final position L2 error
        rot_err: final rotation error
        collected_data: dict of time series arrays from the simulation
    """
    np.random.seed(seed)

    if example not in EXAMPLE_CLASSES:
        raise ValueError(
            f"Unknown example '{example}'. "
            f"Valid choices: {list(EXAMPLE_CLASSES.keys())}")
    ExampleClass = EXAMPLE_CLASSES[example]

    args = Args()
    args.mu = mu
    args.num_iterations = num_iterations
    args.plot_sim_time = 1e9  # disable auto plot/close
    args.plot_keys = []
    args.cache_kernels = True
    args.experiment = solver
    args.constant_force = force_vec
    args.num_spheres = 4
    args.sphere_radius = 0.005
    args.use_sphere_slope = False
    args.seed = seed

    viewer = newton.viewer.ViewerNull(num_frames=num_frames)
    ex = ExampleClass(viewer, args=args)

    while ex.sim_time < num_frames * ex.frame_dt:
        ex.step()

    recompute_rot_error_unwrapped(ex.collected_data)
    pos_err_l2 = ex.collected_data[COLLECTOR_TYPES.POS_L2_E][-1]
    rot_err = ex.collected_data[COLLECTOR_TYPES.ROT_E][-1][0]
    return pos_err_l2, rot_err, collected_data_to_arrays(ex.collected_data)


def collected_data_to_arrays(collected_data):
    """Convert collected_data dict-of-lists to dict-of-numpy-arrays."""
    arrays = {}
    for key, value in collected_data.items():
        if isinstance(value, list) and len(value) > 0:
            arrays[key] = np.array(value)
        else:
            arrays[key] = value
    return arrays


def save_results(out_path, results):
    """Save results dict to .npz, creating directories as needed."""
    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    np.savez(out_path, **results)
    if getattr(save_results, "_last_path", None) != out_path:
        save_results._last_path = out_path
        print(f"Saving results to {out_path}")
