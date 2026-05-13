# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Shapes
#
# Shows how to programmatically creates a variety of
# collision shapes using the newton.ModelBuilder() API.
#
# Command: python -m newton.examples basic_shapes
#
###########################################################################
import warp as wp
import newton
import numpy as np
from newton.examples.hiro.example_poc import Example


def collect_trajectories(parser, render=True):
    """Run each experiment and collect body_q for every frame.

    Args:
        experiments (list[str]): list of experiment names (e.g. ['gt','b1']).
        max_frames (int): number of frames to run each experiment.
        viewer: an optional viewer instance passed to Example.
        render (bool): whether to call example.render() each frame.

    Returns:
        dict: mapping experiment -> ndarray of shape (frames, q_dim)
    """
    trajectories = {}


    for exp in ["gt", "b1", "b2", "b3", "ours"]:
        viewer, args = newton.examples.init(parser)
        if exp not in args.experiments:
            continue
        
        print(f"Running experiment '{exp}' for {args.frames} frames...")
        example = Example(viewer, experiment=exp)
        frames = []
        for frame in range(args.frames):
            example.step()
            if render and viewer is not None:
                example.render()
            if exp == 'b2':
                q = np.asarray(example.state_0.particle_q)
                com = np.mean(q, axis=0)
                delta = np.array([0.0780692, 0.0, -0.0209191], dtype=float)
                com_corrected = com + delta
                frames.append(com_corrected.copy())
            else:
                q = np.asarray(example.state_0.body_q)[0][:3] # xyz 
                frames.append(q.copy())

        trajectories[exp] = np.stack(frames, axis=0)
        viewer.close()

    return trajectories


def compare_trajectories(trajectories, baseline='gt'):
    """Compare each trajectory to a baseline and print simple metrics.

    Currently computes mean L1 and L2 differences across all frames and degrees
    of freedom between each experiment and the baseline.
    """

    base = trajectories[baseline]
    print('\nTrajectory comparison report:')
    for exp, traj in trajectories.items():
        # align lengths if needed
        m = min(base.shape[0], traj.shape[0])
        diff = traj[:m] - base[:m]
        np.savetxt(f"newton/examples/hiro/traj_diff_{exp}_vs_{baseline}.txt", diff)
        l1 = np.mean(np.abs(diff))
        l2 = np.sqrt(np.mean(diff ** 2))
        print(f"  {exp:6s} vs {baseline}: mean L1={l1:.6f}, RMS L2={l2:.6f}")


def main():
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--experiments",
        help="One or more experiments (e.g. --experiments gt b3 b1) or 'all'",
        nargs='+',
        type=str,
        choices=["gt", "b1", "b2", "b3", "ours"],
        default=["gt"],
    )
    parser.add_argument(
        "--frames",
        help="Number of frames to run per experiment",
        type=int,
        default=300,
    )

    # Show particles in the viewer by default
    # viewer.show_particles = True

    trajectories = collect_trajectories(parser)

    compare_trajectories(trajectories, baseline='gt')


if __name__ == "__main__":
    main()
