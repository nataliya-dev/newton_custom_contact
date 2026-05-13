import argparse

import matplotlib

matplotlib.use("Agg")  # PNG-only; save to disk, never popup
import matplotlib.pyplot as plt
import numpy as np


def compare_experiments(experiments, tag, keys):
    # Load data for each experiment into a mutable dict so we can normalize shapes.
    raw_npz = {exp: np.load(f"outputs/{exp}.npz") for exp in experiments}
    all_data = {}
    for exp, npz in raw_npz.items():
        all_data[exp] = {}
        # copy requested keys and normalize 1D arrays to (N,1)
        for key in keys:
            if key in npz:
                data = npz[key]
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                all_data[exp][key] = data

        # copy frame_dt (scalar) for time calculation
        if 'frame_dt' in npz:
            all_data[exp]['frame_dt'] = float(npz['frame_dt'].tolist() if hasattr(
                npz['frame_dt'], 'tolist') else npz['frame_dt'])
        else:
            # default to 1.0 if not present
            all_data[exp]['frame_dt'] = 1.0
    rows = len(keys)

    # Determine how many axes each key has (max across experiments).
    key_axes = {}
    for key in keys:
        max_axes = 0
        for exp in experiments:
            data = all_data[exp][key]
            axes = data.shape[1]
            if axes > max_axes:
                max_axes = axes
        key_axes[key] = max_axes

    # Determine number of columns per key (based on available axes). No L2 columns are added.
    cols_per_key = key_axes.copy()
    cols = max(cols_per_key.values()) if cols_per_key else 0
    if cols == 0:
        raise ValueError("No axis data found for provided keys.")
    fig = plt.figure(figsize=(4 * cols, 2.5 * rows))
    legend_handles = {}

    for k, key in enumerate(keys):
        axes_for_key = cols_per_key[key]
        for axis in range(cols):
            ax_idx = k * cols + axis + 1
            ax = plt.subplot(rows, cols, ax_idx)

            plotted_any = False
            for exp in experiments:
                data = all_data[exp][key]
                # data is guaranteed to be 2D now
                N = data.shape[0]
                time = np.arange(N) * all_data[exp]["frame_dt"]
                exp_name = exp.split('/')[1]

                # Only plot real axes that exist in this dataset
                if axis < data.shape[1]:
                    y = data[:, axis]
                else:
                    # nothing to plot for this experiment on this axis
                    continue

                line,  = ax.plot(time, y, label=exp_name, linewidth=1)
                if exp_name not in legend_handles:
                    legend_handles[exp_name] = line
                plotted_any = True

            # If no experiment had data for this subplot, hide it
            if not plotted_any:
                ax.set_visible(False)
                continue

            # Titles for the first row
            if k == 0:
                if axis == 0:
                    ax.set_title("Axis X")
                elif axis == 1:
                    ax.set_title("Axis Y")
                elif axis == 2:
                    ax.set_title("Axis Z")

            if axis == 0:
                ax.set_ylabel(key)
            if k == rows - 1:
                ax.set_xlabel("Time (s)")

            ax.grid()
    plt.tight_layout()

    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc='upper right',
            fontsize='small',
        )

    out_path = f"outputs/{tag}_experiment_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved comparison plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        nargs='+',
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default=""
    )

    args = parser.parse_args()
    keys = ['L_solver', 'L_analytic', 'w_solver', 'w_analytic',
            'w_err', 'tau_array', 'rot_err', 'pos_err', 'pos_l2_err']
    compare_experiments(args.experiment, args.tag, keys)
