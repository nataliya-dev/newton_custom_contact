import numpy as np
import matplotlib

matplotlib.use("Agg")  # PNG-only; examples always save to disk, never popup
import matplotlib.pyplot as plt

def plot_all_data(data_dict, keys, out_path=None):
    # Filter out empty entries
    non_empty_items = {
        k: v for k, v in data_dict.items()
        if v is not None and k != 'frame_dt' and k in keys and len(v) > 0
    }

    if not non_empty_items:
        print("No non-empty data to plot.")
        return

    # Infer time length from the first non-empty key
    first_key = next(iter(non_empty_items))
    N = len(non_empty_items[first_key])
    time = np.arange(N) * data_dict['frame_dt']

    num_plots = len(non_empty_items)
    plt.figure(figsize=(8, 3 * num_plots))

    for i, (key, values) in enumerate(non_empty_items.items(), start=1):
        arr = np.array(values)
        plt.subplot(num_plots, 1, i)

        # Scalar vs vector handling
        if arr.ndim == 1:
            plt.plot(time, arr)
            plt.legend([key])
        else:
            plt.plot(time, arr)
            labels = [f"{key}[{j}]" for j in range(arr.shape[1])]
            plt.legend(labels)

        plt.title(f"{key} over Time")
        plt.xlabel("Time (s)")
        plt.grid()

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
