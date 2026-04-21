import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os

# Add parent directory to sys.path to allow importing modules from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import truss as t

x = np.linspace(0, 30, 20)
y = np.linspace(0, 25, 20)
X, Y = np.meshgrid(x, y)
combinations = np.vstack([X.ravel(), Y.ravel()]).T


def try_fab(do, di):
    try:
        do1, di1 = t.make_fabricable(do / 1000, di / 1000)
    except Exception:
        return np.nan, np.nan
    return 1000 * do1, 1000 * di1


Z = [try_fab(np.array([do]), np.array([di])) for do, di in combinations]


def plot_dual_heatmaps(x_vals, y_vals, z_vals):
    # Shape aligned to (rows, columns) -> (Y, X) based on meshgrid flattening
    shape = (len(y_vals), len(x_vals))

    z1 = np.array([np.squeeze(item[0]) for item in z_vals], dtype=float).reshape(shape)
    z2 = np.array([np.squeeze(item[1]) for item in z_vals], dtype=float).reshape(shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # X and Y labels appropriately mapped to the columns and rows
    sns.heatmap(
        z1,
        ax=axes[0],
        annot=True,
        fmt=".2g",
        xticklabels=np.round(x_vals, 2),
        yticklabels=np.round(y_vals, 2),
        cmap="viridis",
    )
    axes[0].invert_yaxis()  # Places the lowest Y value at the bottom
    axes[0].set_title("Function Result 1")
    axes[0].set_xlabel("X Axis")
    axes[0].set_ylabel("Y Axis")

    sns.heatmap(
        z2,
        ax=axes[1],
        annot=True,
        fmt=".2g",
        xticklabels=np.round(x_vals, 2),
        yticklabels=np.round(y_vals, 2),
        cmap="plasma",
    )
    axes[1].invert_yaxis()  # Places the lowest Y value at the bottom
    axes[1].set_title("Function Result 2")
    axes[1].set_xlabel("X Axis")
    axes[1].set_ylabel("Y Axis")

    plt.tight_layout()
    plt.show()


plot_dual_heatmaps(x, y, Z)
