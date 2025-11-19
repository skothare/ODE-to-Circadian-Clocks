import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(t, y, labels=None, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(y.shape[0]):
        label = labels[i] if labels else f"y{i}"
        ax.plot(t, y[i], label=label)
    ax.legend()
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time (h)")
    return ax


def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight")
