import matplotlib.pyplot as plt
import numpy as np


def plot_data_and_score(raw_data, score):
    f, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(raw_data)
    ax[0].set_title("time series")
    ax[1].plot(score, "r")
    ax[1].set_title("change score")
    x_grid, y_grid = np.meshgrid(np.arange(len(score)), np.linspace(*ax[0].get_ylim()))
    if np.max(score):
        z_grid = (score/np.max(score))[x_grid]
    else:
        z_grid = score[x_grid]
    ax[0].contourf(x_grid, y_grid, z_grid, alpha=0.5, cmap="BuPu")
    ax[1].set_xlim(ax[0].get_xlim())
    return f
