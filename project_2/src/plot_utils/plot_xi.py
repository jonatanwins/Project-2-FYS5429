import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from typing import Optional, Dict
from sindyLibrary import get_row_context

def plot_sindy_coefficients(
    xi,
    library_hparams,
    title: Optional[str] = "Discovered Coefficients",
    second_order: bool = False,
    save_figure: bool = False,
    save_path: Optional[str] = None,
):
    """Plots and optionally saves the SINDy coefficients.

    Args:
        xi (ndarray): Array of coefficients to plot.
        library_hparams (Dict): Hyperparameters for the sindy library for the model
        title (Optional[str]): Title of the plot.
        save_figure (bool): If True, save the plot to the specified path or default path.
        save_path (Optional[str]): Path to save the figure.
    """
    Xi_plot = xi.copy()
    Xi_plot = np.abs(Xi_plot)

    max_val = round(max(jnp.concatenate(Xi_plot)), 0)

    row_labels = [f"${x}$" for x in get_row_context(library_hparams, second_order=second_order)]
    n_labels = len(row_labels)
    middle_index = n_labels // 2

    # Defining the y-ticks and labels to include ellipsis
    yticks = [0, 1, 2, n_labels - 2, n_labels - 1]
    yticks_labels = [row_labels[0], row_labels[1], row_labels[2], row_labels[-2], row_labels[-1]]

    if n_labels > 5:
        yticks = [0, 1, 2, middle_index, n_labels - 2, n_labels - 1]
        yticks_labels = [row_labels[0], row_labels[1], row_labels[2], fr"$\vdots$", row_labels[-2], row_labels[-1]]

    plt.figure(figsize=(1, 2))
    plt.imshow(Xi_plot, interpolation="none", cmap="Reds")
    plt.title(title)
    plt.xticks([])
    plt.yticks(yticks, labels=yticks_labels, fontsize=7)
    plt.tick_params(axis='y', which='both', length=0)  # Disable ticks on the y-axis
    plt.tight_layout()
    plt.clim([0, max_val])
    plt.colorbar()

    if save_figure:
        save_path = save_path or "."
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_').lower()}.png"))

    plt.show()


def compare_sindy_coefficients(
    true_xi,
    discovered_xi,
    library_hparams,
    second_order: bool = False,
    save_figure: bool = False,
    save_path: Optional[str] = None,
):
    """Plots and optionally saves the true and discovered SINDy coefficients side by side.

    Args:
        true_xi (ndarray): Array of true coefficients.
        discovered_xi (ndarray): Array of discovered coefficients.
        save_figure (bool): If True, save the plot to the specified path or default path.
        save_path (Optional[str]): Path to save the figure.
    """
    true_xi_plot = true_xi.copy()
    true_xi_plot = np.abs(true_xi_plot)

    discovered_xi_plot = discovered_xi.copy()
    discovered_xi_plot = np.abs(discovered_xi_plot)

    row_labels = [f"${x}$" for x in get_row_context(library_hparams, second_order=second_order)]
    n_labels = len(row_labels)
    middle_index = n_labels // 2

    # Defining the y-ticks and labels to include ellipsis
    yticks = [0, 1, 2, n_labels - 2, n_labels - 1]
    yticks_labels = [row_labels[0], row_labels[1], row_labels[2], row_labels[-2], row_labels[-1]]

    if n_labels > 4:
        yticks = [0, 1, 2, middle_index, n_labels - 2, n_labels - 1]
        yticks_labels = [row_labels[0], row_labels[1], row_labels[2], fr"$\vdots$", row_labels[-2], row_labels[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot true coefficients
    ax = axes[0]
    cax = ax.imshow(true_xi_plot, interpolation="none", cmap="Reds")
    ax.set_title("True Coefficients")
    ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels, fontsize=12)
    ax.tick_params(axis='y', which='both', length=0)  # Disable ticks on the y-axis
    fig.colorbar(cax, ax=ax)

    # Plot discovered coefficients
    ax = axes[1]
    cax = ax.imshow(discovered_xi_plot, interpolation="none", cmap="Reds")
    ax.set_title("Discovered Coefficients")
    ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels, fontsize=12)
    fig.colorbar(cax, ax=ax)

    plt.tick_params(axis='y', which='both', length=0)  # Disable ticks on the y-axis
    plt.tight_layout()

    if save_figure:
        save_path = save_path or "."
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "compare_sindy_coefficients.png"))

    plt.show()


if __name__ == "__main__":
    plt.style.use("plot_settings.mplstyle")

    # Example coefficients
    true_xi = np.zeros((10, 3))
    true_xi[0, 1] = 1.0
    true_xi[3, 1] = 1.8
    true_xi[5, 2] = 2.9


    discovered_xi = np.zeros((10, 3))
    discovered_xi[1, 0] = 0.9
    discovered_xi[3, 1] = 1.7
    discovered_xi[5, 2] = 2.8

    n_states = 3
    poly_order = 2
    include_sine = False
    include_constant = True
    lib_kwargs = {
        "n_states": n_states,
        "poly_order": poly_order,
        "include_sine": include_sine,
        "include_constant": include_constant,
    }
    # Plot comparison without saving
    compare_sindy_coefficients(true_xi, discovered_xi, library_hparams=lib_kwargs, second_order=False)
    plot_sindy_coefficients(true_xi, library_hparams=lib_kwargs, second_order=True)
