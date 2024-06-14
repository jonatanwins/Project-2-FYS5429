import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from typing import Optional, Dict
from sindyLibrary import get_row_context


def plot_sindy_coefficients(
    xi,
    library_hparams,
    title: Optional[str] = "Discovered Coefficients",
    second_order: bool = False,
    save_figure: bool = False,
    folder_path: Optional[str] = None,
    file_name: Optional[str] = None,
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

    row_labels = [
        f"${x}$" for x in get_row_context(library_hparams, second_order=second_order)
    ]
    
    row_sums = [sum(x) for x in Xi_plot]

    yticks = [0]
    yticks_labels = [r"$1$"]
    for i in range(1, len(row_labels)):
        if row_sums[i] != 0:
            yticks.append(i)
            yticks_labels.append(row_labels[i])

    """
    # Maybe remove?
    n_labels = len(row_labels)
    middle_index = n_labels // 2

    # Defining the y-ticks and labels to include ellipsis
    yticks = [0, 1, 2, n_labels - 2, n_labels - 1]
    yticks_labels = [
        row_labels[0],
        row_labels[1],
        row_labels[2],
        row_labels[-2],
        row_labels[-1],
    ]

    if n_labels > 5:
        yticks = [0, 1, 2, middle_index, n_labels - 2, n_labels - 1]
        yticks_labels = [
            row_labels[0],
            row_labels[1],
            row_labels[2],
            rf"$\vdots$",
            row_labels[-2],
            row_labels[-1],
        ]
    """

    # plt.figure(figsize=(1, 2))
    plt.imshow(Xi_plot, interpolation="none", cmap="Reds")
    plt.title(title)
    plt.xticks([])
    plt.yticks(yticks, labels=yticks_labels, fontsize=12)
    plt.tick_params(axis="y", which="both", length=0)  # Disable ticks on the y-axis
    plt.tight_layout()
    plt.clim([0, max_val])
    plt.colorbar()

    if save_figure:
        if folder_path and file_name:
            os.makedirs(folder_path, exist_ok=True)
            save_path = os.path.join(folder_path, file_name)
        else:
            raise ValueError(
                "Both folder_path and file_name must be provided if save_figure is True."
            )

        plt.savefig(save_path)

    plt.show()


def compare_sindy_coefficients(
    true_xi,
    discovered_xi,
    library_hparams,
    second_order: bool = False,
    save_figure: bool = False,
    file_name: Optional[str] = None,
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
    discovered_xi_plot = discovered_xi.copy()

    row_labels = [
        f"${x}$" for x in get_row_context(library_hparams, second_order=second_order)
    ]

    true_row_sums = [sum(x) for x in true_xi_plot]
    discovered_row_sums = [sum(x) for x in discovered_xi_plot]

    yticks = [0]
    yticks_labels = [r"$1$"]
    for i in range(1, len(row_labels)):
        if True: #true_row_sums[i] != 0 or discovered_row_sums[i] != 0:
            yticks.append(i)
            yticks_labels.append(row_labels[i])

    true_xi_plot = np.abs(true_xi_plot)
    #true_xi_plot[true_xi_plot==0] = -100

    discovered_xi_plot = np.abs(discovered_xi_plot)
    #discovered_xi_plot[discovered_xi_plot==0] = -100

    max_val = max(round(max(jnp.concatenate(true_xi_plot)), 0), round(max(jnp.concatenate(discovered_xi_plot)), 0))
    

    """
    # Maybe remove?
    n_labels = len(row_labels)
    middle_index = n_labels // 2

    # Defining the y-ticks and labels to include ellipsis
    yticks = [0, 1, 2, n_labels - 2, n_labels - 1]
    yticks_labels = [
        row_labels[0],
        row_labels[1],
        row_labels[2],
        row_labels[-2],
        row_labels[-1],
    ]

    if n_labels > 5:
        yticks = [0, 1, 2, middle_index, n_labels - 2, n_labels - 1]
        yticks_labels = [
            row_labels[0],
            row_labels[1],
            row_labels[2],
            rf"$\vdots$",
            row_labels[-2],
            row_labels[-1],
        ]
    """

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 4.5))

    # Plot true coefficients
    ax = axes[0]
    cax = ax.imshow(true_xi_plot, interpolation="none", cmap="YlOrBr", norm=LogNorm())
    cax.set_clim(0.1, max_val)
    ax.set_title("True Coefficients")
    ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels, fontsize=10)
    ax.tick_params(axis="y", which="both", length=0)  # Disable ticks on the y-axis
    #fig.colorbar(cax, ax=ax)

    # Plot discovered coefficients
    ax = axes[1]
    cax = ax.imshow(discovered_xi_plot, interpolation="none", cmap="YlOrBr", norm=LogNorm())
    cax.set_clim(0.1, max_val)
    ax.set_title("Discovered Coefficients")
    ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels, fontsize=10)

    cbar = fig.colorbar(cax, ax=ax, ticks=[0.1, np.floor(max_val//8), np.floor(max_val//4), np.floor(max_val//2), np.floor(max_val)])
    cbar.ax.set_yticklabels(['0', f'{np.floor(max_val//8)}', f'{np.floor(max_val//4)}', f'{np.floor(max_val//2)}', f'{np.floor(max_val)}'])

    plt.tick_params(axis="y", which="both", length=0)  # Disable ticks on the y-axis
    plt.tight_layout()

    if save_figure:
        save_path = save_path or "."
        os.makedirs(save_path, exist_ok=True)
        if file_name:
            plt.savefig(os.path.join(save_path, file_name))
        else:
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
    compare_sindy_coefficients(
        true_xi, discovered_xi, library_hparams=lib_kwargs, second_order=False
    )
    plot_sindy_coefficients(true_xi, library_hparams=lib_kwargs, second_order=True)
