import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from typing import Optional
from itertools import product

def plot_sindy_coefficients(xi, library_hparams, title: Optional[str] = "Discovered Coefficients", save_figure: bool = False, save_path: Optional[str] = None):
    """Plots and optionally saves the SINDy coefficients.
    
    Args:
        xi (ndarray): Array of coefficients to plot.
        library_hparams (Dict): Hyperparameters for the sindy library for the model
        title (Optional[str]): Title of the plot.
        save_figure (bool): If True, save the plot to the specified path or default path.
        save_path (Optional[str]): Path to save the figure.
    """
    Xi_plot = xi.copy()
    Xi_plot[Xi_plot == 0] = np.inf
    Xi_plot = np.abs(Xi_plot)

    max_val = round(max(jnp.concatenate(Xi_plot))+5, -1)

    n_states = library_hparams['n_states']
    poly_order = library_hparams['poly_order']
    include_constant = library_hparams['include_constant']
    include_sine = library_hparams['include_sine']

    terms = jnp.diag(jnp.full(n_states, 1))
    if include_constant:
        terms = jnp.concatenate([jnp.zeros((1,n_states)), terms], axis=0)

    if poly_order > 1:
        degrees = jnp.array(list(product(range(poly_order + 1), repeat=n_states)))
        sums = jnp.sum(degrees, axis=1)
        degrees = degrees[(sums <= poly_order) & (sums > 1)][::-1]
        degrees = jnp.array(sorted(degrees, key= lambda x: sum(x)))
        terms = jnp.concatenate([terms, degrees], axis=0)

    if include_sine:
        terms = jnp.concatenate([terms, jnp.diag(jnp.full(n_states, -1))], axis=0)

    row_labels = []
    for row in terms:
        if sum(row) == 0:
            row_labels.append(r'$1$')
        else:
            label = r"$"
            for i, deg in enumerate(row):
                if deg == 1:
                    label += f"z_{i}"
                if deg > 1:
                    label += f"z_{i}^{int(deg)}"
                if deg == -1:
                    label += f"sin(z_{i})"
            label += r"$"
            row_labels.append(label)

    plt.figure(figsize=(1, 2))
    plt.imshow(Xi_plot, interpolation='none', cmap='Reds')
    plt.title(title)
    plt.xticks([])
    plt.yticks([n for n in range(0,20)], labels=row_labels)
    plt.tight_layout()
    # plt.axis('off')
    plt.clim([0, max_val])
    plt.colorbar()
    
    if save_figure:
        save_path = save_path or "."
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_').lower()}.png"))
    
    plt.show()

def compare_sindy_coefficients(true_xi, discovered_xi, save_figure: bool = False, save_path: Optional[str] = None):
    """Plots and optionally saves the true and discovered SINDy coefficients side by side.
    
    Args:
        true_xi (ndarray): Array of true coefficients.
        discovered_xi (ndarray): Array of discovered coefficients.
        save_figure (bool): If True, save the plot to the specified path or default path.
        save_path (Optional[str]): Path to save the figure.
    """
    true_xi_plot = true_xi.copy()
    true_xi_plot[true_xi_plot == 0] = np.inf
    
    discovered_xi_plot = discovered_xi.copy()
    discovered_xi_plot[discovered_xi_plot == 0] = np.inf
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot true coefficients
    ax = axes[0]
    cax = ax.imshow(true_xi_plot, interpolation='none')
    ax.set_title("True Coefficients")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.colorbar(cax, ax=ax)
    
    # Plot discovered coefficients
    ax = axes[1]
    cax = ax.imshow(discovered_xi_plot, interpolation='none')
    ax.set_title("Discovered Coefficients")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.colorbar(cax, ax=ax)
    
    plt.tight_layout()
    
    if save_figure:
        save_path = save_path or "."
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "compare_sindy_coefficients.png"))
    
    plt.show()

if __name__ == "__main__":
    plt.style.use("plot_settings.mplstyle")
    
    # Example coefficients
    true_xi = np.zeros((10, 10))
    true_xi[0, 1] = 1.0
    true_xi[1, 3] = 2.0
    true_xi[2, 5] = 3.0
    
    discovered_xi = np.zeros((10, 10))
    discovered_xi[0, 1] = 1.0
    discovered_xi[1, 3] = 1.8
    discovered_xi[2, 5] = 2.9
    discovered_xi[3, 7] = 0.5  # False positive

    # Plot comparison without saving
    compare_sindy_coefficients(true_xi, discovered_xi)
