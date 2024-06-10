import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_sindy_coefficients(xi, title: Optional[str] = "Discovered Coefficients", save_figure: bool = False, save_path: Optional[str] = None):
    """Plots and optionally saves the SINDy coefficients.
    
    Args:
        xi (ndarray): Array of coefficients to plot.
        title (Optional[str]): Title of the plot.
        save_figure (bool): If True, save the plot to the specified path or default path.
        save_path (Optional[str]): Path to save the figure.
    """
    Xi_plot = xi.copy()
    Xi_plot[Xi_plot == 0] = np.inf
    
    plt.figure(figsize=(1, 2))
    plt.imshow(Xi_plot, interpolation='none')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.clim([-10, 30])
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
