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


if __name__ == "__main__":
    plt.style.use("plot_settings.mplstyle")
    # Example coefficients
    xi = np.random.randn(10, 10)

    # Plot without saving
    plot_sindy_coefficients(xi)

    # Plot with custom title and save the plot
    plot_sindy_coefficients(xi, title="Custom Title", save_figure=True, save_path="./figures")