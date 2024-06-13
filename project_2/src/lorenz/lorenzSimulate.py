import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_single_trajectory(z_sim, title: Optional[str] = "Discovered Dynamics", save_figure: bool = False, save_path: Optional[str] = None):
    fig = plt.figure(figsize=(5, 3))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.plot(z_sim[:, 0], z_sim[:, 1], z_sim[:, 2], linewidth=0.5)
    ax1.set_title(title)
    ax1.tick_params(axis='both', which='major')
    ax1.axis('off')
    
    if save_figure:
        save_path = save_path or "."
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_').lower()}.png"))

    plt.show()

def plot_xyz_trajectories(z_sim, time, title_x: Optional[str] = "X Trajectory", title_y: Optional[str] = "Y Trajectory", title_z: Optional[str] = "Z Trajectory", save_figure: bool = False, save_path: Optional[str] = None):
    fig, axs = plt.subplots(3, 1, figsize=(9, 12))
    
    axs[0].plot(time, z_sim[:, 0], 'r', linewidth=2)
    axs[0].set_title(title_x)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X')

    axs[1].plot(time, z_sim[:, 1], 'g', linewidth=2)
    axs[1].set_title(title_y)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Y')

    axs[2].plot(time, z_sim[:, 2], 'b', linewidth=2)
    axs[2].set_title(title_z)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Z')

    plt.tight_layout()
    
    if save_figure:
        save_path = save_path or "."
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"{title_x.replace(' ', '_').lower()}.png"))
        fig.savefig(os.path.join(save_path, f"{title_y.replace(' ', '_').lower()}.png"))
        fig.savefig(os.path.join(save_path, f"{title_z.replace(' ', '_').lower()}.png"))

    plt.show()

if __name__ == "__main__":
    from lorenzData import lorenz_coefficients
    from sindySimulate import sindy_simulate

    # LIBRARY KWARGS
    poly_order = 3
    include_sine = False
    include_constant = True

    lib_kwargs = {
        "poly_order": poly_order,
        "include_sine": include_sine,
        "include_constant": include_constant
    }

    # Get true Lorenz coefficients
    lorenz_coeff = lorenz_coefficients(sigma=10.0, beta=8 / 3, rho=28.0, **lib_kwargs)

    # Simulation parameters
    initial_condition = np.array([0.0, 0.0, 25.0])
    time = np.linspace(0, 10, 1000)

    z_sim = sindy_simulate(initial_condition, time, lorenz_coeff, **lib_kwargs)
    
    plot_single_trajectory(z_sim, title="3D Trajectory")
    plot_xyz_trajectories(z_sim, time, title_x="X Trajectory", title_y="Y Trajectory", title_z="Z Trajectory")
