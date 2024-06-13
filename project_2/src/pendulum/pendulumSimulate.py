import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_z_zdot(z_sim_list, title: Optional[str] = "Discovered Dynamics", save_figure: bool = False, save_path: Optional[str] = None):
    plt.figure(figsize=(4,3))
    for sim in z_sim_list:
        plt.plot(sim[:,0].T, sim[:,1].T, linewidth=1, color='blue')
    plt.axis('equal')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.xlabel(r"$z$")
    plt.ylabel(r"$\dot{z}$", rotation=0)
    plt.tick_params(axis='both', which='major')
    
    if save_figure:
        save_path = save_path or "."
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{title.replace(' ', '_').lower()}.png"))

    plt.show()