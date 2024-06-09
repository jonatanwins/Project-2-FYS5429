import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from sindySimulate import sindy_simulate
from jax import jit
import jax.numpy as jnp

def generate_initial_conditions(n_ics, ic_means=np.array([0, 0, 25]), ic_widths=2 * np.array([36, 48, 41])):
    """
    Generate initial conditions for training sample.
    """
    return ic_widths * (np.random.rand(n_ics, 3) - 0.5) + ic_means

def coordinate_transformation(xi, test_ic, sindy_coefficients):
    a1 = 1
    a2 = sindy_coefficients[2, 0] / xi[2, 0]
    a3 = np.sqrt(-xi[5, 2] / xi[6, 1] * a2 ** 2)
    b3 = -xi[0, 2] / xi[3, 2]

    sindy_coefficients_transformed = np.zeros(xi.shape)
    sindy_coefficients_transformed[1, 0] = xi[1, 0]
    sindy_coefficients_transformed[2, 0] = xi[2, 0] * a2 / a1
    sindy_coefficients_transformed[1, 1] = xi[6, 1] * a1 / a2 * b3
    sindy_coefficients_transformed[2, 1] = xi[2, 1]
    sindy_coefficients_transformed[6, 1] = xi[6, 1] * a1 * a3 / a2
    sindy_coefficients_transformed[3, 2] = xi[3, 2]
    sindy_coefficients_transformed[5, 2] = xi[5, 2] * a1 * a2 / a3

    z0_transformed = np.array([test_ic[0, 0] / a1, test_ic[0, 1] / a2, (test_ic[0, 2] - b3) / a3])

    return z0_transformed, sindy_coefficients_transformed

def simulate_ode(test_ic, t, xi, xi_transformed=None, params=None):
    lib_kwargs = {"poly_order": params['poly_order'], "include_sine": params['include_sine'], "n_states": params['input_dim']}
    z_sim = sindy_simulate(test_ic, t, xi, **lib_kwargs)

    if xi_transformed is not None:
        z0_transformed, sindy_coefficients_transformed = coordinate_transformation(xi_transformed, test_ic, xi)
        z_sim_transformed = sindy_simulate(z0_transformed, t, sindy_coefficients_transformed, **lib_kwargs)
        return z_sim, z_sim_transformed

    return z_sim

def plot_sindy_coefficients(xi, xi_transformed=None, xi_true=None):
    def plot_coefficients(coefficients, title):
        Xi_plot = coefficients.copy()
        Xi_plot[Xi_plot == 0] = np.inf
        plt.figure(figsize=(1, 2))
        plt.imshow(Xi_plot, interpolation='none')
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.clim([-10, 30])
        plt.colorbar()

    plot_coefficients(xi, "Discovered Coefficients")
    
    if xi_transformed is not None:
        plot_coefficients(xi_transformed, "Transformed Coefficients")
    
    if xi_true is not None:
        plot_coefficients(xi_true, "True Coefficients")

    plt.show()

def plot_single_trajectory(t, z_sim, z_sim_transformed=None, true_z=None):
    fig = plt.figure(figsize=(9, 3) if z_sim_transformed is not None else (3, 3))
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(z_sim[:, 0], z_sim[:, 1], z_sim[:, 2], linewidth=2)
    ax1.set_title('Discovered Dynamics')
    plt.axis('off')
    
    if z_sim_transformed is not None:
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot(z_sim_transformed[:, 0], z_sim_transformed[:, 1], z_sim_transformed[:, 2], linewidth=2)
        ax2.set_title('Transformed Dynamics')
        plt.axis('off')
    
    if true_z is not None:
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot(true_z[:, 0], true_z[:, 1], true_z[:, 2], linewidth=2)
        ax3.set_title('True Dynamics')
        plt.axis('off')
    
    plt.show()

def create_out_of_distribution_initial_conditions(n_ics, inDist_ic_widths, outDist_extra_width, noise_strength=1e-6):
    full_width = inDist_ic_widths + outDist_extra_width
    ics = np.zeros((n_ics, 3))
    i = 0
    
    while i < n_ics:
        ic = np.array([np.random.uniform(-full_width[0], full_width[0]),
                       np.random.uniform(-full_width[1], full_width[1]),
                       np.random.uniform(-full_width[2], full_width[2]) + 25])
        
        if ((ic[0] > -inDist_ic_widths[0]) and (ic[0] < inDist_ic_widths[0]) and
            (ic[1] > -inDist_ic_widths[1]) and (ic[1] < inDist_ic_widths[1]) and
            (ic[2] > 25 - inDist_ic_widths[2]) and (ic[2] < 25 + inDist_ic_widths[2])):
            continue
        else:
            ics[i] = ic
            i += 1
    
    # Add noise to the initial conditions
    noisy_ics = ics + noise_strength * np.random.normal(size=ics.shape)
    
    return noisy_ics


if __name__ == "__main__":
    from trainer import SINDy_trainer
    import json

    # Load the trained model
    checkpoint_path = "checkpoints/kathleenReplica_1"
    exmp_input = jnp.arange(128).reshape(1, 128)
    trainer = SINDy_trainer.load_from_checkpoint(checkpoint_path, exmp_input)

    # Load hyperparameters
    with open(f"{checkpoint_path}/hparams.json", "r") as f:
        hparams = json.load(f)

    # Get SINDy coefficients and mask from the trained model
    sindy_coefficients = trainer.state.params["sindy_coefficients"]
    mask = trainer.state.mask

    # Generate minimal test data
    t = np.arange(0, 1, 0.1)  # minimal time span
    z0 = np.array([[-8, 7, 27]])  # single initial condition

    # Simulate ODE with the discovered coefficients
    z_sim = simulate_ode(z0, t, sindy_coefficients, params=hparams)

    # Plot SINDy coefficients
    plot_sindy_coefficients(sindy_coefficients)

    # Plot the single trajectory
    plot_single_trajectory(t, z_sim)
