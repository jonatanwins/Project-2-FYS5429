import sys
sys.path.append("../")
from trainer import SINDy_trainer
import jax.numpy as jnp
from lorenzData import generate_lorenz_train_data, generate_lorenz_test_data
import matplotlib.pyplot as plt
import numpy as np
from sindy_utils import sindy_simulate, create_sindy_library
from loss import loss_dynamics_x_factory, loss_dynamics_z_factory, recon_loss_factory
from jax import jit
import json

### MODEL YOU WANT TO LOAD
model = "kathleenReplica_1"
### MODEL YOU WANT TO LOAD

def get_checkpoint_path():
    try:
        # Check if running in a Jupyter notebook
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Running in a notebook
            return "checkpoints/version_0"
        else:
            # Running in some other shell
            return "src/lorenz/checkpoints/version_0"
    except NameError:
        # Running in some other shell
        return "src/lorenz/checkpoints/version_0"

checkpoint_path = get_checkpoint_path()

exmp_input = jnp.arange(128).reshape(1, 128)
trainer = SINDy_trainer.load_from_checkpoint(checkpoint_path, exmp_input)

#get hparams.json file from checkpoint_path
with open(checkpoint_path + "/hparams.json", "r") as f:
    hparams = json.load(f)

# Access the parameters from the loaded model state
sindy_coefficients = trainer.state.params["sindy_coefficients"]
mask = trainer.state.mask

t = np.arange(0, 10, 1)
z0 = np.array([[-8, 7, 27]])

# get lorenz data from non-random ics
test_data = generate_lorenz_test_data(
    z0,
    t,
    hparams["input_dim"],
    linear=False,
    normalization=np.array([1 / 40, 1 / 40, 1 / 40]),
)

# Simulate the system using the discovered dynamics
xi = sindy_coefficients * mask
z_sim = sindy_simulate(test_data["z"][0], t, xi, hparams["poly_order"], hparams["include_sine"])
lorenz_sim = sindy_simulate(test_data["z"][0], t, test_data["sindy_coefficients"], hparams["poly_order"], hparams["include_sine"])

def transform_and_simulate(xi, test_data):
    """
    Transform and simulate the system using the discovered dynamics.
    NOTE: ONLY WORKS FOR GOOD RESULTS?
    """
    a1 = 1
    a2 = test_data["sindy_coefficients"][2, 0] / xi[2, 0]
    a3 = np.sqrt(-xi[5, 2] / xi[6, 1] * a2**2)
    b3 = -xi[0, 2] / xi[3, 2]

    sindy_coefficients_transformed = np.zeros(xi.shape)
    sindy_coefficients_transformed[1, 0] = xi[1, 0]
    sindy_coefficients_transformed[2, 0] = xi[2, 0] * a2 / a1
    sindy_coefficients_transformed[1, 1] = xi[6, 1] * a1 / a2 * b3
    sindy_coefficients_transformed[2, 1] = xi[2, 1]
    sindy_coefficients_transformed[6, 1] = xi[6, 1] * a1 * a3 / a2
    sindy_coefficients_transformed[3, 2] = xi[3, 2]
    sindy_coefficients_transformed[5, 2] = xi[5, 2] * a1 * a2 / a3

    z0_transformed = np.array(
        [
            test_data["z"][0, 0] / a1,
            test_data["z"][0, 1] / a2,
            (test_data["z"][0, 2] - b3) / a3,
        ]
    )

    # Simulate transformed system
    z_sim_transformed = sindy_simulate(
        z0_transformed,
        t,
        sindy_coefficients_transformed,
        hparams["poly_order"],
        hparams["include_sine"],
    )
    return z_sim_transformed, sindy_coefficients_transformed

def plot_results(z_sim, z_sim_transformed, lorenz_sim, test_data, xi, sindy_coefficients_transformed):
    # Plot the simulated results
    fig1 = plt.figure(figsize=(3, 3))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot(z_sim[:, 0], z_sim[:, 1], z_sim[:, 2], linewidth=2)
    plt.axis("off")
    ax1.view_init(azim=120)

    fig2 = plt.figure(figsize=(3, 3))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot(
        z_sim_transformed[:, 0],
        z_sim_transformed[:, 1],
        z_sim_transformed[:, 2],
        linewidth=2,
    )
    plt.axis("off")
    ax2.view_init(azim=120)

    fig3 = plt.figure(figsize=(3, 3))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot(lorenz_sim[:, 0], lorenz_sim[:, 1], lorenz_sim[:, 2], linewidth=2)
    plt.xticks([])
    plt.axis("off")
    ax3.view_init(azim=120)

    # Plot time series comparison
    plt.figure(figsize=(3, 3))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, test_data["z"][:, i], color="#888888", linewidth=2)
        plt.plot(t, z_sim[:, i], "--", linewidth=2)
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")

    # Plot SINDy coefficients
    Xi_plot = xi
    Xi_plot[Xi_plot == 0] = np.inf
    plt.figure(figsize=(1, 2))
    plt.imshow(Xi_plot, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.clim([-10, 30])

    Xi_transformed_plot = np.copy(sindy_coefficients_transformed)
    Xi_transformed_plot[Xi_transformed_plot == 0] = np.inf
    plt.figure(figsize=(1, 2))
    plt.imshow(Xi_transformed_plot, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.clim([-10, 30])

    Xi_true_plot = np.copy(test_data["sindy_coefficients"])
    Xi_true_plot[Xi_true_plot == 0] = np.inf
    Xi_true_plot[6, 1] = -1.0
    Xi_true_plot[5, 2] = 1.0
    plt.figure(figsize=(1, 2))
    plt.imshow(Xi_true_plot, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.clim([-10, 30])

    plt.show()

# Function to calculate metrics

def calculate_metrics(trainer, test_data):
    recon_loss_fn = jit(recon_loss_factory())
    loss_dynamics_x_fn = jit(loss_dynamics_x_factory(trainer.autoencoder.decoder))
    loss_dynamics_z_fn = jit(loss_dynamics_z_factory(trainer.autoencoder.encoder))

    x_hat = trainer.model.apply({"params": trainer.state.params}, test_data["x"])

    decoder_x_error = recon_loss_fn(jnp.array(test_data["x"]), x_hat)

    decoder_dx_error = loss_dynamics_x_fn(
        trainer.state.params, 
        test_data["z"], 
        test_data["dx"], 
        create_sindy_library(hparams["poly_order"], hparams["include_sine"], n_states=hparams["latent_dim"])(test_data["z"]),
        trainer.state.params["sindy_coefficients"], 
        trainer.state.mask
    )
    sindy_dz_error = loss_dynamics_z_fn(
        trainer.state.params, 
        test_data["x"], 
        test_data["dx"], 
        create_sindy_library(hparams["poly_order"], hparams["include_sine"], n_states=hparams["latent_dim"])(test_data["z"]),
        trainer.state.params["sindy_coefficients"], 
        trainer.state.mask
    )

    print("Decoder relative error: %f" % decoder_x_error)
    print("Decoder relative SINDy error: %f" % decoder_dx_error)
    print("SINDy relative error, z: %f" % sindy_dz_error)

# Assuming some logic to check if the results are good
results_good = True  # Placeholder for actual condition

if results_good:
    z_sim_transformed, sindy_coefficients_transformed = transform_and_simulate(xi, test_data)
    plot_results(z_sim, z_sim_transformed, lorenz_sim, test_data, xi, sindy_coefficients_transformed)

calculate_metrics(trainer, test_data)
