import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sindyLibrary import sindy_simulate, create_sindy_library
from lorenzData import generate_lorenz_data
from dev.loss_old import loss_dynamics_x_factory, loss_dynamics_z_factory, recon_loss_factory
from jax import jit
import jax.numpy as jnp

def coordinate_transformation(xi, test_data):
    a1 = 1
    a2 = test_data['sindy_coefficients'][2, 0] / xi[2, 0]
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

    z0_transformed = np.array([test_data['z'][0, 0] / a1, test_data['z'][0, 1] / a2, (test_data['z'][0, 2] - b3) / a3])

    return z0_transformed, sindy_coefficients_transformed

def simulate_ode(test_data, t, xi, xi_transformed=None, params=None):
    z_sim = sindy_simulate(test_data['z'][0], t, xi, params['poly_order'], params['loss_params']['include_sine'])

    if xi_transformed is not None:
        z0_transformed, sindy_coefficients_transformed = coordinate_transformation(xi_transformed, test_data)
        z_sim_transformed = sindy_simulate(z0_transformed, t, sindy_coefficients_transformed, params['poly_order'], params['loss_params']['include_sine'])
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

def plot_single_trajectory(t, z_sim, z_sim_transformed=None, test_data=None):
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
    
    if test_data is not None:
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot(test_data['z'][:, 0], test_data['z'][:, 1], test_data['z'][:, 2], linewidth=2)
        ax3.set_title('True Dynamics')
        plt.axis('off')
    
    plt.show()

def create_out_of_distribution_initial_conditions(n_ics, inDist_ic_widths, outDist_extra_width, t, params, noise_strength=1e-6):
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
    
    test_data = generate_lorenz_data(ics, t, params['input_dim'], linear=False, normalization=np.array([1/40, 1/40, 1/40]))
    test_data['x'] = test_data['x'].reshape((-1, params['input_dim']))
    test_data['x'] += noise_strength * np.random.normal(size=test_data['x'].shape)
    test_data['dx'] = test_data['dx'].reshape((-1, params['input_dim']))
    test_data['dx'] += noise_strength * np.random.normal(size=test_data['dx'].shape)
    
    return test_data

def calculate_losses(trainer, test_data, params):
    recon_loss_fn = jit(recon_loss_factory())
    loss_dynamics_x_fn = jit(loss_dynamics_x_factory(trainer.autoencoder.decoder))
    loss_dynamics_z_fn = jit(loss_dynamics_z_factory(trainer.autoencoder.encoder))

    x_hat = trainer.model.apply({"params": trainer.state.params}, test_data["x"])

    decoder_x_error = recon_loss_fn(jnp.array(test_data["x"]), x_hat)

    decoder_dx_error = loss_dynamics_x_fn(
        trainer.state.params, 
        test_data["z"], 
        test_data["dx"], 
        create_sindy_library(params["poly_order"], params["loss_params"]["include_sine"], n_states=params["latent_dim"])(test_data["z"]),
        trainer.state.params["sindy_coefficients"], 
        trainer.state.mask
    )
    sindy_dz_error = loss_dynamics_z_fn(
        trainer.state.params, 
        test_data["x"], 
        test_data["dx"], 
        create_sindy_library(params["poly_order"], params["loss_params"]["include_sine"], n_states=params["latent_dim"])(test_data["z"]),
        trainer.state.params["sindy_coefficients"], 
        trainer.state.mask
    )

    print("Decoder relative error: %f" % decoder_x_error)
    print("Decoder relative SINDy error: %f" % decoder_dx_error)
    print("SINDy relative error, z: %f" % sindy_dz_error)

    return decoder_x_error, decoder_dx_error, sindy_dz_error
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

    # Get sindy coefficients and mask from the trained model
    sindy_coefficients = trainer.state.params["sindy_coefficients"]
    mask = trainer.state.mask

    # Generate minimal test data
    t = np.arange(0, 1, 0.01)  # minimal time span
    z0 = np.array([[-8, 7, 27]])  # single initial condition
    test_data = generate_lorenz_data(z0, t, hparams["input_dim"], linear=False, normalization=np.array([1/40, 1/40, 1/40]))
    test_data['x'] = test_data['x'].reshape((-1, hparams['input_dim']))
    test_data['dx'] = test_data['dx'].reshape((-1, hparams['input_dim']))

    # Simulate ODE with the discovered coefficients
    z_sim = simulate_ode(test_data, t, sindy_coefficients, params=hparams)

    # Plot sindy coefficients
    plot_sindy_coefficients(sindy_coefficients)

    # Plot the single trajectory
    plot_single_trajectory(t, z_sim, test_data=test_data)

    # Calculate losses
    calculate_losses(trainer, test_data, hparams)
