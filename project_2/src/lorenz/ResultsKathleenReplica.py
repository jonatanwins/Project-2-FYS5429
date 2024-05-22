# %%
import sys

sys.path.append("../")
from trainer import Trainer
import jax.numpy as jnp
from autoencoder import Autoencoder
from lorenzUtils import generate_lorenz_data
import matplotlib.pyplot as plt
import numpy as np

# Assuming `exmp_input` is already defined or you know how to reconstruct it
exmp_input = jnp.arange(128).reshape(1, 128)


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
trainer = Trainer.load_from_checkpoint(checkpoint_path, exmp_input)

# PATH IS DIFFRENT FROM TERMINAL AND IDE /src/lorenz/checkpoints/version_0 vs checkpoints/version_0
# trainer = Trainer.load_from_checkpoint("checkpoints/version_0", exmp_input)

# Access the parameters from the loaded model state
sindy_coefficients = trainer.state.params["sindy_coefficients"]

mask = trainer.state.mask

t = np.arange(0, 10, 1)
z0 = np.array([[-8, 7, 27]])
params = {"input_dim": 128, "latent_dim": 3, "poly_order": 3, "include_sine": False}

# get lorentz data from non random ics
test_data = generate_lorenz_data(
    z0,
    t,
    params["input_dim"],
    linear=False,
    normalization=np.array([1 / 40, 1 / 40, 1 / 40]),
)


# %%
from sindy_utils import sindy_simulate

# Assuming trainer.state.params['sindy_coefficients'] contains the coefficients
xi = sindy_coefficients * mask

# Simulate the system using the discovered dynamics
z_sim = sindy_simulate(
    test_data["z"][0], t, xi, params["poly_order"], params["include_sine"]
)
lorenz_sim = sindy_simulate(
    test_data["z"][0],
    t,
    test_data["sindy_coefficients"],
    params["poly_order"],
    params["include_sine"],
)


# %%
# Transform the SINDy coefficients
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
    params["poly_order"],
    params["include_sine"],
)


# %%

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


# %%
# Assuming test_data contains the necessary fields and trainer has the required outputs
decoder_x_error = np.mean(
    (test_data["x"] - trainer.state.params["x_decoded"]) ** 2
) / np.mean(test_data["x"] ** 2)
decoder_dx_error = np.mean(
    (test_data["dx"] - trainer.state.params["dx_decoded"]) ** 2
) / np.mean(test_data["dx"] ** 2)
sindy_dz_error = np.mean(
    (trainer.state.params["dz"] - trainer.state.params["dz_predict"]) ** 2
) / np.mean(trainer.state.params["dz"] ** 2)

print("Decoder relative error: %f" % decoder_x_error)
print("Decoder relative SINDy error: %f" % decoder_dx_error)
print("SINDy relative error, z: %f" % sindy_dz_error)
