import jax.numpy as jnp
from jax import random, jit

# Simplified small-scale test
input_dim = 16
latent_dim = 3

key = random.PRNGKey(0)
x = random.normal(key, (5, input_dim))
dx = random.normal(key, (5, input_dim))
ddx = random.normal(key, (5, input_dim))
mask = jnp.ones((latent_dim, latent_dim))

# Define simplified dummy functions for encoder and decoder
def encoder_dummy(params, x):
    return x[:, :latent_dim]

def decoder_dummy(params, z):
    return z

# Simplified SINDy library
def sindy_library_fn(z):
    return jnp.concatenate([z, z**2], axis=1)

# Example of simplified loss function components
def simplified_loss_fn(x, dx, ddx, mask):
    z = encoder_dummy(None, x)
    x_hat = decoder_dummy(None, z)
    
    dphi_dx = jnp.eye(latent_dim)  # Simplified jacobian
    dpsi_dz = jnp.eye(latent_dim)  # Simplified jacobian
    ddphi_dx2 = jnp.zeros((latent_dim, latent_dim, latent_dim))  # Simplified hessian
    dpsi_dz2 = jnp.zeros((latent_dim, latent_dim, latent_dim))  # Simplified hessian

    dx_in_z = dphi_dx @ dx.T
    theta = sindy_library_fn(z)
    xi = jnp.ones((theta.shape[1], latent_dim))  # Simplified coefficients

    ddz_reconstructed = theta @ (mask * xi)
    ddz_in_x = (dpsi_dz2 @ dx_in_z) @ dx_in_z + dpsi_dz @ ddz_reconstructed

    loss = jnp.mean((ddz_in_x.T - ddx) ** 2)
    return loss

# JIT compile the simplified loss function
jitted_simplified_loss_fn = jit(simplified_loss_fn)

# Run the simplified test
loss_val = jitted_simplified_loss_fn(x, dx, ddx, mask)
print("Simplified second-order loss:", loss_val)