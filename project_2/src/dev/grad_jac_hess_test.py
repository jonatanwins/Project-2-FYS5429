#test script for gradient, jacobian and hessian jax

import jax.numpy as jnp
from jax import grad, jacobian, hessian

# Define dummy encoder and decoder functions
def dummy_encoder(params, x):
    """
    Encoder-like function: reduces dimensionality
    Args:
        params: Parameters of the function (not used in this simple example)
        x: Input vector (larger dimension)
    Returns:
        Reduced dimension vector
    """
    return jnp.tanh(x[:2])  # Reduces dimensionality to 2

def dummy_decoder(params, z):
    """
    Decoder-like function: increases dimensionality
    Args:
        params: Parameters of the function (not used in this simple example)
        z: Input vector (smaller dimension)
    Returns:
        Increased dimension vector
    """
    return jnp.tanh(jnp.concatenate([z, z]))  # Increases dimensionality to 4

# Create dummy input vectors
x = jnp.array([1.0, 2.0, 3.0, 4.0])
z = jnp.array([0.5, -0.5])

# Dummy parameters (not used in these simple examples)
params = None

# Calculate first-order gradient using grad
grad_encoder = grad(lambda x: jnp.sum(dummy_encoder(params, x)))
grad_decoder = grad(lambda z: jnp.sum(dummy_decoder(params, z)))

# Calculate Jacobian using jacobian
jacobian_encoder = jacobian(dummy_encoder, argnums=1)
jacobian_decoder = jacobian(dummy_decoder, argnums=1)

# Calculate Hessian using hessian
hessian_encoder = hessian(lambda x: jnp.sum(dummy_encoder(params, x)))
hessian_decoder = hessian(lambda z: jnp.sum(dummy_decoder(params, z)))

# Print results
print("Input x:", x)
print("Encoder output:", dummy_encoder(params, x))
print("Encoder gradient:", grad_encoder(x))
print("Encoder Jacobian:\n", jacobian_encoder(params, x))
print("Encoder Hessian:\n", hessian_encoder(x))

print("\nInput z:", z)
print("Decoder output:", dummy_decoder(params, z))
print("Decoder gradient:", grad_decoder(z))
print("Decoder Jacobian:\n", jacobian_decoder(params, z))
print("Decoder Hessian:\n", hessian_decoder(z))
