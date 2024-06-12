import jax.numpy as jnp
from jax import jacfwd, jacrev, vmap
from typing import Tuple, Callable
from jax import Array
from type_utils import ModelLayers
from flax import linen as nn
from sindyLibrary import sindy_library_factory

def recon_loss_single(x: Array, x_hat: Array) -> Array:
    """
    Computes the reconstruction loss for a single sample.

    Args:
        x: Original input array.
        x_hat: Reconstructed input array.

    Returns:
        Mean squared error between the original and reconstructed input.
    """
    return jnp.mean((x - x_hat) ** 2)

def loss_regularization(xi_masked: Array) -> Array:
    """
    Computes the regularization loss.

    Args:
        xi_masked: Masked SINDy coefficient array.

    Returns:
        Mean absolute value of the masked SINDy coefficients.
    """
    return jnp.mean(jnp.abs(xi_masked))

def loss_dynamics_x_single(dx: Array, theta: Array, dpsi_dz_val: Array, xi_masked: Array) -> Array:
    """
    Computes the dynamics loss in the x-space for a single sample.

    Args:
        dx: Derivative of x.
        theta: SINDy library matrix.
        dpsi_dz_val: Jacobian of the decoder.
        xi_masked: Masked SINDy coefficients.

    Returns:
        Mean squared error of the dynamics in x-space.
    """
    sindy_dz_in_x = dpsi_dz_val @ (theta @ xi_masked)
    return jnp.mean((dx - sindy_dz_in_x) ** 2)

def loss_dynamics_z_single(theta: Array, dx_in_z: Array, xi_masked: Array) -> Array:
    """
    Computes the dynamics loss in the z-space for a single sample.

    Args:
        theta: SINDy library matrix.
        dx_in_z: Derivative of x in z-space.
        xi_masked: Masked SINDy coefficients.

    Returns:
        Mean squared error of the dynamics in z-space.
    """
    return jnp.mean((dx_in_z - theta @ xi_masked) ** 2)

def calculate_outputs_and_derivatives_factory(autoencoder: nn.Module) -> Callable:
    """
    Factory function to calculate outputs and their derivatives for the autoencoder.

    Args:
        autoencoder: Autoencoder module.

    Returns:
        A function to calculate outputs and their derivatives.
    """
    
    def autoencoder_apply(params: ModelLayers, x: Array) -> Tuple[Array, Array]:
        """Apply the autoencoder to the input."""
        return autoencoder.apply({"params": params}, x)

    def dphi_dx(p, x):
        """Calculate the Jacobian of the encoder."""
        return jacrev(lambda x: autoencoder.encoder.apply({"params": p}, x))(x)

    def dpsi_dz(p, z):
        """Calculate the Jacobian of the decoder."""
        return jacfwd(lambda z: autoencoder.decoder.apply({"params": p}, z))(z)

    def calculate_outputs_and_derivatives(params: ModelLayers, x: Array, dx: Array) -> Tuple:
        """
        Calculate the latent representation, reconstruction, and derivatives.

        Args:
            params: Model parameters.
            x: Input array.
            dx: Derivative of the input array.

        Returns:
            Latent representation, reconstruction, derivative in z-space, and decoder Jacobian.
        """
        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        z, x_hat = autoencoder_apply(params, x)
        
        dphi_dx_val = dphi_dx(encoder_params, x)
        dx_in_z = jnp.dot(dphi_dx_val, dx)
        dpsi_dz_val = dpsi_dz(decoder_params, z)

        return z, x_hat, dx_in_z, dpsi_dz_val

    return calculate_outputs_and_derivatives

def first_order_loss_fn_factory(autoencoder: nn.Module, loss_weights: Tuple[float, float, float, float] = (1, 1e-4, 0, 1e-5),  regularization: bool = True, **library_kwargs) -> Callable:
    """
    Factory function to create the first-order loss function.

    Args:
        autoencoder: Autoencoder module.
        loss_weights: Tuple of loss_weights for different loss components.
        regularization: Boolean indicating if regularization should be used.
        **library_kwargs: Additional arguments for the SINDy library.

    Returns:
        A first-order loss function.
    """
    
    calculate_output_and_derivatives = calculate_outputs_and_derivatives_factory(autoencoder)
    sindy_library_fn = sindy_library_factory(**library_kwargs)
    
    recon_weight, x_weight, z_weight, reg_weight = loss_weights

    if not regularization:
        reg_weight = 0
    
    def first_order_loss_fn_single(params: ModelLayers, x: Array, dx: Array, mask: Array) -> Array:
        """
        Computes the first-order loss for a single sample.

        Args:
            params: Model parameters.
            x: Input array.
            dx: Derivative of the input array.
            mask: Mask array for SINDy coefficients.

        Returns:
            Array of loss components for the sample.
        """
        z, x_hat, dx_in_z, dpsi_dz_val = calculate_output_and_derivatives(params, x, dx)

        theta = sindy_library_fn(z)
        xi = params["sindy_coefficients"]
        xi_masked = mask * xi

        recon_loss_val = recon_loss_single(x, x_hat)
        x_dynamics_loss_val = loss_dynamics_x_single(dx, theta, dpsi_dz_val, xi_masked)
        z_dynamics_loss_val = loss_dynamics_z_single(theta, dx_in_z, xi_masked)

        return jnp.array([recon_loss_val, x_dynamics_loss_val, z_dynamics_loss_val])

    def first_order_loss_fn(params: ModelLayers, batch: Tuple, mask: Array) -> Tuple[float, dict]:
        """
        Computes the first-order loss for a batch of samples.

        Args:
            params: Model parameters.
            batch: Tuple containing input array and its derivative.
            mask: Mask array for SINDy coefficients.

        Returns:
            Total loss and a dictionary of loss components.
        """
        x, dx = batch

        loss_components = vmap(first_order_loss_fn_single, in_axes=(None, 0, 0, None))(params, x, dx, mask)
        mean_loss_components = jnp.mean(loss_components, axis=0)

        recon_loss_val, x_dynamics_loss_val, z_dynamics_loss_val = mean_loss_components

        total_loss = recon_weight * recon_loss_val + x_weight * x_dynamics_loss_val + z_weight * z_dynamics_loss_val
        loss_dict = {
            "loss": total_loss,
            "reconstruction": recon_weight * recon_loss_val,
            "dynamics_x": x_weight * x_dynamics_loss_val,
            "dynamics_z": z_weight * z_dynamics_loss_val,
        }

        if reg_weight > 0:
            loss_reg = reg_weight * loss_regularization(mask * params["sindy_coefficients"])
            total_loss += loss_reg
            loss_dict["regularization"] = loss_reg

        loss_dict["loss"] = total_loss
        return total_loss, loss_dict
    
    return first_order_loss_fn

def test():
    from autoencoder import Encoder, Decoder, Autoencoder
    from trainer import TrainState
    from sindyLibrary import library_size
    from jax import random, jit
    import optax

    key = random.PRNGKey(0)
    input_dim = 128
    latent_dim = 3
    poly_order = 3
    include_sine = False
    include_constant = True

    # Configure SINDy library
    lib_kwargs = {'n_states': latent_dim, 'poly_order': poly_order, 'include_sine': include_sine, 'include_constant': include_constant}
    lib_size = library_size(**lib_kwargs)

    # Define the autoencoder
    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32])
    decoder = Decoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32])
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32], encoder=encoder, decoder=decoder, lib_size=lib_size)

    # Create some random data
    key, subkey = random.split(key)
    x = random.normal(subkey, (10, input_dim))
    key, subkey = random.split(key)
    dx = random.normal(subkey, (10, input_dim))

    variables = autoencoder.init(subkey, x)

    state = TrainState(
        step=0,
        apply_fn=autoencoder.apply,
        params=variables["params"],
        rng=subkey,
        tx=None,
        opt_state=None,
        mask=variables['params']['sindy_coefficients'],
    )

    optimizer = optax.adam(1e-3)

    state = TrainState.create(
        apply_fn=state.apply_fn,
        params=state.params,
        tx=optimizer,
        rng=state.rng,
        mask=state.mask
    )

    # First-order dynamics test
    loss_fn_first_order = first_order_loss_fn_factory(autoencoder, loss_weights=(1, 1, 40, 1), regularization=True, **lib_kwargs)
    loss_first_order, losses_first_order = loss_fn_first_order(state.params, (x, dx), state.mask)
    print("First-order loss:", loss_first_order)
    print("First-order loss components:", losses_first_order)

    # Jitted first-order dynamics test
    jitted_loss_fn_first_order = jit(loss_fn_first_order)
    loss_first_order_jit, losses_first_order_jit = jitted_loss_fn_first_order(state.params, (x, dx), state.mask)
    print("Jitted first-order loss:", loss_first_order_jit)
    print("Jitted first-order loss components:", losses_first_order_jit)

    # Additional test points with varying data shapes
    for batch_size in [20, 50, 100]:
        key, subkey = random.split(key)
        x = random.normal(subkey, (batch_size, input_dim))
        key, subkey = random.split(key)
        dx = random.normal(subkey, (batch_size, input_dim))

        loss_first_order, losses_first_order = loss_fn_first_order(state.params, (x, dx), state.mask)
        print(f"First-order loss for batch size {batch_size}:", loss_first_order)
        print(f"First-order loss components for batch size {batch_size}:", losses_first_order)

        # Jitted test for each batch size
        loss_first_order_jit, losses_first_order_jit = jitted_loss_fn_first_order(state.params, (x, dx), state.mask)
        print(f"Jitted first-order loss for batch size {batch_size}:", loss_first_order_jit)
        print(f"Jitted first-order loss components for batch size {batch_size}:", losses_first_order_jit)

if __name__ == "__main__":
    test()