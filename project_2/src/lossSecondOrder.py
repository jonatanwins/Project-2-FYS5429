import jax.numpy as jnp
from jax import jacfwd, jacrev, vmap, jvp, vjp
from typing import Tuple, Callable
from jax import Array
from type_utils import ModelLayers
from flax import linen as nn
from sindyLibrary import sindy_library_factory
import functools

# """
# Functions for getting jacobian and value of a function. Might not add any performance benefit.
# https://github.com/google/jax/pull/762
# """
# def value_and_jacfwd(f, x):
#   pushfwd = functools.partial(jvp, f, (x,))
#   basis = jnp.eye(x.size, dtype=x.dtype).reshape(x.size, *x.shape)
#   y, jac = vmap(pushfwd, out_axes=(None, 0))(basis)
#   return y, jac

# def value_and_jacrev(f, x):
#   y, pullback = vjp(f, x)
#   print(y.shape, jnp.array(pullback(jnp.ones_like(y))).shape)

#   basis = jnp.eye(y.size, dtype=y.dtype)
#   jac = vmap(pullback)(basis)
#   return y, jnp.stack(jac, axis=-1)

def recon_loss_single(x: Array, x_hat: Array) -> Array:
    """
    Calculate reconstruction loss as mean squared error.

    Args:
        x (Array): Original input data.
        x_hat (Array): Reconstructed data from the autoencoder.

    Returns:
        Array: Mean squared error between original and reconstructed data.
    """
    return jnp.mean((x - x_hat) ** 2)

def loss_regularization(xi_masked: Array) -> Array:
    """
    Calculate regularization loss as mean absolute error.

    Args:
        xi_masked (Array): Masked SINDy coefficients.

    Returns:
        Array: Mean absolute error of the masked SINDy coefficients.
    """
    return jnp.mean(jnp.abs(xi_masked))

def loss_dynamics_x_second_order_single(ddx: Array, theta: Array, dx_in_z: Array, dpsi_dz_val: Array, ddpsi_ddz_val: Array, xi_masked: Array) -> Array:
    """
    Calculate the second-order dynamics loss in the original space.

    Args:
        ddx (Array): Second derivative of x.
        theta (Array): SINDy library function output.
        dx_in_z (Array): First derivative of x in the latent space.
        dpsi_dz_val (Array): Jacobian of the decoder.
        ddpsi_ddz_val (Array): Second-order Jacobian of the decoder.
        xi_masked (Array): Masked SINDy coefficients.

    Returns:
        Array: Mean squared error of the second-order dynamics in the original space.
    """
    ddz_reconstructed = theta @ xi_masked
    ddz_in_x = (ddpsi_ddz_val @ dx_in_z) @ dx_in_z + dpsi_dz_val @ ddz_reconstructed
    return jnp.mean((ddz_in_x - ddx) ** 2)

def loss_dynamics_z_second_order_single(dx: Array, ddx: Array, theta: Array, ddphi_ddx_val: Array, dphi_dx_val: Array, xi_masked: Array) -> Array:
    """
    Calculate the second-order dynamics loss in the latent space.

    Args:
        dx (Array): First derivative of x.
        ddx (Array): Second derivative of x.
        theta (Array): SINDy library function output.
        ddphi_ddx_val (Array): Second-order Jacobian of the encoder.
        dphi_dx_val (Array): Jacobian of the encoder.
        xi_masked (Array): Masked SINDy coefficients.

    Returns:
        Array: Mean squared error of the second-order dynamics in the latent space.
    """
    ddx_in_z = (ddphi_ddx_val @ dx) @ dx + dphi_dx_val @ ddx
    return jnp.mean((ddx_in_z - theta @ xi_masked) ** 2)

def calculate_outputs_and_derivatives_second_order_factory(autoencoder: nn.Module) -> Callable:
    """
    Factory function to create a function that calculates outputs and derivatives for second-order dynamics.

    Args:
        autoencoder (nn.Module): The autoencoder model.

    Returns:
        Callable: Function to calculate outputs and derivatives.
    """
    
    def autoencoder_apply(params: ModelLayers, x: Array) -> Tuple[Array, Array]:
        return autoencoder.apply({"params": params}, x)

    def dphi_dx(p, x):
        return jacrev(lambda x: autoencoder.encoder.apply({"params": p}, x))(x)
    
    def ddphi_ddx(p, x):
        return jacfwd(lambda x: dphi_dx(p, x))(x)
    
    # def dphi_and_ddphi_ddx(p, x):
    #     return value_and_jacrev(lambda x: dphi_dx(p, x), x)


    def dpsi_dz(p, z):
        return jacfwd(lambda z: autoencoder.decoder.apply({"params": p}, z))(z)
    
    def ddpsi_ddz(p, z):
        return jacfwd(lambda z: dpsi_dz(p, z))(z)

    # def dpsi_and_ddpsi_ddz(p, z):
    #     return value_and_jacfwd(lambda z: dpsi_dz(p, z), z)

    def calculate_outputs_and_derivatives(params: ModelLayers, x: Array, dx: Array) -> Tuple:
        """
        Calculate the outputs and derivatives for second-order dynamics.

        Args:
            params (ModelLayers): Model parameters.
            x (Array): Input data.
            dx (Array): First derivative of input data.

        Returns:
            Tuple: Outputs and derivatives including latent representation, reconstructed input, Jacobians, and more.
        """
        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        z, x_hat = autoencoder_apply(params, x)
        
        dphi_dx_val = dphi_dx(encoder_params, x)
        ddphi_ddx_val = ddphi_ddx(encoder_params, x)
        # dphi_dx_val, ddphi_ddx_val = dphi_and_ddphi_ddx(encoder_params, x)
        
        dx_in_z = jnp.dot(dphi_dx_val, dx)

        dpsi_dz_val = dpsi_dz(decoder_params, z)
        ddpsi_ddz_val = ddpsi_ddz(decoder_params, z)
        # dpsi_dz_val, ddpsi_ddz_val = dpsi_and_ddpsi_ddz(decoder_params, z)

        return z, x_hat, dphi_dx_val, ddphi_ddx_val, dx_in_z, dpsi_dz_val, ddpsi_ddz_val

    return calculate_outputs_and_derivatives

def second_order_loss_fn_factory(autoencoder: nn.Module, loss_weights: Tuple[float, float, float, float] = (1, 1e-4, 0, 1e-5), regularization: bool = True, **library_kwargs) -> Callable:
    """
    Factory function to create a second-order loss function.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        loss_weights (Tuple[float, float, float, float]): loss_Weights for the different components of the loss.
        regularization (bool): Whether to include regularization loss.
        **library_kwargs: Additional keyword arguments for the SINDy library function.

    Returns:
        Callable: Second-order loss function.
    """
    
    calculate_outputs_and_derivatives = calculate_outputs_and_derivatives_second_order_factory(autoencoder)
    sindy_library_fn = sindy_library_factory(**library_kwargs)
    
    recon_weight, x_weight, z_weight, reg_weight = loss_weights

    if not regularization:
        reg_weight = 0

    def second_order_loss_fn_single(params: ModelLayers, x: Array, dx: Array, ddx: Array, mask: Array) -> Array:
        """
        Calculate the second-order loss for a single data point.

        Args:
            params (ModelLayers): Model parameters.
            x (Array): Input data.
            dx (Array): First derivative of input data.
            ddx (Array): Second derivative of input data.
            mask (Array): Mask for the SINDy coefficients.

        Returns:
            Array: Components of the second-order loss.
        """
        z, x_hat, dphi_dx_val, ddphi_ddx_val, dx_in_z, dpsi_dz_val, ddpsi_ddz_val = calculate_outputs_and_derivatives(params, x, dx)

        theta = sindy_library_fn(jnp.concatenate([z, dx_in_z], axis=0))
        xi = params["sindy_coefficients"]
        xi_masked = mask * xi

        recon_loss_val = recon_loss_single(x, x_hat)
        x_dynamics_loss_val = loss_dynamics_x_second_order_single(ddx, theta, dx_in_z, dpsi_dz_val, ddpsi_ddz_val, xi_masked)
        z_dynamics_loss_val = loss_dynamics_z_second_order_single(dx, ddx, theta, ddphi_ddx_val, dphi_dx_val, xi_masked)

        return jnp.array([recon_loss_val, x_dynamics_loss_val, z_dynamics_loss_val])

    def second_order_loss_fn(params: ModelLayers, batch: Tuple, mask: Array) -> Tuple[float, dict]:
        """
        Calculate the total second-order loss for a batch of data.

        Args:
            params (ModelLayers): Model parameters.
            batch (Tuple): Batch of data containing input, first derivative, and second derivative.
            mask (Array): Mask for the SINDy coefficients.

        Returns:
            Tuple[float, dict]: Total loss and dictionary of loss components.
        """
        x, dx, ddx = batch

        loss_components = vmap(second_order_loss_fn_single, in_axes=(None, 0, 0, 0, None))(params, x, dx, ddx, mask)
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
    
    return second_order_loss_fn


if __name__ == "__main__":
    from autoencoder import Encoder, Decoder, Autoencoder
    from trainer import TrainState
    from sindyLibrary import library_size
    from jax import random, jit
    import optax

    # Initialize random key and model parameters
    key = random.PRNGKey(0)
    input_dim = 128
    latent_dim = 3
    poly_order = 3
    include_sine = False
    include_constant = True

    # Create some random data
    key, subkey = random.split(key)
    x = random.normal(subkey, (10, input_dim))
    key, subkey = random.split(key)
    dx = random.normal(subkey, (10, input_dim))

    # Second-order case
    sindy_input_features = latent_dim * 2
    lib_kwargs = {'n_states': sindy_input_features, 'poly_order': poly_order, 'include_sine': include_sine, 'include_constant': include_constant}
    lib_size = library_size(**lib_kwargs)

    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32])
    decoder = Decoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32])
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32], encoder=encoder, decoder=decoder, lib_size=lib_size)

    # Create some random data for second-order
    key, subkey = random.split(key)
    ddx = random.normal(subkey, (10, input_dim))

    variables = autoencoder.init(subkey, x)

    state = TrainState(
        step=0,
        apply_fn=autoencoder.apply,
        params=variables["params"],
        rng=subkey,
        tx=None,
        opt_state=None,
        mask=variables['params']['sindy_coefficients'], # mask should be initialized with 1s just as sindy coefficients
    )

    optimizer = optax.adam(1e-3)

    state = TrainState.create(
        apply_fn=state.apply_fn,
        params=state.params,
        tx=optimizer,
        rng=state.rng,
        mask=state.mask
    )

    # Second-order dynamics test
    loss_fn_second_order = second_order_loss_fn_factory(autoencoder, loss_weights=(1, 1, 40, 1), regularization=True, **lib_kwargs)
    loss_second_order, losses_second_order = loss_fn_second_order(state.params, (x, dx, ddx), state.mask)
    print("Second-order loss:", loss_second_order)
    print("Second-order loss components:", losses_second_order)

    # Jitted second-order dynamics test
    jitted_loss_fn_second_order = jit(loss_fn_second_order)
    loss_second_order_jit, losses_second_order_jit = jitted_loss_fn_second_order(state.params, (x, dx, ddx), state.mask)
    print("Jitted second-order loss:", loss_second_order_jit)
    print("Jitted second-order loss components:", losses_second_order_jit)
