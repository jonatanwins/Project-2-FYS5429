import jax.numpy as jnp
from jax import jacfwd, jacrev, vmap
from sindy_utils import create_sindy_library
from typing import Tuple
from jax import Array
from type_utils import ModelLayers, ModelParams
from flax import linen as nn

def loss_recon_single(x: Array, x_hat: Array) -> Array:
    """
    Reconstruction loss for a single data point

    Arguments:
        x {Array} -- Original data
        x_hat {Array} -- Reconstructed data

    Returns:
        Array -- Reconstruction loss
    """
    return jnp.linalg.norm(x - x_hat) ** 2

def loss_dynamics_x_single(params: ModelParams, decoder: nn.Module, z: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array) -> Array:
    """
    Loss for the dynamics in x for a single data point

    Arguments:
        params {ModelParams} -- Model parameters
        decoder {nn.Module} -- Decoder
        z {Array} -- Latent space
        dx_dt {Array} -- Time derivative of x
        theta {Array} -- SINDy library
        xi {Array} -- SINDy coefficients
        mask {Array} -- Mask

    Returns:
        Array -- Loss dynamics in x
    """
    def psi(params, z):
        return decoder.apply({"params": params}, z)

    dpsi_dz = jacfwd(psi, argnums=1)
    dx_dt_sindy_recon = jnp.dot(dpsi_dz(params, z), theta @ (mask * xi))
    return jnp.linalg.norm(dx_dt - dx_dt_sindy_recon) ** 2

def loss_dynamics_z_single(params: ModelParams, encoder: nn.Module, x: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array) -> Array:
    """
    Loss for the dynamics in z for a single data point

    Arguments:
        params {ModelParams} -- Model parameters
        encoder {nn.Module} -- Encoder
        x {Array} -- Original data
        dx_dt {Array} -- Time derivative of x
        theta {Array} -- SINDy library
        xi {Array} -- SINDy coefficients
        mask {Array} -- Mask

    Returns:
        Array -- Loss dynamics in z
    """
    def phi(params, x):
        return encoder.apply({"params": params}, x)

    dphi_dx = jacrev(phi, argnums=1)
    dx_in_z = jnp.dot(dphi_dx(params, x), dx_dt)
    return jnp.linalg.norm(dx_in_z - theta @ (mask * xi)) ** 2

def loss_dynamics_x_second_order_single(params: ModelParams, decoder: nn.Module,    z: Array, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
    """
    Second-order loss for the dynamics in x for a single data point

    Arguments:
        params {ModelParams} -- Model parameters
        decoder {nn.Module} -- Decoder
        z {Array} -- Latent space
        x {Array} -- Original data
        dx {Array} -- First derivative of x
        ddx {Array} -- Second derivative of x
        theta {Array} -- SINDy library
        xi {Array} -- SINDy coefficients
        mask {Array} -- Mask

    Returns:
        Array -- Second-order loss dynamics in x
    """
    def psi(params, z):
        return decoder.apply({"params": params}, z)

    dpsi_dz = jacfwd(psi, argnums=1)
    ddpsi_dz2 = jacfwd(lambda z: dpsi_dz(params, z))

    ddx_sindy_recon = jnp.dot(ddpsi_dz2(z), theta @ (mask * xi))
    return jnp.linalg.norm(ddx - ddx_sindy_recon) ** 2

def loss_dynamics_z_second_order_single(params: ModelParams, encoder: nn.Module, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
    """
    Second-order loss for the dynamics in z for a single data point

    Arguments:
        params {ModelParams} -- Model parameters
        encoder {nn.Module} -- Encoder
        x {Array} -- Original data
        dx {Array} -- First derivative of x
        ddx {Array} -- Second derivative of x
        theta {Array} -- SINDy library
        xi {Array} -- SINDy coefficients
        mask {Array} -- Mask

    Returns:
        Array -- Second-order loss dynamics in z
    """     
    def phi(params, x):
        return encoder.apply({"params": params}, x)

    J_f = jacrev(phi, argnums=1)

    def y_derivative(x, dx, ddx):
        J_f_x = J_f(params, x)
        dJ_f_dx = jacrev(lambda x: J_f(params, x))(x)
        dy_dt = (dJ_f_dx @ dx) @ dx + J_f_x @ ddx
        return dy_dt

    ddphi_dz2 = y_derivative(x, dx, ddx)
    return jnp.linalg.norm(ddphi_dz2 - theta @ (mask * xi)) ** 2

def loss_regularization(xi: Array, mask: Array) -> Array:
    """
    Regularization loss

    Arguments:
        xi {Array} -- SINDy coefficients

    Returns:
        Array -- L1 norm of input
    """
    return jnp.linalg.norm(xi * mask, ord=1)

def base_loss_fn_first_order(params: ModelLayers, batch: Tuple, autoencoder: nn.Module, mask: Array, sindy_library, weights, v_loss_recon, v_loss_dynamics_x, v_loss_dynamics_z):
    """
    Base loss function without regularization for first-order derivatives

    Args:
        params (ModelLayers): Model parameters
        autoencoder (nn.Module): Autoencoder model
        batch (Tuple): Tuple of x and dx
        mask (Array): Mask
        sindy_library: SINDy library function
        weights: Tuple of weights for the loss components

    Returns:
        Tuple: Total loss and dictionary of losses
    """
    recon_weight, dx_weight, dz_weight, reg_weight = weights
    x, dx = batch

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    z, x_hat = autoencoder.apply({"params": params}, x)
    theta = sindy_library(z)
    xi = params["sindy_coefficients"]

    encoder_params = params["encoder"]
    decoder_params = params["decoder"]

    # Compute losses across the entire batch
    loss_reconstruction = jnp.mean(v_loss_recon(x, x_hat))
    loss_dynamics_x_part = jnp.mean(v_loss_dynamics_x(
        decoder_params, decoder, z, dx, theta, xi, mask
    ))
    loss_dynamics_z_part = jnp.mean(v_loss_dynamics_z(
        encoder_params, encoder, x, dx, theta, xi, mask
    ))

    total_loss = (
        recon_weight * loss_reconstruction
        + dx_weight * loss_dynamics_x_part
        + dz_weight * loss_dynamics_z_part
    )

    loss_dict = {
        "loss": total_loss,
        "reconstruction": recon_weight * loss_reconstruction,
        "dynamics_x": dx_weight * loss_dynamics_x_part,
        "dynamics_z": dz_weight * loss_dynamics_z_part,
    }

    return total_loss, loss_dict

def base_loss_fn_second_order(params: ModelLayers, batch: Tuple, autoencoder: nn.Module, mask: Array, sindy_library, weights, v_loss_recon, v_loss_dynamics_x, v_loss_dynamics_z):
    """
    Base loss function without regularization for second-order derivatives

    Args:
        params (ModelLayers): Model parameters
        autoencoder (nn.Module): Autoencoder model
        batch (Tuple): Tuple of x, dx, and ddx
        mask (Array): Mask
        sindy_library: SINDy library function
        weights: Tuple of weights for the loss components

    Returns:
        Tuple: Total loss and dictionary of losses
    """
    recon_weight, dx_weight, dz_weight, reg_weight = weights
    x, dx, ddx = batch

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    z, x_hat = autoencoder.apply({"params": params}, x)
    theta = sindy_library(z)
    xi = params["sindy_coefficients"]

    encoder_params = params["encoder"]
    decoder_params = params["decoder"]

    # Compute losses across the entire batch
    loss_reconstruction = jnp.mean(v_loss_recon(x, x_hat))
    loss_dynamics_x_part = jnp.mean(v_loss_dynamics_x(
        decoder_params, decoder, z, x, dx, ddx, theta, xi, mask
    ))
    loss_dynamics_z_part = jnp.mean(v_loss_dynamics_z(
        encoder_params, encoder, x, dx, ddx, theta, xi, mask
    ))

    total_loss = (
        recon_weight * loss_reconstruction
        + dx_weight * loss_dynamics_x_part
        + dz_weight * loss_dynamics_z_part
    )

    loss_dict = {
        "loss": total_loss,
        "reconstruction": recon_weight * loss_reconstruction,
        "dynamics_x": dx_weight * loss_dynamics_x_part,
        "dynamics_z": dz_weight * loss_dynamics_z_part,
    }

    return total_loss, loss_dict

def loss_fn_factory(latent_dim: int, poly_order: int, include_sine: bool = False, weights: tuple = (1, 1, 40, 1), regularization: bool = True, second_order: bool = False):
    """
    Create a loss function for different sindy libraries

    Args:
        latent_dim (int): dimension of latent space
        poly_order (int): polynomial order
        include_sine (bool, optional): Include sine functions in the library. Defaults to False.
        weights (tuple, optional): Weights for the loss functions. Defaults to (1, 1, 40, 1).
        regularization (bool, optional): Whether to include regularization loss. Defaults to True.
        second_order (bool, optional): Whether to use second-order derivatives. Defaults to False.

    Returns:
        Callable: Loss function
    """
    sindy_library = create_sindy_library(poly_order, include_sine, n_states=latent_dim)

    v_loss_recon = vmap(loss_recon_single, in_axes=(0, 0))

    if second_order:
        v_loss_dynamics_x = vmap(loss_dynamics_x_second_order_single, in_axes=(None, None, 0, None, None, 0, 0, None, None)) #args are params, decoder, z, x, dx, ddx, theta, xi, mask
        v_loss_dynamics_z = vmap(loss_dynamics_z_second_order_single, in_axes=(None, None, 0, 0, 0, 0, None, None)) #args are params, encoder, x, dx, ddx, theta, xi, mask
        base_loss_fn = base_loss_fn_second_order
    else:
        v_loss_dynamics_x = vmap(loss_dynamics_x_single, in_axes=(None, None, 0, 0, 0, None, None)) #args are params, decoder, z, dx, theta, xi, mask
        v_loss_dynamics_z = vmap(loss_dynamics_z_single, in_axes=(None, None, 0, 0, 0, None, None)) #args are params, encoder, x, dx, theta, xi, mask
        base_loss_fn = base_loss_fn_first_order

    if regularization:
        def loss_fn_with_reg(params: ModelLayers, batch: Tuple, autoencoder: nn.Module, mask: Array):
            total_loss, loss_dict = base_loss_fn(params, batch, autoencoder, mask, sindy_library, weights, v_loss_recon, v_loss_dynamics_x, v_loss_dynamics_z)
            xi = params["sindy_coefficients"]
            loss_reg = loss_regularization(xi, mask)
            total_loss += weights[3] * loss_reg
            loss_dict["loss"] = total_loss
            loss_dict["regularization"] = weights[3] * loss_reg
            return total_loss, loss_dict

        return loss_fn_with_reg
    else:
        def loss_fn(params: ModelLayers, batch: Tuple, autoencoder: nn.Module, mask: Array):
            return base_loss_fn(params, batch, autoencoder, mask, sindy_library, weights, v_loss_recon, v_loss_dynamics_x, v_loss_dynamics_z)

        return loss_fn

if __name__ == "__main__":
    from autoencoder import Encoder, Decoder, Autoencoder
    from trainer import TrainState
    from sindy_utils import library_size
    from jax import random
    import optax

    key = random.PRNGKey(0)
    input_dim = 128
    latent_dim = 3
    poly_order = 3
    lib_size = library_size(latent_dim, poly_order)

    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32])
    decoder = Decoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32])
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32], encoder=encoder, decoder=decoder, lib_size=lib_size)

    # Create some random data
    key, subkey = random.split(key)
    x = random.normal(subkey, (10, input_dim))

    key, subkey = random.split(key)
    dx = random.normal(subkey, (10, input_dim))
    ddx = random.normal(subkey, (10, input_dim))

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

    # First-order loss
    loss_fn_first_order = loss_fn_factory(latent_dim, poly_order, include_sine=False, weights=(1, 1, 40, 1), second_order=False)
    loss_first_order, losses_first_order = loss_fn_first_order(state.params, (x, dx), autoencoder, state.mask)
    print("First-order total loss:", loss_first_order)
    print("First-order individual losses:", losses_first_order)

    # Second-order loss
    loss_fn_second_order = loss_fn_factory(latent_dim, poly_order, include_sine=False, weights=(1, 1, 40, 1), second_order=True)
    loss_second_order, losses_second_order = loss_fn_second_order(state.params, (x, dx, ddx), autoencoder, state.mask)
    print("Second-order total loss:", loss_second_order)
    print("Second-order individual losses:", losses_second_order)
