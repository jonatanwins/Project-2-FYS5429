import jax.numpy as jnp
from jax import jacobian, vmap
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

def loss_dynamics_dx_single(params: ModelParams, decoder: nn.Module, z: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array) -> Array:
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

    jacobian_fn = jacobian(psi, argnums=1)
    dpsi_dt = jnp.dot(jacobian_fn(params, z), theta @ (mask * xi))
    return jnp.linalg.norm(dx_dt - dpsi_dt) ** 2

def loss_dynamics_dz_single(params: ModelParams, encoder: nn.Module, x: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array) -> Array:
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
    
    jacobian_fn = jacobian(phi, argnums=1)
    dphi_dt = jnp.dot(jacobian_fn(params, x), dx_dt)
    return jnp.linalg.norm(dphi_dt - theta @ (mask * xi)) ** 2

def loss_regularization(xi: Array) -> Array:
    """
    Regularization loss

    Arguments:
        xi {Array} -- SINDy coefficients

    Returns:
        Array -- L1 norm of input
    """
    return jnp.linalg.norm(xi, ord=1)


def create_loss_fn(latent_dim: int, poly_order: int, include_sine: bool = False, weights: tuple = (1, 1, 40, 1), regularization: bool = True):
    """
    Create a loss function for different sindy libraries

    Args:
        latent_dim (int): dimension of latent space
        poly_order (int): polynomial order
        include_sine (bool, optional): Include sine functions in the library. Defaults to False.
        weights (tuple, optional): Weights for the loss functions. Defaults to (1, 1, 40, 1).
        regularization (bool, optional): Whether to include regularization loss. Defaults to True.

    Returns:
        Callable: Loss function
    """
    sindy_library = create_sindy_library(poly_order, include_sine, n_states=latent_dim)
    recon_weight, dx_weight, dz_weight, reg_weight = weights

    # Vectorize individual losses using vmap
    v_loss_recon = vmap(loss_recon_single, in_axes=(0, 0))
    v_loss_dynamics_dx = vmap(loss_dynamics_dx_single, in_axes=(None, None, 0, 0, 0, None, None))
    v_loss_dynamics_dz = vmap(loss_dynamics_dz_single, in_axes=(None, None, 0, 0, 0, None, None))


    def base_loss_fn(params: ModelLayers, batch: Tuple, autoencoder: nn.Module, mask: Array):
        """
        Base loss function without regularization

        Args:
            params (ModelLayers): Model parameters
            autoencoder (nn.Module): Autoencoder model
            batch (Tuple): Tuple of features and target
            mask (Array): Mask

        Returns:
            Tuple: Total loss and dictionary of losses
        """
        features, target = batch

        encoder = autoencoder.encoder
        decoder = autoencoder.decoder

        z, x_hat = autoencoder.apply({"params": params}, features)
        theta = sindy_library(z)
        xi = params["sindy_coefficients"]

        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        # Compute losses across the entire batch
        loss_reconstruction = jnp.mean(v_loss_recon(features, x_hat))
        loss_dynamics_dx_part = jnp.mean(v_loss_dynamics_dx(
            decoder_params, decoder, z, target, theta, xi, mask
        ))
        loss_dynamics_dz_part = jnp.mean(v_loss_dynamics_dz(
            encoder_params, encoder, features, target, theta, xi, mask
        ))
        
        total_loss = (
            recon_weight * loss_reconstruction
            + dx_weight * loss_dynamics_dx_part
            + dz_weight * loss_dynamics_dz_part
        )

        loss_dict = {
            "loss": total_loss,
            "reconstruction": recon_weight * loss_reconstruction,
            "dynamics_dx": dx_weight * loss_dynamics_dx_part,
            "dynamics_dz": dz_weight * loss_dynamics_dz_part,
        }

        return total_loss, loss_dict

    if regularization:
        def loss_fn_with_reg(params: ModelLayers, batch: Tuple, autoencoder: nn.Module, mask: Array):
            total_loss, loss_dict = base_loss_fn(params, batch, autoencoder, mask)
            xi = params["sindy_coefficients"]
            loss_reg = loss_regularization(xi)
            total_loss += reg_weight * loss_reg
            loss_dict["loss"] = total_loss
            loss_dict["regularization"] = reg_weight * loss_reg
            return total_loss, loss_dict

        return loss_fn_with_reg
    else:
        return base_loss_fn


if __name__ == "__main__":
    # lets thest the loss function
    from jax import random, tree_map
    import jax.numpy as jnp
    from sindy_utils import library_size
    from autoencoder import Autoencoder, Encoder, Decoder
    from flax import linen as nn
    from flax.core.frozen_dict import freeze
    from trainer import TrainState
    import optax

    key = random.PRNGKey(0)
    input_dim = 128
    latent_dim = 3
    lib_size = library_size(3, 3)
    poly_order = 3
    

    encoder = Encoder(input_dim=input_dim,
                      latent_dim=latent_dim, widths=[32, 32])
    decoder = Decoder(input_dim=input_dim,
                      latent_dim=latent_dim, widths=[32, 32])
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, lib_size=lib_size, widths=[
        32, 32], encoder=encoder, decoder=decoder)

    # create some random data
    key, subkey = random.split(key)
    features = random.normal(subkey, (10, input_dim))

    key, subkey = random.split(key)
    target = random.normal(subkey, (10, input_dim))

    variables = autoencoder.init(subkey, features)

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

    loss_fn = create_loss_fn(latent_dim,poly_order, include_sine=False, weights=(1, 1, 40, 1))


    loss, losses = loss_fn(
        state.params, (features, target), autoencoder, state.mask)
    print(loss)
    print(losses)
