import jax.numpy as jnp
from jax import jacobian, vmap, jacfwd, jacrev
from sindy_utils import create_sindy_library
from typing import Tuple
from jax import Array
from type_utils import ModelLayers, ModelParams, Callable
from flax import linen as nn

def recon_loss_factory()->Callable:
    """
    Factory function to create a reconstruction loss function
    Returns:
        Callable -- Reconstruction loss function
    """
    def recon_residual(x: Array, x_hat: Array) -> Array:
        """
        Reconstruction loss for a single data point

        Arguments:
            x {Array} -- Original data
            x_hat {Array} -- Reconstructed data

        Returns:
            Array -- Reconstruction loss
        """
        return (x - x_hat) ** 2
    
    v_recon_residual = vmap(recon_residual, in_axes=(0, 0))

    def recon_loss(x: Array, x_hat: Array) -> Array:
        """
        Reconstruction loss for the entire batch

        Arguments:
            params {ModelLayers} -- Model parameters
            x {Array} -- Original data
            x_hat {Array} -- Reconstructed data

        Returns:
            Array -- Reconstruction loss
        """
        return jnp.mean(v_recon_residual(x, x_hat))

    return recon_loss

def loss_dynamics_dx_factory(decoder: nn.Module):

    def psi(params,z):
        return decoder.apply({"params": params}, z)
    
    def dynamixs_dx_residual(params: ModelParams, z: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array):
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

        jacobian_fn = jacobian(psi, argnums=1)
        dpsi_dt = jnp.dot(jacobian_fn(params, z), theta @ (mask * xi))
        return (dx_dt - dpsi_dt) ** 2
    
    v_dynamics_dx_residual = vmap(dynamixs_dx_residual, in_axes=(None, 0, 0, 0, None, None))

    def loss_dynamics_dx(params: ModelParams, z: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array) -> Array:
        """
        Loss for the dynamics in x for the entire batch

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
        return jnp.mean(v_dynamics_dx_residual(params, z, dx_dt, theta, xi, mask))
    
    return loss_dynamics_dx

def loss_dynamics_dz_factory(encoder: nn.Module):

    def phi(params, x):
        return encoder.apply({"params": params}, x)
    
    def dynamics_dz_residual(params: ModelParams, x: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array):
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
        jacobian_fn = jacobian(phi, argnums=1)
        dphi_dt = jnp.dot(jacobian_fn(params, x), dx_dt)
        return (dphi_dt - theta @ (mask * xi)) ** 2
    
    v_dynamics_dz_residual = vmap(dynamics_dz_residual, in_axes=(None, 0, 0, 0, None, None))

    def loss_dynamics_dz(params: ModelParams, x: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array) -> Array:
        """
        Loss for the dynamics in z for the entire batch

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
        return jnp.mean(v_dynamics_dz_residual(params, x, dx_dt, theta, xi, mask))
    
    return loss_dynamics_dz

def loss_regularization_factory()->Callable:
    """
    Factory function to create a regularization loss function
    Returns:
        Callable -- Regularization loss function
    """
    def loss_regularization(xi: Array) -> Array:
        """
        Regularization loss

        Arguments:
            xi {Array} -- SINDy coefficients

        Returns:
            Array -- L1 norm of input
        """
        return jnp.mean(jnp.abs(xi))
    
    return loss_regularization

def loss_fn_factory(autoencoder: nn.Module, latent_dim: int, poly_order: int, include_sine: bool = False, weights: tuple = (1, 1, 40, 1), regularization: bool = True):
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

    # Unpacking autoencoder
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    # Vectorize individual losses using vmap
    loss_reconstruction = recon_loss_factory()
    loss_dynamics_dx = loss_dynamics_dx_factory(decoder)
    loss_dynamics_dz = loss_dynamics_dz_factory(encoder)
    loss_regularization = loss_regularization_factory()


    def base_loss_fn(params: ModelLayers, batch: Tuple, mask: Array):
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

        ### Calculate z and x_hat
        z, x_hat = autoencoder.apply({"params": params}, features)
        theta = sindy_library(z)
        xi = params["sindy_coefficients"]

        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        # Compute losses across the entire batch
        recon_loss = recon_weight * loss_reconstruction(features, x_hat)
        dx_loss = dx_weight * loss_dynamics_dx(decoder_params, z, target, theta, xi, mask)
        dz_loss = dz_weight * loss_dynamics_dz(encoder_params, features, target, theta, xi, mask)
        total_loss = (
            recon_loss
            + dx_loss
            + dz_loss
        )

        loss_dict = {
            "loss": total_loss,
            "reconstruction": recon_loss,
            "dynamics_dx": dx_loss,
            "dynamics_dz": dz_loss,
        }
        return total_loss, loss_dict

    if regularization:
        def loss_fn_with_reg(params: ModelLayers, batch: Tuple, mask: Array):
            total_loss, loss_dict = base_loss_fn(params, batch, mask)
            xi = params["sindy_coefficients"]
            loss_reg = reg_weight * loss_regularization(xi)
            total_loss += loss_reg
            loss_dict["loss"] = total_loss
            loss_dict["regularization"] = loss_reg
            return total_loss, loss_dict

        return loss_fn_with_reg
    else:
        return base_loss_fn
    
def loss_dynamics_x_second_order_single(decoderparams: ModelParams, decoder: nn.Module,  encoderparams:ModelParams ,encoder: nn.Module, z: Array, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
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
    def psi(z):
        return decoder.apply({"params": decoderparams}, z)
    
    def phi(x):
        return encoder.apply({"params": encoderparams}, x)
    
    dphi_dx = jacrev(phi)

    dx_in_z = dphi_dx(x) @ dx

    dpsi_dz = jacfwd(psi)
    ddpsi_dz2 = jacfwd(dpsi_dz)
    
    ddz_in_x = (ddpsi_dz2(z) @ dx_in_z) @ dx_in_z + dpsi_dz(z) @ (theta @ (mask * xi)) #chain rule (theta @ (mask * xi) is the SINDy reconstruction of ddz

    return jnp.linalg.norm(ddz_in_x - ddx) ** 2

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
    def phi(x):
        return encoder.apply({"params": params}, x)

    dphi_dx = jacrev(phi)
    ddphi_dx2 = jacrev(dphi_dx) #second derivative/hessian

    ddx_in_z = (ddphi_dx2(x) @ dx) @ dx + dphi_dx(x) @ ddx #chain rule

    return jnp.linalg.norm(ddx_in_z - theta @ (mask * xi)) ** 2



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

    loss_fn = loss_fn_factory(autoencoder, latent_dim,poly_order, include_sine=False, weights=(1, 1, 40, 1))


    loss, losses = loss_fn(
        state.params, (features, target), state.mask)
    print(loss)
    print(losses)
    print(loss.shape)
