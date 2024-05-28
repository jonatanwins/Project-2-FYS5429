import jax.numpy as jnp
from jax import jacobian, vmap, jacfwd, jacrev
from sindy_utils import create_sindy_library
from typing import Tuple, Callable
from jax import Array
from type_utils import ModelLayers, ModelParams
from flax import linen as nn

def recon_loss_factory() -> Callable:
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
            x {Array} -- Original data
            x_hat {Array} -- Reconstructed data

        Returns:
            Array -- Reconstruction loss
        """
        return jnp.mean(v_recon_residual(x, x_hat))

    return recon_loss

def loss_dynamics_x_factory(decoder: nn.Module):

    def psi(params, z):
        return decoder.apply({"params": params}, z)
    
    def dynamics_dx_residual(params: ModelParams, z: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array):
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
        dpsi_dz = jacfwd(psi, argnums=1)
        dpsi_dt = dpsi_dz(params, z) @ (theta @ (mask * xi))
        return (dx_dt - dpsi_dt) ** 2
    
    v_dynamics_dx_residual = vmap(dynamics_dx_residual, in_axes=(None, 0, 0, 0, None, None))

    def loss_dynamics_x(params: ModelParams, z: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array) -> Array:
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
    
    return loss_dynamics_x

def loss_dynamics_z_factory(encoder: nn.Module):

    def phi(params, x):
        return encoder.apply({"params": params}, x)
    
    def dynamics_z_residual(params: ModelParams, x: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array):
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
        dphi_dx = jacrev(phi, argnums=1)
        dx_in_z = dphi_dx(params, x) @ dx_dt

        return (dx_in_z - theta @ (mask * xi)) ** 2
    
    v_dynamics_z_residual = vmap(dynamics_z_residual, in_axes=(None, 0, 0, 0, None, None))

    def loss_dynamics_z(params: ModelParams, x: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array) -> Array:
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
        return jnp.mean(v_dynamics_z_residual(params, x, dx_dt, theta, xi, mask))
    
    return loss_dynamics_z

def loss_regularization_factory() -> Callable:
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

def loss_dynamics_x_second_order_factory(decoder: nn.Module, encoder: nn.Module):

    def psi(z, decoderparams):
        return decoder.apply({"params": decoderparams}, z)
    
    def phi(x, encoderparams):
        return encoder.apply({"params": encoderparams}, x)

    def loss_dynamics_x_second_order_residual(decoder_params: ModelParams, encoder_params: ModelParams, z: Array, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
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
        dphi_dx = jacrev(phi, argnums=1)
        dx_in_z = dphi_dx(x, encoder_params) @ dx

        dpsi_dz = jacfwd(psi)
        ddpsi_dz2 = jacfwd(dpsi_dz)
        
        ddz_in_x = (ddpsi_dz2(z, decoder_params) @ dx_in_z) @ dx_in_z + dpsi_dz(z, decoder_params) @ (theta @ (mask * xi)) # chain rule (theta @ (mask * xi) is the SINDy reconstruction of ddz

        return (ddz_in_x - ddx) ** 2

    v_loss_dynamics_x_second_order_single = vmap(loss_dynamics_x_second_order_residual, in_axes=(None, None, 0, 0, 0, 0, 0, None, None))

    def loss_dynamics_x_second_order(decoder_params: ModelParams, encoder_params:ModelParams, z: Array, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
        """
        Second-order loss for the dynamics in x for the entire batch

        Arguments:
            params {ModelParams} -- Model parameters
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
        return jnp.mean(v_loss_dynamics_x_second_order_single(decoder_params, encoder_params, z, x, dx, ddx, theta, xi, mask))
    
    return loss_dynamics_x_second_order

def loss_dynamics_z_second_order_factory(encoder: nn.Module):
     
    def phi(x, params):
        return encoder.apply({"params": params}, x)

    def loss_dynamics_z_second_order_single(params: ModelParams, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
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
        dphi_dx = jacrev(phi, argnums=1)
        ddphi_dx2 = jacrev(dphi_dx, argnums=1) # second derivative/hessian

        ddx_in_z = (ddphi_dx2(x, params) @ dx) @ dx + dphi_dx(x, params) @ ddx # chain rule

        return (ddx_in_z - theta @ (mask * xi)) ** 2

    v_loss_dynamics_z_second_order_single = vmap(loss_dynamics_z_second_order_single, in_axes=(None, 0, 0, 0, 0, None, None))

    def loss_dynamics_z_second_order(params: ModelParams, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
        """
        Second-order loss for the dynamics in z for the entire batch

        Arguments:
            params {ModelParams} -- Model parameters
            x {Array} -- Original data
            dx {Array} -- First derivative of x
            ddx {Array} -- Second derivative of x
            theta {Array} -- SINDy library
            xi {Array} -- SINDy coefficients
            mask {Array} -- Mask

        Returns:
            Array -- Second-order loss dynamics in z
        """
        return jnp.mean(v_loss_dynamics_z_second_order_single(params, x, dx, ddx, theta, xi, mask))
    
    return loss_dynamics_z_second_order

def loss_fn_factory(autoencoder: nn.Module, latent_dim: int, poly_order: int, include_sine: bool = False, weights: tuple = (1, 1, 40, 1), regularization: bool = True, second_order: bool = False):
    """
    Create a loss function for different SINDy libraries

    Args:
        latent_dim (int): dimension of latent space
        poly_order (int): polynomial order
        include_sine (bool, optional): Include sine functions in the library. Defaults to False.
        weights (tuple, optional): Weights for the loss functions. Defaults to (1, 1, 40, 1).
        regularization (bool, optional): Whether to include regularization loss. Defaults to True.
        second_order (bool, optional): Whether to include second-order dynamics. Defaults to False.

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
    loss_regularization = loss_regularization_factory()

    if second_order:
        loss_dynamics_dx = loss_dynamics_x_second_order_factory(decoder, encoder)
        loss_dynamics_dz = loss_dynamics_z_second_order_factory(encoder)
    else:
        loss_dynamics_dx = loss_dynamics_x_factory(decoder)
        loss_dynamics_dz = loss_dynamics_z_factory(encoder)

    def base_loss_fn_first_order(params: ModelLayers, batch: Tuple, mask: Array):
        """
        Base loss function for first-order dynamics

        Args:
            params (ModelLayers): Model parameters
            batch (Tuple): Tuple of x and dx
            mask (Array): Mask

        Returns:
            Tuple: Total loss and dictionary of losses
        """
        x, dx = batch

        # Calculate z and x_hat
        z, x_hat = autoencoder.apply({"params": params}, x)
        theta = sindy_library(z)
        xi = params["sindy_coefficients"]

        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        # Compute losses across the entire batch
        recon_loss = recon_weight * loss_reconstruction(x, x_hat)
        dx_loss = dx_weight * loss_dynamics_dx(decoder_params, z, dx, theta, xi, mask)
        dz_loss = dz_weight * loss_dynamics_dz(encoder_params, x, dx, theta, xi, mask)
        
        total_loss = recon_loss + dx_loss + dz_loss

        loss_dict = {
            "loss": total_loss,
            "reconstruction": recon_loss,
            "dynamics_dx": dx_loss,
            "dynamics_dz": dz_loss,
        }
        
        return total_loss, loss_dict

    def base_loss_fn_second_order(params: ModelLayers, batch: Tuple, mask: Array):
        """
        Base loss function for second-order dynamics

        Args:
            params (ModelLayers): Model parameters
            batch (Tuple): Tuple of x, dx, and ddx
            mask (Array): Mask

        Returns:
            Tuple: Total loss and dictionary of losses
        """
        x, dx, ddx = batch

        # Calculate z and x_hat
        z, x_hat = autoencoder.apply({"params": params}, x)
        theta = sindy_library(z)
        xi = params["sindy_coefficients"]

        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        # Compute losses across the entire batch
        recon_loss = recon_weight * loss_reconstruction(x, x_hat)
        dx_loss = dx_weight * loss_dynamics_dx(decoder_params, encoder_params, z, x, dx, ddx, theta, xi, mask)
        dz_loss = dz_weight * loss_dynamics_dz(encoder_params, x, dx, ddx, theta, xi, mask)
        
        total_loss = recon_loss + dx_loss + dz_loss

        loss_dict = {
            "loss": total_loss,
            "reconstruction": recon_loss,
            "dynamics_dx": dx_loss,
            "dynamics_dz": dz_loss,
        }
        
        return total_loss, loss_dict

    if regularization:
        def loss_fn_with_reg_first_order(params: ModelLayers, batch: Tuple, mask: Array):
            total_loss, loss_dict = base_loss_fn_first_order(params, batch, mask)
            xi = params["sindy_coefficients"]
            loss_reg = reg_weight * loss_regularization(xi)
            total_loss += loss_reg
            loss_dict["loss"] = total_loss
            loss_dict["regularization"] = loss_reg
            return total_loss, loss_dict

        def loss_fn_with_reg_second_order(params: ModelLayers, batch: Tuple, mask: Array):
            total_loss, loss_dict = base_loss_fn_second_order(params, batch, mask)
            xi = params["sindy_coefficients"]
            loss_reg = reg_weight * loss_regularization(xi)
            total_loss += loss_reg
            loss_dict["loss"] = total_loss
            loss_dict["regularization"] = loss_reg
            return total_loss, loss_dict

        return loss_fn_with_reg_second_order if second_order else loss_fn_with_reg_first_order
    else:
        return base_loss_fn_second_order if second_order else base_loss_fn_first_order

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

    loss_fn = loss_fn_factory(autoencoder, latent_dim, poly_order, include_sine=False, weights=(1, 1, 40, 1), second_order=True)

    loss, losses = loss_fn(state.params, (x, dx, ddx), state.mask)
    print(loss)
    print(losses)
    print(loss.shape)

