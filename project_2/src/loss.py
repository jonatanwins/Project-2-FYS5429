import jax.numpy as jnp
from jax import jacobian, vmap, jacfwd, jacrev
from sindyLibrary import sindy_library_factory
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
    
    def dynamics_x_residual(params: ModelParams, z: Array, dx_dt: Array, theta: Array, xi: Array, mask: Array):
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
        #jacobian_fn of decoder with respect to z
        dpsi_dz = jacfwd(psi, argnums=1)
        #sindy dz prediction in x space
        sindy_dz_in_x = dpsi_dz(params, z) @ (theta @ (mask * xi))
        return (dx_dt - sindy_dz_in_x) ** 2
    
    v_dynamics_x_residual = vmap(dynamics_x_residual, in_axes=(None, 0, 0, 0, None, None))

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
        return jnp.mean(v_dynamics_x_residual(params, z, dx_dt, theta, xi, mask))
    
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
        #jacrev of encoder with respect to x
        dphi_dx = jacrev(phi, argnums=1)
        #dx from data in z space 
        dx_in_z = dphi_dx(params, x) @ dx_dt

        return (dx_in_z - theta @ (mask * xi)) ** 2 #dx_in_z - sindy prediction for dz
    
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
    def loss_regularization(xi: Array, mask: Array) -> Array:
        """
        Regularization loss

        Arguments:
            xi {Array} -- SINDy coefficients

        Returns:
            Array -- L1 norm of input
        """
        return jnp.mean(jnp.abs(xi*mask))
    
    return loss_regularization

def loss_dynamics_x_second_order_factory(decoder: nn.Module, encoder: nn.Module):

    def psi(decoderparams, z):
        return decoder.apply({"params": decoderparams}, z)
    
    def phi(encoderparams, x):
        return encoder.apply({"params": encoderparams}, x)

    def loss_dynamics_x_second_order_residual(decoder_params: ModelParams, encoder_params: ModelParams, z: Array, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
        """
        Second-order loss for the dynamics in x for a single data point

        Arguments:
            decoder_params {ModelParams} -- Decoder parameters
            encoder_params {ModelParams} -- Encoder parameters
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
        # Compute the Jacobian of the encoder with respect to x
        dphi_dx = jacrev(phi, argnums=1)
        dphi_dx_val = dphi_dx(encoder_params, x)

        # Transform dx from x-space to z-space
        dx_in_z = dphi_dx_val @ dx

        # Compute the Jacobian and Hessian of the decoder with respect to z
        dpsi_dz = jacfwd(psi, argnums=1)
        dpsi_dz_val = dpsi_dz(decoder_params, z)

        dpsi_dz2 = jacfwd(dpsi_dz, argnums=1)
        dpsi_dz2_val = dpsi_dz2(decoder_params, z)

        # SINDy reconstruction of ddz
        ddz_reconstructed = theta @ (mask * xi)

        # Apply the chain rule for the second-order derivative
        ddz_in_x = (dpsi_dz2_val @ dx_in_z) @ dx_in_z + dpsi_dz_val @ ddz_reconstructed

        return (ddz_in_x - ddx) ** 2

    v_loss_dynamics_x_second_order_residual = vmap(loss_dynamics_x_second_order_residual, in_axes=(None, None, 0, 0, 0, 0, 0, None, None))

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
        return jnp.mean(v_loss_dynamics_x_second_order_residual(decoder_params, encoder_params, z, x, dx, ddx, theta, xi, mask))
    
    return loss_dynamics_x_second_order

def loss_dynamics_z_second_order_factory(encoder: nn.Module):
     
    def phi(params, x):
        return encoder.apply({"params": params}, x)

    def loss_dynamics_z_second_order_residual(params: ModelParams, x: Array, dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array) -> Array:
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
        # Jacobian of encoder with respect to x     
        dphi_dx = jacrev(phi, argnums=1)
        # Hessian of encoder with respect to x
        ddphi_dx2 = jacrev(dphi_dx, argnums=1) 

        ddx_in_z = (ddphi_dx2(params, x) @ dx) @ dx + dphi_dx(params, x) @ ddx # chain rule

        return (ddx_in_z - theta @ (mask * xi)) ** 2

    v_loss_dynamics_z_second_order_residual = vmap(loss_dynamics_z_second_order_residual, in_axes=(None, 0, 0, 0, 0, None, None))

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
        return jnp.mean(v_loss_dynamics_z_second_order_residual(params, x, dx, ddx, theta, xi, mask))
    
    return loss_dynamics_z_second_order


def loss_fn_factory(autoencoder: nn.Module, weights: Tuple[float, float, float, float] = (1, 1, 40, 1), regularization: bool = True, second_order: bool = False, **library_kwargs) -> Callable:
    """
    Create a loss function for different SINDy libraries.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        latent_dim (int): Dimension of latent space.
        poly_order (int): Polynomial order.
        include_sine (bool, optional): Include sine functions in the library. Defaults to False.
        weights (tuple, optional): Weights for the loss functions. Defaults to (1, 1, 40, 1).
        regularization (bool, optional): Whether to include regularization loss. Defaults to True.
        second_order (bool, optional): Whether to include second-order dynamics. Defaults to False.

    Returns:
        Callable: Loss function.
    """
    if second_order:
        library_kwargs['n_states'] *= 2

    sindy_library = sindy_library_factory(**library_kwargs)
    recon_weight, x_weight, z_weight, reg_weight = weights

    # Unpacking autoencoder
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    # Vectorize individual losses using vmap
    loss_reconstruction = recon_loss_factory()
    loss_regularization = loss_regularization_factory()

    if second_order:
        loss_dynamics_x = loss_dynamics_x_second_order_factory(decoder, encoder)
        loss_dynamics_z = loss_dynamics_z_second_order_factory(encoder)
    else:
        loss_dynamics_x = loss_dynamics_x_factory(decoder)
        loss_dynamics_z = loss_dynamics_z_factory(encoder)


    def loss_fn(params: ModelLayers, batch: Tuple, mask: Array) -> Tuple[float, dict]:
        if second_order:
            x, dx, ddx = batch
        else:
            x, dx = batch

        # Calculate z and x_hat
        z, x_hat = autoencoder.apply({"params": params}, x)
        theta = sindy_library(z)
        xi = params["sindy_coefficients"]

        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        # Compute reconstruction loss
        recon_loss = recon_weight * loss_reconstruction(x, x_hat)
        
        # Compute dynamics losses
        if second_order:
            x_dynamics_loss = x_weight * loss_dynamics_x(decoder_params, encoder_params, z, x, dx, ddx, theta, xi, mask)
            z_dynamics_loss = z_weight * loss_dynamics_z(encoder_params, x, dx, ddx, theta, xi, mask)
        else:
            x_dynamics_loss = x_weight * loss_dynamics_x(decoder_params, z, dx, theta, xi, mask)
            z_dynamics_loss = z_weight * loss_dynamics_z(encoder_params, x, dx, theta, xi, mask)

        total_loss = recon_loss + x_dynamics_loss + z_dynamics_loss

        loss_dict = {
            "loss": total_loss,
            "reconstruction": recon_loss,
            "dynamics_x": x_dynamics_loss,
            "dynamics_z": z_dynamics_loss,
        }

        if regularization:
            loss_reg = reg_weight * loss_regularization(xi, mask)
            total_loss += loss_reg
            loss_dict["regularization"] = loss_reg

        loss_dict["loss"] = total_loss
        return total_loss, loss_dict

    return loss_fn
if __name__ == "__main__":
    from autoencoder import Encoder, Decoder, Autoencoder
    from trainer import TrainState
    from sindyLibrary import library_size
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

