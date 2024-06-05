import jax.numpy as jnp
from jax import jacobian, vmap, jacfwd, jacrev
from sindyLibrary import sindy_library_factory
from typing import Tuple, Callable
from jax import Array
from type_utils import ModelLayers, ModelParams
from flax import linen as nn

def recon_loss(x: Array, x_hat: Array) -> Array:
    """
    Compute the reconstruction loss as the mean squared error between x and x_hat.

    Args:
        x (Array): Original input array.
        x_hat (Array): Reconstructed input array.

    Returns:
        Array: Reconstruction loss.
    """
    def recon_residual(x: Array, x_hat: Array) -> Array:
        return (x - x_hat) ** 2
    
    v_recon_residual = vmap(recon_residual, in_axes=(0, 0))
    return jnp.mean(v_recon_residual(x, x_hat))


def loss_regularization(xi: Array, mask: Array) -> Array:
    """
    Compute the regularization loss.

    Args:
        xi (Array): SINDy coefficients.
        mask (Array): Mask for SINDy coefficients.

    Returns:
        Array: Regularization loss.
    """
    return jnp.mean(jnp.abs(xi * mask))


def loss_dynamics_x(dx: Array, theta: Array, xi: Array, mask: Array, dpsi_dz_val: Array) -> Array:
    """
    Compute the dynamics loss in x space.

    Args:
        z (Array): Latent space representation.
        dx (Array): Derivative of x with respect to time.
        theta (Array): SINDy library.
        xi (Array): SINDy coefficients.
        mask (Array): Mask for SINDy coefficients.
        dpsi_dz_val (Array): Jacobian of the decoder with respect to z.

    Returns:
        Array: Dynamics loss in x space.
    """
    def dynamics_x_residual(dx: Array, theta: Array, xi: Array, mask: Array, dpsi_dz_val: Array):
        sindy_dz_in_x = dpsi_dz_val @ (theta @ (mask * xi))
        return (dx - sindy_dz_in_x) ** 2
    
    v_dynamics_x_residual = vmap(dynamics_x_residual, in_axes=(0, 0, None, None, 0))
    return jnp.mean(v_dynamics_x_residual(dx, theta, xi, mask, dpsi_dz_val))

def loss_dynamics_z(theta: Array, xi: Array, mask: Array, dx_in_z: Array) -> Array:
    """
    Compute the dynamics loss in z space.

    Args:
        theta (Array): SINDy library.
        xi (Array): SINDy coefficients.
        mask (Array): Mask for SINDy coefficients.
        dx_in_z (Array): Derivative of x in z space.

    Returns:
        Array: Dynamics loss in z space.
    """
    def dynamics_z_residual(theta: Array, xi: Array, mask: Array, dx_in_z: Array):
        return (dx_in_z - theta @ (mask * xi)) ** 2
    
    v_dynamics_z_residual = vmap(dynamics_z_residual, in_axes=(0, None, None, 0))
    return jnp.mean(v_dynamics_z_residual(theta, xi, mask, dx_in_z))


def loss_dynamics_x_second_order(ddx: Array, theta: Array, xi: Array, mask: Array, dx_in_z: Array, dpsi_dz_val: Array, dpsi_dz2_val: Array) -> Array:
    """
    Compute the second-order dynamics loss in x space.

    Args:
        z (Array): Latent space representation.
        x (Array): Original input array.
        dx (Array): First derivative of x.
        ddx (Array): Second derivative of x.
        theta (Array): SINDy library.
        xi (Array): SINDy coefficients.
        mask (Array): Mask for SINDy coefficients.
        dx_in_z (Array): Derivative of x in z space.
        dpsi_dz_val (Array): Jacobian of the decoder with respect to z.
        dpsi_dz2_val (Array): Hessian of the decoder with respect to z.

    Returns:
        Array: Second-order dynamics loss in x space.
    """
    def loss_dynamics_x_second_order_residual(ddx: Array, theta: Array, xi: Array, mask: Array, dx_in_z: Array, dpsi_dz_val: Array, dpsi_dz2_val: Array):
        ddz_reconstructed = theta @ (mask * xi)
        ddz_in_x = (dpsi_dz2_val @ dx_in_z) @ dx_in_z + dpsi_dz_val @ ddz_reconstructed
        return (ddz_in_x - ddx) ** 2

    v_loss_dynamics_x_second_order_residual = vmap(loss_dynamics_x_second_order_residual, in_axes=(0, 0, None, None, 0, 0, 0))
    return jnp.mean(v_loss_dynamics_x_second_order_residual(ddx, theta, xi, mask, dx_in_z, dpsi_dz_val, dpsi_dz2_val))

def loss_dynamics_z_second_order(dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array, ddphi_dx2_val: Array, dphi_dx_val: Array) -> Array:
    """
    Compute the second-order dynamics loss in z space.

    Args:
        x (Array): Original input array.
        dx (Array): First derivative of x.
        ddx (Array): Second derivative of x.
        theta (Array): SINDy library.
        xi (Array): SINDy coefficients.
        mask (Array): Mask for SINDy coefficients.
        dx_in_z (Array): Derivative of x in z space.
        ddphi_dx2_val (Array): Hessian of the encoder with respect to x.
        dphi_dx_val (Array): Jacobian of the encoder with respect to x.

    Returns:
        Array: Second-order dynamics loss in z space.
    """
    def loss_dynamics_z_second_order_residual(dx: Array, ddx: Array, theta: Array, xi: Array, mask: Array, ddphi_dx2_val: Array, dphi_dx_val: Array):
        ddx_in_z = (ddphi_dx2_val @ dx) @ dx + dphi_dx_val @ ddx
        return (ddx_in_z - theta @ (mask * xi)) ** 2

    v_loss_dynamics_z_second_order_residual = vmap(loss_dynamics_z_second_order_residual, in_axes=(0, 0, 0, None, None, 0, 0))
    return jnp.mean(v_loss_dynamics_z_second_order_residual(dx, ddx, theta, xi, mask, ddphi_dx2_val, dphi_dx_val))

def loss_fn_factory(autoencoder: nn.Module, weights: Tuple[float, float, float, float] = (1, 1, 40, 1), regularization: bool = True, second_order: bool = False, **library_kwargs) -> Callable:
    """
    Compute the overall loss function combining reconstruction, dynamics, and regularization losses.

    Args:
        autoencoder (nn.Module): Autoencoder model containing encoder and decoder.
        weights (Tuple[float, float, float, float], optional): Weights for different loss components. Defaults to (1, 1, 40, 1).
        regularization (bool, optional): Whether to include regularization loss. Defaults to True.
        second_order (bool, optional): Whether to include second-order dynamics. Defaults to False.
        **library_kwargs: Additional arguments for SINDy library.

    Returns:
        Callable: Loss function.
    """
    if second_order:
        library_kwargs['n_states'] *= 2

    sindy_library_fn = sindy_library_factory(**library_kwargs)
    recon_weight, x_weight, z_weight, reg_weight = weights

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    ####Encoder and Decoder func######
    def phi(params, x):
        return encoder.apply({"params": params}, x)
    
    def psi(params, z):
        return decoder.apply({"params": params}, z)
    
    ### Derivatives of encoder and decoder ####
    
    # Jacobians of encoder and decoder
    dphi_dx = jacrev(phi, argnums=1)
    dpsi_dz = jacfwd(psi, argnums=1)

    # Hessians of encoder and decoder
    ddphi_dx2 = jacrev(dphi_dx, argnums=1)
    ddpsi_dz2 = jacfwd(dpsi_dz, argnums=1)
    

    def loss_fn(params: ModelLayers, batch: Tuple, mask: Array) -> Tuple[float, dict]:
        if second_order:
            x, dx, ddx = batch
        else:
            x, dx = batch

        z, x_hat = autoencoder.apply({"params": params}, x)
        
        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        dx_in_z = vmap(lambda x, dx: dphi_dx(encoder_params, x) @ dx)(x, dx) # dx in z space

        if second_order:
            theta = vmap(sindy_library_fn)(jnp.concatenate([z, dx_in_z], axis=1))
        else:
            theta = vmap(sindy_library_fn)(z)

        xi = params["sindy_coefficients"]

        recon_loss_val = recon_weight * recon_loss(x, x_hat)

        dpsi_dz_val = vmap(lambda z: dpsi_dz(decoder_params, z))(z)

        if second_order:
            ddphi_dx2_val = vmap(lambda x: ddphi_dx2(encoder_params, x))(x)
            dpsi_dz2_val = vmap(lambda z: ddpsi_dz2(decoder_params, z))(z)

            x_dynamics_loss_val = x_weight * loss_dynamics_x_second_order(ddx, theta, xi, mask, dx_in_z, dpsi_dz_val, dpsi_dz2_val)
            z_dynamics_loss_val = z_weight * loss_dynamics_z_second_order(dx, ddx, theta, xi, mask, dx_in_z, ddphi_dx2_val, dphi_dx)
        else:
            x_dynamics_loss_val = x_weight * loss_dynamics_x(dx, theta, xi, mask, dpsi_dz_val)
            z_dynamics_loss_val = z_weight * loss_dynamics_z(theta, xi, mask, dx_in_z)

        total_loss = recon_loss_val + x_dynamics_loss_val + z_dynamics_loss_val

        loss_dict = {
            "loss": total_loss,
            "reconstruction": recon_loss_val,
            "dynamics_x": x_dynamics_loss_val,
            "dynamics_z": z_dynamics_loss_val,
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
    from jax import random, jit
    import optax

    key = random.PRNGKey(0)
    input_dim = 128

    latent_dim = 3
    poly_order = 3
    include_sine = False
    include_constant = True

    lib_kwargs = {'n_states': latent_dim, 'poly_order': poly_order, 'include_sine': include_sine, 'include_constant': include_constant}

    lib_size = library_size(**lib_kwargs)

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

    # First-order dynamics test
    loss_fn_first_order = loss_fn_factory(autoencoder, weights=(1, 1, 40, 1), second_order=False, **lib_kwargs)
    loss_first_order, losses_first_order = loss_fn_first_order(state.params, (x, dx), state.mask)
    print("First-order loss:", loss_first_order)
    print("First-order loss components:", losses_first_order)

    # Jitted first-order dynamics test
    jitted_loss_fn_first_order = jit(loss_fn_first_order)
    loss_first_order_jit, losses_first_order_jit = jitted_loss_fn_first_order(state.params, (x, dx), state.mask)
    print("Jitted first-order loss:", loss_first_order_jit)
    print("Jitted first-order loss components:", losses_first_order_jit)

    key = random.PRNGKey(0)
    input_dim = 128

    latent_dim = 3
    poly_order = 3
    include_sine = False
    include_constant = True

    lib_kwargs = {'n_states': latent_dim*2, 'poly_order': poly_order, 'include_sine': include_sine, 'include_constant': include_constant}

    lib_size = library_size(**lib_kwargs)

    encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32])
    decoder = Decoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32])
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, widths=[32, 32], encoder=encoder, decoder=decoder, lib_size=lib_size)


    # Second-order dynamics test
    loss_fn_second_order = loss_fn_factory(autoencoder, weights=(1, 1, 40, 1), second_order=True, **lib_kwargs)
    loss_second_order, losses_second_order = loss_fn_second_order(state.params, (x, dx, ddx), state.mask)
    print("Second-order loss:", loss_second_order)
    print("Second-order loss components:", losses_second_order)

    # Jitted second-order dynamics test
    jitted_loss_fn_second_order = jit(loss_fn_second_order)
    loss_second_order_jit, losses_second_order_jit = jitted_loss_fn_second_order(state.params, (x, dx, ddx), state.mask)
    print("Jitted second-order loss:", loss_second_order_jit)
    print("Jitted second-order loss components:", losses_second_order_jit)
