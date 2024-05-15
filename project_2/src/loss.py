import jax.numpy as jnp
from jax import jacobian, grad
from sindy_utils import create_sindy_library, add_sine
from typing import Tuple
from jax import Array
from type_utils import ModelLayers, ModelParams
from flax import linen as nn
from jax import tree_map
from jax import vmap


def loss_recon(x: Array, x_hat: Array) -> Array:
    """
    Reconstruction loss

    Arguments:
        x {Array} -- Original data
        x_hat {Array} -- Reconstructed data

    Returns:
        Array -- Reconstruction loss
    """
    return jnp.mean(jnp.linalg.norm(x - x_hat, axis=1) ** 2)

# params in a dictionary of collections


def loss_dynamics_dx(params: ModelParams,
                     decoder: nn.module,
                     z: Array, dx_dt: Array,
                     theta: Array,
                     xi: Array,
                     mask: Array
                     ):
    """
    Loss for the dynamics in x

    Arguments:
        params {NestedModelLayers} -- Model parameters
        decoder {nn.module} -- Decoder
        z {Array} -- Latent space
        dx_dt {Array} -- Time derivative of x
        theta {Array} -- SINDy library
        xi {Array} -- SINDy coefficients
        mask {Array} -- Mask

    Returns
        Array -- Loss dynamics in x

    """

    def psi(params, z):
        return decoder.apply({"params": params}, z)

    jacobian_fn = vmap(jacobian(psi, argnums = 1), in_axes = (None, 0))

    dpsi_dt = jnp.einsum('ijk,ik->ij', jacobian_fn(params, z), theta @ (mask * xi))
    
    #print(f"We find the shape of the dpsidt function to be {dpsi_dt.shape}")
    #jacobian_phi = jacobian(phi, argnums=1)

    return jnp.mean(jnp.linalg.norm(dx_dt - dpsi_dt, axis=1)** 2)


def loss_dynamics_dz(params: ModelParams,
                     encoder: nn.module,
                     x: Array,
                     dx_dt: Array,
                     theta: Array,
                     xi: Array,
                     mask: Array
                     ):
    """
    Loss for the dynamics in z

    arguments:
        params {NestedModelLayers} -- Model parameters
        encoder {nn.module} -- Encoder
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
    
    jacobian_fn = vmap(jacobian(phi, argnums = 1), in_axes = (None, 0))

    dphi_dt = jnp.einsum('ijk,ik->ij', jacobian_fn(params, x), dx_dt)
    
    #print(f"We find the shape of the dphidt function to be {dphi_dt.shape}")
    #jacobian_phi = jacobian(phi, argnums=1)

    return jnp.mean(jnp.linalg.norm(dphi_dt -theta @ (mask * xi), axis=1)** 2)


def loss_regularization(xi: Array):
    """
    Regularization loss

    Arguments:
        xi {Array} -- SINDy coefficients

    Returns:
        Array -- L1 norm of input
    """
    return jnp.linalg.norm(xi, ord=1)


def create_loss_fn(latent_dim: int, poly_order: int, include_sine: bool = False, weights: tuple = (1, 1, 40, 1), batchsize: int = 128):
    """
    Create a loss function for different sindy libraries

    Args:
        lib_size (int): number of columns in sindy library (different functions)
        poly_order (int): polynomial order
        include_sine (bool, optional): Include sine functions in the library. Defaults to False.
        weights (tuple, optional): Weights for the loss functions. Defaults to (1, 1, 40, 1).

    Returns:
        Callable: Loss function
    """
    sindy_library = create_sindy_library(
        poly_order, include_sine, n_states=latent_dim)
    recon_weight, dx_weight, dz_weight, reg_weight = weights

    def loss_fn(params: ModelLayers,
                batch: Tuple,
                autoencoder: nn.module,
                mask: Array):
        """
        Total loss function

        args:
            params: Model parameters
            autoencoder: Autoencoder model
            batch: Tuple of features and target
            mask: Mask

        returns:
            Tuple: Total loss and dictionary of losses

        """
        features, target = batch

        encoder = autoencoder.encoder
        decoder = autoencoder.decoder

        z, x_hat = autoencoder.apply({"params": params}, features)
        #print(f"We have the shape of z to be {z.shape}")

        theta = sindy_library(z)
        #print(f"We have the shape of theta to be {theta.shape}")

        xi = params["sindy_coefficients"]

        encoder_params = params["encoder"]
        decoder_params = params["decoder"]

        loss_reconstruction = loss_recon(features, x_hat)
        #print(f"The reconstruction loss is {loss_reconstruction}, with type {type(loss_reconstruction)}")

        loss_dynamics_dx_part = loss_dynamics_dx(
            decoder_params, decoder, z, target, theta, xi, mask
        )
        loss_dynamics_dz_part = loss_dynamics_dz(
            encoder_params, encoder, features, target, theta, xi, mask
        )

        loss_reg = loss_regularization(xi)
        
        total_loss = (
            recon_weight * loss_reconstruction
            + dx_weight * loss_dynamics_dx_part
            + dz_weight * loss_dynamics_dz_part
            + reg_weight * loss_reg
        )
        return total_loss, {
            "loss": total_loss,
            "reconstruction": loss_reconstruction,
            "dynamics_dx": loss_dynamics_dx_part,
            "dynamics_dz": loss_dynamics_dz_part,
            "regularization": loss_reg,
        }

    return loss_fn


# %% [markdown]

# Reconstruction Loss (`Lrecon`)

# $$ L_{ \text{recon} } = \frac{1}{m} \sum_{i=1}^{m}  ||x_i - \psi(\phi(x_i))||^2_2  $$

# Dynamics in `x` Loss (`Ldx/dt`)
# $$ L_{dx/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \dot{x}_i - (\nabla_z \psi(\phi(x_i))) \Theta(\phi(x_i))^T \Xi
# \right\|^2_2 $$

# Dynamics in `z` Loss (`Ldz/dt`)
# $$ L_{dz/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \nabla_x \phi(x_i) \dot{x}_i - \Theta(\phi(x_i))^T \Xi
# \right\|^2_2 $$

# Regularization Loss (`Lreg`)
# $$ L_{\text{reg}} = \frac{1}{pd} \| \Xi \|_1 $$


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

    loss_fn = create_loss_fn(latent_dim,poly_order, include_sine=False, batchsize=10)


    loss, losses = loss_fn(
        state.params, (features, target), autoencoder, state.mask)
    print(loss)
    print(losses)
