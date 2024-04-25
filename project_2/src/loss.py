import jax.numpy as jnp
from jax import grad
from sindy_utils import library_size, sindy_library, add_sine


def loss_recon(x, x_hat):
    """
    Reconstruction loss
    """
    return jnp.mean(jnp.linalg.norm(x - x_hat, axis=1)**2)


def loss_dynamics_dx(params, decoder, z, dx_dt, theta, xi, mask):
    """
    Loss for the dynamics in x
    """

    def psi(z, params): return decoder.apply(params, z)
    grad_psi = grad(psi, argnums=1)

    return jnp.mean(jnp.linalg.norm(dx_dt - jnp.dot(grad_psi(params, z), theta @ mask*xi ), axis=1)**2)


def loss_dynamics_dz(params, encoder, x, dx_dt, theta, xi, mask):
    """
    Loss for the dynamics in z
    """

    def phi(x, params): return encoder.apply(params, x)
    grad_phi = grad(phi, argnums=1)

    return jnp.mean(jnp.linalg.norm(jnp.dot(grad_phi(params, x), dx_dt) - theta @ mask*xi, axis=1)**2)


def loss_regularization(xi):
    """
    Regularization loss
    """
    return jnp.linalg.norm(xi, ord=1)



def create_loss_fn(lib_size, poly_order: int, include_sine: bool = False):
    #need to be able to have different libraries for different models
    
    if include_sine:
        sindy_library = lambda features: add_sine(features, sindy_library(features, poly_order, lib_size))
    else:
        sindy_library = lambda features: sindy_library(features, poly_order, lib_size)
        
    def loss_fn(model, state, batch):
        """
        Total loss function
        """
        features, target = batch 
        
        encoder = model.encoder
        decoder = model.decoder
        
        z, x_hat = model.apply(state.params, features)
        
        theta = sindy_library(z)
        
        xi = state.params['sindy_coefficients']
        mask = state.mask

        encoder_params = {'params': state.params['encoder']}
        decoder_params = {'params': state.params['decoder']}

        loss_reconstruction = loss_recon(features, x_hat)

        loss_dynamics_dx_part = loss_dynamics_dx(decoder_params, decoder, features, target, theta, xi, mask)
        loss_dynamics_dz_part = loss_dynamics_dz(encoder_params, encoder, features, target, theta, xi, mask)
        loss_reg = loss_regularization(xi)

        total_loss = loss_reconstruction + loss_dynamics_dx_part + loss_dynamics_dz_part + loss_reg
        return total_loss, {'loss': total_loss, 'reconstruction': loss_reconstruction, 'dynamics_dx': loss_dynamics_dx_part, 'dynamics_dz': loss_dynamics_dz_part, 'regularization': loss_reg}                                                                                                              
    
    return loss_fn

# %% [markdown]

# Reconstruction Loss (`Lrecon`)

# $$ L_{ \text{recon} } = \frac{1}{m} \sum_{i=1}^{m}  ||x_i - \psi(\phi(x_i))||^2_2  $$

# Dynamics in `x` Loss (`Ldx/dt`)
# $$ L_{dx/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \dot{x}_i - (\nabla_z \psi(\phi(x_i))) \Theta(\phi(x_i))^T \Xi \right\|^2_2 $$

# Dynamics in `z` Loss (`Ldz/dt`)
# $$ L_{dz/dt} = \frac{1}{m} \sum_{i=1}^{m} \left\| \nabla_x \phi(x_i) \dot{x}_i - \Theta(\phi(x_i))^T \Xi \right\|^2_2 $$

# Regularization Loss (`Lreg`)
# $$ L_{\text{reg}} = \frac{1}{pd} \| \Xi \|_1 $$
