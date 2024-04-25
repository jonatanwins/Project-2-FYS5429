from flax import linen as nn



class Encoder(nn.Module):
    input_dim: int
    latent_dim: int
    widths: list
    activation: nn.activation = nn.relu
    initializer: nn.initializers = nn.initializers.glorot_normal()

    @nn.compact
    def __call__(self, x):
        for width in self.widths:
            x = nn.Dense(width, self.initializer)(x)
            x = self.activation(x)
        z = nn.Dense(self.latent_dim)(x)
        return z


class Decoder(nn.Module):
    input_dim: int
    latent_dim: int
    widths: list
    activation: nn.activation = nn.relu
    initializer: nn.initializers = nn.initializers.glorot_normal()

    @nn.compact
    def __call__(self, z):
        for width in reversed(self.widths):
            z = nn.Dense(width, self.initializer)(z)
            z = self.activation(z)
        x_decode = nn.Dense(self.input_dim)(z)
        return x_decode


class Autoencoder(nn.Module):
    input_dim: int
    latent_dim: int
    lib_size: int
    widths: list
    encoder: Encoder
    decoder: Decoder
    train: bool = True

    def setup(self):

        self.sindy_coefficients = self.param(
            "sindy_coefficients",
           nn.initializers.constant(1.0),
            (self.lib_size, self.latent_dim),
        )

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat




if __name__ == "__main__":

    from jax import random, tree_map
    import jax.numpy as jnp
    from flax.core.frozen_dict import freeze, FrozenDict

    #lets create an instance of the autoencoder and make sure it works
    key = random.PRNGKey(0)
    input_dim = 128
    latent_dim = 2
    lib_size = 3
    widths = [60, 40, 20]
    x = jnp.ones((1, input_dim))
    encoder = Encoder(input_dim, latent_dim, widths)
    decoder = Decoder(input_dim, latent_dim, widths)
    model = Autoencoder(input_dim, latent_dim, lib_size, widths, encoder, decoder)
    params = model.init(key, x)
    z, x_hat = model.apply(params, x)
    print(z.shape, x_hat.shape)
    #lets have a look at the params
    #print(type(params)) 
    print(tree_map(jnp.shape, params))
    print("=================================================")
    #print(tree_map(jnp.shape, params['params']))
    encoder_params = {'params': params['params']['encoder']}
    print(tree_map(jnp.shape, encoder_params))
    print(type(encoder_params))
    z = model.encoder.apply(encoder_params, x)
    print(z)

