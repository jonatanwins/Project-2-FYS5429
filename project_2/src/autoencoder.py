from flax import linen as nn
from flax.linen import initializers
from jax import Array

class Encoder(nn.Module):
    input_dim: int
    latent_dim: int
    widths: list
    activation: str = 'sigmoid'
    weight_initializer: str = 'xavier_uniform'
    bias_initializer: str = 'constant'

    def setup(self):
        self.activation_fn = getattr(nn, self.activation)
        self.weight_initializer_fn = getattr(initializers, self.weight_initializer)()
        self.bias_initializer_fn = getattr(initializers, self.bias_initializer)(0)

    @nn.compact
    def __call__(self, x: Array):
        for width in self.widths:
            x = nn.Dense(width, kernel_init=self.weight_initializer_fn, bias_init = self.bias_initializer_fn)(x)
            x = self.activation_fn(x)
        z = nn.Dense(self.latent_dim, kernel_init=self.weight_initializer_fn, bias_init = self.bias_initializer_fn)(x)
        return z


class Decoder(nn.Module):
    input_dim: int
    latent_dim: int
    widths: list
    activation: str = 'sigmoid'
    weight_initializer: str = 'xavier_uniform'
    bias_initializer: str = 'constant'

    def setup(self):
        self.activation_fn = getattr(nn, self.activation)
        self.weight_initializer_fn = getattr(initializers, self.weight_initializer)()
        self.bias_initializer_fn = getattr(initializers, self.bias_initializer)(0)

    @nn.compact
    def __call__(self, z):
        for width in reversed(self.widths):
            z = nn.Dense(width, kernel_init=self.weight_initializer_fn, bias_init = self.bias_initializer_fn)(z)
            z = self.activation_fn(z)
        x_decode = nn.Dense(self.input_dim, kernel_init=self.weight_initializer_fn, bias_init = self.bias_initializer_fn)(z)
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
    import numpy as np

    key = random.PRNGKey(0)
    input_dim = 128
    latent_dim = 2
    lib_size = 3
    widths = [60, 40, 20]
    x = jnp.ones((1, input_dim))
    encoder = Encoder(input_dim, latent_dim, widths)
    decoder = Decoder(input_dim, latent_dim, widths)
    model = Autoencoder(input_dim, latent_dim, lib_size, widths, encoder, decoder)

    # Split the key for consistent initialization
    key1, key2 = random.split(key)
    params = model.init(key1, x)
    z, x_hat = model.apply(params, x)
    print(z.shape, x_hat.shape)
    print(type(params))
    print(tree_map(jnp.shape, params))
    print("=================================================")
    encoder_params = {"params": params["params"]["encoder"]}
    print(tree_map(jnp.shape, encoder_params))
    print(type(encoder_params))
    z = model.encoder.apply(encoder_params, x)
    print(z)