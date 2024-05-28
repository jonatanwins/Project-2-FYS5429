"""
jax translation of the lorenz utils. Currently in 32bit precision.
"""

import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.ode import odeint
import numpy as np


def legendre_polynomials(n, x):
    """Generate the first n Legendre polynomials at x."""
    polys = [jnp.ones_like(x), x]
    for i in range(2, n):
        p_i = ((2 * i - 1) * x * polys[i - 1] - (i - 1) * polys[i - 2]) / i
        polys.append(p_i)
    return jnp.array(polys[:n])


def lorenz_ode(z, t, sigma=10, beta=8/3, rho=28):
    x, y, z = z
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return jnp.array([dx, dy, dz])


@jit
def simulate_lorenz_odeint(z0, t, sigma=10.0, beta=8 / 3, rho=28.0):
    z = odeint(lorenz_ode, z0, t, sigma, beta, rho)
    dz = vmap(lorenz_ode, in_axes=(0, None, None, None, None))(
        z, t, sigma, beta, rho)
    return z, dz


def generate_modes(n_points):
    y_spatial = jnp.linspace(-1, 1, n_points)
    # Generate first 6 Legendre polynomials
    modes = legendre_polynomials(6, y_spatial)
    return modes, y_spatial


def generate_lorenz_data_linear(ic, t, modes, normalization, sigma=10, beta=8 / 3, rho=28):
    z, dz = simulate_lorenz_odeint(
        ic, t, sigma=sigma, beta=beta, rho=rho)
    z *= normalization
    dz *= normalization

    # Only use the first three modes for the linear case
    modes_linear = modes[:3]
    x = jnp.einsum('ij,jk->ik', modes_linear.T, z.T)
    dx = jnp.einsum('ij,jk->ik', modes_linear.T, dz.T)

    return x.T, dx.T, z, dz


def generate_lorenz_data_nonlinear(ic, t, modes, normalization, sigma=10, beta=8 / 3, rho=28):
    z, dz = simulate_lorenz_odeint(
        ic, t, sigma=sigma, beta=beta, rho=rho)
    z *= normalization
    dz *= normalization

    modes_linear = modes[:3]
    modes_cubic = modes[3:]

    x = jnp.einsum('ij,jk->ik', modes_linear.T, z.T) + \
        jnp.einsum('ij,jk->ik', modes_cubic.T, (z.T)**3)
    dx = jnp.einsum('ij,jk->ik', modes_linear.T, dz.T) + \
        jnp.einsum('ij,jk,jk->ik', modes_cubic.T, 3 * (z.T)**2, dz.T)

    return x.T, dx.T, z, dz


def get_lorenz_test_data(n_ics, noise_strength=0, linear=True):
    t = jnp.arange(0, 5, 0.02)
    input_dim = 128

    ic_means = jnp.array([0, 0, 25])
    ic_widths = 2 * jnp.array([36, 48, 41])
    ics = ic_widths * (np.random.rand(n_ics, 3) - 0.5) + ic_means

    modes, y_spatial = generate_modes(input_dim)

    normalization = jnp.array([1 / 40, 1 / 40, 1 / 40])
    if linear:
        data_per_ic = vmap(lambda ic: generate_lorenz_data_linear(
            ic, t, modes, normalization, 10, 8 / 3, 28))(ics)
    else:
        data_per_ic = vmap(lambda ic: generate_lorenz_data_nonlinear(
            ic, t, modes, normalization, 10, 8 / 3, 28))(ics)

    x, dx, z, dz = data_per_ic

    x = x.reshape((-1, input_dim)) + noise_strength * \
        np.random.randn(n_ics * t.size, input_dim)
    dx = dx.reshape((-1, input_dim)) + noise_strength * \
        np.random.randn(n_ics * t.size, input_dim)

    return {'t': t, 'y_spatial': y_spatial, 'modes': modes, 'x': x, 'dx': dx, 'z': z, 'dz': dz}


def get_lorenz_train_data(n_ics, noise_strength=0, linear=True):
    t = jnp.arange(0, 5, 0.02)
    input_dim = 128

    ic_means = jnp.array([0, 0, 25])
    ic_widths = 2 * jnp.array([36, 48, 41])
    ics = ic_widths * (np.random.rand(n_ics, 3) - 0.5) + ic_means

    modes, y_spatial = generate_modes(input_dim)

    normalization = jnp.array([1 / 40, 1 / 40, 1 / 40])
    if linear:
        data_per_ic = vmap(lambda ic: generate_lorenz_data_linear(
            ic, t, modes, normalization, 10, 8 / 3, 28))(ics)
    else:
        data_per_ic = vmap(lambda ic: generate_lorenz_data_nonlinear(
            ic, t, modes, normalization, 10, 8 / 3, 28))(ics)

    x, dx, z, dz = data_per_ic

    x = x.reshape((-1, input_dim)) + noise_strength * \
        np.random.randn(n_ics * t.size, input_dim)
    dx = dx.reshape((-1, input_dim)) + noise_strength * \
        np.random.randn(n_ics * t.size, input_dim)

    return {'x': x, 'dx': dx}


if __name__ == "__main__":
    data = get_lorenz_test_data(1, linear=False)
    print(f't: {data["t"].shape}, y_spatial: {data["y_spatial"].shape}, modes: {data["modes"].shape}, x: {data["x"].shape}, dx: {data["dx"].shape}, z: {data["z"].shape}, dz: {data["dz"].shape}')
