import jax.numpy as jnp
from jax import vmap
from scipy.special import binom
from scipy.integrate import odeint, ODEintWarning
from scipy.integrate import solve_ivp
from itertools import product
from jax import Array
import warnings

def library_size(n: int, poly_order: int, use_sine: bool = False, include_constant=True) -> int:
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l

def polynomial_degrees(n_states: int, poly_order: int) -> Array:
    degrees = jnp.array(list(product(range(poly_order + 1), repeat=n_states)))
    sums = jnp.sum(degrees, axis=1)
    degrees = degrees[sums <= poly_order]
    return degrees

def create_sindy_library(poly_order: int, include_sine: bool = False, n_states: int = 3, second_order:bool=False) -> jnp.ndarray:

    def polynomial(x, degree):
        return jnp.prod(x ** degree)
    
    def polynomial_features(X, degrees):
        all_polynomials = vmap(polynomial, in_axes=(None, 0))
        all_features = vmap(all_polynomials, in_axes=(0, None))
        return all_features(X, degrees)
    
    degrees = polynomial_degrees(n_states, poly_order)

    def sindy_library(features: Array) -> Array:
        return polynomial_features(features, degrees)
    
    if include_sine == True:
        return lambda features: add_sine(features, sindy_library(features))
    
    return sindy_library

def add_sine(features: Array, library: Array) -> Array:
    sine = jnp.sin(features)
    library = jnp.concatenate([library, sine], axis=1)
    return library

def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    n = x0.size
    sindy_library = create_sindy_library(poly_order, include_sine, n)
    
    def f(t, x):
        library_features = sindy_library(jnp.array(x).reshape((1, n)))
        return jnp.dot(library_features, Xi).reshape((n,))
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        sol = solve_ivp(f, (t[0], t[-1]), x0, t_eval=t)
    
    return sol.y.T

def sindy_simulate_order2(x0, dx0, t, Xi, poly_order, include_sine):
    n = 2 * x0.size
    l = Xi.shape[0]
    
    Xi_order1 = jnp.zeros((l, n))
    for i in range(n // 2):
        Xi_order1 = Xi_order1.at[2 * (i + 1), i].set(1.0)
        Xi_order1 = Xi_order1.at[:, i + n // 2].set(Xi[:, i])
    
    initial_condition = jnp.concatenate((x0, dx0))
    x = sindy_simulate(initial_condition, t, Xi_order1, poly_order, include_sine)
    return x

# Example usage and testing code
if __name__ == "__main__":
    from jax import random

    key = random.PRNGKey(1)
    X = jnp.array([1, 2, 3, 4]).reshape(2, -1)
    print("Input Data:\n", X)

    my_function = create_sindy_library(poly_order=2, include_sine=False, n_states=2)
    print("Polynomial Degrees:\n", polynomial_degrees(2, 2))
    print("SINDy Library Features:\n", my_function(X))

    # Testing sindy_simulate
    t = jnp.linspace(0, 10, 100)
    x0 = jnp.array([1.0, 2.0])
    Xi = jnp.array([[0.5, 0.0], [0.0, 0.5], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    print(f"Xi shape: {Xi.shape}")
    x_simulated = sindy_simulate(x0, t, Xi, poly_order=2, include_sine=False)
    print("SINDy Simulation Results:\n", x_simulated)

    # Testing sindy_simulate_order2
    dx0 = jnp.array([0.0, 0.0])
    x_simulated_order2 = sindy_simulate_order2(x0, dx0, t, Xi, poly_order=2, include_sine=False)
    print("SINDy Second-Order Simulation Results:\n", x_simulated_order2)