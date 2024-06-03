import warnings
from sindyLibrary import create_sindy_library, polynomial_degrees
import jax.numpy as jnp
from scipy.integrate import solve_ivp

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