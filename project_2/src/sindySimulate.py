from sindyLibrary import sindy_library_factory, polynomial_degrees
import jax.numpy as jnp
from jax import vmap
from scipy.integrate import solve_ivp

def sindy_simulate(x0, t, Xi, **library_kwargs):
    n = x0.size
    library_kwargs["n_states"] = n
    sindy_library = vmap(sindy_library_factory(**library_kwargs))
    
    def f(t, x):
        library_features = sindy_library(jnp.array(x).reshape((1, n)))
        return jnp.dot(library_features, Xi).reshape((n,))
    
    sol = solve_ivp(f, (t[0], t[-1]), x0, t_eval=t)
    
    return sol.y.T


def test():
    # Example usage and testing code
    from jax import random
    from jax import vmap

    key = random.PRNGKey(1)
    X = jnp.array([1, 2, 3, 4]).reshape(2, -1)
    print("Input Data:\n", X)

    my_function = vmap(sindy_library_factory(poly_order=2, include_sine=False, n_states=2))
    print("Polynomial Degrees:\n", polynomial_degrees(2, 2))
    print("SINDy Library Features:\n", my_function(X))

    # Testing sindy_simulate
    t = jnp.linspace(0, 10, 100)
    x0 = jnp.array([1.0, 2.0])
    Xi = jnp.array([[0.5, 0.0], [0.0, 0.5], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    print(f"Xi shape: {Xi.shape}")
    x_simulated = sindy_simulate(x0, t, Xi, poly_order=2, include_sine=False)
    print("SINDy Simulation Results:\n", x_simulated)

if __name__ == "__main__":
    test()