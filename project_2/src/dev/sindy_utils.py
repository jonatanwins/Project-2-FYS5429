import jax.numpy as jnp
from jax import vmap
from itertools import product
from jax import Array
from scipy.special import binom

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
    degrees = degrees[(sums <= poly_order) & (sums > 1)][::-1]
    return degrees

def add_polynomial_features(features: Array, degrees: Array) -> Array:
    def polynomial(x, degree):
        return jnp.prod(x ** degree, axis=-1)
    
    def polynomial_features(X, degrees):
        all_polynomials = vmap(polynomial, in_axes=(0, None))
        return all_polynomials(X, degrees)
    
    return polynomial_features(features, degrees)

def add_sine(features: Array) -> Array:
    return jnp.sin(features)

def create_sindy_library(poly_order: int, include_sine: bool = False, add_constant: bool = True, n_states: int = 3) -> callable:
    degrees = polynomial_degrees(n_states, poly_order)
    print("Polynomial degrees:", degrees)  # Print polynomial degrees for verification

    def sindy_library(features: Array) -> Array:
        library = features
        if add_constant:
            ones = jnp.ones((features.shape[0], 1))
            library = jnp.concatenate([ones, library], axis=1)
        polynomial_library = add_polynomial_features(features, degrees)
        library = jnp.concatenate([library, polynomial_library], axis=1)
        if include_sine:
            sine_features = add_sine(features)
            library = jnp.concatenate([library, sine_features], axis=1)
        return library
    
    return sindy_library

def test_sindy_library():
    # Define test configurations
    test_cases = [
        {"poly_order": 2, "n_states": 3, "include_sine": False, "add_constant": True},
        {"poly_order": 2, "n_states": 2, "include_sine": True, "add_constant": True},
        {"poly_order": 3, "n_states": 4, "include_sine": False, "add_constant": False},
    ]

    for case in test_cases:
        n_states = case["n_states"]
        
        # Generate test data with the correct shape
        test_features = jnp.array([jnp.arange(1, n_states + 1, dtype=float) + i for i in range(2)])
        larger_test_features = jnp.array([jnp.arange(1, n_states + 1, dtype=float) + i for i in range(3)])

        print(f"Testing with config: {case}")
        sindy_lib = create_sindy_library(case["poly_order"], case["include_sine"], case["add_constant"], case["n_states"])
        library = sindy_lib(test_features)
        larger_library = sindy_lib(larger_test_features)
        print("Test features:")    
        print(test_features)
        print("Library:")
        print(library)
        print("Larger test features:")    
        print(larger_test_features)
        print("Larger Library:")
        print(larger_library)

if __name__ == "__main__":
    test_sindy_library()
