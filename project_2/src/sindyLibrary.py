import jax.numpy as jnp
from jax import vmap
from scipy.special import binom
from itertools import product
from jax import Array

def library_size(n_states: int, poly_order: int, include_sine: bool = False, include_constant: bool = True) -> int:
    """
    Calculate the size of the library based on the given parameters.

    Args:
        poly_order (int): The highest order of polynomials to include.
        include_sine (bool): Whether to include sine terms.
        n_states (int): The number of state variables.
        include_constant (bool): Whether to include a constant term.

    Returns:
        int: The total number of terms in the library.
    """
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n_states + k - 1, k))
    if include_sine:
        l += n_states
    if not include_constant:
        l -= 1
    return l

def polynomial_degrees(n_states: int, poly_order: int) -> Array:
    """
    Generate all possible polynomial degrees up to a given order.

    Args:
        n_states (int): The number of state variables.
        poly_order (int): The highest order of polynomials to include.

    Returns:
        Array: An array of polynomial degrees.
    """
    degrees = jnp.array(list(product(range(poly_order + 1), repeat=n_states)))
    sums = jnp.sum(degrees, axis=1)
    degrees = degrees[(sums <= poly_order) & (sums > 1)][::-1]
    return degrees

def polynomial(x: Array, degree: Array) -> Array:
    """
    Compute the polynomial for a given degree.

    Args:
        x (Array): The input array.
        degree (Array): The degree array.

    Returns:
        Array: The computed polynomial value.
    """
    return jnp.prod(x ** degree)

def polynomial_features(x: Array, degrees: Array) -> Array:
    """
    Compute polynomial features for a given set of degrees.

    Args:
        x (Array): The input array.
        degrees (Array): The degrees array.

    Returns:
        Array: The polynomial features.
    """
    return vmap(lambda degree: polynomial(x, degree))(degrees)

def add_polynomials(x: Array, library: Array, degrees: Array) -> Array:
    """
    Add polynomial terms to the library.

    Args:
        x (Array): The input array.
        library (Array): The existing library.
        degrees (Array): The degrees array.

    Returns:
        Array: The updated library with polynomial terms.
    """
    polynomial_library = polynomial_features(x, degrees)
    return jnp.concatenate([library, polynomial_library], axis=0)

def add_sine(x: Array, library: Array) -> Array:
    """
    Add sine terms to the library.

    Args:
        x (Array): The input array.
        library (Array): The existing library.

    Returns:
        Array: The updated library with sine terms.
    """
    sine = jnp.sin(x)
    return jnp.concatenate([library, sine], axis=0)

def polynomial_transform_factory(poly_order: int, n_states: int) -> callable:
    """
    Factory function to create a polynomial transform function based on the provided parameters.

    Args:
        poly_order (int): The highest order of polynomials to include.
        n_states (int): The number of state variables.

    Returns:
        callable: A function that adds polynomial terms to the library.
    """
    degrees = polynomial_degrees(n_states, poly_order)
    if n_states == 1:
        degrees = degrees[::-1]
    return lambda x, library: add_polynomials(x, library, degrees)

def sine_transform_factory() -> callable:
    """
    Factory function to create a sine transform function.

    Returns:
        callable: A function that adds sine terms to the library.
    """
    return lambda x, library: add_sine(x, library)

def sindy_library_factory(poly_order: int = 1, n_states: int = 1, include_sine: bool = False, include_constant: bool = True) -> callable:
    """
    Factory function to create a SINDy library function based on the provided parameters.

    Args:
        poly_order (int): The highest order of polynomials to include.
        n_states (int): The number of state variables.
        include_sine (bool): Whether to include sine terms.
        include_constant (bool): Whether to include a constant term.

    Returns:
        callable: A function that generates the SINDy library for given input.
    """
    polynomial_transform = polynomial_transform_factory(poly_order, n_states)
    sine_transform = sine_transform_factory()
    constant_transform = lambda x, library: jnp.concatenate([jnp.ones((1,)), library], axis=0)

    def sindy_library(x: jnp.ndarray) -> jnp.ndarray:
        """
        Generate the SINDy library for given input x.

        Args:
            x (jnp.ndarray): The input array.

        Returns:
            jnp.ndarray: The SINDy library.
        """
        library = x
        if poly_order > 1:
            library = polynomial_transform(x, library)
        if include_constant:
            library = constant_transform(x, library)
        if include_sine:
            library = sine_transform(x, library)
        return library

    return sindy_library

def test_sindy_library() -> None:
    """
    Test the SINDy library with various configurations to ensure correctness.
    """
    test_cases = [
        {"poly_order": 2, "n_states": 3, "include_sine": False, "include_constant": True},
        {"poly_order": 2, "n_states": 2, "include_sine": True, "include_constant": True},
        {"poly_order": 3, "n_states": 4, "include_sine": False, "include_constant": False},
        {"poly_order": 2, "n_states": 1, "include_sine": False, "include_constant": True},
        {"poly_order": 1, "n_states": 1, "include_sine": False, "include_constant": False}
    ]
    #test_cases = [test_cases[1]]

    for case in test_cases:
        n_states = case["n_states"]

        test_features = jnp.array([jnp.arange(1, n_states + 1, dtype=float) + i for i in range(2)])
        larger_test_features = jnp.array([jnp.arange(1, n_states + 1, dtype=float) + i for i in range(3)])


        print(f"Testing with config: {case}")
        sindy_lib = sindy_library_factory(case["poly_order"], n_states, case["include_sine"], case["include_constant"])
        sindy_lib = vmap(sindy_lib)
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

        lib_size = library.shape[1]
        lib_size_larger = larger_library.shape[1]
        lib_size_func = library_size(case["poly_order"], case["include_sine"], case["n_states"], case["include_constant"])

        print("Library size from function: ", lib_size_func)
        print("Library size Larger input : ", lib_size_larger)
        print("Library size test input: ", lib_size)

if __name__ == "__main__":
    test_sindy_library()
