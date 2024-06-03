import jax.numpy as jnp
from jax import vmap
from scipy.special import binom
from itertools import product
from jax import Array
from jax import jit

def library_size(poly_order: int, include_sine: bool = False, include_basic_fractions: bool = False, include_intermediate_fractions: bool = False, include_three_body: bool = False, n_states: int = 3, include_constant: bool = True) -> int:
    """
    Calculate the size of the library based on the number of states, polynomial order, and various options for additional features.
    
    Parameters:
    poly_order (int): Maximum order of polynomials.
    include_sine (bool, optional): If True, includes sine functions in the library. Defaults to False.
    include_basic_fractions (bool, optional): If True, includes basic fractions in the library. Defaults to False.
    include_intermediate_fractions (bool, optional): If True, includes intermediate fractions in the library. Defaults to False.
    include_three_body (bool, optional): If True, includes three-body fractions in the library. Defaults to False.
    n_states (int, optional): Number of states. Defaults to 3.
    include_constant (bool, optional): If True, includes a constant term in the library. Defaults to True.
    
    Returns:
    int: Size of the library.
    """
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n_states + k - 1, k))
    if include_sine:
        l += n_states
    if include_basic_fractions:
        l += n_states ** 2
    if include_intermediate_fractions:
        l += 3 * n_states ** 2
    if include_three_body and n_states == 3:
        l += n_states * (n_states - 1) * (n_states - 2) // 6
    if not include_constant:
        l -= 1
    return l

def polynomial_degrees(n_states: int, poly_order: int) -> Array:
    """
    Generate all combinations of polynomial degrees for the given number of states and polynomial order.
    
    Parameters:
    n_states (int): Number of states.
    poly_order (int): Maximum order of polynomials.
    
    Returns:
    Array: Array of polynomial degrees.
    """
    degrees = jnp.array(list(product(range(poly_order + 1), repeat=n_states)))
    sums = jnp.sum(degrees, axis=1)
    degrees = degrees[(sums <= poly_order) & (sums > 1)][::-1]
    return degrees

def polynomial(x, degree):
    return jnp.prod(x ** degree)

def polynomial_features(X, degrees):
    all_polynomials = vmap(polynomial, in_axes=(None, 0))
    all_features = vmap(all_polynomials, in_axes=(0, None))
    return all_features(X, degrees)

def add_polynomials(features: Array, library: Array, degrees) -> Array:
    """
    Add polynomial features to the SINDy library.
    
    Parameters:
    features (Array): Input features.
    library (Array): Existing SINDy library.
    
    Returns:
    Array: Updated SINDy library with polynomial features.
    """
    polynomial_library = polynomial_features(features, degrees)
    library = jnp.concatenate([library, polynomial_library], axis=1)
    return library

def add_sine(features: Array, library: Array) -> Array:
    """
    Add sine functions to the SINDy library.
    
    Parameters:
    features (Array): Input features.
    library (Array): Existing SINDy library.
    
    Returns:
    Array: Updated SINDy library with sine functions.
    """
    sine = jnp.sin(features)
    library = jnp.concatenate([library, sine], axis=1)
    return library

def add_basic_fractions(features: Array, library: Array) -> Array:
    """
    Add basic fractions to the SINDy library.

    Parameters:
    features (Array): Input features.
    library (Array): Existing SINDy library.

    Returns:
    Array: Updated SINDy library with basic fractions.
    """
    def single_sample_fractions(sample: Array):
        denominators = jnp.repeat(sample, len(sample))
        numerators = jnp.tile(sample, len(sample))

        def divide(n, d):
            return n / d

        vectorized_divide = vmap(divide)
        fractions = vectorized_divide(numerators, denominators)
        return fractions

    basic_fractions = vmap(single_sample_fractions)(features)
    return jnp.concatenate([library, basic_fractions], axis=1)

def add_intermediate_fractions(features: Array, library: Array) -> Array:
    """
    Add intermediate fractions to the SINDy library.
    
    Parameters:
    features (Array): Input features.
    library (Array): Existing SINDy library.
    
    Returns:
    Array: Updated SINDy library with intermediate fractions.
    """
    def single_sample_intermediate_fractions(sample: Array):
        n = len(sample)
        fractions = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    if k != i and k != j:
                        fractions.append((sample[i] - sample[j]) / sample[k])
                        fractions.append((sample[j] - sample[i]) / sample[k])
                        fractions.append(sample[i] * sample[j] / sample[k])
        return jnp.array(fractions)

    intermediate_fractions = vmap(single_sample_intermediate_fractions)(features)
    return jnp.concatenate([library, intermediate_fractions], axis=1)

def add_three_body_fractions(features: Array, library: Array) -> Array:
    """
    Add three-body fractions to the SINDy library.
    
    Parameters:
    features (Array): Input features (assumed to be positions).
    library (Array): Existing SINDy library.
    
    Returns:
    Array: Updated SINDy library with three-body fractions.
    """
    def single_sample_three_body_fractions(sample: Array):
        r12 = sample[0] - sample[1]
        r13 = sample[0] - sample[2]
        r23 = sample[1] - sample[2]

        norm_r12 = jnp.linalg.norm(r12)
        norm_r13 = jnp.linalg.norm(r13)
        norm_r23 = jnp.linalg.norm(r23)

        three_body_fractions = jnp.concatenate([
            - r12 / norm_r12 ** 3 - r13 / norm_r13 ** 3,
            - r23 / norm_r23 ** 3 - r12 / norm_r12 ** 3,
            - r13 / norm_r13 ** 3 - r23 / norm_r23 ** 3
        ])
        return three_body_fractions

    three_body_fractions = vmap(single_sample_three_body_fractions)(features)
    return jnp.concatenate([library, three_body_fractions], axis=1)

def sindy_library_factory(poly_order: int, include_sine: bool = False, include_basic_fractions: bool = False, include_intermediate_fractions: bool = False, include_three_body: bool = False, n_states: int = 3, include_constant: bool = True) -> jnp.ndarray:
    """
    Create a SINDy (Sparse Identification of Nonlinear Dynamics) library for the given polynomial order and states.
    
    Parameters:
    poly_order (int): Maximum order of polynomials.
    include_sine (bool, optional): If True, includes sine functions in the library. Defaults to False.
    include_basic_fractions (bool, optional): If True, includes basic fractions in the library. Defaults to False.
    include_intermediate_fractions (bool, optional): If True, includes intermediate fractions in the library. Defaults to False.
    include_three_body (bool, optional): If True, includes three-body fractions in the library. Defaults to False.
    n_states (int, optional): Number of states. Defaults to 3.
    include_constant (bool, optional): If True, includes a constant term in the library. Defaults to True.
    
    Returns:
    function: Function to generate the SINDy library.
    """
    degrees = polynomial_degrees(n_states, poly_order)
    if n_states == 1: 
        degrees = degrees[::-1] # Reverse the order of the polynomial degrees for 1D case, more natural ordering

    def sindy_library(features: Array) -> Array:
        library = features
        if poly_order > 1:
            library = add_polynomials(features, library, degrees)
        if include_constant:
            ones = jnp.ones((features.shape[0], 1))
            library = jnp.concatenate([ones, library], axis=1)
        if include_sine:
            library = add_sine(features, library)
        if include_basic_fractions:
            library = add_basic_fractions(features, library)
        if include_intermediate_fractions:
            library = add_intermediate_fractions(features, library)
        if include_three_body and n_states == 3:
            library = add_three_body_fractions(features, library)
        return library

    return sindy_library

def test_sindy_library():
    # Define test configurations
    test_cases = [
        {"poly_order": 2, "n_states": 3, "include_sine": False, "include_basic_fractions": True, "include_intermediate_fractions": False, "include_three_body": False, "include_constant": True},
        {"poly_order": 2, "n_states": 2, "include_sine": True, "include_basic_fractions": False, "include_intermediate_fractions": True, "include_three_body": False, "include_constant": True},
        {"poly_order": 3, "n_states": 4, "include_sine": False, "include_basic_fractions": False, "include_intermediate_fractions": False, "include_three_body": True, "include_constant": False},
        {"poly_order": 2, "n_states": 1, "include_sine": False, "include_basic_fractions": False, "include_intermediate_fractions": False, "include_three_body": False, "include_constant": True},  # Basic test case without any fractions
    ]
    #test_cases = [test_cases[0]]  # Select a single test case for now

    for case in test_cases:
        n_states = case["n_states"]
        
        # Generate test data with the correct shape
        test_features = jnp.array([jnp.arange(1, n_states + 1, dtype=float) + i for i in range(2)])
        larger_test_features = jnp.array([jnp.arange(1, n_states + 1, dtype=float) + i for i in range(3)])

        print(f"Testing with config: {case}")
        sindy_lib = sindy_library_factory(case["poly_order"], case["include_sine"], case["include_basic_fractions"], case["include_intermediate_fractions"], case["include_three_body"], case["n_states"], case["include_constant"])
        sindy_lib = jit(sindy_lib)
        library = sindy_lib(test_features)
        larger_library = sindy_lib(larger_test_features)
        # print("Test features:")    
        # print(test_features)
        # print("Library:")
        # print(library)
        # print("Larger test features:")    
        # print(larger_test_features)
        # print("Larger Library:")
        #print(larger_library)
        lib_size = library.shape[1]
        lib_size_larger = larger_library.shape[1]
        lib_size_func = library_size(case["poly_order"], case["include_sine"], case["include_basic_fractions"], case["include_intermediate_fractions"], case["include_three_body"], case["n_states"], case["include_constant"])

        #assert library size is equal to second dimension of library
        print("Library size from function: ", lib_size_func)
        print("Library size Larger input : ", lib_size_larger)
        print("Library size test input: ", lib_size)
        #assert lib_size == lib_size_func == lib_size_larger

if __name__ == "__main__":
    test_sindy_library()
