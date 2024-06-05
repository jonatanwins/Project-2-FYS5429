import jax.numpy as jnp
from jax import vmap
from scipy.special import binom
from itertools import product
from jax import Array
from jax import jit

def library_size(poly_order: int, include_sine: bool = False, include_basic_fractions: bool = False, include_intermediate_fractions: bool = False, include_three_body_fractions: bool = False, n_states: int = 3, include_constant: bool = True) -> int:
    """
    Calculate the size of the library based on the given parameters.

    Args:
        poly_order (int): The highest order of polynomials to include.
        include_sine (bool): Whether to include sine terms.
        include_basic_fractions (bool): Whether to include basic fraction terms.
        include_intermediate_fractions (bool): Whether to include intermediate fraction terms.
        include_three_body_fractions (bool): Whether to include three-body fraction terms (only for n_states=3).
        n_states (int): The number of state variables.
        include_constant (bool): Whether to include a constant term.

    Returns:
        int: The total number of terms in the library.
    """
    l = 0
    # Calculate the number of polynomial terms
    for k in range(poly_order + 1):
        l += int(binom(n_states + k - 1, k))
    # Add terms for sine functions if included
    if include_sine:
        l += n_states
    # Add terms for basic fractions if included
    if include_basic_fractions:
        l += n_states * (n_states - 1)
    # Add terms for intermediate fractions if included
    if include_intermediate_fractions:
        l += 3 * n_states ** 2
    # Add terms for three-body fractions if included
    if include_three_body_fractions:
        l += 3*2 #this is very wrong, as we essentialy need to onlu work with r12, r13, r23 as features for the 3body problem. Mega issue, might KMS
    # Remove the constant term if not included
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
    # Filter the degrees to include only those with sum <= poly_order and > 1
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
    library = jnp.concatenate([library, polynomial_library], axis=0)
    return library

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
    library = jnp.concatenate([library, sine], axis=0)
    return library

def add_basic_fractions(x: jnp.ndarray, library: jnp.ndarray, numerator_indices: jnp.ndarray, denominator_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Add basic fraction terms to the library.

    Args:
        x (jnp.ndarray): The input array.
        library (jnp.ndarray): The existing library.
        numerator_indices (jnp.ndarray): Precomputed indices for numerators.
        denominator_indices (jnp.ndarray): Precomputed indices for denominators.

    Returns:
        jnp.ndarray: The updated library with basic fraction terms.
    """
    numerators = x[numerator_indices]
    denominators = x[denominator_indices]

    # Calculate the fractions
    fractions = numerators / denominators

    return jnp.concatenate([library, fractions], axis=0)
  

def add_intermediate_fractions(x: jnp.ndarray, library: jnp.ndarray, diag_mask: jnp.ndarray, triu_indices: tuple, num_features:int, num_x_repeat:int, num_triag_elements:int) -> jnp.ndarray:
    """
    Add intermediate fraction terms to the library.

    Args:
        x (jnp.ndarray): The input array.
        library (jnp.ndarray): The existing library.
        diag_mask (jnp.ndarray): Precomputed diagonal mask.
        triu_indices (tuple): Precomputed upper triangular indices.

    Returns:
        jnp.ndarray: The updated library with intermediate fraction terms.
    """

    sample_col = x[:, jnp.newaxis]
    sample_row = x[jnp.newaxis, :]

    differences_matrix = sample_col - sample_row

    differences = differences_matrix[diag_mask]

    # Ensure correct shapes for broadcasting
    differences_tiled = jnp.tile(differences, num_features)
    x_repeat = jnp.repeat(x, num_x_repeat)

    fractions_1 = jnp.divide(differences_tiled, x_repeat)
    fractions_2 = jnp.divide(x_repeat, differences_tiled)

    product_matrix = sample_col * sample_row
    upper_tri_products = product_matrix[triu_indices].flatten()

    # Ensure correct shapes for broadcasting
    upper_tri_products_tiled = jnp.tile(upper_tri_products, num_features)
    sample_repeat = jnp.repeat(x, num_triag_elements)

    fractions_3 = jnp.divide(upper_tri_products_tiled, sample_repeat)
    fractions_4 = jnp.divide(sample_repeat, upper_tri_products_tiled)

    fractions = jnp.concatenate([fractions_1, fractions_2, fractions_3, fractions_4])
    return jnp.concatenate([library, fractions], axis=0)

def add_three_body_fractions(x: Array, library: Array) -> Array:
    """
    Add three-body fraction terms to the library for the case of n_states=3.

    Args:
        x (Array): The input array.
        library (Array): The existing library.

    Returns:
        Array: The updated library with three-body fraction terms.
    """
    r12 = x[0] - x[1]
    r13 = x[0] - x[2]
    r23 = x[1] - x[2]

    norm_r12 = jnp.linalg.norm(r12)
    norm_r13 = jnp.linalg.norm(r13)
    norm_r23 = jnp.linalg.norm(r23)

    three_body_fractions = jnp.concatenate([
        - r12 / norm_r12 ** 3 - r13 / norm_r13 ** 3,
        - r23 / norm_r23 ** 3 - r12 / norm_r12 ** 3,
        - r13 / norm_r13 ** 3 - r23 / norm_r23 ** 3
    ])
    return jnp.concatenate([library, three_body_fractions], axis=0)

def sindy_library_factory(poly_order: int = 1, n_states: int = 1, include_sine: bool = False, include_basic_fractions: bool = False, include_intermediate_fractions: bool = False, include_three_body_fractions: bool = False, include_constant: bool = True) -> jnp.ndarray:
    """
    Factory function to create a SINDy library function based on the provided parameters.

    Args:
        poly_order (int): The highest order of polynomials to include.
        n_states (int): The number of state variables.
        include_sine (bool): Whether to include sine terms.
        include_basic_fractions (bool): Whether to include basic fraction terms.
        include_intermediate_fractions (bool): Whether to include intermediate fraction terms.
        include_three_body_fractions (bool): Whether to include three-body fraction terms (only for n_states=3).
        include_constant (bool): Whether to include a constant term.

    Returns:
        jnp.ndarray: A function that generates the SINDy library for given input.
    """
    if n_states != 3 and include_three_body_fractions:
        raise ValueError("Three-body fractions are only supported for n_states=3.")
    
    degrees = polynomial_degrees(n_states, poly_order)
    if n_states == 1:
        degrees = degrees[::-1]

    # Create diag mask for intermediate fractions. Needs to be calculated at compile time for jit
    diag_mask = (jnp.ones((n_states, n_states)) - jnp.eye(n_states)).astype(bool)
    triu_indices = jnp.triu_indices(n_states, k=0)
    num_triag_elements = n_states * (n_states + 1) // 2 #+1 for including diagonal
    num_features = n_states 
    num_off_diag_elms = (n_states**2)-n_states
    num_x_repeat = num_off_diag_elms 

    # Precompute numerator and denominator indices for basic fractions
    numerator_indices = jnp.repeat(jnp.arange(n_states), n_states - 1)
    denominator_indices = jnp.concatenate([jnp.concatenate([jnp.arange(i), jnp.arange(i + 1, n_states)]) for i in range(n_states)])

    # Define lambda functions for each transformation
    polynomial_transform = lambda x, library: add_polynomials(x, library, degrees)
    constant_transform = lambda x, library: jnp.concatenate([jnp.ones((1,)), library], axis=0)
    sine_transform = lambda x, library: add_sine(x, library)
    basic_fractions_transform = lambda x, library: add_basic_fractions(x, library, numerator_indices, denominator_indices)
    intermediate_fractions_transform = lambda x, library: add_intermediate_fractions(x, library, diag_mask, triu_indices, num_features, num_x_repeat, num_triag_elements)
    three_body_fractions_transform = lambda x, library: add_three_body_fractions(x, library)

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
        if include_basic_fractions:
            library = basic_fractions_transform(x, library)
        if include_intermediate_fractions:
            library = intermediate_fractions_transform(x, library)
        if include_three_body_fractions:
            library = three_body_fractions_transform(x, library)
        return library

    return sindy_library

def test_sindy_library() -> None:
    """
    Test the SINDy library with various configurations to ensure correctness.
    """
    test_cases = [
        {"poly_order": 2, "n_states": 3, "include_sine": False, "include_basic_fractions": True, "include_intermediate_fractions": False, "include_three_body_fractions": False, "include_constant": True},
        {"poly_order": 2, "n_states": 2, "include_sine": True, "include_basic_fractions": False, "include_intermediate_fractions": True, "include_three_body_fractions": False, "include_constant": True},
        {"poly_order": 3, "n_states": 4, "include_sine": False, "include_basic_fractions": False, "include_intermediate_fractions": False, "include_three_body_fractions": True, "include_constant": False},
        {"poly_order": 2, "n_states": 1, "include_sine": False, "include_basic_fractions": False, "include_intermediate_fractions": False, "include_three_body_fractions": False, "include_constant": True},
        {"poly_order": 1, "n_states": 1, "include_sine": False, "include_basic_fractions": False, "include_intermediate_fractions": True, "include_three_body_fractions": False, "include_constant": False}
    ]
    test_cases = [test_cases[-1]]

    for case in test_cases:
        n_states = case["n_states"]

        test_features = jnp.array([jnp.arange(1, n_states + 1, dtype=float) + i for i in range(2)])
        larger_test_features = jnp.array([jnp.arange(1, n_states + 1, dtype=float) + i for i in range(3)])
        test_features = jnp.array([[2]], dtype=float)
        #larger_test_features = jnp.array([[2, 7, 11]], dtype=float)

        print(f"Testing with config: {case}")
        sindy_lib = sindy_library_factory(case["poly_order"], n_states, case["include_sine"], case["include_basic_fractions"], case["include_intermediate_fractions"], case["include_three_body_fractions"], case["include_constant"])
        #sindy_lib = jit(vmap(sindy_lib))
        sindy_lib = vmap(sindy_lib)
        library = sindy_lib(test_features)
        larger_library = sindy_lib(larger_test_features)
        
        print("Test features:")    
        print(test_features)
        print("Library:")
        print(library)
        #print("Larger test features:")    
        #print(larger_test_features)
        #print("Larger Library:")
        #print(larger_library)

        lib_size = library.shape[1]
        lib_size_larger = larger_library.shape[1]
        lib_size_func = library_size(case["poly_order"], case["include_sine"], case["include_basic_fractions"], case["include_intermediate_fractions"], case["include_three_body_fractions"], case["n_states"], case["include_constant"])

        print("Library size from function: ", lib_size_func)
        print("Library size Larger input : ", lib_size_larger)
        print("Library size test input: ", lib_size)

if __name__ == "__main__":
    test_sindy_library()
