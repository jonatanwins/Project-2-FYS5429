import jax.numpy as jnp
from jax import vmap
from scipy.special import binom
from itertools import product
from jax import Array


def library_size(
    n_states: int,
    poly_order: int,
    include_sine: bool = False,
    include_constant: bool = True,
) -> int:
    """
    Calculate the size of the library based on the given parameters.

    Args:
        n_states (int): The number of state variables.
        poly_order (int): The highest order of polynomials to include.
        include_sine (bool): Whether to include sine terms.
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
    degrees = jnp.array(sorted(degrees, key=lambda x: sum(x)))
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
    return jnp.prod(x**degree)


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


def polynomial_transform_factory(n_states: int, poly_order: int) -> callable:
    """
    Factory function to create a polynomial transform function based on the provided parameters.

    Args:
        n_states (int): The number of state variables.
        poly_order (int): The highest order of polynomials to include.

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


def sindy_library_factory(
    n_states: int = 1,
    poly_order: int = 1,
    include_sine: bool = False,
    include_constant: bool = True,
) -> callable:
    """
    Factory function to create a SINDy library function based on the provided parameters.

    Args:
        n_states (int): The number of state variables.
        poly_order (int): The highest order of polynomials to include.
        include_sine (bool): Whether to include sine terms.
        include_constant (bool): Whether to include a constant term.

    Returns:
        callable: A function that generates the SINDy library for given input.
    """
    polynomial_transform = polynomial_transform_factory(n_states, poly_order)
    sine_transform = sine_transform_factory()
    constant_transform = lambda x, library: jnp.concatenate(
        [jnp.ones((1,)), library], axis=0
    )

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
        {
            "poly_order": 2,
            "n_states": 3,
            "include_sine": False,
            "include_constant": True,
        },
        {
            "poly_order": 2,
            "n_states": 2,
            "include_sine": True,
            "include_constant": True,
        },
        {
            "poly_order": 3,
            "n_states": 4,
            "include_sine": False,
            "include_constant": False,
        },
        {
            "poly_order": 2,
            "n_states": 1,
            "include_sine": False,
            "include_constant": True,
        },
        {
            "poly_order": 1,
            "n_states": 1,
            "include_sine": False,
            "include_constant": False,
        },
    ]
    # test_cases = [test_cases[1]]

    for case in test_cases:
        n_states = case["n_states"]

        test_features = jnp.array(
            [jnp.arange(1, n_states + 1, dtype=float) + i for i in range(2)]
        )
        larger_test_features = jnp.array(
            [jnp.arange(1, n_states + 1, dtype=float) + i for i in range(3)]
        )

        print(f"Testing with config: {case}")
        sindy_lib = sindy_library_factory(
            case["n_states"],
            case["poly_order"],
            case["include_sine"],
            case["include_constant"],
        )
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
        lib_size_func = library_size(
            case["n_states"],
            case["poly_order"],
            case["include_sine"],
            case["include_constant"],
        )

        print("Library size from function: ", lib_size_func)
        print("Library size Larger input : ", lib_size_larger)
        print("Library size test input: ", lib_size)


def get_row_context(library_hparams, second_order=False):
    """Returns a list of strings with the term for each row of the library
        if second_order is True, then for each term the first n_states//2 
        are the z terms and the last are \\dot{z} (dz) terms
    Args:
        library_hparams (Dict): Hyperparameters for the sindy library for the model
        second_order (bool, optional): If the library is second order. Defaults to False.
    """
    n_states = library_hparams["n_states"]
    poly_order = library_hparams["poly_order"]
    include_constant = library_hparams["include_constant"]
    include_sine = library_hparams["include_sine"]

    terms = jnp.diag(jnp.full(n_states, 1))
    if include_constant:
        terms = jnp.concatenate([jnp.zeros((1, n_states)), terms], axis=0)

    if poly_order > 1:
        degrees = jnp.array(list(product(range(poly_order + 1), repeat=n_states)))
        sums = jnp.sum(degrees, axis=1)
        degrees = degrees[(sums <= poly_order) & (sums > 1)][::-1]
        degrees = jnp.array(sorted(degrees, key=lambda x: sum(x)))
        terms = jnp.concatenate([terms, degrees], axis=0)

    if include_sine:
        terms = jnp.concatenate([terms, jnp.diag(jnp.full(n_states, -1))], axis=0)

    if second_order:
        term_context = []
        for row in terms:
            if sum(row) == 0:
                term_context.append(r"1")
            else:
                label = ""
                for i, deg in enumerate(row):
                    if i < n_states // 2:
                        if deg == 1:
                            label += f"z_{i+1}"
                        if deg > 1:
                            label += f"z_{i+1}^{int(deg)}"
                        if deg == -1:
                            label += f"sin(z_{i+1})"
                    else:
                        j = i - n_states // 2
                        if deg == 1:
                            label += rf"\dot{{z}}_{j+1}"
                        if deg > 1:
                            label += rf"\dot{{z}}_{j+1}^{int(deg)}"
                        if deg == -1:
                            label += rf"sin(\dot{{z}}_{j+1})"

                term_context.append(label)
        return term_context
    else:
        term_context = []
        for row in terms:
            if sum(row) == 0:
                term_context.append(r"1")
            else:
                label = ""
                for i, deg in enumerate(row):
                    if deg == 1:
                        label += f"z_{i+1}"
                    if deg > 1:
                        label += f"z_{i+1}^{int(deg)}"
                    if deg == -1:
                        label += f"sin(z_{i+1})"
                term_context.append(label)

        return term_context


if __name__ == "__main__":
    # test_sindy_library()
    # test of polynomial_degrees
    print(polynomial_degrees(3, 3))
    #test term context
    print(get_row_context({"n_states": 3, "poly_order": 3, "include_sine": False, "include_constant": True}))
    print(get_row_context({"n_states": 2, "poly_order": 3, "include_sine": True, "include_constant": True}, True))