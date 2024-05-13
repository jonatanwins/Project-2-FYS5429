import jax.numpy as jnp
from jax import vmap
from scipy.special import binom
from scipy.integrate import odeint
from itertools import product
from jax import Array


def library_size(
    n: int, poly_order: int, use_sine: bool = False, include_constant=True
) -> int:
    """
    Calculate the size of the library of functions for the given number of
    states and polynomial order

    Args:
        n: int, number of states/features
        poly_order: int, the maximum order of the polynomial terms
        use_sine: bool, whether to include the sine terms
        include_constant: bool, whether to include the constant term

    Returns:
        l: int, the size of the library

    -Iterates through each polynomial order and finds the number of
    combinations with replacement ( n + k - 1 choose k)

    -If use_sine is True, then it adds n sine terms.

    -If include_constant is False, then it subtracts 1 from the total size
    of the library
    """
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l

def polynomial_degrees(n_states: int, poly_order: int) -> Array:
    """
    Generate the polynomial orders for the SINDy model.

    Args:
        n: int, the number of states
        poly_order: int, the maximum order of the polynomial terms

    Returns:
        degrees: jnp.array of shape (l, n) where l is the size of the library
        and n is the number of states

    -Generates all possible combinations of polynomial orders for the given
    number of states and polynomial order.
    """
    degrees = jnp.array(list(product(range(poly_order + 1), repeat=n_states)))
    sums = jnp.sum(degrees, axis=1)
    degrees = degrees[sums <= poly_order]
    return degrees


def create_sindy_library(poly_order: int, 
                         include_sine: bool = False, 
                         n_states: int = 3,
                         ) -> jnp.ndarray:
    
    def polynomial(x, degree):
        return jnp.prod(x ** degree)
    
    def polynomial_features(X, degrees):
        all_polynomials = vmap(polynomial, in_axes=(None, 0))
        
        all_features = vmap(all_polynomials, in_axes=(0, None))
        
        return all_features(X, degrees)
    
    degrees = polynomial_degrees(n_states, poly_order)

    def sindy_library(
        features: Array
    ) -> Array:
        """
        Construct the library of functions for the SINDy model.

        Args:
            features: jnp.ndarray, the input features
        
        Returns:
            library: jnp.ndarray, the library of functions

        """
        return polynomial_features(features, degrees)
    
    if include_sine ==  True:
        return lambda features: add_sine(features, sindy_library(features))
    
    return sindy_library



def add_sine(features: Array, library: Array) -> Array:
    """
    Add sine terms to the library of functions.

    Args:
        features: jnp.ndarray, the input features
        library: jnp.ndarray, the existing library
    
    returns:
        library: jnp.ndarray, the library with sine terms added
    """
    sine = jnp.sin(features)
    library = jnp.concatenate([library, sine], axis=1)
    return library


def sindy_fit(RHS, LHS, coefficient_threshold):
    """

    Fit the SINDy model coefficients using sequential thresholding.

    Args:
        RHS: jnp.ndarray, right-hand side of the SINDy model
            - library matrix of candidate functions
        LHS: jnp.ndarray, left-hand side of the SINDy model
            - matrix of time derivatives
        coefficient_threshold: float, the threshold below which
            coefficients are set to zero

    Returns:
        Xi: jnp.ndarray, the SINDy model coefficients

    """
    m, n = LHS.shape
    Xi = jnp.linalg.lstsq(RHS, LHS, rcond=None)[0]

    for k in range(10):
        small_inds = jnp.abs(Xi) < coefficient_threshold
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:, i]
            if jnp.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds, i] = jnp.linalg.lstsq(RHS[:, big_inds], LHS[:, i], rcond=None)[
                0
            ]
    return Xi


def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    """
    Simulate the discovered dynamical system from initial conditions using the
        SINDy coefficients.

    Args:
        x0: jnp.ndarray, initial state of the system
        t: jnp.ndarray, time points where the solution is sought
            (must be 1D array)
        Xi: jnp.ndarray, matrix of SINDy coefficients used for simulation
        poly_order: int, the polynomial order used in the function library
        include_sine: bool, whether to include sine in the function library

    Returns:
        x: jnp.ndarray, array of model states over time points

    """

    n = x0.size
    sindy_library = create_sindy_library(poly_order, include_sine, n)

    lib_size = library_size(n, poly_order, include_sine)

    def f(x, t):
        return jnp.dot(
            sindy_library(jnp.array(x).reshape((1, n)), poly_order, lib_size=lib_size),
            Xi,
        ).reshape((n,))

    x = odeint(f, x0, t)
    return x


def sindy_simulate_order2(x0, dx0, t, Xi, poly_order, include_sine):
    """
    Simulate the second-order dynamical system
        specified by the SINDy coefficients.

    Args:
        x0: jnp.ndarray, initial state vector of the system
        dx0: jnp.ndarray, initial derivative of the state vector
        t: jnp.ndarray, time points for the simulation
        Xi: jnp.ndarray, SINDy coefficients for the first-order system
        poly_order: int, order of the polynomials in the library
        include_sine: bool, flag to include sine function in the library

    Returns:
        x: jnp.ndarray, the simulated states of the system
            at the requested time points

    """

    n = 2 * x0.size
    l = Xi.shape[0]

    Xi_order1 = jnp.zeros((l, n))
    for i in range(n // 2):
        Xi_order1[2 * (i + 1), i] = 1.0
        Xi_order1[:, i + n // 2] = Xi[:, i]

    x = sindy_simulate(
        jnp.concatenate((x0, dx0)), t, Xi_order1, poly_order, include_sine
    )
    return x

if __name__ == "__main__":
    from jax import random
    key = random.PRNGKey(1)
    X = jnp.array([1,2,3,4]).reshape(2,-1)
    print(X)
    my_function = create_sindy_library(poly_order = 2, include_sine = False, n_states=2)
    print(my_function(X))
    print(polynomial_degrees(2,2))

    # # print(my_function(X).shape)
    # print(polynomial_degrees(3, 3))


