from itertools import product
from jax import Array
import jax.numpy as jnp
from sindy_utils import polynomial_degrees
from jax import vmap

def sine_lib(n_states):
    """
    sine library 
    """
    lib = []
    for i in range(n_states):
        funcs = [lambda x: x]
        funcs[i] = jnp.sin
        lib.append(tuple(funcs))
    return lib


def get_symbolic_expression(xi: Array, poly_order: int = 3, include_sine: bool = False):
    """
    Generate the expression for the SINDy model.
    
    Args:
        xi: jnp.array of shape (l, n) where l is the number of functions in the library and n is the number of states
        poly_order: int, the maximum order of the polynomial terms
        include_sine: bool, whether to include sine functions in the library
        
    Returns:
        expression: callable function for the dynamics discoverd by SINDy
    """
    def polynomial(x, degree):
        return jnp.prod(x ** degree)

    poly_terms = polynomial_degrees(xi.shape[1], poly_order)

    terms = []
    for state in range(xi.shape[1]):
        state_terms = poly_terms[xi[:, state] != 0] 
        terms.append(state_terms)

    def dynamics(x):
        return
        
    return 
    
# def get_expression(xi: Array, poly_order: int = 3, include_sine: bool = False):
#     """
#     Generate the expression for the SINDy model.
    
#     Args:
#         xi: jnp.array of shape (l, n) where l is the number of functions in the library and n is the number of states
#         poly_order: int, the maximum order of the polynomial terms
#         include_sine: bool, whether to include sine functions in the library
        
#     Returns:
#         expression: jnp.array of shape (l, n) representing the coefficients of the SINDy model
#     """

#     expression = xi
#     return expression