
import jax.numpy as jnp
from jax import vmap
from itertools import product
from sklearn.linear_model import LinearRegression

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

def polynomial_degrees(n_states: int, poly_order: int) -> jnp.ndarray:
    degrees = jnp.array(list(product(range(poly_order + 1), repeat=n_states)))
    sums = jnp.sum(degrees, axis=1)
    degrees = degrees[sums <= poly_order]
    return degrees

def get_symbolic_expression(xi: jnp.ndarray, n_states: int = 3, poly_order: int = 3, include_sine: bool = False):
    """
    Generate the expression for the SINDy model.
    
    Args:
        xi: jnp.ndarray of shape (l, n) where l is the number of functions in the library and n is the number of states
        poly_order: int, the maximum order of the polynomial terms
        include_sine: bool, whether to include sine functions in the library
        
    Returns:
        expression: callable function for the dynamics discovered by SINDy
    """
    def polynomial(x, degree):
        return jnp.prod(x ** degree, axis=-1)
    
    poly_terms = polynomial_degrees(n_states=n_states, poly_order=poly_order)

    # Only keep the necessary polynomial terms based on non-zero coefficients in xi
    non_zero_indices = jnp.nonzero(xi)
    unique_terms = jnp.unique(non_zero_indices[0])
    required_poly_terms = poly_terms[unique_terms]

    # Filter the non-zero coefficients
    required_xi = xi[unique_terms, :]

    # Vectorize the polynomial function over degrees using vmap
    vectorized_polynomial = vmap(polynomial, in_axes=(None, 0))

    def dynamics(x):
        features = vectorized_polynomial(x, required_poly_terms).T
        return features @ required_xi

    return dynamics

# Test case
if __name__ == "__main__":
    # Generating test data
    import numpy as np
    n_samples = 100
    np.random.seed(0)
    X = np.random.rand(n_samples, 2)  # Two features
    X = jnp.array(X)

    # Known polynomial relationship (example: y = 1 + 2*x1 + 3*x2 + 4*x1^2 + 5*x1*x2 + 6*x2^2)
    def true_function(X):
        return 1 + 2*X[:, 0] + 3*X[:, 1] + 4*X[:, 0]**2 + 5*X[:, 0]*X[:, 1] + 6*X[:, 1]**2

    y = true_function(X)

    # Fit a linear regression model
    poly_terms = polynomial_degrees(2, 2)
    poly_features = jnp.apply_along_axis(lambda x_row: jnp.array([jnp.prod(x_row ** degree) for degree in poly_terms]), 1, X)
    model = LinearRegression().fit(poly_features, y)
    xi = model.coef_
    xi = jnp.array(xi).reshape(-1, 1)

    # Generate the symbolic expression
    expression = get_symbolic_expression(xi, n_states=2, poly_order=2)

    # Test the generated expression
    y_pred = expression(X)

    # Print the results
    print("True values: ", y[:5])
    print("Predicted values: ", y_pred[:5].flatten())

    
