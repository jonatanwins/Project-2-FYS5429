import jax.numpy as jnp
from jax import vmap

# Define the vector
vector = jnp.array([1, 2, 3, 4, 5, 6])

# Generate pairs in the desired order
numerators = jnp.repeat(vector, len(vector))
denominators = jnp.tile(vector, len(vector))

# Define a function to perform division
def divide(n, d):
    return n / d

# Vectorize the division function using vmap
vectorized_divide = vmap(divide)

# Calculate all possible fractions
fractions = vectorized_divide(numerators, denominators)

print(fractions)

