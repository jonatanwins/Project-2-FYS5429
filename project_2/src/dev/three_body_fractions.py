import jax
import jax.numpy as jnp

#check how tile function works
x = jnp.array([1, 2, 3])
print(jnp.tile(x, 3))  # [1, 2, 3, 1, 2, 3]