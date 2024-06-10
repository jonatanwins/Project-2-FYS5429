import jax
import jax.numpy as jnp
from jax import jit

def create_jax_batches(data, batch_size):
    """
    Convert Lorenz data to JAX arrays and create batches.

    Arguments:
        batch_size - Size of each batch.
        data - Dictionary containing 'x', 'dx' arrays, and optionally 'ddx'.

    Returns:
        batches - List of tuples. Each tuple contains a batch of 'x' and 'dx' arrays,
                  and optionally 'ddx' if provided.
    """
    x = jnp.array(data['x'])
    dx = jnp.array(data['dx'])
    
    # Check if 'ddx' is provided
    ddx = jnp.array(data['ddx']) if 'ddx' in data else None

    # Calculate the number of batches
    num_samples = x.shape[0]
    num_batches = num_samples // batch_size

    # Create the batches
    batches = []
    for i in range(num_batches):
        x_batch = x[i * batch_size: (i + 1) * batch_size]
        dx_batch = dx[i * batch_size: (i + 1) * batch_size]
        if ddx is not None:
            ddx_batch = ddx[i * batch_size: (i + 1) * batch_size]
            batches.append((x_batch, dx_batch, ddx_batch))
        else:
            batches.append((x_batch, dx_batch))
    
    return jnp.array(batches, dtype=object)

def shuffle_jax_batches_factory(second_order=False):
    """
    Factory function to create a jitted shuffle_jax_batches function.

    Arguments:
        second_order - Boolean indicating if ddx is included in the data.

    Returns:
        A jitted function to shuffle the JAX batches.
    """
    if second_order:
        @jit
        def shuffle_jax_batches(jax_batches, rng_key):
            """
            Shuffle the JAX batches while keeping the (x, dx, ddx) pairs intact.

            Arguments:
                jax_batches - JAX array of shape (num_batches, 3, batch_size, input_dim).
                rng_key - JAX random key for shuffling.

            Returns:
                shuffled_batches - JAX array of shuffled batches.
            """
            # Separate x, dx, and ddx from the batches
            x_batches = jax_batches[:, 0]
            dx_batches = jax_batches[:, 1]
            ddx_batches = jax_batches[:, 2]

            # Concatenate all batches
            x_all = jnp.concatenate(x_batches, axis=0)
            dx_all = jnp.concatenate(dx_batches, axis=0)
            ddx_all = jnp.concatenate(ddx_batches, axis=0)

            # Get the number of samples and batch size
            num_samples = x_all.shape[0]
            batch_size = x_batches.shape[1]

            # Generate a random permutation of indices
            perm = jax.random.permutation(rng_key, num_samples)

            # Shuffle the arrays
            x_shuffled = x_all[perm]
            dx_shuffled = dx_all[perm]
            ddx_shuffled = ddx_all[perm]

            # Calculate the number of full batches
            num_batches = num_samples // batch_size

            # Select only the samples that fit into full batches
            x_shuffled = x_shuffled[:num_batches * batch_size]
            dx_shuffled = dx_shuffled[:num_batches * batch_size]
            ddx_shuffled = ddx_shuffled[:num_batches * batch_size]

            # Split the arrays into batches
            x_batches = jnp.reshape(x_shuffled, (num_batches, batch_size, -1))
            dx_batches = jnp.reshape(dx_shuffled, (num_batches, batch_size, -1))
            ddx_batches = jnp.reshape(ddx_shuffled, (num_batches, batch_size, -1))

            # Stack the batches together
            shuffled_batches = jnp.stack((x_batches, dx_batches, ddx_batches), axis=1)

            return shuffled_batches

    else:
        @jit
        def shuffle_jax_batches(jax_batches, rng_key):
            """
            Shuffle the JAX batches while keeping the (x, dx) pairs intact.

            Arguments:
                jax_batches - JAX array of shape (num_batches, 2, batch_size, input_dim).
                rng_key - JAX random key for shuffling.

            Returns:
                shuffled_batches - JAX array of shuffled batches.
            """
            # Separate x and dx from the batches
            x_batches = jax_batches[:, 0]
            dx_batches = jax_batches[:, 1]

            # Concatenate all batches
            x_all = jnp.concatenate(x_batches, axis=0)
            dx_all = jnp.concatenate(dx_batches, axis=0)

            # Get the number of samples and batch size
            num_samples = x_all.shape[0]
            batch_size = x_batches.shape[1]

            # Generate a random permutation of indices
            perm = jax.random.permutation(rng_key, num_samples)

            # Shuffle the arrays
            x_shuffled = x_all[perm]
            dx_shuffled = dx_all[perm]

            # Calculate the number of full batches
            num_batches = num_samples // batch_size

            # Select only the samples that fit into full batches
            x_shuffled = x_shuffled[:num_batches * batch_size]
            dx_shuffled = dx_shuffled[:num_batches * batch_size]

            # Split the arrays into batches
            x_batches = jnp.reshape(x_shuffled, (num_batches, batch_size, -1))
            dx_batches = jnp.reshape(dx_shuffled, (num_batches, batch_size, -1))

            # Stack the batches together
            shuffled_batches = jnp.stack((x_batches, dx_batches), axis=1)

            return shuffled_batches

    return shuffle_jax_batches
