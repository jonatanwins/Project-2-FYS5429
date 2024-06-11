import jax
import jax.numpy as jnp
from jax import jit

def create_jax_batches_factory(second_order=False):
    """
    Factory function to create a function that converts data to JAX arrays and creates batches.

    Arguments:
        second_order - Boolean indicating if ddx is included in the data.

    Returns:
        A function to create JAX batches.
    """
    if second_order:
        def create_jax_batches(data, batch_size):
            """
            Convert Lorenz data to JAX arrays and create batches for second-order data.

            Arguments:
                batch_size - Size of each batch.
                data - Dictionary containing 'x', 'dx', and 'ddx' arrays.

            Returns:
                batches - List of tuples. Each tuple contains a batch of 'x', 'dx', and 'ddx' arrays.
            """
            x = jnp.array(data['x'])
            dx = jnp.array(data['dx'])
            ddx = jnp.array(data['ddx'])

            # Calculate the number of batches
            num_samples = x.shape[0]
            num_batches = num_samples // batch_size

            # Create the batches
            batches = []
            for i in range(num_batches):
                x_batch = x[i * batch_size: (i + 1) * batch_size]
                dx_batch = dx[i * batch_size: (i + 1) * batch_size]
                ddx_batch = ddx[i * batch_size: (i + 1) * batch_size]
                batches.append((x_batch, dx_batch, ddx_batch))
            
            return jnp.array(batches, dtype=jnp.float64) #if float

    else:
        def create_jax_batches(data, batch_size):
            """
            Convert Lorenz data to JAX arrays and create batches for first-order data.

            Arguments:
                batch_size - Size of each batch.
                data - Dictionary containing 'x' and 'dx' arrays.

            Returns:
                batches - List of tuples. Each tuple contains a batch of 'x' and 'dx' arrays.
            """
            x = jnp.array(data['x'])
            dx = jnp.array(data['dx'])

            # Calculate the number of batches
            num_samples = x.shape[0]
            num_batches = num_samples // batch_size

            # Create the batches
            batches = []
            for i in range(num_batches):
                x_batch = x[i * batch_size: (i + 1) * batch_size]
                dx_batch = dx[i * batch_size: (i + 1) * batch_size]
                batches.append((x_batch, dx_batch))
            
            return jnp.array(batches, dtype=jnp.float64) #if float64 is not available it will default to float32

    return create_jax_batches

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
            input_dim = x_all.shape[1]

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
            x_batches = x_shuffled.reshape((num_batches, batch_size, input_dim))
            dx_batches = dx_shuffled.reshape((num_batches, batch_size, input_dim))
            ddx_batches = ddx_shuffled.reshape((num_batches, batch_size, input_dim))

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
            input_dim = x_all.shape[1]

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
            x_batches = x_shuffled.reshape((num_batches, batch_size, input_dim))
            dx_batches = dx_shuffled.reshape((num_batches, batch_size, input_dim))

            # Stack the batches together
            shuffled_batches = jnp.stack((x_batches, dx_batches), axis=1)

            return shuffled_batches

    return shuffle_jax_batches


def test_first_order():
    import numpy as np
    from jax import random
    
    # Create smaller dummy data with integers for easier tracking
    num_samples = 10
    batch_size = 3
    
    # First-order data
    data_first_order = {
        'x': np.array([[i,i] for i in range(num_samples)]),
        'dx': np.array([[i*10,i*10] for i in range(num_samples)])
    }
    
    # Create batch function
    create_jax_batches_first = create_jax_batches_factory(second_order=False)
    
    # Create batches
    batches_first = create_jax_batches_first(data_first_order, batch_size)
    
    print("First-order batches:")
    print(batches_first)
    #print shape
    print(f"Non-shuffled shape : {batches_first.shape}")
    
    # Iterate over batches
    print("\nIterating over first-order batches:")
    for batch in batches_first:
        print("Batch")
        x, dx = batch
        print(f"x: {x}")
        print(f"dx: {dx}")
    
    # Create shuffle function
    shuffle_jax_batches_first = shuffle_jax_batches_factory(second_order=False)
    
    # Shuffle the batches
    rng_key = random.PRNGKey(0)
    shuffled_batches_first = shuffle_jax_batches_first(batches_first, rng_key)
    
    print("\nShuffled first-order batches:")
    print(shuffled_batches_first)

    print(f"Shuffled array shape : {shuffled_batches_first.shape}")    
    # Iterate over shuffled batches
    print("\nIterating over shuffled first-order batches:")
    for batch in shuffled_batches_first:
        print("Batch")
        x, dx = batch
        print(f"x: {x}")
        print(f"dx: {dx}")

def test_second_order():
    import numpy as np
    from jax import random
    
    # Create smaller dummy data with integers for easier tracking
    num_samples = 10
    batch_size = 2
    
    # Second-order data
    data_second_order = {
        'x': np.array([[i,i] for i in range(num_samples)]),
        'dx': np.array([[i*10, i*10] for i in range(num_samples)]),
        'ddx': np.array([[i*100, i*100] for i in range(num_samples)])
    }
    
    # Create batch function
    create_jax_batches_second = create_jax_batches_factory(second_order=True)
    
    # Create batches
    batches_second = create_jax_batches_second(data_second_order, batch_size)
    
    print("Second-order batches:")
    print(batches_second)
    
    # Iterate over batches
    print("\nIterating over second-order batches:")
    for batch in batches_second:
        print("Batch")
        x, dx, ddx = batch
        print(f"x: {x}")
        print(f"dx: {dx}")
        print(f"ddx: {ddx}")

    # Create shuffle function
    shuffle_jax_batches_second = shuffle_jax_batches_factory(second_order=True)
    
    # Shuffle the batches
    rng_key = random.PRNGKey(0)
    shuffled_batches_second = shuffle_jax_batches_second(batches_second, rng_key)
    
    print("\nShuffled second-order batches:")
    print(shuffled_batches_second)
    
    # Iterate over shuffled batches
    print("\nIterating over shuffled second-order batches:")
    for batch in shuffled_batches_second:
        x, dx, ddx = batch
        print(f"x: {x}")
        print(f"dx: {dx}")
        print(f"ddx: {ddx}")


if __name__ == "__main__":
    #test_first_order()
    test_second_order()