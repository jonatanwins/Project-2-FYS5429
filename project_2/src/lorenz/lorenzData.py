"""
Large parts of this module is fetched from the SindyAutoencoder repository
https://github.com/kpchamp/SindyAutoencoders/blob/master/examples/lorenz/example_lorenz.py

Some slight modifications where made to troubleshoot memory bugs and modernize the code
"""


import sys
sys.path.append("../")
from sindyLibrary import library_size
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import legendre
from torch.utils.data import Dataset
import jax.numpy as jnp
from jax import jit
import jax

def lorenz(t, z, sigma=10, beta=8/3, rho=28):
    x, y, z = z
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def simulate_lorenz(z0, t, sigma=10.0, beta=8 / 3, rho=28.0):
    """
    Simulate the Lorenz dynamics.

    Arguments:
        z0 - Initial condition in the form of a 3-value list or array.
        t - Array of time points at which to simulate.
        sigma, beta, rho - Lorenz parameters

    Returns:
        z, dz - Arrays of the trajectory values and their 1st derivatives.
    """
    f = lambda t, z: lorenz(t, z, sigma=sigma, beta=beta, rho=rho)

    sol = solve_ivp(f, [t[0], t[-1]], z0, t_eval=t, vectorized=True)

    z = sol.y.T

    dt = t[1] - t[0]
    dz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(t[i], z[i])
    return z, dz

def generate_lorenz_data(
    ics, t, n_points, linear=True, normalization=None, sigma=10, beta=8 / 3, rho=28
):
    """
    Generate high-dimensional Lorenz data set.

    Arguments:
        ics - Nx3 array of N initial conditions
        t - array of time points over which to simulate
        n_points - size of the high-dimensional dataset created
        linear - Boolean value. If True, high-dimensional dataset is a linear combination
        of the Lorenz dynamics. If False, the dataset also includes cubic modes.
        normalization - Optional 3-value array for rescaling the 3 Lorenz variables.
        sigma, beta, rho - Parameters of the Lorenz dynamics.

    Returns:
        data - Dictionary containing elements of the dataset. This includes the time points (t),
        spatial mapping (y_spatial), high-dimensional modes used to generate the full dataset
        (modes), low-dimensional Lorenz dynamics (z, along with 1st derivatives dz),
        high-dimensional dataset (x, along with 1st derivatives dx), and
        the true Lorenz coefficient matrix for SINDy.
    """

    n_ics = ics.shape[0]
    n_steps = t.size

    d = 3
    z = np.zeros((n_ics, n_steps, d))
    dz = np.zeros(z.shape)
    for i in range(n_ics):
        z[i], dz[i] = simulate_lorenz(
            ics[i], t, sigma=sigma, beta=beta, rho=rho
        )

    if normalization is not None:
        z *= normalization
        dz *= normalization

    n = n_points
    L = 1
    y_spatial = np.linspace(-L, L, n)

    modes = np.zeros((2 * d, n))
    for i in range(2 * d):
        modes[i] = legendre(i)(y_spatial)
    x1 = np.zeros((n_ics, n_steps, n))
    x2 = np.zeros((n_ics, n_steps, n))
    x3 = np.zeros((n_ics, n_steps, n))
    x4 = np.zeros((n_ics, n_steps, n))
    x5 = np.zeros((n_ics, n_steps, n))
    x6 = np.zeros((n_ics, n_steps, n))

    x = np.zeros((n_ics, n_steps, n))
    dx = np.zeros(x.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            x1[i, j] = modes[0] * z[i, j, 0]
            x2[i, j] = modes[1] * z[i, j, 1]
            x3[i, j] = modes[2] * z[i, j, 2]
            x4[i, j] = modes[3] * z[i, j, 0] ** 3
            x5[i, j] = modes[4] * z[i, j, 1] ** 3
            x6[i, j] = modes[5] * z[i, j, 2] ** 3

            x[i, j] = x1[i, j] + x2[i, j] + x3[i, j]
            if not linear:
                x[i, j] += x4[i, j] + x5[i, j] + x6[i, j]

            dx[i, j] = (
                modes[0] * dz[i, j, 0] + modes[1] *
                dz[i, j, 1] + modes[2] * dz[i, j, 2]
            )
            if not linear:
                dx[i, j] += (
                    modes[3] * 3 * (z[i, j, 0] ** 2) * dz[i, j, 0]
                    + modes[4] * 3 * (z[i, j, 1] ** 2) * dz[i, j, 1]
                    + modes[5] * 3 * (z[i, j, 2] ** 2) * dz[i, j, 2]
                )

    if normalization is None:
        sindy_coefficients = lorenz_coefficients(
            [1, 1, 1], sigma=sigma, beta=beta, rho=rho
        )
    else:
        sindy_coefficients = lorenz_coefficients(
            normalization, sigma=sigma, beta=beta, rho=rho
        )

    data = {}
    data["t"] = t
    data["y_spatial"] = y_spatial
    data["modes"] = modes
    data["x"] = x
    data["dx"] = dx
    data["z"] = z
    data["dz"] = dz
    data["sindy_coefficients"] = sindy_coefficients.astype(np.float32)

    return data

def get_lorenz_train_data(n_ics, noise_strength=0):
    """
    Generate a set of Lorenz training data for multiple random initial conditions.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.

    Return:
        data - Dictionary containing elements of the dataset.
    """
    t = np.arange(0, 5, 0.02)
    n_steps = t.size
    input_dim = 128

    ic_means = np.array([0, 0, 25])
    ic_widths = 2 * np.array([36, 48, 41])

    # Generate initial conditions
    ics = ic_widths * (np.random.rand(n_ics, 3) - 0.5) + ic_means

    # Generate Lorenz data with the specified initial conditions
    data = generate_lorenz_data(ics, t, input_dim, noise_strength)
    x = data["x"].copy()
    dx = data["dx"].copy()
    del data
    x = x.reshape(
        (-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    dx = dx.reshape(
        (-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)

    del t, n_steps, input_dim, ic_means, ic_widths

    return {'x': x, 'dx': dx}

def get_lorenz_test_data(n_ics, noise_strength=0):
    """
    Generate a set of Lorenz test data for multiple random initial conditions.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.

    Return:
        data - Dictionary containing elements of the dataset.
    """
    t = np.arange(0, 5, 0.02)
    n_steps = t.size
    input_dim = 128

    ic_means = np.array([0, 0, 25])
    ic_widths = 2 * np.array([36, 48, 41])

    # Generate initial conditions
    ics = ic_widths * (np.random.rand(n_ics, 3) - 0.5) + ic_means

    # Generate Lorenz data with the specified initial conditions
    data = generate_lorenz_data(
        ics, t, input_dim, linear=False, normalization=np.array([1 / 40, 1 / 40, 1 / 40])
    )

    data["x"] = data["x"].reshape(
        (-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data["dx"] = data["dx"].reshape(
        (-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data["z"] = data["z"].reshape((-1, 3))
    data["dz"] = data["dz"].reshape((-1, 3))

    return data

def lorenz_coefficients(normalization, poly_order=3, sigma=10.0, beta=8 / 3, rho=28.0):
    """
    Generate the SINDy coefficient matrix for the Lorenz system.

    Arguments:
        normalization - 3-element list of array specifying scaling of each Lorenz variable
        poly_order - Polynomial order of the SINDy model.
        sigma, beta, rho - Parameters of the Lorenz system
    """
    Xi = np.zeros((library_size(3, poly_order), 3))
    Xi[1, 0] = -sigma
    Xi[2, 0] = sigma * normalization[0] / normalization[1]
    Xi[1, 1] = rho * normalization[1] / normalization[0]
    Xi[2, 1] = -1
    Xi[6, 1] = -normalization[1] / (normalization[0] * normalization[2])
    Xi[3, 2] = -beta
    Xi[5, 2] = normalization[2] / (normalization[0] * normalization[1])
    return Xi



def create_jax_batches(data, batch_size):
    """
    Convert Lorenz data to JAX arrays and create batches.

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
    
    # # Handle the remaining samples if any
    # if num_samples % batch_size != 0:
    #     x_batch = x[num_batches * batch_size:]
    #     dx_batch = dx[num_batches * batch_size:]
    #     batches.append((x_batch, dx_batch))

    return jnp.array(batches)

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
    
    # Stack the x and dx batches together
    shuffled_batches = jnp.stack((x_batches, dx_batches), axis=1)
    
    return shuffled_batches

# Test the function
if __name__ == "__main__":
    # Generate the training data
    training_data = get_lorenz_train_data(2)
    print("Training Data Keys:", training_data.keys())
    print(f"x shape: {training_data['x'].shape}, dx shape: {training_data['dx'].shape}")

    # Specify the batch size
    batch_size = 32

    # Create JAX batches
    jax_batches = create_jax_batches(batch_size, training_data)

    # Print some information about the batches
    print(f"Number of batches: {jax_batches.shape[0]}")
    print(f"Shape of the first batch x: {jax_batches[0][0].shape}, dx: {jax_batches[0][1].shape}")
    print(jax_batches.shape) 

    # Create a random key
    rng_key = jax.random.PRNGKey(42)

    # Shuffle the batches and print some information
    shuffled_batches = shuffle_jax_batches(jax_batches, rng_key)
    print(f"Number of shuffled batches: {shuffled_batches.shape[0]}")
    print(f"Shape of the first shuffled batch x: {shuffled_batches[0][0].shape}, dx: {shuffled_batches[0][1].shape}")
    print(shuffled_batches.shape) 