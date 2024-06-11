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
    ###lib kwargs for true coefficients
    poly_order = 3
    include_sine = False
    include_constant = True

    lib_kwargs = {'poly_order': poly_order, 'include_sine': include_sine, 'include_constant': include_constant}

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
            [1, 1, 1], sigma=sigma, beta=beta, rho=rho, **lib_kwargs
        )
    else:
        sindy_coefficients = lorenz_coefficients(
            normalization, sigma=sigma, beta=beta, rho=rho, **lib_kwargs
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

def get_lorenz_data(n_ics, noise_strength=0, test_data=False):
    """
    Generate a set of Lorenz data for multiple random initial conditions. 
    Used for training and validation data.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.
        test_data - Boolean indicating whether to generate test data (True) or train data (False).

    Return:
        x - Data array of shape (n_ics * n_steps, input_dim)
        dx - Derivative data array of shape (n_ics * n_steps, input_dim)
        t - Time array
    """
    t = np.arange(0, 5, 0.02)
    n_steps = t.size
    input_dim = 128

    ic_means = np.array([0, 0, 25])
    ic_widths = 2 * np.array([36, 48, 41])

    # Generate initial conditions
    ics = ic_widths * (np.random.rand(n_ics, 3) - 0.5) + ic_means

    data = generate_lorenz_data(ics, t, input_dim, noise_strength, normalization=np.array([1 / 40, 1 / 40, 1 / 40]))

    x = data["x"].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    dx = data["dx"].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)

    return {"x": x, "dx": dx, "t": t}


def out_of_distro_ics(n_ics, inDist_ic_widths=np.array([36, 48, 41]), outDist_extra_width=np.array([10, 10, 10]), ic_means=np.array([0, 0, 25])):
    """
    Generate out-of-distribution initial conditions (OOD ICs).

    Arguments:
        n_ics - Integer specifying the number of OOD initial conditions to generate.
        inDist_ic_widths - Array specifying the widths of in-distribution initial conditions.
        outDist_extra_width - Array specifying the extra width for out-of-distribution initial conditions.
        ic_means - Array specifying the mean values for the initial conditions (default is [0, 0, 25]).

    Return:
        ics - Array of out-of-distribution initial conditions.
    """
    full_width = inDist_ic_widths + outDist_extra_width
    ics = np.zeros((n_ics, 3))
    i = 0
    
    while i < n_ics:
        ic = np.array([np.random.uniform(-full_width[0], full_width[0]),
                       np.random.uniform(-full_width[1], full_width[1]),
                       np.random.uniform(-full_width[2], full_width[2]) + ic_means[2]])
        
        if ((ic[0] > -inDist_ic_widths[0]) and (ic[0] < inDist_ic_widths[0]) and
            (ic[1] > -inDist_ic_widths[1]) and (ic[1] < inDist_ic_widths[1]) and
            (ic[2] > ic_means[2] - inDist_ic_widths[2]) and (ic[2] < ic_means[2] + inDist_ic_widths[2])):
            continue
        else:
            ics[i] = ic
            i += 1
    
    return ics

def get_lorenz_OutOfDistro_data(n_ics, noise_strength=0, inDist_ic_widths=np.array([36, 48, 41]), outDist_extra_width=np.array([10, 10, 10])):
    """
    Generate a set of Lorenz out-of-distribution (OOD) data for multiple random initial conditions.

    Arguments:
        n_ics - Integer specifying the number of OOD initial conditions to use.
        noise_strength - Amount of noise to add to the data.
        inDist_ic_widths - Array specifying the widths of in-distribution initial conditions.
        outDist_extra_width - Array specifying the extra width for out-of-distribution initial conditions.

    Return:
        x - Data array of shape (n_ics * n_steps, input_dim)
        dx - Derivative data array of shape (n_ics * n_steps, input_dim)
        t - Time array
    """
    t = np.arange(0, 5, 0.02)
    n_steps = t.size
    input_dim = 128

    # Generate out-of-distribution initial conditions (OOD ICs)
    ics = out_of_distro_ics(n_ics, inDist_ic_widths, outDist_extra_width)

    # Generate Lorenz data with the specified initial conditions
    data = generate_lorenz_data(
        ics, t, input_dim, linear=False, normalization=np.array([1 / 40, 1 / 40, 1 / 40])
    )

    x = data["x"].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    dx = data["dx"].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)

    return {"x": x, "dx": dx, "t": t}


def lorenz_coefficients(normalization=(1,1,1), sigma=10.0, beta=8 / 3, rho=28.0, **library_kwargs):
    """
    Generate the SINDy coefficient matrix for the Lorenz system.

    Arguments:
        normalization - 3-element list of array specifying scaling of each Lorenz variable
        poly_order - Polynomial order of the SINDy model.
        sigma, beta, rho - Parameters of the Lorenz system
    """
    Xi = np.zeros((library_size(3, **library_kwargs), 3))
    Xi[1, 0] = -sigma
    Xi[2, 0] = sigma * normalization[0] / normalization[1]
    Xi[1, 1] = rho * normalization[1] / normalization[0]
    Xi[2, 1] = -1
    Xi[6, 1] = -normalization[1] / (normalization[0] * normalization[2])
    Xi[3, 2] = -beta
    Xi[5, 2] = normalization[2] / (normalization[0] * normalization[1])
    return Xi


# Test the function
if __name__ == "__main__":
    # Generate the training data
    training_data = get_lorenz_data(2)
    print("Training Data Keys:", training_data.keys())
    print(f"x shape: {training_data['x'].shape}, dx shape: {training_data['dx'].shape}")