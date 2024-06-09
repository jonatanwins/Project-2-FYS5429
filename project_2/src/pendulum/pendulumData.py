import numpy as np
from scipy.integrate import odeint


def get_pendulum_data(n_ics):
    t,x,dx,ddx,z = generate_pendulum_data(n_ics)
    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics*t.size, -1))
    data['dx'] = dx.reshape((n_ics*t.size, -1))
    data['ddx'] = ddx.reshape((n_ics*t.size, -1))
    data['z'] = z.reshape((n_ics*t.size, -1))[:,0:1]
    data['dz'] = z.reshape((n_ics*t.size, -1))[:,1:2]

    return data

def get_pendulum_training_data(n_ics):
    t,x,dx,ddx,z = generate_pendulum_data(n_ics)
    data = {}
    data['x'] = x.reshape((n_ics*t.size, -1))
    data['dx'] = dx.reshape((n_ics*t.size, -1))
    data['ddx'] = ddx.reshape((n_ics*t.size, -1))
    #explicity del unused variables to free up memory, might be useless idk
    del t, z
    return data


def generate_pendulum_data(n_ics):
    f  = lambda z, t : [z[1], -np.sin(z[0])]
    t = np.arange(0, 10, .02)

    z = np.zeros((n_ics,t.size,2))
    dz = np.zeros(z.shape)

    z1range = np.array([-np.pi,np.pi])
    z2range = np.array([-2.1,2.1])
    i = 0
    while (i < n_ics):
        z0 = np.array([(z1range[1]-z1range[0])*np.random.rand()+z1range[0],
            (z2range[1]-z2range[0])*np.random.rand()+z2range[0]])
        if np.abs(z0[1]**2/2. - np.cos(z0[0])) > .99:
            continue
        z[i] = odeint(f, z0, t)
        dz[i] = np.array([f(z[i,j], t[j]) for j in range(len(t))])
        i += 1

    x,dx,ddx = pendulum_to_movie(z, dz)

    return t,x,dx,ddx,z


def pendulum_to_movie(z, dz):
    n_ics = z.shape[0]
    n_samples = z.shape[1]
    n = 51
    y1,y2 = np.meshgrid(np.linspace(-1.5,1.5,n),np.linspace(1.5,-1.5,n))
    create_image = lambda theta : np.exp(-((y1-np.cos(theta-np.pi/2))**2 + (y2-np.sin(theta-np.pi/2))**2)/.05)
    argument_derivative = lambda theta,dtheta : -1/.05*(2*(y1 - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta \
                                                      + 2*(y2 - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta)
    argument_derivative2 = lambda theta,dtheta,ddtheta : -2/.05*((np.sin(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta**2 \
                                                               + (y1 - np.cos(theta-np.pi/2))*np.cos(theta-np.pi/2)*dtheta**2 \
                                                               + (y1 - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*ddtheta \
                                                               + (-np.cos(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta**2 \
                                                               + (y2 - np.sin(theta-np.pi/2))*(np.sin(theta-np.pi/2))*dtheta**2 \
                                                               + (y2 - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*ddtheta)
        
    x = np.zeros((n_ics, n_samples, n, n))
    dx = np.zeros((n_ics, n_samples, n, n))
    ddx = np.zeros((n_ics, n_samples, n, n))
    for i in range(n_ics):
        for j in range(n_samples):
            z[i,j,0] = wrap_to_pi(z[i,j,0])
            x[i,j] = create_image(z[i,j,0])
            dx[i,j] = (create_image(z[i,j,0])*argument_derivative(z[i,j,0], dz[i,j,0]))
            ddx[i,j] = create_image(z[i,j,0])*((argument_derivative(z[i,j,0], dz[i,j,0]))**2 \
                            + argument_derivative2(z[i,j,0], dz[i,j,0], dz[i,j,1]))
            
    return x,dx,ddx


def wrap_to_pi(z):
    z_mod = z % (2*np.pi)
    subtract_m = (z_mod > np.pi) * (-2*np.pi)
    return z_mod + subtract_m




import jax
import jax.numpy as jnp
from jax import jit
from typing import Dict, Tuple

def create_jax_batches(batch_size: int, data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert Pendulum data to JAX arrays and create batches.

    Arguments:
        batch_size - Size of each batch.
        data - Dictionary containing 'x' and 'dx' arrays.

    Returns:
        batches - JAX array of shape (num_batches, 2, batch_size, input_dim).
    """
    x = jnp.array(data['x'])
    dx = jnp.array(data['dx'])
    
    # Calculate the number of batches
    num_samples = x.shape[0]
    num_batches = num_samples // batch_size

    # Create the batches
    x_batches = jnp.reshape(x[:num_batches * batch_size], (num_batches, batch_size, -1))
    dx_batches = jnp.reshape(dx[:num_batches * batch_size], (num_batches, batch_size, -1))
    
    # Stack the x and dx batches together
    batches = jnp.stack((x_batches, dx_batches), axis=1)

    return batches

@jit
def shuffle_jax_batches(jax_batches: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
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

# Test the functions
if __name__ == "__main__":
    # Number of initial conditions
    n_ics = 5

    # Generate pendulum training data
    training_data = get_pendulum_training_data(n_ics)

    # Specify the batch size
    batch_size = 32

    # Create JAX batches
    jax_batches = create_jax_batches(batch_size, training_data)
    print(f"Number of batches: {jax_batches.shape[0]}")
    print(f"Shape of the first batch x: {jax_batches[0][0].shape}, dx: {jax_batches[0][1].shape}")
    print(jax_batches.shape) # <- this is the input

    # Create a random key
    rng_key = jax.random.PRNGKey(42)

    # Shuffle the batches and print some information
    shuffled_batches = shuffle_jax_batches(jax_batches, rng_key)
    print(f"Number of shuffled batches: {shuffled_batches.shape[0]}")
    print(f"Shape of the first shuffled batch x: {shuffled_batches[0][0].shape}, dx: {shuffled_batches[0][1].shape}")
    print(shuffled_batches.shape) # <- this is the output


