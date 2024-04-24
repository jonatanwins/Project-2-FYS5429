import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ThreeBodyData:
    def __init__(self, t_span, num_points, method='RK45', rtol=1e-9, atol=1e-9):
        
        self.G = 6.67430e-11  # Gravitational constant (in m^3 kg^-1 s^-2)

        self.mass_sun = 1.989e30  # Mass of the sun (in kg)

        # Masses of the three bodies (in kilograms)
        self.m1 = 1.1 * self.mass_sun  # Alpha Centauri A
        self.m2 = 0.907 * self.mass_sun  # Alpha Centauri B
        self.m3 = 0.123 * self.mass_sun  # Proxima Centauri

        self.t_span = t_span
        self.t_eval = np.linspace(t_span[0], t_span[1], num_points)

        # ODE solver stuff
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.initial_conditions = None
        self.solution = None


    def three_body_equations(self, t, y):
        """
        Compute the derivatives for the three-body problem at a given time t and state y.

        This method calculates the acceleration of each body due to the gravitational forces exerted by the other bodies
        using Newton's law of universal gravitation. The resulting differential equations govern the motion of the bodies.

        Parameters:
        - t (float): Current time point for which the derivatives are being computed.
        - y (array): Current state of the system, containing position and velocity vectors for all three bodies.

        Returns:
        - array: Derivatives of the state vector, including velocities and accelerations for all bodies.
        """
        r1, r2, r3, v1, v2, v3 = y.reshape(6, -1)
        r12 = r1 - r2
        r13 = r1 - r3
        r23 = r2 - r3
        a1 = -self.G * self.m2 * r12 / np.linalg.norm(r12)**3 - self.G * self.m3 * r13 / np.linalg.norm(r13)**3
        a2 = -self.G * self.m3 * r23 / np.linalg.norm(r23)**3 - self.G * self.m1 * (-r12) / np.linalg.norm(r12)**3
        a3 = -self.G * self.m1 * (-r13) / np.linalg.norm(r13)**3 - self.G * self.m2 * (-r23) / np.linalg.norm(r23)**3
        return np.concatenate((v1, v2, v3, a1, a2, a3)).flatten()


    def solve_ode(self):
        """
        Solve the ordinary differential equations (ODEs) for the three-body problem using initial conditions.

        This method utilizes the 'solve_ivp' function from the scipy.integrate library to numerically integrate the 
        equations of motion for a three-body system over a specified time span. The solver integrates the system of
        differential equations defined by the 'three_body_equations' method from the initial time to the final time 
        specified in 't_span'. The integration method can be chosen through the 'method' attribute (e.g., 'RK45' for 
        explicit Runge-Kutta method of order 5(4)).

        The solver's options such as relative tolerance ('rtol') and absolute tolerance ('atol') ensure the accuracy
        of the solution.

        Returns:
        - Solution object: An object containing the time points ('t'), the numerical solutions ('y'), and other
                        information about the solver's performance. The object also contains a boolean 'success'
                        flag indicating whether the solver succeeded. If the solver fails, the function logs the
                        issue and returns `None`.
        """
        try:
            self.solution = solve_ivp(
                self.three_body_equations, self.t_span, self.initial_conditions,
                t_eval=self.t_eval, method=self.method, rtol=self.rtol, atol=self.atol
            )
            if not self.solution.success:
                logging.warning("ODE solver did not converge.")
            return self.solution
        except Exception as e:
            logging.error(f"Failed to solve ODE: {e}")
            return None


    def create_high_dimensional_dataset(self, num_modes=6, poly_order=128, domain=[-1, 1]):
        """
        Create a row-major high-dimensional dataset using Legendre polynomials and the three-body problem solution.
        
        Args:
            num_modes (int): Number of spatial modes to use (fixed at 6 for this example).
            poly_order (int): The number of points in the spatial domain to evaluate the polynomials.
            domain (list): The domain over which to evaluate the Legendre polynomials.
            
        Returns:
            X (np.ndarray): The high-dimensional dataset in row-major format, where each row is a time step.
        """
        # Ensure the solution has been computed and is successful
        if self.solution is None or not self.solution.success:
            raise ValueError("The ODE solution must be successfully calculated before creating the high-dimensional dataset.")

        num_samples = self.solution.y.shape[1]

        # Generate grid points where Legendre polynomials will be evaluated
        grid_points = np.linspace(domain[0], domain[1], poly_order)

        # Evaluate Legendre polynomials on the grid
        legendre_polys = np.polynomial.legendre.legval(grid_points, np.identity(num_modes))

        # Allocate space for the high-dimensional dataset
        X = np.zeros((num_samples, num_modes * poly_order))

        # Apply the transformation to each sample
        for sample in range(num_samples):
            r = self.solution.y[:, sample]  # State variables at this time step
            for mode in range(num_modes):
                mod_j = mode%3
                if mode < int(num_modes/2):
                    term = r[mode] * legendre_polys[mode]
                else:
                    term = (r[mode] * legendre_polys[mode])**3
                    
                X[sample, mod_j*poly_order:(mod_j+1)*poly_order] = term
        return X
    

    def generate_datasets(self, position_bounds, velocity_bounds, num_initial_conditions, 
                        num_modes, poly_order, train_split, val_split, 
                        test_split, high_dim=True, plot=False):        
        """
        Generate datasets based on the three-body problem using sampled initial conditions, with an option to create high-dimensional datasets.

        This method samples initial conditions within specified bounds for position and velocity, solves the three-body equations,
        and optionally transforms the solution into a high-dimensional dataset using Legendre polynomials.

        Args:
            position_bounds (dict): Min and max ranges for position components of each body.
            velocity_bounds (dict): Min and max ranges for velocity components of each body.
            num_initial_conditions (int): Number of initial condition samples.
            num_modes (int): Number of spatial modes for Legendre polynomial transformation.
            poly_order (int): Order of Legendre polynomials.
            train_split (float): Proportion of data for training.
            val_split (float): Proportion of data for validation.
            test_split (float): Proportion of data for testing.
            high_dim (bool): If True, creates a high-dimensional dataset; otherwise uses raw simulation data.
            plot (bool): If True, displays an animation of the body trajectories.

        Returns:
            tuple: Contains training, validation, and testing dataset instances of ThreeBodyHighDimDataset.
        """
        
        # Sample initial conditions
        initial_conditions = self.sample_initial_conditions(num_initial_conditions, position_bounds, velocity_bounds)
        
        data_list = []
        for i in tqdm(range(num_initial_conditions), desc="Generating datasets"):
            self.initial_conditions = initial_conditions[i]
            solution = self.solve_ode()  # Solve ODE


            if plot:  # For testing
                if self.solution is None or not self.solution.success:
                    logging.error("Solution is not available or failed. Cannot plot.")
                    return
                positions = self.solution.y[:9].reshape(3, 3, -1)
                self.plot_animation(positions=positions, title=f'Animated Trajectories of Alpha Centauri System Bodies {i}')

            if solution is not None and solution.success:
                if high_dim:
                    # Transform the solution into high-dimensional dataset
                    data = self.create_high_dimensional_dataset(num_modes, poly_order)
                else:
                    # Use the solution directly
                    data = solution.y
                data_list.append(data)
            else:
                logging.warning(f"ODE solver did not converge for initial condition set {i}. Skipping this dataset.")

        if not data_list:  # Check if the list is empty
            raise RuntimeError("No valid solutions were generated. Check the ODE solver and initial conditions.")

        # Combine all data into one array
        data_combined = np.concatenate(data_list, axis=1)

        # Now split the combined data into train, validation, and test sets
        num_samples = data_combined.shape[1]
        train_size = int(num_samples * train_split)
        val_size = int(num_samples * val_split) 

        train_data = data_combined[:, 0 : train_size]
        val_data = data_combined[:, train_size : train_size + val_size]
        test_data = data_combined[:, train_size + val_size:]

        # # Use sklearn's train_test_split to split the data
        # train_data, test_val_data = train_test_split(data_combined, train_size=train_size, shuffle=False)
        # val_data, test_data = train_test_split(test_val_data, train_size=val_size, shuffle=False)

        np.save(f'ThreeBodyDataset_train.npy', train_data)
        np.save(f'ThreeBodyDataset_val.npy', val_data)
        np.save(f'ThreeBodyDataset_test.npy', test_data)

        # The current data has shape (features, samples)

        # Create PyTorch datasets (transposed to get (samples, features))
        train_dataset = ThreeBodyDataset(train_data.T)
        val_dataset = ThreeBodyDataset(val_data.T)
        test_dataset = ThreeBodyDataset(test_data.T)

        return train_dataset, val_dataset, test_dataset

        
    def sample_initial_conditions(self, num_samples, position_bounds, velocity_bounds):
        """
        Sample initial conditions uniformly within the specified bounds for position and velocity.

        Args:
            num_samples (int): Number of samples to generate.
            position_bounds (dict): Dictionary specifying the range for x, y, and z components of position.
            velocity_bounds (dict): Dictionary specifying the range for u, v, and w components of velocity.

        Returns:
            initial_conditions (np.ndarray): Array of sampled initial conditions with shape (num_samples, 18).
        """
        initial_conditions = np.empty((num_samples, 18))

        for i in range(num_samples):
            # Sample positions and velocities within the specified bounds.
            r1_0 = [np.random.uniform(*position_bounds['x']),
                    np.random.uniform(*position_bounds['y']),
                    np.random.uniform(*position_bounds['z'])]
            r2_0 = [np.random.uniform(*position_bounds['x']),
                    np.random.uniform(*position_bounds['y']),
                    np.random.uniform(*position_bounds['z'])]
            r3_0 = [np.random.uniform(*position_bounds['x']),
                    np.random.uniform(*position_bounds['y']),
                    np.random.uniform(*position_bounds['z'])]
            
            v1_0 = [np.random.uniform(*velocity_bounds['u']),
                    np.random.uniform(*velocity_bounds['v']),
                    np.random.uniform(*velocity_bounds['w'])]
            v2_0 = [np.random.uniform(*velocity_bounds['u']),
                    np.random.uniform(*velocity_bounds['v']),
                    np.random.uniform(*velocity_bounds['w'])]
            v3_0 = [np.random.uniform(*velocity_bounds['u']),
                    np.random.uniform(*velocity_bounds['v']),
                    np.random.uniform(*velocity_bounds['w'])]

            # Concatenate position and velocity vectors for each celestial body.
            initial_conditions[i, :] = np.concatenate((r1_0, r2_0, r3_0, v1_0, v2_0, v3_0))

        return initial_conditions


    def plot_animation(self, positions, save_as_gif=False, filename='three_body_animation.gif', 
                       title='Animated Trajectories of Alpha Centauri System Bodies'):
        """
        Plot an animation of the three-body simulation trajectories.

        This method visualizes the trajectories of three bodies in a 3D space over time, based on the
        numerical solutions obtained from the differential equations of the system. It can also save
        the animation as a GIF file.

        Parameters:
        - positions (array): The positions of the three stars, shape = (3, 3, num_samples)
        - save_as_gif (bool): If True, the animation will be saved as a GIF file.
        - filename (str): The filename for the GIF file if save_as_gif is True.
        - title (str): The title of the plot.

        Returns:
        - None: This method does not return anything but shows a matplotlib plot or saves it as a GIF.
        """
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        lines = [ax.plot([], [], [], '-', lw=2, label=f'Body {i+1}')[0] for i in range(3)]

        x_max = np.max(positions[:, 0, :]) * 1.1    
        x_min = np.min(positions[:, 0, :]) * 1.1

        y_max = np.max(positions[:, 1, :]) * 1.1
        y_min = np.min(positions[:, 1, :]) * 1.1

        z_max = np.max(positions[:, 2, :]) * 1.1
        z_min = np.min(positions[:, 2, :]) * 1.1

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(title)

        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return lines

        def animate(i):
            for idx, line in enumerate(lines):
                line.set_data(positions[idx, 0, :i], positions[idx, 1, :i])
                line.set_3d_properties(positions[idx, 2, :i])
            return lines

        ani = FuncAnimation(fig, animate, init_func=init, frames=len(self.t_eval), interval=30, blit=True)
        plt.legend()

        if save_as_gif:
            try:
                ani.save(filename, writer='pillow', fps=30)
            except Exception as e:
                logging.error(f"Failed to save GIF: {e}")

        plt.show()


    def load_random_simulation_and_animate(self, num_initial_conditions):
        """
        Load and animate a random simulation from each dataset (train, val, test).

        Args:
            num_initial_conditions (int): Total number of simulations.
        """
        datasets = ['train', 'val', 'test']
        for dataset in datasets:
            # Load dataset
            data = np.load(f'ThreeBodyDataset_{dataset}.npy')

            # Select a random simulation
            random_index = random.randint(0, num_initial_conditions - 1)
            simulation_data = data[:, random_index:random_index + 365] # 365 time steps

            positions = simulation_data[:9].reshape(3, 3, -1)

            # Plot animation
            self.plot_animation(positions=positions, save_as_gif=False, filename=f'ThreeBodyDataset_{dataset}_random_simulation.gif',
                                title=f'Random {dataset.capitalize()} Simulation')
            
            print(f'Animation for a random {dataset} simulation saved as ThreeBodyDataset_{dataset}_random_simulation.gif')



class ThreeBodyDataset(Dataset):
    def __init__(self, data):

        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == '__main__':
    # Configure logging for debugging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Define the initial conditions range for r (positions) and v (velocities)
    position_bounds = {
        'x': [0, 1e11],  # Range for the x component of all position vectors
        'y': [0, 1e11],  # Range for the y component of all position vectors
        'z': [0, 1e11]   # Range for the z component of all position vectors
    }

    velocity_bounds = {
        'u': [-2e4, 2e4],  # Range for the u component of all velocity vectors
        'v': [-2e4, 2e4],  # Range for the v component of all velocity vectors
        'w': [-2e4, 2e4]   # Range for the w component of all velocity vectors
    }


    # Specify the number of samples and the train/val/test split proportions
    num_initial_conditions = 20  # Total number of different initial conditions aka simulations
    train_split, val_split, test_split = 0.8, 0.1, 0.1


    SECONDS_PER_DAY = 86400
    t_span = (0, 365.25 * SECONDS_PER_DAY)
    num_points = 365

    high_dim = False

    sim = ThreeBodyData(t_span, num_points)
    # Generate the datasets
    train_dataset, val_dataset, test_dataset = sim.generate_datasets(
        position_bounds,
        velocity_bounds,
        num_initial_conditions,
        num_modes=6,
        poly_order=3,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        high_dim=high_dim
    )


    # Create DataLoaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # Now use the load_random_simulation_and_animate method
    sim.load_random_simulation_and_animate(num_initial_conditions)
