import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import legendre
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from torch.utils.data import Dataset, DataLoader
from scipy.stats import qmc


# For nice plots
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['font.size'] = 9
plt.rcParams['figure.figsize'] = (3.4, 3.5)
plt.rcParams['pdf.fonttype'] = 42


class DataSim:
    def __init__(self, t_span, num_points, system_type='three_body', dim=9, method="RK45", rtol=1e-9, atol=1e-9):
        """
        Initializes the DataSim class with specified simulation parameters.

        Parameters:
        - t_span (tuple): The time span for the ODE solver (start, end).
        - num_points (int): Number of time points to evaluate within the time span.
        - system_type (str): Type of dynamical system to simulate ('three_body' or 'lorenz').
        - dim (int): Dimension of the system (number of variables involved).
        - method (str): Numerical method for solving the ODE ('RK45', 'RK23', etc.).
        - rtol (float): Relative tolerance for the ODE solver.
        - atol (float): Absolute tolerance for the ODE solver.
        """
        self.dim = dim
        self.system_type = system_type
        self.G = 6.67430e-11  # Gravitational constant (in m^3 kg^-1 s^-2)
        self.mass_sun = 1.989e30  # Mass of the sun (in kg)
        self.m1 = 1.1 * self.mass_sun  # Alpha Centauri A
        self.m2 = 0.907 * self.mass_sun  # Alpha Centauri B
        self.m3 = 0.123 * self.mass_sun  # Proxima Centauri
        self.t_span = t_span
        self.t_eval = np.linspace(t_span[0], t_span[1], num_points)
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.ode_system = None


    def three_body_equations(self, t, y):
        """
        Defines the differential equations for the three-body problem.

        Parameters:
        - t (float): Current time point (not used directly in calculations here).
        - y (ndarray): Current values of the system's state variables.

        Returns:
        - ndarray: Derivatives of the state variables.
        """
        r1, r2, r3, v1, v2, v3 = y.reshape(6, -1)
        r12 = r1 - r2
        r13 = r1 - r3
        r23 = r2 - r3
        a1 = (
            -self.G * self.m2 * r12 / np.linalg.norm(r12) ** 3
            - self.G * self.m3 * r13 / np.linalg.norm(r13) ** 3
        )
        a2 = (
            -self.G * self.m3 * r23 / np.linalg.norm(r23) ** 3
            - self.G * self.m1 * (-r12) / np.linalg.norm(r12) ** 3
        )
        a3 = (
            -self.G * self.m1 * (-r13) / np.linalg.norm(r13) ** 3
            - self.G * self.m2 * (-r23) / np.linalg.norm(r23) ** 3
        )
        return np.concatenate((v1, v2, v3, a1, a2, a3)).flatten()


    def lorenz_equations(self, t, y, sigma=10, rho=28, beta=8/3):
        """
        Defines the Lorenz system's differential equations.

        Parameters:
        - t (float): Current time point (not used directly in calculations here).
        - y (ndarray): Current values of the system's state variables (x, y, z).
        - sigma (float): Parameter sigma in the Lorenz equations.
        - rho (float): Parameter rho in the Lorenz equations.
        - beta (float): Parameter beta in the Lorenz equations.

        Returns:
        - ndarray: Derivatives of the state variables.
        """
        x, y, z = y
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return np.array([dx_dt, dy_dt, dz_dt])


    def set_ode_system(self, ode_system):
        """
        Sets the ODE system to solve.

        Parameters:
        - ode_system (callable): A function that takes time t and state y and returns dy/dt.
        """
        self.ode_system = ode_system


    def solve_ode(self, plot_trajectory=False):
        """
        Solves the ODE using the specified initial conditions and solver settings.

        Parameters:
        - plot_trajectory (bool): If True, plot the trajectory after solving.

        Returns:
        - OdeResult: An object representing the solution.
        """        
        solution = solve_ivp(self.ode_system, self.t_span, self.initial_conditions, t_eval=self.t_eval, method=self.method, rtol=self.rtol, atol=self.atol)
        if not solution.success:
            logging.warning("ODE solver did not converge.")
        if plot_trajectory:
            self.static_plot(solution)
        return solution

    
    def sample_initial_conditions(self, num_samples, bounds):
        """
        Samples initial conditions for the dynamical system.

        Parameters:
        - num_samples (int): Number of samples to generate.
        - bounds (list of tuples): List of tuples representing the bounds (min, max) for each variable.

        Returns:
        - ndarray: Array of sampled initial conditions.
        """

        num_vars = len(bounds)  # Total number of variables, should be 18 for three bodies with 6 variables each
        initial_conditions = np.empty((num_samples, num_vars))

        for attempt in range(100):  # Limit to prevent infinite loops
            sampler = qmc.LatinHypercube(d=num_vars, optimization="random-cd", seed=0)
            sample = sampler.random(n=num_samples)

            lower_bounds = np.array([b[0] for b in bounds])
            upper_bounds = np.array([b[1] for b in bounds])
            scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)

            # Check the dist. between bodies to make sure they are not to close
            min_distance = 1e3  # Set minimum allowable distance between any two bodies for the three-body problem
            if self.system_type == 'three_body':
                distances_okay = True
                for i in range(num_samples):
                    # Extract positions for each of the three bodies within a single sample
                    positions = scaled_sample[i, :9].reshape(3, 3)  # Assume positions are the first three components for each body

                    # Check distances between each pair of bodies within the sample
                    for j in range(3):
                        for k in range(j + 1, 3):
                            distance = np.linalg.norm(positions[j] - positions[k])
                            if distance < min_distance:
                                distances_okay = False
                                break
                        if not distances_okay:
                            break
                    if not distances_okay:
                        break
                if distances_okay:
                    initial_conditions = scaled_sample
                    break
        return initial_conditions


    def static_plot(self, solution):
        """
        Plots the trajectory of the dynamical system in 3D phase space or 2D time series.

        Parameters:
        - solution (OdeResult): The result object from scipy's solve_ivp method.
        - title (str): Title of the plot.
        """
        plt.rc('axes', edgecolor='none', axisbelow=True, grid=True) # facecolor='whitesmoke'
        plt.rc('grid', color='w', linestyle='solid')
        plt.rc('lines', linewidth=2)


        if self.system_type == 'lorenz':

            x, y, z = solution.y

            # from mayavi import mlab
            # mlab.figure(bgcolor=(1, 1, 1))  # Optional: Set background color to white
            # mlab.plot3d(x, y, z, self.t_eval, tube_radius=0.5, colormap='viridis')
            # mlab.show()  # This should be the last line

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.xaxis.set_pane_color((230/255, 230/255, 230/255, 1))
            ax.yaxis.set_pane_color((230/255, 230/255, 230/255, 1))
            ax.zaxis.set_pane_color((230/255, 230/255, 230/255, 1))
            ax.plot(x, y, z, label='Lorenz Attractor', lw=1.5, color='mediumpurple') # mediumpurple, royalblue

            # Draw lines along the back edges
            x_bounds = [-20, 20]
            y_bounds = [-20, 20]
            z_bounds = [-5, 50]
            ax.plot([x_bounds[0], x_bounds[0]], [y_bounds[1], y_bounds[1]], [z_bounds[0], z_bounds[1]], color=(0.39, 0.39, 0.39, 0.5), lw=0.5)
            ax.plot([x_bounds[0], x_bounds[0]], [y_bounds[0], y_bounds[1]], [z_bounds[0], z_bounds[0]], color=(0.39, 0.39, 0.39, 0.5), lw=0.5)
            ax.plot([x_bounds[0], x_bounds[1]], [y_bounds[1], y_bounds[1]], [z_bounds[0], z_bounds[0]], color=(0.39, 0.39, 0.39, 0.5), lw=0.5)
            ax.plot([x_bounds[1], x_bounds[1]], [y_bounds[0], y_bounds[0]], [z_bounds[0], z_bounds[0]], color=(0.39, 0.39, 0.39, 0.5), lw=0.5)

            ax.set_title('Lorenz System Phase Space', pad=-0.4)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')
            ax.tick_params(axis='both', which='major')

            ax.set_yticks([-20, 0.0, 20])
            ax.set_xticks([-20, 0.0, 20])

            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.set_zlim(-5, 50)

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)
            plt.subplots_adjust(left=0.15, right=0.8, top=0.95, bottom=0.1) 
            plt.savefig('lorenz_static.pdf')
            plt.show()
            plt.close(fig)
            

        elif self.system_type == 'three_body':
            positions = solution.y[:9].reshape(3, 3, -1)  # Assuming the first 9 are positions

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.xaxis.set_pane_color((230/255, 230/255, 230/255, 1))
            ax.yaxis.set_pane_color((230/255, 230/255, 230/255, 1))
            ax.zaxis.set_pane_color((230/255, 230/255, 230/255, 1))

            colors = ['cornflowerblue', 'crimson', 'limegreen']
            alphas = [1.0, 0.8, 1.0]

            labels = ['Body 1', 'Body 2', 'Body 3']
            for i, color in enumerate(colors):
                ax.plot(positions[i][0], positions[i][1], positions[i][2], label=labels[i], color=color, alpha=alphas[i])
            
            # Draw lines along the back edges
            x_bounds = [-1.2, 1.2]
            y_bounds = [-0.4, 0.4]
            z_bounds = [-0.05, 0.052]
            ax.plot([x_bounds[0], x_bounds[0]], [y_bounds[1], y_bounds[1]], [z_bounds[0], z_bounds[1]], color=(0.39, 0.39, 0.39, 0.5), lw=0.5)
            ax.plot([x_bounds[0], x_bounds[0]], [y_bounds[0], y_bounds[1]], [z_bounds[0], z_bounds[0]], color=(0.39, 0.39, 0.39, 0.5), lw=0.5)
            ax.plot([x_bounds[0], x_bounds[1]], [y_bounds[1], y_bounds[1]], [z_bounds[0], z_bounds[0]], color=(0.39, 0.39, 0.39, 0.5), lw=0.5)
            ax.plot([x_bounds[1], x_bounds[1]], [y_bounds[0], y_bounds[0]], [z_bounds[0], z_bounds[0]], color=(0.39, 0.39, 0.39, 0.5), lw=0.5)


            ax.set_title('Three-body, Figure-8 Trajectory', pad=-0.4)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position', labelpad=7)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, handlelength=2, handletextpad=0.5, columnspacing=1)

            ax.set_yticks([-0.4, 0.0, 0.4])

            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.4, 0.4)
            ax.set_zlim(-0.05, 0.052)

            # plt.tight_layout()
            plt.subplots_adjust(left=0.15, right=0.8, top=0.95, bottom=0.1) 
            plt.savefig('three_body_figure8.pdf')
            plt.show()
            plt.close(fig)


    def plot_animation(
        self,
        positions,
        save_as_gif=False,
        filename="three_body_animation.gif",
        title="Animated Trajectories of Alpha Centauri System Bodies",
    ):
        """
        Animates the trajectories of bodies in a 3D plot.

        Parameters:
        - positions (array): The positions of the bodies to animate.
        - save_as_gif (bool): Whether to save the animation as a GIF.
        - filename (str): Filename for the GIF if saved.
        - title (str): Title of the plot.

        Returns:
        - None: Shows a matplotlib plot or saves it as a GIF.
        """

        fig = plt.figure() # figsize=(10, 8)
        ax = fig.add_subplot(111, projection="3d")

        lines = [
            ax.plot([], [], [], "-", lw=2, label=f"Body {i+1}")[0] for i in range(int(self.dim/3))
        ]

        x_max = np.max(positions[:, 0, :]) * 1.1
        x_min = np.min(positions[:, 0, :]) * 1.1

        y_max = np.max(positions[:, 1, :]) * 1.1
        y_min = np.min(positions[:, 1, :]) * 1.1

        z_max = np.max(positions[:, 2, :]) * 1.1
        z_min = np.min(positions[:, 2, :]) * 1.1

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
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

        ani = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(self.t_eval),
            interval=30,
            blit=True,
        )
        plt.legend()

        if save_as_gif:
            try:
                ani.save(filename, writer="pillow", fps=30)
            except Exception as e:
                logging.error(f"Failed to save GIF: {e}")

        plt.show()
        plt.close(fig)


    def generate_datasets(self, ode_system, bounds, num_initial_conditions, high_dim, high_dim_trans=True, include_nonlinear=True, plot=False):
        """
        Generates datasets by solving the ODE system with sampled initial conditions.

        Parameters:
        - ode_system (callable): ODE system to solve.
        - bounds (list): Bounds for sampling initial conditions.
        - num_initial_conditions (int): Number of initial conditions to sample.
        - high_dim (int): Dimension of the high-dimensional feature space.
        - high_dim_trans (bool): Whether to transform the data into high-dimensional space.
        - include_nonlinear (bool): Whether to include nonlinear terms in the transformation.
        - plot (bool): Whether to plot the trajectories.

        Returns:
        - None: Populates internal variables with the simulation results.
        """
        self.set_ode_system(ode_system)
        initial_conditions = self.sample_initial_conditions(num_initial_conditions, bounds)

        num_modes = 2 * self.dim
        xi = np.linspace(-1, 1, high_dim)
        all_modes = np.array([legendre(i)(xi) for i in range(num_modes)])

        num_linear_modes = int(num_modes/2)
        modes = all_modes if include_nonlinear else all_modes[:num_linear_modes]

        n_steps = len(self.t_eval)
        self.solutions = np.zeros((num_initial_conditions, n_steps, self.dim))
        self.high_dim_data_array = np.zeros((num_initial_conditions, n_steps, high_dim))

        # Iterate over the initial conditions, simulate 3Body data and transform it into a high-dimensional dataset
        for i in tqdm(range(num_initial_conditions), desc="Generating datasets"):
            self.initial_conditions = initial_conditions[i]
            solution = self.solve_ode().y[:self.dim]
            self.solutions[i] = solution.T

            if plot:
                positions = solution.reshape(3, 3, -1) if self.system_type == 'three_body' else solution.reshape(3, -1)[np.newaxis, :, :]
                self.plot_animation(positions=positions, title=f"Animated Trajectories - {self.system_type.capitalize()} System - IC number: {i}")

            if high_dim_trans:
                for idx, mode in enumerate(modes):
                    mod_idx = idx % self.dim 
                    for j in range(n_steps):
                        term = mode * self.solutions[i][j, mod_idx] if idx < self.dim else mode * self.solutions[i][j, mod_idx] ** 3
                        self.high_dim_data_array[i, j] = term

 
    def test_figure_eight(self, anim, static_plot):
        """
        Test the three-body problem with initial conditions known to form a stable figure-8 configuration
        with equal masses, and animate the results.
        """
        # Normalize G for the three-body problem
        self.G = 1.0  # Set gravitational constant to 1 for normalization
        self.m1 = self.m2 = self.m3 = 1.0  # Set masses to 1 for the Figure-8 configuration

        # Figure-8 initial conditions
        initial_conditions = np.array([
            -0.97000436,  0.24308753, 0,  # Body 1 position
             0.97000436, -0.24308753, 0,  # Body 2 position
             0.0,         0.0,        0,  # Body 3 position
             0.466203685, 0.43236573, 0,  # Body 1 velocity
             0.466203685, 0.43236573, 0,  # Body 2 velocity
            -0.93240737, -0.86473146, 0   # Body 3 velocity
        ])
        self.initial_conditions = initial_conditions
        self.set_ode_system(self.three_body_equations)
        result = self.solve_ode(plot_trajectory=static_plot)
        positions = result.y[:9].reshape(3, 3, -1)  # Reshape to (n_bodies, n_dimensions, n_points)
        if anim:
            self.plot_animation(positions, title="Figure-8 Configuration")
        return result.y, result.t


    def test_lorenz_stable(self, anim, static_plot):
        """
        Test the Lorenz system with initial conditions near a stable orbit within the attractor and animate the results.
        """
        # Stable initial conditions for the Lorenz system
        initial_conditions = np.array([1, 1, 1])

        self.initial_conditions = initial_conditions
        self.set_ode_system(self.lorenz_equations)
        result = self.solve_ode(plot_trajectory=static_plot)
        positions = result.y.reshape(1, 3, -1)  # Reshape for consistency with plotting method
        if anim:
            self.plot_animation(positions, title="Lorenz System Trajectory")
        return result.y, result.t

    
if __name__ == "__main__":
    """ Test of implementation """
    # Three-body problem
    sim_three_body = DataSim((0, 2.2), 1000, 'three_body')
    figure_eight_trajectory, time_points = sim_three_body.test_figure_eight(anim = False, static_plot=True)

    # Lorenz system
    sim_lorenz = DataSim((0, 30), 4000, 'lorenz', dim=3)
    lorenz_trajectory, time_points = sim_lorenz.test_lorenz_stable(anim = False, static_plot=True)


    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    n_ics = 10  # Number of initial conditions


    """ Three-Body Problem """
    t_span_three_body = (0, 365 * 86400)
    num_points_three_body = 365

    # Bounds setup for the three-body problem
    num_bodies = 3
    position_bounds = [(-1e6, 1e6)] * 3  # x, y, z bounds for one body 
    velocity_bounds = [(-3e4, 3e4)] * 3  # u, v, w bounds for one body 
    three_body_bounds = position_bounds * num_bodies + velocity_bounds * num_bodies

    three_body_sim = DataSim(t_span_three_body, num_points_three_body, system_type='three_body', dim=9)
    three_body_sim.generate_datasets(three_body_sim.three_body_equations, three_body_bounds, n_ics, high_dim=128, plot=True)

    three_body_sim_high_dim_data = three_body_sim.high_dim_data_array
    three_body_sim_low_dim_data = three_body_sim.solutions


    """ Lorenz System """
    t_span_lorenz = (0, 5)
    num_points_lorenz =  1000

    # Bounds setup for the Lorenz system
    ic_means = np.array([0, 0, 25])
    ic_widths = 2 * np.array([36, 48, 41])
    lorenz_bounds = [(mean - width/2, mean + width/2) for mean, width in zip(ic_means, ic_widths)]


    lorenz_sim = DataSim(t_span_lorenz, num_points_lorenz, system_type='lorenz', dim=3)
    lorenz_sim.generate_datasets(lorenz_sim.lorenz_equations, lorenz_bounds, n_ics, high_dim=50, plot=True)

    lorenz_sim_high_dim_data = lorenz_sim.high_dim_data_array
    lorenz_sim_low_dim_data = lorenz_sim.solutions
