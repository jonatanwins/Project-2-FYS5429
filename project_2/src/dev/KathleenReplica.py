# %% [markdown]
# This notebook reproduces the Lorentz results from [Champion et. al](https://www.pnas.org/doi/full/10.1073/pnas.1906995116). The data generation is specified in the  [appendix](https://www.pnas.org/action/downloadSupplement?doi=10.1073%2Fpnas.1906995116&file=pnas.1906995116.sapp.pdf) of [Champion et. al](https://www.pnas.org/doi/full/10.1073/pnas.1906995116), and is restated here. 
# 
# 
# The data was syntheticaly generated using the governing lorentz
# equations
# 
# 
# $$
# \begin{aligned}
# \dot{z}_1 =& \sigma (z_2 - z_1) \\
# 
# 
# 
# \dot{z}_2 =& z_1 (\rho - z_3) - z_2 \\
# 
# 
# 
# \dot{z}_3 =& z_1 z_2 - \beta z_3 \\
# 
# \end{aligned}
# $$
# 
# with the standard paramater values $\sigma = 10$, $\beta = \frac{8}{3}$ and $\rho = 28$

# %% [markdown]
# This data is then trnsformed using the first 6 legendre polynomials.
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

plt.style.use("../plot_utils/plot_settings.mplstyle")

x = np.linspace(-1, 1, 128)

# Plot the first 6 Legendre polynomials
for n in range(6):
    u_n = legendre(n)
    y = u_n(x)
    # Plot the polynomial
    #make the 
    plt.plot(x, y, label=f'$u_{n}(x)$', marker='o', markersize=2)


plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('First 6 Legendre Polynomials')
plt.legend()
plt.grid(True)



# %% [markdown]
# Each datapoint $\boldsymbol{z} = [z_1, z_2, z_3]$ is mapped to a $\boldsymbol{x}$ by

# %% [markdown]
# $$
# \boldsymbol{x}(t) = \boldsymbol{u_1} z_1(t) +  \boldsymbol{u_2}z_2(t) +  \boldsymbol{u_3}z_3(t) +  \boldsymbol{u_4}z_1(t)^3 +  \boldsymbol{u_5}z_2(t)^3 +  \boldsymbol{u_6} z_3(t)^3 
# $$

# %% [markdown]
# Our $\boldsymbol{u}_n \in \mathbb{R}^{128}$ vectors correspond precisely to the ones depicted above; these are constructed by sampling the $n$-th Legendre polynomial at 128 equally spaced points within the interval $[-1, 1]$.

# %% [markdown]
# From the [appendix](https://www.pnas.org/action/downloadSupplement?doi=10.1073%2Fpnas.1906995116&file=pnas.1906995116.sapp.pdf) 
# "To generate our data set, we simulate the system with 2048 initial conditions for the training set, 20 for the validation set, and 100 for the test set. For each initial condition we integrate the system forward in time from t = 0 to t = 5 with a spacing of ∆t = 0.02 to obtain 250 samples. Initial conditions 93 are chosen randomly from a uniform distribution over $z_1$ ∈ [−36, 36], $z_2$ ∈ [−48, 48], $z_3$ ∈ [−16, 66]. This results in a training 94 set with 512,000 total samples"

# %% [markdown]
# The functions for generating such data is taken from [Kathleens github](https://github.com/kpchamp/SindyAutoencoders) and simply wrapped in some simple functions to create pytorch dataloaders.

# %%
#GPU STUFF
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


import __main__
#set __spec__ to None to avoid breaking the multiprocessing
#check if main has attribute __spec__ and set to none
if hasattr(__main__, '__spec__'):
    __main__.__spec__ = None
# #do the same for __file__
# if hasattr(__main__, '__file__'):
#     __main__.__file__ = None
# #do the same for __loader__
# if hasattr(__main__, '__loader__'):
#     __main__.__loader__ = None


# %%
import sys
sys.path.append('../')
from lorenzTorchData import get_lorenz_dataloader, get_random_sample # type: ignore -goofy linitng issue


# generate training, validation, testing data
noise_strength = 1e-6

# training_data = get_lorenz_dataloader(204, noise_strength=noise_strength, batch_size=8000)
# validation_data = get_lorenz_dataloader(20, train=False, noise_strength=noise_strength, batch_size=800)
# testing_data = get_lorenz_dataloader(100, train=False, noise_strength=noise_strength, batch_size= 800)

from dummy_data import get_dummy_dataloader

num_datapoints = 1000000  # 1 million data points
num_datapoints_val = 10000 
num_features = 128
batch_size = 64
shuffle = True
num_workers = 0
persistent_workers = False

training_data = get_dummy_dataloader(num_datapoints, num_features, batch_size, shuffle, num_workers, persistent_workers)
validation_data = get_dummy_dataloader(num_datapoints_val, num_features, batch_size, shuffle, num_workers, persistent_workers)



# %%
from jax import random
from sindy_utils import library_size
from trainer import Trainer

# Set random key and hyperparameters
key = random.PRNGKey(0)
input_dim = 128
latent_dim = 3
poly_order = 3
widths = [64, 32]

# Get example input from training_data loader
x, dx = get_random_sample(training_data)

# Loss function parameters
loss_params = {
    'latent_dim': latent_dim,
    'poly_order': poly_order,
    'include_sine': False,
    'weights': (1, 1, 40, 1)
}

# Optimizer and logger hyperparameters
optimizer_hyperparams = {'optimizer' : "adam"}
logger_params = {}

# Initialize trainer
trainer = Trainer(
    input_dim=input_dim,
    latent_dim=latent_dim,
    poly_order=poly_order,
    widths=widths,
    exmp_input=x,
    optimizer_hparams=optimizer_hyperparams,
    loss_params=loss_params,  # Pass the loss parameters here
    seed=42,
    logger_params=logger_params,
    check_val_every_n_epoch=1000,
    update_mask_every_n_epoch=500,
    debug=True
)

# %%
trainer.train_model(training_data, validation_data, num_epochs=10000)

# %%
sys.path.append("../plot_utils")
from metrics import RunMetrics # type: ignore -goofy linitng issue
from plot_metrics import plot_metrics # type: ignore -goofy linitng issue

log_dir = trainer.log_dir

metrics = RunMetrics(log_dir)

# Plot a single run
plot_metrics(metrics, metric_names=["train/loss", "val/loss"], title="Loss Metrics")


# %% [markdown]
# ## Get the sparse representation of the model ($\Xi$ )

# %%
mask = trainer.state.mask
xi = trainer.state.params['sindy_coefficients']
xi = xi * mask

print(xi)

# %%
from sindy_utils import get_expression

expression = get_expression(xi, poly_order=poly_order, include_sine=False)
#print(expression)


