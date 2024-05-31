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
%matplotlib inline
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
import jax
from jax.lib import xla_bridge

print(f"JAX is using: {xla_bridge.get_backend().platform}")
devices = jax.devices()
print(f"Number of devices: {len(devices)}")
for device in devices:
    print(device)


# %%
import sys
sys.path.append('../')
from lorenzData import get_lorenz_test_data, get_lorenz_train_data
from lorenzDataSets import LorenzTrainDataset, LorenzTestDataset
from data_utils import get_random_sample, JaxDocsLoader # type: ignore -goofy linitng issue

# Define all arguments as variables
n_ics_training = 2048
n_ics_validation = 20
#n_ics_testing = 100

noise_strength = 1e-6
batch_size_training = 8000
batch_size_validation = 800
#batch_size_testing = 800

num_workers = 0
pin_memory = False
drop_last = False
timeout = 0
worker_init_fn = None

#get_lorenz_train_data gets x, dx. Which is all we need for the losses/training. z, dz from
# get_lorenz_test_data could be interesting. But, not necessary for now.

training_data = get_lorenz_train_data(n_ics_training, noise_strength)
training_data_set = LorenzTrainDataset(training_data)
training_data_loader = JaxDocsLoader(training_data_set, batch_size=batch_size_training, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)

validation_data = get_lorenz_train_data(n_ics_validation, noise_strength)
validation_data_set = LorenzTrainDataset(validation_data)
validation_data_loader = JaxDocsLoader(validation_data_set, batch_size=batch_size_validation, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)



# %%
from jax import random
from sindy_utils import library_size
from trainer import SINDy_trainer

# Set random key and hyperparameters
key = random.PRNGKey(0)
input_dim = 128
latent_dim = 3
poly_order = 3
widths = [64, 32]

# Get example input from training_data loader
x, dx = get_random_sample(training_data_loader)

# Define hyperparameters dictionary
hparams = {
    'input_dim': input_dim,
    'latent_dim': latent_dim,
    'poly_order': poly_order,
    'widths': widths,
    'activation': 'tanh',
    'weight_initializer': 'xavier_uniform',
    'bias_initializer': 'zeros',
    'optimizer_hparams': {'optimizer': "adam"},
    'loss_params': {
        'latent_dim': latent_dim,
        'poly_order': poly_order,
        'include_sine': False,
        'weights': (1, 1e-4, 1e-6, 1e-5)
    },
    'seed': 42,
    'update_mask_every_n_epoch': 500,
    'coefficient_threshold': 0.1
}

# Define other parameters dictionary
trainer_params = {
    'exmp_input': x,
    'logger_params': {},
    'enable_progress_bar': True,
    'debug': False,
    'check_val_every_n_epoch': 100
}


# %%
# Merge dictionaries
params = {**hparams, **trainer_params}

# Initialize trainer
trainer = SINDy_trainer(**params)

# %%
trainer.train_model(training_data_loader, validation_data_loader, num_epochs=9000, final_epochs=1000)

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
xi_og = trainer.state.params['sindy_coefficients']
xi_og

# %%
from sindy_utils import get_expression

expression = get_expression(xi, poly_order=poly_order, include_sine=False)
#print(expression)


