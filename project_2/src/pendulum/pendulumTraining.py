# %% [markdown]
# This notebook reproduces the nonlinear Pendulum results from [Champion et. al](https://www.pnas.org/doi/full/10.1073/pnas.1906995116). The data generation is specified in the  [appendix](https://www.pnas.org/action/downloadSupplement?doi=10.1073%2Fpnas.1906995116&file=pnas.1906995116.sapp.pdf) of [Champion et. al](https://www.pnas.org/doi/full/10.1073/pnas.1906995116), and is restated here.

# %% [markdown]
# The nonlinear pendulum equation is given by:
# 
# $$
# \ddot{z} = -\sin(z)
# $$
# 
# Here $z$ denotes the angle between the vertical and the pendulum.

# %% [markdown]
# As per the appendix, "We generate synthetic video of the pendulum in two spatial dimensions by creating high-dimensional snapshots given by"
# 
# $$
# x(y_1, y_2, t) = \exp \left( -20 \left(  (y_1 - \cos(z(t)) - \pi/2)\right)^2 + (y_2  - \sin(z(t)) - \pi/2)^2 \right)
# $$
# 
# The spatial discretization is $y_1, y_2 \in [-1.5, 1.5]$ with 51 grid points in each dimension, resulting in snapshots $x(t) \in \mathbb{R}^{2601}$.

# %% [markdown]
# To generate a training set, we simulate the pendulum equation from 100 randomly chosen initial conditions with $z(0) \in [-\pi, \pi]$ and $\dot{z}(0) \in [-2.1, 2.1]$. The initial conditions are selected from a uniform distribution in the specified range but are restricted to conditions for which the pendulum does not have enough energy to do a full loop. This condition is determined by checking that $|\dot{z}(0)^2/2 - \cos(z(0))| \le 0.99$.

# %%
import numpy as np
import matplotlib.pyplot as plt
 

plt.style.use("../plot_utils/plot_settings.mplstyle")

# Import necessary functions for data generation
from pendulumData import get_pendulum_train_data, get_pendulum_test_data
from pendulumDataSets import PendulumTrainDataset, PendulumTestDataset
from data_utils import get_random_sample, JaxDocsLoader

# Define all arguments as variables
n_ics_training = 100
n_ics_validation = 20

noise_strength = 1e-6
batch_size_training = 800
batch_size_validation = 80

num_workers = 0
pin_memory = False
drop_last = False
timeout = 0
worker_init_fn = None

# Get training data
training_data = get_pendulum_train_data(n_ics_training, noise_strength)
training_data_set = PendulumTrainDataset(training_data)
training_data_loader = JaxDocsLoader(training_data_set, batch_size=batch_size_training, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)

# Get validation data
validation_data = get_pendulum_train_data(n_ics_validation, noise_strength)
validation_data_set = PendulumTrainDataset(validation_data)
validation_data_loader = JaxDocsLoader(validation_data_set, batch_size=batch_size_validation, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)

# %%
from jax import random
from sindy_utils import library_size
from trainer import SINDy_trainer

# Set random key and hyperparameters
key = random.PRNGKey(0)
input_dim = 2601
latent_dim = 2
poly_order = 3
widths = [128, 64]

# Get example input from training_data loader
x, dx = get_random_sample(training_data)

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
    'seed': 42
}

# Define other parameters dictionary
trainer_params = {
    'exmp_input': x,
    'logger_params': {},
    'enable_progress_bar': True,
    'debug': False,
    'check_val_every_n_epoch': 100,
    'update_mask_every_n_epoch': 500,
    'coefficient_threshold': 0.1
}

# Merge dictionaries
params = {**hparams, **trainer_params}

# Initialize trainer
trainer = SINDy_trainer(**params)

# %%
trainer.train_model(training_data, validation_data, num_epochs=5000, final_epochs=1000)

# %%
sys.path.append("../plot_utils")
from metrics import RunMetrics # type: ignore -goofy linting issue
from plot_metrics import plot_metrics # type: ignore -goofy linting issue

log_dir = trainer.log_dir

metrics = RunMetrics(log_dir)

# Plot a single run
plot_metrics(metrics, metric_names=["train/loss", "val/loss"], title="Loss Metrics")

# %% [markdown]
# ## Get the sparse representation of the model ($\Xi$)

# %%
mask = trainer.state.mask
xi = trainer.state.params['sindy_coefficients']
xi = xi * mask

print(xi)

