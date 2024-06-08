import jax
from jax import random
from jax.lib import xla_bridge
import sys

sys.path.append('../')

#from sindyLibrary import library_size
from trainer import SINDy_trainer
from lorenzData import get_lorenz_train_data, LorenzDataset
from data_utils import get_random_sample, JaxDocsLoader


seed = int(sys.argv[1])
print(f"Seed: {seed}")

print(f"JAX is using: {xla_bridge.get_backend().platform}")
devices = jax.devices()
print(f"Number of devices: {len(devices)}")
for device in devices:
    print(device)

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
training_data_set = LorenzDataset(training_data)
training_data_loader = JaxDocsLoader(training_data_set, batch_size=batch_size_training, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)

validation_data = get_lorenz_train_data(n_ics_validation, noise_strength)
validation_data_set = LorenzDataset(validation_data)
validation_data_loader = JaxDocsLoader(validation_data_set, batch_size=batch_size_validation, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)

# # Set random key and hyperparameters
# key = random.PRNGKey(0)
input_dim = 128
latent_dim = 3
poly_order = 3
widths = [64, 32]

initial_epochs = 10001
final_epochs = 1001

# Get example input from training_data loader
x, dx = get_random_sample(training_data_loader)

# Define hyperparameters dictionary
hparams = {
    'input_dim': input_dim,
    'latent_dim': latent_dim,
    'poly_order': poly_order,
    'widths': widths,
    'activation': 'sigmoid',
    'weight_initializer': 'xavier_uniform',
    'bias_initializer': 'zeros',
    'optimizer_hparams': {'optimizer': "adam"},
    'include_sine': False,  # Extracted from loss_params
    'loss_weights': (1, 1e-4, 0, 1e-5),  # Extracted from loss_params['weights']
    'seed': seed,
    'update_mask_every_n_epoch': 500,
    'coefficient_threshold': 0.1,
    'regularization': True,  # Added default value
    'second_order': False,  # Added default value
    'include_constant': True  # Added default value
}

# Define other parameters dictionary
trainer_params = {
    'exmp_input': x,
    'logger_params': {},
    'enable_progress_bar': True,
    'debug': False,
    'check_val_every_n_epoch': 100
}

# Merge dictionaries
params = {**hparams, **trainer_params}

# Initialize trainer
trainer = SINDy_trainer(**params)



trainer.train_model(training_data_loader, validation_data_loader, num_epochs=initial_epochs, final_epochs=final_epochs)