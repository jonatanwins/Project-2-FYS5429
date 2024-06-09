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
from lorenzData import get_lorenz_test_data, get_lorenz_train_data, create_jax_batches


# Define all arguments as variables
n_ics_training = 2
n_ics_validation = 2
#n_ics_testing = 100

noise_strength = 1e-6
batch_size_training = 8
batch_size_validation = 8
#batch_size_testing = 800

num_workers = 0
pin_memory = False
drop_last = False
timeout = 0
worker_init_fn = None

#get_lorenz_train_data gets x, dx. Which is all we need for the losses/training. z, dz from
# get_lorenz_test_data could be interesting. But, not necessary for now.

training_data = get_lorenz_train_data(n_ics_training, noise_strength)
train_loader = create_jax_batches(training_data, batch_size_training)

#training_data_set = LorenzDataset(training_data)
#training_data_loader = JaxDocsLoader(training_data_set, batch_size=batch_size_training, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)

validation_data = get_lorenz_train_data(n_ics_validation, noise_strength)
validation_loader = create_jax_batches(validation_data, batch_size_validation)

#validation_data_set = LorenzDataset(validation_data)
#validation_data_loader = JaxDocsLoader(validation_data_set, batch_size=batch_size_validation, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn)



# %%
input_dim = 128
latent_dim = 3
poly_order = 3
widths = [64, 32]

seed = 69 # OBS: This seed is used for the initial weights of the model

initial_epochs = 10001
final_epochs = 1001

# Get example input from training_data loader
x, dx = train_loader[0]

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


# %%
from trainer import SINDy_trainer

# Merge dictionaries
params = {**hparams, **trainer_params}

# Initialize trainer
trainer = SINDy_trainer(**params)

# %%

trainer.train_model(train_loader, validation_loader, num_epochs=10001, final_epochs=1000)
