import jax
import sys
from jax.lib import xla_bridge

sys.path.append('../')
from lorenzData import get_lorenz_data
from data_utils import create_jax_batches_factory
from trainer import SINDy_trainer

create_jax_batches = create_jax_batches_factory(second_order=False)

if __name__ == "__main__":

    seed = int(sys.argv[1])

    #Check if jax is using GPU
    print(f"JAX is using: {xla_bridge.get_backend().platform}")
    devices = jax.devices()
    print(f"Number of devices: {len(devices)}")
    for device in devices:
        print(device)
    print("Performing simulation for seed: ", seed)

    # Set up training and validation data sets as arrays
    n_ics_training = 2048
    n_ics_validation = 20

    noise_strength = 1e-6
    batch_size_training = 8000
    batch_size_validation = 5000

    training_data = get_lorenz_data(n_ics_training, noise_strength)
    train_loader = create_jax_batches(training_data, batch_size_training)


    validation_data = get_lorenz_data(n_ics_validation, noise_strength)
    validation_loader = create_jax_batches(validation_data, batch_size_validation)

    # Define hyperparameters
    input_dim = 128
    latent_dim = 3
    poly_order = 3
    widths = [64, 32]

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
        'check_val_every_n_epoch': 400
    }

    # Merge dictionaries
    params = {**hparams, **trainer_params}

    # Initialize trainer
    trainer = SINDy_trainer(**params)

    trainer.train_model(train_loader, validation_loader, num_epochs=10001, final_epochs=1001)
