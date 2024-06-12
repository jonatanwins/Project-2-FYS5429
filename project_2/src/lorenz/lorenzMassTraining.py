import jax
import gc
from jax.lib import xla_bridge
from lorenzData import get_lorenz_data, get_lorenz_OutOfDistro_data
from data_utils import create_jax_batches_factory
from trainer import SINDy_trainer

jax.config.update("jax_enable_x64", True)

def run_simulation(seed, activation='sigmoid', lr_schedule=False):
    create_jax_batches = create_jax_batches_factory(second_order=False)

    # Check if JAX is using GPU
    print(f"JAX is using: {xla_bridge.get_backend().platform}")
    devices = jax.devices()
    print(f"Number of devices: {len(devices)}")
    for device in devices:
        print(device)
    print("Performing simulation for seed: ", seed)

    # Set up training and validation data sets as arrays
    n_ics_training = 2048
    n_ics_validation = 20
    n_ics_testing = 100

    noise_strength = 1e-6
    batch_size_training = 8000
    batch_size_validation = 5000
    batch_size_testing = 5000

    training_data = get_lorenz_data(n_ics_training, noise_strength)
    train_loader = create_jax_batches(training_data, batch_size_training)

    validation_data = get_lorenz_data(n_ics_validation)  # no noise for val data
    validation_loader = create_jax_batches(validation_data, batch_size_validation)

    out_dist_testing_data = get_lorenz_OutOfDistro_data(n_ics_testing)  # no noise for testing
    out_dist_testing_loader = create_jax_batches(out_dist_testing_data, batch_size_testing)

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
        'activation': activation,
        'weight_initializer': 'xavier_uniform',
        'bias_initializer': 'zeros',
        'optimizer_hparams': {'optimizer': "adam", 'lr_schedule': lr_schedule}, 
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
        'logger_params': {'logger_name': 'KathleenReplicas'},
        'enable_progress_bar': True,
        'debug': False,
        'check_val_every_n_epoch': 400
    }

    # Merge dictionaries
    params = {**hparams, **trainer_params}

    # Initialize trainer
    trainer = SINDy_trainer(**params)

    # Train model
    trainer.train_model(train_loader, validation_loader, out_dist_testing_loader, num_epochs=initial_epochs, final_epochs=final_epochs)

    # Clean up to free memory
    del training_data, train_loader, validation_data, validation_loader, out_dist_testing_data, out_dist_testing_loader, trainer
    gc.collect()

if __name__ == "__main__":
    # Run simulations for different seeds
    for seed in [99, 444, 1414, 2020, 2001]:
        run_simulation(seed, activation='sigmoid')
    for seed in [707, 747, 777, 1881, 2019]:
        run_simulation(seed, activation='sigmoid', lr_schedule=True)
