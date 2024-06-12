import jax
from jax.lib import xla_bridge
from pendulumData import get_pendulum_data
import sys
sys.path.append('../')
from data_utils import create_jax_batches_factory
from trainer import SINDy_trainer

#jax.config.update('enable_x64', True)


if __name__ == "__main__":

    seed = int(sys.argv[1])

    print(f"JAX is using: {xla_bridge.get_backend().platform}")
    devices = jax.devices()
    print(f"Number of devices: {len(devices)}")
    for device in devices:
        print(device)

    # Create JAX data loaders
    create_jax_batches = create_jax_batches_factory(second_order=True)


    # Set up training and validation data sets as arrays
    n_ics_training = 100
    n_ics_validation = 10

    batch_size_training = 1024
    batch_size_validation = 250

    training_data = get_pendulum_data(n_ics_training)
    train_loader = create_jax_batches(training_data, batch_size_training)


    validation_data = get_pendulum_data(n_ics_validation)
    validation_loader = create_jax_batches(validation_data, batch_size_validation)


    # Define hyperparameters
    input_dim = 2601
    latent_dim = 1
    poly_order = 3
    widths = [128, 64, 32]

    initial_epochs = 10001
    final_epochs = 1001

    # Get example input from training_data loader
    x, dx, ddx = train_loader[0]

    # Define hyperparameters dictionary
    hparams = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'poly_order': poly_order,
        'widths': widths,
        'activation': 'sigmoid',
        'weight_initializer': 'xavier_uniform',
        'bias_initializer': 'zeros',
        'optimizer_hparams': {'optimizer': "adam", "lr": 1e-4},
        'include_sine': True, #important  
        'loss_weights': (1, 5e-4, 5e-5, 1e-5),  # Note different weights than Lorenz
        'seed': seed, #importiante 
        'update_mask_every_n_epoch': 500,
        'coefficient_threshold': 0.1,
        'regularization': True,  
        'second_order': True,   #second order True WIIIIIIII
        'include_constant': True  
    }

    # Define other parameters dictionary
    trainer_params = {
        'exmp_input': x,
        'logger_params': {'logger_name': 'KathleenReplicas'},
        'enable_progress_bar': True,
        'debug': False,
        'check_val_every_n_epoch': 100,
    }

    # Merge dictionaries
    params = {**hparams, **trainer_params}

    # Initialize trainer
    trainer = SINDy_trainer(**params)

    trainer.train_model(train_loader, validation_loader, num_epochs=initial_epochs, final_epochs=final_epochs)