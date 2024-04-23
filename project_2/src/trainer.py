# not using package style imports to avoid easier use in google colab
from UvAutils.Basetrainer import TrainerModule
from sindy_autoencoder import SindyAutoencoder
from typing import Any, Callable, Dict, Tuple
import jax.numpy as jnp
from flax.training import train_state

# import our loss functions
from loss import loss_recon, loss_dynamics_dz, loss_regularization


class TrainState(train_state.TrainState):
    #batch_stats: Any = None
    rng: Any = None
    mask: jnp.ndarray = None



def update_mask(coefficients, threshold=0.1):
    return jnp.where(jnp.abs(coefficients) >= threshold, 1, 0)


class Trainer(TrainerModule):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, **kwargs):
        super().__init__(model, train_loader, val_loader,
                         optimizer, loss_fn, device, **kwargs)

    def create_functions(self) -> Tuple[Callable[[TrainState, Any], Tuple[TrainState, Dict]],
                                        Callable[[TrainState, Any], Tuple[TrainState, Dict]]]:
        """
        This function is used to create the train_step and eval_step functions which
        calculate the loss and update the model parameters ect. for one batch of data.


        Returns:
            Tuple[Callable[[TrainState, Any], Tuple[TrainState, Dict]],
                  Callable[[TrainState, Any], Tuple[TrainState, Dict]]]:
            train_step, eval_step

        """
        def train_step(state: TrainState,
                       batch: Any):
            metrics = {}

            mask = state.mask

                        
            masked_coefficients = state.params['sindy_coefficients'] *  mask
            
            state = state.replace(mask=mask)
            
           
            return state, metrics

        def eval_step(state: TrainState,
                      batch: Any):
            metrics = {}
            return metrics

        return train_step, eval_step


if __name__ == "__main__":
    # how one might use the trainer

    model_hparams = {'input_dim': 2,
                     'latent_dim': 2, 'widths': [60, 40, 20, 3], }
    optimizer_hparams = {'learning_rate': 1e-3}

    # 12 examples of 124 dimensions. Example input should not be created manualy like this, but rather fetched from a pytorch dataloader,
    # that way batchdim ect is always correct
    exmp_input = jnp.ones((12, 124))



    trainer = TrainerModule(
        SindyAutoencoder,  model_hparams, optimizer_hparams, exmp_input)
    # trainer.train(train_loader, val_loader, test_loader: Optional, num_epochs)  #data loader object should be created my jouval and daniel.
    # now everything is stored in logger, that supposedly can plot the matrics nicely for us?
    # trainer.logger.plot()? -this is just a wonky suggestion

