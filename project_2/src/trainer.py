# not using package style imports to avoid easier use in google colab
from UvAutils.Basetrainer import TrainerModule
from sindy_autoencoder import SindyAutoencoder
from typing import Any, Callable, Dict, Tuple
import jax.numpy as jnp
from flax.training import train_state
from jax import value_and_grad
from flax import nn



class TrainState(train_state.TrainState):
    #batch_stats: Any = None
    rng: Any = None
    mask: jnp.ndarray = None



def update_mask(coefficients, threshold=0.1):
    return jnp.where(jnp.abs(coefficients) >= threshold, 1, 0)


class Trainer(TrainerModule):
    def __init__(
        self,
        model_class: nn.Module,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any],
        exmp_input: Any,
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 500,
        update_mask_every_n_epoch: int = 500,
        loss_fn: Callable[[TrainState, Any], Tuple[jnp.ndarray, Dict]] = lambda state, batch: (0, {})
    ):
        super().__init__(model_class, model_hparams, optimizer_hparams, exmp_input, seed, logger_params, enable_progress_bar, debug, check_val_every_n_epoch)
        self.update_mask_every_n_epoch = update_mask_every_n_epoch
        self.loss_fn = loss_fn
        self.config.update({'update_mask_every_n_epoch': update_mask_every_n_epoch, 'loss_fn': loss_fn})


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
    
            val_grad_fn = value_and_grad(self.loss_fn, has_aux=True)
            (loss, metrics), grad = val_grad_fn(state, batch)
            optimizer = self.optimizer.apply_gradient(grad)
            state = state.apply(optimizer=optimizer)



            return state, metrics

        def eval_step(state: TrainState,
                      batch: Any):
            
            (loss, metrics) = self.loss_fn(state, batch)

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

