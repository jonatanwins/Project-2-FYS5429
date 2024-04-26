# not using package style imports to avoid easier use in google colab
from UvAutils.Basetrainer import TrainerModule
from typing import Any, Callable, Dict, Tuple, Iterator, Optional
import jax.numpy as jnp
from flax.training import train_state
from jax import value_and_grad
from flax import linen as nn


class TrainState(train_state.TrainState):
    # batch_stats: Any = None
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
        loss_fn: Callable[
            [TrainState, Any], Tuple[jnp.ndarray, Dict]
        ] = lambda state, batch: (0, {}),
    ):
        super().__init__(
            model_class,
            model_hparams,
            optimizer_hparams,
            exmp_input,
            seed,
            logger_params,
            enable_progress_bar,
            debug,
            check_val_every_n_epoch,
        )
        self.update_mask_every_n_epoch = update_mask_every_n_epoch
        self.loss_fn = loss_fn
        self.config.update(
            {"update_mask_every_n_epoch": update_mask_every_n_epoch, "loss_fn": loss_fn}
        )

    def create_functions(
        self,
    ) -> Tuple[
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
    ]:
        """
        This function is used to create the train_step and eval_step functions which
        calculate the loss and update the model parameters ect. for one batch of data.


        Returns:
            Tuple[Callable[[TrainState, Any], Tuple[TrainState, Dict]],
                  Callable[[TrainState, Any], Tuple[TrainState, Dict]]]:
            train_step, eval_step

        """

        def train_step(state: TrainState, batch: Any):

            val_grad_fn = value_and_grad(self.loss_fn, has_aux=True)
            (loss, metrics), grad = val_grad_fn(state, batch)
            optimizer = self.optimizer.apply_gradient(grad)
            state = state.apply(optimizer=optimizer)

            return state, metrics

        def eval_step(state: TrainState, batch: Any):

            (loss, metrics) = self.loss_fn(state, batch)

            return metrics

        return train_step, eval_step

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Optional[Iterator] = None,
        num_epochs: int = 500,
    ) -> Dict[str, Any]:
        """
        Starts a training loop for the given number of epochs.

        Args:
          train_loader: Data loader of the training set.
          val_loader: Data loader of the validation set.
          test_loader: If given, best model will be evaluated on the test set.
          num_epochs: Number of epochs for which to train the model.

        Returns:
          A dictionary of the train, validation and evt. test metrics for the
          best model on the validation set.
        """
        # Create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Prepare training loop
        self.on_training_start()
        best_eval_metrics = None
        for epoch_idx in self.tracker(range(1, num_epochs + 1), desc="Epochs"):
            train_metrics = self.train_epoch(train_loader)
            self.logger.log_metrics(train_metrics, step=epoch_idx)
            self.on_training_epoch_end(epoch_idx)
            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix="val/")
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loader)
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f"eval_epoch_{str(epoch_idx).zfill(3)}", eval_metrics)
                # Save best model
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(step=epoch_idx)
                    self.save_metrics("best_eval", eval_metrics)
            ##NEW LINES- only difference from UvA code in this method
            if epoch_idx % self.update_mask_every_n_epoch == 0:
                self.state.mask = update_mask(self.state.params["sindy_coefficients"])
            ##END NEW LINES
        # Test best model if possible
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(test_loader, log_prefix="test/")
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics("test", test_metrics)
            best_eval_metrics.update(test_metrics)
        # Close logger
        self.logger.finalize("success")
        return best_eval_metrics
