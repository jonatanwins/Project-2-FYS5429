# not using package style imports to avoid easier use in google colab
from UvAutils.Basetrainer import TrainerModule
from typing import Any, Callable, Dict, Tuple, Iterator, Optional
import jax.numpy as jnp
from flax.training import train_state
from jax import value_and_grad
from flax import linen as nn
from jax import random


class TrainState(train_state.TrainState):
    rng: Any = None,
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
        loss_fn: Callable = lambda params, batch, model, mask: (0, None)
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
        Callable[[TrainState, Tuple], Tuple[TrainState, Tuple]],
        Callable[[TrainState, Tuple], Tuple[TrainState, Tuple]],
    ]:
        """
        This function is used to create the train_step and eval_step functions which
        calculate the loss and update the model parameters ect. for one batch of data.


        Returns:
            Tuple[Callable[[TrainState, Any], Tuple[TrainState, Dict]],
                  Callable[[TrainState, Any], Tuple[TrainState, Dict]]]:
            train_step, eval_step

        """

        def train_step(state: TrainState, batch: Tuple):

            def loss_fn(params): return self.loss_fn(params,
                                                     batch, self.model, state.mask)

            val_grad_fn = value_and_grad(loss_fn, has_aux=True)
            (loss, metrics), grads = val_grad_fn(state.params)
            state = state.apply_gradients(grads=grads)

            return state, metrics

        def eval_step(state: TrainState, batch: Any):

            (loss, metrics) = self.loss_fn(
                state.params, batch, self.model, state.mask)

            return metrics

        return train_step, eval_step
