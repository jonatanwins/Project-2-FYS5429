"""
Large parts of this module was created by Phillip Lippe (Revision cf18eb5d, 2022), and is part of 
Guide 4: Research Projects with JAX https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html
which is part of UVA DEEP LEARNING COURSE https://uvadlc.github.io/
"""

from autoencoder import Autoencoder, Encoder, Decoder

import os
from typing import Any, Optional, Tuple, Iterator, Dict, Callable
import json
import time
from tqdm import tqdm
from copy import copy
from collections import defaultdict
import pickle

import jax
from jax import jit, random, value_and_grad
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import jax.numpy as jnp

from pytorch_lightning.loggers import TensorBoardLogger

from loss import loss_fn_factory
from sindy_utils import library_size


class TrainState(train_state.TrainState):
    mask: jnp.ndarray = None
    rng: Any = None


@jit
def update_mask(coefficients, threshold=0.1):
    return jnp.where(jnp.abs(coefficients) >= threshold, 1, 0)


class SINDy_trainer:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        poly_order: int,
        widths: list,
        exmp_input: Any,
        activation: str = 'tanh',
        weight_initializer: str = 'xavier_uniform',
        bias_initializer: str = 'zeros',
        loss_factory: Callable[[int, int, bool, tuple, bool], Callable] = loss_fn_factory,
        optimizer_hparams: Dict[str, Any] = {},
        loss_params: Dict[str, Any] = {},
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 500,
        update_mask_every_n_epoch: int = 500,
        coefficient_threshold: float = 0.1, 
    ):
        """
        Trainer module for holding all parts required for training a model. 
        This includes the model, optimizer, loss function, and the training loop.

        Args:
            input_dim (int): Input dimension of the data.
            latent_dim (int): Dimension of the latent space.
            poly_order (int): Polynomial order for the SINDy library.
            widths (list): List of layer widths for the encoder and decoder. Assumes the same for both.
            exmp_input (Any): Example input to initialize the autoencoder model.
            activation (str): Activation function for the encoder and decoder. Defaults to 'tanh'.
            weight_initializer (str): Weight initializer for the encoder and decoder. Defaults to 'xavier_uniform'.
            bias_initializer (str): Bias initializer for the encoder and decoder. Defaults to 'zeros'.
            loss_factory (Callable[[int, int, bool, tuple, bool], Callable], optional): Factory function for the loss function. Defaults to loss_fn_factory.
            optimizer_hparams (Dict[str, Any], optional): Hyperparameters for the optimizer. Defaults to {}.
            loss_params (Dict[str, Any], optional): Parameters for the loss function. Defaults to {}.
            seed (int, optional): Random seed. Defaults to 42.
            logger_params (Dict[str, Any], optional): Parameters for the logger. Defaults to None.
            enable_progress_bar (bool, optional): Whether to enable progress bar. Defaults to True.
            debug (bool, optional): Whether to jit the loss functions. Defaults to False.
            check_val_every_n_epoch (int, optional): Check validation every n epoch. Defaults to 500.
            update_mask_every_n_epoch (int, optional): Update mask every n epoch. Defaults to 500.
            coefficient_threshold (float, optional): Threshold for updating the mask. Defaults to 0.1.
        """
        self.seed = seed

        ### Hyperparameters for autoencoder setup
        self.model_hparams = {
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'poly_order': poly_order,
            'widths': widths,
            'activation': activation,
            'weight_initializer': weight_initializer,
            'bias_initializer': bias_initializer,
        }

        ### Setting up library size for the model parameters
        lib_size = library_size(self.model_hparams['latent_dim'], poly_order=self.model_hparams['poly_order'], use_sine=False)
        self.model_hparams['lib_size'] = lib_size

        ### Store model parameters
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.update_mask_every_n_epoch = update_mask_every_n_epoch
        self.loss_params = loss_params
        self.logger_params = logger_params

        # Store hyperparameters for trainer and model
        self.hparams = {
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "poly_order": poly_order,
                "widths": widths,
                "activation": activation,
                "weight_initializer": weight_initializer,
                "bias_initializer": bias_initializer,
                "optimizer_hparams": optimizer_hparams,
                "loss_params": loss_params,
                "update_mask_every_n_epoch": update_mask_every_n_epoch,
                "coefficient_threshold": coefficient_threshold,
        }

        ### Define the autoencoder model
        self._init_autoencoder()
        ### Initialize the autoencoder, the SINDy coefficients, and define self.state with TrainState
        self._init_model_state(exmp_input)

        ### Define the loss function from the factory
        self.loss_fn = loss_factory(autoencoder=self.model, **self.loss_params)
        
        self.create_jitted_functions()

    
    def _init_autoencoder(self):
        """
        Initialize the autoencoder and add it to the loss params.
        """
        # Initialize Encoder and Decoder
        encoder = Encoder(self.model_hparams['input_dim'], self.model_hparams['latent_dim'], self.model_hparams['widths'])
        decoder = Decoder(self.model_hparams['input_dim'], self.model_hparams['latent_dim'], self.model_hparams['widths'])

        # Initialize Autoencoder
        self.model = Autoencoder(
            input_dim=self.model_hparams['input_dim'],
            latent_dim=self.model_hparams['latent_dim'],
            lib_size=self.model_hparams['lib_size'],
            widths=self.model_hparams['widths'],
            encoder=encoder,
            decoder=decoder
        )

    def _init_model_state(self, exmp_input: Any):
        """
        Initialize the flax autoencoder model with the example input and random seed. Also initializes the SINDy coefficients

        Args:
            exmp_input (Any): Example input to initialize the model, with correct input shape

        """
        ### Split initialization rng
        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)

        ### Set correct example input for initialization
        exmp_input = [exmp_input] if not isinstance(exmp_input, (list, tuple)) else exmp_input
        ### Initialize model parameters
        variables = self.model.init(init_rng, exmp_input)

        ### Optimizer state
        self.state = TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=variables["params"],
            rng=model_rng,
            tx=None,
            opt_state=None,
            mask=variables['params']['sindy_coefficients'],
        )

    def init_logger(self, logger_params: Optional[Dict] = None):
        """
        Initialize the tensorboard logger for logging the training metrics and model checkpoints.
        (see https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#tensorboard-logging)

        default log_dir: checkpoints/version_{x} (x is auto-incremented (if not specified) for each run (1,2,3..))

        Args:
            logger_params (Optional[Dict], optional): Parameters for the logger. Mainly includes folder and file name params.
        """
        if logger_params is None:
            logger_params = dict()
        log_dir = logger_params.get("log_dir", None)
        if not log_dir:
            log_dir = logger_params.get("base_log_dir", "checkpoints/")
            if "logger_name" in logger_params:
                log_dir = os.path.join(log_dir, logger_params["logger_name"])
            version = None
        else:
            version = ""
        self.logger = TensorBoardLogger(
            save_dir=log_dir, version=version, name="")
        log_dir = self.logger.log_dir
        if not os.path.isfile(os.path.join(log_dir, "hparams.json")):
            os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
            with open(os.path.join(log_dir, "hparams.json"), "w") as f:
                json.dump(self.hparams, f, indent=4)
        self.log_dir = log_dir 

    def print_tabulate(self, exmp_input: Any):
        """
        Print the model paramater summary using the tabulate function.
        """
        print(self.model.tabulate(random.PRNGKey(0), exmp_input))

    def init_optimizer(self,
                   num_epochs: int,
                   num_steps_per_epoch: int):
        """
        Initializes the optimizer and learning rate scheduler.
        Defaults to no warmup and a constant learning rate with Adam.
        No weight decay or gradient clipping by default.

        Args:
        num_epochs: Number of epochs the model will be trained for.
        num_steps_per_epoch: Number of training steps per epoch.
        """
        hparams = copy(self.optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop('optimizer', 'adam')
        if optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{optimizer_name}"'

        # Initialize learning rate
        lr = hparams.pop('lr', 1e-3)
        use_lr_schedule = hparams.pop('lr_schedule', False)

        if use_lr_schedule:
            # Initialize learning rate scheduler
            warmup = hparams.pop('warmup', 0)
            lr = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=int(num_epochs * num_steps_per_epoch),
                end_value=0.01 * lr
            )

        # Clip gradients at max value, and optionally apply weight decay
        transf = [optax.clip_by_global_norm(hparams.pop('gradient_clip', 1.0))]
        if opt_class == optax.sgd and 'weight_decay' in hparams:  # Weight decay is integrated in adamw
            transf.append(optax.add_decayed_weights(hparams.pop('weight_decay', 0.0)))

        # Combine transformations and optimizer
        optimizer = optax.chain(
            *transf,
            opt_class(lr, **hparams)
        )

        # Initialize training state
        self.state = TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,
            tx=optimizer,
            rng=self.state.rng,
            mask=self.state.mask,
        )


    def create_jitted_functions(self):
        """
        Create the jitted training and evaluation functions if debug=False. Otherwise, skip jitting.
        """
        train_step, eval_step = self.create_functions()
        if self.debug:
            print("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = jit(train_step)
            self.eval_step = jit(eval_step)

    def create_functions(
        self,
    ) -> Tuple[
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
    ]:
        """
        create training and evaluation functions for the model. Based on the loss function. 
        """
        ### Defining value and grad function for the loss function
        val_grad_fn = value_and_grad(self.loss_fn, has_aux=True, argnums=0)
        ### Training step
        ### The loss function takes in params, batch, and state.mask
        def train_step(state: TrainState, batch: Tuple):

            (loss, metrics), grads = val_grad_fn(state.params, batch, state.mask)
            state = state.apply_gradients(grads=grads)
            return state, metrics

        def eval_step(state: TrainState, batch: Any):
            (loss, metrics) = self.loss_fn(state.params, batch, state.mask)
            return metrics

        return train_step, eval_step
    ### Maybe in future enable train step scheduler for this for varying step size
    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        num_epochs: int = 5000,
        final_epochs: int = 500,
    ) -> Dict[str, Any]:
        """
        Train the model using the training and validation loaders. Optionally, evaluate the model on a test loader.

        Args:
            train_loader (Iterator): Training data loader
            val_loader (Iterator): Validation data loader
            test_loader (Optional[Iterator], optional): Test data loader. Defaults to None.
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 500.
        
        """
        ### Initialize the logger
        self.init_logger(self.logger_params)

        ### Initialize the optimizer
        self.init_optimizer(num_epochs + final_epochs, len(train_loader))
        best_eval_metrics = None

        #### Initial training loop
        for epoch_idx in self.tracker(range(1, num_epochs + 1), desc="Epochs"):
            starting = time.time()
            train_metrics = self.train_epoch(train_loader)
            self.logger.log_metrics(train_metrics, step=epoch_idx)

            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix="val/")
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f"eval_epoch_{str(epoch_idx).zfill(3)}", eval_metrics)
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(step=epoch_idx)
                    self.save_metrics("best_eval", eval_metrics)

            if epoch_idx % self.update_mask_every_n_epoch == 0:
                new_mask = update_mask(self.state.params["sindy_coefficients"])
                self.state = self.state.replace(mask=new_mask)
            
        print(f"Completed {num_epochs} epochs. Starting final training loop without regularization.")
        new_loss_params = self.loss_params.copy()  # Create a copy of the loss parameters
        new_loss_params['regularization'] = False  # Update the copy with regularization=False
        self.loss_fn = loss_fn_factory(autoencoder=self.model, **new_loss_params)  # Create the new loss function
        self.create_jitted_functions()  # Recreate the jitted functions with the new loss function

        #### Final training loop
        print(f"Beginning final training loop.")
        for epoch_idx in self.tracker(range(1, final_epochs + 1), desc="Final Epochs without regularization"):
            train_metrics = self.train_epoch(train_loader)
            overall_epoch_idx = epoch_idx + num_epochs  
            self.logger.log_metrics(train_metrics, step=overall_epoch_idx)

            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix="val/")
                self.logger.log_metrics(eval_metrics, step=overall_epoch_idx)
                self.save_metrics(f"eval_epoch_{str(overall_epoch_idx).zfill(3)}", eval_metrics)
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(step=overall_epoch_idx)
                    self.save_metrics("best_eval", eval_metrics)
            
            if epoch_idx % self.update_mask_every_n_epoch == 0:
                new_mask = update_mask(self.state.params["sindy_coefficients"])
                self.state = self.state.replace(mask=new_mask)
        
        self.logger.finalize("success")
        
        return best_eval_metrics


    def train_epoch(self, train_loader: Iterator) -> Dict[str, Any]:
        """
        Train the model for one epoch using the training data loader. Called from train_model.

        Args:
            train_loader (Iterator): Training data loader

        """
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        #start_time = time.time()
        for batch in train_loader:
            #print("training batch")
            self.state, step_metrics = self.train_step(self.state, batch)
            for key in step_metrics:
                metrics["train/" + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        #print(f"Inner training loop time: {time.time() - start_time}")
        #metrics["epoch_time"] = time.time() - start_time
        return metrics

    def eval_model(self, data_loader: Iterator, log_prefix: Optional[str] = "") -> Dict[str, Any]:
        """
        Evaluate the model using the data loader. Called from train_model.

        Args:
            data_loader (Iterator): Data loader for evaluation
            log_prefix (Optional[str], optional): Prefix for the logging keys. Defaults to "".

        """
        metrics = defaultdict(float)
        num_elements = 0
        for batch in data_loader:
            step_metrics = self.eval_step(self.state, batch)
            batch_size = batch[0].shape[0] if isinstance(
                batch, (list, tuple)) else batch.shape[0]
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements += batch_size
        metrics = {log_prefix +
                   key: (metrics[key] / num_elements).item() for key in metrics}
        return metrics

    def is_new_model_better(self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]) -> bool:
        """
        Check if the new model is better than the old model based on the validation loss.

        Args:
            new_metrics (Dict[str, Any]): New metrics
            old_metrics (Dict[str, Any]): Old metrics

        Returns:
            bool: True if the new model is better, False otherwise
        """
        if old_metrics is None:
            return True

        # Compare only the validation loss
        new_loss = new_metrics.get("val/loss")
        old_loss = old_metrics.get("val/loss")

        if new_loss is not None and old_loss is not None:
            return new_loss < old_loss

        # If for some reason the loss is not in the metrics, return False as a fallback
        assert False, f"No known metrics to log on: {new_metrics}"

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """
        Create a tqdm progress bar if enable_progress_bar is True. Otherwise, return the iterator.

        Args:
            iterator (Iterator): Iterator to track

        Returns:
            Iterator: Tracked iterator
        """
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """
        Save the metrics to a JSON file.

        Args:
            filename (str): Filename to save the metrics
            metrics (Dict[str, Any]): Metrics to save
        """

        metrics_dir = os.path.join(self.log_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        file_path = os.path.join(metrics_dir, f"{filename}.json")
        print(f"Saving metrics to: {file_path}")
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)

    def save_model(self, step: int = 0):
        absolute_log_dir = os.path.abspath(self.log_dir)
        checkpoints.save_checkpoint(
            ckpt_dir=absolute_log_dir,
            target={"params": self.state.params,
                    "opt_state": self.state.opt_state},
            step=step,
            overwrite=True,
        )
        model_state = {
            "params": self.state.params,
            "opt_state": self.state.opt_state,
        }
        with open(os.path.join(absolute_log_dir, 'model_state.pkl'), 'wb') as f:
            pickle.dump(model_state, f)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, exmp_input: Any) -> 'SINDy_trainer':
        """
        Load the trainer from a checkpoint. Required for loading the model and optimizer state.

        Args:
            checkpoint (str): Path to the checkpoint
            exmp_input (Any): Example input to initialize the model

        Returns:
            Trainer: Trainer object
        """
        checkpoint = os.path.abspath(checkpoint)
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), f"Could not find hparams file"

        with open(hparams_file, "r") as f:
            hparams = json.load(f)

        # Create the trainer instance with the loaded hyperparameters
        trainer = cls(
            input_dim=hparams["input_dim"],
            latent_dim=hparams["latent_dim"],
            poly_order=hparams["poly_order"],
            widths=hparams["widths"],
            exmp_input=exmp_input,
            activation=hparams["activation"],
            weight_initializer=hparams["weight_initializer"],
            bias_initializer=hparams["bias_initializer"],
            optimizer_hparams=hparams.get("optimizer_hparams", {}),
            loss_params=hparams.get("loss_params", {}),
            seed=hparams.get("seed", 42),
            logger_params=hparams.get("logger_params", {}),
            enable_progress_bar=hparams.get("enable_progress_bar", True),
            debug=hparams.get("debug", False),
            check_val_every_n_epoch=hparams.get("check_val_every_n_epoch", 500),
            update_mask_every_n_epoch=hparams.get("update_mask_every_n_epoch", 500),
            coefficient_threshold=hparams.get("coefficient_threshold", 0.1),
        )

        # Load the model and optimizer state from the checkpoint
        trainer.load_model(checkpoint)

        return trainer



    def load_model(self, checkpoint: str):
        """
        Load model and optimizer state from the checkpoint.

        Args:
            checkpoint (str): Path to the checkpoint
        """
        checkpoint = os.path.abspath(checkpoint)
        with open(os.path.join(checkpoint, 'model_state.pkl'), 'rb') as f:
            model_state = pickle.load(f)

        self.state = self.state.replace(
            params=model_state['params'],
            opt_state=model_state['opt_state'],
        )
