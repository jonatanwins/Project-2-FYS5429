import sys
sys.path.append('../')
from autoencoder import Autoencoder, Encoder, Decoder

import os
from typing import Any, Optional, Tuple, Iterator, Dict, Callable
import json
import time
from tqdm import tqdm
from copy import copy
from collections import defaultdict

import jax
from jax import random, value_and_grad
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import jax.numpy as jnp

from pytorch_lightning.loggers import TensorBoardLogger

class TrainState(train_state.TrainState):
    mask: jnp.ndarray = None
    rng: Any = None

def update_mask(coefficients, threshold=0.1):
    return jnp.where(jnp.abs(coefficients) >= threshold, 1, 0)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any],
        exmp_input: Any,
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_n_epoch: int = 500,
        update_mask_every_n_epoch: int = 500,
        loss_fn: Callable = lambda params, batch, model, mask: (0, None),
        **kwargs,
    ):
        """
        Trainer module for holding all parts required for training a model. 
        This includes the model, optimizer, loss function, and the training loop.

        Args:
            model (nn.Module): Flax model to be trained
            model_hparams (Dict[str, Any]): Hyperparameters for the model
            optimizer_hparams (Dict[str, Any]): Hyperparameters for the optimizer
            exmp_input (Any): Example input to initialize the model
            seed (int, optional): Random seed. Defaults to 42.
            logger_params (Dict[str, Any], optional): Parameters for the logger. Defaults to None.
            enable_progress_bar (bool, optional): Whether to enable progress bar. Defaults to True.
            debug (bool, optional): Whether to jit the loss funcions. Defaults to False.
            check_val_every_n_epoch (int, optional): Check validation every n epoch. Defaults to 500.
            update_mask_every_n_epoch (int, optional): Update mask every n epoch. Defaults to 500.
            loss_fn (Callable, optional): Loss function. Defaults to lambda params, batch, model, mask: (0, None).
        """
        self.model = model
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.update_mask_every_n_epoch = update_mask_every_n_epoch
        self.loss_fn = loss_fn

        self.exmp_input = exmp_input
        json_model_hparams = self.json_serializable(copy(model_hparams))
        self.config = {
             "model_hparams": json_model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "logger_params": logger_params,
            "enable_progress_bar": self.enable_progress_bar,
            "debug": self.debug,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "seed": self.seed,
            "update_mask_every_n_epoch": self.update_mask_every_n_epoch,
            "loss_fn": loss_fn,
        }
        self.config.update(kwargs)

        self.print_tabulate(exmp_input)
        self.init_logger(logger_params)
        self.create_jitted_functions()
        self.init_model(exmp_input)
    

    def json_serializable(self, obj: dict):
        """
        Convert the model parameters to a JSON serializable format. For storing the model
        """
        for key, value in obj.items():
            if isinstance(value, nn.Module):
                obj[key] = {
                    "class": value.__class__.__name__,
                    "params": {k: getattr(value, k) for k in value.__annotations__.keys()}
                }
            elif isinstance(value, dict):
                obj[key] = self.json_serializable(value)
        return obj

    @staticmethod
    def instantiate_from_dict(class_dict: dict):
        """
        Instantiate a class from a dictionary. Required for loading the model from a checkpoint. Inverse
        of above method.

        Args:
            class_dict (dict): Dictionary containing the class name and parameters
        """
        class_name = class_dict['class']
        params = class_dict['params']

        if class_name == "Encoder":
            return Encoder(**params)
        elif class_name == "Decoder":
            return Decoder(**params)
        else:
            raise ValueError(f"Unknown class name: {class_name}")

    def init_logger(self, logger_params: Optional[Dict] = None):
        """
        Initialize the tensorboard logger for logging the training metrics and model checkpoints.
        (see https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#tensorboard-logging)

        Args:
            logger_params (Optional[Dict], optional): Parameters for the logger. Defaults to None.
        """
        if logger_params is None:
            logger_params = dict()
        log_dir = logger_params.get("log_dir", None)
        if not log_dir:
            log_dir  = logger_params.get("base_log_dir", "checkpoints/")
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
                json.dump(self.config, f, indent=4)
        self.log_dir = log_dir

    def init_model(self, exmp_input: Any):
        """
        Initialize the flax model with the example input and random seed.

        Args:
            exmp_input (Any): Example input to initialize the model, with correct input shape

        """
        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)
        exmp_input = [exmp_input] if not isinstance(exmp_input, (list, tuple)) else exmp_input
        variables = self.run_model_init(exmp_input, init_rng)
        self.state = TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=variables["params"],
            rng=model_rng,
            tx=None,
            opt_state=None,
            mask=variables['params']['sindy_coefficients'],
        )

    def run_model_init(self, exmp_input: Any, init_rng: Any) -> Dict:
        """
        Run the flax model init function to initialize the model parameters.
        """
        return self.model.init(init_rng, exmp_input)

    def print_tabulate(self, exmp_input: Any):
        """
        Print the model paramater summary using the tabulate function.
        """
        print(self.model.tabulate(random.PRNGKey(0), exmp_input))

    def init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        """
        Initialize the optimizer with the hyperparameters. Defaults to AdamW with a warmup cosine decay schedule.

        Args:
            num_epochs (int): Number of epochs to train the model (used for the learning rate schedule)
            num_steps_per_epoch (int): Number of steps per epoch (used for the learning rate schedule)
        
        """
        hparams = copy(self.optimizer_hparams)
        optimizer_name = hparams.pop("optimizer", "adamw")
        if optimizer_name.lower() == "adam":
            opt_class = optax.adam
        elif optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        elif optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'
        lr = hparams.pop("lr", 1e-3)
        warmup = hparams.pop("warmup", 0)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup,
            decay_steps=int(num_epochs * num_steps_per_epoch),
            end_value=0.01 * lr,
        )
        transf = [optax.clip_by_global_norm(hparams.pop("gradient_clip", 1.0))]
        if opt_class == optax.sgd and "weight_decay" in hparams:
            transf.append(optax.add_decayed_weights(hparams.pop("weight_decay", 0.0)))
        optimizer = optax.chain(*transf, opt_class(lr_schedule, **hparams))
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
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

    def create_functions(
        self,
    ) -> Tuple[
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
        Callable[[TrainState, Any], Tuple[TrainState, Dict]],
    ]:
        """
        Create the training and evaluation functions for the model. Based on the loss function. 
        """
        def train_step(state: TrainState, batch: Tuple):
            def loss_fn(params):
                return self.loss_fn(params, batch, self.model, state.mask)

            val_grad_fn = value_and_grad(loss_fn, has_aux=True)
            (loss, metrics), grads = val_grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, metrics

        def eval_step(state: TrainState, batch: Any):
            (loss, metrics) = self.loss_fn(state.params, batch, self.model, state.mask)
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
        Train the model using the training and validation loaders. Optionally, evaluate the model on a test loader.

        Args:
            train_loader (Iterator): Training data loader
            val_loader (Iterator): Validation data loader
            test_loader (Optional[Iterator], optional): Test data loader. Defaults to None.
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 500.
        
        """
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval_metrics = None
        for epoch_idx in self.tracker(range(1, num_epochs + 1), desc="Epochs"):
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
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(test_loader, log_prefix="test/")
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics("test", test_metrics)
            best_eval_metrics.update(test_metrics)
        self.logger.finalize("success")
        return best_eval_metrics

    def train_epoch(self, train_loader: Iterator) -> Dict[str, Any]:
        """
        Train the model for one epoch using the training data loader.

        Args:
            train_loader (Iterator): Training data loader

        """
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        for batch in train_loader:
            self.state, step_metrics = self.train_step(self.state, batch)
            for key in step_metrics:
                metrics["train/" + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics["epoch_time"] = time.time() - start_time
        return metrics

    def eval_model(self, data_loader: Iterator, log_prefix: Optional[str] = "") -> Dict[str, Any]:
        """
        Evaluate the model using the data loader.

        Args:
            data_loader (Iterator): Data loader for evaluation
            log_prefix (Optional[str], optional): Prefix for the logging keys. Defaults to "".

        """
        metrics = defaultdict(float)
        num_elements = 0
        for batch in data_loader:
            step_metrics = self.eval_step(self.state, batch)
            batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements += batch_size
        metrics = {log_prefix + key: (metrics[key] / num_elements).item() for key in metrics}
        return metrics

    def is_new_model_better(self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]) -> bool:
        if old_metrics is None:
            return True
        for key, is_larger in [("val/val_metric", False), ("val/loss", False)]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f"No known metrics to log on: {new_metrics}"

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
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
            target={"params": self.state.params},
            step=step,
            overwrite=True,
        )

    def load_model(self):
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=state_dict["params"],
            tx=self.state.tx if self.state.tx else optax.sgd(0.1),
            rng=self.state.rng,
        )

    def bind_model(self):
        params = {"params": self.state.params}
        return self.model.bind(params)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, exmp_input: Any) -> Any:
        checkpoint = os.path.abspath(checkpoint)
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), "Could not find hparams file"
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        model_hparams = hparams.get("model_hparams", {})
        encoder = cls.instantiate_from_dict(model_hparams.pop("encoder"))
        decoder = cls.instantiate_from_dict(model_hparams.pop("decoder"))
        model = Autoencoder(
            input_dim=model_hparams["input_dim"],
            latent_dim=model_hparams["latent_dim"],
            lib_size=model_hparams["lib_size"],
            widths=model_hparams["widths"],
            encoder=encoder,
            decoder=decoder,
            train=True
        )
        trainer = cls(
            model=model, 
            model_hparams=hparams.get("model_hparams", {}),
            optimizer_hparams=hparams.get("optimizer_hparams", {}),
            exmp_input=exmp_input,
            seed=hparams.get("seed", 42),
            logger_params=hparams.get("logger_params", {}),
            enable_progress_bar=hparams.get("enable_progress_bar", True),
            debug=hparams.get("debug", False),
            check_val_every_n_epoch=hparams.get("check_val_every_n_epoch", 100),
            update_mask_every_n_epoch=hparams.get("update_mask_every_n_epoch", 500),
            loss_fn=hparams.get("loss_fn", lambda params, batch, model, mask: (0, None))
        )
        trainer.load_model()
        return trainer