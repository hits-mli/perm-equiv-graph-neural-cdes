import os
import pydantic
import logging
import random
import time
import typing as tp
import hashlib
import json

import exca as xk
import numpy as np

from pydantic.types import Discriminator
import torch

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import wandb

from tgb.nodeproppred.evaluate import Evaluator

from configs import (
    WandBConfig,
    GraphNeuralCDECfg,
    GraphNeuralODECfg,
    PGTGraphNeuralCDECfg,
    PGTGraphNeuralODECfg,
    PGTSTGraphNeuralCDECfg,
    TGBSTGraphNeuralODECfg,
    TGBDataSetCfg,
    PGTDataSetCfg,
    OptimiserCfg,
    MSELossCfg,
    L1LossCfg,
)

from models import *

logging.basicConfig(level=logging.INFO)


def mse_loss(model, data_i):
    if isinstance(model, PGTGraphNeuralODE):
        (
            t_i,
            adj_coeffs_i,
            adj_list_i,
            x_coeffs,
            x0_i,
            label_i,
        ) = data_i
        y_pred = model(t_i, adj_coeffs_i, adj_list_i, x_coeffs, x0_i)
    else:
        (
            t_i,
            adj_coeffs_i,
            x_coeffs,
            x0_i,
            label_i,
        ) = data_i
        y_pred = model(t_i, adj_coeffs_i, x_coeffs, x0_i)
    y_pred = jnp.reshape(y_pred, (y_pred.shape[0], 1))
    return jnp.mean((y_pred - label_i) ** 2)


class Trainer(pydantic.BaseModel):
    """
    Trainer class for training a model using the specified configurations.
    """

    wandb: WandBConfig = pydantic.Field(..., description="WandB configuration")
    model: (
        TGBSTGraphNeuralODECfg
        | PGTGraphNeuralCDECfg
        | PGTGraphNeuralODECfg
        | PGTSTGraphNeuralCDECfg
        | GraphNeuralCDECfg
        | GraphNeuralODECfg
    ) = pydantic.Field(..., discriminator="name", description="Model configuration")
    epochs: int = 2000
    patience: int = -1
    min_epochs: int = 100

    seed: int = 1234
    log_freq: int = 10
    eval_freq: int = 10

    checkpoint_dir: str = pydantic.Field(
        default=".checkpoints/", description="Directory to save checkpoints"
    )
    checkpoint_name: str = pydantic.Field(..., description="Name of the checkpoint")

    dataset: PGTDataSetCfg | TGBDataSetCfg = pydantic.Field(
        ..., discriminator="name", description="Dataset configuration"
    )
    optimiser: OptimiserCfg = pydantic.Field(..., description="Optimiser configuration")
    loss: MSELossCfg | L1LossCfg = pydantic.Field(
        ..., discriminator="name", description="Loss configuration"
    )

    infra: xk.TaskInfra = xk.TaskInfra()

    logger_name: str = pydantic.Field(description="Name of Logger")

    # TODO: use same function as in data hashing
    def _hash_config(self) -> str:
        """
        Hashes the configuration.

        Returns:
            str: The hashed configuration.
        """
        config_dict = self.model_dump(exclude={"cache_dir"})
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def save_model_pickle(self, model, config_hash):
        """
        Save the model and optimizer state using pickle.
        """
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{config_hash}.eqx",  # Include the hash and epoch in the name
            )
            eqx.tree_serialise_leaves(checkpoint_path, model)
            return checkpoint_path  # Optionally return the checkpoint path

    def run_initialisations(self):
        """
        Initialize random seeds for reproducibility.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @infra.apply
    def run(self):
        """
        Run the training process.
        """
        self.run_initialisations()

        config_hash = self._hash_config()

        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.INFO)
        logger.info(self)

        trainer_key = jr.PRNGKey(self.seed)
        train_data_key, val_data_key, test_data_key, model_key, optimiser_key = (
            jr.split(trainer_key, 5)
        )
        train_data_loader = self.dataset.get_training_data(train_data_key)
        val_data_loader = self.dataset.get_validation_data(val_data_key)
        test_data_loader = self.dataset.get_test_data(test_data_key)

        logger.info("Data loading done")

        model = self.model.build(model_key)
        logger.info("Built model")

        total_parameters = sum(
            [
                x.size
                for x in jax.tree_util.tree_leaves(model)
                if isinstance(x, jax.Array)
            ]
        )
        logger.info(f"Total nuber of parameters: {total_parameters}")
        wandb.log({"num_params": total_parameters})

        optimiser, schedule = self.optimiser.build(optimiser_key)
        opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))

        # loss = self.loss.build()
        loss = mse_loss

        best_validation_loss, test_loss, best_epoch = jnp.inf, jnp.inf, 0
        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss, max_grad, max_update = 0.0, 0.0, 0.0
            start_time = time.time()

            for train_batch_dict in train_data_loader:
                if isinstance(model, PGTGraphNeuralODE):
                    data_i = (
                        jnp.array(train_batch_dict["t"]),
                        jnp.array(train_batch_dict["graph_path_coeffs"]),
                        jnp.array(train_batch_dict["A"]),
                        jnp.array(train_batch_dict["x_coeffs"]),
                        jnp.array(train_batch_dict["true_y0"]),
                        jnp.array(train_batch_dict["true_y"]),
                    )
                else:
                    data_i = (
                        jnp.array(train_batch_dict["t"]),
                        jnp.array(train_batch_dict["graph_path_coeffs"]),
                        jnp.array(train_batch_dict["x_coeffs"]),
                        jnp.array(train_batch_dict["true_y0"]),
                        jnp.array(train_batch_dict["true_y"]),
                    )

                batch_train_loss, model, opt_state, batch_max_grad, batch_max_update = (
                    make_step(model, optimiser, loss, data_i, opt_state)
                )
                train_loss += batch_train_loss
                max_grad = max(max_grad, batch_max_grad)
                max_update = max(max_update, batch_max_update)

            train_loss /= len(train_data_loader)

            end_time = time.time()

            wandb.log({"train_loss": train_loss})
            wandb.log({"train_step_time": (end_time - start_time)})
            wandb.log({"max_grad": max_grad})
            wandb.log({"max_update": max_update})

            if epoch == 0 or (epoch + 1) % self.log_freq == 0:
                logger.info(
                    f"Epoch: {epoch + 1:04d}, Train Loss: {train_loss}, Train Step Time: {end_time - start_time:.4f}s, Learning Rate: {schedule(epoch)}"
                )

            if (epoch + 1) % self.eval_freq == 0:
                validation_loss = 0.0

                start_time = time.time()
                for val_batch_dict in val_data_loader:
                    if isinstance(model, PGTGraphNeuralODE):
                        val_data_i = (
                            jnp.array(val_batch_dict["t"]),
                            jnp.array(val_batch_dict["graph_path_coeffs"]),
                            jnp.array(val_batch_dict["A"]),
                            jnp.array(val_batch_dict["x_coeffs"]),
                            jnp.array(val_batch_dict["true_y0"]),
                            jnp.array(val_batch_dict["true_y"]),
                        )
                    else:
                        val_data_i = (
                            jnp.array(val_batch_dict["t"]),
                            jnp.array(val_batch_dict["graph_path_coeffs"]),
                            jnp.array(val_batch_dict["x_coeffs"]),
                            jnp.array(val_batch_dict["true_y0"]),
                            jnp.array(val_batch_dict["true_y"]),
                        )

                    validation_loss += loss(model, val_data_i)

                validation_loss /= len(val_data_loader)

                end_time = time.time()

                logger.info(
                    f"Epoch: {epoch + 1:04d}, Validation Loss: {validation_loss}, Inference time: {end_time - start_time:.4f}s"
                )

                wandb.log({"validation_loss": validation_loss})
                wandb.log({"validation_step_time": (end_time - start_time)})

                if validation_loss < best_validation_loss:
                    patience_counter = 0
                    best_validation_loss = validation_loss
                    best_epoch = epoch

                    if self.checkpoint_dir:
                        self.save_model_pickle(model, config_hash)
                        logger.info(f"Model saved at epoch {epoch}")

                        test_loss = 0.0
                        for test_batch_dict in test_data_loader:
                            if isinstance(model, PGTGraphNeuralODE):
                                test_data_i = (
                                    jnp.array(test_batch_dict["t"]),
                                    jnp.array(test_batch_dict["graph_path_coeffs"]),
                                    jnp.array(test_batch_dict["A"]),
                                    jnp.array(test_batch_dict["x_coeffs"]),
                                    jnp.array(test_batch_dict["true_y0"]),
                                    jnp.array(test_batch_dict["true_y"]),
                                )
                            else:
                                test_data_i = (
                                    jnp.array(test_batch_dict["t"]),
                                    jnp.array(test_batch_dict["graph_path_coeffs"]),
                                    jnp.array(test_batch_dict["x_coeffs"]),
                                    jnp.array(test_batch_dict["true_y0"]),
                                    jnp.array(test_batch_dict["true_y"]),
                                )

                            test_loss += loss(model, test_data_i)
                        test_loss /= len(test_data_loader)
                else:
                    patience_counter += 1
                    if (
                        self.patience > 0
                        and patience_counter * self.eval_freq >= self.patience
                        and epoch > self.min_epochs
                    ):
                        logger.info("Early stopping")
                        break

        logger.info(
            f"Best validation loss: {best_validation_loss}, Corresponding test loss: {test_loss}, Epoch: {best_epoch}"
        )
        wandb.log(
            {
                "best_validation_loss": best_validation_loss,
                "corr_test_loss": test_loss,
                "best_epoch": best_epoch,
            }
        )


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    optimiser: optax.GradientTransformation,
    loss: tp.Callable[
        [
            eqx.Module,
            tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
        jnp.ndarray,
    ],
    data_i: tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    opt_state: optax.OptState,
) -> tp.Tuple[jnp.ndarray, eqx.Module, optax.OptState, jnp.ndarray, jnp.ndarray]:
    """
    Perform a single training step.

    Args:
        model (eqx.Module): The model to train.
        optimiser (optax.GradientTransformation): The optimiser to use.
        loss (tp.Callable): The loss function.
        data_i (tp.Tuple): The data for the step.
        opt_state (optax.OptState): The optimiser state.

    Returns:
        tuple : The loss, updated model, updated optimiser state, max gradient, and max update.
    """
    loss, grads = eqx.filter_value_and_grad(loss)(model, data_i)

    # Flatten gradients and compute max absolute value
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    max_grad = jnp.max(jnp.abs(jnp.concatenate([g.ravel() for g in flat_grads])))

    updates, opt_state = optimiser.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    # Flatten updates and compute max absolute value
    flat_updates, _ = jax.tree_util.tree_flatten(updates)
    max_update = jnp.max(jnp.abs(jnp.concatenate([u.ravel() for u in flat_updates])))
    return loss, model, opt_state, max_grad, max_update
