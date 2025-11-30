import os
import pydantic
import logging
import random
import time
import typing as tp

import exca as xk
import numpy as np

import torch

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import wandb

from configs import *

logging.basicConfig(level=logging.INFO)


class Trainer(pydantic.BaseModel):
    """
    Trainer class for training a model using the specified configurations.
    """

    wandb: WandBConfig = pydantic.Field(..., description="WandB configuration")
    model: GraphNeuralCDECfg | GraphNeuralODECfg = pydantic.Field(
        ..., discriminator="name", description="Model configuration"
    )
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

    dataset: ODEDataSetCfg = pydantic.Field(..., description="Dataset configuration")
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
        train_data_dict = self.dataset.get_training_data(train_data_key)
        val_data_dict = self.dataset.get_validation_data(val_data_key)
        test_data_dict = self.dataset.get_test_data(test_data_key)
        logger.info("Data loading done")

        model = self.model.build(model_key)
        optimiser, schedule = self.optimiser.build(optimiser_key)
        opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))

        loss = self.loss.build()
        validation_loss_fn = self.loss.build_validation_loss()

        best_validation_loss, corr_test_loss, best_epoch = jnp.inf, jnp.inf, 0
        patience_counter = 0

        for epoch in range(self.epochs):
            start_time = time.time()
            if isinstance(model, GraphNeuralODE):
                data_i = (
                    train_data_dict["train_t"],
                    train_data_dict["train_graph_path_coeffs"],
                    train_data_dict["train_true_y"],
                    train_data_dict["true_y0"],
                    train_data_dict["A"],
                    train_data_dict["events_time"],
                )
            elif isinstance(model, GraphNeuralCDE):
                data_i = (
                    train_data_dict["train_t"],
                    train_data_dict["train_graph_path_coeffs"],
                    train_data_dict["train_true_y"],
                    train_data_dict["true_y0"],
                )
            else:
                raise ValueError(f"Model {self.model.name} not supported")

            train_loss, model, opt_state, max_grad, max_update = make_step(
                model, optimiser, loss, data_i, opt_state
            )

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
                if isinstance(model, GraphNeuralODE):
                    val_data_i = (
                        train_data_dict["t"],
                        train_data_dict["val_graph_path_coeffs"],
                        train_data_dict["true_y"],
                        train_data_dict["true_y0"],
                        train_data_dict["A"],
                        train_data_dict["events_time"],
                    )
                elif isinstance(model, GraphNeuralCDE):
                    val_data_i = (
                        train_data_dict["t"],
                        train_data_dict["val_graph_path_coeffs"],
                        train_data_dict["true_y"],
                        train_data_dict["true_y0"],
                    )

                total_validation_loss, total_validation_loss_l1 = validation_loss_fn(
                    model, val_data_i
                )

                validation_ids_total = jnp.hstack(
                    [train_data_dict["id_test_inter"], train_data_dict["id_test_extra"]]
                )
                validation_loss_inter = jnp.mean(
                    total_validation_loss[:, train_data_dict["id_test_inter"]]
                )
                validation_loss_extra = jnp.mean(
                    total_validation_loss[:, train_data_dict["id_test_extra"]]
                )
                validation_loss_total = jnp.mean(
                    total_validation_loss[:, validation_ids_total]
                )

                validation_loss_l1_total = jnp.mean(total_validation_loss_l1)

                logger.info(
                    f"Epoch: {epoch + 1:04d}, Validation Loss Inter: {validation_loss_inter}, Validation Loss Extra: {validation_loss_extra}, Validation Loss Total: {validation_loss_total}, Inference time: {end_time - start_time:.4f}s"
                )

                if isinstance(model, GraphNeuralODE):
                    separate_val_data_i = (
                        val_data_dict["t"],
                        val_data_dict["test_graph_path_coeffs"],
                        val_data_dict["true_y"],
                        val_data_dict["true_y0"],
                        val_data_dict["A"],
                        val_data_dict["events_time"],
                    )
                elif isinstance(model, GraphNeuralCDE):
                    separate_val_data_i = (
                        val_data_dict["t"],
                        val_data_dict["test_graph_path_coeffs"],
                        val_data_dict["true_y"],
                        val_data_dict["true_y0"],
                    )

                separate_val_loss_total, separate_val_loss_l1_total = (
                    validation_loss_fn(model, separate_val_data_i)
                )
                separate_val_loss_total = jnp.mean(separate_val_loss_total)
                separate_val_loss_l1_total = jnp.mean(separate_val_loss_l1_total)

                wandb.log({"validation_loss_inter": validation_loss_inter})
                wandb.log({"validation_loss_extra": validation_loss_extra})
                wandb.log({"validation_loss_total": validation_loss_total})
                wandb.log({"validation_loss_l1_total": validation_loss_l1_total})
                wandb.log({"separate_val_loss_total": separate_val_loss_total})
                wandb.log({"separate_val_loss_l1_total": separate_val_loss_l1_total})

                wandb.log({"validation_step_time": (end_time - start_time)})

                if separate_val_loss_total < best_validation_loss:
                    patience_counter = 0
                    best_validation_loss = separate_val_loss_total
                    best_epoch = epoch

                    if self.checkpoint_dir:
                        checkpoint_path = self.save_model_pickle(model, config_hash)
                        logger.info(
                            f"Model saved at epoch {epoch} to {checkpoint_path}"
                        )

                    if isinstance(model, GraphNeuralODE):
                        test_data_i = (
                            test_data_dict["t"],
                            test_data_dict["test_graph_path_coeffs"],
                            test_data_dict["true_y"],
                            test_data_dict["true_y0"],
                            test_data_dict["A"],
                            test_data_dict["events_time"],
                        )
                    elif isinstance(model, GraphNeuralCDE):
                        test_data_i = (
                            test_data_dict["t"],
                            test_data_dict["test_graph_path_coeffs"],
                            test_data_dict["true_y"],
                            test_data_dict["true_y0"],
                        )

                    test_loss_total, test_loss_l1_total = validation_loss_fn(
                        model, test_data_i
                    )

                    corr_test_loss = jnp.mean(test_loss_total)
                    corr_test_l1_loss = jnp.mean(test_loss_l1_total)
                    wandb.log({"test_loss": corr_test_loss})
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
            f"Best validation loss: {best_validation_loss}, Corresponding test loss: {corr_test_loss}, Corresponding L1 test loss: {corr_test_l1_loss}, Epoch: {best_epoch}"
        )
        wandb.log(
            {
                "validation_loss": best_validation_loss,
                "corr_test_loss": corr_test_loss,
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
