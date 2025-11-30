import pydantic
import typing as tp

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from models import *


class MSELossCfg(pydantic.BaseModel):
    """
    Configuration for Mean Squared Error (MSE) loss.
    """

    name: tp.Literal["MSE"] = pydantic.Field(...)

    model_config = pydantic.ConfigDict(extra="forbid")

    @staticmethod
    def mse_loss(
        model: eqx.Module,
        data_i: tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jax.Array:
        """
        Compute the Mean Squared Error (MSE) loss.

        Args:
            model (eqx.Module): The model to evaluate.
            data_i (tp.Tuple): The data tuple for the step.

        Returns:
            jax.Array: The computed MSE loss.
        """
        if isinstance(model, GraphNeuralODE):
            t_i, coeffs_adj_i, label_i, x0_i, adjacency_list, events_time = data_i
            pred_y = jax.vmap(model)(
                t_i, coeffs_adj_i, x0_i, adjacency_list, events_time
            )
        elif isinstance(model, GraphNeuralCDE):
            t_i, coeffs_adj_i, label_i, x0_i = data_i
            pred_y = jax.vmap(model)(t_i, coeffs_adj_i, x0_i)

        pred_y = jnp.squeeze(pred_y, axis=-1)
        return jnp.mean((pred_y - label_i) ** 2)

    @staticmethod
    def validation_mse_loss(
        model: eqx.Module,
        data_i: tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tp.Tuple[jax.Array, jax.Array]:
        """
        Compute the validation Mean Squared Error (MSE) loss, additionallt returns Mean Absolute Error (MAE).

        Args:
            model (eqx.Module): The model to evaluate.
            data_i (tp.Tuple): The data tuple for the step.

        Returns:
            jax.Array: The computed MSE loss.
        """
        if isinstance(model, GraphNeuralODE):
            t_i, coeffs_adj_i, label_i, x0_i, adjacency_list, events_time = data_i
            pred_y = jax.vmap(model)(
                t_i, coeffs_adj_i, x0_i, adjacency_list, events_time
            )
        elif isinstance(model, GraphNeuralCDE):
            t_i, coeffs_adj_i, label_i, x0_i = data_i
            pred_y = jax.vmap(model)(t_i, coeffs_adj_i, x0_i)

        pred_y = jnp.squeeze(pred_y, axis=-1)
        return jnp.mean((pred_y - label_i) ** 2, axis=-1), jnp.mean(
            jnp.abs(pred_y - label_i), axis=-1
        )

    def build(self) -> tp.Callable:
        """
        Build the MSE loss function.

        Returns:
            tp.Callable: The MSE loss function.
        """
        return self.mse_loss

    def build_validation_loss(self) -> tp.Callable:
        """
        Build the validation MSE loss function.

        Returns:
            tp.Callable: The validation MSE loss function.
        """
        return self.validation_mse_loss


class L1LossCfg(pydantic.BaseModel):
    """
    Configuration for Mean Absolute Error (MSE) or L1 loss.
    """

    name: tp.Literal["L1"] = pydantic.Field(...)

    model_config = pydantic.ConfigDict(extra="forbid")

    @staticmethod
    def l1_loss(
        model: eqx.Module,
        data_i: tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jax.Array:
        """
        Compute the Mean Absolute Error (L1) loss.

        Args:
            model (eqx.Module): The model to evaluate.
            data_i (tp.Tuple): The data tuple for the step.

        Returns:
            jax.Array: The computed L1 loss.
        """
        if isinstance(model, GraphNeuralODE):
            t_i, coeffs_adj_i, label_i, x0_i, adjacency_list, events_time = data_i
            pred_y = jax.vmap(model)(
                t_i, coeffs_adj_i, x0_i, adjacency_list, events_time
            )
        elif isinstance(model, GraphNeuralCDE):
            t_i, coeffs_adj_i, label_i, x0_i = data_i
            pred_y = jax.vmap(model)(t_i, coeffs_adj_i, x0_i)

        pred_y = jnp.squeeze(pred_y, axis=-1)
        return jnp.mean(jnp.abs((pred_y - label_i)))

    @staticmethod
    def validation_l1_loss(
        model: eqx.Module,
        data_i: tp.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tp.Tuple[jax.Array, None]:
        """
        Compute the validation Mean Absolute Error (L1) loss.

        Args:
            model (eqx.Module): The model to evaluate.
            data_i (tp.Tuple): The data tuple for the step.

        Returns:
            jax.Array: The computed L1 loss.
        """
        if isinstance(model, GraphNeuralODE):
            t_i, coeffs_adj_i, label_i, x0_i, adjacency_list, events_time = data_i
            pred_y = jax.vmap(model)(
                t_i, coeffs_adj_i, x0_i, adjacency_list, events_time
            )
        elif isinstance(model, GraphNeuralCDE):
            t_i, coeffs_adj_i, label_i, x0_i = data_i
            pred_y = jax.vmap(model)(t_i, coeffs_adj_i, x0_i)

        pred_y = jnp.squeeze(pred_y, axis=-1)
        return jnp.mean(jnp.abs((pred_y - label_i)), axis=-1), None

    def build(self) -> tp.Callable:
        """
        Build the L1 loss function.

        Returns:
            tp.Callable: The L1 loss function.
        """
        return self.l1_loss

    def build_validation_loss(self) -> tp.Callable:
        """
        Build the validation L1 loss function.

        Returns:
            tp.Callable: The validation L1 loss function.
        """
        return self.validation_l1_loss
