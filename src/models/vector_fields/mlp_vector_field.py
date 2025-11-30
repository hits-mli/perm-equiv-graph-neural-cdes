import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import diffrax
import equinox as eqx

import math

from .layers import ConvLayer


class MLPVectorField(eqx.Module):
    mlp: eqx.Module

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        *,
        key: jr.PRNGKey,
        **kwargs,
    ):
        super(MLPVectorField).__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 1,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jnn.relu,
            key=key,
        )

    def __call__(self, t, y, args):
        # append t to every row of y
        y = jnp.concatenate([y, t[:, None]], axis=-1)

        return jax.vmap(self.mlp)(y)
