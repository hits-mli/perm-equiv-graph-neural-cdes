import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array
import equinox as eqx


class ConstVectorField(eqx.Module):
    bias: Array
    data_embed_dim: int
    num_nodes: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        data_embed_dim: int,
        num_nodes: int,
        *,
        key: jr.PRNGKey,
        **kwargs,
    ):
        super(ConstVectorField).__init__(**kwargs)

        lim = 1
        self.bias = jr.uniform(key, (output_dim,), minval=-lim, maxval=lim)
        self.data_embed_dim = data_embed_dim
        self.num_nodes = num_nodes

    def __call__(self, t, y, args):
        return self.bias
