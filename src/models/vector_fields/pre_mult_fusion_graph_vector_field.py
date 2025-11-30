import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import math

from .layers import ConvPreMultFusionLayer


class PreMultFusionGraphVectorField(eqx.Module):
    """
    A Graph Neural CDE vector field for which leverages a premultiplication fusion of the adjacency and its derivative.

    Attributes:
        gnn_layers (list[ConvPreMultFusionLayer]): List of convolutional layers in the GNN.
    """

    gnn_layers: list[ConvPreMultFusionLayer]
    data_embed_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        data_embed_dim: int,
        num_layers: int,
        *,
        key: jr.PRNGKey,
        num_nodes: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_embed_dim = data_embed_dim

        gnn_layers = []
        for _ in range(num_layers - 1):
            tempkey, key = jr.split(key, 2)
            gnn_layers.append(
                ConvPreMultFusionLayer(
                    input_dim=input_dim, output_dim=hidden_dim, key=tempkey
                )
            )
            input_dim = hidden_dim
        tempkey, key = jr.split(key, 2)
        gnn_layers.append(
            ConvPreMultFusionLayer(
                input_dim=input_dim, output_dim=output_dim, key=tempkey
            )
        )
        self.gnn_layers = gnn_layers

    def __call__(self, t: float, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """
        Computes the vector field for the given time and state.

        Args:
            t (float): Current time.
            y (jnp.ndarray): Current state.
            args (tuple): Additional arguments, including control adjacency matrix.

        Returns:
            jnp.ndarray: Computed vector field.
        """
        node_features, control_adj = y, args
        adj, adj_derivative = control_adj.evaluate(t), control_adj.derivative(t)

        for i, layer in enumerate(self.gnn_layers):
            node_features = layer(node_features, adj[..., -1], adj_derivative[..., -1])
            if i < len(self.gnn_layers) - 1:
                node_features = jax.nn.relu(node_features)

        t_gradient = jnp.mean(adj_derivative[:, :, 0], axis=0)  # Shape: [nodes]
        out = jnp.einsum("i, ij -> ij", t_gradient, node_features)
        return out
