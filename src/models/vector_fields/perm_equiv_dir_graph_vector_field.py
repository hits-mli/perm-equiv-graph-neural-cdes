import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from .layers import ConvEquivFusionDirectedLayer
from ..neural_nets import IdxEncoder


class PermEquivDirGraphVectorField(eqx.Module):
    gnn_layers: list[ConvEquivFusionDirectedLayer]
    # gnn_layers: list[ConvEquivFusionLayer]
    data_embed_dim: int
    num_nodes: int
    idx_enc: IdxEncoder
    msg_func_adj: eqx.nn.MLP
    msg_func_adj_deriv: eqx.nn.MLP
    enc_idx: bool

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        data_embed_dim: int,
        num_nodes: int,
        enc_idx: bool = False,
        enc_type: str = "mlp",
        idx_dim: int = 512,
        *,
        key: jr.PRNGKey,
        **kwargs,
    ):
        """
        Initializes the GraphVectorField.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output features.
            num_layers (int): Number of layers in the GNN.
            key (jr.PRNGKey): JAX random key for initializing layers.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)

        gnn_layers = []
        for _ in range(num_layers - 1):
            tempkey, key = jr.split(key, 2)
            gnn_layers.append(
                ConvEquivFusionDirectedLayer(
                    input_dim=input_dim, output_dim=hidden_dim, key=tempkey
                )
            )
            input_dim = hidden_dim
        tempkey, key = jr.split(key, 2)
        gnn_layers.append(
            ConvEquivFusionDirectedLayer(
                input_dim=input_dim, output_dim=output_dim, key=tempkey
            )
        )
        self.gnn_layers = gnn_layers
        self.data_embed_dim = data_embed_dim
        self.num_nodes = num_nodes

        idx_enc_key, msg_func_adj_key, msg_func_adj_deriv_key = jr.split(key, 3)

        self.enc_idx = enc_idx
        self.idx_enc = IdxEncoder(num_nodes, idx_dim, key=idx_enc_key, type=enc_type)
        self.msg_func_adj = eqx.nn.MLP(
            in_size=2 * idx_dim + 1,
            out_size=1,
            width_size=8,
            depth=2,
            key=msg_func_adj_key,
        )
        self.msg_func_adj_deriv = eqx.nn.MLP(
            in_size=2 * idx_dim + 1,
            out_size=1,
            width_size=8,
            depth=2,
            key=msg_func_adj_deriv_key,
        )

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
        adj, adj_derivative, t_gradient = (
            control_adj.evaluate(t)[..., -1],
            control_adj.derivative(t)[..., -1],
            control_adj.derivative(t)[..., 0],
        )

        if self.enc_idx:
            pairwise_emb = self.idx_enc()

            # print(src_idx_encodings.shape, dst_idx_encodings.shape, message_passing_matrix[:, :, None].shape)

            adj = jax.vmap(jax.vmap(self.msg_func_adj))(
                jnp.concat([adj[:, :, None], pairwise_emb], axis=-1)
            )
            adj = jnp.squeeze(adj, axis=-1)

            adj_derivative = jax.vmap(jax.vmap(self.msg_func_adj_deriv))(
                jnp.concat(
                    [adj_derivative[:, :, None], pairwise_emb],
                    axis=-1,
                )
            )
            adj_derivative = jnp.squeeze(adj_derivative, axis=-1)

        for i, layer in enumerate(self.gnn_layers):
            node_features = layer(node_features, adj, adj_derivative)
            if i < len(self.gnn_layers) - 1:
                node_features = jax.nn.relu(node_features)

        t_gradient = jnp.mean(t_gradient, axis=0)  # Shape: [nodes]
        out = jnp.einsum("i, ij -> ij", t_gradient, node_features)
        return out
