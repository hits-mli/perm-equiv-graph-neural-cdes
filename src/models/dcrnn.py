import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from typing import List, Optional, Tuple


class DiffusionGCN(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    node_num: int
    dim_in: int
    dim_out: int
    order: int

    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        order: int,
        *,
        key: jax.random.PRNGKey,
    ):
        self.node_num = node_num
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.order = order
        num_matrices = dim_in * (order + 1)
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (num_matrices, dim_out)) * jnp.sqrt(
            2.0 / num_matrices
        )
        self.bias = jnp.zeros((dim_out,))

    def __call__(self, x: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        # x: [N, D]
        # adj: [N, N]
        N, D = x.shape
        out = [x]
        x0 = x
        supports = [adj]
        for support in supports:
            x1 = jnp.einsum("ij,jk->ik", support, x0)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = 2 * jnp.einsum("ij,jk->ik", support, x1) - x0
                out.append(x2)
                x0, x1 = x1, x2
        h = jnp.concatenate(out, axis=-1)  # [N, D_concat]
        h = h.reshape(N, -1)
        h = h @ self.weight + self.bias
        return h.reshape(N, self.dim_out)


class DCGRUCell(eqx.Module):
    gate: DiffusionGCN
    update: DiffusionGCN
    hidden_dim: int
    num_node: int

    def __init__(
        self,
        num_node: int,
        input_dim: int,
        hidden_dim: int,
        order: int,
        *,
        key: jax.random.PRNGKey,
    ):
        k1, k2 = jax.random.split(key)
        self.gate = DiffusionGCN(
            num_node, input_dim + hidden_dim, 2 * hidden_dim, order, key=k1
        )
        self.update = DiffusionGCN(
            num_node, input_dim + hidden_dim, hidden_dim, order, key=k2
        )
        self.hidden_dim = hidden_dim
        self.num_node = num_node

    def __call__(
        self, x: jnp.ndarray, adj: jnp.ndarray, state: jnp.ndarray
    ) -> jnp.ndarray:
        inp = jnp.concatenate([x, state], axis=-1)
        z_r = jax.nn.sigmoid(self.gate(inp, adj))
        z, r = jnp.split(z_r, 2, axis=-1)
        candidate = jnp.concatenate([x, z * state], axis=-1)
        hc = jnp.tanh(self.update(candidate, adj))
        return r * state + (1 - r) * hc


class DCRNNModelSingleStep(eqx.Module):
    encoder_cells: List[DCGRUCell]
    decoder_cells: List[DCGRUCell]
    projection: nn.Linear
    num_layers: int
    num_node: int
    input_dim: int
    hidden_dim: int
    output_dim: int

    def __init__(
        self,
        num_node: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        order: int,
        num_layers: int = 1,
        *,
        key: jax.random.PRNGKey,
    ):
        self.num_layers = num_layers
        self.num_node = num_node
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        keys = jax.random.split(key, 2 * num_layers + 1)
        self.encoder_cells = [
            DCGRUCell(
                num_node,
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                order,
                key=keys[i],
            )
            for i in range(num_layers)
        ]
        self.decoder_cells = [
            DCGRUCell(
                num_node,
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                order,
                key=keys[num_layers + i],
            )
            for i in range(num_layers)
        ]
        self.projection = nn.Linear(hidden_dim, output_dim, key=keys[-1])

    def init_hidden(self) -> List[jnp.ndarray]:
        return [
            jnp.zeros((self.num_node, self.hidden_dim)) for _ in range(self.num_layers)
        ]

    def encode(
        self, source: jnp.ndarray, adj: jnp.ndarray, init_states: List[jnp.ndarray]
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Encode input history.
        Args:
            source: [T, N, D]
            adj: [T, N, N]
            init_states: list of initial h per layer
        Returns:
            current: [T, N, hidden_dim]
            encoder_states: list of final h per layer ([N, hidden_dim])
        """
        encoder_states = []
        current = source
        for i, cell in enumerate(self.encoder_cells):
            h = init_states[i]
            outputs = []
            for t in range(current.shape[0]):
                h = cell(current[t], adj[t], h)
                outputs.append(h)
            encoder_states.append(h)
            current = jnp.stack(outputs, axis=0)
        return current, encoder_states

    def decode(
        self, targets: jnp.ndarray, adj: jnp.ndarray, init_states: List[jnp.ndarray]
    ) -> jnp.ndarray:
        T, N, D = targets.shape
        h = targets[0]  # GO symbol
        for i, cell in enumerate(self.decoder_cells):
            h = cell(h, adj[-1], init_states[i])
        h_flat = h.reshape(N, -1)
        out_flat = jax.vmap(self.projection)(h_flat)
        if self.output_dim == 1:
            return out_flat.reshape(1, N)
        else:
            return out_flat.reshape(1, N, self.output_dim)

    def __call__(self, source: jnp.ndarray, adj: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            source: [T, N, D]
            adj: [T, N, N]
        Returns:
            outputs: [1, N, D]
        """
        # 1) init and encode
        init_state = self.init_hidden()
        _, encoder_hidden = self.encode(source, adj, init_state)
        # 2) create GO_Symbol (last observed) and dummy
        GO_Symbol = source[-1:, :, :]
        dummy = GO_Symbol  # filler to match [2, N, D]
        targets = jnp.concatenate([GO_Symbol, dummy], axis=0)
        return self.decode(targets, adj, encoder_hidden)  # [1, N, D]


# Example usage:
# supports = [torch.tensor(adj1), torch.tensor(adj2)]
# model = DCRNNModelSingleStep(supports, num_nodes, input_dim, hidden_dim, order, num_layers)
# source = torch.randn(batch_size, history_length, num_nodes, input_dim)
# prediction = model(source)  # shape: (batch_size, 1, num_nodes, input_dim)
