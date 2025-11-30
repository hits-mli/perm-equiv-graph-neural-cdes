import equinox as eqx
import jax
import jax.numpy as jnp


class IdxEncoder(eqx.Module):

    type: str
    num_nodes: int
    module: eqx.nn.Embedding | eqx.nn.MLP | eqx.nn.Linear
    dst_grid: jax.Array | None
    src_grid: jax.Array | None

    def __init__(
        self, num_nodes: int, out_channels: int, *, key: jax.random.PRNGKey, type: str
    ):
        super().__init__()

        self.type = type
        self.num_nodes = num_nodes

        if type == "linear":
            self.module = eqx.nn.Linear(
                in_features=1,
                out_features=out_channels,
                key=key,
            )
        elif type == "mlp":
            self.module = eqx.nn.MLP(
                in_size=1,
                out_size=out_channels,
                width_size=8,
                depth=2,
                key=key,
            )
        elif type == "emb":
            self.module = eqx.nn.Embedding(
                num_embeddings=num_nodes, embedding_size=out_channels, key=key
            )

        if type in ["linear", "mlp"]:
            self.dst_grid = jnp.tile((jnp.arange(num_nodes) + 1), (num_nodes, 1))
            self.src_grid = self.dst_grid.T
        else:
            self.dst_grid, self.src_grid = None, None

    def __call__(self) -> jax.Array:
        if self.type == "emb":
            emb = jax.vmap(self.module)(jnp.arange(self.num_nodes))
        elif self.type in ["linear", "mlp"]:
            emb = jax.vmap(self.module)(jnp.arange(self.num_nodes)[:, None])

        # emb_i will be broadcast along axis 1 -> shape (n, 1, d)
        emb_i = emb[:, None, :]  # (n, 1, d)
        # emb_j will be broadcast along axis 0 -> shape (1, n, d)
        emb_j = emb[None, :, :]  # (1, n, d)

        return jnp.concatenate(
            [
                emb_i.repeat(self.num_nodes, axis=1),
                emb_j.repeat(self.num_nodes, axis=0),
            ],
            axis=-1,
        )
