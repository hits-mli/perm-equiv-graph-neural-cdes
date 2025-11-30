import jax.numpy as jnp
import equinox as eqx


class CDEWrapperVectorField(eqx.Module):
    vector_field: eqx.Module
    hidden_dim: int

    def __init__(
        self,
        vector_field: eqx.Module,
        hidden_dim: int,
    ):
        super(CDEWrapperVectorField).__init__()

        self.vector_field = vector_field
        self.hidden_dim = hidden_dim

    def __call__(self, t: float, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        control_adj, control_data = args
        out = self.vector_field(t, y, control_adj).reshape(
            -1, self.hidden_dim, self.vector_field.data_embed_dim, 2
        )
        # out = self.vector_field(t, y, control_adj).reshape(-1, self.hidden_dim, 2)
        out = jnp.einsum("nmlk,nlk->nm", out, control_data.derivative(t))
        return out
