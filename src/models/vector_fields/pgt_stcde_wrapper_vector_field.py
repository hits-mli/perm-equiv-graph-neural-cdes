import jax.numpy as jnp
import equinox as eqx


class PGTSTCDEWrapperVectorField(eqx.Module):
    f_func: eqx.Module
    g_func: eqx.Module
    data_dim: int

    def __init__(self, f_func: eqx.Module, g_func: eqx.Module, data_dim: int):
        super(PGTSTCDEWrapperVectorField).__init__()

        self.f_func = f_func
        self.g_func = g_func
        self.data_dim = data_dim

    def __call__(self, t: float, y: jnp.ndarray, args: tuple) -> jnp.ndarray:
        h, z = y[0], y[1]
        control_data = args

        vector_field_f = self.f_func(h).reshape(
            -1, self.f_func.hidden_dim, self.data_dim
        )

        vector_field_g = self.g_func(z).reshape(
            -1, self.g_func.hidden_dim, self.f_func.hidden_dim
        )

        vector_field_fg = jnp.einsum("nml,nlk->nmk", vector_field_g, vector_field_f)

        dh = jnp.einsum(
            "nml,nl->nm", vector_field_f, control_data.derivative(t)[..., -1]
        )
        dz = jnp.einsum(
            "nml,nl->nm", vector_field_fg, control_data.derivative(t)[..., -1]
        )

        return jnp.stack([dh, dz], axis=0)
