import equinox as eqx
import jax
import jax.numpy as jnp


class FinalTanhF(eqx.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int

    linear_in: eqx.nn.Linear
    linears: list[eqx.nn.Linear]
    linear_out: eqx.nn.Linear

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, *, key
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Split key for each module
        key, key_linear_in, key_linears, key_linear_out = jax.random.split(key, 4)

        # First linear layer maps hidden_channels -> hidden_hidden_channels
        self.linear_in = eqx.nn.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            key=key_linear_in,
        )

        # Create a list of (num_hidden_layers - 1) linear layers
        keys = jax.random.split(key_linears, num_layers - 1)
        self.linears = [
            eqx.nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                key=k,
            )
            for k in keys
        ]

        # Final layer maps hidden_hidden_channels -> (input_channels * hidden_channels)
        self.linear_out = eqx.nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
            key=key_linear_out,
        )

    def __call__(self, z):
        z = jax.vmap(self.linear_in)(z)
        z = jax.nn.relu(z)

        for linear in self.linears:
            z = jax.vmap(linear)(z)
            z = jax.nn.relu(z)

        z = jax.vmap(self.linear_out)(z)
        z = jnp.tanh(z)
        return z
