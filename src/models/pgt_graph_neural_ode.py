import typing as tp

import equinox as eqx
import jax.random as jr
import jax
import jax.numpy as jnp
import diffrax
import pydantic

from . import vector_fields


class PGTGraphNeuralODE(eqx.Module):
    cfg: pydantic.BaseModel = eqx.field(static=True)
    method: str
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    vector_field: eqx.Module
    interpolation: tp.Literal["linear", "cubic"]
    controller: diffrax.PIDController

    def __init__(
        self,
        cfg: pydantic.BaseModel,
        vector_field: eqx.Module,
        interpolation: tp.Literal["linear", "cubic"],
        model_key: jr.PRNGKey,
        **kwargs,
    ):
        """
        Initialize the GraphNeuralODE model.

        Args:
            cfg (pydantic.BaseModel): Configuration model.
            vector_field (eqx.Module): Vector field module.
            interpolation (tp.Literal["linear", "cubic"]): Interpolation method.
            model_key (jr.PRNGKey): Random key for model initialization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.cfg = cfg
        self.vector_field = vector_field
        self.interpolation = interpolation

        # TODO: pass input dimension
        encoder_key, decoder_key, data_encoder_key = jr.split(model_key, 3)

        self.encoder = eqx.nn.MLP(
            in_size=self.cfg.data_dim,
            out_size=self.cfg.hidden_dim,
            width_size=16,
            depth=2,
            key=encoder_key,
        )
        self.decoder = eqx.nn.MLP(
            in_size=self.cfg.hidden_dim,
            out_size=self.cfg.feature_dim,
            width_size=16,
            depth=2,
            key=decoder_key,
        )

        self.method = getattr(diffrax, self.cfg.method)()
        self.controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    def __call__(
        self,
        ts: jax.Array,
        coeffs_adj: jax.Array,
        adj_list: jax.Array,
        x_coeffs: jax.Array,
        x0: jax.Array,
        evolving_out: bool = False,
    ) -> jax.Array:
        """
        Forward pass of the GraphNeuralODE model.

        Args:
            ts (jax.Array): Sequence of time points.
            coeffs_adj (jax.Array): Sequence of control coefficients.
            adj_list (jax.Array): Adjacency list.
            x_coeffs (jax.Array): Sequence of control coefficients for node features.
            x0 (jax.Array): Initial state.
            evolving_out (bool): Flag to determine if the output should be saved at all time points.

        Returns:
            jax.Array: The output of the model.
        """
        if self.interpolation == "linear":
            control_adj = diffrax.LinearInterpolation(ts, coeffs_adj)
        elif self.interpolation == "cubic":
            control_adj = diffrax.CubicInterpolation(ts, coeffs_adj)

        term = diffrax.ODETerm(self.vector_field)

        dt0 = None
        y0 = jax.vmap(self.encoder)(x0)

        if evolving_out:
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)

        latent_node_path = diffrax.diffeqsolve(
            terms=term,
            solver=self.method,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            args=[control_adj, adj_list],
            stepsize_controller=self.controller,
            saveat=saveat,
        )

        if self.cfg.return_sequence:
            output = jax.vmap(jax.vmap(self.decoder, in_axes=0), in_axes=0)(
                latent_node_path.ys
            )
        else:
            output = jax.vmap(self.decoder)(latent_node_path.ys[-1])

        return output
