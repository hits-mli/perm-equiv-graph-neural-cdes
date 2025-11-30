from re import X
import typing as tp

import equinox as eqx
import jax
import jax.random as jr
import jax.numpy as jnp
import diffrax
import pydantic

from . import vector_fields


class PGTSTGraphNeuralCDE(eqx.Module):
    cfg: pydantic.BaseModel = eqx.field(static=True)
    method: str
    encoder_h: eqx.nn.MLP
    encoder_z: eqx.nn.MLP
    decoder: eqx.nn.MLP
    interpolation: tp.Literal["linear", "cubic"]
    data_dim: int
    controller: diffrax.PIDController
    wrapped_vector_field: eqx.Module

    f_func: eqx.Module
    g_func: eqx.Module

    def __init__(
        self,
        cfg: pydantic.BaseModel,
        f_func: eqx.Module,
        g_func: eqx.Module,
        interpolation: tp.Literal["linear", "cubic"],
        data_dim: int,
        model_key: jr.PRNGKey,
        **kwargs,
    ):
        """
        Initialize the GraphNeuralCDE model.

        Args:
            cfg (pydantic.BaseModel): Configuration model.
            vector_field (eqx.Module): Vector field module.
            interpolation (tp.Literal["linear", "cubic"]): Interpolation method.
            model_key (jr.PRNGKey): Random key for model initialization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.cfg = cfg
        self.f_func = f_func
        self.g_func = g_func
        self.interpolation = interpolation
        self.data_dim = data_dim

        # TODO: pass input dimension
        decoder_h_key, decoder_z_key, decoder_key, data_encoder_key = jr.split(
            model_key, 4
        )

        self.encoder_h = eqx.nn.MLP(
            in_size=self.cfg.data_dim,
            out_size=f_func.hidden_dim,
            width_size=16,
            depth=2,
            key=decoder_h_key,
        )
        self.encoder_z = eqx.nn.MLP(
            in_size=self.cfg.data_dim,
            out_size=g_func.hidden_dim,
            width_size=16,
            depth=2,
            key=decoder_z_key,
        )

        self.decoder = eqx.nn.MLP(
            in_size=f_func.hidden_dim,
            out_size=self.cfg.feature_dim,
            width_size=16,
            depth=2,
            key=decoder_key,
        )

        self.method = getattr(diffrax, self.cfg.method)()
        self.controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

        self.wrapped_vector_field = vector_fields.PGTSTCDEWrapperVectorField(
            self.f_func, self.g_func, self.data_dim
        )

    def __call__(
        self,
        ts: jax.Array,
        coeffs_adj: jax.Array,
        x_coeffs: jax.Array,
        x0: jax.Array,
        evolving_out: bool = False,
    ) -> jax.Array:
        """
        Forward pass of the GraphNeuralCDE model.

        Args:
            ts (jax.Array): Sequence of time points.
            coeffs_adj (jax.Array): Sequence of control coefficients.
            x_t (jax.Array): .
            x0 (jax.Array): Initial state.
            evolving_out (bool): Flag to determine if the output should be saved at all time points.

        Returns:
            jax.Array: The output of the model.
        """

        if self.interpolation == "linear":
            control_data = diffrax.LinearInterpolation(ts, x_coeffs)
        elif self.interpolation == "cubic":
            control_data = diffrax.CubicInterpolation(ts, x_coeffs)

        term = diffrax.ODETerm(self.wrapped_vector_field)

        dt0 = None

        h0 = jax.vmap(self.encoder_h)(x0)
        z0 = jax.vmap(self.encoder_z)(x0)
        y0 = jnp.stack([h0, z0], axis=0)

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
            args=control_data,
            stepsize_controller=self.controller,
            saveat=saveat,
            max_steps=4096,
        )

        output = jax.vmap(self.decoder)(latent_node_path.ys[-1][1])

        return output
