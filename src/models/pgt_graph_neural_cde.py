import typing as tp

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pydantic

from . import vector_fields


class PGTGraphNeuralCDE(eqx.Module):
    cfg: pydantic.BaseModel = eqx.field(static=True)
    method: str
    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    vector_field: eqx.Module
    interpolation: tp.Literal["linear", "cubic"]
    controller: diffrax.ConstantStepSize
    wrapped_vector_field: eqx.Module

    def __init__(
        self,
        cfg: pydantic.BaseModel,
        vector_field: eqx.Module,
        interpolation: tp.Literal["linear", "cubic"],
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
            key=encoder_key,
        )

        self.method = getattr(diffrax, self.cfg.method)()

        self.controller = diffrax.ConstantStepSize()
        # self.controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

        self.wrapped_vector_field = vector_fields.CDEWrapperVectorField(
            self.vector_field, self.cfg.hidden_dim
        )

        assert not isinstance(
            self.vector_field, vector_fields.GNODEFloorVectorField
        ), "GNODEFloorVectorField is not supported for GraphNeuralCDE."

    def __call__(
        self,
        ts: jax.Array,
        coeffs_adj: jax.Array,
        x_coeffs: jax.Array,
        x0: jax.Array,
        evolving_out: bool = False,
        global_readout: bool = True,  # make this class var
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
            control_adj = diffrax.LinearInterpolation(ts, coeffs_adj)
            control_data = diffrax.LinearInterpolation(ts, x_coeffs)
        elif self.interpolation == "cubic":
            control_adj = diffrax.CubicInterpolation(ts, coeffs_adj)
            control_data = diffrax.CubicInterpolation(ts, x_coeffs)

        term = diffrax.ODETerm(self.wrapped_vector_field)

        dt0 = 0.1

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
            args=[control_adj, control_data],
            stepsize_controller=self.controller,
            saveat=saveat,
        )

        output = jax.vmap(self.decoder)(latent_node_path.ys[-1])

        if global_readout:
            return jnp.sum(output, axis=0)
        else:
            return output

    # def __call__(
    #     self,
    #     ts: jax.Array,
    #     coeffs_adj: jax.Array,
    #     x_coeffs: jax.Array,
    #     x0: jax.Array,
    #     evolving_out: bool = False,
    #     global_readout: bool = True,  # make this class var
    #     **kwargs,
    # ) -> jax.Array:
    #     """
    #     Forward pass of the GraphNeuralCDE model.

    #     Args:
    #         ts (jax.Array): Sequence of time points.
    #         coeffs_adj (jax.Array): Sequence of control coefficients.
    #         x0 (jax.Array): Initial state.
    #         evolving_out (bool): Flag to determine if the output should be saved at all time points.

    #     Returns:
    #         jax.Array: The output of the model.
    #     """
    #     if self.interpolation == "linear":
    #         control_adj = diffrax.LinearInterpolation(ts, coeffs_adj)
    #     elif self.interpolation == "cubic":
    #         control_adj = diffrax.CubicInterpolation(ts, coeffs_adj)

    #     term = diffrax.ODETerm(self.vector_field)

    #     dt0 = 0.01
    #     y0 = jax.vmap(self.encoder)(x0)

    #     if evolving_out:
    #         saveat = diffrax.SaveAt(ts=ts)
    #     else:
    #         saveat = diffrax.SaveAt(t1=True)

    #     latent_node_path = diffrax.diffeqsolve(
    #         terms=term,
    #         solver=self.method,
    #         t0=ts[0],
    #         t1=ts[-1],
    #         dt0=dt0,
    #         y0=y0,
    #         args=control_adj,
    #         stepsize_controller=self.controller,
    #         saveat=saveat,
    #     )

    #     if self.cfg.return_sequence:
    #         output = jax.vmap(jax.vmap(self.final_linear, in_axes=0), in_axes=0)(
    #             latent_node_path.ys
    #         )
    #     else:
    #         output = jax.vmap(self.decoder)(latent_node_path.ys[-1])

    #     if global_readout:
    #         return jnp.sum(output, axis=0)
    #     else:
    #         return output
