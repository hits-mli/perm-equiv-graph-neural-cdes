import typing as tp

import equinox as eqx
import jax.random as jr
import jax
import diffrax
import pydantic

from .vector_fields import GNODEFloorVectorField


class GraphNeuralODE(eqx.Module):
    cfg: pydantic.BaseModel = eqx.field(static=True)
    method: str
    initial_linear: eqx.nn.MLP
    final_linear: eqx.nn.Linear
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
        initial_linear_key, final_linear_key = jr.split(model_key, 2)
        self.initial_linear = eqx.nn.Linear(
            in_features=1, out_features=self.cfg.hidden_dim, key=initial_linear_key
        )
        self.final_linear = eqx.nn.Linear(
            in_features=self.cfg.hidden_dim, out_features=1, key=final_linear_key
        )

        self.method = getattr(diffrax, self.cfg.method)()
        self.controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

        assert isinstance(
            self.vector_field, GNODEFloorVectorField
        ), "GNODEFloorVectorField is the only supported vector field for GraphNeuralODE."

    def __call__(
        self,
        ts: jax.Array,
        coeffs_adj: jax.Array,
        x0: jax.Array,
        adjacency_list: tp.Optional[jax.Array] = None,
        events_time: tp.Optional[jax.Array] = None,
        evolving_out: bool = True,
    ) -> jax.Array:
        """
        Forward pass of the GraphNeuralCDE model.

        Args:
            ts (jax.Array): Sequence of time points.
            coeffs_adj (jax.Array): Sequence of control coefficients.
            x0 (jax.Array): Initial state.
            evolving_out (bool): Flag to determine if the output should be saved at all time points.

        Returns:
            jax.Array: The output of the model.
        """
        if self.interpolation == "linear":
            control_adj = diffrax.LinearInterpolation(ts, coeffs_adj)
        elif self.interpolation == "cubic":
            control_adj = diffrax.CubicInterpolation(ts, coeffs_adj)

        if isinstance(self.vector_field, GNODEFloorVectorField):
            args = control_adj, adjacency_list, events_time
        else:
            args = control_adj

        term = diffrax.ODETerm(self.vector_field)

        dt0 = None
        y0 = jax.vmap(self.initial_linear)(x0)

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
            args=args,
            stepsize_controller=self.controller,
            saveat=saveat,
        )

        if self.cfg.return_sequence:
            output = jax.vmap(jax.vmap(self.final_linear, in_axes=0), in_axes=0)(
                latent_node_path.ys
            )
        else:
            output = self.final_linear(latent_node_path.ys[-1])

        return output
