import pydantic
import typing as tp

import jax.random as jr

from .vector_field_configs import VectorFieldCfg
from .neural_nets_configs import NeuralNetsCfg

from models import (
    GraphNeuralCDE,
    TGBGraphNeuralCDE,
    TGBGraphNeuralODE,
    TGBSTGraphNeuralCDE,
    GraphNeuralODE,
    PGTGraphNeuralODE,
    PGTGraphNeuralCDE,
    PGTSTGraphNeuralCDE,
    STIDGCN,
    ASTGCN,
    DCRNNModelSingleStep,
)


class GraphNeuralCDECfg(pydantic.BaseModel):
    """
    Configuration for the Graph Neural CDE model.
    """

    name: tp.Literal["graph_neural_cde"] = pydantic.Field(
        ..., description="Name of Model"
    )
    vector_field: VectorFieldCfg
    hidden_dim: int = 64
    interpolation: tp.Literal["linear", "rectilinear", "cubic", "cubic_hermite"] = (
        "linear"
    )
    use_initial: bool = True
    method: tp.Literal[
        "Tsit5",
        "Kvaerno3",
    ] = pydantic.Field(default="Tsit5")
    return_sequence: bool = True

    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self, model_key: jr.PRNGKey) -> GraphNeuralCDE:
        """
        Builds the GraphNeuralCDE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralCDE: The initialized GraphNeuralCDE model.
        """
        model_key, vf_key = jr.split(model_key, 2)

        vector_field = self.vector_field.build(vf_key)
        return GraphNeuralCDE(self, vector_field, self.interpolation, model_key)


class TGBGraphNeuralCDECfg(pydantic.BaseModel):
    """
    Configuration for the Graph Neural CDE model.
    """

    name: tp.Literal["tgb_graph_neural_cde"] = pydantic.Field(
        ..., description="Name of Model"
    )
    vector_field: VectorFieldCfg
    hidden_dim: int = 64

    interpolation: tp.Literal["linear", "rectilinear", "cubic", "cubic_hermite"] = (
        "linear"
    )
    use_initial: bool = True
    method: tp.Literal[
        "Tsit5",
        "Kvaerno3",
        "Kvaerno5",
    ] = pydantic.Field(default="Tsit5")
    return_sequence: bool = True

    model_config = pydantic.ConfigDict(extra="forbid")
    use_mlps: bool = False

    def build(self, model_key: jr.PRNGKey) -> GraphNeuralCDE:
        """
        Builds the GraphNeuralCDE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralCDE: The initialized GraphNeuralCDE model.
        """
        model_key, vf_key = jr.split(model_key, 2)

        vector_field = self.vector_field.build(vf_key)
        return TGBGraphNeuralCDE(self, vector_field, self.interpolation, model_key)


class GraphNeuralODECfg(pydantic.BaseModel):
    """
    Configuration for the Graph Neural ODE model.
    """

    name: tp.Literal["graph_neural_ode"] = pydantic.Field(
        ..., description="Name of Model"
    )
    vector_field: VectorFieldCfg
    hidden_dim: int = 64

    interpolation: tp.Literal["linear", "rectilinear", "cubic", "cubic_hermite"] = (
        "linear"
    )
    use_initial: bool = True
    method: tp.Literal[
        "Tsit5",
        "Kvaerno3",
    ] = pydantic.Field(default="Tsit5")
    return_sequence: bool = True

    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self, model_key: jr.PRNGKey) -> GraphNeuralODE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """
        model_key, vf_key = jr.split(model_key, 2)

        vector_field = self.vector_field.build(vf_key)
        return GraphNeuralODE(self, vector_field, self.interpolation, model_key)


class TGBGraphNeuralODECfg(pydantic.BaseModel):
    """
    Configuration for the Graph Neural ODE model.
    """

    name: tp.Literal["tgb_graph_neural_ode"] = pydantic.Field(
        ..., description="Name of Model"
    )
    vector_field: VectorFieldCfg
    hidden_dim: int = 64

    interpolation: tp.Literal["linear", "rectilinear", "cubic", "cubic_hermite"] = (
        "linear"
    )
    use_initial: bool = True
    method: tp.Literal[
        "Tsit5",
        "Kvaerno3",
    ] = pydantic.Field(default="Tsit5")
    return_sequence: bool = True

    model_config = pydantic.ConfigDict(extra="forbid")
    use_mlps: bool = False

    def build(self, model_key: jr.PRNGKey) -> TGBGraphNeuralODE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """
        model_key, vf_key = jr.split(model_key, 2)

        vector_field = self.vector_field.build(vf_key)
        return TGBGraphNeuralODE(self, vector_field, self.interpolation, model_key)


class TGBSTGraphNeuralODECfg(pydantic.BaseModel):
    """
    Configuration for the Graph Neural ODE model.
    """

    name: tp.Literal["tgb_st_graph_neural_cde"] = pydantic.Field(
        ..., description="Name of Model"
    )
    f_func: NeuralNetsCfg
    g_func: VectorFieldCfg
    interpolation: tp.Literal["linear", "rectilinear", "cubic", "cubic_hermite"] = (
        "linear"
    )
    use_initial: bool = True
    method: tp.Literal[
        "Tsit5", "Kvaerno3", "Kvaerno4", "Kvaerno5", "Dopri5", "Dopri8"
    ] = pydantic.Field(default="Tsit5")
    return_sequence: bool = True

    model_config = pydantic.ConfigDict(extra="forbid")
    use_mlps: bool = False

    def build(self, model_key: jr.PRNGKey) -> TGBSTGraphNeuralCDE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """
        model_key, f_func_key, g_func_key = jr.split(model_key, 3)

        f_func = self.f_func.build(f_func_key)
        g_func = self.g_func.build(g_func_key, hidden_dim_f=self.f_func.hidden_dim)
        return TGBSTGraphNeuralCDE(
            self,
            f_func,
            g_func,
            self.interpolation,
            self.f_func.data_embed_dim,
            model_key,
        )


class PGTGraphNeuralCDECfg(pydantic.BaseModel):
    """
    Configuration for the Graph Neural ODE model.
    """

    name: tp.Literal["pgt_graph_neural_cde"] = pydantic.Field(
        ..., description="Name of Model"
    )
    vector_field: VectorFieldCfg
    hidden_dim: int = 64
    data_dim: int
    feature_dim: int

    interpolation: tp.Literal["linear", "rectilinear", "cubic", "cubic_hermite"] = (
        "linear"
    )
    use_initial: bool = True
    method: tp.Literal[
        "Tsit5",
        "Kvaerno3",
    ] = pydantic.Field(default="Tsit5")
    return_sequence: bool = True

    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self, model_key: jr.PRNGKey) -> PGTGraphNeuralCDE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """
        model_key, vf_key = jr.split(model_key, 2)

        vector_field = self.vector_field.build(vf_key)
        return PGTGraphNeuralCDE(self, vector_field, self.interpolation, model_key)


class PGTGraphNeuralODECfg(pydantic.BaseModel):
    """
    Configuration for the Graph Neural ODE model.
    """

    name: tp.Literal["pgt_graph_neural_ode"] = pydantic.Field(
        ..., description="Name of Model"
    )
    vector_field: VectorFieldCfg
    hidden_dim: int = 64
    data_dim: int
    feature_dim: int

    interpolation: tp.Literal["linear", "rectilinear", "cubic", "cubic_hermite"] = (
        "linear"
    )
    use_initial: bool = True
    method: tp.Literal[
        "Tsit5",
        "Kvaerno3",
    ] = pydantic.Field(default="Tsit5")
    return_sequence: bool = True

    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self, model_key: jr.PRNGKey) -> PGTGraphNeuralODE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """
        model_key, vf_key = jr.split(model_key, 2)

        vector_field = self.vector_field.build(vf_key)
        return PGTGraphNeuralODE(self, vector_field, self.interpolation, model_key)


class PGTSTGraphNeuralCDECfg(pydantic.BaseModel):
    """
    Configuration for the Graph Neural ODE model.
    """

    name: tp.Literal["pgt_st_graph_neural_cde"] = pydantic.Field(
        ..., description="Name of Model"
    )
    f_func: NeuralNetsCfg
    g_func: VectorFieldCfg
    interpolation: tp.Literal["linear", "rectilinear", "cubic", "cubic_hermite"] = (
        "linear"
    )
    use_initial: bool = True
    method: tp.Literal[
        "Tsit5", "Kvaerno3", "Kvaerno4", "Kvaerno5", "Dopri5", "Dopri8"
    ] = pydantic.Field(default="Tsit5")
    return_sequence: bool = True

    data_dim: int
    feature_dim: int

    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self, model_key: jr.PRNGKey) -> PGTSTGraphNeuralCDE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """
        model_key, f_func_key, g_func_key = jr.split(model_key, 3)

        f_func = self.f_func.build(f_func_key)
        g_func = self.g_func.build(g_func_key, hidden_dim_f=self.f_func.hidden_dim)
        return PGTSTGraphNeuralCDE(
            self,
            f_func,
            g_func,
            self.interpolation,
            self.data_dim,
            model_key,
        )


class StidGCNCfg(pydantic.BaseModel):
    """
    Configuration for the StidGCN model.
    """

    name: tp.Literal["stid_gcn"] = pydantic.Field(..., description="Name of Model")
    input_dim: int = pydantic.Field(..., description="Hidden dimension of the model")
    num_nodes: int = pydantic.Field(..., description="Number of nodes in the graph")
    num_time_steps: int = pydantic.Field(
        ..., description="Number of time steps in the graph"
    )
    channels: int = pydantic.Field(..., description="Number of channels in the graph")
    output_len: int = pydantic.Field(default=1, description="Output length")
    granularity: int = pydantic.Field(..., description="Granularity of the graph")
    dropout: float = pydantic.Field(..., description="Dropout rate of the graph")
    memory_dim1: int = pydantic.Field(..., description="Memory dimension 1")
    memory_dim2: int = pydantic.Field(..., description="Memory dimension 2")

    def build(self, model_key: jr.PRNGKey) -> PGTSTGraphNeuralCDE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """

        return STIDGCN(
            self.input_dim,
            self.num_nodes,
            self.num_time_steps,
            self.channels,
            self.output_len,
            self.granularity,
            self.dropout,
            self.memory_dim1,
            self.memory_dim2,
            key=model_key,
        )


class ASTGCNCfg(pydantic.BaseModel):
    """
    Configuration for the ASTGCN model.
    """

    name: tp.Literal["astgcn"] = pydantic.Field(..., description="Name of Model")
    input_dim: int = pydantic.Field(..., description="Hidden dimension of the model")
    output_dim: int = pydantic.Field(
        default=1, description="Output dimension of the model"
    )
    num_nodes: int = pydantic.Field(..., description="Number of nodes in the graph")
    node_embed_dim: int = pydantic.Field(
        ..., description="Dimension of node embeddings"
    )
    num_time_steps: int = pydantic.Field(
        ..., description="Number of time steps in the graph"
    )
    channels: int = pydantic.Field(..., description="Number of channels in the graph")
    K: int = pydantic.Field(..., description="Order of Chebyshev polynomials")

    def build(self, model_key: jr.PRNGKey) -> PGTSTGraphNeuralCDE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """

        backbones = [
            {
                "K": self.K,
                "num_of_chev_filters": self.input_dim,
                "num_of_time_filters": self.input_dim,
                "time_conv_strides": self.num_time_steps,
            },
            {
                "K": self.K,
                "num_of_chev_filters": self.input_dim,
                "num_of_time_filters": self.input_dim,
                "time_conv_strides": 1,
            },
        ]

        return ASTGCN(
            num_for_prediction=self.output_dim,
            backbones=backbones,
            num_vertices=self.num_nodes,
            node_embed_dim=self.node_embed_dim,
            num_features=self.input_dim,
            num_timesteps=self.num_time_steps,
            key=model_key,
        )


class DCRNNCfg(pydantic.BaseModel):
    """
    Configuration for the DCRNN model.
    """

    name: tp.Literal["dcrnn"] = pydantic.Field(..., description="Name of Model")
    input_dim: int = pydantic.Field(..., description="Hidden dimension of the model")
    num_nodes: int = pydantic.Field(..., description="Number of nodes in the graph")
    hidden_dim: int = pydantic.Field(..., description="Hidden dimension of the model")
    output_dim: int = pydantic.Field(
        default=1, description="Output dimension of the model"
    )
    num_time_steps: int = pydantic.Field(
        ..., description="Number of time steps in the graph"
    )
    order: int = pydantic.Field(..., description="Order of Chebyshev polynomials")
    num_layers: int = pydantic.Field(..., description="Number of layers in the model")

    def build(self, model_key: jr.PRNGKey) -> PGTSTGraphNeuralCDE:
        """
        Builds the GraphNeuralODE model.

        Args:
            model_key (jr.PRNGKey): The random key for model initialization.

        Returns:
            GraphNeuralODE: The initialized GraphNeuralODE model.
        """

        return DCRNNModelSingleStep(
            num_node=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            order=self.order,
            num_layers=self.num_layers,
            key=model_key,
        )
