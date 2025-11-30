import typing as tp

import equinox as eqx
import pydantic

from models import neural_nets


class NeuralNetsCfg(pydantic.BaseModel):
    """
    Configuration for a neural network.
    """

    name: tp.Literal["FinalTanhF",] = pydantic.Field(..., description="Name of Model")
    hidden_dim: int = pydantic.Field(64, description="Dimension of the hidden layers")
    data_embed_dim: int = pydantic.Field(
        16, description="Dimension of the data embedding"
    )
    num_layers: int = pydantic.Field(4, description="Number of layers in the model")
    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self, nn_key: str) -> eqx.Module:
        """
        Builds and returns an instance of the vector field.

        Args:
            vf_key (str): The key for the vector field.

        Returns:
            eqx.Module: An instance of the vector field.
        """
        neural_net_cls = getattr(neural_nets, self.name)
        return neural_net_cls(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim * self.data_embed_dim,
            num_layers=self.num_layers,
            key=nn_key,
        )
