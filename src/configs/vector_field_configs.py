import pydantic
import typing as tp

import equinox as eqx

from models import vector_fields


class VectorFieldCfg(pydantic.BaseModel):
    """
    Configuration for a constant learning rate schedule.
    """

    name: tp.Literal[
        "ConstVectorField",
        "GNODEVectorField",
        "GNODEFloorVectorField",
        "PermEquivGraphVectorField",
        "PermEquivDirGraphVectorField",
        "GraphVectorField",
        "PreMultFusionGraphVectorField",
        "STGraphVectorField",
    ] = pydantic.Field(..., description="Name of Model")
    hidden_dim: int = pydantic.Field(64, description="Dimension of the hidden layers")
    data_embed_dim: int = pydantic.Field(
        16, description="Dimension of the data embedding"
    )
    node_embed_dim: int = pydantic.Field(
        16, description="Dimension of the node embedding"
    )
    num_layers: int = pydantic.Field(4, description="Number of layers in the model")
    model_config = pydantic.ConfigDict(extra="forbid")
    use_control: bool = pydantic.Field(True, description="Whether to use control term")

    cheb_k: int = pydantic.Field(3, description="Chebyshev polynomial order")
    num_nodes: int = 0

    enc_idx: bool = False
    enc_type: str = "mlp"
    idx_dim: int = 512

    def build(self, vf_key: str, hidden_dim_f: int = None) -> eqx.Module:
        """
        Builds and returns an instance of the vector field.

        Args:
            vf_key (str): The key for the vector field.

        Returns:
            eqx.Module: An instance of the vector field.
        """
        vector_field_cls = getattr(vector_fields, self.name)
        if self.use_control:
            if self.name == "STGraphVectorField":
                return vector_field_cls(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim * hidden_dim_f,
                    num_layers=self.num_layers,
                    node_embed_dim=self.node_embed_dim,
                    num_nodes=self.num_nodes,
                    cheb_k=self.cheb_k,
                    key=vf_key,
                )
            else:
                if not self.enc_idx:
                    return vector_field_cls(
                        input_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        # output_dim=self.hidden_dim,
                        output_dim=self.hidden_dim * self.data_embed_dim * 2,
                        num_layers=self.num_layers,
                        data_embed_dim=self.data_embed_dim,
                        num_nodes=self.num_nodes,
                        key=vf_key,
                    )
                else:
                    return vector_field_cls(
                        input_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        # output_dim=self.hidden_dim * self.data_embed_dim * 2,
                        output_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        data_embed_dim=self.data_embed_dim,
                        num_nodes=self.num_nodes,
                        enc_idx=self.enc_idx,
                        enc_type=self.enc_type,
                        idx_dim=self.idx_dim,
                        key=vf_key,
                    )
        else:
            if self.name == "GNODEFloorVectorField":
                return vector_field_cls(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    key=vf_key,
                )
            else:
                return vector_field_cls(
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    data_embed_dim=self.data_embed_dim,
                    num_nodes=self.num_nodes,
                    key=vf_key,
                )
