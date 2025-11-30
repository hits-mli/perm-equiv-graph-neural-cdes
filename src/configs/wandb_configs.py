import pydantic


class WandBConfig(pydantic.BaseModel):
    """
    Configuration for Weights&Biases parameters and logging.
    """

    project: str = pydantic.Field(..., description="Name of Weights&Biases Proejct")

    model_config = pydantic.ConfigDict(extra="forbid")  # <- Add this line
