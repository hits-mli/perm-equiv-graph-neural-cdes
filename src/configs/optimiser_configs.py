import pydantic
import typing as tp

import optax


class ConstantScheduleCfg(pydantic.BaseModel):
    """
    Configuration for the Constant Schedule in Optax.
    """

    name: tp.Literal["constant_schedule"] = pydantic.Field(
        default="constant_schedule", description="Name of the schedule"
    )
    value: float = 0.01

    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self) -> optax.Schedule:
        """Builds the constant schedule."""
        schedule_cls = getattr(optax, self.name)
        return schedule_cls(value=self.value)


class WarmupCosineDecayScheduleCfg(pydantic.BaseModel):
    """
    Configuration for the Warmup Cosine Decay Schedule in Optax.
    """

    name: tp.Literal["warmup_cosine_decay_schedule", "sgd", "adamw"] = pydantic.Field(
        ..., description="Name of the schedule"
    )
    init_value: float = 0.0
    peak_value: float = 1.0
    warmup_steps: int = 50
    decay_steps: int = 1_000
    end_value: float = 0.0

    model_config = pydantic.ConfigDict(extra="forbid")

    def build(self) -> optax.Schedule:
        """Builds the warmup cosine decay schedule."""
        schedule_cls = getattr(optax, self.name)
        return schedule_cls(
            init_value=self.init_value,
            peak_value=self.peak_value,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.end_value,
        )


class OptimiserCfg(pydantic.BaseModel):
    """
    Configuration for the Optimiser in Optax.
    """

    name: tp.Literal["adam", "sgd", "adamw"] = pydantic.Field(
        ..., description="Name of the optimizer"
    )
    weight_decay: float = 0.0

    schedule: tp.Union[ConstantScheduleCfg, WarmupCosineDecayScheduleCfg] = (
        pydantic.Field(..., discriminator="name")
    )
    gradient_clipping: bool = pydantic.Field(..., description="Gradient clipping")

    model_config = pydantic.ConfigDict(extra="forbid")

    def build(
        self, optimiser_key: str
    ) -> tp.Tuple[optax.GradientTransformation, optax.Schedule]:
        """Builds the optimizer and schedule."""
        optimiser_cls = getattr(optax, self.name)
        schedule = self.schedule.build()
        optimiser = optimiser_cls(
            learning_rate=schedule,
            weight_decay=self.weight_decay,
        )
        if self.gradient_clipping:
            optimiser = optax.chain(
                optax.clip_by_global_norm(1.0),
                optimiser,
            )
        return (
            optimiser,
            schedule,
        )
