from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .discrete_gust import DiscreteGustParameters


class ConstantWindConfig(BaseModel):
    """Configuration of the constant wind."""

    model_config = ConfigDict(extra="forbid")

    wind_speed: float = Field(default=0.0, ge=0.0)

    wind_direction: float = Field(default=0.0)

    wind_elevation: float = Field(default=0.0, ge=-90.0, le=90.0)


class TurbulenceConfig(BaseModel):
    """Configuration of atmospheric turbulence."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["dryden"] = "dryden"

    severity: Literal["light", "moderate", "severe"] = "moderate"


class WindConfig(BaseModel):
    """Configuration of the complete wind model."""

    model_config = ConfigDict(extra="forbid")

    constant_wind: ConstantWindConfig = Field(default_factory=ConstantWindConfig)

    turbulence: TurbulenceConfig | None = None

    discrete_gusts: list[DiscreteGustParameters] = Field(default_factory=list)
