# 3rd party libraries
import numpy as np
from pydantic import BaseModel, field_validator
from typing_extensions import Any

# VAHSimulator library
from .dynamics import (
    Dynamics6DOF,
    Dynamics3DOF_PointMass,
    StraightManeuver,
    PitchOverManeuver,
    GravityTurnManeuver,
    PullUpManeuver,
)
from .end_condition import EndConditions
from .vehicle_state import VehicleState

class FlightPhase(BaseModel):

    model_config = {"arbitrary_types_allowed": True}
        
    actions: list[dict[str, Any]] = []
    end_condition: EndConditions
    dynamics: Dynamics6DOF | Dynamics3DOF_PointMass | None = None
    maneuver: StraightManeuver | PitchOverManeuver | GravityTurnManeuver | PullUpManeuver | None = None

class MissionPlan(BaseModel):

    model_config = {
        "arbitrary_types_allowed": True
    }
        
    initial_state: VehicleState
    flight_phases: list[FlightPhase]

    @field_validator("initial_state", mode="before")
    @classmethod
    def _dict_to_vehicle_state(cls, v: dict[str, Any]) -> VehicleState:
        if isinstance(v, dict):
            return VehicleState.from_dict(v)
        return v
