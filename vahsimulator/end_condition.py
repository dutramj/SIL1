# 3rd party libraries
import numpy as np
from pydantic import BaseModel, PrivateAttr
from typing_extensions import Literal, Union, Optional

# VAHSimulator library
from .vehicle_state import VehicleState
from .aerodynamics import aero_state


class BaseEndCondition(BaseModel):

    model_config = {"arbitrary_types_allowed": True}

    def initialize(self, state: VehicleState, aero_state: aero_state) -> bool:
        pass

    def evaluate(self, state: VehicleState, aero_state: aero_state) -> bool:
        pass


class DurationEndCondition(BaseEndCondition):

    type: Literal['duration']
    value: np.float64
    _t0: np.float64 = PrivateAttr()

    def initialize(self, state: VehicleState, aero_state: aero_state) -> bool:
        self._t0 = state.time

    def evaluate(self, state: VehicleState, aero_state: aero_state) -> bool:
        if state.time - self._t0 > self.value:
            return True
        return False


class TimeEndCondition(BaseEndCondition):

    type: Literal['time']
    value: np.float64

    def evaluate(self, state: VehicleState, aero_state: aero_state) -> bool:
        if state.time > self.value:
            return True
        return False


class LaunchTowerEndCondition(BaseEndCondition):

    type: Literal['launch_tower']
    value: np.float64
    _alt0: np.float64 = PrivateAttr()

    def initialize(self, state: VehicleState, aero_state: aero_state) -> bool:
        self._alt0 = state.alt

    def evaluate(self, state: VehicleState, aero_state: aero_state) -> bool:
        if state.alt - self._alt0 > self.value:
            return True
        return False


class TargetGammaEndCondition(BaseEndCondition):

    type: Literal['target_gamma']
    target: np.float64
    direction: Literal["rising", "falling"]

    _prev_error: Optional[np.float64] = None

    def evaluate(self, state: VehicleState, aero_state: aero_state) -> bool:

        target = np.deg2rad(self.target)
        error = state.gamma - target

        if self._prev_error is None:
            self._prev_error = error
            return False

        prev = self._prev_error
        self._prev_error = error

        if self.direction == "rising":
            return prev < 0 and error >= 0

        if self.direction == "falling":
            return prev > 0 and error <= 0

        return False
    
class TargetDynamicPressureCondition(BaseEndCondition):

    type: Literal['target_qdyn']
    target: np.float64
    direction: Literal["rising", "falling"]

    _prev_error: Optional[np.float64] = None

    def evaluate(self, state: VehicleState, aero_state: aero_state) -> bool:
        
        error = aero_state.Q - self.target

        if self._prev_error is None:
            self._prev_error = error
            return False

        prev = self._prev_error
        self._prev_error = error

        if self.direction == "rising":
            return prev < 0 and error >= 0

        if self.direction == "falling":
            return prev > 0 and error <= 0

        return False
    
class TargetAltitudeCondition(BaseEndCondition):

    type: Literal['target_altitude']
    target: np.float64
    direction: Literal["rising", "falling"]

    _prev_error: Optional[np.float64] = None

    def evaluate(self, state: VehicleState, aero_state: aero_state) -> bool:
        
        error = state.alt - self.target

        if self._prev_error is None:
            self._prev_error = error
            return False

        prev = self._prev_error
        self._prev_error = error

        if self.direction == "rising":
            return prev < 0 and error >= 0

        if self.direction == "falling":
            return prev > 0 and error <= 0

        return False

EndConditions = Union[
    DurationEndCondition,
    TimeEndCondition,
    LaunchTowerEndCondition,
    TargetGammaEndCondition,
    TargetDynamicPressureCondition,
    TargetAltitudeCondition,
]
