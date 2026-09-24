# Python standard libraries
from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import fields

# 3rd party libraries
import numpy as np
from pydantic import BaseModel, PrivateAttr
from typing_extensions import Any

# VAHSimulator library
from ..aerodynamics import AeroState
from ..loads import Loads
from ..mass_properties import MassPropertiesData
from ..vehicle_state import VehicleState
from .. import performance_decorator


class DynamicsBase(BaseModel, ABC):

    _rk4_step: Callable = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:

        # Dummy call of rk4_step to force compilation
        state_dummy = VehicleState.from_vector(
            time=np.float64(0.0),
            interp_time=np.float64(0.0),
            roll_ned=np.float64(0.0),
            pitch_ned=np.float64(0.0),
            yaw_ned=np.float64(0.0),
            state_array = np.arange(1, 14, dtype=np.float64).reshape(-1, 1),
        )
        mpd_dummy = MassPropertiesData(*np.arange(1, len(MassPropertiesData._fields) + 1, dtype=np.float64))
        loads_dummy = Loads(*np.arange(1, len(fields(Loads)) + 1, dtype=np.float64))
        self._rk4_step(
            state_dummy.vector,
            loads_dummy.vector,
            np.float64(0.01),
            mpd_dummy,
        )

    @performance_decorator.time_execution_stats
    def _state_integration(self, state_vector: np.ndarray, loads:Loads, mpd: MassPropertiesData, dt: np.float64) -> np.ndarray:
        new_state_vector = self._rk4_step(
            state_vector,
            loads.vector,
            dt,
            mpd,
        )
        return new_state_vector

    @abstractmethod
    def step(
            self,
            state: VehicleState,
            aero_state: AeroState,
            mpd: MassPropertiesData,
            total_loads: Loads,
            dt: np.float64,
        ) -> VehicleState:
       ...
