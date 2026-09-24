# Python standard libraries
from __future__ import annotations
from abc import ABC, abstractmethod

# 3rd party libraries
import numpy as np
from typing_extensions import Literal, Any
from pydantic import BaseModel, PrivateAttr

# VAHSimulator library
from ..aerodynamics import AeroState
from ..optimize import OptimizationVariable
from ..pid_controller import PIDGains, PIDController
from ..vehicle_state import VehicleState


class ManeuverState(BaseModel, ABC):

    model_config = {
        "arbitrary_types_allowed": True
    }

    _t0: np.float64 = PrivateAttr()
    _tf: np.float64 = PrivateAttr()

    def initialize(self, t0: np.float64, tf: np.float64) -> np.ndarray:
        self._t0 = t0
        self._tf = tf

    @abstractmethod
    def evaluate(self, context: ManeuverContext, state: VehicleState, aero_state: AeroState, dt: np.float64) -> np.ndarray:
        ...
 

class StraightManeuver(ManeuverState):

    type: Literal["straight"] = "straight"

    def evaluate(self, context: ManeuverContext, state: VehicleState, aero_state: AeroState, dt: np.float64) -> np.ndarray:
        context.reset()
        return np.zeros(3, dtype=np.float64)
 

class PitchManeuver(ManeuverState, ABC):

    @abstractmethod
    def _generate_alpha_reference(self, time: np.float64) -> np.float64:
        ...

    def evaluate(self, context: ManeuverContext, state: VehicleState, aero_state: AeroState, dt: np.float64) -> np.ndarray:

        gains = PIDGains(
            kp=context._base_gains.kp/dt,
            ki=context._base_gains.ki/dt,
            kd=context._base_gains.kd/dt,
        )
        context._alpha_controller.gains = gains

        alpha_setpoint = self._generate_alpha_reference(state.time)

        angular_rate_q = context._alpha_controller.step(
            setpoint=alpha_setpoint,
            measurement=np.deg2rad(aero_state.alpha),
            dt=dt,
        )

        return np.array([0.0, angular_rate_q, 0.0], dtype=np.float64)
 

class GravityTurnManeuver(PitchManeuver):

    type: Literal["gravity_turn"] = "gravity_turn"

    def _generate_alpha_reference(self, time: np.float64) -> np.float64:
        
        return np.float64(0.0)


class PullUpManeuver(PitchManeuver):

    type: Literal["pull_up"] = "pull_up"
    t1: np.float64 | OptimizationVariable = np.float64(0.0)
    tm: np.float64 | OptimizationVariable
    tp: np.float64 | OptimizationVariable
    t2: np.float64 | OptimizationVariable = np.inf
    alpha_max: np.float64 | OptimizationVariable

    def initialize(self, t0: np.float64, tf: np.float64) -> np.ndarray:
        super().initialize(t0, tf)

        if self.t2 == np.inf:
            if np.inf and tf == np.inf:
                raise Exception('t2 must be specified por maneuver whenever the maneuver duration is unknown!')
            else:
                self.t2 = self._tf - self._t0

    def _generate_alpha_reference(self, time: np.float64) -> np.float64:

        t  = time - self._t0
        t1 = self.t1
        t2 = self.t2
        if (t < t1 or t > t2):
            return np.float64(0.0)

        tm = self.tm
        tp = self.tp
        if (t < tm):
            phi = np.pi*(t - t1)/((tm - t1)*(t2 - t)/(t2-tm) + (t-t1))
        elif (t > tp):
            phi = np.pi*(t - t1)/((tp - t1)*(t2 - t)/(t2-tp) + (t-t1))
        else:
            phi = np.pi/2
        
        return np.deg2rad(self.alpha_max) * (np.sin(phi)**2)


class PitchOverManeuver(PullUpManeuver):

    type: Literal["pitch_over"] = "pitch_over"

    def _generate_alpha_reference(self, time: np.float64) -> np.float64:
        return -super()._generate_alpha_reference(time)


class ManeuverContext(BaseModel):

    state: ManeuverState | None = None
    _base_gains: PIDGains = PrivateAttr()
    _alpha_controller: PIDController = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
 
        self._base_gains = PIDGains(
            kp=np.float64(1.5),
            ki=np.float64(2.0),
            kd=np.float64(0.0),
        )

        self._alpha_controller = PIDController(
            gains=self._base_gains,
        )

    def reset(self) -> None:
        self._alpha_controller.reset()

    def transition_to(self, state: ManeuverState):
        self.state = state

    def evaluate(self, *args, **kwargs) -> np.ndarray:
        return self.state.evaluate(self, *args, **kwargs)
 