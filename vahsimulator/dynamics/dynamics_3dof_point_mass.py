# STEVENS, Brian L.; LEWIS, Frank L.; JOHNSON, Eric N. Aircraft control and simulation: dynamics, controls design, and autonomous systems. John Wiley & Sons, 1992.

# Python standard libraries
from __future__ import annotations

# 3rd party libraries
import numpy as np
import numba
from typing_extensions import Literal, Any
from pydantic import Field
from dataclasses import replace

# VAHSimulator library
from ..aerodynamics import AeroState
from ..parameters import earth_rate__rad_s
from . import DynamicsBase
from .. import performance_decorator
from ..runge_kutta import rk4_factory
from ..vehicle_state import VehicleState
from .maneuver import (
    ManeuverContext,
    ManeuverState,
)
from ..mass_properties import MassPropertiesData
from ..loads import Loads
from ..utils import eci_to_body_quaternion, skew_matrix, skew_matrix_angular_velocity, compute_euler_angles_ned


@numba.njit(cache=True)
def diff_eq(state_vector: np.ndarray, loads_vector: np.ndarray, mpd: MassPropertiesData) -> np.ndarray:
    ## Initialization and variables construction
    # Unpacking state variables
    x_I = state_vector[0, 0]
    y_I = state_vector[1, 0]
    z_I = state_vector[2, 0]
    q0 = state_vector[3, 0]
    q1 = state_vector[4, 0]
    q2 = state_vector[5, 0]
    q3 = state_vector[6, 0]
    u = state_vector[7, 0]
    v = state_vector[8, 0]
    w = state_vector[9, 0]
    p = state_vector[10, 0]
    q = state_vector[11, 0]
    r = state_vector[12, 0]

    p_I        = np.array([[x_I], [y_I], [z_I]])    # Inertial position
    quaternion = np.array([[q0], [q1], [q2], [q3]]) # Inertial attitude quaternion representation
    V_B        = np.array([[u], [v], [w]])          # Relative Velocity with Relation to Earth's surface (ECEF)
    omega_B    = np.array([[p], [q], [r]])          # Inertial Angular velocity wrt ECI represented in body frame

    # Normalizing the quaternion
    quaternion /= np.linalg.norm(quaternion)

    # Getting external forces represented in body frame
    F_B = loads_vector[0:3]

    # Creating Earth's angular velocity vector wrt ECI frame represented in ECI frame
    omega_E = np.array([[0.], [0.], [earth_rate__rad_s]])

    # Representing Earth's angular velocity in body axis
    LBI = eci_to_body_quaternion(quaternion)
    omega_E_B = LBI @ omega_E

    # Computing relative angular velocity (relative to ECI) represented in body frame
    omega_R = omega_B - omega_E_B
    omega_T = omega_R + 2 * omega_E_B

    # Computing angular velocity skew matrices
    omega_E_sm = skew_matrix(omega_E)

    omega_T_sm = skew_matrix(omega_T)

    # Computing angular velocity skew matrix for quaternion derivative computation
    omega_Q_sm = skew_matrix_angular_velocity(omega_B)

    # Computing body to ECI matrix
    LIB = np.transpose(LBI)

    ## Dynamics
    # Position
    p_I_dot = omega_E_sm @ p_I + LIB @ V_B # Equation 1.3-2

    # Orientation
    q_dot = -0.5 * omega_Q_sm @ quaternion # Equation 1.4-29

    # Translational motion
    V_B_dot = (1 / mpd.mass) * F_B - omega_T_sm @ V_B # Equation 1.3-8

    # Rotational motion
    omega_B_dot = np.zeros((3, 1), dtype=np.float64)

    ## Creating the state array derivative
    x_dot = np.array(
        [[p_I_dot[0, 0]], [p_I_dot[1, 0]], [p_I_dot[2, 0]],            # Inertial position time derivative
         [q_dot[0, 0]], [q_dot[1, 0]], [q_dot[2, 0]], [q_dot[3, 0]],   # quaternion wrt ECI time derivative
         [V_B_dot[0, 0]], [V_B_dot[1, 0]], [V_B_dot[2, 0]],            # Relative Velocity time derivative
         [omega_B_dot[0, 0]], [omega_B_dot[1, 0]], [omega_B_dot[2, 0]] # Angular velocity wrt ECI time derivative
        ],
        dtype=np.float64
    )

    return x_dot


class Dynamics3DOF_PointMass(DynamicsBase):

    type: Literal["3DOF_PointMass"] = "3DOF_PointMass"
    maneuver: ManeuverContext = Field(
        default_factory=lambda: ManeuverContext()
    )

    def model_post_init(self, __context: Any) -> None:
        self._rk4_step = rk4_factory(diff_eq)
        super().model_post_init(__context)
    
    def set_maneuver(self, maneuver: ManeuverState):
        self.maneuver.transition_to(maneuver)
    
    @performance_decorator.time_execution_stats
    def step(
            self,
            state: VehicleState,
            aero_state: AeroState,
            mpd: MassPropertiesData,
            total_loads: Loads,
            dt: np.float64,
            gimbal_lock_protection: bool,
        ) -> VehicleState:

        state_vector = state.vector
        omega_r = self.maneuver.evaluate(
            state=state,
            aero_state=aero_state,
            dt=dt,
        )

        omega_e = np.array([[0.,], [0.,], [earth_rate__rad_s]], dtype=np.float64)
        LBI = eci_to_body_quaternion(state.quaternion)
        omega_e_b = LBI @ omega_e
        omega_b = omega_r + omega_e_b.reshape(3,)

        previous_euler_angles_ned__rad = np.array([state.roll_ned[0], state.pitch_ned[0], state.yaw_ned[0]])

        state_vector[10:13,0] = omega_b
        new_state_vector = self._state_integration(state_vector, total_loads, mpd, dt)
        new_time = state.time + dt
        new_interp_time = state.interp_time + dt

        euler_angles_ned__rad, gimbal_lock_protection = compute_euler_angles_ned(
            new_state_vector, 
            new_time,
            previous_euler_angles_ned__rad,
            gimbal_lock_protection
        )

        roll_ned, pitch_ned, yaw_ned = euler_angles_ned__rad

        new_state = VehicleState.from_vector(new_time, new_interp_time, roll_ned, pitch_ned, yaw_ned, new_state_vector)

        return new_state, gimbal_lock_protection
