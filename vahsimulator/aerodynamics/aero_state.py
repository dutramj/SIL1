# Python standard libraries
from __future__ import annotations
from dataclasses import dataclass


# 3rd party libraries
import numpy as np
from typing_extensions import Self

# VAHSimulator library
from ..vehicle_state import VehicleState
from ..atmosphere import AtmosphereData

@dataclass(slots=True, frozen=True)
class AeroState:
    
    alpha : np.float64 = 0.0  # angle of attack
    alpha_dot : np.float64 = 0.0  # derivative of angle of attack
    alpha_total : np.float64 = 0.0  # total angle of attack
    beta : np.float64 = 0.0  # sideslip angle
    Va : np.float64 = 0.0  # true airspeed
    mach : np.float64 = 0.0  # mach number
    Q : np.float64 = 0.0  # dynamic pressure
    Q_Va: np.float64 = 0.0 # dynamic pressure over true airspeed
    x_cp : np.float64 = 0.0  # center of pressure location

    @classmethod
    def from_data(cls, aero_state_previous: AeroState, state: VehicleState, atmosphere_data: AtmosphereData, wb: np.ndarray, dt: np.float64) -> Self:

        vGb = state.velocity_body
        vAb = np.add(vGb, -wb[0:3])  # airspeed in the body frame [u'; v'; w']
        Va = np.sqrt(vAb[0, 0]**2 + vAb[1, 0]**2 + vAb[2, 0]**2)  # true airspeed
        Va = max(Va, 1e-6)

        alpha = np.rad2deg(np.atan2(vAb[2, 0], vAb[0, 0]))
        beta = np.rad2deg(np.asin(vAb[1, 0] / Va))

        alpha_total = np.rad2deg(np.acos(vAb[0, 0] / Va))

        alpha_dot = ((alpha - aero_state_previous.alpha) / dt)

        mach = Va / atmosphere_data.speed_of_sound_m_s

        Q = 0.5 * atmosphere_data.density_kg_m3 * Va**2
        Q_Va = 0.5 * atmosphere_data.density_kg_m3 * Va

        return cls(
            alpha = alpha,
            alpha_dot = alpha_dot,
            alpha_total = alpha_total,
            beta = beta,
            Va = Va,
            mach = mach,
            Q = Q,
            Q_Va = Q_Va,
        )
