# https://www.mathworks.com/help/nav/ref/gpssensor-system-object.html

# Python standard libraries
from pathlib import Path

# 3rd party libraries
import numpy as np
from numpy import exp
from numpy.linalg import norm
from numpy.random import normal
from typing_extensions import Self
import yaml

# VAHSimulator library
from .vehicle_state import VehicleState
from .utils import eci_to_enu, enu_to_eci
from .parameters import lat_ref, lon_ref, alt_ref
from . import performance_decorator


class GNSS:
    def __init__(self, params, seed=None):
        if seed is not None:
            np.random.seed(seed + 6)
            
        self.sample_rate = params['sample_rate']
        self.horizontal_position_accuracy = params['horizontal_std_noise_position']
        self.vertical_position_accuracy = params['vertical_std_noise_position']
        self.std_noise_velocity = params['std_noise_velocity']
        self.std_noise_yaw = params['std_noise_yaw']

        self.gauss_markov_time_constant = 0.999

        self.horizontal_position_noise = np.zeros((2, 1))
        self.vertical_position_noise = 0.0
        self.dt = 0

    @classmethod
    def from_yaml(cls, gnss_file: Path, seed) -> Self:

        gnss_params = yaml.safe_load(open(gnss_file))
        return cls(gnss_params, seed)

    def _simulate(self, position_eci, velocity_eci, yaw_eci, time):
        phi = exp(-self.dt*self.gauss_markov_time_constant)
        self.horizontal_position_noise = phi*self.horizontal_position_noise + normal(0, self.horizontal_position_accuracy, (2, 1))
        self.vertical_position_noise = phi*self.vertical_position_noise + normal(0, self.vertical_position_accuracy)

        position_enu = eci_to_enu(position_eci[0, 0], position_eci[1, 0], position_eci[2, 0], lat_ref, lon_ref, alt_ref, time)
        
        gnss_position_enu = np.zeros((3, 1))
        gnss_position_enu[0:2] = position_enu[0:2] + self.horizontal_position_noise
        gnss_position_enu[2] = position_enu[2] + self.vertical_position_noise

        gnss_position_eci = enu_to_eci(gnss_position_enu[0, 0], gnss_position_enu[1, 0], gnss_position_enu[2, 0], lat_ref, lon_ref, alt_ref, time)

        gnss_velocity = velocity_eci + normal(0, self.std_noise_velocity, (3, 1))
        gnss_yaw = yaw_eci + np.deg2rad(normal(0, self.std_noise_yaw))

        if self.dt < (1.0 / self.sample_rate) or gnss_position_enu[2, 0] > 18000.0 or norm(gnss_velocity) > 515.0:
            return None, None, None
        
        self.dt = 0.0

        return gnss_position_eci, gnss_velocity, gnss_yaw

    @performance_decorator.time_execution_stats
    def step(self, state: VehicleState, dt: np.float64):
        self.dt += dt
        gnss_position, gnss_velocity, gnss_yaw = self._simulate(state.position_eci, state.velocity_eci, state.yaw_eci, state.time)

        return gnss_position, gnss_velocity, gnss_yaw
