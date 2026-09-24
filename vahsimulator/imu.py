# J. A. Farrell, F. O. Silva, F. Rahman and J. Wendel, "Inertial Measurement Unit Error Modeling Tutorial: Inertial Navigation System State Estimation with Real-Time Sensor Calibration," in IEEE Control Systems Magazine, vol. 42, no. 6, pp. 40-66, Dec. 2022, doi: 10.1109/MCS.2022.3209059.
# Groves, P.D. Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems; Artech House: London, UK, 2013.

# Python standard libraries
from pathlib import Path
from abc import ABC, abstractmethod

# 3rd party libraries
import numpy as np
from numpy import sqrt, log, exp, pi
from numpy.random import randn
from typing_extensions import Self
import yaml

from . import performance_decorator
from .loads import Loads
from .mass_properties import MassPropertiesData
from .vehicle_state import VehicleState


class IMUModel(ABC):

    def _inertial_data(self, state: VehicleState, loads: Loads, grav_acc, mpd : MassPropertiesData):

        state_array = state.vector

        fs_b = np.add(loads.force / mpd.mass, -grav_acc)
        gyro_b = state_array[10:13]

        return fs_b, gyro_b
    
    @abstractmethod
    def evaluate(
        self,
        state: VehicleState,
        mpd: MassPropertiesData,
        total_loads: Loads,
        grav_acc,
        dt,
    ):
        ...

class IdealIMU(IMUModel):

    def evaluate(
        self,
        state: VehicleState,
        mpd: MassPropertiesData,
        total_loads: Loads,
        grav_acc,
        dt,
    ):
        fs_b, gyro_b = self._inertial_data(state, total_loads, grav_acc, mpd)
        return fs_b, gyro_b

class RealIMU(IMUModel):

    def __init__(self, imu_params, seed=None):

        if seed is not None:
            np.random.seed(seed + 8)

        self.accel_params = imu_params['accel']
        self.gyro_params = imu_params['gyro']
        self.ba = np.zeros((3, 1))
        self.bg = np.zeros((3, 1))
        self.dt = 0
        self.min_sample_time = imu_params['min_sample_time']

        # accelerometer
        if 'repeatibility' in self.accel_params:
            self.bas = self.accel_params['repeatibility']*randn(3, 1)
        else:
            self.bas = np.zeros((3, 1))
        
        if 'misalignment' in self.accel_params:
            self.Ma = self.accel_params['misalignment']*randn(3, 3)
        else:
            self.Ma = np.zeros((3, 3))
        if 'scale_factor' in self.accel_params:
            Sa = self.accel_params['scale_factor']*randn(3, 1)
        else:
            Sa = np.zeros((3, 1))
        self.Ma[0, 0] = Sa[0, 0]
        self.Ma[1, 1] = Sa[1, 0]
        self.Ma[2, 2] = Sa[2, 0]
        self.Ma = np.add(np.eye(3), self.Ma)
        
        tba = self.accel_params['bias_stability_tb']
        self.ua = 1.0 / tba
        Sba = 2*self.accel_params['bias_stability_std']**2*log(2) / (pi*0.4365**2*tba)
        self.Qba = sqrt(Sba*(1- exp(-2*self.ua*self.min_sample_time)) / (2*self.ua))

        self.nd_std_a = self.accel_params['noise_density']*sqrt(1.0 / self.min_sample_time)  # noise density

        if 'random_walk' in self.accel_params:
            self.rw_std_a = self.accel_params['random_walk']*sqrt(self.min_sample_time)
        else:
            self.rw_std_a = 0
        self.rw_a = np.zeros((3, 1))  # random walk
        
        self.bia = np.zeros((3, 1))  # bias instability

        # gyroscope
        if 'repeatibility' in self.gyro_params:
            self.bgs = self.gyro_params['repeatibility']*randn(3, 1)
        else:
            self.bgs = np.zeros((3, 1))
        
        if 'misalignment' in self.gyro_params:
            self.Mg = self.gyro_params['misalignment']*randn(3, 3)
        else:
            self.Mg = np.zeros((3, 3))
        if 'scale_factor' in self.gyro_params:
            Sg = self.gyro_params['scale_factor']*randn(3, 1)
        else:
            Sg = np.zeros((3, 1))
        self.Mg[0, 0] = Sg[0, 0]
        self.Mg[1, 1] = Sg[1, 0]
        self.Mg[2, 2] = Sg[2, 0]
        self.Mg = np.add(np.eye(3), self.Mg)

        if 'g_dependent_bias' in self.gyro_params:
            self.Gg = self.gyro_params['g_dependent_bias']*randn(3, 3)
        else:
            self.Gg = np.zeros((3, 3))
        
        tbg = self.gyro_params['bias_stability_tb']
        self.ug = 1.0 / tbg
        Sbg = 2*self.gyro_params['bias_stability_std']**2*log(2) / (pi*0.4365**2*tbg)
        self.Qbg = sqrt(Sbg*(1- exp(-2*self.ug*self.min_sample_time)) / (2*self.ug))

        self.nd_std_g = self.gyro_params['noise_density']*sqrt(1.0 / self.min_sample_time)  # noise density

        if 'random_walk' in self.gyro_params:
            self.rw_std_g = self.gyro_params['random_walk']*sqrt(self.min_sample_time)
        else:
            self.rw_std_g = 0
        self.rw_g = np.zeros((3, 1))  # random walk

        self.big = np.zeros((3, 1))  # bias instability

    @classmethod
    def from_yaml(cls, imu_file: Path, seed) -> Self:

        imu_params = yaml.safe_load(open(imu_file))
        return cls(imu_params, seed)

    def _simulate(self, fs_b, gyro_b):
        if self.dt < self.min_sample_time:
            return None, None
        
        dt = max(self.dt, self.min_sample_time)
        
        # accelerometer
        self.bia = np.add(exp(-self.ua*dt)*self.bia, self.Qba*randn(3, 1))
        self.rw_a = np.add(self.rw_a, self.rw_std_a*randn(3, 1))
        self.ba = np.add(self.bas, np.add(self.bia, self.rw_a))
        wn_a = self.nd_std_a*randn(3, 1)

        acc_b_ms = self.ba + np.matmul(self.Ma, fs_b) + wn_a
        acc_b_ms = np.clip(acc_b_ms, -self.accel_params['range'], self.accel_params['range'])

        # gyroscope
        self.big = np.add(exp(-self.ug*dt)*self.big, self.Qbg*randn(3, 1))
        self.rw_g = np.add(self.rw_g, self.rw_std_g*randn(3, 1))
        self.bg = np.add(self.bgs, np.add(np.add(self.big, self.Gg.dot(fs_b)), self.rw_g))
        wn_g = self.nd_std_g*randn(3, 1)

        gyro_b_ms = self.bg + np.matmul(self.Mg, gyro_b) + np.matmul(self.Gg, fs_b) + wn_g
        gyro_b_ms = np.clip(gyro_b_ms, -self.gyro_params['range'], self.gyro_params['range'])

        self.dt = 0

        return acc_b_ms, gyro_b_ms
    
    @performance_decorator.time_execution_stats
    def evaluate(
            self,
            state: VehicleState,
            mpd: MassPropertiesData,
            total_loads: Loads,
            grav_acc,
            dt,
        ):
        self.dt += dt
        fs_b, gyro_b = self._inertial_data(state, total_loads, grav_acc, mpd)
        fs_b_ms, gyro_b_ms = self._simulate(fs_b, gyro_b)

        return fs_b_ms, gyro_b_ms
