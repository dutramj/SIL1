# BRYSON, Arthur Earl. Control of spacecraft and aircraft. Princeton: Princeton University Press, 1994.

# 3rd party libraries
import numpy as np
from numpy import copysign

from .loads import Loads

# Simulator library
from .parameters import g0
from . import performance_decorator
from .vehicle_state import VehicleState
from .mass_properties import MassPropertiesData


class RCS:
    def __init__(self, config_data: dict, seed=None):
        if seed is not None:
            np.random.seed(seed + 10)

        self.rcs_data = config_data['rcs_parameters']

        idx = next(
            (i for i, d in enumerate(self.rcs_data) if d.get("phase_id") == 1),
            None  # returned if no match is found
        )
        
        self._enabled = False

        self.rcs_F_max       = self.rcs_data[idx]['rcs_F_max']
        self.rcs_F_min       = self.rcs_data[idx]['rcs_F_min']
        self.rcs_long_arm    = self.rcs_data[idx]['rcs_long_arm']
        self.rcs_lat_arm     = self.rcs_data[idx]['rcs_lat_arm']
        self.rcs_Isp         = self.rcs_data[idx]['rcs_Isp']
        self.rcs_mass        = self.rcs_data[idx]['rcs_initial_mass']
        self.rcs_deadband    = self.rcs_data[idx]['rcs_deadband']
        self.rcs_tau         = self.rcs_data[idx]['rcs_tau']
        self.rcs_hysteresis  = self.rcs_data[idx]['rcs_hysteresis']

        self.rcs_F = self.rcs_F_max
        
        self.t_int   = 0.0
        self.dt_bc   = 0.0

        self.rcs_initial_mass = self.rcs_mass
        self.reset_mass_comp = False
        self.previous_phase = 1

        self.rcs_min_pulse = self.rcs_data[idx]['rcs_min_pulse']
        self.rcs_delay     = self.rcs_data[idx]['rcs_delay']

        self.dt_rcs_roll = self.rcs_min_pulse
        self.dt_rcs_pitch = self.rcs_min_pulse
        self.dt_rcs_yaw = self.rcs_min_pulse

        self.dt_delay_roll = self.rcs_delay
        self.dt_delay_pitch = self.rcs_delay
        self.dt_delay_yaw = self.rcs_delay

        self.rcs_roll_cmd = 0.0

        self.prev_error_roll = 0.0
        self.prev_error_pitch = 0.0
        self.prev_error_yaw = 0.0
        
        self.prev_roll_out = 0.0
        self.prev_pitch_out = 0.0
        self.prev_yaw_out = 0.0

        self.rcs_loads = Loads(0.0, 0.0, 0.0 , 0.0, 0.0 , 0.0)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def _update_propellant_mass(self):
        if self.t_int > 0.0:
            self.rcs_mass = (self.rcs_initial_mass*self.rcs_Isp*g0 / self.t_int - self.rcs_F_min) / (self.rcs_Isp*g0 / self.t_int + (self.rcs_F_max - self.rcs_F_min) / self.rcs_initial_mass)
            self.rcs_F = (self.rcs_F_max - self.rcs_F_min)*self.rcs_mass / self.rcs_initial_mass + self.rcs_F_min
            # self.rcs_F *= np.random.uniform(0.95, 1.05)  # add noise 
    
    def _schmitt_trigger(self, input_new, input):	
        output = 0.0
        trend = copysign(1, input_new - input)
        side = copysign(1, input)
        trigger = (self.rcs_deadband*side + self.rcs_hysteresis*trend) / 2.0

        if(input >= trigger and side == 1):
            output = 1.0
        elif(input <= trigger and side == -1):
            output = -1.0
        else:
            output = 0.0

        return output
    
    def _rcs_control(self, state: VehicleState, x_cg, p, q, r, pitch_cmd, yaw_cmd) -> Loads:

        error_roll = self.rcs_roll_cmd - (self.rcs_tau*p + state.roll_ned[0])
        error_pitch = pitch_cmd - (self.rcs_tau*q + state.pitch_ned[0])
        error_yaw = yaw_cmd - (self.rcs_tau*r + state.yaw_ned[0])

        roll_out_new = self._schmitt_trigger(error_roll, self.prev_error_roll)
        pitch_out_new = self._schmitt_trigger(error_pitch, self.prev_error_pitch)
        yaw_out_new = self._schmitt_trigger(error_yaw, self.prev_error_yaw)

        # add noise
        rcs_delay_noisy = self.rcs_delay  # *np.random.uniform(0.95, 1.05)
        rcs_min_pulse_noisy = self.rcs_min_pulse  # *np.random.uniform(0.95, 1.05)

        roll_out = self.prev_roll_out
        if roll_out_new != self.prev_roll_out:
            can_change = False
            
            if self.prev_roll_out == 0.0:
                if self.dt_delay_roll >= rcs_delay_noisy:
                    can_change = True
            else:
                if self.dt_rcs_roll >= rcs_min_pulse_noisy:
                    roll_out_new = 0.0
                    can_change = True
            
            if can_change:
                roll_out = roll_out_new
                self.dt_rcs_roll = 0.0
                if roll_out == 0.0:
                    self.dt_delay_roll = 0.0
        
        pitch_out = self.prev_pitch_out
        if pitch_out_new != self.prev_pitch_out:
            can_change = False
            
            if self.prev_pitch_out == 0.0:
                if self.dt_delay_pitch >= rcs_delay_noisy:
                    can_change = True
            else:
                if self.dt_rcs_pitch >= rcs_min_pulse_noisy:
                    pitch_out_new = 0.0
                    can_change = True
            
            if can_change:
                pitch_out = pitch_out_new
                self.dt_rcs_pitch = 0.0
                if pitch_out == 0.0:
                    self.dt_delay_pitch = 0.0
                    
        yaw_out = self.prev_yaw_out
        if yaw_out_new != self.prev_yaw_out:
            can_change = False
            
            if self.prev_yaw_out == 0.0:
                if self.dt_delay_yaw >= rcs_delay_noisy:
                    can_change = True
            else:
                if self.dt_rcs_yaw >= rcs_min_pulse_noisy:
                    yaw_out_new = 0.0
                    can_change = True
            
            if can_change:
                yaw_out = yaw_out_new
                self.dt_rcs_yaw = 0.0
                if yaw_out == 0.0:
                    self.dt_delay_yaw = 0.0

        self.prev_error_roll = error_roll
        self.prev_error_pitch = error_pitch
        self.prev_error_yaw = error_yaw
        self.prev_roll_out = roll_out
        self.prev_pitch_out = pitch_out
        self.prev_yaw_out = yaw_out

        L = roll_out*self.rcs_F*self.rcs_lat_arm # Considering a doublet
        M = pitch_out*self.rcs_F*(self.rcs_long_arm - x_cg)
        N = yaw_out*self.rcs_F*(self.rcs_long_arm - x_cg)

        loads = Loads( 0.0, 0.0, 0.0 ,L, M , N )
        self.t_int += abs(roll_out)*self.dt_bc + abs(pitch_out)*self.dt_bc + abs(yaw_out)*self.dt_bc
        self._update_propellant_mass()

        return loads
    
    @performance_decorator.time_execution_stats
    def step(self, state: VehicleState, mpd: MassPropertiesData, phase, dt, gyro_b, pitch_cmd, yaw_cmd):

        # Loading data from configuration file
        idx = next(
            (i for i, d in enumerate(self.rcs_data) if d.get("phase_id") == phase),
            None  # returned if no match is found
        )
        
        if self.previous_phase != phase:
            reset_flag = True
        else:
            reset_flag = False
        
        self.previous_phase = phase

        comp_init_mass = self.rcs_data[idx]['rcs_compute_initial_mass']
        if not comp_init_mass:
            self.rcs_initial_mass = self.rcs_data[idx]['rcs_initial_mass']

            if reset_flag:
                self.t_int = 0.0 # Reseting integration time in case we reset the initial mass when changing phases    
                self.rcs_mass = self.rcs_initial_mass

        min_altitude         = self.rcs_data[idx]['rcs_min_alt']

        self.rcs_F_max       = self.rcs_data[idx]['rcs_F_max']
        self.rcs_F_min       = self.rcs_data[idx]['rcs_F_min']
        self.rcs_long_arm    = self.rcs_data[idx]['rcs_long_arm']
        self.rcs_lat_arm     = self.rcs_data[idx]['rcs_lat_arm']
        self.rcs_Isp         = self.rcs_data[idx]['rcs_Isp']
        self.rcs_deadband    = self.rcs_data[idx]['rcs_deadband']
        self.rcs_tau         = self.rcs_data[idx]['rcs_tau']
        self.rcs_hysteresis  = self.rcs_data[idx]['rcs_hysteresis']

        if not self._enabled:
            self.rcs_loads = Loads(0.0, 0.0, 0.0 , 0.0, 0.0 , 0.0)
        
        else:
            if state.alt < min_altitude:
                self.rcs_loads = Loads(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            else:
                self.dt_rcs_roll += dt
                self.dt_rcs_pitch += dt
                self.dt_rcs_yaw += dt
                self.dt_bc += dt

                if self.prev_roll_out == 0.0:
                    self.dt_delay_roll += dt
                if self.prev_pitch_out == 0.0:
                    self.dt_delay_pitch += dt  
                if self.prev_yaw_out == 0.0:
                    self.dt_delay_yaw += dt
                
                if gyro_b is not None:
                    p = np.rad2deg(gyro_b[0, 0])
                    q = np.rad2deg(gyro_b[1, 0])
                    r = np.rad2deg(gyro_b[2, 0])

                    self.rcs_loads = self._rcs_control(state, mpd.x_cg, p, q, r, pitch_cmd, yaw_cmd)
                    self.dt_bc = 0.0

        return self.rcs_loads, self.rcs_mass
