# ZIPFEL, Peter H. Modeling and simulation of aerospace vehicle dynamics. AIAA, 2007.

# 3rd party libraries
import numpy as np

# VAHSimulator library
from . import performance_decorator
from .aerodynamics.aero_state import AeroState
from .vehicle_state import VehicleState
from .control_laws import (
    control_heading,
    control_roll,
    control_gamma,
    control_rate,
    compute_tvc_factor,
)


class AerodynamicControl:
    def __init__(self, dt, wn_rc=2.0, zeta_rc=0.9):

        self._enabled = False
        
        self.wn_rc = wn_rc
        self.zeta_rc = zeta_rc

        # canard parameters
        self.delta_p_aero_control = 0.0
        self.delta_q_aero_control = 0.0
        self.delta_r_aero_control = 0.0

        self.dt = dt

        # Blending factor
        self.tvc_factor = 0.0  # 0=TVC only, 1=canard only

        self.heading_factor = -0.95  # Factor to reduce heading gain
        
        self.wgam = 5.0   # Natural frequency - rad/s
        self.zgam = 0.7   # Damping - ND
        self.pgam = 0.7   # Real pole location - rad/s
        
        self.heading_cmd = np.deg2rad(0.0)  # heading command, rad

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @performance_decorator.time_execution_stats
    def step(self, state: VehicleState, aero_state: AeroState, config_data: dict, phase, fb, wb, grav_acc, Na, Nd, Ma, Mq, Md_canard, LLda, LLp, gamma_cmd, pitch_cmd, yaw_cmd):

        if not self._enabled:
            return tuple(np.zeros(3, dtype=np.float64))

        transition_time = config_data['control_interlocks'][0]['transition_time']
        action_time = config_data['control_interlocks'][0]['action_time']
        
        control_surface_data = config_data['aero_controls_parameters']
        
        # Finding phase index
        idx = next(
            (i for i, d in enumerate(control_surface_data) if d.get("phase_id") == phase),
            None  # returned if no match is found
        )

        self.max_cs_deflect = control_surface_data[idx]['max_deflection']
        self.max_cs_rate    = control_surface_data[idx]['max_rate']
        self.wn_rc          = control_surface_data[idx]['wn_rc']
        self.zeta_rc        = control_surface_data[idx]['zeta_rc']
        
        # Control Flags
        control_roll_flag    = control_surface_data[idx]['control_roll']
        control_heading_flag = control_surface_data[idx]['control_heading']
        control_gamma_flag   = control_surface_data[idx]['control_gamma']
        control_rate_flag    = control_surface_data[idx]['control_rate']
        tvc_factor_flag      = control_surface_data[idx]['tvc_factor']
        use_ref_heading_flag = control_surface_data[idx]['use_ref_heading']

        # Comptuing TVC factor
        if tvc_factor_flag:
            self.tvc_factor = compute_tvc_factor(state.time, transition_time, action_time)
        else:
            self.tvc_factor = 1.0
        
        # Defining heading and roll control parameters:
        self.heading_cmd = np.deg2rad(yaw_cmd) if use_ref_heading_flag else np.float64(0.0)
        

        if wb is not None and fb is not None:
            p = wb[0, 0]
            q = wb[1, 0]
            r = wb[2, 0]

            # Roll control command
            roll_cmd = control_heading(state, grav_acc, aero_state.Va, self.zeta_rc, self.wn_rc, self.heading_cmd, self.heading_factor) if control_heading_flag else np.float64(0.0)
        
            # Roll control
            self.delta_p_aero_control = control_roll(state, aero_state, roll_cmd, self.max_cs_deflect) if control_roll_flag else np.float64(0.0)

            if control_rate_flag:
                q_cmd, self.delta_r_aero_control = control_rate(state, aero_state, pitch_cmd, yaw_cmd, self.max_cs_deflect)                
            
            else:
                q_cmd, self.delta_r_aero_control = 0.0, 0.0
            
            self.delta_q_aero_control = control_gamma(gamma_cmd, state, q, aero_state.Va, aero_state.Q, Na, Nd, Ma, Mq, Md_canard, self.max_cs_deflect) if control_gamma_flag else np.float64(q_cmd)
            
        else:
            self.delta_p_aero_control, self.delta_q_aero_control, self.delta_r_aero_control = 0.0, 0.0, 0.0

        
        return self.delta_p_aero_control, self.delta_q_aero_control, self.delta_r_aero_control
