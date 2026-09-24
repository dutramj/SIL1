# Python standard libraries

# 3rd party libraries
import numpy as np

# VAHSimulator library
from .tvc import TVCBase
from vahsimulator.utils import integrate
from .. import performance_decorator


class TVCSimplified(TVCBase):
    def __init__(self, dt, seed=None):
        if seed is not None:
            np.random.seed(seed + 11)

        self.dt = dt
        self.tvc_mode = 0  # 0: no TVC, 1: 2nd-order TVC
        self.wn = 100.0  # natural frequency, rad/s
        self.zeta = 0.7  # damping ratio
        self.G = 1.0  # gain for the TVC system

        self.delta_q = 0.0  # pitch nozzle deflection, rad
        self.delta_r = 0.0  # yaw nozzle deflection, rad

        self.delta_q_dot = 0.0  # pitch nozzle deflection rate, rad/s
        self.delta_r_dot = 0.0  # yaw nozzle deflection rate, rad/s

        self.delta_q_dot_dot = 0.0  # pitch nozzle deflection angular acceleration, rad/s²
        self.delta_r_dot_dot = 0.0  # yaw nozzle deflection angular acceleration, rad/s²
    
    def _compute_dynamics(self, delta_cmd, delta, delta_dot, delta_dot_dot):
        if abs(delta) >= np.deg2rad(self.max_tvc_deflect):
            delta = np.clip(delta, -np.deg2rad(self.max_tvc_deflect), np.deg2rad(self.max_tvc_deflect))
            if delta*delta_dot > 0:
                delta_dot = 0.0
            
        iflag = False
        if abs(delta_dot) >= np.deg2rad(self.max_tvc_rate):
            iflag = True
            delta_dot = np.clip(delta_dot, -np.deg2rad(self.max_tvc_rate), np.deg2rad(self.max_tvc_rate))
        
        delta_dot_ = delta_dot
        delta = integrate(delta_dot_, delta_dot, delta, self.dt)
        delta = np.clip(delta, -np.deg2rad(self.max_tvc_deflect), np.deg2rad(self.max_tvc_deflect))

        delta_dot_dot_ = self.wn**2*(delta_cmd - delta) - 2*self.zeta*self.wn*delta_dot
        delta_dot = integrate(delta_dot_dot_, delta_dot_dot, delta_dot, self.dt)

        if iflag and delta_dot*delta_dot_dot > 0:
            delta_dot_dot = 0.0
        
        return delta, delta_dot, delta_dot_dot

    def _2nd_order_dynamics(self, delta_q_cmd, delta_r_cmd):
        delta_q_cmd = np.deg2rad(self.G*delta_q_cmd)
        delta_r_cmd = np.deg2rad(self.G*delta_r_cmd)
        
        # pitch dynamics
        self.delta_q, self.delta_q_dot, self.delta_q_dot_dot = self._compute_dynamics(delta_q_cmd, self.delta_q, self.delta_q_dot, self.delta_q_dot_dot)
        # yaw dynamics
        self.delta_r, self.delta_r_dot, self.delta_r_dot_dot = self._compute_dynamics(delta_r_cmd, self.delta_r, self.delta_r_dot, self.delta_r_dot_dot)

    def set_dt(self, dt):
        self.dt = dt

    @performance_decorator.time_execution_stats
    def step(self, config_data: dict, delta_q_cmd, delta_r_cmd, phase):
        tvc_data = config_data['tvc_parameters']
        
        idx = next(
            (i for i, d in enumerate(tvc_data) if d.get("phase_id") == phase),
            None  # returned if no match is found
        )

        self.max_tvc_deflect = tvc_data[idx]['max_tvc_deflect']
        self.max_tvc_rate    = tvc_data[idx]['max_tvc_rate']

        self._2nd_order_dynamics(delta_q_cmd, delta_r_cmd)
        return self.delta_q, self.delta_r
