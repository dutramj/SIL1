# ZIPFEL, Peter H. Modeling and simulation of aerospace vehicle dynamics. AIAA, 2007.

# 3rd party libraries
import numpy as np

# VAHSimulator library
from .aerodynamics.aero_state import AeroState
from .utils import integrate
from . import performance_decorator
from .vehicle_state import VehicleState
from .control_laws import compute_tvc_factor

class ControlTVC:
    def __init__(self, config_data:dict, dt):

        self._enabled = False

        self.wn_ac_tvc = 5.0
        self.preal_ac_tvc = 0.9
        self.zeta_ac_tvc = 0.9

        self.z_tvc = 0.0
        self.zd_tvc = 0.0
        self.y_tvc = 0.0
        self.yd_tvc = 0.0

        self.delta_q_tvc = 0.0
        self.delta_r_tvc = 0.0
        
        self.dt = dt

        # Blending factor
        self.tvc_factor = 0.0  # 0=TVC only, 1=canard only

        control_interlocks = config_data['control_interlocks'][0]

        self.transition_time = control_interlocks['transition_time']
        self.action_time = control_interlocks['action_time']

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def _control_accel(self, q, r, fy, fz, V, Q, Na, Ma, Mq, Md_tvc, an_cmd, al_cmd):
        """
        Controls the acceleration of the system by calculating pitch and yaw control commands.
        """
        # ===== LOW-Q PROTECTION =====
        Q_min = 10000.0
        Q_scale = 1.0
        
        if Q < Q_min and V < 100.0:
            Q_scale = 0.3 + 0.7*(Q / Q_min)
        
        # ===== TVC CHANNEL GAINS =====
        wn_ac_tvc = max(0.1, self.wn_ac_tvc + 0.1e-5*(Q - 10000.0))
        preal_ac_tvc = max(0.1, self.preal_ac_tvc + 1.0e-5*(Q - 10000.0)*1.5)
        zeta_ac_tvc = self.zeta_ac_tvc
        
        # Apply low-Q scaling
        wn_ac_tvc = wn_ac_tvc*Q_scale
        preal_ac_tvc = preal_ac_tvc*Q_scale

        if abs(Md_tvc) < 1e-6:
            Md_tvc = np.sign(Md_tvc)*1e-6

        GI_tvc = preal_ac_tvc*wn_ac_tvc**2 / (Na*Md_tvc)
        k2_tvc = (2*zeta_ac_tvc*wn_ac_tvc + preal_ac_tvc + Mq - Na / V) / Md_tvc
        k1_tvc = (wn_ac_tvc**2 + 2*zeta_ac_tvc*wn_ac_tvc*preal_ac_tvc + Ma + Mq*Na / V - k2_tvc*Md_tvc*Na / V) / (Na*Md_tvc)

        # PITCH - TVC Channel
        zd_tvc_ = an_cmd + fz
        self.z_tvc = integrate(zd_tvc_, self.zd_tvc, self.z_tvc, self.dt)
        self.zd_tvc = zd_tvc_
        
        self.delta_q_tvc = np.rad2deg(k1_tvc*fz - k2_tvc*q + GI_tvc*self.z_tvc)        
        self.delta_q_tvc = np.clip(self.delta_q_tvc*(1.0 - self.tvc_factor), -self.max_tvc_deflect, self.max_tvc_deflect)
        
        # ===== YAW CHANNELS =====
        # YAW - TVC Channel
        yd_tvc_ = al_cmd - fy
        self.y_tvc = integrate(yd_tvc_, self.yd_tvc, self.y_tvc, self.dt)
        self.yd_tvc = yd_tvc_

        self.delta_r_tvc = np.rad2deg(-k1_tvc*fy - k2_tvc*r + GI_tvc*self.y_tvc)
        self.delta_r_tvc = np.clip(self.delta_r_tvc*(1.0 - self.tvc_factor), -self.max_tvc_deflect, self.max_tvc_deflect)

    @performance_decorator.time_execution_stats
    def step(self, state: VehicleState, aero_state: AeroState, config_data: dict, phase, fb, wb, Na, Ma, Mq, Md_tvc, an_cmd, al_cmd):

        if not self._enabled:
            return tuple(np.zeros(2, dtype=np.float64))
        
        self.tvc_factor = compute_tvc_factor(state.time, self.transition_time, self.action_time)

        tvc_data = config_data['tvc_parameters']
        
        idx = next(
            (i for i, d in enumerate(tvc_data) if d.get("phase_id") == phase),
            None  # returned if no match is found
        )

        self.max_tvc_deflect = tvc_data[idx]['max_tvc_deflect']

        if wb is not None and fb is not None:
            q = wb[1, 0]
            r = wb[2, 0]
            fy = fb[1, 0]
            fz = fb[2, 0]

            if phase == 1:
                self._control_accel(q, r, fy, fz, aero_state.Va, aero_state.Q, Na, Ma, Mq, Md_tvc, an_cmd, al_cmd)
            else:
                self.delta_q_tvc, self.delta_r_tvc = 0.0, 0.0

        return self.delta_q_tvc, self.delta_r_tvc
