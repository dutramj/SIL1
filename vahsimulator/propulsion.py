# DA SILVA, Adolfo Graciano. Análise e projeto do sistema de controle de atitude do veículo lançador de satélite (VLS). 2014.

# 3rd party libraries
import numpy as np
import pandas as pd
from numpy import cos, sin

# VAHSimulator library
from .utils import skew_matrix, eci_to_body_quaternion
from . import performance_decorator
from .vehicle_state import VehicleState
from .mass_properties import MassPropertiesData
from .atmosphere import AtmosphereData
from .interpolate import LinearInterpolator1D
from .loads import Loads
from .parameters import earth_rate__rad_s

class Propulsion:
    def __init__(self, phase, config_data: dict, thrust_applic_point_unc=0.0, thrust_vec_y_pos_unc=0.0, thrust_vec_z_pos_unc=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed + 9)
        
        self.phase = phase  # 1- complete rocket, 2- intermediate stage with fairing, 3- intermediate stage without fairing, 4- 

        propulsion_data = config_data['propulsion_properties']
        geometric_data = config_data['geometric_data']
        self.propulsion_data = self._get_propulsion_data(propulsion_data, geometric_data)

        # Finding index for first flight phase
        idx = self.propulsion_data['phase'].index(1)

        # Initializing thrust
        self.Th_k = max(0.0, self.propulsion_data['Thrust_Curve'][idx].evaluate([0.])[0, 0])

        self.thrust_applic_point_unc = thrust_applic_point_unc
        self.thrust_vec_y_pos_unc = thrust_vec_y_pos_unc
        self.thrust_vec_z_pos_unc = thrust_vec_z_pos_unc

    
    @performance_decorator.time_execution_stats
    def _initialize_propulsion_interpolator(self, data):
        if data['type'] == 'curve':
            file_name = data['file_name']
            df = pd.read_csv(file_name)

        else:
            initial_data = data['initial_values']
            final_data = data['final_values']

            thrust_data = {
                "Time_s": [initial_data['time'], final_data['time']],
                "Thrust_N": [initial_data['thrust'], final_data['thrust']],

            }

            df = pd.DataFrame(thrust_data)

        return LinearInterpolator1D.from_df(df,'Time_s')


    def _get_propulsion_data(self, propulsion_data, geometric_data):
        n_phases = len(propulsion_data)

        propulsion_data_per_phase = {
            'phase': [],
            'Nozzle_Exhaust_Area': [],
            'Thrust_Application_Point': [],
            'Thrust_Curve': [],
        }

        for phase in range(n_phases):
            data = propulsion_data[phase]
            data_geom = geometric_data[phase]
            phase_id = data['phase_id']

            propulsion_data_per_phase['phase'].append(phase_id)
            propulsion_data_per_phase['Nozzle_Exhaust_Area'].append(data_geom['other_parameters']['nozzle_exhaust_area'])
            propulsion_data_per_phase['Thrust_Application_Point'].append(data_geom['other_parameters']['x_thrust'])
  

            # Getting propellant data
            propulsion_data_per_phase['Thrust_Curve'].append(
                self._initialize_propulsion_interpolator(data['propulsion_curve']))

        return propulsion_data_per_phase
        

    def _dynamic_properties(self, time):
        """
        Thrust at time t.
        """
        idx = self.propulsion_data['phase'].index(self.phase)

        self.Th_k = max(0.0, self.propulsion_data['Thrust_Curve'][idx].evaluate([time])[0, 0])

    
    def _exhaust_velocity(self, mdot_kgs_1):
        if mdot_kgs_1 != 0 and self.Th_k > 0.0:
            v_e = -self.Th_k / mdot_kgs_1  # exhaust velocity, m/s
        else:
            v_e = 0.0
        return v_e


    def _propulsive_forces_moments(self, state: VehicleState, pressure, By, Bz, x_cg_m, mdot_kgs_1) -> Loads:
        """
        pressure: local pressure, Pa
        By: actuator displacement in yaw
        Bz: actuator displacement in pitch
        """

        # Getting propulsion properties
        idx      = self.propulsion_data['phase'].index(self.phase)
        Ae       = self.propulsion_data['Nozzle_Exhaust_Area'][idx]
        x_thrust = self.propulsion_data['Thrust_Application_Point'][idx]

        # Applying uncertainty
        thrust_x_pos = x_thrust +  self.thrust_applic_point_unc

        # Thrust correction due to atmospheric pressure
        Th = max(0.0, self.Th_k - pressure * Ae)  # (2.28)
                
        # propulsive forces
        Fex = Th*cos(By)*cos(Bz)
        Fey = Th*sin(By)*cos(Bz)
        Fez = -Th*sin(Bz)

        # propulsive moments
            
        lc = np.array([[x_cg_m - thrust_x_pos], [self.thrust_vec_y_pos_unc], [self.thrust_vec_z_pos_unc]]) 
        lc_sm = skew_matrix(lc)
        Me = np.matmul(lc_sm, np.array([[Fex], [Fey], [Fez]]))

        re = lc  # distance between the CG and the gas outlet point, which is approximately lc
        re_sm = lc_sm

        wbl = np.array(
            [
                [state.p_r],
                [state.q_r],
                [state.r_r]
            ]
        )

        wbl_sm = skew_matrix(wbl)

        if Th > 1e-12:
            Maj = mdot_kgs_1*np.matmul(re_sm, np.matmul(wbl_sm, re))  # jet damping moment (2.37)
        else:
            Maj = np.zeros((3, 1))

        L = Me[0, 0] + Maj[0, 0]
        M = Me[1, 0] + Maj[1, 0]
        N = Me[2, 0] + Maj[2, 0]

        loads = Loads(Fex, Fey, Fez, L, M, N)

        return loads
    
    @performance_decorator.time_execution_stats
    def evaluate(self, state: VehicleState, mpd: MassPropertiesData, atmosphere: AtmosphereData, phase, By, Bz) -> tuple[Loads, float]:
        self.phase = phase
        self._dynamic_properties(state.interp_time)
        prop_forces_moments = self._propulsive_forces_moments(state, atmosphere.pressure_Pa, By, Bz, mpd.x_cg, mpd.mass_dot)
        v_e = self._exhaust_velocity(mpd.mass_dot)
        return prop_forces_moments, v_e
