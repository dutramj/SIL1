# 3rd party libraries
import numpy as np
from numpy import copysign
import pandas as pd

# VAHSimulator library
from . import AerodynamicCoreInterpolator
from . import ControlSurfacesSetInterpolator
from .. import performance_decorator

from ..vehicle_state import VehicleState
from ..mass_properties import MassPropertiesData
from ..atmosphere import AtmosphereData
from ..loads import Loads
from .aero_state import AeroState
from ..parameters import earth_rate__rad_s
from ..utils import eci_to_body_quaternion


class Aerodynamics:

    def __init__(
            self,
            phase,
            config_data:dict,
            drag_unc=0.0,
            seed=None
        ) -> None:

        if seed is not None:
            np.random.seed(seed + 1)
        
        self.drag_unc = np.random.normal(0.0, drag_unc / 100.0)

        # Aero loads
        self.Na = 1e-7  # normal force slope derivative - m/s²
        self.Nd = 1e-7  # normal force derivative with pitch control - m/s²
        self.Ma = 1e-7  # pitch moment derivative - 1/s²
        self.Mq = 1e-7  # pitch damping derivative - 1/s²
        self.Md_tvc = 1e-7  # pitch control derivative - 1/s²
        self.Md_canard = 1e-7  # pitch control derivative - 1/s²
        self.LLda = 1e-7  # roll control derivative - 1/s²
        self.LLp = 1e-7  # roll damping derivative - 1/s

        # aerodynamics coefficients
        self.CA = 0.0  # axial force coefficient

        self.CY = 0.0  # lateral force coefficient
        self.CYB = 0.0  # derivative of the lateral force coefficient with respect to the sideslip angle (beta)
        self.CYP = 0.0  # derivative of the lateral force coefficient with respect to the roll rate (p)
        self.CYR = 0.0  # derivative of the lateral force coefficient with respect to the yaw rate (r)
        self.CYDR = 0.0  # derivative of the lateral force coefficient with respect to the equivalent deflection in yaw

        self.CN = 0.0  # normal force coefficient
        self.CNA = 0.0  # derivative of the normal force coefficient with respect to the angle of attack (alpha)
        self.CNQ = 0.0  # derivative of the normal force coefficient with respect to the pitch rate (q)
        self.CNAD = 0.0  # derivative of the normal force coefficient with respect to the rate of change of the angle of attack
        self.CNDQ = 0.0  # derivative of the normal force coefficient with respect to the equivalent deflection in pitch

        self.CLL = 0.0  # rolling moment coefficient
        self.CLLB = 0.0  # derivative of the rolling moment coefficient with respect to the sideslip angle (beta)
        self.CLLP = 0.0  # derivative of the rolling moment coefficient with respect to the roll rate (p)
        self.CLLR = 0.0  # derivative of the rolling moment coefficient with respect to the yaw rate (r)
        self.CLLDP = 0.0  # derivative of the rolling moment coefficient with respect to the equivalent deflection in roll

        self.CM = 0.0  # pitching moment coefficient
        self.CMA = 0.0  # derivative of the pitching moment coefficient with respect to the angle of attack (alpha)
        self.CMQ = 0.0  # derivative of the pitching moment coefficient with respect to the pitch rate (q)
        self.CMAD = 0.0  # derivative of the pitching moment coefficient with respect to the rate of change of the angle of attack
        self.CMDQ = 0.0  # derivative of the pitching moment coefficient with respect to the equivalent deflection in pitch
        
        self.CLN = 0.0  # yawing moment coefficient
        self.CLNB = 0.0  # derivative of the yawing moment coefficient with respect to the sideslip angle (beta)
        self.CLNR = 0.0  # derivative of the yawing moment coefficient with respect to the yaw rate (r)
        self.CLNP = 0.0  # derivative of the yawing moment coefficient with respect to the roll rate (p)
        self.CLNDR = 0.0  # derivative of the yawing moment coefficient with respect to the equivalent deflection in yaw

        self.DCA  = 0.0  # axial force coefficient increment due to fins/canards
        self.DCY  = 0.0  # lateral force coefficient increment due to fins/canards
        self.DCN  = 0.0  # normal force coefficient increment due to fins/canards
        self.DCLL = 0.0  # rolling moment coefficient increment due to fins/canards
        self.DCM  = 0.0  # pitching moment coefficient increment due to fins/canards
        self.DCLN = 0.0  # yaw moment coefficient increment due to fins/canards
        
        self.geometry = config_data['geometric_data']
        self.aerodecks = config_data['aerodecks']
        self.interpolators = self._initialize_interpolators()
        self.phase = phase

    def _initialize_interpolators(self) -> dict:
        n_phases = len(self.aerodecks)

        interpolators = dict()

        for phase in range(n_phases):
            df_total = pd.read_csv(self.aerodecks[phase]["core_bank"])
            df_control = pd.read_csv(self.aerodecks[phase]["control_delta_bank"])

            interpolators[self.aerodecks[phase]["phase_id"]] = (
                AerodynamicCoreInterpolator(df_total),
                ControlSurfacesSetInterpolator(
                    ['fin_1', 'fin_2', 'fin_3', 'fin_4'],
                    df_control,
                ),
            )

        return interpolators

    @performance_decorator.time_execution_stats
    def _get_current_model(self) -> tuple[AerodynamicCoreInterpolator, ControlSurfacesSetInterpolator, float, float, float, float]:
        # Geometry
        
        idx = next(
            (i for i, d in enumerate(self.geometry) if d.get("phase_id") == self.phase),
            None  # returned if no match is found
        )
        geometry_phase         = self.geometry[idx]
        ref_area               = geometry_phase['main_parameters']['S_ref']
        ref_long               = geometry_phase['main_parameters']['l_ref_1']
        ref_lat                = geometry_phase['main_parameters']['l_ref_2']
        ref_thrust_application = geometry_phase['other_parameters']['x_thrust']
        
        # Interpolators
        interpolator, interpolator_control = self.interpolators[self.phase]
        
        return interpolator, interpolator_control, ref_area, ref_long, ref_lat, ref_thrust_application

    @performance_decorator.time_execution_stats
    def _aerodynamics_forces_moments(
            self,
            state: VehicleState,
            aero_state: AeroState,
            delta_1,
            delta_2,
            delta_3,
            delta_4,
            x_cg,
            w_w
        ) -> tuple[Loads,float]:
        
        interpolators, interpolators_control, A_ref, l_ref_1, l_ref_2, _ = self._get_current_model()

        p = state.p_r + w_w[0, 0]
        q = state.q_r + w_w[1, 0]
        r = state.r_r + w_w[2, 0]

        beta_rad = np.deg2rad(aero_state.beta)
        alpha_dot_rad = np.deg2rad(aero_state.alpha_dot)

        pl_2V = (p*l_ref_2) / (2*aero_state.Va) if aero_state.Va > 1e-7 else 0.0
        ql_2V = (q*l_ref_1) / (2*aero_state.Va) if aero_state.Va > 1e-7 else 0.0
        rl_2V = (r*l_ref_2) / (2*aero_state.Va) if aero_state.Va > 1e-7 else 0.0
        al_2V = (alpha_dot_rad*l_ref_1) / (2*aero_state.Va) if aero_state.Va > 1e-7 else 0.0

        coefficients = interpolators.evaluate(
            points=[
                [aero_state.alpha, aero_state.beta, aero_state.mach, state.alt],
            ],
        )
        coefficients = {key: value[0] for key, value in coefficients.items()}
        
        # Computing control contributions
        coefficients_control_df = interpolators_control.evaluate(
            points=np.asarray([[aero_state.alpha, aero_state.beta, aero_state.mach]]),
            fin_commands={
                'fin_1': delta_1,
                'fin_2': delta_2,
                'fin_3': delta_3,
                'fin_4': delta_4,
            },
        )
        control_contributions = {key: value[0] for key, value in coefficients_control_df['total'].items()}

        x_ref = coefficients['XCG']

        # Computing final coefficients
        # Axial force coefficient
        self.CA_stat = coefficients['CA']
        self.DCA     = control_contributions['DCA']
        self.CA      = self.CA_stat + self.DCA
        self.CA     *= (1.0 + self.drag_unc)

        # Lateral force coefficient
        self.CYB = coefficients['CYB']
        self.CYP = coefficients['CYP']
        self.CYR = coefficients['CYR']
        self.DCY = control_contributions['DCY']
        self.CY  = self.CYB*beta_rad + self.CYP*pl_2V + self.CYR*rl_2V + self.DCY

        # Normal force coefficient
        self.CN_stat = coefficients['CN']
        self.CNA     = coefficients['CNA']
        self.CNQ     = coefficients['CNQ']
        self.CNAD    = coefficients['CNAD']
        self.DCN     = control_contributions['DCN']
        self.CN      = self.CN_stat + self.CNQ*ql_2V + self.CNAD*al_2V + self.DCN

        # Rolling moment coefficient
        self.CLLB = coefficients['CLLB']
        self.CLLP = coefficients['CLLP']
        self.CLLR = coefficients['CLLR']
        self.DCLL = control_contributions['DCLL']
        self.CLL  = self.CLLB*beta_rad + self.CLLP*pl_2V + self.CLLR*rl_2V + self.DCLL

        # Pitching moment coefficient
        self.CM_stat  = coefficients['CM']
        self.CMA      = coefficients['CMA']
        self.CMQ      = coefficients['CMQ']
        self.CMAD     = coefficients['CMAD']
        self.DCM      = control_contributions['DCM']
        ## Correction due to CG position
        self.CM_stat += self.CN_stat * (x_cg - x_ref) / l_ref_1
        self.CMA     += self.CNA * (x_cg - x_ref) / l_ref_1
        self.CMQ     += self.CNQ * (x_cg - x_ref) / l_ref_1
        self.CMAD    += self.CNAD * (x_cg - x_ref) / l_ref_1
        self.DCM     += self.DCN * (x_cg - x_ref) / l_ref_1
        
        self.CM       = self.CM_stat + self.CMQ*ql_2V + self.CMAD*al_2V + self.DCM

        # yawing moment coefficient
        self.CLNB  = coefficients['CLNB']
        self.CLNR  = coefficients['CLNR']
        self.CLNP  = coefficients['CLNP']
        self.DCLN  = control_contributions['DCLN']
        ## Correction due to CG position
        self.CLNB += self.CYB * (x_cg - x_ref) / l_ref_1
        self.CLNP += self.CYP * (x_cg - x_ref) / l_ref_1
        self.CLNR += self.CYR * (x_cg - x_ref) / l_ref_1
        self.DCLN += self.DCY * (x_cg - x_ref) / l_ref_1

        self.CLN   = self.CLNB*beta_rad + self.CLNP*pl_2V + self.CLNR*rl_2V + self.DCLN          

        # Computing Static Margin
        static_margin = - self.CMA / self.CNA

        QS = aero_state.Q*A_ref

        Fb = np.array([[-self.CA*QS], [self.CY*QS], [-self.CN*QS]])
        Fx = Fb[0, 0]
        Fy = Fb[1, 0]
        Fz = Fb[2, 0]

        Mb = np.array([[self.CLL*QS*l_ref_2], [self.CM*QS*l_ref_1], [self.CLN*QS*l_ref_2]])
        L = Mb[0, 0]
        M = Mb[1, 0]
        N = Mb[2, 0]

        loads = Loads(Fx, Fy, Fz, L, M, N)

        return loads, static_margin
    
    # To be removed once control needs for aerodynamic derivatives are deleted
    @performance_decorator.time_execution_stats
    def _aerodynamic_derivatives_coefficients(
            self, 
            aero_state: AeroState
        ) -> None:

        if self.phase == 1:
            self.CYDR  =  6.755e-03
            self.CNDQ  = -6.755e-03
            self.CMDQ  = -1.554e-02
            self.CLLDP =  4.827e-03

        elif self.phase == 2:
            self.CYDR  =  5.655e-03
            self.CNDQ  = -5.655e-03
            self.CMDQ  =  9.091e-03
            self.CLLDP =  4.755e-03

        elif self.phase == 3:
            self.CYDR  =  3.755e-03
            self.CNDQ  = -3.755e-03
            self.CMDQ  =  1.373e-03
            self.CLLDP =  8.273e-04
        
        else:
            mach_values = [
                4.0, 4.5, 5.0, 5.5, 
                6.0, 6.5, 7.0, 7.5, 
                8.0, 8.5, 9.0, 9.5, 
                10., 10.5
            ]

            CNDQ_values = [
                0.499188, 0.459004, 0.426246, 0.403500, 
                0.387148, 0.373962, 0.363156, 0.353664,
                0.346192, 0.339832, 0.334390, 0.329692, 
                0.325604, 0.322026
            ]
            
            CMDQ_values = [
                -0.446476, -0.409584, -0.379198, -0.358354,
                -0.343602, -0.331486, -0.321548, -0.312820,
                -0.305940, -0.300084, -0.295068, -0.290732, 
                -0.286956, -0.283654
            ]
            
            CLLDP_values = [
                0.219280, 0.201668, 0.187490, 0.177640,
                0.170498, 0.164736, 0.160012, 0.155864, 
                0.152572, 0.149796, 0.147432, 0.145380,
                0.143594, 0.142034
            ]
            
            self.CYDR  = 0.0
            self.CNDQ  = 2.0 * np.interp(aero_state.mach, mach_values, CNDQ_values)
            self.CMDQ  = 2.0 * np.interp(aero_state.mach, mach_values, CMDQ_values)
            self.CLLDP = 2.0 * np.interp(aero_state.mach, mach_values, CLLDP_values)

    @performance_decorator.time_execution_stats
    def _aerodynamics_derivatives(
            self,
            aero_state: AeroState,
            mass_kg,
            Ixx_kgm2,
            Iyy_kgm2,
            prop_loads: Loads,
            x_cg_m
        ) -> None:

        if np.abs(aero_state.alpha_total) < 20.0:
            _, _, A_ref, l_ref_1, l_ref_2, x_thrust = self._get_current_model()

            QS = aero_state.Q*A_ref
            QS_Va = aero_state.Q_Va*A_ref

            ## Normal force derivative
            # CN Alpha
            self.Na = (QS / mass_kg) * self.CNA
            if abs(self.Na) < 1e-7:
                self.Na = copysign(1, self.Na)*1e-7
            
            # CN control
            self.Nd = (QS / mass_kg) * np.rad2deg(self.CNDQ)
            if abs(self.Nd) < 1e-7:
                self.Nd = copysign(1, self.Nd)*1e-7

            ## Pitch moment derivatives
            # CM Alpha
            self.Ma = (QS * l_ref_1 / Iyy_kgm2) * self.CMA

            # CM Pitch rate
            self.Mq = (QS_Va * l_ref_1**2 / (2 * Iyy_kgm2)) * self.CMQ          
            
            if self.phase != 4:
                self.Md_tvc = (x_cg_m - x_thrust) * prop_loads.force_norm / Iyy_kgm2
                if abs(self.Md_tvc) < 1e-7:
                    self.Md_tvc = copysign(1, self.Md_tvc)*1e-7

                self.Md_canard = QS * l_ref_1 * np.rad2deg(self.CMDQ) / Iyy_kgm2
                if abs(self.Md_canard) < 1e-7:
                    self.Md_canard = copysign(1, self.Md_canard)*1e-7
                
            else:
                self.Md_tvc = 1e-7

                self.Md_canard = (QS * l_ref_1 / Iyy_kgm2) * self.CMDQ
                if abs(self.Md_canard) < 1e-7:
                    self.Md_canard = copysign(1, self.Md_canard)*1e-7
            
            # dimensional roll control derivative - 1/s²
            self.LLda = (QS * l_ref_2 / Ixx_kgm2) * np.rad2deg(self.CLLDP)
            if abs(self.LLda) < 1e-7:
                self.LLda = copysign(1, self.LLda)*1e-7

            # dimensional roll damping derivative - 1/s
            self.LLp = (QS_Va * l_ref_2 / Ixx_kgm2) * (l_ref_2 / 2) * self.CLLP
    
    @performance_decorator.time_execution_stats
    def evaluate(
            self,
            state: VehicleState,
            aero_state: AeroState,
            mpd: MassPropertiesData,
            atmosphere: AtmosphereData,
            phase,
            prop_loads: Loads,
            wb,
            dt,
            delta_1,
            delta_2,
            delta_3,
            delta_4,
        ) -> None:

        self.phase = phase
        new_aero_state = AeroState.from_data(aero_state, state, atmosphere, wb, dt)
        aero_loads = Loads(
            np.float64 (0.0),
            np.float64 (0.0),
            np.float64 (0.0),
            np.float64 (0.0),
            np.float64 (0.0),
            np.float64 (0.0)
        )

        aero_loads, static_margin = self._aerodynamics_forces_moments(
            state, aero_state,
            delta_1, delta_2, delta_3, delta_4,
            mpd.x_cg, wb[3:6]
        )
        self._aerodynamics_derivatives(
            aero_state,
            mpd.mass, mpd.Ixx, mpd.Iyy, prop_loads, mpd.x_cg
        )

        return (
            aero_loads, new_aero_state, static_margin, 
            self.Na, self.Nd, self.Ma, self.Mq, self.Md_tvc, self.Md_canard, self.LLda, self.LLp, 
            self.CA, self.CY, self.CN, self.CLL, self.CM, self.CLN, self.DCA, self.DCY, self.DCN, self.DCLL, self.DCM, self.DCLN
        )
