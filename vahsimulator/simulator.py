# 3rd party libraries
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import ClassVar, Any
from dataclasses import replace

# VAHSimulator library
from .aerodynamics.aerodynamics import Aerodynamics
from .aerodynamics.aero_state import AeroState
from .atmosphere import AtmosphereBase, atmosphere_factory
from .control_surfaces_set import ControlSurfacesSet
from .control_aerodynamics import AerodynamicControl
from .control_tvc import ControlTVC
from .dynamics import (
    DynamicsBase,
)
from .end_condition import DurationEndCondition
from .gravity import Gravity
from .guidance import Guidance
from .imu import (
    IMUModel,
    IdealIMU,
)
from .mass_properties import MassProperties
from .mission_plan import MissionPlan
from .propulsion import Propulsion
from .rcs import RCS
from .recorder import Recorder
from .tvc.tvc import tvc_factory
from .tvc.tvc_base import TVCBase
from .vehicle_state import VehicleState
from .wind import (
    CompositeWind,
    WindConfig,
    WindModel,
)
from . import performance_decorator
from .loads import Loads
from .launching_reference import LaunchReference

class Simulator(BaseModel):

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def resolve_atmosphere_model(cls, values: Any) -> Any:
        """Resolve the atmospheric model configuration.

        Parameters
        ----------
        values : Any
            Raw simulator configuration provided to Pydantic.

        Returns
        -------
        Any
            Simulator configuration with the atmospheric model
            instantiated.
        """

        if not isinstance(values, dict):
            return values

        atmosphere_model = values.get("atmosphere_model")

        if isinstance(atmosphere_model, dict):
            values = values.copy()
            values["atmosphere_model"] = atmosphere_factory(
                atmosphere_model
            )

        return values
    

    # Class variables
    _instance_counter: ClassVar[int] = 0

    # Required inputs
    atmosphere_model: AtmosphereBase
    perturbations: dict
    sample_period: np.float64
    config_parameters: dict

    # Wind Model
    wind: WindConfig = Field(
        default_factory=WindConfig,
    )

    wind_model: WindModel = Field(
        default_factory=CompositeWind,
    )

    # Optional inputs
    seed: int | None = None

    # Runtime state
    phase_id: int = 1
    previous_phase_id: int = 1
    reset_interp_time: bool = False

    # Modules (injected)
    dynamics: DynamicsBase | None = None
    guidance: Guidance | None = None
    control_aerodynamics: AerodynamicControl | None = None
    control_tvc: ControlTVC | None = None
    imu: IMUModel = Field(
        default_factory=lambda: IdealIMU()
    )

    # Private runtime attributes
    _instance_id: int = PrivateAttr()
    _state: VehicleState | None = PrivateAttr(default_factory=None)
    _aero_state: AeroState = PrivateAttr()

    _aerodynamics: Aerodynamics = PrivateAttr()
    _recorder: Recorder = PrivateAttr()
    _gravity: Gravity = PrivateAttr()
    _tvc: TVCBase = PrivateAttr()
    _rcs: RCS = PrivateAttr()
    _control_surfaces_set: ControlSurfacesSet = PrivateAttr()
    _mass_properties: MassProperties = PrivateAttr()
    _propulsion: Propulsion = PrivateAttr()

    # Runtime vectors
    By: float = 0.0
    Bz: float = 0.0
    pn_dsrd: np.ndarray = Field(default_factory=lambda: np.zeros((3, 1)))

    rcs_loads: Loads = Field(
        default_factory=lambda: Loads(
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
        )
    )

    delta_1: float = 0.0
    delta_2: float = 0.0
    delta_3: float = 0.0
    delta_4: float = 0.0

    error_delta_1: float | None = None
    error_delta_2: float | None = None
    error_delta_3: float | None = None
    error_delta_4: float | None = None

    error_Bz: float | None = None
    error_By: float | None = None

    gimbal_lock_protection: bool | None = None

    def model_post_init(self, __context: Any) -> None:

        # instance id
        self._instance_id = Simulator._instance_counter
        Simulator._instance_counter += 1

        # RNG seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # perturbations
        (
            self.error_delta_1,
            self.error_delta_2,
            self.error_delta_3,
            self.error_delta_4,
        ) = np.random.normal(
            0.0, self.perturbations["control_surface_misalignment"], size=4
        )

        self.error_Bz, self.error_By = np.random.normal(
            0.0,
            np.deg2rad(self.perturbations["thrust_misalignment"]),
            size=2,
        )

        # module construction
        self._aerodynamics = Aerodynamics(
            phase=self.phase_id,
            config_data=self.config_parameters,
            drag_unc=self.perturbations['drag'],
            seed=self.seed
        )
        self._recorder = Recorder()
        self._gravity = Gravity()
        self._tvc = tvc_factory(
            dt=self.sample_period,
            config=self.config_parameters,
        )
        self._rcs = RCS(config_data=self.config_parameters, seed=self.seed)
        self._control_surfaces_set = ControlSurfacesSet(dt=self.sample_period,
            config_data=self.config_parameters, seed=self.seed)

        self._mass_properties = MassProperties(
            dt=self.sample_period,
            config_data=self.config_parameters,
            prop_weight_unc=self.perturbations["propellant_weight"],
            inert_weight_unc=self.perturbations["inert_weight"],
            payload_weight_unc=self.perturbations["payload_weight"],
            cg_offset_unc=self.perturbations["cg_offset"],
            seed=self.seed,
        )

        self._propulsion = Propulsion(
            phase=self.phase_id,
            config_data=self.config_parameters,
            thrust_applic_point_unc=self.perturbations["thrust_vector_x_offset"],
            thrust_vec_y_pos_unc=self.perturbations["thrust_vector_y_offset"],
            thrust_vec_z_pos_unc=self.perturbations["thrust_vector_z_offset"],
            seed=self.seed,
        )

        self.wind_model.initialize(
            config_data=self.config_parameters,
            dt=self.sample_period,
            wind_config=self.wind,
            seed=self.seed,
        )
        
        self.control_aerodynamics = AerodynamicControl(dt=self.sample_period)
        self.control_tvc = ControlTVC(config_data=self.config_parameters, dt=self.sample_period)
        self.guidance = Guidance(config_data=self.config_parameters, dt=self.sample_period, dt_bc=self.sample_period)  

    @performance_decorator.time_execution_stats
    def step(self) -> None:
        # Reseting interpolation time in case of phase changing
        if self.reset_interp_time:
           self._state = replace(self._state, interp_time=np.float64(0.0))

        # vvvvv Getting Max Control Surfaces deflection
        control_surface_data = self.config_parameters['aero_controls_parameters']
        
        # Finding phase index
        idx = next(
            (i for i, d in enumerate(control_surface_data) if d.get("phase_id") == self.phase_id),
            None  # returned if no match is found
        )

        max_deflection      = control_surface_data[idx]['max_deflection']
        max_deflection_rate = control_surface_data[idx]['max_rate']
        cs_config_type      = control_surface_data[idx]['control_surface_type']

        # vvvvv Physics Simulation vvvvv
        atmosphere_data = self.atmosphere_model.evaluate(self._state.alt)
        wb = self.wind_model.evaluate(state=self._state, launch_reference=self._launching_ref, Va=self._aero_state.Va, phase=self.phase_id)
        mass_properties_data = self._mass_properties.evaluate(self._state, self.phase_id)

        prop_loads, v_e = self._propulsion.evaluate(self._state, mass_properties_data, atmosphere_data, self.phase_id, self.By + self.error_By, self.Bz + self.error_Bz)
        aero_loads, self._aero_state, static_margin, Na, Nd, Ma, Mq, Md_tvc, Md_canard, LLda, LLp, CA, CY, CN, CLL, CM, CLN, DCA, DCY, DCN, DCLL, DCM, DCLN = self._aerodynamics.evaluate(self._state, self._aero_state, mass_properties_data, atmosphere_data, self.phase_id, prop_loads, wb, self.sample_period, self.delta_1, self.delta_2, self.delta_3, self.delta_4)
        grav_loads, grav_acc = self._gravity.evaluate(self._state, mass_properties_data)

        total_loads = aero_loads + prop_loads + grav_loads + self.rcs_loads

        downrange, crossrange, displacement = self._state.ned_displacements(self._launching_ref)

        # vvvvv Sensors vvvvv
        fs_b_m, gyro_b_m = self.imu.evaluate(self._state, mass_properties_data, total_loads, grav_acc, self.sample_period)
    
        # vvvvv On-board Computer vvvvv
        an_cmd, al_cmd, self.pn_dsrd, gamma_cmd, pitch_cmd, yaw_cmd = self.guidance.step(self._launching_ref, self._state, self.phase_id, fs_b_m, v_e)
        delta_q_tvc, delta_r_tvc = self.control_tvc.step(self._state, self._aero_state, self.config_parameters, self.phase_id, fs_b_m, gyro_b_m, Na, Ma, Mq, Md_tvc, an_cmd, al_cmd)
        delta_p_control_surface, delta_q_control_surface, delta_r_control_surface = self.control_aerodynamics.step(self._state, self._aero_state, self.config_parameters, self.phase_id, fs_b_m, gyro_b_m, grav_acc, Na, Nd, Ma, Mq, Md_canard, LLda, LLp, gamma_cmd, pitch_cmd, yaw_cmd)
                            
        # vvvvv Actuators vvvvv
        self.Bz, self.By = self._tvc.step(self.config_parameters, delta_q_tvc, delta_r_tvc, self.phase_id)
        if self.phase_id in [2, 3, 4]:
            self.Bz, self.By = 0.0, 0.0
        self.rcs_loads, rcs_mass = self._rcs.step(self._state, mass_properties_data, self.phase_id, self.sample_period, gyro_b_m, pitch_cmd, yaw_cmd)
        self.delta_1, self.delta_2, self.delta_3, self.delta_4 = self._control_surfaces_set.step(delta_p_control_surface, delta_q_control_surface, delta_r_control_surface, self.error_delta_1, self.error_delta_2, self.error_delta_3, self.error_delta_4, cs_config_type)
 
        self._recorder.append({
            'TIME__s': self._state.time,
            'PHASE': self.phase_id,
            'INTERP_TIME__s': self._state.interp_time,

            # AtmosphereData
            'SPEED_OF_SOUND__m_s': atmosphere_data.speed_of_sound_m_s,
            'TEMPERATURE__K': atmosphere_data.temperature_K,
            'PRESSURE__Pa': atmosphere_data.pressure_Pa,
            'DENSITY__kg_m3': atmosphere_data.density_kg_m3,

            # Wind
            'WIND_BODY_X__m_s': wb[0,0],
            'WIND_BODY_Y__m_s': wb[1,0],
            'WIND_BODY_Z__m_s': wb[2,0],
            'WIND_BODY_P__rad_s': wb[3,0],
            'WIND_BODY_Q__rad_s': wb[4,0],
            'WIND_BODY_R__rad_s': wb[5,0],

            # Propulsion
            'PROP_FORCE_X__N': prop_loads.fx,
            'PROP_FORCE_Y__N': prop_loads.fy,
            'PROP_FORCE_Z__N': prop_loads.fz,
            'PROP_MOMENT_X__Nm': prop_loads.l,
            'PROP_MOMENT_Y__Nm': prop_loads.m,
            'PROP_MOMENT_Z__Nm': prop_loads.n,
            'THRUST__N': prop_loads.force_norm,
            'MASS__kg': self._mass_properties.mass,
            'IXX__kgm2': self._mass_properties.Ixx,
            'IYY__kgm2': self._mass_properties.Iyy,
            'IZZ__kgm2': self._mass_properties.Izz,
            'IXY__kgm2': self._mass_properties.Ixy,
            'IXZ__kgm2': self._mass_properties.Ixz,
            'IYZ__kgm2': self._mass_properties.Iyz,
            'IXX_RATE__kgm2_s': self._mass_properties.Ixx_dot,
            'IYY_RATE__kgm2_s': self._mass_properties.Iyy_dot,
            'IZZ_RATE__kgm2_s': self._mass_properties.Izz_dot,
            'X_CG__m': self._mass_properties.x_cg,
            'EXHAUST_VELOCITY__m_s': v_e,
            'MASS_FLOW_RATE__kg_s': self._mass_properties.mass_dot,

            # Aerodynamics
            'AERO_FORCE_X__N': aero_loads.fx,
            'AERO_FORCE_Y__N': aero_loads.fy,
            'AERO_FORCE_Z__N': aero_loads.fz,
            'AERO_MOMENT_X__Nm': aero_loads.l,
            'AERO_MOMENT_Y__Nm': aero_loads.m,
            'AERO_MOMENT_Z__Nm': aero_loads.n,
            'TRUE_AIRSPEED__m_s': self._aero_state.Va,
            'DYNAMIC_PRESSURE__Pa': self._aero_state.Q,
            'ANGLE_OF_ATTACK__deg': self._aero_state.alpha,
            'SIDESLIP_ANGLE__deg': self._aero_state.beta,
            'ALPHA_DOT_deg__s': self._aero_state.alpha_dot,
            'MACH_NUMBER': self._aero_state.mach,
            'STATIC_MARGIN': static_margin,
            'NA__m_s2': Na,
            'ND__m_s2': Nd,
            'MA__1_s2': Ma,
            'MQ__1_s2': Mq,
            'MD__1_s2': Md_canard,
            'MD_TVC__1_s2': Md_tvc,
            'LLDA__1_s2': LLda,
            'LLP__1_s2': LLp,
            'CA': CA,
            'CY': CY,
            'CN': CN,
            'CLL': CLL,
            'CM': CM,
            'CLN': CLN,
            'DCA': DCA,
            'DCY': DCY,
            'DCN': DCN,
            'DCLL': DCLL,
            'DCM': DCM,
            'DCLN': DCLN,

            # Gravity
            'GRAVITY_FORCE_X__N': grav_loads.fx,
            'GRAVITY_FORCE_Y__N': grav_loads.fy,
            'GRAVITY_FORCE_Z__N': grav_loads.fz,

            # Equations of Motion
            'STATE_ECI_POS_X__m': self._state.x[0],
            'STATE_ECI_POS_Y__m': self._state.y[0],
            'STATE_ECI_POS_Z__m': self._state.z[0],
            'STATE_QUAT_W': self._state.q0[0],
            'STATE_QUAT_X': self._state.q1[0],
            'STATE_QUAT_Y': self._state.q2[0],
            'STATE_QUAT_Z': self._state.q3[0],
            'STATE_BODY_VEL_U__m_s': self._state.u[0],
            'STATE_BODY_VEL_V__m_s': self._state.v[0],
            'STATE_BODY_VEL_W__m_s': self._state.w[0],
            'SPECIFIC_FORCE_X__m_s2': fs_b_m[0,0],
            'SPECIFIC_FORCE_Y__m_s2': fs_b_m[1,0],
            'SPECIFIC_FORCE_Z__m_s2': fs_b_m[2,0],
            'BODY_RATE_P__rad_s': gyro_b_m[0,0],
            'BODY_RATE_Q__rad_s': gyro_b_m[1,0],
            'BODY_RATE_R__rad_s': gyro_b_m[2,0],
            'LATITUDE__deg': np.rad2deg(self._state.lat),
            'LONGITUDE__deg': np.rad2deg(self._state.lon),
            'ALTITUDE__m': self._state.alt,
            'HEADING__deg': np.rad2deg(self._state.heading),
            'FLIGHT_PATH_ANGLE__deg': np.rad2deg(self._state.gamma),
            'ROLL_ECI__deg': np.rad2deg(self._state.roll_eci),
            'PITCH_ECI__deg': np.rad2deg(self._state.pitch_eci),
            'YAW_ECI__deg': np.rad2deg(self._state.yaw_eci),
            'ROLL_NED__deg': np.rad2deg(self._state.roll_ned[0]),
            'PITCH_NED__deg': np.rad2deg(self._state.pitch_ned[0]),
            'YAW_NED__deg': np.rad2deg(self._state.yaw_ned[0]),
            'DOWNRANGE__m': downrange[0],
            'CROSSRANGE__m': crossrange[0],
            'DISPLACEMENT__m': displacement[0],
            'VEL_X_ECI__m_s': self._state.vx_eci,
            'VEL_Y_ECI__m_s': self._state.vy_eci,
            'VEL_Z_ECI__m_s': self._state.vz_eci,
            'VEL_X_ECEF__m_s': self._state.vx_ecef,
            'VEL_Y_ECEF__m_s': self._state.vy_ecef,
            'VEL_Z_ECEF__m_s': self._state.vz_ecef,
            'VEL_X_NED__m_s': self._state.vx_ned,
            'VEL_Y_NED__m_s': self._state.vy_ned,
            'VEL_Z_NED__m_s': self._state.vz_ned,

            # IMU
            'IMU_SPECIFIC_FORCE_X__m_s2': fs_b_m[0][0] if fs_b_m is not None else None,
            'IMU_SPECIFIC_FORCE_Y__m_s2': fs_b_m[1][0] if fs_b_m is not None else None,
            'IMU_SPECIFIC_FORCE_Z__m_s2': fs_b_m[2][0] if fs_b_m is not None else None,
            'IMU_BODY_RATE_P__rad_s': gyro_b_m[0][0] if gyro_b_m is not None else None,
            'IMU_BODY_RATE_Q__rad_s': gyro_b_m[1][0] if gyro_b_m is not None else None,
            'IMU_BODY_RATE_R__rad_s': gyro_b_m[2][0] if gyro_b_m is not None else None,

            # Guidance
            'NORMAL_ACC_CMD__m_s2': an_cmd,
            'LATERAL_ACC_CMD__m_s2': al_cmd,
            'GAMMA_CMD__deg': np.rad2deg(gamma_cmd),
            'PITCH_CMD__deg': pitch_cmd,
            'YAW_CMD__deg': yaw_cmd,
            'TARGET_POS_NED_X__m': self.pn_dsrd[0,0],
            'TARGET_POS_NED_Y__m': self.pn_dsrd[1,0],
            'TARGET_POS_NED_Z__m': self.pn_dsrd[2,0],

            # Control
            'DELTA_Q_TVC_CMD__deg': delta_q_tvc,
            'DELTA_R_TVC_CMD__deg': delta_r_tvc,
            'DELTA_P_CONTROL_SURFACE_CMD__deg': delta_p_control_surface,
            'DELTA_Q_CONTROL_SURFACE_CMD__deg': delta_q_control_surface,
            'DELTA_R_CONTROL_SURFACE_CMD__deg': delta_r_control_surface,

            # TVC
            'TVC_DELTA_Q__deg': np.rad2deg(self.Bz),
            'TVC_DELTA_R__deg': np.rad2deg(self.By),

            # RCS
            'RCS_MOMENT_X__Nm': self.rcs_loads.l,
            'RCS_MOMENT_Y__Nm': self.rcs_loads.m,
            'RCS_MOMENT_Z__Nm': self.rcs_loads.n,
            'RCS_MASS__kg': rcs_mass,

            # Canard Set
            'CONTROL_SURFACE_DELTA_1__deg': self.delta_1,
            'CONTROL_SURFACE_DELTA_2__deg': self.delta_2,
            'CONTROL_SURFACE_DELTA_3__deg': self.delta_3,
            'CONTROL_SURFACE_DELTA_4__deg': self.delta_4,

            'CONTROL_SURFACE_TYPE': cs_config_type,

        })

        self._state, self.gimbal_lock_protection = self.dynamics.step(self._state, self._aero_state, mass_properties_data, total_loads, self.sample_period, self.gimbal_lock_protection)
        
    def set_phase_id(self, phase_id: int) -> None:
        self.phase_id = phase_id

    @property
    def modules(self) -> dict[str, object]:
        return {
            "atmosphere_model": self.atmosphere_model,
            "control_surfaces": self.control_aerodynamics,
            "control_tvc": self.control_tvc,
            "dynamics": self.dynamics,
            "guidance": self.guidance,
            "imu": self.imu,
            "rcs": self._rcs,
            "simulator": self,
        }

    @performance_decorator.time_execution
    def run(self, mission_plan: MissionPlan) -> pd.DataFrame:

        # state initialization
        self._state = mission_plan.initial_state
        self._launching_ref = LaunchReference(
            lat=self._state.lat,
            lon=self._state.lon,
            alt=self._state.alt,
            azimuth=self._state.yaw_ned,
        )

        if np.abs(np.pi/2 - np.abs(self._state.pitch_ned)) <= np.deg2rad(5.0):
            self.gimbal_lock_protection = True
        else:
            self.gimbal_lock_protection = False

        # Wind and aedynamics may be initialized after defining initial state        
        self.wind_model.evaluate_ic(self._state, self._launching_ref)

        Va = self.wind_model.get_tas()
        self._aero_state = AeroState(Va=Va)

        for flight_phase in mission_plan.flight_phases:

            if flight_phase.dynamics is not None:
                self.dynamics = flight_phase.dynamics

            if flight_phase.maneuver is not None:
                if isinstance(flight_phase.end_condition, DurationEndCondition):
                    tf = self._state.time + flight_phase.end_condition.value
                else:
                    tf = np.inf
                flight_phase.maneuver.initialize(
                    t0=self._state.time,
                    tf=tf,
                )
                self.dynamics.set_maneuver(flight_phase.maneuver)

            for action in flight_phase.actions:
                module = self.modules[action["target"]]
                function = getattr(module, action["command"])
                kwargs = action.get("arguments", {})
                function(**kwargs)

            flight_phase.end_condition.initialize(self._state, self._aero_state)
            while not flight_phase.end_condition.evaluate(self._state, self._aero_state):
                # Reseting interpolation time for time-dependent interpolations in case phase changes
                if self.phase_id == self.previous_phase_id + 1:
                    self.reset_interp_time = True
                    self.previous_phase_id = self.phase_id
                else:
                    self.reset_interp_time = False                

                self.step()
            
            # To record Final Values
            self.step()

        sim_df = self._recorder.to_df()
        self._recorder.clear()

        return sim_df
