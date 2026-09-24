# Python standard libraries
from dataclasses import dataclass, field

# 3rd party libraries
from typing_extensions import Self
import numpy as np

# VAHSimulator library
from .utils import (
    body_to_eci_quaternion,
    ecef_to_eci,
    ecef_to_ned_matrix,
    eci_to_body_quaternion,
    eci_to_geodetic,
    eci_to_ecef,
    eci_to_ned_attitude,
    euler_to_quaternion,
    geodetic_to_eci,
    ned_to_ecef_matrix,
    quaternion_to_euler_321,
    compute_downrange_crossrange,
)
from .parameters import earth_rate__rad_s
from . import performance_decorator
from .launching_reference import LaunchReference

@dataclass(slots=True, frozen=True)
class VehicleState:
    x : np.float64
    y : np.float64
    z : np.float64
    q0 : np.float64
    q1 : np.float64
    q2 : np.float64
    q3 : np.float64
    u : np.float64
    v : np.float64
    w : np.float64
    p : np.float64
    q : np.float64
    r : np.float64
    roll_ned: np.float64
    pitch_ned: np.float64
    yaw_ned: np.float64
    time : np.float64 = np.float64(0.0)
    interp_time: np.float64 = np.float64(0.0)
    lat: np.float64 = field(init=False)
    lon: np.float64 = field(init=False)
    alt: np.float64 = field(init=False)
    vx_ecef: np.float64 = field(init=False)
    vz_ecef: np.float64 = field(init=False)
    vy_ecef: np.float64 = field(init=False)
    vx_eci: np.float64 = field(init=False)
    vy_eci: np.float64 = field(init=False)
    vz_eci: np.float64 = field(init=False)
    vx_ned: np.float64 = field(init=False)
    vy_ned: np.float64 = field(init=False)
    vz_ned: np.float64 = field(init=False)
    gamma: np.float64 = field(init=False)
    heading: np.float64 = field(init=False)
    roll_eci: np.float64 = field(init=False)
    pitch_eci: np.float64 = field(init=False)
    yaw_eci: np.float64 = field(init=False)
    q0_ned: np.float64 = field(init=False)
    q1_ned: np.float64 = field(init=False)
    q2_ned: np.float64 = field(init=False)
    q3_ned: np.float64 = field(init=False)
    p_r: np.float64 = field(init=False)
    q_r: np.float64 = field(init=False)
    r_r: np.float64 = field(init=False)

    @performance_decorator.time_execution_stats
    def __post_init__(self) -> None:
        self.__normalize_quaternion()
        self.__compute_geodetic_position()
        self.__compute_ecef_velocity()
        self.__compute_inertial_velocity()
        self.__compute_ned_velocity()
        self.__compute_flight_path_angles()
        self.__compute_euler_angles()
        self.__compute_ned_quaternion()
        self.__compute_relative_angular_velocity()

    @performance_decorator.time_execution_stats
    def __normalize_quaternion(self) -> None:
        
        q = np.array([self.q0, self.q1, self.q2, self.q3], dtype=np.float64)
        norm = np.linalg.norm(q)
        q = q/norm
        object.__setattr__(self, "q0", q[0])
        object.__setattr__(self, "q1", q[1])
        object.__setattr__(self, "q2", q[2])
        object.__setattr__(self, "q3", q[3])

    @performance_decorator.time_execution_stats
    def __compute_euler_angles(self) -> None:
        roll_eci, pitch_eci, yaw_eci = quaternion_to_euler_321(self.quaternion)
        object.__setattr__(self, "roll_eci", roll_eci)
        object.__setattr__(self, "pitch_eci", pitch_eci)
        object.__setattr__(self, "yaw_eci", yaw_eci)

    @performance_decorator.time_execution_stats
    def __compute_ned_quaternion(self):
        q0_ned, q1_ned, q2_ned, q3_ned = eci_to_ned_attitude(
                    self.quaternion,
                    self.lat,
                    self.lon,
                    self.time,
                )
        object.__setattr__(self, "q0_ned", q0_ned)
        object.__setattr__(self, "q1_ned", q1_ned)
        object.__setattr__(self, "q2_ned", q2_ned)
        object.__setattr__(self, "q3_ned", q3_ned)      

    @performance_decorator.time_execution_stats
    def __compute_geodetic_position(self) -> None:
        lat, lon, alt = eci_to_geodetic(self.x[0], self.y[0], self.z[0], self.time)
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "alt", alt)

    @performance_decorator.time_execution_stats
    def __compute_ecef_velocity(self) -> None:
        u, v, w = self.velocity_body
        V_B = np.array([[u[0]], [v[0]], [w[0]]])

        V_ECEF = self.dcm_eci_to_ecef @ self.dcm_body_to_eci @ V_B

        object.__setattr__(self, "vx_ecef", V_ECEF[0, 0])
        object.__setattr__(self, "vy_ecef", V_ECEF[1, 0])
        object.__setattr__(self, "vz_ecef", V_ECEF[2, 0])

    @performance_decorator.time_execution_stats
    def __compute_inertial_velocity(self) -> None:
        V_ECEF = np.array([[self.vx_ecef],
                           [self.vy_ecef],
                           [self.vz_ecef]])

        V_rel_I = self.dcm_ecef_to_eci @ V_ECEF

        xi, yi, _ = self.position_eci

        omega_cross_r = np.array([[-earth_rate__rad_s * xi[0]], [earth_rate__rad_s * yi[0]], [0.0]])

        V_I = V_rel_I + omega_cross_r

        object.__setattr__(self, "vx_eci", V_I[0, 0])
        object.__setattr__(self, "vy_eci", V_I[1, 0])
        object.__setattr__(self, "vz_eci", V_I[2, 0])

    @performance_decorator.time_execution_stats
    def __compute_ned_velocity(self) -> None:
        v_ecef = np.array([
            [self.vx_ecef],
            [self.vy_ecef],
            [self.vz_ecef]
        ])
        
        v_ned = self.dcm_ecef_to_ned @ v_ecef

        object.__setattr__(self, "vx_ned", v_ned[0, 0])
        object.__setattr__(self, "vy_ned", v_ned[1, 0])
        object.__setattr__(self, "vz_ned", v_ned[2, 0])

    @performance_decorator.time_execution_stats
    def __compute_flight_path_angles(self):
        v_ned = np.array([
            [self.vx_ned],
            [self.vy_ned],
            [self.vz_ned]
        ])

        V = np.linalg.norm(v_ned)

        if V < 1e-6:
            object.__setattr__(self, "heading", np.nan)
            object.__setattr__(self, "gamma", np.nan)

        # heading angle (azimuth from north)
        heading = np.atan2(v_ned[1, 0], v_ned[0, 0])  
        if heading > np.pi:
            heading -= 2*np.pi

        # elevation (flight path angle)
        denom = np.sqrt(v_ned[0, 0]**2 + v_ned[1, 0]**2)
        if denom > 1e-10:
            gamma = np.atan2(-v_ned[2, 0], denom)
        else:
            # Edge case: vertical flight
            if v_ned[2, 0] > 0:
                gamma = -np.pi / 2  # Diving straight down
            elif v_ned[2, 0] < 0:
                gamma = np.pi / 2   # Climbing straight up
            else:
                gamma = 0.0

        object.__setattr__(self, "heading", heading)
        object.__setattr__(self, "gamma", gamma)

    @performance_decorator.time_execution_stats
    def ned_displacements(self, launching_ref: LaunchReference) -> tuple[np.float64, np.float64, np.float64]:
        lat_deg = np.rad2deg(self.lat)
        lon_deg = np.rad2deg(self.lon)
        alt_m   = self.alt

        return compute_downrange_crossrange(
            lat_deg,
            lon_deg,
            alt_m,
            launching_ref,
            )

    def __compute_relative_angular_velocity(self):
        # Rotation Matrix - ECI to body frame
        quaternion = self.quaternion
        LBI = eci_to_body_quaternion(quaternion)

        # Creating Earth's angular rotation vector with relation to ECI expressed in body frame
        omega_e = np.array([[0.], [0.], [earth_rate__rad_s]])
        omega_e_b = LBI @ omega_e

        # Computing the angular rates relatie to ECEF
        p_r = np.float64(self.p[0] - omega_e_b[0, 0])
        q_r = np.float64(self.q[0] - omega_e_b[1, 0])
        r_r = np.float64(self.r[0] - omega_e_b[2, 0])

        object.__setattr__(self, "p_r", p_r)
        object.__setattr__(self, "q_r", q_r)
        object.__setattr__(self, "r_r", r_r)

    @property
    def vector(self) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.z,
            self.q0,
            self.q1,
            self.q2,
            self.q3,
            self.u,
            self.v,
            self.w,
            self.p,
            self.q,
            self.r,
        ], dtype=np.float64)
    
    @property
    def position_eci(self) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.z,
        ], dtype=np.float64)

    @property
    def position_geodetic(self) -> np.ndarray:
        return np.array([
            self.lat,
            self.lon,
            self.alt,
        ], dtype=np.float64)

    @property
    def quaternion(self) -> np.ndarray:
        return np.array([
            self.q0,
            self.q1,
            self.q2,
            self.q3,
        ], dtype=np.float64)

    @property
    def dcm_body_to_eci(self) -> np.ndarray:
        return body_to_eci_quaternion(self.quaternion)

    def dcm_eci_to_body(self) -> np.ndarray:
        return eci_to_body_quaternion(self.quaternion)

    @property
    def dcm_eci_to_ecef(self) -> np.ndarray:        
        return eci_to_ecef(earth_rate__rad_s * self.time)

    @property
    def dcm_ecef_to_eci(self) -> np.ndarray:        
        return ecef_to_eci(earth_rate__rad_s * self.time)

    @property
    def dcm_ecef_to_ned(self) -> np.ndarray:
        return ecef_to_ned_matrix(self.lat, self.lon)

    @property
    def dcm_ned_to_ecef(self) -> np.ndarray:
        return ned_to_ecef_matrix(self.lat, self.lon)

    @property
    def velocity_body(self) -> np.ndarray:
        return np.array([
            self.u,
            self.v,
            self.w,
        ], dtype=np.float64)

    @property
    def velocity_eci(self) -> np.ndarray:
        return np.array([
            self.vx_eci,
            self.vy_eci,
            self.vz_eci,
        ], dtype=np.float64)

    @property
    def angular_rate_body(self) -> np.ndarray:
        return np.array([
            self.p,
            self.q,
            self.r,
        ], dtype=np.float64)

    @classmethod
    @performance_decorator.time_execution_stats
    def from_vector(cls, time, interp_time, roll_ned, pitch_ned, yaw_ned, state_array: np.ndarray) -> Self:
        return cls(
            time=time,
            interp_time=interp_time,
            roll_ned=[roll_ned],
            pitch_ned=[pitch_ned],
            yaw_ned=[yaw_ned],
            *state_array
        )
    
    @classmethod
    def from_dict(cls, x0_dict: dict) -> Self:

        ## Position in ECI
        xi, yi, zi = geodetic_to_eci(x0_dict['lat'], x0_dict['lon'], x0_dict['alt'], np.float64(0.0))
        quat_ned = euler_to_quaternion(x0_dict['roll'], x0_dict['pitch'], x0_dict['yaw'])
        quat_ned = quat_ned / np.linalg.norm(quat_ned)
        LNB = body_to_eci_quaternion(quat_ned)
        LEN = ned_to_ecef_matrix(x0_dict['lat'], x0_dict['lon'])
        LIE = ecef_to_eci(0.0)
        LIB = np.matmul(LIE, np.matmul(LEN, LNB))

        roll = np.atan2(LIB[2, 1], LIB[2, 2])
        if abs(LIB[2, 0]) > 1:
            LIB[2, 0] = np.sign(LIB[2, 0])
        pitch = -np.asin(LIB[2, 0])
        yaw = np.atan2(LIB[1, 0], LIB[0, 0])

        q = euler_to_quaternion(roll, pitch, yaw)
        q /= np.linalg.norm(q)

        # Angular velocity
        LBI = eci_to_body_quaternion(q)
        w_eci = np.array([[0], [0], [earth_rate__rad_s]])
        w_body = np.matmul(LBI, w_eci)
        
        x0 = {
            'x' : [xi],
            'y' : [yi],
            'z' : [zi],
            'q0' : [q[0, 0]],
            'q1' : [q[1, 0]],
            'q2' : [q[2, 0]],
            'q3' : [q[3, 0]],
            'u' : [x0_dict['airspeed_u']],
            'v' : [x0_dict['airspeed_v']],
            'w' : [x0_dict['airspeed_w']],
            'p' : [w_body[0, 0]],
            'q' : [w_body[1, 0]],
            'r' : [w_body[2, 0]],
            'roll_ned': [x0_dict['roll']],
            'pitch_ned': [x0_dict['pitch']],
            'yaw_ned': [x0_dict['yaw']],
        }
        return cls(**x0)
