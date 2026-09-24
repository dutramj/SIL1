# CHRISTOPHERSEN, Henrik B. et al. A compact guidance, navigation, and control system for unmanned aerial vehicles. Journal of aerospace computing, information, and communication, v. 3, n. 5, p. 187-213, 2006.
# FARRELL, Jay. Aided navigation: GPS with high rate sensors. McGraw-Hill, Inc., 2008.

# Python standard libraries
from pathlib import Path

# 3rd party libraries
import numpy as np
from numpy import sin
from numpy.linalg import norm, inv
from typing_extensions import Self
import yaml

# VAHSimulator library
from .utils import body_to_eci_quaternion, eci_to_body_quaternion, eci_to_geodetic, geodetic_to_geocentric_latitude, eci_to_ecef, skew_matrix, quaternion_to_euler_321, euler_to_quaternion
from .parameters import we, GM, a, J2
from .vehicle_state import VehicleState
from . import performance_decorator


I3x3 = np.eye(3)
I10x10 = np.eye(10)
I16x16 = np.eye(16)


class Navigation:
    
    def __init__(self, nav_params: dict, initial_state: VehicleState):

        self.x_nav = np.zeros((16, 1))

        # attitude
        roll, pitch, yaw = quaternion_to_euler_321(initial_state.quaternion)
        roll_nav = roll + nav_params['std_roll_pitch']*np.random.randn()
        pitch_nav = pitch + nav_params['std_roll_pitch']*np.random.randn()
        yaw_nav = yaw + nav_params['std_yaw']*np.random.randn()
        self.x_nav[0:4] = euler_to_quaternion(roll_nav, pitch_nav, yaw_nav)
        self.x_nav[0:4] = self.x_nav[0:4] / norm(self.x_nav[0:4])
        # position
        self.x_nav[4:7] = np.add(initial_state.position_eci, nav_params['std_position']*np.random.randn(3, 1))
        # velocity
        LIB = body_to_eci_quaternion(initial_state.quaternion)
        wei = np.array([[0], [0], [we]])
        wei_sm = skew_matrix(wei)
        self.x_nav[7:10] = np.add(np.add(np.matmul(LIB, initial_state.velocity_body), np.matmul(wei_sm, initial_state.position_eci)), nav_params['std_velocity']*np.random.randn(3, 1))
        # biases
        self.x_nav[10:16] = np.zeros((6, 1))

        self.P = np.zeros((16, 16))
        
        self.Q = np.eye(4)

        self.R = np.zeros((10,10))  # measurements noise covariance matrix
        self.H = np.concatenate((I10x10, np.zeros((10, 6))), axis=1)
        self.grav_mag = 0.0

    @classmethod
    def from_yaml(cls, nav_file: Path, initial_state: VehicleState) -> Self:

        nav_params = yaml.safe_load(open(nav_file))
        
        return cls(nav_params, initial_state)

    def _gravity_wgs84(self, lat, alt, pe):
        px = pe[0, 0]
        py = pe[1, 0]
        pz = pe[2, 0]
        r = norm(pe)
        lat = geodetic_to_geocentric_latitude(lat, alt)
        
        return -GM / r**2*np.array([[(1.0 + 3.0 / 2.0*(a / r)**2*J2*(1.0 - 5*sin(lat)**2))*px / r],
                                      [(1.0 + 3.0 / 2.0*(a / r)**2*J2*(1.0 - 5*sin(lat)**2))*py / r],
                                      [(1.0 + 3.0 / 2.0*(a / r)**2*J2*(3.0 - 5*sin(lat)**2))*pz / r]])

    def _covariance_derivative(self, P, Q, LIB, a, w, q_):
        # F matrix
        fx = a[0, 0]
        fy = a[1, 0]
        fz = a[2, 0]
        q0 = q_[0, 0]
        q1 = q_[1, 0]
        q2 = q_[2, 0]
        q3 = q_[3, 0]
        p = w[0, 0]
        q = w[1, 0]
        r = w[2, 0]

        F11 = 0.5*np.array([[0.0, -p, -q, -r],
                            [p, 0.0, r, -q],
                            [q, -r, 0.0, p],
                            [r, q, -p, 0.0]])

        F15 = 0.5*np.array([[q1, q2, q3],
                            [-q0, q3, -q2],
                            [-q3, -q0, q1],
                            [q2, -q1, -q0]])
        ax = -fx
        ay = -fy
        az = -fz
        F31 = 2.0*np.array([[q3*ay - q2*az, -(q2*ay + q3*az), 2*q2*ax - q1*ay - q0*az, 2*q3*ax + q0*ay - q1*az],
                            [q1*az - q3*ax, 2*q1*ay - q2*ax + q0*az, -(q1*ax + q3*az), 2*q3*ay - q0*ax - q2*az],
                            [q2*ax - q1*ay, 2*q1*az - q3*ax - q0*ay, 2*q2*az - q3*ay + q0*ax, -(q1*ax + q2*ay)]])
        F34 = -1*LIB

        F1 = np.concatenate((F11, np.zeros((4, 9)), F15), axis=1)
        F2 = np.concatenate((np.zeros((3, 7)), I3x3, np.zeros((3, 6))), axis=1)
        F3 = np.concatenate((F31, np.zeros((3, 6)), F34, np.zeros((3, 3))), axis=1)
        F4 = np.zeros((6, 16))
        F = np.concatenate((F1, F2, F3, F4), axis=0)

        # Lyapunov equation
        dPdt = np.add(np.matmul(F, P), np.add(np.matmul(P, np.transpose(F)), Q))

        return dPdt
    
    def _covariance_integration(self, P, Q, LIB, a, w, q_, dt):
        dPdt = self._covariance_derivative(P, Q, LIB, a, w, q_)
        return P + dt*dPdt

    def _imu_state_derivative(self, x, a, w):
        """Equations for propagating the estimates of the evolving IMU state."""
        q_ = x[0:4]
        p = w[0, 0]
        q = w[1, 0]
        r = w[2, 0]
        Om = np.array([[0.0, -p, -q, -r],
                       [p, 0.0, r, -q],
                       [q, -r, 0.0, p],
                       [r, q, -p, 0.0]])

        dqdt = np.ravel(0.5*np.matmul(Om, q_))
        dpdt = np.ravel(x[7:10])
        dvdt = np.ravel(a)
        dbadt = np.ravel(np.zeros((3, 1)))
        dbgdt = np.ravel(np.zeros((3, 1)))

        dxdt = np.array([dqdt[0], dqdt[1], dqdt[2], dqdt[3], 
                         dpdt[0], dpdt[1], dpdt[2], 
                         dvdt[0], dvdt[1], dvdt[2], 
                         dbadt[0], dbadt[1], dbadt[2], 
                         dbgdt[0], dbgdt[1], dbgdt[2]])
        dxdt = np.reshape(dxdt, (16, 1))

        return dxdt

    def _imu_state_integration(self, x, a, w, dt):
        """4th order Runge-Kutta numerical integration of the IMU state."""
        k1 = self._imu_state_derivative(x, a, w)
        k2 = self._imu_state_derivative(np.add(x, 0.5*dt*k1), a, w)
        k3 = self._imu_state_derivative(np.add(x, 0.5*dt*k2), a, w)
        k4 = self._imu_state_derivative(np.add(x, dt*k3), a, w)

        dxdt = (1.0 / 6.0)*np.add(k1, np.add(2.0*k2, np.add(2.0*k3, k4)))
        x = np.add(x, dt*dxdt)
        x[0:4] = x[0:4] / norm(x[0:4])

        return x
    
    def _propagate(self, dt, t, am, wm):
        """
        For each IMU measurement received, propagate the state and covariance.

        Inputs:
            dt: time step
            am: measurements of accelerometers
            wm: measurements of gyroscopes
        """
        q = self.x_nav[0:4]
        pi = self.x_nav[4:7]
        vi = self.x_nav[7:10]
        ba = self.x_nav[10:13]
        bg = self.x_nav[13:16]

        xi = pi[0, 0]
        yi = pi[1, 0]
        zi = pi[2, 0]
        
        lat, _, h = eci_to_geodetic(xi, yi, zi, t)
        LEI = eci_to_ecef(we*t)
        pe = np.matmul(LEI, pi)
        g_e = self._gravity_wgs84(lat, h, pe)

        # wei = np.array([[0], [0], [we]])
        # wei_sm = skew_matrix(wei)
        # wee = np.matmul(LEI, wei)
        # wee_sm = skew_matrix(wee)

        wee = np.array([[0], [0], [we]])
        wee_sm = skew_matrix(wee)
        g_e = np.add(g_e, -np.matmul(wee_sm, np.matmul(wee_sm, pe)))
        
        LIE = np.transpose(LEI)
        g_i = np.matmul(LIE, g_e)
        # g_i = np.add(g_i, -2*np.matmul(wei_sm, vi))
        LBI = eci_to_body_quaternion(q)
        g_b = np.matmul(LBI, g_i)  # gravity in body frame
        self.grav_mag = norm(g_b)

        LIB = body_to_eci_quaternion(q)
        am_b = np.add(am, -ba)
        a = np.matmul(LIB, np.add(am_b, g_b))
        w = np.add(wm, -bg)
        # w = np.add(w, -np.matmul(LBI, wee))

        # propagate the state estimate
        self.x_nav = self._imu_state_integration(self.x_nav, a, w, dt)
        self.P = self._covariance_integration(self.P, self.Q, LIB, am_b, w, q, dt)

    def _update(self, zp, zv, zy):
        """When GNSS-position (zp), GNSS-velocity (zv) and GNSS-yaw (zy) become available, performe an EKF update.

        Inputs:
            zp: measure of GNSS-position
            zv: measure of GNSS-velocity
            zy: measure of GNSS-yaw
        """
        # inovation
        roll, pitch, _ = quaternion_to_euler_321(self.x_nav[0:4])
        zq = euler_to_quaternion(roll, pitch, zy)
        zq = zq / norm(zq)
        z = np.concatenate((zq, zp, zv), axis=0)
        dk = np.add(z, -np.matmul(self.H, self.x_nav))

        # Kalman gain
        HPH = np.matmul(self.H, np.matmul(self.P, np.transpose(self.H)))
        S = np.add(HPH, self.R)
        invS = inv(S)
        K = np.matmul(self.P, np.matmul(np.transpose(self.H), invS))

        # state update
        self.x_nav = np.add(self.x_nav, np.matmul(K, dk))
        self.x_nav[0:4] = self.x_nav[0:4] / norm(self.x_nav[0:4])

        # state covariance update
        self.P = np.matmul(np.add(I16x16, -np.matmul(K, self.H)), self.P)

    @performance_decorator.time_execution_stats
    def step(self, dt, t, am, wm, zp, zv, zy):
        if am is not None and wm is not None:
            self._propagate(dt, t, am, wm)
            if zp is not None:
                self._update(zp, zv, zy)

        return self.x_nav, self.P, self.grav_mag
