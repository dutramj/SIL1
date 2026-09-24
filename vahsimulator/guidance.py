# Jaggers, R., An Explicit Solution to the Exo-atmospheric Powered Flight Guidance and Trajectory Optimization Problem for Rocket Propelled Vehicles, AIAA Paper, Vol. 1051, 1977.
# https://en.wikipedia.org/wiki/Proportional_navigation

# Python standard libraries
from copy import copy

# 3rd party libraries
import numpy as np
from numpy import sin, cos, log, sqrt, atan2
from numpy.linalg import norm

# VAHSimulator library
from .utils import eci_to_body_quaternion, skew_matrix, eci_to_ned, ecef_to_ned_matrix, euler_to_quaternion
from .parameters import GM, Re_equatorial, earth_rate__rad_s
from . import performance_decorator
from .vehicle_state import VehicleState
from .launching_reference import LaunchReference


class Guidance:
    def __init__(self, config_data: dict, dt, dt_bc):

        self._enabled = False

        self.dt = dt
        self.dt_bc = dt_bc

        # Getting vehicle guidance parameters
        control_interlocks  = config_data['control_interlocks'][0]
        guidance_parameters = config_data['guidance_parameters'][0] 

        self.transition_time = control_interlocks['transition_time']
        self.action_time     = control_interlocks['action_time']
        self.max_lmb         = guidance_parameters['max_turn_rate']
        self.max_acc         = guidance_parameters['max_acceleration']
        

        # Proportional Navigation parameters
        self.N = 20.0  # proportional navigation constant
        self.an_cmd = 0.0
        self.al_cmd = 0.0

        self.orb_position_dsrd = Re_equatorial + 35000.0  # desired orbital position, m
        self.orb_velocity_dsrd = 2359.2145  # desired orbital velocity, m/s
        self.path_angle_dsrd = 0.0  # desired path angle, deg

        self.pi_pred = np.ones((3, 1))*1e-7
        self.vi_pred = np.ones((3, 1))*1e-7

        self.tb = self.action_time  # time remaining until burnout, s
        self.L = 0.0  # velocity to be gained, m/s

        self.v_go = np.zeros((3, 1))  # velocity to go, m/s
        self.t_go = 1.0  # time to go, s
        self.prev_t_go = 1.0  # previous time to go, s
        self.r_go = np.ones((3, 1))*1e-7  # range to go, m
        self.tau = 0.0  # characteristic time, s

        self.pi_dsrd = np.ones((3, 1))*1e-7  # desired inertial position, m
        self.v_dsrd = np.ones((3, 1))*1e-7  # desired inertial velocity, m/s
        self.pn_dsrd = np.ones((3, 1))*1e-7  # desired inertial position in NED, m

        self.p_bias = np.zeros((3, 1))  # position bias, m
        self.p_grav = np.zeros((3, 1))  # position loss due to gravity, m
        self.u_d = np.zeros((3, 1))  # unit vector of pi_pred and pi_dsrd
        self.u_y = np.zeros((3, 1))  # unit vector normal to trajectory plane
        self.u_z = np.zeros((3, 1))  # unit vector in traj plane, normal to desired inertial pos
        self.first_step = True
        self.initialized = False

        # Gamma command parameters
        self.gamma_cmd = 0.0
        self.alt_target = 50000.0
        
        self.gain_alt = 0.000018
        self.gain_alt_rate = 0.0009
        
        self.prev_alt = None
        self.alt_rate = 0.0
        
        # Attitude alignment with velocity vector
        self.pitch_cmd = 0.0  # Pitch command for RCS
        self.yaw_cmd = 0.0    # Yaw command for RCS

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def _kepler_projector(self, pi, vi):
        sqrt_GM = sqrt(GM)
        ro = norm(pi)
        vo = norm(vi)
        rvo = np.dot(pi.T, vi)[0, 0]
        a1 = vo*vo / GM
        sa = ro / (2.0 - ro*a1)
        if sa < 0.0:
            # print(f'Guidance: {self.t:.2f} s, Kepler projector failed, negative semi-major axis')
            return None, None
        smua = sqrt_GM*sqrt(sa)
        mdot = smua / (sa*sa)

        dm = mdot*self.t_go
        de = dm  # initialize eccentricity
        a11 = rvo / smua
        a21 = (sa - ro) / sa
        count20 = 0
        adm = np.inf
        while True:
            cde = 1.0 - cos(de)
            sde = sin(de)
            dmn = de + a11*cde - a21*sde
            dmerr = dm - dmn

            adm = abs(dmerr) / mdot
            dmde = 1.0 + a11*sde - a21*(1.0 - cde)
            de = de + dmerr / dmde
            count20 += 1
            if count20 > 20:
                # print(f"Guidance: {self.t:.2f} s, Kepler projector failed: convergence issue")
                return None, None
            if adm < 1e-7:
                break
        
        # projected position
        fk = (ro - sa*cde) / ro
        gk = (dm + sde - de) / mdot
        pi_pred = np.add(pi*fk, vi*gk)

        # projected velocity
        rp = norm(pi_pred)
        fdk = -smua*sde / ro
        gdk = rp - sa*cde
        vi_pred = np.add(pi*(fdk / rp), vi*(gdk / rp))
        
        return pi_pred, vi_pred
    
    def _end_state_predictor(self, d_lmb_dt, u_lmb, L, S, J, Q, H, P, J_L, q_, pi, vi):
        d_lmb_dt_sq = np.dot(d_lmb_dt.T, d_lmb_dt)[0, 0]
        v_th = u_lmb*(L - 0.5*d_lmb_dt_sq*(H - J*J_L))  # velocity gained due to thrust

        pi_th = np.add(u_lmb*(S - 0.5*d_lmb_dt_sq*(P - J_L*(Q + q_))), d_lmb_dt*q_)  # displacement gained due to thrust

        self.p_bias = np.add(self.r_go, -pi_th)

        pi_0 = np.add(pi, np.add(-pi_th*0.1, -v_th*(self.t_go / 30.0))) 
        vi_0 = np.add(vi, np.add(pi_th*(1.2 / self.t_go), -v_th*0.1)) 

        pi_1, vi_1 = self._kepler_projector(pi_0, vi_0)

        # gravity corrections
        if pi_1 is not None and vi_1 is not None:
            v_grav = np.add(vi_1, -vi_0)
            self.p_grav = np.add(pi_1, np.add(-pi_0, -vi_0*self.t_go)) 

            self.pi_pred = np.add(pi, np.add(vi*self.t_go, np.add(self.p_grav, pi_th))) 
            self.vi_pred = np.add(vi, np.add(v_grav, v_th))

    def _range_to_go(self, pi, vi, S, u_lmb):
        self.p_grav = self.p_grav*(self.t_go / self.prev_t_go)**2

        r_go_ = np.add(self.pi_dsrd, np.add(-np.add(pi, np.add(vi*self.t_go, self.p_grav)), -self.p_bias))

        r_go_x = np.dot(r_go_.T, self.u_d)[0, 0]
        r_go_y = np.dot(r_go_.T, self.u_y)[0, 0]
        r_go_xy = np.add(r_go_x*self.u_d, r_go_y*self.u_y)

        num = np.dot(r_go_xy.T, u_lmb)[0, 0]
        denom = np.dot(u_lmb.T, self.u_z)[0, 0]
        if abs(denom) > 0.0:
            r_go_z = (S - num) / denom
            self.r_go = np.add(r_go_xy, r_go_z*self.u_z)

    def _turn_rate(self, S, Q, J_L, pi, vi):
        u_lmb = self.v_go / norm(self.v_go)  # unit thrust vector in direction of velocity to go

        if self.first_step:
            self.first_step = False
            self.r_go = u_lmb*S
        
        self._range_to_go(pi, vi, S, u_lmb)

        denom = (Q - S*J_L)
        if denom != 0.0:
            d_lmb_dt = np.add(self.r_go, -S*u_lmb) / denom
        else:
            d_lmb_dt = np.zeros((3, 1))
        
        d_lmb_dt_mag = norm(d_lmb_dt)
        if d_lmb_dt_mag >= self.max_lmb:
            d_lmb_dt = (d_lmb_dt / d_lmb_dt_mag)*self.max_lmb

        return u_lmb, d_lmb_dt

    def _integrals(self, v_e):
        """
        Calculate the integrals for the linear tangent guidance law.
        Inputs:
            transition_time_remaining: time remaining until burnout, s
            L: velocity to be gained, m/s
            v_e: effective exhaust velocity, m/s
        
        Outputs:
            S: thrust integral, position - m   
            J: thrust integral, velocity*time - m   
            Q: thrust integral, position*time - m*s   
            H: thrust integral, velocity*time^2 - m*s   
            P: thrust integral, position*time^2 - m*s²
            J/L: time remaining - s
            t_lmb: time of thrust integration - s 
            q_: conditioned Q-integral - m*s
        """
        x = self.tb / self.tau
        
        if x == 2:
            a1 = 1.0 / (1.0 - 0.5*x*1.001)
        else:
            a1 = 1.0 / (1.0 - 0.5*x)
        
        if x == 1:
            a2 = 1.0 / (1.0 - x*1.001)
            # print(f'Guidance: {self.t:.2f} s, end-state cannot be reached')
        else:
            a2 = 1.0 / (1.0 - x)
        
        aa = v_e / self.tau  # longitudinal acceleration of booster, m/s²

        a1x = 4.0*a1 - a2 - 3.0
        a2xsq = 2.0*a2 - 4.0*a1 + 2.0
        S = (aa*self.tb**2 / 2.0)*(1.0 + a1x / 3.0 + a2xsq / 6.0)
        J = (aa*self.tb**2 / 2.0)*(1.0 + a1x*(2.0 / 3.0) + a2xsq / 2.0)
        Q = (aa*self.tb**3 / 6.0)*(1.0 + a1x / 2.0 + a2xsq*0.3)
        P = (aa*self.tb**4 / 12.0)*(1.0 + a1x*0.6 + a2xsq*0.4)
        H = J*self.t_go - Q

        if self.L < 1e-7:
            self.L = 1e-7

        J_L = J / self.L
        q_ = Q - S*J_L
        
        return S, J, Q, H, P, J_L, q_

    def _time_to_go(self, time, v_e):
        V_go = norm(self.v_go)
        self.prev_t_go = copy(self.t_go)

        self.tau = self.tau - (time - self.action_time)
        
        self.tb = self.action_time - time

        self.L = 0.0
        if self.tau > 0 and self.tb / self.tau < 1.0:
            self.L = -v_e*log(1.0 - self.tb / self.tau)  # L-integral (velocity to be gained - m/s)
            if self.L < V_go:
                self.t_go = self.tb

    def _end_state_correction(self, time, pi, vi):
        self.u_d = self.pi_pred / norm(self.pi_pred)
        self.pi_dsrd = self.orb_position_dsrd*self.u_d  # desired inertial position, m
        self.pn_dsrd = eci_to_ned(self.pi_dsrd[0, 0], self.pi_dsrd[1, 0], self.pi_dsrd[2, 0], self.lat_ref, self.lon_ref, self.alt_ref, time).reshape((3, 1))

        Vi = skew_matrix(vi)
        Y = np.matmul(Vi, pi)
        self.u_y = Y / norm(Y)

        U_d = skew_matrix(self.u_d)
        self.u_z = np.matmul(U_d, self.u_y)
        self.u_z = self.u_z / norm(self.u_z)

        # velocity-to-be-gained
        self.v_dsrd = np.add(self.u_d*sin(np.deg2rad(self.path_angle_dsrd)), self.u_z*cos(np.deg2rad(self.path_angle_dsrd)))*self.orb_velocity_dsrd
        v_miss = np.add(self.vi_pred, -self.v_dsrd)
        self.v_go = np.add(self.v_go, -v_miss)
    
        self.is_pi_dsrd_computed = True
    
    def _proportional_navigation(self, state: VehicleState, fs_b, v_e):
        pi = state.position_eci
        q = state.quaternion
        LBI = eci_to_body_quaternion(q)
        LIB = np.transpose(LBI)
        wei = np.array([[0], [0], [earth_rate__rad_s]])
        wei_sm = skew_matrix(wei)
        ve = np.matmul(LIB, state.velocity_body)
        vi = np.add(ve, np.matmul(wei_sm, pi))  # velocity written in ECI frame, m/s

        q_ned = euler_to_quaternion(state.roll_ned[0], state.pitch_ned[0], state.yaw_ned[0])
        q_ned /= norm(q_ned)
        LBN = eci_to_body_quaternion(q_ned)
        if state.time < 50.0:
            self.yaw_cmd = np.rad2deg(state.yaw_ned[0])
                
        LNE = ecef_to_ned_matrix(self.lat_ref, self.lon_ref)
        vn = np.matmul(LNE, ve)  # velocity in NED frame, m/s
        V = norm(vn)

        fs_i = np.matmul(LIB, fs_b)
        Fs_i = norm(fs_i)
        self.tau = v_e / Fs_i

        if state.time < 10.0:
            # initialization
            if not self.initialized:
                self.pi_pred = pi
                self.vi_pred = vi
                self._end_state_correction(state.time, pi, vi)
                self.is_pi_dsrd_computed = False
                self.initialized = True
            else:
                self.v_go = np.add(self.v_go, -fs_i*self.dt_bc)
            
            # time-to-go
            self._time_to_go(state.time, v_e)

            # thrust integrals
            S, J, Q, H, P, J_L, q_ = self._integrals(v_e)

            # turn rate
            u_lmb, d_lmb_dt = self._turn_rate(S, Q, J_L, pi, vi)

            # end-state prediction
            self._end_state_predictor(d_lmb_dt, u_lmb, self.L, S, J, Q, H, P, J_L, q_, pi, vi)

            # end-state correction
            self._end_state_correction(state.time, pi, vi)

        ## Proportional Navigation
        # relative position
        self.pn_dsrd[0, 0] = np.abs(self.pn_dsrd[0, 0]) #+ vn[0, 0]*self.dt
        self.pn_dsrd[1, 0] = np.abs(self.pn_dsrd[1, 0]) #+ vn[1, 0]*self.dt
        pn = eci_to_ned(pi[0, 0], pi[1, 0], pi[2, 0], self.lat_ref, self.lon_ref, self.alt_ref, state.time).reshape((3, 1))  # current inertial position in NED, m
        ptb = np.add(self.pn_dsrd, -pn)  # relative position in NED, m

        # relative velocity
        vtb = -vn
        Vtb = norm(vtb)

        omega = np.cross(ptb.ravel(), vtb.ravel()) / np.dot(ptb.ravel(), ptb.ravel())
        an = (-self.N*Vtb / V)*np.cross(vn.ravel(), omega.ravel())
        ab = np.matmul(LBN, an.reshape((3, 1)))
        self.al_cmd = ab[1][0]
        self.an_cmd = -ab[2][0]

        # structural total acceleration limiter
        acc = sqrt(self.al_cmd**2 + self.an_cmd**2)
        if acc > self.max_acc:
            acc = self.max_acc
        
            phi = atan2(self.an_cmd, self.al_cmd)
            if abs(self.an_cmd) < 1e-7 and abs(self.al_cmd) < 1e-7:
                phi = 0.0
            self.an_cmd = acc*sin(phi)
            self.al_cmd = acc*cos(phi)

    def _gamma_commanded(self, alt, dt):
        if self.prev_alt is not None and dt > 0:
            self.alt_rate = (alt - self.prev_alt) / dt
        self.prev_alt = alt

        alt_error = (self.alt_target - alt)
        gamma_p = self.gain_alt*alt_error
        gamma_d = -self.gain_alt_rate*self.alt_rate 
        self.gamma_cmd = np.deg2rad(gamma_p + gamma_d)
            
    def _velocity_alignment_guidance(self, state: VehicleState):
        """
        Calculate pitch and yaw commands to align vehicle with velocity vector.
        This minimizes angle of attack and sideslip angle.
        """
        lat, lon, _ = state.position_geodetic
        
        q = state.quaternion
        LBI = eci_to_body_quaternion(q)
        LIB = np.transpose(LBI)
        v_ecef = np.matmul(LIB, state.velocity_body)
        LNE = ecef_to_ned_matrix(lat, lon)
        v_ned = np.matmul(LNE, v_ecef)
        vn, ve, vd = v_ned[0, 0], v_ned[1, 0], v_ned[2, 0]
        
        v_horizontal = np.sqrt(vn**2 + ve**2)
        gamma_velocity = np.arctan2(vd, v_horizontal)
        
        heading = np.arctan2(ve, vn) 
        if heading > np.pi:
            heading -= 2*np.pi
        
        # Command vehicle to align with velocity vector
        self.pitch_cmd = -np.rad2deg(gamma_velocity)
        self.yaw_cmd = np.rad2deg(heading)

    def set_dt(self, dt):
        self.dt = dt
            
    @performance_decorator.time_execution_stats
    def step(self, launching_ref: LaunchReference, state: VehicleState, phase, fs_b, v_e):
        """
        Updates the guidance step based on the current state, target state, and velocity.

        Args:
            x (numpy.ndarray): Current state vector.
            fs_b (numpy.ndarray or None): Specific force vector in the body frame.
            v_e (numpy.ndarray): Exhaust velocity.

        Returns:
            numpy.ndarray: Commanded thrust vector in the body frame.
        """
        if not self._enabled:
            return (
                np.float64(0.0),
                np.float64(0.0),
                np.zeros((3, 1), dtype=np.float64),
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
            )
        
        self.lat_ref = launching_ref.lat
        self.lon_ref = launching_ref.lon
        self.alt_ref = launching_ref.alt
        
        if fs_b is not None and state.time > 0.3:
            if phase == 1:
                if state.time < self.transition_time:
                    self._proportional_navigation(state, fs_b, v_e)
                else:
                    self._velocity_alignment_guidance(state)
                    self.an_cmd, self.al_cmd = 0.0, 0.0
            else:
                _, _, alt = state.position_geodetic
                self._gamma_commanded(alt, self.dt_bc)
                self._velocity_alignment_guidance(state)
        return self.an_cmd, self.al_cmd, self.pn_dsrd, self.gamma_cmd, self.pitch_cmd, self.yaw_cmd
