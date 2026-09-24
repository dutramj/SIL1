# ZIPFEL, Peter H. Modeling and simulation of aerospace vehicle dynamics. AIAA, 2007.

# 3rd party libraries
import numpy as np
from numpy.linalg import inv, norm
from numpy import sqrt, atan2

# VAHSimulator library
from .utils import (
    ecef_to_ned_matrix,
    eci_to_body_quaternion
)
from .vehicle_state import VehicleState
from .aerodynamics.aero_state import AeroState


def control_heading(state: VehicleState, grav_acc, V, zeta_rc, wn_rc, heading_cmd, heading_factor):
    """
    Heading hold control - generates roll angle command.
    
    Args:
        x: state vector
        grav_mag: magnitude of gravitational acceleration (m/s²)
        V: true airspeed (m/s)
        
    Returns:
        phi_cmd: roll angle command (radians)
    """
    lat, lon, _ = state.position_geodetic

    q = state.quaternion
    LBI = eci_to_body_quaternion(q)
    LIB = np.transpose(LBI)
    v_ecef = np.matmul(LIB, state.velocity_body)
    
    LNE = ecef_to_ned_matrix(lat, lon)
    v_ned = np.matmul(LNE, v_ecef)
    
    heading = atan2(v_ned[1, 0], v_ned[0, 0])
    if heading > np.pi:
        heading -= 2*np.pi
    
    grav_mag = norm(grav_acc)
    Kpsi = (V / grav_mag)*zeta_rc*wn_rc*(1.0 - zeta_rc**2)*(1.0 + heading_factor)
    phi_cmd = Kpsi*(heading_cmd - heading)

    return phi_cmd

def control_roll(state: VehicleState, aero_state: AeroState, roll_cmd, max_delta):

    # gains
    Kp = 0.7 # (2*zeta_rc*wn_rc + LLp) / LLda
    Kphi = 1.0 # (wn_rc**2) / LLda

    # roll position error
    error_roll = Kphi*(roll_cmd - state.roll_ned[0])
    delta_p = np.rad2deg(error_roll - Kp*state.p[0])
    delta_p = np.clip(delta_p, -max_delta, max_delta)

    control_factor = np.interp(aero_state.Q, [1000., 2000.], [0.0, 1.0])

    return delta_p * control_factor

def control_rate(state: VehicleState, aero_state: AeroState, pitch_cmd, yaw_cmd, max_delta):
    K_p_pitch = 5.0
    K_d_pitch = 0.3
    K_p_yaw = 2.0
    K_d_yaw = 0.2    

    error_pitch = np.deg2rad(pitch_cmd) - state.pitch_ned[0]
    delta_q = K_p_pitch * error_pitch - K_d_pitch * state.q[0]
    delta_q = np.clip(delta_q, -max_delta, max_delta)
    
    error_yaw = np.deg2rad(yaw_cmd) - state.yaw_ned[0]
    delta_r = K_p_yaw * error_yaw - K_d_yaw * state.r[0]
    delta_r = np.clip(delta_r, -max_delta, max_delta)

    control_factor = np.interp(aero_state.Q, [1000., 2000.], [0.0, 1.0])

    delta_q *= control_factor
    delta_r *= control_factor    

    return delta_q, delta_r

def control_gamma(gamma_cmd, state: VehicleState, q, V, Q, Na, Nd, Ma, Mq, Md, max_delta):

    if V < 1.0:
        V = 1.0
    
    if abs(Md) < 1e-6:
        Md = np.sign(Md)*1e-6
    
    lat, lon, _ = state.position_geodetic
    
    q_ = state.quaternion

    LBI = eci_to_body_quaternion(q_)
    LIB = np.transpose(LBI)
    v_ecef = np.matmul(LIB, state.velocity_body)

    LNE = ecef_to_ned_matrix(lat, lon)
    v_ned = np.matmul(LNE, v_ecef)

    denom = sqrt(v_ned[0, 0]**2 + v_ned[1, 0]**2)
    if denom > 1e-10:
        gamma = atan2(-v_ned[2, 0], denom)
    else:
        if v_ned[2, 0] > 0:
            gamma = -np.pi / 2.0
        elif v_ned[2, 0] < 0:
            gamma = np.pi / 2.0
        else:
            gamma = 0.0

    wgam = max(3.0, 25.12 + 0.5e-5*Q)
    pgam = max(1.0, 10.0 + 1.0e-5*Q)
    zgam = 0.9

    # compute c_
    am = 2.0*zgam*wgam + pgam
    bm = wgam**2 + 2.0*zgam*wgam*pgam
    cm = wgam**2*pgam
    
    DP = np.array([[Md, 0.0, Nd / V],
                    [Md*Na / V - Nd*Ma / V, Md, -Mq*Nd / V],
                    [0.0, Md*Na / V - Nd*Ma / V, Md*Na / V - Nd*Ma / V]])
    
    DD = np.array([[am + Mq - Na / V],
                    [bm + Ma + Mq*Na / V],
                    [cm]])

    if abs(np.linalg.det(DP)) < 1e-10:
        delta_e = 0.0
        return delta_e
    
    DPI = inv(DP)
    
    Kgam = np.matmul(DPI, DD)

    # compute k
    AA = np.array([[Mq, Ma, -Ma],
                    [1.0, 0.0, 0.0],
                    [0.0, Na / V, -Na / V]])
    
    BB = np.array([[Md], [0.0], [Nd / V]])
    
    DUM33 = np.add(AA, -np.matmul(BB, np.transpose(Kgam)))
    
    det_DUM33 = np.linalg.det(DUM33)
    if abs(det_DUM33) < 1e-10:
        Kp = 2.0
        delta_e = np.rad2deg(Kp*(gamma_cmd - gamma))
        delta_e = np.clip(delta_e, -max_delta, max_delta)
        return delta_e
    
    IDUM33 = inv(DUM33)
    DUM3 = np.matmul(IDUM33, BB)
    HH = np.array([[0.0], [0.0], [1.0]])
    denom = np.dot(HH.T, DUM3)[0, 0]
    if abs(denom) < 1e-10:
        denom = np.sign(denom)*1e-10
    
    # Feedforward gain
    k = -1.0 / denom    

    # Control law with feedforward and feedback terms
    thtc = k*gamma_cmd
    qqf = Kgam[0, 0]*q
    thtbgf = Kgam[1, 0]*state.pitch_ned[0]
    thtugf = Kgam[2, 0]*gamma
    
    delec = thtc - (qqf + thtbgf + thtugf)
    delta_e = np.rad2deg(delec)
    delta_e = np.clip(delta_e, -max_delta, max_delta)

    return delta_e

def compute_tvc_factor(t, transition_time, action_time):
    """
    Compute smooth blending factor between TVC and canard
    Returns: 0.0 for pure TVC, 1.0 for pure canard
    """
    if t < transition_time:
        return 0.0
    elif t >= action_time:
        return 1.0
    else:
        progress = (t - transition_time) / (action_time - transition_time)
        return progress
