# Python standard libraries
import subprocess
import sys
import signal
from math import pi, atan2, asin
import ctypes
import logging

# 3rd party libraries
import numpy as np
from numpy.linalg import norm

# VAHSimulator library
import vahsimulator
from vahsimulator.parameters import lat_ref, lon_ref, alt_ref, we
from vahsimulator.utils import euler_to_quaternion, geodetic_to_eci, eci_to_body_quaternion, body_to_eci_quaternion, ned_to_ecef_matrix, ecef_to_eci


logging.basicConfig(level=logging.DEBUG)

# Prevent system sleep
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)

SEED = 42
simulator = None


def handle_exit(signum, frame):
    global simulator
    try:
        simulator.recorder.stop()
    except:
        pass
    print()
    print("\nStarting Monte Carlo analysis...")
    subprocess.Popen(["poetry", "run", "py", ".\\tests\\plot_record.py"], shell=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    sys.exit(0)


def run():
    global simulator, SEED

    # IMU parameters
    dt = 1.0 / 200.0
    dt_imu = 1.0 / 200.0

    gyro_params = {
        'range': 500.0*pi / 180.0,  # °/s to rad/s
        # 'misalignment' : 0.01*pi / 180.0,  # ° to rad
        # 'scale_factor' : 0.1 / 100.0,  # % to fraction
        # 'repeatability' : 0.1*pi / 180.0,  # °/s to rad/s
        'bias_stability_std' : 1.0*pi / 180.0 / 3600.0,  # °/h to rad/s
        # 'random_walk' : 1.0e-07,  # rad/s^(3/2)
        # 'g_dependent_bias' : 0.001*pi / 180.0 / 1 / 9.80665,  # # °/s/g to rad/s/m/s²
        'noise_density' : 0.0001*pi / 180.0,  # °/s/√Hz to rad/s/√Hz
        'bias_stability_tb' : 100,  # s
        'min_sample_time': dt_imu,  # s
    }

    accel_params = {
        'range': 50.0*9.80665,  # g to m/s²
        # 'misalignment' : 0.01*pi / 180.0,  # ° to rad
        # 'scale_factor' : 0.1 / 100.0,  # % to fraction
        # 'repeatability' : 1.0*1e-3*9.80665,  # mg to m/s²
        'bias_stability_std' : 1.0*1e-6*9.80665,  # µg to m/s²
        # 'random_walk' : 1.0e-07,  # m/s^(5/2)
        'noise_density' : 100*1e-6*9.80665,  # µg/√Hz to m√s/s³
        'bias_stability_tb' : 500,  # s
        'min_sample_time': dt_imu,  # s
    }

    # Perturbations parameters
    pert_params = {
        'fin_misalignment': 0.0,  # °
        'canard_misalignment': 0.0,  # °
        'fin_14x_misalignment': 0.0,  # °
        'drag': 0.0,  # %
        'propellant_weight': 0.0,  # %
        'inert_weight': 0.0,  # %
        'payload_weight': 0.0,  # %
        'thrust_misalignment': 0.0,  # °
        'cg_offset': 0.0,  # m
        'thrust_vector_x_offset': 0.0, # m
        'thrust_vector_y_offset': 0.0, # m
        'thrust_vector_z_offset': 0.0, # m
    }

    # GNSS parameters
    gnss_params = {
        'sample_rate': 10.0,  # Hz
        'horizontal_std_noise_position': 1.0,  # m
        'vertical_std_noise_position': 1.0,  # m
        'std_noise_velocity': 0.1,  # m/s
        'std_noise_yaw': 1.0  # °
    }

    # Navigation parameters
    nav_params = {
        'std_roll_pitch': 0.0*pi / 180.0,  # ° to rad 
        'std_yaw': 0.0*pi / 180.0,  # ° to rad
        'std_position': 0.0,  # m
        'std_velocity': 0.0  # m/s
    }

    simulator = vahsimulator.Simulator(gyro_params=gyro_params, accel_params=accel_params, gnss_params=gnss_params, nav_params=nav_params, pert_params=pert_params, sim_id=0, dt=dt, seed=SEED)
    
    # Initial state
    # Option 1: Define from Euler angles, body velocity, and altitude
    t = 0.0
    phase = 1

    ## position in ECI
    xi, yi, zi = geodetic_to_eci(lat_ref, lon_ref, alt_ref, t)
    
    ## attitude
    roll_ned = np.deg2rad(0.0)
    pitch_ned = np.deg2rad(89.999)  
    yaw_ned = np.deg2rad(0.0)
    quat_ned = euler_to_quaternion(roll_ned, pitch_ned, yaw_ned)
    quat_ned = quat_ned / norm(quat_ned)
    LNB = body_to_eci_quaternion(quat_ned)
    LEN = ned_to_ecef_matrix(lat_ref, lon_ref)
    LIE = ecef_to_eci(0.0)
    LIB = np.matmul(LIE, np.matmul(LEN, LNB))
    
    roll = atan2(LIB[2, 1], LIB[2, 2])
    if abs(LIB[2, 0]) > 1:
        LIB[2, 0] = np.sign(LIB[2, 0])
    pitch = -asin(LIB[2, 0])
    yaw = atan2(LIB[1, 0], LIB[0, 0])

    q = euler_to_quaternion(roll, pitch, yaw)
    q /= norm(q)

    ## velocity in ECEF projected to body frame
    v_body = np.zeros((3, 1))

    ## angular velocity
    LBI = eci_to_body_quaternion(q)
    w_eci = np.array([[0], [0], [we]])
    w_body = np.matmul(LBI, w_eci)

    x0 = {
        'x' : [xi],  # 0: xi
        'y' : [yi],  # 1: yi
        'z' : [zi],  # 2: zi
        'q0' : [q[0, 0]],  # 3: q0
        'q1' : [q[1, 0]],  # 4: q1
        'q2' : [q[2, 0]],  # 5: q2
        'q3' : [q[3, 0]],  # 6: q3
        'u' : [v_body[0, 0]],  # 7: u
        'v' : [v_body[1, 0]],  # 8: v
        'w' : [v_body[2, 0]],  # 9: w
        'p' : [w_body[0, 0]],  # 10: p
        'q' : [w_body[1, 0]],  # 11: q
        'r' : [w_body[2, 0]],  # 12: r
    }

    # # Option 2: Define state directly
    # t = 91.16999999998957  # s
    # phase = 2
    # x0 = np.array([[ 4.67074360e+06],
    #                [-4.41816626e+06],
    #                [-1.76127374e+05],
    #                [ 7.96938733e-01],
    #                [-4.29841494e-01],
    #                [-4.05793156e-01],
    #                [-1.24325621e-01],
    #                [ 2.68770620e+03],
    #                [ 7.97982400e+01],
    #                [ 3.29649453e+01],
    #                [ 1.84146567e-03],
    #                [ 1.29589018e-02],
    #                [-1.15147981e-02]])
    
    # # Option 3: Define from aerodynamic angles, path angle, heading, and Mach        
    # t = 300.0  # s
    # phase = 3
    # alt = 50000.0  # m
    # alpha = -1.0  # deg
    # beta = 1.0  # deg
    # path_angle = -10.0  # deg (gamma)
    # heading = 0.0  # deg (psi)
    # mach = 10.0
    # roll_rate = 0.0  # deg/s
    # pitch_rate = 0.0  # deg/s
    # yaw_rate = 0.0  # deg/s
    
    # ## position in ECI frame
    # xi, yi, zi = geodetic_to_eci(lat_ref, lon_ref, alt, t)
    
    # ## velocity in NED frame from path angles
    # atmosphere = Atmosphere()
    # speed_sound, _, _, _ = atmosphere.step(alt)
    # Va = mach*speed_sound  # true airspeed
    
    # alpha_rad = np.deg2rad(alpha)
    # beta_rad = np.deg2rad(beta)
    # gamma_rad = np.deg2rad(path_angle)
    # psi_rad = np.deg2rad(heading)
    
    # # Velocity in NED frame
    # v_ned = np.array([[Va*np.cos(gamma_rad)*np.cos(psi_rad)],
    #                   [Va*np.cos(gamma_rad)*np.sin(psi_rad)],
    #                   [-Va*np.sin(gamma_rad)]])
    
    # # Wind frame to body frame: pitch = gamma + alpha, yaw = psi, roll = 0 (if beta = 0)
    # theta = gamma_rad + alpha_rad  # pitch in NED
    # phi = np.arcsin(np.sin(beta_rad) / np.cos(alpha_rad)) if abs(np.cos(alpha_rad)) > 1e-6 else 0.0  # roll in NED
    # psi = psi_rad + beta_rad  # yaw in NED
    
    # quat_ned = euler_to_quaternion(phi, theta, psi)
    # quat_ned = quat_ned / norm(quat_ned)
    
    # # Transform to ECI frame
    # LNB = body_to_eci_quaternion(quat_ned)
    # LEN = ned_to_ecef_matrix(lat_ref, lon_ref)
    # LIE = ecef_to_eci(t)
    # LIB = np.matmul(LIE, np.matmul(LEN, LNB))
    
    # roll = atan2(LIB[2, 1], LIB[2, 2])
    # if abs(LIB[2, 0]) > 1:
    #     LIB[2, 0] = np.sign(LIB[2, 0])
    # pitch = -asin(LIB[2, 0])
    # yaw = atan2(LIB[1, 0], LIB[0, 0])
    
    # q = euler_to_quaternion(roll, pitch, yaw)
    # q /= norm(q)
    
    # ## velocity in body frame
    # quat_ned_conj = np.array([[quat_ned[0, 0]], [-quat_ned[1, 0]], [-quat_ned[2, 0]], [-quat_ned[3, 0]]])
    # LBN = body_to_eci_quaternion(quat_ned_conj)
    # v_body = np.matmul(LBN, v_ned)
    
    # ## angular velocity in body frame
    # LBI = eci_to_body_quaternion(q)
    # w_eci = np.array([[0], [0], [we]])
    # w_body = np.matmul(LBI, w_eci)
    # w_body = np.add(w_body, np.array([[np.deg2rad(roll_rate)], [np.deg2rad(pitch_rate)], [np.deg2rad(yaw_rate)]]))
    
    # x0 = np.array([
    #     [xi],  # 0: xi
    #     [yi],  # 1: yi
    #     [zi],  # 2: zi
    #     [q[0, 0]],  # 3: q0
    #     [q[1, 0]],  # 4: q1
    #     [q[2, 0]],  # 5: q2
    #     [q[3, 0]],  # 6: q3
    #     [v_body[0, 0]],  # 7: u
    #     [v_body[1, 0]],  # 8: v
    #     [v_body[2, 0]],  # 9: w
    #     [w_body[0, 0]],  # 10: p
    #     [w_body[1, 0]],  # 11: q
    #     [w_body[2, 0]],  # 12: r
    # ])

    simulator.set_initial_state(x0, phase, seed=SEED)

    simulator.run()
    
    print('\n\n', t, '\n\n', x)
    try:
        print()
        simulator.recorder.stop()
    except:
        pass


subprocess.Popen(["poetry", "run", "py", ".\\tests\\plots_vah_3D.py"], shell=True)

signal.signal(signal.SIGINT, handle_exit)

if __name__ == "__main__":
    run()

try:
    simulator.recorder.stop()
except:
    pass

print("\nStarting Monte Carlo analysis...")
subprocess.Popen(["poetry", "run", "py", ".\\tests\\plot_record.py"], shell=True)

ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
