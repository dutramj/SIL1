import subprocess
import sys
import signal
from math import pi, asin, atan2
import ctypes
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
from numpy.linalg import norm

import vahsimulator
from vahsimulator.parameters import lat_ref, lon_ref, alt_ref, we
from vahsimulator.utils import geodetic_to_eci, euler_to_quaternion, body_to_eci_quaternion, ned_to_ecef_matrix, ecef_to_eci, eci_to_body_quaternion

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)

plot_process = None

def handle_exit(signum, frame):
    global plot_process
    if plot_process is not None:
        try:
            plot_process.terminate()
            plot_process.wait(timeout=2)
        except:
            try:
                plot_process.kill()
            except:
                pass
    print("\nStarting Monte Carlo analysis...")
    subprocess.Popen(["poetry", "run", "py",  ".\\tests\\plot_record_mc.py"], shell=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    sys.exit(0)


def run_single_episode(episode, gyro_params, accel_params, gnss_params, pert_params, nav_params, dt, dt_imu, seed):
    simulator = vahsimulator.Simulator(gyro_params=gyro_params, accel_params=accel_params, gnss_params=gnss_params, 
                    nav_params=nav_params, sim_id=episode, pert_params=pert_params, dt=dt, seed=seed+episode)

    t = 0.0
    phase = 1

    xi, yi, zi = geodetic_to_eci(lat_ref, lon_ref, alt_ref, t)
    
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

    v_body = np.zeros((3, 1))

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

    simulator.set_initial_state(x0, phase, seed=seed+episode)
    
    simulator.run()

    try:
        simulator.recorder.stop()
    except:
        pass
    
    return episode


def run_parallel(episodes, num_threads=4):
    dt = 1.0 / 500.0
    dt_imu = 1.0 / 200.0
    seed = 42

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

    gyro_params = {
        'range': 500.0 * pi / 180.0,
        'misalignment' : 0.1 * pi / 180.0,
        'scale_factor' : 0.01 / 100.0,
        'repeatability' : 3.0 * pi / 180.0 / 3600.0,
        'bias_stability_std' : 3.0 * pi / 180.0 / 3600.0,
        'random_walk' : 0.01 * pi / 180.0 / 60.0,
        'noise_density' : 0.001 * pi / 180.0,
        'bias_stability_tb' : 500.0,
        'min_sample_time': dt_imu,
    }

    accel_params = {
        'range': 500.0 * 9.80665,
        'misalignment' : 0.1 * pi / 180.0,
        'scale_factor' : 0.01 / 100.0,
        'repeatability' : 3.0 * 1e-6 * 9.80665,
        'bias_stability_std' : 3.0 * 1e-6 * 9.80665,
        'random_walk' : 1.0 / 1000.0 / 60.0,
        'noise_density' : 1.0 * 1e-6 * 9.80665,
        'bias_stability_tb' : 500.0,
        'min_sample_time': dt_imu,
    }

    gnss_params = {
        'sample_rate': 10.0,
        'horizontal_std_noise_position': 1.0,
        'vertical_std_noise_position': 1.0,
        'std_noise_velocity': 0.1,
        'std_noise_yaw': 1.0
    }

    nav_params = {
        'std_roll_pitch': 0.0*pi / 180.0,
        'std_yaw': 0.0*pi / 180.0,
        'std_position': 0.0,
        'std_velocity': 0.0
    }

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                run_single_episode, episode, gyro_params, accel_params, gnss_params, pert_params, 
                nav_params, dt, dt_imu, seed
            )
            for episode in range(episodes)
        ]
        
        for future in tqdm(as_completed(futures), total=episodes, desc='Monte Carlo Episodes'):
            try:
                result = future.result()
            except Exception as e:
                print(f"Episode failed with error: {e}")


signal.signal(signal.SIGINT, handle_exit)

if __name__ == "__main__":
    plot_process = subprocess.Popen(["poetry", "run", "py",  ".\\tests\\plots_vah_3D.py"], shell=True)
    episodes = 200
    num_threads = 2
    
    try:
        run_parallel(episodes, num_threads)
    finally:
        try:
            plot_process.terminate()
            plot_process.wait(timeout=5)
        except:
            try:
                plot_process.kill()
            except:
                pass
    
    print("\nStarting Monte Carlo analysis...")
    subprocess.Popen(["poetry", "run", "py",  ".\\tests\\plot_record_mc.py"], shell=True)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
