# Python standard libraries
import csv
import time
from math import radians
import subprocess
import ctypes

# 3rd party libraries
import numpy as np

# VAHSimulator library
from vahsimulator.communication import Communication

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)


class SimulationReplay:
    def __init__(self, csv_filepath, sim_id=0):
        self.csv_filepath = csv_filepath
        self.comm = Communication(sim_id=sim_id)
        self.data_rows = []
        self.column_map = {}
        
    def load_data(self):
        print(f"Loading data from {self.csv_filepath}...")
        
        with open(self.csv_filepath, 'r') as file:
            reader = csv.reader(file)
            
            header = next(reader)
            self.column_map = {col: idx for idx, col in enumerate(header)}
            
            for row in reader:
                converted_row = []
                for val in row:
                    if val == '' or val == 'None':
                        converted_row.append(0.0)
                    else:
                        try:
                            converted_row.append(float(val))
                        except ValueError:
                            converted_row.append(0.0)
                self.data_rows.append(converted_row)
        
        print(f"Loaded {len(self.data_rows)} data points")
        
    def get_column_value(self, row, column_name):
        if column_name in self.column_map:
            return row[self.column_map[column_name]]
        else:
            return 0.0
            
    def extract_state_vectors(self, row):
        x = np.array([
            [self.get_column_value(row, 'STATE_ECI_POS_X__m')],      # xi
            [self.get_column_value(row, 'STATE_ECI_POS_Y__m')],      # yi  
            [self.get_column_value(row, 'STATE_ECI_POS_Z__m')],      # zi
            [self.get_column_value(row, 'STATE_QUAT_W')],            # qw
            [self.get_column_value(row, 'STATE_QUAT_X')],            # qx
            [self.get_column_value(row, 'STATE_QUAT_Y')],            # qy
            [self.get_column_value(row, 'STATE_QUAT_Z')],            # qz
            [self.get_column_value(row, 'STATE_BODY_VEL_U__m_s')],   # u
            [self.get_column_value(row, 'STATE_BODY_VEL_V__m_s')],   # v
            [self.get_column_value(row, 'STATE_BODY_VEL_W__m_s')],   # w
            [self.get_column_value(row, 'STATE_BODY_RATE_P__rad_s')], # p
            [self.get_column_value(row, 'STATE_BODY_RATE_Q__rad_s')], # q
            [self.get_column_value(row, 'STATE_BODY_RATE_R__rad_s')]  # r
        ])
        
        x_nav = np.array([
            [self.get_column_value(row, 'NAV_QUAT_W')],              # qw
            [self.get_column_value(row, 'NAV_QUAT_X')],              # qx
            [self.get_column_value(row, 'NAV_QUAT_Y')],              # qy
            [self.get_column_value(row, 'NAV_QUAT_Z')],              # qz
            [self.get_column_value(row, 'NAV_POS_ECI_X__m')],        # xi
            [self.get_column_value(row, 'NAV_POS_ECI_Y__m')],        # yi
            [self.get_column_value(row, 'NAV_POS_ECI_Z__m')],        # zi
            [self.get_column_value(row, 'NAV_VEL_ECI_X__m_s')],      # vxi
            [self.get_column_value(row, 'NAV_VEL_ECI_Y__m_s')],      # vyi
            [self.get_column_value(row, 'NAV_VEL_ECI_Z__m_s')],      # vzi
            [self.get_column_value(row, 'NAV_ACC_BIAS_X__m_s2')],    # bax
            [self.get_column_value(row, 'NAV_ACC_BIAS_Y__m_s2')],    # bay
            [self.get_column_value(row, 'NAV_ACC_BIAS_Z__m_s2')],    # baz
            [self.get_column_value(row, 'NAV_GYRO_BIAS_X__rad_s')],  # bgx
            [self.get_column_value(row, 'NAV_GYRO_BIAS_Y__rad_s')],  # bgy
            [self.get_column_value(row, 'NAV_GYRO_BIAS_Z__rad_s')]   # bgz
        ])
        
        pn_pred = np.array([
            [self.get_column_value(row, 'TARGET_POS_NED_X__m')],
            [self.get_column_value(row, 'TARGET_POS_NED_Y__m')],
            [self.get_column_value(row, 'TARGET_POS_NED_Z__m')]
        ])
        
        return x, x_nav, pn_pred
        
    def replay(self, time_scale=1.0):
        if not self.data_rows:
            print("No data loaded. Call load_data() first.")
            return
            
        print(f"Starting replay with time scale: {time_scale}x")
        print("Press Ctrl+C to stop replay")
        
        start_real_time = time.time()
        start_sim_time = None
        
        try:
            for i, row in enumerate(self.data_rows):
                t = self.get_column_value(row, 'TIME__s')
                
                if start_sim_time is None:
                    start_sim_time = t
                
                sim_elapsed = t - start_sim_time
                target_real_elapsed = sim_elapsed / time_scale
                
                while time.time() - start_real_time < target_real_elapsed:
                    continue
                
                x, x_nav, pn_pred = self.extract_state_vectors(row)
                
                phase = int(self.get_column_value(row, 'PHASE'))
                
                # TVC angles
                By = radians(self.get_column_value(row, 'TVC_DELTA_R__deg'))
                Bz = radians(self.get_column_value(row, 'TVC_DELTA_Q__deg'))
                
                # Canard deflections
                delta_1 = self.get_column_value(row, 'CANARD_DELTA_1__deg')
                delta_2 = self.get_column_value(row, 'CANARD_DELTA_2__deg')
                delta_3 = self.get_column_value(row, 'CANARD_DELTA_3__deg')
                delta_4 = self.get_column_value(row, 'CANARD_DELTA_4__deg')
                
                # Fin deflections
                delta_left = self.get_column_value(row, 'FIN_DELTA_LEFT__deg')
                delta_right = self.get_column_value(row, 'FIN_DELTA_RIGHT__deg')
                
                # Aerodynamic parameters
                alpha = self.get_column_value(row, 'ANGLE_OF_ATTACK__deg')
                beta = self.get_column_value(row, 'SIDESLIP_ANGLE__deg')
                mach = self.get_column_value(row, 'MACH_NUMBER')
                
                # Thrust
                Th = self.get_column_value(row, 'THRUST__N')
                
                self.comm.step(t, phase, x, x_nav, By, Bz, delta_1, delta_2, delta_3, delta_4, delta_left, delta_right, alpha, beta, mach, pn_pred, Th)
                
                if i % 100 == 0:
                    alt = self.get_column_value(row, 'ALTITUDE__m')
                    print(f"t={t:.3f}s, alt={alt:.1f}m, step={i+1}/{len(self.data_rows)}, phase: {phase}", end='\r')
                    
        except KeyboardInterrupt:
            print("\nReplay stopped by user")
        except Exception as e:
            print(f"Error during replay: {e}")
        finally:
            self.comm.stop()
            print("Replay finished")


def main():
    import os
    
    csv_path = os.path.join('records', 'simulation_data_0.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please make sure simulation_data_01.csv exists in the records folder")
        return
    
    replay = SimulationReplay(csv_path, sim_id=1)
    replay.load_data()
    replay.replay(time_scale=1.0)


subprocess.Popen(["poetry", "run", "py",  ".\\tests\\plots_vah_3D.py"], shell=True)

if __name__ == "__main__":
    for i in range(1):
        main()

ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)