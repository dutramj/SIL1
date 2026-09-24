# Environment variables
# import os
# os.environ['NUMBA_DEBUG_CACHE'] = '1'
# os.environ['NUMBA_DISABLE_JIT'] = '1'

# Python standard libraries
from pathlib import Path
import logging
from multiprocessing import Process
import time
import pandas as pd

# VAHSimulator library
import vahsimulator as vah
from vahsimulator.utils import clean_pycache

def write_csv(df: pd.DataFrame, filepath: Path):
    time_start = time.perf_counter()
    df.to_csv(filepath, index=False)
    print(f"Writing to CSV took {time.perf_counter()-time_start:.4g} seconds")


def main():

    MISSION_FILE = 'rato_mission_3dof_skip_glide.yaml'
    LOG_FILE = "simulation_data_0.csv"
        
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("numba").setLevel(logging.WARNING)

    print("*** Building simulator ***")
    mission_file = (Path.cwd() / "mission" / MISSION_FILE).resolve()
    time_start = time.perf_counter()
    mission_config = vah.load_config(mission_file=mission_file)
    simulator = vah.Simulator(**mission_config['simulator'])
    mission_plan = vah.MissionPlan(**mission_config['mission_plan'])
    print(f"Simulator building took {time.perf_counter()-time_start:.4g} seconds")

    print("*** Running simulation ***")
    sim_df = simulator.run(mission_plan)

    print("*** Writing to CSV file ***")
    log_file = (Path.cwd() / "records" / LOG_FILE).resolve()
    p = Process(target=write_csv, args=(sim_df, log_file))
    p.start()

    print("*** Plotting results ***")
    vah.plot_all(sim_df)


if __name__ == "__main__":
    flag = True
    main_directory = Path(__file__).resolve().parent.parent
    # clean_pycache(main_directory, flag=flag)
    main()
