# Environment variables
# import os
# os.environ['NUMBA_DEBUG_CACHE'] = '1'
# os.environ['NUMBA_DISABLE_JIT'] = '1'

# Python standard libraries
from pathlib import Path
import logging
from multiprocessing import Process
import time
import sys
from datetime import datetime

# 3rd party libraries
import pandas as pd
import colorama

# VAHSimulator library
import vahsimulator as vah
from vahsimulator.utils import clean_pycache


class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def write_csv(df: pd.DataFrame, filepath: Path):
    time_start = time.perf_counter()
    df.to_csv(filepath, index=False)
    print(f"Writing to CSV took {time.perf_counter()-time_start:.4g} seconds")

def main():
    
    MISSION_FILE = "rato_optimize_skip_glide.yaml"
    RESULTS_FILE = "simulation_data_opt_"
    LOG_FILE_PREFIX = "optimization_log_"

    # ------------------------------------------------------------------
    # Console log
    # ------------------------------------------------------------------
    records_dir = Path.cwd() / "records"
    records_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_log = records_dir / f"{RESULTS_FILE}{timestamp}.csv"
    console_log = records_dir / f"{LOG_FILE_PREFIX}{timestamp}.log"

    csv_file = (results_log).resolve()

    log_fp = open(console_log, "w", encoding="utf-8")

    # Duplicate stdout/stderr
    sys.stdout = Tee(sys.__stdout__, log_fp)
    sys.stderr = Tee(sys.__stderr__, log_fp)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,   # ensures any previous logging config is replaced
    )

    logging.getLogger("numba").setLevel(logging.WARNING)

    print(f"Console log saved to: {console_log}")

    try:

        print("*** Building optimization ***")

        mission_file = (Path.cwd() / "mission" / MISSION_FILE).resolve()

        time_start = time.perf_counter()

        mission_config = vah.load_config(mission_file=mission_file)

        simulator = vah.Simulator(**mission_config["simulator"])

        # Loads MissionPlan while registering optimization variables
        opt_variables_registry = []

        mission_plan = vah.MissionPlan.model_validate(
            mission_config["mission_plan"],
            context={"opt_variables_registry": opt_variables_registry},
        )

        trajectory_optimizer = vah.TrajectoryOptimizer(
            simulator=simulator,
            mission_plan=mission_plan,
            opt_variables_registry=opt_variables_registry,
            **mission_config["optimizer"],
        )

        print(
            f"Optimization building took "
            f"{time.perf_counter()-time_start:.4g} seconds"
        )

        print("*** Running optimization ***")

        opt_mission_plan = trajectory_optimizer.evaluate()

        if opt_mission_plan is None:
            print(f"{colorama.Fore.RED}NO OPTIMAL TRAJECTORY FOUND!")
            return

        print("*** Running simulation of optimal trajectory ***")

        sim_df = simulator.run(opt_mission_plan)

        print("*** Writing to CSV file ***")
        p = Process(target=write_csv, args=(sim_df, csv_file))
        p.start()

        print("*** Plotting results ***")

        vah.plot_all(sim_df)

        # Wait for CSV writing to finish
        p.join()

    finally:
        log_fp.close()


if __name__ == "__main__":
    flag = True
    main_directory = Path(__file__).resolve().parent.parent
    clean_pycache(main_directory, flag=flag)
    main()