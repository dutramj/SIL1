# Environment variables
import os
os.environ['VAHSIM_PERFORMANCE_DECORATOR_STATS_ENABLE'] = '1'
# os.environ['NUMBA_DEBUG_CACHE'] = '1'
# os.environ['NUMBA_DISABLE_JIT'] = '1'

# Python standard libraries
from pathlib import Path
import logging
import os

# VAHSimulator library
import vahsimulator as vah


def main():

    MISSION_FILE = 'rato_mission_3dof_point_mass.yaml'
        
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("numba").setLevel(logging.WARNING)

    mission_file = (Path.cwd() / "mission" / MISSION_FILE).resolve()
    mission_config = vah.load_config(mission_file=mission_file)
    simulator = vah.Simulator(**mission_config['simulator'])
    mission_plan = vah.MissionPlan(**mission_config['mission_plan'])
    simulator.run(mission_plan)

if __name__ == "__main__":
    main()
