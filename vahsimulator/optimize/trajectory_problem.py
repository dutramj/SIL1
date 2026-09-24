# Python standard libraries
import logging

# 3rd party libraries
from pymoo.core.problem import Problem
import multiprocessing
from functools import partial
import numpy as np
from typing_extensions import Any

# VAHSimulator library
from .constraints import OptimizationConstraint
from ..mission_plan import MissionPlan
from .objectives import OptimizationObjective
from .optimization_variable import OptimizationVariable
from ..simulator import Simulator


def opt_worker(
        x: Any,
        constraints: list[OptimizationConstraint],
        mission_plan: MissionPlan,
        objectives: list[OptimizationObjective],
        opt_variables_registry: list[OptimizationVariable],
        simulator: Simulator,
    ):

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    OptimizationVariable.set_values(opt_variables_registry, x)
    
    sim_df = simulator.run(mission_plan)
    
    logger.debug(x)
    
    f = []
    for objective in objectives:
        result = objective.evaluate(sim_df)
        f += result
        
    g = []
    for constraint in constraints:
        result = constraint.evaluate(sim_df)
        g += result
        
    return f, g


class TrajectoryProblem(Problem):
    
    constraints: list[OptimizationConstraint]
    mission_plan: MissionPlan
    n_processes: int
    objectives: list[OptimizationObjective]
    opt_variables_registry: list[OptimizationVariable]
    simulator: Simulator

    def __init__(
            self,
            mission_plan: MissionPlan,
            objectives: list[OptimizationObjective],
            opt_variables_registry: list[OptimizationVariable],
            simulator: Simulator,
            constraints: list[OptimizationConstraint] = [],
            n_processes: int = 4,
        ):

        self.constraints = constraints
        self.mission_plan = mission_plan
        self.n_processes = n_processes
        self.objectives = objectives
        self.opt_variables_registry = opt_variables_registry
        self.simulator = simulator

        super().__init__(
            n_var=len(opt_variables_registry),
            n_obj=len(self.objectives),
            n_ieq_constr=sum(constraint.count for constraint in self.constraints),
            xl=OptimizationVariable.lower_bounds(self.opt_variables_registry),
            xu=OptimizationVariable.upper_bounds(self.opt_variables_registry),
        )

    def _evaluate(self, X: Any, out: Any):

        func = partial(
            opt_worker,
            constraints = self.constraints,
            mission_plan=self.mission_plan,
            objectives=self.objectives,
            opt_variables_registry=self.opt_variables_registry,
            simulator=self.simulator,
        )

        with multiprocessing.Pool(self.n_processes) as pool:
            result_obj = pool.map_async(func, X)

            try:
                while not result_obj.ready():
                    result_obj.wait(timeout=0.2)
                results = result_obj.get()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise

        F = []
        G = []
        for f, g in results:
            F.append(f)
            G.append(g)

        out["F"] = np.array(F)
        out["G"] = np.array(G)
