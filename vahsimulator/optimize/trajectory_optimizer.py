# Python standard libraries
import logging

# 3rd party libraries
from pydantic import BaseModel, field_validator, PrivateAttr
from typing_extensions import Any
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# VAHSimulator library
from .. import performance_decorator
from .constraints import OptimizationConstraint
from ..mission_plan import MissionPlan
from .objectives import OptimizationObjective
from .optimization_variable import OptimizationVariable
from ..simulator import Simulator
from .trajectory_problem import TrajectoryProblem


logger = logging.getLogger(__name__)


class TrajectoryOptimizer(BaseModel):

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
    }

    algorithm: NSGA2
    constraints: list[OptimizationConstraint] = []
    mission_plan: MissionPlan
    n_gen: int = 100
    objectives: list[OptimizationObjective]
    opt_variables_registry: list[OptimizationVariable]
    seed: int = 1
    simulator: Simulator
    
    _trajectory_problem: TrajectoryProblem = PrivateAttr()

    @field_validator("algorithm", mode="before")
    @classmethod
    def load_algorithm(cls, v: Any) -> NSGA2:
        if isinstance(v, dict):
            return NSGA2(**v)
        return v

    def model_post_init(self, __context: Any) -> None:

        extra_kwargs = self.model_extra or {}

        self._trajectory_problem = TrajectoryProblem(
            constraints=self.constraints,
            mission_plan=self.mission_plan,
            objectives=self.objectives,
            opt_variables_registry=self.opt_variables_registry,
            simulator=self.simulator,
            **extra_kwargs,
        )

    @performance_decorator.time_execution
    def evaluate(self) -> MissionPlan | None:

        result = minimize(
            self._trajectory_problem,
            self.algorithm,
            termination=('n_gen', self.n_gen),
            seed=self.seed,
            verbose=True
        )

        opt_variables = result.X

        logger.debug('** OPTIMIZATION RESULTS **')

        if opt_variables is None:
            logger.warning('Optimal trajectory not found!')
            return None

        logger.debug(f'opt_variables: {opt_variables}')

        if opt_variables.ndim == 2:
            logger.warning('More than one solution found! Picking first one of the list')
            opt_variables = opt_variables[0,:]

        OptimizationVariable.set_values(self.opt_variables_registry, opt_variables)

        return self.mission_plan
