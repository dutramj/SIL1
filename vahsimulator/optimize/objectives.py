# Python standard libraries
from abc import ABC, abstractmethod
import logging

# 3rd party libraries
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing_extensions import Literal, Union, ClassVar


logger = logging.getLogger(__name__)


class BaseObjective(BaseModel, ABC):

    model_config = {"arbitrary_types_allowed": True}

    _FAILURE_PENALTY: ClassVar[np.float64] = np.float64(1e12)

    def evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        f_list = self._evaluate(df)

        for i, f in enumerate(f_list):
            if not np.isfinite(f):
                f_list[i] = self._FAILURE_PENALTY

        return f_list

    @abstractmethod
    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:
        ...


class ApogeeObjective(BaseObjective):

    type: Literal["apogee"]
    target: np.float64

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        apogee = df['ALTITUDE__m'].max()
        logger.debug(f"apogee: {apogee}")

        f = (apogee - self.target) ** 2

        return [f]


class MaxFinalAltitudeObjective(BaseObjective):

    type: Literal["max_final_altitude"]

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        max_alt_f = df['ALTITUDE__m'].iloc[-1]
        logger.debug(f"max_alt_f: {max_alt_f}")

        f = -(max_alt_f) ** 2

        return [f]


class MaxFinalMachObjective(BaseObjective):

    type: Literal["max_final_mach"]

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        mach = df['MACH_NUMBER'].iloc[-1]
        logger.debug(f"mach_f: {mach}")

        f = -(mach) ** 2

        return [f]


class TargetFinalMachObjective(BaseObjective):

    type: Literal["target_final_mach"]
    target: np.float64

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        mach_f = df['MACH_NUMBER'].iloc[-1]
        logger.debug(f"mach_f: {mach_f}")

        f = (mach_f - self.target) ** 2

        return [f]

class MaxFinalMachAndAltitudeObjective(BaseObjective):

    type: Literal["max_final_mach_and_altitude"]
    mach_target: np.float64
    alt_target: np.float64
    mach_weight : np.float64
    alt_weight: np.float64

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        mach_f = df['MACH_NUMBER'].iloc[-1]
        alt_f = df['ALTITUDE__m'].iloc[-1]

        f = -(self.mach_weight * ((mach_f / self.mach_target - 1) ** 2) +  self.alt_weight * ((alt_f / self.alt_target - 1) ** 2))

        logger.debug(f"Cost Function: {f:6.3f}")

        return [f]

OptimizationObjective = Union[
    ApogeeObjective,
    MaxFinalAltitudeObjective,
    MaxFinalMachObjective,
    TargetFinalMachObjective,
    MaxFinalMachAndAltitudeObjective,
]
