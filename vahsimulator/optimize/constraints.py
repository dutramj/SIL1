# Python standard libraries
from abc import ABC, abstractmethod
import logging

# 3rd party libraries
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Literal, Union, ClassVar, Annotated


logger = logging.getLogger(__name__)


class BaseConstraint(BaseModel, ABC):

    model_config = {"arbitrary_types_allowed": True}

    _FAILURE_PENALTY: ClassVar[np.float64] = np.float64(1e12)

    def evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        g_list = self._evaluate(df)

        for i, g in enumerate(g_list):
            if not np.isfinite(g):
                g_list[i] = self._FAILURE_PENALTY

        return g_list

    @abstractmethod
    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:
        ...


class FinalGammaConstraint(BaseConstraint):

    type: Literal["final_gamma"]
    count: ClassVar[int] = 2
    target: np.float64
    tolerance: np.float64

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        gamma_f = df['FLIGHT_PATH_ANGLE__deg'].iloc[-1]
        logger.debug(f"gamma_f: {gamma_f}")

        lower_bound = self.target - self.tolerance
        upper_bound = self.target + self.tolerance

        g1 = lower_bound - gamma_f
        g2 = gamma_f - upper_bound

        return [g1, g2]


class FinalMachConstraint(BaseConstraint):

    type: Literal["final_mach"]
    count: ClassVar[int] = 2
    lower_bound: np.float64
    upper_bound: np.float64

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        mach_f = df['MACH_NUMBER'].iloc[-1]
        logger.debug(f"mach_f: {mach_f}")

        g1 = self.lower_bound - mach_f
        g2 = mach_f - self.upper_bound

        return [g1, g2]


class AltitudeEnvelopeConstraint(BaseConstraint):

    type: Literal["altitude_envelope"]
    count: ClassVar[int] = 2
    target: np.float64
    tolerance: np.float64

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        alt_f = df['ALTITUDE__m'].iloc[-1]
        logger.debug(f"alt_f: {alt_f}")

        lower_bound = self.target - self.tolerance
        upper_bound = self.target + self.tolerance

        g1 = lower_bound - alt_f
        g2 = alt_f - upper_bound

        return [g1, g2]


class FairingOpeningConstraint(BaseConstraint):

    type: Literal["fairing_opening_conditions"]
    count: ClassVar[int] = 1
    max_q: Annotated[np.float64, Field(gt=0.0)]

    def _evaluate(self, df: pd.DataFrame) -> list[np.float64]:

        apogee_idx = df['ALTITUDE__m'].idxmax()
        Q_apogee = df["DYNAMIC_PRESSURE__Pa"][apogee_idx]
        g = Q_apogee - self.max_q

        logger.debug(f"Q_apogee: {Q_apogee}")

        return [g]


OptimizationConstraint = Union[
    FinalGammaConstraint,
    FinalMachConstraint,
    AltitudeEnvelopeConstraint,
    FairingOpeningConstraint,
]
