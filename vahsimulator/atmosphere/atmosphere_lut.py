from __future__ import annotations

import pandas as pd
from typing import Any

from pydantic import field_validator

from .. import performance_decorator
from ..interpolate import LinearInterpolator1D
from .atmosphere_base import AtmosphereBase
from .atmosphere_data import AtmosphereData


class AtmosphereLUT(AtmosphereBase):
    """Modelo atmosférico baseado em tabela de consulta.

    O modelo utiliza interpolação linear para obter as propriedades
    atmosféricas a partir da altitude.

    Parameters
    ----------
    interpolator : LinearInterpolator1D or pandas.DataFrame
        Interpolador contendo o perfil atmosférico ou DataFrame a
        partir do qual o interpolador será construído.

    Notes
    -----
    O perfil deve possuir as colunas:

    - ``altitude_m``
    - ``temperature_K``
    - ``pressure_Pa``
    - ``density_kg_m3``
    - ``speed_of_sound_m_s``
    """

    interpolator: LinearInterpolator1D

    @field_validator("interpolator", mode="before")
    @classmethod
    def df_to_interpolator(cls, value: Any) -> LinearInterpolator1D:
        """Converte um DataFrame em interpolador linear.

        Parameters
        ----------
        value : Any
            DataFrame contendo o perfil atmosférico ou um
            ``LinearInterpolator1D`` já construído.

        Returns
        -------
        LinearInterpolator1D
            Interpolador pronto para utilização.
        """

        if isinstance(value, pd.DataFrame):
            return LinearInterpolator1D.from_df(value, key="altitude_m")

        return value

    @performance_decorator.time_execution_stats
    def evaluate(self, altitude_m: float) -> AtmosphereData:
        """Avalia as propriedades atmosféricas em uma altitude.

        Parameters
        ----------
        altitude_m : float
            Altitude do veículo [m].

        Returns
        -------
        AtmosphereData
            Propriedades atmosféricas interpoladas.
        """

        result = self.interpolator.evaluate([altitude_m])

        return AtmosphereData(
            temperature_K=result[0, 0],
            pressure_Pa=result[0, 1],
            density_kg_m3=result[0, 2],
            speed_of_sound_m_s=result[0, 3],
        )
