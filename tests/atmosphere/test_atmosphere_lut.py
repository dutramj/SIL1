from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vahsimulator.atmosphere import AtmosphereData, AtmosphereLUT
from vahsimulator.interpolate import LinearInterpolator1D


def _create_atmosphere_dataframe() -> pd.DataFrame:
    """Cria uma tabela atmosférica simples para os testes.

    Returns
    -------
    pandas.DataFrame
        Perfil atmosférico com duas altitudes.
    """

    return pd.DataFrame(
        {
            "altitude_m": [0.0, 1000.0],
            "temperature_K": [300.0, 280.0],
            "pressure_Pa": [100000.0, 90000.0],
            "density_kg_m3": [1.2, 1.0],
            "speed_of_sound_m_s": [347.0, 335.0],
        }
    )


def test_atmosphere_lut_converts_dataframe_to_interpolator() -> None:
    """Verifica a conversão automática de DataFrame para interpolador.

    Raises
    ------
    AssertionError
        Caso o ``DataFrame`` não seja convertido para
        ``LinearInterpolator1D``.
    """

    dataframe = _create_atmosphere_dataframe()

    atmosphere = AtmosphereLUT(interpolator=dataframe)

    assert isinstance(atmosphere.interpolator, LinearInterpolator1D)


def test_atmosphere_lut_accepts_existing_interpolator() -> None:
    """Verifica que um interpolador existente é aceito diretamente.

    Raises
    ------
    AssertionError
        Caso a instância fornecida seja substituída ou rejeitada.
    """

    dataframe = _create_atmosphere_dataframe()

    interpolator = LinearInterpolator1D.from_df(dataframe, key="altitude_m")

    atmosphere = AtmosphereLUT(interpolator=interpolator)

    assert atmosphere.interpolator is interpolator


@pytest.mark.parametrize(
    "altitude_m, expected_temperature_K, expected_pressure_Pa",
    [(0.0, 300.0, 100000.0), (1000.0, 280.0, 90000.0)],
)
def test_atmosphere_lut_returns_table_values(
    altitude_m: float, expected_temperature_K: float, expected_pressure_Pa: float
) -> None:
    """Verifica os valores do LUT nos pontos da tabela.

    Parameters
    ----------
    altitude_m : float
        Altitude de avaliação [m].
    expected_temperature_K : float
        Temperatura esperada [K].
    expected_pressure_Pa : float
        Pressão esperada [Pa].

    Raises
    ------
    AssertionError
        Caso os valores retornados sejam diferentes dos valores
        armazenados na tabela.
    """

    atmosphere = AtmosphereLUT(interpolator=_create_atmosphere_dataframe())

    result = atmosphere.evaluate(altitude_m)

    assert isinstance(result, AtmosphereData)
    assert result.temperature_K == pytest.approx(expected_temperature_K)
    assert result.pressure_Pa == pytest.approx(expected_pressure_Pa)


def test_atmosphere_lut_interpolates_between_table_points() -> None:
    """Verifica a interpolação linear entre dois pontos da tabela.

    Para altitude de 500 m, os valores esperados são exatamente a média
    dos valores correspondentes a 0 m e 1000 m.

    Raises
    ------
    AssertionError
        Caso a interpolação não produza os valores esperados.
    """

    atmosphere = AtmosphereLUT(interpolator=_create_atmosphere_dataframe())

    result = atmosphere.evaluate(500.0)

    assert result.temperature_K == pytest.approx(290.0)
    assert result.pressure_Pa == pytest.approx(95000.0)
    assert result.density_kg_m3 == pytest.approx(1.1)
    assert result.speed_of_sound_m_s == pytest.approx(341.0)


def test_atmosphere_lut_returns_numpy_scalar_values() -> None:
    """Verifica que os valores retornados são escalares numéricos.

    O teste evita que o ``AtmosphereData`` receba acidentalmente arrays
    de dimensão um provenientes do interpolador.

    Raises
    ------
    AssertionError
        Caso alguma propriedade retornada não seja escalar.
    """

    atmosphere = AtmosphereLUT(interpolator=_create_atmosphere_dataframe())

    result = atmosphere.evaluate(500.0)

    assert np.ndim(result.temperature_K) == 0
    assert np.ndim(result.pressure_Pa) == 0
    assert np.ndim(result.density_kg_m3) == 0
    assert np.ndim(result.speed_of_sound_m_s) == 0
