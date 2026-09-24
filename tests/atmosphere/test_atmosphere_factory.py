from __future__ import annotations

import pandas as pd
import pytest

from vahsimulator.atmosphere import (
    AtmosphereCOESA1976,
    AtmosphereLUT,
    atmosphere_factory,
)


def _create_atmosphere_dataframe() -> pd.DataFrame:
    """Cria uma tabela atmosférica mínima para os testes.

    Returns
    -------
    pandas.DataFrame
        Tabela contendo propriedades atmosféricas em duas altitudes.
    """

    return pd.DataFrame(
        {
            "altitude_m": [0.0, 1000.0],
            "temperature_K": [288.15, 281.65],
            "pressure_Pa": [101325.0, 89874.6],
            "density_kg_m3": [1.225, 1.112],
            "speed_of_sound_m_s": [340.294, 336.434],
        }
    )


def test_atmosphere_factory_creates_lut_model() -> None:
    """Verifica a criação de um modelo atmosférico LUT.

    Raises
    ------
    AssertionError
        Caso a fábrica não retorne uma instância de ``AtmosphereLUT``.
    """

    config = {"type": "lut", "interpolator": _create_atmosphere_dataframe()}

    atmosphere = atmosphere_factory(config)

    assert isinstance(atmosphere, AtmosphereLUT)


def test_atmosphere_factory_creates_coesa1976_model() -> None:
    """Verifica a criação do modelo COESA 1976.

    Raises
    ------
    AssertionError
        Caso a fábrica não retorne uma instância de
        ``AtmosphereCOESA1976``.
    """

    config = {"type": "coesa1976"}

    atmosphere = atmosphere_factory(config)

    assert isinstance(atmosphere, AtmosphereCOESA1976)


def test_atmosphere_factory_raises_for_unknown_model() -> None:
    """Verifica o tratamento de modelos atmosféricos desconhecidos.

    Raises
    ------
    AssertionError
        Caso a fábrica não lance ``ValueError`` para um tipo
        não suportado.
    """

    config = {"type": "unknown_model"}

    with pytest.raises(ValueError, match="Unknown atmosphere model"):
        atmosphere_factory(config)


def test_atmosphere_factory_requires_model_type() -> None:
    """Verifica que o campo ``type`` é obrigatório.

    Raises
    ------
    AssertionError
        Caso a ausência do campo ``type`` não gere ``KeyError``.
    """

    config = {"interpolator": _create_atmosphere_dataframe()}

    with pytest.raises(KeyError):
        atmosphere_factory(config)


def test_atmosphere_factory_ignores_interpolator_for_coesa1976() -> None:
    """Verifica o tratamento de configuração extra para COESA 1976.

    O modelo COESA 1976 não utiliza um interpolador de tabela.
    Ainda assim, uma configuração contendo ``interpolator`` deve continuar
    sendo aceita pela fábrica.

    Raises
    ------
    AssertionError
        Caso a fábrica falhe ao criar o modelo devido à presença do
        campo ``interpolator``.
    """

    config = {"type": "coesa1976", "interpolator": _create_atmosphere_dataframe()}

    atmosphere = atmosphere_factory(config)

    assert isinstance(atmosphere, AtmosphereCOESA1976)
