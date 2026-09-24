from __future__ import annotations

import pytest

from vahsimulator.atmosphere import AtmosphereData


def test_atmosphere_data_stores_properties() -> None:
    """Verifica o armazenamento das propriedades atmosféricas.

    O teste garante que os quatro campos do ``AtmosphereData`` são
    armazenados sem alteração dos valores fornecidos.

    Raises
    ------
    AssertionError
        Caso qualquer propriedade seja armazenada incorretamente.
    """

    data = AtmosphereData(
        temperature_K=288.15,
        pressure_Pa=101325.0,
        density_kg_m3=1.225,
        speed_of_sound_m_s=340.294,
    )

    assert data.temperature_K == pytest.approx(288.15)
    assert data.pressure_Pa == pytest.approx(101325.0)
    assert data.density_kg_m3 == pytest.approx(1.225)
    assert data.speed_of_sound_m_s == pytest.approx(340.294)


def test_atmosphere_data_accepts_zero_values() -> None:
    """Verifica que valores nulos físicos são aceitos.

    O teste utiliza zero como valor para todas as propriedades para
    verificar que o objeto não impõe restrições físicas adicionais
    além das definidas pelo próprio modelo de dados.

    Raises
    ------
    AssertionError
        Caso o ``AtmosphereData`` rejeite ou altere os valores.
    """

    data = AtmosphereData(
        temperature_K=0.0, pressure_Pa=0.0, density_kg_m3=0.0, speed_of_sound_m_s=0.0
    )

    assert data.temperature_K == 0.0
    assert data.pressure_Pa == 0.0
    assert data.density_kg_m3 == 0.0
    assert data.speed_of_sound_m_s == 0.0
