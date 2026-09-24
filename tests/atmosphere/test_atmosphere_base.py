from __future__ import annotations

import pytest

from vahsimulator.atmosphere import AtmosphereBase, AtmosphereData


class DummyAtmosphere(AtmosphereBase):
    """Implementação mínima de atmosfera utilizada nos testes."""

    def evaluate(self, altitude_m: float) -> AtmosphereData:
        """Retorna propriedades atmosféricas constantes.

        Parameters
        ----------
        altitude_m : float
            Altitude geométrica [m].

        Returns
        -------
        AtmosphereData
            Propriedades atmosféricas constantes utilizadas apenas
            para teste.
        """

        return AtmosphereData(
            temperature_K=300.0,
            pressure_Pa=100000.0,
            density_kg_m3=1.2,
            speed_of_sound_m_s=347.0,
        )


def test_atmosphere_base_is_abstract() -> None:
    """Verifica que ``AtmosphereBase`` não pode ser instanciada diretamente.

    Raises
    ------
    AssertionError
        Caso a classe base possa ser instanciada diretamente.
    """

    with pytest.raises(TypeError):
        AtmosphereBase()


def test_atmosphere_base_can_be_implemented() -> None:
    """Verifica o contrato mínimo de implementação da classe base.

    O teste utiliza uma implementação fictícia para garantir que uma
    classe derivada pode implementar ``evaluate`` e ser utilizada como
    um modelo atmosférico.

    Raises
    ------
    AssertionError
        Caso a implementação derivada não possa ser criada ou
        utilizada conforme o contrato definido.
    """

    atmosphere = DummyAtmosphere()

    result = atmosphere.evaluate(1000.0)

    assert isinstance(result, AtmosphereData)
    assert result.temperature_K == pytest.approx(300.0)
    assert result.pressure_Pa == pytest.approx(100000.0)
    assert result.density_kg_m3 == pytest.approx(1.2)
    assert result.speed_of_sound_m_s == pytest.approx(347.0)


def test_atmosphere_base_evaluate_contract() -> None:
    """Verifica que ``evaluate`` recebe altitude em metros.

    O teste confirma que a implementação concreta recebe diretamente
    a altitude fornecida e retorna um ``AtmosphereData``.

    Raises
    ------
    AssertionError
        Caso o método não respeite o contrato de entrada e saída.
    """

    atmosphere = DummyAtmosphere()

    result = atmosphere.evaluate(50000.0)

    assert isinstance(result, AtmosphereData)
