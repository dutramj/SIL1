from __future__ import annotations

import numpy as np
import pytest

from vahsimulator.wind.discrete_gust import (
    DiscreteGustParameters,
    DiscreteGustWind,
    GustComponentParameters,
    GustVector,
)


@pytest.fixture
def gust_parameters() -> DiscreteGustParameters:
    """Cria parâmetros padrão para os testes.

    Returns
    -------
    DiscreteGustParameters
        Parâmetros de uma rajada de teste.
    """
    return DiscreteGustParameters(
        leading_edge_altitude_m=500.0,
        half_width=100.0,
        risk=0.01,
        longitudinal=GustComponentParameters(
            standard_deviation=10.0, length_scale=1000.0
        ),
        lateral=GustComponentParameters(standard_deviation=6.0, length_scale=900.0),
        vertical=GustComponentParameters(standard_deviation=4.0, length_scale=800.0),
    )


@pytest.fixture
def gust(gust_parameters: DiscreteGustParameters) -> DiscreteGustWind:
    """Cria uma rajada de teste.

    Returns
    -------
    DiscreteGustWind
        Modelo de rajada discreta.
    """
    return DiscreteGustWind(gust_parameters)


def test_total_length(gust_parameters: DiscreteGustParameters) -> None:
    """Verifica o comprimento total da rajada."""
    assert gust_parameters.total_length_m == pytest.approx(200.0)


def test_leading_edge_altitude(gust: DiscreteGustWind) -> None:
    """Verifica a altitude da borda dianteira."""
    assert gust.leading_edge_altitude_m == pytest.approx(500.0)


def test_trailing_edge_altitude(gust: DiscreteGustWind) -> None:
    """Verifica a altitude da borda traseira."""
    assert gust.trailing_edge_altitude_m == pytest.approx(700.0)


@pytest.mark.parametrize(
    "altitude, expected",
    [(499.999, False), (500.0, True), (600.0, True), (700.0, True), (700.001, False)],
)
def test_is_active(gust: DiscreteGustWind, altitude: float, expected: bool) -> None:
    """Verifica a região de atuação da rajada."""
    assert gust.is_active(altitude) is expected


def test_profile_before_gust(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica o perfil antes da rajada."""
    result = gust.profile(distance_m=-1.0, component=gust_parameters.longitudinal)

    assert result == pytest.approx(0.0)


def test_profile_at_leading_edge(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica o perfil na borda dianteira."""
    result = gust.profile(distance_m=0.0, component=gust_parameters.longitudinal)

    assert result == pytest.approx(0.0)


def test_profile_at_half_width(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica a magnitude máxima da rajada.

    No modelo de Leahy (2008), o valor máximo da rajada não é igual
    ao desvio-padrão da turbulência. A magnitude máxima ``Vm`` é
    calculada a partir do desvio-padrão, comprimento de escala,
    half-width e risco de excedência.

    Parameters
    ----------
    gust : DiscreteGustWind
        Modelo de rajada discreta.
    gust_parameters : DiscreteGustParameters
        Parâmetros utilizados no teste.
    """
    result = gust.profile(
        distance_m=gust_parameters.half_width, component=gust_parameters.longitudinal
    )

    expected = gust._gust_magnitude(gust_parameters.longitudinal)

    assert result == pytest.approx(expected)


def test_profile_at_trailing_edge(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica o perfil na borda traseira."""
    result = gust.profile(
        distance_m=gust_parameters.total_length_m,
        component=gust_parameters.longitudinal,
    )

    assert result == pytest.approx(0.0)


def test_profile_after_gust(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica o perfil após a rajada."""
    result = gust.profile(
        distance_m=gust_parameters.total_length_m + 1.0,
        component=gust_parameters.longitudinal,
    )

    assert result == pytest.approx(0.0)


def test_profile_quarter_width(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica o perfil em um quarto da largura total.

    Para ``x = half_width / 2``, o perfil 1-cosseno assume metade
    da magnitude máxima da rajada.

    Parameters
    ----------
    gust : DiscreteGustWind
        Modelo de rajada discreta.
    gust_parameters : DiscreteGustParameters
        Parâmetros utilizados no teste.
    """
    result = gust.profile(
        distance_m=gust_parameters.half_width / 2.0,
        component=gust_parameters.longitudinal,
    )

    gust_magnitude = gust._gust_magnitude(gust_parameters.longitudinal)

    expected = 0.5 * gust_magnitude

    assert result == pytest.approx(expected)


def test_evaluate_components_at_leading_edge(gust: DiscreteGustWind) -> None:
    """Verifica as componentes na borda dianteira."""
    result = gust.evaluate_components(altitude_m=gust.leading_edge_altitude_m)

    np.testing.assert_allclose(result, np.zeros((3, 1)), atol=1e-12)


def test_evaluate_components_at_center(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica as componentes no centro da rajada.

    No centro da rajada, cada componente deve assumir sua respectiva
    magnitude máxima calculada pelo modelo de Leahy (2008).

    Parameters
    ----------
    gust : DiscreteGustWind
        Modelo de rajada discreta.
    gust_parameters : DiscreteGustParameters
        Parâmetros utilizados no teste.
    """
    altitude = gust_parameters.leading_edge_altitude_m + gust_parameters.half_width

    result = gust.evaluate_components(altitude)

    expected = np.array(
        [
            [gust._gust_magnitude(gust_parameters.longitudinal)],
            [gust._gust_magnitude(gust_parameters.lateral)],
            [gust._gust_magnitude(gust_parameters.vertical)],
        ]
    )

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_evaluate_components_outside_gust(gust: DiscreteGustWind) -> None:
    """Verifica que a rajada é nula fora da região ativa."""
    result = gust.evaluate_components(800.0)

    np.testing.assert_allclose(result, np.zeros((3, 1)), atol=1e-12)


def test_from_dict() -> None:
    """Verifica a criação a partir de dicionário."""
    config = {
        "leading_edge_altitude_m": 500.0,
        "half_width": 100.0,
        "risk": 0.01,
        "longitudinal": {"standard_deviation": 10.0, "length_scale": 1000.0},
        "lateral": {"standard_deviation": 6.0, "length_scale": 900.0},
        "vertical": {"standard_deviation": 4.0, "length_scale": 800.0},
    }

    gust = DiscreteGustWind.from_dict(config)

    assert gust.leading_edge_altitude_m == pytest.approx(500.0)
    assert gust.trailing_edge_altitude_m == pytest.approx(700.0)


def test_gust_vector_as_array() -> None:
    """Verifica a conversão de GustVector para array."""
    gust = GustVector(longitudinal=1.0, lateral=2.0, vertical=3.0)

    result = gust.as_array()

    expected = np.array([[1.0], [2.0], [3.0]])

    np.testing.assert_array_equal(result, expected)


def test_leahy_reference_case_longitudinal_gust_magnitude() -> None:
    """Verifica o caso de referência apresentado por Leahy (2008).

    O caso de referência utiliza turbulência severa a 10 km de altitude,
    com os seguintes parâmetros para a componente longitudinal:

    - desvio-padrão: 7.72 m/s;
    - comprimento de escala: 1230 m;
    - half-width da rajada: 500 m;
    - risco de excedência: 1%.

    Segundo Leahy (2008), esses parâmetros devem resultar em uma
    magnitude máxima de rajada de aproximadamente 14.83 m/s.

    References
    ----------
    Leahy, F. B. (2008).
    ``Discrete Gust Model for Launch Vehicle Assessments``.
    13th Conference on Aviation, Range and Aerospace Meteorology,
    American Meteorological Society.
    """
    parameters = DiscreteGustParameters(
        leading_edge_altitude_m=10000.0,
        half_width=500.0,
        risk=0.01,
        longitudinal=GustComponentParameters(
            standard_deviation=7.72, length_scale=1230.0
        ),
        lateral=GustComponentParameters(standard_deviation=4.67, length_scale=1100.0),
        vertical=GustComponentParameters(standard_deviation=4.67, length_scale=1100.0),
    )

    gust = DiscreteGustWind(parameters)

    magnitude = gust.profile(distance_m=500.0, component=parameters.longitudinal)

    assert magnitude == pytest.approx(14.83, rel=0.0, abs=0.01)


def test_leahy_gust_magnitude_fixture(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica a magnitude calculada pelo modelo de Leahy.

    Para os parâmetros do fixture, a magnitude máxima longitudinal
    esperada é aproximadamente 10.96678034 m/s.

    Parameters
    ----------
    gust : DiscreteGustWind
        Modelo de rajada discreta.
    gust_parameters : DiscreteGustParameters
        Parâmetros utilizados no teste.
    """
    result = gust._gust_magnitude(gust_parameters.longitudinal)

    expected = 10.966780340065966

    assert result == pytest.approx(expected, rel=0.0, abs=1e-10)
