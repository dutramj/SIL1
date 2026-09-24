from __future__ import annotations

import numpy as np
import pytest

from vahsimulator.launching_reference import LaunchReference
from vahsimulator.wind.discrete_gust import (
    DiscreteGustParameters,
    DiscreteGustWind,
    GustComponentParameters,
)
from vahsimulator.wind.discrete_gust_list import DiscreteGustList


@pytest.fixture
def gust_parameters() -> DiscreteGustParameters:
    """Cria parâmetros de uma rajada de teste.

    Returns
    -------
    DiscreteGustParameters
        Parâmetros da rajada.
    """
    return DiscreteGustParameters(
        leading_edge_altitude_m=500.0,
        half_width=100.0,
        risk=0.01,
        longitudinal=GustComponentParameters(
            standard_deviation=10.0, length_scale=1000.0
        ),
        lateral=GustComponentParameters(standard_deviation=5.0, length_scale=1000.0),
        vertical=GustComponentParameters(standard_deviation=3.0, length_scale=1000.0),
    )


@pytest.fixture
def gust(gust_parameters: DiscreteGustParameters) -> DiscreteGustWind:
    """Cria uma rajada de teste.

    Returns
    -------
    DiscreteGustWind
        Modelo da rajada.
    """
    return DiscreteGustWind(gust_parameters)


@pytest.fixture
def gust_list(gust: DiscreteGustWind) -> DiscreteGustList:
    """Cria uma lista contendo uma rajada.

    Parameters
    ----------
    gust : DiscreteGustWind
        Rajada utilizada no teste.

    Returns
    -------
    DiscreteGustList
        Lista contendo a rajada.
    """
    return DiscreteGustList([gust])


@pytest.fixture
def launch_reference() -> LaunchReference:
    """Cria uma referência de lançamento.

    Returns
    -------
    LaunchReference
        Referência com azimute de lançamento de 0 graus.
    """
    return LaunchReference(
        lat=np.float64(0.0),
        lon=np.float64(0.0),
        alt=np.float64(0.0),
        azimuth=np.float64(0.0),
    )


def test_empty_list() -> None:
    """Verifica uma lista sem rajadas."""
    gust_list = DiscreteGustList()

    assert len(gust_list) == 0

    result = gust_list.evaluate_components(600.0)

    np.testing.assert_array_equal(result, np.zeros((3, 1)))


def test_constructor(gust: DiscreteGustWind) -> None:
    """Verifica a construção da lista."""
    gust_list = DiscreteGustList([gust])

    assert len(gust_list) == 1
    assert gust_list.gusts[0] is gust


def test_from_parameters(gust_parameters: DiscreteGustParameters) -> None:
    """Verifica a criação a partir de parâmetros."""
    gust_list = DiscreteGustList.from_parameters([gust_parameters])

    assert len(gust_list) == 1
    assert gust_list.gusts[0].leading_edge_altitude_m == pytest.approx(500.0)


def test_from_dict() -> None:
    """Verifica a criação a partir de dicionário."""
    config = [
        {
            "leading_edge_altitude_m": 500.0,
            "half_width": 100.0,
            "risk": 0.01,
            "longitudinal": {"standard_deviation": 10.0, "length_scale": 1000.0},
            "lateral": {"standard_deviation": 5.0, "length_scale": 1000.0},
            "vertical": {"standard_deviation": 3.0, "length_scale": 1000.0},
        }
    ]

    gust_list = DiscreteGustList.from_dict(config)

    assert len(gust_list) == 1


def test_evaluate_components(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica as componentes de uma rajada no centro do evento.

    Parameters
    ----------
    gust : DiscreteGustWind
        Modelo de rajada discreta.
    gust_parameters : DiscreteGustParameters
        Parâmetros utilizados no teste.
    """
    altitude = gust_parameters.leading_edge_altitude_m + gust_parameters.half_width

    gust_list = DiscreteGustList([gust])

    result = gust_list.evaluate_components(altitude)

    expected = np.array(
        [
            [gust._gust_magnitude(gust_parameters.longitudinal)],
            [gust._gust_magnitude(gust_parameters.lateral)],
            [gust._gust_magnitude(gust_parameters.vertical)],
        ]
    )

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_multiple_gusts_are_summed(gust_parameters: DiscreteGustParameters) -> None:
    """Verifica a soma de múltiplas rajadas ativas."""
    gust_1 = DiscreteGustWind(gust_parameters)

    gust_2_parameters = gust_parameters.model_copy(
        update={"leading_edge_altitude_m": 500.0}
    )
    gust_2 = DiscreteGustWind(gust_2_parameters)

    gust_list = DiscreteGustList([gust_1, gust_2])

    altitude = 600.0

    result = gust_list.evaluate_components(altitude)

    expected_single = gust_1.evaluate_components(altitude)
    expected = 2.0 * expected_single

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_ned_to_wind_frame_at_zero_degrees(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica a transformação para vento vindo do Norte.

    Para ``wind_direction_deg = 0 deg``, o vento vem do Norte e
    sopra para o Sul. Portanto:

    * longitudinal positivo -> Sul;
    * lateral positivo -> Oeste;
    * vertical positivo -> Up, portanto negativo em NED Down.

    Parameters
    ----------
    gust : DiscreteGustWind
        Modelo de rajada discreta.
    gust_parameters : DiscreteGustParameters
        Parâmetros utilizados no teste.
    """
    gust_list = DiscreteGustList([gust])

    altitude = gust_parameters.leading_edge_altitude_m + gust_parameters.half_width

    result = gust_list.evaluate_ned(altitude_m=altitude, wind_direction_deg=0.0)

    longitudinal = gust._gust_magnitude(gust_parameters.longitudinal)
    lateral = gust._gust_magnitude(gust_parameters.lateral)
    vertical = gust._gust_magnitude(gust_parameters.vertical)

    expected = np.array([[-longitudinal], [-lateral], [-vertical]], dtype=np.float64)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_ned_to_wind_frame_at_ninety_degrees(
    gust: DiscreteGustWind, gust_parameters: DiscreteGustParameters
) -> None:
    """Verifica a transformação para vento vindo do Leste.

    Para ``wind_direction_deg = 90 deg``, o vento vem do Leste e
    sopra para o Oeste. Portanto:

    * longitudinal positivo -> Oeste;
    * lateral positivo -> Norte;
    * vertical positivo -> Up, portanto negativo em NED Down.

    Parameters
    ----------
    gust : DiscreteGustWind
        Modelo de rajada discreta.
    gust_parameters : DiscreteGustParameters
        Parâmetros utilizados no teste.
    """
    gust_list = DiscreteGustList([gust])

    altitude = gust_parameters.leading_edge_altitude_m + gust_parameters.half_width

    result = gust_list.evaluate_ned(altitude_m=altitude, wind_direction_deg=90.0)

    longitudinal = gust._gust_magnitude(gust_parameters.longitudinal)
    lateral = gust._gust_magnitude(gust_parameters.lateral)
    vertical = gust._gust_magnitude(gust_parameters.vertical)

    expected = np.array([[lateral], [-longitudinal], [-vertical]], dtype=np.float64)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_active_gusts(gust_list: DiscreteGustList) -> None:
    """Verifica a identificação das rajadas ativas."""
    active = gust_list.active_gusts(600.0)

    assert len(active) == 1
    assert active[0] is gust_list.gusts[0]


def test_no_active_gusts(gust_list: DiscreteGustList) -> None:
    """Verifica altitude fora da região de atuação."""
    active = gust_list.active_gusts(800.0)

    assert active == []
