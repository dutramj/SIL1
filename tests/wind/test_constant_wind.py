from __future__ import annotations

import numpy as np
import pytest

from vahsimulator.vehicle_state import VehicleState
from vahsimulator.wind.constant_wind import ConstantWind


@pytest.fixture
def vehicle_state() -> VehicleState:
    """Cria um estado de veículo representativo para os testes.

    Returns
    -------
    VehicleState
        Estado do veículo em posição e atitude de referência.
    """
    return VehicleState.from_dict(
        {
            "lat": np.deg2rad(0.0),
            "lon": np.deg2rad(0.0),
            "alt": 1000.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "airspeed_u": 100.0,
            "airspeed_v": 0.0,
            "airspeed_w": 0.0,
        }
    )


@pytest.fixture
def constant_wind() -> ConstantWind:
    """Cria uma instância de vento constante.

    Returns
    -------
    ConstantWind
        Modelo de vento constante.
    """
    return ConstantWind()


def test_initial_state(constant_wind: ConstantWind) -> None:
    """Verifica o estado inicial do modelo."""
    assert constant_wind.wind_ned.shape == (3, 1)
    np.testing.assert_array_equal(constant_wind.wind_ned, np.zeros((3, 1)))


def test_zero_wind(constant_wind: ConstantWind) -> None:
    """Verifica vento nulo."""
    constant_wind.initialize(wind_speed=0.0, wind_direction=0.0, wind_elevation=0.0)

    np.testing.assert_allclose(
        constant_wind.evaluate_ned(), np.zeros((3, 1)), atol=1e-12
    )


@pytest.mark.parametrize(
    "direction, expected",
    [
        (0.0, [-10.0, 0.0, 0.0]),
        (90.0, [0.0, -10.0, 0.0]),
        (180.0, [10.0, 0.0, 0.0]),
        (270.0, [0.0, 10.0, 0.0]),
    ],
)
def test_wind_direction(
    constant_wind: ConstantWind, direction: float, expected: list[float]
) -> None:
    """Verifica a convenção de direção do vento."""
    constant_wind.initialize(
        wind_speed=10.0, wind_direction=direction, wind_elevation=0.0
    )

    result = constant_wind.evaluate_ned()

    np.testing.assert_allclose(result.ravel(), expected, atol=1e-12)


@pytest.mark.parametrize(
    "elevation, expected_vertical",
    [(0.0, 0.0), (30.0, -5.0), (90.0, -10.0), (-30.0, 5.0)],
)
def test_wind_elevation(
    constant_wind: ConstantWind, elevation: float, expected_vertical: float
) -> None:
    """Verifica a convenção de elevação do vento."""
    constant_wind.initialize(
        wind_speed=10.0, wind_direction=0.0, wind_elevation=elevation
    )

    result = constant_wind.evaluate_ned()

    assert result[2, 0] == pytest.approx(expected_vertical, abs=1e-12)


def test_wind_magnitude_is_preserved(constant_wind: ConstantWind) -> None:
    """Verifica que a magnitude do vento é preservada."""
    constant_wind.initialize(wind_speed=37.5, wind_direction=123.0, wind_elevation=27.0)

    result = constant_wind.evaluate_ned()

    assert np.linalg.norm(result) == pytest.approx(37.5, abs=1e-12)


def test_evaluate_returns_copy(constant_wind: ConstantWind) -> None:
    """Verifica que evaluate_ned não expõe o estado interno."""
    constant_wind.initialize(wind_speed=10.0, wind_direction=0.0, wind_elevation=0.0)

    result = constant_wind.evaluate_ned()

    result[0, 0] = 999.0

    assert constant_wind.wind_ned[0, 0] != 999.0


def test_state_argument_is_ignored(
    constant_wind: ConstantWind, vehicle_state: VehicleState
) -> None:
    """Verifica que o estado não altera o vento constante."""
    constant_wind.initialize(wind_speed=20.0, wind_direction=45.0, wind_elevation=10.0)

    result_without_state = constant_wind.evaluate_ned()
    result_with_state = constant_wind.evaluate_ned(vehicle_state)

    np.testing.assert_array_equal(result_without_state, result_with_state)


def test_reinitialize_updates_wind(constant_wind: ConstantWind) -> None:
    """Verifica que uma nova inicialização substitui o vento anterior."""
    constant_wind.initialize(wind_speed=10.0, wind_direction=0.0, wind_elevation=0.0)

    first = constant_wind.evaluate_ned()

    constant_wind.initialize(wind_speed=20.0, wind_direction=90.0, wind_elevation=0.0)

    second = constant_wind.evaluate_ned()

    assert not np.array_equal(first, second)
    np.testing.assert_allclose(second.ravel(), [0.0, -20.0, 0.0], atol=1e-12)
