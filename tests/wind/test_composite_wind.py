from __future__ import annotations

import numpy as np
import pytest

from vahsimulator.launching_reference import LaunchReference
from vahsimulator.vehicle_state import VehicleState
from vahsimulator.wind.composite_wind import CompositeWind
from vahsimulator.wind.constant_wind import ConstantWind
from vahsimulator.wind.discrete_gust import DiscreteGustWind
from vahsimulator.wind.discrete_gust_list import DiscreteGustList
from vahsimulator.wind.dryden import DrydenTurbulence
from vahsimulator.wind.wind_config import WindConfig


@pytest.fixture
def config_data() -> dict:
    """Fornece uma configuração mínima para o ``CompositeWind``.

    Returns
    -------
    dict
        Configuração contendo dados geométricos para duas fases.
    """
    return {
        "geometric_data": [
            {"phase_id": 1, "main_parameters": {"l_ref_2": 2.0}},
            {"phase_id": 2, "main_parameters": {"l_ref_2": 3.0}},
        ]
    }


@pytest.fixture
def vehicle_state() -> VehicleState:
    """Cria um estado de veículo para os testes.

    Returns
    -------
    VehicleState
        Estado em voo nivelado a 1000 m de altitude.
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
def launch_reference() -> LaunchReference:
    """Cria uma referência de lançamento.

    Returns
    -------
    LaunchReference
        Referência com azimute de zero graus.
    """
    return LaunchReference(
        lat=np.float64(0.0),
        lon=np.float64(0.0),
        alt=np.float64(0.0),
        azimuth=np.float64(0.0),
    )


@pytest.fixture
def wind_config() -> WindConfig:
    """Cria uma configuração de vento constante.

    Returns
    -------
    WindConfig
        Configuração sem turbulência e sem rajadas.
    """
    return WindConfig(
        constant_wind={
            "wind_speed": 10.0,
            "wind_direction": 180.0,
            "wind_elevation": 0.0,
        }
    )


@pytest.fixture
def wind_config_with_gust() -> WindConfig:
    """Cria uma configuração com vento constante e rajada discreta.

    Returns
    -------
    WindConfig
        Configuração contendo vento constante e uma rajada.
    """
    return WindConfig(
        constant_wind={
            "wind_speed": 10.0,
            "wind_direction": 180.0,
            "wind_elevation": 0.0,
        },
        discrete_gusts=[
            {
                "leading_edge_altitude_m": 900.0,
                "half_width": 100.0,
                "risk": 0.01,
                "longitudinal": {"standard_deviation": 4.0, "length_scale": 1000.0},
                "lateral": {"standard_deviation": 2.0, "length_scale": 900.0},
                "vertical": {"standard_deviation": 3.0, "length_scale": 800.0},
            }
        ],
    )


class TestCompositeWind:
    """Testes do modelo composto de vento."""

    def test_default_initial_state(self) -> None:
        """Verifica o estado inicial do modelo."""
        model = CompositeWind()

        assert isinstance(model.constant_wind, ConstantWind)
        assert isinstance(model.discrete_gusts, DiscreteGustList)

        assert model.dryden is None
        assert model.config_data == {}
        assert model.dt == 0.0
        assert model.get_tas() == pytest.approx(0.0)

    def test_initialize_constant_wind(
        self, config_data: dict, wind_config: WindConfig
    ) -> None:
        """Verifica a inicialização do vento constante.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config : WindConfig
            Configuração do vento.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        assert model.dt == pytest.approx(0.01)
        assert model.config_data is config_data
        assert model.dryden is None

        expected_wind = np.array([[10.0], [0.0], [0.0]], dtype=np.float64)

        assert np.allclose(model.constant_wind.wind_ned, expected_wind)

    def test_initialize_discrete_gusts(
        self, config_data: dict, wind_config_with_gust: WindConfig
    ) -> None:
        """Verifica a inicialização das rajadas discretas.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config_with_gust : WindConfig
            Configuração contendo uma rajada.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data,
            dt=0.01,
            wind_config=wind_config_with_gust,
            seed=123,
        )

        assert len(model.discrete_gusts) == 1
        assert isinstance(model.discrete_gusts.gusts[0], DiscreteGustWind)

    def test_initialize_without_turbulence(
        self, config_data: dict, wind_config: WindConfig
    ) -> None:
        """Verifica que a ausência de turbulência mantém Dryden desativado.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config : WindConfig
            Configuração sem turbulência.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        assert model.dryden is None

    def test_initialize_dryden(self, config_data: dict) -> None:
        """Verifica a criação do modelo Dryden pela configuração.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        """
        wind_config = WindConfig(
            constant_wind={
                "wind_speed": 10.0,
                "wind_direction": 180.0,
                "wind_elevation": 0.0,
            },
            turbulence={"type": "dryden", "severity": "moderate"},
        )

        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        assert isinstance(model.dryden, DrydenTurbulence)
        assert model.dryden.turbulence == "moderate"
        assert model.dryden.dt == pytest.approx(0.01)
        assert model.dryden.b == pytest.approx(2.0)

    def test_initialize_replaces_previous_dryden(self, config_data: dict) -> None:
        """Verifica que ``initialize()`` recria o Dryden.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        """
        config_with_dryden = WindConfig(
            turbulence={"type": "dryden", "severity": "light"}
        )

        config_without_dryden = WindConfig()

        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=config_with_dryden, seed=123
        )

        assert model.dryden is not None

        model.initialize(
            config_data=config_data,
            dt=0.02,
            wind_config=config_without_dryden,
            seed=123,
        )

        assert model.dryden is None

    def test_evaluate_environment_ned_with_constant_wind(
        self,
        config_data: dict,
        wind_config: WindConfig,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
    ) -> None:
        """Verifica o ambiente NED contendo somente vento constante.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config : WindConfig
            Configuração do vento.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        environment_ned = model._evaluate_environment_ned(state=vehicle_state)

        expected = np.array([[10.0], [0.0], [0.0]], dtype=np.float64)

        assert np.allclose(environment_ned, expected)

    def test_evaluate_environment_ned_sums_constant_wind_and_gust(
        self,
        config_data: dict,
        wind_config_with_gust: WindConfig,
        vehicle_state: VehicleState,
    ) -> None:
        """Verifica a soma do vento constante com a rajada.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config_with_gust : WindConfig
            Configuração contendo uma rajada.
        vehicle_state : VehicleState
            Estado do veículo.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data,
            dt=0.01,
            wind_config=wind_config_with_gust,
            seed=123,
        )

        environment_ned = model._evaluate_environment_ned(state=vehicle_state)

        constant_wind_ned = model.constant_wind.evaluate_ned(vehicle_state)

        discrete_gust_ned = model.discrete_gusts.evaluate_ned(
            altitude_m=vehicle_state.alt,
            wind_direction_deg=model.constant_wind.wind_direction,
        )

        expected = constant_wind_ned + discrete_gust_ned

        np.testing.assert_allclose(environment_ned, expected, atol=1e-12)

    def test_evaluate_without_dryden(
        self,
        config_data: dict,
        wind_config: WindConfig,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
    ) -> None:
        """Verifica a saída sem turbulência Dryden.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config : WindConfig
            Configuração sem turbulência.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        result = model.evaluate(
            state=vehicle_state, launch_reference=launch_reference, Va=100.0, phase=1
        )

        environment_ned = model._evaluate_environment_ned(state=vehicle_state)

        environment_body = model.transformer.ned_to_body(
            wind_ned=environment_ned,
            roll_rad=vehicle_state.roll_ned,
            pitch_rad=vehicle_state.pitch_ned,
            yaw_rad=vehicle_state.yaw_ned,
        )

        expected = np.vstack((environment_body, np.zeros((3, 1), dtype=np.float64)))

        assert result.shape == (6, 1)

        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_evaluate_with_discrete_gust(
        self,
        config_data: dict,
        wind_config_with_gust: WindConfig,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
    ) -> None:
        """Verifica a inclusão de uma rajada discreta na saída.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config_with_gust : WindConfig
            Configuração contendo uma rajada.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data,
            dt=0.01,
            wind_config=wind_config_with_gust,
            seed=123,
        )

        result = model.evaluate(
            state=vehicle_state, launch_reference=launch_reference, Va=100.0, phase=1
        )

        environment_ned = model.constant_wind.evaluate_ned(
            vehicle_state
        ) + model.discrete_gusts.evaluate_ned(
            altitude_m=vehicle_state.alt,
            wind_direction_deg=model.constant_wind.wind_direction,
        )

        environment_body = model.transformer.ned_to_body(
            wind_ned=environment_ned,
            roll_rad=vehicle_state.roll_ned,
            pitch_rad=vehicle_state.pitch_ned,
            yaw_rad=vehicle_state.yaw_ned,
        )

        expected = np.vstack((environment_body, np.zeros((3, 1), dtype=np.float64)))

        assert result.shape == (6, 1)

        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_evaluate_ic_calculates_tas(
        self,
        config_data: dict,
        wind_config: WindConfig,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
    ) -> None:
        """Verifica o cálculo da TAS no instante inicial.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config : WindConfig
            Configuração do vento.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        print("\n--- VehicleState ---")
        print("u =", vehicle_state.u)
        print("v =", vehicle_state.v)
        print("w =", vehicle_state.w)
        print("velocity_body =", vehicle_state.velocity_body)

        print("\n--- Attitude NED ---")
        print("roll_ned =", vehicle_state.roll_ned)
        print("pitch_ned =", vehicle_state.pitch_ned)
        print("yaw_ned =", vehicle_state.yaw_ned)

        print("\n--- Position ---")
        print("lat =", vehicle_state.lat)
        print("lon =", vehicle_state.lon)
        print("alt =", vehicle_state.alt)

        environment_ned = model._evaluate_environment_ned(state=vehicle_state)

        print("\n--- Environment NED ---")
        print(environment_ned)

        environment_body = model.transformer.ned_to_body(
            wind_ned=environment_ned,
            roll_rad=vehicle_state.roll_ned,
            pitch_rad=vehicle_state.pitch_ned,
            yaw_rad=vehicle_state.yaw_ned,
        )

        print("\n--- Environment BODY ---")
        print(environment_body)

        air_velocity_body = vehicle_state.velocity_body - environment_body

        print("\n--- Air velocity BODY ---")
        print(air_velocity_body)

        print("\n--- TAS ---")
        print(np.linalg.norm(air_velocity_body))

        model.evaluate_ic(state=vehicle_state, launch_reference=launch_reference)

        expected_tas = 90.0

        assert model.get_tas() == pytest.approx(expected_tas)

    def test_get_tas_returns_numpy_float64(
        self,
        config_data: dict,
        wind_config: WindConfig,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
    ) -> None:
        """Verifica o tipo retornado por ``get_tas()``.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        wind_config : WindConfig
            Configuração do vento.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        """
        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        model.evaluate_ic(state=vehicle_state, launch_reference=launch_reference)

        assert isinstance(model.get_tas(), np.float64)

    def test_evaluate_ic_initializes_dryden(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
    ) -> None:
        """Verifica a inicialização do Dryden em ``evaluate_ic()``.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        """
        wind_config = WindConfig(
            constant_wind={
                "wind_speed": 10.0,
                "wind_direction": 180.0,
                "wind_elevation": 0.0,
            },
            turbulence={"type": "dryden", "severity": "moderate"},
        )

        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        assert model.dryden is not None
        assert model.dryden._initialized is False

        model.evaluate_ic(state=vehicle_state, launch_reference=launch_reference)

        assert model.dryden._initialized is True
        assert model.dryden.Va == pytest.approx(90.0)

    def test_evaluate_dryden_output_is_added_to_environment(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifica a soma das componentes do Dryden ao ambiente.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        monkeypatch : pytest.MonkeyPatch
            Ferramenta para substituir o resultado do Dryden.
        """
        wind_config = WindConfig(
            constant_wind={
                "wind_speed": 10.0,
                "wind_direction": 180.0,
                "wind_elevation": 0.0,
            },
            turbulence={"type": "dryden", "severity": "moderate"},
        )

        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        dryden_output = np.array(
            [[1.0], [2.0], [3.0], [0.1], [0.2], [0.3]], dtype=np.float64
        )

        def fake_evaluate(
            state: VehicleState, Va: float, phase: int, config_data: dict
        ) -> np.ndarray:
            """Retorna uma saída determinística para o teste."""
            return dryden_output.copy()

        assert model.dryden is not None

        monkeypatch.setattr(model.dryden, "evaluate", fake_evaluate)

        result = model.evaluate(
            state=vehicle_state, launch_reference=launch_reference, Va=100.0, phase=1
        )

        environment_ned = model._evaluate_environment_ned(state=vehicle_state)

        environment_body = model.transformer.ned_to_body(
            wind_ned=environment_ned,
            roll_rad=vehicle_state.roll_ned,
            pitch_rad=vehicle_state.pitch_ned,
            yaw_rad=vehicle_state.yaw_ned,
        )

        expected = np.vstack(
            (environment_body + dryden_output[0:3], dryden_output[3:6])
        )

        assert result.shape == (6, 1)
        assert np.allclose(result, expected)

    def test_evaluate_does_not_transform_dryden_angular_components(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifica que as componentes angulares do Dryden são preservadas.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        monkeypatch : pytest.MonkeyPatch
            Ferramenta para substituir o resultado do Dryden.
        """
        wind_config = WindConfig(turbulence={"type": "dryden", "severity": "moderate"})

        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        dryden_output = np.array(
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=np.float64
        )

        def fake_evaluate(
            state: VehicleState, Va: float, phase: int, config_data: dict
        ) -> np.ndarray:
            """Retorna uma saída determinística para o teste."""
            return dryden_output.copy()

        assert model.dryden is not None

        monkeypatch.setattr(model.dryden, "evaluate", fake_evaluate)

        result = model.evaluate(
            state=vehicle_state, launch_reference=launch_reference, Va=100.0, phase=1
        )

        expected_angular = np.array([[4.0], [5.0], [6.0]], dtype=np.float64)

        assert np.allclose(result[3:6], expected_angular)

    def test_evaluate_passes_arguments_to_dryden(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        launch_reference: LaunchReference,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifica os argumentos encaminhados ao Dryden.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        vehicle_state : VehicleState
            Estado do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        monkeypatch : pytest.MonkeyPatch
            Ferramenta para substituir o método Dryden.
        """
        wind_config = WindConfig(turbulence={"type": "dryden", "severity": "moderate"})

        model = CompositeWind()

        model.initialize(
            config_data=config_data, dt=0.01, wind_config=wind_config, seed=123
        )

        captured = {}

        def fake_evaluate(
            state: VehicleState, Va: float, phase: int, config_data: dict
        ) -> np.ndarray:
            """Captura os argumentos recebidos pelo Dryden."""
            captured["state"] = state
            captured["Va"] = Va
            captured["phase"] = phase
            captured["config_data"] = config_data

            return np.zeros((6, 1), dtype=np.float64)

        assert model.dryden is not None

        monkeypatch.setattr(model.dryden, "evaluate", fake_evaluate)

        model.evaluate(
            state=vehicle_state, launch_reference=launch_reference, Va=123.0, phase=2
        )

        assert captured["state"] is vehicle_state
        assert captured["Va"] == pytest.approx(123.0)
        assert captured["phase"] == 2
        assert captured["config_data"] is config_data
