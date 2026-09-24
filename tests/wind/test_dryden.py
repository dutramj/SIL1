from __future__ import annotations

import numpy as np
import pytest

from vahsimulator.vehicle_state import VehicleState
from vahsimulator.wind.dryden import DrydenTurbulence


@pytest.fixture
def config_data() -> dict:
    """Fornece uma configuração mínima para os testes do Dryden.

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
    """Cria um estado de veículo válido para os testes.

    Returns
    -------
    VehicleState
        Estado com altitude de 1000 m e velocidade longitudinal
        de 100 m/s.
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
def constant_wind_body() -> np.ndarray:
    """Fornece um vento constante nulo na referência body.

    Returns
    -------
    numpy.ndarray
        Vetor de vento ``3 x 1``.
    """
    return np.zeros((3, 1), dtype=np.float64)


class TestDrydenTurbulence:
    """Testes do modelo de turbulência de Dryden."""

    @pytest.mark.parametrize("turbulence", ["light", "moderate", "severe"])
    def test_valid_turbulence_levels(self, turbulence: str) -> None:
        """Verifica a criação para níveis válidos de turbulência.

        Parameters
        ----------
        turbulence : str
            Nível de turbulência.
        """
        model = DrydenTurbulence(turbulence)

        assert model.turbulence == turbulence

    def test_invalid_turbulence_level(self) -> None:
        """Verifica a rejeição de um nível de turbulência inválido."""
        with pytest.raises(ValueError, match="turbulence deve ser"):
            DrydenTurbulence("invalid")

    def test_default_initial_state(self) -> None:
        """Verifica os valores iniciais do modelo."""
        model = DrydenTurbulence()

        assert model.dt == 0.0
        assert model.b == 0.0
        assert model.Va == 1.0

        assert model.Lu == 0.0
        assert model.Lv == 0.0
        assert model.Lw == 0.0

        assert model.sigma_u == 0.0
        assert model.sigma_v == 0.0
        assert model.sigma_w == 0.0

        assert model.w6 == 0.0

        assert model.u_w is None
        assert model.v_w is None
        assert model.w_w is None
        assert model.p_w is None
        assert model.q_w is None
        assert model.r_w is None

        assert model._initialized is False

    def test_initialize_updates_sample_time_and_reference_length(
        self, config_data: dict
    ) -> None:
        """Verifica ``dt`` e o comprimento de referência após inicialização.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=10)

        assert model.dt == pytest.approx(0.01)
        assert model.b == pytest.approx(2.0)

    @pytest.mark.parametrize(
        "turbulence,minimum_factor,maximum_factor",
        [
            ("light", 2.2, 4.3),
            ("moderate", 2.0 * 2.2, 2.0 * 4.3),
            ("severe", 3.0 * 2.2, 3.0 * 4.3),
        ],
    )
    def test_initialize_w6_range(
        self,
        config_data: dict,
        turbulence: str,
        minimum_factor: float,
        maximum_factor: float,
    ) -> None:
        """Verifica a faixa do parâmetro ``w6``.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        turbulence : str
            Nível de turbulência.
        minimum_factor : float
            Limite inferior esperado para ``w6``.
        maximum_factor : float
            Limite superior esperado para ``w6``.
        """
        model = DrydenTurbulence(turbulence)

        model.initialize(config_data=config_data, dt=0.01, seed=123)

        assert minimum_factor <= model.w6 <= maximum_factor

    def test_initialize_requires_phase_one(self) -> None:
        """Verifica erro quando ``phase_id=1`` não está configurada."""
        model = DrydenTurbulence()

        config_data = {
            "geometric_data": [{"phase_id": 2, "main_parameters": {"l_ref_2": 3.0}}]
        }

        with pytest.raises(ValueError, match="Não foi encontrada phase_id=1"):
            model.initialize(config_data=config_data, dt=0.01, seed=1)

    def test_update_parameters_below_304_8_m(self, config_data: dict) -> None:
        """Verifica os parâmetros abaixo de 304,8 m."""
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model._update_dryden_parameters(100.0)

        expected_lw = 100.0
        expected_lu = 100.0 / (0.177 + 0.000823 * 100.0) ** 1.2

        expected_sigma_w = 0.1 * model.w6
        expected_sigma_uv = 0.1 * model.w6 / (0.177 + 0.000823 * 100.0) ** 0.4

        assert model.Lw == pytest.approx(expected_lw)
        assert model.Lu == pytest.approx(expected_lu)
        assert model.Lv == pytest.approx(expected_lu)

        assert model.sigma_w == pytest.approx(expected_sigma_w)
        assert model.sigma_u == pytest.approx(expected_sigma_uv)
        assert model.sigma_v == pytest.approx(expected_sigma_uv)

    def test_update_parameters_above_609_6_m(self, config_data: dict) -> None:
        """Verifica os parâmetros acima de 609,6 m."""
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model._update_dryden_parameters(1000.0)

        assert model.Lu == pytest.approx(533.4)
        assert model.Lv == pytest.approx(533.4)
        assert model.Lw == pytest.approx(533.4)

        assert model.sigma_u == pytest.approx(3.0)
        assert model.sigma_v == pytest.approx(3.0)
        assert model.sigma_w == pytest.approx(3.0)

    def test_update_parameters_negative_altitude_is_clamped(
        self, config_data: dict
    ) -> None:
        """Verifica o tratamento de altitude negativa."""
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model._update_dryden_parameters(-100.0)

        assert model.Lw == pytest.approx(1e-3)

        assert np.isfinite(model.Lu)
        assert np.isfinite(model.Lv)
        assert np.isfinite(model.sigma_u)
        assert np.isfinite(model.sigma_v)
        assert np.isfinite(model.sigma_w)

    def test_create_filters(self, config_data: dict) -> None:
        """Verifica a criação dos seis filtros Dryden."""
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model._update_dryden_parameters(1000.0)
        model.Va = 100.0
        model._create_filters()

        assert model.u_w is not None
        assert model.v_w is not None
        assert model.w_w is not None
        assert model.p_w is not None
        assert model.q_w is not None
        assert model.r_w is not None

    def test_evaluate_ic_initializes_model(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        constant_wind_body: np.ndarray,
    ) -> None:
        """Verifica a inicialização dos filtros no instante inicial.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado inicial do veículo.
        constant_wind_body : numpy.ndarray
            Vento constante na referência body.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model.evaluate_ic(state=vehicle_state, constant_wind_body=constant_wind_body)

        assert model._initialized is True
        assert model.Va == pytest.approx(100.0)

        assert model.u_w is not None
        assert model.v_w is not None
        assert model.w_w is not None
        assert model.p_w is not None
        assert model.q_w is not None
        assert model.r_w is not None

    def test_evaluate_ic_accounts_for_constant_wind(
        self, config_data: dict, vehicle_state: VehicleState
    ) -> None:
        """Verifica que o vento constante é subtraído da velocidade.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        constant_wind_body = np.array([[20.0], [0.0], [0.0]], dtype=np.float64)

        model.evaluate_ic(state=vehicle_state, constant_wind_body=constant_wind_body)

        assert model.Va == pytest.approx(80.0)

    def test_evaluate_ic_limits_airspeed_to_one_m_s(
        self, config_data: dict, vehicle_state: VehicleState
    ) -> None:
        """Verifica o limite inferior de velocidade aerodinâmica.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        constant_wind_body = np.asarray(vehicle_state.velocity_body, dtype=float)

        model.evaluate_ic(state=vehicle_state, constant_wind_body=constant_wind_body)

        assert model.Va == pytest.approx(1.0)

    def test_evaluate_returns_six_by_one_array(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        constant_wind_body: np.ndarray,
    ) -> None:
        """Verifica as dimensões da saída do modelo.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        constant_wind_body : numpy.ndarray
            Vento constante na referência body.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model.evaluate_ic(state=vehicle_state, constant_wind_body=constant_wind_body)

        np.random.seed(100)

        turbulence = model.evaluate(
            state=vehicle_state, Va=100.0, phase=1, config_data=config_data
        )

        assert isinstance(turbulence, np.ndarray)
        assert turbulence.shape == (6, 1)

    def test_evaluate_returns_finite_values(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        constant_wind_body: np.ndarray,
    ) -> None:
        """Verifica que a saída do modelo contém valores finitos.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        constant_wind_body : np.ndarray
            Vento constante na referência body.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model.evaluate_ic(state=vehicle_state, constant_wind_body=constant_wind_body)

        np.random.seed(100)

        turbulence = model.evaluate(
            state=vehicle_state, Va=100.0, phase=1, config_data=config_data
        )

        assert np.all(np.isfinite(turbulence))

    def test_evaluate_updates_reference_length_for_current_phase(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        constant_wind_body: np.ndarray,
    ) -> None:
        """Verifica a atualização de ``b`` para a fase atual.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        constant_wind_body : np.ndarray
            Vento constante na referência body.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model.evaluate_ic(state=vehicle_state, constant_wind_body=constant_wind_body)

        np.random.seed(100)

        model.evaluate(state=vehicle_state, Va=100.0, phase=2, config_data=config_data)

        assert model.b == pytest.approx(3.0)

    def test_evaluate_invalid_phase(
        self, config_data: dict, vehicle_state: VehicleState
    ) -> None:
        """Verifica erro para uma fase inexistente.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        """
        model = DrydenTurbulence()

        with pytest.raises(ValueError, match="Não foi encontrada phase_id=99"):
            model.evaluate(
                state=vehicle_state, Va=100.0, phase=99, config_data=config_data
            )

    def test_evaluate_zero_altitude_returns_zero(self, config_data: dict) -> None:
        """Verifica saída nula para altitude igual a zero.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        """
        state = VehicleState.from_dict(
            {
                "lat": np.deg2rad(0.0),
                "lon": np.deg2rad(0.0),
                "alt": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "airspeed_u": 100.0,
                "airspeed_v": 0.0,
                "airspeed_w": 0.0,
            }
        )

        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        turbulence = model.evaluate(
            state=state, Va=100.0, phase=1, config_data=config_data
        )

        assert turbulence.shape == (6, 1)
        assert np.array_equal(turbulence, np.zeros((6, 1)))

    def test_evaluate_above_80_km_returns_zero(self, config_data: dict) -> None:
        """Verifica saída nula acima de 80 km.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        """
        state = VehicleState.from_dict(
            {
                "lat": np.deg2rad(0.0),
                "lon": np.deg2rad(0.0),
                "alt": 80001.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "airspeed_u": 100.0,
                "airspeed_v": 0.0,
                "airspeed_w": 0.0,
            }
        )

        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        turbulence = model.evaluate(
            state=state, Va=100.0, phase=1, config_data=config_data
        )

        assert turbulence.shape == (6, 1)
        assert np.array_equal(turbulence, np.zeros((6, 1)))

    def test_evaluate_at_80_km_is_active(
        self,
        config_data: dict,
        vehicle_state: VehicleState,
        constant_wind_body: np.ndarray,
    ) -> None:
        """Verifica que 80 km ainda pertence à faixa ativa.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado inicial do veículo.
        constant_wind_body : numpy.ndarray
            Vento constante na referência body.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        model.evaluate_ic(state=vehicle_state, constant_wind_body=constant_wind_body)

        state_80km = VehicleState.from_dict(
            {
                "lat": np.deg2rad(0.0),
                "lon": np.deg2rad(0.0),
                "alt": 80000.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "airspeed_u": 100.0,
                "airspeed_v": 0.0,
                "airspeed_w": 0.0,
            }
        )

        np.random.seed(100)

        turbulence = model.evaluate(
            state=state_80km, Va=100.0, phase=1, config_data=config_data
        )

        assert turbulence.shape == (6, 1)
        assert np.all(np.isfinite(turbulence))

    def test_evaluate_initializes_filters_lazily(
        self, config_data: dict, vehicle_state: VehicleState
    ) -> None:
        """Verifica a criação automática dos filtros em ``evaluate()``.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        assert model.u_w is None
        assert model.v_w is None
        assert model.w_w is None
        assert model.p_w is None
        assert model.q_w is None
        assert model.r_w is None

        np.random.seed(100)

        turbulence = model.evaluate(
            state=vehicle_state, Va=100.0, phase=1, config_data=config_data
        )

        assert turbulence.shape == (6, 1)

        assert model.u_w is not None
        assert model.v_w is not None
        assert model.w_w is not None
        assert model.p_w is not None
        assert model.q_w is not None
        assert model.r_w is not None

    def test_evaluate_limits_low_airspeed(
        self, config_data: dict, vehicle_state: VehicleState
    ) -> None:
        """Verifica o limite inferior de ``Va`` em ``evaluate()``.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        np.random.seed(100)

        turbulence = model.evaluate(
            state=vehicle_state, Va=0.0, phase=1, config_data=config_data
        )

        assert model.Va == pytest.approx(1.0)
        assert turbulence.shape == (6, 1)
        assert np.all(np.isfinite(turbulence))

    def test_evaluate_changes_with_random_input(
        self, config_data: dict, vehicle_state: VehicleState
    ) -> None:
        """Verifica que a saída responde à entrada aleatória.

        Parameters
        ----------
        config_data : dict
            Configuração utilizada pelo modelo.
        vehicle_state : VehicleState
            Estado do veículo.
        """
        model = DrydenTurbulence()

        model.initialize(config_data=config_data, dt=0.01, seed=1)

        np.random.seed(100)

        first_output = model.evaluate(
            state=vehicle_state, Va=100.0, phase=1, config_data=config_data
        )

        np.random.seed(200)

        second_output = model.evaluate(
            state=vehicle_state, Va=100.0, phase=1, config_data=config_data
        )

        assert not np.array_equal(first_output, second_output)
