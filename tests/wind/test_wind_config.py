from __future__ import annotations

import pytest
from pydantic import ValidationError

from vahsimulator.wind.discrete_gust import DiscreteGustParameters
from vahsimulator.wind.wind_config import (
    ConstantWindConfig,
    TurbulenceConfig,
    WindConfig,
)


class TestConstantWindConfig:
    """Testes para ``ConstantWindConfig``."""

    def test_default_values(self) -> None:
        """Verifica os valores padrão da configuração."""
        config = ConstantWindConfig()

        assert config.wind_speed == 0.0
        assert config.wind_direction == 0.0
        assert config.wind_elevation == 0.0

    def test_custom_values(self) -> None:
        """Verifica a criação com valores personalizados."""
        config = ConstantWindConfig(
            wind_speed=10.0, wind_direction=180.0, wind_elevation=5.0
        )

        assert config.wind_speed == 10.0
        assert config.wind_direction == 180.0
        assert config.wind_elevation == 5.0

    def test_zero_wind_speed_is_valid(self) -> None:
        """Verifica que velocidade de vento nula seja aceita."""
        config = ConstantWindConfig(wind_speed=0.0)

        assert config.wind_speed == 0.0

    def test_negative_wind_speed_is_rejected(self) -> None:
        """Verifica que velocidade de vento negativa seja rejeitada."""
        with pytest.raises(ValidationError):
            ConstantWindConfig(wind_speed=-1.0)

    @pytest.mark.parametrize("elevation", [-90.0, 0.0, 90.0])
    def test_valid_wind_elevation(self, elevation: float) -> None:
        """Verifica os limites válidos do ângulo de elevação."""
        config = ConstantWindConfig(wind_elevation=elevation)

        assert config.wind_elevation == elevation

    @pytest.mark.parametrize("elevation", [-90.001, 90.001])
    def test_invalid_wind_elevation(self, elevation: float) -> None:
        """Verifica rejeição de elevação fora do intervalo válido."""
        with pytest.raises(ValidationError):
            ConstantWindConfig(wind_elevation=elevation)

    def test_extra_field_is_rejected(self) -> None:
        """Verifica que campos não especificados sejam rejeitados."""
        with pytest.raises(ValidationError):
            ConstantWindConfig(wind_speed=10.0, invalid_field=1.0)


class TestTurbulenceConfig:
    """Testes para ``TurbulenceConfig``."""

    def test_default_values(self) -> None:
        """Verifica os valores padrão."""
        config = TurbulenceConfig()

        assert config.type == "dryden"
        assert config.severity == "moderate"

    @pytest.mark.parametrize("severity", ["light", "moderate", "severe"])
    def test_valid_severity(self, severity: str) -> None:
        """Verifica os níveis de severidade válidos."""
        config = TurbulenceConfig(severity=severity)

        assert config.severity == severity

    def test_invalid_severity_is_rejected(self) -> None:
        """Verifica rejeição de severidade inválida."""
        with pytest.raises(ValidationError):
            TurbulenceConfig(severity="invalid")

    def test_valid_type(self) -> None:
        """Verifica o tipo de turbulência Dryden."""
        config = TurbulenceConfig(type="dryden")

        assert config.type == "dryden"

    def test_invalid_type_is_rejected(self) -> None:
        """Verifica rejeição de tipo de turbulência inválido."""
        with pytest.raises(ValidationError):
            TurbulenceConfig(type="invalid")

    def test_extra_field_is_rejected(self) -> None:
        """Verifica que campos não especificados sejam rejeitados."""
        with pytest.raises(ValidationError):
            TurbulenceConfig(type="dryden", invalid_field="value")


class TestWindConfig:
    """Testes para ``WindConfig``."""

    def test_default_values(self) -> None:
        """Verifica os valores padrão da configuração completa."""
        config = WindConfig()

        assert isinstance(config.constant_wind, ConstantWindConfig)
        assert config.constant_wind.wind_speed == 0.0
        assert config.constant_wind.wind_direction == 0.0
        assert config.constant_wind.wind_elevation == 0.0

        assert config.turbulence is None
        assert config.discrete_gusts == []

    def test_constant_wind_configuration(self) -> None:
        """Verifica configuração somente com vento constante."""
        config = WindConfig(
            constant_wind={
                "wind_speed": 10.0,
                "wind_direction": 180.0,
                "wind_elevation": 0.0,
            }
        )

        assert config.constant_wind.wind_speed == 10.0
        assert config.constant_wind.wind_direction == 180.0
        assert config.constant_wind.wind_elevation == 0.0

    def test_turbulence_configuration(self) -> None:
        """Verifica configuração de turbulência."""
        config = WindConfig(turbulence={"type": "dryden", "severity": "moderate"})

        assert config.turbulence is not None
        assert config.turbulence.type == "dryden"
        assert config.turbulence.severity == "moderate"

    def test_discrete_gust_configuration(self) -> None:
        """Verifica configuração de uma rajada discreta."""
        config = WindConfig(
            discrete_gusts=[
                {
                    "leading_edge_altitude_m": 900.0,
                    "half_width": 100.0,
                    "risk": 0.01,
                    "longitudinal": {"standard_deviation": 4.0, "length_scale": 1000.0},
                    "lateral": {"standard_deviation": 2.0, "length_scale": 900.0},
                    "vertical": {"standard_deviation": 3.0, "length_scale": 800.0},
                }
            ]
        )

        assert len(config.discrete_gusts) == 1
        assert isinstance(config.discrete_gusts[0], DiscreteGustParameters)

        gust = config.discrete_gusts[0]

        assert gust.leading_edge_altitude_m == 900.0
        assert gust.half_width == 100.0
        assert gust.risk == 0.01

        assert gust.longitudinal.standard_deviation == 4.0
        assert gust.longitudinal.length_scale == 1000.0

        assert gust.lateral.standard_deviation == 2.0
        assert gust.lateral.length_scale == 900.0

        assert gust.vertical.standard_deviation == 3.0
        assert gust.vertical.length_scale == 800.0

    def test_multiple_discrete_gusts(self) -> None:
        """Verifica configuração de múltiplas rajadas."""
        config = WindConfig(
            discrete_gusts=[
                {
                    "leading_edge_altitude_m": 900.0,
                    "half_width": 100.0,
                    "risk": 0.01,
                    "longitudinal": {"standard_deviation": 4.0, "length_scale": 1000.0},
                    "lateral": {"standard_deviation": 2.0, "length_scale": 900.0},
                    "vertical": {"standard_deviation": 3.0, "length_scale": 800.0},
                },
                {
                    "leading_edge_altitude_m": 1500.0,
                    "half_width": 150.0,
                    "risk": 0.005,
                    "longitudinal": {"standard_deviation": 5.0, "length_scale": 1200.0},
                    "lateral": {"standard_deviation": 2.5, "length_scale": 1000.0},
                    "vertical": {"standard_deviation": 3.5, "length_scale": 900.0},
                },
            ]
        )

        assert len(config.discrete_gusts) == 2

        assert config.discrete_gusts[0].leading_edge_altitude_m == 900.0
        assert config.discrete_gusts[0].half_width == 100.0
        assert config.discrete_gusts[0].risk == 0.01

        assert config.discrete_gusts[0].longitudinal.standard_deviation == 4.0
        assert config.discrete_gusts[0].longitudinal.length_scale == 1000.0

        assert config.discrete_gusts[1].leading_edge_altitude_m == 1500.0
        assert config.discrete_gusts[1].half_width == 150.0
        assert config.discrete_gusts[1].risk == 0.005

        assert config.discrete_gusts[1].longitudinal.standard_deviation == 5.0
        assert config.discrete_gusts[1].longitudinal.length_scale == 1200.0

    def test_complete_configuration(self) -> None:
        """Verifica configuração com todos os componentes de vento."""
        config = WindConfig(
            constant_wind={
                "wind_speed": 10.0,
                "wind_direction": 180.0,
                "wind_elevation": 0.0,
            },
            turbulence={"type": "dryden", "severity": "moderate"},
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

        assert config.constant_wind.wind_speed == 10.0
        assert config.constant_wind.wind_direction == 180.0
        assert config.constant_wind.wind_elevation == 0.0

        assert config.turbulence is not None
        assert config.turbulence.type == "dryden"
        assert config.turbulence.severity == "moderate"

        assert len(config.discrete_gusts) == 1
        assert isinstance(config.discrete_gusts[0], DiscreteGustParameters)

    def test_empty_discrete_gust_list(self) -> None:
        """Verifica que uma lista vazia de rajadas seja aceita."""
        config = WindConfig(discrete_gusts=[])

        assert config.discrete_gusts == []

    def test_turbulence_can_be_disabled(self) -> None:
        """Verifica que a turbulência possa ser desabilitada."""
        config = WindConfig(turbulence=None)

        assert config.turbulence is None

    def test_extra_field_is_rejected(self) -> None:
        """Verifica que campos desconhecidos sejam rejeitados."""
        with pytest.raises(ValidationError):
            WindConfig(invalid_field="value")

    def test_nested_extra_field_is_rejected(self) -> None:
        """Verifica rejeição de campo desconhecido em vento constante."""
        with pytest.raises(ValidationError):
            WindConfig(constant_wind={"wind_speed": 10.0, "invalid_field": 1.0})
