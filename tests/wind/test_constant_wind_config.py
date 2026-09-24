from __future__ import annotations

import pytest
from pydantic import ValidationError

from vahsimulator.wind.wind_config import (
    ConstantWindConfig,
    TurbulenceConfig,
    WindConfig,
)


class TestConstantWindConfig:
    """Testes para a configuração do vento constante."""

    def test_default_values(self) -> None:
        """Verifica os valores padrão da configuração."""
        config = ConstantWindConfig()

        assert config.wind_speed == 0.0
        assert config.wind_direction == 0.0
        assert config.wind_elevation == 0.0

    def test_custom_values(self) -> None:
        """Verifica a criação com valores personalizados."""
        config = ConstantWindConfig(
            wind_speed=25.0, wind_direction=90.0, wind_elevation=10.0
        )

        assert config.wind_speed == 25.0
        assert config.wind_direction == 90.0
        assert config.wind_elevation == 10.0

    def test_negative_wind_speed_is_rejected(self) -> None:
        """Verifica que velocidade negativa é rejeitada."""
        with pytest.raises(ValidationError):
            ConstantWindConfig(wind_speed=-1.0)

    def test_invalid_wind_elevation_is_rejected(self) -> None:
        """Verifica que elevação fora do intervalo é rejeitada."""
        with pytest.raises(ValidationError):
            ConstantWindConfig(wind_elevation=91.0)

        with pytest.raises(ValidationError):
            ConstantWindConfig(wind_elevation=-91.0)

    def test_extra_fields_are_rejected(self) -> None:
        """Verifica que campos não especificados são rejeitados."""
        with pytest.raises(ValidationError):
            ConstantWindConfig(unknown_parameter=1.0)


class TestTurbulenceConfig:
    """Testes para a configuração de turbulência."""

    def test_default_values(self) -> None:
        """Verifica os valores padrão."""
        config = TurbulenceConfig()

        assert config.type == "dryden"
        assert config.severity == "moderate"

    @pytest.mark.parametrize("severity", ["light", "moderate", "severe"])
    def test_valid_severity(self, severity: str) -> None:
        """Verifica os níveis válidos de severidade."""
        config = TurbulenceConfig(severity=severity)

        assert config.severity == severity

    def test_invalid_severity_is_rejected(self) -> None:
        """Verifica que severidade inválida é rejeitada."""
        with pytest.raises(ValidationError):
            TurbulenceConfig(severity="invalid")

    def test_invalid_type_is_rejected(self) -> None:
        """Verifica que tipo de turbulência inválido é rejeitado."""
        with pytest.raises(ValidationError):
            TurbulenceConfig(type="invalid")


class TestWindConfig:
    """Testes para a configuração completa de vento."""

    def test_default_configuration(self) -> None:
        """Verifica a configuração padrão."""
        config = WindConfig()

        assert isinstance(config.constant_wind, ConstantWindConfig)
        assert config.turbulence is None
        assert config.discrete_gusts == []

    def test_constant_wind_configuration(self) -> None:
        """Verifica a configuração de vento constante."""
        config = WindConfig(
            constant_wind={
                "wind_speed": 20.0,
                "wind_direction": 180.0,
                "wind_elevation": 5.0,
            }
        )

        assert config.constant_wind.wind_speed == 20.0
        assert config.constant_wind.wind_direction == 180.0
        assert config.constant_wind.wind_elevation == 5.0

    def test_turbulence_configuration(self) -> None:
        """Verifica a configuração de turbulência."""
        config = WindConfig(turbulence={"type": "dryden", "severity": "severe"})

        assert config.turbulence is not None
        assert config.turbulence.type == "dryden"
        assert config.turbulence.severity == "severe"

    def test_extra_fields_are_rejected(self) -> None:
        """Verifica que campos extras são rejeitados."""
        with pytest.raises(ValidationError):
            WindConfig(unknown_parameter=1.0)
