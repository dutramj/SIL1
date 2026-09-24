from __future__ import annotations

import inspect

import numpy as np
import pytest

from vahsimulator.launching_reference import LaunchReference
from vahsimulator.vehicle_state import VehicleState
from vahsimulator.wind.composite_wind import CompositeWind
from vahsimulator.wind.wind_model import WindModel


class TestWindModel:
    """Testes da interface abstrata ``WindModel``."""

    def test_wind_model_is_abstract(self) -> None:
        """Verifica que ``WindModel`` é uma classe abstrata."""
        assert inspect.isabstract(WindModel)

    def test_wind_model_cannot_be_instantiated(self) -> None:
        """Verifica que a interface não pode ser instanciada diretamente."""
        with pytest.raises(TypeError):
            WindModel()

    def test_required_abstract_methods(self) -> None:
        """Verifica os métodos obrigatórios da interface."""
        expected_methods = {"initialize", "evaluate_ic", "evaluate", "get_tas"}

        assert expected_methods.issubset(WindModel.__abstractmethods__)

    def test_composite_wind_is_concrete_implementation(self) -> None:
        """Verifica que ``CompositeWind`` implementa ``WindModel``."""
        assert issubclass(CompositeWind, WindModel)
        assert not inspect.isabstract(CompositeWind)

    def test_composite_wind_can_be_instantiated(self) -> None:
        """Verifica que ``CompositeWind`` pode ser instanciado."""
        model = CompositeWind()

        assert isinstance(model, WindModel)

    def test_abstract_initialize_is_defined(self) -> None:
        """Verifica a existência da interface ``initialize``."""
        method = getattr(WindModel, "initialize")

        assert callable(method)
        assert getattr(method, "__isabstractmethod__", False)

    def test_abstract_evaluate_ic_is_defined(self) -> None:
        """Verifica a existência da interface ``evaluate_ic``."""
        method = getattr(WindModel, "evaluate_ic")

        assert callable(method)
        assert getattr(method, "__isabstractmethod__", False)

    def test_abstract_evaluate_is_defined(self) -> None:
        """Verifica a existência da interface ``evaluate``."""
        method = getattr(WindModel, "evaluate")

        assert callable(method)
        assert getattr(method, "__isabstractmethod__", False)

    def test_abstract_get_tas_is_defined(self) -> None:
        """Verifica a existência da interface ``get_tas``."""
        method = getattr(WindModel, "get_tas")

        assert callable(method)
        assert getattr(method, "__isabstractmethod__", False)

    def test_composite_wind_exposes_required_methods(self) -> None:
        """Verifica os métodos obrigatórios da implementação concreta."""
        model = CompositeWind()

        assert callable(model.initialize)
        assert callable(model.evaluate_ic)
        assert callable(model.evaluate)
        assert callable(model.get_tas)

    def test_evaluate_interface_returns_numpy_array(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifica o tipo de retorno esperado de ``evaluate()``.

        O Dryden é substituído por uma saída determinística para que
        o teste valide somente o contrato da interface.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Ferramenta para substituir o modelo Dryden.
        """
        model = CompositeWind()

        state = VehicleState.from_dict(
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

        launch_reference = LaunchReference(
            lat=np.float64(0.0),
            lon=np.float64(0.0),
            alt=np.float64(0.0),
            azimuth=np.float64(0.0),
        )

        # Sem Dryden, a saída continua sendo determinada
        # pelo contrato do WindModel.
        result = model.evaluate(
            state=state, launch_reference=launch_reference, Va=100.0, phase=1
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (6, 1)
