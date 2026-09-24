from __future__ import annotations

import numpy as np

from ..launching_reference import LaunchReference
from ..vehicle_state import VehicleState
from .constant_wind import ConstantWind
from .discrete_gust_list import DiscreteGustList
from .dryden import DrydenTurbulence
from .wind_config import WindConfig
from .wind_frame import WindFrameTransformer
from .wind_model import WindModel


class CompositeWind(WindModel):
    """
    Modelo composto de vento.

    Combina:
    * vento médio;
    * rajadas discretas;
    * turbulência Dryden opcional.

    Notes
    -----
    O vento médio e as rajadas discretas são inicialmente expressos
    no referencial NED.

    A transformação NED -> body é realizada uma única vez pelo
    ``WindFrameTransformer``.

    O Dryden gera suas componentes de turbulência diretamente nos
    eixos do veículo e, portanto, não passa novamente pela
    transformação NED -> body.
    """

    def __init__(self, dryden: DrydenTurbulence | None = None) -> None:
        self.constant_wind = ConstantWind()
        self.discrete_gusts = DiscreteGustList()
        self.transformer = WindFrameTransformer()
        self.dryden = dryden

        self.config_data: dict = {}
        self.dt = 0.0
        self._tas = np.float64(0.0)

    def initialize(
        self,
        config_data: dict,
        dt: float,
        wind_config: WindConfig,
        seed: int | None = None,
    ) -> None:
        """
        Inicializa os modelos de vento.

        Parameters
        ----------
        config_data : dict
            Configuração geral do simulador.
        dt : float
            Período de amostragem.
        wind_config : WindConfig
            Configuração dos modelos de vento.
        seed : int, optional
            Semente do gerador aleatório.
        """
        self.config_data = config_data
        self.dt = float(dt)

        self.constant_wind.initialize(
            wind_speed=wind_config.constant_wind.wind_speed,
            wind_direction=wind_config.constant_wind.wind_direction,
            wind_elevation=wind_config.constant_wind.wind_elevation,
        )

        self.discrete_gusts = DiscreteGustList.from_parameters(
            wind_config.discrete_gusts
        )

        self.dryden = None

        if wind_config.turbulence is not None:
            if wind_config.turbulence.type == "dryden":
                self.dryden = DrydenTurbulence(
                    turbulence=wind_config.turbulence.severity
                )

                self.dryden.initialize(config_data=config_data, dt=dt, seed=seed)

    def _evaluate_environment_ned(self, state: VehicleState) -> np.ndarray:
        """Avalia o vento ambiental no referencial NED.

        O vento ambiental é composto pela soma do vento constante e das
        rajadas discretas. A orientação das componentes longitudinal e
        lateral das rajadas é definida pela direção do vento constante.

        Parameters
        ----------
        state : VehicleState
            Estado atual do veículo.

        Returns
        -------
        numpy.ndarray
            Vetor de vento NED com dimensão ``(3, 1)``.
        """
        constant_wind_ned = self.constant_wind.evaluate_ned(state)

        discrete_gust_ned = self.discrete_gusts.evaluate_ned(
            altitude_m=state.alt, wind_direction_deg=self.constant_wind.wind_direction
        )

        return constant_wind_ned + discrete_gust_ned

    def _ned_to_body(self, state: VehicleState, wind_ned: np.ndarray) -> np.ndarray:
        """
        Transforma uma velocidade de vento NED para body.

        Parameters
        ----------
        state : VehicleState
            Estado atual do veículo.
        wind_ned : numpy.ndarray
            Velocidade do vento no referencial NED.

        Returns
        -------
        numpy.ndarray
            Velocidade do vento no referencial body.
        """
        return self.transformer.ned_to_body(
            wind_ned=wind_ned,
            roll_rad=state.roll_ned,
            pitch_rad=state.pitch_ned,
            yaw_rad=state.yaw_ned,
        )

    def evaluate_ic(
        self, state: VehicleState, launch_reference: LaunchReference
    ) -> None:
        """
        Avalia o vento no instante inicial e calcula a TAS.

        Parameters
        ----------
        state : VehicleState
            Estado inicial do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        """
        environment_ned = self._evaluate_environment_ned(state=state)

        environment_body = self._ned_to_body(state=state, wind_ned=environment_ned)

        velocity_body = np.asarray(state.velocity_body, dtype=np.float64)

        air_velocity_body = velocity_body - environment_body

        self._tas = np.float64(np.linalg.norm(air_velocity_body))

        if self.dryden is not None:
            self.dryden.evaluate_ic(state=state, constant_wind_body=environment_body)

    def evaluate(
        self,
        state: VehicleState,
        launch_reference: LaunchReference,
        Va: float,
        phase: int,
    ) -> np.ndarray:
        """
        Avalia o vento total no referencial body.

        Parameters
        ----------
        state : VehicleState
            Estado atual do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        Va : float
            Velocidade verdadeira do veículo.
        phase : int
            Identificador da fase de voo.

        Returns
        -------
        numpy.ndarray
            Vetor de vento body ``[u, v, w, p, q, r]``.
        """
        environment_ned = self._evaluate_environment_ned(state=state)

        environment_body = self._ned_to_body(state=state, wind_ned=environment_ned)

        angular_body = np.zeros((3, 1), dtype=np.float64)

        if self.dryden is not None:
            dryden_wind = self.dryden.evaluate(
                state=state, Va=Va, phase=phase, config_data=self.config_data
            )

            environment_body += dryden_wind[0:3]
            angular_body = dryden_wind[3:6]

        return self.transformer.assemble_wind(
            wind_body=environment_body, angular_wind_body=angular_body
        )

    def get_tas(self) -> np.float64:
        """
        Retorna a velocidade verdadeira calculada.

        Returns
        -------
        numpy.float64
            True Airspeed.
        """
        return np.float64(self._tas)
