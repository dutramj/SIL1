from __future__ import annotations

import numpy as np

from ..vehicle_state import VehicleState


class ConstantWind:
    """
    Modelo de vento médio constante.

    O vento é definido na referência NED e permanece constante durante
    a simulação.

    Notes
    -----
    ``wind_direction`` é interpretado como a direção de onde o vento
    vem, em graus. Por isso, é adicionada uma rotação de ``180 deg``
    para obter a direção do vetor de velocidade do vento.

    ``wind_elevation`` é positivo para cima.
    """

    def __init__(self) -> None:
        self.wind_ned = np.zeros((3, 1))

        self.wind_speed = 0.0
        self.wind_direction = 0.0
        self.wind_elevation = 0.0

    def initialize(
        self,
        wind_speed: float = 0.0,
        wind_direction: float = 0.0,
        wind_elevation: float = 0.0,
    ) -> None:
        """
        Inicializa o vento constante.

        Parameters
        ----------
        wind_speed : float, optional
            Velocidade do vento, em m/s.
        wind_direction : float, optional
            Direção de onde o vento vem, em graus.
        wind_elevation : float, optional
            Elevação do vento, em graus.
        """
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.wind_elevation = wind_elevation

        direction_rad = np.deg2rad(wind_direction) + np.pi
        elevation_rad = -np.deg2rad(wind_elevation)

        self.wind_ned = np.array(
            [
                [wind_speed * np.cos(direction_rad) * np.cos(elevation_rad)],
                [wind_speed * np.sin(direction_rad) * np.cos(elevation_rad)],
                [wind_speed * np.sin(elevation_rad)],
            ],
            dtype=float,
        )

    def evaluate_ned(self, state: VehicleState | None = None) -> np.ndarray:
        """
        Retorna o vento médio na referência NED.

        Parameters
        ----------
        state : VehicleState or None, optional
            Estado atual do veículo. Não utilizado pelo modelo
            constante, mas mantido para compatibilidade de interface.

        Returns
        -------
        numpy.ndarray
            Vetor ``3 x 1`` contendo ``[V_N, V_E, V_D]``.
        """
        return self.wind_ned.copy()
