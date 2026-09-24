"""Gerenciamento dos modelos de sensores do VAHSimulator.

Este módulo coordena a execução dos modelos de sensores, converte o
estado verdadeiro do simulador nas grandezas físicas necessárias pelos
modelos de sensores e agrupa as medições produzidas em ``SensorsData``.
"""

from __future__ import annotations

import numpy as np

from ..gnss import GNSS
from ..imu import IMUModel
from ..loads import Loads
from ..mass_properties import MassPropertiesData
from ..vehicle_state import VehicleState
from .sensors_data import SensorsData


class SensorsManager:
    """Coordena os modelos de sensores do simulador.

    Parameters
    ----------
    imu : IMUModel
        Modelo de IMU utilizado pelo simulador.
    gnss : GNSS or None, optional
        Modelo GNSS utilizado pelo simulador.

    Notes
    -----
    Esta classe é responsável por conectar os modelos físicos do
    simulador aos modelos de sensores. Os modelos individuais de
    sensores não precisam conhecer ``VehicleState``, ``Loads`` ou
    ``MassPropertiesData``.
    """

    def __init__(
        self,
        imu: IMUModel,
        gnss: GNSS | None = None,
    ) -> None:
        self.imu = imu
        self.gnss = gnss

    def evaluate(
        self,
        phase_id: int,
        state: VehicleState,
        mass_properties_data: MassPropertiesData,
        total_loads: Loads,
        grav_acc_m_s2: np.ndarray,
        dt_s: float,
    ) -> SensorsData:
        """Atualiza os sensores e retorna suas medições.

        Parameters
        ----------
        phase_id: int
            Estágio atual do veículo.
        state : VehicleState
            Estado verdadeiro do veículo.
        mass_properties_data : MassPropertiesData
            Dados de massa e propriedades inerciais do veículo.
        total_loads : Loads
            Cargas totais aplicadas ao veículo.
        grav_acc_m_s2 : numpy.ndarray
            Aceleração gravitacional no referencial do corpo, em m/s².
        dt_s : float
            Intervalo de simulação desde a última chamada, em segundos.

        Returns
        -------
        SensorsData
            Conjunto das novas medições disponíveis neste instante.
        """
        specific_force_b_m_s2 = (
            total_loads.force / mass_properties_data.mass
            - grav_acc_m_s2
        )

        state_array = state.vector
        angular_rate_b_rad_s = state_array[10:13]

        imu_data = self.imu.evaluate(
            specific_force_b_m_s2=specific_force_b_m_s2,
            angular_rate_b_rad_s=angular_rate_b_rad_s,
            time_s=state.time,
            dt_s=dt_s,
        )

        gnss_data = None

        if self.gnss is not None:
            gnss_data = self.gnss.step(
                state=state,
                time_s=state.time,
                dt_s=dt_s,
            )

        return SensorsData(
            time_s=state.time,
            phase_id=phase_id,
            imu=imu_data,
            gnss=gnss_data,
        )