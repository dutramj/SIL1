"""Estruturas de dados dos sensores do veículo.

Este módulo define as estruturas de dados utilizadas para transportar
as medições dos sensores do VAHSimulator para o sistema de GNC.

As estruturas deste módulo representam exclusivamente medições que
estariam disponíveis para um sistema de GNC embarcado. Estados verdadeiros
(truth), como ``VehicleState``, não devem ser armazenados nestas estruturas.

References
----------
Farrell, J. A., Silva, F. O., Rahman, F., & Wendel, J. (2022).
"Inertial Measurement Unit Error Modeling Tutorial: Inertial Navigation
System State Estimation with Real-Time Sensor Calibration".
IEEE Control Systems Magazine, 42(6), 40-66.
doi:10.1109/MCS.2022.3209059

Groves, P. D. (2013).
Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems.
Artech House.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class IMUData:
    """Dados de medição da Unidade de Medição Inercial (IMU).

    Os dados representam as medições efetivamente disponibilizadas ao
    sistema de GNC, e não os valores verdadeiros utilizados internamente
    pelo modelo de dinâmica do simulador.

    Parameters
    ----------
    time_s : float
        Instante de aquisição da medição, em segundos.
    specific_force_b_m_s2 : numpy.ndarray
        Força específica medida nos eixos do corpo, em m/s².
        O vetor possui os componentes ``[fx, fy, fz]``.
    angular_rate_b_rad_s : numpy.ndarray
        Velocidade angular medida nos eixos do corpo, em rad/s.
        O vetor possui os componentes ``[p, q, r]``.

    Notes
    -----
    Os vetores são representados como ``numpy.ndarray`` com três
    componentes. A convenção dos eixos é a mesma utilizada pelo
    VAHSimulator.

    A classe é ``frozen`` para impedir a substituição dos campos após
    a criação do objeto. Os arrays armazenados, entretanto, continuam
    sendo objetos mutáveis do NumPy.
    """

    time_s: float
    specific_force_b_m_s2: np.ndarray
    angular_rate_b_rad_s: np.ndarray


@dataclass(frozen=True, slots=True)
class GNSSData:
    """Dados de medição do receptor GNSS.

    Parameters
    ----------
    time_s : float
        Instante de aquisição da medição, em segundos.
    position_eci_m : numpy.ndarray
        Posição medida no referencial ECI, em metros.
        O vetor possui os componentes ``[x, y, z]``.
    velocity_eci_m_s : numpy.ndarray
        Velocidade medida no referencial ECI, em m/s.
        O vetor possui os componentes ``[vx, vy, vz]``.
    yaw_eci_rad : float
        Ângulo de yaw medido, em radianos.

    Notes
    -----
    Os dados representam a medição disponibilizada ao GNC e não o
    estado verdadeiro utilizado internamente pelo simulador.
    """

    time_s: float
    position_eci_m: np.ndarray
    velocity_eci_m_s: np.ndarray
    yaw_eci_rad: float


@dataclass(frozen=True, slots=True)
class SensorsData:
    """Conjunto de medições disponíveis para o GNC.

    Esta estrutura representa a interface de dados entre os modelos
    de sensores do VAHSimulator e o sistema de GNC.

    Parameters
    ----------
    phase_id: int
        Estágio atual do veículo.
    time_s : float
        Instante de referência do conjunto de medições, em segundos.
    imu : IMUData or None, optional
        Medição da IMU disponível no instante ``time_s``.
        ``None`` indica que não houve uma nova medição da IMU nesse
        instante.

    Notes
    -----
    Sensores diferentes podem operar em frequências de amostragem
    diferentes. Portanto, a ausência de uma medição em determinado
    instante é representada explicitamente por ``None``.

    Novos sensores, como GNSS, poderão ser adicionados posteriormente
    sem alterar a interface dos sensores existentes.
    """

    phase_id: int
    time_s: float
    imu: IMUData | None = None
