"""Gerenciamento dos effectors do VAHSimulator.

Este módulo coordena os modelos físicos responsáveis por transformar
comandos de controle em forças, momentos e deflexões aplicadas ao
veículo.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mass_properties import MassPropertiesData
from ..vehicle_state import VehicleState
from control_surfaces_set import ControlSurfacesSet
from rcs import RCS
from tvc import TVC


class EffectorsManager:
    """Coordena os effectors do veículo.

    Parameters
    ----------
    tvc : TVC or None, optional
        Modelo do sistema de controle vetorial de empuxo.
    control_surfaces_set : ControlSurfacesSet or None, optional
        Modelo do conjunto de superfícies de controle.
    rcs : RCS or None, optional
        Modelo do sistema de controle por reação.
    """

    def __init__(
        self,
        tvc: TVC | None = None,
        control_surfaces_set: ControlSurfacesSet | None = None,
        rcs: RCS | None = None,
    ) -> None:
        self.tvc = tvc
        self.control_surfaces_set = control_surfaces_set
        self.rcs = rcs

    def evaluate(
        self,
        state: VehicleState,
        mass_properties_data: MassPropertiesData,
        phase_id: int,
        dt_s: float,
        gyro_b_rad_s: np.ndarray,
        delta_q_tvc: float,
        delta_r_tvc: float,
        pitch_cmd: float,
        yaw_cmd: float,
        delta_p_control_surface: float,
        delta_q_control_surface: float,
        delta_r_control_surface: float,
        error_delta_1: float,
        error_delta_2: float,
        error_delta_3: float,
        error_delta_4: float,
        cs_config_type: str,
        config_parameters: dict[str, Any],
    ) -> tuple[float, float, Any, float, float, float, float, float]:
        """Atualiza os effectors do veículo."""
        ...