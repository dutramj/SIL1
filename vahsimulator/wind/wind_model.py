from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..launching_reference import LaunchReference
from ..vehicle_state import VehicleState


class WindModel(ABC):
    """
    Interface para o modelo global de vento utilizado pelo simulador.

    O ``WindModel`` é responsável por fornecer o vento resultante na
    referência do corpo do veículo. Os modelos individuais de vento,
    como vento médio, turbulência Dryden e rajadas discretas, são
    componentes internos do modelo composto.

    Notes
    -----
    O modelo deve retornar um vetor ``6 x 1`` contendo:

    ``[u_w, v_w, w_w, p_w, q_w, r_w]``

    onde as três primeiras componentes são velocidades relativas do
    vento na referência do corpo e as três últimas são velocidades
    angulares da perturbação.

    """

    @abstractmethod
    def initialize(
        self, config_data: dict, dt: float, wind_config, seed: int | None = None
    ) -> None:
        """
        Inicializa o modelo.

        Parameters
        ----------
        config_data : dict
            Dados gerais de configuração do simulador.
        dt : float
            Período de amostragem.
        wind_config : WindConfig
            Configuração específica do modelo de vento.
        seed : int or None, optional
            Semente para geração pseudoaleatória.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_ic(
        self, state: VehicleState, launch_reference: LaunchReference
    ) -> None:
        """
        Inicializa o estado do modelo no instante inicial.

        Parameters
        ----------
        state : VehicleState
            Estado inicial do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        state: VehicleState,
        launch_reference: LaunchReference,
        Va: float,
        phase: int,
    ) -> np.ndarray:
        """
        Avalia o vento na condição atual.

        Parameters
        ----------
        state : VehicleState
            Estado atual do veículo.
        launch_reference : LaunchReference
            Referência de lançamento.
        Va : float
            Velocidade aerodinâmica.
        phase : int
            Identificador da fase de voo.

        Returns
        -------
        numpy.ndarray
            Vetor ``6 x 1`` com vento linear e angular na referência
            do corpo.
        """
        raise NotImplementedError

    @abstractmethod
    def get_tas(self) -> np.float64:
        """
        Retorna a velocidade verdadeira do ar.

        Returns
        -------
        numpy.float64
            True Airspeed (TAS).
        """
        raise NotImplementedError
